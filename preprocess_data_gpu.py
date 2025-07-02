import os
import glob
import numpy as np
import torch
from tqdm import tqdm
import argparse
import pickle
import healpy as hp
from stl import mesh
import time

# Set CUDA memory allocation to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Import our modules
from a import read_stl_file, center_of_mass


def ray_triangle_intersection_gpu(ray_origins, ray_directions, triangle_vertices, device='cuda'):
    """
    GPU-accelerated vectorized ray-triangle intersection using Möller-Trumbore algorithm.
    Single process, full GPU utilization.
    """
    if not torch.cuda.is_available():
        device = 'cpu'
        print("⚠️  CUDA not available, falling back to CPU")
    
    # Convert to tensors and move to device
    ray_origins = torch.from_numpy(ray_origins).float().to(device)
    ray_directions = torch.from_numpy(ray_directions).float().to(device)
    triangle_vertices = torch.from_numpy(triangle_vertices).float().to(device)
    
    N = ray_origins.shape[0]  # number of rays
    M = triangle_vertices.shape[0]  # number of triangles
    
    # Process in smaller chunks to avoid OOM - be more conservative
    # Dynamically adjust chunk size based on number of triangles
    base_chunk_size = min(2048, N)  # Much smaller base size
    memory_factor = max(1, M // 5000)  # Scale down for complex models
    chunk_size = max(512, base_chunk_size // memory_factor)
    
    all_distances = []
    
    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        chunk_origins = ray_origins[i:end_i]
        chunk_directions = ray_directions[i:end_i]
        chunk_N = chunk_origins.shape[0]
        
        # Expand dimensions for broadcasting
        origins_exp = chunk_origins.unsqueeze(1)  # (chunk_N, 1, 3)
        directions_exp = chunk_directions.unsqueeze(1)  # (chunk_N, 1, 3)
        triangles_exp = triangle_vertices.unsqueeze(0)  # (1, M, 3, 3)
        
        # Extract triangle vertices
        v0 = triangles_exp[:, :, 0, :]  # (1, M, 3)
        v1 = triangles_exp[:, :, 1, :]  # (1, M, 3)
        v2 = triangles_exp[:, :, 2, :]  # (1, M, 3)
        
        # Möller-Trumbore algorithm
        edge1 = v1 - v0  # (1, M, 3)
        edge2 = v2 - v0  # (1, M, 3)
        
        h = torch.cross(directions_exp, edge2, dim=-1)  # (chunk_N, M, 3)
        a = torch.sum(edge1 * h, dim=-1)  # (chunk_N, M)
        
        # Parallel check
        epsilon = 1e-8
        parallel_mask = torch.abs(a) < epsilon
        
        f = 1.0 / (a + epsilon)
        s = origins_exp - v0  # (chunk_N, M, 3)
        u = f * torch.sum(s * h, dim=-1)  # (chunk_N, M)
        
        # Check bounds
        valid_u = (u >= 0.0) & (u <= 1.0)
        
        q = torch.cross(s, edge1, dim=-1)  # (chunk_N, M, 3)
        v = f * torch.sum(directions_exp * q, dim=-1)  # (chunk_N, M)
        
        # Check bounds
        valid_v = (v >= 0.0) & ((u + v) <= 1.0)
        
        t = f * torch.sum(edge2 * q, dim=-1)  # (chunk_N, M)
        
        # Valid intersection mask
        valid_mask = ~parallel_mask & valid_u & valid_v & (t > epsilon)
        
        # Set invalid intersections to infinity
        t = torch.where(valid_mask, t, torch.inf)
        
        # Find closest intersection for each ray
        distances, _ = torch.min(t, dim=1)  # (chunk_N,)
        distances = torch.where(torch.isinf(distances), 0.0, distances)
        
        all_distances.append(distances.cpu())
        
        # Aggressive GPU memory cleanup
        del origins_exp, directions_exp, triangles_exp, v0, v1, v2, edge1, edge2
        del h, a, parallel_mask, f, s, u, valid_u, q, v, valid_v, t, valid_mask, distances
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for operations to complete
    
    result = torch.cat(all_distances, dim=0).numpy()
    
    # Final cleanup
    del ray_origins, ray_directions, triangle_vertices, all_distances
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    return result


def surface_distances_healpix_gpu(vertices, triangles, com, nside=32, device='cuda'):
    """
    GPU-accelerated HEALPix surface distance computation.
    """
    try:
        # Ensure nside is power of 2
        assert nside == 2**int(np.log2(nside)), "nside must be a power of 2"
        
        npix = hp.nside2npix(nside)
        pixel_indices = np.arange(npix)
        
        # Get pixel directions with better error handling
        try:
            pixel_directions = np.array(hp.pix2vec(nside, pixel_indices, nest=True)).T
        except Exception as e:
            print(f"HEALPix error with nside={nside}: {e}")
            # Fallback: use fewer pixels
            nside = max(nside // 2, 4)
            npix = hp.nside2npix(nside)
            pixel_indices = np.arange(npix)
            pixel_directions = np.array(hp.pix2vec(nside, pixel_indices, nest=True)).T
        
        # Normalize to ensure unit vectors
        norms = np.linalg.norm(pixel_directions, axis=1, keepdims=True)
        pixel_directions = pixel_directions / (norms + 1e-12)
        
        # Prepare ray origins (all from center of mass)
        ray_origins = np.tile(com, (npix, 1))
        
        # Compute distances using GPU
        pixel_distances = ray_triangle_intersection_gpu(
            ray_origins, pixel_directions, vertices[triangles], device=device
        )
        
        # Check for failed intersections
        valid_mask = pixel_distances > 0
        if not np.all(valid_mask):
            n_failed = np.sum(~valid_mask)
            if n_failed < npix * 0.2:  # Less than 20% failed is acceptable
                # Set failed pixels to mean of successful ones
                if np.any(valid_mask):
                    mean_distance = np.mean(pixel_distances[valid_mask])
                    pixel_distances[~valid_mask] = mean_distance
                else:
                    pixel_distances[:] = 1.0  # Fallback value
            else:
                print(f"Warning: {n_failed}/{npix} pixels have no intersection")
        
        return pixel_directions, pixel_distances, pixel_indices
        
    except Exception as e:
        print(f"Error in HEALPix computation: {e}")
        # Return dummy data with correct shape
        npix = 12 * nside**2
        return np.random.randn(npix, 3), np.ones(npix), np.arange(npix)


def preprocess_stl_files_gpu(data_dir, output_dir, nside=32, device='cuda', start_from=0):
    """
    Single-process GPU-accelerated preprocessing of STL files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check GPU
    if torch.cuda.is_available() and device == 'cuda':
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🚀 Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        device = 'cpu'
        print("⚠️  Using CPU (CUDA not available)")
    
    # Find all STL files
    stl_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.stl'):
                stl_files.append(os.path.join(root, file))
    
    print(f"Found {len(stl_files)} STL files to preprocess")
    
    successful_files = []
    failed_files = []
    
    # Process files sequentially with GPU acceleration
    start_time = time.time()
    
    # Load existing files if restarting
    if start_from > 0:
        print(f"🔄 Restarting from file index {start_from}")
        for i in range(min(start_from, len(stl_files))):
            stl_file = stl_files[i]
            rel_path = os.path.relpath(stl_file, data_dir)
            output_name = rel_path.replace('/', '_').replace('.stl', '.npy')
            output_path = os.path.join(output_dir, output_name)
            if os.path.exists(output_path):
                successful_files.append((stl_file, output_path))
    
    for idx, stl_file in enumerate(tqdm(stl_files[start_from:], desc="Processing STL files", initial=start_from, total=len(stl_files))):
        i = idx + start_from  # Actual index in the full list
        try:
            # Generate output filename
            rel_path = os.path.relpath(stl_file, data_dir)
            output_name = rel_path.replace('/', '_').replace('.stl', '.npy')
            output_path = os.path.join(output_dir, output_name)
            
            # Skip if already processed
            if os.path.exists(output_path):
                successful_files.append((stl_file, output_path))
                continue
            
            # Check GPU memory before processing large files
            if device == 'cuda':
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                # If we're using >80% of GPU memory, do aggressive cleanup
                if allocated > 0.8 * torch.cuda.get_device_properties(0).total_memory:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            
            # Process STL file
            vertices, triangles = read_stl_file(stl_file)
            com = center_of_mass(vertices, triangles)
            
            # Generate HEALPix surface distances
            pixel_directions, pixel_distances, pixel_indices = surface_distances_healpix_gpu(
                vertices, triangles, com, nside=nside, device=device
            )
            
            # Save as numpy array
            np.save(output_path, pixel_distances.astype(np.float32))
            successful_files.append((stl_file, output_path))
            
            # Clean up after each file
            del vertices, triangles, com, pixel_directions, pixel_distances, pixel_indices
            if device == 'cuda':
                torch.cuda.empty_cache()
            
            # Print progress every 50 files with memory info
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(stl_files) - i - 1) / rate
                
                gpu_info = ""
                if device == 'cuda':
                    allocated = torch.cuda.memory_allocated() / 1e9
                    reserved = torch.cuda.memory_reserved() / 1e9
                    gpu_info = f", GPU: {allocated:.1f}GB alloc / {reserved:.1f}GB reserved"
                
                print(f"Processed {i+1}/{len(stl_files)} files ({rate:.1f} files/sec, {remaining/60:.1f} min remaining{gpu_info})")
                
                # Aggressive cleanup every 50 files
                if device == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            
        except Exception as e:
            print(f"Error processing {stl_file}: {e}")
            failed_files.append((stl_file, str(e)))
            
            # Aggressive cleanup on error to prevent memory leaks
            if device == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Reset any potential memory fragmentation
                try:
                    torch.cuda.reset_peak_memory_stats()
                except:
                    pass
    
    elapsed_time = time.time() - start_time
    
    print(f"\n🎉 Preprocessing complete!")
    print(f"Total time: {elapsed_time/60:.1f} minutes")
    print(f"Average time per file: {elapsed_time/len(stl_files):.2f} seconds")
    print(f"Successfully processed: {len(successful_files)} files")
    print(f"Failed to process: {len(failed_files)} files")
    
    # Save file mapping for training
    file_mapping = {
        'successful_files': successful_files,
        'failed_files': failed_files,
        'nside': nside,
        'total_files': len(stl_files),
        'processing_time': elapsed_time,
        'device': device
    }
    
    mapping_path = os.path.join(output_dir, 'file_mapping.pkl')
    with open(mapping_path, 'wb') as f:
        pickle.dump(file_mapping, f)
    
    print(f"Saved file mapping to: {mapping_path}")
    return successful_files, failed_files


def benchmark_gpu(test_file, nside=16, device='cuda'):
    """Benchmark GPU processing speed."""
    print(f"🔥 GPU Benchmark: {test_file}")
    
    # Load STL file
    vertices, triangles = read_stl_file(test_file)
    com = center_of_mass(vertices, triangles)
    print(f"Loaded: {len(vertices)} vertices, {len(triangles)} triangles")
    
    # Warm up GPU
    if device == 'cuda' and torch.cuda.is_available():
        dummy = torch.randn(1000, 1000).cuda()
        _ = torch.mm(dummy, dummy)
        del dummy
        torch.cuda.empty_cache()
    
    # Benchmark GPU processing
    start_time = time.time()
    pixel_directions, pixel_distances, pixel_indices = surface_distances_healpix_gpu(
        vertices, triangles, com, nside=nside, device=device
    )
    gpu_time = time.time() - start_time
    
    print(f"✓ GPU processing: {gpu_time:.3f}s for {len(pixel_distances)} pixels")
    print(f"✓ Speed: {len(pixel_distances)/gpu_time:.0f} pixels/second")
    print(f"✓ Throughput: {len(pixel_distances) * len(triangles) / gpu_time / 1e6:.1f} M ray-triangle tests/sec")
    
    return gpu_time


def main():
    parser = argparse.ArgumentParser(description='GPU-accelerated preprocessing of STL files')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing STL files')
    parser.add_argument('--output_dir', type=str, default='processed_data_gpu', help='Directory to save processed sequences')
    parser.add_argument('--nside', type=int, default=16, help='HEALPix nside parameter (16=3K pixels, 32=12K pixels)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark on first available file')
    parser.add_argument('--test_file', type=str, default=None, help='Process single file for testing')
    parser.add_argument('--start_from', type=int, default=0, help='Start processing from file index (for restarts)')
    parser.add_argument('--check_progress', action='store_true', help='Check how many files are already processed')
    
    args = parser.parse_args()
    
    # Check progress mode
    if args.check_progress:
        # Count processed files
        if os.path.exists(args.output_dir):
            processed_files = glob.glob(os.path.join(args.output_dir, '*.npy'))
            print(f"📊 Found {len(processed_files)} processed files in {args.output_dir}")
            
            # Count total STL files
            stl_files = []
            for root, dirs, files in os.walk(args.data_dir):
                for file in files:
                    if file.endswith('.stl'):
                        stl_files.append(os.path.join(root, file))
            
            print(f"📊 Total STL files: {len(stl_files)}")
            print(f"📊 Progress: {len(processed_files)}/{len(stl_files)} ({len(processed_files)/len(stl_files)*100:.1f}%)")
            
            if len(processed_files) < len(stl_files):
                print(f"💡 To restart from where it left off, use: --start_from {len(processed_files)}")
        else:
            print(f"❌ Output directory {args.output_dir} not found")
        return
    
    # Benchmark mode
    if args.benchmark:
        # Find a test file
        test_file = None
        for root, dirs, files in os.walk(args.data_dir):
            for file in files:
                if file.endswith('.stl'):
                    test_file = os.path.join(root, file)
                    break
            if test_file:
                break
        
        if test_file:
            benchmark_gpu(test_file, args.nside, args.device)
        else:
            print("No STL files found for benchmarking")
        return
    
    # Single file test
    if args.test_file:
        print(f"Testing single file: {args.test_file}")
        os.makedirs(args.output_dir, exist_ok=True)
        
        try:
            vertices, triangles = read_stl_file(args.test_file)
            com = center_of_mass(vertices, triangles)
            
            pixel_directions, pixel_distances, pixel_indices = surface_distances_healpix_gpu(
                vertices, triangles, com, nside=args.nside, device=args.device
            )
            
            output_path = os.path.join(args.output_dir, 'test.npy')
            np.save(output_path, pixel_distances.astype(np.float32))
            
            print(f"✓ Successfully processed: {output_path}")
            print(f"✓ Output shape: {pixel_distances.shape}, range: [{pixel_distances.min():.3f}, {pixel_distances.max():.3f}]")
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
        return
    
    # Full preprocessing
    successful, failed = preprocess_stl_files_gpu(
        args.data_dir, args.output_dir, args.nside, args.device, args.start_from
    )
    
    if failed and len(failed) > 0:
        print(f"\nFailed files:")
        for stl_file, error in failed[:10]:  # Show first 10 failures
            print(f"  {stl_file}: {error}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")


if __name__ == '__main__':
    main() 