import os
import glob
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import pickle

# Import our modules
from a import read_stl_file, center_of_mass, surface_distances_healpix
from c import (mae_1d_tiny_patch64, mae_1d_small_patch64, mae_1d_base_patch64, mae_1d_large_patch64,
               mae_1d_tiny_patch64_short, mae_1d_small_patch64_short)


class PreprocessedDataset(Dataset):
    """Fast dataset that loads preprocessed HEALPix sequences from disk."""
    
    def __init__(self, processed_dir, normalize=True):
        self.processed_dir = processed_dir
        self.normalize = normalize
        
        # Load file mapping
        mapping_path = os.path.join(processed_dir, 'file_mapping.pkl')
        with open(mapping_path, 'rb') as f:
            file_mapping = pickle.load(f)
        
        self.successful_files = file_mapping['successful_files']
        self.npy_files = [output_path for _, output_path in self.successful_files]
        
        print(f"Found {len(self.npy_files)} preprocessed files")
        
    def __len__(self):
        return len(self.npy_files)
    
    def __getitem__(self, idx):
        npy_file = self.npy_files[idx]
        
        # Load preprocessed sequence
        pixel_distances = np.load(npy_file)
        
        # Convert to torch tensor: (1, seq_len) for single channel 1D sequence
        sequence = torch.from_numpy(pixel_distances).float().unsqueeze(0)
        
        if self.normalize:
            # Normalize to zero mean and unit variance
            sequence = (sequence - sequence.mean()) / (sequence.std() + 1e-8)
        
        return sequence


class STLDataset(Dataset):
    """Dataset that processes STL files into HEALPix surface distance sequences."""
    
    def __init__(self, data_dir, nside=32, normalize=True):
        self.data_dir = data_dir
        self.nside = nside
        self.normalize = normalize
        
        # Find all STL files
        self.stl_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.stl'):
                    self.stl_files.append(os.path.join(root, file))
        
        print(f"Found {len(self.stl_files)} STL files")
        
        # Calculate sequence length for this nside
        self.seq_len = 12 * nside**2
        print(f"Using nside={nside}, sequence length={self.seq_len}")
        
    def __len__(self):
        return len(self.stl_files)
    
    def __getitem__(self, idx):
        stl_file = self.stl_files[idx]
        
        try:
            # Process STL file
            vertices, triangles = read_stl_file(stl_file)
            com = center_of_mass(vertices, triangles)
            
            # Generate HEALPix surface distances (using NESTED order for spatial locality)
            pixel_directions, pixel_distances, pixel_indices = surface_distances_healpix(
                vertices, triangles, com, nside=self.nside
            )
            
            # Convert to torch tensor: (1, seq_len) for single channel 1D sequence
            sequence = torch.from_numpy(pixel_distances).float().unsqueeze(0)
            
            if self.normalize:
                # Normalize to zero mean and unit variance
                sequence = (sequence - sequence.mean()) / (sequence.std() + 1e-8)
            
            return sequence
            
        except Exception as e:
            print(f"Error processing {stl_file}: {e}")
            # Return a dummy sequence if processing fails
            seq_len = 12 * self.nside**2
            return torch.randn(1, seq_len)


def train_mae(model, dataloader, optimizer, device, epoch):
    """Train the MAE model for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, sequences in enumerate(progress_bar):
        sequences = sequences.to(device)
        
        # Forward pass
        loss, pred, mask = model(sequences, mask_ratio=0.75)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        avg_loss = total_loss / num_batches
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train 1D MAE on HEALPix surface distances')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing STL files')
    parser.add_argument('--processed_dir', type=str, default=None, help='Directory containing preprocessed data (faster)')
    parser.add_argument('--model_size', type=str, default='tiny', choices=['tiny', 'small', 'base', 'large'], help='Model size')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--nside', type=int, default=16, help='HEALPix nside parameter (16=3072 pixels, 32=12288 pixels)')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create dataset and dataloader
    if args.processed_dir and os.path.exists(args.processed_dir):
        print(f"Using preprocessed data from: {args.processed_dir}")
        dataset = PreprocessedDataset(args.processed_dir)
    else:
        # Check if GPU preprocessed data exists
        gpu_processed_dir = 'processed_data_gpu'
        fast_processed_dir = 'processed_data_fast'
        
        if os.path.exists(gpu_processed_dir):
            print(f"Using GPU preprocessed data from: {gpu_processed_dir}")
            dataset = PreprocessedDataset(gpu_processed_dir)
        elif os.path.exists(fast_processed_dir):
            print(f"Using CPU preprocessed data from: {fast_processed_dir}")
            dataset = PreprocessedDataset(fast_processed_dir)
        else:
            print(f"No preprocessed data found. Please run preprocessing first:")
            print(f"  uv run preprocess_data_gpu.py --data_dir {args.data_dir} --nside {args.nside}")
            print(f"Falling back to slow on-the-fly processing from: {args.data_dir}")
            dataset = STLDataset(args.data_dir, nside=args.nside)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Create model based on nside and model size
    seq_len = 12 * args.nside**2
    
    if seq_len == 3072:  # nside=16
        if args.model_size == 'tiny':
            model = mae_1d_tiny_patch64_short()
        elif args.model_size == 'small':
            model = mae_1d_small_patch64_short()
        else:
            print(f"Only tiny/small models available for nside=16, using small")
            model = mae_1d_small_patch64_short()
    elif seq_len == 12288:  # nside=32
        if args.model_size == 'tiny':
            model = mae_1d_tiny_patch64()
        elif args.model_size == 'small':
            model = mae_1d_small_patch64()
        elif args.model_size == 'base':
            model = mae_1d_base_patch64()
        else:  # large
            model = mae_1d_large_patch64()
    else:
        raise ValueError(f"No model available for sequence length {seq_len} (nside={args.nside})")
    
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: {args.model_size}, Total params: {total_params:,}, Trainable: {trainable_params:,}')
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    
    # Training loop
    print(f'Starting training for {args.epochs} epochs...')
    for epoch in range(args.epochs):
        avg_loss = train_mae(model, dataloader, optimizer, device, epoch + 1)
        print(f'Epoch {epoch + 1}/{args.epochs}, Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'mae_{args.model_size}_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'args': args
            }, checkpoint_path)
            print(f'Saved checkpoint: {checkpoint_path}')
    
    # Save final model
    final_path = os.path.join(args.save_dir, f'mae_{args.model_size}_final.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'args': args
    }, final_path)
    print(f'Saved final model: {final_path}')


if __name__ == '__main__':
    main() 