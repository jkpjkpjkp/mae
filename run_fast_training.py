#!/usr/bin/env python3
"""
Fast MAE training script that automatically runs preprocessing if needed.
"""

import os
import subprocess
import argparse
import sys


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error: {description} failed with return code {result.returncode}")
        sys.exit(1)
    print(f"✓ {description} completed successfully")


def main():
    parser = argparse.ArgumentParser(description='Fast MAE training with automatic preprocessing')
    
    # Preprocessing args
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing STL files')
    parser.add_argument('--nside', type=int, default=32, help='HEALPix nside parameter')
    parser.add_argument('--skip_preprocessing', action='store_true', help='Skip preprocessing step')
    
    # Training args
    parser.add_argument('--model_size', type=str, default='base', choices=['tiny', 'small', 'base', 'large'], help='Model size')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    processed_dir = 'processed_data_gpu'
    
    # Step 1: Check if preprocessing is needed
    if not args.skip_preprocessing:
        if not os.path.exists(processed_dir) or len(os.listdir(processed_dir)) < 10:
            print(f"Preprocessed data not found or incomplete. Starting fast preprocessing...")
            
            # Build GPU preprocessing command
            preprocess_cmd = f"uv run preprocess_data_gpu.py --data_dir {args.data_dir} --output_dir {processed_dir} --nside {args.nside}"
            
            run_command(preprocess_cmd, "GPU preprocessing (single-threaded but blazing fast!)")
        else:
            print(f"✓ Preprocessed data found in {processed_dir}")
    else:
        print("Skipping preprocessing as requested")
    
    # Step 2: Run training
    train_cmd = f"uv run train_mae.py --processed_dir {processed_dir} --model_size {args.model_size} --batch_size {args.batch_size} --epochs {args.epochs} --lr {args.lr} --nside {args.nside}"
    
    run_command(train_cmd, "MAE training")
    
    print(f"\n{'='*60}")
    print("🎉 Fast MAE training pipeline completed successfully!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main() 