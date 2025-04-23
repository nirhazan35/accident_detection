import os
import time
import json
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from data_processor import prepare_dataloaders, VideoDataset

def test_dataloader_configs(config_path):
    """
    Test different DataLoader configurations to find the optimal setup
    
    Args:
        config_path (str): Path to the config JSON file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Base directories
    accident_dir = config['data']['accident_dir']
    non_accident_dir = config['data']['non_accident_dir']
    
    # Test different numbers of workers
    worker_counts = [0, 1, 2, 4, 8]
    batch_sizes = [4, 8, 16, 32]
    pin_memory_options = [True, False]
    
    results = []
    
    # First test: number of workers with default settings
    print("Testing different worker counts...")
    for num_workers in worker_counts:
        start_time = time.time()
        
        train_loader, _, _ = prepare_dataloaders(
            accident_dir=accident_dir,
            non_accident_dir=non_accident_dir,
            batch_size=config['training']['batch_size'],
            num_frames=config['data']['num_frames'],
            frame_interval=config['data']['frame_interval'],
            train_ratio=0.3,  # Use smaller portion for testing
            val_ratio=0.0,
            num_workers=num_workers,
            balance_classes=False  # Disable for benchmarking
        )
        
        # Time how long it takes to iterate through the data
        data_loading_time = 0
        batch_count = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for batch_idx, (videos, labels) in enumerate(train_loader):
            # Only process 10 batches for speed
            if batch_idx >= 10:
                break
                
            # Simulate sending to GPU
            batch_start = time.time()
            videos = videos.to(device)
            labels = labels.to(device)
            data_loading_time += time.time() - batch_start
            batch_count += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_batch_time = data_loading_time / batch_count if batch_count > 0 else 0
        
        results.append({
            'test_type': 'workers',
            'num_workers': num_workers,
            'batch_size': config['training']['batch_size'],
            'pin_memory': True,
            'total_time': total_time,
            'avg_batch_time': avg_batch_time
        })
        
        print(f"Workers: {num_workers}, Total time: {total_time:.2f}s, Avg batch time: {avg_batch_time:.4f}s")
    
    # Second test: different batch sizes with best worker count
    best_worker_count = min(results, key=lambda x: x['total_time'])['num_workers']
    print(f"\nBest worker count: {best_worker_count}")
    print("Testing different batch sizes...")
    
    batch_results = []
    for batch_size in batch_sizes:
        start_time = time.time()
        
        train_loader, _, _ = prepare_dataloaders(
            accident_dir=accident_dir,
            non_accident_dir=non_accident_dir,
            batch_size=batch_size,
            num_frames=config['data']['num_frames'],
            frame_interval=config['data']['frame_interval'],
            train_ratio=0.3,
            val_ratio=0.0,
            num_workers=best_worker_count,
            balance_classes=False
        )
        
        # Time how long it takes to iterate through the data
        data_loading_time = 0
        batch_count = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for batch_idx, (videos, labels) in enumerate(train_loader):
            # Process up to 200MB of data
            if batch_idx * batch_size >= 50:
                break
                
            # Simulate sending to GPU
            batch_start = time.time()
            videos = videos.to(device)
            labels = labels.to(device)
            data_loading_time += time.time() - batch_start
            batch_count += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_batch_time = data_loading_time / batch_count if batch_count > 0 else 0
        
        batch_results.append({
            'test_type': 'batch_size',
            'num_workers': best_worker_count,
            'batch_size': batch_size,
            'pin_memory': True,
            'total_time': total_time,
            'avg_batch_time': avg_batch_time
        })
        
        print(f"Batch size: {batch_size}, Total time: {total_time:.2f}s, Avg batch time: {avg_batch_time:.4f}s")
    
    results.extend(batch_results)
    
    # Third test: pin_memory options with best worker count and batch size
    best_batch_size = min(batch_results, key=lambda x: x['avg_batch_time'])['batch_size']
    print(f"\nBest batch size: {best_batch_size}")
    print("Testing pin_memory options...")
    
    for pin_memory in pin_memory_options:
        start_time = time.time()
        
        # We need to manually create the dataset and data loader since the prepare_dataloaders function
        # doesn't let us control pin_memory
        accident_videos = []
        for root, _, files in os.walk(accident_dir):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mov')):
                    accident_videos.append(os.path.join(root, file))
        
        non_accident_videos = []
        for root, _, files in os.walk(non_accident_dir):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mov')):
                    non_accident_videos.append(os.path.join(root, file))
        
        all_videos = accident_videos[:50] + non_accident_videos[:50]  # Limit for testing
        all_labels = [1] * len(accident_videos[:50]) + [0] * len(non_accident_videos[:50])
        
        dataset = VideoDataset(
            all_videos, all_labels, config['data']['num_frames'], config['data']['frame_interval']
        )
        
        train_loader = DataLoader(
            dataset,
            batch_size=best_batch_size,
            shuffle=True,
            num_workers=best_worker_count,
            pin_memory=pin_memory
        )
        
        # Time how long it takes to iterate through the data
        data_loading_time = 0
        batch_count = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for batch_idx, (videos, labels) in enumerate(train_loader):
            if batch_idx >= 10:
                break
                
            # Simulate sending to GPU
            batch_start = time.time()
            videos = videos.to(device)
            labels = labels.to(device)
            data_loading_time += time.time() - batch_start
            batch_count += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_batch_time = data_loading_time / batch_count if batch_count > 0 else 0
        
        results.append({
            'test_type': 'pin_memory',
            'num_workers': best_worker_count,
            'batch_size': best_batch_size,
            'pin_memory': pin_memory,
            'total_time': total_time,
            'avg_batch_time': avg_batch_time
        })
        
        print(f"Pin memory: {pin_memory}, Total time: {total_time:.2f}s, Avg batch time: {avg_batch_time:.4f}s")
    
    # Print final recommendations
    worker_results = [r for r in results if r['test_type'] == 'workers']
    batch_results = [r for r in results if r['test_type'] == 'batch_size']
    pin_memory_results = [r for r in results if r['test_type'] == 'pin_memory']
    
    best_worker = min(worker_results, key=lambda x: x['total_time'])
    best_batch = min(batch_results, key=lambda x: x['avg_batch_time'])
    best_pin = min(pin_memory_results, key=lambda x: x['avg_batch_time'])
    
    print("\n--- RECOMMENDATIONS ---")
    print(f"Best num_workers: {best_worker['num_workers']}")
    print(f"Best batch_size: {best_batch['batch_size']}")
    print(f"Best pin_memory: {best_pin['pin_memory']}")
    
    print("\nUpdate your config.json with these values for optimal performance.")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot worker results
    plt.subplot(3, 1, 1)
    workers = [r['num_workers'] for r in worker_results]
    times = [r['total_time'] for r in worker_results]
    plt.bar(workers, times)
    plt.title('DataLoader Performance by Number of Workers')
    plt.xlabel('Number of Workers')
    plt.ylabel('Total Time (s)')
    plt.xticks(workers)
    
    # Plot batch size results
    plt.subplot(3, 1, 2)
    batch_sizes = [r['batch_size'] for r in batch_results]
    avg_times = [r['avg_batch_time'] for r in batch_results]
    plt.bar(batch_sizes, avg_times)
    plt.title('DataLoader Performance by Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Avg Batch Time (s)')
    plt.xticks(batch_sizes)
    
    # Plot pin_memory results
    plt.subplot(3, 1, 3)
    pin_options = [str(r['pin_memory']) for r in pin_memory_results]
    pin_times = [r['avg_batch_time'] for r in pin_memory_results]
    plt.bar(pin_options, pin_times)
    plt.title('DataLoader Performance by pin_memory Setting')
    plt.xlabel('pin_memory')
    plt.ylabel('Avg Batch Time (s)')
    
    plt.tight_layout()
    plt.savefig('dataloader_optimization.png')
    plt.close()
    
    print("\nResults chart saved to 'dataloader_optimization.png'")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize DataLoader settings')
    parser.add_argument('--config', type=str, default='./configs/config.json', 
                        help='Path to configuration file')
    args = parser.parse_args()
    
    test_dataloader_configs(args.config) 