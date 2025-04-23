import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import random
from tqdm import tqdm
import math
from torchvision import transforms
from PIL import Image

class VideoDataset(Dataset):
    """
    Dataset for loading video frames from videos 
    in the accidents and non_accidents folders
    """
    def __init__(self, video_paths, labels, num_frames=32, frame_interval=4, transform=None, augment=False):
        """
        Args:
            video_paths (list): List of video file paths
            labels (list): List of labels (1 for accident, 0 for non-accident)
            num_frames (int): Number of frames to extract from each video
            frame_interval (int): Interval between frames
            transform (callable, optional): Optional transform to be applied on frames
            augment (bool): Whether to use data augmentation
        """
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.transform = transform
        self.augment = augment
        
        # Define augmentation transforms
        self.aug_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Extract frames from video
        frames = extract_frames(
            video_path, 
            self.num_frames, 
            self.frame_interval
        )
        
        # Apply augmentation if enabled
        if self.augment and random.random() > 0.5:  # 50% chance of augmentation
            aug_frames = []
            for frame in frames:
                # Convert from [C, H, W] to [H, W, C] for augmentation
                frame_hwc = np.transpose(frame, (1, 2, 0))
                # Apply augmentations
                aug_frame = self.aug_transforms(frame_hwc * 255.0).numpy()  # Scale back to 0-255 for ToPILImage
                aug_frames.append(aug_frame)
            frames = np.array(aug_frames)
        
        # Convert to tensor [num_frames, C, H, W]
        frames_tensor = torch.FloatTensor(frames)
        
        return frames_tensor, label

def extract_frames(video_path, num_frames=32, frame_interval=4):
    """
    Extract frames from a video file
    
    Args:
        video_path (str): Path to the video file
        num_frames (int): Number of frames to extract
        frame_interval (int): Interval between frames
    
    Returns:
        list: List of frames as numpy arrays
    """
    frames = []
    
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            # Return empty frames array with the correct dimensions
            return np.zeros((num_frames, 3, 224, 224), dtype=np.float32)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to extract - adjusted to use whatever frames are available
        if frame_count <= num_frames:
            # If we have fewer frames than requested, use all available frames
            indices = np.arange(frame_count)
        else:
            # Otherwise sample evenly across the video
            indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        
        # Extract frames at calculated indices
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Error reading frame {i} from {video_path}")
                frames.append(np.zeros((3, 224, 224), dtype=np.float32))
                continue
            
            # Resize frame to 224x224
            frame = cv2.resize(frame, (224, 224))
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            
            # Convert to [C, H, W] format
            frame = np.transpose(frame, (2, 0, 1))
            
            frames.append(frame)
        
        cap.release()
        
        # If we couldn't extract enough frames, pad with zeros
        if len(frames) < num_frames:
            padding = num_frames - len(frames)
            for _ in range(padding):
                frames.append(np.zeros((3, 224, 224), dtype=np.float32))
        
        return np.array(frames)
    
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return np.zeros((num_frames, 3, 224, 224), dtype=np.float32)

def prepare_dataloaders(accident_dir, non_accident_dir, batch_size=8, 
                        num_frames=32, frame_interval=4, 
                        train_ratio=0.7, val_ratio=0.3, 
                        num_workers=4, balance_classes=True):
    """
    Prepare data loaders for training, validation, and testing
    
    Args:
        accident_dir (str): Directory containing accident videos
        non_accident_dir (str): Directory containing non-accident videos
        batch_size (int): Batch size
        num_frames (int): Number of frames to extract per video
        frame_interval (int): Interval between frames
        train_ratio (float): Ratio of data for training
        val_ratio (float): Ratio of data for validation
        num_workers (int): Number of workers for data loading
        balance_classes (bool): Whether to use class balancing
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    print("Loading video paths...")
    
    # Get all video paths and labels
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
    
    num_accident = len(accident_videos)
    num_non_accident = len(non_accident_videos)
    
    print(f"Found {num_accident} accident videos")
    print(f"Found {num_non_accident} non-accident videos")
    
    # Split each class separately to maintain class distribution in splits
    # Accident videos
    random.shuffle(accident_videos)
    accident_train_size = int(num_accident * train_ratio)
    accident_val_size = int(num_accident * val_ratio)
    
    accident_train = accident_videos[:accident_train_size]
    accident_val = accident_videos[accident_train_size:accident_train_size + accident_val_size]
    accident_test = accident_videos[accident_train_size + accident_val_size:]
    
    # Non-accident videos
    random.shuffle(non_accident_videos)
    non_accident_train_size = int(num_non_accident * train_ratio)
    non_accident_val_size = int(num_non_accident * val_ratio)
    
    non_accident_train = non_accident_videos[:non_accident_train_size]
    non_accident_val = non_accident_videos[non_accident_train_size:non_accident_train_size + non_accident_val_size]
    non_accident_test = non_accident_videos[non_accident_train_size + non_accident_val_size:]
    
    # Combine train, val, test sets
    train_videos = accident_train + non_accident_train
    train_labels = [1] * len(accident_train) + [0] * len(non_accident_train)
    
    val_videos = accident_val + non_accident_val
    val_labels = [1] * len(accident_val) + [0] * len(non_accident_val)
    
    test_videos = accident_test + non_accident_test
    test_labels = [1] * len(accident_test) + [0] * len(non_accident_test)
    
    # Shuffle
    train_indices = list(range(len(train_videos)))
    random.shuffle(train_indices)
    train_videos = [train_videos[i] for i in train_indices]
    train_labels = [train_labels[i] for i in train_indices]
    
    val_indices = list(range(len(val_videos)))
    random.shuffle(val_indices)
    val_videos = [val_videos[i] for i in val_indices]
    val_labels = [val_labels[i] for i in val_indices]
    
    print(f"Train set: {len(train_videos)} videos ({train_labels.count(1)} accident, {train_labels.count(0)} non-accident)")
    print(f"Validation set: {len(val_videos)} videos ({val_labels.count(1)} accident, {val_labels.count(0)} non-accident)")
    print(f"Test set: {len(test_videos)} videos ({test_labels.count(1)} accident, {test_labels.count(0)} non-accident)")
    
    # Create datasets with augmentation for training
    train_dataset = VideoDataset(
        train_videos, train_labels, num_frames, frame_interval, augment=True
    )
    
    val_dataset = VideoDataset(
        val_videos, val_labels, num_frames, frame_interval, augment=False
    )
    
    test_dataset = VideoDataset(
        test_videos, test_labels, num_frames, frame_interval, augment=False
    )
    
    # Create weighted sampler for training to address class imbalance
    if balance_classes:
        # Calculate class weights
        class_counts = [train_labels.count(0), train_labels.count(1)]
        weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        
        # Assign weight to each sample
        sample_weights = [weights[label] for label in train_labels]
        
        # Create sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_labels),
            replacement=True
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=sampler,  # Use weighted sampler
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader 