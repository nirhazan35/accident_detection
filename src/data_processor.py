import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import random
from tqdm import tqdm
import torchvision.transforms as transforms

class VideoDataset(Dataset):
    """Dataset for loading video frames for accident detection"""
    
    def __init__(self, video_paths, labels, num_frames=32, frame_interval=4, transform=None):
        """
        Initialize VideoDataset
        
        Args:
            video_paths (list): List of paths to video files
            labels (list): List of labels (1 for accident, 0 for non-accident)
            num_frames (int): Number of frames to extract from each video
            frame_interval (int): Interval between frames to extract
            transform (callable, optional): Optional transform to be applied on frames
        """
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video frames
        frames = self._load_video(video_path)
        
        # Convert label to tensor
        label_tensor = torch.tensor(label, dtype=torch.float)
        
        return frames, label_tensor
    
    def _load_video(self, video_path):
        """
        Load video and extract frames
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            torch.Tensor: Tensor of shape (num_frames, C, H, W)
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame indices to extract
        if frame_count <= self.num_frames * self.frame_interval:
            # If video has fewer frames than required, duplicate frames
            indices = np.linspace(0, frame_count - 1, self.num_frames, dtype=int)
        else:
            # Randomly select a starting point and extract consecutive frames
            max_start_idx = frame_count - self.num_frames * self.frame_interval
            start_idx = random.randint(0, max_start_idx)
            indices = np.array([start_idx + i * self.frame_interval for i in range(self.num_frames)])
        
        # Extract frames
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                # If frame read failed, create a black frame
                frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image and apply transform
            pil_img = Image.fromarray(frame)
            transformed_img = self.transform(pil_img)
            
            frames.append(transformed_img)
        
        cap.release()
        
        # Stack frames into a tensor
        frames_tensor = torch.stack(frames)
        
        return frames_tensor

def get_transforms(phase):
    if phase == 'train':
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

def prepare_dataloaders(accident_dir, non_accident_dir, batch_size=8, num_frames=32, 
                         frame_interval=4, num_workers=4, train_ratio=0.7, val_ratio=0.15):
    """
    Prepare train, validation, and test dataloaders
    
    Args:
        accident_dir (str): Directory containing accident videos
        non_accident_dir (str): Directory containing non-accident videos
        batch_size (int): Batch size
        num_frames (int): Number of frames to extract from each video
        frame_interval (int): Interval between frames
        num_workers (int): Number of workers for data loading
        train_ratio (float): Proportion of data for training
        val_ratio (float): Proportion of data for validation
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Get all video paths and labels
    video_paths = []
    labels = []
    
    # Process accident videos (label 1)
    accident_videos = [os.path.join(accident_dir, f) for f in os.listdir(accident_dir) 
                      if f.endswith(('.mp4', '.avi', '.mov'))]
    video_paths.extend(accident_videos)
    labels.extend([1] * len(accident_videos))
    
    # Process non-accident videos (label 0)
    non_accident_videos = [os.path.join(non_accident_dir, f) for f in os.listdir(non_accident_dir) 
                          if f.endswith(('.mp4', '.avi', '.mov'))]
    video_paths.extend(non_accident_videos)
    labels.extend([0] * len(non_accident_videos))
    
    # Create dataset
    dataset = VideoDataset(
        video_paths=video_paths,
        labels=labels,
        num_frames=num_frames,
        frame_interval=frame_interval
    )
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create dataloaders
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

def extract_video_features(video_path, feature_extractor, device, num_frames=32, frame_interval=4):
    """
    Extract features from a video using the feature extractor
    
    Args:
        video_path (str): Path to video file
        feature_extractor (nn.Module): Feature extractor model
        device (torch.device): Device to run inference on
        num_frames (int): Number of frames to extract
        frame_interval (int): Interval between frames
        
    Returns:
        torch.Tensor: Extracted features of shape (num_frames, feature_dim)
    """
    # Set up transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to extract
    if frame_count <= num_frames * frame_interval:
        indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    else:
        max_start_idx = frame_count - num_frames * frame_interval
        start_idx = random.randint(0, max_start_idx)
        indices = np.array([start_idx + i * frame_interval for i in range(num_frames)])
    
    # Extract frames
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            # If frame read failed, create a black frame
            frame_height, frame_width = 224, 224
            frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image and apply transform
        pil_img = Image.fromarray(frame)
        transformed_img = transform(pil_img)
        
        frames.append(transformed_img)
    
    cap.release()
    
    # Stack frames into a tensor and move to device
    frames_tensor = torch.stack(frames).to(device)
    
    # Extract features
    with torch.no_grad():
        feature_extractor.eval()
        features = feature_extractor(frames_tensor.unsqueeze(0))
        features = features.squeeze(0)
    
    return features 