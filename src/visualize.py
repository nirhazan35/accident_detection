#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
from PIL import Image
from torchvision import transforms
from sklearn.manifold import TSNE
import seaborn as sns
from tqdm import tqdm
import torch.nn.functional as F

from model import get_model

class ModelVisualizer:
    """Visualize model attention and predictions for a video"""
    
    def __init__(self, config_path, model_path, device=None):
        """
        Initialize visualizer
        
        Args:
            config_path (str): Path to configuration file
            model_path (str): Path to trained model weights
            device (torch.device, optional): Device to run model on
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Extract model components
        self.feature_extractor = self.model.feature_extractor if hasattr(self.model, 'feature_extractor') else None
        
        # Set parameters
        self.num_frames = self.config['data'].get('num_frames', 32)
        
        # Define transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_path):
        """Load trained model"""
        model = get_model(self.config['model'])
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model
    
    def _extract_frames(self, video_path):
        """
        Extract frames from video
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            tuple: (original_frames, processed_frames, frame_indices)
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames evenly
        frame_indices = np.linspace(0, frame_count - 1, self.num_frames, dtype=int)
        
        # Extract frames
        original_frames = []
        processed_frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                # If frame read failed, use previous frame or black frame
                if original_frames:
                    frame = original_frames[-1]
                else:
                    frame = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # Store original frame
            original_frames.append(frame)
            
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            processed_frame = self.transform(pil_img)
            processed_frames.append(processed_frame)
        
        cap.release()
        
        return original_frames, processed_frames, frame_indices
    
    def _get_attention_weights(self, processed_frames):
        """
        Extract attention weights from model
        
        Args:
            processed_frames (list): List of processed frames
            
        Returns:
            tuple: (prediction, attention_weights)
        """
        frames_tensor = torch.stack(processed_frames).unsqueeze(0).to(self.device)
        
        # Forward pass to extract attention
        self.model.eval()
        
        # Register hooks to capture attention
        attention_weights = []
        
        def get_attention_hook(name):
            def hook(module, input, output):
                # For transformer attention
                if isinstance(output, tuple) and len(output) > 1:
                    # Some transformer implementations return attention weights
                    attention_weights.append(output[1].detach().cpu())
                
            return hook
        
        # Try to register hooks at common attention locations
        hooks = []
        if hasattr(self.model, 'transformer_encoder'):
            for i, layer in enumerate(self.model.transformer_encoder.layers):
                if hasattr(layer, 'self_attn'):
                    hooks.append(layer.self_attn.register_forward_hook(
                        get_attention_hook(f'transformer_layer_{i}')))
        
        # Forward pass
        with torch.no_grad():
            output = self.model(frames_tensor)
            prediction = output.item()
            
            # If no attention weights were captured via hooks, compute manually
            if not attention_weights and hasattr(self.model, 'transformer_encoder'):
                # Extract LSTM output
                if hasattr(self.model, 'lstm'):
                    features = frames_tensor
                    if hasattr(self.model, 'feature_extractor') and not self.model.use_pretrained_features:
                        # Reshape for CNN processing
                        b, t, c, h, w = frames_tensor.shape
                        features = frames_tensor.view(b * t, c, h, w)
                        features = self.model.feature_extractor(features)
                        features = features.view(b, t, -1)
                    
                    lstm_out, _ = self.model.lstm(features)
                    
                    # Compute self-attention weights manually
                    sim = torch.matmul(lstm_out, lstm_out.transpose(-2, -1))
                    attn_weights = F.softmax(sim / np.sqrt(lstm_out.size(-1)), dim=-1)
                    attention_weights.append(attn_weights.cpu())
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Process attention weights
        if attention_weights:
            # Use the last layer's attention weights
            attn = attention_weights[-1].squeeze(0)
        else:
            # If no attention weights were captured, use identity matrix
            attn = torch.eye(self.num_frames)
        
        return prediction, attn
    
    def visualize_attention(self, video_path, output_dir, save_video=True):
        """
        Visualize attention for a video
        
        Args:
            video_path (str): Path to video file
            output_dir (str): Directory to save visualizations
            save_video (bool): Whether to save the attention visualization as a video
            
        Returns:
            float: Model prediction (0-1)
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract frames
        print(f"Extracting frames from {video_path}...")
        original_frames, processed_frames, frame_indices = self._extract_frames(video_path)
        
        # Get prediction and attention weights
        print("Computing attention weights...")
        prediction, attention_weights = self._get_attention_weights(processed_frames)
        
        # Plot attention matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_weights.numpy(), cmap='viridis')
        plt.title(f'Attention Matrix (Prediction: {prediction:.3f})')
        plt.xlabel('Frame (Target)')
        plt.ylabel('Frame (Query)')
        plt.savefig(os.path.join(output_dir, 'attention_matrix.png'))
        plt.close()
        
        # Plot frame importance (attention sum for each frame)
        frame_importance = attention_weights.sum(dim=0).numpy()
        
        plt.figure(figsize=(12, 4))
        plt.bar(range(len(frame_importance)), frame_importance)
        plt.title(f'Frame Importance (Prediction: {prediction:.3f})')
        plt.xlabel('Frame Index')
        plt.ylabel('Attention Sum')
        plt.savefig(os.path.join(output_dir, 'frame_importance.png'))
        plt.close()
        
        # Visualize attention on original frames
        print("Creating frame visualizations...")
        
        # Normalize frame importance for visualization
        normalized_importance = (frame_importance - frame_importance.min()) / (frame_importance.max() - frame_importance.min() + 1e-8)
        
        # Create directory for frame visualizations
        frames_dir = os.path.join(output_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        # Process each frame with attention overlay
        attention_frames = []
        
        for i, (frame, importance) in enumerate(zip(original_frames, normalized_importance)):
            # Create a copy of the frame
            viz_frame = frame.copy()
            
            # Add attention overlay
            overlay = np.ones_like(viz_frame) * np.array([255, 0, 0], dtype=np.uint8)  # Red overlay
            overlay_opacity = importance * 0.5  # Scale importance to 0-0.5 range
            
            viz_frame = cv2.addWeighted(viz_frame, 1.0, overlay, overlay_opacity, 0)
            
            # Add frame index and importance
            cv2.putText(
                viz_frame,
                f"Frame {i} (Imp: {importance:.3f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Save individual frame
            cv2.imwrite(os.path.join(frames_dir, f'frame_{i:03d}.png'), viz_frame)
            
            # Add to list for video
            attention_frames.append(viz_frame)
        
        # Save visualization video
        if save_video:
            print("Creating attention visualization video...")
            video_path = os.path.join(output_dir, 'attention_visualization.mp4')
            
            # Get frame dimensions
            height, width = attention_frames[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, 2, (width, height))
            
            # Write frames
            for frame in attention_frames:
                video_writer.write(frame)
            
            video_writer.release()
        
        print(f"Visualization completed. Results saved to {output_dir}")
        print(f"Model prediction: {prediction:.4f}")
        
        return prediction
    
    def visualize_feature_space(self, video_paths, labels, output_dir):
        """
        Visualize the feature space learned by the model using t-SNE
        
        Args:
            video_paths (list): List of video paths
            labels (list): List of labels (1 for accident, 0 for non-accident)
            output_dir (str): Directory to save visualizations
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Collect features
        features_list = []
        predictions_list = []
        
        for video_path in tqdm(video_paths, desc="Extracting features"):
            # Extract frames
            _, processed_frames, _ = self._extract_frames(video_path)
            
            # Create input tensor
            frames_tensor = torch.stack(processed_frames).unsqueeze(0).to(self.device)
            
            # Extract features and make prediction
            with torch.no_grad():
                # Extract CNN features if using raw frames
                if not self.model.use_pretrained_features and hasattr(self.model, 'feature_extractor'):
                    # Reshape for CNN processing
                    b, t, c, h, w = frames_tensor.shape
                    x = frames_tensor.view(b * t, c, h, w)
                    
                    # Extract features
                    features = self.model.feature_extractor(x)
                    features = features.view(b, t, -1).mean(dim=1).cpu().numpy()
                else:
                    # Use raw frames as features
                    features = frames_tensor.mean(dim=1).cpu().numpy()
                
                # Make prediction
                output = self.model(frames_tensor)
                prediction = output.item()
            
            features_list.append(features.squeeze())
            predictions_list.append(prediction)
        
        # Convert to numpy arrays
        features_array = np.array(features_list)
        predictions_array = np.array(predictions_list)
        labels_array = np.array(labels)
        
        # Perform t-SNE
        print("Performing t-SNE dimensionality reduction...")
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features_array)
        
        # Plot t-SNE
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot colored by true labels
        scatter = plt.scatter(
            features_2d[:, 0],
            features_2d[:, 1],
            c=labels_array,
            s=50,
            alpha=0.8,
            cmap='coolwarm'
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('True Label (1=Accident, 0=Non-Accident)')
        
        plt.title('t-SNE Visualization of Video Features')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, 'tsne_true_labels.png'))
        plt.close()
        
        # Plot t-SNE colored by predictions
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot colored by predictions
        scatter = plt.scatter(
            features_2d[:, 0],
            features_2d[:, 1],
            c=predictions_array,
            s=50,
            alpha=0.8,
            cmap='coolwarm'
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Model Prediction (0-1)')
        
        plt.title('t-SNE Visualization of Video Features with Model Predictions')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, 'tsne_predictions.png'))
        plt.close()
        
        print(f"t-SNE visualization completed. Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Visualize model attention and predictions')
    parser.add_argument('--config', type=str, default='configs/config.json', help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--output', type=str, default='visualization_results', help='Directory to save visualizations')
    parser.add_argument('--feature_space', action='store_true', help='Visualize feature space (requires accident and non-accident directories)')
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ModelVisualizer(args.config, args.model)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process video if provided
    if args.video:
        video_output_dir = os.path.join(args.output, os.path.basename(args.video).split('.')[0])
        visualizer.visualize_attention(args.video, video_output_dir)
    
    # Visualize feature space if requested
    if args.feature_space:
        # Load configuration
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # Get accident and non-accident directories
        accident_dir = config['data']['accident_dir']
        non_accident_dir = config['data']['non_accident_dir']
        
        # Check if directories exist
        if not os.path.exists(accident_dir) or not os.path.exists(non_accident_dir):
            print(f"Error: Accident or non-accident directory not found. Please check the configuration.")
            return
        
        # Get video paths and labels
        accident_videos = [os.path.join(accident_dir, f) for f in os.listdir(accident_dir) 
                          if f.endswith(('.mp4', '.avi', '.mov'))]
        non_accident_videos = [os.path.join(non_accident_dir, f) for f in os.listdir(non_accident_dir) 
                              if f.endswith(('.mp4', '.avi', '.mov'))]
        
        # Limit to a reasonable number of videos for visualization
        max_videos = 100
        accident_videos = accident_videos[:max_videos//2]
        non_accident_videos = non_accident_videos[:max_videos//2]
        
        video_paths = accident_videos + non_accident_videos
        labels = [1] * len(accident_videos) + [0] * len(non_accident_videos)
        
        # Visualize feature space
        feature_space_dir = os.path.join(args.output, 'feature_space')
        visualizer.visualize_feature_space(video_paths, labels, feature_space_dir)

if __name__ == '__main__':
    main() 