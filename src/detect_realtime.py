#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import torch
import argparse
import numpy as np
import json
import time
from collections import deque
from PIL import Image
from torchvision import transforms

from model import get_model

class AccidentDetector:
    """Real-time accident detection system"""
    
    def __init__(self, config_path, model_path, threshold=0.5):
        """
        Initialize the accident detector
        
        Args:
            config_path (str): Path to configuration file
            model_path (str): Path to trained model
            threshold (float): Decision threshold for accident detection
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Initialize parameters
        self.frame_buffer = deque(maxlen=self.config['data']['num_frames'])
        self.feature_buffer = []
        self.num_frames = self.config['data']['num_frames']
        self.threshold = threshold
        self.config_path = config_path
        self.model_path = model_path
        
        # Transform for preprocessing frames
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_path):
        """
        Load the trained model
        
        Args:
            model_path (str): Path to model checkpoint
            
        Returns:
            torch.nn.Module: Loaded model
        """
        # Load model configuration from config
        model = get_model(self.config['model'])
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        
        print(f"Model loaded from {model_path}")
        return model
    
    def preprocess_frame(self, frame):
        """
        Preprocess a frame for model input
        
        Args:
            frame (numpy.ndarray): BGR frame from OpenCV
            
        Returns:
            torch.Tensor: Processed frame tensor
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # Apply transformations
        processed_frame = self.transform(pil_image)
        
        return processed_frame
    
    def process_frame(self, frame):
        """
        Process a single frame
        
        Args:
            frame (numpy.ndarray): BGR frame from OpenCV
            
        Returns:
            tuple: (processed_frame, prediction, prediction_time)
        """
        # Start prediction timing
        start_time = time.time()
        
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # Add to buffer
        self.frame_buffer.append(processed_frame)
        
        # Make prediction if buffer is full
        if len(self.frame_buffer) == self.num_frames:
            # Prepare input tensor
            frames_tensor = torch.stack(list(self.frame_buffer)).unsqueeze(0).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                output = self.model(frames_tensor)
                prediction = output.item()
        else:
            # Buffer not yet full
            prediction = 0.0
        
        # End prediction timing
        pred_time = time.time() - start_time
        
        # Draw on frame
        result_frame = self.draw_prediction(frame, prediction)
        
        return result_frame, prediction, pred_time
    
    def draw_prediction(self, frame, prediction):
        """
        Draw prediction information on the frame
        
        Args:
            frame (numpy.ndarray): Original BGR frame
            prediction (float): Model prediction (0-1)
            
        Returns:
            numpy.ndarray: Annotated frame
        """
        # Make a copy of the frame
        result = frame.copy()
        
        # Get frame dimensions
        height, width = result.shape[:2]
        
        # Draw prediction bar
        bar_height = 30
        bar_width = int(width * 0.6)
        bar_x = int((width - bar_width) / 2)
        bar_y = height - 50
        
        # Draw bar background
        cv2.rectangle(result, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), -1)
        
        # Draw prediction level
        pred_width = int(bar_width * prediction)
        
        # Color based on prediction level (green to yellow to red)
        if prediction < 0.3:
            color = (0, 255, 0)  # Green
        elif prediction < 0.6:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        cv2.rectangle(result, (bar_x, bar_y), (bar_x + pred_width, bar_y + bar_height), color, -1)
        
        # Draw bar border
        cv2.rectangle(result, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        
        # Add text
        cv2.putText(
            result, 
            f"Accident Risk: {prediction:.2f}", 
            (bar_x, bar_y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, 
            (255, 255, 255), 
            2
        )
        
        # Add warning if prediction exceeds threshold
        if prediction >= self.threshold:
            # Draw warning text
            cv2.putText(
                result,
                "ACCIDENT DETECTED",
                (int(width / 2 - 150), int(height / 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )
            
            # Draw red border
            border_thickness = 5
            cv2.rectangle(
                result,
                (border_thickness, border_thickness),
                (width - border_thickness, height - border_thickness),
                (0, 0, 255),
                border_thickness
            )
        
        return result
    
    def process_video(self, source, output_path=None):
        """
        Process video from source
        
        Args:
            source (str): Path to video file or camera index (0 for webcam)
            output_path (str, optional): Path to save output video
        """
        # Open video source
        if source == '0' or source == 'webcam':
            cap = cv2.VideoCapture(0)
        else:
            # For file inputs, standardize the video first
            if os.path.isfile(source) and source != '0':
                print(f"Standardizing input video: {source}")
                # Create temporary file for standardized video
                std_source = f"{os.path.splitext(source)[0]}_std.mp4"
                
                # Standardize the video
                if standardize_video(source, std_source, resolution=(480, 360), fps=25, bitrate=1500):
                    print(f"Using standardized video: {std_source}")
                    source = std_source
                else:
                    print(f"Standardization failed, using original video")
            
            cap = cv2.VideoCapture(source)
        
        # Check if video opened successfully
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize video writer if output path is specified
        video_writer = None
        if output_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Clear frame buffer
        self.frame_buffer.clear()
        
        # Start processing loop
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                result_frame, prediction, pred_time = self.process_frame(frame)
                
                # Save frame if writer is initialized
                if video_writer:
                    video_writer.write(result_frame)
                
                # Display result
                cv2.imshow('Accident Detection', result_frame)
                
                # Print prediction info
                print(f"Prediction: {prediction:.4f}, Time: {pred_time*1000:.1f}ms", end="\r")
                
                # Check for exit key (q or ESC)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 27 is ESC
                    break
        
        finally:
            # Clean up
            if cap:
                cap.release()
            if video_writer:
                video_writer.release()
                
            # Remove temporary standardized video if it was created
            if os.path.isfile(source) and '_std.mp4' in source and os.path.exists(source):
                try:
                    # Only remove it if it's a temporary standardized file
                    if source != '0' and source != 'webcam' and source.endswith('_std.mp4'):
                        os.remove(source)
                        print(f"Removed temporary standardized video: {source}")
                except Exception as e:
                    print(f"Error removing temporary file: {e}")
                
            cv2.destroyAllWindows()

def standardize_video(input_path, output_path, resolution=(480, 360), fps=25, bitrate=1500):
    """
    Standardize video dimensions, frame rate, codec and quality
    
    Args:
        input_path (str): Path to input video
        output_path (str): Path to save standardized video
        resolution (tuple): Width and height for output video
        fps (int): Frames per second for output video
        bitrate (int): Bitrate in kbps for output video
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # First check if ffmpeg is available
        try:
            import subprocess
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Build ffmpeg command
            width, height = resolution
            cmd = [
                'ffmpeg', '-y', '-i', input_path,
                '-c:v', 'libx264', '-preset', 'medium',
                '-b:v', f'{bitrate}k',
                '-vf', f'scale={width}:{height}',
                '-r', str(fps),
                '-pix_fmt', 'yuv420p',
                '-an',  # Remove audio
                output_path
            ]
            
            # Run ffmpeg
            process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Check if output file exists and has content
            if process.returncode != 0:
                print(f"ffmpeg error: {process.stderr.decode()}")
                return False
                
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                print(f"Output file is empty or doesn't exist: {output_path}")
                return False
                
            return True
            
        except (subprocess.SubprocessError, FileNotFoundError):
            print("ffmpeg not available, falling back to OpenCV")
            
            # Fall back to OpenCV if ffmpeg is not available
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                print(f"Could not open input video: {input_path}")
                return False
                
            # Get video properties
            width, height = resolution
            
            # Define codec
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
            
            # Create output video writer
            out = cv2.VideoWriter(
                output_path, fourcc, fps, (width, height)
            )
            
            if not out.isOpened():
                print(f"Could not create output video writer: {output_path}")
                cap.release()
                return False
            
            # Process each frame
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Resize frame
                resized_frame = cv2.resize(frame, (width, height))
                
                # Write frame
                out.write(resized_frame)
                frame_count += 1
            
            # Release resources
            cap.release()
            out.release()
            
            # Verify output
            if frame_count == 0:
                print(f"No frames were processed: {input_path}")
                return False
                
            return True
            
    except Exception as e:
        print(f"Error standardizing video {input_path}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description='Real-time accident detection')
    parser.add_argument('--config', type=str, default='configs/config.json', help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--source', type=str, default='webcam', help='Video source (file path or "webcam")')
    parser.add_argument('--output', type=str, help='Output video path')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    args = parser.parse_args()
    
    # Create detector
    detector = AccidentDetector(args.config, args.model, args.threshold)
    
    # Process video
    detector.process_video(args.source, args.output)

if __name__ == '__main__':
    main() 