#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import shutil
import json
from PIL import Image
import torch
import torchvision.transforms as transforms

def validate_video(video_path, num_frames=32, frame_interval=4, min_valid_frames=16):
    """
    Validate a video file by checking if it can be opened and frames can be read
    
    Args:
        video_path (str): Path to video file
        num_frames (int): Number of frames to check
        frame_interval (int): Interval between frames
        min_valid_frames (int): Minimum number of valid frames required
        
    Returns:
        tuple: (is_valid, issue_description, valid_frame_count)
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Could not open video", 0
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return False, "Zero frame count", 0
        
        # Check if we can extract frames
        valid_frames = 0
        total_check = min(frame_count, num_frames)
        indices = np.linspace(0, frame_count - 1, total_check, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                valid_frames += 1
        
        cap.release()
        
        # Validate results
        if valid_frames < min_valid_frames:
            return False, f"Only {valid_frames}/{total_check} frames could be read", valid_frames
            
        return True, "Valid", valid_frames
    
    except Exception as e:
        return False, f"Exception: {str(e)}", 0

def extract_keyframes(video_path, output_path, num_frames=32):
    """
    Extract keyframes from a video and save them as a new video
    
    Args:
        video_path (str): Path to source video
        output_path (str): Path to save fixed video
        num_frames (int): Number of frames to extract
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
            
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Extract evenly spaced frames
        indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        valid_frames = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                valid_frames.append(frame)
                out.write(frame)
        
        # If we didn't get enough frames, duplicate the last one
        while len(valid_frames) < num_frames and len(valid_frames) > 0:
            out.write(valid_frames[-1])
        
        # Release resources
        cap.release()
        out.release()
        
        return len(valid_frames) > 0
    
    except Exception as e:
        print(f"Error extracting keyframes from {video_path}: {e}")
        return False

def process_video_directory(src_dir, valid_dir, invalid_dir, fixed_dir, num_frames=32):
    """
    Process all videos in a directory and organize them
    
    Args:
        src_dir (str): Source directory with videos
        valid_dir (str): Directory to store valid videos
        invalid_dir (str): Directory to store invalid videos
        fixed_dir (str): Directory to store fixed videos
        num_frames (int): Number of frames to check
        
    Returns:
        dict: Statistics about processing
    """
    # Create directories if they don't exist
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(invalid_dir, exist_ok=True)
    os.makedirs(fixed_dir, exist_ok=True)
    
    videos = [f for f in os.listdir(src_dir) 
              if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    stats = {
        "total": len(videos),
        "valid": 0,
        "invalid": 0,
        "fixed": 0,
        "unfixable": 0
    }
    
    results = []
    
    for video in tqdm(videos, desc=f"Processing {os.path.basename(src_dir)}"):
        video_path = os.path.join(src_dir, video)
        is_valid, issue, valid_frames = validate_video(video_path, num_frames=num_frames)
        
        result = {
            "filename": video,
            "is_valid": is_valid,
            "issue": issue,
            "valid_frames": valid_frames
        }
        
        if is_valid:
            # Copy to valid directory
            shutil.copy2(video_path, os.path.join(valid_dir, video))
            stats["valid"] += 1
        else:
            # Copy to invalid directory
            shutil.copy2(video_path, os.path.join(invalid_dir, video))
            stats["invalid"] += 1
            
            # Try to fix the video
            fixed_path = os.path.join(fixed_dir, f"fixed_{video}")
            if extract_keyframes(video_path, fixed_path, num_frames):
                result["fixed"] = True
                stats["fixed"] += 1
            else:
                result["fixed"] = False
                stats["unfixable"] += 1
        
        results.append(result)
    
    return stats, results

def main():
    parser = argparse.ArgumentParser(description="Validate and organize videos for accident detection")
    parser.add_argument("--accident_dir", type=str, default="data/accidents", 
                      help="Directory with accident videos")
    parser.add_argument("--non_accident_dir", type=str, default="data/non_accidents", 
                      help="Directory with non-accident videos")
    parser.add_argument("--output_dir", type=str, default="data/processed", 
                      help="Output directory for processed videos")
    parser.add_argument("--num_frames", type=int, default=32, 
                      help="Number of frames to check per video")
    parser.add_argument("--min_valid_frames", type=int, default=16, 
                      help="Minimum number of valid frames for a valid video")
    args = parser.parse_args()
    
    # Create output directory structure
    os.makedirs(args.output_dir, exist_ok=True)
    
    valid_accident_dir = os.path.join(args.output_dir, "valid_accidents")
    invalid_accident_dir = os.path.join(args.output_dir, "invalid_accidents")
    fixed_accident_dir = os.path.join(args.output_dir, "fixed_accidents")
    
    valid_non_accident_dir = os.path.join(args.output_dir, "valid_non_accidents")
    invalid_non_accident_dir = os.path.join(args.output_dir, "invalid_non_accidents")
    fixed_non_accident_dir = os.path.join(args.output_dir, "fixed_non_accidents")
    
    # Process accident videos
    print("\nProcessing accident videos...")
    accident_stats, accident_results = process_video_directory(
        args.accident_dir, 
        valid_accident_dir, 
        invalid_accident_dir, 
        fixed_accident_dir,
        args.num_frames
    )
    
    # Process non-accident videos
    print("\nProcessing non-accident videos...")
    non_accident_stats, non_accident_results = process_video_directory(
        args.non_accident_dir, 
        valid_non_accident_dir, 
        invalid_non_accident_dir, 
        fixed_non_accident_dir,
        args.num_frames
    )
    
    # Combine results for training
    print("\nPreparing training data...")
    train_accident_dir = os.path.join(args.output_dir, "train_accidents")
    train_non_accident_dir = os.path.join(args.output_dir, "train_non_accidents")
    
    os.makedirs(train_accident_dir, exist_ok=True)
    os.makedirs(train_non_accident_dir, exist_ok=True)
    
    # Copy valid and fixed accident videos to train directory
    for video in os.listdir(valid_accident_dir):
        shutil.copy2(os.path.join(valid_accident_dir, video), 
                    os.path.join(train_accident_dir, video))
    
    for video in os.listdir(fixed_accident_dir):
        # Only copy if the fix was successful (file exists and has content)
        video_path = os.path.join(fixed_accident_dir, video)
        if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
            shutil.copy2(video_path, os.path.join(train_accident_dir, video))
    
    # Copy valid and fixed non-accident videos to train directory
    for video in os.listdir(valid_non_accident_dir):
        shutil.copy2(os.path.join(valid_non_accident_dir, video), 
                    os.path.join(train_non_accident_dir, video))
    
    for video in os.listdir(fixed_non_accident_dir):
        # Only copy if the fix was successful
        video_path = os.path.join(fixed_non_accident_dir, video)
        if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
            shutil.copy2(video_path, os.path.join(train_non_accident_dir, video))
    
    # Save statistics
    all_stats = {
        "accident": accident_stats,
        "non_accident": non_accident_stats,
        "training_data": {
            "accidents": len(os.listdir(train_accident_dir)),
            "non_accidents": len(os.listdir(train_non_accident_dir))
        }
    }
    
    with open(os.path.join(args.output_dir, "validation_stats.json"), 'w') as f:
        json.dump(all_stats, f, indent=4)
    
    # Save detailed results
    all_results = {
        "accident": accident_results,
        "non_accident": non_accident_results
    }
    
    with open(os.path.join(args.output_dir, "validation_results.json"), 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # Print summary
    print("\n=== Video Validation Summary ===")
    print(f"Accident videos: {accident_stats['total']} total, {accident_stats['valid']} valid, {accident_stats['fixed']} fixed, {accident_stats['unfixable']} unfixable")
    print(f"Non-accident videos: {non_accident_stats['total']} total, {non_accident_stats['valid']} valid, {non_accident_stats['fixed']} fixed, {non_accident_stats['unfixable']} unfixable")
    print(f"\nTraining data prepared: {all_stats['training_data']['accidents']} accident videos and {all_stats['training_data']['non_accidents']} non-accident videos")
    print(f"\nTo use the validated data for training, update your config.json with:")
    print(f'"accident_dir": "{os.path.abspath(train_accident_dir)}",')
    print(f'"non_accident_dir": "{os.path.abspath(train_non_accident_dir)}"')

if __name__ == "__main__":
    main() 