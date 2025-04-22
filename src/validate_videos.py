#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import shutil
import json
import logging
from PIL import Image
import torch
import torchvision.transforms as transforms
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('video_validation.log')
    ]
)
logger = logging.getLogger(__name__)

def get_available_codec():
    """Find an available codec for video writing"""
    codecs = ['mp4v', 'avc1', 'XVID', 'MJPG']
    for codec in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            # Test if codec works by creating a small test file
            test_file = "test_codec.mp4"
            writer = cv2.VideoWriter(
                test_file, fourcc, 30.0, (640, 480), isColor=True
            )
            if writer.isOpened():
                writer.release()
                if os.path.exists(test_file):
                    os.remove(test_file)
                return codec
        except Exception:
            continue
    
    # Fallback to raw frames if no codec works
    logger.warning("No suitable video codec found. Will save individual frames.")
    return None

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
        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            return False, "File does not exist or is empty", 0
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Could not open video", 0
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if frame_count <= 0:
            cap.release()
            return False, "Zero frame count", 0
            
        if frame_width <= 0 or frame_height <= 0:
            cap.release()
            return False, f"Invalid dimensions: {frame_width}x{frame_height}", 0
        
        # Check if we can extract frames
        valid_frames = 0
        total_check = min(frame_count, num_frames)
        indices = np.linspace(0, frame_count - 1, total_check, dtype=int)
        
        # Also check frame variance to detect corruption where frames are readable but corrupted
        frame_sizes = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                valid_frames += 1
                # Check frame size/complexity as a basic quality measure
                frame_size = sys.getsizeof(frame.tobytes())
                frame_sizes.append(frame_size)
        
        cap.release()
        
        # Check frame size variance (if we have enough frames)
        if len(frame_sizes) >= 3:
            variance = np.var(frame_sizes)
            mean_size = np.mean(frame_sizes)
            if variance < 0.01 * mean_size:  # Very low variance could indicate duplicated frames
                return False, f"Low frame variance, possibly corrupted: {variance}/{mean_size}", valid_frames
        
        # Validate results
        if valid_frames < min_valid_frames:
            return False, f"Only {valid_frames}/{total_check} frames could be read", valid_frames
            
        return True, "Valid", valid_frames
    
    except Exception as e:
        logger.error(f"Error validating {video_path}: {str(e)}")
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
            logger.error(f"Could not open video: {video_path}")
            return False
            
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Ensure valid dimensions
        if width <= 0 or height <= 0:
            logger.error(f"Invalid dimensions {width}x{height} for {video_path}")
            cap.release()
            return False
            
        # Ensure valid framerate
        if fps <= 0:
            fps = 30.0  # Default to 30 fps if invalid
        
        # Find available codec
        codec = get_available_codec()
        valid_frames = []
        
        # Extract evenly spaced frames
        indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        
        # First pass: collect valid frames
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                valid_frames.append(frame)
        
        if len(valid_frames) == 0:
            logger.error(f"No valid frames extracted from {video_path}")
            cap.release()
            return False
        
        # If we have fewer frames than requested, duplicate the last one
        while len(valid_frames) < num_frames:
            valid_frames.append(valid_frames[-1].copy())
        
        # Write frames to output
        if codec:
            # Use video codec
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in valid_frames:
                out.write(frame)
            
            out.release()
        else:
            # Fallback: save as image sequence
            frames_dir = output_path + "_frames"
            os.makedirs(frames_dir, exist_ok=True)
            
            for i, frame in enumerate(valid_frames):
                frame_path = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
                cv2.imwrite(frame_path, frame)
            
            # Create a metadata file
            with open(os.path.join(frames_dir, "metadata.json"), 'w') as f:
                json.dump({
                    "original_file": video_path,
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "frame_count": len(valid_frames)
                }, f, indent=4)
            
            # Set the output_path to the frames directory
            with open(output_path, 'w') as f:
                f.write(frames_dir)
        
        cap.release()
        
        # Validate the output
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            is_valid, _, _ = validate_video(output_path, num_frames=min(5, num_frames), min_valid_frames=min(3, num_frames))
            if not is_valid:
                logger.warning(f"Generated video failed validation: {output_path}")
                return False
            return True
        else:
            logger.error(f"Failed to create output video {output_path}")
            return False
        
    except Exception as e:
        logger.error(f"Error extracting keyframes from {video_path}: {str(e)}")
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
    # Check if source directory exists
    if not os.path.exists(src_dir):
        logger.error(f"Source directory does not exist: {src_dir}")
        return {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "fixed": 0,
            "unfixable": 0
        }, []
    
    # Create directories if they don't exist
    for dir_path in [valid_dir, invalid_dir, fixed_dir]:
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory {dir_path}: {str(e)}")
            return {
                "total": 0,
                "valid": 0,
                "invalid": 0,
                "fixed": 0,
                "unfixable": 0
            }, []
    
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
        
        try:
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
        except Exception as e:
            logger.error(f"Error processing {video_path}: {str(e)}")
            result["error"] = str(e)
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
    parser.add_argument("--log_file", type=str, default="video_validation.log",
                      help="Log file path")
    args = parser.parse_args()
    
    # Update log file
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)
    
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting video validation with arguments: {args}")
    
    # Create output directory structure
    os.makedirs(args.output_dir, exist_ok=True)
    
    valid_accident_dir = os.path.join(args.output_dir, "valid_accidents")
    invalid_accident_dir = os.path.join(args.output_dir, "invalid_accidents")
    fixed_accident_dir = os.path.join(args.output_dir, "fixed_accidents")
    
    valid_non_accident_dir = os.path.join(args.output_dir, "valid_non_accidents")
    invalid_non_accident_dir = os.path.join(args.output_dir, "invalid_non_accidents")
    fixed_non_accident_dir = os.path.join(args.output_dir, "fixed_non_accidents")
    
    # Process accident videos
    logger.info("\nProcessing accident videos...")
    accident_stats, accident_results = process_video_directory(
        args.accident_dir, 
        valid_accident_dir, 
        invalid_accident_dir, 
        fixed_accident_dir,
        args.num_frames
    )
    
    # Process non-accident videos
    logger.info("\nProcessing non-accident videos...")
    non_accident_stats, non_accident_results = process_video_directory(
        args.non_accident_dir, 
        valid_non_accident_dir, 
        invalid_non_accident_dir, 
        fixed_non_accident_dir,
        args.num_frames
    )
    
    # Combine results for training
    logger.info("\nPreparing training data...")
    train_accident_dir = os.path.join(args.output_dir, "train_accidents")
    train_non_accident_dir = os.path.join(args.output_dir, "train_non_accidents")
    
    os.makedirs(train_accident_dir, exist_ok=True)
    os.makedirs(train_non_accident_dir, exist_ok=True)
    
    # Copy valid and fixed accident videos to train directory
    for video in os.listdir(valid_accident_dir):
        try:
            shutil.copy2(os.path.join(valid_accident_dir, video), 
                        os.path.join(train_accident_dir, video))
        except Exception as e:
            logger.error(f"Error copying valid accident video {video}: {str(e)}")
    
    for video in os.listdir(fixed_accident_dir):
        try:
            # Only copy if the fix was successful (file exists and has content)
            video_path = os.path.join(fixed_accident_dir, video)
            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                shutil.copy2(video_path, os.path.join(train_accident_dir, video))
        except Exception as e:
            logger.error(f"Error copying fixed accident video {video}: {str(e)}")
    
    # Copy valid and fixed non-accident videos to train directory
    for video in os.listdir(valid_non_accident_dir):
        try:
            shutil.copy2(os.path.join(valid_non_accident_dir, video), 
                        os.path.join(train_non_accident_dir, video))
        except Exception as e:
            logger.error(f"Error copying valid non-accident video {video}: {str(e)}")
    
    for video in os.listdir(fixed_non_accident_dir):
        try:
            # Only copy if the fix was successful
            video_path = os.path.join(fixed_non_accident_dir, video)
            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                shutil.copy2(video_path, os.path.join(train_non_accident_dir, video))
        except Exception as e:
            logger.error(f"Error copying fixed non-accident video {video}: {str(e)}")
    
    # Save statistics
    all_stats = {
        "accident": accident_stats,
        "non_accident": non_accident_stats,
        "training_data": {
            "accidents": len(os.listdir(train_accident_dir)),
            "non_accidents": len(os.listdir(train_non_accident_dir))
        }
    }
    
    try:
        with open(os.path.join(args.output_dir, "validation_stats.json"), 'w') as f:
            json.dump(all_stats, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving validation stats: {str(e)}")
    
    # Save detailed results
    all_results = {
        "accident": accident_results,
        "non_accident": non_accident_results
    }
    
    try:
        with open(os.path.join(args.output_dir, "validation_results.json"), 'w') as f:
            json.dump(all_results, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving validation results: {str(e)}")
    
    # Print summary
    summary = "\n=== Video Validation Summary ===\n"
    summary += f"Accident videos: {accident_stats['total']} total, {accident_stats['valid']} valid, {accident_stats['fixed']} fixed, {accident_stats['unfixable']} unfixable\n"
    summary += f"Non-accident videos: {non_accident_stats['total']} total, {non_accident_stats['valid']} valid, {non_accident_stats['fixed']} fixed, {non_accident_stats['unfixable']} unfixable\n"
    summary += f"\nTraining data prepared: {all_stats['training_data']['accidents']} accident videos and {all_stats['training_data']['non_accidents']} non-accident videos\n"
    summary += f"\nTo use the validated data for training, update your config.json with:\n"
    summary += f'"accident_dir": "{os.path.abspath(train_accident_dir)}",\n'
    summary += f'"non_accident_dir": "{os.path.abspath(train_non_accident_dir)}"\n'
    
    logger.info(summary)
    print(summary)
    
    # Return stats for programmatic use
    return all_stats

if __name__ == "__main__":
    main() 