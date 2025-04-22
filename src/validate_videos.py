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
import math
import glob
import concurrent.futures
import yaml

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

def validate_video(video_path, num_frames=32, min_valid_frames=16):
    """
    Validate a video file by checking if it can be opened and frames can be read
    
    Args:
        video_path (str): Path to video file
        num_frames (int): Number of frames to check
        min_valid_frames (int): Minimum number of valid frames required
        
    Returns:
        tuple: ("valid" or "invalid", error message if any)
    """
    try:
        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            return "invalid", "File does not exist or is empty"
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "invalid", "Could not open video"
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if frame_count <= 0:
            cap.release()
            return "invalid", "Zero frame count"
            
        if frame_width <= 0 or frame_height <= 0:
            cap.release()
            return "invalid", f"Invalid dimensions: {frame_width}x{frame_height}"
        
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
                return "invalid", f"Low frame variance, possibly corrupted: {variance}/{mean_size}"
        
        # Validate results
        if valid_frames < min_valid_frames:
            return "invalid", f"Only {valid_frames}/{total_check} frames could be read (minimum {min_valid_frames})"
            
        return "valid", ""
    
    except Exception as e:
        logger.error(f"Error validating {video_path}: {str(e)}")
        return "invalid", f"Exception: {str(e)}"

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
        # First, try to get detailed info about the video to diagnose issues
        logger.info(f"Starting to fix video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video for fixing: {video_path}")
            return False
            
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Log video properties for diagnosis
        logger.info(f"Video properties: frame_count={frame_count}, dimensions={width}x{height}, fps={fps}")
        
        # Ensure valid dimensions
        if width <= 0 or height <= 0:
            logger.error(f"Invalid dimensions {width}x{height} for {video_path}")
            cap.release()
            return False
            
        # Ensure valid framerate
        if fps <= 0:
            fps = 30.0  # Default to 30 fps if invalid
            logger.warning(f"Invalid fps, using default value: {fps}")
        
        # Try different fixing methods - Method 1: Using OpenCV VideoWriter
        success = try_fix_with_videowriter(cap, output_path, num_frames, width, height, fps)
        
        if success:
            logger.info(f"Successfully fixed video using VideoWriter: {output_path}")
            return True
            
        # If VideoWriter method failed, try Method 2: Save as image sequence
        logger.warning(f"VideoWriter method failed, trying image sequence method")
        success = try_fix_with_image_sequence(cap, output_path, num_frames, width, height, fps)
        
        cap.release()
        
        if success:
            logger.info(f"Successfully fixed video using image sequence: {output_path}")
            return True
        else:
            logger.error(f"All fixing methods failed for {video_path}")
            return False
        
    except Exception as e:
        logger.error(f"Error extracting keyframes from {video_path}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def try_fix_with_videowriter(cap, output_path, num_frames, width, height, fps):
    """
    Try to fix video using OpenCV VideoWriter
    
    Args:
        cap: OpenCV VideoCapture object
        output_path: Path to save fixed video
        num_frames: Number of frames to extract
        width: Frame width
        height: Frame height
        fps: Frames per second
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Reset cap position
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Find available codec
        codec = get_available_codec()
        if not codec:
            logger.error("No suitable video codec found for VideoWriter")
            return False
            
        logger.info(f"Using codec: {codec}")
        
        # Make sure dimensions are even (required by some codecs)
        width = width if width % 2 == 0 else width + 1
        height = height if height % 2 == 0 else height + 1
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            logger.error(f"Failed to create VideoWriter with codec {codec}, dimensions {width}x{height}")
            return False
        
        # Extract evenly spaced frames
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, frame_count - 1, min(num_frames, frame_count), dtype=int)
        valid_frames = []
        
        logger.info(f"Extracting {len(indices)} frames...")
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                # Ensure frame dimensions match output dimensions
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                valid_frames.append(frame)
                out.write(frame)
                
        logger.info(f"Extracted {len(valid_frames)} valid frames")
        
        # If we didn't get enough frames, duplicate the last one
        if len(valid_frames) < num_frames and len(valid_frames) > 0:
            logger.info(f"Duplicating last frame to reach {num_frames} frames")
            for _ in range(num_frames - len(valid_frames)):
                out.write(valid_frames[-1])
        
        # Release resources
        out.release()
        
        # Validate the output exists and has content
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            logger.error(f"Output video file is empty or doesn't exist: {output_path}")
            return False
            
        # Try to open the output video to verify it works
        test_cap = cv2.VideoCapture(output_path)
        if not test_cap.isOpened():
            logger.error(f"Cannot open output video for verification: {output_path}")
            test_cap.release()
            return False
            
        # Check if we can read at least one frame
        ret, frame = test_cap.read()
        test_cap.release()
        
        if not ret or frame is None:
            logger.error(f"Cannot read frames from output video: {output_path}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error in try_fix_with_videowriter: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def try_fix_with_image_sequence(cap, output_path, num_frames, width, height, fps):
    """
    Try to fix video by saving individual frames
    
    Args:
        cap: OpenCV VideoCapture object
        output_path: Path to save fixed video
        num_frames: Number of frames to extract
        width: Frame width
        height: Frame height
        fps: Frames per second
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Reset cap position
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Create directory for frames
        frames_dir = output_path + "_frames"
        os.makedirs(frames_dir, exist_ok=True)
        
        # Extract evenly spaced frames
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, frame_count - 1, min(num_frames, frame_count), dtype=int)
        valid_frames = []
        
        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                # Ensure frame dimensions are consistent
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                
                frame_path = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                valid_frames.append(frame_path)
        
        # If we didn't get enough frames, duplicate the last one
        if len(valid_frames) < num_frames and len(valid_frames) > 0:
            last_frame = cv2.imread(valid_frames[-1])
            for i in range(len(valid_frames), num_frames):
                frame_path = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
                cv2.imwrite(frame_path, last_frame)
                valid_frames.append(frame_path)
        
        # Create a metadata file
        metadata_path = os.path.join(frames_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                "original_file": video_path,
                "fps": fps,
                "width": width,
                "height": height,
                "frame_count": len(valid_frames),
                "frames": [os.path.basename(f) for f in valid_frames]
            }, f, indent=4)
        
        # Try to create a video from the frames as an alternative
        alternative_video_path = os.path.join(frames_dir, "compiled_video.mp4")
        try:
            # Try with automatic codec selection
            compile_success = compile_frames_to_video(valid_frames, alternative_video_path, fps, width, height)
            if compile_success:
                # Use the compiled video instead
                shutil.copy2(alternative_video_path, output_path)
                logger.info(f"Successfully compiled frames to video: {output_path}")
                return True
        except Exception as e:
            logger.warning(f"Could not compile frames to video: {str(e)}")
        
        # If we couldn't create a video, create a simple text pointer to the frames dir
        with open(output_path, 'w') as f:
            f.write(frames_dir)
            
        return len(valid_frames) > 0
        
    except Exception as e:
        logger.error(f"Error in try_fix_with_image_sequence: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def compile_frames_to_video(frame_paths, output_path, fps, width, height):
    """Try to compile frames into a video file using ffmpeg if available"""
    try:
        # First try with ffmpeg if available
        import subprocess
        
        # Check if ffmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            
            # Create a text file with frame paths
            frame_list = "frames_list.txt"
            with open(frame_list, 'w') as f:
                for frame_path in frame_paths:
                    f.write(f"file '{frame_path}'\n")
            
            # Use ffmpeg to create video
            cmd = [
                'ffmpeg', '-y', '-r', str(fps), 
                '-f', 'concat', '-safe', '0', '-i', frame_list,
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', 
                '-vf', f'scale={width}:{height}',
                output_path
            ]
            
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            
            # Clean up
            if os.path.exists(frame_list):
                os.remove(frame_list)
                
            return os.path.exists(output_path) and os.path.getsize(output_path) > 0
            
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("ffmpeg not available, falling back to OpenCV")
            
        # Fallback to OpenCV if ffmpeg is not available
        codec = get_available_codec() or 'mp4v'  # Default to mp4v if no codec found
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            logger.error(f"Failed to create VideoWriter for frame compilation")
            return False
            
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            if frame is not None:
                # Ensure frame dimensions match
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                out.write(frame)
                
        out.release()
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
        
    except Exception as e:
        logger.error(f"Error compiling frames to video: {str(e)}")
        return False

def process_video_directory(src_dir, valid_dir, invalid_dir=None, fixed_dir=None, num_frames=32, min_valid_frames=16, parallel=True):
    """
    Process all videos in a directory, validate them, and organize them into valid, invalid, and fixed directories.
    
    When using the simplified directory structure, only provide the valid_dir parameter:
    - Valid videos will be saved directly to valid_dir with their original filenames
    - Invalid videos will be fixed and saved to valid_dir with "fixed_" prefix
    - Original invalid videos won't be saved anywhere
    
    Args:
        src_dir (str): Source directory containing videos
        valid_dir (str): Directory for valid videos and fixed videos when using simplified structure
        invalid_dir (str, optional): Directory for invalid videos. If None, invalid videos are not saved.
        fixed_dir (str, optional): Directory for fixed videos. If None, fixed videos are saved to valid_dir.
        num_frames (int): Number of frames to check in each video
        min_valid_frames (int): Minimum number of valid frames required
        parallel (bool): Whether to process videos in parallel
        
    Returns:
        tuple: (stats_dict, results_list) where stats_dict contains counts of processed videos
               and results_list contains detailed results for each video
    """
    os.makedirs(valid_dir, exist_ok=True)
    if invalid_dir:
        os.makedirs(invalid_dir, exist_ok=True)
    
    # If fixed_dir is None, we'll save fixed videos to valid_dir
    if fixed_dir:
        os.makedirs(fixed_dir, exist_ok=True)
    else:
        fixed_dir = valid_dir  # Use valid_dir for fixed videos
    
    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_files.extend(glob.glob(os.path.join(src_dir, f'*{ext}')))
    
    if not video_files:
        logger.warning(f"No video files found in {src_dir}")
        return {"total": 0, "valid": 0, "invalid": 0, "fixed": 0, "unfixable": 0}, []
    
    logger.info(f"Found {len(video_files)} videos in {src_dir}")
    
    results = []
    valid_count = 0
    invalid_count = 0
    fixed_count = 0
    unfixable_count = 0
    
    def process_video_file(video_path):
        video_name = os.path.basename(video_path)
        valid_path = os.path.join(valid_dir, video_name)
        invalid_path = os.path.join(invalid_dir, video_name) if invalid_dir else None
        
        # When using simplified structure (fixed_dir == valid_dir), don't add "fixed_" prefix
        # to keep all videos with their original names in the simplified directory
        if fixed_dir == valid_dir:
            fixed_path = os.path.join(fixed_dir, video_name)
        else:
            fixed_path = os.path.join(fixed_dir, f"fixed_{video_name}")
            
        # Create temporary path for standardized video
        standardized_path = os.path.join(os.path.dirname(video_path), f"std_{video_name}")
        
        # Standardize video first
        standardize_success = standardize_video(
            video_path, 
            standardized_path, 
            resolution=(480, 360),
            fps=25,
            bitrate=1500
        )
        
        # If standardization failed, use original video
        source_video = standardized_path if standardize_success else video_path
        
        # Validate (standardized) video
        status, error_msg = validate_video(source_video, num_frames=num_frames, min_valid_frames=min_valid_frames)
        
        result = {
            "file": video_name,
            "status": status,
            "error": error_msg,
            "fixed": False,
            "standardized": standardize_success
        }
        
        if status == "valid":
            try:
                # If video was standardized successfully, copy that instead of original
                src_to_copy = standardized_path if standardize_success else video_path
                shutil.copy2(src_to_copy, valid_path)
                logger.info(f"Valid video: {video_name}")
            except Exception as e:
                logger.error(f"Error copying valid video {video_name}: {e}")
                result["error"] = f"Error copying: {str(e)}"
        else:  # Invalid video
            if invalid_dir:
                try:
                    shutil.copy2(video_path, invalid_path)
                    logger.info(f"Invalid video: {video_name} - {error_msg}")
                except Exception as e:
                    logger.error(f"Error copying invalid video {video_name}: {e}")
            else:
                logger.info(f"Invalid video: {video_name} - {error_msg}")
            
            # Try to fix video (using standardized version if available)
            try:
                fixed = extract_keyframes(source_video, fixed_path, num_frames)
                if fixed:
                    result["fixed"] = True
                    logger.info(f"Fixed video: {video_name}")
                else:
                    logger.warning(f"Could not fix video: {video_name}")
            except Exception as e:
                logger.error(f"Error fixing video {video_name}: {e}")
                result["error"] += f" (Fix error: {str(e)})"
        
        # Clean up temporary standardized file
        if standardize_success and os.path.exists(standardized_path):
            try:
                os.remove(standardized_path)
            except Exception as e:
                logger.warning(f"Could not remove temporary file {standardized_path}: {e}")
                
        return result
    
    if parallel and len(video_files) > 1:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            process_results = list(executor.map(process_video_file, video_files))
            results.extend(process_results)
    else:
        for video_path in video_files:
            result = process_video_file(video_path)
            results.append(result)
    
    # Count results
    for result in results:
        if result["status"] == "valid":
            valid_count += 1
        else:  # invalid
            invalid_count += 1
            if result["fixed"]:
                fixed_count += 1
            else:
                unfixable_count += 1
    
    stats = {
        "total": len(video_files),
        "valid": valid_count,
        "invalid": invalid_count,
        "fixed": fixed_count,
        "unfixable": unfixable_count
    }
    
    logger.info(f"Processed {len(video_files)} videos: {valid_count} valid, {invalid_count} invalid, {fixed_count} fixed, {unfixable_count} unfixable")
    
    return stats, results

def main():
    # HARDCODED PATHS - MODIFY THESE VALUES
    src_accident_dir = "data/accidents"
    src_non_accident_dir = "data/non_accidents"
    output_dir = "data/processed"
    
    # HARDCODED PARAMETERS
    num_frames = 32
    min_valid_frames = 16
    parallel = True
    
    # Command line arguments for dry-run and verbosity only
    parser = argparse.ArgumentParser(description="Validate and process videos for accident detection")
    parser.add_argument("--dry-run", action="store_true", help="Only validate videos without processing")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Validate directories exist
    if not os.path.exists(src_accident_dir):
        logger.error(f"Source directory for accident videos does not exist: {src_accident_dir}")
        return 1
        
    if not os.path.exists(src_non_accident_dir):
        logger.error(f"Source directory for non-accident videos does not exist: {src_non_accident_dir}")
        return 1
    
    # Create output directory structure
    processed_accident_dir = os.path.join(output_dir, "accidents")
    processed_non_accident_dir = os.path.join(output_dir, "non_accidents")
    
    os.makedirs(processed_accident_dir, exist_ok=True)
    os.makedirs(processed_non_accident_dir, exist_ok=True)
    
    logger.info(f"Source accident videos: {src_accident_dir}")
    logger.info(f"Source non-accident videos: {src_non_accident_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    if args.dry_run:
        logger.info("Running in dry-run mode - only validating videos")
        process_dry_run(src_accident_dir, src_non_accident_dir, num_frames, min_valid_frames)
    else:
        logger.info("Processing accident videos...")
        process_video_directory(
            src_dir=src_accident_dir,
            valid_dir=processed_accident_dir,
            num_frames=num_frames,
            min_valid_frames=min_valid_frames,
            parallel=parallel
        )
        
        logger.info("Processing non-accident videos...")
        process_video_directory(
            src_dir=src_non_accident_dir,
            valid_dir=processed_non_accident_dir,
            num_frames=num_frames,
            min_valid_frames=min_valid_frames,
            parallel=parallel
        )
        
        logger.info("Video processing complete")
    
    return 0

def process_dry_run(accident_dir, non_accident_dir, num_frames, min_valid_frames):
    """Process videos in dry-run mode (validation only)"""
    video_files = []
    for dir_path in [accident_dir, non_accident_dir]:
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend(glob.glob(os.path.join(dir_path, f'*{ext}')))
    
    stats = {"total": 0, "valid": 0, "invalid": 0}
    results = []
    
    for i, video_path in enumerate(video_files):
        video_name = os.path.basename(video_path)
        logger.info(f"Validating video {i+1}/{len(video_files)}: {video_name}")
        
        status, error_msg = validate_video(video_path, num_frames, min_valid_frames)
        
        result = {
            "file": video_name,
            "status": status,
            "error": error_msg
        }
        
        if status == "valid":
            stats["valid"] += 1
        else:
            stats["invalid"] += 1
            
        results.append(result)
        print(f"{video_name}: {status}" + (f" - {error_msg}" if error_msg else ""))
    
    stats["total"] = len(video_files)
    logger.info(f"Validated {stats['total']} videos: {stats['valid']} valid, {stats['invalid']} invalid")
    
    return stats, results

def copy_processed_videos(valid_dir, fixed_dir, output_dir):
    """
    Copy all valid and fixed videos to the output directory
    
    Args:
        valid_dir (str): Directory with valid videos
        fixed_dir (str): Directory with fixed videos
        output_dir (str): Output directory for all processed videos
    """
    # Copy valid videos
    if os.path.exists(valid_dir):
        for video in os.listdir(valid_dir):
            try:
                source_path = os.path.join(valid_dir, video)
                dest_path = os.path.join(output_dir, video)
                shutil.copy2(source_path, dest_path)
                logger.info(f"Copied valid video: {video}")
            except Exception as e:
                logger.error(f"Error copying valid video {video}: {str(e)}")
    
    # Copy fixed videos
    if os.path.exists(fixed_dir):
        for video in os.listdir(fixed_dir):
            try:
                source_path = os.path.join(fixed_dir, video)
                # Only copy if the fix was successful (file exists and has content)
                if os.path.exists(source_path) and os.path.getsize(source_path) > 0:
                    dest_path = os.path.join(output_dir, video)
                    shutil.copy2(source_path, dest_path)
                    logger.info(f"Copied fixed video: {video}")
            except Exception as e:
                logger.error(f"Error copying fixed video {video}: {str(e)}")

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
                logger.error(f"ffmpeg error: {process.stderr.decode()}")
                return False
                
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                logger.error(f"Output file is empty or doesn't exist: {output_path}")
                return False
                
            return True
            
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("ffmpeg not available, falling back to OpenCV")
            
            # Fall back to OpenCV if ffmpeg is not available
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                logger.error(f"Could not open input video: {input_path}")
                return False
                
            # Get video properties
            width, height = resolution
            
            # Find available codec
            codec = get_available_codec() or 'avc1'  # Use H.264 when available
            fourcc = cv2.VideoWriter_fourcc(*codec)
            
            # Create output video writer
            out = cv2.VideoWriter(
                output_path, fourcc, fps, (width, height)
            )
            
            if not out.isOpened():
                logger.error(f"Could not create output video writer: {output_path}")
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
                logger.error(f"No frames were processed: {input_path}")
                return False
                
            return True
            
    except Exception as e:
        logger.error(f"Error standardizing video {input_path}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    main() 