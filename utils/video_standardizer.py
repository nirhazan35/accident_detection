#!/usr/bin/env python
import os
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm
import logging
import tempfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_ffmpeg():
    """Check if FFmpeg is installed."""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

def standardize_video(input_path, output_path):
    """
    Standardize a video to 480x360 resolution, 1500kbps bitrate, and 15fps,
    while maintaining the original length and play speed.
    
    Args:
        input_path: Path to the input video.
        output_path: Path to save the standardized video.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    if not os.path.exists(input_path):
        logger.error(f"Input file does not exist: {input_path}")
        return False
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # FFmpeg command to standardize the video
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-vf', 'scale=480:360,fps=15',
        '-b:v', '1500k',
        '-movflags', '+faststart',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-strict', 'experimental',
        '-y',  # Overwrite output file if it exists
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logger.error(f"Error standardizing {input_path}: {result.stderr}")
            return False
        logger.info(f"Successfully standardized {input_path} to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Exception while standardizing {input_path}: {str(e)}")
        return False

def process_directory(input_dir, output_dir):
    """Process all videos in a directory and its subdirectories."""
    if not check_ffmpeg():
        logger.error("FFmpeg is not installed. Please install FFmpeg first.")
        return
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(list(input_dir.glob(f'**/*{ext}')))
    
    logger.info(f"Found {len(video_files)} video files to process.")
    
    success_count = 0
    failed_files = []
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        # Create the same directory structure in the output directory
        rel_path = video_file.relative_to(input_dir)
        output_file = output_dir / rel_path
        
        # Create parent directories if they don't exist
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        if standardize_video(str(video_file), str(output_file)):
            success_count += 1
        else:
            failed_files.append(str(video_file))
    
    logger.info(f"Standardization complete. {success_count} videos standardized successfully.")
    if failed_files:
        logger.warning(f"{len(failed_files)} videos failed to standardize:")
        for file in failed_files:
            logger.warning(f"  - {file}")

def main():
    parser = argparse.ArgumentParser(description="Standardize videos for model training")
    parser.add_argument("--input", "-i", required=True, help="Input directory containing videos")
    parser.add_argument("--output", "-o", required=True, help="Output directory for standardized videos")
    args = parser.parse_args()
    
    process_directory(args.input, args.output)

if __name__ == "__main__":
    main() 