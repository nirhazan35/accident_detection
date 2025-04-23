# Video Standardization for Model Training

This script standardizes all videos to have the same properties for consistent model training:
- Resolution: 480x360
- Bitrate: 1500kbps
- Frame Rate: 15fps
- Maintains original length and play speed

## Prerequisites

- Python 3.6+
- FFmpeg installed on your system

To install FFmpeg:
- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt install ffmpeg`
- Windows: Download from [FFmpeg.org](https://ffmpeg.org/download.html) and add to PATH

## Usage

```bash
# Basic usage
python video_standardizer.py --input /path/to/input/videos --output /path/to/output/videos

# Example with relative paths
python video_standardizer.py --input data/accidents --output data/processed/accidents
```

## Features

**Video Standardization**: Converts all videos to the same resolution, bitrate, and frame rate while maintaining the original length and play speed.

## Output

The script maintains the same directory structure from input to output. For example, if you have:
```
data/accidents/video1.mp4
data/accidents/day1/video2.mp4
```

After processing, you'll get:
```
data/processed/accidents/video1.mp4
data/processed/accidents/day1/video2.mp4
```

A log of all processed videos, including success and failures, will be displayed in the console output. 