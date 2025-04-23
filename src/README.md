# Video Processing for Accident Detection

This module provides tools for validating, fixing, and standardizing videos for use in accident detection machine learning systems.

## Features

- Video validation with customizable criteria
- Automatic fixing of corrupted/invalid videos
- Standardization of videos to consistent format (resolution, FPS, bitrate, codec)
- Parallel processing for improved performance
- Support for both CLI and programmatic usage

## Installation

This module requires:
- Python 3.6+
- OpenCV
- FFmpeg (optional but recommended)

## Usage

### Command Line Usage

The system can be used via the `process_videos.py` script:

```bash
# Basic usage with config file
python process_videos.py --config video_config.json

# Process specific directory
python process_videos.py --input-dir ../data/videos --valid-dir ../data/valid_videos

# Only standardize videos
python process_videos.py --input-dir ../data/videos --standardize-only

# Dry run (validate without copying)
python process_videos.py --input-dir ../data/videos --dry-run
```

### Configuration

You can configure the system using a JSON config file like `video_config.json`:

```json
{
    "input_dir": "../data/accidents",
    "valid_dir": "../data/processed/valid",
    "invalid_dir": "../data/processed/invalid",
    "fixed_dir": "../data/processed/fixed",
    
    "num_frames": 32,
    "min_valid_frames": 24,
    "standard_height": 360,
    
    "target_width": 480,
    "target_height": 360,
    "target_fps": 15,
    "target_bitrate": "1500k",
    "codec": "H.264",
    
    "parallel": true,
    "standardize_valid": true,
    "recursive": false
}
```

### API Usage

You can use the modules programmatically:

```python
from video_processing import process_video_directory, standardize_video

# Process a directory of videos
results = process_video_directory(
    "input_dir",
    "output_dir",
    num_frames=32,
    min_valid_frames=24,
    standardize_valid=True
)

# Standardize a single video
standardize_video(
    "input.mp4", 
    "output.mp4", 
    target_width=480,
    target_height=360,
    target_fps=15
)
```

## Module Structure

- `video_processing/`: Main package
  - `__init__.py`: Package initialization and exports
  - `cli.py`: Command line interface
  - `extractor.py`: Keyframe extraction and video repair
  - `processors.py`: Directory processing functions
  - `standardizer.py`: Video standardization functions
  - `utils.py`: Utility functions
  - `validator.py`: Video validation functions

## Parameters

### Validation Parameters

- `num_frames`: Number of frames to sample for validation (default: 32)
- `min_valid_frames`: Minimum valid frames required (default: 24)
- `standard_height`: Height for resizing during fixing (default: 360)

### Standardization Parameters

- `target_width`: Target width (default: 480)
- `target_height`: Target height (default: 360)
- `target_fps`: Target frames per second (default: 15)
- `target_bitrate`: Target bitrate (default: "1500k")
- `codec`: Video codec (default: "H.264") 