# Accident Detection System

A machine learning system for detecting accidents in video footage using a hybrid LSTM-Transformer architecture.

## Overview

This system analyzes video sequences and identifies potential accident events. It's designed to work with various types of videos and can be deployed for traffic monitoring, workplace safety, and other surveillance applications.

## Directory Structure

```
.
├── configs/                          # Configuration files
│   └── config.json                   # Main configuration file for training
├── data/                             # Data directory
│   ├── accidents/                    # Folder with raw accident videos
│   ├── non_accidents/                # Folder with raw non-accident videos
│   └── processed/                    # Processed data
│       ├── accidents/                # Processed accident videos
│       └── non_accidents/            # Processed non-accident videos
├── models/                           # Trained models
├── src/                              # Source code
│   ├── check_video_codecs.py         # Diagnostic tool for video codecs
│   ├── model.py                      # Model architecture
│   ├── train.py                      # Training script
│   ├── utils.py                      # Utility functions
│   └── validate_videos.py            # Video validation script
└── README.md                         # This file
```

## Project Workflow

### 1. Prepare Your Data

Place your raw videos in the following directories:
- Accident videos: `data/accidents/`
- Non-accident videos: `data/non_accidents/`

### 2. Validate and Process Videos

Run the video validation script to check for corrupted videos and process them:

```bash
python src/validate_videos.py
```

This will:
1. Check all videos for corruption
2. Try to fix corrupted videos
3. Save all processed videos (both valid and fixed) to:
   - `data/processed/accidents/`
   - `data/processed/non_accidents/`
4. Update the config file with the paths to the processed videos

To check system compatibility for video processing:

```bash
python src/check_video_codecs.py
```

### 3. Train the Model (Separate Step)

After video validation is complete, you can train the model:

```bash
python src/train.py --config configs/config.json
```

Or combine validation and training in one step:

```bash
python src/validate_videos.py --run_training
```

## Project Structure

```
accident_detection/
├── configs/             # Configuration files
├── data/                # Data directory
│   ├── accidents/       # Accident videos
│   ├── non_accidents/   # Non-accident videos
│   └── processed/       # Processed videos (after validation)
├── models/              # Saved model checkpoints
├── output/              # Training outputs
├── src/                 # Source code
│   ├── data_processor.py   # Data loading and processing
│   ├── detect_realtime.py  # Real-time accident detection
│   ├── evaluate.py         # Model evaluation
│   ├── model.py            # Model architecture
│   ├── train.py            # Training script
│   ├── validate_videos.py  # Video validation script
│   └── visualize.py        # Visualization utilities
└── utils/               # Utility scripts
```

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/accident_detection.git
   cd accident_detection
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Data Preparation and Training

The recommended workflow is to separate the validation and training steps:

### Step 1: Validate and Fix Videos

First, process your raw videos to fix any corrupted files:

```
python src/validate_videos.py --accident_dir data/accidents --non_accident_dir data/non_accidents
```

This will:
- Validate all videos in both directories
- Fix corrupted videos when possible
- Create a clean dataset in `data/processed/`
- Update your config.json to use the processed videos
- Generate detailed reports on video validation

After this step, your `config.json` will be updated to point to the validated videos.

### Step 2: Train the Model

Once validation is complete, train the model:

```
python src/train.py --config configs/config.json
```

The training process includes:
- Data loading and preprocessing from validated videos
- Model training with early stopping
- Performance visualization
- Model checkpointing

The training output will be saved to the output directory, including:
- Model checkpoints
- Training history plots
- Evaluation metrics
- Final trained model

### Alternative Workflows

If needed, you can also:

1. **Validate and train in one step**:
   ```
   python src/validate_videos.py --accident_dir data/accidents --non_accident_dir data/non_accidents --run_training
   ```
   
2. **Skip validation** (if you've already validated):
   ```
   python src/validate_videos.py --skip_validation --run_training
   ```

## Running Real-Time Detection

To perform real-time accident detection on video streams:

```
python src/detect_realtime.py --config_path configs/config.json --model_path models/best_model.pt --source video_file.mp4
```

Options for the source parameter:
- Path to a video file
- Camera index (e.g., 0 for webcam)
- RTSP stream URL

## Evaluation

To evaluate a trained model:

```
python src/evaluate.py --config configs/config.json --model_path models/best_model.pt
```

## References

- [Original research paper](https://example.com/paper-url)
- [Dataset source](https://example.com/dataset-url)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Command Line Usage

There are two ways to use the video validation script:

### 1. Using Command Line Arguments

```bash
python src/validate_videos.py --accident-dir data/accidents --non-accident-dir data/non_accidents --output-dir data/processed
```

### 2. Using Configuration File (Recommended)

First, create a default configuration file:

```bash
python src/validate_videos.py --create-config config.yaml
```

Then run the script with the config file:

```bash
python src/validate_videos.py --config config.yaml
```

### Additional Options

- `--dry-run`: Only validate videos without processing them
- `--num-frames`: Number of frames to check in each video (default: 32)
- `--min-valid-frames`: Minimum valid frames required (default: 16)
- `--parallel`: Process videos in parallel (default: True)
- `--no-parallel`: Disable parallel processing
- `--update-config`: Path to config JSON file to update with processed paths
- `--verbose`: Enable verbose logging

### Example Workflow

1. Create a default configuration file:
   ```bash
   python src/validate_videos.py --create-config config.yaml
   ```

2. Edit the config file to set your data directories and parameters:
   ```yaml
   data_dir: /path/to/your/data
   raw_dirs:
     accidents: raw/accidents
     non_accidents: raw/non_accidents
   processed_dirs:
     accidents: processed/accidents
     non_accidents: processed/non_accidents
   num_frames: 32
   min_valid_frames: 16
   parallel: true
   verbose: false
   ```

3. Run the validation script:
   ```bash
   python src/validate_videos.py --config config.yaml
   ```

4. Update your training configuration with the processed video paths:
   ```bash
   python src/validate_videos.py --config config.yaml --update-config configs/config.json
   ```

5. Train your model:
   ```bash
   python src/train.py --config configs/config.json
   ```

## Simplified Directory Structure

After running the validation script, your data will be organized as follows:

```
/data_dir
├── raw/
│   ├── accidents/            # Original accident videos
│   └── non_accidents/        # Original non-accident videos
└── processed/
    ├── accidents/            # Processed (valid + fixed) accident videos
    └── non_accidents/        # Processed (valid + fixed) non-accident videos
```

All valid original videos and successfully fixed videos will be saved to the `processed` directory with their original names, ready for training.

## Checking System Compatibility

You can check your system compatibility for video processing:

```bash
python src/check_video_codecs.py
```

This will test available video codecs and report any issues that might affect the video processing. 