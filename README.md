# Accident Detection System

A machine learning system for detecting accidents in video footage using a hybrid LSTM-Transformer architecture.

## Overview

This system analyzes video sequences and identifies potential accident events. It's designed to work with various types of videos and can be deployed for traffic monitoring, workplace safety, and other surveillance applications.

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