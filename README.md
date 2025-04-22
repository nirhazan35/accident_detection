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

## Data Preparation

1. Organize your videos into two folders:
   - `data/accidents/` - Videos containing accidents
   - `data/non_accidents/` - Videos without accidents

2. Run the video validation script to process and fix corrupted videos:
   ```
   python src/validate_videos.py --accident_dir data/accidents --non_accident_dir data/non_accidents
   ```

   This will:
   - Validate all videos in both directories
   - Fix corrupted videos when possible
   - Create a clean dataset in `data/processed/`
   - Generate reports on video validation

## Training

1. Train the model using the processed data:
   ```
   python src/train.py --config configs/processed_config.json
   ```

   The training process includes:
   - Data loading and preprocessing
   - Model training with early stopping
   - Performance visualization
   - Model checkpointing

2. The training output will be saved to the directory specified in the config file, including:
   - Model checkpoints
   - Training history plots
   - Evaluation metrics

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