# LSTM-Transformer Accident Detection

This project implements a deep learning system for detecting accidents in videos using a hybrid LSTM-Transformer architecture.

## Overview

The system processes video clips and predicts whether they contain an accident event. It combines the strengths of:

- **CNN** for feature extraction from individual frames
- **LSTM** for temporal pattern recognition
- **Transformer** for capturing long-range dependencies and attention mechanisms

This hybrid approach allows the model to effectively analyze temporal patterns while maintaining awareness of the entire sequence context.

## Project Structure

```
.
├── configs/               # Configuration files
│   └── config.json        # Main configuration
├── data/                  # Data directory (not included in repo)
│   ├── accidents/         # Accident videos
│   └── non_accidents/     # Non-accident videos
├── models/                # Saved models
├── notebooks/             # Jupyter notebooks for analysis
├── output/                # Output directory for results
├── src/                   # Source code
│   ├── data_processor.py  # Data loading and processing
│   ├── model.py           # Model architecture
│   ├── train.py           # Training script
│   └── evaluate.py        # Evaluation script
└── README.md              # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/lstm-transformer-accident-detection.git
cd lstm-transformer-accident-detection
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Prepare your data by organizing accident and non-accident videos into their respective directories.

## Usage

### Training

To train the model:

```bash
python src/train.py --config configs/config.json
```

You can customize the training parameters by editing the `config.json` file.

### Evaluation

To evaluate a trained model:

```bash
python src/evaluate.py --config configs/config.json --model models/best_model.pt
```

### Real-time Detection

For real-time detection using a webcam:

```bash
python src/detect_realtime.py --config configs/config.json --model models/best_model.pt
```

## Model Architecture

The architecture consists of:

1. **Feature Extraction**: A pre-trained CNN (ResNet-18) extracts visual features from each video frame.
2. **Temporal Processing**: A bidirectional LSTM processes the sequence of frame features.
3. **Contextual Understanding**: A Transformer encoder with self-attention mechanism further processes the outputs.
4. **Classification**: A fully connected layer with sigmoid activation provides the final prediction.

## Performance

On our test dataset, the model achieves:
- Accuracy: ~92%
- Precision: ~89%
- Recall: ~93%
- F1 Score: ~91%

## Citation

If you use this code in your research, please cite:

```
@misc{lstm-transformer-accident-detection,
  author = {Your Name},
  title = {LSTM-Transformer for Video Accident Detection},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/yourusername/lstm-transformer-accident-detection}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 