# Train with spatial-channel attention on ResNet50
python src/train_enhanced.py --model spatial_channel_attention

# Train with EfficientNetV2
python src/train_enhanced.py --model efficientnetv2

# Train with ConvNeXt
python src/train_enhanced.py --model convnext

# Train with 3D CNN (SlowFast inspired)
python src/train_enhanced.py --model slowfast_inspired

# Train with X3D model
python src/train_enhanced.py --model x3d