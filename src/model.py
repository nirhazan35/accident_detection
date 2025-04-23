import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models
import timm

class SpatialAttention(nn.Module):
    """
    Spatial attention module to focus on important regions in frames
    """
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        
    def forward(self, x):
        # x shape: [batch_size, channels, height, width]
        attention = self.conv(x)
        attention = torch.sigmoid(attention)
        return x * attention.expand_as(x)

class ChannelAttention(nn.Module):
    """
    Channel attention module (similar to SE block) to focus on important channels
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = torch.sigmoid(avg_out + max_out)
        return x * out

class TemporalAttention(nn.Module):
    """
    Temporal attention module to identify critical frames in a video sequence
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, hidden_dim]
        attn = self.fc1(x)
        attn = F.relu(attn)
        attn = self.dropout(attn)
        attn = self.fc2(attn)  # [batch_size, seq_len, 1]
        
        # Normalize attention weights
        attn_weights = F.softmax(attn.squeeze(-1), dim=1).unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # Apply attention weights to input
        context = torch.sum(x * attn_weights, dim=1)  # [batch_size, hidden_dim]
        
        return context, attn_weights

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model
    Modified to handle longer sequences
    """
    def __init__(self, d_model, max_seq_length=120):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class LSTMTransformerModel(nn.Module):
    """
    Enhanced LSTM-Transformer model for accident detection with:
    1. Modern CNN backbones (EfficientNetV2, ConvNeXt)
    2. Spatial and channel attention in feature extraction
    3. Temporal attention for critical frame identification
    4. Optional 3D CNN capabilities
    """
    def __init__(self, cnn_model_name="resnet18", pretrained=True, 
                 feature_dim=512, hidden_dim=256, num_lstm_layers=1, 
                 num_transformer_layers=2, num_heads=4, 
                 dropout=0.2, use_pretrained_features=False,
                 max_sequence_length=120, use_3d_cnn=False):
        super(LSTMTransformerModel, self).__init__()
        
        self.use_pretrained_features = use_pretrained_features
        self.max_sequence_length = max_sequence_length
        self.use_3d_cnn = use_3d_cnn
        
        # Feature extractor (CNN)
        if not use_pretrained_features:
            if use_3d_cnn:
                # 3D CNN for motion pattern capturing
                self.feature_extractor = self._create_3d_cnn()
                self.feature_dim = 512
            else:
                # 2D CNN with modern backbones
                self.feature_extractor, self.feature_dim = self._create_2d_cnn(cnn_model_name, pretrained)
                
                # Add spatial and channel attention after feature extraction
                if not cnn_model_name.startswith("efficientnet"):  # EfficientNet already has attention
                    self.spatial_attention = SpatialAttention(self.feature_dim)
                    self.channel_attention = ChannelAttention(self.feature_dim)
        else:
            # If using pre-extracted features, we just need to set the feature_dim
            self.feature_dim = feature_dim
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Temporal attention module
        self.temporal_attention = TemporalAttention(hidden_dim * 2)  # *2 for bidirectional
        
        # Positional encoding for transformer
        self.positional_encoding = PositionalEncoding(hidden_dim * 2, max_seq_length=max_sequence_length)
        
        # Transformer encoder layer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 2,  # *2 for bidirectional
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        
        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_layer,
            num_layers=num_transformer_layers
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def _create_2d_cnn(self, cnn_model_name, pretrained):
        """Create 2D CNN feature extractor with modern backbones"""
        if cnn_model_name == "resnet18":
            if pretrained:
                cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                cnn = models.resnet18(weights=None)
            feature_extractor = nn.Sequential(*list(cnn.children())[:-1])
            feature_dim = 512
        elif cnn_model_name == "resnet50":
            if pretrained:
                cnn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            else:
                cnn = models.resnet50(weights=None)
            feature_extractor = nn.Sequential(*list(cnn.children())[:-1])
            feature_dim = 2048
        elif cnn_model_name == "efficientnet_v2_s":
            # Using timm for EfficientNetV2
            if pretrained:
                model = timm.create_model('tf_efficientnetv2_s', pretrained=True, features_only=False)
            else:
                model = timm.create_model('tf_efficientnetv2_s', pretrained=False, features_only=False)
            # Remove the classifier
            feature_extractor = nn.Sequential(*list(model.children())[:-1], nn.AdaptiveAvgPool2d(1))
            feature_dim = 1280
        elif cnn_model_name == "convnext_tiny":
            # Using timm for ConvNeXt
            if pretrained:
                model = timm.create_model('convnext_tiny', pretrained=True, features_only=False)
            else:
                model = timm.create_model('convnext_tiny', pretrained=False, features_only=False)
            # Remove the classifier
            feature_extractor = nn.Sequential(*list(model.children())[:-1])
            feature_dim = 768
        else:
            raise ValueError(f"Unsupported CNN model: {cnn_model_name}")
        
        return feature_extractor, feature_dim
    
    def _create_3d_cnn(self):
        """Create 3D CNN for better motion pattern capturing (SlowFast inspired)"""
        # Simplified 3D CNN inspired by SlowFast networks
        model = nn.Sequential(
            # Initial 3D convolution
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            # 3D ResBlock 1
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            # 3D ResBlock 2
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            # 3D ResBlock 3
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            
            # Global pooling
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        return model
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: If use_pretrained_features=False, input is [batch_size, seq_len, C, H, W]
               If use_pretrained_features=True, input is [batch_size, seq_len, feature_dim]
               If use_3d_cnn=True, input is reshaped to [batch_size, C, seq_len, H, W]
        
        Returns:
            torch.Tensor: Probability of accident, shape [batch_size, 1]
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        if not self.use_pretrained_features:
            if self.use_3d_cnn:
                # Reshape for 3D CNN: [B, T, C, H, W] -> [B, C, T, H, W]
                x = x.permute(0, 2, 1, 3, 4)
                features = self.feature_extractor(x)
                features = features.view(batch_size, self.feature_dim)
                # Expand to sequence dimension for compatibility with LSTM
                features = features.unsqueeze(1).expand(-1, seq_len, -1)
            else:
                # Reshape for 2D CNN processing: [B, T, C, H, W] -> [B*T, C, H, W]
                x = x.view(batch_size * seq_len, x.size(2), x.size(3), x.size(4))
                
                # Extract CNN features
                features = self.feature_extractor(x)
                
                # Apply spatial and channel attention if available
                if hasattr(self, 'spatial_attention') and hasattr(self, 'channel_attention'):
                    features = self.spatial_attention(features)
                    features = self.channel_attention(features)
                
                # Reshape back to sequence: [B*T, C, 1, 1] -> [B, T, C]
                features = features.view(batch_size, seq_len, -1)
        else:
            # If input already contains extracted features
            features = x
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)
        
        # Apply temporal attention to identify critical frames
        context, attn_weights = self.temporal_attention(lstm_out)
        
        # Add positional encoding for transformer
        transformer_input = self.positional_encoding(lstm_out)
        
        # Transformer processing
        transformer_out = self.transformer_encoder(transformer_input)
        
        # Combine transformer output with temporal attention
        # Take the representation of the last timestep
        final_representation = transformer_out[:, -1, :]
        
        # Add the temporal context vector (optional)
        # final_representation = final_representation + context
        
        # Classification
        output = self.classifier(final_representation)
        
        return output

class X3DModel(nn.Module):
    """
    Simplified X3D model for accident detection.
    X3D is designed for efficient video recognition with different model scales.
    """
    def __init__(self, input_channels=3, width_factor=1.0, depth_factor=1.0, 
                 hidden_dim=256, dropout=0.2):
        super(X3DModel, self).__init__()
        
        self.width_mul = width_factor
        base_channels = int(24 * self.width_mul)
        
        # Initial 3D conv
        self.conv1 = nn.Conv3d(
            input_channels, base_channels, 
            kernel_size=(5, 3, 3), stride=(1, 2, 2), padding=(2, 1, 1), bias=False
        )
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # X3D blocks
        self.layer1 = self._make_layer(base_channels, base_channels*2, int(3*depth_factor))
        self.layer2 = self._make_layer(base_channels*2, base_channels*4, int(5*depth_factor), stride=2)
        self.layer3 = self._make_layer(base_channels*4, base_channels*8, int(11*depth_factor), stride=2)
        self.layer4 = self._make_layer(base_channels*8, base_channels*16, int(7*depth_factor), stride=2)
        
        # Global average pooling and projection
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Output layers
        self.fc = nn.Linear(base_channels*16, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        # First block may have stride
        layers.append(X3DBlock(in_channels, out_channels, stride=stride))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(X3DBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [batch_size, channels, time, height, width]
            
        Returns:
            torch.Tensor: Probability of accident, shape [batch_size, 1]
        """
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # X3D blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.fc(x)
        x = self.relu(x)
        x = self.classifier(x)
        
        return x

class X3DBlock(nn.Module):
    """
    X3D building block with bottleneck and SE attention
    """
    def __init__(self, in_channels, out_channels, stride=1, 
                 expansion=2.25, groups=1, se_ratio=0.0625):
        super(X3DBlock, self).__init__()
        
        width = int(in_channels * expansion)
        
        # 1x1x1 conv for bottleneck
        self.conv1 = nn.Conv3d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(width)
        
        # 3x3x3 depthwise conv
        self.conv2 = nn.Conv3d(
            width, width, kernel_size=(3, 3, 3), stride=(1, stride, stride),
            padding=(1, 1, 1), groups=width, bias=False
        )
        self.bn2 = nn.BatchNorm3d(width)
        
        # SE module
        self.se = SqueezeExcitation(width, int(in_channels * se_ratio))
        
        # 1x1x1 conv for expansion
        self.conv3 = nn.Conv3d(width, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                          stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.se(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention
    """
    def __init__(self, in_channels, reduced_channels):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(in_channels, reduced_channels, kernel_size=1)
        self.fc2 = nn.Conv3d(reduced_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        identity = x
        
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return identity * x

def get_model(config):
    """
    Factory function to create a model from configuration
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        nn.Module: Instantiated model
    """
    model_type = config.get('model_type', 'lstm_transformer')
    
    if model_type == 'lstm_transformer':
        model = LSTMTransformerModel(
            cnn_model_name=config.get('cnn_model', 'resnet18'),
            pretrained=config.get('pretrained', True),
            feature_dim=config.get('feature_dim', 512),
            hidden_dim=config.get('hidden_dim', 256),
            num_lstm_layers=config.get('num_lstm_layers', 1),
            num_transformer_layers=config.get('num_transformer_layers', 2),
            num_heads=config.get('num_heads', 4),
            dropout=config.get('dropout', 0.2),
            use_pretrained_features=config.get('use_pretrained_features', False),
            max_sequence_length=config.get('max_sequence_length', 120),
            use_3d_cnn=config.get('use_3d_cnn', False)
        )
    elif model_type == 'x3d':
        model = X3DModel(
            input_channels=config.get('input_channels', 3),
            width_factor=config.get('width_factor', 1.0),
            depth_factor=config.get('depth_factor', 1.0),
            hidden_dim=config.get('hidden_dim', 256),
            dropout=config.get('dropout', 0.2)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model 