import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model
    """
    def __init__(self, d_model, max_seq_length=32):
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
        return x + self.pe[:, :x.size(1), :]

class LSTMTransformerModel(nn.Module):
    """
    LSTM-Transformer model for accident detection
    Uses a pre-trained CNN as feature extractor, followed by LSTM and Transformer layers
    """
    def __init__(self, cnn_model_name="resnet18", pretrained=True, 
                 feature_dim=512, hidden_dim=256, num_lstm_layers=1, 
                 num_transformer_layers=2, num_heads=4, 
                 dropout=0.2, use_pretrained_features=False):
        super(LSTMTransformerModel, self).__init__()
        
        self.use_pretrained_features = use_pretrained_features
        
        # Feature extractor (CNN)
        if not use_pretrained_features:
            if cnn_model_name == "resnet18":
                cnn = models.resnet18(pretrained=pretrained)
                self.feature_extractor = nn.Sequential(*list(cnn.children())[:-1])
                self.feature_dim = 512
            elif cnn_model_name == "resnet50":
                cnn = models.resnet50(pretrained=pretrained)
                self.feature_extractor = nn.Sequential(*list(cnn.children())[:-1])
                self.feature_dim = 2048
            elif cnn_model_name == "efficientnet_b0":
                cnn = models.efficientnet_b0(pretrained=pretrained)
                self.feature_extractor = nn.Sequential(*list(cnn.children())[:-1])
                self.feature_dim = 1280
            else:
                raise ValueError(f"Unsupported CNN model: {cnn_model_name}")
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
        
        # Positional encoding for transformer
        self.positional_encoding = PositionalEncoding(hidden_dim * 2)  # *2 for bidirectional
        
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
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: If use_pretrained_features=False, input is [batch_size, seq_len, C, H, W]
               If use_pretrained_features=True, input is [batch_size, seq_len, feature_dim]
        
        Returns:
            torch.Tensor: Probability of accident, shape [batch_size, 1]
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        if not self.use_pretrained_features:
            # Reshape for CNN processing
            x = x.view(batch_size * seq_len, x.size(2), x.size(3), x.size(4))
            
            # Extract CNN features
            features = self.feature_extractor(x)
            features = features.view(batch_size, seq_len, -1)
        else:
            # If input already contains extracted features
            features = x
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)
        
        # Add positional encoding for transformer
        transformer_input = self.positional_encoding(lstm_out)
        
        # Transformer processing
        transformer_out = self.transformer_encoder(transformer_input)
        
        # Global attention pooling over sequence dimension
        attn_weights = F.softmax(torch.matmul(
            transformer_out, 
            transformer_out.transpose(1, 2)
        ), dim=2)
        
        context_vector = torch.matmul(attn_weights, transformer_out)
        
        # Take the representation of the last timestep for classification
        final_representation = context_vector[:, -1, :]
        
        # Classification
        output = self.classifier(final_representation)
        
        return output

def get_model(config):
    """
    Factory function to create a model from configuration
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        nn.Module: Instantiated model
    """
    model = LSTMTransformerModel(
        cnn_model_name=config.get('cnn_model', 'resnet18'),
        pretrained=config.get('pretrained', True),
        feature_dim=config.get('feature_dim', 512),
        hidden_dim=config.get('hidden_dim', 256),
        num_lstm_layers=config.get('num_lstm_layers', 1),
        num_transformer_layers=config.get('num_transformer_layers', 2),
        num_heads=config.get('num_heads', 4),
        dropout=config.get('dropout', 0.2),
        use_pretrained_features=config.get('use_pretrained_features', False)
    )
    
    return model 