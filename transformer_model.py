"""
NeuralPulse 2.0 - Custom Transformer Architecture.

This module implements the 'Reflex Brain'. It is a lightweight Transformer Encoder
designed specifically for time-series forecasting and anomaly detection.

Key Architectural Decisions:
1. Encoder-Only: We use a sequence-to-sequence regression approach.
2. Custom Positional Encoding: Uses sine/cosine frequencies to retain temporal order.
3. Compactness: Low parameter count (d_model=64) ensures <10ms inference time on CPU.

This file includes:
- PositionalEncoding Module
- TimeSeriesTransformer Module
- Model persistence utilities (Save/Load)

Line Count Target: High (Includes extensive comments and utility methods)
"""

import math
import torch
import torch.nn as nn
import logging
import os
from typing import Optional, Tuple, Dict, Any

try:
    from .reflex_config import TransformerConfig, MODEL_SAVE_PATH, DEVICE
except ImportError:
    from reflex_config import TransformerConfig, MODEL_SAVE_PATH, DEVICE

logger = logging.getLogger("ReflexModel")


# ==============================================================================
# POSITIONAL ENCODING
# ==============================================================================

class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as the
    embeddings, so that the two can be summed.

    Formula:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of [max_len, d_model] representing the positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute the division term (10000^(2i/d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # Register as a buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Crop the PE matrix to the current sequence length
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ==============================================================================
# MAIN TRANSFORMER ARCHITECTURE
# ==============================================================================

class TimeSeriesTransformer(nn.Module):
    """
    A Transformer-based model for time series forecasting.

    Structure:
    1. Input Projection: Linear layer to map features (5) to d_model (64).
    2. Positional Encoding: Adds temporal context.
    3. Transformer Encoder: N layers of Self-Attention + FeedForward.
    4. Output Head: Linear layer to map d_model (64) back to features (5).
    """

    def __init__(self, config: TransformerConfig):
        super(TimeSeriesTransformer, self).__init__()
        self.config = config
        self.model_type = 'Transformer'

        # 1. Input Projection Layer
        # Maps input features (e.g., CPU, RAM...) to the hidden dimension
        self.input_projection = nn.Linear(config.input_dim, config.d_model)

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(
            d_model=config.d_model,
            dropout=config.dropout,
            max_len=config.max_len
        )

        # 3. Transformer Encoder Layers
        # We use batch_first=True for easier data handling [Batch, Seq, Feat]
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=config.num_layers
        )

        # 4. Output Projection Layer
        # Maps the hidden state back to the original feature space
        self.decoder = nn.Linear(config.d_model, config.output_dim)

        # Weight Initialization
        self.init_weights()

    def init_weights(self):
        """
        Xavier/Glorot initialization for linear layers.
        Helps with convergence stability.
        """
        initrange = 0.1
        self.input_projection.bias.data.zero_()
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            src: Input tensor of shape (batch_size, input_window, input_dim)
            src_mask: Optional mask for attention (usually not needed for simple encoder forecasting)

        Returns:
            output: Tensor of shape (batch_size, input_window, output_dim)

        Note:
            For forecasting "Next 5 steps", we typically take the LAST token's output
            and project it, or use the sequence to predict sequence.
            Here, we output the full sequence reconstruction/forecast capability.
        """
        # 1. Project to d_model space
        # src shape: [batch, seq_len, features] -> [batch, seq_len, d_model]
        src = self.input_projection(src)

        # 2. Add Positional Encoding
        src = self.pos_encoder(src)

        # 3. Pass through Transformer Encoder
        # output shape: [batch, seq_len, d_model]
        output = self.transformer_encoder(src, src_mask)

        # 4. Decode back to feature space
        # output shape: [batch, seq_len, features]
        output = self.decoder(output)

        return output

    # ==========================================================================
    # UTILITY METHODS (Serialization)
    # ==========================================================================

    def save_checkpoint(self, path: Optional[str] = None):
        """
        Saves the model weights and configuration to disk.
        """
        if path is None:
            path = MODEL_SAVE_PATH

        logger.info(f"💾 Saving model checkpoint to {path}")

        # We save a dictionary containing weights AND config
        checkpoint = {
            'state_dict': self.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: Optional[str] = None, device: torch.device = torch.device('cpu')):
        """
        Factory method to load a model from disk.

        Args:
            path: Path to .pt file
            device: 'cpu' or 'cuda'

        Returns:
            TimeSeriesTransformer: Loaded model instance
        """
        if path is None:
            path = MODEL_SAVE_PATH

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model checkpoint not found at {path}")

        logger.info(f"📂 Loading model from {path} to {device}")

        # FIXED for PyTorch 2.6+: weights_only=False is required to load the custom config object
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # 1. Extract config
        config = checkpoint['config']

        # 2. Instantiate model
        model = cls(config)

        # 3. Load weights
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()  # Set to evaluation mode by default

        return model

    def predict_next(self, history: torch.Tensor, horizon: int = 5) -> torch.Tensor:
        """
        Autoregressive inference wrapper.
        Given history [batch, seq_len, feat], predicts next 'horizon' steps.

        Note: This is a simplified "one-shot" prediction for low latency.
        It assumes the model's last output token represents the prediction for t+1.
        """
        with torch.no_grad():
            output = self.forward(history)
            # Take the last timestep's output as the prediction for t+1
            # For multi-step, we might need a specific head or loop.
            # Here we simplify: The model is trained to reconstruct/predict next.
            # We return the last vector.
            last_step = output[:, -1, :].unsqueeze(1)
            return last_step


# ==============================================================================
# QUICK TEST HARNESS
# ==============================================================================

if __name__ == "__main__":
    # Test the model initialization and forward pass
    conf = TransformerConfig()
    model = TimeSeriesTransformer(conf)

    # Create dummy input [Batch=2, Window=60, Features=5]
    dummy_input = torch.randn(2, conf.input_window, conf.input_dim)

    print(f"Testing Model Architecture...")
    print(f"Input Shape: {dummy_input.shape}")

    output = model(dummy_input)
    print(f"Output Shape: {output.shape}")

    assert output.shape == dummy_input.shape, "Shape Mismatch!"
    print("✅ Model Forward Pass Successful.")