"""
NeuralPulse 2.0 - Reflex Brain Configuration.

This module acts as the central command center for the Phase 01 Reflex System.
It defines the hyperparameters for the Transformer model, training settings,
data processing pipelines, and anomaly detection thresholds.

Robustness features:
- Automatic path resolution (works regardless of where script is run).
- Centralized device management (CPU/CUDA/MPS).
- Feature mapping to ensure input vectors match the simulation schema.
"""

import os
import torch
import logging
from pathlib import Path
from dataclasses import dataclass

# Setup Logging for Phase 01
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [REFLEX_BRAIN] - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ReflexConfig")


# ==============================================================================
# PATH MANAGEMENT
# ==============================================================================

def get_project_root() -> Path:
    """
    Intelligently finds the project root directory.
    Assumes this file is at neuralpulse-core/01_reflex_brain/reflex_config.py
    """
    current_file = Path(__file__).resolve()
    # Go up two levels: 01_reflex_brain -> neuralpulse-core
    return current_file.parent.parent


PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "00_simulation" / "data"
MODEL_DIR = PROJECT_ROOT / "01_reflex_brain" / "checkpoints"
LOG_DIR = PROJECT_ROOT / "01_reflex_brain" / "logs"

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# File Targets
TRAIN_DATA_PATH = DATA_DIR / "train_normal.csv"
TEST_DATA_PATH = DATA_DIR / "test_anomaly.csv"
MODEL_SAVE_PATH = MODEL_DIR / "observer_model.pt"
SCALER_SAVE_PATH = MODEL_DIR / "scaler_state.pkl"


# ==============================================================================
# MODEL HYPERPARAMETERS
# ==============================================================================

@dataclass
class TransformerConfig:
    """
    Configuration for the Time-Series Transformer Encoder.
    Optimized for low-latency inference (<10ms).
    """
    # Input/Output Dimensions
    input_dim: int = 5  # CPU, RAM, Latency, Error, OPS
    output_dim: int = 5  # Predicting the same features (Forecasting)

    # Sequence Lengths
    input_window: int = 60  # Lookback window (e.g., 60 steps = 5 minutes)
    forecast_horizon: int = 5  # Predict next 5 steps

    # Architecture Specs
    d_model: int = 64  # Hidden dimension (Keep small for speed)
    nhead: int = 4  # Number of attention heads
    num_layers: int = 2  # Number of transformer encoder layers
    dim_feedforward: int = 128  # FFN expansion dimension
    dropout: float = 0.1  # Regularization

    # Activation
    activation: str = "gelu"  # Modern activation function

    # Positional Encoding
    max_len: int = 5000  # Max sequence length support


# ==============================================================================
# TRAINING HYPERPARAMETERS
# ==============================================================================

@dataclass
class TrainingConfig:
    """
    Configuration for the training loop.
    """
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    epochs: int = 20

    # Early Stopping
    patience: int = 5
    min_delta: float = 0.0001

    # Data Splitting
    validation_split: float = 0.2

    # Workers
    num_workers: int = 0  # Set to 0 for maximum compatibility on Windows


# ==============================================================================
# ANOMALY DETECTION SETTINGS
# ==============================================================================

@dataclass
class AnomalyConfig:
    """
    Settings for the thresholding logic.
    """
    # The percentile of training error to use as the threshold.
    # e.g., 99.9th percentile ensures < 0.1% false positives on clean data.
    threshold_percentile: float = 99.9

    # Smoothing factor for error calculation (Exponential Moving Average)
    error_smoothing: float = 0.5

    # Minimum error required to trigger (prevents triggering on noise)
    min_error_cutoff: float = 0.05


# ==============================================================================
# DEVICE CONFIGURATION
# ==============================================================================

def get_device() -> torch.device:
    """
    Automatically selects the best available hardware.
    Prioritizes CUDA -> MPS (Mac) -> CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"🚀 Acceleration: CUDA Enabled ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("🚀 Acceleration: Apple Metal (MPS) Enabled")
    else:
        device = torch.device("cpu")
        logger.warning("⚠️ Acceleration: None. Running on CPU (Slower training).")
    return device


DEVICE = get_device()
TX_CONF = TransformerConfig()
TRAIN_CONF = TrainingConfig()
ANOM_CONF = AnomalyConfig()

# Check for data existence
if not TRAIN_DATA_PATH.exists():
    logger.error(f"❌ CRITICAL: Training data not found at {TRAIN_DATA_PATH}")
    logger.error("   Run 'python 00_simulation/generator.py' first!")
else:
    logger.info(f"✅ Training data found at {TRAIN_DATA_PATH}")

# ==============================================================================
# FEATURE MAPPING (The Fix)
# ==============================================================================
# These must match the headers in train_normal.csv exactly
FEATURE_COLUMNS = [
    'cpu_percent',
    'memory_percent',
    'latency_ms',
    'error_rate',
    'requests_per_second'
]
