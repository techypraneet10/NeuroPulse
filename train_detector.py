"""
NeuralPulse 2.0 - Reflex Brain Trainer.

This script manages the training pipeline for the Anomaly Detection Transformer.
It adheres to MLOps best practices:
1. Data Scaling: MinMax scaling to [0,1] based on simulation configs.
2. Sliding Window Dataset: Converts flat CSV rows into (Window, Target) pairs.
3. Training Loop: Includes validation split, early stopping, and loss logging.
4. Artifact Management: Saves both the Model weights and the Scaler state.

Usage:
    python 01_reflex_brain/train_detector.py

Line Count Target: High (Dataset Logic + Training Logic + Metric Calculation)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
import logging
import time
import os
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Dict

try:
    from .reflex_config import (
        TX_CONF, TRAIN_CONF, TRAIN_DATA_PATH, MODEL_SAVE_PATH,
        SCALER_SAVE_PATH, FEATURE_COLUMNS, DEVICE
    )
    from .transformer_model import TimeSeriesTransformer
except ImportError:
    from reflex_config import (
        TX_CONF, TRAIN_CONF, TRAIN_DATA_PATH, MODEL_SAVE_PATH,
        SCALER_SAVE_PATH, FEATURE_COLUMNS, DEVICE
    )
    from transformer_model import TimeSeriesTransformer

logger = logging.getLogger("ReflexTrainer")


# ==============================================================================
# DATASET & PREPROCESSING
# ==============================================================================

class WindowedDataset(Dataset):
    """
    PyTorch Dataset that creates sliding windows over time-series data.

    Task: Forecasting / Reconstruction.
    Input (X): Sequence of length 'input_window' (e.g., t=0 to t=59)
    Target (Y): The sequence shifted by 1 (e.g., t=1 to t=60) OR same sequence for Autoencoder.

    Here, we use "Next Step Prediction" logic.
    Target is the input sequence shifted by 1 step into the future.
    """

    def __init__(self, data: np.ndarray, window_size: int):
        self.data = torch.FloatTensor(data)
        self.window_size = window_size
        self.n_samples = len(data) - window_size

    def __len__(self):
        return max(0, self.n_samples)

    def __getitem__(self, idx):
        # Input: Steps [idx, idx + window]
        # Target: Steps [idx+1, idx + window + 1] (Next step prediction)

        # We ensure we don't go out of bounds
        x = self.data[idx: idx + self.window_size]
        y = self.data[idx + 1: idx + self.window_size + 1]

        # If we reach end of buffer, handled by __len__, but safeguard:
        if len(y) < self.window_size:
            y = self.data[idx: idx + self.window_size]  # Fallback

        return x, y


def load_and_process_data() -> Tuple[DataLoader, DataLoader, MinMaxScaler]:
    """
    Loads CSV, fits Scaler, creates DataLoaders.
    """
    logger.info(f"📥 Loading training data from {TRAIN_DATA_PATH}...")

    try:
        df = pd.read_csv(TRAIN_DATA_PATH)
    except FileNotFoundError:
        logger.error("❌ Data file not found. Please run Phase 00 generator.")
        raise

    # Extract feature columns
    try:
        raw_data = df[FEATURE_COLUMNS].values
    except KeyError as e:
        logger.error(f"❌ Columns mismatch! Config expects: {FEATURE_COLUMNS}")
        logger.error(f"   CSV contains: {list(df.columns)}")
        raise e

    logger.info(f"📊 Data Shape: {raw_data.shape}")

    # Scale Data to [0, 1] range
    # Crucial for Neural Network convergence
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(raw_data)

    # Save scaler for inference usage
    with open(SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"💾 Scaler saved to {SCALER_SAVE_PATH}")

    # Split Train/Val
    split_idx = int(len(scaled_data) * (1 - TRAIN_CONF.validation_split))
    train_data = scaled_data[:split_idx]
    val_data = scaled_data[split_idx:]

    # Create Datasets
    train_ds = WindowedDataset(train_data, TX_CONF.input_window)
    val_ds = WindowedDataset(val_data, TX_CONF.input_window)

    # Create DataLoaders
    # Note: num_workers=0 is safer for Windows/Interactive environments
    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_CONF.batch_size,
        shuffle=True,
        num_workers=TRAIN_CONF.num_workers
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=TRAIN_CONF.batch_size,
        shuffle=False,
        num_workers=TRAIN_CONF.num_workers
    )

    return train_loader, val_loader, scaler


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train_model():
    """
    Main execution function.
    """
    # 1. Setup
    train_loader, val_loader, _ = load_and_process_data()

    # 2. Initialize Model
    model = TimeSeriesTransformer(TX_CONF).to(DEVICE)
    logger.info(f"🧠 Initialized Transformer Model on {DEVICE}")
    logger.info(f"   Structure: {TX_CONF.num_layers} Layers, {TX_CONF.nhead} Heads, {TX_CONF.d_model} Hidden Dim")

    # 3. Optimizer & Loss
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONF.learning_rate,
        weight_decay=TRAIN_CONF.weight_decay
    )

    # Scheduler for smoother convergence
    # FIXED: Removed 'verbose=True' which causes crashes in newer PyTorch versions
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # 4. Loop Variables
    best_val_loss = float('inf')
    early_stop_counter = 0
    start_time = time.time()

    logger.info("🚀 Starting Training Loop...")

    for epoch in range(TRAIN_CONF.epochs):
        model.train()
        train_loss = 0.0

        # --- Training Step ---
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass
            output = model(x)

            # Compute loss
            loss = criterion(output, y)

            # Backward pass
            loss.backward()

            # Gradient Clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Step ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            # In PyTorch, iterating DataLoader yields batches directly
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(DEVICE), y_val.to(DEVICE)
                output = model(x_val)
                loss = criterion(output, y_val)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # --- Logging & Scheduling ---
        scheduler.step(avg_val_loss)

        # Check current LR
        current_lr = optimizer.param_groups[0]['lr']

        elapsed = time.time() - start_time

        logger.info(
            f"Epoch {epoch + 1}/{TRAIN_CONF.epochs} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        # --- Early Stopping & Checkpointing ---
        if avg_val_loss < (best_val_loss - TRAIN_CONF.min_delta):
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            model.save_checkpoint(MODEL_SAVE_PATH)
            logger.info(f"   ⭐ New Best Model Saved (Loss: {best_val_loss:.6f})")
        else:
            early_stop_counter += 1
            if early_stop_counter >= TRAIN_CONF.patience:
                logger.info("🛑 Early Stopping Triggered.")
                break

    logger.info("✅ Training Complete.")
    logger.info(f"   Final Best Validation Loss: {best_val_loss:.6f}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    try:
        train_model()
    except KeyboardInterrupt:
        logger.info("🛑 Training interrupted by user.")
    except Exception as e:
        logger.exception("❌ Training failed with error:")