"""
NeuralPulse 2.0 - Reflex Inference Engine.

This module is the runtime component of the Reflex Brain.
It is responsible for:
1. Loading the trained Transformer and Scaler.
2. Maintaining a rolling buffer of real-time data (60 steps).
3. Predicting the 'expected' state of the system.
4. Comparing 'expected' vs 'actual' to compute an Anomaly Score.
5. Emitting a standardized Vector Alert if the score exceeds threshold.

Usage:
    Can be imported by the Orchestration layer (Phase 4) or run standalone
    to process the test_anomaly.csv dataset.

Line Count Target: High (Includes buffering logic and threshold calculation)
"""

import torch
import numpy as np
import pandas as pd
import pickle
import logging
import time
from collections import deque
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

try:
    from .reflex_config import (
        MODEL_SAVE_PATH, SCALER_SAVE_PATH, TEST_DATA_PATH,
        TX_CONF, ANOM_CONF, FEATURE_COLUMNS, DEVICE
    )
    from .transformer_model import TimeSeriesTransformer
except ImportError:
    from reflex_config import (
        MODEL_SAVE_PATH, SCALER_SAVE_PATH, TEST_DATA_PATH,
        TX_CONF, ANOM_CONF, FEATURE_COLUMNS, DEVICE
    )
    from transformer_model import TimeSeriesTransformer

logger = logging.getLogger("ReflexInference")


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class VectorAlert:
    """
    The standardized output format of the Reflex Brain.
    Sent to the Reasoning Brain (Phase 2) for analysis.
    """
    timestamp: float
    is_anomaly: bool
    anomaly_score: float  # Overall MSE magnitude
    primary_metric: str  # The metric with highest deviation (e.g., 'latency')
    metric_values: Dict[str, float]  # The actual raw values
    deviations: Dict[str, float]  # The prediction errors per metric

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "status": "CRITICAL" if self.is_anomaly else "NORMAL",
            "score": round(self.anomaly_score, 4),
            "primary_cause": self.primary_metric,
            "metrics": self.metric_values,
            "deviation_vector": self.deviations
        }


# ==============================================================================
# REFLEX BRAIN CLASS
# ==============================================================================

class ReflexBrain:
    """
    The Inference Agent.

    lifecycle:
    1. init() -> Loads model & scaler.
    2. ingest() -> Takes one timestep of data.
    3. process() -> Returns Alert (if buffer full).
    """

    def __init__(self):
        self.device = DEVICE
        self.model = None
        self.scaler = None
        self.buffer = deque(maxlen=TX_CONF.input_window)

        # Thresholding state
        self.threshold = ANOM_CONF.min_error_cutoff

        # Load Resources
        self._load_resources()

    def _load_resources(self):
        """Loads trained artifacts from disk."""
        logger.info("🧠 Loading Reflex Brain resources...")

        # 1. Load Scaler
        try:
            with open(SCALER_SAVE_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info("   ✅ Scaler loaded.")
        except FileNotFoundError:
            logger.error(f"   ❌ Scaler not found at {SCALER_SAVE_PATH}. Train first!")
            raise

        # 2. Load Model
        try:
            self.model = TimeSeriesTransformer.load_checkpoint(MODEL_SAVE_PATH, self.device)
            self.model.eval()  # Inference mode
            logger.info("   ✅ Transformer model loaded.")
        except Exception as e:
            logger.error(f"   ❌ Failed to load model: {e}")
            raise

    def ingest_data(self, metrics: Dict[str, float]) -> Optional[VectorAlert]:
        """
        Main entry point for the stream.

        Args:
            metrics: Dict containing 'cpu', 'ram', 'latency', 'error_rate', 'ops'

        Returns:
            VectorAlert if buffer is full and processed, else None.
        """
        # 1. Extract and Order Features based on Config
        try:
            raw_vector = [metrics[col] for col in FEATURE_COLUMNS]
        except KeyError as e:
            logger.error(f"Missing feature column: {e}")
            return None

        # 2. Add to Buffer
        self.buffer.append(raw_vector)

        # 3. Check if ready to infer
        if len(self.buffer) < TX_CONF.input_window:
            return None  # Warming up

        # 4. Run Inference
        return self._detect_anomaly()

    def _detect_anomaly(self) -> VectorAlert:
        """
        Internal method: Runs the model on the buffer and computes error.
        """
        # 1. Prepare Input Tensor
        # Convert buffer to numpy array [60, 5]
        input_data = np.array(list(self.buffer))

        # Normalize using the loaded scaler
        scaled_input = self.scaler.transform(input_data)

        # Convert to Tensor [1, 60, 5]
        tensor_input = torch.FloatTensor(scaled_input).unsqueeze(0).to(self.device)

        # 2. Model Prediction
        with torch.no_grad():
            # Get reconstruction/forecast
            # The model outputs [1, 60, 5]. We care most about the LAST timestep
            # (reconstructing the current moment given context)
            prediction_tensor = self.model(tensor_input)

        # 3. Compute Error (MSE per feature)
        # We compare the last timestep of input (Actual) vs last timestep of output (Predicted)
        actual_last_step = tensor_input[:, -1, :]
        predicted_last_step = prediction_tensor[:, -1, :]

        # Squared Error per feature
        error_vector = torch.pow(actual_last_step - predicted_last_step, 2).cpu().numpy()[0]

        # Total Mean Squared Error
        total_mse = np.mean(error_vector)

        # 4. Construct Alert
        # Identify which metric is most broken
        max_error_idx = np.argmax(error_vector)
        primary_cause = FEATURE_COLUMNS[max_error_idx]

        # Create deviation dictionary
        deviations = {feat: float(err) for feat, err in zip(FEATURE_COLUMNS, error_vector)}

        # Get raw values of current step
        current_raw = {feat: val for feat, val in zip(FEATURE_COLUMNS, list(self.buffer)[-1])}

        # Check Threshold
        is_anomaly = total_mse > self.threshold

        return VectorAlert(
            timestamp=time.time(),
            is_anomaly=is_anomaly,
            anomaly_score=float(total_mse),
            primary_metric=primary_cause,
            metric_values=current_raw,
            deviations=deviations
        )


# ==============================================================================
# TEST RUNNER (Offline Mode)
# ==============================================================================

def run_test_on_file():
    """
    Runs the Reflex Brain against the test_anomaly.csv file generated in Phase 0.
    Simulates a live stream.
    """
    logger.info("🧪 Starting Offline Test Run on 'test_anomaly.csv'...")

    if not TEST_DATA_PATH.exists():
        logger.error("Test data not found.")
        return

    # Load test data
    df = pd.read_csv(TEST_DATA_PATH)
    brain = ReflexBrain()

    anomalies_detected = 0
    start_time = time.time()

    print("\n" + "=" * 60)
    print(f"{'TIMESTAMP':<10} | {'STATUS':<10} | {'SCORE':<8} | {'CAUSE':<15} | {'LATENCY':<8}")
    print("=" * 60)

    for i, row in df.iterrows():
        # Simulate Stream: Convert row to dict
        metrics = row.to_dict()

        # Ingest
        alert = brain.ingest_data(metrics)

        # Process output
        if alert:
            # FIX: Use 'latency_ms' instead of 'latency' to match CSV schema
            latency_val = row.get('latency_ms', row.get('latency', 0))

            if alert.is_anomaly:
                anomalies_detected += 1
                print(
                    f"{i:<10} | 🔴 ALERT   | {alert.anomaly_score:.4f}   | {alert.primary_metric:<15} | {latency_val:.0f}ms")
            elif i % 200 == 0:
                print(f"{i:<10} | 🟢 Normal  | {alert.anomaly_score:.4f}   | -               | {latency_val:.0f}ms")

        # Optional: Sleep to simulate real-time
        # time.sleep(0.01)

    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"✅ Test Complete.")
    print(f"   Processed {len(df)} rows in {elapsed:.2f}s")
    print(f"   Total Anomalies Detected: {anomalies_detected}")


if __name__ == "__main__":
    run_test_on_file()