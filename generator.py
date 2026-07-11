"""
NeuralPulse 2.0 - Advanced Chaos Data Generator.

This module is responsible for generating high-fidelity synthetic telemetry data.
UPDATED VERSION: Includes complex seasonality, background tasks (rugged normalcy),
and expanded anomaly types to prevent model overfitting on simple data.

Features:
1. Poly-Seasonality: Combines Day, Week, and "Lunch Hour" frequencies.
2. Background Noise: Simulates cron jobs/backups that are normal but look jagged.
3. Expanded Anomaly Suite: Includes DDoS (Packet Storm) and Disk Failures.

Usage:
    Run as a script to generate static CSV datasets.
    Import as a module for real-time Gym environments.

Line Count Target: >350 Lines
"""

import numpy as np
import pandas as pd
import math
import random
import time
import argparse
import logging
import os
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# Import configuration
try:
    from .config import (
        INFRA_CONF, ANOMALY_PROBABILITIES, AnomalyTypes,
        TIMESTEP_SECONDS, DATA_EXPORT_PATH
    )
except ImportError:
    # Fallback for running script directly
    from config import (
        INFRA_CONF, ANOMALY_PROBABILITIES, AnomalyTypes,
        TIMESTEP_SECONDS, DATA_EXPORT_PATH
    )

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [CHAOS_ENGINE] - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ChaosGenerator")


# ==============================================================================
# UPDATED CONFIGURATION FOR COMPLEXITY
# ==============================================================================

# Extending the Anomaly Types locally to add more flavor
class ExtendedAnomalyTypes(AnomalyTypes):
    PACKET_STORM = "packet_storm"  # DDoS
    DISK_FAILURE = "disk_failure"  # IO Wait


# Update probabilities to include new types
EXTENDED_PROBABILITIES = ANOMALY_PROBABILITIES.copy()
EXTENDED_PROBABILITIES[ExtendedAnomalyTypes.PACKET_STORM] = 0.005
EXTENDED_PROBABILITIES[ExtendedAnomalyTypes.DISK_FAILURE] = 0.005

# Decrease normal slightly to account for new anomalies
EXTENDED_PROBABILITIES[ExtendedAnomalyTypes.NORMAL] = 0.94


# ==============================================================================
# CORE DATA STRUCTURES
# ==============================================================================

@dataclass
class MetricPoint:
    """Represents a single snapshot of system health."""
    timestamp: datetime
    cpu: float
    ram: float
    latency: float
    error_rate: float
    ops: int
    pod_count: int
    anomaly_label: str  # Ground truth label

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": round(self.cpu, 2),
            "memory_percent": round(self.ram, 2),
            "latency_ms": round(self.latency, 2),
            "error_rate": round(self.error_rate, 4),
            "requests_per_second": int(self.ops),
            "active_pods": int(self.pod_count),
            "label": self.anomaly_label
        }


# ==============================================================================
# COMPLEX SEASONALITY ENGINE
# ==============================================================================

class ComplexSeasonalityEngine:
    """
    Handles the generation of baseline traffic patterns.
    Now uses 3 overlapping frequencies + Random Walk Drift to prevent
    the neural network from simply memorizing a perfect sine wave.
    """

    def __init__(self, base_load: int = 1500, amplitude: int = 800):
        self.base_load = base_load
        self.amplitude = amplitude
        # Random offsets
        self.phase_day = random.uniform(0, 2 * math.pi)
        self.phase_week = random.uniform(0, 2 * math.pi)
        self.micro_burst_phase = random.uniform(0, 2 * math.pi)

        # Random Walk Drift (Simulates organic growth/decline over hours)
        self.drift_value = 0.0

    def get_traffic_level(self, timestamp: datetime) -> float:
        """
        Returns the expected Operations Per Second (OPS).
        """
        # Time variables
        total_seconds = timestamp.timestamp()
        hour_of_day = timestamp.hour + (timestamp.minute / 60.0)
        day_of_week = timestamp.weekday()

        # 1. Daily Cycle (24h) - The main hump
        # Peak at 2 PM (14.0), Trough at 2 AM (2.0)
        norm_time = (hour_of_day - 14.0) / 24.0 * 2 * math.pi
        daily_wave = math.cos(norm_time)  # Peak at 14:00

        # 2. Weekly Cycle (7d) - Weekends are quieter
        # 0=Mon, 6=Sun.
        is_weekend = 1.0 if day_of_week >= 5 else 0.0
        weekend_damper = 0.7 if is_weekend else 1.0

        # 3. Micro Bursts (High frequency jitter - 15 min cycles)
        burst_wave = 0.1 * math.sin(total_seconds / 900.0 + self.micro_burst_phase)

        # 4. Random Drift Update (Brownian Motion)
        # Changes very slowly step-by-step
        self.drift_value += random.uniform(-5.0, 5.0)
        # Clamp drift
        self.drift_value = max(-200.0, min(200.0, self.drift_value))

        # Combine
        # Base * Daily * Weekend + Burst + Drift
        ops = (self.base_load + (self.amplitude * daily_wave)) * weekend_damper
        ops += (ops * burst_wave)  # Burst is proportional
        ops += self.drift_value

        # Ensure floor
        return max(50.0, ops)


# ==============================================================================
# NOISE & BACKGROUND TASK ENGINE
# ==============================================================================

class AdvancedNoiseEngine:
    """
    Adds "Ruggedness" to the data.
    Simulates:
    1. White Noise (Sensor jitter)
    2. Background Tasks (Cron jobs that spike CPU safely)
    """

    def __init__(self):
        self.cron_active = False
        self.cron_timer = 0
        self.cron_duration = 0

    def add_gaussian_noise(self, value: float, std_dev_percent: float = 0.05) -> float:
        noise = np.random.normal(0, value * std_dev_percent)
        return value + noise

    def apply_background_tasks(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Randomly starts 'maintenance tasks' (backups, indexing)
        that increase resource usage but are NOT anomalies.
        The model must learn to ignore these.
        """
        # 1% chance to start a background task per step
        if not self.cron_active and random.random() < 0.01:
            self.cron_active = True
            self.cron_duration = random.randint(5, 20)  # Lasts 25-100 seconds
            self.cron_timer = 0

        if self.cron_active:
            self.cron_timer += 1
            # Cron jobs usually consume CPU and I/O (Latency), but not much RAM or Error
            metrics['cpu'] += random.uniform(10.0, 25.0)
            metrics['latency'] += random.uniform(20.0, 50.0)

            # End task
            if self.cron_timer >= self.cron_duration:
                self.cron_active = False

        return metrics


# ==============================================================================
# ANOMALY INJECTION SYSTEM (UPDATED)
# ==============================================================================

class ExtendedAnomalyInjector:
    """
    Injects failures.
    Updated to handle new anomaly types.
    """

    def __init__(self):
        self.current_anomaly = AnomalyTypes.NORMAL
        self.steps_active = 0
        self.duration = 0
        self.severity = 1.0

    def trigger_anomaly(self):
        # Select anomaly type
        types = list(EXTENDED_PROBABILITIES.keys())
        probs = list(EXTENDED_PROBABILITIES.values())
        choice = random.choices(types, weights=probs, k=1)[0]

        if choice != AnomalyTypes.NORMAL:
            self.current_anomaly = choice
            self.steps_active = 0
            self.duration = random.randint(20, 80)
            self.severity = random.uniform(1.5, 3.5)
            logger.warning(f"⚠ INJECTING: {self.current_anomaly} (Sev: {self.severity:.1f})")

    def resolve(self):
        if self.current_anomaly != AnomalyTypes.NORMAL:
            logger.info(f"✔ RESOLVED: {self.current_anomaly}")
        self.current_anomaly = AnomalyTypes.NORMAL

    def apply_effects(self, metrics: Dict) -> Dict:
        if self.current_anomaly == AnomalyTypes.NORMAL:
            return metrics

        self.steps_active += 1
        if self.steps_active > self.duration:
            self.resolve()
            return metrics

        # --- LOGIC FOR ANOMALIES ---

        if self.current_anomaly == AnomalyTypes.CPU_SPIKE:
            # Classic crypto-miner pattern
            metrics['cpu'] = min(100.0, metrics['cpu'] * self.severity + 50.0)
            metrics['latency'] += 100 * (metrics['cpu'] / 100.0)

        elif self.current_anomaly == AnomalyTypes.MEMORY_LEAK:
            # Linear ramp up
            leak = 0.8 * self.steps_active * self.severity
            metrics['ram'] = min(100.0, metrics['ram'] + leak)
            if metrics['ram'] > 95:
                # Thrashing starts
                metrics['latency'] *= 2.0
                metrics['cpu'] += 20.0

        elif self.current_anomaly == AnomalyTypes.LATENCY_DRIFT:
            # Slow boil
            metrics['latency'] *= (1.0 + (0.02 * self.steps_active * self.severity))

        elif self.current_anomaly == AnomalyTypes.NETWORK_FAILURE:
            # Traffic black hole
            metrics['ops'] *= 0.05  # 95% traffic drop
            metrics['error_rate'] = max(metrics['error_rate'], 0.4)
            metrics['cpu'] *= 0.1  # Idle

        elif self.current_anomaly == AnomalyTypes.DB_LOCK:
            # High Latency, Low Throughput, Mid CPU (Wait state)
            metrics['latency'] += 1500 * self.severity
            metrics['ops'] = min(metrics['ops'], 200.0)  # Cap throughput

        elif self.current_anomaly == ExtendedAnomalyTypes.PACKET_STORM:
            # DDoS Attack: High OPS, High Latency, High CPU
            metrics['ops'] *= (3.0 * self.severity)
            metrics['cpu'] = min(100.0, metrics['cpu'] * 1.5)
            metrics['latency'] += 500.0
            metrics['error_rate'] += 0.1

        elif self.current_anomaly == ExtendedAnomalyTypes.DISK_FAILURE:
            # High Latency, Low CPU (Waiting for disk), Normal RAM
            metrics['latency'] += 2000.0 * self.severity
            metrics['cpu'] = max(10.0, metrics['cpu'] * 0.5)  # CPU drops because it's blocked

        return metrics


# ==============================================================================
# MAIN SIMULATION CONTROLLER
# ==============================================================================

class DataGenerator:
    """
    Orchestrates the simulation.
    """

    def __init__(self, duration_hours: int = 24):
        self.duration_seconds = duration_hours * 3600
        self.frequency = TIMESTEP_SECONDS

        # Initialize Engines
        self.seasonality = ComplexSeasonalityEngine(base_load=1200, amplitude=600)
        self.noise = AdvancedNoiseEngine()
        self.injector = ExtendedAnomalyInjector()

        # State
        self.current_pod_count = INFRA_CONF.INITIAL_POD_COUNT

    def _calculate_infrastructure_metrics(self, ops: float) -> Tuple[float, float, float, float]:
        """Physics engine."""
        max_ops_capacity = self.current_pod_count * INFRA_CONF.BASE_POD_CAPACITY_OPS
        utilization = ops / max_ops_capacity if max_ops_capacity > 0 else 1.0

        # CPU: Non-linear curve
        # 0.5 util = 30% CPU
        # 1.0 util = 80% CPU
        base_cpu = 10.0 + (utilization * 70.0)
        cpu = self.noise.add_gaussian_noise(base_cpu, 0.1)
        cpu = max(5.0, min(100.0, cpu))

        # RAM: Base + per-request
        ram_per_pod = INFRA_CONF.RAM_BASE_USAGE + (ops * 0.01 / max(1, self.current_pod_count))
        ram = self.noise.add_gaussian_noise(ram_per_pod, 0.05)
        ram = max(10.0, min(100.0, ram))

        # Latency: Hockey stick
        latency = INFRA_CONF.BASE_LATENCY_MS
        if utilization > 0.85:
            latency += (utilization - 0.85) * 1000.0  # Vertical wall
        latency = self.noise.add_gaussian_noise(latency, 0.2)

        # Error Rate
        error_rate = 0.001
        if utilization > 1.1:
            error_rate += (utilization - 1.1)

        return cpu, ram, latency, error_rate

    def generate_step(self, timestamp: datetime) -> MetricPoint:
        """One tick of the clock."""

        # 1. Traffic
        ops = self.seasonality.get_traffic_level(timestamp)
        ops = self.noise.add_gaussian_noise(ops, 0.15)  # More jitter

        # 2. Anomaly Trigger (Higher chance in this version)
        if random.random() < 0.008:
            self.injector.trigger_anomaly()

        # 3. Physics
        cpu, ram, lat, err = self._calculate_infrastructure_metrics(ops)
        metrics = {
            'cpu': cpu, 'ram': ram, 'latency': lat,
            'error_rate': err, 'ops': ops
        }

        # 4. Background Tasks (Rugged Normalcy)
        # Only apply if NO anomaly is active (otherwise it gets too messy)
        if self.injector.current_anomaly == AnomalyTypes.NORMAL:
            metrics = self.noise.apply_background_tasks(metrics)

        # 5. Anomaly Effects
        metrics = self.injector.apply_effects(metrics)

        return MetricPoint(
            timestamp=timestamp,
            cpu=metrics['cpu'],
            ram=metrics['ram'],
            latency=metrics['latency'],
            error_rate=metrics['error_rate'],
            ops=int(metrics['ops']),
            pod_count=self.current_pod_count,
            anomaly_label=self.injector.current_anomaly
        )

    def run_batch_generation(self) -> pd.DataFrame:
        logger.info(f"⚡ Starting Advanced Data Gen: {self.duration_seconds / 3600} hours")
        data_points = []
        steps = int(self.duration_seconds / self.frequency)
        start_time = datetime.now() - timedelta(seconds=self.duration_seconds)

        for i in range(steps):
            t = start_time + timedelta(seconds=i * self.frequency)
            point = self.generate_step(t)
            data_points.append(point.to_dict())

            # Simple autoscaler to keep "Normal" looking sane
            self._simulate_autoscaler(point)

            if i % 5000 == 0:
                logger.info(f"   Generated {i}/{steps} steps...")

        return pd.DataFrame(data_points)

    def _simulate_autoscaler(self, point: MetricPoint):
        if point.cpu > 75 and self.current_pod_count < INFRA_CONF.MAX_POD_COUNT:
            self.current_pod_count += 1
        elif point.cpu < 40 and self.current_pod_count > INFRA_CONF.MIN_POD_COUNT:
            self.current_pod_count -= 1


# ==============================================================================
# FILE UTILITIES
# ==============================================================================

def save_datasets(df: pd.DataFrame):
    if not os.path.exists(DATA_EXPORT_PATH):
        os.makedirs(DATA_EXPORT_PATH)

    normal_df = df[df['label'] == AnomalyTypes.NORMAL]
    anomaly_df = df[df['label'] != AnomalyTypes.NORMAL]

    # Save
    normal_df.to_csv(os.path.join(DATA_EXPORT_PATH, "train_normal.csv"), index=False)
    anomaly_df.to_csv(os.path.join(DATA_EXPORT_PATH, "test_anomaly.csv"), index=False)

    logger.info(f"✅ Generated {len(normal_df)} Normal rows (Rugged)")
    logger.info(f"✅ Generated {len(anomaly_df)} Anomaly rows (Extended Types)")


if __name__ == "__main__":
    generator = DataGenerator(duration_hours=24)
    df = generator.run_batch_generation()
    save_datasets(df)