"""
NeuralPulse 2.0 - Simulation Configuration & Physics Constants.

This module acts as the centralized configuration store for the simulation layer.
It defines the 'laws of physics' for the simulated e-commerce environment,
including thresholds for anomalies, resource consumption rates, scaling delays,
and reward function weights for the Reinforcement Learning agent.

By centralizing these values, we ensure consistency between the Data Generator
and the RL Environment.
"""

import dataclasses
from typing import Dict, Tuple, List

# ==============================================================================
# GLOBAL SIMULATION SETTINGS
# ==============================================================================
SIMULATION_SEED = 42
TIMESTEP_SECONDS = 5  # Each simulation step represents 5 seconds of real time
DEFAULT_DURATION_HOURS = 24

# ==============================================================================
# METRIC BOUNDARIES (NORMALIZATION LIMITS)
# ==============================================================================
# We define the min/max values for normalization to keep state space in [0, 1]
CPU_MIN_MAX = (0.0, 100.0)
RAM_MIN_MAX = (0.0, 100.0)
LATENCY_MIN_MAX = (10.0, 5000.0)  # ms
ERROR_RATE_MIN_MAX = (0.0, 1.0)  # 0% to 100%
OPS_MIN_MAX = (0, 10000)  # Orders per second


# ==============================================================================
# INFRASTRUCTURE PHYSICS
# ==============================================================================
@dataclasses.dataclass(frozen=True)
class InfrastructureConfig:
    """
    Defines the baseline performance characteristics of the simulated cluster.
    """
    # Baseline capacities
    BASE_POD_CAPACITY_OPS: int = 50  # Orders per second per pod
    INITIAL_POD_COUNT: int = 10
    MAX_POD_COUNT: int = 100
    MIN_POD_COUNT: int = 2

    # Latency physics
    BASE_LATENCY_MS: float = 45.0
    LATENCY_PENALTY_PER_OPS_OVERLOAD: float = 15.0  # ms added per unit of overload

    # Resource consumption
    CPU_PER_OPS: float = 0.5  # % CPU used per order
    RAM_BASE_USAGE: float = 20.0  # % RAM used just by existing
    RAM_PER_CONNECTION: float = 0.05  # % RAM per active user connection

    # Scaling physics (Simulating "Cold Starts")
    SCALE_UP_DELAY_STEPS: int = 3  # Time to spin up new pods
    SCALE_DOWN_DELAY_STEPS: int = 1

    # Database physics
    DB_CONNECTION_LIMIT: int = 5000
    DB_LATENCY_FACTOR: float = 0.002  # Latency added per active connection


INFRA_CONF = InfrastructureConfig()


# ==============================================================================
# ANOMALY DEFINITIONS
# ==============================================================================
class AnomalyTypes:
    NORMAL = "normal"
    CPU_SPIKE = "cpu_spike"
    MEMORY_LEAK = "memory_leak"
    LATENCY_DRIFT = "latency_drift"
    NETWORK_FAILURE = "network_failure"
    DB_LOCK = "db_lock"


ANOMALY_PROBABILITIES = {
    AnomalyTypes.NORMAL: 0.95,
    AnomalyTypes.CPU_SPIKE: 0.01,
    AnomalyTypes.MEMORY_LEAK: 0.01,
    AnomalyTypes.LATENCY_DRIFT: 0.015,
    AnomalyTypes.NETWORK_FAILURE: 0.005,
    AnomalyTypes.DB_LOCK: 0.01
}

# ==============================================================================
# ACTION SPACE (RL AGENT)
# ==============================================================================
# 0: Do Nothing
# 1: Scale Up Small (+2 Pods)
# 2: Scale Up Large (+10 Pods)
# 3: Scale Down Small (-2 Pods)
# 4: Reroute Traffic (Reduces Load by 30% but costs money)
# 5: Clear Cache (Reduces RAM but causes temporary latency spike)
ACTION_NAMES = {
    0: "DO_NOTHING",
    1: "SCALE_UP_SMALL",
    2: "SCALE_UP_LARGE",
    3: "SCALE_DOWN_SMALL",
    4: "REROUTE_TRAFFIC",
    5: "CLEAR_CACHE"
}


# ==============================================================================
# REWARD FUNCTION CONFIGURATION
# ==============================================================================
@dataclasses.dataclass(frozen=True)
class RewardWeights:
    """
    Weights for the RL reward function components.
    The goal is to minimize latency and error rate while optimizing costs.
    """
    # Penalties (Negative values)
    LATENCY_VIOLATION_PENALTY: float = -2.0  # Per step if latency > SLA
    ERROR_RATE_PENALTY: float = -10.0  # Per step if errors > 0
    CRASH_PENALTY: float = -100.0  # If system goes totally down

    # Costs (Negative values)
    COST_PER_POD: float = -0.05  # Operational cost per pod
    REROUTE_COST: float = -1.0  # High cost for external rerouting

    # Bonuses (Positive values)
    STABILITY_BONUS: float = 0.1  # Small reward for staying within SLA
    RECOVERY_BONUS: float = 5.0  # Reward for fixing an anomaly

    # SLA Thresholds
    MAX_ACCEPTABLE_LATENCY: float = 200.0  # ms
    MAX_ACCEPTABLE_ERROR_RATE: float = 0.01  # 1%


REWARD_CONF = RewardWeights()

# ==============================================================================
# LOGGING & DEBUG
# ==============================================================================
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATA_EXPORT_PATH = "00_simulation/data/"
MODEL_SAVE_PATH = "03_adaptive_brain/policy_registry/"


def get_feature_names() -> List[str]:
    """Returns the ordered list of features used in the state vector."""
    return [
        "cpu_percent",
        "memory_percent",
        "latency_ms",
        "error_rate",
        "requests_per_second",
        "active_pods",
        "anomaly_flag"  # 1 if anomaly is active (simulation knowledge), usually hidden
    ]

# End of Configuration