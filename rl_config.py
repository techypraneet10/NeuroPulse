"""
NeuralPulse 2.0 - Adaptive Brain Configuration.

This module defines the hyperparameters for the Reinforcement Learning (RL) Agent.
It controls:
1. PPO Algorithm Settings (Learning Rate, Gamma, Clip Range).
2. Reward Function Sensitivity (Penalty weights for latency vs cost).
3. Action Space Definitions.
4. Path Management for Model Persistence.

Robustness Features:
- Centralized Path Resolution.
- Device Agnostic (CPU/CUDA).
- Tensorboard Logging Config.

Line Count Target: High (Detailed Param Documentation)
"""

import os
import torch
import logging
from pathlib import Path
from dataclasses import dataclass

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [ADAPTIVE_BRAIN] - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RLConfig")


# ==============================================================================
# PATH MANAGEMENT
# ==============================================================================
def get_project_root() -> Path:
    current_file = Path(__file__).resolve()
    return current_file.parent.parent


PROJECT_ROOT = get_project_root()
MODEL_DIR = PROJECT_ROOT / "03_adaptive_brain" / "policy_registry"
LOG_DIR = PROJECT_ROOT / "03_adaptive_brain" / "logs"

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

MODEL_PATH = MODEL_DIR / "ppo_agent.zip"
BEST_MODEL_PATH = MODEL_DIR / "best_model.zip"


# ==============================================================================
# RL HYPERPARAMETERS (PPO)
# ==============================================================================
@dataclass
class PPOConfig:
    """
    Configuration for the Proximal Policy Optimization algorithm.
    Tuned for stability in continuous control environments.
    """
    policy_type: str = "MlpPolicy"  # Multi-Layer Perceptron (Dense NN)
    learning_rate: float = 3e-4  # Standard start point
    n_steps: int = 2048  # Steps per update buffer
    batch_size: int = 64
    n_epochs: int = 10  # Optimization passes per update
    gamma: float = 0.99  # Discount factor (Future reward importance)
    gae_lambda: float = 0.95  # Generalized Advantage Estimation
    clip_range: float = 0.2  # PPO Clipping (Prevents drastic policy shifts)
    ent_coef: float = 0.01  # Entropy coefficient (Encourages exploration)
    vf_coef: float = 0.5  # Value Function coefficient
    max_grad_norm: float = 0.5  # Gradient clipping


# ==============================================================================
# TRAINING SETTINGS
# ==============================================================================
@dataclass
class TrainConfig:
    total_timesteps: int = 50_000  # Total simulation steps to train
    eval_freq: int = 1000  # How often to evaluate the agent
    save_freq: int = 5000  # How often to save checkpoints
    verbose: int = 1  # 0=Silent, 1=Info, 2=Debug


# ==============================================================================
# ACTION SPACE MAPPING
# ==============================================================================
# Must match 00_simulation/config.py
ACTION_MAP = {
    0: "DO_NOTHING",
    1: "SCALE_UP_SMALL",  # +2 Pods
    2: "SCALE_UP_LARGE",  # +10 Pods
    3: "SCALE_DOWN_SMALL",  # -2 Pods
    4: "REROUTE_TRAFFIC",  # Costly but instant latency fix
    5: "CLEAR_CACHE"  # Risky (CPU spike) but frees RAM
}


# ==============================================================================
# REWARD WEIGHTS
# ==============================================================================
@dataclass
class RewardConfig:
    """
    Fine-tuning the reward signal is critical.
    The agent wants to MAXIMIZE this score.
    """
    # Base Penalties
    latency_penalty_weight: float = -0.5  # Per ms over threshold
    error_penalty_weight: float = -50.0  # Per % of error rate
    cpu_penalty_weight: float = -0.1  # Per % CPU over 90%

    # Cost Penalties
    action_cost_scale: float = 1.0  # Multiplier for action costs
    pod_cost_weight: float = -0.05  # Cost per active pod per step

    # Bonuses
    stability_bonus: float = 2.0  # Reward for being "Green"
    recovery_bonus: float = 10.0  # Reward for exiting "Red" state


# ==============================================================================
# DEVICE CONFIGURATION
# ==============================================================================
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = get_device()
PPO_CONF = PPOConfig()
TRAIN_CONF = TrainConfig()
REWARD_CONF = RewardConfig()