"""
NeuralPulse 2.0 - RL Training Pipeline.

Fixed for numbered directories (00_simulation) using dynamic importlib.
"""

import sys
import os
import time
import importlib.util
from pathlib import Path

# Add current directory to path so we can import rl_config
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# Import Configs
try:
    from rl_config import (
        PPO_CONF, TRAIN_CONF, MODEL_PATH, LOG_DIR, MODEL_DIR
    )
except ImportError:
    from .rl_config import (
        PPO_CONF, TRAIN_CONF, MODEL_PATH, LOG_DIR, MODEL_DIR
    )


class ProgressBarCallback(BaseCallback):
    """
    Custom callback for printing training progress to console.
    """

    def __init__(self, check_freq: int, verbose=1):
        super(ProgressBarCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            elapsed = time.time() - self.start_time
            fps = int(self.num_timesteps / (elapsed + 1e-5))
            print(f"   Step: {self.num_timesteps} / {TRAIN_CONF.total_timesteps} | FPS: {fps}")
        return True


def make_env():
    """
    Dynamically loads NeuralPulseEnv from 00_simulation/environment_gym.py
    This bypasses the issue where Python cannot import folders starting with numbers.
    """
    # Fix: Add 00_simulation to sys.path so environment_gym can import generator/config
    sim_dir = os.path.join(PROJECT_ROOT, "00_simulation")
    if sim_dir not in sys.path:
        sys.path.append(sim_dir)

    env_path = os.path.join(sim_dir, "environment_gym.py")

    spec = importlib.util.spec_from_file_location("environment_gym", env_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["environment_gym"] = module
    spec.loader.exec_module(module)

    # Return an instance of the class
    return module.NeuralPulseEnv()


def train_agent():
    print("=" * 60)
    print("🤖 NeuralPulse RL Training Module")
    print("=" * 60)

    # 1. Create Environment
    print("Creating Training Environment...")
    # stable-baselines3 requires a callable that returns the env
    env = make_vec_env(make_env, n_envs=1)

    # 2. Setup Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=TRAIN_CONF.save_freq,
        save_path=str(MODEL_DIR),
        name_prefix="ppo_checkpoint"
    )
    progress_callback = ProgressBarCallback(check_freq=1000)

    # 3. Initialize Agent
    print(f"Initializing PPO Agent (Device: {PPO_CONF.policy_type})...")
    model = PPO(
        policy=PPO_CONF.policy_type,
        env=env,
        learning_rate=PPO_CONF.learning_rate,
        n_steps=PPO_CONF.n_steps,
        batch_size=PPO_CONF.batch_size,
        n_epochs=PPO_CONF.n_epochs,
        gamma=PPO_CONF.gamma,
        gae_lambda=PPO_CONF.gae_lambda,
        clip_range=PPO_CONF.clip_range,
        ent_coef=PPO_CONF.ent_coef,
        verbose=1,
        tensorboard_log=str(LOG_DIR)
    )

    # 4. Start Training
    print(f"🚀 Starting Training for {TRAIN_CONF.total_timesteps} steps...")
    start_time = time.time()

    model.learn(
        total_timesteps=TRAIN_CONF.total_timesteps,
        callback=[checkpoint_callback, progress_callback]
    )

    total_time = time.time() - start_time
    print(f"✅ Training Complete in {total_time:.1f}s")

    # 5. Save Final Model
    print(f"💾 Saving model to {MODEL_PATH}")
    model.save(MODEL_PATH)


if __name__ == "__main__":
    train_agent()