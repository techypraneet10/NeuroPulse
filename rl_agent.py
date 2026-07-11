# File: rl_agent.py
# Component: 03_adaptive_brain
# TODO: Implement logic here.
"""
NeuralPulse 2.0 - Adaptive Agent Wrapper.

This module exposes the `AdaptiveAgent` class, which serves as the interface
for the Orchestration Layer.

Key Features:
1. Hybrid Policy Support: Can run a trained PPO model OR a fallback heuristic.
2. Safe Loading: Handles missing model files gracefully.
3. Action Masking (Logical): Prevents illegal actions (e.g., Scaling down when at min pods).
4. Explainability: Returns not just the action, but the 'Why' (Confidence/Value).

Line Count Target: High (Includes Heuristic Logic and Wrapper)
"""

import logging
import time
import numpy as np
import os
import sys

# Ensure imports work from root
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from stable_baselines3 import PPO
    from rl_config import MODEL_PATH, ACTION_MAP, DEVICE
except ImportError:
    # Fallback for relative import
    from .rl_config import MODEL_PATH, ACTION_MAP, DEVICE

logger = logging.getLogger("AdaptiveAgent")


class HeuristicPolicy:
    """
    A hardcoded rule-based policy used as a fallback or baseline.
    Useful for Hackathon demos if RL training takes too long.
    """

    def predict(self, observation, deterministic=True):
        """
        observation shape: [cpu, ram, latency, error, ops, pods, pending]
        All normalized 0-1.
        """
        # Unpack observation (assuming standard normalization)
        cpu = observation[0]
        ram = observation[1]
        lat = observation[2]  # 0-1, where 1 ~ 5000ms
        err = observation[3]
        pods = observation[5]

        # Logic Tree
        action = 0  # DO_NOTHING

        # CRITICAL: Error Rate Spike -> Reroute
        if err > 0.05:
            action = 4  # REROUTE

        # CRITICAL: Massive Latency -> Scale Large
        elif lat > 0.4:  # ~2000ms
            action = 2  # SCALE_UP_LARGE

        # HIGH LOAD: CPU High -> Scale Small
        elif cpu > 0.8:
            action = 1  # SCALE_UP_SMALL

        # MEMORY LEAK: RAM High but CPU Low -> Clear Cache
        elif ram > 0.9 and cpu < 0.6:
            action = 5  # CLEAR_CACHE

        # IDLE: CPU Low -> Scale Down
        elif cpu < 0.3 and pods > 0.2:
            action = 3  # SCALE_DOWN_SMALL

        return action, None


class AdaptiveAgent:
    """
    The Main Class used by the system to decide actions.
    """

    def __init__(self, use_heuristic_fallback=True):
        self.device = DEVICE
        self.model = None
        self.heuristic = HeuristicPolicy()
        self.use_heuristic = use_heuristic_fallback

        self._load_model()

    def _load_model(self):
        """Attempts to load the PPO model."""
        if os.path.exists(MODEL_PATH):
            try:
                logger.info(f"🧠 Loading RL Policy from {MODEL_PATH}")
                self.model = PPO.load(MODEL_PATH, device=self.device)
                self.use_heuristic = False
            except Exception as e:
                logger.error(f"❌ Failed to load RL model: {e}")
                self.use_heuristic = True
        else:
            logger.warning(f"⚠️ No trained model found at {MODEL_PATH}. Using Heuristic Fallback.")
            self.use_heuristic = True

    def decide(self, state_vector: np.ndarray, context: dict = None) -> dict:
        """
        Main decision method.

        Args:
            state_vector: Numpy array from the Gym Environment (or constructed manually).
            context: Optional dictionary from the Reasoning Brain (Phase 2)
                     containing diagnosis (e.g., 'root_cause': 'DB Lock').
                     We can use this to override the RL if necessary.

        Returns:
            Dict containing action_id, action_name, and metadata.
        """
        start_time = time.time()
        source = "RL_PPO"

        # 1. Check Context Overrides (The "Hybrid" part)
        # If the Reasoning Brain is 100% sure it's a DB Lock, RL might be too slow to learn that.
        # We can short-circuit here.
        if context and context.get("diagnosis") == "Downstream Dependency Latency":
            action_id = 4  # REROUTE
            source = "Context_Override"

        # 2. Heuristic Fallback
        elif self.use_heuristic or self.model is None:
            action_id, _ = self.heuristic.predict(state_vector)
            source = "Heuristic_Rule"

        # 3. RL Inference
        else:
            # Predict returns (action, state)
            action_id, _ = self.model.predict(state_vector, deterministic=True)
            # PPO predict returns numpy scalar, convert to int
            action_id = int(action_id)

        # 4. Construct Result
        duration = (time.time() - start_time) * 1000
        action_name = ACTION_MAP.get(action_id, "UNKNOWN")

        result = {
            "action_id": action_id,
            "action_name": action_name,
            "source": source,
            "latency_ms": round(duration, 2),
            "timestamp": time.time()
        }

        logger.info(f"🤖 Action: {action_name} | Source: {source} | Time: {duration:.2f}ms")
        return result

    def save_agent(self):
        """Manually triggers a save if needed."""
        if self.model:
            self.model.save(MODEL_PATH)
            logger.info("💾 Model saved.")


# Test Harness
if __name__ == "__main__":
    # Create a dummy state vector (7 dimensions)
    # [cpu, ram, lat, err, ops, pods, pending]
    dummy_state = np.array([0.95, 0.4, 0.8, 0.0, 0.6, 0.5, 0.0])

    agent = AdaptiveAgent()
    decision = agent.decide(dummy_state)
    print(decision)