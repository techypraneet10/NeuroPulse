import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
from typing import Tuple, Dict, Any, List, Optional
from collections import deque
from datetime import datetime, timedelta

# Import Simulation Logic
try:
    from .generator import DataGenerator, MetricPoint, AnomalyTypes
    from .config import (
        INFRA_CONF, REWARD_CONF, ACTION_NAMES,
        CPU_MIN_MAX, RAM_MIN_MAX, LATENCY_MIN_MAX,
        ERROR_RATE_MIN_MAX, OPS_MIN_MAX
    )
except ImportError:
    from generator import DataGenerator, MetricPoint, AnomalyTypes
    from config import (
        INFRA_CONF, REWARD_CONF, ACTION_NAMES,
        CPU_MIN_MAX, RAM_MIN_MAX, LATENCY_MIN_MAX,
        ERROR_RATE_MIN_MAX, OPS_MIN_MAX
    )

logger = logging.getLogger("NeuralPulse_Env")


# ==============================================================================
# STATE MANAGER & NORMALIZER
# ==============================================================================

class StateManager:
    """
    Handles the transformation of raw infrastructure metrics into a normalized
    state vector suitable for a Neural Network.

    State Vector Definition (7 dimensions):
    [0] CPU Load (0-1)
    [1] RAM Load (0-1)
    [2] Latency (0-1)
    [3] Error Rate (0-1)
    [4] Traffic Demand (0-1)
    [5] Active Pods (0-1)
    [6] Pending Actions (0-1) - Represents scale operations in progress
    """

    def __init__(self):
        self.raw_state = {}

    def normalize(self, value: float, min_val: float, max_val: float) -> float:
        """MinMax scaler with clipping."""
        norm = (value - min_val) / (max_val - min_val + 1e-5)
        return np.clip(norm, 0.0, 1.0)

    def construct_observation(self, metrics: MetricPoint, pending_actions: int) -> np.ndarray:
        """Creates the numpy observation array."""

        obs = np.array([
            self.normalize(metrics.cpu, *CPU_MIN_MAX),
            self.normalize(metrics.ram, *RAM_MIN_MAX),
            self.normalize(metrics.latency, *LATENCY_MIN_MAX),
            self.normalize(metrics.error_rate, *ERROR_RATE_MIN_MAX),
            self.normalize(metrics.ops, *OPS_MIN_MAX),
            self.normalize(metrics.pod_count, INFRA_CONF.MIN_POD_COUNT, INFRA_CONF.MAX_POD_COUNT),
            self.normalize(pending_actions, -5, 5)  # Heuristic normalization for pending actions
        ], dtype=np.float32)

        return obs


# ==============================================================================
# PHYSICS ENGINE FOR ACTIONS
# ==============================================================================

class ActionPhysicsEngine:
    """
    Simulates the delay and side-effects of operations actions.
    Real infrastructure doesn't scale instantly. This class manages a buffer
    of 'pending changes' that apply after N timesteps.
    """

    def __init__(self):
        # A queue of tuples: (timesteps_remaining, pod_change_amount)
        self.pending_scales = deque()
        self.reroute_active = False
        self.reroute_timer = 0

    def register_action(self, action_id: int) -> Tuple[str, float]:
        """
        Processes the intent of an action and schedules it.
        Returns a description and the immediate cost.
        """
        cost = 0.0
        desc = ACTION_NAMES[action_id]

        if action_id == 0:  # DO_NOTHING
            pass

        elif action_id == 1:  # SCALE_UP_SMALL (+2)
            # Schedule +2 pods in X steps
            self.pending_scales.append([INFRA_CONF.SCALE_UP_DELAY_STEPS, 2])
            cost = REWARD_CONF.COST_PER_POD * 2

        elif action_id == 2:  # SCALE_UP_LARGE (+10)
            self.pending_scales.append([INFRA_CONF.SCALE_UP_DELAY_STEPS + 1, 10])
            cost = REWARD_CONF.COST_PER_POD * 10

        elif action_id == 3:  # SCALE_DOWN_SMALL (-2)
            self.pending_scales.append([INFRA_CONF.SCALE_DOWN_DELAY_STEPS, -2])
            cost = 0  # Saving money, but action itself is free

        elif action_id == 4:  # REROUTE_TRAFFIC
            if not self.reroute_active:
                self.reroute_active = True
                self.reroute_timer = 12  # Active for 1 minute (12 steps)
                cost = REWARD_CONF.REROUTE_COST

        elif action_id == 5:  # CLEAR_CACHE
            # Immediate effect, handled in main step, just return cost
            cost = -0.5

        return desc, cost

    def update_physics(self) -> int:
        """
        Advances time for pending actions.
        Returns the net change in pod count for this step.
        """
        net_pod_change = 0

        # Process scaling queue
        # Iterate backwards to allow removal
        for i in range(len(self.pending_scales) - 1, -1, -1):
            item = self.pending_scales[i]
            item[0] -= 1  # Decrement timer

            if item[0] <= 0:
                net_pod_change += item[1]
                del self.pending_scales[i]

        # Process reroute timer
        if self.reroute_active:
            self.reroute_timer -= 1
            if self.reroute_timer <= 0:
                self.reroute_active = False

        return net_pod_change


# ==============================================================================
# MAIN GYM ENVIRONMENT
# ==============================================================================

class NeuralPulseEnv(gym.Env):
    """
    The main environment class compatible with Stable-Baselines3.
    """
    metadata = {'render_modes': ['human', 'console']}

    def __init__(self, render_mode: Optional[str] = None):
        super(NeuralPulseEnv, self).__init__()

        self.render_mode = render_mode

        # 1. Define Action Space (Discrete 6 actions)
        self.action_space = spaces.Discrete(len(ACTION_NAMES))

        # 2. Define Observation Space (7 normalized metrics)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )

        # 3. Initialize Components
        self.generator = DataGenerator(duration_hours=24)
        self.physics = ActionPhysicsEngine()
        self.state_manager = StateManager()

        # 4. Simulation State
        self.current_step = 0
        self.max_steps = 1000  # Episode length
        self.history = []
        self.cumulative_reward = 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Resets the environment to a clean state for a new episode.
        """
        super().reset(seed=seed)

        # Reset generator with a new random seed implicit in seasonality
        self.generator = DataGenerator(duration_hours=24)
        self.physics = ActionPhysicsEngine()

        self.current_step = 0
        self.cumulative_reward = 0.0
        self.history = []

        # Get initial data point
        initial_point = self.generator.generate_step(datetime.now())

        # Construct observation
        obs = self.state_manager.construct_observation(initial_point, 0)

        info = {"status": "Environment Reset"}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Executes one time step within the environment.

        Logic Flow:
        1. Parse Agent Action -> Register in Physics Engine.
        2. Evolve Environment Physics (Pending scales apply).
        3. Get new Traffic/Chaos from Generator.
        4. Apply 'Real' System Physics (How many pods do we actually have now?).
        5. Calculate Reward based on metrics.
        6. Return Observation.
        """
        self.current_step += 1

        # --- 1. Physics Update (Actions applied) ---
        action_desc, action_cost = self.physics.register_action(action)
        pod_change = self.physics.update_physics()

        # Update generator's internal pod count (the reality)
        new_pod_count = self.generator.current_pod_count + pod_change
        # Clip to hardware limits
        new_pod_count = max(INFRA_CONF.MIN_POD_COUNT,
                            min(INFRA_CONF.MAX_POD_COUNT, new_pod_count))
        self.generator.current_pod_count = new_pod_count

        # --- 2. Traffic Generation ---
        # We advance time by 5 seconds
        current_time = datetime.now() + timedelta(seconds=self.current_step * 5)
        raw_point = self.generator.generate_step(current_time)

        # --- 3. Apply Action Effects (Rerouting, Cache Clearing) ---
        # Modify the raw point based on active mitigation strategies

        if self.physics.reroute_active:
            # Rerouting reduces load on our servers, lowering OPS and Latency
            raw_point.ops = int(raw_point.ops * 0.7)
            raw_point.latency = raw_point.latency * 0.8
            # But maybe adds a slight "processed elsewhere" flag (not simulated here)

        if action == 5:  # CLEAR_CACHE
            # Reduces RAM, but spikes CPU momentarily
            raw_point.ram = raw_point.ram * 0.5
            raw_point.cpu = min(100.0, raw_point.cpu + 15.0)

        # --- 4. Reward Calculation ---
        reward = 0.0

        # A. Performance Penalties
        if raw_point.latency > REWARD_CONF.MAX_ACCEPTABLE_LATENCY:
            # Penalty proportional to how bad it is
            overshoot = raw_point.latency - REWARD_CONF.MAX_ACCEPTABLE_LATENCY
            reward += REWARD_CONF.LATENCY_VIOLATION_PENALTY * (1 + overshoot / 1000)

        if raw_point.error_rate > REWARD_CONF.MAX_ACCEPTABLE_ERROR_RATE:
            reward += REWARD_CONF.ERROR_RATE_PENALTY * (raw_point.error_rate * 100)

        if raw_point.cpu > 95:
            reward += -1.0  # Danger zone penalty

        # B. Cost Penalties
        reward += action_cost  # The cost of the action taken
        reward += (raw_point.pod_count * REWARD_CONF.COST_PER_POD)  # Running cost

        # C. Stability Bonuses
        if raw_point.latency < REWARD_CONF.MAX_ACCEPTABLE_LATENCY and raw_point.error_rate < 0.01:
            reward += REWARD_CONF.STABILITY_BONUS

        self.cumulative_reward += reward

        # --- 5. Termination Logic ---
        terminated = False
        truncated = False

        if self.current_step >= self.max_steps:
            truncated = True

        # Fail state: System crash
        if raw_point.error_rate > 0.5 or raw_point.latency > 10000:
            terminated = True
            reward += REWARD_CONF.CRASH_PENALTY
            logger.error("💀 SYSTEM CRASH SIMULATED. Episode Terminated.")

        # --- 6. Observation Construction ---
        pending_count = len(self.physics.pending_scales)
        obs = self.state_manager.construct_observation(raw_point, pending_count)

        # Info dict for debugging/UI
        info = {
            "metrics": raw_point.to_dict(),
            "action_taken": action_desc,
            "cost": action_cost,
            "reward": reward,
            "anomaly_active": raw_point.anomaly_label != AnomalyTypes.NORMAL
        }

        self.history.append(info)

        if self.render_mode == "console":
            self.render(raw_point, action_desc, reward)

        return obs, reward, terminated, truncated, info

    def render(self, point: MetricPoint, action: str, reward: float):
        """Simple console visualization."""
        print(f"\nstep={self.current_step} | {point.timestamp.strftime('%H:%M:%S')}")
        print(f"ACT: {action} | RWD: {reward:.2f}")
        print(f"PODS: {point.pod_count} | OPS: {point.ops}")
        print(f"CPU: {point.cpu:.1f}% | RAM: {point.ram:.1f}%")
        print(f"LAT: {point.latency:.0f}ms | ERR: {point.error_rate:.3f}")
        if point.anomaly_label != "normal":
            print(f"🚨 ANOMALY: {point.anomaly_label}")
        print("-" * 30)


# ==============================================================================
# TEST HARNESS
# ==============================================================================

if __name__ == "__main__":
    # Quick test to verify environment integrity
    env = NeuralPulseEnv(render_mode="console")
    obs, _ = env.reset()

    print("Environment Integrity Check...")
    print(f"Obs Shape: {obs.shape}")
    print(f"Action Space: {env.action_space.n}")

    # Run a few random steps
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            break

    print("\n✅ Environment Check Passed.")