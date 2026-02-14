"""Very simple placeholder policy for smoke testing EnergyPlus rollouts.

Use this before you have a trained policy checkpoint.
"""

from __future__ import annotations

import numpy as np


class SimpleRLModel:
    def __init__(self):
        # Assumes obs = [zone_temp, outdoor_temp]
        self.weights = np.array([0.5, 0.2], dtype=np.float32)
        self.bias = 20.0

    def get_action(self, state_vec: np.ndarray) -> np.ndarray:
        state_vec = np.asarray(state_vec, dtype=np.float32).reshape(-1)
        
        # Assume obs = [outside_temp, time_features..., zone_temps..., trends...]
        # For placeholder: Average all to simulate "effective" state
        avg_state = np.mean(state_vec) if len(state_vec) > 0 else 20.0  # Fallback
        
        # Simple linear: weights on avg_state (placeholder)
        setpoint = float(np.dot(self.weights, np.array([avg_state, avg_state])) + self.bias)  # Fake 2-dim
        action = float(np.clip(setpoint, 18.0, 26.0))
        
        # For multi-zone: Repeat the action for each zone (or customize per-zone later)
        num_zones = len(state_vec) // 3 if len(state_vec) > 2 else 1  # Rough guess: ~3 features/zone
        return np.full(num_zones, action, dtype=np.float32)  # Vector [action, action, ...]

    def update(self, trajectory: dict) -> None:
        total_reward = float(np.sum(trajectory.get("rewards", []))) if trajectory else 0.0
        if total_reward < -1000:
            self.weights += np.random.normal(0, 0.01, size=self.weights.shape).astype(np.float32)
            self.bias += float(np.random.normal(0, 0.1))
        print(f"[simple] Updated weights: {self.weights}, bias: {self.bias}")
