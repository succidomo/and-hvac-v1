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

    def get_action(self, state_vec: np.ndarray) -> float:
        state_vec = np.asarray(state_vec, dtype=np.float32).reshape(-1)
        if state_vec.shape[0] != 2:
            raise ValueError(f"SimpleRLModel expects obs_dim=2, got {state_vec.shape[0]}")
        setpoint = float(np.dot(self.weights, state_vec) + self.bias)
        return float(np.clip(setpoint, 18.0, 26.0))

    def update(self, trajectory: dict) -> None:
        total_reward = float(np.sum(trajectory.get("rewards", []))) if trajectory else 0.0
        if total_reward < -1000:
            self.weights += np.random.normal(0, 0.01, size=self.weights.shape).astype(np.float32)
            self.bias += float(np.random.normal(0, 0.1))
        print(f"[simple] Updated weights: {self.weights}, bias: {self.bias}")
