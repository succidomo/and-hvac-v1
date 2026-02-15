"""Very simple placeholder policy for smoke testing EnergyPlus rollouts.

Use this before you have a trained policy checkpoint.
"""

from __future__ import annotations

import numpy as np


class SimpleRLModel:
    def __init__(self):
        # Assumes obs includes multiple zones — weights on avg for placeholder
        self.weights = np.array([0.5, 0.2], dtype=np.float32)
        self.bias = 20.0

    def get_action(self, state_vec: np.ndarray) -> np.ndarray:
        state_vec = np.asarray(state_vec, dtype=np.float32).reshape(-1)
        print(f"[simple_dbg] Input obs shape: {state_vec.shape}")  # Debug to confirm obs

        # For placeholder: Average the entire obs as "effective state"
        avg_state = np.mean(state_vec) if len(state_vec) > 0 else 20.0
        
        # Simple linear on avg (fake 2-dim input)
        setpoint = float(np.dot(self.weights, np.array([avg_state, avg_state])) + self.bias)
        action = float(np.clip(setpoint, 18.0, 26.0))
        
        # Guess num_zones from obs length (e.g., ~3 features/zone after shared)
        num_features_per_zone = 3  # Adjust based on your obs (Tzone + 2 trends)
        shared_features = 5  # Toa + 4 time
        num_zones = max(1, (len(state_vec) - shared_features) // num_features_per_zone)
        
        # Return vector: same action repeated for each zone
        action_vec = np.full(num_zones, action, dtype=np.float32)
        print(f"[simple_dbg] Returning action_vec: {action_vec}")  # Confirm vector
        
        return action_vec

    def update(self, trajectory: dict) -> None:
        total_reward = float(np.sum(trajectory.get("rewards", []))) if trajectory else 0.0
        if total_reward < -1000:
            self.weights += np.random.normal(0, 0.01, size=self.weights.shape).astype(np.float32)
            self.bias += float(np.random.normal(0, 0.1))
        print(f"[simple] Updated weights: {self.weights}, bias: {self.bias}")