"""Very simple placeholder policy for smoke testing EnergyPlus rollouts.

Path A contract:
- get_action(obs) returns a vector of normalized actions in [-1, 1], length = num_zones
- worker maps each action to (heat_sp, cool_sp) via _map_action_to_setpoints()

This is NOT a learner — just a stable deterministic baseline for docker smoke tests.
"""

from __future__ import annotations

import numpy as np


class SimpleRLModel:
    def __init__(self, num_zones: int = 1, target_temp_c: float = 22.5, gain: float = 0.5):
        self.num_zones = int(max(1, num_zones))
        self.target_temp_c = float(target_temp_c)
        self.gain = float(gain)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Return normalized actions in [-1,1] (one per zone).

        Expected obs layout (worker Path A):
        [Tz_1..Tz_N] + [Toa] + [sin_tod, cos_tod] + trends + [sin_doy, cos_doy] + [occ_flag]
        """
        x = np.asarray(obs, dtype=np.float32).reshape(-1)
        n = self.num_zones

        actions = np.zeros(n, dtype=np.float32)

        # Per-zone proportional control in normalized action space
        if x.size >= n:
            tz = x[:n]  # per-zone temperatures
            for i in range(n):
                tz_i = float(tz[i]) if np.isfinite(tz[i]) else self.target_temp_c
                err = self.target_temp_c - tz_i  # positive if too cold (raise setpoint)
                a_i = float(np.clip(self.gain * err / 4.0, -1.0, 1.0))
                actions[i] = a_i
        else:
            # Fallback: same default action for all
            actions.fill(0.0)

        return actions

    def update(self, trajectory: dict) -> None:
        # no-op placeholder (kept for interface compatibility)
        return
