"""Rollout writing utilities for Andruix EnergyPlus workers.

Writes a single-episode chunk to:
- rollout_<id>.npz  (TD3-style replay arrays)
- rollout_<id>.json (metadata)
- rollout_<id>.done (simple completion marker)

Keep this separate so the simulation runner can stay focused on callbacks + control.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np


@dataclass
class RolloutMeta:
    rollout_id: str
    n_steps: int
    episode_return: float


class RolloutWriter:
    def __init__(self, rollout_dir: str | Path, rollout_id: str, obs_dim: int, act_dim: int):
        self.rollout_dir = Path(rollout_dir)
        self.rollout_dir.mkdir(parents=True, exist_ok=True)
        self.rollout_id = rollout_id
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        self.obs: list[np.ndarray] = []
        self.act: list[np.ndarray] = []
        self.rew: list[np.float32] = []
        self.next_obs: list[np.ndarray] = []
        self.done: list[np.float32] = []

        # Optional time/energy indexing for stitching rollouts later
        self.day_of_year: list[np.int32] = []
        self.minute_of_day: list[np.int32] = []
        self.energy_meter: list[np.float32] = []

    def append(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: float,
        *,
        day_of_year: int | None = None,
        minute_of_day: int | None = None,
        energy_meter: float | None = None,
    ) -> None:
        """Append one transition.

        Shapes:
          obs: (obs_dim,)
          act: (act_dim,)
          next_obs: (obs_dim,)

        Extra (optional):
          day_of_year: 1..366
          minute_of_day: 0..1439
          energy_meter: raw meter value for this timestep (whatever EnergyPlus reports)
        """
        obs = np.asarray(obs, dtype=np.float32).reshape(self.obs_dim)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(self.obs_dim)
        act = np.asarray(act, dtype=np.float32).reshape(self.act_dim)

        self.obs.append(obs)
        self.act.append(act)
        self.rew.append(np.float32(rew))
        self.next_obs.append(next_obs)
        self.done.append(np.float32(done))

        self.day_of_year.append(np.int32(-1 if day_of_year is None else int(day_of_year)))
        self.minute_of_day.append(np.int32(-1 if minute_of_day is None else int(minute_of_day)))
        self.energy_meter.append(np.float32(np.nan if energy_meter is None else float(energy_meter)))


    def to_npz_dict(self) -> dict[str, np.ndarray]:
        if self.obs:
            obs = np.stack(self.obs).astype(np.float32)
            next_obs = np.stack(self.next_obs).astype(np.float32)
        else:
            obs = np.zeros((0, self.obs_dim), np.float32)
            next_obs = np.zeros((0, self.obs_dim), np.float32)

        if self.act:
            act = np.stack(self.act).astype(np.float32)
        else:
            act = np.zeros((0, self.act_dim), np.float32)

        rew = np.asarray(self.rew, np.float32)
        done = np.asarray(self.done, np.float32)
        return {
            "obs": obs, 
            "act": act, 
            "rew": rew, 
            "next_obs": next_obs, 
            "done": done, 
            "day_of_year": np.asarray(self.day_of_year, np.int32),
            "minute_of_day": np.asarray(self.minute_of_day, np.int32),
            "energy_meter": np.asarray(self.energy_meter, np.float32),
            }

    def write(self) -> Path:
        out_npz = self.rollout_dir / f"rollout_{self.rollout_id}.npz"
        payload = self.to_npz_dict()
        np.savez_compressed(out_npz, **payload)

        meta = RolloutMeta(
            rollout_id=self.rollout_id,
            n_steps=int(len(payload["rew"])),
            episode_return=float(np.sum(payload["rew"])) if len(payload["rew"]) else 0.0,
        )
        (self.rollout_dir / f"rollout_{self.rollout_id}.json").write_text(json.dumps(meta.__dict__, indent=2))
        # Write marker last
        (self.rollout_dir / f"rollout_{self.rollout_id}.done").write_text("ok\n")
        return out_npz
