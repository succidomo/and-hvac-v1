"""Rollout writing utilities for Andruix EnergyPlus workers.

Writes a single-episode chunk to:
- rollout_<id>.npz  (TD3-style replay arrays)
- rollout_<id>.json (metadata)
- rollout_<id>.done (simple completion marker)

Keep this separate so the simulation runner can stay focused on callbacks + control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import numpy as np

from dataclasses import dataclass, field
from typing import Iterable, Tuple
import math

try:
    import pandas as pd
except Exception:
    pd = None


@dataclass
class RolloutMeta:
    rollout_id: str
    n_steps: int
    episode_return: float

    zones: List[str]
    obs_dim: int
    act_dim: int

    start_mmdd: Optional[str] = None
    end_mmdd: Optional[str] = None

    reward_mode: Optional[str] = None
    reward_scale: Optional[float] = None

    obs_flags: Optional[Dict[str, Any]] = None
    policy_fingerprint: Optional[str] = None


class RolloutWriter:
    def __init__(
        self, 
        rollout_dir: str | Path, 
        rollout_id: str, 
        obs_dim: int, 
        act_dim: int,
        *,
        zones: Optional[List[str]] = None,
        start_mmdd: Optional[str] = None,
        end_mmdd: Optional[str] = None,
        reward_mode: Optional[str] = None,
        reward_scale: Optional[float] = None,
        obs_flags: Optional[Dict[str, Any]] = None,
        policy_fingerprint: Optional[str] = None,
    ):
        self.rollout_dir = Path(rollout_dir)
        self.rollout_dir.mkdir(parents=True, exist_ok=True)
        self.rollout_id = rollout_id
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        self.zones = list(zones) if zones else []
        self.start_mmdd = start_mmdd
        self.end_mmdd = end_mmdd
        self.reward_mode = reward_mode
        self.reward_scale = reward_scale
        self.obs_flags = dict(obs_flags) if obs_flags else None
        self.policy_fingerprint = policy_fingerprint

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
            zones=list(self.zones),
            obs_dim=int(self.obs_dim),
            act_dim=int(self.act_dim),
            start_mmdd=self.start_mmdd,
            end_mmdd=self.end_mmdd,
            reward_mode=self.reward_mode,
            reward_scale=self.reward_scale,
            obs_flags=self.obs_flags,
            policy_fingerprint=self.policy_fingerprint,
        )
        (self.rollout_dir / f"rollout_{self.rollout_id}.json").write_text(
            json.dumps(meta.__dict__, indent=2)
        )

        (self.rollout_dir / f"rollout_{self.rollout_id}.done").write_text("ok\n")
        return out_npz


@dataclass
class WorkerTimeseriesWriter:
    """
    Writes a per-timestep timeseries file for a single worker rollout.

    Output:
      timeseries_<rollout_id>.parquet  (preferred)
      timeseries_<rollout_id>.csv      (fallback if parquet deps missing)

    Usage pattern (from callbacks):
      ts.append_step(
        step_idx=...,
        day_of_year=...,
        minute_of_day=...,
        outside_air_c=...,
        hvac_energy_kwh=...,
        reward=...,
        zone_temps_c={...},
        zone_setpoints_cool_c={...},
        zone_setpoints_heat_c={...},
        zone_actions_norm={...},
      )
    """
    out_dir: str | Path
    rollout_id: str
    zones: list[str] = field(default_factory=list)

    # Optional meta for traceability
    policy_fingerprint: str | None = None
    image_tag: str | None = None

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._rows: list[dict] = []

    @staticmethod
    def _safe_zone_col(prefix: str, zone: str) -> str:
        # Keep it predictable + filesystem/column friendly
        z = zone.strip().replace(" ", "_")
        return f"{prefix}__{z}"

    def append_step(
        self,
        *,
        step_idx: int,
        day_of_year: int | None,
        minute_of_day: int | None,
        outside_air_c: float | None = None,
        hvac_energy_kwh: float | None = None,
        reward: float | None = None,
        zone_temps_c: dict[str, float] | None = None,
        zone_setpoints_heat_c: dict[str, float] | None = None,
        zone_setpoints_cool_c: dict[str, float] | None = None,
        zone_actions_norm: dict[str, float] | None = None,
        extra_scalars: dict[str, float] | None = None,
    ) -> None:
        row = {
            "step_idx": int(step_idx),
            "day_of_year": int(-1 if day_of_year is None else day_of_year),
            "minute_of_day": int(-1 if minute_of_day is None else minute_of_day),
            "outside_air_c": (float(outside_air_c) if outside_air_c is not None else math.nan),
            "hvac_energy_kwh": (float(hvac_energy_kwh) if hvac_energy_kwh is not None else math.nan),
            "reward": (float(reward) if reward is not None else math.nan),
        }

        # Zones
        if zone_temps_c:
            for z, v in zone_temps_c.items():
                row[self._safe_zone_col("tz_c", z)] = float(v)
        if zone_setpoints_heat_c:
            for z, v in zone_setpoints_heat_c.items():
                row[self._safe_zone_col("heat_sp_c", z)] = float(v)
        if zone_setpoints_cool_c:
            for z, v in zone_setpoints_cool_c.items():
                row[self._safe_zone_col("cool_sp_c", z)] = float(v)
        if zone_actions_norm:
            for z, v in zone_actions_norm.items():
                row[self._safe_zone_col("act_norm", z)] = float(v)

        if self.policy_fingerprint:
            row["policy_hash"] = self.policy_fingerprint

        # Easy extension point: add any scalar metric without changing schema code
        if extra_scalars:
            for k, v in extra_scalars.items():
                row[str(k)] = float(v)

        self._rows.append(row)

    def write(self) -> Path:
        """
        Writes parquet if possible, otherwise writes csv.
        Returns the written file path.
        """
        base = self.out_dir / f"timeseries_{self.rollout_id}"
        parquet_path = base.with_suffix(".parquet")
        csv_path = base.with_suffix(".csv")

        if pd is None:
            # No pandas available: fallback to CSV via python stdlib
            import csv
            # Union of keys across rows
            keys = sorted({k for r in self._rows for k in r.keys()})
            with csv_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(self._rows)
            return csv_path

        df = pd.DataFrame(self._rows)

        # Prefer parquet (fast + small). This requires either pyarrow or fastparquet.
        try:
            df.to_parquet(parquet_path, index=False)
            return parquet_path
        except Exception:
            # If parquet deps missing, csv fallback
            df.to_csv(csv_path, index=False)
            return csv_path