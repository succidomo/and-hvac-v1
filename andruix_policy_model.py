"""Worker-side policy inference utilities (TD3 actor) for Andruix rollouts.

This module is meant to be imported by rollout runners (e.g., run_sim_train.py),
so the main runner file can stay focused on EnergyPlus orchestration + rollout writing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence, Optional

import numpy as np
import torch
import torch.nn as nn


def mlp(in_dim: int, out_dim: int, hidden: Sequence[int] = (256, 256)) -> nn.Module:
    """Simple MLP: Linear+ReLU stacks ending with Linear."""
    layers: list[nn.Module] = []
    last = int(in_dim)
    for h in hidden:
        layers.append(nn.Linear(last, int(h)))
        layers.append(nn.ReLU(inplace=True))
        last = int(h)
    layers.append(nn.Linear(last, int(out_dim)))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """TD3 actor network (tanh squashed) matching learner-side architecture."""

    def __init__(self, obs_dim: int, act_dim: int, act_limit: float, hidden: Sequence[int] = (256, 256)):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, hidden=hidden)
        self.act_limit = float(act_limit)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs)) * self.act_limit


class TorchPolicyModel:
    """Loads a TD3 policy snapshot (policy.pt) and produces actions from observations.

    Expected checkpoint format:
      { "meta": {...}, "td3": { "obs_dim": int, "act_dim": int, "cfg": {...}, "actor": state_dict, ... } }

    Action mapping (worker output -> setpoint):
      - normalized (default): actor output in [-act_limit,+act_limit] mapped to [sp_min, sp_max]
      - direct: actor output interpreted directly as setpoint

    Configure via env:
      ANDRUIX_ACTION_MODE=normalized|direct
    """

    def __init__(
        self,
        policy_path: str | os.PathLike | None,
        device: str | None = None,
        sp_min: float = 18.0,
        sp_max: float = 26.0,
        default_sp: float = 22.0,
    ):
        self.policy_path = Path(policy_path) if policy_path else None
        self.sp_min = float(sp_min)
        self.sp_max = float(sp_max)
        self.default_sp = float(default_sp)

        self.action_mode = os.environ.get("ANDRUIX_ACTION_MODE", "normalized").strip().lower()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.actor: Optional[Actor] = None
        self.obs_dim: Optional[int] = None
        self.act_dim: Optional[int] = None
        self.act_limit: float = 1.0

        self._load_or_fallback()

    def _load_or_fallback(self) -> None:
        if not self.policy_path or not self.policy_path.exists():
            print(f"[policy] No policy found at {self.policy_path}; using default setpoint={self.default_sp}")
            return

        try:
            ckpt = torch.load(self.policy_path, map_location="cpu")
            td3 = ckpt.get("td3", ckpt)
            cfg = td3.get("cfg", {}) or {}

            self.obs_dim = int(td3.get("obs_dim"))
            self.act_dim = int(td3.get("act_dim"))
            self.act_limit = float(cfg.get("act_limit", 1.0))

            actor = Actor(self.obs_dim, self.act_dim, self.act_limit)
            actor.load_state_dict(td3["actor"], strict=True)
            actor.eval()
            actor.to(self.device)

            self.actor = actor

            meta = ckpt.get("meta", {})
            note = meta.get("note", "")
            print(
                f"[policy] Loaded policy '{self.policy_path}' obs_dim={self.obs_dim} act_dim={self.act_dim} "
                f"act_limit={self.act_limit} mode={self.action_mode} note='{note}'"
            )
        except Exception as e:
            self.actor = None
            self.obs_dim = None
            self.act_dim = None
            print(f"[policy] Failed to load policy at {self.policy_path}: {e}. Using default setpoint={self.default_sp}")

    @torch.no_grad()
    def get_action(self, state_vec: np.ndarray) -> float:
        """Return a zone setpoint (°C) given an observation vector."""
        if self.actor is None:
            return float(np.clip(self.default_sp, self.sp_min, self.sp_max))

        s = np.asarray(state_vec, dtype=np.float32).reshape(1, -1)
        if self.obs_dim is not None and s.shape[1] != self.obs_dim:
            raise ValueError(f"state_vec dim mismatch: got {s.shape[1]} expected {self.obs_dim}")

        obs_t = torch.from_numpy(s).to(self.device)
        act_t = self.actor(obs_t)
        act = act_t.detach().cpu().numpy().reshape(-1)
        a0 = float(act[0])  # this worker currently drives one setpoint actuator

        if self.action_mode == "direct":
            setpoint = a0
        else:
            denom = self.act_limit if abs(self.act_limit) > 1e-6 else 1.0
            a_norm = float(np.clip(a0 / denom, -1.0, 1.0))
            mid = 0.5 * (self.sp_min + self.sp_max)
            half = 0.5 * (self.sp_max - self.sp_min)
            setpoint = mid + a_norm * half

        return float(np.clip(setpoint, self.sp_min, self.sp_max))
