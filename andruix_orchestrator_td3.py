#!/usr/bin/env python3
"""
andruix_orchestrator_td3.py

Single-VM POC skeleton:
- Learner (TD3) runs on host (uses GPU if available).
- Rollout workers run in Docker containers (CPU) and write trajectories to a shared host directory.

CONTRACT (worker container):
- Reads:   /shared/policy/latest/policy.pt   (optional; if missing, worker can use random/heuristic)
- Writes:  /shared/rollouts/inbox/<rollout_id>/
            - rollout_<rollout_id>.npz    (required)
            - rollout_<rollout_id>.json   (optional)
            - rollout_<rollout_id>.done   (required; written last as a completion marker)

rollout_*.npz format (minimum):
- obs:      float32 [T, obs_dim]
- act:      float32 [T, act_dim]
- rew:      float32 [T, 1] or [T]
- next_obs: float32 [T, obs_dim]
- done:     float32 [T, 1] or [T]  (0.0/1.0)

Usage (example):
  python andruix_orchestrator_td3.py \
    --shared-root /data/andruix \
    --image andruix/eplus:latest \
    --max-workers 2 \
    --rollout-days 2 \
    --train-steps-per-rollout 500

Notes:
- Keep rollouts CPU-only. Let learner use the GPU.
- This is a skeleton: plug in your exact worker args/env-vars and your exact trajectory schema if different.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib

from torch.utils.tensorboard import SummaryWriter


# -----------------------------
# Paths / layout helpers
# -----------------------------
def ensure_dirs(shared_root: Path) -> Dict[str, Path]:
    policy_dir = shared_root / "policy" / "latest"
    inbox_dir = shared_root / "rollouts" / "inbox"
    done_dir = shared_root / "rollouts" / "done"
    logs_dir = shared_root / "logs"

    policy_dir.mkdir(parents=True, exist_ok=True)
    inbox_dir.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return {
        "policy_dir": policy_dir,
        "inbox_dir": inbox_dir,
        "done_dir": done_dir,
        "logs_dir": logs_dir,
    }

def _sha1(a: np.ndarray) -> str:
    # stable fingerprint for “are these arrays identical?”
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:10]

def atomic_save_torch(obj: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, out_path)  # atomic on same filesystem


# -----------------------------
# Replay buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int = 2_000_000):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add_batch(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
    ) -> int:
        """Add a rollout batch. Returns number of transitions added."""
        obs = np.asarray(obs, dtype=np.float32)
        act = np.asarray(act, dtype=np.float32)
        rew = np.asarray(rew, dtype=np.float32).reshape(-1, 1)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        done = np.asarray(done, dtype=np.float32).reshape(-1, 1)

        n = obs.shape[0]
        if n == 0:
            return 0

        # If batch > capacity, keep the last chunk
        if n > self.capacity:
            obs, act, rew, next_obs, done = obs[-self.capacity :], act[-self.capacity :], rew[-self.capacity :], next_obs[-self.capacity :], done[-self.capacity :]
            n = self.capacity

        end = self.ptr + n
        if end <= self.capacity:
            sl = slice(self.ptr, end)
            self.obs[sl] = obs
            self.act[sl] = act
            self.rew[sl] = rew
            self.next_obs[sl] = next_obs
            self.done[sl] = done
        else:
            # wrap
            first = self.capacity - self.ptr
            self.obs[self.ptr :] = obs[:first]
            self.act[self.ptr :] = act[:first]
            self.rew[self.ptr :] = rew[:first]
            self.next_obs[self.ptr :] = next_obs[:first]
            self.done[self.ptr :] = done[:first]

            rem = n - first
            self.obs[:rem] = obs[first:]
            self.act[:rem] = act[first:]
            self.rew[:rem] = rew[first:]
            self.next_obs[:rem] = next_obs[first:]
            self.done[:rem] = done[first:]

        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)
        return n

    def sample(self, batch_size: int, device: torch.device):
        if self.size < batch_size:
            raise RuntimeError(f"Not enough data to sample: size={self.size}, batch={batch_size}")
        idx = np.random.randint(0, self.size, size=batch_size)

        obs = torch.from_numpy(self.obs[idx]).to(device)
        act = torch.from_numpy(self.act[idx]).to(device)
        rew = torch.from_numpy(self.rew[idx]).to(device)
        next_obs = torch.from_numpy(self.next_obs[idx]).to(device)
        done = torch.from_numpy(self.done[idx]).to(device)
        return obs, act, rew, next_obs, done


# -----------------------------
# TD3 Networks
# -----------------------------
def mlp(in_dim: int, out_dim: int, hidden: Tuple[int, ...] = (256, 256)) -> nn.Module:
    layers: List[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_limit: float):
        super().__init__()
        self.net = mlp(obs_dim, act_dim)
        self.act_limit = float(act_limit)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # tanh squash to [-act_limit, act_limit]
        return torch.tanh(self.net(obs)) * self.act_limit


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.q = mlp(obs_dim + act_dim, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.q(x)


@dataclass
class TD3Config:
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    batch_size: int = 256
    act_limit: float = 1.0  # scale to your action bounds
    expl_noise: float = 0.1  # rollout exploration (worker-side, optional)


class TD3Learner:
    def __init__(self, obs_dim: int, act_dim: int, cfg: TD3Config, device: torch.device):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg = cfg
        self.device = device

        self.actor = Actor(obs_dim, act_dim, cfg.act_limit).to(device)
        self.actor_targ = Actor(obs_dim, act_dim, cfg.act_limit).to(device)
        self.actor_targ.load_state_dict(self.actor.state_dict())

        self.q1 = Critic(obs_dim, act_dim).to(device)
        self.q2 = Critic(obs_dim, act_dim).to(device)
        self.q1_targ = Critic(obs_dim, act_dim).to(device)
        self.q2_targ = Critic(obs_dim, act_dim).to(device)
        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.critic_lr)

        self.total_updates = 0

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(self.device).unsqueeze(0)
        a = self.actor(obs_t).cpu().numpy()[0]
        return a

    def update(self, rb: ReplayBuffer, steps: int) -> Dict[str, float]:
        metrics = {"q_loss": 0.0, "actor_loss": 0.0}
        if steps <= 0:
            return metrics

        for _ in range(steps):
            obs, act, rew, next_obs, done = rb.sample(self.cfg.batch_size, self.device)

            with torch.no_grad():
                noise = (torch.randn_like(act) * self.cfg.policy_noise).clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
                next_act = (self.actor_targ(next_obs) + noise).clamp(-self.cfg.act_limit, self.cfg.act_limit)

                q1_t = self.q1_targ(next_obs, next_act)
                q2_t = self.q2_targ(next_obs, next_act)
                q_t = torch.min(q1_t, q2_t)
                target = rew + self.cfg.gamma * (1.0 - done) * q_t

            q1 = self.q1(obs, act)
            q2 = self.q2(obs, act)
            q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

            self.q_opt.zero_grad(set_to_none=True)
            q_loss.backward()
            self.q_opt.step()

            actor_loss = torch.tensor(0.0, device=self.device)

            # delayed policy update
            if self.total_updates % self.cfg.policy_delay == 0:
                actor_loss = -self.q1(obs, self.actor(obs)).mean()
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                self.actor_opt.step()

                # polyak averaging
                self._soft_update(self.actor, self.actor_targ)
                self._soft_update(self.q1, self.q1_targ)
                self._soft_update(self.q2, self.q2_targ)

            self.total_updates += 1
            metrics["q_loss"] += float(q_loss.detach().cpu().item())
            metrics["actor_loss"] += float(actor_loss.detach().cpu().item())

        metrics["q_loss"] /= steps
        metrics["actor_loss"] /= steps
        return metrics

    def _soft_update(self, net: nn.Module, targ: nn.Module):
        tau = self.cfg.tau
        with torch.no_grad():
            for p, tp in zip(net.parameters(), targ.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)

    def state_dict(self) -> Dict:
        return {
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "cfg": self.cfg.__dict__,
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "total_updates": self.total_updates,
        }


# -----------------------------
# Docker orchestration
# -----------------------------
@dataclass
class WorkerSpec:
    image: str
    shared_root: Path
    rollout_id: str
    rollout_days: int
    seed: int
    extra_env: Dict[str, str]
    cpus: Optional[float] = None
    mem: Optional[str] = None  # e.g. "8g"


def docker_run_worker(spec: WorkerSpec) -> str:
    """
    Launches one rollout container. Returns container_id.
    Customize env vars/args to match your run_sim_base.py entrypoint.
    """
    # Where the worker will write
    rollout_dir = spec.shared_root / "rollouts" / "inbox" / spec.rollout_id
    rollout_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "ANDRUIX_ROLLOUT_ID": spec.rollout_id,
        "ANDRUIX_ROLLOUT_DAYS": str(spec.rollout_days),
        "ANDRUIX_SEED": str(spec.seed),
        # Worker should read policy from here (inside container)
        "ANDRUIX_POLICY_PATH": "/shared/policy/latest/policy.pt",
        # Worker should write here (inside container)
        "ANDRUIX_OUT_DIR": f"/shared/rollouts/inbox/{spec.rollout_id}"
    }
    env.update(spec.extra_env or {})

    cmd = ["docker", "run", "-d", "--name", f"andruix_rollout_{spec.rollout_id}", "--label", f"andruix.rollout_id={spec.rollout_id}"]
    # Resource caps (optional)
    if spec.cpus is not None:
        cmd += ["--cpus", str(spec.cpus)]
    if spec.mem is not None:
        cmd += ["--memory", spec.mem]

    # Mount shared root
    cmd += ["-v", f"{str(spec.shared_root)}:/shared"]

    # Env vars
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]

    # Image
    cmd += [spec.image]

    cmd += [
        "--rollout-id", spec.rollout_id,
        "--rollout-dir", f"/shared/rollouts/inbox/{spec.rollout_id}",
        "--outdir", f"/shared/results/{spec.rollout_id}",
        "--start-date", spec.extra_env["ANDRUIX_START_MMDD"],
        "--end-date", spec.extra_env["ANDRUIX_END_MMDD"],
        "--policy-kind", spec.extra_env.get("ANDRUIX_POLICY_KIND", "torch"),
        "--reward-mode", spec.extra_env.get("ANDRUIX_REWARD_MODE", "raw"),
        "--reward-scale", spec.extra_env.get("ANDRUIX_REWARD_SCALE", "3600000"),
    ]


    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        return out  # container id
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"docker run failed:\n{e.output}") from e


def docker_ps(container_id: str) -> bool:
    """Return True if container is still running."""
    try:
        out = subprocess.check_output(["docker", "ps", "-q", "--no-trunc"], stderr=subprocess.STDOUT, text=True)
        # NOTE: Some environments alias 'docker' weirdly; leaving as "docker" below.
    except Exception:
        pass

    try:
        out = subprocess.check_output(["docker", "ps", "-q", "--no-trunc"], stderr=subprocess.STDOUT, text=True)
        running = set([x.strip() for x in out.splitlines() if x.strip()])
        return container_id in running
    except subprocess.CalledProcessError:
        return False


def docker_stop(container_id: str, timeout_s: int = 10) -> None:
    try:
        subprocess.check_call(["docker", "stop", "-t", str(timeout_s), container_id], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        # might already be stopped
        pass


def docker_get_logs(container_id: str) -> str:
    """Return docker logs for a container (stdout+stderr). Container must still exist."""
    try:
        return subprocess.check_output(["docker", "logs", container_id], stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        return e.output or ""

def docker_rm(container_id: str) -> None:
    """Remove a stopped container."""
    try:
        subprocess.check_call(["docker", "rm", "-f", container_id], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        pass

# -----------------------------
# Rollout ingestion
# -----------------------------
def list_completed_rollouts(inbox_dir: Path) -> List[Path]:
    """Return rollout directories that appear complete.

    New worker contract: rollout_<id>.done marker written last.
    Legacy support: DONE marker + traj.npz.
    """
    rdirs = set()

    # New style: rollout_*.done
    for p in inbox_dir.glob("*/rollout_*.done"):
        rdirs.add(p.parent)

    # Legacy style: DONE marker
    for p in inbox_dir.glob("*/DONE"):
        rdirs.add(p.parent)

    return sorted(rdirs)



def load_traj_npz(traj_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a rollout npz safely (ensures file handle is closed; important on Windows)."""
    with np.load(traj_path) as data:
        obs = data["obs"]
        act = data["act"]
        rew = data["rew"]
        next_obs = data["next_obs"]
        done = data["done"]
    return obs, act, rew, next_obs, done


# -----------------------------
# Main orchestrator loop
# -----------------------------
class Orchestrator:
    def __init__(
        self,
        shared_root: Path,
        image: str,
        obs_dim: int,
        act_dim: int,
        cfg: TD3Config,
        max_workers: int,
        rollout_days: int,
        train_steps_per_rollout: int,
        publish_every_rollouts: int,
        min_replay_before_train: int,
        worker_cpus: Optional[float] = None,
        worker_mem: Optional[str] = None,
        extra_env: Optional[Dict[str, str]] = None,
        tb_logdir: Optional[Path] = None,
        tb_run_name: Optional[str] = None,
        tb_flush_secs: int = 10
    ):
        self.shared_root = shared_root
        self.paths = ensure_dirs(shared_root)
        self.image = image
        self.max_workers = int(max_workers)
        self.rollout_days = int(rollout_days)
        self.train_steps_per_rollout = int(train_steps_per_rollout)
        self.publish_every_rollouts = int(publish_every_rollouts)
        self.min_replay_before_train = int(min_replay_before_train)
        self.worker_cpus = worker_cpus
        self.worker_mem = worker_mem
        self.extra_env = extra_env or {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learner = TD3Learner(obs_dim, act_dim, cfg, self.device)
        self.rb = ReplayBuffer(obs_dim, act_dim, capacity=2_000_000)

        # TensorBoard
        run_name = tb_run_name or time.strftime('%Y%m%d_%H%M%S')
        logdir = (tb_logdir or (self.shared_root / 'tb')) / run_name
        logdir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(logdir), flush_secs=int(tb_flush_secs))
        self.tb_logdir = logdir
        # Log static run config
        self.writer.add_text('run/config', json.dumps({
            'image': self.image,
            'obs_dim': obs_dim,
            'act_dim': act_dim,
            'max_workers': self.max_workers,
            'rollout_days': self.rollout_days,
            'train_steps_per_rollout': self.train_steps_per_rollout,
            'publish_every_rollouts': self.publish_every_rollouts,
            'min_replay_before_train': self.min_replay_before_train,
            'td3_cfg': self.learner.cfg.__dict__,
            'extra_env': self.extra_env,
        }, indent=2))
        self.writer.add_text('run/logdir', str(self.tb_logdir))

        self.active: Dict[str, str] = {}  # rollout_id -> container_id
        self.rollouts_ingested = 0
        self._stop = False

        # Write an initial policy snapshot so workers can start
        self.publish_policy(version_note="init")

    def publish_policy(self, version_note: str = "") -> None:
        policy_path = self.paths["policy_dir"] / "policy.pt"
        meta = {
            "published_at_unix": time.time(),
            "rollouts_ingested": self.rollouts_ingested,
            "total_updates": self.learner.total_updates,
            "note": version_note,
            "device": str(self.device),
        }
        obj = {
            "meta": meta,
            "td3": self.learner.state_dict(),
            # Convenience keys for rollout workers
            "actor": self.learner.actor.state_dict(),
            "obs_dim": self.learner.obs_dim,
            "act_dim": self.learner.act_dim,
            "act_limit": self.learner.cfg.act_limit,
        }
        atomic_save_torch(obj, policy_path)

        # TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('policy/published', 1, self.rollouts_ingested)
            self.writer.add_scalar('policy/total_updates', self.learner.total_updates, self.rollouts_ingested)
            if version_note:
                self.writer.add_text('policy/note', version_note, self.rollouts_ingested)

        # Optional: also write a small JSON sidecar for human debugging
        with open(self.paths["policy_dir"] / "policy_meta.json.tmp", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        os.replace(self.paths["policy_dir"] / "policy_meta.json.tmp", self.paths["policy_dir"] / "policy_meta.json")

    def launch_more_workers_if_needed(self) -> None:
        while len(self.active) < self.max_workers and not self._stop:
            rollout_id = uuid.uuid4().hex[:12]
            seed = int.from_bytes(os.urandom(4), "little", signed=False)

            spec = WorkerSpec(
                image=self.image,
                shared_root=self.shared_root,
                rollout_id=rollout_id,
                rollout_days=self.rollout_days,
                seed=seed,
                extra_env=self.extra_env,
                cpus=self.worker_cpus,
                mem=self.worker_mem,
            )
            cid = docker_run_worker(spec)
            self.active[rollout_id] = cid
            print(f"[orchestrator] launched rollout {rollout_id} -> container {cid}")

    
    def reap_finished_containers(self) -> None:
        """Capture logs for finished containers and remove them."""
        dead = []
        for rid, cid in self.active.items():
            if not docker_ps(cid):
                dead.append(rid)

        for rid in dead:
            cid = self.active.pop(rid, "")
            # Capture docker logs BEFORE removing container
            try:
                logs_text = docker_get_logs(cid)
                log_path = self.paths["logs_dir"] / f"worker_{rid}.log"
                header = (
                    f"===== Andruix worker logs ====="
                    f"rollout_id: {rid}"
                    f"container_id: {cid}"
                    f"captured_at_unix: {time.time()}"
                    f"==============================="
                )
                log_path.write_text(header + logs_text, encoding="utf-8")
                print(f"[orchestrator] wrote worker logs -> {log_path}")
            except Exception as e:
                print(f"[orchestrator] WARNING: failed to capture logs for rollout {rid} cid={cid}: {e}")

            # Remove container (we intentionally do NOT use --rm so we can capture logs)
            docker_rm(cid)
            print(f"[orchestrator] container ended for rollout {rid} (cid={cid})")

    def ingest_completed_rollouts(self) -> int:
        inbox_dir = self.paths["inbox_dir"]
        done_dir = self.paths["done_dir"]
        completed = list_completed_rollouts(inbox_dir)
        if not completed:
            return 0

        ingested = 0
        for rdir in completed:            # New worker artifact names: rollout_<id>.npz (+ rollout_<id>.done)
            npz_path = next(iter(sorted(rdir.glob("rollout_*.npz"))), None)
            if npz_path is None:
                # Legacy fallback
                legacy_npz = rdir / "traj.npz"
                npz_path = legacy_npz if legacy_npz.exists() else None

            if npz_path is None:
                print(f"[orchestrator] WARNING: completion marker present but no rollout npz in {rdir}")
                # move aside anyway to avoid infinite loop
                shutil.move(str(rdir), str(done_dir / rdir.name))
                continue

            try:
                obs, act, rew, next_obs, done = load_traj_npz(npz_path)

                # act debug delete me later
                act_mean = float(np.mean(act))
                act_std  = float(np.std(act))
                act_min  = float(np.min(act))
                act_max  = float(np.max(act))

                rew_sha = _sha1(rew)
                act_sha = _sha1(act)

                print(
                    f"[orchestrator] rollout={rdir.name} "
                    f"act(mean={act_mean:.4f}, std={act_std:.4f}, min={act_min:.4f}, max={act_max:.4f}) "
                    f"sha(rew={rew_sha}, act={act_sha})"
                )
                # act debug delete me later

                added = self.rb.add_batch(obs, act, rew, next_obs, done)
                ingested += 1
                self.rollouts_ingested += 1

                # Reward statistics for debugging ingestion
                rew_arr = np.asarray(rew, dtype=np.float32)
                nan_ct = int(np.isnan(rew_arr).sum())
                if nan_ct:
                    print(f"[orchestrator] WARNING: rollout={rdir.name} has {nan_ct} NaN rewards")
                rew_mean = float(np.nanmean(rew_arr))
                rew_std = float(np.nanstd(rew_arr))
                rew_min = float(np.nanmin(rew_arr))
                rew_max = float(np.nanmax(rew_arr))
                rew_sum = float(np.nansum(rew_arr))
                nsteps = int(rew_arr.shape[0])
                print(
                    f"[orchestrator] ingested rollout={rdir.name} transitions={added} buffer_size={self.rb.size} "
                    f"steps={nsteps} return={rew_sum:.4f} mean={rew_mean:.6f} std={rew_std:.6f} "
                    f"min={rew_min:.6f} max={rew_max:.6f} npz={npz_path.name}"
                )

                # TensorBoard rollout metrics
                if self.writer is not None:
                    ep_return = float(np.sum(rew))
                    self.writer.add_scalar('rollout/transitions', int(added), self.rollouts_ingested)
                    self.writer.add_scalar('rollout/episode_return', ep_return, self.rollouts_ingested)
                    self.writer.add_scalar('rollout/mean_reward', float(np.mean(rew)), self.rollouts_ingested)
                    self.writer.add_scalar('rollout/min_reward', float(np.min(rew)), self.rollouts_ingested)
                    self.writer.add_scalar('rollout/max_reward', float(np.max(rew)), self.rollouts_ingested)
                    self.writer.add_scalar('buffer/size', int(self.rb.size), self.rollouts_ingested)
            except Exception as e:
                print(f"[orchestrator] ERROR ingesting {rdir}: {e}")
                # move aside for inspection
            finally:
                # Move rollout folder to done (even if ingestion failed, to avoid reprocessing loop)
                target = done_dir / rdir.name
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)
                shutil.move(str(rdir), str(target))

        return ingested

    def maybe_train(self, newly_ingested: int) -> None:
        if newly_ingested <= 0:
            return
        if self.rb.size < self.min_replay_before_train:
            print(f"[learner] buffer {self.rb.size} < min {self.min_replay_before_train}, skipping train")
            return

        steps = self.train_steps_per_rollout * newly_ingested
        metrics = self.learner.update(self.rb, steps=steps)
        print(
            f"[learner] updates={steps} total_updates={self.learner.total_updates} "
            f"q_loss={metrics['q_loss']:.4f} actor_loss={metrics['actor_loss']:.4f} device={self.device}"
        )

        # TensorBoard training metrics (x-axis: total_updates)
        if self.writer is not None:
            self.writer.add_scalar('train/q_loss', metrics['q_loss'], self.learner.total_updates)
            self.writer.add_scalar('train/actor_loss', metrics['actor_loss'], self.learner.total_updates)
            self.writer.add_scalar('train/steps_this_update', int(steps), self.learner.total_updates)

        if self.publish_every_rollouts > 0 and (self.rollouts_ingested % self.publish_every_rollouts == 0):
            self.publish_policy(version_note=f"rollouts={self.rollouts_ingested}")

    def stop(self) -> None:
        self._stop = True
        print("[orchestrator] stopping... stopping active containers")

        # Stop and capture logs for any active containers
        for rid, cid in list(self.active.items()):
            print(f"[orchestrator] stopping rollout {rid} cid={cid}")

            # Stop container (best-effort)
            try:
                docker_stop(cid, timeout_s=10)
            except Exception as e:
                print(f"[orchestrator] WARNING: docker_stop failed rid={rid} cid={cid}: {e}")

            # Capture logs (best-effort)
            try:
                logs_text = docker_get_logs(cid)
                log_path = self.paths["logs_dir"] / f"worker_{rid}.log"
                header = (
                    "===== Andruix worker logs (stopped) =====\n"
                    f"rollout_id: {rid}\n"
                    f"container_id: {cid}\n"
                    f"captured_at_unix: {time.time()}\n"
                    "========================================\n"
                )
                log_path.write_text(header + logs_text, encoding="utf-8")
                print(f"[orchestrator] wrote worker logs -> {log_path}")
            except Exception as e:
                print(f"[orchestrator] WARNING: failed to capture logs rid={rid} cid={cid}: {e}")

            # Remove container (best-effort)
            try:
                docker_rm(cid)
            except Exception as e:
                print(f"[orchestrator] WARNING: docker_rm failed rid={rid} cid={cid}: {e}")

        # Clear after loop
        self.active.clear()

        # Flush TB writer
        if hasattr(self, "writer") and self.writer is not None:
            try:
                self.writer.flush()
                self.writer.close()
            except Exception as e:
                print(f"[orchestrator] WARNING: failed to close writer: {e}")


    def run_forever(self, poll_s: float = 2.0) -> None:
        def _sig_handler(signum, frame):
            self.stop()

        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)

        print(f"[orchestrator] device={self.device} max_workers={self.max_workers} rollout_days={self.rollout_days}")
        while not self._stop:
            self.launch_more_workers_if_needed()
            self.reap_finished_containers()
            newly_ingested = self.ingest_completed_rollouts()
            self.maybe_train(newly_ingested)
            time.sleep(poll_s)

        print("[orchestrator] exited.")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--shared-root", type=str, required=True, help="Host directory shared with containers, e.g. /data/andruix")
    p.add_argument("--image", type=str, required=True, help="Rollout worker docker image, e.g. andruix/eplus:latest")

    # TensorBoard
    p.add_argument('--tb-logdir', type=str, default=None, help='TensorBoard log root (default: <shared-root>/tb)')
    p.add_argument('--tb-run-name', type=str, default=None, help='TensorBoard run name (default: timestamp)')
    p.add_argument('--tb-flush-secs', type=int, default=10, help='TensorBoard flush seconds')

    # You must set these to match your environment
    p.add_argument("--obs-dim", type=int, required=True)
    p.add_argument("--act-dim", type=int, required=True)
    p.add_argument("--act-limit", type=float, default=1.0)

    p.add_argument("--max-workers", type=int, default=2)
    p.add_argument("--rollout-days", type=int, default=2)
    p.add_argument("--train-steps-per-rollout", type=int, default=500, help="Gradient steps per ingested rollout")
    p.add_argument("--publish-every-rollouts", type=int, default=10)
    p.add_argument("--min-replay-before-train", type=int, default=10_000)

    # Optional resource caps per container
    p.add_argument("--worker-cpus", type=float, default=None)
    p.add_argument("--worker-mem", type=str, default=None)

    # TD3 knobs (starter defaults)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--policy-noise", type=float, default=0.2)
    p.add_argument("--noise-clip", type=float, default=0.5)
    p.add_argument("--policy-delay", type=int, default=2)
    p.add_argument("--actor-lr", type=float, default=1e-3)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)

    # Pass-through env to containers (repeatable key=value)
    p.add_argument("--env", action="append", default=[], help="Extra env var for workers, format KEY=VALUE (repeatable)")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    shared_root = Path(args.shared_root).resolve()

    extra_env: Dict[str, str] = {}
    for item in args.env:
        if "=" not in item:
            raise ValueError(f"--env must be KEY=VALUE, got: {item}")
        k, v = item.split("=", 1)
        extra_env[k] = v

    cfg = TD3Config(
        gamma=args.gamma,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_delay=args.policy_delay,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        batch_size=args.batch_size,
        act_limit=args.act_limit,
    )

    orch = Orchestrator(
        shared_root=shared_root,
        image=args.image,
        obs_dim=args.obs_dim,
        act_dim=args.act_dim,
        cfg=cfg,
        max_workers=args.max_workers,
        rollout_days=args.rollout_days,
        train_steps_per_rollout=args.train_steps_per_rollout,
        publish_every_rollouts=args.publish_every_rollouts,
        min_replay_before_train=args.min_replay_before_train,
        worker_cpus=args.worker_cpus,
        worker_mem=args.worker_mem,
        extra_env=extra_env,
        tb_logdir=(Path(args.tb_logdir).resolve() if args.tb_logdir else None),
        tb_run_name=args.tb_run_name,
        tb_flush_secs=args.tb_flush_secs
    )
    orch.run_forever(poll_s=2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
