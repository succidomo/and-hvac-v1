#!/usr/bin/env python3
"""
stitch_rollouts.py

Stitch Andruix rollout chunks (rollout_*.npz) into a single time-ordered dataset.

Expected NPZ keys (minimum for stitching):
  - obs, act, rew, next_obs, done
Optional (recommended):
  - day_of_year (int32), minute_of_day (int32), energy_meter (float32)

If day_of_year/minute_of_day are missing, the script will fall back to:
  - per-rollout step_index (0..N-1) and sort by filename order.

Outputs:
  - CSV (default) or Parquet (if pandas+pyarrow available)
  - Summary JSON with aggregates

Usage:
  python stitch_rollouts.py --rollouts-dir /path/to/rollouts --out stitched.csv
  python stitch_rollouts.py --rollouts-dir /path/to/rollouts --out stitched.parquet --format parquet
  python stitch_rollouts.py --rollouts-dir /path/to/rollouts --plot --plot-out stitched.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np


def _safe_load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _rollout_id_from_filename(path: str) -> str:
    name = Path(path).name
    # rollout_<id>.npz -> <id>
    if name.startswith("rollout_") and name.endswith(".npz"):
        return name[len("rollout_"):-len(".npz")]
    return Path(path).stem


def _has_time(payload: Dict[str, np.ndarray]) -> bool:
    return ("day_of_year" in payload) and ("minute_of_day" in payload)


def _validate_shapes(payload: Dict[str, np.ndarray]) -> Tuple[int, int, int]:
    """Return (T, obs_dim, act_dim). Raises if inconsistent."""
    obs = payload["obs"]
    act = payload["act"]
    T = int(obs.shape[0])

    if obs.ndim != 2:
        raise ValueError(f"obs must be 2D (T,obs_dim). Got shape {obs.shape}")
    obs_dim = int(obs.shape[1])

    if act.ndim == 1:
        act_dim = 1
        if act.shape[0] != T:
            raise ValueError(f"act length mismatch: {act.shape} vs T={T}")
    elif act.ndim == 2:
        if act.shape[0] != T:
            raise ValueError(f"act shape mismatch: {act.shape} vs T={T}")
        act_dim = int(act.shape[1])
    else:
        raise ValueError(f"act must be 1D or 2D. Got shape {act.shape}")

    for k in ["rew", "done"]:
        if payload[k].shape[0] != T:
            raise ValueError(f"{k} length mismatch: {payload[k].shape} vs T={T}")

    if payload["next_obs"].shape[0] != T:
        raise ValueError(f"next_obs length mismatch: {payload['next_obs'].shape} vs T={T}")
    if payload["next_obs"].ndim != 2 or payload["next_obs"].shape[1] != obs_dim:
        raise ValueError(f"next_obs must be (T,{obs_dim}). Got {payload['next_obs'].shape}")

    return T, obs_dim, act_dim


def _flatten_obs_act(obs: np.ndarray, act: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Return column dicts for obs_i and act_i."""
    obs_cols = {f"obs_{i}": obs[:, i].astype(np.float32) for i in range(obs.shape[1])}
    if act.ndim == 1:
        act = act.reshape(-1, 1)
    act_cols = {f"act_{i}": act[:, i].astype(np.float32) for i in range(act.shape[1])}
    return obs_cols, act_cols


def stitch_rollouts(rollout_paths: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Return (columns_dict, summary_dict)."""
    rows: Dict[str, List[np.ndarray]] = {}
    per_rollout: List[Dict[str, Any]] = []

    have_time_any = False
    total_steps = 0
    total_return = 0.0

    for idx, p in enumerate(rollout_paths):
        payload = _safe_load_npz(p)
        required = {"obs", "act", "rew", "next_obs", "done"}
        missing = required - set(payload.keys())
        if missing:
            raise ValueError(f"{p} missing required keys: {sorted(missing)}")

        T, obs_dim, act_dim = _validate_shapes(payload)

        rid = _rollout_id_from_filename(p)
        obs = payload["obs"]
        act = payload["act"]
        rew = payload["rew"].astype(np.float32)
        done = payload["done"].astype(np.float32)

        obs_cols, act_cols = _flatten_obs_act(obs, act)

        # time columns
        if _has_time(payload):
            doy = payload["day_of_year"].astype(np.int32)
            mod = payload["minute_of_day"].astype(np.int32)
            energy = payload.get("energy_meter", np.full((T,), np.nan, np.float32)).astype(np.float32)
            have_time_any = True
        else:
            doy = np.full((T,), -1, np.int32)
            mod = np.full((T,), -1, np.int32)
            energy = np.full((T,), np.nan, np.float32)

        # always include stable per-file ordering fallback
        file_index = np.full((T,), idx, np.int32)
        step_index = np.arange(T, dtype=np.int32)

        cols: Dict[str, np.ndarray] = {
            "rollout_id": np.array([rid] * T, dtype=object),
            "file_index": file_index,
            "step_index": step_index,
            "day_of_year": doy,
            "minute_of_day": mod,
            "energy_meter": energy,
            "reward": rew,
            "done": done,
        }
        cols.update(obs_cols)
        cols.update(act_cols)

        for k, arr in cols.items():
            rows.setdefault(k, []).append(arr)

        ep_ret = float(np.sum(rew)) if T else 0.0
        per_rollout.append({
            "rollout_id": rid,
            "path": str(p),
            "n_steps": int(T),
            "episode_return": ep_ret,
            "has_time": bool(_has_time(payload)),
        })

        total_steps += int(T)
        total_return += ep_ret

    # concat columns
    stitched: Dict[str, np.ndarray] = {}
    for k, parts in rows.items():
        if k == "rollout_id":
            stitched[k] = np.concatenate(parts, axis=0).astype(object)
        else:
            stitched[k] = np.concatenate(parts, axis=0)

    # sort
    if have_time_any and (stitched["day_of_year"] >= 0).any():
        # Sort by (day_of_year, minute_of_day, file_index, step_index)
        keys = np.lexsort((stitched["step_index"], stitched["file_index"], stitched["minute_of_day"], stitched["day_of_year"]))
    else:
        keys = np.lexsort((stitched["step_index"], stitched["file_index"]))
    for k in stitched:
        stitched[k] = stitched[k][keys]

    summary = {
        "n_rollouts": int(len(rollout_paths)),
        "total_steps": int(total_steps),
        "total_return": float(total_return),
        "have_time_any": bool(have_time_any),
        "per_rollout": per_rollout,
    }
    return stitched, summary


def to_pandas(stitched: Dict[str, np.ndarray]):
    import pandas as pd  # optional dependency
    df = pd.DataFrame({k: (v if v.dtype != object else v.astype(str)) for k, v in stitched.items()})
    # Add a derived timestamp for convenience if time exists
    if "day_of_year" in df.columns and "minute_of_day" in df.columns:
        # Use a dummy non-leap year; you can override later if you want.
        year = 2021
        valid = (df["day_of_year"] >= 1) & (df["minute_of_day"] >= 0)
        if valid.any():
            # vectorized: start at Jan 1 then add days/minutes
            base = np.datetime64(f"{year}-01-01T00:00")
            df.loc[valid, "timestamp"] = base + (df.loc[valid, "day_of_year"].astype("int64") - 1).values.astype("timedelta64[D]") + df.loc[valid, "minute_of_day"].astype("int64").values.astype("timedelta64[m]")
    return df


def write_outputs(stitched: Dict[str, np.ndarray], summary: Dict[str, Any], out_path: Path, fmt: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[ok] wrote summary: {summary_path}")

    fmt = fmt.lower().strip()
    if fmt == "csv":
        try:
            import pandas as pd
            df = to_pandas(stitched)
            df.to_csv(out_path, index=False)
        except Exception as e:
            # fallback: numpy.savetxt-like is awkward for mixed types; require pandas for CSV
            raise RuntimeError(f"CSV output requires pandas. Install pandas or use --format npz. Error: {e}")
        print(f"[ok] wrote csv: {out_path}")
        return

    if fmt == "parquet":
        try:
            df = to_pandas(stitched)
            df.to_parquet(out_path, index=False)
        except Exception as e:
            raise RuntimeError(f"Parquet output requires pandas + pyarrow (recommended) or fastparquet. Error: {e}")
        print(f"[ok] wrote parquet: {out_path}")
        return

    if fmt == "npz":
        np.savez_compressed(out_path, **stitched)
        print(f"[ok] wrote npz: {out_path}")
        return

    raise ValueError(f"Unknown --format {fmt!r}. Use csv|parquet|npz.")


def maybe_plot(stitched: Dict[str, np.ndarray], plot_out: Path) -> None:
    """
    Plot a couple quick series:
      - energy_meter vs stitched index
      - reward vs stitched index
    If time exists, x-axis is day_of_year + minute_of_day/1440.
    """
    import matplotlib.pyplot as plt

    n = stitched["reward"].shape[0]
    if n == 0:
        print("[plot] no rows to plot")
        return

    if ("day_of_year" in stitched) and ("minute_of_day" in stitched) and (stitched["day_of_year"] >= 1).any():
        x = stitched["day_of_year"].astype(np.float64) + stitched["minute_of_day"].astype(np.float64) / 1440.0
        xlab = "day_of_year + minute/1440"
    else:
        x = np.arange(n)
        xlab = "stitched_index"

    # Energy plot
    if "energy_meter" in stitched:
        y = stitched["energy_meter"].astype(np.float64)
        plt.figure()
        plt.plot(x, y)
        plt.xlabel(xlab)
        plt.ylabel("energy_meter (raw)")
        plt.title("Energy meter vs time")
        plt.tight_layout()
        p1 = plot_out.with_name(plot_out.stem + "_energy" + plot_out.suffix)
        plt.savefig(p1, dpi=150)
        plt.close()
        print(f"[plot] wrote {p1}")

    # Reward plot
    plt.figure()
    plt.plot(x, stitched["reward"].astype(np.float64))
    plt.xlabel(xlab)
    plt.ylabel("reward")
    plt.title("Reward vs time")
    plt.tight_layout()
    p2 = plot_out.with_name(plot_out.stem + "_reward" + plot_out.suffix)
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f"[plot] wrote {p2}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts-dir", required=True, help="Directory containing rollout_*.npz files.")
    ap.add_argument("--pattern", default="rollout_*.npz", help="Glob pattern inside rollouts-dir.")
    ap.add_argument("--out", required=True, help="Output path (csv/parquet/npz depending on --format).")
    ap.add_argument("--format", default="csv", choices=["csv", "parquet", "npz"], help="Output format.")
    ap.add_argument("--plot", action="store_true", help="Generate quick plots (energy + reward).")
    ap.add_argument("--plot-out", default="stitched.png", help="Plot output basename (suffix used).")
    args = ap.parse_args()

    rollouts_dir = Path(args.rollouts_dir)
    paths = sorted(glob.glob(str(rollouts_dir / args.pattern)))
    if not paths:
        raise SystemExit(f"No rollouts found in {rollouts_dir} matching {args.pattern!r}")

    stitched, summary = stitch_rollouts(paths)
    write_outputs(stitched, summary, Path(args.out), args.format)

    if args.plot:
        maybe_plot(stitched, Path(args.plot_out))


if __name__ == "__main__":
    main()
