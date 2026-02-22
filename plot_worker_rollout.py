#!/usr/bin/env python3
"""
plot_worker_rollout.py

Plot per-worker timeseries artifacts written by WorkerTimeseriesWriter.

Inputs:
  - timeseries_<rollout_id>.parquet (preferred)
  - timeseries_<rollout_id>.csv     (fallback)

Outputs:
  - temps_setpoints.png
  - energy_reward.png

Examples:
  python plot_worker_rollout.py --path ./shared/results/test123/timeseries_test123.parquet \
    --zones "CORE_BOTTOM,PERIMETER_BOT_ZN_3" --resample 15min --outdir ./shared/results/test123

  python plot_worker_rollout.py --path ./shared/results/test123/timeseries_test123.parquet \
    --top-zones 3 --resample 1H
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ZONE_PREFIXES = {
    "tz": "tz_c__",
    "heat": "heat_sp_c__",
    "cool": "cool_sp_c__",
    "act": "act_norm__",
}


def _detect_zones(df: pd.DataFrame) -> List[str]:
    zones = []
    for c in df.columns:
        if c.startswith(ZONE_PREFIXES["tz"]):
            zones.append(c[len(ZONE_PREFIXES["tz"]):])
    # de-dupe, preserve stable order
    seen = set()
    out = []
    for z in zones:
        if z not in seen:
            out.append(z)
            seen.add(z)
    return out


def _select_zones(all_zones: List[str], zones_arg: str | None, top_n: int) -> List[str]:
    if zones_arg:
        wanted = [z.strip().replace(" ", "_") for z in zones_arg.split(",") if z.strip()]
        # keep only ones that exist
        return [z for z in wanted if z in all_zones]

    # Default: pick first N zones (stable) to avoid clutter
    return all_zones[: max(1, top_n)]


def _load_timeseries(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix} (expected .parquet or .csv)")


def _build_time_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """
    We don't have absolute timestamps; we build a synthetic datetime index that is
    monotonic and supports resampling.

    Use step_idx if present; otherwise use row number.
    """
    if "step_idx" in df.columns:
        step = pd.to_numeric(df["step_idx"], errors="coerce").fillna(0).astype(int)
    else:
        step = pd.Series(np.arange(len(df)), name="step_idx")

    # 1-minute synthetic spacing (resampling still works; it's a relative timeline)
    start = pd.Timestamp("2000-01-01 00:00:00")
    return start + pd.to_timedelta(step, unit="min")


def _apply_resample(df: pd.DataFrame, rule: str | None) -> pd.DataFrame:
    if not rule:
        return df

    # Resample numeric columns with mean; energy_kwh often better as sum per bucket.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Heuristic: sum energy per period, mean everything else
    sum_cols = [c for c in numeric_cols if c in ("hvac_energy_kwh",)]
    mean_cols = [c for c in numeric_cols if c not in sum_cols]

    agg = {}
    for c in mean_cols:
        agg[c] = "mean"
    for c in sum_cols:
        agg[c] = "sum"

    return df.resample(rule).agg(agg)


def _plot_temps_setpoints(df: pd.DataFrame, zones: List[str], outpath: Path) -> None:
    fig = plt.figure()
    ax = plt.gca()

    # Plot OAT as thin context line (if present)
    if "outside_air_c" in df.columns:
        ax.plot(df.index, df["outside_air_c"], label="OAT (C)")

    for z in zones:
        tz = f"{ZONE_PREFIXES['tz']}{z}"
        hs = f"{ZONE_PREFIXES['heat']}{z}"
        cs = f"{ZONE_PREFIXES['cool']}{z}"

        if tz in df.columns:
            ax.plot(df.index, df[tz], label=f"Tz {z}")
        if hs in df.columns:
            ax.plot(df.index, df[hs], label=f"HeatSP {z}")
        if cs in df.columns:
            ax.plot(df.index, df[cs], label=f"CoolSP {z}")

    ax.set_title("Zone Temps + Setpoints (with OAT)")
    ax.set_xlabel("time (synthetic)")
    ax.set_ylabel("deg C")
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _plot_energy_reward(df: pd.DataFrame, outpath: Path) -> None:
    fig = plt.figure()
    ax1 = plt.gca()

    # Energy
    if "hvac_energy_kwh" in df.columns:
        ax1.plot(df.index, df["hvac_energy_kwh"], label="HVAC energy (kWh/step)")
        ax1.set_ylabel("kWh/step (or kWh per resample bucket if resampled)")
    else:
        ax1.set_ylabel("")

    ax1.set_xlabel("time (synthetic)")

    # Reward on secondary axis
    if "reward" in df.columns:
        ax2 = ax1.twinx()
        ax2.plot(df.index, df["reward"], label="reward")
        ax2.set_ylabel("reward")

        # Combined legend
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
    else:
        ax1.legend(loc="best", fontsize=8)

    ax1.set_title("Energy + Reward")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to timeseries_<rollout_id>.parquet (or .csv)")
    ap.add_argument("--zones", default=None, help="Comma-separated zone list. If omitted uses --top-zones.")
    ap.add_argument("--top-zones", type=int, default=2, help="If --zones not provided, plot first N zones found.")
    ap.add_argument("--resample", default=None, help='Pandas resample rule (e.g. "15min", "1H", "D").')
    ap.add_argument("--outdir", default=None, help="Output directory. Defaults to the file's parent directory.")
    args = ap.parse_args()

    path = Path(args.path)
    outdir = Path(args.outdir) if args.outdir else path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    df = _load_timeseries(path)

    # Build synthetic time index for resampling/plotting
    df = df.copy()
    df.index = _build_time_index(df)

    # Ensure numeric columns are numeric
    for c in df.columns:
        if c == "step_idx":
            continue
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Apply resample if requested
    df = _apply_resample(df, args.resample)

    # Detect zones and select
    all_zones = _detect_zones(df)
    if not all_zones:
        raise RuntimeError("No zones detected. Expected columns like tz_c__<ZONE> in the timeseries file.")

    zones = _select_zones(all_zones, args.zones, args.top_zones)
    if not zones:
        raise RuntimeError(f"No matching zones. Available zones: {all_zones}")

    # Plots
    _plot_temps_setpoints(df, zones, outdir / "temps_setpoints.png")
    _plot_energy_reward(df, outdir / "energy_reward.png")

    print(f"Wrote:\n  {outdir / 'temps_setpoints.png'}\n  {outdir / 'energy_reward.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())