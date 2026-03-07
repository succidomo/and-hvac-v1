import argparse
import os
from pathlib import Path

import numpy as np
import uuid
from collections import deque

from pyenergyplus.api import EnergyPlusAPI

from eplus_runperiod_utils import rewrite_first_runperiod
from andruix_policy_model import TorchPolicyModel
from andruix_rollout_writer import RolloutWriter, WorkerTimeseriesWriter
from andruix_simple_model import SimpleRLModel


class RLController:
    """EnergyPlus callback controller.

    Step 1.2 change (trend features):
      - Observation includes time features AND short-horizon *trend* features from past measurements
        (no external forecast required).

    Default obs (all toggles ON):
      [
        Tzone,
        Toa,
        sin(tod), cos(tod),
        sin(doy), cos(doy),
        occ_flag,
        dTzone_60m, dToa_60m,
        dTzone_15m, dToa_15m,
      ]

    NOTE: Existing torch policies trained with earlier obs_dim will NOT be compatible.
          Start a fresh training run with orchestrator --obs-dim matching this worker.
    """

    def __init__(
        self,
        api: EnergyPlusAPI,
        state,
        zone_names: str,
        outdir: Path,
        rollout_dir: Path,
        rollout_id: str,
        policy_path: str | os.PathLike | None = None,
        policy_kind: str = "torch",
        energy_meter_name: str = "Electricity:Building",
        dump_api_available_csv: bool = True,
        debug_meter_every_n_steps: int = 20,
        reward_scale: float = 1,

        # --- Observation features ---
        include_occ_flag: bool = True,
        include_doy_features: bool = True,
        include_trend_60m: bool = True,
        include_trend_15m: bool = True,

        # --- Action semantics (single action -> thermostat centerline) ---
        sp_center_min_occ: float = 21.0,
        sp_center_max_occ: float = 24.0,
        sp_center_min_unocc: float = 18.0,
        sp_center_max_unocc: float = 28.0,
        deadband_occ: float = 1.0,
        deadband_unocc: float = 2.0,
        occupied_start_minute: int = 8 * 60,
        occupied_end_minute: int = 18 * 60,

        # --- Reward shaping: comfort + hard constraints ---
        comfort_low: float = 21.0,
        comfort_high: float = 25.0,
        comfort_weight: float = 0.1,
        hard_low: float = 15.0,
        hard_high: float = 30.0,
        hard_weight: float = 5.0,
        slew_weight: float = 0.0,

        # --- Setpoint safety clamps ---
        min_heat_sp: float = 15.0,
        max_cool_sp: float = 30.0,
        min_deadband: float = 0.5,
        control_minutes: int = 15,
    ):
        self.api = api
        self.state = state
        self.ZONES = [z.strip() for z in str(zone_names).split(',') if z.strip()]
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.rollout_dir = Path(rollout_dir)
        self.rollout_dir.mkdir(parents=True, exist_ok=True)
        self.rollout_id = rollout_id

        # Observation config
        self.include_occ_flag = bool(include_occ_flag)
        self.include_doy_features = bool(include_doy_features)
        self.include_trend_60m = bool(include_trend_60m)
        self.include_trend_15m = bool(include_trend_15m)

        # History buffer (absolute simulation minutes -> temps)
        self._hist = deque(maxlen=3000)

        # Compute obs_dim / act_dim (Path A)
        # obs = [Tz_1..Tz_N] + [Toa] + [sin_tod, cos_tod] + trends + [sin_doy, cos_doy] + [occ_flag]
        n_z = len(self.ZONES)
        obs_dim = n_z  # zone temps
        obs_dim += 1   # outside air temp
        obs_dim += 2   # sin_tod, cos_tod (always included)
        if self.include_trend_60m:
            obs_dim += 2
        if self.include_trend_15m:
            obs_dim += 2
        if self.include_doy_features:
            obs_dim += 2
        if self.include_occ_flag:
            obs_dim += 1
        self.obs_dim = int(obs_dim)
        self.act_dim = int(n_z)

        # IMPORTANT: For get_meter_handle(), pass just the meter name (no ",hourly").
        self.energy_meter_name = energy_meter_name.split(",")[0].strip()
        self.dump_api_available_csv = dump_api_available_csv
        self.debug_meter_every_n_steps = debug_meter_every_n_steps
        self.reward_scale = float(reward_scale) if reward_scale else 1.0

        # ---- Control interval (aggregate SystemTimesteps into one RL step) ----
        self.control_minutes = int(control_minutes) if control_minutes else 15

        # reward/accounting bucket (what accumulators belong to)
        self._bucket_id = None

        # action bucket (what held action has been applied for)
        self._action_bucket_id = None

        # staged boundary info from begin-callback until end-callback closes old bucket
        self._pending_boundary = None

        # bucket accumulators (sum over SystemTimesteps)
        self._bucket_rew = 0.0
        self._bucket_energy_kwh_sum = 0.0
        self._bucket_sys_steps = 0
        self._bucket_uncomfort_min = 0.0

        # minute tracking to avoid over-counting sub-minute callbacks
        self._bucket_last_mod = None
        self._last_uncomfortable = False
        self._last_occupied = False

        # weight for minutes-uncomfortable (env configurable)
        self.uncomfort_min_weight = float(os.environ.get("ANDRUIX_UNCOMFORT_MIN_WEIGHT", "1.0"))

        # last seen time (for trend history de-dupe)
        self._last_hist_abs_minute = None

        # debug counters
        self.sys_step_count = 0      # raw EnergyPlus SystemTimesteps
        self.step_count = 0          # RL control steps (15-min buckets)

        # “current policy outputs that are being held”
        self._held_act = np.zeros((len(self.ZONES),), dtype=np.float32)
        self._held_heat_sp = {z: None for z in self.ZONES}
        self._held_cool_sp = {z: None for z in self.ZONES}

        # last known state snapshot (so finalize() can write last bucket row)
        self._last_zone_temps = None
        self._last_outside_temp = None
        self._last_doy = None
        self._last_mod = None
        self._bucket_oat_sum = 0.0
        self._bucket_oat_count = 0
        self._bucket_mean_oat = 0.0

        # ---- Action mapping config ----
        self.sp_center_min_occ = float(sp_center_min_occ)
        self.sp_center_max_occ = float(sp_center_max_occ)
        self.sp_center_min_unocc = float(sp_center_min_unocc)
        self.sp_center_max_unocc = float(sp_center_max_unocc)
        self.deadband_occ = float(deadband_occ)
        self.deadband_unocc = float(deadband_unocc)
        self.occupied_start_minute = int(occupied_start_minute)
        self.occupied_end_minute = int(occupied_end_minute)

        # ---- Reward shaping config ----
        self.comfort_low = float(comfort_low)
        self.comfort_high = float(comfort_high)
        self.comfort_weight = float(comfort_weight)
        self.hard_low = float(hard_low)
        self.hard_high = float(hard_high)
        self.hard_weight = float(hard_weight)
        self.slew_weight = float(slew_weight)

        # ---- Safety clamps ----
        self.min_heat_sp = float(min_heat_sp)
        self.max_cool_sp = float(max_cool_sp)
        self.min_deadband = float(min_deadband)

        self.policy_kind = (policy_kind or "simple").lower()
        self._is_torch_policy = self.policy_kind == "torch"

        print(f"Energy Meter used for reward fuction: {self.energy_meter_name}")

        # Worker-side policy (loads once at init, no per-timestep RPC)
        if self._is_torch_policy:
            print("TorchPolicyModel..... loaded")
            self.rl_model = TorchPolicyModel(policy_path)
        else:
            print("SimpleRLModel..... loaded")
            self.rl_model = SimpleRLModel(num_zones=len(self.ZONES))

        # TD3-style replay writer
        self.rollout_writer = RolloutWriter(
            self.rollout_dir, 
            self.rollout_id, 
            obs_dim=self.obs_dim, 
            act_dim=self.act_dim,
            zones=self.ZONES,
            start_mmdd=os.getenv("EPLUS_START_MMDD"),
            end_mmdd=os.getenv("EPLUS_END_MMDD"),
            reward_mode=os.getenv("ANDRUIX_REWARD_MODE"),
            reward_scale=float(os.getenv("ANDRUIX_REWARD_SCALE", "0") or 0) or None,
            obs_flags={
                "include_occ": bool(int(os.getenv("ANDRUIX_OBS_OCC", "0"))),
                "no_doy": bool(int(os.getenv("ANDRUIX_OBS_NO_DOY", "0"))),
                "no_trend_15m": bool(int(os.getenv("ANDRUIX_OBS_NO_TREND_15M", "0"))),
                "no_trend_60m": bool(int(os.getenv("ANDRUIX_OBS_NO_TREND_60M", "0"))),
            },
            policy_fingerprint=getattr(self.rl_model, "policy_hash", None),
        )

        # Per-worker debug timeseries writer (parquet/csv) keyed by rollout_id
        self.ts_writer = WorkerTimeseriesWriter(
            out_dir=self.outdir,          # writes to /shared/results/<rollout_id>/ (your --outdir)
            rollout_id=self.rollout_id,
            zones=self.ZONES,
            image_tag=os.getenv("ANDRUIX_IMAGE_TAG") or None,
            policy_fingerprint=getattr(self.rl_model, "policy_hash", None),
        )
        self._ts_pending = None  # begin-callback stash; flushed in end-callback

        # Handle readiness / sentinels
        self.handles_ready = False
        self.api_dumped = False
        self.facility_elec_meter_handle = -1
        self.heat_sp_handles = {}
        self.cool_sp_handles = {}
        self.room_temp_handles = {}         # zone_name → handle
        self.outside_temp_handle = None      # shared
        self.step_count = 0
        self.dbg_count = 0
        self.init_attempts = 0

        self._prev_obs = None
        self._prev_act = None
        self._pending_rew = None
        self._pending_energy = None
        self._prev_day_of_year = None
        self._prev_minute_of_day = None

        # Last applied setpoints (for comfort + slew penalty)
        self._last_heat_sp = {zone: None for zone in self.ZONES}
        self._last_cool_sp = {zone: None for zone in self.ZONES}
        self._prev_center_sp = {zone: None for zone in self.ZONES}
        self._last_center_sp = {zone: None for zone in self.ZONES}

        # List of (variable_name, key) pairs we want
        variables_to_request = [
            ("Site Outdoor Air Drybulb Temperature", "Environment"),
        ]

        # Add one entry per zone for zone mean air temperature
        for zone_name in self.ZONES:
            variables_to_request.append(("Zone Mean Air Temperature", zone_name))

        # Now request them all
        for var_name, key in variables_to_request:
            self.api.exchange.request_variable(state, var_name, key)

        # Callbacks
        self.api.runtime.callback_after_new_environment_warmup_complete(state, self._try_init_handles)

        # Act only on 15-min bucket boundaries (runs every SystemTimestep, but we gate it)
        self.api.runtime.callback_begin_system_timestep_before_predictor(
            state, self.begin_system_timestep_callback
        )

        # Reward accumulation (every SystemTimestep)
        self.api.runtime.callback_end_system_timestep_after_hvac_reporting(
            state, self.end_system_timestep_callback
        )

    # ---- init / readiness ----
    def _try_init_handles(self, state):
        if self.handles_ready:
            return
        if not self.api.exchange.api_data_fully_ready(state):
            return
        if self.api.exchange.warmup_flag(state):
            return

        if self.dump_api_available_csv and not self.api_dumped:
            try:
                csv_bytes = self.api.exchange.list_available_api_data_csv(state)
                (self.outdir / 'api_available.csv').write_bytes(csv_bytes)
                self.api_dumped = True
                print(f"[init] Wrote {self.outdir / 'api_available.csv'}")
            except Exception as e:
                print(f"[init] WARNING: failed to write api_available.csv: {e}")
                self.api_dumped = True

        # Shared handles
        self.outside_temp_handle = self.api.exchange.get_variable_handle(state, 'Site Outdoor Air Drybulb Temperature', 'Environment')
        self.facility_elec_meter_handle = self.api.exchange.get_meter_handle(state, self.energy_meter_name)

        # Per-zone handles
        for zone in self.ZONES:
            self.heat_sp_handles[zone] = self.api.exchange.get_actuator_handle(state, 'Zone Temperature Control', 'Heating Setpoint', zone)
            self.cool_sp_handles[zone] = self.api.exchange.get_actuator_handle(state, 'Zone Temperature Control', 'Cooling Setpoint', zone)
            self.room_temp_handles[zone] = self.api.exchange.get_variable_handle(state, 'Zone Mean Air Temperature', zone)

        # Validate
        missing: list[str] = []
        if self.outside_temp_handle == -1:
            missing.append('outside_temp_handle')
        if self.facility_elec_meter_handle == -1:
            missing.append(f"meter({self.energy_meter_name})")
        for z in self.ZONES:
            if self.room_temp_handles.get(z, -1) == -1:
                missing.append(f"room_temp_handle[{z}]")
            if self.heat_sp_handles.get(z, -1) == -1:
                missing.append(f"heat_sp_handle[{z}]")
            if self.cool_sp_handles.get(z, -1) == -1:
                missing.append(f"cool_sp_handle[{z}]")

        self.init_attempts += 1
        if missing:
            if self.init_attempts % 50 == 0:
                print(f"[init] Missing handles (will retry): {missing}")
            return

        self.handles_ready = True
        self._hist.clear()
        print(f"[init] Handles resolved ✔ meter='{self.energy_meter_name}' obs_dim={self.obs_dim} act_dim={self.act_dim}")

    def _ensure_ready(self, state) -> bool:
        if not self.api.exchange.api_data_fully_ready(state):
            return False
        if not self.handles_ready:
            self._try_init_handles(state)
            return False
        return True

    def _get_time_index(self, state):
        """Return (day_of_year, minute_of_day, abs_minute) from EnergyPlus simulation clock."""
        try:
            doy = int(self.api.exchange.day_of_year(state))
            hr = int(self.api.exchange.hour(state))
            mn = int(self.api.exchange.minutes(state))
            if hr >= 24:
                hr = 23
                mn = 59
            mod = max(0, min(1439, hr * 60 + mn))
            abs_minute = (max(1, doy) - 1) * 1440 + mod
            return doy, mod, abs_minute
        except Exception:
            return None, None, None

    def _is_occupied(self, minute_of_day: int | None) -> bool:
        if minute_of_day is None:
            return True
        start = int(self.occupied_start_minute)
        end = int(self.occupied_end_minute)
        m = int(minute_of_day)
        if start == end:
            return True
        if start < end:
            return start <= m < end
        return m >= start or m < end

    # ---- trend features ----
    def _push_history(self, abs_minute, avg_zone_temp, outside_temp):
        if abs_minute is None:
            return
        try:
            tz = float(avg_zone_temp)
            toa = float(outside_temp)
        except Exception:
            return
        if not (np.isfinite(tz) and np.isfinite(toa)):
            return
        self._hist.append((int(abs_minute), tz, toa))

    def _lookup_at_or_before(self, target_abs_minute: int):
        for t, tz, toa in reversed(self._hist):
            if t <= target_abs_minute and np.isfinite(tz) and np.isfinite(toa):
                return float(tz), float(toa)
        return None

    def _trend_deltas(self, abs_minute: int | None, room_temp: float, outside_temp: float) -> list[float]:
        feats: list[float] = []

        # Not enough info yet
        if abs_minute is None or not self._hist:
            if self.include_trend_60m:
                feats += [0.0, 0.0]
            if self.include_trend_15m:
                feats += [0.0, 0.0]
            return feats

        cur_tz = float(room_temp)
        cur_toa = float(outside_temp)

        # If current values are bad, don't emit NaNs
        if not (np.isfinite(cur_tz) and np.isfinite(cur_toa)):
            if self.include_trend_60m:
                feats += [0.0, 0.0]
            if self.include_trend_15m:
                feats += [0.0, 0.0]
            return feats

        if self.include_trend_60m:
            past = self._lookup_at_or_before(int(abs_minute) - 60)
            if past is None:
                feats += [0.0, 0.0]
            else:
                p_tz, p_toa = past
                if not (np.isfinite(p_tz) and np.isfinite(p_toa)):
                    feats += [0.0, 0.0]
                else:
                    feats += [cur_tz - p_tz, cur_toa - p_toa]

        if self.include_trend_15m:
            past = self._lookup_at_or_before(int(abs_minute) - 15)
            if past is None:
                feats += [0.0, 0.0]
            else:
                p_tz, p_toa = past
                if not (np.isfinite(p_tz) and np.isfinite(p_toa)):
                    feats += [0.0, 0.0]
                else:
                    feats += [cur_tz - p_tz, cur_toa - p_toa]

        return feats


    def _make_obs_multi_zone(self, zone_temps_dict, outside_temp, doy, mod, abs_minute):
        """Observation layout (Path A):
        [Tz_1..Tz_N] + [Toa] + [sin_tod, cos_tod] +
        [dTzAvg_60, dToa_60] + [dTzAvg_15, dToa_15] +
        [sin_doy, cos_doy] + [occ_flag]
        """
        obs_parts: list[float] = []

        # 1) Zone temps in the provided zone order
        tz_list: list[float] = []
        for z in self.ZONES:
            v = zone_temps_dict.get(z, None)
            if v is None:
                tz_list.append(float('nan'))
                print(f"[make_obs_bad] step={self.step_count} | zone={z} | missing value → NaN")
                continue

            try:
                tz = float(v)
                if not np.isfinite(tz):
                    raise ValueError(f"non-finite value: {tz}")
                tz_list.append(tz)
            except (ValueError, TypeError) as e:
                tz_list.append(float('nan'))
                print(f"[make_obs_bad] step={self.step_count} | zone={z} | value={v!r} → NaN | error={str(e)}")

        obs_parts.extend(tz_list)

        # 2) Outside air temp
        try:
            toa = float(outside_temp)
            if not np.isfinite(toa):
                raise ValueError("non-finite Toa")
        except (ValueError, TypeError) as e:
            print(f"[make_obs_bad] Toa conversion failed  value={outside_temp!r}  error={str(e)}")
            toa = float('nan')
        obs_parts.append(toa)

        # 3) Time-of-day sin/cos (always included)
        if mod is None:
            sin_tod, cos_tod = 0.0, 0.0
        else:
            tod = float(mod) / 1440.0
            sin_tod = float(np.sin(2.0 * np.pi * tod))
            cos_tod = float(np.cos(2.0 * np.pi * tod))
        obs_parts.extend([sin_tod, cos_tod])

        # 4) Shared trend deltas using avg zone temp + outside temp
        if len(tz_list) > 0:
            tz_avg = float(np.nanmean(tz_list))
        else:
            tz_avg = float('nan')
            print(f"[make_obs_bad] tz_list was empty → returning NaN")

        obs_parts.extend(self._trend_deltas(abs_minute, tz_avg, toa))

        # 5) Day-of-year sin/cos (optional)
        if self.include_doy_features:
            if doy is None:
                obs_parts.extend([0.0, 0.0])
            else:
                frac = float(doy) / 365.0
                obs_parts.extend([float(np.sin(2.0 * np.pi * frac)), float(np.cos(2.0 * np.pi * frac))])

        # 6) Occupancy flag (optional)
        occupied = self._is_occupied(mod)
        if self.include_occ_flag:
            obs_parts.append(1.0 if occupied else 0.0)

        return np.asarray(obs_parts, dtype=np.float32), occupied

    def _coerce_action_vec(self, act, n: int) -> np.ndarray:
        """Minimal contract enforcement for actions.
        - ensures shape (n,)
        - replaces NaN/Inf with 0
        - clips to [-1, 1] because _map_action_to_setpoints expects normalized actions
        """
        try:
            a = np.asarray(act, dtype=np.float32).reshape(-1)
        except Exception:
            a = np.zeros((n,), dtype=np.float32)

        if a.size == 1 and n > 1:
            a = np.full((n,), float(a[0]), dtype=np.float32)
        elif a.size < n:
            pad = np.zeros((n - a.size,), dtype=np.float32)
            a = np.concatenate([a, pad], axis=0)
        elif a.size > n:
            a = a[:n]

        a = np.where(np.isfinite(a), a, 0.0).astype(np.float32)
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def _coerce_action(self, act) -> float:
        """Back-compat scalar coercion."""
        return float(self._coerce_action_vec(act, 1)[0])

    def _map_action_to_setpoints(self, a: float, occupied: bool) -> tuple[float, float, float]:
        if occupied:
            cmin, cmax = self.sp_center_min_occ, self.sp_center_max_occ
            deadband = self.deadband_occ
        else:
            cmin, cmax = self.sp_center_min_unocc, self.sp_center_max_unocc
            deadband = self.deadband_unocc

        center = float(cmin + 0.5 * (a + 1.0) * (cmax - cmin))
        heat = center - 0.5 * float(deadband)
        cool = center + 0.5 * float(deadband)

        heat = max(heat, self.min_heat_sp)
        cool = min(cool, self.max_cool_sp)

        if (cool - heat) < self.min_deadband:
            mid = 0.5 * (heat + cool)
            heat = mid - 0.5 * self.min_deadband
            cool = mid + 0.5 * self.min_deadband
            if heat < self.min_heat_sp:
                heat = self.min_heat_sp
                cool = heat + self.min_deadband
            if cool > self.max_cool_sp:
                cool = self.max_cool_sp
                heat = cool - self.min_deadband

        center = 0.5 * (heat + cool)
        return float(heat), float(cool), float(center)

    def _temp_violation(self, temp_c: float, low: float, high: float) -> float:
        below = max(0.0, float(low) - float(temp_c))
        above = max(0.0, float(temp_c) - float(high))
        return below + above
    
    def _bucket_from_abs_minute(self, abs_minute: int) -> int:
        # aligned 15-min buckets across the whole run
        return int(abs_minute) // int(self.control_minutes)

    def _read_zone_temps(self, state) -> dict[str, float]:
        d: dict[str, float] = {}
        for z in self.ZONES:
            h = self.room_temp_handles.get(z, -1)
            if h == -1:
                d[z] = float("nan")
                continue
            v = self.api.exchange.get_variable_value(state, h)
            d[z] = float(v) if v is not None else float("nan")
        return d

    def _read_outside_temp(self, state) -> float:
        v = self.api.exchange.get_variable_value(state, self.outside_temp_handle)
        return float(v) if v is not None else float("nan")

    def _apply_action_vec(self, state, a_norm: np.ndarray, occupied: bool) -> None:
        # Apply setpoints + update held setpoints for re-assertion
        for i, zone in enumerate(self.ZONES):
            heat_sp, cool_sp, center_sp = self._map_action_to_setpoints(float(a_norm[i]), occupied)

            self.api.exchange.set_actuator_value(state, self.heat_sp_handles[zone], heat_sp)
            self.api.exchange.set_actuator_value(state, self.cool_sp_handles[zone], cool_sp)

            self._held_heat_sp[zone] = float(heat_sp)
            self._held_cool_sp[zone] = float(cool_sp)

            # Track for slew (only changes when we apply a new action)
            self._last_heat_sp[zone] = heat_sp
            self._last_cool_sp[zone] = cool_sp
            if self._last_center_sp[zone] is not None:
                self._prev_center_sp[zone] = self._last_center_sp[zone]
            self._last_center_sp[zone] = center_sp

    def _select_action_vec(self, obs: np.ndarray) -> np.ndarray:
        a_pol = np.asarray(self.rl_model.get_action(obs), dtype=np.float32).reshape(-1)
        if a_pol.size == 1:
            a_pol = np.full((len(self.ZONES),), float(a_pol.item()), dtype=np.float32)

        explore_noise = float(os.environ.get("ANDRUIX_EXPLORE_NOISE", "0.0"))
        a_raw = a_pol.copy()
        if explore_noise > 0.0:
            a_raw = a_raw + np.random.normal(0.0, explore_noise, size=len(self.ZONES)).astype(np.float32)

        a_norm = self._coerce_action_vec(a_raw, n=len(self.ZONES))

        # (optional) saturation log
        if self.step_count % 10 == 0:
            sat = float(np.mean(np.abs(a_norm) >= 0.98))
            print(f"[a_ctl] step={self.step_count} sat_frac={sat:.2f} a_norm={np.round(a_norm,3)}")

        return a_norm

    def _write_bucket_timeseries(
        self,
        doy: int,
        mod: int,
        outside_temp: float,
        zone_temps: dict[str, float],
        reward_sum: float,
        energy_kwh_sum: float,
        occupied: bool,
        held_act: np.ndarray | None = None,
        held_heat_sp: dict[str, float] | None = None,
        held_cool_sp: dict[str, float] | None = None,
    ) -> None:
        act_vec = self._held_act if held_act is None else held_act
        heat_map = self._held_heat_sp if held_heat_sp is None else held_heat_sp
        cool_map = self._held_cool_sp if held_cool_sp is None else held_cool_sp

        self.ts_writer.append_step(
            step_idx=int(self.step_count),
            day_of_year=int(doy),
            minute_of_day=int(mod),
            outside_air_c=float(outside_temp),
            hvac_energy_kwh=float(energy_kwh_sum),
            reward=float(reward_sum),
            zone_temps_c={k: float(v) for k, v in zone_temps.items()},
            zone_setpoints_heat_c={z: float(heat_map[z]) for z in self.ZONES if heat_map[z] is not None},
            zone_setpoints_cool_c={z: float(cool_map[z]) for z in self.ZONES if cool_map[z] is not None},
            zone_actions_norm={z: float(act_vec[i]) for i, z in enumerate(self.ZONES)},
            extra_scalars={"occupied": 1.0 if occupied else 0.0},
        )

    def _finalize_bucket(
        self,
        next_obs: np.ndarray,
        next_doy: int,
        next_mod: int,
        next_zone_temps: dict[str, float],
        next_outside_temp: float,
        next_occupied: bool,
        prev_act_snapshot: np.ndarray,
        prev_heat_sp_snapshot: dict[str, float],
        prev_cool_sp_snapshot: dict[str, float],
    ) -> None:
        energy_sum = float(self._bucket_energy_kwh_sum)
        uncomfort_min = float(self._bucket_uncomfort_min)

        energy_term_bucket = energy_sum / float(self.reward_scale)
        comfort_term_bucket = float(self.uncomfort_min_weight) * uncomfort_min
        self._bucket_rew = -energy_term_bucket - comfort_term_bucket

        print(
            f"[bucket] doy={next_doy} min={next_mod} energy_kwh_sum={energy_sum:.3f} "
            f"uncomfort_min={uncomfort_min:.1f} reward={self._bucket_rew:.3f} "
            f"bucket_sys_steps={self._bucket_sys_steps}"
        )

        if self._prev_obs is not None and self._prev_act is not None:
            self.rollout_writer.append(
                obs=self._prev_obs,
                act=np.asarray(prev_act_snapshot, dtype=np.float32),
                rew=float(self._bucket_rew),
                next_obs=np.asarray(next_obs, dtype=np.float32),
                done=0.0,
                day_of_year=self._prev_day_of_year,
                minute_of_day=self._prev_minute_of_day,
                energy_meter=float(energy_sum),
                outside_air_c=float(self._bucket_mean_oat),
            )

        self.step_count += 1

        self._write_bucket_timeseries(
            next_doy,
            next_mod,
            next_outside_temp,
            next_zone_temps,
            self._bucket_rew,
            energy_sum,
            next_occupied,
            held_act=np.asarray(prev_act_snapshot, dtype=np.float32),
            held_heat_sp=prev_heat_sp_snapshot,
            held_cool_sp=prev_cool_sp_snapshot,
        )

    # ---- callbacks ----
    def begin_system_timestep_callback(self, state):
        if not self._ensure_ready(state):
            return

        doy, mod, abs_minute = self._get_time_index(state)
        if abs_minute is None or doy is None or mod is None:
            return

        zone_temps = self._read_zone_temps(state)
        outside_temp = self._read_outside_temp(state)
        self._bucket_oat_sum += outside_temp
        self._bucket_oat_count += 1
        self._bucket_mean_oat = self._bucket_oat_sum / max(self._bucket_oat_count, 1)

        # store last snapshot for finalize()
        self._last_zone_temps = zone_temps
        self._last_outside_temp = outside_temp
        self._last_doy = doy
        self._last_mod = mod

        # push history once per *minute* (avoid duplicates from sub-minute system timesteps)
        if self._last_hist_abs_minute != abs_minute:
            avg_zone_temp = float(np.nanmean(list(zone_temps.values()))) if zone_temps else float("nan")
            self._push_history(abs_minute, avg_zone_temp, outside_temp)
            self._last_hist_abs_minute = abs_minute

        obs, occupied = self._make_obs_multi_zone(zone_temps, outside_temp, doy, mod, abs_minute)
        bucket_id = self._bucket_from_abs_minute(abs_minute)

        # first ever bucket -> pick initial action
        if self._bucket_id is None:
            self._bucket_id = bucket_id
            self._action_bucket_id = bucket_id

            self._prev_obs = obs
            self._prev_day_of_year = doy
            self._prev_minute_of_day = mod

            a_norm = self._select_action_vec(obs)
            self._held_act = a_norm.astype(np.float32)
            self._prev_act = self._held_act.copy()
            self._apply_action_vec(state, self._held_act, occupied)
            return

        # same action bucket -> just re-assert held setpoints
        if bucket_id == self._action_bucket_id:
            for z in self.ZONES:
                if self._held_heat_sp[z] is not None:
                    self.api.exchange.set_actuator_value(state, self.heat_sp_handles[z], float(self._held_heat_sp[z]))
                if self._held_cool_sp[z] is not None:
                    self.api.exchange.set_actuator_value(state, self.cool_sp_handles[z], float(self._held_cool_sp[z]))
            return

        # ---- new action bucket reached in BEGIN callback ----
        # Stage the boundary, but DO NOT finalize/reset reward bucket here.
        prev_act_snapshot = np.asarray(self._held_act, dtype=np.float32).copy()
        prev_heat_sp_snapshot = dict(self._held_heat_sp)
        prev_cool_sp_snapshot = dict(self._held_cool_sp)

        a_norm = self._select_action_vec(obs)
        new_act = a_norm.astype(np.float32)

        # apply the new action immediately for the new bucket
        self._held_act = new_act.copy()
        self._apply_action_vec(state, self._held_act, occupied)
        self._action_bucket_id = bucket_id

        self._pending_boundary = {
            "bucket_id": int(bucket_id),
            "obs": np.asarray(obs, dtype=np.float32).copy(),
            "doy": int(doy),
            "mod": int(mod),
            "zone_temps": {k: float(v) for k, v in zone_temps.items()},
            "outside_temp": float(outside_temp),
            "occupied": bool(occupied),
            "new_act": new_act.copy(),
            "prev_act_snapshot": prev_act_snapshot,
            "prev_heat_sp_snapshot": prev_heat_sp_snapshot,
            "prev_cool_sp_snapshot": prev_cool_sp_snapshot,
        }
        return

    def end_system_timestep_callback(self, state):
        if not self._ensure_ready(state):
            return
        if self._bucket_id is None:
            return  # control loop hasn’t initialized yet

        self.sys_step_count += 1

        doy, mod, abs_minute = self._get_time_index(state)
        zone_temps = self._read_zone_temps(state)
        outside_temp = self._read_outside_temp(state)

        raw_val = self.api.exchange.get_meter_value(state, self.facility_elec_meter_handle)
        if raw_val == 0.0 and self.api.exchange.api_error_flag(state):
            print("[meter] api_error_flag=True reading meter; handle likely invalid")
            return

        # EnergyPlus meter is Joules over the *system timestep*
        energy_kwh = float(raw_val) / 3_600_000.0
        self._bucket_energy_kwh_sum += float(energy_kwh)

        occupied = self._is_occupied(mod)
        uncomfortable_now = False

        if occupied and (self.comfort_weight > 0.0 or self.hard_weight > 0.0):
            for _, t in zone_temps.items():
                if np.isnan(t):
                    continue

                v_comf = self._temp_violation(t, self.comfort_low, self.comfort_high)
                if v_comf > 0.0:
                    uncomfortable_now = True


        # 3) only increment minutes when minute-of-day advances
        # add the *previous* minute's uncomfortable state into the bucket
        if self._bucket_last_mod is None:
            self._bucket_last_mod = int(mod)
            self._last_uncomfortable = bool(uncomfortable_now)
            self._last_occupied = bool(occupied)
        else:
            mod_i = int(mod)
            prev_mod = int(self._bucket_last_mod)

            if mod_i != prev_mod:
                # handle minute wrap-around at midnight
                delta_min = mod_i - prev_mod
                if delta_min < 0:
                    delta_min += 1440

                if self._last_uncomfortable and self._last_occupied:
                    self._bucket_uncomfort_min += float(delta_min)

                self._bucket_last_mod = mod_i
                self._last_uncomfortable = bool(uncomfortable_now)
                self._last_occupied = bool(occupied)

        # accumulate into current 15-min bucket
        self._bucket_sys_steps += 1

        if self.debug_meter_every_n_steps and (self.sys_step_count % self.debug_meter_every_n_steps == 0):
            avg_tz = float(np.nanmean(list(zone_temps.values()))) if zone_temps else float("nan")
            print(
                f"[sys {self.sys_step_count}] dayofyr={doy} min={mod} raw={raw_val:.2f} "
                f"kWh={energy_kwh:.4f} avg_Tz={avg_tz:.2f}C "
                f"occ={occupied} outside_temp={outside_temp:.2f} bucket_energy_kwh_sum={self._bucket_energy_kwh_sum:.4f} "
                f"bucket={self._bucket_id} bucket_rew_sum={self._bucket_rew:.2f}"
            )

        end_bucket_id = self._bucket_from_abs_minute(abs_minute)

        if end_bucket_id != self._bucket_id:
            pb = self._pending_boundary

            if pb is None or int(pb["bucket_id"]) != int(end_bucket_id):
                print(
                    f"[bucket-warn] end bucket advanced to {end_bucket_id} but pending boundary "
                    f"was missing or mismatched; using current snapshot fallback"
                )

                pb = {
                    "bucket_id": int(end_bucket_id),
                    "obs": self._prev_obs if self._prev_obs is not None else np.zeros((self.obs_dim,), dtype=np.float32),
                    "doy": int(doy),
                    "mod": int(mod),
                    "zone_temps": {k: float(v) for k, v in zone_temps.items()},
                    "outside_temp": float(outside_temp),
                    "occupied": bool(occupied),
                    "new_act": np.asarray(self._held_act, dtype=np.float32).copy(),
                    "prev_act_snapshot": np.asarray(self._prev_act, dtype=np.float32).copy()
                        if self._prev_act is not None else np.asarray(self._held_act, dtype=np.float32).copy(),
                    "prev_heat_sp_snapshot": dict(self._held_heat_sp),
                    "prev_cool_sp_snapshot": dict(self._held_cool_sp),
                }

            # finalize old bucket AFTER its last end-of-system-timestep energy has been accumulated
            self._finalize_bucket(
                next_obs=pb["obs"],
                next_doy=pb["doy"],
                next_mod=pb["mod"],
                next_zone_temps=pb["zone_temps"],
                next_outside_temp=pb["outside_temp"],
                next_occupied=pb["occupied"],
                prev_act_snapshot=pb["prev_act_snapshot"],
                prev_heat_sp_snapshot=pb["prev_heat_sp_snapshot"],
                prev_cool_sp_snapshot=pb["prev_cool_sp_snapshot"],
            )

            # promote staged new bucket to current reward bucket
            self._bucket_id = int(end_bucket_id)
            self._prev_obs = np.asarray(pb["obs"], dtype=np.float32).copy()
            self._prev_act = np.asarray(pb["new_act"], dtype=np.float32).copy()
            self._prev_day_of_year = int(pb["doy"])
            self._prev_minute_of_day = int(pb["mod"])

            # reset accumulators for the new bucket
            self._bucket_rew = 0.0
            self._bucket_energy_kwh_sum = 0.0
            self._bucket_sys_steps = 0
            self._bucket_uncomfort_min = 0.0
            self._bucket_last_mod = int(mod)
            self._last_uncomfortable = bool(uncomfortable_now)
            self._last_occupied = bool(occupied)
            self._bucket_mean_oat = 0.0
            self._bucket_oat_count = 0
            self._bucket_oat_sum = 0.0

            self._pending_boundary = None

    def finalize_and_write_rollout(self):
        """
        Flush the final (possibly partial) bucket.

        We treat end-of-window as a truncation (done=0.0) by default so TD3 still bootstraps.
        If you ever want a true terminal episode, set ANDRUIX_TERMINAL_END=1.
        """
        terminal_end = (os.environ.get("ANDRUIX_TERMINAL_END", "0") == "1")

        # Only flush if we actually have an (obs, act) to close out
        if self._prev_obs is not None and self._prev_act is not None and self._bucket_id is not None:
            # If we never accumulated anything since the last boundary, don't emit a fake final step
            if int(self._bucket_sys_steps) > 0:
                energy_sum = float(self._bucket_energy_kwh_sum)
                uncomfort_min = float(self._bucket_uncomfort_min)

                energy_term_bucket = energy_sum / float(self.reward_scale)
                comfort_term_bucket = float(self.uncomfort_min_weight) * uncomfort_min
                final_rew = -energy_term_bucket - comfort_term_bucket

                # Build an end observation from the last snapshot (preferred)
                obs_end = self._prev_obs
                end_doy = self._prev_day_of_year if self._prev_day_of_year is not None else -1
                end_mod = self._prev_minute_of_day if self._prev_minute_of_day is not None else -1
                occ_end = self._is_occupied(end_mod if end_mod >= 0 else None)

                if (
                    self._last_zone_temps is not None
                    and self._last_outside_temp is not None
                    and self._last_doy is not None
                    and self._last_mod is not None
                ):
                    end_doy = int(self._last_doy)
                    end_mod = int(self._last_mod)
                    # abs_minute consistent with _get_time_index()
                    mod_clamped = max(0, min(1439, end_mod))
                    abs_minute = (max(1, end_doy) - 1) * 1440 + mod_clamped

                    obs_end, occ_end = self._make_obs_multi_zone(
                        self._last_zone_temps,
                        float(self._last_outside_temp),
                        end_doy,
                        mod_clamped,
                        abs_minute,
                    )

                done_final = 1.0 if terminal_end else 0.0

                print(
                    f"[bucket-final] doy={end_doy} min={end_mod} energy_kwh_sum={energy_sum:.3f} "
                    f"uncomfort_min={uncomfort_min:.1f} reward={final_rew:.3f} bucket_sys_steps={self._bucket_sys_steps} "
                    f"done={done_final:.1f}"
                )

                # Write final replay transition for the last held action
                self.rollout_writer.append(
                    obs=self._prev_obs,
                    act=np.asarray(self._prev_act, dtype=np.float32),
                    rew=float(final_rew),
                    next_obs=obs_end,
                    done=float(done_final),
                    day_of_year=self._prev_day_of_year,
                    minute_of_day=self._prev_minute_of_day,
                    energy_meter=float(energy_sum),
                    outside_air_c=float(self._bucket_mean_oat),
                )

                # Mirror the boundary behavior: this was a completed control step
                self.step_count += 1

                # Final timeseries row (use last snapshot if available)
                if self._last_zone_temps is not None and self._last_outside_temp is not None and end_doy != -1 and end_mod != -1:
                    self._write_bucket_timeseries(
                        int(end_doy),
                        int(end_mod),
                        float(self._last_outside_temp),
                        self._last_zone_temps,
                        float(final_rew),
                        float(energy_sum),
                        bool(occ_end),
                    )

        out_npz = self.rollout_writer.write()
        print(f"[rollout] wrote {out_npz}")

        ts_path = self.ts_writer.write()
        print(f"[timeseries] wrote {ts_path}")


def parse_args():
    p = argparse.ArgumentParser(description="EnergyPlus + PyEnergyPlus callback runner (Step 1.2: time + trend features)")
    p.add_argument("--idf", default=os.environ.get("EPLUS_IDF", "/home/guser/models/IECC_OfficeMedium_STD2021_Denver_RL_HIGH_ENERGY.idf"))
    p.add_argument("--epw", default=os.environ.get("EPLUS_EPW", "/home/guser/weather/5B_USA_CO_BOULDER.epw"))
    p.add_argument("--outdir", default=os.environ.get("EPLUS_OUTDIR", "/home/guser/results/"))
    p.add_argument("--zones", default=os.environ.get("EPLUS_ZONE", "PERIMETER_BOT_ZN_3"))
    p.add_argument("--energy-meter", default=os.environ.get("EPLUS_METER", "Electricity:Building"))
    p.add_argument("--dump-api-csv", action="store_true", default=os.environ.get("EPLUS_DUMP_API_CSV", "1") == "1")
    p.add_argument("--rollout-dir", default=os.environ.get("ROLLOUT_DIR", "/home/guser/rollouts"))
    p.add_argument("--rollout-id", default=os.environ.get("ROLLOUT_ID", str(uuid.uuid4())))

    p.add_argument("--start-date", default=os.environ.get("EPLUS_START_MMDD", ""))
    p.add_argument("--end-date", default=os.environ.get("EPLUS_END_MMDD", ""))

    p.add_argument("--policy-kind", default=os.environ.get("ANDRUIX_POLICY_KIND", "torch"), choices=["simple", "torch"])
    p.add_argument("--policy-path", default=os.environ.get("ANDRUIX_POLICY_PATH", os.environ.get("POLICY_PATH", "")))
    p.add_argument("--reward-scale", type=float, default=float(os.environ.get("ANDRUIX_REWARD_SCALE", "1")))

    # Observation feature toggles
    p.add_argument("--include-occ-flag", action="store_true", default=os.environ.get("ANDRUIX_OBS_OCC", "1") == "1")
    p.add_argument("--no-doy-features", action="store_true", default=os.environ.get("ANDRUIX_OBS_NO_DOY", "0") == "1")
    p.add_argument("--no-trend-60m", action="store_true", default=os.environ.get("ANDRUIX_OBS_NO_TREND_60M", "0") == "1")
    p.add_argument("--no-trend-15m", action="store_true", default=os.environ.get("ANDRUIX_OBS_NO_TREND_15M", "0") == "1")

    # Step 1 controls
    p.add_argument("--sp-center-min-occ", type=float, default=float(os.environ.get("ANDRUIX_SP_CENTER_MIN_OCC", "21.0")))
    p.add_argument("--sp-center-max-occ", type=float, default=float(os.environ.get("ANDRUIX_SP_CENTER_MAX_OCC", "24.0")))
    p.add_argument("--sp-center-min-unocc", type=float, default=float(os.environ.get("ANDRUIX_SP_CENTER_MIN_UNOCC", "18.0")))
    p.add_argument("--sp-center-max-unocc", type=float, default=float(os.environ.get("ANDRUIX_SP_CENTER_MAX_UNOCC", "28.0")))
    p.add_argument("--deadband-occ", type=float, default=float(os.environ.get("ANDRUIX_DEADBAND_OCC", "1.0")))
    p.add_argument("--deadband-unocc", type=float, default=float(os.environ.get("ANDRUIX_DEADBAND_UNOCC", "2.0")))
    p.add_argument("--occupied-start-minute", type=int, default=int(os.environ.get("ANDRUIX_OCC_START_MIN", str(8 * 60))))
    p.add_argument("--occupied-end-minute", type=int, default=int(os.environ.get("ANDRUIX_OCC_END_MIN", str(18 * 60))))

    p.add_argument("--comfort-low", type=float, default=float(os.environ.get("ANDRUIX_COMFORT_LOW", "21.0")))
    p.add_argument("--comfort-high", type=float, default=float(os.environ.get("ANDRUIX_COMFORT_HIGH", "24.0")))
    p.add_argument("--comfort-weight", type=float, default=float(os.environ.get("ANDRUIX_COMFORT_W", "0.1")))
    p.add_argument("--hard-low", type=float, default=float(os.environ.get("ANDRUIX_HARD_LOW", "15.0")))
    p.add_argument("--hard-high", type=float, default=float(os.environ.get("ANDRUIX_HARD_HIGH", "30.0")))
    p.add_argument("--hard-weight", type=float, default=float(os.environ.get("ANDRUIX_HARD_W", "5.0")))
    p.add_argument("--slew-weight", type=float, default=float(os.environ.get("ANDRUIX_SLEW_W", "0.0")))

    p.add_argument("--min-heat-sp", type=float, default=float(os.environ.get("ANDRUIX_MIN_HEAT_SP", "15.0")))
    p.add_argument("--max-cool-sp", type=float, default=float(os.environ.get("ANDRUIX_MAX_COOL_SP", "30.0")))
    p.add_argument("--min-deadband", type=float, default=float(os.environ.get("ANDRUIX_MIN_DEADBAND", "0.5")))
    p.add_argument("--control-minutes", type=int, default=int(os.environ.get("ANDRUIX_CONTROL_MINUTES", "15")),
)
    return p.parse_args()


def main():
    args = parse_args()

    # Optional: windowed run by rewriting RunPeriod
    idf_path = Path(args.idf)
    if args.start_date and args.end_date:
        outdir_for_idf = Path(args.outdir)
        outdir_for_idf.mkdir(parents=True, exist_ok=True)
        idf_path = outdir_for_idf / f"runperiod_{args.start_date.replace('/', '-')}_{args.end_date.replace('/', '-')}.idf"
        rewrite_first_runperiod(Path(args.idf), idf_path, args.start_date, args.end_date)

    idf = idf_path
    epw = Path(args.epw)
    outdir = Path(args.outdir)
    if not idf.exists():
        raise FileNotFoundError(f"IDF not found: {idf}")
    if not epw.exists():
        raise FileNotFoundError(f"EPW not found: {epw}")
    outdir.mkdir(parents=True, exist_ok=True)

    api = EnergyPlusAPI()
    state = api.state_manager.new_state()

    controller = RLController(
        api,
        state,
        zone_names=args.zones,
        outdir=outdir,
        energy_meter_name=args.energy_meter,
        rollout_dir=Path(args.rollout_dir),
        rollout_id=args.rollout_id,
        policy_path=args.policy_path,
        policy_kind=args.policy_kind,
        dump_api_available_csv=args.dump_api_csv,
        debug_meter_every_n_steps=20,
        reward_scale=args.reward_scale,

        include_occ_flag=bool(args.include_occ_flag),
        include_doy_features=(not bool(args.no_doy_features)),
        include_trend_60m=(not bool(args.no_trend_60m)),
        include_trend_15m=(not bool(args.no_trend_15m)),

        sp_center_min_occ=args.sp_center_min_occ,
        sp_center_max_occ=args.sp_center_max_occ,
        sp_center_min_unocc=args.sp_center_min_unocc,
        sp_center_max_unocc=args.sp_center_max_unocc,
        deadband_occ=args.deadband_occ,
        deadband_unocc=args.deadband_unocc,
        occupied_start_minute=args.occupied_start_minute,
        occupied_end_minute=args.occupied_end_minute,
        comfort_low=args.comfort_low,
        comfort_high=args.comfort_high,
        comfort_weight=args.comfort_weight,
        hard_low=args.hard_low,
        hard_high=args.hard_high,
        hard_weight=args.hard_weight,
        slew_weight=args.slew_weight,
        min_heat_sp=args.min_heat_sp,
        max_cool_sp=args.max_cool_sp,
        min_deadband=args.min_deadband,
        control_minutes=args.control_minutes,
    )

    eplus_args = ["-w", str(epw), "-d", str(outdir), str(idf)]
    api.runtime.run_energyplus(state, eplus_args)
    controller.finalize_and_write_rollout()


if __name__ == "__main__":
    main()