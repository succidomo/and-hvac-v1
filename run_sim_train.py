import argparse
import os
from pathlib import Path

import numpy as np
import uuid
from collections import deque

from pyenergyplus.api import EnergyPlusAPI

from eplus_runperiod_utils import rewrite_first_runperiod
from andruix_policy_model import TorchPolicyModel
from andruix_rollout_writer import RolloutWriter
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
        zone_name: str,
        outdir: Path,
        rollout_dir: Path,
        rollout_id: str,
        policy_path: str | os.PathLike | None = None,
        policy_kind: str = "torch",
        energy_meter_name: str = "Electricity:Building",
        dump_api_available_csv: bool = True,
        debug_meter_every_n_steps: int = 20,
        reward_mode: str = "delta",
        reward_scale: float = 1e6,

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
        comfort_high: float = 24.0,
        comfort_weight: float = 0.1,
        hard_low: float = 15.0,
        hard_high: float = 30.0,
        hard_weight: float = 5.0,
        slew_weight: float = 0.0,

        # --- Setpoint safety clamps ---
        min_heat_sp: float = 15.0,
        max_cool_sp: float = 30.0,
        min_deadband: float = 0.5,
    ):
        self.api = api
        self.state = state
        self.ZONE = zone_name
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
        self._hist = deque()  # items: (abs_minute:int, Tzone:float, Toa:float)
        self._max_trend_window = 0
        if self.include_trend_60m:
            self._max_trend_window = max(self._max_trend_window, 60)
        if self.include_trend_15m:
            self._max_trend_window = max(self._max_trend_window, 15)

        # Compute obs_dim
        # base: Tzone, Toa, sin_tod, cos_tod => 4
        obs_dim = 4
        if self.include_doy_features:
            obs_dim += 2
        if self.include_occ_flag:
            obs_dim += 1
        if self.include_trend_60m:
            obs_dim += 2
        if self.include_trend_15m:
            obs_dim += 2
        self.obs_dim = obs_dim
        self.act_dim = 1

        # IMPORTANT: For get_meter_handle(), pass just the meter name (no ",hourly").
        self.energy_meter_name = energy_meter_name.split(",")[0].strip()
        self.dump_api_available_csv = dump_api_available_csv
        self.debug_meter_every_n_steps = debug_meter_every_n_steps
        self.reward_mode = (reward_mode or "delta").lower()
        self.reward_scale = float(reward_scale) if reward_scale else 1.0

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

        # Worker-side policy (loads once at init, no per-timestep RPC)
        if self._is_torch_policy:
            print("TorchPolicyModel..... loaded")
            self.rl_model = TorchPolicyModel(policy_path)
        else:
            print("SimpleRLModel..... loaded")
            self.rl_model = SimpleRLModel()

        # TD3-style replay writer
        self.rollout_writer = RolloutWriter(self.rollout_dir, self.rollout_id, obs_dim=self.obs_dim, act_dim=self.act_dim)

        # Handle readiness / sentinels
        self.handles_ready = False
        self.api_dumped = False
        self.room_temp_handle = -1
        self.outside_temp_handle = -1
        self.facility_elec_meter_handle = -1
        self.heat_sp_handle = -1
        self.cool_sp_handle = -1

        self.last_meter_val = None
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
        self._last_heat_sp = None
        self._last_cool_sp = None
        self._last_center_sp = None

        # Request variables you plan to read
        for var, key in [
            ("Zone Mean Air Temperature", self.ZONE),
            ("Site Outdoor Air Drybulb Temperature", "Environment"),
        ]:
            self.api.exchange.request_variable(state, var, key)

        # Callbacks
        self.api.runtime.callback_after_new_environment_warmup_complete(state, self._try_init_handles)
        self.api.runtime.callback_begin_zone_timestep_before_init_heat_balance(state, self.begin_timestep_callback)
        self.api.runtime.callback_end_system_timestep_after_hvac_reporting(state, self.end_system_timestep_callback)

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
                (self.outdir / "api_available.csv").write_bytes(csv_bytes)
                self.api_dumped = True
                print(f"[init] Wrote {self.outdir / 'api_available.csv'}")
            except Exception as e:
                print(f"[init] WARNING: failed to write api_available.csv: {e}")
                self.api_dumped = True

        self.room_temp_handle = self.api.exchange.get_variable_handle(state, "Zone Mean Air Temperature", self.ZONE)
        self.outside_temp_handle = self.api.exchange.get_variable_handle(state, "Site Outdoor Air Drybulb Temperature", "Environment")
        self.facility_elec_meter_handle = self.api.exchange.get_meter_handle(state, self.energy_meter_name)
        self.heat_sp_handle = self.api.exchange.get_actuator_handle(state, "Zone Temperature Control", "Heating Setpoint", self.ZONE)
        self.cool_sp_handle = self.api.exchange.get_actuator_handle(state, "Zone Temperature Control", "Cooling Setpoint", self.ZONE)

        missing = [name for name, h in {
            "room_temp_handle": self.room_temp_handle,
            "outside_temp_handle": self.outside_temp_handle,
            f"meter({self.energy_meter_name})": self.facility_elec_meter_handle,
            "heat_sp_handle": self.heat_sp_handle,
            "cool_sp_handle": self.cool_sp_handle,
        }.items() if h == -1]

        self.init_attempts += 1
        if missing:
            if self.init_attempts % 50 == 0:
                print(f"[init] Missing handles (will retry): {missing}")
            return

        self.handles_ready = True
        self.last_meter_val = None
        self._hist.clear()
        print(f"[init] Handles resolved ✔ meter='{self.energy_meter_name}' obs_dim={self.obs_dim}")

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
    def _push_history(self, abs_minute: int | None, room_temp: float, outside_temp: float):
        if abs_minute is None:
            return
        self._hist.append((int(abs_minute), float(room_temp), float(outside_temp)))
        if self._max_trend_window <= 0:
            return
        cutoff = int(abs_minute) - int(self._max_trend_window) - 1
        while self._hist and self._hist[0][0] < cutoff:
            self._hist.popleft()

    def _lookup_at_or_before(self, target_abs_minute: int) -> tuple[float, float] | None:
        # Scan from newest backwards to find the closest value at or before target time.
        for t, tz, toa in reversed(self._hist):
            if t <= target_abs_minute:
                return float(tz), float(toa)
        return None

    def _trend_deltas(self, abs_minute: int | None, room_temp: float, outside_temp: float) -> list[float]:
        """Return trend deltas [dTzone_60, dToa_60, dTzone_15, dToa_15] depending on toggles."""
        feats: list[float] = []
        if abs_minute is None or not self._hist:
            # Not enough info yet
            if self.include_trend_60m:
                feats += [0.0, 0.0]
            if self.include_trend_15m:
                feats += [0.0, 0.0]
            return feats

        cur_tz = float(room_temp)
        cur_toa = float(outside_temp)

        if self.include_trend_60m:
            past = self._lookup_at_or_before(int(abs_minute) - 60)
            if past is None:
                feats += [0.0, 0.0]
            else:
                p_tz, p_toa = past
                feats += [cur_tz - p_tz, cur_toa - p_toa]

        if self.include_trend_15m:
            past = self._lookup_at_or_before(int(abs_minute) - 15)
            if past is None:
                feats += [0.0, 0.0]
            else:
                p_tz, p_toa = past
                feats += [cur_tz - p_tz, cur_toa - p_toa]

        return feats

    def _make_obs(self, room_temp: float, outside_temp: float, doy: int | None, mod: int | None, abs_minute: int | None) -> tuple[np.ndarray, bool]:
        # time-of-day features
        if mod is None:
            mod = 0
        ang_tod = 2.0 * np.pi * (float(mod) / 1440.0)
        sin_tod = float(np.sin(ang_tod))
        cos_tod = float(np.cos(ang_tod))

        feats: list[float] = [float(room_temp), float(outside_temp), sin_tod, cos_tod]

        if self.include_doy_features:
            if doy is None:
                doy = 1
            ang_doy = 2.0 * np.pi * ((float(doy) - 1.0) / 365.0)
            feats += [float(np.sin(ang_doy)), float(np.cos(ang_doy))]

        occupied = self._is_occupied(mod)
        if self.include_occ_flag:
            feats.append(1.0 if occupied else 0.0)

        # trend features (from history)
        if self.include_trend_60m or self.include_trend_15m:
            feats += self._trend_deltas(abs_minute, room_temp, outside_temp)

        return np.asarray(feats, dtype=np.float32), occupied

    # ---- action mapping / penalties ----
    def _coerce_action(self, act) -> float:
        try:
            a = float(np.asarray(act, dtype=np.float32).reshape(-1)[0])
        except Exception:
            a = 0.0
        if not np.isfinite(a):
            a = 0.0
        return float(np.clip(a, -1.0, 1.0))

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

    # ---- callbacks ----
    def begin_timestep_callback(self, state):
        if not self._ensure_ready(state):
            return

        room_temp = self.api.exchange.get_variable_value(state, self.room_temp_handle)
        outside_temp = self.api.exchange.get_variable_value(state, self.outside_temp_handle)
        doy, mod, abs_minute = self._get_time_index(state)

        # update history buffer BEFORE building obs (so the 15/60-min lookups can find older data)
        self._push_history(abs_minute, room_temp, outside_temp)

        obs, occupied = self._make_obs(room_temp, outside_temp, doy, mod, abs_minute)

        # Finalize last transition
        if self._prev_obs is not None and self._prev_act is not None and self._pending_rew is not None:
            self.rollout_writer.append(
                obs=self._prev_obs,
                act=np.asarray([self._prev_act], dtype=np.float32),
                rew=float(self._pending_rew),
                next_obs=obs,
                done=0.0,
                day_of_year=self._prev_day_of_year,
                minute_of_day=self._prev_minute_of_day,
                energy_meter=self._pending_energy,
            )
            self._pending_rew = None
            self._pending_energy = None

        self._pending_energy = None
        self._prev_day_of_year = None
        self._prev_minute_of_day = None

        # Policy action (normalized)
        obs_for_policy = obs if self._is_torch_policy else np.asarray([room_temp, outside_temp], dtype=np.float32)
        
        a_norm_raw = self.rl_model.get_action(obs)

        explore_noise = 0.1  # Or make configurable via env var/self.cfg
        noise = np.random.normal(0, explore_noise)
        a_norm_noisy = a_norm_raw + noise

        a_norm = self._coerce_action(a_norm_noisy)

        heat_sp, cool_sp, center_sp = self._map_action_to_setpoints(a_norm, occupied)

        if self.step_count % 50 == 0:
            fixed = os.environ.get("ANDRUIX_FIXED_SP_C")
            print(f"[dbg] fixed={fixed} a_norm_raw={a_norm_raw} act={a_norm} obs0={obs[0]:.2f} obs1={obs[1]:.2f}")

        if self.dbg_count < 10:
            print(f"[dbg] a_norm_raw={a_norm_raw:.4f} a_norm={a_norm:.4f} center_sp={center_sp:.2f} heat={heat_sp:.2f} cool={cool_sp:.2f}")
            self.dbg_count += 1



        self.api.exchange.set_actuator_value(state, self.heat_sp_handle, heat_sp)
        self.api.exchange.set_actuator_value(state, self.cool_sp_handle, cool_sp)

        self._last_heat_sp = heat_sp
        self._last_cool_sp = cool_sp
        self._last_center_sp = center_sp

        self._prev_obs = obs
        self._prev_act = a_norm
        self._prev_day_of_year = doy
        self._prev_minute_of_day = mod

    def end_system_timestep_callback(self, state):
        if not self._ensure_ready(state):
            return

        raw_val = self.api.exchange.get_meter_value(state, self.facility_elec_meter_handle)
        if raw_val == 0.0 and self.api.exchange.api_error_flag(state):
            print("[meter] api_error_flag=True reading meter; handle likely invalid")
            return

        delta = 0.0
        if self.last_meter_val is not None:
            delta = float(raw_val) - float(self.last_meter_val)
            if delta < 0.0:
                delta = 0.0
        self.last_meter_val = float(raw_val)

        use_val = float(raw_val) if self.reward_mode == "raw" else float(delta)
        scale = self.reward_scale if self.reward_scale != 0.0 else 1.0
        energy_kwh = use_val / scale

        temp_now = self.api.exchange.get_variable_value(state, self.room_temp_handle)
        occupied = self._is_occupied(self._prev_minute_of_day)

        comfort_pen = 0.0
        if occupied and self.comfort_weight > 0.0:
            v = self._temp_violation(temp_now, self.comfort_low, self.comfort_high)
            comfort_pen = self.comfort_weight * (v * v)

        hard_pen = 0.0
        if self.hard_weight > 0.0:
            v = self._temp_violation(temp_now, self.hard_low, self.hard_high)
            hard_pen = self.hard_weight * (v * v)

        slew_pen = 0.0
        if self.slew_weight > 0.0 and self._last_center_sp is not None:
            if getattr(self, "_prev_center_sp", None) is not None:
                d = float(self._last_center_sp) - float(self._prev_center_sp)
                slew_pen = self.slew_weight * (d * d)
            self._prev_center_sp = float(self._last_center_sp)

        rew = -energy_kwh - comfort_pen - hard_pen - slew_pen

        self.step_count += 1
        if self.debug_meter_every_n_steps and (self.step_count % self.debug_meter_every_n_steps == 0):
            meter_str = f"raw={raw_val:.2f}" if self.reward_mode == "raw" else f"delta={delta:.2f}"
            print(
                f"[step {self.step_count}] {meter_str} kWh={energy_kwh:.4f} "
                f"Tz={temp_now:.2f}C occ={occupied} "
                f"Hsp={self._last_heat_sp:.1f} Csp={self._last_cool_sp:.1f} "
                f"pen(c={comfort_pen:.3f}, h={hard_pen:.3f}, s={slew_pen:.3f}) rew={rew:.4f}"
            )

        self._pending_rew = rew
        self._pending_energy = energy_kwh

    def finalize_and_write_rollout(self):
        if self._prev_obs is not None and self._prev_act is not None and self._pending_rew is not None:
            self.rollout_writer.append(
                obs=self._prev_obs,
                act=np.asarray([self._prev_act], dtype=np.float32),
                rew=float(self._pending_rew),
                next_obs=self._prev_obs,
                done=1.0,
                day_of_year=self._prev_day_of_year,
                minute_of_day=self._prev_minute_of_day,
                energy_meter=self._pending_energy,
            )
            self._pending_rew = None
            self._pending_energy = None

        out_npz = self.rollout_writer.write()
        print(f"[rollout] wrote {out_npz}")


def parse_args():
    p = argparse.ArgumentParser(description="EnergyPlus + PyEnergyPlus callback runner (Step 1.2: time + trend features)")
    p.add_argument("--idf", default=os.environ.get("EPLUS_IDF", "/home/guser/models/IECC_OfficeMedium_STD2021_Denver_RL_BASELINE_1_0.idf"))
    p.add_argument("--epw", default=os.environ.get("EPLUS_EPW", "/home/guser/weather/5B_USA_CO_BOULDER.epw"))
    p.add_argument("--outdir", default=os.environ.get("EPLUS_OUTDIR", "/home/guser/results/"))
    p.add_argument("--zone", default=os.environ.get("EPLUS_ZONE", "PERIMETER_BOT_ZN_3"))
    p.add_argument("--energy-meter", default=os.environ.get("EPLUS_METER", "Electricity:Building"))
    p.add_argument("--dump-api-csv", action="store_true", default=os.environ.get("EPLUS_DUMP_API_CSV", "1") == "1")
    p.add_argument("--rollout-dir", default=os.environ.get("ROLLOUT_DIR", "/home/guser/rollouts"))
    p.add_argument("--rollout-id", default=os.environ.get("ROLLOUT_ID", str(uuid.uuid4())))

    p.add_argument("--start-date", default=os.environ.get("EPLUS_START_DATE", ""))
    p.add_argument("--end-date", default=os.environ.get("EPLUS_END_DATE", ""))

    p.add_argument("--policy-kind", default=os.environ.get("ANDRUIX_POLICY_KIND", "torch"), choices=["simple", "torch"])
    p.add_argument("--policy-path", default=os.environ.get("ANDRUIX_POLICY_PATH", os.environ.get("POLICY_PATH", "")))

    p.add_argument("--reward-mode", default=os.environ.get("ANDRUIX_REWARD_MODE", "delta"), choices=["raw", "delta"])
    p.add_argument("--reward-scale", type=float, default=float(os.environ.get("ANDRUIX_REWARD_SCALE", "1000000.0")))

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
        zone_name=args.zone,
        outdir=outdir,
        energy_meter_name=args.energy_meter,
        rollout_dir=Path(args.rollout_dir),
        rollout_id=args.rollout_id,
        policy_path=args.policy_path,
        policy_kind=args.policy_kind,
        dump_api_available_csv=args.dump_api_csv,
        debug_meter_every_n_steps=1,
        reward_mode=args.reward_mode,
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
    )

    eplus_args = ["-w", str(epw), "-d", str(outdir), str(idf)]
    api.runtime.run_energyplus(state, eplus_args)
    controller.finalize_and_write_rollout()


if __name__ == "__main__":
    main()
