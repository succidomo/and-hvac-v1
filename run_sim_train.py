import argparse
import os
from pathlib import Path
import time

import numpy as np
import json
import uuid
from pyenergyplus.api import EnergyPlusAPI

import torch
from eplus_runperiod_utils import parse_mmdd, rewrite_first_runperiod
from andruix_policy_model import TorchPolicyModel
from andruix_rollout_writer import RolloutWriter
from andruix_simple_model import SimpleRLModel


class RLController:
    def __init__(
        self,
        api: EnergyPlusAPI,
        state,
        zone_name: str,
        outdir: Path,
        rollout_dir: Path,
        rollout_id: str,
        policy_path: str | os.PathLike | None = None,
        policy_kind: str = "simple",
        energy_meter_name: str = "Electricity:Building",
        dump_api_available_csv: bool = True,
        debug_meter_every_n_steps: int = 20,
        reward_mode: str = "delta",
        reward_scale: float = 1e6,

        # --- Action semantics (single action -> thermostat centerline) ---
        # We treat the policy output as a normalized action a \in [-1, 1].
        # We map it to a thermostat "centerline" temperature, then derive
        # heating/cooling setpoints via a deadband. This avoids the current
        # behavior of setting *both* heating and cooling setpoints to the same
        # absolute value.
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
        # Worker-side policy (loads once at init, no per-timestep RPC)
        # Default path matches the orchestrator contract: /shared/policy/latest/policy.pt
        if (policy_kind or "").lower() == "torch":
            self.rl_model = TorchPolicyModel(policy_path)
        else:
            self.rl_model = SimpleRLModel()
        
        self.trajectory = {"states": [], "actions": [], "rewards": []}
        # TD3-style replay writer (obs, act, rew, next_obs, done)
        self.rollout_writer = RolloutWriter(self.rollout_dir, self.rollout_id, obs_dim=2, act_dim=1)
        # Handle readiness / sentinels to avoid AttributeError spam
        self.handles_ready = False
        self.api_dumped = False

        self.room_temp_handle = -1
        self.outside_temp_handle = -1
        self.facility_elec_meter_handle = -1
        self.heat_sp_handle = -1
        self.cool_sp_handle = -1

        self.last_meter_val = None
        self.step_count = 0
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
        self._last_occupied = None


        # Metrics
        self._cum_occ_comfort_violation_degmin = 0.0
        self._cum_hard_violation_degmin = 0.0

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
        self.api.runtime.callback_after_new_environment_warmup_complete(
            state, self._try_init_handles
        )

        self.api.runtime.callback_begin_zone_timestep_before_init_heat_balance(
            state, self.begin_timestep_callback
        )

        self.api.runtime.callback_end_system_timestep_after_hvac_reporting(
            state, self.end_system_timestep_callback
        )


    # ---- init / readiness ----
    def _try_init_handles(self, state):
        """Safe init attempt: no raising inside ctypes callback. Retries until ready."""
        if self.handles_ready:
            return

        if not self.api.exchange.api_data_fully_ready(state):
            return
        
        if self.api.exchange.warmup_flag(state):
            return


        # Dump available API data once (helps confirm exact meter spelling)
        if self.dump_api_available_csv and not self.api_dumped:
            try:
                csv_bytes = self.api.exchange.list_available_api_data_csv(state)  # returns bytes
                (self.outdir / "api_available.csv").write_bytes(csv_bytes)
                self.api_dumped = True
                print(f"[init] Wrote {self.outdir / 'api_available.csv'}")

                # Quick sanity: print a few meter rows so we can see exact names/keys
                text = csv_bytes.decode("utf-8", errors="replace")
                meter_lines = [ln for ln in text.splitlines() if ln.startswith("Meter,")]
                print("[init] First meters exposed by API:")
                for ln in meter_lines[:25]:
                    print("   ", ln)

            except Exception as e:
                print(f"[init] WARNING: failed to write api_available.csv: {e}")
                self.api_dumped = True  # prevent endless retry spam if desired


        # Resolve handles
        self.room_temp_handle = self.api.exchange.get_variable_handle(
            state, "Zone Mean Air Temperature", self.ZONE
        )
        self.outside_temp_handle = self.api.exchange.get_variable_handle(
            state, "Site Outdoor Air Drybulb Temperature", "Environment"
        )

        self.facility_elec_meter_handle = self.api.exchange.get_meter_handle(
            state, self.energy_meter_name
        )

        self.heat_sp_handle = self.api.exchange.get_actuator_handle(
            state, "Zone Temperature Control", "Heating Setpoint", self.ZONE
        )
        self.cool_sp_handle = self.api.exchange.get_actuator_handle(
            state, "Zone Temperature Control", "Cooling Setpoint", self.ZONE
        )

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
        print(f"[init] Handles resolved ✔ meter='{self.energy_meter_name}'")

    def _ensure_ready(self, state) -> bool:
        if not self.api.exchange.api_data_fully_ready(state):
            return False
        if not self.handles_ready:
            self._try_init_handles(state)
            return False
        return True
    
    def _get_time_index(self, state):
        """Return (day_of_year, minute_of_day) from EnergyPlus simulation clock."""
        try:
            doy = int(self.api.exchange.day_of_year(state))
            hr = int(self.api.exchange.hour(state))
            mn = int(self.api.exchange.minutes(state))
            if hr >= 24:
                hr = 23
                mn = 59

            mod = max(0, min(1439, hr*60 + mn))

            return doy, mod
        except Exception:
            return None, None

    def _is_occupied(self, minute_of_day: int | None) -> bool:
        """Simple occupancy heuristic for reward shaping + bounds.

        We keep this intentionally simple for v1 (no schedule handle parsing):
        occupied if minute_of_day is within [occupied_start_minute, occupied_end_minute).
        """
        if minute_of_day is None:
            return True
        start = int(self.occupied_start_minute)
        end = int(self.occupied_end_minute)
        m = int(minute_of_day)
        if start == end:
            return True
        if start < end:
            return start <= m < end
        # Handle wrap-around (e.g. 22:00 -> 06:00)
        return m >= start or m < end

    def _coerce_action(self, act) -> float:
        """Coerce various action shapes to a scalar float and clip to [-1, 1]."""
        try:
            a = float(np.asarray(act, dtype=np.float32).reshape(-1)[0])
        except Exception:
            a = 0.0
        # Worker expects normalized action; enforce a hard clip for safety.
        if not np.isfinite(a):
            a = 0.0
        return float(np.clip(a, -1.0, 1.0))

    def _map_action_to_setpoints(self, a: float, occupied: bool) -> tuple[float, float, float]:
        """Map normalized action a\in[-1,1] -> (heat_sp, cool_sp, center_sp) in °C."""
        if occupied:
            cmin, cmax = self.sp_center_min_occ, self.sp_center_max_occ
            deadband = self.deadband_occ
        else:
            cmin, cmax = self.sp_center_min_unocc, self.sp_center_max_unocc
            deadband = self.deadband_unocc

        # Map [-1,1] -> [cmin,cmax]
        center = float(cmin + 0.5 * (a + 1.0) * (cmax - cmin))

        # Derive setpoints via deadband around the centerline
        heat = center - 0.5 * float(deadband)
        cool = center + 0.5 * float(deadband)

        # Safety clamps
        heat = max(heat, self.min_heat_sp)
        cool = min(cool, self.max_cool_sp)

        # Enforce minimum deadband
        if (cool - heat) < self.min_deadband:
            mid = 0.5 * (heat + cool)
            heat = mid - 0.5 * self.min_deadband
            cool = mid + 0.5 * self.min_deadband
            # Re-clamp while preserving the minimum deadband as best we can
            if heat < self.min_heat_sp:
                heat = self.min_heat_sp
                cool = heat + self.min_deadband
            if cool > self.max_cool_sp:
                cool = self.max_cool_sp
                heat = cool - self.min_deadband

        center = 0.5 * (heat + cool)
        return float(heat), float(cool), float(center)
    def _temp_violation(self, temp_c: float, low: float, high: float) -> float:
        """Return degrees-C violation outside [low, high] (0 if inside)."""
        below = max(0.0, float(low) - float(temp_c))
        above = max(0.0, float(temp_c) - float(high))
        return below + above

    # ---- callbacks ----
    def begin_timestep_callback(self, state):
        if not self._ensure_ready(state):
            return

        room_temp = self.api.exchange.get_variable_value(state, self.room_temp_handle)
        outside_temp = self.api.exchange.get_variable_value(state, self.outside_temp_handle)
        obs = np.array([room_temp, outside_temp], dtype=np.float32)

        doy, mod = self._get_time_index(state)

        occupied = self._is_occupied(mod)



        # If we have reward from the previous system timestep, we can now finalize last transition:
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

        # --- Policy action (normalized) ---
        act_raw = self.rl_model.get_action(obs)
        act = self._coerce_action(act_raw)

        # Map to thermostat setpoints
        heat_sp, cool_sp, center_sp = self._map_action_to_setpoints(act, occupied)

        # Apply to heating and cooling setpoints
        self.api.exchange.set_actuator_value(state, self.heat_sp_handle, heat_sp)
        self.api.exchange.set_actuator_value(state, self.cool_sp_handle, cool_sp)

        self._last_heat_sp = heat_sp
        self._last_cool_sp = cool_sp
        self._last_center_sp = center_sp
        self._last_occupied = occupied
        self._last_occupied = occupied

        self._prev_obs = obs
        self._prev_act = act
        self._prev_day_of_year = doy
        self._prev_minute_of_day = mod

    
    def end_system_timestep_callback(self, state):
        if not self._ensure_ready(state):
            return

        raw_val = self.api.exchange.get_meter_value(state, self.facility_elec_meter_handle)

        # Disambiguate a 0.0 return (could be valid or could be invalid handle)
        if raw_val == 0.0 and self.api.exchange.api_error_flag(state):
            print("[meter] api_error_flag=True reading meter; handle likely invalid")
            return

        # Meter interpretation:
        # - Many EnergyPlus meters are cumulative over the run. In that case, delta is what you want per step.
        # - If your chosen meter is already per-timestep, 'raw' may be fine.
        delta = None
        if self.last_meter_val is None:
            delta = 0.0
        else:
            delta = float(raw_val) - float(self.last_meter_val)
            # If meter ever resets/backtracks, clamp to 0 for safety
            if delta < 0.0:
                delta = 0.0
        self.last_meter_val = float(raw_val)

        use_val = float(raw_val) if self.reward_mode == "raw" else float(delta)
        scale = self.reward_scale if self.reward_scale != 0.0 else 1.0
        energy_kwh = use_val / scale

        # --- Comfort + constraint penalties ---
        # Measure zone temperature at the end of the system timestep.
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
            # Penalize large setpoint changes (reduces bang-bang behavior)
            if getattr(self, "_prev_center_sp", None) is not None:
                d = float(self._last_center_sp) - float(self._prev_center_sp)
                slew_pen = self.slew_weight * (d * d)
            self._prev_center_sp = float(self._last_center_sp)

        rew = -energy_kwh - comfort_pen - hard_pen - slew_pen

        # Optional debug print (helps validate penalties + action mapping)
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
        # If sim ended and we still have a pending reward, close out with a terminal transition
        if self._prev_obs is not None and self._prev_act is not None and self._pending_rew is not None:
            self.rollout_writer.append(
                obs=self._prev_obs,
                act=np.asarray([self._prev_act], dtype=np.float32),
                rew=float(self._pending_rew),
                next_obs=self._prev_obs,  # terminal
                done=1.0,
                day_of_year=self._prev_day_of_year,
                minute_of_day=self._prev_minute_of_day,
                energy_meter=self._pending_energy,
            )
            self._pending_rew = None
            self._pending_energy = None
        self._pending_energy = None
        self._prev_day_of_year = None
        self._prev_minute_of_day = None

        out_npz = self.rollout_writer.write()
        print(f"[rollout] wrote {out_npz}")

def parse_args():
    p = argparse.ArgumentParser(description="EnergyPlus + PyEnergyPlus callback runner")
    p.add_argument(
        "--idf",
        default=os.environ.get("EPLUS_IDF", "/home/guser/models/IECC_OfficeMedium_STD2021_Denver_RL_BASELINE_1_0.idf"),
    )
    p.add_argument(
        "--epw",
        default=os.environ.get("EPLUS_EPW", "/home/guser/weather/5B_USA_CO_BOULDER.epw"),
    )
    p.add_argument(
        "--outdir",
        default=os.environ.get("EPLUS_OUTDIR", "/home/guser/results/"),
    )
    p.add_argument(
        "--zone",
        default=os.environ.get("EPLUS_ZONE", "PERIMETER_BOT_ZN_3"),
    )
    p.add_argument(
        "--energy-meter",
        default=os.environ.get("EPLUS_METER", "Electricity:Building"),
        help="Meter name for get_meter_handle(). Do NOT include reporting frequency (e.g., no ',hourly').",
    )
    p.add_argument(
        "--dump-api-csv",
        action="store_true",
        default=os.environ.get("EPLUS_DUMP_API_CSV", "1") == "1",
    )
    p.add_argument(
        "--rollout-dir", 
        default=os.environ.get("ROLLOUT_DIR", "/home/guser/rollouts")
    )
    p.add_argument(
        "--rollout-id", 
        default=os.environ.get("ROLLOUT_ID", 
        str(uuid.uuid4()))
    )

    p.add_argument(
        "--start-date",
        default=os.environ.get("EPLUS_START_DATE", ""),
        help="Optional: start date for windowed run in MM/DD (e.g., 09/01).",
    )
    p.add_argument(
        "--end-date",
        default=os.environ.get("EPLUS_END_DATE", ""),
        help="Optional: end date for windowed run in MM/DD (e.g., 09/07).",
    )

    p.add_argument(
        "--policy-kind",
        default=os.environ.get("ANDRUIX_POLICY_KIND", "simple"),
        choices=["simple", "torch"],
        help="Which policy implementation to use on the worker. Use simple for smoke tests.",
    )
    p.add_argument(
        "--reward-mode",
        default=os.environ.get("ANDRUIX_REWARD_MODE", "delta"),
        choices=["raw", "delta"],
        help="Reward uses raw meter value ('raw') or per-step delta of the meter ('delta'). Delta is recommended if the meter is cumulative.",
    )
    p.add_argument(
        "--reward-scale",
        type=float,
        default=float(os.environ.get("ANDRUIX_REWARD_SCALE", "1000000.0")),
        help="Divide reward by this scale factor to keep magnitudes reasonable (e.g., 1e6 or 1e9).",
    )

    # --- Step 1: action semantics + comfort/constraint penalties ---
    p.add_argument(
        "--sp-center-min-occ",
        type=float,
        default=float(os.environ.get("ANDRUIX_SP_CENTER_MIN_OCC", "21.0")),
        help="Occupied min centerline setpoint (°C) for action mapping.",
    )
    p.add_argument(
        "--sp-center-max-occ",
        type=float,
        default=float(os.environ.get("ANDRUIX_SP_CENTER_MAX_OCC", "24.0")),
        help="Occupied max centerline setpoint (°C) for action mapping.",
    )
    p.add_argument(
        "--sp-center-min-unocc",
        type=float,
        default=float(os.environ.get("ANDRUIX_SP_CENTER_MIN_UNOCC", "18.0")),
        help="Unoccupied min centerline setpoint (°C) for action mapping.",
    )
    p.add_argument(
        "--sp-center-max-unocc",
        type=float,
        default=float(os.environ.get("ANDRUIX_SP_CENTER_MAX_UNOCC", "28.0")),
        help="Unoccupied max centerline setpoint (°C) for action mapping.",
    )
    p.add_argument(
        "--deadband-occ",
        type=float,
        default=float(os.environ.get("ANDRUIX_DEADBAND_OCC", "1.0")),
        help="Occupied thermostat deadband width (°C).",
    )
    p.add_argument(
        "--deadband-unocc",
        type=float,
        default=float(os.environ.get("ANDRUIX_DEADBAND_UNOCC", "2.0")),
        help="Unoccupied thermostat deadband width (°C).",
    )
    p.add_argument(
        "--occupied-start-minute",
        type=int,
        default=int(os.environ.get("ANDRUIX_OCC_START_MIN", str(8 * 60))),
        help="Minute-of-day when occupied period starts (default 480 = 08:00).",
    )
    p.add_argument(
        "--occupied-end-minute",
        type=int,
        default=int(os.environ.get("ANDRUIX_OCC_END_MIN", str(18 * 60))),
        help="Minute-of-day when occupied period ends (default 1080 = 18:00).",
    )

    p.add_argument(
        "--comfort-low",
        type=float,
        default=float(os.environ.get("ANDRUIX_COMFORT_LOW", "21.0")),
        help="Occupied comfort lower bound (°C).",
    )
    p.add_argument(
        "--comfort-high",
        type=float,
        default=float(os.environ.get("ANDRUIX_COMFORT_HIGH", "24.0")),
        help="Occupied comfort upper bound (°C).",
    )
    p.add_argument(
        "--comfort-weight",
        type=float,
        default=float(os.environ.get("ANDRUIX_COMFORT_W", "0.1")),
        help="Penalty weight for occupied comfort violations (kWh-equivalent per °C^2 per timestep).",
    )

    p.add_argument(
        "--hard-low",
        type=float,
        default=float(os.environ.get("ANDRUIX_HARD_LOW", "15.0")),
        help="Hard lower bound (°C) applied all the time.",
    )
    p.add_argument(
        "--hard-high",
        type=float,
        default=float(os.environ.get("ANDRUIX_HARD_HIGH", "30.0")),
        help="Hard upper bound (°C) applied all the time.",
    )
    p.add_argument(
        "--hard-weight",
        type=float,
        default=float(os.environ.get("ANDRUIX_HARD_W", "5.0")),
        help="Penalty weight for hard constraint violations (kWh-equivalent per °C^2 per timestep).",
    )
    p.add_argument(
        "--slew-weight",
        type=float,
        default=float(os.environ.get("ANDRUIX_SLEW_W", "0.0")),
        help="Penalty weight for setpoint changes (°C^2 per timestep).",
    )

    p.add_argument(
        "--min-heat-sp",
        type=float,
        default=float(os.environ.get("ANDRUIX_MIN_HEAT_SP", "15.0")),
        help="Minimum allowed heating setpoint (°C).",
    )
    p.add_argument(
        "--max-cool-sp",
        type=float,
        default=float(os.environ.get("ANDRUIX_MAX_COOL_SP", "30.0")),
        help="Maximum allowed cooling setpoint (°C).",
    )
    p.add_argument(
        "--min-deadband",
        type=float,
        default=float(os.environ.get("ANDRUIX_MIN_DEADBAND", "0.5")),
        help="Minimum enforced thermostat deadband (°C).",
    )

    p.add_argument(
        "--policy-path",
        default=os.environ.get("ANDRUIX_POLICY_PATH", os.environ.get("POLICY_PATH", "")),
        help="Path to policy.pt (TD3 snapshot). Defaults to ANDRUIX_POLICY_PATH (or POLICY_PATH).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Optional: run a shorter simulation by rewriting the IDF RunPeriod to a custom date window.
    # This is the cleanest way to make a Batch rollout worker run (say) 7 days and then exit.
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
        debug_meter_every_n_steps=20,
        reward_mode=args.reward_mode,
        reward_scale=args.reward_scale,

        # Step 1: action semantics + comfort/constraint penalties
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
