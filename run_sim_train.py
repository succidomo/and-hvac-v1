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

    # ---- callbacks ----
    def begin_timestep_callback(self, state):
        if not self._ensure_ready(state):
            return

        room_temp = self.api.exchange.get_variable_value(state, self.room_temp_handle)
        outside_temp = self.api.exchange.get_variable_value(state, self.outside_temp_handle)
        obs = np.array([room_temp, outside_temp], dtype=np.float32)

        doy, mod = self._get_time_index(state)



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

        act = self.rl_model.get_action(obs)

        # Apply to both heating and cooling setpoints
        self.api.exchange.set_actuator_value(state, self.heat_sp_handle, act)
        self.api.exchange.set_actuator_value(state, self.cool_sp_handle, act)

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
        rew = -energy_kwh

        # Optional debug print
        # self.step_count += 1
        # if self.debug_meter_every_n_steps and (self.step_count % self.debug_meter_every_n_steps == 0) and self.reward_mode == "raw":
        #     print(f"[meter] raw={raw_val:.3f} reward={rew:.6f} mode={self.reward_mode} scale={scale:g}")
        # elif self.debug_meter_every_n_steps and (self.step_count % self.debug_meter_every_n_steps == 0) and self.reward_mode == "delta":
        #     print(f"[meter] delta={delta:.3f} reward={rew:.6f} mode={self.reward_mode} scale={scale:g}")

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
    )

    eplus_args = ["-w", str(epw), "-d", str(outdir), str(idf)]
    api.runtime.run_energyplus(state, eplus_args)
    controller.finalize_and_write_rollout()


if __name__ == "__main__":
    main()
