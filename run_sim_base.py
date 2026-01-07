import argparse
import os
from pathlib import Path
import time

import numpy as np
from pyenergyplus.api import EnergyPlusAPI


# Simple RL model (placeholder for a neural network)
class SimpleRLModel:
    def __init__(self):
        self.weights = np.array([0.5, 0.2], dtype=np.float32)
        self.bias = 20.0

    def get_action(self, state_vec: np.ndarray) -> float:
        setpoint = float(np.dot(self.weights, state_vec) + self.bias)
        return max(min(setpoint, 26.0), 18.0)

    def update(self, trajectory):
        total_reward = sum(trajectory["rewards"])
        if total_reward < -1000:
            self.weights += np.random.normal(0, 0.01, size=self.weights.shape).astype(np.float32)
            self.bias += float(np.random.normal(0, 0.1))
        print(f"Updated weights: {self.weights}, bias: {self.bias}")


class RLController:
    def __init__(
        self,
        api: EnergyPlusAPI,
        state,
        zone_name: str,
        outdir: Path,
        energy_meter_name: str = "Electricity:Building",
        dump_api_available_csv: bool = True,
        debug_meter_every_n_steps: int = 20,
    ):
        self.api = api
        self.state = state
        self.ZONE = zone_name
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        # IMPORTANT: For get_meter_handle(), pass just the meter name (no ",hourly").
        self.energy_meter_name = energy_meter_name.split(",")[0].strip()

        self.dump_api_available_csv = dump_api_available_csv
        self.debug_meter_every_n_steps = debug_meter_every_n_steps

        self.rl_model = SimpleRLModel()
        self.trajectory = {"states": [], "actions": [], "rewards": []}

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

    # ---- callbacks ----
    def begin_timestep_callback(self, state):
        if not self._ensure_ready(state):
            return

        room_temp = self.api.exchange.get_variable_value(state, self.room_temp_handle)
        outside_temp = self.api.exchange.get_variable_value(state, self.outside_temp_handle)
        state_vec = np.array([room_temp, outside_temp], dtype=np.float32)

        setpoint = self.rl_model.get_action(state_vec)

        # Apply to both heating and cooling setpoints
        self.api.exchange.set_actuator_value(state, self.heat_sp_handle, setpoint)
        self.api.exchange.set_actuator_value(state, self.cool_sp_handle, setpoint)

        cur_time = self.api.exchange.current_sim_time(state)
        print(
            f"[{cur_time:8.2f} h] RL setpoint ➜ {setpoint:5.1f} °C "
            f"(room {room_temp:5.1f} °C | outdoor {outside_temp:5.1f} °C)"
        )

        self.trajectory["states"].append(state_vec)
        self.trajectory["actions"].append(setpoint)

    def end_system_timestep_callback(self, state):
        if not self._ensure_ready(state):
            return

        raw_val = self.api.exchange.get_meter_value(state, self.facility_elec_meter_handle)

        # Disambiguate a 0.0 return (could be valid or could be invalid handle)
        if raw_val == 0.0 and self.api.exchange.api_error_flag(state):
            print("[meter] api_error_flag=True reading meter; handle likely invalid")
            return

        # API note: this is currently instantaneous, not cumulative
        # (so delta logic isn't needed; treat raw_val as your step energy)
        reward = -raw_val
        self.trajectory["rewards"].append(reward)

        cur_time = self.api.exchange.current_sim_time(state)
        print(f"[{cur_time:8.2f} h] Meter raw={raw_val} J  reward={reward}")



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
    return p.parse_args()


def main():
    args = parse_args()

    idf = Path(args.idf)
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
        dump_api_available_csv=args.dump_api_csv,
        debug_meter_every_n_steps=20,
    )

    eplus_args = ["-w", str(epw), "-d", str(outdir), str(idf)]
    print("Running EnergyPlus with args:", " ".join(eplus_args))
    api.runtime.run_energyplus(state, eplus_args)

    episode_reward = float(sum(controller.trajectory["rewards"]))
    print(f"Episode reward: {episode_reward}")
    controller.rl_model.update(controller.trajectory)


if __name__ == "__main__":
    main()
