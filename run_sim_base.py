import argparse
import os
from pathlib import Path
from functools import partial

import numpy as np
from pyenergyplus.api import EnergyPlusAPI


# Simple RL model (placeholder for a neural network)
class SimpleRLModel:
    def __init__(self):
        # Random initial weights for a linear policy: setpoint = w0 * room_temp + w1 * outside_temp + b
        self.weights = np.array([0.5, 0.2])  # [w0, w1]
        self.bias = 20.0  # b

    def get_action(self, state_vec: np.ndarray) -> float:
        # State: [room_temp, outside_temp]
        setpoint = float(np.dot(self.weights, state_vec) + self.bias)
        # Clip setpoint between 18°C and 26°C
        return max(min(setpoint, 26.0), 18.0)

    def update(self, trajectory):
        # Placeholder update: adjust weights based on total reward (simplified)
        total_reward = sum(trajectory["rewards"])
        if total_reward < -1000:  # Arbitrary threshold for negative reward (high energy use)
            self.weights += np.random.normal(0, 0.01, size=self.weights.shape)
            self.bias += float(np.random.normal(0, 0.1))
        print(f"Updated weights: {self.weights}, bias: {self.bias}")


class RLController:
    def __init__(
        self,
        api: EnergyPlusAPI,
        state,
        zone_name: str,
        energy_meter_name: str = "Electricity:Facility",
    ):
        self.api = api
        self.state = state
        self.rl_model = SimpleRLModel()
        self.trajectory = {"states": [], "actions": [], "rewards": []}

        self.handles_ready = False
        self.room_temp_handle = -1
        self.outside_temp_handle = -1
        self.facility_elec_meter_handle = -1
        self.setpoint_handle = -1
        self.clg_handle = -1

        self.ZONE = zone_name
        self.energy_meter_name = energy_meter_name

        # Tell EnergyPlus which variables you want to track
        for var, key in [
            ("Zone Mean Air Temperature", self.ZONE),
            ("Site Outdoor Air Drybulb Temperature", "Environment"),
            # You had this, but note: depending on model/reporting, this key may not exist as written.
            ("Zone Air System Sensible Heating Energy", "Environment"),
        ]:
            self.api.exchange.request_variable(state, var, key)

        # Defer handle look-ups until the engine is ready
        self.api.runtime.callback_after_new_environment_warmup_complete(
            state,
            partial(RLController.init_handles, self),
        )

        # Main timestep callbacks
        self.api.runtime.callback_begin_zone_timestep_before_init_heat_balance(
            state,
            self.begin_timestep_callback,
        )

        self.api.runtime.callback_end_zone_timestep_after_zone_reporting(
            state,
            self.end_timestep_callback,
        )

    def init_handles(self, state):

        if not self.api.exchange.api_data_fully_ready(state):
            return
        
        csv = self.api.exchange.list_available_api_data_csv(state)
        (Path(self.outdir) / "api_available.csv").write_text(csv)


        # If already ready, do nothing
        if self.handles_ready:
            return

        self.room_temp_handle = self.api.exchange.get_variable_handle(
            state, "Zone Mean Air Temperature", self.ZONE
        )
        self.outside_temp_handle = self.api.exchange.get_variable_handle(
            state, "Site Outdoor Air Drybulb Temperature", "Environment"
        )

        # You set energy_handle twice in your original code; keeping meter handle as the one used.
        self.facility_elec_meter_handle = self.api.exchange.get_meter_handle(
            state, self.energy_meter_name  # should be "Electricity:Facility"
        )

        self.setpoint_handle = self.api.exchange.get_actuator_handle(
            state, "Zone Temperature Control", "Heating Setpoint", self.ZONE
        )
        self.clg_handle = self.api.exchange.get_actuator_handle(
            state, "Zone Temperature Control", "Cooling Setpoint", self.ZONE
        )

        # Sanity check
        missing = []
        for name, h in {
            "room temp": self.room_temp_handle,
            "outside": self.outside_temp_handle,
            "facility_elec_meter_handle": self.facility_elec_meter_handle,
            "heat SP": self.setpoint_handle,
            "cool SP": self.clg_handle,
        }.items():
            if h == -1:
                missing.append(name)

        if missing:
            print(f"[init_handles] Not ready / missing handles: {missing}. Will retry.")
            return
        
        self.handles_ready = True
        self.last_meter_val = None

        print("All EMS handles resolved ✔")

    def begin_timestep_callback(self, state):

        if not self.api.exchange.api_data_fully_ready(state):
            return
        
        if not self.handles_ready:
            self.init_handles(state)
            return

        room_temp = self.api.exchange.get_variable_value(state, self.room_temp_handle)
        outside_temp = self.api.exchange.get_variable_value(state, self.outside_temp_handle)
        state_vec = np.array([room_temp, outside_temp], dtype=np.float32)

        setpoint = self.rl_model.get_action(state_vec)

        # Apply to both heating and cooling setpoints (simple for now)
        self.api.exchange.set_actuator_value(state, self.setpoint_handle, setpoint)
        self.api.exchange.set_actuator_value(state, self.clg_handle, setpoint)

        cur_time = self.api.exchange.current_sim_time(state)  # hours since start
        print(
            f"[{cur_time:8.2f} h] RL setpoint ➜ {setpoint:5.1f} °C "
            f"(room {room_temp:5.1f} °C | outdoor {outside_temp:5.1f} °C)"
        )

        self.trajectory["states"].append(state_vec)
        self.trajectory["actions"].append(setpoint)

    def end_timestep_callback(self, state):

        if not self.api.exchange.api_data_fully_ready(state):
            return
        
        if not self.handles_ready:
            self.init_handles(state)
            return

        val = self.api.exchange.get_meter_value(state, self.facility_elec_meter_handle)
        if self.last_meter_val is None:
            step_j = 0.0
        else:
            step_j = max(0.0, val - self.last_meter_val)
        self.last_meter_val = val
        reward = -step_j

        self.trajectory["rewards"].append(reward)

        cur_time = self.api.exchange.current_sim_time(state)
        print(f"[{cur_time:8.2f} h] Energy penalty {step_j} J")


def parse_args():
    p = argparse.ArgumentParser(description="EnergyPlus + PyEnergyPlus callback hello-world runner")
    p.add_argument(
        "--idf",
        default=os.environ.get(
            "EPLUS_IDF", "/home/guser/models/IECC_OfficeMedium_STD2021_Denver_RL_BASELINE_1_0.idf"
        ),
        help="Path to the IDF model",
    )
    p.add_argument(
        "--epw",
        default=os.environ.get("EPLUS_EPW", "/home/guser/weather/5B_USA_CO_BOULDER.epw"),
        help="Path to the EPW weather file",
    )
    p.add_argument(
        "--outdir",
        default=os.environ.get("EPLUS_OUTDIR", "/home/guser/results/"),
        help="Output directory for EnergyPlus results",
    )
    p.add_argument(
        "--zone",
        default=os.environ.get("EPLUS_ZONE", "PERIMETER_BOT_ZN_3"),
        help="Zone name used for variable/actuator handles",
    )
    p.add_argument(
        "--energy-meter",
        default=os.environ.get("EPLUS_METER", "Electricity:Facility,hourly"),
        help="EnergyPlus meter name for reward/penalty",
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
    controller = RLController(api, state, zone_name=args.zone, energy_meter_name=args.energy_meter)

    # Run simulation for one episode
    eplus_args = [
        "-w",
        str(epw),
        "-d",
        str(outdir),
        str(idf),
    ]

    print("Running EnergyPlus with args:", " ".join(eplus_args))
    api.runtime.run_energyplus(state, eplus_args)

    print(f"Episode reward: {sum(controller.trajectory['rewards'])}")
    controller.rl_model.update(controller.trajectory)


if __name__ == "__main__":
    main()
