from pyenergyplus.api import EnergyPlusAPI
import numpy as np
from functools import partial

# Simple RL model (placeholder for a neural network)
class SimpleRLModel:
    def __init__(self):
        # Random initial weights for a linear policy: setpoint = w0 * room_temp + w1 * outside_temp + b
        self.weights = np.array([0.5, 0.2])  # [w0, w1]
        self.bias = 20.0  # b

    def get_action(self, state):
        # State: [room_temp, outside_temp]
        setpoint = np.dot(self.weights, state) + self.bias
        return max(min(setpoint, 26.0), 18.0)  # Clip setpoint between 18°C and 26°C

    def update(self, trajectory):
        # Placeholder update: adjust weights based on total reward (simplified)
        total_reward = sum(trajectory['rewards'])
        if total_reward < -1000:  # Arbitrary threshold for negative reward (high energy use)
            self.weights += np.random.normal(0, 0.01, size=self.weights.shape)
            self.bias += np.random.normal(0, 0.1)
        print(f"Updated weights: {self.weights}, bias: {self.bias}")

class RLController:
    def __init__(self, api, state):
        self.api = api
        self.state = state
        self.rl_model = SimpleRLModel()
        self.trajectory = {'states': [], 'actions': [], 'rewards': []}

        self.ZONE = "PERIMETER_BOT_ZN_3"

        # 1️⃣ Tell EnergyPlus which variables you want to track
        for var, key in [
            ("Zone Mean Air Temperature", self.ZONE),
            ("Site Outdoor Air Drybulb Temperature", "Environment"),
            ("Zone Air System Sensible Heating Energy", "Environment")
        ]:
            api.exchange.request_variable(state, var, key)
            
        

       # 2 Defer handle look-ups until the engine is ready
        api.runtime.callback_after_new_environment_warmup_complete(
	    state,	
            partial(RLController.init_handles, self)
        )

        # Main timestep callbacks (unchanged)
        api.runtime.callback_begin_zone_timestep_before_init_heat_balance(
	    state,
            self.begin_timestep_callback
        )
        api.runtime.callback_end_zone_timestep_after_zone_reporting(
            state,
            self.end_timestep_callback
        )

    def init_handles(self, state):
        self.room_temp_handle = self.api.exchange.get_variable_handle(
            state, "Zone Mean Air Temperature", self.ZONE)
        self.outside_temp_handle = self.api.exchange.get_variable_handle(
            state, "Site Outdoor Air Drybulb Temperature", "Environment")
        self.energy_handle = self.api.exchange.get_variable_handle(
            state, "Zone Air System Sensible Heating Energy", "Environment")
        self.energy_handle = self.api.exchange.get_meter_handle(state, "General:Cooling:Electricity")

        self.setpoint_handle = self.api.exchange.get_actuator_handle(
            state, "Zone Temperature Control", "Heating Setpoint", self.ZONE)
        self.clg_handle = self.api.exchange.get_actuator_handle(
            state, "Zone Temperature Control", "Cooling Setpoint", self.ZONE)



        # Sanity check
        for name, h in {
            "room temp": self.room_temp_handle,
            "outside":   self.outside_temp_handle,
            "energy":    self.energy_handle,
            "heat SP":   self.setpoint_handle,
            "cool SP":   self.clg_handle,
            "energy meter":   self.energy_handle,		
        }.items():
            if h == -1:
                raise RuntimeError(f"{name} handle not found – "
                                   "check spelling and case in RDD/EDD")

        print("All EMS handles resolved ✔")  # visible confirmation

    def begin_timestep_callback(self, state):
        room_temp    = self.api.exchange.get_variable_value(state, self.room_temp_handle)
        outside_temp = self.api.exchange.get_variable_value(state, self.outside_temp_handle)
        state_vec    = np.array([room_temp, outside_temp])

        setpoint = self.rl_model.get_action(state_vec)
        self.api.exchange.set_actuator_value(state, self.setpoint_handle, setpoint)
        self.api.exchange.set_actuator_value(state, self.clg_handle, setpoint)

        #energy = self.api.exchange.get_meter_value(state, self.energy_handle)
        #print(f"Raw energy (J): {energy}")

        # NEW: echo what the RL agent just wrote
        cur_time = self.api.exchange.current_sim_time(state)  # hours since start
        print(f"[{cur_time:8.2f} h]  RL setpoint ➜ {setpoint:5.1f} °C  "
            f"(room {room_temp:5.1f} °C | outdoor {outside_temp:5.1f} °C)")

        self.trajectory['states'].append(state_vec)
        self.trajectory['actions'].append(setpoint)


    def end_timestep_callback(self, state):
        energy = self.api.exchange.get_meter_value(state, self.energy_handle)
        reward = -energy
        self.trajectory['rewards'].append(reward)

        # NEW: echo the energy penalty for context
        cur_time = self.api.exchange.current_sim_time(state)
        print(f"[{cur_time:8.2f} h]  Energy penalty {energy} J")


def main():
    api = EnergyPlusAPI()
    state = api.state_manager.new_state()
    controller = RLController(api, state)
    
    # Run simulation for one episode (replace paths with your files)
    api.runtime.run_energyplus(state, [
        '-w', '/home/guser/weather/5B_USA_CO_BOULDER.epw',
        '-d', '/home/guser/results/',
        '/home/guser/models/OfficeMedium_STD2021_Denver_RL_BASELINE_1_0.idf'
    ])
    
    # Update model after simulation
    print(f"Episode reward: {sum(controller.trajectory['rewards'])}")
    controller.rl_model.update(controller.trajectory)

    # For POC, you can manually repeat or add a loop for more episodes

if __name__ == "__main__":
    main()
