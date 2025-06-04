import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gridworld import ComponentEnv
from gridworld import MultiAgentEnv
from gridworld.distribution_system import OpenDSSSolver
from gridworld.agents.vehicles import EVChargingEnv
from mpl_toolkits.mplot3d import Axes3D

import plotting_mine


# busses = ['633.1', '634.1', '634.2', '634.3', '645.3', '646.2', '675.2', '675.3', '680.1', '692.2', '611.3', '652.1']
busses = ['634a', '634b', '634c', '645', '675a', '675b', '675c', '670a', '670b', '670c', '684c']

agents = [
    {
        "name": "ev-charging-{}".format(i),
        "bus": busses[i],
        "cls": EVChargingEnv,
        "config": {
            "num_vehicles": 100,
            "minutes_per_step": 15,
            "max_charge_rate_kw": 7.,
            "peak_threshold": 700.,
            "vehicle_multiplier": 1.,
            "rescale_spaces": False,
            "reward_scale": 1e5,
        }
    } for i in range(len(busses))
]


# Bare minimum common config specifies start and stop times, and control
# timedelta.  
common_config = {
    "start_time": "08-12-2020 20:00:00",
    "end_time": "08-13-2020 08:00:00",
    "control_timedelta": pd.Timedelta(900, "s")
}

# PowerFlow configuration.  Note that file paths are relative to 
# "gridworld/distribution_system/data" by default.
pf_config = {
    "cls": OpenDSSSolver,
    "config": {
        "feeder_file": "ieee_13_dss/IEEE13Nodeckt.dss",
        "loadshape_file": "ieee_13_dss/annual_hourly_load_profile.csv",
        "system_load_rescale_factor": 0.9,
    }
}

# Configuration of the multi-agent environment.
env_config = {
    "common_config": common_config,
    "pf_config": pf_config,
    "agents": agents
}

# Create the environment and run an episode using random policy
env = MultiAgentEnv(**env_config)
env.reset()
done = {"__all__": False}
metas = []
states = []
rewards = []
charging_rates = {agent["name"]: [] for agent in agents}

while not done["__all__"]:
    action = {name: space.high for name, space in env.action_space.items()}
    #action = {name: space.sample() for name, space in env.action_space.items()}
    #action = {name: space.low for name, space in env.action_space.items()}
    #print("Action: ", action)
    obs, rew, done, truncated, info = env.step(action)
    states.append(obs)
    metas.append(info)
    rewards.append(rew)
    
    # Collect actions for all EVs
    for agent_name, agent_action in action.items():
        charging_rates[agent_name].append(agent_action)
    
    if "final_info" in info:
        print("Final Info: ", info["final_info"])
        # print("info", info)
        # print(obs)
        print("final_reward", info["final_info"]["episode"]["r"])
    
    # Plot the rewards for each agent
plt.figure(figsize=(12, 6))
for agent_name in rewards[0].keys():
    agent_rewards = [reward[agent_name] for reward in rewards]
    plt.plot(agent_rewards, label=agent_name)

plt.title("Rewards per Agent Over Time")
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.legend(loc="best")
plt.grid(True)
plt.show()

def load_profile_to_dataframe(pf_config, common_config):
    """
    Reads the load profile CSV and converts it into a DataFrame of loads and timestamps
    filtered by the time window specified in common_config.

    Args:
        pf_config (dict): PowerFlow configuration containing the loadshape file path.
        common_config (dict): Common configuration containing the start and end times.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered load profile with 'time' and 'load' columns.
    """
    # Load the load profile CSV (no headers)
    loadshape_file = pf_config["config"]["loadshape_file"]
    print("loadshape_file", loadshape_file)
    load_profile_path = f"gridworld/distribution_system/data/{loadshape_file}"
    print("load_profile_path", load_profile_path)
    load_profile = pd.read_csv(load_profile_path, header=None, names=["load"])

    # Generate a 'time' column assuming hourly data for the year
    start_of_year = pd.Timestamp("2020-01-01 00:00:00")  # Adjust year if needed
    load_profile["time"] = [
        start_of_year + pd.Timedelta(hours=i) for i in range(len(load_profile))
    ]

    # Filter the load profile to match the time window in common_config
    start_time = pd.to_datetime(common_config["start_time"])
    end_time = pd.to_datetime(common_config["end_time"])
    filtered_load_profile = load_profile[
        (load_profile["time"] >= start_time) & (load_profile["time"] <= end_time)
    ]

    return filtered_load_profile


def ev_load_to_dataframe(metas, env):    
    """
    Converts the 'real_power_consumed' from metas and 'base_load' from env.history into a DataFrame of total EV load,
    total base load, and timesteps.

    Args:
        metas (list): A list of metadata dictionaries containing 'real_power_consumed' for each timestep.
        env: The environment object containing the 'base_load' history.

    Returns:
        pd.DataFrame: A DataFrame containing 'time', 'total_ev_load', and 'total_base_load' columns.
    """
    # Extract real power consumed and calculate total EV load for each timestep
    total_ev_load = [
        sum(agent_data.get("real_power_consumed", 0) for agent_data in info.values())
        for info in metas
    ]

    # Extract base load from env.history["base_load"] and calculate total base load for each timestep
    total_base_load = [
        np.sum(base_load[1][:, 0]) for base_load in env.history["base_load"]
    ]

    # Generate a 'time' column assuming timesteps are evenly spaced
    start_time = pd.to_datetime(common_config["start_time"])
    timestep_delta = common_config["control_timedelta"]
    time = [start_time + i * timestep_delta for i in range(len(total_ev_load))]

    # Create the DataFrame
    ev_load_df = pd.DataFrame({
        "time": time,
        "total_ev_load": total_ev_load,
        "total_load": total_base_load
    })

    return ev_load_df

def power_losses_to_dataframe(env):
    """
    Extracts power losses (real and reactive) from the environment's history and returns them as a DataFrame.

    Args:
        env: The environment object containing the power flow solver history.

    Returns:
        pd.DataFrame: A DataFrame with 'time' as the index and columns 'real_power_loss' and 'reactive_power_loss'.
    """
    # Extract power losses from env.history["losses"]
    real_power_losses = [loss[0] for loss in env.history["losses"]]
    reactive_power_losses = [loss[1] for loss in env.history["losses"]]

    # Convert to kW
    real_power_losses = np.array(real_power_losses) / 1000.0
    reactive_power_losses = np.array(reactive_power_losses) / 1000.0

    # Generate a 'time' column assuming timesteps are evenly spaced
    start_time = pd.to_datetime(common_config["start_time"])
    timestep_delta = common_config["control_timedelta"]
    time = [start_time + i * timestep_delta for i in range(len(real_power_losses))]

    # Create the DataFrame
    power_losses_df = pd.DataFrame({
        "time": time,
        "real_power_loss": real_power_losses,
        "reactive_power_loss": reactive_power_losses
    }).set_index("time")

    return power_losses_df

#base_load = env.pf_solver._obtain_base_load_info
#print("base load", base_load)

# Collect load profile data
#filtered_load_profile = load_profile_to_dataframe(pf_config, common_config)
ev_load_df = ev_load_to_dataframe(metas, env)
power_losses_df = power_losses_to_dataframe(env)
#print("Power losses DataFrame:\n", power_losses_df)
#ev_load_df["total_load"] = ev_load_df["total_ev_load"] + ev_load_df["total_base_load"]
#[print("Meta", meta) for meta in metas]

# # Add the EV load to the filtered load profile DataFrame
# filtered_load_profile = filtered_load_profile.merge(ev_load_df, on="time", how="left").fillna(0)
# filtered_load_profile["total_load"] = filtered_load_profile["load"] + filtered_load_profile["total_ev_load"]

plotting_mine.plot_2x2_nodal_voltages_and_load_profiles(env, charging_rates, ev_load_df, power_losses_df)
#plotting_mine.plot_3d_nodal_voltages_and_charging_rates(env, charging_rates)

