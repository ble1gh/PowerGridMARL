import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gridworld import ComponentEnv
from gridworld import MultiAgentEnv
from gridworld.distribution_system import OpenDSSSolver
from gridworld.agents.vehicles import EVChargingEnv
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_nodal_voltages_and_charging_rates(env, charging_rates):
    """
    Plots two 3D graphs:
    1. Nodal voltages over time.
    2. EV charging rates over time.

    Args:
        env: The environment object containing voltage and timestamp history.
        charging_rates (dict): A dictionary where keys are agent names and values are lists of actions over timesteps.
    """
    # Plot 1: Nodal Voltages Over Time
    voltage_data = env.history["voltage"]
    timestamps = env.history["timestamp"]

    # Convert voltage data to a DataFrame for easier manipulation
    df = pd.DataFrame(voltage_data, index=timestamps)

    # Convert timestamps to numeric values (e.g., seconds since the start)
    numeric_timestamps = (df.index - df.index[0]).total_seconds()

    # Create a 3D plot for nodal voltages
    fig1 = plt.figure(figsize=(12, 8))
    ax1 = fig1.add_subplot(111, projection='3d')

    for idx, node in enumerate(df.columns):
        x = [idx] * len(df)  # Node index (x-axis), repeated for all timesteps
        y = numeric_timestamps  # Timestamps converted to numeric (y-axis)
        z = df[node].values  # Voltage values (z-axis)

        # Plot the 3D line for the node
        ax1.plot(x, y, z, label=f"Node {node}")

    # Add labels and title for nodal voltages
    ax1.set_xlabel("Node Index")
    ax1.set_ylabel("Time (seconds)")
    ax1.set_zlabel("Voltage (p.u.)")
    ax1.set_title("Nodal Voltages Over Time")
    #ax1.legend(loc="upper right", bbox_to_anchor=(1.2, 1))
    plt.tight_layout()

    # Plot 2: EV Charging Rates Over Time
    fig2 = plt.figure(figsize=(12, 8))
    ax2 = fig2.add_subplot(111, projection='3d')

    for idx, (agent_name, actions) in enumerate(charging_rates.items()):
        # Ensure actions are flattened to 1D
        actions = np.array(actions).flatten()

        # Generate x, y, z coordinates
        x = [idx] * len(actions)  # Agent index (x-axis), repeated for all timesteps
        y = list(range(len(actions)))  # Timestep (y-axis)
        z = actions  # Action values (z-axis)

        # Plot the 3D line for the agent
        ax2.plot(x, y, z, label=agent_name)

    # Add labels and title for charging rates
    ax2.set_xlabel("Agent Index")
    ax2.set_ylabel("Timestep")
    ax2.set_zlabel("Action Value")
    ax2.set_title("EV Actions Over Time")
    ax2.legend(loc="upper right", bbox_to_anchor=(1.2, 1))
    plt.tight_layout()

    # Show both plots
    plt.show()

def plot_2d_nodal_voltages_and_charging_rates(env, charging_rates, load_profile_df):
    """
    Plots three 2D graphs in a single figure with subplots:
    1. Nodal voltages over time (collapsed along the index axis).
    2. EV charging rates over time (collapsed along the index axis).
    3. Load profile over time.

    Args:
        env: The environment object containing voltage and timestamp history.
        charging_rates (dict): A dictionary where keys are agent names and values are lists of actions over timesteps.
        load_profile_df (pd.DataFrame): A DataFrame containing 'time' and 'load' columns for the load profile.
    """
    # Plot 1: Nodal Voltages Over Time
    voltage_data = env.history["voltage"]
    timestamps = env.history["timestamp"]

    # Convert voltage data to a DataFrame for easier manipulation
    df = pd.DataFrame(voltage_data, index=timestamps)

    # Convert timestamps to numeric values (e.g., seconds since the start)
    numeric_timestamps = (df.index - df.index[0]).total_seconds()

    # Create a single figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

    # Subplot 1: Nodal Voltages Over Time
    for node in df.columns:
        y = df[node].values  # Voltage values (y-axis)
        ax1.plot(numeric_timestamps, y, label=f"Node {node}")

    # Add labels and title for nodal voltages
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Voltage (p.u.)")
    ax1.set_title("Nodal Voltages Over Time")
#    ax1.legend(loc="upper right", bbox_to_anchor=(1.2, 1))

    # Subplot 2: EV Charging Rates Over Time
    for agent_name, actions in charging_rates.items():
        y = np.array(actions).flatten()  # Action values (y-axis)
        x = list(range(len(y)))  # Timestep (x-axis)
        ax2.plot(x, y, label=agent_name)

    # Add labels and title for charging rates
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Action Value")
    ax2.set_title("EV Actions Over Time")
    ax2.legend(loc="upper right", bbox_to_anchor=(1.2, 1))

    # Subplot 3: Load Profile Over Time
    ax3.plot(load_profile_df["time"], load_profile_df["load"], label="Load Profile", color="green")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Load (kW)")
    ax3.set_title("Load Profile Over Time")
    ax3.legend(loc="upper right")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the combined figure
    plt.show()

def plot_2x2_nodal_voltages_and_load_profiles(env, charging_rates, load_profile_df, power_losses):
    """
    Plots four 2D graphs in a single figure with a 2x2 grid:
    1. Nodal voltages over time (top-left).
    2. Power losses over time (bottom-left).
    3. Base load profile over time (top-right).
    4. EV load profile over time (bottom-right).

    Args:
        env: The environment object containing voltage and timestamp history.
        charging_rates (dict): A dictionary where keys are agent names and values are lists of actions over timesteps.
        load_profile_df (pd.DataFrame): A DataFrame containing 'time', 'total_ev_load' and 'total_load' columns.
        power_losses (pd.DataFrame): A DataFrame with 'real_power_loss' and 'reactive_power_loss', and 'time' as the index.
    """
    # Ensure 'time' is in datetime format
    load_profile_df["time"] = pd.to_datetime(load_profile_df["time"])
    power_losses.index = pd.to_datetime(power_losses.index)

    # Check for missing values and handle them
    load_profile_df.fillna(0, inplace=True)
    power_losses.fillna(0, inplace=True)

    # Plot 1: Nodal Voltages Over Time
    voltage_data = env.history["voltage"]
    timestamps = env.history["timestamp"]

    # Convert timestamps to datetime format
    timestamps = pd.to_datetime(timestamps)

    # Convert voltage data to a DataFrame for easier manipulation
    df = pd.DataFrame(voltage_data, index=timestamps)

    # Convert timestamps to numeric values (e.g., seconds since the start)
    numeric_timestamps = (df.index - df.index[0]).total_seconds()

    # Create a single figure with a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Subplot 1 (Top-Left): Nodal Voltages Over Time
    ax1 = axes[0, 0]
    for node in df.columns:
        y = df[node].values  # Voltage values (y-axis)
        ax1.plot(timestamps, y, label=f"Node {node}")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Voltage (p.u.)")
    ax1.set_title("Nodal Voltages")
    # ax1.legend(loc="upper right", bbox_to_anchor=(1.2, 1))

    # Subplot 2 (Bottom-Left): Power Losses Over Time
    ax2 = axes[1, 0]
    ax2.plot(power_losses.index, power_losses["real_power_loss"], label="Real Power Loss", color="red")
    ax2.plot(power_losses.index, power_losses["reactive_power_loss"], label="Reactive Power Loss", color="purple")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Power Loss (kW / kVAR)")
    ax2.set_title("Power Losses")
    ax2.legend(loc="upper right")

    # Subplot 3 (Top-Right): Base Load Profile Over Time
    ax3 = axes[0, 1]
    ax3.plot(load_profile_df["time"], load_profile_df["total_load"], label="Total Load Profile", color="blue")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Load (kW)")
    ax3.set_title("Total Load Profile")
    ax3.legend(loc="upper right")

    # Subplot 4 (Bottom-Right): EV Load Profile Over Time
    ax4 = axes[1, 1]
    ax4.plot(load_profile_df["time"], load_profile_df["total_ev_load"], label="EV Load Profile", color="green")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Load (kW)")
    ax4.set_title("EV Load Profile")
    ax4.legend(loc="upper right")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Return the figure object for further manipulation or saving
    return fig

def ev_load_to_dataframe(metas, env):
    """
    Converts the 'real_power_consumed' from metas and 'base_load' from env.history into a DataFrame of total EV load,
    total base load, and timesteps.
    """
    # Filter out None values from metas
    metas = [info for info in metas if info is not None]

    # Extract real power consumed and calculate total EV load for each timestep
    total_ev_load = [
        sum(agent_data.get("real_power_consumed", 0) for agent_data in info.values() if agent_data is not None)
        for info in metas
    ]

    # Extract base load from env.history["base_load"] and calculate total base load for each timestep
    total_base_load = [
        np.sum(base_load[1][:, 0]) for base_load in env.history["base_load"]
    ]

    # Generate a 'time' column assuming timesteps are evenly spaced
    start_time = pd.to_datetime(env.common_config["start_time"])
    timestep_delta = env.common_config["control_timedelta"]
    time = [start_time + i * timestep_delta for i in range(len(total_ev_load))]

    # Create the DataFrame
    ev_load_df = pd.DataFrame({
        "time": time,
        "total_ev_load": total_ev_load,
        "total_load": total_base_load,
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