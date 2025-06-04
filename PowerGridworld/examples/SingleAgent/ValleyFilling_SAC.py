import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gridworld import ComponentEnv
from gridworld import MultiAgentEnv
from gridworld.distribution_system import OpenDSSSolver
from gridworld.agents.vehicles import EVChargingEnv
from mpl_toolkits.mplot3d import Axes3D

import plotting_mine
from plotting_mine import plot_2x2_nodal_voltages_and_load_profiles


busses = ['634a', '634b', '634c', '645', '675a', '675b', '675c', '670a', '670b', '670c', '684c']
# busses = ['633.1', '634.1', '634.2', '634.3', '645.3', '646.2', '675.2', '675.3', '680.1', '692.2', '611.3', '652.1']

agents = [
    {
        "name": "ev-charging-{}".format(i),
        "bus": busses[i],
        "cls": EVChargingEnv,
        "config": {
            "num_vehicles": 70,
            "minutes_per_step": 15,
            "max_charge_rate_kw": 7.,
            "peak_threshold": 700.,
            "vehicle_multiplier": 1.,
            "rescale_spaces": False,
            "unserved_penalty": 0.0,
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

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from gymnasium.spaces import Box, Dict, flatten_space, flatten
import wandb


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "PowerGridworld_valleyfilling_SAC"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "PowerGridworld-v0"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 4
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 1e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 5e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 10
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

from gymnasium import Env

class MultiAgentEnvWrapper(MultiAgentEnv,gym.Env):
    def __init__(self, multi_agent_env):
        self.multi_agent_env = multi_agent_env

        # Combine observation and action spaces
        #self.observation_space = Dict(multi_agent_env.observation_space)
        self.observation_space = Box(
            low=np.concatenate([space.low for space in multi_agent_env.observation_space.values()]),
            high=np.concatenate([space.high for space in multi_agent_env.observation_space.values()]),
            dtype=np.float32,  # Ensure float32 for compatibility
        )
        self.action_space = Box(
            low=np.concatenate([space.low for space in multi_agent_env.action_space.values()]),
            high=np.concatenate([space.high for space in multi_agent_env.action_space.values()]),
            dtype=np.float32,  # Ensure float32 for compatibility
        )

        # print("Observation space:", self.observation_space.shape)
        # print("Single observation space shape:", self.observation_space.shape)

        # Define single_observation_space and single_action_space for vectorized environments
        self.single_observation_space = flatten_space(self.observation_space)
        self.single_observation_space.dtype = np.float32  # Force dtype to float32
        self.single_action_space = self.action_space

        # Debugging: Print single_observation_space
        # print("Single observation space:", self.single_observation_space)
        # print("Single observation space shape:", self.single_observation_space.shape)
        # print("Single action space:", self.single_action_space)
        # print("Single action space shape:", self.single_action_space.shape)

    def reset(self, seed=None, options=None, **kwargs):
        """Reset the wrapped environment."""
        obs, info = self.multi_agent_env.reset(seed=seed, options=options, **kwargs)

        # Flatten the dictionary observation into a single array
        flattened_obs = np.concatenate([obs[key].astype(np.float32) for key in sorted(obs.keys())])
        return flattened_obs, info

    def step(self, action):
        split_actions = {}
        start_idx = 0
        for agent, space in self.multi_agent_env.action_space.items():
            action_size = space.shape[0]
            split_actions[agent] = action[start_idx:start_idx + action_size]
            start_idx += action_size

        obs, self.agent_rewards, dones, truncated, infos = self.multi_agent_env.step(split_actions)

        # Debugging: Print observation shapes
        # for agent, observation in obs.items():
        #     print(f"Agent: {agent}, Observation shape: {observation.shape}")

        flattened_obs = np.concatenate([obs[key].astype(np.float32) for key in sorted(obs.keys())])
        dones = all(dones.values())

        # # Ensure "final_info" is always included in infos
        # if "final_info" not in infos:
        #     infos["final_info"] = None

        # Aggregate rewards into a single float
        total_reward = sum(self.agent_rewards.values())

        # Ensure "final_observation" is included in infos
        if "final_observation" in infos:
            infos["final_observation"] = {
                key: np.concatenate([obs[key] for key in sorted(obs.keys())])
                for key in infos["final_observation"]
            }

        return flattened_obs, total_reward, dones, truncated, infos

    def render(self, mode="human"):
        """
        Renders the environment by plotting:
        1. Nodal voltages, power losses, and load profiles (2x2 plot).
        2. Rewards per agent over time.
        Saves the plots to WandB under the `videos/{run_name}` directory.
        """
        if mode != "human":
            raise NotImplementedError("Only 'human' mode is supported for rendering.")

    def close(self):
        return self.multi_agent_env.close()

def render(rend_env, actor, device, run_name=None):
    """
    Renders the environment by plotting:
    1. Nodal voltages, power losses, and load profiles (2x2 plot).
    2. Rewards per agent over time.
    Saves the plots to WandB under the `videos/{run_name}` directory.
    """
    import matplotlib.pyplot as plt
    import os
    from plotting_mine import plot_2x2_nodal_voltages_and_load_profiles

    if run_name is None:
        run_name = "default_run"

    # Ensure the videos directory exists
    video_dir = f"videos/{run_name}"
    os.makedirs(video_dir, exist_ok=True)

    # Collect data for plotting
    charging_rates = {agent: [] for agent in rend_env.multi_agent_env.agents}
    metas = []
    rewards = []
    done = False

    # Perform a single deterministic rollout
    obs = rend_env.reset()
    while not done:
        # Use the current policy to generate deterministic actions
        with torch.no_grad():
            if isinstance(obs, tuple):
                obs = obs[0]
            obs = torch.Tensor(obs).to(device).unsqueeze(0)
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()
            actions = actions.flatten()

        # Step the environment
        obs, rew, done, truncated, info = rend_env.step(actions)
        agent_reward_dict = rend_env.agent_rewards  # Access self.agent_rewards within rend_env


        if info is None:
            print("Warning: info is None at this timestep.")

        # Collect rewards and metadata
        metas.append(info)
        rewards.append(agent_reward_dict)

        # Collect charging rates for each agent
        for agent_name, action in zip(rend_env.multi_agent_env.agents, actions):
            charging_rates[agent_name].append(action)

    # Filter out None values from metas
    metas = [info for info in metas if info is not None]

    # Prepare data for plotting
    ev_load_df = plotting_mine.ev_load_to_dataframe(metas, rend_env.multi_agent_env)
    power_losses_df = power_losses_to_dataframe(rend_env.multi_agent_env)

    # Plot 3D charging rates
    plot_3d_charging_rates(charging_rates, video_dir)
    plt.savefig(f"{video_dir}/charging_rates.png")
    plt.close()

    # Plot 2x2 nodal voltages and load profiles
    plot_2x2_nodal_voltages_and_load_profiles(rend_env.multi_agent_env, charging_rates, ev_load_df, power_losses_df)
    plt.savefig(f"{video_dir}/nodal_voltages_and_load_profiles.png")
    plt.close()

    # Plot rewards per agent over time
    plt.figure(figsize=(12, 6))
    # print("Rewards:", rewards)
    for agent_name in rewards[0].keys():
        agent_rewards = [reward[agent_name] for reward in rewards]
        plt.plot(agent_rewards, label=agent_name)

    plt.title("Rewards per Agent Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(f"{video_dir}/rewards_per_agent.png")
    plt.close()

    # Log the plots to WandB
    if run_name:
        wandb.log({
            "nodal_voltages_and_load_profiles": wandb.Image(f"{video_dir}/nodal_voltages_and_load_profiles.png"),
            "rewards_per_agent": wandb.Image(f"{video_dir}/rewards_per_agent.png"),
            "charging_rates": wandb.Image(f"{video_dir}/charging_rates.png"),   
        })
        print(f"Plots saved to {video_dir} and logged to WandB.")

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

def make_env():
    def thunk():
        env = MultiAgentEnv(**env_config)
        env = MultiAgentEnvWrapper(env)
        # if capture_video and idx == 0:
        #     env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # env.action_space.seed(seed)
        return env
    return thunk

def plot_3d_charging_rates(charging_rates, video_dir):
    """
    Creates a 3D plot of EV charging rates over time for each agent.

    Args:
        charging_rates (dict): A dictionary where keys are agent names and values are lists of actions (charging rates) over timesteps.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    # Create a 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create a 2D plot for EV charging rates over time
    plt.figure(figsize=(12, 8))

    # Iterate over agents and their charging rates
    for agent_name, rates in charging_rates.items():
        # Ensure rates are a 1D array
        rates = np.array(rates).flatten()

        # Plot the charging rates over time for the agent
        plt.plot(range(len(rates)), rates, label=agent_name)

    # Add labels, legend, and title
    plt.xlabel("Timestep")
    plt.ylabel("Charging Rate (percentage)")
    plt.title("EV Charging Rates Over Time")
    plt.legend()
    plt.grid(True)

    return fig

# # Create the multi-agent environment
# multi_agent_env = MultiAgentEnv(**env_config)

# # Wrap the environment
# env = MultiAgentEnvWrapper(multi_agent_env)

# # Test reset
# obs = env.reset()
# print("Observation shape:", obs.shape)

# # Test step
# action = env.action_space.sample()
# next_obs, rewards, dones, infos = env.step(action)
# print("Next observation shape:", next_obs.shape)
# print("Rewards:", rewards)
# print("Dones:", dones)


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))

        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("Using device:", device)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env() for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    rend_env = MultiAgentEnv(**env_config)
    rend_env = MultiAgentEnvWrapper(rend_env)

    # print("Observation space:", envs.single_observation_space)
    # print("Action space:", envs.single_action_space)
    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if terminations.all() or truncations.all():  # Check if the episode is done
            # Debugging: Print the infos dictionary
            # print(f"Infos at global_step={global_step}: {infos}")
            if "_final_info" in infos:
                # print(f"Structure of final_info at global_step={global_step}: {infos['episode']['r']}")
                if infos is not None:
                    # print(f"global_step={global_step}, episodic_return={infos['episode']['r']}, episodic_length={infos['episode']['l']}")
                    writer.add_scalar("charts/episodic_return", np.mean(infos["episode"]["r"]), global_step)
                    writer.add_scalar("charts/episodic_length", np.mean(infos["episode"]["l"]), global_step)
                    print(f"global_step={global_step}, episodic_return={infos['episode']['r']}, episodic_length={infos['episode']['l']}")
                    # print("Final Info:", infos["final_info"])
                else:
                    print(f"Warning: 'final_info' is None at global_step={global_step}.")
            else:
                print(f"Warning: 'final_info' key not found in infos at global_step={global_step}.")

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                render(rend_env, actor=actor, device=device, run_name=run_name)

                # deterministic rollout
                with torch.no_grad():
                    deterministic_actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
                

            #if global_step % 1000 == 0:
                # # Ensure the 'models/' directory exists
                # os.makedirs("models", exist_ok=True)
                # # Save model parameters
                # torch.save({'actor_state_dict': actor.state_dict(),
                #     'qf1_state_dict': qf1.state_dict(),
                #     'qf2_state_dict': qf2.state_dict(),
                #     'qf1_target_state_dict': qf1_target.state_dict(),
                #     'qf2_target_state_dict': qf2_target.state_dict(),
                #     'actor_optimizer_state_dict': actor_optimizer.state_dict(),
                #     'q_optimizer_state_dict': q_optimizer.state_dict(),
                #     'log_alpha': log_alpha if args.autotune else None,
                #     'a_optimizer_state_dict': a_optimizer.state_dict() if args.autotune else None,
                # }, f"models/{run_name}.pth")
                


    envs.close()
    writer.close()