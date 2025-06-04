import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gridworld import ComponentEnv
from gridworld import MultiAgentEnv
from gridworld.distribution_system import OpenDSSSolver
from gridworld.agents.vehicles import EVChargingEnv
from mpl_toolkits.mplot3d import Axes3D


busses = ['671', '634a', '634b', '634c', '645', '675a', '675b', '675c', '670a', '670b', '670c', '684c']

agents = [
    {
        "name": "ev-charging-{}".format(i),
        "bus": busses[i],
        "cls": EVChargingEnv,
        "config": {
            "num_vehicles": 70,
            "minutes_per_step": 15,
            "max_charge_rate_kw": 7.,
            "peak_threshold": 250.,
            "vehicle_multiplier": 1.,
            "rescale_spaces": False
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


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "PowerGridworld_valleyfilling_SAC"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "PowerGridworld-v0"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
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
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
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
        flattened_obs = np.concatenate([obs[key] for key in sorted(obs.keys())])
        return flattened_obs, info

    def step(self, action):
        split_actions = {}
        start_idx = 0
        for agent, space in self.multi_agent_env.action_space.items():
            action_size = space.shape[0]
            split_actions[agent] = action[start_idx:start_idx + action_size]
            start_idx += action_size

        obs, rewards, dones, truncated, infos = self.multi_agent_env.step(split_actions)
        flattened_obs = np.concatenate([obs[key] for key in sorted(obs.keys())])
        dones["__all__"] = all(dones.values())

        # Aggregate rewards into a single float
        total_reward = sum(rewards.values())

        # Ensure "final_observation" is included in infos
        if "final_observation" in infos:
            infos["final_observation"] = {
                key: np.concatenate([obs[key] for key in sorted(obs.keys())])
                for key in infos["final_observation"]
            }

        return flattened_obs, total_reward, dones, truncated, infos

    def render(self, mode="human"):
        return self.multi_agent_env.render(mode)

    def close(self):
        return self.multi_agent_env.close()

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = MultiAgentEnv(**env_config)
            env = MultiAgentEnvWrapper(env)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = MultiAgentEnv(**env_config)
            env = MultiAgentEnvWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk