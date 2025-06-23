from abc import abstractmethod
from collections import defaultdict

import logging
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd

from gridworld.base import ComponentEnv, MultiComponentEnv
from gridworld.log import logger

from torchrl.envs import EnvBase

from torchrl.data import Composite, Bounded, CompositeSpec, Binary, Categorical
from tensordict import TensorDict
import torch

from gymnasium.spaces import Box

bus_name_mapping = {
    "634a": "634.1",
    "634b": "634.2",
    "634c": "634.3",
    "645": "645.2",
    "675a": "675.1",
    "675b": "675.2",
    "675c": "675.3",
    "670a": "670.1",
    "670b": "670.2",
    "670c": "670.3",
    "684c": "684.3",
    "611": "611.3",
    "652": "652.1",
    "671": "671.1.2.3",  # Example for a 3-phase bus
}


class MultiAgentEnv(EnvBase):
    """This class implements the multi-agent environment created from a list
    of agents of either type, ComponentEnv or MultiComponentEnv."""

    def __init__(
            self,
            common_config: dict = {},
            pf_config: dict = {},
            agents: list = None,
            max_episode_steps: int = None,
            rescale_spaces: bool = True,
            **kwargs
    ):
        super().__init__()

        self.common_config = common_config
        self.rescale_spaces = rescale_spaces
        assert len(agents) > 0, "need at least one agent!"

        # TODO:  If we required certain keys in this config dict, we need
        # to do some simple checking and raise a helpful error.
        self.start_time = pd.Timestamp(common_config["start_time"])
        self.end_time = pd.Timestamp(common_config["end_time"])
        self.control_timedelta = common_config["control_timedelta"]

        # Likewise here, need helpful checking and errors.
        self.pf_config = pf_config
        self.max_episode_steps = max_episode_steps if max_episode_steps is not None else np.inf
        
        self.episode_step = None
        self.time = None
        self.history = None
        self.voltages = None
        self.obs_dict = {}
        self._global_reward_components = {}

        # Call the agent constructors with both common and agent-specific 
        # configuration arguments.
        self.agents = []
        for a in agents:

            # Top-level name argument overrides a name in the config.  The
            # constructor will error out otherwise because it gets two values
            # of the argument, one from config dict and one from the agent name.
            _config = a["config"]
            if "name" in a["config"]:
                _config = {k: v for k, v in _config.items() if k != "name"}
                logger.warning(
                    f"ignoring 'name' in config dict in favor of constructor argument")
            
            # Call the constructor and append to the agent list.
            new_agent = a["cls"](name=a["name"], **_config, **self.common_config)
            self.agents.append(new_agent)

        # Keep track of which bus each agent is attached to.    
        self.agent_name_bus_map = {a["name"]: a["bus"] for a in agents} 
        
        # Create a list of agent names and ensure they are unique.
        self.agent_names = list(set([a.name for a in self.agents]))
        assert len(self.agent_names) == len(agents), "all agents need unique names"

        # Instantiate the powerflow solver.
        self.pf_solver = pf_config["cls"](**pf_config["config"])

        # Create the gym observation and action spaces.
        try:
            self.observation_space = {
                agent.name: Box(
                    low=agent.observation_space.low,
                    high=agent.observation_space.high,
                    dtype=np.float64
                )
                for agent in self.agents
            }
        except Exception as e:
            print("Exception while building observation_spec:", e)
            raise

        self.action_space = {
            agent.name: agent.action_space for agent in self.agents}
        
        n_agents = len(self.agents)
        obs_dim = self.agents[0].observation_space.shape[0]+1  # +1 for voltage
        act_dim = self.agents[0].action_space.shape[0]

        # Get bounds from any agent (they are all the same)
        obs_low = torch.as_tensor(self.agents[0].observation_space.low, dtype=torch.float32)
        obs_high = torch.as_tensor(self.agents[0].observation_space.high, dtype=torch.float32)
        # Append 0 to obs_low and inf to obs_high for the voltage dimension
        obs_low = torch.cat([obs_low, torch.tensor([0.0], dtype=torch.float32)])
        obs_high = torch.cat([obs_high, torch.tensor([float('inf')], dtype=torch.float32)])
        # print("obs_low shape:", obs_low.shape)
        # print("obs_high shape:", obs_high.shape)
        act_high = torch.as_tensor(self.agents[0].action_space.high, dtype=torch.float32)
        act_low = torch.as_tensor(self.agents[0].action_space.low, dtype=torch.float32)

        # First create a per-agent spec without batching
        agent_obs_spec = Composite({
            "observation": Bounded(
                low=obs_low,
                high=obs_high,
                shape=(obs_dim,),
                dtype=torch.float32
            )
        })

        # Then expand it to match your agent count
        self.observation_spec = Composite({
            "agents": agent_obs_spec.expand(n_agents)
        })
        # print("Observation spec:", self.observation_spec)

        # Same for action spec
        agent_act_spec = Composite({
            "action": Bounded(
                low=act_low,
                high=act_high,
                shape=(act_dim,),
                dtype=torch.float32
            )
        })

        self.action_spec = Composite({
            "agents": agent_act_spec.expand(n_agents)
        })

        # Define reward spec
        agent_reward_spec = Composite({
            "reward": Bounded(
                low=float("-inf"),
                high=float("inf"),
                shape=(1,),
                dtype=torch.float32
            ),
            "episode_reward": Bounded(
                low=float("-inf"),
                high=float("inf"),
                shape=(1,),
                dtype=torch.float32
            )
        })

        # Expand to all agents
        self.reward_spec = Composite({
            "agents": agent_reward_spec.expand(n_agents)
        })

        self.done_spec = Composite({
            "agents": Composite({
                "done": Categorical(n=2, dtype=torch.bool, shape=(1,)),
                "terminated": Categorical(n=2, dtype=torch.bool, shape=(1,)),
                "truncated": Categorical(n=2, dtype=torch.bool, shape=(1,)),
            }).expand(n_agents),
            "done": Categorical(n=2, dtype=torch.bool, shape=(1,)),
            "terminated": Categorical(n=2, dtype=torch.bool, shape=(1,)),
            "truncated": Categorical(n=2, dtype=torch.bool, shape=(1,)),
        })

        self.info_spec = Composite({
            "agents": Composite({
                "agent_info": Composite({
                    "real_power_unserved": Bounded(low=float(0.0), high=float("inf"), shape=(), dtype=torch.float32),
                    "peak_reward": Bounded(low=float("-inf"), high=float(0.0), shape=(), dtype=torch.float32),
                })
            }).expand(n_agents),
            "info": Composite({
                "power_loss_reward": Bounded(low=float("-inf"), high=float(0.0), shape=(), dtype=torch.float32),
                "voltage_reward": Bounded(low=float("-inf"), high=float(0.0), shape=(), dtype=torch.float32),
                "load_penalty": Bounded(low=float("-inf"), high=float(0.0), shape=(), dtype=torch.float32),
            })
        })


    def close(self, raise_if_closed=True):
        """Clean up resources used by the environment.
    
        Args:
            raise_if_closed (bool): If True, raising an error if the environment
                is already closed is allowed. Default is True.
        """
        # Add a check for whether the environment is already closed
        if hasattr(self, '_closed') and self._closed:
            if raise_if_closed:
                raise RuntimeError("Trying to close an environment that is already closed")
            return
            
        # Close the power flow solver if it has a close method
        if hasattr(self.pf_solver, "close"):
            try:
                self.pf_solver.close()
            except Exception as e:
                logger.warning(f"Error closing power flow solver: {e}")
                
        # Close all agents if they have a close method
        for agent in self.agents:
            if hasattr(agent, "close"):
                try:
                    agent.close()
                except Exception as e:
                    logger.warning(f"Error closing agent {agent.name}: {e}")
        
        # Mark the environment as closed
        self._closed = True
        
        # Log that the environment has been closed
        logger.info("MultiAgentEnv has been closed.")
    
    # Remove the '*' from the signature to allow positional arguments
    def _reset(self, tensordict=None, **kwargs):
        # Get observations from environment logic
        obs_dict = self._reset_logic(**kwargs)

        # Reset info reward components
        self._global_reward_components = {}
        self.episode_reward = 0
        
        agent_obs = torch.stack([
            torch.as_tensor(obs_dict[agent_name], dtype=torch.float32)
            for agent_name in self.agent_names
        ])
        
        # Create TensorDict matching spec structure 
        obs_td = TensorDict({
            "agents": TensorDict({
                "observation": agent_obs,
            }, batch_size=[len(self.agent_names)])
        }, batch_size=[])
        # print(f"Observation on Reset: {obs_td}")
        
        return obs_td

    def _step(self, tensordict=None):
        # Extract actions from tensordict
        actions = tensordict["agents"]["action"].clone()
        
        # Convert actions to dictionary format for env logic
        action_dict = {
            self.agent_names[i]: actions[i].cpu().numpy() 
            for i in range(len(self.agent_names))
        }
        
        # Call environment step logic
        obs, rewards, dones, truncated, per_agent_info = self._step_logic(action_dict)
        
        # Stack all agent observations into a single tensor [n_agents, obs_dim]
        agent_obs = torch.stack([
            torch.as_tensor(obs[agent_name], dtype=torch.float32) 
            for agent_name in self.agent_names
        ])
        
        # Create per-agent reward tensor
        agent_rewards = torch.tensor([
            rewards[agent_name] for agent_name in self.agent_names
        ], dtype=torch.float32).unsqueeze(-1)
        
        # Get done and truncated as boolean tensors
        agent_dones = torch.tensor([
            dones[agent_name] for agent_name in self.agent_names
        ], dtype=torch.bool).unsqueeze(-1)
        
        agent_truncs = torch.tensor([
            truncated[agent_name] if isinstance(truncated, dict) else truncated 
            for agent_name in self.agent_names
        ], dtype=torch.bool).unsqueeze(-1)
        
        # Create per-agent episode reward tensor
        agent_episode_rewards = torch.tensor([
            self.obs_dict["episode_reward"].get(agent_name, 0) 
            for agent_name in self.agent_names
        ], dtype=torch.float32).unsqueeze(-1)
        
        # Create per-agent info dictionary
        per_agent_info_td = TensorDict({
            "real_power_unserved": torch.tensor([
                per_agent_info[agent_name].get("real_power_unserved", 0) 
                for agent_name in self.agent_names
            ], dtype=torch.float32),
            "peak_reward": torch.tensor([
                per_agent_info[agent_name].get("peak_reward", 0) 
                for agent_name in self.agent_names
            ], dtype=torch.float32),
        }, batch_size=[len(self.agent_names)])
        
        # Create group-level info dictionary
        group_info_td = TensorDict({
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in self._global_reward_components.items()
        }, batch_size=[])

        # Create output TensorDict
        next_obs = TensorDict({
            "agents": TensorDict({
                "observation": agent_obs,  # Per-agent observations
                "reward": agent_rewards,  # Per-agent rewards
                "episode_reward": agent_episode_rewards,  # Per-agent episode rewards
                "terminated": agent_dones,  # Per-agent termination flags
                "truncated": agent_truncs,  # Per-agent truncated flags
                "done": agent_dones | agent_truncs,  # Per-agent done flags
                "agent_info": per_agent_info_td,  # Per-agent info
            }, batch_size=[len(self.agent_names)]),
            "info": group_info_td,  # Group-level info
            "done": (agent_dones.squeeze(-1) | agent_truncs.squeeze(-1)).any().reshape(1),
            "terminated": agent_dones.any().reshape(1),
            "truncated": agent_truncs.any().reshape(1),
        }, batch_size=[])

        # print(f"Next Obs reward: {next_obs['agents']['reward']}")
        # print(f"Next Obs episode reward: {next_obs['agents']['episode_reward']}")
    
        return next_obs

    def get_external_obs_vars(
        self, 
        agent: Union[ComponentEnv, MultiComponentEnv],
        seed
    ) -> dict:
        """These are external variables to the agents, need to implement how
        they get this data so it can be passed to their reset/step methods
        and added to the observation space.  Currently, a user will have to
        overwrite the method to give agents access to other quantities.
        TODO: Design an interface for a user to customize this."""

        kwargs = {}

        # Get the bus voltage at the agent's bus.
        if "bus_voltage" in agent.obs_labels:
            kwargs["bus_voltage"] = self.pf_solver.get_bus_voltage_by_name(
                self.agent_name_bus_map[agent.name])

        # Get the maximum voltage across all buses.
        if "max_voltage" in agent.obs_labels:
            kwargs["max_voltage"] = max(list(self.voltages.values()))

        # Get the minimum voltage across all buses.
        if "min_voltage" in agent.obs_labels:
            kwargs["min_voltage"] = min(list(self.voltages.values()))

        return kwargs

    
    @abstractmethod
    def reward_transform(self, agent_rewards: dict) -> dict:
        """Function to transform the agent rewards based on centralized view.
        Pass-through by default but can be overwrittent for custom rewards."""
        return agent_rewards


    def _reset_logic(self, seed=None, options=None, **kwargs) -> Tuple[Dict[str, any], dict]:
        """Reset the environment and return the initial observations for all agents."""
        self.episode_step = 0
        self.time = self.start_time
        self.history = {"timestamp": [], "voltage": [], "agent_power_p": [], "base_load": [], "losses": []}
        self.episode_reward = 0
        self.obs_dict = {}

        # Run OpenDSS to have voltage info
        self.pf_solver.calculate_power_flow(current_time=self.time)
        self.voltages = self.pf_solver.get_bus_voltages()
        self.base_load = self.pf_solver._obtain_base_load_info()
        self.losses = self.pf_solver.get_losses()

        # Reset the controllable agents and collect their obs arrays
        for agent in self.agents:
            kwargs = self.get_external_obs_vars(agent, seed=seed)
            _ = agent.reset(**kwargs)

        # Return observations and an empty info dictionary
        obs = self.get_obs()
        obs = self.obs_transform(obs)
        return obs


    def get_obs(self) -> Dict[str, any]:
        obs = {}
        for agent in self.agents:
            kwargs = self.get_external_obs_vars(agent, seed=None)
            obs[agent.name], _ = agent.get_obs(**kwargs)
        return obs


    def _step_logic(self, action: Dict[str, any]) -> Tuple[dict, dict, dict, dict]:
        self.episode_step += 1
        self.time += pd.Timedelta(seconds=self.control_timedelta)

        # Initialize agent outputs.
        obs, rew, done, meta = {}, {}, {}, {}
        load_p, load_q = {}, {}
        agent_power_p = []

        # For each agent, call the step method and inject any external variables
        # as keyword arguments. Accumulate the real/reactive power from each
        # agent for use in power flow calculation.
        for agent in self.agents:
            name = agent.name
            kwargs = self.get_external_obs_vars(agent, seed=None)
            obs[name], rew[name], done[name], meta[name] = agent.step(
                action=action[name], **kwargs
            )

            load_bus = self.agent_name_bus_map[name]
            agent_p_consumed = agent.real_power
            agent_q_consumed = agent.reactive_power
            agent_power_p.append(agent_p_consumed)

            if load_bus in load_p.keys():
                load_p[load_bus] += agent_p_consumed
                load_q[load_bus] += agent_q_consumed
            else:
                load_p[load_bus] = agent_p_consumed
                load_q[load_bus] = agent_q_consumed

        # Call power flow solver and update the bus voltages.
        self.pf_solver.calculate_power_flow(
            current_time=self.time,
            p_controllable_consumed=load_p,
            q_controllable_consumed=load_q
        )
        self.voltages = self.pf_solver.get_bus_voltages()
        self.base_load = self.pf_solver._obtain_base_load_info()
        self.losses = self.pf_solver.get_losses()

        # Update history dict.
        self.history["timestamp"].append(self.time)
        self.history["voltage"].append(self.voltages.copy())
        self.history["agent_power_p"].append(agent_power_p)
        self.history["base_load"].append(self.base_load)
        self.history["losses"].append(self.losses)

        # Check for terminal condition.
        any_done = np.any(list(done.values()))
        max_steps_reached = (self.episode_step == self.max_episode_steps - 1)
        time_up = self.time >= self.end_time
        done = any_done or max_steps_reached or time_up
        # print(f"Episode step: {self.episode_step}, Done: {done}, Time up: {time_up}, Max steps reached: {max_steps_reached}")

        # Create the dones dict that will be returned.
        dones = {a.name: done for a in self.agents}

        # Transform rewards and meta
        rew = self.reward_transform(rew)
        # meta = self.meta_transform(meta)
        obs = self.obs_transform(obs)

        # Extract agent-specific reward components from meta
        per_agent_info = {}
        for agent_name, agent_meta in meta.items():
            if agent_name in self.agent_names:  # Ensure it's an actual agent
                # Extract "real_power_unserved" and "peak_reward" if available
                if isinstance(agent_meta, dict):
                    per_agent_info[agent_name] = {}
                    if "real_power_unserved" in agent_meta:
                        per_agent_info[agent_name]["real_power_unserved"] = agent_meta["real_power_unserved"]
                    if "peak_reward" in agent_meta:
                        per_agent_info[agent_name]["peak_reward"] = agent_meta["peak_reward"]

        # Update meta with global reward components
        if hasattr(self, '_global_reward_components'):
            meta["reward_components"] = self._global_reward_components

        for agent_name in rew:
            if "episode" not in meta[agent_name]:
                meta[agent_name]["episode"] = {"r": 0, "l": 0}
            meta[agent_name]["episode"]["r"] += rew[agent_name]
            meta[agent_name]["episode"]["l"] = self.episode_step

        truncated = {a.name: False for a in self.agents}

        return obs, rew, dones, truncated, per_agent_info

    def obs_transform(self, obs_dict) -> dict:
        """Function to transform the agent observations based on centralized view."""
        for agent_name in obs_dict:
            # Ensure the observation array is of type float32
            obs_dict[agent_name] = obs_dict[agent_name].astype(np.float32)

            # Get the bus name in numeric form
            bus_name = self.agent_name_bus_map[agent_name]
            numeric_bus_name = bus_name_mapping.get(bus_name, bus_name)  # Default to original if not found

            # Get the voltage for the agent's bus and ensure it is a float32
            voltage = float(self.voltages[numeric_bus_name])

            # Append the voltage to the observation
            obs_dict[agent_name] = np.append(obs_dict[agent_name], voltage).astype(np.float32)

        return obs_dict

    def reward_transform(self, rew_dict) -> dict:
        """Function to transform the agent rewards based on centralized view."""
    
        # Calculate the power loss reward
        power_loss_reward = -self.losses[0] / 1e5

        # Calculate voltage violation reward
        voltage_reward = 0
        # Check if any voltage is below 0.95 p.u.
        # If so, calculate the total voltage difference from 0.95 p.u.
        if np.any(np.array(list(self.voltages.values())) < 0.95):
            voltage_differences = [0.95 - v for v in self.voltages.values() if v < 0.95]
            total_voltage_difference = sum(voltage_differences)
            voltage_reward = -total_voltage_difference * 1e3

        # Calculate load penalty
        total_load = []
        for item in self.history['base_load'][:]:
            bus_names, data_array = item
            load = data_array[:, 0].sum()
            total_load.append(load)

        # Convert the list of arrays to a 2D numpy array
        total_load_array = np.array(total_load)
        
        # Load stability reward
        # print(f"base_load: {self.history['base_load'][:]}")
        load_penalty = -np.linalg.norm(total_load_array)/1e5
        # print(f"Load penalty: {load_penalty}")
        
        # print(f"Voltage reward: {voltage_reward}")

        # Store global reward components for group-level logging
        self._global_reward_components = {
            "power_loss_reward": float(power_loss_reward),
            "voltage_reward": float(voltage_reward),
            "load_penalty": float(load_penalty),
        }

        # Add global rewards to each agent's reward individually
        for agent_name in rew_dict:
            if isinstance(rew_dict[agent_name], (int, float)):
                rew_dict[agent_name] += power_loss_reward + voltage_reward
            else:
                logger.warning(f"Reward for agent {agent_name} is not a number: {rew_dict[agent_name]}")

        # Track individual agent rewards for the episode
        for agent_name in rew_dict:
            if "episode_reward" not in self.obs_dict:
                self.obs_dict["episode_reward"] = {}
            self.obs_dict["episode_reward"][agent_name] = (
                self.obs_dict["episode_reward"].get(agent_name, 0) + rew_dict[agent_name]
            )

        return rew_dict


    # def meta_transform(self, meta) -> dict:
    #     """Function to augment the agent meta info based on centralized view.
    #     Pass-through by default.
    #     """
    #     return meta


    @property
    def agent_dict(self) -> Dict[str, ComponentEnv]:
        return {a.name: a for a in self.agents}
    
    # def _reset(self, *args, **kwargs):
    #     # Implement your reset logic or call your existing reset logic
    #     return self.reset(*args, **kwargs)

    # def _step(self, *args, **kwargs):
    #     # Implement your step logic or call your existing step logic
    #     return self.step(*args, **kwargs)

    def _set_seed(self, seed: int):
        # Implement your seeding logic if needed
        self.seed = seed
        np.random.seed(seed)
        # If your agents or other components need seeding, do it here
