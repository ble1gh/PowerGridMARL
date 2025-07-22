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
import warnings

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

        # Compute max_episode_steps based on time range and control interval
        total_seconds = (self.end_time - self.start_time).total_seconds()
        step_seconds = pd.Timedelta(seconds=self.control_timedelta).total_seconds()
        self.max_episode_steps = int(total_seconds // step_seconds) + 1
        
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
                    "energy_remaining": Bounded(low=float(0.0), high=float("inf"), shape=(), dtype=torch.float32),
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
            "energy_remaining": torch.tensor([
                sum(per_agent_info[agent_name].get("energy_remaining", {}).values()) 
                if isinstance(per_agent_info[agent_name].get("energy_remaining", {}), dict)
                else 0.0
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
        self.history = {
            "timestamp": [], "voltage": [], "agent_power_p": [], 
            "total_load": [], "losses": [], "actions": [], 
            "reward_components": [], "per_agent_info": [], "agent_rewards": []
        }
        self.episode_reward = 0
        self.obs_dict = {}

        # Run OpenDSS to have voltage info
        self.pf_solver.calculate_power_flow(current_time=self.time)
        self.voltages = self.pf_solver.get_bus_voltages()
        self.total_load = self.pf_solver._obtain_base_load_info()
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
        self.total_load = self.pf_solver._obtain_base_load_info()
        self.losses = self.pf_solver.get_losses()

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
                per_agent_info[agent_name] = {}
                if isinstance(agent_meta, dict):
                    if "energy_remaining" in agent_meta:
                        per_agent_info[agent_name]["energy_remaining"] = agent_meta["energy_remaining"]
                        if isinstance(agent_meta["energy_remaining"], (int, float)) and agent_meta["energy_remaining"] == 0:
                            warnings.warn(
                                f"[MultiAgentEnv] 'energy_remaining' for agent '{agent_name}' is 0 (default) at step {self.episode_step}.",
                                UserWarning,
                            )
                    elif "energy_remaining" not in agent_meta:
                        warnings.warn(
                            f"[MultiAgentEnv] 'energy_remaining' missing in meta for agent '{agent_name}' at step {self.episode_step}.",
                            UserWarning,
                        )
                    if "peak_reward" in agent_meta:
                        per_agent_info[agent_name]["peak_reward"] = agent_meta["peak_reward"]
                    else:
                        warnings.warn(
                            f"[MultiAgentEnv] 'peak_reward' missing in meta for agent '{agent_name}' at step {self.episode_step}.",
                            UserWarning,
                        )
                    if "real_energy_unserved" in agent_meta:
                        per_agent_info[agent_name]["real_energy_unserved_reward"] = agent_meta["real_energy_unserved"]
                    else:
                        warnings.warn(
                            f"[MultiAgentEnv] 'real_energy_unserved' missing in meta for agent '{agent_name}' at step {self.episode_step}.",
                            UserWarning,
                        )

        # Update meta with global reward components
        if hasattr(self, '_global_reward_components'):
            meta["reward_components"] = self._global_reward_components

        for agent_name in rew:
            if "episode" not in meta[agent_name]:
                meta[agent_name]["episode"] = {"r": 0, "l": 0}
            meta[agent_name]["episode"]["r"] += rew[agent_name]
            meta[agent_name]["episode"]["l"] = self.episode_step

        truncated = {a.name: False for a in self.agents}

        # Update history dict.
        self.history["timestamp"].append(self.time)
        self.history["voltage"].append(self.voltages.copy())
        self.history["agent_power_p"].append(agent_power_p)
        self.history["total_load"].append(self.total_load)
        self.history["losses"].append(self.losses)
        self.history["actions"].append(action)
        self.history["reward_components"].append(self._global_reward_components.copy())
        self.history["per_agent_info"].append(per_agent_info)
        self.history["agent_rewards"].append(rew.copy())


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
        for item in self.history['total_load'][:]:
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

    @property
    def agent_dict(self) -> Dict[str, ComponentEnv]:
        return {a.name: a for a in self.agents}
    
    def _set_seed(self, seed: int):
        # Implement your seeding logic if needed
        self.seed = seed
        np.random.seed(seed)
        # If your agents or other components need seeding, do it here
    
    def render_rollout_fig(self):
        """
        Generates plots from the episode history and logs them to WandB.
        This method is called from close() when an evaluation episode ends.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed, skipping plot generation.")
            return

        if not self.history or not self.history["timestamp"]:
            logger.info("No history to plot, skipping render.")
            return

        # --- Data Processing ---
        timestamps = pd.to_datetime(self.history["timestamp"])
        
        # Actions
        actions_df = pd.DataFrame([{agent: a[0] for agent, a in step_action.items()} for step_action in self.history["actions"]], index=timestamps)
        
        # Voltages
        voltages_df = pd.DataFrame(self.history["voltage"], index=timestamps)
        
        # Power Losses
        losses_df = pd.DataFrame([{"real": l[0], "reactive": l[1]} for l in self.history["losses"]], index=timestamps)
        
        # Loads
        ev_load = [sum(p) for p in self.history["agent_power_p"]]
        # Assuming the history value is the total load.
        # The history stores a tuple: (bus_names, load_array). We sum the real power column.
        total_load = [l[1][:, 0].sum() for l in self.history["total_load"]]
        load_df = pd.DataFrame({"Agent Load": ev_load, "Total Load": total_load}, index=timestamps)

        # Per-Agent Info with warnings if missing
        # Build a flat list of all vehicles for legend
        energy_remaining_records = []
        vehicle_labels = set()
        agent_colors = {}
        import matplotlib.pyplot as plt
        color_map = plt.get_cmap('tab10')
        
        # Create color mapping based on bus connections
        # Get unique buses and map them to colors
        unique_buses = list(set(self.agent_name_bus_map.values()))
        bus_color_map = {bus: color_map(i % 10) for i, bus in enumerate(unique_buses)}
        
        # Map agents to colors based on their bus
        agent_color_list = {agent: bus_color_map[self.agent_name_bus_map[agent]] for agent in self.agent_names}

        for t, step_info in enumerate(self.history["per_agent_info"]):
            # For each agent, get their per-vehicle dict
            for agent in self.agent_names:
                info = step_info.get(agent, {})
                er_dict = info.get("energy_remaining", None)
                if er_dict is None:
                    warnings.warn(
                        f"[render_rollout_fig] 'energy_remaining' missing for agent '{agent}' at timestep {t}.",
                        UserWarning,
                    )
                    continue
                if isinstance(er_dict, dict):
                    for veh_id, val in er_dict.items():
                        energy_remaining_records.append({
                            "timestamp": timestamps[t],
                            "agent": agent,
                            "vehicle": veh_id,
                            "energy_remaining": val
                        })
                        vehicle_labels.add((agent, veh_id))
                        # if val == 0:
                        #     warnings.warn(
                        #         f"[render_rollout_fig] 'energy_remaining' for agent '{agent}', vehicle '{veh_id}' is 0 at timestep {t}.",
                        #         UserWarning,
                        #     )
                else:
                    warnings.warn(
                        f"[render_rollout_fig] 'energy_remaining' for agent '{agent}' at timestep {t} is not a dict.",
                        UserWarning,
                    )

        # Build a DataFrame for plotting
        energy_remaining_long = pd.DataFrame(energy_remaining_records)
        # Pivot to have columns as (agent, vehicle)
        energy_remaining_pivot = energy_remaining_long.pivot_table(
            index="timestamp",
            columns=["agent", "vehicle"],
            values="energy_remaining"
        )

        # Agent rewards and unserved reward data
        agent_reward_data = []
        unserved_reward_data = []
        for t, (step_rewards, step_info) in enumerate(zip(self.history["agent_rewards"], self.history["per_agent_info"])):
            ar_row = {}
            ur_row = {}
            for agent in self.agent_names:
                # Get agent reward for this timestep
                ar_row[agent] = step_rewards.get(agent, 0)
                
                # Get unserved reward info
                info = step_info.get(agent, {})
                if "real_energy_unserved_reward" not in info:
                    warnings.warn(
                        f"[render_rollout_fig] 'real_energy_unserved_reward' missing for agent '{agent}' at timestep {t}.",
                        UserWarning,
                    )
                ur_row[agent] = info.get("real_energy_unserved_reward", 0)
            agent_reward_data.append(ar_row)
            unserved_reward_data.append(ur_row)
        agent_reward_df = pd.DataFrame(agent_reward_data, index=timestamps)
        unserved_reward_df = pd.DataFrame(unserved_reward_data, index=timestamps)

        # # debugging
        # print(energy_remaining_df)
        # print(energy_remaining_df.dtypes)

        # Global Reward Components
        reward_comp_df = pd.DataFrame(self.history["reward_components"], index=timestamps)

        # --- Plotting ---
        fig, axes = plt.subplots(4, 2, figsize=(20, 24))
        fig.suptitle("Evaluation Rollout", fontsize=16)
        plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.05, hspace=0.3, wspace=0.25)
        
        # Plot 1: Agent Actions
        ax_actions = axes[0, 0]
        plotted_buses = set()
        for agent in self.agent_names:
            if agent in actions_df.columns:
                bus = self.agent_name_bus_map[agent]
                # Only add legend entry for the first agent on each bus
                label = f"Bus {bus}" if bus not in plotted_buses else None
                ax_actions.plot(actions_df.index, actions_df[agent], 
                              label=label, 
                              color=agent_color_list[agent])
                plotted_buses.add(bus)
        ax_actions.set_title("Agent Actions")
        ax_actions.set_ylabel("Action Value")
        ax_actions.grid(True)
        ax_actions.legend(loc='best', fontsize='small')

        # Plot 2: Nodal Voltages
        ax_voltages = axes[0, 1]
        
        # Create a comprehensive color mapping for all buses in the voltage data
        # First, get all bus names from the voltage data
        all_voltage_buses = list(voltages_df.columns)
        
        # Create a color mapping that includes both agent buses and voltage buses
        extended_color_map = {}
        color_index = 0
        
        # First assign colors to buses with agents
        for agent in self.agent_names:
            agent_bus = self.agent_name_bus_map[agent]
            if agent_bus not in extended_color_map:
                extended_color_map[agent_bus] = color_map(color_index % 10)
                color_index += 1
        
        # Then assign colors to remaining voltage buses
        for bus in all_voltage_buses:
            if bus not in extended_color_map:
                # Check if this bus corresponds to an agent bus through the mapping
                mapped_bus = None
                for mapped_name, full_name in bus_name_mapping.items():
                    if bus == mapped_name or bus == full_name:
                        # Find if any agent is connected to the mapped bus
                        for agent_bus in self.agent_name_bus_map.values():
                            if agent_bus == full_name or agent_bus == mapped_name:
                                mapped_bus = agent_bus
                                break
                        break
                
                if mapped_bus and mapped_bus in extended_color_map:
                    extended_color_map[bus] = extended_color_map[mapped_bus]
                else:
                    extended_color_map[bus] = color_map(color_index % 10)
                    color_index += 1
        
        # Plot all voltage buses with their assigned colors
        for bus in voltages_df.columns:
            color = extended_color_map[bus]
            ax_voltages.plot(voltages_df.index, voltages_df[bus], 
                           label=f"Bus {bus}", color=color)
        ax_voltages.set_title("Nodal Voltages")
        ax_voltages.set_ylabel("Voltage (p.u.)")
        ax_voltages.grid(True)
        ax_voltages.legend(loc='best', fontsize='small')

        # Plot 3: Power Losses
        losses_df.plot(ax=axes[1, 0])
        axes[1, 0].set_title("Power Losses")
        axes[1, 0].set_ylabel("Power (kW/kVAR)")
        axes[1, 0].grid(True)

        # Plot 4: Total and Agent Load
        load_df.plot(ax=axes[1, 1])
        axes[1, 1].set_title("System Load")
        axes[1, 1].set_ylabel("Power (kW)")
        axes[1, 1].grid(True)

        # Plot 5: Energy Remaining (per Vehicle, colored by agent's bus)
        ax_er = axes[2, 0]
        plotted_buses = set()
        # Since each agent now controls only one EV, simplify the plotting
        for (agent, vehicle) in energy_remaining_pivot.columns:
            color = agent_color_list[agent]
            bus = self.agent_name_bus_map[agent]
            # Only add legend entry for the first agent on each bus
            label = f"Bus {bus}" if bus not in plotted_buses else None
            ax_er.plot(
                energy_remaining_pivot.index,
                energy_remaining_pivot[(agent, vehicle)],
                label=label,
                color=color,
                linestyle='-',
                marker=None,
                alpha=0.7
            )
            plotted_buses.add(bus)
        ax_er.set_title("Remaining Energy Need (by Vehicle)")
        ax_er.set_ylabel("Energy Remaining (kWh)")
        ax_er.grid(True)
        ax_er.legend(loc='best', fontsize='small', ncol=2)

        # Plot 6: Agent Rewards (per Agent)
        ax_rewards = axes[2, 1]
        plotted_buses = set()
        for agent in self.agent_names:
            if agent in agent_reward_df.columns:
                bus = self.agent_name_bus_map[agent]
                # Only add legend entry for the first agent on each bus
                label = f"Bus {bus}" if bus not in plotted_buses else None
                ax_rewards.plot(agent_reward_df.index, agent_reward_df[agent], 
                              label=label, 
                              color=agent_color_list[agent])
                plotted_buses.add(bus)
        ax_rewards.set_title("Agent Rewards (per Agent)")
        ax_rewards.set_ylabel("Reward")
        ax_rewards.grid(True)
        ax_rewards.legend(loc='best', fontsize='small')

        # Plot 7: Global Reward Components
        reward_comp_df.plot(ax=axes[3, 0])
        axes[3, 0].set_title("Global Reward Components")
        axes[3, 0].set_ylabel("Reward Value")
        axes[3, 0].grid(True)

        # Plot 8: Unserved Reward (per Agent)
        ax_unserved = axes[3, 1]
        plotted_buses = set()
        for agent in self.agent_names:
            if agent in unserved_reward_df.columns:
                bus = self.agent_name_bus_map[agent]
                # Only add legend entry for the first agent on each bus
                label = f"Bus {bus}" if bus not in plotted_buses else None
                ax_unserved.plot(unserved_reward_df.index, unserved_reward_df[agent], 
                               label=label, 
                               color=agent_color_list[agent])
                plotted_buses.add(bus)
        ax_unserved.set_title("Unserved Reward (per Agent)")
        ax_unserved.set_ylabel("Reward")
        ax_unserved.grid(True)
        ax_unserved.legend(loc='best', fontsize='small')
        ax_unserved.set_visible(True)  # Make sure this subplot is visible

        return fig