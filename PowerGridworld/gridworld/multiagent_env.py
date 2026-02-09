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

        # Get global reward configs
        self.power_loss_penalty = float(common_config.get("power_loss_penalty", 1e-5))
        if "power_loss_penalty" not in common_config:
            logger.info(f"Using default power_loss_penalty: {self.power_loss_penalty}")

        self.voltage_penalty = float(common_config.get("voltage_penalty", 1e3))
        if "voltage_penalty" not in common_config:
            logger.info(f"Using default voltage_penalty: {self.voltage_penalty}")

        self.load_2norm_penalty = float(common_config.get("load_2norm_penalty", 0.0))
        if "load_2norm_penalty" not in common_config:
            logger.info(f"Using default load_2norm_penalty: {self.load_2norm_penalty}")
        
        self.tracking_reward_penalty = float(common_config.get("tracking_reward_penalty", 1.0))
        if "tracking_reward_penalty" not in common_config:
            logger.info(f"Using default tracking_reward_penalty: {self.tracking_reward_penalty}")

        self.signal_tracking = common_config.get("signal_tracking", False)
        if "signal_tracking" not in common_config:
            logger.info(f"Using default signal_tracking: {self.signal_tracking}")
        
        self.track_total_load = common_config.get("track_total_load", False)
        if "track_total_load" not in common_config:
            logger.info(f"Using default track_total_load: {self.track_total_load}")
        
        self.setpoint = float(common_config.get("setpoint", 200.0))
        if "setpoint" not in common_config:
            logger.info(f"Using default setpoint: {self.setpoint}")

        self.vpp_reward = common_config.get("vpp_reward", False)
        if "vpp_reward" not in common_config:
            logger.info(f"Using default vpp_reward: {self.vpp_reward}")
        
        self.vpp_setpoint = float(common_config.get("vpp_setpoint", 0.0))
        if "vpp_setpoint" not in common_config:
            logger.info(f"Using default vpp_setpoint: {self.vpp_setpoint}")
        
        self.vpp_reward_penalty = float(common_config.get("vpp_reward_penalty", 1.0))
        if "vpp_reward_penalty" not in common_config:
            logger.info(f"Using default vpp_reward_penalty: {self.vpp_reward_penalty}")
            
        self.cooperative_voltage = common_config.get("cooperative_voltage", True)
        if "cooperative_voltage" not in common_config:
            logger.info(f"Using default cooperative_voltage: {self.cooperative_voltage}")
            
        self.include_load_in_agent_obs = common_config.get("include_load_in_agent_obs", True)
        if "include_load_in_agent_obs" not in common_config:
            logger.info(f"Using default include_load_in_agent_obs: {self.include_load_in_agent_obs}")

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
        # Observation dimension: base + voltage + setpoint + (optionally) base load
        obs_dim = self.agents[0].observation_space.shape[0] + 2  # +1 for voltage, +1 for setpoint
        if self.include_load_in_agent_obs:
            obs_dim += 1  # +1 for base load
        act_dim = self.agents[0].action_space.shape[0]

        # Get bounds from any agent (they are all the same)
        obs_low = torch.as_tensor(self.agents[0].observation_space.low, dtype=torch.float32)
        obs_high = torch.as_tensor(self.agents[0].observation_space.high, dtype=torch.float32)
        # Append 0 to obs_low and inf to obs_high for the voltage dimension
        obs_low = torch.cat([obs_low, torch.tensor([0.0], dtype=torch.float32)])
        obs_high = torch.cat([obs_high, torch.tensor([float('inf')], dtype=torch.float32)])
        # Append bounds for the setpoint dimension (can be any positive value)
        obs_low = torch.cat([obs_low, torch.tensor([0.0], dtype=torch.float32)])
        obs_high = torch.cat([obs_high, torch.tensor([float('inf')], dtype=torch.float32)])
        # Optionally append bounds for the normalized base load dimension
        if self.include_load_in_agent_obs:
            obs_low = torch.cat([obs_low, torch.tensor([0.0], dtype=torch.float32)])
            obs_high = torch.cat([obs_high, torch.tensor([1.0], dtype=torch.float32)])
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
                "load_2norm_penalty": Bounded(low=float("-inf"), high=float(0.0), shape=(), dtype=torch.float32),
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
        # logger.info("MultiAgentEnv has been closed.")
    
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

        # Pass the current simulation timestamp so time-of-day-aware agents
        # (e.g. PVEnv) can look up the correct profile value.
        kwargs["current_time"] = self.time

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
                    # EV-specific fields
                    if "energy_remaining" in agent_meta:
                        per_agent_info[agent_name]["energy_remaining"] = agent_meta["energy_remaining"]
                        if isinstance(agent_meta["energy_remaining"], (int, float)) and agent_meta["energy_remaining"] == 0:
                            warnings.warn(
                                f"[MultiAgentEnv] 'energy_remaining' for agent '{agent_name}' is 0 (default) at step {self.episode_step}.",
                                UserWarning,
                            )
                    if "peak_reward" in agent_meta:
                        per_agent_info[agent_name]["peak_reward"] = agent_meta["peak_reward"]
                    if "real_energy_unserved" in agent_meta:
                        per_agent_info[agent_name]["real_energy_unserved_reward"] = agent_meta["real_energy_unserved"]
                    # Storage-specific fields
                    if "state_of_charge" in agent_meta:
                        per_agent_info[agent_name]["state_of_charge"] = float(agent_meta["state_of_charge"])
                    # PV-specific fields
                    if "real_power" in agent_meta:
                        per_agent_info[agent_name]["pv_real_power"] = float(agent_meta["real_power"])
        # Also record each agent's real_power (for VPP / load plots)
        for agent in self.agents:
            if agent.name in per_agent_info:
                per_agent_info[agent.name]["real_power"] = float(agent.real_power)
                # EV agents expose max_real_power (needed for VPP curtailment calc)
                if hasattr(agent, 'max_real_power'):
                    per_agent_info[agent.name]["max_real_power"] = float(agent.max_real_power)

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
        # Get the current total base load (this is system-wide information)
        # Use the sum of real power from base load only (excluding controllable agents)
        total_base_load = float(self.pf_solver.normalized_load_coefficient)  # normalized load coefficient
        
        for agent_name in obs_dict:
            # Ensure the observation array is of type float32
            obs_dict[agent_name] = obs_dict[agent_name].astype(np.float32)

            # Get the bus name and get voltage using the power flow solver's built-in method
            bus_name = self.agent_name_bus_map[agent_name]
            voltage = self.pf_solver.get_bus_voltage_by_name(bus_name)
            
            # Handle both single-phase (float) and three-phase (list) voltages
            if isinstance(voltage, list):
                # For three-phase buses, use the average voltage
                voltage = float(np.mean(voltage))
            else:
                voltage = float(voltage)

            # Append the voltage to the observation
            obs_dict[agent_name] = np.append(obs_dict[agent_name], voltage).astype(np.float32)
            
            # Append the setpoint to the observation
            obs_dict[agent_name] = np.append(obs_dict[agent_name], self.setpoint).astype(np.float32)
            
            # Optionally append the total base load to the observation
            if self.include_load_in_agent_obs:
                obs_dict[agent_name] = np.append(obs_dict[agent_name], total_base_load).astype(np.float32)


        return obs_dict

    def reward_transform(self, rew_dict) -> dict:
        """Function to transform the agent rewards based on centralized view."""
        
        # Calculate total agent load (sum of all agent real power at current timestep)
        total_agent_load = float(sum([agent.real_power for agent in self.agents]))
        
        # Calculate base load (sum of all base load real power at current timestep)
        # Use per-node forecast coefficients if available, else fall back to scalar
        if hasattr(self.pf_solver, 'forecast_load_coefficients') and self.pf_solver.forecast_load_coefficients is not None:
            current_step_load = self.pf_solver.forecast_load_coefficients[:, np.newaxis] * \
                self.pf_solver.base_load * self.pf_solver.system_load_rescale_factor
        else:
            current_step_load = self.pf_solver.normalized_load_coefficient * self.pf_solver.base_load * \
                self.pf_solver.system_load_rescale_factor
        base_load = float(current_step_load[:, 0].sum())  # Sum of real power column

        # Tracking reward (quadratic penalty on normalized error)
        tracking_reward = 0.0
        tracking_error = 0.0
        tracking_error_norm = 0.0
        if self.signal_tracking:
            if self.track_total_load:
                tracking_error = (total_agent_load + base_load) - self.setpoint
            else:
                tracking_error = total_agent_load - self.setpoint
            # Normalize by setpoint magnitude (avoid div by zero)
            denom = max(abs(self.setpoint), 1.0)
            tracking_error_norm = tracking_error / denom
            # Quadratic (L2) penalty encourages matching setpoint while giving smooth gradient
            tracking_reward = -(tracking_error_norm ** 2) * self.tracking_reward_penalty

        # VPP (Virtual Power Plant) reward
        # Measures how well agents collectively hit a power production setpoint.
        # - PV / Energy Storage: VPP contribution = -real_power (net injection to grid)
        # - EV: VPP contribution = max_real_power - real_power (demand response)
        vpp_reward_value = 0.0
        total_vpp_production = 0.0
        if self.vpp_reward:
            for i, agent in enumerate(self.agents):
                # Skip inactive agents if using variable agent counts
                if hasattr(self, 'active_mask') and not self.active_mask[i]:
                    continue
                
                if hasattr(agent, 'max_real_power'):
                    # EV-type agent: demand response = max possible draw - actual draw
                    total_vpp_production += (agent.max_real_power - agent.real_power)
                else:
                    # PV / Storage: net power injection = -real_power
                    total_vpp_production += (-agent.real_power)
            
            # Quadratic penalty on normalized error from VPP setpoint
            vpp_error = total_vpp_production - self.vpp_setpoint
            vpp_denom = max(abs(self.vpp_setpoint), 1.0)
            vpp_error_norm = vpp_error / vpp_denom
            vpp_reward_value = -(vpp_error_norm ** 2) * self.vpp_reward_penalty
    
        # Calculate the power loss reward using instance variable
        power_loss_reward = -self.losses[0] * self.power_loss_penalty

        # Calculate voltage violation reward
        voltage_reward = 0
        violating_buses = {}
        # Check if any voltage is below 0.95 p.u.
        # If so, calculate the total voltage difference from 0.95 p.u.
        if np.any(np.array(list(self.voltages.values())) < 0.95):
            violating_buses = {b: 0.95 - v for b, v in self.voltages.items() if v < 0.95}
            total_voltage_difference = sum(violating_buses.values())
            voltage_reward = -total_voltage_difference * self.voltage_penalty
        
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
        load_2norm_penalty = -np.linalg.norm(total_load_array) * self.load_2norm_penalty

        # Store global reward components for group-level logging
        self._global_reward_components = {
            "power_loss_reward": float(power_loss_reward),
            "voltage_reward": float(voltage_reward),
            "load_2norm_penalty": float(load_2norm_penalty),
            "tracking_reward": float(tracking_reward),
            "vpp_reward": float(vpp_reward_value),
            "vpp_production": float(total_vpp_production),
        }

        # Add global rewards to each agent's reward individually
        for agent_name in rew_dict:
            if isinstance(rew_dict[agent_name], (int, float)):
                # Base global rewards (loss, load stability, tracking, VPP)
                rew_dict[agent_name] += 100 + power_loss_reward + load_2norm_penalty + tracking_reward + vpp_reward_value
                
                # Voltage Reward Logic
                if self.cooperative_voltage:
                    # Apply total system voltage penalty to everyone
                    rew_dict[agent_name] += voltage_reward
                else:
                    # Apply local voltage penalty only
                    # Use the map we created in __init__
                    bus_name = self.agent_name_bus_map.get(agent_name)
                    
                    if bus_name:
                         try:
                             v = self.pf_solver.get_bus_voltage_by_name(bus_name)
                             # Handle list (e.g. 3-phase bus)
                             if isinstance(v, (list, np.ndarray)):
                                 v = float(np.mean(v))
                             else:
                                 v = float(v)
                                 
                             if v < 0.95:
                                 local_penalty = -(0.95 - v) * self.voltage_penalty
                                 rew_dict[agent_name] += local_penalty
                         except Exception as e:
                             # Fallback or silent ignore if bus not found in solver
                             pass
                    # If bus has no violation, or bus unknown, 0 penalty added
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
        Dispatches to render_rollout_fig_vpp() when VPP reward is active.
        """
        # VPP dispatch — returns a list of figures instead of a single figure
        if getattr(self, "vpp_reward", False):
            return self.render_rollout_fig_vpp()

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed, skipping plot generation.")
            return

        if not self.history or not self.history["timestamp"]:
            logger.info("No history to plot, skipping render.")
            return

        # Determine active agents
        active_agents = self.agent_names
        if hasattr(self, "active_mask") and self.active_mask is not None:
             try:
                 mask = self.active_mask.cpu().numpy() if hasattr(self.active_mask, "cpu") else self.active_mask
                 mask = mask.astype(bool)
                 if len(mask) == len(self.agent_names):
                     active_agents = [name for i, name in enumerate(self.agent_names) if mask[i]]
             except Exception as e:
                 logger.warning(f"Could not filter active agents in render_rollout_fig: {e}")

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
            for agent in active_agents:
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
            for agent in active_agents:
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
        fig, axes = plt.subplots(4, 2, figsize=(20, 24), tight_layout=True)
        fig.suptitle(f"Evaluation Rollout (N_agents={len(active_agents)})", fontsize=16)
        
        # Plot 1: Agent Actions
        ax_actions = axes[0, 0]
        plotted_buses = set()
        for agent in active_agents:
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
        ax_actions.set_ylim(0, 1)  # Fixed y-axis scale for actions
        ax_actions.grid(True)
        ax_actions.legend(loc='best', fontsize='small')

        # Plot 2: Nodal Voltages
        ax_voltages = axes[0, 1]
        
        # Create a mapping from voltage bus names to agent buses using the same logic as get_bus_voltage_by_name
        agent_voltage_bus_map = {}
        for agent_name in active_agents:
            agent_bus = self.agent_name_bus_map[agent_name]
            try:
                # Use the same conversion logic as get_bus_voltage_by_name
                PHASE_MAP = {'a': '.1', 'b': '.2', 'c': '.3'}
                
                # Handle single-phase with letter notation (e.g., "634a")
                if agent_bus[-1] in PHASE_MAP.keys():
                    converted_bus_name = agent_bus.replace(agent_bus[-1], PHASE_MAP[agent_bus[-1]])
                    if converted_bus_name in voltages_df.columns:
                        agent_voltage_bus_map[converted_bus_name] = agent_bus
                else:
                    # Check for all possible phases and collect them
                    for phase_ext in ['.1', '.2', '.3']:
                        test_name = agent_bus + phase_ext
                        if test_name in voltages_df.columns:
                            agent_voltage_bus_map[test_name] = agent_bus
                            break  # Only map the first matching phase for this agent
            except:
                # If there's an error, skip this agent
                continue
        
        # Plot buses with agents (colored and labeled)
        plotted_agent_buses = set()
        for bus in voltages_df.columns:
            if bus in agent_voltage_bus_map:
                # This bus has agents connected - use color and add to legend
                agent_bus = agent_voltage_bus_map[bus]
                
                # Find the agent and get its color
                agent_name = None
                for name, mapped_bus in self.agent_name_bus_map.items():
                    if mapped_bus == agent_bus:
                        agent_name = name
                        break
                
                if agent_name:
                    color = agent_color_list[agent_name]
                    # Only add legend entry once per agent bus
                    label = f"Bus {agent_bus}" if agent_bus not in plotted_agent_buses else None
                    ax_voltages.plot(voltages_df.index, voltages_df[bus], 
                                   label=label, color=color, linewidth=2)
                    plotted_agent_buses.add(agent_bus)
                else:
                    # Fallback to gray if we can't find the agent
                    ax_voltages.plot(voltages_df.index, voltages_df[bus], 
                                   color='lightgray', alpha=0.6, linewidth=1)
            else:
                # This bus has no agents - make it gray with no legend
                ax_voltages.plot(voltages_df.index, voltages_df[bus], 
                               color='lightgray', alpha=0.6, linewidth=1)
        
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
        for agent in active_agents:
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
        for agent in active_agents:
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

    def render_rollout_fig_vpp(self):
        """
        VPP-specific evaluation plots.  Returns a list of two figures:

        Figure 1 (3×2 grid):
          Row 0: Nodal Voltages | EV Actions
          Row 1: PV Actions     | Storage Actions
          Row 2: System Load + VPP Tracking | Rewards per Agent

        Figure 2 (2×2 grid):
          Row 0: Remaining Energy Need per EV | Storage State of Charge
          Row 1: PV Real Power Output          | EV Curtailment (VPP Contribution)
        """
        import matplotlib.pyplot as plt

        if not self.history or not self.history["timestamp"]:
            logger.info("No history to plot, skipping VPP render.")
            return []

        # ---- Active agents ----
        active_agents = self.agent_names
        if hasattr(self, "active_mask") and self.active_mask is not None:
            try:
                mask = self.active_mask.cpu().numpy() if hasattr(self.active_mask, "cpu") else self.active_mask
                mask = mask.astype(bool)
                if len(mask) == len(self.agent_names):
                    active_agents = [name for i, name in enumerate(self.agent_names) if mask[i]]
            except Exception as e:
                logger.warning(f"Could not filter active agents in render_rollout_fig_vpp: {e}")

        # Partition agents by type
        ev_agents = [a for a in active_agents if a.startswith("EV-")]
        pv_agents = [a for a in active_agents if a.startswith("PV-")]
        storage_agents = [a for a in active_agents if a.startswith("Storage-")]

        # ---- Timestamps ----
        timestamps = pd.to_datetime(self.history["timestamp"])

        # ---- Color mapping (one color per bus) ----
        color_map = plt.get_cmap('tab10')
        unique_buses = list(set(self.agent_name_bus_map.values()))
        bus_color_map = {bus: color_map(i % 10) for i, bus in enumerate(unique_buses)}
        agent_color = {agent: bus_color_map[self.agent_name_bus_map[agent]] for agent in self.agent_names}

        # ---- Data extraction ----
        # Actions (scalar per agent per step)
        actions_df = pd.DataFrame(
            [{agent: a[0] for agent, a in step.items()} for step in self.history["actions"]],
            index=timestamps)

        # Voltages
        voltages_df = pd.DataFrame(self.history["voltage"], index=timestamps)

        # Per-agent real_power time series
        agent_real_power = {a: [] for a in active_agents}
        # EV energy remaining, Storage SoC, PV real power, EV curtailment
        ev_energy_records = []
        ev_curtailment = {a: [] for a in ev_agents}  # max_real_power - real_power
        storage_soc = {a: [] for a in storage_agents}
        pv_power = {a: [] for a in pv_agents}

        for t, step_info in enumerate(self.history["per_agent_info"]):
            for agent in active_agents:
                info = step_info.get(agent, {})
                agent_real_power[agent].append(info.get("real_power", 0.0))
            for agent in ev_agents:
                info = step_info.get(agent, {})
                er = info.get("energy_remaining", {})
                if isinstance(er, dict):
                    for veh_id, val in er.items():
                        ev_energy_records.append({
                            "timestamp": timestamps[t], "agent": agent,
                            "vehicle": veh_id, "energy_remaining": val})
                # EV VPP curtailment = max possible draw − actual draw
                max_rp = info.get("max_real_power", 0.0)
                rp = info.get("real_power", 0.0)
                ev_curtailment[agent].append(max_rp - rp)
            for agent in storage_agents:
                info = step_info.get(agent, {})
                storage_soc[agent].append(info.get("state_of_charge", np.nan))
            for agent in pv_agents:
                info = step_info.get(agent, {})
                pv_power[agent].append(info.get("pv_real_power", 0.0))

        # Total load & aggregate agent impact (VPP production)
        total_load = [l[1][:, 0].sum() for l in self.history["total_load"]]
        vpp_production = [rc.get("vpp_production", 0.0) for rc in self.history["reward_components"]]
        vpp_setpoint_val = getattr(self, "vpp_setpoint", 0.0)

        # Agent rewards
        agent_reward_data = []
        for step_rewards in self.history["agent_rewards"]:
            agent_reward_data.append({a: step_rewards.get(a, 0) for a in active_agents})
        agent_reward_df = pd.DataFrame(agent_reward_data, index=timestamps)

        # Reward components
        reward_comp_df = pd.DataFrame(self.history["reward_components"], index=timestamps)

        # ---- Helper: voltage bus mapping ----
        PHASE_MAP = {'a': '.1', 'b': '.2', 'c': '.3'}
        agent_voltage_bus_map = {}
        for agent_name in active_agents:
            agent_bus = self.agent_name_bus_map[agent_name]
            try:
                if agent_bus[-1] in PHASE_MAP:
                    converted = agent_bus[:-1] + PHASE_MAP[agent_bus[-1]]
                    if converted in voltages_df.columns:
                        agent_voltage_bus_map[converted] = agent_bus
                else:
                    for ext in ['.1', '.2', '.3']:
                        if agent_bus + ext in voltages_df.columns:
                            agent_voltage_bus_map[agent_bus + ext] = agent_bus
                            break
            except Exception:
                continue

        # ==================================================================
        # FIGURE 1: System-level view  (3 rows × 2 cols)
        # ==================================================================
        fig1, axes1 = plt.subplots(3, 2, figsize=(20, 18), tight_layout=True)
        fig1.suptitle(f"VPP Evaluation – System View (N_agents={len(active_agents)})", fontsize=16)

        # (0,0) Nodal Voltages
        ax = axes1[0, 0]
        plotted = set()
        for bus in voltages_df.columns:
            if bus in agent_voltage_bus_map:
                ab = agent_voltage_bus_map[bus]
                name = next((n for n, b in self.agent_name_bus_map.items() if b == ab), None)
                if name:
                    label = f"Bus {ab}" if ab not in plotted else None
                    ax.plot(voltages_df.index, voltages_df[bus], label=label,
                            color=agent_color.get(name, 'gray'), linewidth=2)
                    plotted.add(ab)
                else:
                    ax.plot(voltages_df.index, voltages_df[bus], color='lightgray', alpha=0.5, linewidth=1)
            else:
                ax.plot(voltages_df.index, voltages_df[bus], color='lightgray', alpha=0.5, linewidth=1)
        ax.set_title("Nodal Voltages")
        ax.set_ylabel("Voltage (p.u.)")
        ax.axhline(0.95, color='red', linestyle='--', alpha=0.5, label='0.95 p.u.')
        ax.grid(True); ax.legend(loc='best', fontsize='small')

        # (0,1) EV Actions
        ax = axes1[0, 1]
        plotted = set()
        for agent in ev_agents:
            if agent in actions_df.columns:
                bus = self.agent_name_bus_map[agent]
                label = f"{agent} ({bus})" if agent not in plotted else None
                ax.plot(actions_df.index, actions_df[agent], label=label, color=agent_color[agent])
                plotted.add(agent)
        ax.set_title("EV Charging Actions")
        ax.set_ylabel("Action [0-1]"); ax.set_ylim(-0.05, 1.05)
        ax.grid(True); ax.legend(loc='best', fontsize='small')

        # (1,0) PV Actions
        ax = axes1[1, 0]
        plotted = set()
        for agent in pv_agents:
            if agent in actions_df.columns:
                bus = self.agent_name_bus_map[agent]
                label = f"{agent} ({bus})" if agent not in plotted else None
                ax.plot(actions_df.index, actions_df[agent], label=label, color=agent_color[agent])
                plotted.add(agent)
        ax.set_title("PV Curtailment Actions")
        ax.set_ylabel("Action [0-1]"); ax.set_ylim(-0.05, 1.05)
        ax.grid(True); ax.legend(loc='best', fontsize='small')

        # (1,1) Storage Actions
        ax = axes1[1, 1]
        plotted = set()
        for agent in storage_agents:
            if agent in actions_df.columns:
                bus = self.agent_name_bus_map[agent]
                label = f"{agent} ({bus})" if agent not in plotted else None
                ax.plot(actions_df.index, actions_df[agent], label=label, color=agent_color[agent])
                plotted.add(agent)
        ax.set_title("Storage Actions")
        ax.set_ylabel("Action [-1, 1]"); ax.set_ylim(-1.05, 1.05)
        ax.grid(True); ax.legend(loc='best', fontsize='small')

        # (2,0) System Load + VPP tracking
        ax = axes1[2, 0]
        ax.plot(timestamps, total_load, label="Total Load (kW)", color='black')
        ax.plot(timestamps, vpp_production, label="VPP Production (kW)", color='tab:green', linewidth=2)
        ax.axhline(vpp_setpoint_val, color='tab:green', linestyle='--', alpha=0.7,
                    label=f"VPP Setpoint ({vpp_setpoint_val:.0f} kW)")
        ax.set_title("System Load & VPP Tracking")
        ax.set_ylabel("Power (kW)")
        ax.grid(True); ax.legend(loc='best', fontsize='small')

        # (2,1) Rewards per Agent
        ax = axes1[2, 1]
        plotted = set()
        for agent in active_agents:
            if agent in agent_reward_df.columns:
                bus = self.agent_name_bus_map[agent]
                label = f"{agent} ({bus})" if agent not in plotted else None
                ax.plot(agent_reward_df.index, agent_reward_df[agent],
                        label=label, color=agent_color[agent])
                plotted.add(agent)
        ax.set_title("Agent Rewards")
        ax.set_ylabel("Reward")
        ax.grid(True); ax.legend(loc='best', fontsize='small')

        # ==================================================================
        # FIGURE 2: Agent-detail view  (2 rows × 2 cols)
        # ==================================================================
        fig2, axes2 = plt.subplots(2, 2, figsize=(20, 12), tight_layout=True)
        fig2.suptitle("VPP Evaluation – Agent Detail", fontsize=16)

        # (0,0) Remaining Energy Need per EV
        ax = axes2[0, 0]
        if ev_energy_records:
            er_long = pd.DataFrame(ev_energy_records)
            er_pivot = er_long.pivot_table(index="timestamp", columns=["agent", "vehicle"],
                                           values="energy_remaining")
            plotted = set()
            for (agent, vehicle) in er_pivot.columns:
                bus = self.agent_name_bus_map[agent]
                label = f"{agent} ({bus})" if agent not in plotted else None
                ax.plot(er_pivot.index, er_pivot[(agent, vehicle)],
                        label=label, color=agent_color[agent], alpha=0.7)
                plotted.add(agent)
        ax.set_title("Remaining Energy Need per EV")
        ax.set_ylabel("Energy (kWh)")
        ax.grid(True); ax.legend(loc='best', fontsize='small')

        # (0,1) Storage State of Charge
        ax = axes2[0, 1]
        for agent in storage_agents:
            bus = self.agent_name_bus_map[agent]
            ax.plot(timestamps, storage_soc[agent],
                    label=f"{agent} ({bus})", color=agent_color[agent])
        ax.set_title("Energy Storage – State of Charge")
        ax.set_ylabel("SoC (kWh)")
        ax.grid(True); ax.legend(loc='best', fontsize='small')

        # (1,0) PV Real Power Output
        ax = axes2[1, 0]
        for agent in pv_agents:
            bus = self.agent_name_bus_map[agent]
            # PV real_power is ≤0 (generation). Plot magnitude for clarity.
            vals = [-v for v in pv_power[agent]]
            ax.plot(timestamps, vals,
                    label=f"{agent} ({bus})", color=agent_color[agent])
        ax.set_title("PV Real Power Output")
        ax.set_ylabel("Generation (kW)")
        ax.grid(True); ax.legend(loc='best', fontsize='small')

        # (1,1) EV Curtailment (VPP contribution = max_draw − actual_draw)
        ax = axes2[1, 1]
        for agent in ev_agents:
            bus = self.agent_name_bus_map[agent]
            ax.plot(timestamps, ev_curtailment[agent],
                    label=f"{agent} ({bus})", color=agent_color[agent])
        ax.set_title("EV Curtailment (VPP Contribution)")
        ax.set_ylabel("Power (kW)")
        ax.grid(True); ax.legend(loc='best', fontsize='small')

        return [fig1, fig2]