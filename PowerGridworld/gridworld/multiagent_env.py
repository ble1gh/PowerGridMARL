from abc import abstractmethod
from collections import defaultdict

import logging
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd

from gridworld.base import ComponentEnv, MultiComponentEnv
from gridworld.log import logger

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv as Env
except ImportError:
    Env = object
    logger.warning("rllib MultiAgentEnv not found, using generic object class")

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


class MultiAgentEnv(Env):
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
        self.observation_space = {
            agent.name: Box(
                low=np.concatenate([agent.observation_space.low, [0]]),
                high=np.concatenate([agent.observation_space.high, [np.inf]]),
                dtype=np.float32
            )
            for agent in self.agents
        }
        self.action_space = {
            agent.name: agent.action_space for agent in self.agents}

    def close(self):
        """Clean up resources used by the environment."""
        # Close the power flow solver if it has a close method
        if hasattr(self.pf_solver, "close"):
            self.pf_solver.close()

        # Close all agents if they have a close method
        for agent in self.agents:
            if hasattr(agent, "close"):
                agent.close()

        # Log that the environment has been closed
        logger.info("MultiAgentEnv has been closed.")

    @abstractmethod
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


    def reset(self, seed=None, options=None, **kwargs) -> Tuple[Dict[str, any], dict]:
        """Reset the environment and return the initial observations for all agents."""
        self.episode_step = 0
        self.time = self.start_time
        self.history = {"timestamp": [], "voltage": [], "agent_power_p": [], "base_load": [], "losses": []}
        self.episode_reward = 0

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
        return obs, {}


    def get_obs(self) -> Dict[str, any]:
        obs = {}
        for agent in self.agents:
            kwargs = self.get_external_obs_vars(agent, seed=None)
            obs[agent.name], _ = agent.get_obs(**kwargs)
        return obs


    def step(self, action: Dict[str, any]) -> Tuple[dict, dict, dict, dict]:
        self.episode_step += 1
        self.time += self.control_timedelta
        self.obs_dict = {}

        # Initialize agent outputs.
        obs, rew, done, meta = {}, {}, {}, {}
        load_p, load_q = {}, {}
        agent_power_p = []

        # For each agent, call the step method and inject any external variables
        # as keyword arguments.  Accumulate the real/reactive power from each
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

        # Check for terminal condition.  Currently, we stop the entire simulation
        # if any agent is `done`, although the RLLib API allows agents to finish
        # at different times.
        any_done = np.any(list(done.values()))  # will fail otherwise
        max_steps_reached = (self.episode_step == self.max_episode_steps - 1)
        time_up = self.time >= self.end_time
        done = any_done or max_steps_reached or time_up
        # print(f"Episode step: {self.episode_step}, Done: {done}, Time up: {time_up}, Max steps reached: {max_steps_reached}")

        # Create the dones dict that will be returned.  We assume all are done
        # or not simulataneously.
        dones = {a.name: done for a in self.agents}
        dones["__all__"] = done

        # Transform rewards and meta
        rew = self.reward_transform(rew)
        meta = self.meta_transform(meta)
        obs = self.obs_transform(obs)

        for agent_name in rew:
            if "episode" not in meta[agent_name]:
                meta[agent_name]["episode"] = {"r": 0, "l": 0}
            meta[agent_name]["episode"]["r"] += rew[agent_name]
            meta[agent_name]["episode"]["l"] = self.episode_step

        # Add final observations for truncated episodes
        if done or max_steps_reached or time_up:
            # meta["final_observation"] = obs
            meta["final_observation"] = obs
            # Calculate the total episodic reward and length
            total_reward = sum(rew.values())
            episode_length = self.episode_step

            # Populate meta["final_info"]
            meta["final_info"] = {
                "episode": {
                    "r": self.episode_reward,  # Total episodic reward
                    "l": episode_length,  # Episode length
                }
            }
            #print(f"Episode finished. Total reward: {self.episode_reward}, Length: {episode_length}")

        truncated = False

        return obs, rew, dones, truncated, meta

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
        """Function to transform the agent rewards based on centralized view.
        Pass-through by default."""
        
        # Calculate the power loss reward
        power_loss_reward = -self.losses[0]/1e5
        # logger.info(f"Power loss reward: {power_loss_reward}")

        # Calculate voltage violation reward
        voltage_reward = 0
        # Check if any voltage is below 0.95 p.u.
        # If so, calculate the total voltage difference from 0.95 p.u.
        if np.any(np.array(list(self.voltages.values())) < 0.95):
            voltage_differences = [0.95 - v for v in self.voltages.values() if v < 0.95]
            total_voltage_difference = sum(voltage_differences)
            voltage_reward = -total_voltage_difference*1e3
        
        # Assuming `output` is the variable containing the data
        total_load = []
        for item in self.history['base_load'][:]:
            bus_names, data_array = item  # Unpack the tuple
            load = data_array[:, 0].sum()  # Extract the first column
            total_load.append(load)

        # Convert the list of arrays to a 2D numpy array
        total_load_array = np.array(total_load)
        
        # Load stability reward
        # print(f"base_load: {self.history['base_load'][:]}")
        load_penalty = -np.linalg.norm(total_load_array)/1e5
        # print(f"Load penalty: {load_penalty}")
        
        # print(f"Voltage reward: {voltage_reward}")

        # Add power_loss_reward to all agent rewards
        for agent_name in rew_dict:
            if isinstance(rew_dict[agent_name], (int, float)):
                rew_dict[agent_name] += power_loss_reward + voltage_reward + load_penalty
            else:
                # Alert if the reward is not a number
                logger.warning(f"Reward for agent {agent_name} is not a number: {rew_dict[agent_name]}")

        # Add up total reward for all agents
        reward_all_vehicles = sum(value for value in rew_dict.values() if isinstance(value, (int, float)))

        self.episode_reward += reward_all_vehicles
        # logger.info(f"Episode reward: {self.episode_reward}")
        # logger.info(f"Agent rewards: {rew_dict}")
        return rew_dict


    def meta_transform(self, meta) -> dict:
        """Function to augment the agent meta info based on centralized view.
        Pass-through by default.
        """
        return meta


    @property
    def agent_dict(self) -> Dict[str, ComponentEnv]:
        return {a.name: a for a in self.agents}
