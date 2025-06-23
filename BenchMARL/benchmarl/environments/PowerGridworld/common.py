#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
import copy
from typing import Callable, Dict, List, Optional

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING

from tensordict import TensorDictBase

from torchrl.data import CompositeSpec
from torchrl.envs import EnvBase
from torchrl.envs.transforms import RewardSum, Transform
#from torchrl.envs.libs import YourTorchRLEnvConstructor

# PowerGridworld environment requirements
import numpy as np
import pandas as pd
from gridworld import ComponentEnv
from gridworld import MultiAgentEnv
from gridworld.distribution_system import OpenDSSSolver
from gridworld.agents.vehicles import EVChargingEnv
from mpl_toolkits.mplot3d import Axes3D

AGENT_CLASS_MAP = {
    "EVChargingEnv": EVChargingEnv,
    # Add other agent types here as needed
}

class PowerGridworldTask(Task):
    # Your task names.
    # Their config will be loaded from conf/task/PowerGridworld

    EVOVERNIGHT13NODE = None  # Loaded automatically from conf/task/PowerGridworld/evovernight13node

    @staticmethod
    def associated_class():
        return PowerGridworldClass


class PowerGridworldClass(TaskClass):
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        # Extract agent types and busses from config
        agent_types = config.get("agents", [])
        busses = config.get("busses", [])
        n_agents = len(agent_types)
        if not busses or len(busses) != n_agents:
            # Default to 13 busses if not specified or mismatch
            busses = ['634a', '634b', '634c', '645', '675a', '675b', '675c', '670a', '670b', '670c', '684c'][:n_agents]
            print(f"Using default busses: {busses} for {n_agents} agents.")

        # Build agent configs
        agents = []
        for i, (agent_type, bus) in enumerate(zip(agent_types, busses)):
            agent_cls = AGENT_CLASS_MAP.get(agent_type)
            if agent_cls is None:
                raise ValueError(f"Unknown agent type: {agent_type}")
            agents.append({
                "name": f"ev-charging-{i}",
                "bus": bus,
                "cls": agent_cls,  # You can map agent_type to class if needed
                "config": {
                    "num_vehicles": config.get("num_vehicles", 70),
                    "minutes_per_step": config.get("minutes_per_step", 15),
                    "max_charge_rate_kw": config.get("max_charge_rate_kw", 7.0),
                    "peak_threshold": config.get("peak_threshold", 700.0),
                    "vehicle_multiplier": config.get("vehicle_multiplier", 1.0),
                    "rescale_spaces": config.get("rescale_spaces", False),
                    "unserved_penalty": config.get("unserved_penalty", 0.0),
                }
            })

        # Common config
        common_config = {
            "start_time": config.get("start_time", "08-12-2020 20:00:00"),
            "end_time": config.get("end_time", "08-13-2020 08:00:00"),
            "control_timedelta": config.get("control_timedelta", 900),
        }

        # Power flow config
        pf_config = {
            "cls": OpenDSSSolver,
            "config": {
                "feeder_file": config.get("feeder_file", "ieee_13_dss/IEEE13Nodeckt.dss"),
                "loadshape_file": config.get("loadshape_file", "ieee_13_dss/annual_hourly_load_profile.csv"),
                "system_load_rescale_factor": config.get("system_load_rescale_factor", 0.7),
            }
        }

        # Compose the environment config
        env_config = {
            "common_config": common_config,
            "pf_config": pf_config,
            "agents": agents,
        }

        # Return a function that creates the environment
        return lambda: MultiAgentEnv(**env_config)
    
    def get_reward_sum_transform(self, env: EnvBase) -> Transform:
        """Define the reward sum transform with proper keys."""
        from torchrl.envs.transforms import RewardSum
    
        # Use flat keys for rewards
        return RewardSum(
            in_keys=[("agents", "reward")],
            out_keys=[("agents", "reward_sum")] # It's good practice to nest the output too
        )
    
    def _reward_spec(self):
        """Return the reward spec for the environment."""
        return self.reward_spec
        
    def supports_continuous_actions(self) -> bool:
        # Does the environment support continuous actions?
        return True

    def supports_discrete_actions(self) -> bool:
        # Does the environment support discrete actions?
        return False

    def has_render(self, env: EnvBase) -> bool:
        # Does the env have a env.render(mode="rgb_array") or env.render() function?
        return False

    def max_steps(self, env: EnvBase) -> int:
        # Maximum number of steps for a rollout during evaluation
        return 100

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        # The group map mapping group names to agent names
        # The data in the tensordict will havebe presented this way
        return {"agents": [agent.name for agent in env.agents]}

    def observation_spec(self, env: EnvBase) -> CompositeSpec:
        # A spec for the observation.
        # Must be a CompositeSpec with one (group_name, observation_key) entry per group.
        return env.full_observation_spec

    def action_spec(self, env: EnvBase) -> CompositeSpec:
        # A spec for the action.
        # If provided, must be a CompositeSpec with one (group_name, "action") entry per group.
        return env.full_action_spec

    def state_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        # A spec for the state.
        # If provided, must be a CompositeSpec with one "state" entry
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        # A spec for the action mask.
        # If provided, must be a CompositeSpec with one (group_name, "action_mask") entry per group.
        return None

    def info_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        # A spec for the info.
        # If provided, must be a CompositeSpec with one (group_name, "info") entry per group (this entry can be composite).
        return None

    @staticmethod
    def env_name() -> str:
        # The name of the environment in the benchmarl/conf/task folder
        return "PowerGridworld"

    def log_info(self, batch: TensorDictBase) -> Dict[str, float]:
        # Optionally return a str->float dict with extra things to log
        # This function has access to the collected batch and is optional
        return {}
