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
from gridworld.agents.pv import PVEnv
from gridworld.agents.energy_storage import EnergyStorageEnv
from mpl_toolkits.mplot3d import Axes3D

AGENT_CLASS_MAP = {
    "EVChargingEnv": EVChargingEnv,
    "PVEnv": PVEnv,
    "EnergyStorageEnv": EnergyStorageEnv,
}

class PowerGridworldTask(Task):
    # Your task names.
    # Their config will be loaded from conf/task/PowerGridworld

    EVOVERNIGHT13NODE = None  # Loaded automatically from conf/task/PowerGridworld/evovernight13node
    EVOVERNIGHT13NODE_SIMPLE = None  # Loaded automatically from conf/task/PowerGridworld/evovernight13node_simple

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
        
        # Get agent counts per node
        EVs_per_node = config.get("EVs_per_node", 0)
        PVs_per_node = config.get("PVs_per_node", 0)
        Storage_per_node = config.get("Storage_per_node", 0)
        
        # Get buses for each agent type
        EV_busses = config.get("EV_busses", [])
        PV_busses = config.get("PV_busses", [])
        Storage_busses = config.get("Storage_busses", [])
        
        # Build agent configs for each type
        agents = []
        
        # Create EV agents
        if EVs_per_node > 0 and EV_busses:
            for bus in EV_busses:
                for copy_num in range(1, EVs_per_node + 1):
                    agent_name = f"EV-{bus}-{copy_num}"
                    agents.append({
                        "name": agent_name,
                        "bus": bus,
                        "cls": EVChargingEnv,
                        "config": {
                            "num_vehicles": config.get("num_vehicles", 1),
                            "minutes_per_step": config.get("minutes_per_step", 15),
                            "max_charge_rate_kw": config.get("max_charge_rate_kw", 7.0),
                            "peak_threshold": config.get("peak_threshold", 700.0),
                            "vehicle_multiplier": config.get("vehicle_multiplier", 1.0),
                            "rescale_spaces": config.get("rescale_spaces", False),
                            "unserved_penalty": config.get("unserved_penalty", 0.0),
                        }
                    })
        
        # Create PV agents
        if PVs_per_node > 0 and PV_busses:
            for bus in PV_busses:
                for copy_num in range(1, PVs_per_node + 1):
                    agent_name = f"PV-{bus}-{copy_num}"
                    agents.append({
                        "name": agent_name,
                        "bus": bus,
                        "cls": PVEnv,
                        "config": {
                            "profile_csv": config.get("pv_profile_csv", "pv_profile.csv"),
                            "scaling_factor": config.get("pv_scaling_factor", 1.0),
                            "rescale_spaces": config.get("rescale_spaces", False),
                            "grid_aware": config.get("pv_grid_aware", False),
                        }
                    })
        
        # Create Energy Storage agents
        if Storage_per_node > 0 and Storage_busses:
            for bus in Storage_busses:
                for copy_num in range(1, Storage_per_node + 1):
                    agent_name = f"Storage-{bus}-{copy_num}"
                    agents.append({
                        "name": agent_name,
                        "bus": bus,
                        "cls": EnergyStorageEnv,
                        "config": {
                            "storage_range": (config.get("storage_range_min", 3.0), config.get("storage_range_max", 50.0)),
                            "initial_storage_mean": config.get("initial_storage_mean", 30.0),
                            "initial_storage_std": config.get("initial_storage_std", 5.0),
                            "charge_efficiency": config.get("charge_efficiency", 0.95),
                            "discharge_efficiency": config.get("discharge_efficiency", 0.9),
                            "max_power": config.get("max_power", 15.0),
                            "rescale_spaces": config.get("rescale_spaces", False),
                        }
                    })

        print(f"Created {len(agents)} agents:")
        print(f"  - {len(EV_busses) * EVs_per_node} EV agents ({EVs_per_node} per node)")
        print(f"  - {len(PV_busses) * PVs_per_node} PV agents ({PVs_per_node} per node)")
        print(f"  - {len(Storage_busses) * Storage_per_node} Storage agents ({Storage_per_node} per node)")

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
        if hasattr(env, "group_map"):
            return env.group_map
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
