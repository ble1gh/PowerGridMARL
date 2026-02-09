#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
import copy
from typing import Callable, Dict, List, Optional

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING

import torch
from tensordict import TensorDictBase

from torchrl.data import CompositeSpec
from torchrl.envs import EnvBase
from torchrl.envs.transforms import RewardSum, Transform
#from torchrl.envs.libs import YourTorchRLEnvConstructor

# PowerGridworldGraph environment requirements
import numpy as np
import pandas as pd
from gridworld import ComponentEnv
from gridworld import MultiAgentEnv
from gridworld.distribution_system import OpenDSSSolver
from gridworld.agents.vehicles import EVChargingEnv
from gridworld.agents.pv import PVEnv
from gridworld.agents.energy_storage import EnergyStorageEnv
from mpl_toolkits.mplot3d import Axes3D

class PaddedMultiAgentEnv(MultiAgentEnv):
    """
    A wrapper around MultiAgentEnv that handles variable agent counts via padding.
    The environment is initialized with the MAXIMUM number of agents.
    On reset, a random subset of agents is activated.
    Inactive agents have their observations and rewards zeroed out.
    """
    def __init__(self, *args, variable_agent_count=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.variable_agent_count = variable_agent_count
        self.max_agents = len(self.agents)
        self.active_mask = torch.ones(self.max_agents, dtype=torch.bool)
        
        # Initialize Graph Structure from Solver
        self._init_graph_structure()
        
    def _init_graph_structure(self):
        # Access the solver from the internal env (MultiAgentEnv has self.pf_solver)
        if hasattr(self, 'pf_solver'):
             # 1. Get detailed connectivity from OpenDSS
             self.grid_nodes, self.adj_dict = self.pf_solver.get_bus_connectivity()
             
             # 2. Convert to torch tensors
             self.adj_tensors = {}
             # Expected keys from opendss.py: 'line', 'transformer', 'switch'
             for k, v in self.adj_dict.items():
                 # Shape: (N_grid, N_grid, F)
                 self.adj_tensors[f"{k}_adjacency"] = torch.from_numpy(v).float()
                 
             # 3. Compute Agent <-> Grid Node mapping
             self.agent_grid_edge_index = self._compute_agent_grid_mapping()
             
             # Store grid node count for reference
             self.n_grid_nodes = len(self.grid_nodes)

             # 4. Update Observation Spec to include static graph info
             # This ensures TorchRL knows about these keys
             if self.agent_grid_edge_index is not None:
                from torchrl.data import Unbounded
                self.observation_spec.set("agent_grid_edge_index", Unbounded(shape=self.agent_grid_edge_index.shape, dtype=torch.long))
                # Add grid node features spec
                # Shape: (N_grid_nodes, 2) [kW, kvar]
                self.observation_spec.set("grid_node_features", Unbounded(shape=(self.n_grid_nodes, 2), dtype=torch.float32))
                # Add participation score spec for agent node features in GNN
                # Shape: (n_agents, 1) - one score per agent
                self.observation_spec.set("participation_score", Unbounded(shape=(self.max_agents, 1), dtype=torch.float32))

        else:
             print("Warning: pf_solver not found in PaddedMultiAgentEnv during init")

    def _get_current_grid_features(self):
        if not hasattr(self, 'pf_solver'):
            return None
        
        # Get load dict
        try:
             load_map = self.pf_solver.get_nodal_load_forecast(self.time)
        except AttributeError:
             # Fallback if method not on solver (e.g. diff solver class)
             return torch.zeros(self.n_grid_nodes, 2, dtype=torch.float32)
        
        features = torch.zeros(self.n_grid_nodes, 2, dtype=torch.float32)
        
        if not hasattr(self, '_grid_node_map'):
             self._grid_node_map = {name: i for i, name in enumerate(self.grid_nodes)}
             
        for bus, load_vec in load_map.items():
            if bus in self._grid_node_map:
                idx = self._grid_node_map[bus]
                features[idx] = torch.from_numpy(load_vec).float()
            else:
                # Try matching phase 1 for 3-phase loads defined on bus root
                # This is a simplification but puts the load *somewhere* in the graph
                for p in ['.1', '.2', '.3', '.a', '.b', '.c']:
                    if f"{bus}{p}" in self._grid_node_map:
                         idx = self._grid_node_map[f"{bus}{p}"]
                         features[idx] = torch.from_numpy(load_vec).float()
                         break
        return features

    def _get_participation_scores(self):
        """Get participation scores for all agents.
        
        For EV agents: returns the initial charge-to-go captured at connection time.
        For other agents: returns 0.0 (can be extended for other agent types).
        
        The participation score is meant to represent 'task difficulty' and remains
        constant from connection until disconnection.
        """
        scores = torch.zeros(self.max_agents, 1, dtype=torch.float32)
        for i, agent in enumerate(self.agents):
            if hasattr(agent, 'participation_score'):
                scores[i, 0] = agent.participation_score
        return scores


    def _compute_agent_grid_mapping(self):
        if not hasattr(self, 'grid_nodes'): return None
        
        node_map = {name: i for i, name in enumerate(self.grid_nodes)}
        edges_src = [] # Agent index
        edges_dst = [] # Grid Node index
        
        for agent_idx, agent in enumerate(self.agents):
            bus = None
            if hasattr(agent, 'bus'):
                bus = agent.bus
            elif hasattr(self, 'agent_name_bus_map') and agent.name in self.agent_name_bus_map:
                bus = self.agent_name_bus_map[agent.name]
            
            if bus is None:
                print(f"Warning: Could not find bus for agent {agent.name}")
                continue
            
            # Logic to match agent bus string to grid nodes
            # Agent bus: '634' or '634.1'
            # Grid nodes: '634.1', '634.2', '634.3'
            
            parts = bus.split('.')
            base = parts[0]
            
            targets = []
            for node_name in self.grid_nodes:
                 n_parts = node_name.split('.')
                 # Check base name match
                 if n_parts[0] == base:
                     # If agent specified phase, require exact match
                     if len(parts) > 1:
                         if node_name == bus:
                             targets.append(node_map[node_name])
                     else:
                         # Connect to all phases of that bus
                         targets.append(node_map[node_name])
            
            for t in targets:
                edges_src.append(agent_idx)
                edges_dst.append(t)
                
        if not edges_src:
            return torch.empty(2, 0, dtype=torch.long)
            
        return torch.stack([torch.tensor(edges_src), torch.tensor(edges_dst)], dim=0).long()

    def _reset(self, tensordict=None, **kwargs):
        if self.variable_agent_count:
            # Randomly select active agents (at least 1)
            n_active = np.random.randint(1, self.max_agents + 1)
            active_indices = np.random.choice(self.max_agents, n_active, replace=False)
            self.active_mask = torch.zeros(self.max_agents, dtype=torch.bool)
            self.active_mask[active_indices] = True
        else:
            self.active_mask = torch.ones(self.max_agents, dtype=torch.bool)
            
        out = super()._reset(tensordict, **kwargs)
        
        # Zero out observations for inactive agents
        if self.variable_agent_count:
            obs = out.get(("agents", "observation"))
            if obs is not None:
                obs[~self.active_mask] = 0.0
                out.set(("agents", "observation"), obs)
                
        # Inject Graph Data into TensorDict
        if hasattr(self, 'adj_tensors'):
            for k, v in self.adj_tensors.items():
                out.set(k, v)
        
        if hasattr(self, 'agent_grid_edge_index') and self.agent_grid_edge_index is not None:
            out.set("agent_grid_edge_index", self.agent_grid_edge_index)
        
        # Add grid node features
        grid_feats = self._get_current_grid_features()
        if grid_feats is not None:
            out.set("grid_node_features", grid_feats)
        
        # Add participation scores for agent node features in GNN
        participation_scores = self._get_participation_scores()
        out.set("participation_score", participation_scores)
                
        return out

    def _step(self, tensordict):
        out = super()._step(tensordict)
        
        if self.variable_agent_count:
            # Zero out next observations
            obs = out.get(("next", "agents", "observation"))
            if obs is not None:
                obs[~self.active_mask] = 0.0
                out.set(("next", "agents", "observation"), obs)
                
            # Zero out rewards
            reward = out.get(("next", "agents", "reward"))
            if reward is not None:
                reward[~self.active_mask] = 0.0
                out.set(("next", "agents", "reward"), reward)
        
        # Inject graph data for next observation
        if hasattr(self, 'agent_grid_edge_index') and self.agent_grid_edge_index is not None:
            out.set("agent_grid_edge_index", self.agent_grid_edge_index)
        
        # Add grid node features
        grid_feats = self._get_current_grid_features()
        if grid_feats is not None:
            out.set("grid_node_features", grid_feats)
        
        # Add participation scores for agent node features in GNN
        participation_scores = self._get_participation_scores()
        out.set("participation_score", participation_scores)
                
        return out

class PowerGridworldGraphTask(Task):
    # Your task names.
    # Their config will be loaded from conf/task/PowerGridworldGraph

    EVOVERNIGHT13NODE = None  # Loaded automatically from conf/task/PowerGridworldGraph/evovernight13node
    EVOVERNIGHT13NODE_SIMPLE = None  # Loaded automatically from conf/task/PowerGridworldGraph/evovernight13node_simple
    EVOVERNIGHT13NODE_VARIABLE = None # Loaded automatically from conf/task/PowerGridworldGraph/evovernight13node_variable

    @staticmethod
    def associated_class():
        return PowerGridworldGraphClass


class PowerGridworldGraphClass(TaskClass):
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
                            "profile_noise_std": config.get("pv_profile_noise_std", 0.0),
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
            # Global penalty parameters 
            "power_loss_penalty": config.get("power_loss_penalty", 1e-4),
            "voltage_penalty": config.get("voltage_penalty", 1e3),
            "cooperative_voltage": config.get("cooperative_voltage", True),
            "load_2norm_penalty": config.get("load_2norm_penalty", 10),
            "tracking_reward_penalty": config.get("tracking_reward_penalty", 1.0),
            # Signal tracking parameters
            "signal_tracking": config.get("signal_tracking", False),
            "track_total_load": config.get("track_total_load", False),
            "setpoint": config.get("setpoint", 200.0),
            "include_load_in_agent_obs": config.get("include_load_in_agent_obs", True),
        }

        # Power flow config
        pf_config = {
            "cls": OpenDSSSolver,
            "config": {
                "feeder_file": config.get("feeder_file", "ieee_13_dss/IEEE13Nodeckt.dss"),
                "loadshape_file": config.get("loadshape_file", "ieee_13_dss/annual_hourly_load_profile.csv"),
                "system_load_rescale_factor": config.get("system_load_rescale_factor", 0.7),
                "load_noise_std": config.get("load_noise_std", 0.0),  # Legacy (backward compat)
                "load_forecast_noise_std": config.get("load_forecast_noise_std", 0.0),
                "load_actual_noise_std": config.get("load_actual_noise_std", 0.0),
            }
        }

        # Compose the environment config
        env_config = {
            "common_config": common_config,
            "pf_config": pf_config,
            "agents": agents,
            "variable_agent_count": config.get("variable_agent_count", False),
        }

        # Return a function that creates the environment
        return lambda: PaddedMultiAgentEnv(**env_config)
    
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
        return "PowerGridworldGraph"

    def log_info(self, batch: TensorDictBase) -> Dict[str, float]:
        # Optionally return a str->float dict with extra things to log
        # This function has access to the collected batch and is optional
        return {}
