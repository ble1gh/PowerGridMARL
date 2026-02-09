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

# PowerGridworldVariable environment requirements
import numpy as np
import pandas as pd
from gridworld import ComponentEnv
from gridworld import MultiAgentEnv
from gridworld.distribution_system import OpenDSSSolver
from gridworld.agents.vehicles import EVChargingEnv
from gridworld.agents.pv import PVEnv
from gridworld.agents.energy_storage import EnergyStorageEnv
from mpl_toolkits.mplot3d import Axes3D

class VariableAgentMultiAgentEnv(MultiAgentEnv):
    """
    A wrapper around MultiAgentEnv that handles variable agent counts via padding.
    The environment is initialized with the MAXIMUM number of agents possible.
    On reset, a random subset of agents is selected based on min/max config.
    Inactive agents have their observations and rewards zeroed out.
    
    The agent set changes between episodes but remains fixed within an episode.
    """
    def __init__(self, *args, 
                 variable_agent_config=None,
                 all_possible_agents=None,
                 **kwargs):
        """
        Args:
            variable_agent_config: Dict with keys:
                - min_EVs, max_EVs: range for EV agent count
                - min_PVs, max_PVs: range for PV agent count  
                - min_Storage, max_Storage: range for Storage agent count
                - EV_busses, PV_busses, Storage_busses: possible buses for each type
                - allow_multiple_agents_per_node: bool
            all_possible_agents: List of all possible agent configs (max set)
        """
        # Store variable agent config before super().__init__
        self.variable_agent_config = variable_agent_config or {}
        self.all_possible_agents = all_possible_agents or []
        
        super().__init__(*args, **kwargs)
        self.max_agents = len(self.agents)
        self.active_mask = torch.ones(self.max_agents, dtype=torch.bool)
        
        # Build agent name to index mapping
        self._agent_name_to_idx = {agent.name: i for i, agent in enumerate(self.agents)}
        
        # Add active_mask to observation spec so it's collected properly
        from torchrl.data import Unbounded
        self.observation_spec.set("active_mask", Unbounded(shape=(self.max_agents,), dtype=torch.bool))

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

                # Add adjacency tensors to observation spec so they are passed to the model
                for k, v in self.adj_tensors.items():
                    self.observation_spec.set(k, Unbounded(shape=v.shape, dtype=torch.float32))

        else:
             print("Warning: pf_solver not found in VariableAgentMultiAgentEnv during init")

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

    def _sample_active_agents(self):
        """
        Sample which agents are active for this episode based on variable_agent_config.
        
        The sampling respects:
        - min/max counts for each agent type
        - available buses for each type
        - allow_multiple_agents_per_node flag
        """
        config = self.variable_agent_config
        
        # Get min/max for each type
        min_evs = config.get('min_EVs', 0)
        max_evs = config.get('max_EVs', 0)
        min_pvs = config.get('min_PVs', 0)
        max_pvs = config.get('max_PVs', 0)
        min_storage = config.get('min_Storage', 0)
        max_storage = config.get('max_Storage', 0)

        # If min and max are equal, skip usage of np.random to ensure consistency
        if min_evs == max_evs and min_pvs == max_pvs and min_storage == max_storage:
             self.active_mask = torch.ones(self.max_agents, dtype=torch.bool)
             return
        
        # Get available buses
        ev_busses = config.get('EV_busses', [])
        pv_busses = config.get('PV_busses', [])
        storage_busses = config.get('Storage_busses', [])
        
        allow_multi = config.get('allow_multiple_agents_per_node', False)
        
        # Initialize mask to all inactive
        self.active_mask = torch.zeros(self.max_agents, dtype=torch.bool)
        
        # Helper to select agents of a given type
        def select_agents_of_type(agent_prefix, busses, min_count, max_count):
            if max_count == 0 or not busses:
                return []
            
            # Find all agents of this type
            type_agents = [(name, idx) for name, idx in self._agent_name_to_idx.items() 
                          if name.startswith(agent_prefix)]
            
            if not type_agents:
                return []
            
            # Sample how many to activate
            actual_max = min(max_count, len(type_agents))
            actual_min = min(min_count, actual_max)
            # n_to_activate = np.random.randint(actual_min, actual_max + 1)
            # print(f"[Sampling] Type {agent_prefix}: Activating {n_to_activate} / {actual_max} agents")
            
            # Use torch random for sampling
            n_to_activate = int(torch.randint(actual_min, actual_max + 1, (1,)).item())
            
            if n_to_activate == 0:
                return []
            
            if allow_multi:
                # Can have multiple agents per node - just sample from all
                perm = torch.randperm(len(type_agents))
                selected_indices = perm[:n_to_activate].tolist()
                return [type_agents[i][1] for i in selected_indices]
            else:
                # No multiple agents per node - sample buses first, then pick one agent per bus
                # Group agents by bus
                bus_to_agents = {}
                for name, idx in type_agents:
                    # Extract bus from name (e.g., "EV-634a-1" -> "634a")
                    parts = name.split('-')
                    if len(parts) >= 2:
                        bus = parts[1]
                        if bus not in bus_to_agents:
                            bus_to_agents[bus] = []
                        bus_to_agents[bus].append(idx)
                
                # Sample buses
                available_busses = [b for b in busses if b in bus_to_agents]
                n_to_select = min(n_to_activate, len(available_busses))
                
                if n_to_select == 0:
                    return []
                
                # Sample buses
                perm_buses = torch.randperm(len(available_busses))
                selected_bus_indices = perm_buses[:n_to_select].tolist()
                selected_busses = [available_busses[i] for i in selected_bus_indices]
                
                # For each selected bus, pick one random agent
                selected = []
                for bus in selected_busses:
                    agents_at_bus = bus_to_agents[bus]
                    # Pick one agent
                    idx_on_bus = int(torch.randint(0, len(agents_at_bus), (1,)).item())
                    selected.append(agents_at_bus[idx_on_bus])
                
                return selected
        
        # Select agents for each type
        ev_selected = select_agents_of_type('EV-', ev_busses, min_evs, max_evs)
        pv_selected = select_agents_of_type('PV-', pv_busses, min_pvs, max_pvs)
        storage_selected = select_agents_of_type('Storage-', storage_busses, min_storage, max_storage)
        
        # Activate selected agents
        for idx in ev_selected + pv_selected + storage_selected:
            self.active_mask[idx] = True
        
        # Ensure at least one agent is active
        if not self.active_mask.any():
            # Fallback: activate first agent
            self.active_mask[0] = True

    def _sample_pv_scaling_factors(self):
        """
        Sample scaling factors for active PV agents from [min, max] range.
        Updates each PV agent's scaling_factor and re-applies noise/scaling.
        """
        config = self.variable_agent_config
        min_scale = config.get('min_pv_scaling_factor', 1.0)
        max_scale = config.get('max_pv_scaling_factor', 1.0)
        
        if min_scale == max_scale:
            # No variation needed
            return
        
        for i, agent in enumerate(self.agents):
            if agent.name.startswith('PV-') and self.active_mask[i]:
                # Sample a new scaling factor uniformly
                # Use torch or np depending on preference, sticking to numpy for continuous params
                sampled_scale = np.random.uniform(min_scale, max_scale)
                agent.scaling_factor = sampled_scale
                # Re-apply noise and scaling with new factor
                if hasattr(agent, '_apply_noise_and_scale'):
                    agent._apply_noise_and_scale()

    def _reset(self, tensordict=None, **kwargs):
        # Sample which agents are active for this episode
        self._sample_active_agents()
        # print(f"DEBUG: Resetting Env. Active Agents: {self.active_mask.sum().item()} / {len(self.active_mask)}")
        
        # Sample PV scaling factors for active PV agents
        self._sample_pv_scaling_factors()
            
        out = super()._reset(tensordict, **kwargs)
        
        # Zero out observations for inactive agents
        obs = out.get(("agents", "observation"))
        if obs is not None:
            obs[~self.active_mask] = 0.0
            out.set(("agents", "observation"), obs)
            
        # Add active mask to tensordict so algorithms can use it
        out.set("active_mask", self.active_mask.clone())
                
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
        # Mask actions for inactive agents BEFORE step to ensure they don't affect physics
        # (e.g. consume power or charge batteries)
        if hasattr(self, 'active_mask'):
            actions = tensordict.get(("agents", "action"))
            if actions is not None:
                # We modify the action in-place so that the environment executes 
                # a no-op (0.0) for inactive agents.
                actions[~self.active_mask] = 0.0
                tensordict.set(("agents", "action"), actions)

        out = super()._step(tensordict)
        
        # Zero out next observations for inactive agents
        obs = out.get(("next", "agents", "observation"))
        if obs is not None:
            obs[~self.active_mask] = 0.0
            out.set(("next", "agents", "observation"), obs)
            
        # Zero out rewards for inactive agents
        reward = out.get(("next", "agents", "reward"))
        if reward is not None:
            reward[~self.active_mask] = 0.0
            out.set(("next", "agents", "reward"), reward)
        
        # Include active mask in output
        out.set("active_mask", self.active_mask.clone())
        
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

class PowerGridworldVariableTask(Task):
    # Your task names.
    # Their config will be loaded from conf/task/PowerGridworldVariable

    EVOVERNIGHT13NODE_SIMPLE = None  # Loaded automatically from conf/task/PowerGridworldVariable/evovernight13node_simple
    EVOVERNIGHT13NODE_NONCOOP = None # Loaded automatically from conf/task/PowerGridworldVariable/evovernight13node_nonCoop
    EVOVERNIGHT13NODE_VPP = None     # Loaded automatically from conf/task/PowerGridworldVariable/evovernight13node_vpp

    @staticmethod
    def associated_class():
        return PowerGridworldVariableClass


class PowerGridworldVariableClass(TaskClass):
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        
        # Get variable agent configuration
        min_evs = config.get("min_EVs", 0)
        max_evs = config.get("max_EVs", 0)
        min_pvs = config.get("min_PVs", 0)
        max_pvs = config.get("max_PVs", 0)
        min_storage = config.get("min_Storage", 0)
        max_storage = config.get("max_Storage", 0)
        
        # Get buses for each agent type
        EV_busses = config.get("EV_busses", [])
        PV_busses = config.get("PV_busses", [])
        Storage_busses = config.get("Storage_busses", [])
        
        allow_multi = config.get("allow_multiple_agents_per_node", False)
        
        # Build MAX agent configs for each type (env will be initialized with max agents)
        # On reset, a subset will be activated based on min/max config
        agents = []
        
        # Create EV agents - one per bus (or more if allow_multiple_agents_per_node)
        if max_evs > 0 and EV_busses:
            if allow_multi:
                # Create max_evs agents, distributing across buses round-robin
                for i in range(max_evs):
                    bus = EV_busses[i % len(EV_busses)]
                    copy_num = (i // len(EV_busses)) + 1
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
            else:
                # One agent per bus (max = number of buses)
                for bus in EV_busses:
                    agent_name = f"EV-{bus}-1"
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
        # Use max_pv_scaling_factor for obs space bounds; actual scaling sampled on reset
        max_pv_scale = config.get("max_pv_scaling_factor", config.get("pv_scaling_factor", 1.0))
        if max_pvs > 0 and PV_busses:
            if allow_multi:
                # Create max_pvs agents, distributing across buses round-robin
                for i in range(max_pvs):
                    bus = PV_busses[i % len(PV_busses)]
                    copy_num = (i // len(PV_busses)) + 1
                    agent_name = f"PV-{bus}-{copy_num}"
                    agents.append({
                        "name": agent_name,
                        "bus": bus,
                        "cls": PVEnv,
                        "config": {
                            "profile_csv": config.get("pv_profile_csv", "pv_profile.csv"),
                            "scaling_factor": max_pv_scale,  # Use max for obs space bounds
                            "profile_noise_std": config.get("pv_profile_noise_std", 0.0),
                            "rescale_spaces": config.get("rescale_spaces", False),
                            "grid_aware": config.get("pv_grid_aware", False),
                        }
                    })
            else:
                for bus in PV_busses:
                    agent_name = f"PV-{bus}-1"
                    agents.append({
                        "name": agent_name,
                        "bus": bus,
                        "cls": PVEnv,
                        "config": {
                            "profile_csv": config.get("pv_profile_csv", "pv_profile.csv"),
                            "scaling_factor": max_pv_scale,  # Use max for obs space bounds
                            "profile_noise_std": config.get("pv_profile_noise_std", 0.0),
                            "rescale_spaces": config.get("rescale_spaces", False),
                            "grid_aware": config.get("pv_grid_aware", False),
                        }
                    })
        
        # Create Energy Storage agents
        if max_storage > 0 and Storage_busses:
            if allow_multi:
                # Create max_storage agents, distributing across buses round-robin
                for i in range(max_storage):
                    bus = Storage_busses[i % len(Storage_busses)]
                    copy_num = (i // len(Storage_busses)) + 1
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
            else:
                for bus in Storage_busses:
                    agent_name = f"Storage-{bus}-1"
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

        n_ev = len([a for a in agents if a['name'].startswith('EV-')])
        n_pv = len([a for a in agents if a['name'].startswith('PV-')])
        n_storage = len([a for a in agents if a['name'].startswith('Storage-')])
        
        print(f"Created {len(agents)} max agents:")
        print(f"  - {n_ev} EV agents (will sample {min_evs}-{max_evs} per episode)")
        print(f"  - {n_pv} PV agents (will sample {min_pvs}-{max_pvs} per episode)")
        print(f"  - {n_storage} Storage agents (will sample {min_storage}-{max_storage} per episode)")

        # Variable agent config to pass to environment
        variable_agent_config = {
            "min_EVs": min_evs,
            "max_EVs": max_evs,
            "min_PVs": min_pvs,
            "max_PVs": max_pvs,
            "min_Storage": min_storage,
            "max_Storage": max_storage,
            "EV_busses": EV_busses,
            "PV_busses": PV_busses,
            "Storage_busses": Storage_busses,
            "allow_multiple_agents_per_node": allow_multi,
            "min_pv_scaling_factor": config.get("min_pv_scaling_factor", config.get("pv_scaling_factor", 1.0)),
            "max_pv_scaling_factor": config.get("max_pv_scaling_factor", config.get("pv_scaling_factor", 1.0)),
        }

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
            # VPP (Virtual Power Plant) reward parameters
            "vpp_reward": config.get("vpp_reward", False),
            "vpp_setpoint": config.get("vpp_setpoint", 0.0),
            "vpp_reward_penalty": config.get("vpp_reward_penalty", 1.0),
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
            "variable_agent_config": variable_agent_config,
        }

        # Return a function that creates the environment
        return lambda: VariableAgentMultiAgentEnv(**env_config)
    
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
        return "PowerGridworldVariable"

    def log_info(self, batch: TensorDictBase) -> Dict[str, float]:
        # Optionally return a str->float dict with extra things to log
        # This function has access to the collected batch and is optional
        logs = {}
        if "active_mask" in batch.keys():
            # active_mask shape: (batch_size, n_agents)
            mask = batch["active_mask"]
            # Convert boolean to float and sum over agents (dim=-1)
            active_counts = mask.float().sum(dim=-1)
            # Log the mean, min, and max active agents in the batch
            logs["counters/num_active_agents_mean"] = active_counts.mean().item()
            logs["counters/num_active_agents_min"] = active_counts.min().item()
            logs["counters/num_active_agents_max"] = active_counts.max().item()
            # print(f"Active Agents in Batch: {active_counts.mean().item():.2f}")
        else:
            # print(f"Active mask not found in batch keys: {batch.keys()}")
            pass
        return logs
