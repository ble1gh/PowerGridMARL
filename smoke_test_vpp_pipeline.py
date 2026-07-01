#!/usr/bin/env python
"""Low-compute VPP pipeline smoke tests.

Default usage is intentionally cheap:

    python smoke_test_vpp_pipeline.py

The default runs import checks, synthetic VPP graph/model contracts, real
PowerGridworldVariable reset/step contracts, and HGTeam module forward checks.
The tiny Experiment.run integration tier is opt-in:

    python smoke_test_vpp_pipeline.py --tier integration
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from types import SimpleNamespace

import torch
import torch_geometric.nn as tgnn
from tensordict import TensorDict
from tensordict import stack as td_stack
from torch import nn
from torchrl.data import Bounded, Composite, Unbounded

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "BenchMARL"))
sys.path.insert(0, os.path.join(ROOT, "PowerGridworld"))

from benchmarl.algorithms import HGTeamHAPPOConfig  # noqa: E402
from benchmarl.algorithms.HGTeamHA import HGTeamHAPPO  # noqa: E402
from benchmarl.environments.PowerGridworldVariable.common import (  # noqa: E402
    PowerGridworldVariableTask,  # noqa: E402
)
from benchmarl.experiment import Experiment, ExperimentConfig  # noqa: E402
from benchmarl.models import (  # noqa: E402
    EdgeBiasedHGT,
    EdgeBiasedHGTConfig,
    EdgeWeightedHGT,
    EdgeWeightedHGTConfig,
    HeteroGNN,
    HeteroGnnConfig,
    MlpConfig,
)

GROUP_COUNTS = {"EV": 4, "PV": 3, "Storage": 2}
OBS_DIMS = {"EV": 8, "PV": 6, "Storage": 5}
N_GRID = 6
ACTION_DIM = 1
GROUPS = tuple(GROUP_COUNTS)


def resolve_device(requested: str, strict_cuda: bool) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        if strict_cuda:
            raise RuntimeError("--device cuda requested, but CUDA is unavailable")
        print("  CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_vpp_critic_gnn_config(hidden_dim: int = 16, num_layers: int = 1) -> HeteroGnnConfig:
    return HeteroGnnConfig(
        topology="adjacency",
        self_loops=True,
        gnn_class=tgnn.TransformerConv,
        gnn_kwargs={"heads": 1, "concat": False, "beta": True},
        grid_edge_keys={
            "line_adjacency": "line_adjacency",
            "transformer_adjacency": "transformer_adjacency",
            "switch_adjacency": "switch_adjacency",
        },
        edge_features_dims={
            "line_adjacency": 3,
            "transformer_adjacency": 3,
            "switch_adjacency": 1,
            "interaction": 0,
            "mapping": 0,
            "mapping_rev": 0,
        },
        node_features_keys={
            "grid_node": "grid_node_features",
            "agents": "participation_score",
        },
        node_features_dims={"grid_node": 2, "agents": 1},
        agent_node_index_key="agent_grid_edge_index",
        exclude_observations_from_node_features=False,
        cat_observations_to_output=False,
        num_layers=num_layers,
        gnn_hidden_dim=hidden_dim,
        norm_class=nn.LayerNorm,
        pos_features=0,
        vel_features=0,
        edge_radius=0,
    )


def build_vpp_edgeweightedhgt_config(
    hidden_dim: int = 16, num_layers: int = 1
) -> EdgeWeightedHGTConfig:
    return EdgeWeightedHGTConfig(
        topology="adjacency",
        self_loops=True,
        grid_edge_keys={
            "line_adjacency": "line_adjacency",
            "transformer_adjacency": "transformer_adjacency",
            "switch_adjacency": "switch_adjacency",
        },
        edge_features_dims={
            "line_adjacency": 3,
            "transformer_adjacency": 3,
            "switch_adjacency": 1,
            "interaction": 0,
            "mapping": 0,
            "mapping_rev": 0,
        },
        node_features_keys={
            "grid_node": "grid_node_features",
            "agents": "participation_score",
        },
        node_features_dims={"grid_node": 2, "agents": 1},
        agent_node_index_key="agent_grid_edge_index",
        exclude_observations_from_node_features=False,
        cat_observations_to_output=False,
        num_layers=num_layers,
        gnn_hidden_dim=hidden_dim,
        heads=2,
        low_rank=4,
        edge_gate_hidden_dim=16,
        edge_gate_num_layers=2,
        zero_init_edge_gates=True,
        norm_class=nn.LayerNorm,
        pos_features=0,
        vel_features=0,
        edge_radius=0,
    )


def build_vpp_edgebiasedhgt_config(
    hidden_dim: int = 16, num_layers: int = 1
) -> EdgeBiasedHGTConfig:
    return EdgeBiasedHGTConfig(
        topology="adjacency",
        self_loops=True,
        grid_edge_keys={
            "line_adjacency": "line_adjacency",
            "transformer_adjacency": "transformer_adjacency",
            "switch_adjacency": "switch_adjacency",
        },
        edge_features_dims={
            "line_adjacency": 3,
            "transformer_adjacency": 3,
            "switch_adjacency": 1,
            "interaction": 0,
            "mapping": 0,
            "mapping_rev": 0,
        },
        node_features_keys={
            "grid_node": "grid_node_features",
            "agents": "participation_score",
        },
        node_features_dims={"grid_node": 2, "agents": 1},
        agent_node_index_key="agent_grid_edge_index",
        exclude_observations_from_node_features=False,
        cat_observations_to_output=False,
        num_layers=num_layers,
        gnn_hidden_dim=hidden_dim,
        heads=2,
        zero_init_edge_projections=True,
        norm_class=nn.LayerNorm,
        pos_features=0,
        vel_features=0,
        edge_radius=0,
    )


def build_vpp_critic_config(critic_model: str, hidden_dim: int = 16, num_layers: int = 1):
    if critic_model == "heterognn":
        return build_vpp_critic_gnn_config(hidden_dim=hidden_dim, num_layers=num_layers)
    if critic_model == "edgeweightedhgt":
        return build_vpp_edgeweightedhgt_config(hidden_dim=hidden_dim, num_layers=num_layers)
    if critic_model == "edgebiasedhgt":
        return build_vpp_edgebiasedhgt_config(hidden_dim=hidden_dim, num_layers=num_layers)
    raise ValueError(f"Unknown critic_model={critic_model}")


def prepare_direct_vpp_gnn_config(config):
    """Mirror HGTeam's per-group placeholder expansion for direct model tests."""
    config.agent_groups = list(GROUPS)
    config.node_types = [*GROUPS, "grid_node"]
    config.node_features_keys = {
        "grid_node": "grid_node_features",
        **{group: f"{group}_participation_score" for group in GROUPS},
    }
    config.node_features_dims = {
        "grid_node": 2,
        **{group: 1 for group in GROUPS},
    }
    return config


def build_vpp_actor_config() -> MlpConfig:
    return MlpConfig(
        num_cells=[32],
        layer_class=nn.Linear,
        activation_class=nn.ReLU,
    )


def make_synthetic_vpp_specs(device: str = "cpu") -> tuple[Composite, Composite, Composite]:
    observation_spec = Composite(device=device)
    action_spec = Composite(device=device)
    output_spec = Composite(device=device)
    for group, n_agents in GROUP_COUNTS.items():
        observation_spec[group] = Composite(
            observation=Bounded(
                low=-10.0,
                high=10.0,
                shape=(OBS_DIMS[group],),
                device=device,
            ),
            active_mask=Unbounded(shape=(), dtype=torch.bool, device=device),
            device=device,
        ).expand(n_agents)
        action_spec[group] = Composite(
            action=Bounded(low=-1.0, high=1.0, shape=(ACTION_DIM,), device=device),
            device=device,
        ).expand(n_agents)
        output_spec[group] = Composite(
            state_value=Unbounded(shape=(1,), device=device),
            device=device,
        ).expand(n_agents)

        observation_spec.set(
            f"{group}_agent_grid_edge_index",
            Unbounded(shape=(2, n_agents), dtype=torch.long, device=device),
        )
        observation_spec.set(
            f"{group}_participation_score",
            Unbounded(shape=(n_agents, 1), device=device),
        )

    observation_spec["grid_node_features"] = Unbounded(shape=(N_GRID, 2), device=device)
    observation_spec["line_adjacency"] = Unbounded(shape=(N_GRID, N_GRID, 3), device=device)
    observation_spec["transformer_adjacency"] = Unbounded(
        shape=(N_GRID, N_GRID, 3), device=device
    )
    observation_spec["switch_adjacency"] = Unbounded(shape=(N_GRID, N_GRID, 1), device=device)
    observation_spec["agent_grid_edge_index"] = Unbounded(
        shape=(2, sum(GROUP_COUNTS.values())), dtype=torch.long, device=device
    )
    observation_spec["participation_score"] = Unbounded(
        shape=(sum(GROUP_COUNTS.values()), 1), device=device
    )
    return observation_spec, action_spec, output_spec


def _mapping_edges(n_agents: int, n_grid: int, batch_size: int, device: str) -> torch.Tensor:
    edges = torch.stack(
        [
            torch.arange(n_agents, device=device),
            torch.arange(n_agents, device=device) % n_grid,
        ],
        dim=0,
    )
    return edges.unsqueeze(0).expand(batch_size, 2, n_agents).clone()


def make_synthetic_vpp_tensordict(batch_size: int = 2, device: str = "cpu") -> TensorDict:
    td = TensorDict({}, batch_size=[batch_size], device=device)
    flat_edges = []
    flat_scores = []
    offset = 0
    for group, n_agents in GROUP_COUNTS.items():
        td.set((group, "observation"), torch.randn(batch_size, n_agents, OBS_DIMS[group], device=device))
        mask = torch.ones(batch_size, n_agents, dtype=torch.bool, device=device)
        mask[:, -1] = False
        td.set((group, "active_mask"), mask)
        td.set((group, "action"), torch.zeros(batch_size, n_agents, ACTION_DIM, device=device))
        td.set(
            f"{group}_agent_grid_edge_index",
            _mapping_edges(n_agents, N_GRID, batch_size, device),
        )
        scores = torch.rand(batch_size, n_agents, 1, device=device)
        td.set(f"{group}_participation_score", scores)
        flat_edges.append(
            torch.stack(
                [
                    torch.arange(offset, offset + n_agents, device=device),
                    torch.arange(n_agents, device=device) % N_GRID,
                ],
                dim=0,
            )
        )
        flat_scores.append(scores)
        offset += n_agents

    td.set("grid_node_features", torch.randn(batch_size, N_GRID, 2, device=device))
    td.set("line_adjacency", torch.zeros(batch_size, N_GRID, N_GRID, 3, device=device))
    td.set("transformer_adjacency", torch.zeros(batch_size, N_GRID, N_GRID, 3, device=device))
    td.set("switch_adjacency", torch.zeros(batch_size, N_GRID, N_GRID, 1, device=device))
    td["line_adjacency"][:, torch.arange(N_GRID - 1), torch.arange(1, N_GRID), :] = torch.tensor(
        [1.0, 0.5, 0.1], device=device
    )
    td["line_adjacency"][:, torch.arange(1, N_GRID), torch.arange(N_GRID - 1), :] = torch.tensor(
        [1.0, 0.5, 0.1], device=device
    )
    td.set("agent_grid_edge_index", torch.cat(flat_edges, dim=-1).unsqueeze(0).expand(batch_size, -1, -1))
    td.set("participation_score", torch.cat(flat_scores, dim=1))
    return td


def assert_vpp_graph_keys(
    td: TensorDict,
    group_names: tuple[str, ...],
    require_static_adjacency: bool = True,
) -> None:
    required_keys = ["grid_node_features", "agent_grid_edge_index"]
    if require_static_adjacency:
        required_keys.extend(
            ["line_adjacency", "transformer_adjacency", "switch_adjacency"]
        )
    for key in required_keys:
        assert key in td.keys(True, True), f"Missing graph key {key}"
        val = td.get(key)
        assert torch.isfinite(val.float()).all(), f"Non-finite values in {key}"
    for group in group_names:
        for key in (
            (group, "observation"),
            (group, "active_mask"),
            f"{group}_agent_grid_edge_index",
            f"{group}_participation_score",
        ):
            assert key in td.keys(True, True), f"Missing VPP key {key}"


def assert_edge_indices_in_bounds(td: TensorDict, group_names: tuple[str, ...]) -> None:
    n_grid = td.get("grid_node_features").shape[-2]
    for group in group_names:
        edge_key = f"{group}_agent_grid_edge_index"
        edges = td.get(edge_key)
        n_agents = td.get((group, "observation")).shape[-2]
        if edges.numel() == 0:
            continue
        src = edges[..., 0, :].reshape(-1)
        dst = edges[..., 1, :].reshape(-1)
        assert int(src.min()) >= 0 and int(src.max()) < n_agents, (
            f"{edge_key} agent indices out of bounds for n_agents={n_agents}"
        )
        assert int(dst.min()) >= 0 and int(dst.max()) < n_grid, (
            f"{edge_key} grid indices out of bounds for n_grid={n_grid}"
        )


def assert_inactive_outputs_zeroed(td: TensorDict, group_names: tuple[str, ...], leaf: str) -> None:
    for group in group_names:
        mask = td.get((group, "active_mask"))
        out = td.get((group, leaf))
        expanded_mask = mask
        while expanded_mask.dim() < out.dim():
            expanded_mask = expanded_mask.unsqueeze(-1)
        inactive = out.masked_select(~expanded_mask.expand_as(out))
        assert torch.allclose(inactive, torch.zeros_like(inactive)), (
            f"Inactive {group}/{leaf} values are not zeroed"
        )


def run_preflight() -> None:
    print("Tier 0: import/dependency preflight")
    for module in ("torch", "tensordict", "torch_geometric", "benchmarl", "gridworld"):
        importlib.import_module(module)
        print(f"  [PASS] import {module}")
    try:
        importlib.import_module("opendssdirect")
        print("  [PASS] import opendssdirect")
    except ModuleNotFoundError as err:
        raise RuntimeError(
            "opendssdirect is required for real VPP env construction. "
            "Load/install the BenchMARL CHPC environment first."
        ) from err
    print(f"  CUDA available: {torch.cuda.is_available()}")


def run_synthetic_gnn_contract(device: str, critic_model: str) -> None:
    print(f"Tier 1: synthetic VPP graph-model contract ({critic_model})")
    observation_spec, action_spec, output_spec = make_synthetic_vpp_specs(device)
    td = make_synthetic_vpp_tensordict(batch_size=2, device=device)
    assert_vpp_graph_keys(td, GROUPS)
    assert_edge_indices_in_bounds(td, GROUPS)

    for use_actions in (False, True):
        config = prepare_direct_vpp_gnn_config(
            build_vpp_critic_config(critic_model, hidden_dim=16, num_layers=1)
        )
        if use_actions:
            efd = dict(config.edge_features_dims or {})
            for src in GROUPS:
                efd[f"{src}_self_interact"] = ACTION_DIM
                for dst in GROUPS:
                    if src != dst:
                        efd[f"{src}_interact_{dst}"] = ACTION_DIM
            efd["interaction"] = ACTION_DIM
            config.edge_features_dims = efd
        model = config.get_model(
            input_spec=observation_spec,
            output_spec=output_spec,
            n_agents=GROUP_COUNTS["EV"],
            centralised=False,
            input_has_agent_dim=True,
            agent_group="EV",
            share_params=True,
            device=device,
            action_spec=action_spec,
        )
        if use_actions:
            model._use_action_edge_features = True
        out = model(td.clone())
        for group, n_agents in GROUP_COUNTS.items():
            val = out.get((group, "state_value"))
            assert val.shape == (2, n_agents, 1), f"{group}: bad output shape {val.shape}"
            assert torch.isfinite(val).all(), f"{group}: non-finite output"
        assert_inactive_outputs_zeroed(out, GROUPS, "state_value")
        print(f"  [PASS] synthetic GNN forward use_action_edge_features={use_actions}")


def build_real_vpp_env(seed: int = 0):
    task = PowerGridworldVariableTask.EVOVERNIGHT13NODE_VPP.get_from_yaml()
    if hasattr(task, "config") and "reward_scale" in task.config:
        task.config["reward_scale"] = 100.0
    env = task.get_env_fun(
        num_envs=1,
        continuous_actions=True,
        seed=seed,
        device="cpu",
    )()
    return task, env


def _zero_actions_from_td(td: TensorDict, group_names: tuple[str, ...]) -> TensorDict:
    for group in group_names:
        obs = td.get((group, "observation"))
        td.set((group, "action"), torch.zeros(*obs.shape[:-1], ACTION_DIM, dtype=obs.dtype))
    return td


def run_real_env_contract() -> tuple[object, object, TensorDict]:
    print("Tier 2: real VPP environment reset/step contract")
    task, env = build_real_vpp_env(seed=7)
    group_names = tuple(task.group_map(env).keys())
    td = env.reset()
    reset_td = td.clone()
    assert_vpp_graph_keys(td, group_names)
    assert_edge_indices_in_bounds(td, group_names)
    flat_active = int(td.get("active_mask").sum().item())
    per_type_active = sum(int(td.get((group, "active_mask")).sum().item()) for group in group_names)
    assert flat_active == per_type_active, "Flat and per-type active masks disagree"
    for group in group_names:
        obs = td.get((group, "observation"))
        mask = td.get((group, "active_mask"))
        assert torch.allclose(obs[~mask], torch.zeros_like(obs[~mask])), (
            f"Inactive {group} observations are not zeroed after reset"
        )

    for _ in range(2):
        td = _zero_actions_from_td(td, group_names)
        stepped = env.step(td)
        next_td = stepped.get("next")
        assert_vpp_graph_keys(next_td, group_names, require_static_adjacency=False)
        assert_edge_indices_in_bounds(next_td, group_names)
        for group in group_names:
            reward = next_td.get((group, "reward"))
            mask = next_td.get((group, "active_mask"))
            if reward is not None:
                assert torch.allclose(reward[~mask], torch.zeros_like(reward[~mask])), (
                    f"Inactive {group} rewards are not zeroed"
                )
        td = next_td
    print("  [PASS] real env reset + 2 steps")
    return task, env, reset_td


def _clone_batched_real_td(td: TensorDict, batch_size: int = 2) -> TensorDict:
    return td_stack([td.clone() for _ in range(batch_size)], 0)


def _assert_graph_model_type(model, requested: str, role: str) -> None:
    if requested == "edgeweightedhgt":
        assert isinstance(model, EdgeWeightedHGT), (
            f"{role} should use EdgeWeightedHGT when requested, got {type(model).__name__}"
        )
    elif requested == "edgebiasedhgt":
        assert isinstance(model, EdgeBiasedHGT), (
            f"{role} should use EdgeBiasedHGT when requested, got {type(model).__name__}"
        )
    else:
        assert isinstance(model, HeteroGNN) and not isinstance(
            model, (EdgeWeightedHGT, EdgeBiasedHGT)
        ), (
            f"{role} should use HeteroGNN when requested, got {type(model).__name__}"
        )


def _build_hgteam_algorithm(task, env, critic_model: str) -> HGTeamHAPPO:
    algo_config = HGTeamHAPPOConfig.get_from_yaml()
    algo_config.gnn_mode = "concat"
    algo_config.heterognn_type = critic_model
    algo_config.embedding_entropy_coef = 0
    algo_config.embedding_diversity_coef = 0
    algo_config.stochastic_z = True
    algo_config.z_dim = 16
    algo_config.hypernet_actor_feature_dim = 32
    algo_config.split_z = False
    algo_config.z_token_dim = 16
    algo_config.z_query_dim = 16
    algo_config.stochastic_z_query = True
    algo_config.scale_lb = 0.0001
    algo_config.lmbda = 0.99
    algo_config.critic_use_other_actions = False
    algo_config.encoder_update_mode = "coop_encoder"
    algo_config.encoder_n_optimizer_steps = 1
    algo_config.encoder_lr = 3e-4
    algo_config.fixed_order = False
    algo_config.use_vib = True
    algo_config.vib_warmup_frames = 1_000_000
    algo_config.vib_beta = 1e-5

    exp_config = SimpleNamespace(
        train_device="cpu",
        buffer_device="cpu",
        share_policy_params=True,
        gamma=0.99,
    )
    mock_experiment = SimpleNamespace(
        config=exp_config,
        model_config=build_vpp_actor_config(),
        critic_model_config=build_vpp_critic_config(critic_model, hidden_dim=16, num_layers=1),
        on_policy=True,
        group_map=task.group_map(env),
        observation_spec=task.observation_spec(env),
        action_spec=task.action_spec(env),
        state_spec=task.state_spec(env),
        action_mask_spec=task.action_mask_spec(env),
        algorithm_config=algo_config,
    )
    return HGTeamHAPPO(experiment=mock_experiment, **algo_config.__dict__)


def run_hgteam_module_contract(critic_model: str) -> None:
    print(f"Tier 3: HGTeam module assembly/forward contract ({critic_model})")
    task, env, td = run_real_env_contract()
    algo = _build_hgteam_algorithm(task, env, critic_model)
    actor_config = build_vpp_actor_config()
    group_names = tuple(task.group_map(env).keys())
    batch = _clone_batched_real_td(td, batch_size=2)

    shared_actor_gnn = None
    for group in group_names:
        policy = algo._get_policy_for_loss(group, actor_config, continuous=True)
        policy_td = batch.clone()
        policy(policy_td)
        action = policy_td.get((group, "action"))
        assert action is not None, f"{group} policy did not write action"
        assert torch.isfinite(action).all(), f"{group} action contains non-finite values"
        if shared_actor_gnn is None:
            shared_actor_gnn = algo._shared_actor_gnn
        else:
            assert algo._shared_actor_gnn is shared_actor_gnn, "Shared actor GNN was not reused"

    assert shared_actor_gnn is not None, "Shared actor GNN was not built"
    _assert_graph_model_type(shared_actor_gnn, critic_model, "Shared actor encoder")

    for group in group_names:
        critic = algo.get_critic(group)
        critic_td = batch.clone()
        critic(critic_td)
        value = critic_td.get((group, "state_value"))
        assert value is not None, f"{group} critic did not write state_value"
        assert torch.isfinite(value).all(), f"{group} state_value contains non-finite values"
        assert value.shape[-2] == len(task.group_map(env)[group]), (
            f"{group} critic value has wrong agent dimension: {value.shape}"
        )

    assert algo._shared_gnn_critic is not None, "Shared critic GNN was not built"
    _assert_graph_model_type(algo._shared_gnn_critic, critic_model, "Shared critic")
    print("  [PASS] policies and critics build/forward without collector loop")


def run_integration_tier(device: str, with_eval: bool, critic_model: str) -> None:
    print(f"Tier 4: tiny Experiment.run integration ({critic_model})")
    algorithm_config = HGTeamHAPPOConfig.get_from_yaml()
    algorithm_config.gnn_mode = "concat"
    algorithm_config.heterognn_type = critic_model
    algorithm_config.embedding_entropy_coef = 0
    algorithm_config.embedding_diversity_coef = 0
    algorithm_config.stochastic_z = True
    algorithm_config.z_dim = 16
    algorithm_config.hypernet_actor_feature_dim = 32
    algorithm_config.split_z = False
    algorithm_config.z_token_dim = 16
    algorithm_config.z_query_dim = 16
    algorithm_config.stochastic_z_query = True
    algorithm_config.scale_lb = 0.0001
    algorithm_config.lmbda = 0.99
    algorithm_config.critic_use_other_actions = False
    algorithm_config.encoder_update_mode = "coop_encoder"
    algorithm_config.encoder_n_optimizer_steps = 1
    algorithm_config.encoder_lr = 3e-4
    algorithm_config.fixed_order = False
    algorithm_config.use_vib = True
    algorithm_config.vib_warmup_frames = 1_000_000
    algorithm_config.vib_beta = 1e-5

    task = PowerGridworldVariableTask.EVOVERNIGHT13NODE_VPP.get_from_yaml()
    if hasattr(task, "config") and "reward_scale" in task.config:
        task.config["reward_scale"] = 100.0

    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.sampling_device = "cpu"
    experiment_config.train_device = device
    experiment_config.collection_policy_device = device
    experiment_config.share_policy_params = True
    experiment_config.evaluation_static = False
    experiment_config.parallel_collection = False
    experiment_config.lr = 3e-4
    experiment_config.evaluation_episodes = 1
    experiment_config.on_policy_n_envs_per_worker = 1
    experiment_config.on_policy_collected_frames_per_batch = 8
    experiment_config.on_policy_minibatch_size = 4
    experiment_config.on_policy_n_minibatch_iters = 1
    experiment_config.max_n_frames = 8
    experiment_config.max_n_iters = 1
    experiment_config.evaluation_interval = 8 if with_eval else 10_000
    experiment_config.loggers = []
    experiment_config.create_json = False
    experiment_config.checkpoint_at_end = False

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=build_vpp_actor_config(),
        critic_model_config=build_vpp_critic_config(critic_model, hidden_dim=16, num_layers=1),
        seed=0,
        config=experiment_config,
    )
    experiment.run()
    print("  [PASS] tiny Experiment.run integration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier",
        choices=("fast", "synthetic", "env", "models", "modules", "integration", "all"),
        default="fast",
        help=(
            "fast=tiers 0-3, synthetic=tier 1, env=tiers 0+2, "
            "models=tiers 0+1+3, modules=tier 3, integration=tier 4"
        ),
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument(
        "--critic-model",
        choices=("heterognn", "edgeweightedhgt", "edgebiasedhgt"),
        default="heterognn",
        help=(
            "Shared VPP graph backbone to smoke-test for both actor encoder and critic; "
            "defaults preserve existing behavior."
        ),
    )
    parser.add_argument("--strict-cuda", action="store_true")
    parser.add_argument("--with-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device, args.strict_cuda)
    if args.tier in ("fast", "synthetic", "env", "models", "modules", "all"):
        run_preflight()
    if args.tier in ("fast", "synthetic", "models", "all"):
        run_synthetic_gnn_contract(device, args.critic_model)
    if args.tier in ("fast", "env", "all"):
        run_real_env_contract()
    if args.tier in ("fast", "models", "modules", "all"):
        run_hgteam_module_contract(args.critic_model)
    if args.tier in ("integration", "all"):
        run_integration_tier(device, args.with_eval, args.critic_model)
    print("All requested VPP smoke tiers passed!")


if __name__ == "__main__":
    main()
