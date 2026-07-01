#!/usr/bin/env python3
"""Smoke tests for the opt-in EdgeWeightedHGT model."""

# ruff: noqa: E402, I001

import sys

sys.path.insert(0, "BenchMARL")
sys.path.insert(0, "PowerGridworld")

import torch
from tensordict import TensorDict
from torchrl.data import Categorical, Composite, Unbounded

from benchmarl.models.edgeweightedHGT import EdgeWeightedHGT


def _build_smacv2_like_spec(groups: dict[str, int], n_enemies: int, entity_dim: int):
    obs_dim = 182
    spec = Composite(device="cpu")
    for group, n_agents in groups.items():
        per_agent = {
            "observation": Unbounded(shape=(obs_dim,)),
            "entity_enemy": Unbounded(shape=(n_enemies, entity_dim)),
            "entity_self": Unbounded(shape=(7,)),
            "move_feats": Unbounded(shape=(4,)),
            "active_mask": Unbounded(shape=(), dtype=torch.bool),
        }
        for ally_group, ally_max in groups.items():
            per_agent[f"entity_{ally_group}_ally"] = Unbounded(
                shape=(ally_max, entity_dim)
            )
        spec[group] = Composite(per_agent, device="cpu").expand(n_agents)
    return spec


def _build_entity_tensordict(groups, n_enemies, entity_dim, batch_dims, include_mask):
    obs_dim = 182
    td = TensorDict({}, batch_size=batch_dims)
    for group, n_agents in groups.items():
        shape_prefix = (*batch_dims, n_agents)
        td.set((group, "observation"), torch.randn(*shape_prefix, obs_dim))
        td.set((group, "entity_enemy"), torch.randn(*shape_prefix, n_enemies, entity_dim))
        td.set((group, "entity_self"), torch.randn(*shape_prefix, 7))
        td.set((group, "move_feats"), torch.randn(*shape_prefix, 4))
        if include_mask:
            td.set((group, "active_mask"), torch.ones(*shape_prefix, dtype=torch.bool))
        for ally_group, ally_max in groups.items():
            td.set(
                (group, f"entity_{ally_group}_ally"),
                torch.randn(*shape_prefix, ally_max, entity_dim),
            )
    return td


def _output_spec(groups: dict[str, int], output_features: int):
    spec = Composite(device="cpu")
    for group, n_agents in groups.items():
        spec[group] = Composite(
            embedding=Unbounded(shape=(output_features,)),
            device="cpu",
        ).expand(n_agents)
    return spec


def _build_model(
    input_spec,
    groups,
    output_features=16,
    topology="full",
    action_spec=None,
    edge_features_dims=None,
    use_action_edge_features=False,
    low_rank=4,
    zero_init_edge_gates=True,
):
    agent_groups = list(groups.keys())
    model = EdgeWeightedHGT(
        topology=topology,
        self_loops=True,
        position_key=None,
        exclude_pos_from_node_features=None,
        velocity_key=None,
        edge_radius=None,
        pos_features=0,
        vel_features=0,
        num_layers=1,
        agent_groups=agent_groups,
        edge_features_dims=edge_features_dims,
        exclude_observations_from_node_features=False,
        cat_observations_to_output=False,
        gnn_hidden_dim=output_features,
        heads=2,
        low_rank=low_rank,
        edge_gate_hidden_dim=16,
        edge_gate_num_layers=2,
        zero_init_edge_gates=zero_init_edge_gates,
        input_spec=input_spec,
        output_spec=_output_spec(groups, output_features),
        agent_group=agent_groups[0],
        input_has_agent_dim=True,
        n_agents=groups[agent_groups[0]],
        centralised=False,
        share_params=True,
        device="cpu",
        action_spec=action_spec if action_spec is not None else Composite(device="cpu"),
        model_index=0,
        is_critic=True,
    )
    if use_action_edge_features:
        model._use_action_edge_features = True
    return model


def _assert_group_outputs(out, model, groups, batch_shape, output_features=16):
    for group, n_agents in groups.items():
        key = (group, "embedding") if group != model.agent_group else model.out_key
        val = out.get(key)
        assert val is not None, f"Missing output for {group}"
        assert val.shape == (*batch_shape, n_agents, output_features), (
            f"{group}: bad output shape {val.shape}"
        )
        assert torch.isfinite(val).all(), f"{group}: non-finite output"


def test_entity_active_mask():
    print("=== Test 1: entity observations with active_mask ===")
    groups = {"stalker": 8, "zealot": 8, "colossus": 4}
    spec = _build_smacv2_like_spec(groups, n_enemies=10, entity_dim=9)
    model = _build_model(spec, groups)
    td = _build_entity_tensordict(groups, 10, 9, batch_dims=(3,), include_mask=True)
    td.get(("colossus", "active_mask"))[:, -2:] = False
    out = model(td)
    _assert_group_outputs(out, model, groups, (3,))
    colossus = out.get(("colossus", "embedding"))
    assert torch.allclose(colossus[:, -2:], torch.zeros_like(colossus[:, -2:]))
    print("  [PASS] active_mask filters edges and zeroes inactive outputs")


def test_entity_without_active_mask():
    print("=== Test 2: entity observations without active_mask ===")
    groups = {"stalker": 8, "zealot": 8, "colossus": 4}
    spec = _build_smacv2_like_spec(groups, n_enemies=10, entity_dim=9)
    model = _build_model(spec, groups)
    td = _build_entity_tensordict(groups, 10, 9, batch_dims=(3,), include_mask=False)
    out = model(td)
    _assert_group_outputs(out, model, groups, (3,))
    print("  [PASS] fallback agent-axis handling works")


def test_vmap_like_extra_dim():
    print("=== Test 3: vmap-like extra leading dimension ===")
    groups = {"stalker": 8, "zealot": 8, "colossus": 4}
    spec = _build_smacv2_like_spec(groups, n_enemies=10, entity_dim=9)
    model = _build_model(spec, groups)
    td = _build_entity_tensordict(groups, 10, 9, batch_dims=(2, 3), include_mask=True)
    out = model(td)
    _assert_group_outputs(out, model, groups, (2, 3))
    print("  [PASS] extra leading dimensions preserve output shape")


def test_flat_obs_only():
    print("=== Test 4: flat VPP-like observations ===")
    groups = {"EV": 10, "PV": 7, "Storage": 5}
    spec = Composite(device="cpu")
    td = TensorDict({}, batch_size=(4,))
    for group, n_agents in groups.items():
        spec[group] = Composite(
            observation=Unbounded(shape=(30,)),
            active_mask=Unbounded(shape=(), dtype=torch.bool),
            device="cpu",
        ).expand(n_agents)
        td.set((group, "observation"), torch.randn(4, n_agents, 30))
        td.set((group, "active_mask"), torch.ones(4, n_agents, dtype=torch.bool))
    model = _build_model(spec, groups)
    out = model(td)
    _assert_group_outputs(out, model, groups, (4,))
    print("  [PASS] flat observations work")


def test_continuous_action_edge_features_and_zero_init():
    print("=== Test 5: continuous action edge features + zero-init vanilla limit ===")
    groups = {"EV": 4, "PV": 3, "Storage": 2}
    action_dim = 1
    spec = Composite(device="cpu")
    action_spec = Composite(device="cpu")
    td_a = TensorDict({}, batch_size=(2,))
    td_b = TensorDict({}, batch_size=(2,))
    for group, n_agents in groups.items():
        spec[group] = Composite(
            observation=Unbounded(shape=(8,)),
            active_mask=Unbounded(shape=(), dtype=torch.bool),
            device="cpu",
        ).expand(n_agents)
        action_spec[group] = Composite(
            action=Unbounded(shape=(action_dim,)),
            device="cpu",
        ).expand(n_agents)
        obs = torch.randn(2, n_agents, 8)
        mask = torch.ones(2, n_agents, dtype=torch.bool)
        td_a.set((group, "observation"), obs)
        td_b.set((group, "observation"), obs.clone())
        td_a.set((group, "active_mask"), mask)
        td_b.set((group, "active_mask"), mask.clone())
        td_a.set((group, "action"), torch.randn(2, n_agents, action_dim))
        td_b.set((group, "action"), torch.zeros(2, n_agents, action_dim))

    model = _build_model(
        spec,
        groups,
        action_spec=action_spec,
        edge_features_dims={"interaction": action_dim},
        use_action_edge_features=True,
        zero_init_edge_gates=True,
    )
    out_a = model(td_a)
    out_b = model(td_b)
    _assert_group_outputs(out_a, model, groups, (2,))
    for group in groups:
        key = (group, "embedding") if group != model.agent_group else model.out_key
        assert torch.allclose(out_a.get(key), out_b.get(key), atol=1e-6), (
            f"{group}: zero-initialized gates should make edge features a no-op"
        )
    print("  [PASS] action edge features are wired and initially no-op")


def test_discrete_action_edge_features():
    print("=== Test 6: discrete action edge features ===")
    groups = {"stalker": 5, "zealot": 4}
    n_actions = 6
    spec = Composite(device="cpu")
    action_spec = Composite(device="cpu")
    td = TensorDict({}, batch_size=(3,))
    for group, n_agents in groups.items():
        spec[group] = Composite(
            observation=Unbounded(shape=(12,)),
            active_mask=Unbounded(shape=(), dtype=torch.bool),
            device="cpu",
        ).expand(n_agents)
        action_spec[group] = Composite(
            action=Categorical(n=n_actions, shape=(), device="cpu"),
            device="cpu",
        ).expand(n_agents)
        td.set((group, "observation"), torch.randn(3, n_agents, 12))
        td.set((group, "active_mask"), torch.ones(3, n_agents, dtype=torch.bool))
        td.set((group, "action"), torch.randint(0, n_actions, (3, n_agents)))
    model = _build_model(
        spec,
        groups,
        action_spec=action_spec,
        edge_features_dims={"interaction": n_actions},
        use_action_edge_features=True,
    )
    out = model(td)
    _assert_group_outputs(out, model, groups, (3,))
    print("  [PASS] categorical actions are converted to one-hot edge features")


def test_low_rank_gradients_and_disabled_modulation():
    print("=== Test 7: low-rank gradients and disabled modulation ===")
    groups = {"EV": 4, "PV": 3}
    action_dim = 2
    spec = Composite(device="cpu")
    action_spec = Composite(device="cpu")
    td = TensorDict({}, batch_size=(2,))
    for group, n_agents in groups.items():
        spec[group] = Composite(
            observation=Unbounded(shape=(8,)),
            active_mask=Unbounded(shape=(), dtype=torch.bool),
            device="cpu",
        ).expand(n_agents)
        action_spec[group] = Composite(
            action=Unbounded(shape=(action_dim,)),
            device="cpu",
        ).expand(n_agents)
        td.set((group, "observation"), torch.randn(2, n_agents, 8))
        td.set((group, "active_mask"), torch.ones(2, n_agents, dtype=torch.bool))
        td.set((group, "action"), torch.randn(2, n_agents, action_dim))

    model = _build_model(
        spec,
        groups,
        action_spec=action_spec,
        edge_features_dims={"interaction": action_dim},
        use_action_edge_features=True,
        zero_init_edge_gates=False,
    )
    out = model(td.clone())
    loss = sum(out.get((group, "embedding")).sum() for group in groups)
    loss.backward()
    gate_grad = any(
        param.grad is not None and torch.isfinite(param.grad).all()
        for conv in model.convs
        for gate in conv.edge_gates.values()
        for param in gate.parameters()
    )
    low_rank_grad = any(
        param.grad is not None and torch.isfinite(param.grad).all()
        for conv in model.convs
        for params in (conv.att_u, conv.att_v, conv.msg_u, conv.msg_v)
        for param in params.values()
    )
    assert gate_grad, "Expected gradients for edge gate networks"
    assert low_rank_grad, "Expected gradients for low-rank factors"

    disabled = _build_model(
        spec,
        groups,
        action_spec=action_spec,
        edge_features_dims={"interaction": action_dim},
        use_action_edge_features=True,
        low_rank=0,
    )
    assert all(len(conv.edge_gates) == 0 for conv in disabled.convs)
    out_disabled = disabled(td.clone())
    _assert_group_outputs(out_disabled, disabled, groups, (2,))
    print("  [PASS] gradients flow and low_rank=0 disables modulation")


if __name__ == "__main__":
    test_entity_active_mask()
    test_entity_without_active_mask()
    test_vmap_like_extra_dim()
    test_flat_obs_only()
    test_continuous_action_edge_features_and_zero_init()
    test_discrete_action_edge_features()
    test_low_rank_gradients_and_disabled_modulation()
    print("All EdgeWeightedHGT smoke tests passed!")
