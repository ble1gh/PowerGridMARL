#!/usr/bin/env python3
"""Smoke test: HeteroGNN _forward with entity-decomposed observations.

Verifies that mixing flat per-agent observations (2D per agent) with
multi-dimensional entity tensors (3D per agent) does not cause dimension
mismatches in torch.cat(input_list, dim=-1).

Tests both the active_mask-present path (primary) and the fallback path
where active_mask is absent from the tensordict (simulates TorchRL's
value estimator .select(*in_keys) stripping it).
"""

import sys

sys.path.insert(0, "BenchMARL")
sys.path.insert(0, "PowerGridworld")

import torch
import torch_geometric.nn
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded


def _build_smacv2_like_spec(groups: dict[str, int], n_enemies: int, entity_dim: int):
    """Build a Composite input spec mimicking SMACv2 entity-decomposed obs."""
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


def _build_tensordict(groups, n_enemies, entity_dim, batch_dims, include_mask):
    """Build a tensordict with the given batch dimensions."""
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


def _build_heterognn(
    input_spec,
    groups,
    output_features=16,
    topology="empty",
    action_spec=None,
    edge_features_dims=None,
    use_action_edge_features=False,
):
    """Instantiate a HeteroGNN with the given input spec."""
    from benchmarl.models.heterognn import HeteroGNN

    agent_groups = list(groups.keys())
    first_group = agent_groups[0]
    n_agents = groups[first_group]

    output_spec = Composite(device="cpu")
    for group, na in groups.items():
        output_spec[group] = Composite(
            embedding=Unbounded(shape=(output_features,)),
            device="cpu",
        ).expand(na)

    model = HeteroGNN(
        topology=topology,
        self_loops=True,
        gnn_class=torch_geometric.nn.TransformerConv,
        gnn_kwargs=None,
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
        input_spec=input_spec,
        output_spec=output_spec,
        agent_group=first_group,
        input_has_agent_dim=True,
        n_agents=n_agents,
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


def test_with_active_mask():
    """Entity obs + active_mask present: primary path (should always work)."""
    print("=== Test 1: HeteroGNN with active_mask present ===")
    groups = {"stalker": 8, "zealot": 8, "colossus": 4}
    n_enemies, entity_dim = 10, 9

    spec = _build_smacv2_like_spec(groups, n_enemies, entity_dim)
    model = _build_heterognn(spec, groups)

    B = 4
    td = _build_tensordict(groups, n_enemies, entity_dim, batch_dims=(B,), include_mask=True)
    td.get(("colossus", "active_mask"))[:, -2:] = False
    out = model(td)

    for group, na in groups.items():
        key = (group, "embedding") if group != "stalker" else model.out_key
        val = out.get(key)
        assert val is not None, f"Missing output for {group}"
        assert val.dim() == 3, f"{group}: expected 3D output, got {val.dim()}D"
        assert val.shape[1] == na, f"{group}: agent dim mismatch"
        if group == "colossus":
            assert torch.allclose(val[:, -2:], torch.zeros_like(val[:, -2:])), (
                "Inactive colossus embeddings were not zeroed"
            )
        print(f"  {group}: output shape {val.shape}")

    print("  PASSED\n")


def test_without_active_mask():
    """Entity obs WITHOUT active_mask: tests the hardened fallback."""
    print("=== Test 2: HeteroGNN without active_mask (fallback path) ===")
    groups = {"stalker": 8, "zealot": 8, "colossus": 4}
    n_enemies, entity_dim = 10, 9

    spec = _build_smacv2_like_spec(groups, n_enemies, entity_dim)
    model = _build_heterognn(spec, groups)

    B = 4
    td = _build_tensordict(groups, n_enemies, entity_dim, batch_dims=(B,), include_mask=False)
    out = model(td)

    for group, na in groups.items():
        key = (group, "embedding") if group != "stalker" else model.out_key
        val = out.get(key)
        assert val is not None, f"Missing output for {group}"
        assert val.dim() == 3, f"{group}: expected 3D output, got {val.dim()}D"
        assert val.shape[1] == na, f"{group}: agent dim mismatch"
        print(f"  {group}: output shape {val.shape}")

    print("  PASSED\n")


def test_vmap_like_extra_dim():
    """Simulates the vmap path: an extra leading dimension on all tensors."""
    print("=== Test 3: HeteroGNN with vmap-like extra leading dim ===")
    groups = {"stalker": 8, "zealot": 8, "colossus": 4}
    n_enemies, entity_dim = 10, 9

    spec = _build_smacv2_like_spec(groups, n_enemies, entity_dim)
    model = _build_heterognn(spec, groups)

    V, B = 2, 4  # vmap dim + batch dim
    td_with = _build_tensordict(groups, n_enemies, entity_dim, batch_dims=(V, B), include_mask=True)
    out_with = model(td_with)
    for group, _na in groups.items():
        key = (group, "embedding") if group != "stalker" else model.out_key
        val = out_with.get(key)
        assert val is not None, f"Missing output for {group}"
        assert val.dim() == 4, f"{group}: expected 4D output with vmap, got {val.dim()}D"
        print(f"  {group} (with mask): output shape {val.shape}")

    td_without = _build_tensordict(groups, n_enemies, entity_dim, batch_dims=(V, B), include_mask=False)
    out_without = model(td_without)
    for group, _na in groups.items():
        key = (group, "embedding") if group != "stalker" else model.out_key
        val = out_without.get(key)
        assert val is not None, f"Missing output for {group}"
        assert val.dim() == 4, f"{group}: expected 4D output with vmap, got {val.dim()}D"
        print(f"  {group} (no mask):   output shape {val.shape}")

    print("  PASSED\n")


def test_flat_obs_only():
    """Flat observations only (PowerGridworld-like) — regression check."""
    print("=== Test 4: Flat observations only (regression) ===")
    groups = {"EV": 10, "PV": 10, "Storage": 10}
    obs_dim = 30

    spec = Composite(device="cpu")
    for group, n_agents in groups.items():
        spec[group] = Composite(
            observation=Unbounded(shape=(obs_dim,)),
            active_mask=Unbounded(shape=(), dtype=torch.bool),
            device="cpu",
        ).expand(n_agents)

    model = _build_heterognn(spec, groups, output_features=16)

    B = 4
    td = TensorDict({}, batch_size=(B,))
    for group, na in groups.items():
        td.set((group, "observation"), torch.randn(B, na, obs_dim))
        td.set((group, "active_mask"), torch.ones(B, na, dtype=torch.bool))
    out = model(td)

    for group, na in groups.items():
        key = (group, "embedding") if group != "EV" else model.out_key
        val = out.get(key)
        assert val is not None, f"Missing output for {group}"
        assert val.shape == (B, na, 16), f"{group}: shape mismatch {val.shape}"
        print(f"  {group}: output shape {val.shape}")

    print("  PASSED\n")


def test_full_topology_unequal_groups():
    """Full topology with unequal groups catches per-group edge-index bugs."""
    print("=== Test 5: Full topology with unequal group sizes ===")
    groups = {"stalker": 8, "zealot": 8, "colossus": 4}
    n_enemies, entity_dim = 10, 9

    spec = _build_smacv2_like_spec(groups, n_enemies, entity_dim)
    model = _build_heterognn(spec, groups, topology="full")

    B = 3
    td = _build_tensordict(groups, n_enemies, entity_dim, batch_dims=(B,), include_mask=True)
    out = model(td)

    for group, na in groups.items():
        key = (group, "embedding") if group != "stalker" else model.out_key
        val = out.get(key)
        assert val is not None, f"Missing output for {group}"
        assert val.shape == (B, na, 16), f"{group}: shape mismatch {val.shape}"
        assert torch.isfinite(val).all(), f"{group}: non-finite output"
        print(f"  {group}: output shape {val.shape}")

    print("  PASSED\n")


def test_vpp_continuous_action_edge_features():
    """Continuous VPP-style actions should still work as edge features."""
    print("=== Test 6: VPP-style continuous action edge features ===")
    groups = {"EV": 10, "PV": 7, "Storage": 5}
    obs_dim = 30
    action_dim = 1

    spec = Composite(device="cpu")
    action_spec = Composite(device="cpu")
    for group, n_agents in groups.items():
        spec[group] = Composite(
            observation=Unbounded(shape=(obs_dim,)),
            active_mask=Unbounded(shape=(), dtype=torch.bool),
            device="cpu",
        ).expand(n_agents)
        action_spec[group] = Composite(
            action=Unbounded(shape=(action_dim,)),
            device="cpu",
        ).expand(n_agents)

    model = _build_heterognn(
        spec,
        groups,
        output_features=16,
        topology="full",
        action_spec=action_spec,
        edge_features_dims={"interaction": action_dim},
        use_action_edge_features=True,
    )

    B = 4
    td = TensorDict({}, batch_size=(B,))
    for group, na in groups.items():
        td.set((group, "observation"), torch.randn(B, na, obs_dim))
        td.set((group, "active_mask"), torch.ones(B, na, dtype=torch.bool))
        td.set((group, "action"), torch.randn(B, na, action_dim))
    out = model(td)

    for group, na in groups.items():
        key = (group, "embedding") if group != "EV" else model.out_key
        val = out.get(key)
        assert val is not None, f"Missing output for {group}"
        assert val.shape == (B, na, 16), f"{group}: shape mismatch {val.shape}"
        assert torch.isfinite(val).all(), f"{group}: non-finite output"
        print(f"  {group}: output shape {val.shape}")

    print("  PASSED\n")


if __name__ == "__main__":
    test_with_active_mask()
    test_without_active_mask()
    test_vmap_like_extra_dim()
    test_flat_obs_only()
    test_full_topology_unequal_groups()
    test_vpp_continuous_action_edge_features()
    print("All HeteroGNN entity-obs smoke tests passed!")
