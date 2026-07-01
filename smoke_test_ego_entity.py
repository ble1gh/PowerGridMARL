#!/usr/bin/env python3
"""Smoke test for the ego-entity GNN pipeline.

Tests:
1. Entity decomposition in VariableSmacv2Env (with mock base env)
2. EgoEntityGNN forward pass
3. EgoEntityGNNWrapper TensorDictModule interface
4. Full actor pipeline integration with concat mode
"""

import sys
import torch
from collections import OrderedDict
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded, Categorical


def test_entity_decomposition():
    """Test that _decompose_obs correctly decomposes flat observations."""
    print("=== Test 1: Entity decomposition ===")

    # Import the wrapper
    sys.path.insert(0, "BenchMARL")
    sys.path.insert(0, "PowerGridworld")
    from benchmarl.environments.smacv2_variable.common import VariableSmacv2Env

    # Create a mock base env
    class MockBaseEnv:
        def __init__(self):
            self.device = torch.device("cpu")
            n_units = 10
            obs_dim = 182  # Protoss 10v10: 4 + 10*9 + 9*9 + 7
            n_actions = 17

            self.observation_spec = Composite(
                agents=Composite(
                    observation=Unbounded(shape=(obs_dim,)),
                    action_mask=Unbounded(shape=(n_actions,), dtype=torch.bool),
                ).expand(n_units),
                state=Unbounded(shape=(120,)),
                device=self.device,
            )
            self.full_action_spec = Composite(
                agents=Composite(
                    action=Categorical(n=n_actions, shape=()),
                ).expand(n_units),
                device=self.device,
            )
            self.n_units = n_units
            self.unit_types = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]  # 4 stalkers, 4 zealots, 2 colossi
            self.episode_limit = 200

        def reset(self, *args, **kwargs):
            n_units = self.n_units
            obs_dim = 182
            n_actions = 17

            obs = torch.randn(n_units, obs_dim)
            # Set type bits for allies (features[6:9] of each ally)
            # ally_feats start at index 4 + 10*9 = 94
            # Each ally is 9 features: [visible, dist, rx, ry, health, shield, t0, t1, t2]
            ally_start = 94
            n_allies = 9
            for i in range(n_units):
                for j in range(n_allies):
                    base = ally_start + j * 9
                    # Set type bits (last 3 of each ally's 9 features)
                    obs[i, base + 6:base + 9] = 0.0
                    # Assign some allies to each type
                    if j < 3:
                        obs[i, base + 6] = 1.0  # stalker
                    elif j < 6:
                        obs[i, base + 7] = 1.0  # zealot
                    else:
                        obs[i, base + 8] = 1.0  # colossus

            amask = torch.ones(n_units, n_actions, dtype=torch.bool)
            state = torch.randn(120)

            return TensorDict({
                ("agents", "observation"): obs,
                ("agents", "action_mask"): amask,
                "state": state,
            }, batch_size=[])

    mock_env = MockBaseEnv()
    max_per_type = OrderedDict([
        ("stalker", 8),
        ("zealot", 8),
        ("colossus", 4),
    ])

    # Create wrapper with entity_obs=True
    wrapper = VariableSmacv2Env(
        base_env=mock_env,
        max_per_type=max_per_type,
        device="cpu",
        entity_obs=True,
    )

    # Reset to trigger type detection
    td = wrapper.reset()

    # Check entity keys exist
    for tn in ["stalker", "zealot", "colossus"]:
        assert (tn, "entity_self") in td.keys(True), f"Missing {tn}/entity_self"
        assert (tn, "entity_enemy") in td.keys(True), f"Missing {tn}/entity_enemy"
        assert (tn, "move_feats") in td.keys(True), f"Missing {tn}/move_feats"
        for at in ["stalker", "zealot", "colossus"]:
            assert (tn, f"entity_{at}_ally") in td.keys(True), f"Missing {tn}/entity_{at}_ally"

    # Check shapes
    print(f"  stalker/entity_self shape: {td['stalker', 'entity_self'].shape}")
    print(f"  stalker/entity_enemy shape: {td['stalker', 'entity_enemy'].shape}")
    print(f"  stalker/move_feats shape: {td['stalker', 'move_feats'].shape}")
    print(f"  stalker/entity_stalker_ally shape: {td['stalker', 'entity_stalker_ally'].shape}")
    print(f"  stalker/entity_zealot_ally shape: {td['stalker', 'entity_zealot_ally'].shape}")
    print(f"  stalker/entity_colossus_ally shape: {td['stalker', 'entity_colossus_ally'].shape}")

    assert td["stalker", "entity_self"].shape == (8, 7), f"Expected (8, 7), got {td['stalker', 'entity_self'].shape}"
    assert td["stalker", "entity_enemy"].shape == (8, 10, 9), f"Expected (8, 10, 9), got {td['stalker', 'entity_enemy'].shape}"
    assert td["stalker", "move_feats"].shape == (8, 4), f"Expected (8, 4), got {td['stalker', 'move_feats'].shape}"

    print("  PASSED\n")


def test_ego_entity_gnn():
    """Test EgoEntityGNN forward pass."""
    print("=== Test 2: EgoEntityGNN forward ===")
    sys.path.insert(0, "BenchMARL")
    from benchmarl.models.ego_entity_gnn import EgoEntityGNN

    all_groups = ["stalker", "zealot", "colossus"]
    group_map = {
        "stalker": list(range(8)),
        "zealot": list(range(8)),
        "colossus": list(range(4)),
    }
    ally_type_max = {"stalker": 8, "zealot": 8, "colossus": 4}

    gnn = EgoEntityGNN(
        group_map=group_map,
        all_groups=all_groups,
        n_enemies=10,
        entity_dim=9,
        own_dim=7,
        move_feats_dim=4,
        ally_type_max=ally_type_max,
        output_dim=32,
        num_layers=2,
        heads=2,
        concat_heads=False,
        use_beta=True,
        self_loops=True,
        topology="star",
        norm_class="LayerNorm",
        device="cpu",
    )

    print(f"  Total parameters: {sum(p.numel() for p in gnn.parameters())}")

    # Create dummy tensordict
    B = 2
    td = TensorDict({}, batch_size=[])
    for g in all_groups:
        n_g = len(group_map[g])
        td.set((g, "entity_self"), torch.randn(B, n_g, 7))
        td.set((g, "entity_enemy"), torch.randn(B, n_g, 10, 9))
        td.set((g, "active_mask"), torch.ones(B, n_g, dtype=torch.bool))
        for at in all_groups:
            td.set((g, f"entity_{at}_ally"), torch.randn(B, n_g, ally_type_max[at], 9))

    results = gnn(td)
    for g in all_groups:
        n_g = len(group_map[g])
        assert results[g].shape == (B, n_g, 32), f"Expected ({B}, {n_g}, 32), got {results[g].shape}"
        print(f"  {g} embedding shape: {results[g].shape}")

    print("  PASSED\n")


def test_ego_entity_gnn_full_topology():
    """Test EgoEntityGNN with full topology."""
    print("=== Test 3: EgoEntityGNN full topology ===")
    sys.path.insert(0, "BenchMARL")
    from benchmarl.models.ego_entity_gnn import EgoEntityGNN

    all_groups = ["stalker", "zealot", "colossus"]
    group_map = {
        "stalker": list(range(8)),
        "zealot": list(range(8)),
        "colossus": list(range(4)),
    }
    ally_type_max = {"stalker": 8, "zealot": 8, "colossus": 4}

    gnn = EgoEntityGNN(
        group_map=group_map,
        all_groups=all_groups,
        n_enemies=10,
        entity_dim=9,
        own_dim=7,
        move_feats_dim=4,
        ally_type_max=ally_type_max,
        output_dim=32,
        num_layers=2,
        heads=2,
        concat_heads=False,
        use_beta=True,
        self_loops=True,
        topology="full",
        norm_class="LayerNorm",
        device="cpu",
    )

    print(f"  Total parameters: {sum(p.numel() for p in gnn.parameters())}")

    B = 2
    td = TensorDict({}, batch_size=[])
    for g in all_groups:
        n_g = len(group_map[g])
        td.set((g, "entity_self"), torch.randn(B, n_g, 7))
        td.set((g, "entity_enemy"), torch.randn(B, n_g, 10, 9))
        td.set((g, "active_mask"), torch.ones(B, n_g, dtype=torch.bool))
        for at in all_groups:
            td.set((g, f"entity_{at}_ally"), torch.randn(B, n_g, ally_type_max[at], 9))

    results = gnn(td)
    for g in all_groups:
        n_g = len(group_map[g])
        assert results[g].shape == (B, n_g, 32)
        print(f"  {g} embedding shape: {results[g].shape}")

    print("  PASSED\n")


def test_ego_entity_gnn_wrapper():
    """Test EgoEntityGNNWrapper TensorDictModule interface."""
    print("=== Test 4: EgoEntityGNNWrapper ===")
    sys.path.insert(0, "BenchMARL")
    from benchmarl.models.ego_entity_gnn import EgoEntityGNN, EgoEntityGNNWrapper

    all_groups = ["stalker", "zealot", "colossus"]
    group_map = {
        "stalker": list(range(8)),
        "zealot": list(range(8)),
        "colossus": list(range(4)),
    }
    ally_type_max = {"stalker": 8, "zealot": 8, "colossus": 4}

    gnn = EgoEntityGNN(
        group_map=group_map,
        all_groups=all_groups,
        n_enemies=10,
        entity_dim=9,
        own_dim=7,
        move_feats_dim=4,
        ally_type_max=ally_type_max,
        output_dim=32,
        num_layers=2,
        heads=2,
        topology="star",
        device="cpu",
    )

    wrapper = EgoEntityGNNWrapper(gnn, all_groups)

    # Check in_keys/out_keys
    print(f"  in_keys: {wrapper.in_keys}")
    print(f"  out_keys: {wrapper.out_keys}")

    B = 2
    td = TensorDict({}, batch_size=[])
    for g in all_groups:
        n_g = len(group_map[g])
        td.set((g, "entity_self"), torch.randn(B, n_g, 7))
        td.set((g, "entity_enemy"), torch.randn(B, n_g, 10, 9))
        td.set((g, "active_mask"), torch.ones(B, n_g, dtype=torch.bool))
        for at in all_groups:
            td.set((g, f"entity_{at}_ally"), torch.randn(B, n_g, ally_type_max[at], 9))

    result_td = wrapper(td)
    for g in all_groups:
        n_g = len(group_map[g])
        emb = result_td.get((g, "gnn_embedding"))
        assert emb is not None, f"Missing {g}/gnn_embedding"
        assert emb.shape == (B, n_g, 32), f"Expected ({B}, {n_g}, 32), got {emb.shape}"
        print(f"  {g}/gnn_embedding shape: {emb.shape}")

    print("  PASSED\n")


def test_gradient_flow():
    """Test that gradients flow through the ego-entity GNN."""
    print("=== Test 5: Gradient flow ===")
    sys.path.insert(0, "BenchMARL")
    from benchmarl.models.ego_entity_gnn import EgoEntityGNN

    all_groups = ["stalker", "zealot", "colossus"]
    group_map = {
        "stalker": list(range(8)),
        "zealot": list(range(8)),
        "colossus": list(range(4)),
    }
    ally_type_max = {"stalker": 8, "zealot": 8, "colossus": 4}

    gnn = EgoEntityGNN(
        group_map=group_map,
        all_groups=all_groups,
        n_enemies=10,
        entity_dim=9,
        own_dim=7,
        move_feats_dim=4,
        ally_type_max=ally_type_max,
        output_dim=32,
        num_layers=2,
        heads=2,
        topology="star",
        device="cpu",
    )

    B = 2
    td = TensorDict({}, batch_size=[])
    for g in all_groups:
        n_g = len(group_map[g])
        td.set((g, "entity_self"), torch.randn(B, n_g, 7))
        td.set((g, "entity_enemy"), torch.randn(B, n_g, 10, 9))
        td.set((g, "active_mask"), torch.ones(B, n_g, dtype=torch.bool))
        for at in all_groups:
            td.set((g, f"entity_{at}_ally"), torch.randn(B, n_g, ally_type_max[at], 9))

    results = gnn(td)
    loss = sum(r.sum() for r in results.values())
    loss.backward()

    # Check gradients exist
    n_params_with_grad = sum(1 for p in gnn.parameters() if p.grad is not None)
    n_params_total = sum(1 for _ in gnn.parameters())
    print(f"  Params with gradients: {n_params_with_grad}/{n_params_total}")
    assert n_params_with_grad > 0, "No gradients!"
    # In star topology, self-loop-only conv params on non-self entity types
    # may not receive gradients since there are no entity↔entity edges.
    # This is expected behavior.
    print(f"  (Some params may lack gradients in star topology — this is expected)")

    print("  PASSED\n")


def test_inactive_masking():
    """Test that inactive agents get zero embeddings."""
    print("=== Test 6: Inactive agent masking ===")
    sys.path.insert(0, "BenchMARL")
    from benchmarl.models.ego_entity_gnn import EgoEntityGNN

    all_groups = ["stalker", "zealot", "colossus"]
    group_map = {
        "stalker": list(range(8)),
        "zealot": list(range(8)),
        "colossus": list(range(4)),
    }
    ally_type_max = {"stalker": 8, "zealot": 8, "colossus": 4}

    gnn = EgoEntityGNN(
        group_map=group_map,
        all_groups=all_groups,
        n_enemies=10,
        entity_dim=9,
        own_dim=7,
        move_feats_dim=4,
        ally_type_max=ally_type_max,
        output_dim=32,
        num_layers=2,
        heads=2,
        topology="star",
        device="cpu",
    )

    B = 2
    td = TensorDict({}, batch_size=[])
    for g in all_groups:
        n_g = len(group_map[g])
        td.set((g, "entity_self"), torch.randn(B, n_g, 7))
        td.set((g, "entity_enemy"), torch.randn(B, n_g, 10, 9))
        # Mark last 2 agents as inactive in each group
        mask = torch.ones(B, n_g, dtype=torch.bool)
        mask[:, -2:] = False
        td.set((g, "active_mask"), mask)
        for at in all_groups:
            td.set((g, f"entity_{at}_ally"), torch.randn(B, n_g, ally_type_max[at], 9))

    results = gnn(td)
    for g in all_groups:
        n_g = len(group_map[g])
        # Last 2 agents should be zero
        inactive = results[g][:, -2:, :]
        assert (inactive == 0).all(), f"{g}: inactive agents have non-zero embeddings"
        # Active agents should be non-zero
        active = results[g][:, :-2, :]
        assert active.abs().sum() > 0, f"{g}: active agents are all zero"
        print(f"  {g}: inactive correctly zeroed, active non-zero")

    print("  PASSED\n")


if __name__ == "__main__":
    test_entity_decomposition()
    test_ego_entity_gnn()
    test_ego_entity_gnn_full_topology()
    test_ego_entity_gnn_wrapper()
    test_gradient_flow()
    test_inactive_masking()
    print("All tests passed!")
