#!/usr/bin/env python
"""Smoke test for smacv2_variable environment wrapper.

Tests Steps 1-3 of the Implementation Timeline:
  1. Skeleton + configs created ✓ (file existence)
  2. Single episode runs, group_map correct, active_mask derived, shapes match
  3. HGTeam GNN forward pass works with SMACv2 variable-composition tensordicts

Uses a FakeSMACv2BaseEnv mock (no StarCraft binary required).
"""

import sys
import traceback

import torch
from tensordict import TensorDict
from torchrl.data import Categorical, Composite, Unbounded
from torchrl.envs import EnvBase


# ====================================================================
#  Mock base environment (mimics SMACv2Env interface)
# ====================================================================


class FakeSMACv2BaseEnv(EnvBase):
    """Mock that produces tensordicts matching SMACv2Env's structure."""

    def __init__(
        self,
        n_units=10,
        n_enemies=10,
        obs_dim=120,
        n_actions=18,
        state_dim=200,
        unit_types=None,
        max_steps=50,
        device="cpu",
    ):
        super().__init__(device=device, batch_size=torch.Size([]))
        self.n_units = n_units
        self._obs_dim = obs_dim
        self._n_actions = n_actions
        self._state_dim = state_dim
        self._step_count = 0
        self._max_steps = max_steps

        # Unit types: list of ints (0=stalker, 1=zealot, 2=colossus)
        if unit_types is None:
            ns = round(n_units * 0.45)
            nz = round(n_units * 0.45)
            nc = n_units - ns - nz
            self.unit_types = [0] * ns + [1] * nz + [2] * nc
        else:
            self.unit_types = unit_types

        # Alive tracking
        self._alive = torch.ones(n_units, dtype=torch.bool, device=device)

        # Specs
        self.observation_spec = Composite(
            agents=Composite(
                observation=Unbounded(shape=(n_units, obs_dim)),
                action_mask=Unbounded(
                    shape=(n_units, n_actions), dtype=torch.bool
                ),
                device=device,
            ),
            state=Unbounded(shape=(state_dim,)),
            info=Composite(
                battle_won=Unbounded(shape=(1,), dtype=torch.bool),
                episode_limit=Unbounded(shape=(1,), dtype=torch.bool),
                device=device,
            ),
            device=device,
        )
        self.action_spec = Composite(
            agents=Composite(
                action=Categorical(n=n_actions, shape=(n_units,)),
                device=device,
            ),
            device=device,
        )
        self.reward_spec = Composite(
            agents=Composite(
                reward=Unbounded(shape=(n_units, 1)),
                device=device,
            ),
            device=device,
        )
        self.done_spec = Composite(
            done=Unbounded(shape=(1,), dtype=torch.bool),
            terminated=Unbounded(shape=(1,), dtype=torch.bool),
            device=device,
        )

    @property
    def group_map(self):
        return {"agents": [f"agent_{i}" for i in range(self.n_units)]}

    @property
    def episode_limit(self):
        return self._max_steps

    def _reset(self, tensordict=None, **kwargs):
        self._step_count = 0
        self._alive = torch.ones(
            self.n_units, dtype=torch.bool, device=self.device
        )

        obs = torch.randn(
            self.n_units, self._obs_dim, device=self.device
        )
        action_mask = torch.ones(
            self.n_units, self._n_actions, dtype=torch.bool, device=self.device
        )
        state = torch.randn(self._state_dim, device=self.device)

        return TensorDict(
            {
                ("agents", "observation"): obs,
                ("agents", "action_mask"): action_mask,
                "state": state,
                ("info", "battle_won"): torch.tensor(
                    [False], device=self.device
                ),
                ("info", "episode_limit"): torch.tensor(
                    [False], device=self.device
                ),
                "done": torch.tensor([False], device=self.device),
                "terminated": torch.tensor([False], device=self.device),
            },
            batch_size=torch.Size([]),
            device=self.device,
        )

    def _step(self, tensordict):
        self._step_count += 1

        # Randomly kill some units after step 5
        if self._step_count > 5:
            kill_mask = torch.rand(self.n_units, device=self.device) < 0.15
            self._alive[kill_mask] = False

        obs = torch.randn(
            self.n_units, self._obs_dim, device=self.device
        )
        obs[~self._alive] = 0.0

        action_mask = torch.ones(
            self.n_units, self._n_actions, dtype=torch.bool, device=self.device
        )
        # Dead units: only noop valid
        action_mask[~self._alive] = False
        action_mask[~self._alive, 0] = True

        state = torch.randn(self._state_dim, device=self.device)
        reward = torch.zeros(
            self.n_units, 1, device=self.device
        )
        reward[self._alive] = 0.1  # small positive reward for alive

        done = self._step_count >= self._max_steps or not self._alive.any()

        return TensorDict(
            {
                ("agents", "observation"): obs,
                ("agents", "action_mask"): action_mask,
                ("agents", "reward"): reward,
                "state": state,
                ("info", "battle_won"): torch.tensor(
                    [not self._alive.any()], device=self.device
                ),
                ("info", "episode_limit"): torch.tensor(
                    [self._step_count >= self._max_steps], device=self.device
                ),
                "done": torch.tensor([done], device=self.device),
                "terminated": torch.tensor([done], device=self.device),
            },
            batch_size=torch.Size([]),
            device=self.device,
        )

    def _set_seed(self, seed):
        torch.manual_seed(seed)


# ====================================================================
#  Test helpers
# ====================================================================


def _random_actions(td, type_names):
    """Generate random valid actions for each type from action_mask."""
    for tn in type_names:
        am = td.get((tn, "action_mask"))
        n = am.shape[0]
        actions = torch.zeros(n, dtype=torch.long, device=am.device)
        for i in range(n):
            valid = am[i].nonzero(as_tuple=True)[0]
            actions[i] = valid[torch.randint(len(valid), (1,))]
        td.set((tn, "action"), actions)
    return td


# ====================================================================
#  Tests
# ====================================================================


def test_file_existence():
    """Step 1: Verify all created files exist and are importable."""
    import importlib

    # common.py
    mod = importlib.import_module(
        "benchmarl.environments.smacv2_variable.common"
    )
    assert hasattr(mod, "Smacv2VariableTask")
    assert hasattr(mod, "Smacv2VariableClass")
    assert hasattr(mod, "VariableSmacv2Env")

    # Task enum has all 4 scenarios
    assert hasattr(mod.Smacv2VariableTask, "PROTOSS_10_VS_10")
    assert hasattr(mod.Smacv2VariableTask, "PROTOSS_10_VS_11")
    assert hasattr(mod.Smacv2VariableTask, "PROTOSS_20_VS_20")
    assert hasattr(mod.Smacv2VariableTask, "PROTOSS_20_VS_23")

    # Registered in BenchMARL
    from benchmarl.environments import task_config_registry

    assert "smacv2_variable/protoss_10_vs_10" in task_config_registry
    assert "smacv2_variable/protoss_20_vs_23" in task_config_registry

    print("  [PASS] File existence & registration")


def test_env_creation():
    """Step 2a: Verify wrapper env creation and spec structure."""
    from benchmarl.environments.smacv2_variable.common import VariableSmacv2Env

    base = FakeSMACv2BaseEnv(n_units=10, obs_dim=120, n_actions=18)
    env = VariableSmacv2Env(
        base_env=base,
        max_per_type={"stalker": 8, "zealot": 8, "colossus": 4},
    )

    # Group map
    gm = env.group_map
    assert set(gm.keys()) == {"stalker", "zealot", "colossus"}
    assert len(gm["stalker"]) == 8
    assert len(gm["zealot"]) == 8
    assert len(gm["colossus"]) == 4

    # Observation spec structure
    obs_spec = env.observation_spec
    for tn in ("stalker", "zealot", "colossus"):
        assert (tn, "observation") in obs_spec.keys(True, True), f"Missing ({tn}, observation)"
        assert (tn, "active_mask") in obs_spec.keys(True, True), f"Missing ({tn}, active_mask)"

    # Participation scores
    for tn in ("stalker", "zealot", "colossus"):
        key = f"{tn}_participation_score"
        assert key in obs_spec.keys(True, True), f"Missing {key}"

    # Flat active mask
    assert "active_mask" in obs_spec.keys(True, True)

    # Action spec
    act_spec = env.action_spec
    for tn in ("stalker", "zealot", "colossus"):
        assert (tn, "action") in act_spec.keys(True, True)

    print("  [PASS] Env creation & specs")


def test_reset_shapes():
    """Step 2b: Verify reset produces correctly shaped tensordicts."""
    from benchmarl.environments.smacv2_variable.common import VariableSmacv2Env

    base = FakeSMACv2BaseEnv(n_units=10, obs_dim=120, n_actions=18)
    env = VariableSmacv2Env(
        base_env=base,
        max_per_type={"stalker": 8, "zealot": 8, "colossus": 4},
    )

    td = env.reset()

    # Per-type observation shapes
    assert td.get(("stalker", "observation")).shape == (8, 120)
    assert td.get(("zealot", "observation")).shape == (8, 120)
    assert td.get(("colossus", "observation")).shape == (4, 120)

    # Per-type action mask shapes
    assert td.get(("stalker", "action_mask")).shape == (8, 18)
    assert td.get(("colossus", "action_mask")).shape == (4, 18)

    # Active masks
    assert td.get(("stalker", "active_mask")).shape == (8,)
    assert td.get(("zealot", "active_mask")).shape == (8,)
    assert td.get(("colossus", "active_mask")).shape == (4,)
    assert td.get("active_mask").shape == (20,)  # 8+8+4

    # Participation scores
    assert td.get("stalker_participation_score").shape == (8, 1)
    assert td.get("colossus_participation_score").shape == (4, 1)

    # State
    assert td.get("state").shape == (200,)

    print("  [PASS] Reset shapes")


def test_active_mask_counts():
    """Step 2c: Verify active mask correctly reflects unit type distribution."""
    from benchmarl.environments.smacv2_variable.common import VariableSmacv2Env

    # 10 units: [0]*5 + [1]*4 + [2]*1 (stalker=5, zealot=4, colossus=1)
    types = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2]
    base = FakeSMACv2BaseEnv(n_units=10, unit_types=types)
    env = VariableSmacv2Env(
        base_env=base,
        max_per_type={"stalker": 8, "zealot": 8, "colossus": 4},
    )

    td = env.reset()

    # At reset, all units are alive
    stalker_active = td.get(("stalker", "active_mask")).sum().item()
    zealot_active = td.get(("zealot", "active_mask")).sum().item()
    colossus_active = td.get(("colossus", "active_mask")).sum().item()
    total_active = td.get("active_mask").sum().item()

    assert stalker_active == 5, f"Expected 5 active stalkers, got {stalker_active}"
    assert zealot_active == 4, f"Expected 4 active zealots, got {zealot_active}"
    assert colossus_active == 1, f"Expected 1 active colossus, got {colossus_active}"
    assert total_active == 10, f"Expected 10 total active, got {total_active}"

    # Padded slots should be inactive
    stalker_mask = td.get(("stalker", "active_mask"))
    assert not stalker_mask[5:].any(), "Padded stalker slots should be inactive"

    colossus_mask = td.get(("colossus", "active_mask"))
    assert not colossus_mask[1:].any(), "Padded colossus slots should be inactive"

    # Inactive observations should be zero
    stalker_obs = td.get(("stalker", "observation"))
    assert (stalker_obs[5:] == 0).all(), "Padded stalker obs should be zero"

    print("  [PASS] Active mask counts")


def test_step_cycle():
    """Step 2d: Verify step produces correct structure and reward broadcasting."""
    from benchmarl.environments.smacv2_variable.common import VariableSmacv2Env

    base = FakeSMACv2BaseEnv(n_units=10, obs_dim=120, n_actions=18, max_steps=20)
    env = VariableSmacv2Env(
        base_env=base,
        max_per_type={"stalker": 8, "zealot": 8, "colossus": 4},
    )

    td = env.reset()

    # Run a few steps
    for step_i in range(10):
        td = _random_actions(td, env.type_names)
        td = env.step(td)
        next_td = td.get("next")

        # Check next has correct keys
        assert next_td.get(("stalker", "observation")).shape == (8, 120)
        assert next_td.get(("zealot", "observation")).shape == (8, 120)
        assert next_td.get(("colossus", "observation")).shape == (4, 120)
        assert next_td.get("done").shape == (1,)

        # Check rewards exist and have correct shape
        assert next_td.get(("stalker", "reward")).shape == (8, 1)
        assert next_td.get(("zealot", "reward")).shape == (8, 1)
        assert next_td.get(("colossus", "reward")).shape == (4, 1)

        # Inactive agents should have zero reward
        for tn in env.type_names:
            mask = next_td.get((tn, "active_mask"))
            rew = next_td.get((tn, "reward"))
            assert (rew[~mask] == 0).all(), (
                f"Step {step_i}: inactive {tn} agents have non-zero reward"
            )

        # Advance to next state for the next iteration
        td = next_td

        if next_td.get("done").item():
            break

    print("  [PASS] Step cycle (multiple steps with death dynamics)")


def test_action_merging():
    """Step 2e: Verify actions merge correctly from per-type back to flat."""
    from benchmarl.environments.smacv2_variable.common import VariableSmacv2Env

    types = [0, 0, 0, 1, 1, 1, 2]  # 3 stalkers, 3 zealots, 1 colossus
    base = FakeSMACv2BaseEnv(n_units=7, n_enemies=7, unit_types=types, obs_dim=80, n_actions=14)
    env = VariableSmacv2Env(
        base_env=base,
        max_per_type={"stalker": 5, "zealot": 5, "colossus": 3},
    )

    td = env.reset()

    # Set specific actions to verify merging
    stalker_actions = torch.tensor([1, 2, 3, 0, 0], dtype=torch.long)
    zealot_actions = torch.tensor([4, 5, 6, 0, 0], dtype=torch.long)
    colossus_actions = torch.tensor([7, 0, 0], dtype=torch.long)

    td.set(("stalker", "action"), stalker_actions)
    td.set(("zealot", "action"), zealot_actions)
    td.set(("colossus", "action"), colossus_actions)

    flat = env._merge_actions(td)
    assert flat.shape == (7,)
    expected = torch.tensor([1, 2, 3, 4, 5, 6, 7])
    assert (flat == expected).all(), f"Expected {expected}, got {flat}"

    print("  [PASS] Action merging")


def test_task_class_specs():
    """Step 2f: Verify TaskClass observation/action/state/info specs."""
    from benchmarl.environments.smacv2_variable.common import (
        Smacv2VariableClass,
        VariableSmacv2Env,
    )

    base = FakeSMACv2BaseEnv(n_units=10, obs_dim=120, n_actions=18)
    env = VariableSmacv2Env(
        base_env=base,
        max_per_type={"stalker": 8, "zealot": 8, "colossus": 4},
    )

    # Create TaskClass instance (config not used in these methods)
    tc = Smacv2VariableClass.__new__(Smacv2VariableClass)

    # group_map
    gm = tc.group_map(env)
    assert set(gm.keys()) == {"stalker", "zealot", "colossus"}

    # observation_spec: should have per-type obs, active_mask, participation
    # but NOT action_mask, state, or info
    obs_spec = tc.observation_spec(env)
    assert ("stalker", "observation") in obs_spec.keys(True, True)
    assert ("stalker", "active_mask") in obs_spec.keys(True, True)
    assert "stalker_participation_score" in obs_spec.keys(True, True)
    # action_mask removed
    assert ("stalker", "action_mask") not in obs_spec.keys(True, True)
    # state removed
    assert "state" not in obs_spec.keys()
    # info removed
    assert "info" not in obs_spec.keys()

    # action_mask_spec
    am_spec = tc.action_mask_spec(env)
    assert ("stalker", "action_mask") in am_spec.keys(True, True)

    # state_spec
    state_spec = tc.state_spec(env)
    assert state_spec is not None
    assert "state" in state_spec.keys()

    # info_spec
    info_spec = tc.info_spec(env)
    assert info_spec is not None

    print("  [PASS] TaskClass specs")


def test_20v20_scenario():
    """Step 2g: Verify 20v20 scaling scenario."""
    from benchmarl.environments.smacv2_variable.common import VariableSmacv2Env

    base = FakeSMACv2BaseEnv(
        n_units=20, n_enemies=20, obs_dim=200, n_actions=26, state_dim=400
    )
    env = VariableSmacv2Env(
        base_env=base,
        max_per_type={"stalker": 15, "zealot": 15, "colossus": 6},
    )

    td = env.reset()
    assert td.get(("stalker", "observation")).shape == (15, 200)
    assert td.get(("colossus", "observation")).shape == (6, 200)
    assert td.get("active_mask").shape == (36,)  # 15+15+6
    assert td.get("active_mask").sum().item() == 20

    # Run a few steps
    for _ in range(5):
        td = _random_actions(td, env.type_names)
        td = env.step(td)
        td = td.get("next")

    print("  [PASS] 20v20 scaling scenario")


def test_gnn_forward_pass():
    """Step 3: Verify HGTeam GNN can forward pass with SMACv2 data.

    Creates a minimal HeteroGNN with the SMACv2 variable-composition spec
    and verifies a forward pass produces correctly shaped output.
    """
    from benchmarl.environments.smacv2_variable.common import VariableSmacv2Env

    base = FakeSMACv2BaseEnv(n_units=10, obs_dim=120, n_actions=18)
    env = VariableSmacv2Env(
        base_env=base,
        max_per_type={"stalker": 8, "zealot": 8, "colossus": 4},
    )

    td = env.reset()

    # Import GNN components
    import torch_geometric.nn as tgnn
    from benchmarl.models.heterognn import HeteroGNN, HeteroGnnConfig

    group_names = list(env.max_per_type.keys())
    hidden_dim = 64
    obs_dim = 120

    # Build edge types: intra-type + inter-type (no grid)
    edge_types = []
    for g1 in group_names:
        edge_types.append((g1, f"{g1}_self_interact", g1))
        for g2 in group_names:
            if g1 != g2:
                edge_types.append((g1, f"{g1}_interact_{g2}", g2))

    # Node types: just agent types (no grid_node)
    node_types = group_names

    # Edge feature dims
    edge_features_dims = {"interaction": 0}
    for g1 in group_names:
        edge_features_dims[f"{g1}_self_interact"] = 0
        for g2 in group_names:
            if g1 != g2:
                edge_features_dims[f"{g1}_interact_{g2}"] = 0

    # Build input/output specs for the GNN (only stalker group for this model instance)
    # Mimic HGTeam's _obs_spec_for_model: group sub-spec with shape=(n_agents,)
    primary_group = "stalker"
    mc = env.max_per_type[primary_group]
    input_spec = Composite(
        {
            primary_group: Composite(
                {"observation": Unbounded(shape=(mc, obs_dim))},
                shape=(mc,),
            ),
        }
    )
    output_spec = Composite(
        {
            primary_group: Composite(
                {"embedding": Unbounded(shape=(mc, hidden_dim))},
                shape=(mc,),
            ),
        }
    )

    # Dummy action spec (required by Model base class)
    action_spec = Composite(
        {
            primary_group: Composite(
                {"action": Categorical(n=18, shape=(mc,))},
                shape=(mc,),
            ),
        }
    )

    # Create the GNN model with all required Model base class args
    gnn = HeteroGNN(
        topology="adjacency",
        self_loops=True,
        gnn_class=tgnn.TransformerConv,
        gnn_kwargs={"heads": 2, "concat": False, "beta": True},
        edge_features_dims=edge_features_dims,
        node_features_keys={},
        node_features_dims={},
        agent_node_index_key=None,  # No grid mapping for SMACv2
        agent_groups=group_names,
        node_types=node_types,
        edge_types=edge_types,
        exclude_observations_from_node_features=False,
        cat_observations_to_output=False,
        num_layers=2,
        prune_non_agent_final_layer=True,
        pos_features=0,
        vel_features=0,
        edge_radius=0,
        position_key=None,
        exclude_pos_from_node_features=None,
        velocity_key=None,
        edge_features_key=None,
        grid_edge_keys={},  # empty dict (non-None) to bypass adjacency check
        # Model base class arguments
        input_spec=input_spec,
        output_spec=output_spec,
        agent_group=primary_group,
        n_agents=mc,
        device="cpu",
        input_has_agent_dim=True,
        centralised=False,
        share_params=False,
        action_spec=action_spec,
        model_index=0,
        is_critic=False,
    )

    # Build input tensordict for the GNN
    gnn_input = TensorDict({}, batch_size=torch.Size([]))
    for tn in group_names:
        gnn_input.set((tn, "observation"), td.get((tn, "observation")))
        gnn_input.set((tn, "active_mask"), td.get((tn, "active_mask")))

    # Forward pass
    gnn_output = gnn(gnn_input)

    # Check output: look for stalker output
    out_keys = list(gnn_output.keys(True, True))
    stalker_out = None
    for key in out_keys:
        val = gnn_output.get(key)
        if isinstance(val, torch.Tensor) and val.shape[0] == mc:
            stalker_out = val
            break

    assert stalker_out is not None, (
        f"No stalker output found. Keys: {out_keys}"
    )
    assert stalker_out.shape[0] == mc, f"Expected {mc} stalker outputs, got {stalker_out.shape}"
    assert stalker_out.ndim == 2, f"Expected 2D output, got shape {stalker_out.shape}"
    assert stalker_out.shape[-1] > 0, "Output feature dim must be > 0"
    assert torch.isfinite(stalker_out).all(), "GNN output contains non-finite values"

    print("  [PASS] GNN forward pass with SMACv2 data")


def test_log_info():
    """Verify log_info produces expected metrics."""
    from benchmarl.environments.smacv2_variable.common import Smacv2VariableClass

    # Build a minimal batch tensordict
    batch = TensorDict(
        {
            ("next", "done"): torch.tensor([[False], [True], [False]]),
            ("next", "info", "battle_won"): torch.tensor(
                [[False], [True], [False]]
            ),
            ("next", "info", "episode_limit"): torch.tensor(
                [[False], [False], [False]]
            ),
            "active_mask": torch.ones(3, 20, dtype=torch.bool),
            ("stalker", "active_mask"): torch.ones(3, 8, dtype=torch.bool),
            ("zealot", "active_mask"): torch.ones(3, 8, dtype=torch.bool),
            ("colossus", "active_mask"): torch.ones(3, 4, dtype=torch.bool),
        },
        batch_size=torch.Size([3]),
    )

    logs = Smacv2VariableClass.log_info(batch)
    assert "collection/info/win_rate" in logs
    assert logs["collection/info/win_rate"] == 1.0  # only done episode won

    print("  [PASS] log_info metrics")


# ====================================================================
#  Main
# ====================================================================


def main():
    passed = 0
    failed = 0
    errors = []

    tests = [
        ("Step 1: File existence & registration", test_file_existence),
        ("Step 2a: Env creation & specs", test_env_creation),
        ("Step 2b: Reset shapes", test_reset_shapes),
        ("Step 2c: Active mask counts", test_active_mask_counts),
        ("Step 2d: Step cycle", test_step_cycle),
        ("Step 2e: Action merging", test_action_merging),
        ("Step 2f: TaskClass specs", test_task_class_specs),
        ("Step 2g: 20v20 scenario", test_20v20_scenario),
        ("Step 2h: log_info metrics", test_log_info),
        ("Step 3: GNN forward pass", test_gnn_forward_pass),
    ]

    for name, test_fn in tests:
        try:
            print(f"\n{name}...")
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, e, traceback.format_exc()))
            print(f"  [FAIL] {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")

    if errors:
        print(f"\nFailures:")
        for name, e, tb in errors:
            print(f"\n--- {name} ---")
            print(tb)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
