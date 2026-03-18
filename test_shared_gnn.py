#!/usr/bin/env python
"""Lightweight smoke-test for the shared actor GNN in HGTeam.

Verifies:
  1. The shared actor GNN is built once and reused across groups.
  2. Cross-group edge types exist in the GNN.
  3. A forward pass through both groups' actor pipelines produces
     embeddings and logits with correct shapes.
  4. Gradients from a dummy loss flow back into the shared GNN.

Usage:
    module load benchmarl/nightly
    python test_shared_gnn.py
"""

import torch
import torch_geometric.nn as tgnn
from types import SimpleNamespace
from tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from benchmarl.algorithms.HGTeam import HGTeam, HGTeamConfig
from benchmarl.models import MlpConfig, HeteroGnnConfig
from benchmarl.models.heterognn import HeteroGNN

DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Tiny mock environment specs: 2 groups, small grid
# ---------------------------------------------------------------------------
N_EV = 3
N_PV = 2
N_GRID = 4
OBS_DIM_EV = 5
OBS_DIM_PV = 4
ACT_DIM = 1  # scalar charge rate

group_map = {
    "EV": [f"EV-{i}" for i in range(N_EV)],
    "PV": [f"PV-{i}" for i in range(N_PV)],
}

# Build observation_spec --------------------------------------------------
observation_spec = Composite()
observation_spec["EV"] = Composite(
    {"observation": Bounded(low=-1, high=1, shape=(OBS_DIM_EV,))},
).expand(N_EV)
observation_spec["PV"] = Composite(
    {"observation": Bounded(low=-1, high=1, shape=(OBS_DIM_PV,))},
).expand(N_PV)

# Root-level graph keys the GNN needs
observation_spec["grid_node_features"] = Unbounded(shape=(N_GRID, 2))
observation_spec["line_adjacency"] = Unbounded(shape=(N_GRID, N_GRID, 3))
observation_spec["transformer_adjacency"] = Unbounded(shape=(N_GRID, N_GRID, 3))
observation_spec["switch_adjacency"] = Unbounded(shape=(N_GRID, N_GRID, 1))
observation_spec["EV_agent_grid_edge_index"] = Unbounded(shape=(2, N_EV))
observation_spec["PV_agent_grid_edge_index"] = Unbounded(shape=(2, N_PV))
observation_spec["EV_participation_score"] = Unbounded(shape=(N_EV, 1))
observation_spec["PV_participation_score"] = Unbounded(shape=(N_PV, 1))

# Build action_spec -------------------------------------------------------
action_spec = Composite()
action_spec["EV"] = Composite(
    {"action": Bounded(low=0, high=1, shape=(ACT_DIM,))},
).expand(N_EV)
action_spec["PV"] = Composite(
    {"action": Bounded(low=0, high=1, shape=(ACT_DIM,))},
).expand(N_PV)

# ---------------------------------------------------------------------------
# Build a minimal mock experiment object
# ---------------------------------------------------------------------------
experiment_config = SimpleNamespace(
    train_device=DEVICE,
    buffer_device=DEVICE,
    share_policy_params=True,
    gamma=0.99,
)

algo_config = HGTeamConfig.get_from_yaml()

actor_model_config = MlpConfig(
    num_cells=[32],
    activation_class=torch.nn.ReLU,
    layer_class=torch.nn.Linear,
)

critic_model_config = HeteroGnnConfig(
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
    node_features_keys={"grid_node": "grid_node_features", "agents": "participation_score"},
    node_features_dims={"grid_node": 2, "agents": 1},
    agent_node_index_key="agent_grid_edge_index",
    exclude_observations_from_node_features=True,
    cat_observations_to_output=False,
    num_layers=2,
    prune_non_agent_final_layer=True,
    gnn_hidden_dim=8,
    pos_features=0,
    vel_features=0,
    edge_radius=0,
)

mock_experiment = SimpleNamespace(
    config=experiment_config,
    model_config=actor_model_config,
    critic_model_config=critic_model_config,
    on_policy=True,
    group_map=group_map,
    observation_spec=observation_spec,
    action_spec=action_spec,
    state_spec=None,
    action_mask_spec=None,
    algorithm_config=algo_config,
)

# ---------------------------------------------------------------------------
# Instantiate HGTeam
# ---------------------------------------------------------------------------
algo = HGTeam(experiment=mock_experiment, **algo_config.__dict__)

# ===========================================================================
# TEST 1: Shared GNN is built once and reused
# ===========================================================================
print("TEST 1: Shared actor GNN is built once and reused ... ", end="")
policy_ev = algo._get_policy_for_loss("EV", actor_model_config, continuous=True)
gnn_after_ev = algo._shared_actor_gnn
assert gnn_after_ev is not None, "Shared actor GNN should be created after first group"

policy_pv = algo._get_policy_for_loss("PV", actor_model_config, continuous=True)
gnn_after_pv = algo._shared_actor_gnn
assert gnn_after_pv is gnn_after_ev, "Same GNN instance must be reused for second group"
print("PASS")

# ===========================================================================
# TEST 2: Cross-group edge types exist
# ===========================================================================
print("TEST 2: Cross-group edge types exist in GNN ... ", end="")
# Find the HeteroGNN inside the shared module
shared_gnn = algo._shared_actor_gnn
hetero_gnns = [m for m in shared_gnn.modules() if isinstance(m, HeteroGNN)]
assert len(hetero_gnns) == 1, f"Expected 1 HeteroGNN, got {len(hetero_gnns)}"
gnn = hetero_gnns[0]

edge_types = gnn.edge_types
# Check for cross-group interaction edges
cross_edges = [(s, r, d) for (s, r, d) in edge_types if s != d and s in group_map and d in group_map]
assert len(cross_edges) >= 2, (
    f"Expected cross-group edges (EV→PV, PV→EV), got: {cross_edges}"
)
print(f"PASS — cross-group edges: {cross_edges}")

# Check both groups appear as node types
assert "EV" in gnn.agent_groups, f"EV not in agent_groups: {gnn.agent_groups}"
assert "PV" in gnn.agent_groups, f"PV not in agent_groups: {gnn.agent_groups}"
print(f"       agent_groups: {gnn.agent_groups}")

# ===========================================================================
# TEST 3: Forward pass produces correct shapes
# ===========================================================================
print("TEST 3: Forward pass produces embeddings and logits ... ", end="")
B = 2  # batch size

# Build a fake TensorDict matching the observation spec
td = TensorDict({
    "EV": TensorDict({
        "observation": torch.randn(B, N_EV, OBS_DIM_EV),
    }, batch_size=[B, N_EV]),
    "PV": TensorDict({
        "observation": torch.randn(B, N_PV, OBS_DIM_PV),
    }, batch_size=[B, N_PV]),
    "grid_node_features": torch.randn(B, N_GRID, 2),
    "line_adjacency": torch.zeros(B, N_GRID, N_GRID, 3),
    "transformer_adjacency": torch.zeros(B, N_GRID, N_GRID, 3),
    "switch_adjacency": torch.zeros(B, N_GRID, N_GRID, 1),
    # Simple mapping: agent i → grid node i (mod N_GRID)
    "EV_agent_grid_edge_index": torch.stack([
        torch.arange(N_EV), torch.arange(N_EV) % N_GRID
    ]).unsqueeze(0).expand(B, 2, N_EV),
    "PV_agent_grid_edge_index": torch.stack([
        torch.arange(N_PV), torch.arange(N_PV) % N_GRID
    ]).unsqueeze(0).expand(B, 2, N_PV),
    "EV_participation_score": torch.rand(B, N_EV, 1),
    "PV_participation_score": torch.rand(B, N_PV, 1),
}, batch_size=[B])

# Make some adjacency edges non-zero in line_adjacency
for i in range(N_GRID - 1):
    td["line_adjacency"][:, i, i + 1, :] = torch.tensor([1.0, 0.5, 0.1])
    td["line_adjacency"][:, i + 1, i, :] = torch.tensor([1.0, 0.5, 0.1])

# Run EV policy forward
td_ev = td.clone()
policy_ev(td_ev)
assert ("EV", "action") in td_ev.keys(True), "EV action not produced"
assert td_ev["EV", "action"].shape == (B, N_EV, ACT_DIM), (
    f"EV action shape wrong: {td_ev['EV', 'action'].shape}"
)
# Check embedding was produced
assert ("EV", "embedding_z") in td_ev.keys(True) or ("EV", "gnn_embedding") in td_ev.keys(True), (
    "No EV embedding produced"
)
print("PASS")
print(f"       EV action shape: {td_ev['EV', 'action'].shape}")

# Also check PV forward
td_pv = td.clone()
policy_pv(td_pv)
assert ("PV", "action") in td_pv.keys(True), "PV action not produced"
assert td_pv["PV", "action"].shape == (B, N_PV, ACT_DIM), (
    f"PV action shape wrong: {td_pv['PV', 'action'].shape}"
)
print(f"       PV action shape: {td_pv['PV', 'action'].shape}")

# ===========================================================================
# TEST 4: Gradients flow through shared GNN from both groups
# ===========================================================================
print("TEST 4: Gradients flow from both groups into shared GNN ... ", end="")

# Zero all grads
for p in shared_gnn.parameters():
    if p.grad is not None:
        p.grad.zero_()

# Forward + backward from EV
td_grad = td.clone()
policy_ev(td_grad)
ev_logits = td_grad["EV", "logits"]
loss_ev = ev_logits.sum()
loss_ev.backward(retain_graph=True)

# Check at least one GNN param has a gradient
gnn_params = list(shared_gnn.parameters())
ev_grad_norms = [p.grad.norm().item() for p in gnn_params if p.grad is not None]
assert len(ev_grad_norms) > 0, "No gradients from EV loss reached the shared GNN"
ev_total = sum(ev_grad_norms)

# Now also backward from PV
td_grad2 = td.clone()
policy_pv(td_grad2)
pv_logits = td_grad2["PV", "logits"]
loss_pv = pv_logits.sum()
loss_pv.backward()

pv_grad_norms = [p.grad.norm().item() for p in gnn_params if p.grad is not None]
pv_total = sum(pv_grad_norms)
assert pv_total > ev_total, (
    "PV backward should have added to EV gradients (accumulated), "
    f"but total went from {ev_total:.6f} to {pv_total:.6f}"
)

print("PASS")
print(f"       GNN grad norm after EV: {ev_total:.6f}")
print(f"       GNN grad norm after EV+PV: {pv_total:.6f}")
print(f"       Shared GNN params: {sum(p.numel() for p in gnn_params)}")

print("\n✅ All tests passed!")
