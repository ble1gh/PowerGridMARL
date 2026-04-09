#!/usr/bin/env python
"""Test that retain_grad() on embedding_z and participation scores works.

This is a minimal test that constructs the actor pipeline for concat mode,
runs a forward pass, computes a loss, calls backward, and checks that
.grad is populated on the retained tensors.
"""

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential

# Minimal mock of the concat-mode pipeline
# Actor: GNN -> EmbeddingProcessor -> Concat -> MLP -> logits


class FakeGNN(nn.Module):
    """Minimal GNN-like module: reads participation scores, outputs gnn_embedding."""

    def __init__(self, input_dim, part_dim, output_dim):
        super().__init__()
        self.lin = nn.Linear(input_dim + part_dim, output_dim)

    def forward(self, obs, participation):
        # participation: (..., n_agents, part_dim)
        combined = torch.cat([obs, participation], dim=-1)
        return self.lin(combined)


class FakeEmbeddingProcessor(nn.Module):
    def forward(self, gnn_embedding):
        return gnn_embedding, None  # z, logvar


def test_retain_grad_embedding_z():
    """Test that embedding_z.retain_grad() captures gradients after backward."""
    torch.manual_seed(42)
    obs_dim, emb_dim, n_agents, batch = 4, 3, 2, 2

    # Build a simple pipeline similar to concat mode
    gnn = FakeGNN(obs_dim, 1, emb_dim)
    gnn_module = TensorDictModule(
        gnn,
        in_keys=[("group", "observation"), "group_participation_score"],
        out_keys=[("group", "gnn_embedding")],
    )

    proc = FakeEmbeddingProcessor()
    proc_module = TensorDictModule(
        proc,
        in_keys=[("group", "gnn_embedding")],
        out_keys=[("group", "embedding_z"), ("group", "embedding_logvar")],
    )

    def concat_fn(obs, emb):
        return torch.cat([obs, emb], dim=-1)

    concat_module = TensorDictModule(
        concat_fn,
        in_keys=[("group", "observation"), ("group", "embedding_z")],
        out_keys=[("group", "concat_input")],
    )

    mlp = nn.Linear(obs_dim + emb_dim, 1)
    mlp_module = TensorDictModule(
        mlp, in_keys=[("group", "concat_input")], out_keys=[("group", "logits")]
    )

    actor = TensorDictSequential(gnn_module, proc_module, concat_module, mlp_module)

    # Build tensordict with participation scores
    td = TensorDict(
        {
            "group": TensorDict(
                {
                    "observation": torch.randn(batch, n_agents, obs_dim),
                },
                batch_size=[batch, n_agents],
            ),
            "group_participation_score": torch.randn(batch, n_agents, 1),
        },
        batch_size=[batch],
    )

    # --- Simulate what HGTeamLoss.forward now does ---

    # PRE: clone participation score and enable grad
    part = td.get("group_participation_score")
    part_clone = part.clone().detach().requires_grad_(True)
    td.set("group_participation_score", part_clone)
    part_clone.retain_grad()

    # FORWARD: run actor
    td = actor(td)

    # POST: retain_grad on embedding_z
    emb_z = td.get(("group", "embedding_z"))
    assert emb_z is not None, "embedding_z not in tensordict"
    assert emb_z.requires_grad, "embedding_z should require grad (computed from params)"
    emb_z.retain_grad()

    # Compute a dummy loss and backward
    logits = td.get(("group", "logits"))
    loss = logits.sum()
    loss.backward()

    # CHECK: gradients should be populated
    assert emb_z.grad is not None, "embedding_z.grad should not be None after backward"
    z_norm = emb_z.grad.norm().item()
    print(f"  embedding_z grad norm: {z_norm:.6f}")
    assert z_norm > 0, "embedding_z grad norm should be > 0"

    assert part_clone.grad is not None, "participation_score.grad should not be None after backward"
    p_norm = part_clone.grad.norm().item()
    print(f"  participation_score grad norm: {p_norm:.6f}")
    assert p_norm > 0, "participation_score grad norm should be > 0"

    print("TEST: retain_grad captures both gradient norms ... PASS")


def test_retain_grad_without_pre_setup():
    """Show that doing retain_grad AFTER forward (like old code) fails for participation."""
    torch.manual_seed(42)
    obs_dim, emb_dim, n_agents, batch = 4, 3, 2, 2

    gnn = FakeGNN(obs_dim, 1, emb_dim)
    gnn_module = TensorDictModule(
        gnn,
        in_keys=[("group", "observation"), "group_participation_score"],
        out_keys=[("group", "gnn_embedding")],
    )

    proc = FakeEmbeddingProcessor()
    proc_module = TensorDictModule(
        proc,
        in_keys=[("group", "gnn_embedding")],
        out_keys=[("group", "embedding_z"), ("group", "embedding_logvar")],
    )

    def concat_fn(obs, emb):
        return torch.cat([obs, emb], dim=-1)

    concat_module = TensorDictModule(
        concat_fn,
        in_keys=[("group", "observation"), ("group", "embedding_z")],
        out_keys=[("group", "concat_input")],
    )

    mlp = nn.Linear(obs_dim + emb_dim, 1)
    mlp_module = TensorDictModule(
        mlp, in_keys=[("group", "concat_input")], out_keys=[("group", "logits")]
    )

    actor = TensorDictSequential(gnn_module, proc_module, concat_module, mlp_module)

    td = TensorDict(
        {
            "group": TensorDict(
                {
                    "observation": torch.randn(batch, n_agents, obs_dim),
                },
                batch_size=[batch, n_agents],
            ),
            "group_participation_score": torch.randn(batch, n_agents, 1),
        },
        batch_size=[batch],
    )

    # WRONG: run forward FIRST, then try to setup grad (old broken approach)
    td = actor(td)

    part = td.get("group_participation_score")
    part_clone = part.clone().detach().requires_grad_(True)
    td.set("group_participation_score", part_clone)
    part_clone.retain_grad()

    logits = td.get(("group", "logits"))
    loss = logits.sum()
    loss.backward()

    # The clone is NOT in the computation graph, so grad is None
    assert part_clone.grad is None, "Expected None — clone after forward is not in graph"
    print("TEST: retain_grad AFTER forward fails for participation (expected) ... PASS")


if __name__ == "__main__":
    test_retain_grad_embedding_z()
    print()
    test_retain_grad_without_pre_setup()
    print()
    print("✅ All gradient tracking tests passed!")
