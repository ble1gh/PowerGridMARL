import os
import sys

import torch

# Ensure BenchMARL is in path
sys.path.append(os.path.join(os.getcwd(), "BenchMARL"))


def test_mat_init_and_forward():
    print("Testing MAT Algorithm initialization and forward pass...")

    # Let's test the MultiAgentTransformer model directly first as it's the core.
    from benchmarl.algorithms.mat import MultiAgentTransformer

    n_agent = 5
    obs_dim = 10
    action_dim = 2

    print(
        f"Creating MultiAgentTransformer(n_agent={n_agent}, obs_dim={obs_dim}, action_dim={action_dim})..."
    )
    model = MultiAgentTransformer(
        n_agent=n_agent,
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_block=2,
        n_embd=32,
        n_head=4,
        action_type="continuous",
        device="cpu",
    )
    print("Model created successfully.")

    # Test Forward Pass
    B = 4
    obs = torch.randn(B, n_agent, obs_dim)
    action = torch.randn(B, n_agent, action_dim)

    print("Testing get_rep...")
    rep = model.get_rep(obs)
    print(f"Rep shape: {rep.shape} (Expected: ({B}, {n_agent}, 32))")
    assert rep.shape == (B, n_agent, 32)

    print("Testing get_value...")
    val = model.get_value(obs)
    print(f"Value shape: {val.shape} (Expected: ({B}, {n_agent}, 1))")
    assert val.shape == (B, n_agent, 1)

    print("Testing get_logits (Parallel training mode)...")
    logits = model.get_logits(rep, action)
    print(f"Logits shape: {logits.shape} (Expected: ({B}, {n_agent}, {action_dim}))")
    assert logits.shape == (B, n_agent, action_dim)

    print("\nMAT Model Unit Tests Passed!")


if __name__ == "__main__":
    test_mat_init_and_forward()
