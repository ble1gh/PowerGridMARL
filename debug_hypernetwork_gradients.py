"""Debug script to verify gradient flow through the heterogeneous GNN pipeline.

Two complementary checks:
  1. PPO-based: run the full ClipPPOLoss forward/backward and inspect where
     gradients land (on the loss module's *_network_params TensorDicts).
  2. Direct: run the actor forward on a batch, compute a dummy scalar loss
     from the output, call backward(), and check .grad on the module params.
     This isolates the GNN architecture from any PPO / functional-param issues.

BUG FIXED: the previous version of this script checked
    loss_module.actor_network.named_parameters()
for .grad attributes after the PPO backward pass.  But TorchRL's ClipPPOLoss
runs the actor *functionally* using loss_module.actor_network_params (a
TensorDict of parameter tensors).  Gradients land on those TensorDict leaves,
NOT necessarily on the nn.Module's .parameters().  The old script therefore
reported ~half the params as "no grad" even though they were receiving
gradients perfectly fine during real training.
"""
import sys
import os
import torch
import torch_geometric.nn as tgnn
from torch import nn

sys.path.append(os.path.join(os.getcwd(), "BenchMARL"))

from benchmarl.algorithms import HGTeamConfig
from benchmarl.models import HeteroGnnConfig, MlpConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.environments.PowerGridworldVariable.common import PowerGridworldVariableTask


# ── helpers ──────────────────────────────────────────────────────────────

def check_gradient_flow_module(model, name="Model"):
    """Check .grad on nn.Module parameters (standard PyTorch)."""
    print(f"\n{'='*70}")
    print(f"Gradient Analysis (module .grad) for: {name}")
    print(f"{'='*70}")
    total = with_grad = zero = no_grad = frozen = 0
    for pname, param in model.named_parameters():
        total += 1
        if not param.requires_grad:
            frozen += 1
            print(f"  🔒 FROZEN:    {pname} | {tuple(param.shape)}")
            continue
        with_grad += 1
        if param.grad is None:
            no_grad += 1
            print(f"  ❌ NO GRAD:   {pname} | {tuple(param.shape)}")
        elif param.grad.norm().item() == 0:
            zero += 1
            print(f"  ⚠️  ZERO GRAD: {pname} | {tuple(param.shape)}")
        else:
            print(f"  ✓  OK:        {pname} | grad_norm={param.grad.norm().item():.6f} | {tuple(param.shape)}")
    ok = with_grad - no_grad - zero
    print(f"\n  Summary: {total} total | {with_grad} trainable | "
          f"{ok} ok | {no_grad} no-grad-attr | {zero} zero-grad | {frozen} frozen")
    return no_grad + zero


def check_gradient_flow_tensordict(td_params, name="Params"):
    """Check .grad on TensorDict parameter leaves (how TorchRL stores them)."""
    print(f"\n{'='*70}")
    print(f"Gradient Analysis (TensorDict leaves) for: {name}")
    print(f"{'='*70}")
    total = zero = no_grad = ok = 0
    for key, param in td_params.items(True, True):
        if not isinstance(param, torch.Tensor):
            continue
        total += 1
        if param.grad is None:
            no_grad += 1
            print(f"  ❌ NO GRAD:   {key} | {tuple(param.shape)}")
        elif param.grad.norm().item() == 0:
            zero += 1
            print(f"  ⚠️  ZERO GRAD: {key} | {tuple(param.shape)}")
        else:
            ok += 1
            print(f"  ✓  OK:        {key} | grad_norm={param.grad.norm().item():.6f} | {tuple(param.shape)}")
    print(f"\n  Summary: {total} total | {ok} ok | {no_grad} no-grad-attr | {zero} zero-grad")
    return no_grad + zero


def check_identity(module, td_params, label="actor"):
    """Verify whether module params and TensorDict leaves are the same objects."""
    mod_params = dict(module.named_parameters())
    td_flat = {}
    for key, val in td_params.items(True, True):
        if isinstance(val, torch.Tensor):
            td_flat[str(key)] = val

    print(f"\n{'='*70}")
    print(f"Identity check: {label}")
    print(f"{'='*70}")
    print(f"  Module params: {len(mod_params)}")
    print(f"  TensorDict leaves: {len(td_flat)}")

    # Build data_ptr → name map for module params
    mod_ptrs = {p.data_ptr(): n for n, p in mod_params.items()}
    shared = 0
    td_only = 0
    for td_key, td_val in td_flat.items():
        ptr = td_val.data_ptr()
        if ptr in mod_ptrs:
            shared += 1
        else:
            td_only += 1
            print(f"  ⚠️  TD-only (not same object as module param): {td_key} | {tuple(td_val.shape)}")
    print(f"  Shared (same tensor object): {shared}")
    print(f"  TD-only (different tensor):  {td_only}")
    if td_only > 0:
        print("  → Gradients on TD-only tensors will NOT appear on module.named_parameters()!")
        print("    This explains the 'NO GRAD' false positives in the old debug script.")


def check_gnn_per_layer(actor):
    """Print per-layer, per-edge-type gradient breakdown for every HeteroGNN."""
    from benchmarl.models.heterognn import HeteroGNN

    print(f"\n{'='*70}")
    print("PER-LAYER / PER-EDGE-TYPE GNN PARAMETER BREAKDOWN")
    print(f"{'='*70}")

    for mod_name, mod in actor.named_modules():
        if not isinstance(mod, HeteroGNN):
            continue
        print(f"\nFound HeteroGNN at: {mod_name}")
        print(f"  node_types: {mod.node_types}")
        print(f"  edge_types: {mod.edge_types}")
        print(f"  prune_non_agent_final_layer: {mod.prune_non_agent_final_layer}")
        print(f"  num_layers: {mod.num_layers}")

        for i, conv in enumerate(mod.convs):
            print(f"\n  --- Layer {i} ---")
            for edge_key, sub_conv in conv.convs.items():
                src, rel, dst = edge_key
                params = list(sub_conv.named_parameters())
                n_ok = sum(1 for _, p in params
                           if p.grad is not None and p.grad.norm().item() > 0)
                n_zero = sum(1 for _, p in params
                             if p.grad is not None and p.grad.norm().item() == 0)
                n_none = sum(1 for _, p in params if p.grad is None)
                print(f"    ({src}, {rel}, {dst}): "
                      f"{len(params)} params | {n_ok} ok | {n_zero} zero | {n_none} none")
                for pn, p in params:
                    if p.grad is not None and p.grad.norm().item() > 0:
                        tag = f"✓ ({p.grad.norm().item():.6f})"
                    elif p.grad is not None:
                        tag = f"⚠️ zero ({p.grad.norm().item():.6f})"
                    else:
                        tag = "❌ None"
                    print(f"      {pn}: {tuple(p.shape)} → {tag}")

        if mod.output_proj is not None:
            print(f"\n  --- output_proj ---")
            for nt, proj in mod.output_proj.items():
                for pn, p in proj.named_parameters():
                    if p.grad is not None and p.grad.norm().item() > 0:
                        tag = f"✓ ({p.grad.norm().item():.6f})"
                    elif p.grad is not None:
                        tag = f"⚠️ zero ({p.grad.norm().item():.6f})"
                    else:
                        tag = "❌ None"
                    print(f"    {nt}.{pn}: {tuple(p.shape)} → {tag}")


# ── main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("HETEROGENEOUS GNN GRADIENT FLOW DEBUGGING")
    print("=" * 70)

    # ── config ───────────────────────────────────────────────────────────
    base_actor_config = MlpConfig(
        num_cells=[128, 128],
        activation_class=nn.Tanh,
        layer_class=nn.Linear,
    )

    critic_gnn_config = HeteroGnnConfig(
        topology="adjacency",
        self_loops=True,
        gnn_class=tgnn.TransformerConv,
        gnn_kwargs={"heads": 4, "concat": False, "beta": True},
        grid_edge_keys={
            "line_adjacency": "line_adjacency",
            "transformer_adjacency": "transformer_adjacency",
            "switch_adjacency": "switch_adjacency",
        },
        edge_features_dims={
            "line_adjacency": 3, "transformer_adjacency": 3,
            "switch_adjacency": 1, "interaction": 0,
            "mapping": 0, "mapping_rev": 0,
        },
        node_features_keys={"grid_node": "grid_node_features"},
        node_features_dims={"grid_node": 2},
        agent_node_index_key="agent_grid_edge_index",
        exclude_observations_from_node_features=False,
        cat_observations_to_output=False,
        num_layers=3,
        gnn_hidden_dim=32,
        pos_features=0, vel_features=0, edge_radius=0,
    )

    algorithm_config = HGTeamConfig(
        share_param_critic=True,
        share_critic_across_groups=True,
        centralised_value_per_agent=True,
        clip_epsilon=0.2,
        entropy_coef=0.05,
        critic_coef=1.0,
        loss_critic_type="l2",
        lmbda=0.95,
        scale_mapping="biased_softplus_1",
        use_tanh_normal=False,
        use_beta=True,
        minibatch_advantage=True,
        beta_min_param=1.0,
        gnn_mode="concat",
        z_dim=32,
        hypernet_actor_feature_dim=64,
        stochastic_z=True,
        embedding_entropy_coef=0.1,
        embedding_diversity_coef=0.01,
        gnn_num_layers=2,
        gnn_heads=2,
        gnn_concat_heads=True,
        gnn_use_beta=True,
        gnn_self_loops=True,
        gnn_agent_node_feature_key="participation_score",
        gnn_agent_node_feature_dim=1,
    )

    task = PowerGridworldVariableTask.EVOVERNIGHT13NODE_SIMPLE.get_from_yaml()

    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.sampling_device = "cpu"
    experiment_config.train_device = "cpu"
    experiment_config.share_policy_params = True
    experiment_config.lr = 3e-4
    experiment_config.evaluation_episodes = 1
    experiment_config.parallel_collection = False
    experiment_config.on_policy_n_envs_per_worker = 2
    experiment_config.on_policy_collected_frames_per_batch = 128
    experiment_config.on_policy_minibatch_size = 64
    experiment_config.on_policy_n_minibatch_iters = 1
    experiment_config.max_n_frames = 128
    experiment_config.evaluation_interval = 128
    experiment_config.loggers = []

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=base_actor_config,
        critic_model_config=critic_gnn_config,
        seed=42,
        config=experiment_config,
    )

    group = "agents"
    loss_module = experiment.losses[group]
    actor = loss_module.actor_network

    # ── collect a batch ──────────────────────────────────────────────────
    print("\n1. Collecting a batch of data …")
    collector = experiment.collector
    batch = next(iter(collector))
    batch = batch.to(experiment_config.train_device)

    # ── show batch content summary ───────────────────────────────────────
    print("\n=== Key tensors in batch ===")
    for key in sorted(str(k) for k in batch.keys(True, True)):
        key_lower = key.lower()
        if any(tok in key_lower for tok in [
            "adjacency", "grid_node", "participation", "observation",
            "agent_grid", "action", "log_prob",
        ]):
            val = batch.get(key)
            if isinstance(val, torch.Tensor):
                nz = (val.abs() > 0).sum().item() if val.is_floating_point() else "N/A"
                print(f"  {key}: shape={tuple(val.shape)}  nonzero={nz}/{val.numel()}")

    # ══════════════════════════════════════════════════════════════════════
    #  CHECK 1 — PPO loss-based gradient analysis  (TensorDict params)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("CHECK 1: PPO loss → backward → TensorDict param gradients")
    print("  (this is how the real training loop works)")
    print("=" * 70)

    algorithm = experiment.algorithm
    batch_ppo = algorithm.process_batch(group, batch.clone())

    # Zero all grads on the module AND on the TensorDict leaves
    for p in loss_module.parameters():
        if p.grad is not None:
            p.grad.zero_()
    for _, p in loss_module.actor_network_params.items(True, True):
        if isinstance(p, torch.Tensor) and p.grad is not None:
            p.grad.zero_()
    for _, p in loss_module.critic_network_params.items(True, True):
        if isinstance(p, torch.Tensor) and p.grad is not None:
            p.grad.zero_()

    loss_td = loss_module(batch_ppo)
    print("\nLoss values:")
    for k, v in loss_td.items():
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            print(f"  {k}: {v.item():.6f}")

    # Backward only loss_objective — this is what the real training loop does
    # for the actor optimizer (see HGTeam._get_parameters and experiment.py)
    loss_td["loss_objective"].backward()

    # Check the TensorDict params (where TorchRL puts gradients)
    bad_actor = check_gradient_flow_tensordict(
        loss_module.actor_network_params,
        "actor_network_params (TensorDict) — after loss_objective.backward()",
    )

    # Also check the module params for comparison
    bad_module = check_gradient_flow_module(
        actor,
        "actor_network (nn.Module) — for comparison only",
    )

    # Are they the same objects?
    check_identity(actor, loss_module.actor_network_params, "actor")

    if bad_module > 0 and bad_actor == 0:
        print("\n  ★ KEY FINDING: TensorDict params all received gradients but")
        print("    module params did not. This is expected behaviour in TorchRL's")
        print("    functional parameter management. The old debug script's 'NO GRAD'")
        print("    reports were FALSE POSITIVES.")

    # ══════════════════════════════════════════════════════════════════════
    #  CHECK 2 — Direct forward/backward (bypass PPO entirely)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("CHECK 2: Direct actor forward → dummy loss → backward")
    print("  (bypasses PPO, tests pure GNN gradient flow through nn.Module)")
    print("=" * 70)

    # Zero all module grads
    actor.zero_grad()

    # Run actor forward directly on the batch
    batch_direct = batch.clone()
    with torch.enable_grad():
        out = actor(batch_direct)

    # Find the actor's output tensor
    logits = None
    for candidate_key in [
        (group, "logits"),
        (group, "loc"),
        (group, "action"),
        (group, "actor_features"),
    ]:
        logits = out.get(candidate_key, None)
        if logits is not None:
            print(f"\n  Output tensor key: {candidate_key}")
            break

    if logits is not None:
        print(f"  Output tensor: shape={tuple(logits.shape)}, "
              f"mean={logits.mean().item():.6f}, std={logits.std().item():.6f}")
        dummy_loss = logits.mean()
        dummy_loss.backward()
        print(f"  Dummy loss: {dummy_loss.item():.6f}")
    else:
        print("  ❌ Could not find output tensor in actor output!")
        print(f"  Available keys: {list(out.keys(True, True))}")
        return

    check_gradient_flow_module(actor, "actor (direct backward — definitive check)")
    check_gnn_per_layer(actor)

    # ── critic check (also direct) ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("CHECK 2b: Direct critic forward → dummy loss → backward")
    print("=" * 70)

    critic = loss_module.critic_network
    critic.zero_grad()

    batch_critic = batch.clone()
    with torch.enable_grad():
        out_c = critic(batch_critic)

    value = None
    for candidate_key in [
        (group, "state_value"),
        (group, "value"),
    ]:
        value = out_c.get(candidate_key, None)
        if value is not None:
            print(f"\n  Output tensor key: {candidate_key}")
            break

    if value is not None:
        print(f"  Output tensor: shape={tuple(value.shape)}, "
              f"mean={value.mean().item():.6f}")
        dummy_loss_c = value.mean()
        dummy_loss_c.backward()
    else:
        print("  ❌ Could not find value tensor in critic output!")
        print(f"  Available keys: {list(out_c.keys(True, True))}")

    check_gradient_flow_module(critic, "critic (direct backward)")
    check_gnn_per_layer(critic)

    # ── final summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DEBUG COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
