import os
import sys
from collections import defaultdict

import torch
import torch_geometric.nn as tgnn

# Ensure BenchMARL import
sys.path.append(os.path.join(os.getcwd(), "BenchMARL"))

from torchrl.envs.utils import step_mdp

from benchmarl.algorithms import HGTeamConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.environments.PowerGridworldVariable.common import PowerGridworldVariableTask
from benchmarl.models import HeteroGnnConfig, TransformerConfig


def patch_tensordict_stack_debug():
    """Wrap TensorDict.maybe_dense_stack to print key dtypes on failure."""
    from tensordict import TensorDict

    orig = TensorDict.maybe_dense_stack

    def debug_maybe_dense_stack(cls, list_of_tensordicts, *args, **kwargs):
        dim = kwargs.get("dim", args[0] if len(args) else 0)
        try:
            # orig may be a classmethod; call underlying function if present
            if hasattr(orig, "__func__"):
                return orig.__func__(cls, list_of_tensordicts, *args, **kwargs)
            return orig(list_of_tensordicts, *args, **kwargs)
        except Exception as exc:
            print("TensorDict.maybe_dense_stack failed; inspecting dtypes...")
            dtype_map = defaultdict(set)
            shape_map = defaultdict(set)
            env_map = defaultdict(set)
            for env_idx, td in enumerate(list_of_tensordicts):
                for key in td.keys(include_nested=True):
                    val = td.get(key)
                    if not isinstance(val, torch.Tensor):
                        continue
                    key_str = key_to_str(key)
                    dtype_map[key_str].add(str(val.dtype))
                    shape_map[key_str].add(tuple(val.shape))
                    env_map[key_str].add((env_idx, str(val.dtype), tuple(val.shape)))
            mismatched = {k: v for k, v in dtype_map.items() if len(v) > 1}
            if not mismatched:
                print("No dtype mismatch found during stack; exception was:", exc)
            else:
                print("Cross-list dtype mismatches before stack:")
                for key, dtypes in sorted(mismatched.items()):
                    shapes = shape_map.get(key, {})
                    env_detail = sorted(env_map.get(key, {}))
                    print(
                        f"  {key}: dtypes={sorted(dtypes)}, shapes={sorted(shapes)}, envs={env_detail}"
                    )
            raise

    TensorDict.maybe_dense_stack = classmethod(debug_maybe_dense_stack)
    return orig


def patch_tensordict_stack_onto_debug():
    """Wrap TensorDict._stack_onto_ to print per-key dtypes on failure."""
    from tensordict import TensorDict

    orig = TensorDict._stack_onto_

    def debug_stack_onto(self, list_of_tensordicts, dim=0):
        try:
            return orig(self, list_of_tensordicts, dim)
        except Exception as exc:
            print("TensorDict._stack_onto_ failed; inspecting dtypes...")
            dtype_map = defaultdict(set)
            shape_map = defaultdict(set)
            env_map = defaultdict(set)
            for env_idx, td in enumerate(list_of_tensordicts):
                for key in td.keys(include_nested=True):
                    val = td.get(key)
                    if not isinstance(val, torch.Tensor):
                        continue
                    key_str = key_to_str(key)
                    dtype_map[key_str].add(str(val.dtype))
                    shape_map[key_str].add(tuple(val.shape))
                    env_map[key_str].add((env_idx, str(val.dtype), tuple(val.shape)))
            mismatched = {k: v for k, v in dtype_map.items() if len(v) > 1}
            if not mismatched:
                print("No dtype mismatch across inputs; exception was:", exc)
            else:
                print("Cross-input dtype mismatches before stack:")
                for key, dtypes in sorted(mismatched.items()):
                    shapes = shape_map.get(key, {})
                    env_detail = sorted(env_map.get(key, {}))
                    print(
                        f"  {key}: dtypes={sorted(dtypes)}, shapes={sorted(shapes)}, envs={env_detail}"
                    )
            # Also print out-dtypes if available
            if self is not None:
                print("Output TensorDict existing dtypes:")
                for key in self.keys(include_nested=True):
                    val = self.get(key)
                    if isinstance(val, torch.Tensor):
                        print(f"  {key_to_str(key)}: dtype={val.dtype}, shape={tuple(val.shape)}")
            raise

    TensorDict._stack_onto_ = debug_stack_onto
    return orig


def patch_transformer_capture_tokens(policy_module, token_store):
    """Monkey-patch the first Transformer found to capture input tokens and positions."""
    from benchmarl.models.transformer import Transformer

    for module in policy_module.modules():
        if isinstance(module, Transformer):
            original = module._build_token_sequence
            original_apply_encoder = module._apply_encoder

            def wrapped_apply_encoder(tokens, key_padding_mask, causal_mask):
                # Make mask dtypes match to silence PyTorch warning
                if key_padding_mask is not None and causal_mask is not None:
                    key_padding_mask = key_padding_mask.to(dtype=torch.bool)
                    causal_mask = causal_mask.to(dtype=torch.bool)
                return original_apply_encoder(tokens, key_padding_mask, causal_mask)

            def wrapped(obs_emb, action):
                tokens, obs_positions = original(obs_emb, action)
                token_store["tokens"] = tokens.detach().cpu()
                token_store["obs_positions"] = obs_positions.detach().cpu()
                token_store["action_present"] = action is not None
                return tokens, obs_positions

            module._build_token_sequence = wrapped
            module._apply_encoder = wrapped_apply_encoder
            token_store["transformer"] = module
            return module
    raise RuntimeError("Transformer module not found in policy.")


def build_configs():
    # Actor: transformer
    actor_model_config = TransformerConfig(
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        dropout=0.0,
        max_seq_len=16,
        use_z_as_query=True,
        append_actions=True,
        norm_first=True,
    )

    # Critic: lightweight GNN
    # Note: grid_node warning is expected here because the actor uses only agent-node outputs
    # from the last GNN layer; grid_node embeddings are not updated as destinations.
    critic_model_config = HeteroGnnConfig(
        topology="adjacency",
        self_loops=True,
        gnn_class=tgnn.TransformerConv,
        gnn_kwargs={"heads": 2, "concat": False, "beta": True},
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
        node_features_keys={"grid_node": "grid_node_features"},
        node_features_dims={"grid_node": 2},
        agent_node_index_key="agent_grid_edge_index",
        exclude_observations_from_node_features=False,
        cat_observations_to_output=False,
        num_layers=1,
        pos_features=0,
        vel_features=0,
        edge_radius=0,
    )

    # Algorithm tweaks
    algorithm_config = HGTeamConfig.get_from_yaml()
    algorithm_config.gnn_mode = "concat"  # produce embedding_z for transformer queries
    algorithm_config.entropy_coef = 0.01
    algorithm_config.embedding_entropy_coef = 0.0
    algorithm_config.embedding_diversity_coef = 0.0

    # Experiment tweaks (small, quick)
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.sampling_device = "cpu"
    experiment_config.train_device = "cpu"
    experiment_config.parallel_collection = True
    experiment_config.on_policy_n_envs_per_worker = 4
    experiment_config.on_policy_collected_frames_per_batch = 8
    experiment_config.on_policy_minibatch_size = 4
    experiment_config.on_policy_n_minibatch_iters = 1
    experiment_config.max_n_frames = 8
    experiment_config.evaluation_interval = 8
    experiment_config.evaluation_episodes = 1
    experiment_config.loggers = []
    experiment_config.create_json = False
    experiment_config.checkpoint_at_end = False

    return actor_model_config, critic_model_config, algorithm_config, experiment_config


def key_to_str(key):
    if isinstance(key, tuple):
        return "/".join(str(k) for k in key)
    return str(key)


def collect_steps_for_dtype_check(experiment, policy, num_steps=3):
    """Run a tiny rollout and record per-step tensordicts before stacking."""
    env = experiment.env_func()
    td = env.reset()
    steps = []

    for _ in range(num_steps):
        with torch.no_grad():
            td = policy(td)

        td_step = env.step(td)
        steps.append(td_step.clone())

        # Prepare next step
        td = step_mdp(td_step)

    return steps


def collect_envwise_steps(experiment, policy, num_envs=2, num_steps=1):
    """Collect steps separately per env to spot cross-env dtype mismatches."""
    envs = [experiment.env_func() for _ in range(num_envs)]
    tds = [env.reset() for env in envs]
    per_env_steps = [[] for _ in range(num_envs)]

    for _ in range(num_steps):
        for idx, (env, td) in enumerate(zip(envs, tds)):
            with torch.no_grad():
                td = policy(td)
            td_step = env.step(td)
            per_env_steps[idx].append(td_step.clone())
            tds[idx] = step_mdp(td_step)

    return per_env_steps


def report_dtype_mismatches(steps):
    dtype_map = defaultdict(set)
    shape_map = defaultdict(set)

    for idx, td in enumerate(steps):
        for key in td.keys(include_nested=True):
            val = td.get(key)
            if not isinstance(val, torch.Tensor):
                continue
            key_str = key_to_str(key)
            dtype_map[key_str].add(str(val.dtype))
            shape_map[key_str].add(tuple(val.shape))

    mismatched = {k: v for k, v in dtype_map.items() if len(v) > 1}

    if not mismatched:
        print("No dtype mismatches detected across steps.")
    else:
        print("Keys with dtype mismatches across steps:")
        for key, dtypes in sorted(mismatched.items()):
            shapes = shape_map.get(key, {})
            print(f"  {key}: dtypes={sorted(dtypes)}, shapes={sorted(shapes)}")


def report_cross_env_dtype_mismatches(per_env_steps):
    if not per_env_steps:
        print("No env steps collected.")
        return

    n_envs = len(per_env_steps)
    n_steps = min(len(steps) for steps in per_env_steps)
    if n_steps == 0:
        print("No steps per env to compare.")
        return

    print(f"Checking dtype mismatches across {n_envs} envs for {n_steps} step(s)...")
    for step_idx in range(n_steps):
        dtype_map = defaultdict(set)
        shape_map = defaultdict(set)
        env_map = defaultdict(set)
        for env_idx, steps in enumerate(per_env_steps):
            td = steps[step_idx]
            for key in td.keys(include_nested=True):
                val = td.get(key)
                if not isinstance(val, torch.Tensor):
                    continue
                key_str = key_to_str(key)
                dtype_map[key_str].add(str(val.dtype))
                shape_map[key_str].add(tuple(val.shape))
                env_map[key_str].add((env_idx, str(val.dtype), tuple(val.shape)))

        mismatched = {k: v for k, v in dtype_map.items() if len(v) > 1}
        if not mismatched:
            print(f"Step {step_idx}: no cross-env dtype mismatches.")
        else:
            print(f"Step {step_idx}: cross-env dtype mismatches:")
            for key, dtypes in sorted(mismatched.items()):
                shapes = shape_map.get(key, {})
                env_detail = sorted(env_map.get(key, {}))
                print(f"  {key}: dtypes={sorted(dtypes)}, shapes={sorted(shapes)}, envs={env_detail}")


def main():
    # Build configs
    actor_model_config, critic_model_config, algorithm_config, experiment_config = build_configs()

    # Task
    task = PowerGridworldVariableTask.EVOVERNIGHT13NODE_VPP.get_from_yaml()

    # Build experiment (this wires specs and transforms, but we won't run training)
    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=actor_model_config,
        critic_model_config=critic_model_config,
        seed=0,
        config=experiment_config,
    )

    # Get collection policy and patch transformer to capture tokens per call
    policy = experiment.algorithm.get_policy_for_collection()
    token_store = {}
    patch_transformer_capture_tokens(policy, token_store)

    # Patch TensorDict stacking to surface mismatching keys when the collector stacks batches
    patch_tensordict_stack_debug()
    patch_tensordict_stack_onto_debug()

    # Tiny single-env rollout to surface cross-step dtype issues
    steps = collect_steps_for_dtype_check(experiment, policy, num_steps=3)
    report_dtype_mismatches(steps)

    # Tiny multi-env rollout to surface cross-env dtype issues
    per_env_steps = collect_envwise_steps(experiment, policy, num_envs=2, num_steps=2)
    report_cross_env_dtype_mismatches(per_env_steps)

    # Attempt one collector batch (will trigger the patched stack if it fails)
    try:
        batch = next(iter(experiment.collector))
        print("Collector stacked without error. Batch shape:", batch.batch_size)
    except Exception as exc:
        print("Collector raised:", exc)

    # Report captured tokens from last policy call for sanity
    tokens = token_store.get("tokens")
    positions = token_store.get("obs_positions")
    has_action = token_store.get("action_present")

    print("Transformer tokens shape:", tokens.shape if tokens is not None else None)
    print("Observation positions:", positions.tolist() if positions is not None else None)
    print("Action token appended:", has_action)
    if tokens is not None:
        # Show first batch/agent tokens
        print("First token slice (batch 0, agent 0):", tokens[0, 0])


if __name__ == "__main__":
    main()