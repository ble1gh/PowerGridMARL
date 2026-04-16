import argparse
import datetime
import os
import sys
from pathlib import Path

import torch
import torch_geometric.nn as tgnn
from torch import nn

# Ensure BenchMARL is in path
sys.path.append(os.path.join(os.getcwd(), "BenchMARL"))

from benchmarl.algorithms import HGTeamConfig, HGTeamHAPPOConfig, HGTeamSACConfig
from benchmarl.environments.PowerGridworldVariable.common import PowerGridworldVariableTask
from benchmarl.environments.smacv2_variable.common import Smacv2VariableTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.experiment.embedding_viz_callback import EmbeddingVizCallback
from benchmarl.models import HeteroGnnConfig, MlpConfig


def parse_args():
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    default_results_dir = Path(__file__).resolve().parent / "results"

    parser = argparse.ArgumentParser(description="Run HGTeam experiment")

    # --- Task / Environment ---
    parser.add_argument(
        "--task",
        type=str,
        default="evovernight13node_vpp",
        choices=[
            "evovernight13node_vpp",
            "protoss_10_vs_10",
            "protoss_10_vs_11",
            "protoss_20_vs_20",
            "protoss_20_vs_23",
        ],
        help="Task to run (PowerGridworld or SMACv2 variable-composition)",
    )

    # --- Experiment ---
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project-name", type=str, default="PowerGridworldVariable_VPP")
    parser.add_argument("--wandb-name", type=str, default=f"HGTeam_{date_str}_{job_id}")
    parser.add_argument(
        "--wandb-group", type=str, default=None, help="WandB group for organizing related runs"
    )
    parser.add_argument(
        "--wandb-tags", type=str, nargs="*", default=None, help="WandB tags for filtering runs"
    )
    parser.add_argument("--results-dir", type=Path, default=default_results_dir)
    parser.add_argument("--max-n-frames", type=int, default=10_000_000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--evaluation-interval", type=int, default=12288)
    parser.add_argument("--evaluation-episodes", type=int, default=5)
    parser.add_argument("--n-envs-per-worker", type=int, default=32)
    parser.add_argument("--frames-per-batch", type=int, default=12288)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--n-minibatch-iters", type=int, default=5)

    # --- Algorithm (HGTeam) ---
    parser.add_argument(
        "--lmbda", type=float, default=None, help="GAE lambda (overrides yaml default if set)"
    )
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument(
        "--gnn-mode",
        type=str,
        default="learned_query",
        choices=["none", "concat", "hypernetwork", "learned_query"],
    )
    parser.add_argument("--embedding-entropy-coef", type=float, default=1.0)
    parser.add_argument("--embedding-diversity-coef", type=float, default=0.001)
    parser.add_argument("--stochastic-z", type=lambda x: x.lower() != "false", default=True)
    parser.add_argument("--z-dim", type=int, default=32)
    parser.add_argument("--hypernet-actor-feature-dim", type=int, default=64)
    parser.add_argument(
        "--gnn-norm-class",
        type=str,
        default="LayerNorm",
        choices=["null", "LayerNorm", "BatchNorm1d", "InstanceNorm1d", "GroupNorm"],
    )
    parser.add_argument("--split-z", type=lambda x: x.lower() != "false", default=False)
    parser.add_argument("--z-token-dim", type=int, default=32)
    parser.add_argument("--z-query-dim", type=int, default=32)
    parser.add_argument("--stochastic-z-query", type=lambda x: x.lower() != "false", default=False)
    parser.add_argument(
        "--scale-lb",
        type=float,
        default=0.0001,
        help="Minimum std for NormalParamExtractor (ignored when use_beta=True)",
    )

    # --- VIB (Variational Information Bottleneck) ---
    parser.add_argument("--use-vib", type=lambda x: x.lower() != "false", default=False)
    parser.add_argument("--vib-beta", type=float, default=0.01)
    parser.add_argument("--vib-warmup-frames", type=int, default=500_000)

    # --- Algorithm selection ---
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "sac", "happo"],
        help="Base RL algorithm: ppo (HGTeam), sac (HGTeamSAC), or happo (HGTeamHAPPO)",
    )

    # --- SAC-specific ---
    parser.add_argument("--alpha-init", type=float, default=1.0)
    parser.add_argument(
        "--target-entropy", type=str, default="auto", help="'auto' or a float value"
    )
    parser.add_argument("--num-qvalue-nets", type=int, default=2)
    parser.add_argument(
        "--min-alpha",
        type=float,
        default=None,
        help="Minimum SAC temperature alpha (prevents alpha→0 NaN)",
    )
    parser.add_argument("--fixed-alpha", type=lambda x: x.lower() != "false", default=False)
    parser.add_argument(
        "--detach-action-from-q", type=lambda x: x.lower() != "false", default=False
    )
    parser.add_argument(
        "--detach-z-from-transformer", type=lambda x: x.lower() != "false", default=True
    )
    parser.add_argument("--critic-use-mu", type=lambda x: x.lower() != "false", default=True)
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-encoder", type=float, default=1e-4)
    parser.add_argument("--lr-critic", type=float, default=3e-4)

    # --- Critic action conditioning ---
    parser.add_argument(
        "--critic-use-other-actions",
        type=lambda x: x.lower() != "false",
        default=None,
        help="Condition critic on other agents' actions via GNN edge features (overrides yaml)",
    )

    # --- HAPPO-specific ---
    parser.add_argument(
        "--encoder-update-mode",
        type=str,
        default="accumulated",
        choices=["accumulated", "separate_forward", "coop_encoder"],
        help="How GNN encoder is updated across sequential group steps",
    )
    parser.add_argument(
        "--fixed-order",
        type=lambda x: x.lower() != "false",
        default=False,
        help="Use fixed (not random) group ordering in HAPPO",
    )
    parser.add_argument(
        "--encoder-n-optimizer-steps",
        type=int,
        default=None,
        help="Phase 2 GNN optimizer steps (default: use --n-minibatch-iters)",
    )

    # --- Off-policy (SAC) experiment config ---
    parser.add_argument("--off-policy-memory-size", type=int, default=1_000_000)
    parser.add_argument("--off-policy-train-batch-size", type=int, default=256)
    parser.add_argument("--off-policy-collected-frames-per-batch", type=int, default=256)
    parser.add_argument("--off-policy-n-optimizer-steps", type=int, default=1)
    parser.add_argument("--off-policy-init-random-frames", type=int, default=10_000)

    # --- Embedding visualization ---
    parser.add_argument(
        "--embedding-viz-interval",
        type=int,
        default=0,
        help="Log embedding t-SNE/PCA/cosine plots every N iters (0=disabled)",
    )
    parser.add_argument("--off-policy-n-envs-per-worker", type=int, default=4)
    parser.add_argument("--polyak-tau", type=float, default=0.005)

    # --- Environment ---
    parser.add_argument("--reward-scale", type=int, default=1000)

    return parser.parse_args()


def main():
    args = parse_args()
    print("Preparing Experiment...")

    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(results_dir / "wandb"))

    # Critic Configuration (Different from Actor)
    # When share_critic_across_groups=True, the critic GNN natively handles
    # heterogeneous agent types as separate node types with their own input
    # dimensions. The "agents" placeholder in node_features_keys/dims is
    # automatically replaced with per-type keys (e.g., EV, PV, Storage) at
    # runtime by HGTeam._get_shared_critic().
    critic_gnn_config = HeteroGnnConfig(
        topology="adjacency",
        self_loops=True,
        gnn_class=tgnn.TransformerConv,
        gnn_kwargs={
            "heads": 2,
            "concat": False,
            "beta": True,
        },
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
            "agents": "participation_score",  # Will be overridden to per-type key
        },
        node_features_dims={
            "grid_node": 2,
            "agents": 1,
        },
        agent_node_index_key="agent_grid_edge_index",  # Will be overridden to per-type key
        exclude_observations_from_node_features=False,
        cat_observations_to_output=False,  # Do not pass observations through
        num_layers=2,
        gnn_hidden_dim=32,
        norm_class=nn.LayerNorm,
        pos_features=0,
        vel_features=0,
        edge_radius=0,
    )

    # critic_model_config = critic_gnn_config

    # HYPERNETWORK CONFIGURATION
    # 1. Define the Base Actor (Stream B)
    # This is the functional part that processes local observations
    # For Hypernetwork, we usually want this to be an MLP or LSTM processing raw obs
    # Transformer actor (tokens = obs plus previous actions; queries come from GNN embedding_z)
    actor_model_config = MlpConfig(
        num_cells=[128, 128, 64],
        layer_class=torch.nn.Linear,
        activation_class=torch.nn.ReLU,
    )
    critic_model_config = critic_gnn_config

    # 2. Algorithm Configuration - load from yaml and override as needed
    if args.algorithm == "sac":
        algorithm_config = HGTeamSACConfig.get_from_yaml()
        # SAC-specific settings
        algorithm_config.alpha_init = args.alpha_init
        target_entropy = args.target_entropy
        algorithm_config.target_entropy = (
            target_entropy if target_entropy == "auto" else float(target_entropy)
        )
        algorithm_config.num_qvalue_nets = args.num_qvalue_nets
        if args.min_alpha is not None:
            algorithm_config.min_alpha = args.min_alpha
        algorithm_config.fixed_alpha = args.fixed_alpha
        algorithm_config.detach_action_from_q = args.detach_action_from_q
        algorithm_config.detach_z_from_transformer = args.detach_z_from_transformer
        algorithm_config.critic_use_mu = args.critic_use_mu
        algorithm_config.lr_actor = args.lr_actor
        algorithm_config.lr_encoder = args.lr_encoder
        algorithm_config.lr_critic = args.lr_critic
    elif args.algorithm == "happo":
        algorithm_config = HGTeamHAPPOConfig.get_from_yaml()
        algorithm_config.entropy_coef = args.entropy_coef
        algorithm_config.encoder_update_mode = args.encoder_update_mode
        algorithm_config.fixed_order = args.fixed_order
        algorithm_config.encoder_lr = args.lr_encoder
        algorithm_config.encoder_n_optimizer_steps = args.encoder_n_optimizer_steps
    else:
        algorithm_config = HGTeamConfig.get_from_yaml()
        algorithm_config.entropy_coef = args.entropy_coef

    # Common HGTeam settings (shared by both PPO and SAC variants)
    if args.critic_use_other_actions is not None:
        algorithm_config.critic_use_other_actions = args.critic_use_other_actions
    if args.lmbda is not None:
        algorithm_config.lmbda = args.lmbda
    algorithm_config.gnn_mode = args.gnn_mode
    algorithm_config.embedding_entropy_coef = args.embedding_entropy_coef
    algorithm_config.embedding_diversity_coef = args.embedding_diversity_coef
    algorithm_config.stochastic_z = args.stochastic_z
    algorithm_config.z_dim = args.z_dim
    algorithm_config.hypernet_actor_feature_dim = args.hypernet_actor_feature_dim
    algorithm_config.gnn_norm_class = None if args.gnn_norm_class == "null" else args.gnn_norm_class
    algorithm_config.split_z = args.split_z
    algorithm_config.z_token_dim = args.z_token_dim
    algorithm_config.z_query_dim = args.z_query_dim
    algorithm_config.stochastic_z_query = args.stochastic_z_query
    algorithm_config.scale_lb = args.scale_lb
    algorithm_config.use_vib = args.use_vib
    algorithm_config.vib_beta = args.vib_beta
    algorithm_config.vib_warmup_frames = args.vib_warmup_frames

    # 3. Task
    _TASK_MAP = {
        "evovernight13node_vpp": lambda: PowerGridworldVariableTask.EVOVERNIGHT13NODE_VPP.get_from_yaml(),
        "protoss_10_vs_10": lambda: Smacv2VariableTask.PROTOSS_10_VS_10.get_from_yaml(),
        "protoss_10_vs_11": lambda: Smacv2VariableTask.PROTOSS_10_VS_11.get_from_yaml(),
        "protoss_20_vs_20": lambda: Smacv2VariableTask.PROTOSS_20_VS_20.get_from_yaml(),
        "protoss_20_vs_23": lambda: Smacv2VariableTask.PROTOSS_20_VS_23.get_from_yaml(),
    }
    task = _TASK_MAP[args.task]()
    if hasattr(task, "config") and "reward_scale" in task.config:
        task.config["reward_scale"] = args.reward_scale

    # 4. Experiment Configuration
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.sampling_device = "cpu"
    experiment_config.train_device = "cuda"
    experiment_config.collection_policy_device = (
        "cuda"  # Run GNN+Transformer on GPU during collection
    )
    experiment_config.share_policy_params = True
    experiment_config.evaluation_static = False
    experiment_config.parallel_collection = True
    experiment_config.max_n_iters = None
    experiment_config.lr = args.lr
    experiment_config.evaluation_episodes = args.evaluation_episodes
    experiment_config.max_n_frames = args.max_n_frames
    experiment_config.evaluation_interval = args.evaluation_interval
    experiment_config.project_name = args.project_name
    experiment_config.loggers = ["wandb"]
    experiment_config.save_folder = str(results_dir)
    wandb_extra = {"name": args.wandb_name}
    if args.wandb_group:
        wandb_extra["group"] = args.wandb_group
    if args.wandb_tags:
        wandb_extra["tags"] = args.wandb_tags
    experiment_config.wandb_extra_kwargs = wandb_extra

    if args.algorithm == "sac":
        # Off-policy experiment settings
        experiment_config.off_policy_memory_size = args.off_policy_memory_size
        experiment_config.off_policy_train_batch_size = args.off_policy_train_batch_size
        experiment_config.off_policy_collected_frames_per_batch = (
            args.off_policy_collected_frames_per_batch
        )
        experiment_config.off_policy_n_optimizer_steps = args.off_policy_n_optimizer_steps
        experiment_config.off_policy_init_random_frames = args.off_policy_init_random_frames
        experiment_config.off_policy_n_envs_per_worker = args.off_policy_n_envs_per_worker
        experiment_config.soft_target_update = True
        experiment_config.polyak_tau = args.polyak_tau
    else:
        # On-policy experiment settings
        experiment_config.on_policy_n_envs_per_worker = args.n_envs_per_worker
        experiment_config.on_policy_collected_frames_per_batch = args.frames_per_batch
        experiment_config.on_policy_minibatch_size = args.minibatch_size
        experiment_config.on_policy_n_minibatch_iters = args.n_minibatch_iters

    callbacks = []
    if args.embedding_viz_interval > 0:
        callbacks.append(
            EmbeddingVizCallback(
                log_every_n_iters=args.embedding_viz_interval,
            )
        )

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=actor_model_config,
        critic_model_config=critic_model_config,
        seed=args.seed,
        config=experiment_config,
        callbacks=callbacks,
    )

    print("\nStarting Run")
    experiment.run()
    print("Run Completed Successfully!")


if __name__ == "__main__":
    main()
