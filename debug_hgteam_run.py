"""Small-scale smoke tests for HGTeam (PPO), HGTeamSAC, and HGTeamHAPPO.

Usage:
    python debug_hgteam_run.py          # run PPO and SAC tests
    python debug_hgteam_run.py ppo      # PPO only
    python debug_hgteam_run.py sac      # SAC only
    python debug_hgteam_run.py happo    # HAPPO only
    python debug_hgteam_run.py all      # run all three tests
"""

import argparse
import sys
import os
import torch
import torch_geometric.nn as tgnn

# Ensure BenchMARL is in path
sys.path.append(os.path.join(os.getcwd(), "BenchMARL"))

from benchmarl.algorithms import HGTeamConfig, HGTeamSACConfig, HGTeamHAPPOConfig
from benchmarl.models import HeteroGnnConfig, MlpConfig, SequenceModelConfig, TransformerConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.environments.PowerGridworldVariable.common import PowerGridworldVariableTask


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _critic_gnn_config():
    """Critic GNN configuration shared by both PPO and SAC tests."""
    return HeteroGnnConfig(
        topology="adjacency",
        self_loops=True,
        gnn_class=tgnn.TransformerConv,
        gnn_kwargs={
            "heads": 1,
            "concat": False,
            "beta": True,
        },
        grid_edge_keys={
            "line_adjacency": "line_adjacency",
            "transformer_adjacency": "transformer_adjacency",
            "switch_adjacency": "switch_adjacency"
        },
        edge_features_dims={
            "line_adjacency": 3,
            "transformer_adjacency": 3,
            "switch_adjacency": 1,
            "interaction": 0,
            "mapping": 0,
            "mapping_rev": 0
        },
        node_features_keys={
            "grid_node": "grid_node_features",
            "agents": "participation_score",
        },
        node_features_dims={
            "grid_node": 2,
            "agents": 1,
        },
        agent_node_index_key="agent_grid_edge_index",
        exclude_observations_from_node_features=False,
        cat_observations_to_output=False,
        num_layers=2,
        gnn_hidden_dim=32,
        pos_features=0, vel_features=0, edge_radius=0
    )


def _actor_model_config():
    """Actor Transformer configuration shared by both PPO and SAC tests."""
    return TransformerConfig(
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        max_seq_len=192,
        use_z_as_query=True,
        append_actions=True,
        norm_first=True,
    )


# ------------------------------------------------------------------
# HGTeam PPO test
# ------------------------------------------------------------------

def run_ppo_test():
    print("=" * 60)
    print("  HGTeam PPO — DEBUG smoke test")
    print("=" * 60)

    algorithm_config = HGTeamConfig.get_from_yaml()
    algorithm_config.entropy_coef = 0.5
    algorithm_config.gnn_mode = "concat"
    algorithm_config.embedding_entropy_coef = 1
    algorithm_config.embedding_diversity_coef = 0.001
    algorithm_config.stochastic_z = True
    algorithm_config.z_dim = 32
    algorithm_config.hypernet_actor_feature_dim = 64

    task = PowerGridworldVariableTask.EVOVERNIGHT13NODE_VPP.get_from_yaml()

    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.sampling_device = "cpu"
    experiment_config.train_device = _device()
    experiment_config.collection_policy_device = _device()
    experiment_config.share_policy_params = True
    experiment_config.lr = 5e-6
    experiment_config.evaluation_episodes = 1
    experiment_config.evaluation_static = False

    # Minimal scale — forces evaluation early so eval-path bugs surface
    experiment_config.parallel_collection = False
    experiment_config.on_policy_n_envs_per_worker = 2
    experiment_config.on_policy_collected_frames_per_batch = 192
    experiment_config.on_policy_minibatch_size = 25
    experiment_config.on_policy_n_minibatch_iters = 2
    experiment_config.max_n_frames = 384
    experiment_config.evaluation_interval = 192

    experiment_config.loggers = []
    experiment_config.create_json = True
    experiment_config.checkpoint_at_end = False

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=_actor_model_config(),
        critic_model_config=_critic_gnn_config(),
        seed=42,
        config=experiment_config,
    )

    print("\nStarting PPO debug run...")
    experiment.run()
    print("PPO debug run completed successfully!\n")


# ------------------------------------------------------------------
# HGTeamSAC test
# ------------------------------------------------------------------

def run_sac_test():
    print("=" * 60)
    print("  HGTeamSAC — DEBUG smoke test")
    print("=" * 60)

    algorithm_config = HGTeamSACConfig.get_from_yaml()
    algorithm_config.gnn_mode = "learned_query"
    algorithm_config.stochastic_z = False
    algorithm_config.z_dim = 32
    algorithm_config.hypernet_actor_feature_dim = 64
    algorithm_config.embedding_entropy_coef = 0.0
    algorithm_config.embedding_diversity_coef = 0.0

    # SAC-specific
    algorithm_config.alpha_init = 0.2
    algorithm_config.target_entropy = 0.0
    algorithm_config.num_qvalue_nets = 2
    algorithm_config.fixed_alpha = False
    algorithm_config.detach_action_from_q = False
    algorithm_config.detach_z_from_transformer = True
    algorithm_config.critic_use_mu = True

    task = PowerGridworldVariableTask.EVOVERNIGHT13NODE_VPP.get_from_yaml()

    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.sampling_device = "cpu"
    experiment_config.train_device = _device()
    experiment_config.collection_policy_device = _device()
    experiment_config.share_policy_params = True
    experiment_config.lr = 1e-4
    experiment_config.evaluation_episodes = 1
    experiment_config.evaluation_static = False

    # Minimal scale for off-policy
    experiment_config.parallel_collection = False
    experiment_config.off_policy_n_envs_per_worker = 2
    experiment_config.off_policy_collected_frames_per_batch = 192
    experiment_config.off_policy_train_batch_size = 64
    experiment_config.off_policy_n_optimizer_steps = 2
    experiment_config.off_policy_memory_size = 1000
    experiment_config.off_policy_init_random_frames = 0
    experiment_config.max_n_frames = 384
    experiment_config.evaluation_interval = 192

    experiment_config.loggers = []
    experiment_config.create_json = True
    experiment_config.checkpoint_at_end = False

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=_actor_model_config(),
        critic_model_config=_critic_gnn_config(),
        seed=42,
        config=experiment_config,
    )

    print("\nStarting SAC debug run...")
    experiment.run()
    print("SAC debug run completed successfully!\n")


# ------------------------------------------------------------------
# HGTeamHAPPO test
# ------------------------------------------------------------------

def run_happo_test():
    print("=" * 60)
    print("  HGTeamHAPPO — DEBUG smoke test")
    print("=" * 60)

    algorithm_config = HGTeamHAPPOConfig.get_from_yaml()
    algorithm_config.entropy_coef = 0.5
    algorithm_config.gnn_mode = "concat"
    algorithm_config.embedding_entropy_coef = 0
    algorithm_config.embedding_diversity_coef = 0
    algorithm_config.stochastic_z = True
    algorithm_config.z_dim = 32
    algorithm_config.hypernet_actor_feature_dim = 64
    algorithm_config.encoder_update_mode = "accumulated"
    algorithm_config.fixed_order = False

    task = PowerGridworldVariableTask.EVOVERNIGHT13NODE_VPP.get_from_yaml()

    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.sampling_device = "cpu"
    experiment_config.train_device = _device()
    experiment_config.collection_policy_device = _device()
    experiment_config.share_policy_params = True
    experiment_config.lr = 5e-6
    experiment_config.evaluation_episodes = 1
    experiment_config.evaluation_static = False

    # Minimal scale
    experiment_config.parallel_collection = False
    experiment_config.on_policy_n_envs_per_worker = 2
    experiment_config.on_policy_collected_frames_per_batch = 192
    experiment_config.on_policy_minibatch_size = 25
    experiment_config.on_policy_n_minibatch_iters = 2
    experiment_config.max_n_frames = 384
    experiment_config.evaluation_interval = 192

    experiment_config.loggers = []
    experiment_config.create_json = True
    experiment_config.checkpoint_at_end = False

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=_actor_model_config(),
        critic_model_config=_critic_gnn_config(),
        seed=42,
        config=experiment_config,
    )

    print("\nStarting HAPPO debug run...")
    experiment.run()
    print("HAPPO debug run completed successfully!\n")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HGTeam debug smoke tests")
    parser.add_argument(
        "mode", nargs="?", default="both",
        choices=["ppo", "sac", "happo", "both", "all"],
        help="Which algorithm to test (default: both = ppo+sac; all = ppo+sac+happo)",
    )
    args = parser.parse_args()

    if args.mode in ("ppo", "both", "all"):
        run_ppo_test()
    if args.mode in ("sac", "both", "all"):
        run_sac_test()
    if args.mode in ("happo", "all"):
        run_happo_test()

    print("All requested debug tests passed!")


if __name__ == "__main__":
    main()
