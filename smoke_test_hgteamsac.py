import argparse
import os
import sys

import torch_geometric.nn as tgnn
from torch import nn

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, "BenchMARL"))
sys.path.append(os.path.join(ROOT, "PowerGridworld"))

from benchmarl.algorithms import HGTeamSACConfig  # noqa: E402
from benchmarl.environments.PowerGridworldVariable.common import (  # noqa: E402
    PowerGridworldVariableTask,
)
from benchmarl.experiment import Experiment, ExperimentConfig  # noqa: E402
from benchmarl.models import EdgeWeightedHGTConfig, HeteroGnnConfig, TransformerConfig  # noqa: E402


def build_critic_model_config(critic_model: str):
    common_kwargs = dict(
        topology="adjacency",
        self_loops=True,
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
            "agents": "participation_score",
        },
        node_features_dims={
            "grid_node": 2,
            "agents": 1,
        },
        agent_node_index_key="agent_grid_edge_index",
        exclude_observations_from_node_features=False,
        cat_observations_to_output=False,
        num_layers=1,
        gnn_hidden_dim=16,
        norm_class=nn.LayerNorm,
        pos_features=0,
        vel_features=0,
        edge_radius=0,
    )
    if critic_model == "heterognn":
        return HeteroGnnConfig(
            gnn_class=tgnn.TransformerConv,
            gnn_kwargs={
                "heads": 1,
                "concat": False,
                "beta": True,
            },
            **common_kwargs,
        )
    if critic_model == "edgeweightedhgt":
        return EdgeWeightedHGTConfig(
            heads=2,
            low_rank=4,
            edge_gate_hidden_dim=16,
            edge_gate_num_layers=2,
            zero_init_edge_gates=True,
            **common_kwargs,
        )
    raise ValueError(f"Unknown critic_model={critic_model}")


def build_experiment(
    encoder_update_mode: str = "coop_encoder", critic_model: str = "heterognn"
) -> Experiment:
    critic_model_config = build_critic_model_config(critic_model)

    actor_model_config = TransformerConfig(
        d_model=64,
        nhead=2,
        num_layers=1,
        dim_feedforward=128,
        dropout=0.0,
        max_seq_len=64,
        use_z_as_query=True,
        append_actions=True,
        norm_first=True,
        prepend_z_token=True,
    )

    algorithm_config = HGTeamSACConfig.get_from_yaml()
    algorithm_config.gnn_mode = "learned_query"
    algorithm_config.embedding_entropy_coef = 0.0
    algorithm_config.embedding_diversity_coef = 0.0
    algorithm_config.stochastic_z = False
    algorithm_config.z_dim = 8
    algorithm_config.hypernet_actor_feature_dim = 16
    algorithm_config.gnn_norm_class = None
    algorithm_config.gnn_num_layers = 1
    algorithm_config.gnn_heads = 1
    algorithm_config.gnn_concat_heads = False
    algorithm_config.gnn_use_beta = True
    algorithm_config.gnn_self_loops = True
    algorithm_config.split_z = False
    algorithm_config.stochastic_z_query = False
    algorithm_config.use_vib = False
    algorithm_config.alpha_init = 1.0
    algorithm_config.target_entropy = "auto"
    algorithm_config.num_qvalue_nets = 2
    algorithm_config.fixed_alpha = False
    algorithm_config.detach_action_from_q = False
    algorithm_config.detach_z_from_transformer = True
    algorithm_config.critic_use_mu = True
    algorithm_config.lr_actor = 3e-4
    algorithm_config.lr_encoder = 1e-4
    algorithm_config.lr_critic = 3e-4
    algorithm_config.encoder_update_mode = encoder_update_mode

    task = PowerGridworldVariableTask.EVOVERNIGHT13NODE_VPP.get_from_yaml()
    task.config["reward_scale"] = 100

    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.sampling_device = "cpu"
    experiment_config.train_device = "cpu"
    experiment_config.collection_policy_device = "cpu"
    experiment_config.share_policy_params = True
    experiment_config.evaluation_static = False
    experiment_config.parallel_collection = False
    experiment_config.max_n_iters = None
    experiment_config.lr = 3e-4
    experiment_config.max_n_frames = 16
    experiment_config.evaluation_interval = 10_000
    experiment_config.evaluation_episodes = 1
    experiment_config.project_name = "PowerGridworldVariable_VPP_smoke"
    experiment_config.loggers = []
    experiment_config.off_policy_memory_size = 16
    experiment_config.off_policy_train_batch_size = 4
    experiment_config.off_policy_collected_frames_per_batch = 4
    experiment_config.off_policy_n_optimizer_steps = 1
    experiment_config.off_policy_init_random_frames = 4
    experiment_config.off_policy_n_envs_per_worker = 1
    experiment_config.soft_target_update = True
    experiment_config.polyak_tau = 0.005

    os.environ.setdefault("WANDB_MODE", "disabled")

    return Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=actor_model_config,
        critic_model_config=critic_model_config,
        seed=0,
        config=experiment_config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a low-frame HGTeamSAC smoke test.")
    parser.add_argument(
        "--encoder-update-mode",
        choices=("accumulated", "separate_forward", "coop_encoder"),
        default="coop_encoder",
        help="HGTeamSAC encoder update schedule to validate.",
    )
    parser.add_argument(
        "--critic-model",
        choices=("heterognn", "edgeweightedhgt"),
        default="heterognn",
        help="Critic graph model to smoke-test; defaults preserve existing behavior.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Building HGTeamSAC smoke-test experiment")
    experiment = build_experiment(
        encoder_update_mode=args.encoder_update_mode,
        critic_model=args.critic_model,
    )
    print(
        "Starting smoke test: 16 frames, CPU-only, no logging, "
        f"encoder_update_mode={args.encoder_update_mode}, critic_model={args.critic_model}"
    )
    experiment.run()
    print("Smoke test completed")


if __name__ == "__main__":
    main()
