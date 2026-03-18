import sys
import os
import torch
import torch_geometric.nn as tgnn
from torch import nn

# Ensure BenchMARL is in path
sys.path.append(os.path.join(os.getcwd(), "BenchMARL"))

from benchmarl.algorithms import HGTeamConfig
from benchmarl.models import HeteroGnnConfig, MlpConfig, SequenceModelConfig, TransformerConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.environments.PowerGridworldVariable.common import PowerGridworldVariableTask

def main():
    print("Preparing DEBUG Experiment...")

    # Critic Configuration (matches full run)
    critic_gnn_config = HeteroGnnConfig(
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

    # Actor: Transformer (matches full run)
    actor_model_config = TransformerConfig(
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

    critic_model_config = critic_gnn_config

    # Algorithm Configuration (matches full run)
    algorithm_config = HGTeamConfig.get_from_yaml()
    algorithm_config.entropy_coef = 0.5
    algorithm_config.gnn_mode = "concat"
    algorithm_config.embedding_entropy_coef = 1
    algorithm_config.embedding_diversity_coef = 0.001
    algorithm_config.stochastic_hypernet = True
    algorithm_config.hypernet_hidden_dim = 32
    algorithm_config.hypernet_feature_dim = 64

    # Task
    task = PowerGridworldVariableTask.EVOVERNIGHT13NODE_VPP.get_from_yaml()

    # DEBUG CONFIG (minimal scale for quick local testing)
    experiment_config = ExperimentConfig.get_from_yaml()

    experiment_config.sampling_device = "cpu"
    experiment_config.train_device = "cuda" if torch.cuda.is_available() else "cpu"
    experiment_config.collection_policy_device = "cuda"  # Run GNN+Transformer on GPU during collection
    experiment_config.share_policy_params = True
    experiment_config.lr = 5e-6
    experiment_config.evaluation_episodes = 1
    experiment_config.evaluation_static = False

    # DEBUG: Small scale — forces evaluation early so eval-path bugs surface
    experiment_config.parallel_collection = False
    experiment_config.on_policy_n_envs_per_worker = 2
    experiment_config.on_policy_collected_frames_per_batch = 200
    experiment_config.on_policy_minibatch_size = 50
    experiment_config.on_policy_n_minibatch_iters = 2
    experiment_config.max_n_frames = 400
    experiment_config.evaluation_interval = 200

    # DEBUG: No logging
    experiment_config.loggers = []
    experiment_config.create_json = True
    experiment_config.checkpoint_at_end = False

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=actor_model_config,
        critic_model_config=critic_model_config,
        seed=42,
        config=experiment_config
    )

    print("\nStarting DEBUG Run (This should finish in ~1 minute)...")
    experiment.run()
    print("DEBUG Run Completed Successfully!")

if __name__ == "__main__":
    main()
