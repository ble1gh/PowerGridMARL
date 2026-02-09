import sys
import os
import torch
import torch_geometric.nn as tgnn
from torch import nn

# Ensure BenchMARL is in path
sys.path.append(os.path.join(os.getcwd(), "BenchMARL"))

from benchmarl.algorithms import HGTeamConfig
from benchmarl.models import HeteroGnnConfig, MlpConfig, SequenceModelConfig
from benchmarl.models.lstm import LstmConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.environments.PowerGridworldVariable.common import PowerGridworldVariableTask

def main():
    print("Preparing DEBUG Experiment...")

    # 1. Model Configuration (Same as full run)
    hetero_gnn_base_config = HeteroGnnConfig(
        topology="adjacency",
        self_loops=True, 
        gnn_class=tgnn.TransformerConv,
        gnn_kwargs={
            "heads": 4,
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
            "grid_node": "grid_node_features"
        },
        node_features_dims={
            "grid_node": 2
        },
        agent_node_index_key="agent_grid_edge_index",
        exclude_observations_from_node_features=True,
        cat_observations_to_output=True,
        num_layers=3,
        pos_features=0, vel_features=0, edge_radius=0
    )

    mlp_head_config = MlpConfig(
        num_cells=[128, 128],
        activation_class=nn.Tanh,
        layer_class=nn.Linear,
    )

    # Critic Configuration (Same as full run)
    critic_gnn_config = HeteroGnnConfig(
        topology="adjacency",
        self_loops=True, 
        gnn_class=tgnn.TransformerConv,
        gnn_kwargs={
            "heads": 4,
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
            "grid_node": "grid_node_features"
        },
        node_features_dims={
            "grid_node": 2
        },
        agent_node_index_key="agent_grid_edge_index",
        exclude_observations_from_node_features=False,
        cat_observations_to_output=False,
        num_layers=3,
        pos_features=0, vel_features=0, edge_radius=0
    )

    # HYPERNETWORK CONFIGURATION (Same as full run)
    # Stream B: Functional Actor (MLP)
    # base_actor_config = MlpConfig(
    #     num_cells=[128, 128, 64],
    #     activation_class=nn.Tanh,
    #     layer_class=nn.Linear,
    # )

    base_actor_config = LstmConfig(
        hidden_size=128,
        n_layers=1,          # 1 layer is standard for RL; 2+ = stacked LSTM, harder to train
        bias=True,
        dropout=0.0,         # No dropout (single layer anyway)
        compile=False,       # torch.compile can help speed but can cause issues with vmap
        mlp_num_cells=[128, 128, 64],           # MLP head after LSTM output
        mlp_layer_class=nn.Linear,
        mlp_activation_class=nn.ReLU,
    )

    actor_model_config = base_actor_config
    critic_model_config = critic_gnn_config

    # 2. Algorithm Configuration - load from yaml and override as needed
    algorithm_config = HGTeamConfig.get_from_yaml()
    
    # Override specific values (same as full run)
    algorithm_config.entropy_coef = 0.05
    algorithm_config.gnn_mode = "concat"  # Options: "none", "concat", "hypernetwork"
    algorithm_config.embedding_entropy_coef = 0.1
    algorithm_config.embedding_diversity_coef = 0.01

    # 3. Task - PowerGridworldVariable
    task = PowerGridworldVariableTask.EVOVERNIGHT13NODE_VPP.get_from_yaml()

    # 4. DEBUG CONFIG (Small limits for quick testing)
    experiment_config = ExperimentConfig.get_from_yaml()
    
    # Apply Overrides Manually (same structure as full run, but smaller)
    experiment_config.sampling_device = "cpu"
    experiment_config.train_device = "cuda"
    experiment_config.share_policy_params = True
    experiment_config.lr = 3e-4
    experiment_config.evaluation_episodes = 1
    
    # DEBUG: Small scale for quick testing
    experiment_config.parallel_collection = False
    experiment_config.on_policy_n_envs_per_worker = 1
    experiment_config.on_policy_collected_frames_per_batch = 100
    experiment_config.on_policy_minibatch_size = 10
    experiment_config.on_policy_n_minibatch_iters = 2
    experiment_config.max_n_frames = 200
    experiment_config.evaluation_interval = 200
    
    # DEBUG: No logging
    experiment_config.loggers = []
    experiment_config.create_json = False
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
