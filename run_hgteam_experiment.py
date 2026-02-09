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
    print("Preparing Experiment...")

    hetero_gnn_base_config = HeteroGnnConfig(
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
        activation_class=nn.ReLU,
        layer_class=nn.Linear,
    )


    # Critic Configuration (Different from Actor)
    critic_gnn_config = HeteroGnnConfig(
        topology="adjacency",
        self_loops=True, 
        gnn_class=tgnn.TransformerConv,
        gnn_kwargs={
            "heads": 3, 
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
        cat_observations_to_output=False, # Do not pass observations through
        num_layers=4, # More layers than actor
        gnn_hidden_dim=32,  # Critical: without this, output_features=1 makes all hidden layers 1×heads=3 dim!
        #norm_class=nn.LayerNorm,  # Add LayerNorm for stability
        pos_features=0, vel_features=0, edge_radius=0
    )

    # critic_model_config = critic_gnn_config

    # HYPERNETWORK CONFIGURATION
    # 1. Define the Base Actor (Stream B)
    # This is the functional part that processes local observations
    # For Hypernetwork, we usually want this to be an MLP or LSTM processing raw obs
    # base_actor_config = MlpConfig(
    #     num_cells=[128, 128, 64],
    #     activation_class=nn.ReLU,
    #     layer_class=nn.Linear
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
    
    # Override specific values if needed
    algorithm_config.entropy_coef = 0.1
    algorithm_config.gnn_mode = "concat"  # Options: "none", "concat", "hypernetwork"
    algorithm_config.embedding_entropy_coef = 1
    algorithm_config.embedding_diversity_coef = 0.001
    algorithm_config.stochastic_hypernet = True
    algorithm_config.hypernet_hidden_dim = 32
    algorithm_config.hypernet_feature_dim = 64
    algorithm_config.gnn_norm_class = "LayerNorm" # Options: null, "LayerNorm", "BatchNorm1d", "InstanceNorm1d", "GroupNorm"

    # 3. Task
    task = PowerGridworldVariableTask.EVOVERNIGHT13NODE_VPP.get_from_yaml()

    # 4. DEBUG CONFIG (Small limits, CPU only, no logging)
    # Load default config and override specific values to avoid missing argument errors
    experiment_config = ExperimentConfig.get_from_yaml()
    
    # Apply Overrides Manually
    experiment_config.sampling_device = "cpu"
    experiment_config.train_device = "cuda"
    experiment_config.share_policy_params = True
    experiment_config.lr = 3e-4
    experiment_config.evaluation_episodes = 10
    experiment_config.evaluation_static = False  # Ensure evaluation sees different agent sets
    
    # Scale Up for GPU Utilization (24 CPUs available)
    experiment_config.parallel_collection = True
    experiment_config.on_policy_n_envs_per_worker = 256
    experiment_config.on_policy_collected_frames_per_batch = 24576 # Increase batch size
    experiment_config.on_policy_minibatch_size = 2000 # minibatches
    experiment_config.on_policy_n_minibatch_iters = 15 # Number of SGD epochs
    experiment_config.max_n_frames = 10_000_000 # Train for longer
    experiment_config.evaluation_interval = 24576
    
    # Logging Overrides
    experiment_config.project_name = "PowerGridworldVariable_VPP"
    experiment_config.loggers = ["wandb"]
    
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    import datetime
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    experiment_config.wandb_extra_kwargs = {
        "name": f"HGTeam_{date_str}_{job_id}"
    }

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=actor_model_config,
        critic_model_config=critic_model_config,
        seed=42,
        config=experiment_config
    )

    print("\nStarting Run")
    experiment.run()
    print("Run Completed Successfully!")

if __name__ == "__main__":
    main()
