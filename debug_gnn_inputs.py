"""Debug script to inspect GNN inputs and gradient flow."""
import sys
import os
import torch
import torch_geometric.nn as tgnn
from torch import nn

sys.path.append(os.path.join(os.getcwd(), "BenchMARL"))

from benchmarl.algorithms import HGTeamConfig
from benchmarl.models import HeteroGnnConfig, MlpConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.environments.PowerGridworldGraph.common import PowerGridworldGraphTask

def main():
    print("=" * 70)
    print("GNN INPUT INSPECTION")
    print("=" * 70)
    
    # Minimal config for debugging
    base_actor_config = MlpConfig(
        num_cells=[128, 128],
        activation_class=nn.Tanh,
        layer_class=nn.Linear
    )
    
    critic_gnn_config = HeteroGnnConfig(
        topology="adjacency",
        self_loops=True, 
        gnn_class=tgnn.TransformerConv,
        gnn_kwargs={"heads": 4, "concat": False, "beta": True},
        grid_edge_keys={
            "line_adjacency": "line_adjacency",
            "transformer_adjacency": "transformer_adjacency",
            "switch_adjacency": "switch_adjacency"
        },
        edge_features_dims={
            "line_adjacency": 3, "transformer_adjacency": 3, "switch_adjacency": 1,
            "interaction": 0, "mapping": 0, "mapping_rev": 0
        },
        node_features_keys={"grid_node": "grid_node_features"},        
        node_features_dims={"grid_node": 2},
        agent_node_index_key="agent_grid_edge_index",
        exclude_observations_from_node_features=False,
        cat_observations_to_output=False,
        num_layers=2,
        pos_features=0, vel_features=0, edge_radius=0
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
        use_hypernetwork=True,
        hypernet_hidden_dim=32,
        hypernet_feature_dim=64,
        stochastic_hypernet=True,
        embedding_entropy_coef=0,
        embedding_diversity_coef=0,
    )

    task = PowerGridworldGraphTask.EVOVERNIGHT13NODE_SIMPLE.get_from_yaml()

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
        config=experiment_config
    )
    
    # Collect data
    batch = next(iter(experiment.collector))
    
    print("\n1. Actor HeteroGNN Config (Parameter Generator):")
    print("-" * 50)
    
    # Find the HeteroGNN in the actor
    actor = experiment.losses["agents"].actor_network
    gnn_module = None
    for name, module in actor.named_modules():
        if "HeteroGNN" in type(module).__name__:
            gnn_module = module
            print(f"Found: {name}")
            print(f"  exclude_observations_from_node_features: {module.exclude_observations_from_node_features}")
            print(f"  cat_observations_to_output: {module.cat_observations_to_output}")
            print(f"  use_dummy_node_features: {module.use_dummy_node_features}")
            print(f"  input_features: {module.input_features}")
            print(f"  output_features: {module.output_features}")
            print(f"  node_types: {module.node_types}")
            print(f"  edge_types: {module.edge_types}")
            break
    
    print("\n2. Batch observation keys:")
    print("-" * 50)
    for k in batch.keys(True, True):
        if "agents" in str(k) or "grid" in str(k):
            val = batch.get(k)
            if isinstance(val, torch.Tensor):
                print(f"  {k}: shape={val.shape}, dtype={val.dtype}")
    
    print("\n3. Tracing GNN forward pass manually:")
    print("-" * 50)
    
    # Hook to capture x_dict
    captured = {}
    
    def hook_fn(module, input, output):
        # Capture the x_dict inside _forward
        pass
    
    # Trace what inputs go to the GNN
    if gnn_module:
        # Direct call
        batch_copy = batch.clone()
        
        # Patch the forward to print x_dict
        original_forward = gnn_module._forward
        
        def patched_forward(tensordict):
            from tensordict.utils import _unravel_key_to_tuple
            import torch.nn.functional as F
            
            device = "cpu"
            x_dict = {}
            
            # Gather inputs for each agent group
            for group in gnn_module.agent_groups:
                observations = []
                if not gnn_module.exclude_observations_from_node_features or gnn_module.cat_observations_to_output:
                    observations = [
                        tensordict.get(in_key)
                        for in_key in gnn_module.in_keys
                        if group in _unravel_key_to_tuple(in_key)
                        and _unravel_key_to_tuple(in_key)[-1] not in (gnn_module.position_key, gnn_module.velocity_key)
                    ]
                
                input_list = []
                if not gnn_module.exclude_observations_from_node_features:
                    input_list.extend(observations)
                
                if not input_list and gnn_module.use_dummy_node_features:
                    # This is the problem!
                    ref = None
                    for k in tensordict.keys(True, True):
                        if group in _unravel_key_to_tuple(k):
                            ref = tensordict.get(k)
                            break
                    
                    batch_size = ref.shape[:-2] if hasattr(ref, 'shape') else tensordict.batch_size
                    n_agents = ref.shape[-2] if hasattr(ref, 'shape') else 1
                    
                    dummy = torch.zeros(*batch_size, n_agents, 1, device=device, dtype=torch.float)
                    print(f"\n  ⚠️  DUMMY NODE FEATURES CREATED for {group}:")
                    print(f"     Shape: {dummy.shape}")
                    print(f"     Values: all zeros (no gradient flow!)")
                    input_list.append(dummy)
                
                if input_list:
                    x_group = torch.cat(input_list, dim=-1)
                    x_dict[group] = x_group
                    print(f"\n  x_dict['{group}']: shape={x_group.shape}, requires_grad={x_group.requires_grad}")
                    print(f"     min={x_group.min().item():.4f}, max={x_group.max().item():.4f}, std={x_group.std().item():.4f}")
            
            # Get grid_node features
            if gnn_module.node_features_keys:
                for node_type, key in gnn_module.node_features_keys.items():
                    full_key = gnn_module._get_key_terminating_with(list(tensordict.keys(True, True)), key, None)
                    if full_key:
                        val = tensordict.get(full_key)
                        x_dict[node_type] = val
                        print(f"\n  x_dict['{node_type}']: shape={val.shape}, requires_grad={val.requires_grad}")
                        print(f"     min={val.min().item():.4f}, max={val.max().item():.4f}, std={val.std().item():.4f}")
            
            return original_forward(tensordict)
        
        gnn_module._forward = patched_forward
        
        # Run actor forward
        print("\nRunning actor forward pass...")
        batch_out = actor(batch_copy)
        
        # Restore
        gnn_module._forward = original_forward
    
    print("\n4. ROOT CAUSE ANALYSIS:")
    print("-" * 50)
    print("""
The HeteroGNN in the hypernetwork is configured with:
  exclude_observations_from_node_features=True
  
This means agent nodes get DUMMY FEATURES (zeros), so:
- Gradients CAN'T flow through constant zero inputs
- Only the mapping_rev edges (grid→agent) can pass gradients
  because they bring information FROM grid nodes TO agents
- All agent→agent edges (interaction) have zero gradients

SOLUTION: Set exclude_observations_from_node_features=False in the 
parameter generator GNN config, OR use a learnable embedding instead
of dummy zeros.
    """)

if __name__ == "__main__":
    main()
