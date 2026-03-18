import sys
import os
import torch
from torch import nn

# Ensure BenchMARL is in path
sys.path.append(os.path.join(os.getcwd(), "BenchMARL"))

from benchmarl.algorithms import MATConfig
from benchmarl.models import MlpConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.environments.PowerGridworldVariable.common import PowerGridworldVariableTask

def main():
    print("Preparing MAT Small Scale Test...")

    actor_model_config = MlpConfig(
        num_cells=[32, 32],
        activation_class=nn.ReLU,
        layer_class=nn.Linear
    )
    
    critic_model_config = MlpConfig(
        num_cells=[32, 32],
        activation_class=nn.ReLU,
        layer_class=nn.Linear
    )

    # Algorithm Configuration
    algorithm_config = MATConfig.get_from_yaml()
    algorithm_config.n_block = 1
    algorithm_config.n_embd = 32
    algorithm_config.n_head = 2

    # Task
    task = PowerGridworldVariableTask.EVOVERNIGHT13NODE_VPP.get_from_yaml()
    
    # Experiment Config - Minimal for quick test
    experiment_config = ExperimentConfig.get_from_yaml()
    
    experiment_config.sampling_device = "cpu"
    experiment_config.train_device = "cpu"  # Safe for login node
    experiment_config.share_policy_params = True
    experiment_config.lr = 1e-4
    experiment_config.evaluation_episodes = 2
    experiment_config.evaluation_static = False
    
    experiment_config.parallel_collection = False # Disable parallel for simple test
    experiment_config.on_policy_n_envs_per_worker = 2
    experiment_config.on_policy_collected_frames_per_batch = 100
    experiment_config.on_policy_minibatch_size = 20 
    experiment_config.on_policy_n_minibatch_iters = 2
    experiment_config.max_n_frames = 200 # Very short run
    experiment_config.evaluation_interval = 200
    
    experiment_config.project_name = "Test_MAT"
    experiment_config.loggers = [] # No logging
    
    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=actor_model_config,
        critic_model_config=critic_model_config,
        seed=42,
        config=experiment_config
    )

    print("\nStarting Test Run")
    experiment.run()
    print("Test Run Completed Successfully!")

if __name__ == "__main__":
    main()
