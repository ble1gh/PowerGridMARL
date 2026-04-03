import sys
import os
import argparse
import datetime
from pathlib import Path
import torch
from torch import nn

# Ensure BenchMARL is in path
sys.path.append(os.path.join(os.getcwd(), "BenchMARL"))

from benchmarl.algorithms import MATConfig
from benchmarl.models import MlpConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.environments.PowerGridworldVariable.common import PowerGridworldVariableTask


def parse_args():
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    default_results_dir = Path(__file__).resolve().parent / "results"

    parser = argparse.ArgumentParser(description="Run MAT experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project-name", type=str, default="PowerGridworldVariable_VPP")
    parser.add_argument("--wandb-name", type=str, default=f"MAT_{date_str}_{job_id}")
    parser.add_argument("--results-dir", type=Path, default=default_results_dir)
    return parser.parse_args()

def main():
    args = parse_args()
    print("Preparing MAT Experiment...")

    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(results_dir / "wandb"))

    # Models: MAT uses its own Transformer, but BenchMARL expects model configs provided.
    # Our MAT implementation ignores these MlpConfigs inside _get_mat_model, 
    # but they are needed to satisfy the Algorithm interface requirements (if any).
    # Specifically, _get_policy_for_loss in MAT wrapper uses _get_mat_model directly,
    # so these are mostly placeholders or used for standard wrappers if we implemented it that way.
    # However, Experiment class requires model_config.
    
    actor_model_config = MlpConfig(
        num_cells=[64, 64],
        activation_class=nn.ReLU,
        layer_class=nn.Linear
    )
    
    critic_model_config = MlpConfig(
        num_cells=[64, 64],
        activation_class=nn.ReLU,
        layer_class=nn.Linear
    )

    # 2. Algorithm Configuration
    algorithm_config = MATConfig.get_from_yaml()
    
    # Custom Overrides
    algorithm_config.entropy_coef = 0.01
    algorithm_config.n_block = 1
    algorithm_config.n_embd = 64
    algorithm_config.n_head = 1

    # 3. Task
    task = PowerGridworldVariableTask.EVOVERNIGHT13NODE_VPP.get_from_yaml()
    
    # Enable unserved penalty if desired (consistent with HGTeam experiments)
    # The default yaml likely has unserved_penalty=1000, reward_scale=1 based on previous fixes.

    # 4. Experiment Config
    experiment_config = ExperimentConfig.get_from_yaml()
    
    experiment_config.sampling_device = "cpu"
    experiment_config.train_device = "cuda"
    experiment_config.share_policy_params = True
    experiment_config.lr = 5e-5 # MAT often uses slightly lower LR
    experiment_config.evaluation_episodes = 10
    experiment_config.evaluation_static = False
    
    experiment_config.parallel_collection = True
    experiment_config.on_policy_n_envs_per_worker = 64
    experiment_config.on_policy_collected_frames_per_batch = 6144
    experiment_config.on_policy_minibatch_size = 1000 
    experiment_config.on_policy_n_minibatch_iters = 10
    experiment_config.max_n_frames = 10_000_000
    experiment_config.evaluation_interval = 6144
    
    experiment_config.project_name = args.project_name
    experiment_config.loggers = ["wandb"]
    experiment_config.save_folder = str(results_dir)
    experiment_config.wandb_extra_kwargs = {
        "name": args.wandb_name
    }

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=actor_model_config,
        critic_model_config=critic_model_config,
        seed=args.seed,
        config=experiment_config
    )

    print("\nStarting Run")
    experiment.run()
    print("Run Completed Successfully!")

if __name__ == "__main__":
    main()
