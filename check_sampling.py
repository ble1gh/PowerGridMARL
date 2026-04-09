import os
import sys

import numpy as np
import torch

# Mocking necessary parts for the test
sys.path.append(os.path.join(os.getcwd(), "BenchMARL"))
from benchmarl.environments.PowerGridworldVariable.common import PowerGridworldVariableTask


def test_sampling():
    print("Initializing Environment...")
    # Load task config properly
    task = PowerGridworldVariableTask.EVOVERNIGHT13NODE_SIMPLE.get_from_yaml()

    # Create environment creation function
    env_fn = task.get_env_fun(
        num_envs=1,
        continuous_actions=True,
        seed=42,  # Match debug script
        device="cpu",
    )

    env = env_fn()
    torch.manual_seed(42)
    np.random.seed(42)

    print("\n--- Testing Sampling with Seed 42 ---")
    td = env.reset()
    active_count = td["active_mask"].sum().item()
    print(f"Seed 42 First Episode Active Count: {active_count}")


if __name__ == "__main__":
    test_sampling()
