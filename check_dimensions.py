# Create a file called check_dimensions.py
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "BenchMARL"))
sys.path.append(os.path.join(os.getcwd(), "PowerGridworld"))

from benchmarl.environments.PowerGridworldVariable.common import PowerGridworldVariableTask

# Create the task and get a sample environment
task = PowerGridworldVariableTask.EVOVERNIGHT13NODE_SIMPLE.get_from_yaml()
env = task.get_env_fun(num_envs=1, continuous_actions=True, seed=42, device="cpu")()

# Reset and get observation
tensordict = env.reset()
print("Environment observation keys:", list(tensordict.keys()))


# Print all tensor shapes in the reset tensordict
def print_tensordict_shapes(td, prefix=""):
    for key, value in td.items():
        if hasattr(value, "shape"):
            print(f"{prefix}{key}: {value.shape}")
        else:
            print(f"{prefix}{key}: {type(value)}")
            if hasattr(value, "items"):
                print_tensordict_shapes(value, prefix + "  ")


print_tensordict_shapes(tensordict)
