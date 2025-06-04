import subprocess

# Command to run gridworld code in its own virtual environment
command = [
    "/opt/anaconda3/envs/gridworld/bin/python",  # Path to Python in the gridworld environment
    "-c",  # Run the following Python code
    "from gridworld.multiagent_env import MultiAgentEnv; print('PowerGridworld is accessible!')"
]

try:
    # Run the command and capture the output
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    print(result.stdout)  # Print the output from the gridworld environment
except subprocess.CalledProcessError as e:
    print(f"Error running gridworld code: {e.stderr}")

# BenchMARL code runs in the current environment
from benchmarl.environments.common import Task
print("BenchMARL is accessible!")