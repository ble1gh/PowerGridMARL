#!/bin/bash
#SBATCH --account=notchpeak-shared-short
#SBATCH --partition=notchpeak-shared-short
#SBATCH --qos=notchpeak-shared-short
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -o MAPPOslurmjob-%j.out-%N
#SBATCH -e MAPPOslurmjob-%j.err-%N

# Enable GPU usage
# SBATCH --gres=gpu:1

# Set up the temporary directory
SCRDIR=/scratch/general/vast/$USER/$SLURM_JOB_ID
mkdir -p $SCRDIR

# Copy the project to scratch directory for faster I/O, excluding old results
echo "Copying project to scratch directory..."
rsync -a --exclude='MASACresults_*' --exclude='results_*' --exclude='MADDPGresults_*' /uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL/ $SCRDIR/PowerGridMARL/
echo "Copy complete. Starting experiment..."
cd $SCRDIR/PowerGridMARL

# Load required modules
module load benchmarl/3.10.3

export PYTHONPATH=$SCRDIR/PowerGridMARL/PowerGridworld:$SCRDIR/PowerGridMARL/BenchMARL:$PYTHONPATH

# Enable GPU usage
# export CUDA_VISIBLE_DEVICES="0"

# Force PyTorch to use GPU with optimized settings for MAPPO
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
# export CUDA_LAUNCH_BLOCKING=0
# export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Optimize CPU-GPU data transfers
# export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1
# export CUDA_CACHE_DISABLE=0

# Optimize for single-node computation with 32 cores - MAPPO can handle more threading
export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20
export NUMBA_NUM_THREADS=8

# Increase process and file limits for parallel environments
ulimit -n 8192    # Increase file descriptor limit
ulimit -u 8192    # Increase process limit

# Show current limits and compare to SLURM allocation
echo "Node: $SLURM_NODELIST"
echo "ulimit -n (open files): $(ulimit -n)"
echo "ulimit -u (max user processes): $(ulimit -u)"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_MEM_PER_NODE: $SLURM_MEM_PER_NODE"

# Print job info
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"

# Check GPU availability
# echo "GPU Info:"
# nvidia-smi
# echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Check if PyTorch can see GPU
# python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"

# Start background GPU monitoring
# nvidia-smi dmon -d 10 -s pucvmet -f $SCRDIR/gpu_usage.log &
# GPU_MONITOR_PID=$!

# MAPPO configuration optimized for 11 agents - on-policy with GAE
python BenchMARL/benchmarl/run.py algorithm=mappo task=PowerGridworld/evovernight13node_simple \
    model=layers/mlp \
    "model.num_cells=[32,32]" \
    algorithm.share_param_critic=False \
    algorithm.clip_epsilon=0.1 \
    algorithm.entropy_coef=0.1 \
    algorithm.critic_coef=0.01 \
    algorithm.lmbda=0.95 \
    experiment.train_device=cpu \
    experiment.sampling_device=cpu \
    experiment.lr=0.00000001 \
    experiment.on_policy_collected_frames_per_batch=4096 \
    experiment.on_policy_n_envs_per_worker=32 \
    experiment.on_policy_n_minibatch_iters=3 \
    experiment.on_policy_minibatch_size=256 \
    experiment.evaluation_interval=4096 \
    experiment.evaluation_episodes=3 \
    experiment.parallel_collection=true \
    experiment.collect_with_grad=false \
    experiment.clip_grad_val=1.0 \
    "experiment.loggers=[wandb]" \
    experiment.create_json=false \
    task.unserved_penalty=10.0 \
    task.reward_scale=0.0001

# Stop GPU monitoring
# kill $GPU_MONITOR_PID 2>/dev/null || true

echo "Job completed at: $(date)"

# Create results directory in home
RESULTS_DIR=/uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL/MAPPOresults_$(date +%Y%m%d)_${SLURM_JOB_ID}
mkdir -p $RESULTS_DIR

# Copy the actual training results from the outputs directory
if [ -d "$SCRDIR/PowerGridMARL/outputs" ]; then
    cp -r $SCRDIR/PowerGridMARL/outputs/* $RESULTS_DIR/ 2>/dev/null || true
fi

# Copy GPU monitoring log back to home directory
# cp $SCRDIR/gpu_usage.log $RESULTS_DIR/ 2>/dev/null || true

# Copy the SLURM script for reference
cp $SCRDIR/PowerGridMARL/EV_Charging_MAPPO_1node.sbatch $RESULTS_DIR/ 2>/dev/null || true

# Copy SLURM output and error files to results directory and clean up from main directory
cp /uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL/MAPPOslurmjob-${SLURM_JOB_ID}.out-* $RESULTS_DIR/ 2>/dev/null || true
cp /uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL/MAPPOslurmjob-${SLURM_JOB_ID}.err-* $RESULTS_DIR/ 2>/dev/null || true

# Clean up SLURM output files from main directory
rm -f /uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL/MAPPOslurmjob-${SLURM_JOB_ID}.out-* 2>/dev/null || true
rm -f /uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL/MAPPOslurmjob-${SLURM_JOB_ID}.err-* 2>/dev/null || true

# Clean up
cd /uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL
rm -rf $SCRDIR

echo "MAPPO training completed successfully!"
