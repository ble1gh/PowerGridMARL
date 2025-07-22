#!/bin/bash
#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu
#SBATCH --qos=notchpeak-gpu
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -o MASACslurmjob-%j.out-%N
#SBATCH -e MASACslurmjob-%j.err-%N

# Set up the temporary directory
SCRDIR=/scratch/general/vast/$USER/$SLURM_JOB_ID
mkdir -p $SCRDIR

# Copy the project to scratch directory for faster I/O, excluding old results
echo "Copying project to scratch directory..."
rsync -a --exclude='MASACresults_*' --exclude='results_*' /uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL/ $SCRDIR/PowerGridMARL/
echo "Copy complete. Starting experiment..."
cd $SCRDIR/PowerGridMARL

# Load required modules
module load benchmarl/3.10.3

export PYTHONPATH=$SCRDIR/PowerGridMARL/PowerGridworld:$SCRDIR/PowerGridMARL/BenchMARL:$PYTHONPATH

# Enable GPU usage
export CUDA_VISIBLE_DEVICES="0"

# Force PyTorch to use GPU with optimized settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Optimize CPU-GPU data transfers
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1
export CUDA_CACHE_DISABLE=0

# Optimize for single-node computation with 32 cores - scale up for 11 agents
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMBA_NUM_THREADS=2

# Increase process and file limits for parallel environments
ulimit -n 4096    # Increase file descriptor limit
ulimit -u 2048    # Increase process limit

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
echo "GPU Info:"
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Check if PyTorch can see GPU
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"

# Start background GPU monitoring
nvidia-smi dmon -d 10 -s pucvmet -f $SCRDIR/gpu_usage.log &
GPU_MONITOR_PID=$!

# Optimized MASAC configuration for 11 agents - scale up parallelization
python BenchMARL/benchmarl/run.py algorithm=masac task=PowerGridworld/evovernight13node_simple \
    model=layers/mlp \
    "model.num_cells=[128,128]" \
    algorithm.share_param_critic=true \
    algorithm.alpha_init=1.0 \
    experiment.train_device=cuda \
    experiment.sampling_device=cpu \
    experiment.buffer_device=cpu \
    experiment.lr=0.0001 \
    experiment.polyak_tau=0.001 \
    experiment.off_policy_n_envs_per_worker=64 \
    experiment.off_policy_collected_frames_per_batch=32000 \
    experiment.off_policy_n_optimizer_steps=800 \
    experiment.off_policy_train_batch_size=4096 \
    experiment.evaluation_interval=32000 \
    experiment.evaluation_episodes=3 \
    experiment.parallel_collection=true \
    experiment.collect_with_grad=false \
    experiment.clip_grad_val=2.0 \
    experiment.off_policy_init_random_frames=5000 \
    "experiment.loggers=[wandb]" \
    experiment.create_json=false

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null || true

echo "Job completed at: $(date)"

# Create results directory in home
RESULTS_DIR=/uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL/MASACresults_$(date +%Y%m%d)_${SLURM_JOB_ID}
mkdir -p $RESULTS_DIR

# Copy the actual training results from the outputs directory
if [ -d "$SCRDIR/PowerGridMARL/outputs" ]; then
    cp -r $SCRDIR/PowerGridMARL/outputs/* $RESULTS_DIR/ 2>/dev/null || true
fi

# Copy GPU monitoring log back to home directory
cp $SCRDIR/gpu_usage.log $RESULTS_DIR/ 2>/dev/null || true

# Copy the SLURM script for reference
cp $SCRDIR/PowerGridMARL/EV_Charging_MASAC_1node.sbatch $RESULTS_DIR/ 2>/dev/null || true

# Copy SLURM output and error files to results directory and clean up from main directory
cp /uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL/MASACslurmjob-${SLURM_JOB_ID}.out-* $RESULTS_DIR/ 2>/dev/null || true
cp /uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL/MASACslurmjob-${SLURM_JOB_ID}.err-* $RESULTS_DIR/ 2>/dev/null || true

# Clean up SLURM output files from main directory
rm -f /uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL/MASACslurmjob-${SLURM_JOB_ID}.out-* 2>/dev/null || true
rm -f /uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL/MASACslurmjob-${SLURM_JOB_ID}.err-* 2>/dev/null || true

# Clean up
cd /uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL
rm -rf $SCRDIR