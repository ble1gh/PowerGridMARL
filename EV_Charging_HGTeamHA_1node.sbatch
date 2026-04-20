#!/bin/bash
#SBATCH --account=rai
#SBATCH --partition=rai-gpu-grn
#SBATCH --qos=rai-gpu-grn
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#
# GPU Request:
# MIG slices available on this partition:
#   gpu:h200_1g.18gb  — 18 GB (grn025, 56 slices) — tight for 768+ minibatch
#   gpu:h200_2g.35gb  — 35 GB (grn026, 16 slices) — good for 1024 minibatch
#   gpu:h200_3g.71gb  — 71 GB (grn026, 8 slices)
#   gpu:h200          — 141 GB full card (grn027-032, 8 each) — long queue
#   gpu:1             — any available GPU (risky: may get 18 GB slice)
# #SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --gres=gpu:h200:1

#SBATCH -o /uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL/results/slurm/HGTeamHAslurmjob-%j.out-%N
#SBATCH -e /uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL/results/slurm/HGTeamHAslurmjob-%j.err-%N
# #SBATCH --exclude=grn030


PROJECT_ROOT=/uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL
RESULTS_ROOT=$PROJECT_ROOT/results
RESULTS_DIR=$RESULTS_ROOT/HGTeamHAresults_$(date +%Y%m%d)_${SLURM_JOB_ID}

mkdir -p "$RESULTS_DIR"


# Set up the temporary directory
SCRDIR=/scratch/general/vast/$USER/$SLURM_JOB_ID
mkdir -p $SCRDIR

# Copy the project to scratch directory for faster I/O, excluding old results
echo "Copying project to scratch directory..."
rsync -a \
    --exclude='*results_*' \
    --exclude='results' \
    --exclude='*slurmjob-*' \
    --exclude='outputs' \
    --exclude='wandb' \
    --exclude='.git' \
    $PROJECT_ROOT/ $SCRDIR/PowerGridMARL/
echo "Copy complete. Starting experiment..."
cd $SCRDIR/PowerGridMARL

# Load required modules
module load benchmarl/nightly

export PYTHONPATH=$SCRDIR/PowerGridMARL/PowerGridworld:$SCRDIR/PowerGridMARL/BenchMARL:$PYTHONPATH
export WANDB_DIR="$RESULTS_DIR/wandb"

# Enable GPU usage — respect SLURM's assignment (required for MIG slices)
# MIG slices use UUID-based device IDs; hardcoding "0" breaks them.
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="0"
fi

# Force PyTorch to use GPU with optimized settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Optimize CPU-GPU data transfers
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1
export CUDA_CACHE_DISABLE=0

# Optimize for single-node computation with 64 cores - maximize CPU utilization
# Set to 1 to avoid thread oversubscription when using 64 parallel environments
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 
export NUMBA_NUM_THREADS=1

# Increase process and file limits for high parallel environments
ulimit -n 8192     # File descriptor limit - increased for more parallel envs
ulimit -u 16384    # Process limit - increased for more parallel envs

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

# Run experiment — HAPPO variant
python run_hgteam_mlp.py \
    --seed 20 \
    --project-name "PowerGridworldVariable_VPP" \
    --wandb-name "HGTeamHA_$(date +%Y%m%d)_${SLURM_JOB_ID}_lmbda0.99_separate_forward_2iter_minib2048_lr3e-4_novib" \
    --results-dir "$RESULTS_DIR" \
    --max-n-frames 50000000 \
    --lr 3e-4 \
    --lr-encoder 3e-4 \
    --encoder-n-optimizer-steps 2 \
    --evaluation-interval 24576 \
    --evaluation-episodes 20 \
    --n-envs-per-worker 32 \
    --frames-per-batch 24576 \
    --minibatch-size 2048 \
    --n-minibatch-iters 2 \
    --entropy-coef 0.01 \
    --algorithm happo \
    --fixed-order false \
    --gnn-mode concat \
    --embedding-entropy-coef 0 \
    --embedding-diversity-coef 0 \
    --stochastic-z true \
    --z-dim 32 \
    --hypernet-actor-feature-dim 64 \
    --gnn-norm-class LayerNorm \
    --split-z false \
    --z-token-dim 32 \
    --z-query-dim 32 \
    --stochastic-z-query true \
    --scale-lb 0.0001 \
    --reward-scale 100000 \
    --use-vib false \
    --vib-warmup-frames 500000 \
    --embedding-viz-interval 5 \
    --lmbda 0.99 \
    --critic-use-other-actions true \
    --encoder-update-mode separate_forward \
    
# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null || true

echo "Job completed at: $(date)"

# Copy the actual training results from the outputs directory
if [ -d "$SCRDIR/PowerGridMARL/outputs" ]; then
    cp -r $SCRDIR/PowerGridMARL/outputs/* $RESULTS_DIR/ 2>/dev/null || true
fi

# Copy GPU monitoring log back to home directory
cp $SCRDIR/gpu_usage.log $RESULTS_DIR/ 2>/dev/null || true

# Copy the SLURM script for reference
cp $SCRDIR/PowerGridMARL/EV_Charging_HGTeamHA_1node.sbatch $RESULTS_DIR/ 2>/dev/null || true

# Clean up
cd $PROJECT_ROOT
rm -rf $SCRDIR
