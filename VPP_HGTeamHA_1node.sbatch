#!/bin/bash
#SBATCH --account=rai
#SBATCH --partition=rai-gpu-grn
#SBATCH --qos=rai-gpu-grn
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
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
#SBATCH -e /uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL/results/slurm/HGTeamHAslurmjob-%j.out-%N
# #SBATCH --exclude=grn030


PROJECT_ROOT=/uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL
RESULTS_ROOT=$PROJECT_ROOT/results
RESULTS_DIR=$RESULTS_ROOT/HGTeamHAresults_$(date +%Y%m%d)_${SLURM_JOB_ID}
SCR_RESULTS_DIR=/scratch/general/vast/$USER/$SLURM_JOB_ID/run_results

mkdir -p "$RESULTS_DIR"
mkdir -p "$SCR_RESULTS_DIR"


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
echo "Scratch results directory: $SCR_RESULTS_DIR"
echo "Scratch checkpoint directory (expected): $SCR_RESULTS_DIR/checkpoints"

# Load required modules
module load benchmarl/nightly

export PYTHONPATH=$SCRDIR/PowerGridMARL/PowerGridworld:$SCRDIR/PowerGridMARL/BenchMARL:$PYTHONPATH
export WANDB_DIR="$SCR_RESULTS_DIR/wandb"

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
    --wandb-name "HGTeamHA_$(date +%Y%m%d)_${SLURM_JOB_ID}_fixedpop_coop_encoder_otheractions_heterognn_H0.1_freeze1M" \
    --results-dir "$SCR_RESULTS_DIR" \
    --max-n-frames 50000000 \
    --lr 3e-4 \
    --lr-encoder 3e-4 \
    --encoder-n-optimizer-steps 1 \
    --evaluation-interval 24576 \
    --evaluation-episodes 20 \
    --n-envs-per-worker 8 \
    --frames-per-batch 24576 \
    --minibatch-size 2048 \
    --n-minibatch-iters 2 \
    --entropy-coef 0.05 \
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
    --scale-lb 0.01 \
    --reward-scale 100000 \
    --use-vib true \
    --vib-warmup-frames 2000000 \
    --vib-beta 0.00001 \
    --embedding-viz-interval 5 \
    --lmbda 0.99 \
    --critic-use-other-actions true \
    --encoder-update-mode coop_encoder \
    --heterognn-type heterognn \
    --encoder-freeze-after-frames 1000000 \
# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null || true

echo "Job completed at: $(date)"

# Copy non-checkpoint artifacts from scratch back to home results.
# Checkpoints are intentionally left on scratch for manual promotion.
if [ -d "$SCR_RESULTS_DIR" ]; then
    rsync -a --exclude='checkpoints' "$SCR_RESULTS_DIR/" "$RESULTS_DIR/" 2>/dev/null || true
fi

# Copy GPU monitoring log back to home directory
cp $SCRDIR/gpu_usage.log $RESULTS_DIR/ 2>/dev/null || true

# Copy the SLURM script for reference
cp $PROJECT_ROOT/VPP_HGTeamHA_1node.sbatch $RESULTS_DIR/ 2>/dev/null || true

# Print manual checkpoint promotion instructions into SLURM .out
echo ""
echo "==== CHECKPOINT LOCATION ===="
echo "Checkpoint directory: $SCR_RESULTS_DIR/checkpoints"
echo "To manually promote the newest checkpoint to home, run:"
echo "mkdir -p \"$RESULTS_DIR/checkpoints_kept\" && cp \"\$(ls -1t \"$SCR_RESULTS_DIR\"/checkpoints/checkpoint_*.pt | head -n 1)\" \"$RESULTS_DIR/checkpoints_kept/\""
echo "To promote a specific checkpoint, run:"
echo "cp \"$SCR_RESULTS_DIR/checkpoints/checkpoint_<FRAMES>.pt\" \"$RESULTS_DIR/checkpoints_kept/\""
echo "============================="

# Clean up
cd $PROJECT_ROOT
# Keep scratch so checkpoints are available for manual promotion.
# Set PURGE_SCRATCH_ON_EXIT=1 in the environment to delete scratch automatically.
if [ "${PURGE_SCRATCH_ON_EXIT:-0}" = "1" ]; then
    rm -rf $SCRDIR
fi
