#!/bin/bash
#SBATCH --account=rai
#SBATCH --partition=rai-gpu-grn
#SBATCH --qos=rai-gpu-grn
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=384GB
#
# GPU Request:
# Generic request (any GPU in partition):
#SBATCH --gres=gpu:1
#
# Specific GPU requests (uncomment one to ensure specific VRAM):
# #SBATCH --gres=gpu:rtx6000:1    # ~24GB VRAM
# #SBATCH --gres=gpu:l40s:1       # ~48GB VRAM
# #SBATCH --gres=gpu:a800:1       # ~80GB VRAM
# #SBATCH --gres=gpu:h100nvl:1       # ~141GB VRAM

#SBATCH -o HGTeamSACslurmjob-%j.out-%N
#SBATCH -e HGTeamSACslurmjob-%j.err-%N
#SBATCH --exclude=grn030


# Set up the temporary directory
SCRDIR=/scratch/general/vast/$USER/$SLURM_JOB_ID
mkdir -p $SCRDIR

# Copy the project to scratch directory for faster I/O, excluding old results
echo "Copying project to scratch directory..."
rsync -a \
    --exclude='*results_*' \
    --exclude='*slurmjob-*' \
    --exclude='outputs' \
    --exclude='wandb' \
    --exclude='.git' \
    /uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL/ $SCRDIR/PowerGridMARL/
echo "Copy complete. Starting experiment..."
cd $SCRDIR/PowerGridMARL

# Load required modules
module load benchmarl/nightly

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

# Optimize for single-node computation - avoid thread oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# Increase process and file limits for high parallel environments
ulimit -n 8192
ulimit -u 16384

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
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')"

# Start background GPU monitoring
nvidia-smi dmon -d 10 -s pucvmet -f $SCRDIR/gpu_usage.log &
GPU_MONITOR_PID=$!

# Run HGTeamSAC experiment
python run_hgteam_mlp.py \
    --algorithm sac \
    --seed 42 \
    --project-name "PowerGridworldVariable_VPP" \
    --wandb-name "HGTeamSAC_$(date +%Y%m%d)_${SLURM_JOB_ID}_2layergnn_mlp" \
    --max-n-frames 10000000 \
    --lr 1e-4 \
    --lr-actor 1e-4 \
    --lr-encoder 1e-4 \
    --lr-critic 1e-4 \
    --evaluation-interval 12288 \
    --evaluation-episodes 5 \
    --reward-scale 100000 \
    --gnn-mode learned_query \
    --embedding-entropy-coef 0 \
    --embedding-diversity-coef 0 \
    --stochastic-z False \
    --z-dim 32 \
    --hypernet-actor-feature-dim 64 \
    --gnn-norm-class LayerNorm \
    --split-z false \
    --z-token-dim 32 \
    --z-query-dim 32 \
    --stochastic-z-query true \
    --use-vib true \
    --alpha-init 0.2 \
    --target-entropy 0.0 \
    --min-alpha 0.01 \
    --num-qvalue-nets 2 \
    --fixed-alpha false \
    --detach-action-from-q false \
    --detach-z-from-transformer true \
    --critic-use-mu true \
    --off-policy-memory-size 500000 \
    --off-policy-train-batch-size 512 \
    --off-policy-collected-frames-per-batch 3072 \
    --off-policy-n-optimizer-steps 2 \
    --off-policy-init-random-frames 10000 \
    --off-policy-n-envs-per-worker 32 \
    --polyak-tau 0.005 \

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null || true

echo "Job completed at: $(date)"

# Create results directory in home
RESULTS_DIR=/uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL/HGTeamSACresults_$(date +%Y%m%d)_${SLURM_JOB_ID}
mkdir -p $RESULTS_DIR

# Copy the actual training results from the outputs directory
if [ -d "$SCRDIR/PowerGridMARL/outputs" ]; then
    cp -r $SCRDIR/PowerGridMARL/outputs/* $RESULTS_DIR/ 2>/dev/null || true
fi

# Copy GPU monitoring log back to home directory
cp $SCRDIR/gpu_usage.log $RESULTS_DIR/ 2>/dev/null || true

# Copy the SLURM script for reference
cp $SCRDIR/PowerGridMARL/EV_Charging_HGTeamSAC_1node.sbatch $RESULTS_DIR/ 2>/dev/null || true

# Copy SLURM output and error files to results directory and clean up from main directory
cp /uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL/HGTeamSACslurmjob-${SLURM_JOB_ID}.out-* $RESULTS_DIR/ 2>/dev/null || true
cp /uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL/HGTeamSACslurmjob-${SLURM_JOB_ID}.err-* $RESULTS_DIR/ 2>/dev/null || true

# Clean up SLURM output files from main directory
rm -f /uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL/HGTeamSACslurmjob-${SLURM_JOB_ID}.out-* 2>/dev/null || true
rm -f /uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL/HGTeamSACslurmjob-${SLURM_JOB_ID}.err-* 2>/dev/null || true

# Clean up
cd /uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL
