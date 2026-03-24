#!/bin/bash
#SBATCH --account=rai
#SBATCH --partition=rai-gpu-grn
#SBATCH --qos=rai-gpu-grn
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=384GB
#SBATCH --gres=gpu:1
#SBATCH -o Ablationslurmjob-%j.out-%N
#SBATCH -e Ablationslurmjob-%j.err-%N
#SBATCH --exclude=grn030

# ── Parameterized ablation experiment ──
# Required env vars (set by launch_all_ablation.sh):
#   SEED            - random seed (e.g. 42)
#   GNN_MODE        - gnn mode: concat | none
#   WANDB_GROUP     - wandb group name (e.g. physical_graph)
#   WANDB_PROJECT   - wandb project name (e.g. HGTeam_CDC_Ablation)
#   WANDB_TAGS      - space-separated tags (e.g. "ablation gnn")
#   USE_VIB         - true | false (default: true)
#   REWARD_SCALE    - reward scale (default: 100000)

SEED=${SEED:-42}
GNN_MODE=${GNN_MODE:-concat}
WANDB_GROUP=${WANDB_GROUP:-default}
WANDB_PROJECT=${WANDB_PROJECT:-HGTeam_CDC_Ablation}
WANDB_TAGS=${WANDB_TAGS:-ablation}
USE_VIB=${USE_VIB:-true}
REWARD_SCALE=${REWARD_SCALE:-100000}

# Set up the temporary directory
SCRDIR=/scratch/general/vast/$USER/$SLURM_JOB_ID
mkdir -p $SCRDIR

# Copy the project to scratch directory for faster I/O
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
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1
export CUDA_CACHE_DISABLE=0

# Optimize for single-node computation
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

ulimit -n 8192
ulimit -u 16384

# Print job info
echo "=== Ablation Experiment ==="
echo "GNN_MODE: $GNN_MODE | SEED: $SEED | GROUP: $WANDB_GROUP"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')"

# Start background GPU monitoring
nvidia-smi dmon -d 10 -s pucvmet -f $SCRDIR/gpu_usage.log &
GPU_MONITOR_PID=$!

# Run experiment
python run_hgteam_mlp.py \
    --seed $SEED \
    --project-name "$WANDB_PROJECT" \
    --wandb-name "HGTeam_${GNN_MODE}_seed${SEED}_${SLURM_JOB_ID}" \
    --wandb-group "$WANDB_GROUP" \
    --wandb-tags $WANDB_TAGS \
    --max-n-frames 10000000 \
    --lr 5e-5 \
    --evaluation-interval 12288 \
    --evaluation-episodes 5 \
    --n-envs-per-worker 32 \
    --frames-per-batch 12288 \
    --minibatch-size 256 \
    --n-minibatch-iters 5 \
    --entropy-coef 0.01 \
    --gnn-mode $GNN_MODE \
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
    --reward-scale $REWARD_SCALE \
    --use-vib $USE_VIB \
    --vib-warmup-frames 500000

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null || true
echo "Job completed at: $(date)"

# Copy results back
RESULTS_DIR=/uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL/Ablationresults_${GNN_MODE}_seed${SEED}_${SLURM_JOB_ID}
mkdir -p $RESULTS_DIR
if [ -d "$SCRDIR/PowerGridMARL/outputs" ]; then
    cp -r $SCRDIR/PowerGridMARL/outputs/* $RESULTS_DIR/ 2>/dev/null || true
fi
cp $SCRDIR/gpu_usage.log $RESULTS_DIR/ 2>/dev/null || true
cp $SCRDIR/PowerGridMARL/launch_ablation.sbatch $RESULTS_DIR/ 2>/dev/null || true

# Copy and clean SLURM logs
cp /uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL/Ablationslurmjob-${SLURM_JOB_ID}.out-* $RESULTS_DIR/ 2>/dev/null || true
cp /uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL/Ablationslurmjob-${SLURM_JOB_ID}.err-* $RESULTS_DIR/ 2>/dev/null || true
rm -f /uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL/Ablationslurmjob-${SLURM_JOB_ID}.out-* 2>/dev/null || true
rm -f /uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL/Ablationslurmjob-${SLURM_JOB_ID}.err-* 2>/dev/null || true

# Clean up scratch
cd /uufs/chpc.utah.edu/common/home/u1175377/PowerGridMARL
rm -rf $SCRDIR
