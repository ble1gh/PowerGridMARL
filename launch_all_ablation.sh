#!/bin/bash
# Launch all ablation experiments for CDC 2026 paper.
# Experiment 1: HGTeam with physical graph (concat GNN) × 5 seeds
# Experiment 2: HGTeam without GNN (none mode) × 3 seeds
#
# Usage: bash launch_all_ablation.sh [--dry-run]

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE — no jobs will be submitted ==="
fi

PROJECT="HGTeam_CDC_Ablation"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

submit() {
    local seed=$1 gnn_mode=$2 group=$3 tags=$4

    echo "  Submitting: gnn_mode=$gnn_mode seed=$seed group=$group"
    if $DRY_RUN; then
        return
    fi

    sbatch \
        --export=SEED=$seed,GNN_MODE=$gnn_mode,WANDB_GROUP=$group,WANDB_PROJECT=$PROJECT,WANDB_TAGS="$tags",USE_VIB=true,REWARD_SCALE=100000 \
        "$SCRIPT_DIR/launch_ablation.sbatch"
}

echo ""
echo "=== Experiment 1: Physical Graph (concat GNN) × 5 seeds ==="
for SEED in 42 123 456 789 1024; do
    submit $SEED concat physical_graph "ablation gnn concat"
done

echo ""
echo "=== Experiment 2: No GNN (baseline) × 3 seeds ==="
for SEED in 42 123 456; do
    submit $SEED none no_gnn "ablation baseline none"
done

echo ""
echo "All jobs submitted. Monitor at: https://wandb.ai — project: $PROJECT"
