#!/bin/bash
# Launch the EMA-median advantage-reference experiments.
#
# Both groups seed the per-group advantage-scale reference with the MEDIAN of
# batch_std over a ~1.5M-frame window (61 iters at 24576 frames/batch), which
# is robust to the large startup transient that otherwise inflates the frozen
# reference (mean seeding).  The two groups differ only in the critic learning
# rate:
#   ema-median            : critic LR x1  (base 3e-4)
#   ema-median-lr1.5e-3   : critic LR x5  (3e-4 -> 1.5e-3), to let the critic
#                           track the policy through the 2-4M EV-critic dip.
#
# Everything else matches the current VPP_HGTeamHA_1node.sbatch config
# (heterognn, coop_encoder, entropy 0.05, encoder freeze at 1M, GAE fix).
#
# Usage: bash launch_ema_median.sh [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SBATCH="$SCRIPT_DIR/VPP_HGTeamHA_1node.sbatch"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN — no jobs will be submitted ==="
fi

SEEDS=(20 21 22)
EMA_REDUCTION=median
EMA_WARMUP_ITERS=61   # ~1.5M frames at 24576 frames/batch

submit() {
    local group=$1 mult=$2 seed=$3
    echo "  submit: group=$group critic_lr_mult=$mult seed=$seed"
    if $DRY_RUN; then
        return 0
    fi
    sbatch \
        --job-name="${group}-s${seed}" \
        --export=ALL,SEED=${seed},WANDB_GROUP=${group},CRITIC_LR_MULT=${mult},EMA_REDUCTION=${EMA_REDUCTION},EMA_WARMUP_ITERS=${EMA_WARMUP_ITERS} \
        "$SBATCH"
}

echo "=== Group: ema-median (critic LR x1) ==="
for s in "${SEEDS[@]}"; do submit ema-median 1.0 "$s"; done

echo ""
echo "=== Group: ema-median-lr1.5e-3 (critic LR x5) ==="
for s in "${SEEDS[@]}"; do submit ema-median-lr1.5e-3 5.0 "$s"; done

echo ""
echo "Submitted ${#SEEDS[@]} + ${#SEEDS[@]} jobs. Monitor: squeue -u \$USER"
