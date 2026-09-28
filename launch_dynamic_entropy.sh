#!/bin/bash
# Launch the dynamic (SAC-style dual-ascent) per-group entropy target sweep.
#
# Replaces the fixed entropy_coef=0.05 bonus with a learned per-group
# temperature alpha_g, adjusted by dual ascent so masked policy entropy
# tracks a constant target_entropy (nats). See .cursor/rules/dynamic-entropy-plan.mdc
# for the full design (calibrated-constant target as the first-step method).
#
# Sweeps target_entropy in {0.25, 0.35, 0.45} x 3 seeds = 9 jobs. Base config
# (heterognn, coop_encoder, freeze1M, GAE fix) matches the current
# VPP_HGTeamHA_1node.sbatch defaults; only dynamic-entropy knobs are set here.
#
# Usage: bash launch_dynamic_entropy.sh [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SBATCH="$SCRIPT_DIR/VPP_HGTeamHA_1node.sbatch"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN — no jobs will be submitted ==="
fi

SEEDS=(20 21 22)
TARGETS=(0.25 0.35 0.45)

submit() {
    local target=$1 seed=$2
    local group="dynH-${target}"
    echo "  submit: group=$group target_entropy=$target seed=$seed"
    if $DRY_RUN; then
        return 0
    fi
    sbatch \
        --job-name="${group}-s${seed}" \
        --export=ALL,SEED=${seed},WANDB_GROUP=${group},DYNAMIC_ENTROPY=true,TARGET_ENTROPY=${target} \
        "$SBATCH"
}

for target in "${TARGETS[@]}"; do
    echo "=== Group: dynH-${target} (target_entropy=${target}) ==="
    for s in "${SEEDS[@]}"; do submit "$target" "$s"; done
    echo ""
done

echo "Submitted $((${#TARGETS[@]} * ${#SEEDS[@]})) jobs. Monitor: squeue -u \$USER"
