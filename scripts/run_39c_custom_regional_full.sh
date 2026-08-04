#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate brats3d

TAG="regional_metrics_full_20260802"

for artifact in blur ghosting noise contrast ringing; do
    echo "============================================================"
    echo "Starting ${artifact}: $(date)"
    echo "============================================================"

    python scripts/25b_evaluate_3d_unet_degraded_full_volume.py \
        --artifact "${artifact}" \
        --min-level 1 \
        --max-level 10 \
        --device cuda:0 \
        --output-tag "${TAG}"

    echo "Completed ${artifact}: $(date)"
done

echo "ALL REGIONAL EVALUATIONS COMPLETED: $(date)"
