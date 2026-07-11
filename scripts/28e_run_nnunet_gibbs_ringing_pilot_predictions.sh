#!/usr/bin/env bash
set -euo pipefail

echo "================================================================================"
echo "Script 28E: Run nnU-Net predictions for Gibbs ringing pilot"
echo "================================================================================"

cd /home/xfh25/brats_segmentation_project

source ~/miniconda3/etc/profile.d/conda.sh
conda activate brats3d

export nnUNet_raw="/home/xfh25/brats_segmentation_project/nnunet/nnUNet_raw"
export nnUNet_preprocessed="/home/xfh25/brats_segmentation_project/nnunet/nnUNet_preprocessed"
export nnUNet_results="/home/xfh25/brats_segmentation_project/nnunet/nnUNet_results"

PILOT_ROOT="/home/xfh25/brats_segmentation_project/nnunet/temporary_degraded_tests/gibbs_ringing_pilot"

LEVELS=(
  gibbs_L1
  gibbs_L2
  gibbs_L3
  gibbs_L4
  gibbs_L5
)

for LEVEL in "${LEVELS[@]}"; do
  INPUT_DIR="${PILOT_ROOT}/${LEVEL}/imagesTs"
  OUTPUT_DIR="${PILOT_ROOT}/${LEVEL}/predictions"

  echo ""
  echo "--------------------------------------------------------------------------------"
  echo "Predicting: ${LEVEL}"
  echo "Input:      ${INPUT_DIR}"
  echo "Output:     ${OUTPUT_DIR}"
  echo "--------------------------------------------------------------------------------"

  mkdir -p "${OUTPUT_DIR}"

  CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict \
    -i "${INPUT_DIR}" \
    -o "${OUTPUT_DIR}" \
    -d 501 \
    -c 3d_fullres \
    -f 0

  COUNT=$(find "${OUTPUT_DIR}" -name "*.nii.gz" | wc -l)
  echo "Prediction count for ${LEVEL}: ${COUNT}"
done

echo ""
echo "================================================================================"
echo "Gibbs ringing pilot prediction complete."
echo "Expected total predictions: 5 levels × 5 patients = 25"
echo "Actual total predictions:"
find "${PILOT_ROOT}" -path "*/predictions/*.nii.gz" | wc -l
echo "================================================================================"
