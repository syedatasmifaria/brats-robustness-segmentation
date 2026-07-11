#!/usr/bin/env bash
set -euo pipefail

echo "================================================================================"
echo "Script 29B: Run nnU-Net predictions for full extended selected degradation test"
echo "================================================================================"

cd /home/xfh25/brats_segmentation_project

source ~/miniconda3/etc/profile.d/conda.sh
conda activate brats3d

export nnUNet_raw="/home/xfh25/brats_segmentation_project/nnunet/nnUNet_raw"
export nnUNet_preprocessed="/home/xfh25/brats_segmentation_project/nnunet/nnUNet_preprocessed"
export nnUNet_results="/home/xfh25/brats_segmentation_project/nnunet/nnUNet_results"

PILOT_ROOT="/home/xfh25/brats_segmentation_project/nnunet/temporary_degraded_tests/extended_full_selected"

CONDITIONS=(
  noise_L6
  noise_L7
  noise_L8
  noise_L9
  noise_L10
  contrast_L6
  contrast_L7
  contrast_L8
  contrast_L9
  contrast_L10
  ringing_L6
  ringing_L7
  ringing_L8
  ringing_L9
  ringing_L10
)

for CONDITION in "${CONDITIONS[@]}"; do
  INPUT_DIR="${PILOT_ROOT}/${CONDITION}/imagesTs"
  OUTPUT_DIR="${PILOT_ROOT}/${CONDITION}/predictions"

  echo ""
  echo "--------------------------------------------------------------------------------"
  echo "Predicting condition: ${CONDITION}"
  echo "Input:  ${INPUT_DIR}"
  echo "Output: ${OUTPUT_DIR}"
  echo "--------------------------------------------------------------------------------"

  mkdir -p "${OUTPUT_DIR}"

  CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict \
    -i "${INPUT_DIR}" \
    -o "${OUTPUT_DIR}" \
    -d 501 \
    -c 3d_fullres \
    -f 0

  COUNT=$(find "${OUTPUT_DIR}" -name "*.nii.gz" | wc -l)
  echo "Prediction count for ${CONDITION}: ${COUNT}"
done

echo ""
echo "================================================================================"
echo "Full extended selected nnU-Net prediction complete."
echo "Expected total predictions: 15 conditions × 74 patients = 1110"
echo "Actual total predictions:"
find "${PILOT_ROOT}" -path "*/predictions/*.nii.gz" | wc -l
echo "================================================================================"
