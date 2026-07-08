#!/bin/bash
set -e

echo "================================================================================"
echo "Script 27C: Run nnU-Net predictions for extended degradation pilot"
echo "================================================================================"

cd /home/xfh25/brats_segmentation_project

export nnUNet_raw="/home/xfh25/brats_segmentation_project/nnunet/nnUNet_raw"
export nnUNet_preprocessed="/home/xfh25/brats_segmentation_project/nnunet/nnUNet_preprocessed"
export nnUNet_results="/home/xfh25/brats_segmentation_project/nnunet/nnUNet_results"

ROOT="nnunet/temporary_degraded_tests/extended_pilot"

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

echo "Conditions to predict: ${#CONDITIONS[@]}"
echo "Expected predictions per condition: 5"
echo "Expected total predictions: 75"
echo "================================================================================"

for CONDITION in "${CONDITIONS[@]}"; do
    INPUT_DIR="$ROOT/$CONDITION/imagesTs"
    OUTPUT_DIR="$ROOT/$CONDITION/predictions"

    echo "--------------------------------------------------------------------------------"
    echo "Running prediction for: $CONDITION"
    echo "Input:  $INPUT_DIR"
    echo "Output: $OUTPUT_DIR"

    mkdir -p "$OUTPUT_DIR"

    echo "Input image count:"
    find "$INPUT_DIR" -name "*.nii.gz" | wc -l

    CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict \
      -i "$INPUT_DIR" \
      -o "$OUTPUT_DIR" \
      -d 501 \
      -c 3d_fullres \
      -f 0 \
      -chk checkpoint_best.pth

    echo "Prediction count for $CONDITION:"
    find "$OUTPUT_DIR" -name "*.nii.gz" | wc -l
done

echo "================================================================================"
echo "Extended pilot predictions complete."
echo "Total prediction masks:"
find "$ROOT" -path "*/predictions/*.nii.gz" | wc -l
echo "Expected total prediction masks: 75"
echo "================================================================================"
