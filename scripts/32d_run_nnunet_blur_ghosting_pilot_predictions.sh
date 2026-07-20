#!/bin/bash
set -e

echo "================================================================================"
echo "Script 32D: Run nnU-Net predictions for blur and ghosting extended pilot"
echo "================================================================================"

cd /home/xfh25/brats_segmentation_project

export nnUNet_raw="/home/xfh25/brats_segmentation_project/nnunet/nnUNet_raw"
export nnUNet_preprocessed="/home/xfh25/brats_segmentation_project/nnunet/nnUNet_preprocessed"
export nnUNet_results="/home/xfh25/brats_segmentation_project/nnunet/nnUNet_results"

ROOT="nnunet/temporary_degraded_tests/blur_ghosting_extended_pilot"

CONDITIONS=(
  blur_L6
  blur_L7
  blur_L8
  blur_L9
  blur_L10
  ghosting_L6
  ghosting_L7
  ghosting_L8
  ghosting_L9
  ghosting_L10
)

echo "Conditions to predict: ${#CONDITIONS[@]}"
echo "Expected input images per condition: 20"
echo "Expected predictions per condition: 5"
echo "Expected total predictions: 50"
echo "================================================================================"

for CONDITION in "${CONDITIONS[@]}"; do
    INPUT_DIR="$ROOT/$CONDITION/imagesTs"
    OUTPUT_DIR="$ROOT/$CONDITION/predictions"

    echo "--------------------------------------------------------------------------------"
    echo "Running prediction for: $CONDITION"
    echo "Input:  $INPUT_DIR"
    echo "Output: $OUTPUT_DIR"

    if [ ! -d "$INPUT_DIR" ]; then
        echo "ERROR: Missing input directory: $INPUT_DIR"
        exit 1
    fi

    INPUT_COUNT=$(find "$INPUT_DIR" -type f -name "*.nii.gz" | wc -l)

    echo "Input image count: $INPUT_COUNT"

    if [ "$INPUT_COUNT" -ne 20 ]; then
        echo "ERROR: Expected 20 input images for $CONDITION, found $INPUT_COUNT"
        exit 1
    fi

    mkdir -p "$OUTPUT_DIR"

    CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict \
      -i "$INPUT_DIR" \
      -o "$OUTPUT_DIR" \
      -d 501 \
      -c 3d_fullres \
      -f 0 \
      -chk checkpoint_best.pth

    PRED_COUNT=$(find "$OUTPUT_DIR" -type f -name "*.nii.gz" | wc -l)

    echo "Prediction count for $CONDITION: $PRED_COUNT"

    if [ "$PRED_COUNT" -ne 5 ]; then
        echo "ERROR: Expected 5 predictions for $CONDITION, found $PRED_COUNT"
        exit 1
    fi
done

TOTAL_PREDICTIONS=$(find "$ROOT" -path "*/predictions/*.nii.gz" | wc -l)

echo "================================================================================"
echo "Blur and ghosting pilot predictions complete."
echo "Total prediction masks: $TOTAL_PREDICTIONS"
echo "Expected total prediction masks: 50"
echo "================================================================================"

if [ "$TOTAL_PREDICTIONS" -ne 50 ]; then
    echo "ERROR: Total prediction count mismatch."
    exit 1
fi
