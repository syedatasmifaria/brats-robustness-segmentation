#!/bin/bash
set -e

echo "================================================================================"
echo "Step 26E-A: Run clean nnU-Net predictions on same 74 test patients"
echo "================================================================================"

cd /home/xfh25/brats_segmentation_project

export nnUNet_raw="/home/xfh25/brats_segmentation_project/nnunet/nnUNet_raw"
export nnUNet_preprocessed="/home/xfh25/brats_segmentation_project/nnunet/nnUNet_preprocessed"
export nnUNet_results="/home/xfh25/brats_segmentation_project/nnunet/nnUNet_results"

INPUT_DIR="nnunet/nnUNet_raw/Dataset501_BraTS2020Multimodal/imagesTs"
OUTPUT_DIR="nnunet/temporary_degraded_tests/final_full/clean/predictions"

mkdir -p "$OUTPUT_DIR"

echo "Input clean images:"
echo "$INPUT_DIR"

echo "Output clean predictions:"
echo "$OUTPUT_DIR"

echo "Counting clean input image files..."
find "$INPUT_DIR" -name "BraTS20_Training_*.nii.gz" | wc -l

echo "Expected image files: 296"
echo "Expected patients: 74"
echo "Running nnU-Net prediction..."
echo "================================================================================"

CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict \
  -i "$INPUT_DIR" \
  -o "$OUTPUT_DIR" \
  -d 501 \
  -c 3d_fullres \
  -f 0 \
  -chk checkpoint_best.pth

echo "================================================================================"
echo "Clean nnU-Net prediction complete."
echo "Counting prediction masks..."
find "$OUTPUT_DIR" -name "BraTS20_Training_*.nii.gz" | wc -l
echo "Expected prediction masks: 74"
echo "================================================================================"
