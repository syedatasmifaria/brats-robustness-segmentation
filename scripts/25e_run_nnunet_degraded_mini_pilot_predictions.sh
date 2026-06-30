#!/usr/bin/env bash

# Script 25E: Run nnU-Net prediction for degraded mini-pilot conditions.
#
# Purpose:
# - Run clean-trained nnU-Net on 25 degraded mini-pilot conditions.
# - Each condition has 2 patients.
# - No training happens here.
#
# Important:
# - Temporary degraded images and predictions are under nnunet/temporary_degraded_tests/
# - Do NOT commit those .nii.gz files to GitHub.

set -e

PROJECT_ROOT="/home/xfh25/brats_segmentation_project"
MINI_ROOT="${PROJECT_ROOT}/nnunet/temporary_degraded_tests/mini_pilot"

export nnUNet_raw="${PROJECT_ROOT}/nnunet/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_ROOT}/nnunet/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_ROOT}/nnunet/nnUNet_results"

GPU_ID=2

echo "============================================================"
echo "Script 25E: nnU-Net degraded mini-pilot prediction"
echo "============================================================"
echo "Project root: ${PROJECT_ROOT}"
echo "Mini-pilot root: ${MINI_ROOT}"
echo "GPU ID: ${GPU_ID}"
echo "nnUNet_raw: ${nnUNet_raw}"
echo "nnUNet_preprocessed: ${nnUNet_preprocessed}"
echo "nnUNet_results: ${nnUNet_results}"
echo "============================================================"
echo

for CONDITION_DIR in "${MINI_ROOT}"/*_L*; do
    if [ ! -d "${CONDITION_DIR}" ]; then
        continue
    fi

    CONDITION_NAME=$(basename "${CONDITION_DIR}")
    INPUT_DIR="${CONDITION_DIR}/imagesTs"
    OUTPUT_DIR="${CONDITION_DIR}/predictions"

    mkdir -p "${OUTPUT_DIR}"

    EXISTING_COUNT=$(find "${OUTPUT_DIR}" -maxdepth 1 -name "BraTS20_Training_*.nii.gz" | wc -l)

    if [ "${EXISTING_COUNT}" -ge 2 ]; then
        echo "Skipping ${CONDITION_NAME}: predictions already exist (${EXISTING_COUNT} files)."
        echo
        continue
    fi

    echo "------------------------------------------------------------"
    echo "Running prediction for condition: ${CONDITION_NAME}"
    echo "Input: ${INPUT_DIR}"
    echo "Output: ${OUTPUT_DIR}"
    echo "------------------------------------------------------------"

    CUDA_VISIBLE_DEVICES=${GPU_ID} nnUNetv2_predict \
        -i "${INPUT_DIR}" \
        -o "${OUTPUT_DIR}" \
        -d 501 \
        -c 3d_fullres \
        -f 0 \
        -chk checkpoint_best.pth

    echo "Finished condition: ${CONDITION_NAME}"
    echo
done

echo "============================================================"
echo "All mini-pilot predictions finished."
echo "============================================================"
