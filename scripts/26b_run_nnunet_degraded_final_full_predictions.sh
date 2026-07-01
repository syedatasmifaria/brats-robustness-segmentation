#!/usr/bin/env bash

# Script 26B: Run nnU-Net prediction for final full degraded evaluation.
#
# Purpose:
# - Run clean-trained nnU-Net on 25 degraded full-test conditions.
# - Each condition has 74 patients.
# - No training happens here.
#
# Important:
# - This evaluates degradation robustness only.
# - Temporary degraded images and predictions are under nnunet/temporary_degraded_tests/
# - Do NOT commit those .nii.gz files to GitHub.

set -e

PROJECT_ROOT="/home/xfh25/brats_segmentation_project"
FINAL_ROOT="${PROJECT_ROOT}/nnunet/temporary_degraded_tests/final_full"

export nnUNet_raw="${PROJECT_ROOT}/nnunet/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_ROOT}/nnunet/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_ROOT}/nnunet/nnUNet_results"

GPU_ID=2
EXPECTED_PATIENTS=74

echo "============================================================"
echo "Script 26B: nnU-Net final full degraded prediction"
echo "============================================================"
echo "Project root: ${PROJECT_ROOT}"
echo "Final root: ${FINAL_ROOT}"
echo "GPU ID: ${GPU_ID}"
echo "Expected patients per condition: ${EXPECTED_PATIENTS}"
echo "nnUNet_raw: ${nnUNet_raw}"
echo "nnUNet_preprocessed: ${nnUNet_preprocessed}"
echo "nnUNet_results: ${nnUNet_results}"
echo "============================================================"
echo

for CONDITION_DIR in "${FINAL_ROOT}"/*_L*; do
    if [ ! -d "${CONDITION_DIR}" ]; then
        continue
    fi

    CONDITION_NAME=$(basename "${CONDITION_DIR}")
    INPUT_DIR="${CONDITION_DIR}/imagesTs"
    OUTPUT_DIR="${CONDITION_DIR}/predictions"

    mkdir -p "${OUTPUT_DIR}"

    EXISTING_COUNT=$(find "${OUTPUT_DIR}" -maxdepth 1 -name "BraTS20_Training_*.nii.gz" | wc -l)

    if [ "${EXISTING_COUNT}" -ge "${EXPECTED_PATIENTS}" ]; then
        echo "Skipping ${CONDITION_NAME}: predictions already exist (${EXISTING_COUNT} files)."
        echo
        continue
    fi

    echo "------------------------------------------------------------"
    echo "Running prediction for condition: ${CONDITION_NAME}"
    echo "Input: ${INPUT_DIR}"
    echo "Output: ${OUTPUT_DIR}"
    echo "Existing predictions: ${EXISTING_COUNT}/${EXPECTED_PATIENTS}"
    echo "------------------------------------------------------------"

    CUDA_VISIBLE_DEVICES=${GPU_ID} nnUNetv2_predict \
        -i "${INPUT_DIR}" \
        -o "${OUTPUT_DIR}" \
        -d 501 \
        -c 3d_fullres \
        -f 0 \
        -chk checkpoint_best.pth

    FINISHED_COUNT=$(find "${OUTPUT_DIR}" -maxdepth 1 -name "BraTS20_Training_*.nii.gz" | wc -l)

    echo "Finished condition: ${CONDITION_NAME}"
    echo "Prediction files now: ${FINISHED_COUNT}/${EXPECTED_PATIENTS}"
    echo
done

echo "============================================================"
echo "All final full degraded predictions finished."
echo "============================================================"
