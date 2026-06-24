#!/usr/bin/env python3
"""
Script 21: Prepare BraTS2020 4-modal data for nnU-Net v2.

Purpose:
- Convert our current train/test CSV setup into nnU-Net v2 dataset format.
- Use all four MRI modalities:
    0000 = FLAIR
    0001 = T1
    0002 = T1ce
    0003 = T2
- Save training images in imagesTr.
- Save training labels in labelsTr.
- Save held-out test images in imagesTs.
- Remap BraTS label 4 to 3 for nnU-Net compatibility.

Important:
This script does NOT train nnU-Net.
It only prepares the dataset structure.

nnU-Net dataset format:
nnUNet_raw/Dataset501_BraTS2020Multimodal/
    dataset.json
    imagesTr/
    labelsTr/
    imagesTs/
"""

import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

TRAIN_CSV = PROJECT_ROOT / "data" / "csvs" / "train_paths.csv"
TEST_CSV = PROJECT_ROOT / "data" / "csvs" / "test_paths.csv"

NNUNET_BASE = PROJECT_ROOT / "nnunet"
NNUNET_RAW = NNUNET_BASE / "nnUNet_raw"
NNUNET_PREPROCESSED = NNUNET_BASE / "nnUNet_preprocessed"
NNUNET_RESULTS = NNUNET_BASE / "nnUNet_results"

DATASET_ID = 501
DATASET_NAME = "BraTS2020Multimodal"
DATASET_FOLDER = NNUNET_RAW / f"Dataset{DATASET_ID}_{DATASET_NAME}"

IMAGESTR = DATASET_FOLDER / "imagesTr"
LABELSTR = DATASET_FOLDER / "labelsTr"
IMAGESTS = DATASET_FOLDER / "imagesTs"

REPORT_DIR = PROJECT_ROOT / "report_materials"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = PROJECT_ROOT / "results" / "21_nnunet_dataset_preparation_summary.csv"
SUMMARY_TXT = REPORT_DIR / "21_nnunet_dataset_preparation_summary.txt"


# =============================================================================
# Settings
# =============================================================================

MODALITIES = [
    ("flair", "0000", "FLAIR"),
    ("t1", "0001", "T1"),
    ("t1ce", "0002", "T1ce"),
    ("t2", "0003", "T2"),
]


# =============================================================================
# Utility functions
# =============================================================================

def make_case_id(patient_id):
    """
    Make a clean nnU-Net case ID.

    Example:
    BraTS20_Training_001 -> BRATS_001
    """
    patient_id = str(patient_id)

    if patient_id.startswith("BraTS20_Training_"):
        suffix = patient_id.replace("BraTS20_Training_", "")
        return f"BRATS_{suffix}"

    # fallback: remove unsafe characters
    safe = patient_id.replace("-", "_").replace(" ", "_")
    return safe


def copy_image(src_path, dst_path):
    """
    Copy a modality image.
    We do not change modality intensities here.
    nnU-Net will handle its own preprocessing later.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    if not src_path.exists():
        raise FileNotFoundError(f"Missing image file: {src_path}")

    shutil.copy2(src_path, dst_path)


def remap_and_save_label(src_seg_path, dst_seg_path):
    """
    BraTS labels are usually:
        0, 1, 2, 4

    nnU-Net expects consecutive labels if dataset.json says:
        0, 1, 2, 3

    So original label 4 becomes 3.
    """
    src_seg_path = Path(src_seg_path)
    dst_seg_path = Path(dst_seg_path)

    if not src_seg_path.exists():
        raise FileNotFoundError(f"Missing segmentation file: {src_seg_path}")

    img = nib.load(str(src_seg_path))
    seg = img.get_fdata().astype(np.int16)

    remapped = np.zeros_like(seg, dtype=np.uint8)
    remapped[seg == 1] = 1
    remapped[seg == 2] = 2
    remapped[seg == 4] = 3

    out_img = nib.Nifti1Image(remapped, affine=img.affine, header=img.header)
    out_img.set_data_dtype(np.uint8)

    nib.save(out_img, str(dst_seg_path))


def write_dataset_json(num_training):
    """
    Write nnU-Net v2 dataset.json.
    """
    dataset_json = {
        "channel_names": {
            "0": "FLAIR",
            "1": "T1",
            "2": "T1ce",
            "3": "T2"
        },
        "labels": {
            "background": 0,
            "tumor_label_1": 1,
            "tumor_label_2": 2,
            "tumor_label_4_remapped": 3
        },
        "numTraining": int(num_training),
        "file_ending": ".nii.gz"
    }

    json_path = DATASET_FOLDER / "dataset.json"

    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)

    return json_path


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("Script 21: Prepare BraTS2020 data for nnU-Net v2")
    print("=" * 80)

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Missing train CSV: {TRAIN_CSV}")

    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Missing test CSV: {TEST_CSV}")

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    print(f"Train patients: {len(train_df)}")
    print(f"Test patients: {len(test_df)}")

    print("\nCreating nnU-Net folders:")
    for folder in [IMAGESTR, LABELSTR, IMAGESTS, NNUNET_PREPROCESSED, NNUNET_RESULTS]:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"  {folder}")

    rows = []

    print("\nPreparing training cases...")
    for idx, row in train_df.iterrows():
        patient_id = row["patient_id"]
        case_id = make_case_id(patient_id)

        print(f"[Train {idx + 1}/{len(train_df)}] {patient_id} -> {case_id}")

        for csv_col, channel_suffix, modality_name in MODALITIES:
            src = Path(row[csv_col])
            dst = IMAGESTR / f"{case_id}_{channel_suffix}.nii.gz"
            copy_image(src, dst)

        label_dst = LABELSTR / f"{case_id}.nii.gz"
        remap_and_save_label(row["seg"], label_dst)

        rows.append({
            "split": "train",
            "patient_id": patient_id,
            "case_id": case_id,
            "images_folder": str(IMAGESTR),
            "label_file": str(label_dst),
        })

    print("\nPreparing held-out test images...")
    for idx, row in test_df.iterrows():
        patient_id = row["patient_id"]
        case_id = make_case_id(patient_id)

        print(f"[Test {idx + 1}/{len(test_df)}] {patient_id} -> {case_id}")

        for csv_col, channel_suffix, modality_name in MODALITIES:
            src = Path(row[csv_col])
            dst = IMAGESTS / f"{case_id}_{channel_suffix}.nii.gz"
            copy_image(src, dst)

        rows.append({
            "split": "test",
            "patient_id": patient_id,
            "case_id": case_id,
            "images_folder": str(IMAGESTS),
            "label_file": "",
        })

    json_path = write_dataset_json(num_training=len(train_df))

    summary_df = pd.DataFrame(rows)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    # Count output files
    num_imagesTr = len(list(IMAGESTR.glob("*.nii.gz")))
    num_labelsTr = len(list(LABELSTR.glob("*.nii.gz")))
    num_imagesTs = len(list(IMAGESTS.glob("*.nii.gz")))

    expected_imagesTr = len(train_df) * 4
    expected_labelsTr = len(train_df)
    expected_imagesTs = len(test_df) * 4

    with open(SUMMARY_TXT, "w") as f:
        f.write("nnU-Net dataset preparation summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset folder: {DATASET_FOLDER}\n")
        f.write(f"Dataset JSON: {json_path}\n\n")
        f.write(f"Train patients: {len(train_df)}\n")
        f.write(f"Test patients: {len(test_df)}\n\n")
        f.write(f"imagesTr files: {num_imagesTr} / expected {expected_imagesTr}\n")
        f.write(f"labelsTr files: {num_labelsTr} / expected {expected_labelsTr}\n")
        f.write(f"imagesTs files: {num_imagesTs} / expected {expected_imagesTs}\n\n")
        f.write("Channel mapping:\n")
        f.write("0000 = FLAIR\n")
        f.write("0001 = T1\n")
        f.write("0002 = T1ce\n")
        f.write("0003 = T2\n\n")
        f.write("Label mapping:\n")
        f.write("0 = background\n")
        f.write("1 = BraTS original label 1\n")
        f.write("2 = BraTS original label 2\n")
        f.write("3 = BraTS original label 4 remapped to 3\n")

    print("\n" + "=" * 80)
    print("nnU-Net dataset preparation complete")
    print("=" * 80)
    print(f"Dataset folder: {DATASET_FOLDER}")
    print(f"dataset.json: {json_path}")
    print(f"Summary CSV: {SUMMARY_CSV}")
    print(f"Summary TXT: {SUMMARY_TXT}")

    print("\nFile counts:")
    print(f"imagesTr: {num_imagesTr} / expected {expected_imagesTr}")
    print(f"labelsTr: {num_labelsTr} / expected {expected_labelsTr}")
    print(f"imagesTs: {num_imagesTs} / expected {expected_imagesTs}")

    if num_imagesTr != expected_imagesTr:
        print("WARNING: imagesTr count mismatch.")
    if num_labelsTr != expected_labelsTr:
        print("WARNING: labelsTr count mismatch.")
    if num_imagesTs != expected_imagesTs:
        print("WARNING: imagesTs count mismatch.")

    print("\nNext step after this:")
    print("Set nnU-Net environment variables and run nnUNetv2_plan_and_preprocess.")
    print("Do not train until preprocessing succeeds.")


if __name__ == "__main__":
    main()
