"""
Script 05c: Prepare 2D test slice list for clean FLAIR U-Net evaluation.

Goal:
- Load test_paths.csv
- Count test axial slices
- Select tumor-containing slices
- Select useful background slices with visible brain tissue
- Save selected test slice information to a CSV

Important:
- This script does NOT train a model.
- This script does NOT save image arrays.
- This script only creates a CSV index of selected test slices.

Input:
- data/csvs/test_paths.csv

Output:
- data/csvs/test_2d_slices_flair.csv
- data/csvs/test_2d_slice_patient_summary_flair.csv
"""

import random
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm


# -----------------------------
# Paths
# -----------------------------
PROJECT_DIR = Path("/home/xfh25/brats_segmentation_project")

TEST_CSV = PROJECT_DIR / "data" / "csvs" / "test_paths.csv"

OUTPUT_DIR = PROJECT_DIR / "data" / "csvs"
OUTPUT_CSV = OUTPUT_DIR / "test_2d_slices_flair.csv"
PATIENT_SUMMARY_CSV = OUTPUT_DIR / "test_2d_slice_patient_summary_flair.csv"


# -----------------------------
# Settings
# -----------------------------
RANDOM_SEED = 42

BACKGROUND_TO_TUMOR_RATIO = 1.0

TUMOR_PIXEL_THRESHOLD = 0

BRAIN_INTENSITY_THRESHOLD = 0.05
MIN_BRAIN_PIXELS = 1000


# -----------------------------
# Reproducibility
# -----------------------------
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_nifti(path):
    """
    Load a NIfTI file and return it as a NumPy array.
    """
    img = nib.load(str(path))
    data = img.get_fdata()
    return data


def normalize_slice(image_slice):
    """
    Normalize one 2D image slice to 0-1.
    """
    image_slice = image_slice.astype(np.float32)

    min_val = np.min(image_slice)
    max_val = np.max(image_slice)

    if max_val - min_val == 0:
        return np.zeros_like(image_slice)

    return (image_slice - min_val) / (max_val - min_val)


def has_enough_brain_tissue(flair_slice):
    """
    Check whether a FLAIR slice has enough visible brain tissue.
    """
    normalized = normalize_slice(flair_slice)

    brain_pixels = np.sum(normalized > BRAIN_INTENSITY_THRESHOLD)

    return brain_pixels >= MIN_BRAIN_PIXELS, brain_pixels


def main():
    print("=" * 80)
    print("Script 05c: Prepare 2D FLAIR test slices")
    print("=" * 80)

    print(f"Reading test CSV from:\n{TEST_CSV}")

    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Could not find test CSV: {TEST_CSV}")

    test_df = pd.read_csv(TEST_CSV)

    print(f"\nNumber of test patients: {len(test_df)}")
    print("\nCSV columns:")
    print(test_df.columns.tolist())

    required_columns = ["patient_id", "flair", "seg"]
    missing_columns = [col for col in required_columns if col not in test_df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in test_paths.csv: {missing_columns}\n"
            f"Available columns are: {test_df.columns.tolist()}"
        )

    selected_rows = []
    patient_summaries = []

    total_slices_all_patients = 0
    total_tumor_slices_all_patients = 0
    total_background_slices_all_patients = 0
    total_useful_background_slices_all_patients = 0
    total_selected_background_slices = 0

    for idx, row in tqdm(
        test_df.iterrows(),
        total=len(test_df),
        desc="Processing test patients"
    ):
        patient_id = row["patient_id"]

        flair_path = row["flair"]
        seg_path = row["seg"]

        flair = load_nifti(flair_path)
        seg = load_nifti(seg_path)

        if flair.shape != seg.shape:
            raise ValueError(
                f"Shape mismatch for {patient_id}: "
                f"FLAIR shape {flair.shape}, SEG shape {seg.shape}"
            )

        num_slices = seg.shape[2]
        total_slices_all_patients += num_slices

        tumor_slices = []
        all_background_slices = []
        useful_background_slices = []

        for slice_idx in range(num_slices):
            flair_slice = flair[:, :, slice_idx]
            mask_slice = seg[:, :, slice_idx]

            tumor_pixels = np.sum(mask_slice > 0)

            if tumor_pixels > TUMOR_PIXEL_THRESHOLD:
                tumor_slices.append(slice_idx)
            else:
                all_background_slices.append(slice_idx)

                enough_brain, brain_pixels = has_enough_brain_tissue(flair_slice)

                if enough_brain:
                    useful_background_slices.append(slice_idx)

        num_tumor_slices = len(tumor_slices)
        num_background_slices = len(all_background_slices)
        num_useful_background_slices = len(useful_background_slices)

        total_tumor_slices_all_patients += num_tumor_slices
        total_background_slices_all_patients += num_background_slices
        total_useful_background_slices_all_patients += num_useful_background_slices

        selected_tumor_slices = tumor_slices

        max_background_to_select = int(num_tumor_slices * BACKGROUND_TO_TUMOR_RATIO)
        max_background_to_select = min(max_background_to_select, num_useful_background_slices)

        if max_background_to_select > 0:
            selected_background_slices = random.sample(
                useful_background_slices,
                k=max_background_to_select
            )
        else:
            selected_background_slices = []

        total_selected_background_slices += len(selected_background_slices)

        patient_summaries.append({
            "patient_id": patient_id,
            "total_slices": num_slices,
            "tumor_slices": num_tumor_slices,
            "all_background_slices": num_background_slices,
            "useful_background_slices": num_useful_background_slices,
            "selected_tumor_slices": len(selected_tumor_slices),
            "selected_background_slices": len(selected_background_slices),
            "selected_total_slices": len(selected_tumor_slices) + len(selected_background_slices)
        })

        for slice_idx in selected_tumor_slices:
            selected_rows.append({
                "patient_id": patient_id,
                "flair_path": flair_path,
                "seg_path": seg_path,
                "slice_idx": slice_idx,
                "has_tumor": 1,
                "slice_type": "tumor",
                "split": "test"
            })

        for slice_idx in selected_background_slices:
            selected_rows.append({
                "patient_id": patient_id,
                "flair_path": flair_path,
                "seg_path": seg_path,
                "slice_idx": slice_idx,
                "has_tumor": 0,
                "slice_type": "useful_background",
                "split": "test"
            })

    selected_df = pd.DataFrame(selected_rows)

    if len(selected_df) == 0:
        raise ValueError("No test slices were selected.")

    selected_df = selected_df.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    patient_summary_df = pd.DataFrame(patient_summaries)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected_df.to_csv(OUTPUT_CSV, index=False)
    patient_summary_df.to_csv(PATIENT_SUMMARY_CSV, index=False)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    print(f"Total test patients: {len(test_df)}")
    print(f"Total axial slices across all test patients: {total_slices_all_patients}")

    print("\nAll slices:")
    print(f"Total tumor-containing slices: {total_tumor_slices_all_patients}")
    print(f"Total background/empty slices: {total_background_slices_all_patients}")
    print(f"Total useful background slices with visible brain tissue: {total_useful_background_slices_all_patients}")

    print("\nSelected for 2D testing:")
    print(f"Tumor slices selected: {total_tumor_slices_all_patients}")
    print(f"Useful background slices selected: {total_selected_background_slices}")
    print(f"Total selected test slices: {len(selected_df)}")

    print(f"\nSaved selected test slice CSV to:\n{OUTPUT_CSV}")
    print(f"\nSaved patient-level test summary CSV to:\n{PATIENT_SUMMARY_CSV}")

    print("\nPreview of selected test slice CSV:")
    print(selected_df.head())

    print("\nSlice class balance:")
    print(selected_df["has_tumor"].value_counts())

    print("\nSlice class percentage:")
    print((selected_df["has_tumor"].value_counts(normalize=True) * 100).round(2))

    print("\nPatient-level selected slice summary:")
    print(patient_summary_df["selected_total_slices"].describe())

    print("\nPatients with zero tumor slices:")
    print((patient_summary_df["tumor_slices"] == 0).sum())

    print("\nPatients with zero useful background slices:")
    print((patient_summary_df["useful_background_slices"] == 0).sum())

    print("\nDone. Script 05c completed successfully.")


if __name__ == "__main__":
    main()