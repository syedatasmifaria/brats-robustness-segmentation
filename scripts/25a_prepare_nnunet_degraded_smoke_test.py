#!/usr/bin/env python3
"""
Script 25A: Prepare a tiny degraded nnU-Net smoke-test dataset.

Purpose:
- Create degraded nnU-Net-style test inputs for a small number of BraTS2020 test patients.
- This is ONLY for evaluation/prediction, not training.
- We use 2 patients and ghosting level 5 as a smoke test.

Important:
- Models were trained on clean images only.
- Degraded images are created only for testing/evaluation.
- Temporary degraded .nii.gz files should NOT be pushed to GitHub.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib


# -----------------------------
# Project paths
# -----------------------------
PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

TEST_CSV = PROJECT_ROOT / "data/csvs/test_paths.csv"

OUT_ROOT = PROJECT_ROOT / "nnunet/temporary_degraded_tests/ghosting_L5"
OUT_IMAGES = OUT_ROOT / "imagesTs"
OUT_LABELS = OUT_ROOT / "labelsTs"

OUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUT_LABELS.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Smoke test settings
# -----------------------------
NUM_PATIENTS = 2

# Ghosting level 5 setting.
# This follows the idea from our custom degradation testing:
# stronger ghosting = more repeated/shifted signal.
GHOST_NUM_REPEATS = 20
GHOST_INTENSITY = 0.55

MODALITIES = [
    ("flair", "0000"),
    ("t1", "0001"),
    ("t1ce", "0002"),
    ("t2", "0003"),
]


def load_nifti(path):
    """Load NIfTI image and return data, affine, and header."""
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine, img.header


def save_nifti(data, affine, header, out_path):
    """Save data as NIfTI while preserving spatial metadata."""
    out_img = nib.Nifti1Image(data.astype(np.float32), affine, header)
    nib.save(out_img, str(out_path))


def save_label_nifti(data, affine, header, out_path):
    """Save segmentation label as integer NIfTI."""
    out_img = nib.Nifti1Image(data.astype(np.uint8), affine, header)
    nib.save(out_img, str(out_path))


def remap_seg_labels(seg):
    """
    BraTS original labels are [0, 1, 2, 4].
    Our nnU-Net dataset used [0, 1, 2, 3], where original label 4 becomes class 3.
    """
    remapped = np.zeros_like(seg, dtype=np.uint8)
    remapped[seg == 1] = 1
    remapped[seg == 2] = 2
    remapped[seg == 4] = 3
    return remapped


def normalize_for_degradation(volume):
    """
    Normalize only for applying the synthetic artifact.
    We later map the degraded image back to the original intensity range.

    Why:
    - The custom degradation functions were easier to control on [0, 1].
    - But nnU-Net was trained on raw-like BraTS intensities, so we should not save
      permanently normalized volumes as inputs.
    """
    brain_mask = volume > 0

    if brain_mask.sum() == 0:
        return volume.copy(), brain_mask, 0.0, 1.0

    brain_values = volume[brain_mask]
    v_min = float(np.percentile(brain_values, 1))
    v_max = float(np.percentile(brain_values, 99))

    if v_max <= v_min:
        return volume.copy(), brain_mask, v_min, v_max

    normalized = np.zeros_like(volume, dtype=np.float32)
    normalized[brain_mask] = (volume[brain_mask] - v_min) / (v_max - v_min)
    normalized = np.clip(normalized, 0.0, 1.0)

    return normalized, brain_mask, v_min, v_max


def restore_original_range(normalized_degraded, brain_mask, v_min, v_max, original_volume):
    """
    Convert degraded [0,1] image back to original intensity scale.
    Background remains 0.
    """
    restored = np.zeros_like(original_volume, dtype=np.float32)

    if v_max <= v_min:
        restored[brain_mask] = original_volume[brain_mask]
        return restored

    restored[brain_mask] = normalized_degraded[brain_mask] * (v_max - v_min) + v_min
    restored[~brain_mask] = 0.0
    return restored


def apply_ghosting_3d(volume_01, brain_mask, num_repeats=20, intensity=0.55):
    """
    Simple synthetic ghosting artifact.

    Logic:
    - Ghosting creates shifted repeated copies of the image along one axis.
    - We simulate this by adding shifted versions of the image.
    - Then we keep values in [0, 1].

    This is not a perfect MRI physics simulator.
    It is a controlled synthetic degradation for robustness testing.
    """
    degraded = volume_01.copy().astype(np.float32)

    # Use axis 1 as the phase-like direction for this synthetic test.
    axis = 1

    # The shift spacing controls how far the ghost copies appear.
    # More repeats means more shifted copies.
    shape_size = degraded.shape[axis]
    shifts = np.linspace(4, max(5, shape_size // 3), num_repeats).astype(int)

    ghost = np.zeros_like(degraded, dtype=np.float32)

    for shift in shifts:
        ghost += np.roll(degraded, shift=shift, axis=axis)
        ghost += np.roll(degraded, shift=-shift, axis=axis)

    ghost = ghost / max(1, 2 * len(shifts))

    degraded = (1.0 - intensity) * degraded + intensity * ghost
    degraded = np.clip(degraded, 0.0, 1.0)

    # Keep background as background.
    degraded[~brain_mask] = 0.0

    return degraded


def main():
    print("=" * 80)
    print("Script 25A: Prepare nnU-Net degraded smoke-test images")
    print("=" * 80)

    df = pd.read_csv(TEST_CSV)
    smoke_df = df.head(NUM_PATIENTS)

    print(f"Test CSV: {TEST_CSV}")
    print(f"Output images: {OUT_IMAGES}")
    print(f"Output labels: {OUT_LABELS}")
    print(f"Number of smoke-test patients: {len(smoke_df)}")
    print(f"Degradation: ghosting L5, repeats={GHOST_NUM_REPEATS}, intensity={GHOST_INTENSITY}")
    print()

    written_images = 0
    written_labels = 0

    for _, row in smoke_df.iterrows():
        patient_id = row["patient_id"]
        print(f"Processing patient: {patient_id}")

        # Save degraded modality files in nnU-Net format.
        for modality_name, channel_id in MODALITIES:
            in_path = Path(row[modality_name])
            volume, affine, header = load_nifti(in_path)

            volume_01, brain_mask, v_min, v_max = normalize_for_degradation(volume)

            degraded_01 = apply_ghosting_3d(
                volume_01=volume_01,
                brain_mask=brain_mask,
                num_repeats=GHOST_NUM_REPEATS,
                intensity=GHOST_INTENSITY,
            )

            degraded_volume = restore_original_range(
                normalized_degraded=degraded_01,
                brain_mask=brain_mask,
                v_min=v_min,
                v_max=v_max,
                original_volume=volume,
            )

            out_name = f"{patient_id}_{channel_id}.nii.gz"
            out_path = OUT_IMAGES / out_name
            save_nifti(degraded_volume, affine, header, out_path)

            print(f"  Saved {modality_name.upper()} -> {out_path.name}")
            written_images += 1

        # Save remapped ground-truth label for later evaluation.
        seg_path = Path(row["seg"])
        seg, seg_affine, seg_header = load_nifti(seg_path)
        seg_remapped = remap_seg_labels(seg)

        label_out_path = OUT_LABELS / f"{patient_id}.nii.gz"
        save_label_nifti(seg_remapped, seg_affine, seg_header, label_out_path)
        print(f"  Saved label -> {label_out_path.name}")
        written_labels += 1

        print()

    print("=" * 80)
    print("Smoke-test degraded nnU-Net input preparation complete.")
    print(f"Images written: {written_images}")
    print(f"Labels written: {written_labels}")
    print("=" * 80)
    print()
    print("Next step after this script:")
    print("Run nnUNetv2_predict on the degraded imagesTs folder.")
    print()
    print("Reminder:")
    print("Do NOT commit temporary degraded .nii.gz files to GitHub.")


if __name__ == "__main__":
    main()
