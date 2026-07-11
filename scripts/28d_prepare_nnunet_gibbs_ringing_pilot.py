#!/usr/bin/env python3
"""
Script 28D: Prepare nnU-Net Gibbs-like ringing pilot inputs.

Purpose:
- Create degraded nnU-Net-compatible input images for a small 5-patient pilot.
- Artifact: gibbs_ringing
- Levels: gibbs_L1 to gibbs_L5
- Modalities: FLAIR, T1, T1ce, T2
- Testing only. No training. No validation fitting.
- Do NOT overwrite old ringing results.

Output:
nnunet/temporary_degraded_tests/gibbs_ringing_pilot/

Expected:
5 patients × 5 levels × 4 modalities = 100 image files
5 labels
"""

from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")
TEST_CSV = PROJECT_ROOT / "data/csvs/test_paths.csv"

OUT_ROOT = PROJECT_ROOT / "nnunet/temporary_degraded_tests/gibbs_ringing_pilot"
LABELS_DIR = OUT_ROOT / "labelsTs"

OUT_ROOT.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# nnU-Net modality mapping
# ------------------------------------------------------------

MODALITIES = [
    ("flair", "0000"),
    ("t1", "0001"),
    ("t1ce", "0002"),
    ("t2", "0003"),
]


# ------------------------------------------------------------
# Gibbs ringing levels
# ------------------------------------------------------------

LEVELS = {
    "gibbs_L1": {"strength": 0.045, "wavelength": 7.0},
    "gibbs_L2": {"strength": 0.075, "wavelength": 6.5},
    "gibbs_L3": {"strength": 0.110, "wavelength": 6.0},
    "gibbs_L4": {"strength": 0.155, "wavelength": 5.5},
    "gibbs_L5": {"strength": 0.210, "wavelength": 5.0},
}


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def normalize_01(volume):
    """
    Normalize MRI volume to [0, 1] using nonzero brain voxels.
    """
    volume = volume.astype(np.float32)
    brain_mask = volume > 0

    if brain_mask.sum() == 0:
        return np.zeros_like(volume, dtype=np.float32), brain_mask

    vals = volume[brain_mask]
    lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)

    volume_clip = np.clip(volume, lo, hi)
    volume_01 = (volume_clip - lo) / (hi - lo + 1e-8)
    volume_01[~brain_mask] = 0.0

    return volume_01.astype(np.float32), brain_mask


def degrade_gibbs_ringing(volume_01, brain_mask, strength=0.10, wavelength=5.0):
    """
    Same Gibbs-like artifact logic used in Scripts 28A and 28B.
    """
    vol = volume_01.astype(np.float32).copy()

    smooth = ndimage.gaussian_filter(vol, sigma=0.5)

    edge = ndimage.gaussian_gradient_magnitude(smooth, sigma=0.7)
    edge = edge * brain_mask

    if edge.max() > 0:
        edge = edge / (edge.max() + 1e-8)

    edge_values = edge[brain_mask]
    threshold = np.percentile(edge_values, 82)
    strong_edges = edge > threshold

    dist = ndimage.distance_transform_edt(~strong_edges)

    ripple = np.sin(2.0 * np.pi * dist / wavelength)
    decay = np.exp(-dist / 7.0)

    edge_band = ndimage.gaussian_filter(strong_edges.astype(np.float32), sigma=2.0)
    if edge_band.max() > 0:
        edge_band = edge_band / (edge_band.max() + 1e-8)

    artifact = strength * ripple * decay * (0.45 + 0.55 * edge_band)
    artifact = artifact * brain_mask

    degraded = vol + artifact
    degraded = np.clip(degraded, 0.0, 1.0)
    degraded[~brain_mask] = 0.0

    return degraded.astype(np.float32)


def remap_seg_to_nnunet(seg):
    """
    BraTS original labels:
    0, 1, 2, 4

    nnU-Net labels:
    0, 1, 2, 3

    So original label 4 becomes 3.
    """
    seg = seg.astype(np.int16)
    out = np.zeros_like(seg, dtype=np.uint8)
    out[seg == 1] = 1
    out[seg == 2] = 2
    out[seg == 4] = 3
    out[seg == 3] = 3  # safe if already remapped
    return out


def save_nifti_like(data, reference_img, out_path):
    """
    Save data using affine/header from reference image.
    """
    out_img = nib.Nifti1Image(data, reference_img.affine, reference_img.header)
    nib.save(out_img, str(out_path))


def case_id_from_patient_id(patient_id):
    """
    BraTS20_Training_001 -> 001
    """
    return str(patient_id).split("_")[-1]


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("=" * 80)
    print("Script 28D: Prepare nnU-Net Gibbs-like ringing pilot inputs")
    print("=" * 80)

    test_df = pd.read_csv(TEST_CSV)

    # Use the same first 5 held-out test patients as the extended pilot style.
    pilot_df = test_df.head(5).copy()

    total_images = 0
    total_labels = 0

    for _, row in pilot_df.iterrows():
        patient_id = row["patient_id"]
        case_num = case_id_from_patient_id(patient_id)

        print(f"\nPreparing patient: {patient_id}")

        # Save shared label once per patient
        seg_img = nib.load(str(row["seg"]))
        seg = seg_img.get_fdata().astype(np.int16)
        seg_remap = remap_seg_to_nnunet(seg)

        label_out = LABELS_DIR / f"BraTS20_Training_{case_num}.nii.gz"
        save_nifti_like(seg_remap.astype(np.uint8), seg_img, label_out)
        total_labels += 1

        for level_name, params in LEVELS.items():
            level_dir = OUT_ROOT / level_name / "imagesTs"
            level_dir.mkdir(parents=True, exist_ok=True)

            for modality_col, channel_id in MODALITIES:
                img_path = Path(row[modality_col])
                img = nib.load(str(img_path))
                vol = img.get_fdata().astype(np.float32)

                vol_01, brain_mask = normalize_01(vol)

                degraded = degrade_gibbs_ringing(
                    vol_01,
                    brain_mask,
                    strength=params["strength"],
                    wavelength=params["wavelength"],
                )

                # nnU-Net test image naming:
                # BRATS_001_0000.nii.gz, BRATS_001_0001.nii.gz, etc.
                out_name = f"BRATS_{case_num}_{channel_id}.nii.gz"
                out_path = level_dir / out_name

                save_nifti_like(degraded.astype(np.float32), img, out_path)
                total_images += 1

    print("\n" + "=" * 80)
    print("Done preparing Gibbs ringing pilot inputs.")
    print(f"Output root: {OUT_ROOT}")
    print(f"Patients: {len(pilot_df)}")
    print(f"Levels: {len(LEVELS)}")
    print(f"Modalities per patient: {len(MODALITIES)}")
    print(f"Total images written: {total_images}")
    print(f"Labels written: {total_labels}")
    print("Expected images: 5 patients × 5 levels × 4 modalities = 100")
    print("Expected labels: 5")
    print("=" * 80)


if __name__ == "__main__":
    main()
