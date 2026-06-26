#!/usr/bin/env python3
"""
Script 22: Analyze degradation strength using MSE and PSNR.

Purpose:
- Load BraTS2020 4-modal test patches.
- Apply current degradation levels to all four modalities.
- Compute MSE and PSNR between clean and degraded patches.
- Save report-ready CSV, TXT summary, and plots.

Important project rule:
- This script does NOT train any model.
- This script does NOT save degraded .nii.gz files.
- Degradation is applied only for evaluation/analysis.
"""

import os
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter


# ============================================================
# Paths and settings
# ============================================================

PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")
TEST_CSV = PROJECT_ROOT / "data/csvs/test_paths.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_DIR = PROJECT_ROOT / "report_materials"

RESULTS_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

PATCH_SIZE = (96, 96, 96)
MODALITIES = ["flair", "t1", "t1ce", "t2"]

# Keep this moderate so the script runs quickly.
# This is for degradation strength analysis, not model testing.
NUM_TEST_PATIENTS = 20
PATCHES_PER_PATIENT = 4
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ============================================================
# Current degradation settings from the existing robustness test
# ============================================================

DEGRADATION_LEVELS = {
    "blur": [
        {"sigma": 0.5},
        {"sigma": 1.0},
        {"sigma": 1.5},
        {"sigma": 2.0},
        {"sigma": 2.5},
    ],
    "noise": [
        {"std": 0.02},
        {"std": 0.04},
        {"std": 0.06},
        {"std": 0.08},
        {"std": 0.10},
    ],
    "contrast": [
        {"factor": 0.90},
        {"factor": 0.70},
        {"factor": 0.50},
        {"factor": 0.40},
        {"factor": 0.30},
    ],
    "ringing": [
        {"keep_fraction": 0.85},
        {"keep_fraction": 0.75},
        {"keep_fraction": 0.65},
        {"keep_fraction": 0.55},
        {"keep_fraction": 0.45},
    ],
    "ghosting": [
        {"num_ghosts": 4, "intensity": 0.15},
        {"num_ghosts": 8, "intensity": 0.25},
        {"num_ghosts": 12, "intensity": 0.35},
        {"num_ghosts": 16, "intensity": 0.45},
        {"num_ghosts": 20, "intensity": 0.55},
    ],
}


# ============================================================
# Loading and preprocessing helpers
# ============================================================

def load_nifti(path):
    """Load a NIfTI file as float32 numpy array."""
    return nib.load(str(path)).get_fdata().astype(np.float32)


def remap_segmentation(seg):
    """
    BraTS original labels are [0, 1, 2, 4].
    Model labels are [0, 1, 2, 3], where original 4 becomes 3.
    """
    seg = seg.astype(np.int16)
    seg_remap = seg.copy()
    seg_remap[seg_remap == 4] = 3
    return seg_remap


def normalize_modality(volume):
    """
    Normalize one MRI modality using nonzero brain voxels.

    We use percentile clipping and min-max scaling to [0, 1].
    This makes MSE/PSNR easier to interpret across modalities.
    """
    volume = volume.astype(np.float32)
    brain_mask = volume > 0

    if brain_mask.sum() == 0:
        return volume

    brain_values = volume[brain_mask]

    low = np.percentile(brain_values, 1)
    high = np.percentile(brain_values, 99)

    volume = np.clip(volume, low, high)

    denom = high - low
    if denom < 1e-8:
        return np.zeros_like(volume, dtype=np.float32)

    volume = (volume - low) / denom
    volume = np.clip(volume, 0.0, 1.0)

    # Keep background at zero.
    volume[~brain_mask] = 0.0

    return volume.astype(np.float32)


def load_patient_4modal(row):
    """
    Load FLAIR, T1, T1ce, T2 and segmentation for one patient.

    Uses the correct CSV columns:
    row["flair"], row["t1"], row["t1ce"], row["t2"], row["seg"]
    """
    channels = []

    for modality in MODALITIES:
        vol = load_nifti(row[modality])
        vol = normalize_modality(vol)
        channels.append(vol)

    image = np.stack(channels, axis=0)  # shape: (4, H, W, D)

    seg = load_nifti(row["seg"])
    seg = remap_segmentation(seg)

    return image, seg


# ============================================================
# Patch sampling helpers
# ============================================================

def get_patch_bounds(center, patch_size, volume_shape):
    """
    Convert a patch center into safe slice bounds.
    """
    starts = []
    ends = []

    for c, p, dim in zip(center, patch_size, volume_shape):
        start = int(c - p // 2)
        end = start + p

        if start < 0:
            start = 0
            end = p

        if end > dim:
            end = dim
            start = dim - p

        starts.append(start)
        ends.append(end)

    return starts, ends


def extract_patch(image, seg, patch_size=PATCH_SIZE):
    """
    Extract a tumor-centered patch when possible.
    If no tumor voxel is found, use a random valid patch.

    image shape: (4, H, W, D)
    seg shape: (H, W, D)
    """
    _, h, w, d = image.shape
    volume_shape = (h, w, d)

    tumor_coords = np.argwhere(seg > 0)

    if len(tumor_coords) > 0:
        center = tumor_coords[np.random.randint(len(tumor_coords))]
    else:
        center = np.array([
            np.random.randint(patch_size[0] // 2, h - patch_size[0] // 2),
            np.random.randint(patch_size[1] // 2, w - patch_size[1] // 2),
            np.random.randint(patch_size[2] // 2, d - patch_size[2] // 2),
        ])

    starts, ends = get_patch_bounds(center, patch_size, volume_shape)

    image_patch = image[
        :,
        starts[0]:ends[0],
        starts[1]:ends[1],
        starts[2]:ends[2],
    ]

    seg_patch = seg[
        starts[0]:ends[0],
        starts[1]:ends[1],
        starts[2]:ends[2],
    ]

    return image_patch.astype(np.float32), seg_patch.astype(np.int16)


# ============================================================
# Degradation functions
# ============================================================

def apply_blur(volume, sigma):
    """Apply Gaussian blur to one 3D modality."""
    degraded = gaussian_filter(volume, sigma=sigma)
    return np.clip(degraded, 0.0, 1.0).astype(np.float32)


def apply_noise(volume, std, rng):
    """Add Gaussian noise to one 3D modality."""
    noise = rng.normal(loc=0.0, scale=std, size=volume.shape).astype(np.float32)
    degraded = volume + noise
    return np.clip(degraded, 0.0, 1.0).astype(np.float32)


def apply_contrast(volume, factor):
    """
    Reduce contrast around the midpoint 0.5.
    factor < 1 means lower contrast.
    """
    degraded = 0.5 + factor * (volume - 0.5)
    return np.clip(degraded, 0.0, 1.0).astype(np.float32)


def apply_ringing(volume, keep_fraction):
    """
    Simulate ringing by removing high frequencies in k-space.

    Lower keep_fraction means stronger frequency truncation.
    """
    fft = np.fft.fftn(volume)
    fft_shift = np.fft.fftshift(fft)

    h, w, d = volume.shape
    center = np.array([h // 2, w // 2, d // 2])

    keep_h = int(h * keep_fraction / 2)
    keep_w = int(w * keep_fraction / 2)
    keep_d = int(d * keep_fraction / 2)

    mask = np.zeros_like(volume, dtype=bool)

    mask[
        center[0] - keep_h:center[0] + keep_h,
        center[1] - keep_w:center[1] + keep_w,
        center[2] - keep_d:center[2] + keep_d,
    ] = True

    fft_shift_filtered = fft_shift * mask
    fft_filtered = np.fft.ifftshift(fft_shift_filtered)
    degraded = np.real(np.fft.ifftn(fft_filtered))

    return np.clip(degraded, 0.0, 1.0).astype(np.float32)


def apply_ghosting(volume, num_ghosts, intensity):
    """
    Simulate simple ghosting by adding shifted copies along one axis.
    """
    degraded = volume.copy().astype(np.float32)

    axis = 1
    shift_step = max(1, volume.shape[axis] // (num_ghosts + 1))

    for i in range(1, num_ghosts + 1):
        shift = i * shift_step
        ghost = np.roll(volume, shift=shift, axis=axis)
        degraded += (intensity / num_ghosts) * ghost

    degraded = degraded / (1.0 + intensity)

    return np.clip(degraded, 0.0, 1.0).astype(np.float32)


def degrade_one_modality(volume, artifact, params, rng):
    """Apply one artifact to one modality."""
    if artifact == "blur":
        return apply_blur(volume, sigma=params["sigma"])

    if artifact == "noise":
        return apply_noise(volume, std=params["std"], rng=rng)

    if artifact == "contrast":
        return apply_contrast(volume, factor=params["factor"])

    if artifact == "ringing":
        return apply_ringing(volume, keep_fraction=params["keep_fraction"])

    if artifact == "ghosting":
        return apply_ghosting(
            volume,
            num_ghosts=params["num_ghosts"],
            intensity=params["intensity"],
        )

    raise ValueError(f"Unknown artifact: {artifact}")


def degrade_all_modalities(image_patch, artifact, params, rng):
    """
    Apply the same degradation setting to all four modalities.

    image_patch shape: (4, 96, 96, 96)
    """
    degraded_channels = []

    for c in range(image_patch.shape[0]):
        degraded = degrade_one_modality(
            volume=image_patch[c],
            artifact=artifact,
            params=params,
            rng=rng,
        )
        degraded_channels.append(degraded)

    return np.stack(degraded_channels, axis=0).astype(np.float32)


# ============================================================
# MSE / PSNR helpers
# ============================================================

def compute_mse(clean, degraded):
    """Mean squared error."""
    return float(np.mean((clean.astype(np.float32) - degraded.astype(np.float32)) ** 2))


def compute_psnr(clean, degraded, data_range=1.0):
    """
    Peak Signal-to-Noise Ratio.

    Since patches are normalized to [0, 1], data_range = 1.0.
    """
    mse = compute_mse(clean, degraded)

    if mse <= 1e-12:
        return float("inf")

    return float(20.0 * math.log10(data_range / math.sqrt(mse)))


# ============================================================
# Main analysis
# ============================================================

def main():
    print("=" * 80)
    print("Script 22: Degradation strength analysis using MSE and PSNR")
    print("=" * 80)

    df = pd.read_csv(TEST_CSV)

    if NUM_TEST_PATIENTS < len(df):
        df = df.sample(n=NUM_TEST_PATIENTS, random_state=RANDOM_SEED).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    print(f"Using test patients: {len(df)}")
    print(f"Patches per patient: {PATCHES_PER_PATIENT}")
    print(f"Patch size: {PATCH_SIZE}")
    print(f"Modalities: {MODALITIES}")
    print()

    rows = []
    rng = np.random.default_rng(RANDOM_SEED)

    for patient_idx, row in df.iterrows():
        patient_id = row["patient_id"]
        print(f"[{patient_idx + 1}/{len(df)}] Loading patient: {patient_id}")

        image, seg = load_patient_4modal(row)

        for patch_idx in range(PATCHES_PER_PATIENT):
            clean_patch, seg_patch = extract_patch(image, seg, PATCH_SIZE)

            for artifact, levels in DEGRADATION_LEVELS.items():
                for level_idx, params in enumerate(levels, start=1):
                    degraded_patch = degrade_all_modalities(
                        image_patch=clean_patch,
                        artifact=artifact,
                        params=params,
                        rng=rng,
                    )

                    # Overall 4-modal MSE/PSNR
                    overall_mse = compute_mse(clean_patch, degraded_patch)
                    overall_psnr = compute_psnr(clean_patch, degraded_patch)

                    rows.append({
                        "patient_id": patient_id,
                        "patch_idx": patch_idx,
                        "artifact": artifact,
                        "level": level_idx,
                        "modality": "all_modalities",
                        "mse": overall_mse,
                        "psnr": overall_psnr,
                        "params": str(params),
                    })

                    # Per-modality MSE/PSNR
                    for modality_idx, modality_name in enumerate(MODALITIES):
                        modality_mse = compute_mse(
                            clean_patch[modality_idx],
                            degraded_patch[modality_idx],
                        )
                        modality_psnr = compute_psnr(
                            clean_patch[modality_idx],
                            degraded_patch[modality_idx],
                        )

                        rows.append({
                            "patient_id": patient_id,
                            "patch_idx": patch_idx,
                            "artifact": artifact,
                            "level": level_idx,
                            "modality": modality_name,
                            "mse": modality_mse,
                            "psnr": modality_psnr,
                            "params": str(params),
                        })

    metrics_df = pd.DataFrame(rows)

    metrics_path = RESULTS_DIR / "22_degradation_strength_mse_psnr_metrics.csv"
    summary_path = REPORT_DIR / "22_degradation_strength_mse_psnr_summary.csv"
    txt_path = REPORT_DIR / "22_degradation_strength_mse_psnr_summary.txt"

    metrics_df.to_csv(metrics_path, index=False)

    summary_df = (
        metrics_df
        .groupby(["artifact", "level", "modality", "params"], as_index=False)
        .agg(
            mean_mse=("mse", "mean"),
            std_mse=("mse", "std"),
            mean_psnr=("psnr", "mean"),
            std_psnr=("psnr", "std"),
            n=("mse", "count"),
        )
    )

    summary_df.to_csv(summary_path, index=False)

    # Report-focused overall summary only
    overall_summary = summary_df[summary_df["modality"] == "all_modalities"].copy()

    with open(txt_path, "w") as f:
        f.write("Script 22: Degradation strength analysis using MSE and PSNR\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Test patients used: {len(df)}\n")
        f.write(f"Patches per patient: {PATCHES_PER_PATIENT}\n")
        f.write(f"Patch size: {PATCH_SIZE}\n")
        f.write("Modalities: FLAIR, T1, T1ce, T2\n")
        f.write("Images normalized to [0, 1] before MSE/PSNR calculation.\n\n")

        f.write("Interpretation guide:\n")
        f.write("- Higher MSE means stronger image change.\n")
        f.write("- Lower PSNR means stronger degradation.\n")
        f.write("- If an artifact has very low MSE and very high PSNR, it may be too weak.\n\n")

        f.write("Overall 4-modal summary:\n")
        f.write("-" * 80 + "\n")

        for _, r in overall_summary.iterrows():
            f.write(
                f"{r['artifact']} level {int(r['level'])}: "
                f"mean MSE={r['mean_mse']:.6f}, "
                f"mean PSNR={r['mean_psnr']:.2f} dB, "
                f"params={r['params']}\n"
            )

    print()
    print("Saved metrics:")
    print(f"  {metrics_path}")
    print(f"  {summary_path}")
    print(f"  {txt_path}")

    # ========================================================
    # Plot MSE curve
    # ========================================================

    plt.figure(figsize=(8, 5))

    for artifact in sorted(overall_summary["artifact"].unique()):
        sub = overall_summary[overall_summary["artifact"] == artifact].sort_values("level")
        plt.plot(sub["level"], sub["mean_mse"], marker="o", label=artifact)

    plt.xlabel("Degradation level")
    plt.ylabel("Mean MSE")
    plt.title("Degradation strength by MSE")
    plt.xticks([1, 2, 3, 4, 5])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    mse_plot_path = REPORT_DIR / "22_degradation_strength_mse_curve.png"
    plt.savefig(mse_plot_path, dpi=300)
    plt.close()

    # ========================================================
    # Plot PSNR curve
    # ========================================================

    plt.figure(figsize=(8, 5))

    for artifact in sorted(overall_summary["artifact"].unique()):
        sub = overall_summary[overall_summary["artifact"] == artifact].sort_values("level")
        plt.plot(sub["level"], sub["mean_psnr"], marker="o", label=artifact)

    plt.xlabel("Degradation level")
    plt.ylabel("Mean PSNR (dB)")
    plt.title("Degradation strength by PSNR")
    plt.xticks([1, 2, 3, 4, 5])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    psnr_plot_path = REPORT_DIR / "22_degradation_strength_psnr_curve.png"
    plt.savefig(psnr_plot_path, dpi=300)
    plt.close()

    print(f"  {mse_plot_path}")
    print(f"  {psnr_plot_path}")

    print()
    print("=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()
