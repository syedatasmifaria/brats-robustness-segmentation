#!/usr/bin/env python3
"""
Script 19: Visualize 4-modal degradation levels.

Purpose:
- Load one held-out BraTS2020 test patient.
- Extract one 4-modal 3D patch:
    FLAIR + T1 + T1ce + T2
- Apply degradation to ALL FOUR MODALITIES.
- Save visual figures showing:
    rows    = modalities
    columns = clean, level 1, level 2, level 3, level 4, level 5

Important:
This script does NOT train or test the model.
It only creates visual evidence that degradations are applied correctly.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import importlib.util


# =============================================================================
# Paths and settings
# =============================================================================

PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

SCRIPT17_PATH = PROJECT_ROOT / "scripts" / "17_test_3d_unet_multimodal_clean_full.py"

TEST_CSV = PROJECT_ROOT / "data" / "csvs" / "test_paths.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = PROJECT_ROOT / "report_materials"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = (96, 96, 96)

# Use first test patient by default.
# Later, if you want, you can change this number.
PATIENT_INDEX = 0

# Which patch to visualize from the selected patient.
PATCH_INDEX = 0

SEED = 42

MODALITY_NAMES = ["FLAIR", "T1", "T1ce", "T2"]


# =============================================================================
# Import Script 17 utilities
# This reuses the same loading, normalization, cropping, and patch extraction.
# =============================================================================

spec = importlib.util.spec_from_file_location("script17", SCRIPT17_PATH)
script17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script17)


# =============================================================================
# Degradation settings
# Same severity definitions as Script 18.
# =============================================================================

DEGRADATION_LEVELS = {
    "blur": [
        {"level": 1, "sigma": 0.5},
        {"level": 2, "sigma": 1.0},
        {"level": 3, "sigma": 1.5},
        {"level": 4, "sigma": 2.0},
        {"level": 5, "sigma": 2.5},
    ],
    "noise": [
        {"level": 1, "std": 0.02},
        {"level": 2, "std": 0.04},
        {"level": 3, "std": 0.06},
        {"level": 4, "std": 0.08},
        {"level": 5, "std": 0.10},
    ],
    "contrast": [
        {"level": 1, "factor": 0.90},
        {"level": 2, "factor": 0.75},
        {"level": 3, "factor": 0.60},
        {"level": 4, "factor": 0.45},
        {"level": 5, "factor": 0.30},
    ],
    "ringing": [
        {"level": 1, "keep_ratio": 0.85},
        {"level": 2, "keep_ratio": 0.75},
        {"level": 3, "keep_ratio": 0.65},
        {"level": 4, "keep_ratio": 0.55},
        {"level": 5, "keep_ratio": 0.45},
    ],
    "ghosting": [
        {"level": 1, "shift": 4, "intensity": 0.15},
        {"level": 2, "shift": 8, "intensity": 0.25},
        {"level": 3, "shift": 12, "intensity": 0.35},
        {"level": 4, "shift": 16, "intensity": 0.45},
        {"level": 5, "shift": 20, "intensity": 0.55},
    ],
}


# =============================================================================
# Degradation functions
# =============================================================================

def stable_seed(*parts):
    text = "_".join(str(p) for p in parts)
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def apply_blur(volume, sigma):
    return gaussian_filter(volume, sigma=sigma).astype(np.float32)


def apply_noise(volume, std, rng):
    noise = rng.normal(loc=0.0, scale=std, size=volume.shape).astype(np.float32)
    return (volume + noise).astype(np.float32)


def apply_contrast(volume, factor):
    return (volume * factor).astype(np.float32)


def apply_ringing(volume, keep_ratio):
    """
    Simulate ringing by keeping only central k-space frequencies.
    Lower keep_ratio = stronger loss of high-frequency detail.
    """
    fft = np.fft.fftn(volume)
    fft_shift = np.fft.fftshift(fft)

    x, y, z = volume.shape
    cx, cy, cz = x // 2, y // 2, z // 2

    rx = max(1, int((x * keep_ratio) / 2))
    ry = max(1, int((y * keep_ratio) / 2))
    rz = max(1, int((z * keep_ratio) / 2))

    mask = np.zeros_like(volume, dtype=np.float32)
    mask[
        cx - rx:cx + rx,
        cy - ry:cy + ry,
        cz - rz:cz + rz,
    ] = 1.0

    filtered = fft_shift * mask
    shifted_back = np.fft.ifftshift(filtered)
    reconstructed = np.fft.ifftn(shifted_back).real

    return reconstructed.astype(np.float32)


def apply_ghosting(volume, shift, intensity):
    """
    Simple ghosting simulation:
    Add shifted copies along one image axis.
    """
    ghost1 = np.roll(volume, shift=shift, axis=1)
    ghost2 = np.roll(volume, shift=-shift, axis=1)

    degraded = volume + intensity * ghost1 + (intensity / 2.0) * ghost2
    degraded = degraded / (1.0 + intensity + intensity / 2.0)

    return degraded.astype(np.float32)


def degrade_all_modalities(image_patch, artifact, params, rng):
    """
    image_patch shape:
        (4, 96, 96, 96)

    Applies selected artifact to all four channels:
        0 = FLAIR
        1 = T1
        2 = T1ce
        3 = T2
    """
    degraded = np.empty_like(image_patch, dtype=np.float32)

    for c in range(image_patch.shape[0]):
        vol = image_patch[c]

        if artifact == "blur":
            out = apply_blur(vol, sigma=params["sigma"])

        elif artifact == "noise":
            out = apply_noise(vol, std=params["std"], rng=rng)

        elif artifact == "contrast":
            out = apply_contrast(vol, factor=params["factor"])

        elif artifact == "ringing":
            out = apply_ringing(vol, keep_ratio=params["keep_ratio"])

        elif artifact == "ghosting":
            out = apply_ghosting(
                vol,
                shift=params["shift"],
                intensity=params["intensity"],
            )

        else:
            raise ValueError(f"Unknown artifact: {artifact}")

        degraded[c] = out.astype(np.float32)

    return degraded


# =============================================================================
# Visualization helpers
# =============================================================================

def robust_display_range(volume):
    """
    Use percentile-based display range so images are visually comparable.
    This prevents one bright voxel from ruining the contrast of the plot.
    """
    nonzero = volume[np.abs(volume) > 1e-8]

    if nonzero.size == 0:
        return -1, 1

    vmin = np.percentile(nonzero, 1)
    vmax = np.percentile(nonzero, 99)

    if abs(vmax - vmin) < 1e-8:
        vmin = nonzero.min()
        vmax = nonzero.max()

    if abs(vmax - vmin) < 1e-8:
        vmin, vmax = -1, 1

    return vmin, vmax


def save_artifact_figure(clean_patch, artifact, patient_id):
    """
    Save one figure for one artifact.

    Rows:
        FLAIR, T1, T1ce, T2

    Columns:
        Clean, L1, L2, L3, L4, L5
    """
    levels = DEGRADATION_LEVELS[artifact]

    all_versions = [("Clean", clean_patch)]

    for params in levels:
        level = params["level"]

        rng_seed = stable_seed(SEED, patient_id, PATCH_INDEX, artifact, level)
        rng = np.random.default_rng(rng_seed)

        degraded_patch = degrade_all_modalities(
            image_patch=clean_patch,
            artifact=artifact,
            params=params,
            rng=rng,
        )

        all_versions.append((f"L{level}", degraded_patch))

    # axial middle slice
    z = clean_patch.shape[-1] // 2

    n_rows = 4
    n_cols = 6

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12))

    for row_idx, modality_name in enumerate(MODALITY_NAMES):
        # Use the clean modality display range for all levels of the same modality.
        # This makes level changes easier to compare.
        clean_slice = clean_patch[row_idx, :, :, z]
        vmin, vmax = robust_display_range(clean_slice)

        for col_idx, (label, patch_version) in enumerate(all_versions):
            img_slice = patch_version[row_idx, :, :, z]

            ax = axes[row_idx, col_idx]
            ax.imshow(img_slice.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
            ax.axis("off")

            if row_idx == 0:
                ax.set_title(label, fontsize=12)

            if col_idx == 0:
                ax.set_ylabel(modality_name, fontsize=12)

    fig.suptitle(
        f"{artifact.upper()} degradation levels across all four modalities\n"
        f"Patient: {patient_id}, patch {PATCH_INDEX}",
        fontsize=16,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    result_path = RESULTS_DIR / f"19_visualize_{artifact}_levels_all_modalities.png"
    report_path = REPORT_DIR / f"19_visualize_{artifact}_levels_all_modalities.png"

    plt.savefig(result_path, dpi=200)
    plt.savefig(report_path, dpi=200)
    plt.close()

    print(f"Saved: {result_path}")
    print(f"Copied: {report_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("Script 19: Visualize degradation levels across all four modalities")
    print("=" * 80)

    if not SCRIPT17_PATH.exists():
        raise FileNotFoundError(f"Missing Script 17: {SCRIPT17_PATH}")

    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Missing test CSV: {TEST_CSV}")

    test_df = pd.read_csv(TEST_CSV)

    if PATIENT_INDEX >= len(test_df):
        raise ValueError(f"PATIENT_INDEX {PATIENT_INDEX} is out of range for {len(test_df)} test patients.")

    row = test_df.iloc[PATIENT_INDEX]
    patient_id = row["patient_id"]

    print(f"Selected patient index: {PATIENT_INDEX}")
    print(f"Selected patient ID: {patient_id}")

    flair = script17.load_nifti(row["flair"])
    t1 = script17.load_nifti(row["t1"])
    t1ce = script17.load_nifti(row["t1ce"])
    t2 = script17.load_nifti(row["t2"])
    seg = script17.load_nifti(row["seg"])

    seg = script17.remap_seg_labels(seg)

    raw_modalities = [flair, t1, t1ce, t2]

    bbox = script17.get_brain_bbox(raw_modalities)

    cropped_modalities = []

    for vol in raw_modalities:
        vol_crop = script17.crop_to_bbox(vol, bbox)
        vol_crop = script17.normalize_nonzero(vol_crop)
        cropped_modalities.append(vol_crop)

    seg_crop = script17.crop_to_bbox(seg, bbox)

    image_4ch = np.stack(cropped_modalities, axis=0).astype(np.float32)

    image_4ch = script17.pad_to_patch_size(image_4ch, PATCH_SIZE)
    seg_crop = script17.pad_to_patch_size(seg_crop, PATCH_SIZE).astype(np.int64)

    centers = script17.choose_patch_centers(seg_crop, PATCH_SIZE, patches_per_patient=4)

    if PATCH_INDEX >= len(centers):
        raise ValueError(f"PATCH_INDEX {PATCH_INDEX} is out of range. Available patches: {len(centers)}")

    center = centers[PATCH_INDEX]
    start = script17.get_patch_start(center, seg_crop.shape, PATCH_SIZE)

    clean_patch = script17.extract_patch(image_4ch, start, PATCH_SIZE)

    print(f"Clean patch shape: {clean_patch.shape}")
    print(f"Patch start: {start}")
    print("Expected patch shape: (4, 96, 96, 96)")
    print("Rows in output figures: FLAIR, T1, T1ce, T2")
    print("Columns in output figures: Clean, Level 1, Level 2, Level 3, Level 4, Level 5")

    for artifact in ["blur", "noise", "contrast", "ringing", "ghosting"]:
        print(f"\nCreating visualization for artifact: {artifact}")
        save_artifact_figure(clean_patch, artifact, patient_id)

    print("\n" + "=" * 80)
    print("Script 19 complete.")
    print("=" * 80)
    print("Open the saved PNG files in report_materials:")
    print("19_visualize_blur_levels_all_modalities.png")
    print("19_visualize_noise_levels_all_modalities.png")
    print("19_visualize_contrast_levels_all_modalities.png")
    print("19_visualize_ringing_levels_all_modalities.png")
    print("19_visualize_ghosting_levels_all_modalities.png")


if __name__ == "__main__":
    main()
