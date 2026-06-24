#!/usr/bin/env python3
"""
Script 18: Degraded testing of full clean 4-modal 3D U-Net.

Purpose:
- Load the full trained clean 4-modal 3D U-Net from Script 16.
- Test on held-out BraTS2020 test patients.
- Use all four modalities:
    FLAIR + T1 + T1ce + T2
- Apply degradation to ALL FOUR MODALITIES during testing only.
- Do NOT train on degraded images.
- Compare degraded performance against clean patch baseline.

Important:
This is patch-based degraded evaluation, matching Script 17.
It does not save degraded .nii.gz volumes.
Degradation is applied dynamically/on-the-fly to the test patches.
"""

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# =============================================================================
# Paths and settings
# =============================================================================

PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

SCRIPT17_PATH = PROJECT_ROOT / "scripts" / "17_test_3d_unet_multimodal_clean_full.py"

TEST_CSV = PROJECT_ROOT / "data" / "csvs" / "test_paths.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "3d_unet_multimodal_clean_full_best.pth"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_CSV = RESULTS_DIR / "18_full_multimodal_3d_degraded_test_metrics.csv"
SUMMARY_CSV = RESULTS_DIR / "18_full_multimodal_3d_degraded_test_summary.csv"
CURVE_PNG = RESULTS_DIR / "18_full_multimodal_3d_degraded_dice_curve.png"
LEVEL5_BAR_PNG = RESULTS_DIR / "18_full_multimodal_3d_level5_dice_drop_bar.png"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PATCH_SIZE = (96, 96, 96)
PATCHES_PER_PATIENT = 4

IN_CHANNELS = 4
NUM_CLASSES = 4
BASE_CHANNELS = 16

SEED = 42


# =============================================================================
# Import Script 17 utilities
# This avoids architecture mismatch. Script 18 reuses the exact fixed UNet3D.
# =============================================================================

spec = importlib.util.spec_from_file_location("script17", SCRIPT17_PATH)
script17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script17)


# =============================================================================
# Degradation settings
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
# Degradation utilities
# =============================================================================

def stable_seed(*parts):
    """
    Create deterministic seed from text/numbers.
    Python's built-in hash is not stable across sessions, so we use hashlib.
    """
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
    Simulate ringing by removing high-frequency information in k-space
    with a hard central frequency mask.

    volume shape: 3D patch
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
        cz - rz:cz + rz
    ] = 1.0

    filtered = fft_shift * mask
    shifted_back = np.fft.ifftshift(filtered)
    reconstructed = np.fft.ifftn(shifted_back).real

    return reconstructed.astype(np.float32)


def apply_ghosting(volume, shift, intensity):
    """
    Simple ghosting simulation:
    Add shifted copies along one in-plane axis.
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

    Apply the same degradation type and level to all four modalities:
        channel 0 = FLAIR
        channel 1 = T1
        channel 2 = T1ce
        channel 3 = T2
    """
    degraded = np.empty_like(image_patch, dtype=np.float32)

    for c in range(image_patch.shape[0]):
        vol = image_patch[c]

        if artifact == "clean":
            out = vol

        elif artifact == "blur":
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
# Model loading
# =============================================================================

def load_model():
    model = script17.UNet3D(
        in_channels=IN_CHANNELS,
        out_channels=NUM_CLASSES,
        base_channels=BASE_CHANNELS,
    ).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded checkpoint using key: model_state_dict")

        if "epoch" in checkpoint:
            print(f"Checkpoint epoch: {checkpoint['epoch']}")
        if "best_val_loss" in checkpoint:
            print(f"Checkpoint best_val_loss: {checkpoint['best_val_loss']}")

    else:
        model.load_state_dict(checkpoint)
        print("Loaded checkpoint as raw state_dict")

    model.eval()
    return model


# =============================================================================
# Plotting
# =============================================================================

def save_dice_curve(summary_df):
    plot_df = summary_df[summary_df["artifact"] != "clean"].copy()

    plt.figure(figsize=(9, 6))

    for artifact in ["blur", "noise", "contrast", "ringing", "ghosting"]:
        sub = plot_df[plot_df["artifact"] == artifact].sort_values("level")
        plt.plot(
            sub["level"],
            sub["mean_whole_tumor_dice"],
            marker="o",
            label=artifact,
        )

    clean_row = summary_df[summary_df["artifact"] == "clean"].iloc[0]
    clean_dice = clean_row["mean_whole_tumor_dice"]

    plt.axhline(clean_dice, linestyle="--", label=f"clean baseline ({clean_dice:.4f})")

    plt.xlabel("Degradation level")
    plt.ylabel("Mean whole tumor Dice")
    plt.title("4-modal 3D U-Net robustness under all-modality degradation")
    plt.xticks([1, 2, 3, 4, 5])
    plt.legend()
    plt.tight_layout()
    plt.savefig(CURVE_PNG, dpi=200)
    plt.close()

    print(f"Saved Dice curve: {CURVE_PNG}")


def save_level5_bar(summary_df):
    clean_dice = summary_df[summary_df["artifact"] == "clean"].iloc[0]["mean_whole_tumor_dice"]

    level5 = summary_df[
        (summary_df["artifact"] != "clean") &
        (summary_df["level"] == 5)
    ].copy()

    level5["dice_drop"] = clean_dice - level5["mean_whole_tumor_dice"]
    level5 = level5.sort_values("dice_drop", ascending=False)

    plt.figure(figsize=(8, 5))
    plt.bar(level5["artifact"], level5["dice_drop"])
    plt.xlabel("Degradation type")
    plt.ylabel("Dice drop from clean baseline")
    plt.title("Level 5 degradation impact on 4-modal 3D U-Net")
    plt.tight_layout()
    plt.savefig(LEVEL5_BAR_PNG, dpi=200)
    plt.close()

    print(f"Saved level-5 drop bar chart: {LEVEL5_BAR_PNG}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("Script 18: Full 4-modal 3D U-Net degraded test")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Test CSV: {TEST_CSV}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Patch size: {PATCH_SIZE}")
    print(f"Patches per patient: {PATCHES_PER_PATIENT}")
    print("Degradation mode: ALL FOUR MODALITIES degraded together")
    print("Training data: clean only")
    print("Testing data: clean + degraded")

    if not SCRIPT17_PATH.exists():
        raise FileNotFoundError(f"Missing Script 17 file: {SCRIPT17_PATH}")

    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Missing test CSV: {TEST_CSV}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {MODEL_PATH}")

    test_df = pd.read_csv(TEST_CSV)
    print(f"Loaded test patients: {len(test_df)}")
    print("CSV columns:", list(test_df.columns))

    required_cols = ["patient_id", "flair", "t1", "t1ce", "t2", "seg"]
    for col in required_cols:
        if col not in test_df.columns:
            raise ValueError(f"Missing required CSV column: {col}")

    model = load_model()

    all_rows = []

    conditions = [{"artifact": "clean", "level": 0, "params": {}}]

    for artifact, levels in DEGRADATION_LEVELS.items():
        for params in levels:
            conditions.append({
                "artifact": artifact,
                "level": params["level"],
                "params": params,
            })

    print(f"Total test conditions including clean: {len(conditions)}")
    print(f"Expected total forward passes: {len(test_df) * PATCHES_PER_PATIENT * len(conditions)}")

    with torch.no_grad():
        for patient_idx, row in test_df.iterrows():
            patient_id = row["patient_id"]

            print(f"\n[{patient_idx + 1}/{len(test_df)}] Preparing patient: {patient_id}")

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

            centers = script17.choose_patch_centers(seg_crop, PATCH_SIZE, PATCHES_PER_PATIENT)

            for patch_idx, center in enumerate(centers):
                start = script17.get_patch_start(center, seg_crop.shape, PATCH_SIZE)

                clean_patch = script17.extract_patch(image_4ch, start, PATCH_SIZE)
                seg_patch = script17.extract_patch(seg_crop, start, PATCH_SIZE)

                for condition in conditions:
                    artifact = condition["artifact"]
                    level = condition["level"]
                    params = condition["params"]

                    rng_seed = stable_seed(SEED, patient_id, patch_idx, artifact, level)
                    rng = np.random.default_rng(rng_seed)

                    test_patch = degrade_all_modalities(
                        image_patch=clean_patch,
                        artifact=artifact,
                        params=params,
                        rng=rng,
                    )

                    x = torch.from_numpy(test_patch).unsqueeze(0).float().to(DEVICE)

                    logits = model(x)
                    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)

                    metrics = script17.compute_metrics(pred, seg_patch)

                    out_row = {
                        "patient_id": patient_id,
                        "patient_index": patient_idx,
                        "patch_index": patch_idx,
                        "patch_start_x": start[0],
                        "patch_start_y": start[1],
                        "patch_start_z": start[2],
                        "artifact": artifact,
                        "level": level,
                        "params": str(params),
                        "all_modalities_degraded": True,
                        **metrics,
                    }

                    all_rows.append(out_row)

            print(f"  Done patient {patient_id}")

    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(METRICS_CSV, index=False)

    print("\n" + "=" * 80)
    print("Saved degraded test metrics")
    print("=" * 80)
    print(f"Metrics CSV: {METRICS_CSV}")
    print(f"Total rows: {len(metrics_df)}")

    group_cols = ["artifact", "level", "params", "all_modalities_degraded"]

    summary_df = (
        metrics_df
        .groupby(group_cols, dropna=False)
        .agg(
            num_patches=("whole_tumor_dice", "count"),

            mean_whole_tumor_dice=("whole_tumor_dice", "mean"),
            std_whole_tumor_dice=("whole_tumor_dice", "std"),

            mean_whole_tumor_iou=("whole_tumor_iou", "mean"),
            std_whole_tumor_iou=("whole_tumor_iou", "std"),

            mean_dice_class_1=("dice_class_1", "mean"),
            mean_dice_class_2=("dice_class_2", "mean"),
            mean_dice_class_3=("dice_class_3", "mean"),

            mean_iou_class_1=("iou_class_1", "mean"),
            mean_iou_class_2=("iou_class_2", "mean"),
            mean_iou_class_3=("iou_class_3", "mean"),

            mean_true_tumor_voxels=("true_tumor_voxels", "mean"),
            mean_pred_tumor_voxels=("pred_tumor_voxels", "mean"),
        )
        .reset_index()
    )

    clean_dice = summary_df[summary_df["artifact"] == "clean"].iloc[0]["mean_whole_tumor_dice"]
    clean_iou = summary_df[summary_df["artifact"] == "clean"].iloc[0]["mean_whole_tumor_iou"]

    summary_df["clean_reference_dice"] = clean_dice
    summary_df["clean_reference_iou"] = clean_iou
    summary_df["whole_tumor_dice_drop"] = clean_dice - summary_df["mean_whole_tumor_dice"]
    summary_df["whole_tumor_iou_drop"] = clean_iou - summary_df["mean_whole_tumor_iou"]

    summary_df = summary_df.sort_values(["artifact", "level"]).reset_index(drop=True)

    summary_df.to_csv(SUMMARY_CSV, index=False)

    print("\n" + "=" * 80)
    print("Degraded test summary")
    print("=" * 80)
    print(f"Clean reference Dice: {clean_dice:.4f}")
    print(f"Clean reference IoU:  {clean_iou:.4f}")

    display_cols = [
        "artifact",
        "level",
        "mean_whole_tumor_dice",
        "whole_tumor_dice_drop",
        "mean_whole_tumor_iou",
        "whole_tumor_iou_drop",
    ]

    print(summary_df[display_cols].to_string(index=False))

    print(f"\nSummary CSV: {SUMMARY_CSV}")

    save_dice_curve(summary_df)
    save_level5_bar(summary_df)

    print("\n" + "=" * 80)
    print("Script 18 complete.")
    print("=" * 80)
    print("Important interpretation:")
    print("This is all-modality degraded testing.")
    print("All four input modalities were degraded together during testing only.")
    print("Compare Dice drops against the clean baseline to identify which artifact hurts most.")


if __name__ == "__main__":
    main()
