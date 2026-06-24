#!/usr/bin/env python3
"""
Script 17: Test full clean 4-modal 3D U-Net on clean held-out test patients.

Purpose:
- Load the full trained 4-modal 3D U-Net checkpoint from Script 16.
- Test on held-out BraTS2020 test patients.
- Use clean images only: FLAIR + T1 + T1ce + T2.
- Compute patch-level Dice and IoU:
    1) Whole tumor: labels 1, 2, 3 combined
    2) Class-wise Dice/IoU for labels 1, 2, 3
- Save CSV metrics, summary CSV, and prediction preview PNG.

Important:
This is still patch-based evaluation, not full-volume sliding-window reconstruction.
That is okay for now because our training was patch-based and this gives us a clean
baseline before degraded testing.
"""

import os
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# =============================================================================
# Paths and settings
# =============================================================================

PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

TEST_CSV = PROJECT_ROOT / "data" / "csvs" / "test_paths.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "3d_unet_multimodal_clean_full_best.pth"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_CSV = RESULTS_DIR / "17_full_multimodal_3d_clean_test_metrics.csv"
SUMMARY_CSV = RESULTS_DIR / "17_full_multimodal_3d_clean_test_summary.csv"
PREVIEW_PNG = RESULTS_DIR / "17_full_multimodal_3d_clean_test_predictions.png"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEED = 42
PATCH_SIZE = (96, 96, 96)
NUM_CLASSES = 4
IN_CHANNELS = 4
BASE_CHANNELS = 16

# Use all held-out test patients.
# To keep evaluation stable and not insanely slow, we use several deterministic patches per patient.
PATCHES_PER_PATIENT = 4

# For preview image
MAX_PREVIEW_PATCHES = 6


# =============================================================================
# Reproducibility
# =============================================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =============================================================================
# Model definition
# Must match Script 16 architecture
# =============================================================================

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, base_channels=16):
        super().__init__()

        self.enc1 = DoubleConv3D(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = DoubleConv3D(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = DoubleConv3D(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv3D(base_channels * 4, base_channels * 8)

        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv3D(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3D(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(base_channels * 2, base_channels)

        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)

        e2 = self.enc2(self.pool1(e1))

        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)


# =============================================================================
# Data utilities
# =============================================================================

def load_nifti(path):
    """Load a NIfTI file as numpy array."""
    return nib.load(str(path)).get_fdata()


def normalize_nonzero(volume):
    """
    Normalize MRI volume using only nonzero brain voxels.

    Why:
    MRI background is usually 0. If we include background in mean/std,
    the normalization becomes garbage. Garbage in, garbage out.
    """
    volume = volume.astype(np.float32)
    mask = volume > 0

    if mask.sum() == 0:
        return volume

    mean = volume[mask].mean()
    std = volume[mask].std()

    if std < 1e-8:
        std = 1.0

    volume[mask] = (volume[mask] - mean) / std
    volume[~mask] = 0.0

    return volume


def remap_seg_labels(seg):
    """
    BraTS original labels are [0, 1, 2, 4].
    Model expects [0, 1, 2, 3].
    So original label 4 becomes class 3.
    """
    seg = seg.astype(np.int64)
    seg_remap = np.zeros_like(seg, dtype=np.int64)
    seg_remap[seg == 1] = 1
    seg_remap[seg == 2] = 2
    seg_remap[seg == 4] = 3
    return seg_remap


def get_brain_bbox(modalities):
    """
    Get shared brain bounding box from all modalities.

    We crop all modalities and segmentation using the same box so they remain aligned.
    """
    brain_mask = np.zeros_like(modalities[0], dtype=bool)

    for vol in modalities:
        brain_mask |= vol > 0

    coords = np.argwhere(brain_mask)

    if coords.shape[0] == 0:
        # fallback: full image
        shape = modalities[0].shape
        return (0, shape[0], 0, shape[1], 0, shape[2])

    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0) + 1

    return (x_min, x_max, y_min, y_max, z_min, z_max)


def crop_to_bbox(volume, bbox):
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    return volume[x_min:x_max, y_min:y_max, z_min:z_max]


def pad_to_patch_size(volume, patch_size, is_seg=False):
    """
    Pad a volume if any dimension is smaller than patch size.
    """
    px, py, pz = patch_size
    x, y, z = volume.shape[-3:]

    pad_x = max(0, px - x)
    pad_y = max(0, py - y)
    pad_z = max(0, pz - z)

    if volume.ndim == 4:
        # shape: C, X, Y, Z
        pad_width = (
            (0, 0),
            (pad_x // 2, pad_x - pad_x // 2),
            (pad_y // 2, pad_y - pad_y // 2),
            (pad_z // 2, pad_z - pad_z // 2),
        )
    else:
        # shape: X, Y, Z
        pad_width = (
            (pad_x // 2, pad_x - pad_x // 2),
            (pad_y // 2, pad_y - pad_y // 2),
            (pad_z // 2, pad_z - pad_z // 2),
        )

    mode = "constant"
    constant_values = 0

    return np.pad(volume, pad_width, mode=mode, constant_values=constant_values)


def get_patch_start(center, shape, patch_size):
    """
    Convert a desired center voxel into a valid patch start index.
    """
    starts = []

    for c, dim, p in zip(center, shape, patch_size):
        start = int(c - p // 2)
        start = max(0, start)
        start = min(start, dim - p)
        starts.append(start)

    return tuple(starts)


def extract_patch(volume, start, patch_size):
    """
    Extract patch from either:
    - 4D image: C, X, Y, Z
    - 3D mask: X, Y, Z
    """
    sx, sy, sz = start
    px, py, pz = patch_size

    if volume.ndim == 4:
        return volume[:, sx:sx + px, sy:sy + py, sz:sz + pz]
    else:
        return volume[sx:sx + px, sy:sy + py, sz:sz + pz]


def choose_patch_centers(seg, patch_size, patches_per_patient):
    """
    Choose deterministic patch centers.

    Strategy:
    - Prefer tumor-centered patches if tumor exists.
    - Also include a central brain patch.
    - This avoids testing only empty background patches.
    """
    shape = seg.shape
    centers = []

    tumor_coords = np.argwhere(seg > 0)

    if tumor_coords.shape[0] > 0:
        # Tumor center
        tumor_center = tumor_coords.mean(axis=0).astype(int)
        centers.append(tuple(tumor_center))

        # Deterministic tumor samples: 25%, 50%, 75% through tumor coordinate list
        idxs = np.linspace(0, tumor_coords.shape[0] - 1, num=max(1, patches_per_patient - 1), dtype=int)
        for idx in idxs:
            centers.append(tuple(tumor_coords[idx]))

    # Always add center patch
    centers.append((shape[0] // 2, shape[1] // 2, shape[2] // 2))

    # Remove duplicates while preserving order
    unique_centers = []
    seen = set()
    for c in centers:
        c = tuple(int(v) for v in c)
        if c not in seen:
            unique_centers.append(c)
            seen.add(c)

    # Keep exactly patches_per_patient centers if possible
    return unique_centers[:patches_per_patient]


# =============================================================================
# Metrics
# =============================================================================

def dice_score(pred_mask, true_mask, eps=1e-8):
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)

    intersection = np.logical_and(pred_mask, true_mask).sum()
    denominator = pred_mask.sum() + true_mask.sum()

    if denominator == 0:
        return np.nan

    return (2.0 * intersection) / (denominator + eps)


def iou_score(pred_mask, true_mask, eps=1e-8):
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)

    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()

    if union == 0:
        return np.nan

    return intersection / (union + eps)


def compute_metrics(pred, true):
    """
    Compute whole tumor and class-wise Dice/IoU.
    """
    metrics = {}

    pred_tumor = pred > 0
    true_tumor = true > 0

    metrics["whole_tumor_dice"] = dice_score(pred_tumor, true_tumor)
    metrics["whole_tumor_iou"] = iou_score(pred_tumor, true_tumor)

    for cls in [1, 2, 3]:
        pred_cls = pred == cls
        true_cls = true == cls

        metrics[f"dice_class_{cls}"] = dice_score(pred_cls, true_cls)
        metrics[f"iou_class_{cls}"] = iou_score(pred_cls, true_cls)

    metrics["true_tumor_voxels"] = int(true_tumor.sum())
    metrics["pred_tumor_voxels"] = int(pred_tumor.sum())

    return metrics


# =============================================================================
# Visualization
# =============================================================================

def save_preview(preview_items, save_path):
    """
    Save a simple prediction preview figure.

    Columns:
    FLAIR slice, T1ce slice, ground truth tumor, predicted tumor
    """
    if len(preview_items) == 0:
        print("No preview items available.")
        return

    n = min(len(preview_items), MAX_PREVIEW_PATCHES)

    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))

    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx in range(n):
        item = preview_items[row_idx]

        image_patch = item["image_patch"]
        true_patch = item["true_patch"]
        pred_patch = item["pred_patch"]
        title = item["title"]

        # middle axial slice of the 3D patch
        z = image_patch.shape[-1] // 2

        flair_slice = image_patch[0, :, :, z]
        t1ce_slice = image_patch[2, :, :, z]
        true_slice = true_patch[:, :, z] > 0
        pred_slice = pred_patch[:, :, z] > 0

        axes[row_idx, 0].imshow(flair_slice.T, cmap="gray", origin="lower")
        axes[row_idx, 0].set_title(f"{title}\nFLAIR")

        axes[row_idx, 1].imshow(t1ce_slice.T, cmap="gray", origin="lower")
        axes[row_idx, 1].set_title("T1ce")

        axes[row_idx, 2].imshow(flair_slice.T, cmap="gray", origin="lower")
        axes[row_idx, 2].imshow(true_slice.T, alpha=0.45, origin="lower")
        axes[row_idx, 2].set_title("Ground truth tumor")

        axes[row_idx, 3].imshow(flair_slice.T, cmap="gray", origin="lower")
        axes[row_idx, 3].imshow(pred_slice.T, alpha=0.45, origin="lower")
        axes[row_idx, 3].set_title("Predicted tumor")

        for col_idx in range(4):
            axes[row_idx, col_idx].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved preview PNG: {save_path}")


# =============================================================================
# Main testing loop
# =============================================================================

def main():
    print("=" * 80)
    print("Script 17: Full clean 4-modal 3D U-Net clean test")
    print("=" * 80)

    print(f"Device: {DEVICE}")
    print(f"Test CSV: {TEST_CSV}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Patch size: {PATCH_SIZE}")
    print(f"Patches per patient: {PATCHES_PER_PATIENT}")

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

    model = UNet3D(
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

    all_rows = []
    preview_items = []

    with torch.no_grad():
        for patient_idx, row in test_df.iterrows():
            patient_id = row["patient_id"]

            print(f"\n[{patient_idx + 1}/{len(test_df)}] Testing patient: {patient_id}")

            flair = load_nifti(row["flair"])
            t1 = load_nifti(row["t1"])
            t1ce = load_nifti(row["t1ce"])
            t2 = load_nifti(row["t2"])
            seg = load_nifti(row["seg"])

            seg = remap_seg_labels(seg)

            raw_modalities = [flair, t1, t1ce, t2]

            bbox = get_brain_bbox(raw_modalities)

            cropped_modalities = []
            for vol in raw_modalities:
                vol_crop = crop_to_bbox(vol, bbox)
                vol_crop = normalize_nonzero(vol_crop)
                cropped_modalities.append(vol_crop)

            seg_crop = crop_to_bbox(seg, bbox)

            image_4ch = np.stack(cropped_modalities, axis=0).astype(np.float32)

            image_4ch = pad_to_patch_size(image_4ch, PATCH_SIZE)
            seg_crop = pad_to_patch_size(seg_crop, PATCH_SIZE).astype(np.int64)

            centers = choose_patch_centers(seg_crop, PATCH_SIZE, PATCHES_PER_PATIENT)

            for patch_idx, center in enumerate(centers):
                start = get_patch_start(center, seg_crop.shape, PATCH_SIZE)

                image_patch = extract_patch(image_4ch, start, PATCH_SIZE)
                seg_patch = extract_patch(seg_crop, start, PATCH_SIZE)

                x = torch.from_numpy(image_patch).unsqueeze(0).float().to(DEVICE)

                logits = model(x)
                pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)

                metrics = compute_metrics(pred, seg_patch)

                out_row = {
                    "patient_id": patient_id,
                    "patient_index": patient_idx,
                    "patch_index": patch_idx,
                    "patch_start_x": start[0],
                    "patch_start_y": start[1],
                    "patch_start_z": start[2],
                    **metrics,
                }

                all_rows.append(out_row)

                print(
                    f"  Patch {patch_idx}: "
                    f"WT Dice={metrics['whole_tumor_dice']:.4f}, "
                    f"WT IoU={metrics['whole_tumor_iou']:.4f}, "
                    f"true voxels={metrics['true_tumor_voxels']}, "
                    f"pred voxels={metrics['pred_tumor_voxels']}"
                )

                if len(preview_items) < MAX_PREVIEW_PATCHES:
                    preview_items.append({
                        "image_patch": image_patch,
                        "true_patch": seg_patch,
                        "pred_patch": pred,
                        "title": f"{patient_id} patch {patch_idx}",
                    })

    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(METRICS_CSV, index=False)

    print("\n" + "=" * 80)
    print("Patch-level metrics saved")
    print("=" * 80)
    print(f"Saved metrics CSV: {METRICS_CSV}")
    print(f"Total evaluated patches: {len(metrics_df)}")

    # Summary using nanmean because class-wise metrics may be NaN when class is absent
    summary = {
        "num_test_patients": len(test_df),
        "patches_per_patient": PATCHES_PER_PATIENT,
        "total_patches": len(metrics_df),

        "mean_whole_tumor_dice": np.nanmean(metrics_df["whole_tumor_dice"]),
        "std_whole_tumor_dice": np.nanstd(metrics_df["whole_tumor_dice"]),
        "mean_whole_tumor_iou": np.nanmean(metrics_df["whole_tumor_iou"]),
        "std_whole_tumor_iou": np.nanstd(metrics_df["whole_tumor_iou"]),

        "mean_dice_class_1": np.nanmean(metrics_df["dice_class_1"]),
        "mean_dice_class_2": np.nanmean(metrics_df["dice_class_2"]),
        "mean_dice_class_3": np.nanmean(metrics_df["dice_class_3"]),

        "mean_iou_class_1": np.nanmean(metrics_df["iou_class_1"]),
        "mean_iou_class_2": np.nanmean(metrics_df["iou_class_2"]),
        "mean_iou_class_3": np.nanmean(metrics_df["iou_class_3"]),

        "mean_true_tumor_voxels": np.nanmean(metrics_df["true_tumor_voxels"]),
        "mean_pred_tumor_voxels": np.nanmean(metrics_df["pred_tumor_voxels"]),
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print("\n" + "=" * 80)
    print("Clean held-out test summary")
    print("=" * 80)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print(f"\nSaved summary CSV: {SUMMARY_CSV}")

    save_preview(preview_items, PREVIEW_PNG)

    print("\n" + "=" * 80)
    print("Script 17 complete.")
    print("=" * 80)
    print("Important interpretation:")
    print("This is the clean 4-modal 3D U-Net held-out test baseline.")
    print("Do NOT start degradation testing until you inspect this output.")
    print("Next later: Script 18 will compare degraded test performance against this clean baseline.")


if __name__ == "__main__":
    main()
