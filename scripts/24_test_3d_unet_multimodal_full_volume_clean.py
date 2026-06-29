#!/usr/bin/env python3
"""
Script 24: Full-volume clean evaluation for custom 4-modal 3D U-Net.

Purpose:
- Load the already trained clean custom 3D U-Net.
- Evaluate it on full BraTS2020 test volumes using sliding-window inference.
- Reconstruct a full-volume prediction for each patient.
- Compute full-volume Dice and IoU.

Important:
- This is TESTING ONLY.
- No retraining.
- No degradation in this script.
- Model was trained on clean patches; this script tests clean full volumes.

Input:
- FLAIR + T1 + T1ce + T2

Output:
- results/24_full_volume_clean_test_metrics.csv
- results/24_full_volume_clean_test_summary.csv
- report_materials/24_full_volume_clean_test_summary.csv
- report_materials/24_full_volume_clean_test_summary.txt
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Paths and settings
# ============================================================

PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

TEST_CSV = PROJECT_ROOT / "data/csvs/test_paths.csv"

MODEL_PATH = PROJECT_ROOT / "models/3d_unet_multimodal_clean_full_best.pth"

RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_DIR = PROJECT_ROOT / "report_materials"

RESULTS_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

OUT_METRICS = RESULTS_DIR / "24_full_volume_clean_test_metrics.csv"
OUT_SUMMARY = RESULTS_DIR / "24_full_volume_clean_test_summary.csv"

REPORT_SUMMARY_CSV = REPORT_DIR / "24_full_volume_clean_test_summary.csv"
REPORT_SUMMARY_TXT = REPORT_DIR / "24_full_volume_clean_test_summary.txt"

MODALITIES = ["flair", "t1", "t1ce", "t2"]

PATCH_SIZE = (96, 96, 96)

# Stride controls overlap.
# Smaller stride = more overlap and smoother prediction, but slower.
# 64 is a reasonable balance for this project.
STRIDE = (64, 64, 64)

NUM_CLASSES = 4
IN_CHANNELS = 4
BASE_CHANNELS = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use None for all 74 test patients.
# For a quick smoke test, temporarily set this to 2.
MAX_TEST_PATIENTS = None


# ============================================================
# Model architecture
#
# IMPORTANT:
# This must match the checkpoint from Script 16.
#
# The checkpoint expects:
# - input_conv.block...
# - down1.block...
# - up1.conv.block...
# - InstanceNorm3d with affine=True
#
# Do NOT rename block to net.
# Do NOT remove affine=True.
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # Safety padding in case dimensions differ by 1 voxel.
        diff_d = skip.size(2) - x.size(2)
        diff_h = skip.size(3) - x.size(3)
        diff_w = skip.size(4) - x.size(4)

        if diff_d != 0 or diff_h != 0 or diff_w != 0:
            x = F.pad(
                x,
                [
                    diff_w // 2, diff_w - diff_w // 2,
                    diff_h // 2, diff_h - diff_h // 2,
                    diff_d // 2, diff_d - diff_d // 2,
                ],
            )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=4, num_classes=4, base_channels=16):
        super().__init__()

        self.input_conv = DoubleConv(in_channels, base_channels)

        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)

        # up1 is deepest, up2 is middle, up3 is final
        self.up1 = Up(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = Up(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up3 = Up(base_channels * 2, base_channels, base_channels)

        self.output_conv = nn.Conv3d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.input_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        logits = self.output_conv(x)
        return logits


# ============================================================
# Data loading and preprocessing
# ============================================================

def load_nifti(path):
    return nib.load(str(path)).get_fdata().astype(np.float32)


def remap_segmentation(seg):
    """
    BraTS original labels are [0, 1, 2, 4].
    Model labels are [0, 1, 2, 3], where original label 4 becomes class 3.
    """
    seg = seg.astype(np.int16)
    seg_remap = seg.copy()
    seg_remap[seg_remap == 4] = 3
    return seg_remap


def normalize_modality(volume):
    """
    Normalize one MRI modality using nonzero brain voxels.

    Steps:
    - find nonzero brain region
    - clip extreme values using 1st and 99th percentiles
    - scale to [0, 1]
    - keep background as 0
    """
    volume = volume.astype(np.float32)
    brain_mask = volume > 0

    if brain_mask.sum() == 0:
        return volume.astype(np.float32)

    brain_values = volume[brain_mask]

    low = np.percentile(brain_values, 1)
    high = np.percentile(brain_values, 99)

    volume = np.clip(volume, low, high)

    denom = high - low
    if denom < 1e-8:
        return np.zeros_like(volume, dtype=np.float32)

    volume = (volume - low) / denom
    volume = np.clip(volume, 0.0, 1.0)

    volume[~brain_mask] = 0.0

    return volume.astype(np.float32)


def load_patient_4modal(row):
    """
    Load FLAIR, T1, T1ce, T2 and segmentation.

    Uses correct CSV columns:
    row["flair"], row["t1"], row["t1ce"], row["t2"], row["seg"]
    """
    channels = []

    for modality in MODALITIES:
        vol = load_nifti(row[modality])
        vol = normalize_modality(vol)
        channels.append(vol)

    image = np.stack(channels, axis=0).astype(np.float32)

    seg = load_nifti(row["seg"])
    seg = remap_segmentation(seg)

    return image, seg


# ============================================================
# Sliding-window helpers
# ============================================================

def get_start_positions(dim_size, patch_size, stride):
    """
    Generate sliding-window start positions that cover the full dimension.

    Example:
    dim_size=155, patch_size=96, stride=64
    positions -> [0, 59]
    because the last patch must end exactly at 155.
    """
    if dim_size <= patch_size:
        return [0]

    positions = list(range(0, dim_size - patch_size + 1, stride))

    last_start = dim_size - patch_size

    if positions[-1] != last_start:
        positions.append(last_start)

    return positions


def sliding_window_predict(model, image):
    """
    Full-volume sliding-window prediction.

    image shape: (4, H, W, D)

    Returns:
    pred shape: (H, W, D), integer class prediction per voxel
    """
    model.eval()

    _, h, w, d = image.shape
    ph, pw, pd = PATCH_SIZE
    sh, sw, sd = STRIDE

    h_starts = get_start_positions(h, ph, sh)
    w_starts = get_start_positions(w, pw, sw)
    d_starts = get_start_positions(d, pd, sd)

    total_patches = len(h_starts) * len(w_starts) * len(d_starts)

    print(f"  Volume shape: {(h, w, d)}")
    print(f"  H starts: {h_starts}")
    print(f"  W starts: {w_starts}")
    print(f"  D starts: {d_starts}")
    print(f"  Total sliding-window patches: {total_patches}")

    prob_accum = np.zeros((NUM_CLASSES, h, w, d), dtype=np.float32)
    count_accum = np.zeros((h, w, d), dtype=np.float32)

    patch_counter = 0

    with torch.no_grad():
        for hs in h_starts:
            for ws in w_starts:
                for ds in d_starts:
                    he = hs + ph
                    we = ws + pw
                    de = ds + pd

                    patch = image[:, hs:he, ws:we, ds:de]

                    if patch.shape != (IN_CHANNELS, ph, pw, pd):
                        raise ValueError(f"Bad patch shape: {patch.shape}")

                    x = torch.from_numpy(patch).unsqueeze(0).float().to(DEVICE)

                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)

                    probs_np = probs.squeeze(0).cpu().numpy().astype(np.float32)

                    prob_accum[:, hs:he, ws:we, ds:de] += probs_np
                    count_accum[hs:he, ws:we, ds:de] += 1.0

                    patch_counter += 1

                    if patch_counter % 10 == 0 or patch_counter == total_patches:
                        print(f"    Predicted patches: {patch_counter}/{total_patches}")

    if np.any(count_accum == 0):
        raise RuntimeError("Some voxels were not covered by sliding-window inference.")

    prob_accum = prob_accum / count_accum[None, :, :, :]

    pred = np.argmax(prob_accum, axis=0).astype(np.int16)

    return pred


# ============================================================
# Metrics
# ============================================================

def dice_binary(pred_mask, true_mask, eps=1e-8):
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)

    intersection = np.logical_and(pred_mask, true_mask).sum()
    denom = pred_mask.sum() + true_mask.sum()

    if denom == 0:
        return 1.0

    return float((2.0 * intersection + eps) / (denom + eps))


def iou_binary(pred_mask, true_mask, eps=1e-8):
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)

    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()

    if union == 0:
        return 1.0

    return float((intersection + eps) / (union + eps))


def compute_patient_metrics(pred, true):
    """
    Compute whole tumor and class-wise Dice/IoU.

    Whole tumor:
    any foreground class > 0
    """
    metrics = {}

    pred_tumor = pred > 0
    true_tumor = true > 0

    metrics["whole_tumor_dice"] = dice_binary(pred_tumor, true_tumor)
    metrics["whole_tumor_iou"] = iou_binary(pred_tumor, true_tumor)

    for cls in [1, 2, 3]:
        pred_cls = pred == cls
        true_cls = true == cls

        metrics[f"dice_class_{cls}"] = dice_binary(pred_cls, true_cls)
        metrics[f"iou_class_{cls}"] = iou_binary(pred_cls, true_cls)

    metrics["true_tumor_voxels"] = int(true_tumor.sum())
    metrics["pred_tumor_voxels"] = int(pred_tumor.sum())

    return metrics


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("Script 24: Full-volume clean evaluation for custom 4-modal 3D U-Net")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Patch size: {PATCH_SIZE}")
    print(f"Stride: {STRIDE}")
    print("No retraining. Testing only.")
    print("=" * 80)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    test_df = pd.read_csv(TEST_CSV)

    if MAX_TEST_PATIENTS is not None:
        test_df = test_df.head(MAX_TEST_PATIENTS).copy()

    print(f"Number of test patients: {len(test_df)}")

    model = UNet3D(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        base_channels=BASE_CHANNELS,
    ).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    # Some scripts save plain state_dict; others may save {"model_state_dict": ...}
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("Model loaded successfully.")
    print()

    all_rows = []

    start_all = time.time()

    for idx, row in test_df.iterrows():
        patient_start = time.time()

        patient_id = row["patient_id"]

        print("-" * 80)
        print(f"[{idx + 1}/{len(test_df)}] Patient: {patient_id}")

        image, true_seg = load_patient_4modal(row)

        pred_seg = sliding_window_predict(model, image)

        metrics = compute_patient_metrics(pred_seg, true_seg)

        elapsed = time.time() - patient_start

        row_out = {
            "patient_id": patient_id,
            "condition": "clean",
            "elapsed_seconds": elapsed,
            **metrics,
        }

        all_rows.append(row_out)

        print(
            f"  Whole tumor Dice: {metrics['whole_tumor_dice']:.4f} | "
            f"IoU: {metrics['whole_tumor_iou']:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(OUT_METRICS, index=False)

    summary = {
        "num_test_patients": len(metrics_df),
        "patch_size": str(PATCH_SIZE),
        "stride": str(STRIDE),
        "condition": "clean",
        "mean_whole_tumor_dice": metrics_df["whole_tumor_dice"].mean(),
        "std_whole_tumor_dice": metrics_df["whole_tumor_dice"].std(),
        "mean_whole_tumor_iou": metrics_df["whole_tumor_iou"].mean(),
        "std_whole_tumor_iou": metrics_df["whole_tumor_iou"].std(),
        "mean_dice_class_1": metrics_df["dice_class_1"].mean(),
        "mean_dice_class_2": metrics_df["dice_class_2"].mean(),
        "mean_dice_class_3": metrics_df["dice_class_3"].mean(),
        "mean_iou_class_1": metrics_df["iou_class_1"].mean(),
        "mean_iou_class_2": metrics_df["iou_class_2"].mean(),
        "mean_iou_class_3": metrics_df["iou_class_3"].mean(),
        "mean_true_tumor_voxels": metrics_df["true_tumor_voxels"].mean(),
        "mean_pred_tumor_voxels": metrics_df["pred_tumor_voxels"].mean(),
        "total_elapsed_minutes": (time.time() - start_all) / 60.0,
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(OUT_SUMMARY, index=False)
    summary_df.to_csv(REPORT_SUMMARY_CSV, index=False)

    with open(REPORT_SUMMARY_TXT, "w") as f:
        f.write("Script 24: Full-volume clean evaluation for custom 4-modal 3D U-Net\n")
        f.write("=" * 80 + "\n\n")

        f.write("Purpose:\n")
        f.write(
            "Evaluate the already trained custom 3D U-Net on full BraTS2020 test "
            "volumes using sliding-window inference.\n\n"
        )

        f.write("Important:\n")
        f.write("- Testing only; no retraining.\n")
        f.write("- Clean images only in this script.\n")
        f.write("- Full-volume prediction reconstructed from overlapping 96x96x96 patches.\n\n")

        f.write("Model:\n")
        f.write(str(MODEL_PATH) + "\n\n")

        f.write("Settings:\n")
        f.write(f"Patch size: {PATCH_SIZE}\n")
        f.write(f"Stride: {STRIDE}\n")
        f.write(f"Test patients: {len(metrics_df)}\n\n")

        f.write("Main full-volume clean results:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean whole tumor Dice: {summary['mean_whole_tumor_dice']:.6f}\n")
        f.write(f"Std whole tumor Dice:  {summary['std_whole_tumor_dice']:.6f}\n")
        f.write(f"Mean whole tumor IoU:  {summary['mean_whole_tumor_iou']:.6f}\n")
        f.write(f"Std whole tumor IoU:   {summary['std_whole_tumor_iou']:.6f}\n\n")

        f.write("Class-wise results:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean Dice class 1: {summary['mean_dice_class_1']:.6f}\n")
        f.write(f"Mean Dice class 2: {summary['mean_dice_class_2']:.6f}\n")
        f.write(f"Mean Dice class 3: {summary['mean_dice_class_3']:.6f}\n")
        f.write(f"Mean IoU class 1:  {summary['mean_iou_class_1']:.6f}\n")
        f.write(f"Mean IoU class 2:  {summary['mean_iou_class_2']:.6f}\n")
        f.write(f"Mean IoU class 3:  {summary['mean_iou_class_3']:.6f}\n\n")

        f.write("Interpretation note:\n")
        f.write(
            "This result is more comparable to nnU-Net-style validation than the "
            "earlier tumor-centered patch result, because this script reconstructs "
            "a full-volume prediction using sliding-window inference. However, "
            "exact direct comparison with nnU-Net still requires attention to "
            "split differences and evaluation protocol differences.\n"
        )

    print()
    print("=" * 80)
    print("Saved:")
    print(f"  {OUT_METRICS}")
    print(f"  {OUT_SUMMARY}")
    print(f"  {REPORT_SUMMARY_CSV}")
    print(f"  {REPORT_SUMMARY_TXT}")
    print("=" * 80)

    print()
    print("Final full-volume clean summary:")
    print(f"  Mean whole tumor Dice: {summary['mean_whole_tumor_dice']:.6f}")
    print(f"  Mean whole tumor IoU:  {summary['mean_whole_tumor_iou']:.6f}")
    print(f"  Total elapsed minutes: {summary['total_elapsed_minutes']:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
