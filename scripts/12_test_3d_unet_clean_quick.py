"""
Script 12: Quick clean test of the 3D U-Net model

Goal:
- Load the quick 3D U-Net model trained in Script 11.
- Sample clean 3D patches from held-out test patients.
- Predict segmentation masks.
- Compute Dice and IoU.
- Save:
    1. Detailed patch-level metrics CSV
    2. Summary CSV
    3. Prediction preview PNG

Important:
- This script does NOT train.
- This script does NOT apply degradation.
- This script tests the quick clean 3D model on clean held-out test patches only.
"""

import os
import random
import warnings

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Paths and settings
# ============================================================

PROJECT_ROOT = "/home/xfh25/brats_segmentation_project"

TEST_CSV = os.path.join(PROJECT_ROOT, "data/csvs/test_paths.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models/3d_unet_flair_clean_quick.pth")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

METRICS_CSV = os.path.join(RESULTS_DIR, "12_quick_3d_clean_test_metrics.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "12_quick_3d_clean_test_summary.csv")
PREVIEW_PNG = os.path.join(RESULTS_DIR, "12_quick_3d_clean_test_predictions.png")

PATCH_SIZE = (96, 96, 96)

# Quick test only, not full evaluation
NUM_TEST_PATIENTS = 20
PATCHES_PER_PATIENT = 2

# Must match Script 11
IN_CHANNELS = 1
OUT_CLASSES = 4
BASE_CHANNELS = 16

RANDOM_SEED = 42


# ============================================================
# 2. Reproducibility
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# 3. 3D U-Net model definition
# ============================================================
# This version matches the Script 11 checkpoint names:
# input_conv, down1, down2, down3, up3, up2, up1, output_conv

class DoubleConv3D(nn.Module):
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


class Down3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)


class Up3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv3D(
            out_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):
        x = self.up(x)

        # Usually not needed for 96x96x96 patches because dimensions divide cleanly.
        # Kept here as safety in case shapes differ by 1 voxel.
        diff_x = skip.size(2) - x.size(2)
        diff_y = skip.size(3) - x.size(3)
        diff_z = skip.size(4) - x.size(4)

        if diff_x != 0 or diff_y != 0 or diff_z != 0:
            x = F.pad(
                x,
                [
                    diff_z // 2, diff_z - diff_z // 2,
                    diff_y // 2, diff_y - diff_y // 2,
                    diff_x // 2, diff_x - diff_x // 2,
                ]
            )

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)

        return x


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_classes=4, base_channels=16):
        super().__init__()

        self.input_conv = DoubleConv3D(in_channels, base_channels)

        self.down1 = Down3D(base_channels, base_channels * 2)
        self.down2 = Down3D(base_channels * 2, base_channels * 4)
        self.down3 = Down3D(base_channels * 4, base_channels * 8)

        # Important:
        # Script 11 used up1 as the deepest decoder block,
        # up2 as the middle decoder block,
        # and up3 as the final decoder block.
        self.up1 = Up3D(
            in_channels=base_channels * 8,
            skip_channels=base_channels * 4,
            out_channels=base_channels * 4
        )

        self.up2 = Up3D(
            in_channels=base_channels * 4,
            skip_channels=base_channels * 2,
            out_channels=base_channels * 2
        )

        self.up3 = Up3D(
            in_channels=base_channels * 2,
            skip_channels=base_channels,
            out_channels=base_channels
        )

        self.output_conv = nn.Conv3d(
            base_channels,
            out_classes,
            kernel_size=1
        )

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
# 4. Helper functions
# ============================================================

def load_nifti(path):
    """Load a NIfTI file as a numpy array."""
    return nib.load(path).get_fdata()


def remap_segmentation_labels(seg):
    """
    BraTS original labels:
        0 = background
        1 = necrotic/non-enhancing tumor
        2 = edema
        4 = enhancing tumor

    Model labels:
        0, 1, 2, 3

    So original label 4 becomes 3.
    """
    seg = seg.astype(np.int64)
    seg_remap = seg.copy()
    seg_remap[seg_remap == 4] = 3
    return seg_remap


def normalize_flair(flair):
    """
    Normalize FLAIR image using nonzero brain voxels.
    Background stays 0.
    """
    flair = flair.astype(np.float32)

    brain_mask = flair > 0

    if brain_mask.sum() == 0:
        return flair

    mean = flair[brain_mask].mean()
    std = flair[brain_mask].std()

    if std < 1e-8:
        std = 1.0

    flair_norm = (flair - mean) / std
    flair_norm[~brain_mask] = 0

    return flair_norm.astype(np.float32)


def get_brain_bbox(flair):
    """
    Find bounding box around the nonzero brain area.
    """
    brain_mask = flair > 0
    coords = np.where(brain_mask)

    if len(coords[0]) == 0:
        return None

    x_min, x_max = coords[0].min(), coords[0].max() + 1
    y_min, y_max = coords[1].min(), coords[1].max() + 1
    z_min, z_max = coords[2].min(), coords[2].max() + 1

    return x_min, x_max, y_min, y_max, z_min, z_max


def crop_to_brain(flair, seg):
    """
    Crop FLAIR and segmentation around the brain.
    """
    bbox = get_brain_bbox(flair)

    if bbox is None:
        return flair, seg

    x_min, x_max, y_min, y_max, z_min, z_max = bbox

    flair_crop = flair[x_min:x_max, y_min:y_max, z_min:z_max]
    seg_crop = seg[x_min:x_max, y_min:y_max, z_min:z_max]

    return flair_crop, seg_crop


def pad_if_needed(volume, patch_size, pad_value=0):
    """
    If cropped volume is smaller than patch size, pad it.
    """
    pad_width = []

    for dim_size, patch_dim in zip(volume.shape, patch_size):
        if dim_size >= patch_dim:
            pad_width.append((0, 0))
        else:
            total_pad = patch_dim - dim_size
            before = total_pad // 2
            after = total_pad - before
            pad_width.append((before, after))

    padded = np.pad(
        volume,
        pad_width=pad_width,
        mode="constant",
        constant_values=pad_value
    )

    return padded


def sample_patch(flair, seg, patch_size, force_tumor=True):
    """
    Sample one 3D patch.

    If force_tumor=True and tumor exists, center patch near a tumor voxel.
    This prevents the test from being mostly empty background.
    """
    px, py, pz = patch_size
    sx, sy, sz = flair.shape

    if force_tumor and np.any(seg > 0):
        tumor_coords = np.array(np.where(seg > 0)).T
        center = tumor_coords[np.random.choice(len(tumor_coords))]
        cx, cy, cz = center

        x_start = int(cx - px // 2)
        y_start = int(cy - py // 2)
        z_start = int(cz - pz // 2)
    else:
        x_start = np.random.randint(0, sx - px + 1)
        y_start = np.random.randint(0, sy - py + 1)
        z_start = np.random.randint(0, sz - pz + 1)

    x_start = max(0, min(x_start, sx - px))
    y_start = max(0, min(y_start, sy - py))
    z_start = max(0, min(z_start, sz - pz))

    x_end = x_start + px
    y_end = y_start + py
    z_end = z_start + pz

    flair_patch = flair[x_start:x_end, y_start:y_end, z_start:z_end]
    seg_patch = seg[x_start:x_end, y_start:y_end, z_start:z_end]

    return flair_patch, seg_patch, (x_start, y_start, z_start)


def dice_score_binary(pred, target, eps=1e-8):
    """
    Binary Dice score.
    Used for whole tumor:
        tumor = labels 1, 2, or 3
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(pred, target).sum()
    total = pred.sum() + target.sum()

    if total == 0:
        return np.nan

    return (2.0 * intersection + eps) / (total + eps)


def iou_score_binary(pred, target, eps=1e-8):
    """
    Binary IoU score.
    Used for whole tumor:
        tumor = labels 1, 2, or 3
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()

    if union == 0:
        return np.nan

    return (intersection + eps) / (union + eps)


def dice_score_for_class(pred, target, class_id, eps=1e-8):
    """
    Dice score for one class.
    """
    pred_class = pred == class_id
    target_class = target == class_id

    intersection = np.logical_and(pred_class, target_class).sum()
    total = pred_class.sum() + target_class.sum()

    if total == 0:
        return np.nan

    return (2.0 * intersection + eps) / (total + eps)


def load_model(device):
    """
    Load trained 3D U-Net model from Script 11.
    """
    model = UNet3D(
        in_channels=IN_CHANNELS,
        out_classes=OUT_CLASSES,
        base_channels=BASE_CHANNELS
    ).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # Supports either direct state_dict or a dictionary containing model_state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print("\nERROR: The model architecture in Script 12 still does not match the saved checkpoint.")
        print("This means Script 11 used a slightly different 3D U-Net structure.")
        print("\nPyTorch error:")
        print(e)
        print("\nFirst few checkpoint keys:")
        for i, key in enumerate(state_dict.keys()):
            print("  ", key)
            if i >= 20:
                break
        raise

    model.eval()
    return model


def predict_patch(model, flair_patch, device):
    """
    Run prediction on one 3D patch.
    """
    x = torch.from_numpy(flair_patch).float()

    # Input shape for Conv3D:
    # (batch, channel, depth, height, width)
    x = x.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1)

    pred_np = pred.squeeze(0).cpu().numpy().astype(np.int64)

    return pred_np


def create_prediction_preview(preview_items, save_path):
    """
    Save prediction preview image.

    Each row:
    1. FLAIR slice
    2. Ground truth mask
    3. Predicted mask
    4. Overlay
    """
    if len(preview_items) == 0:
        print("No preview items available.")
        return

    num_rows = min(len(preview_items), 4)
    fig, axes = plt.subplots(num_rows, 4, figsize=(14, 3.5 * num_rows))

    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx in range(num_rows):
        item = preview_items[row_idx]

        flair_patch = item["flair_patch"]
        seg_patch = item["seg_patch"]
        pred_patch = item["pred_patch"]
        patient_id = item["patient_id"]
        tumor_dice = item["tumor_dice"]

        tumor_per_slice = (seg_patch > 0).sum(axis=(0, 1))

        if tumor_per_slice.max() > 0:
            z = int(np.argmax(tumor_per_slice))
        else:
            z = seg_patch.shape[2] // 2

        flair_slice = flair_patch[:, :, z]
        seg_slice = seg_patch[:, :, z]
        pred_slice = pred_patch[:, :, z]

        axes[row_idx, 0].imshow(flair_slice, cmap="gray")
        axes[row_idx, 0].set_title(f"{patient_id}\nFLAIR slice")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(seg_slice, cmap="viridis", vmin=0, vmax=3)
        axes[row_idx, 1].set_title("Ground truth")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(pred_slice, cmap="viridis", vmin=0, vmax=3)
        axes[row_idx, 2].set_title("Prediction")
        axes[row_idx, 2].axis("off")

        axes[row_idx, 3].imshow(flair_slice, cmap="gray")
        axes[row_idx, 3].imshow(seg_slice > 0, alpha=0.35, cmap="Greens")
        axes[row_idx, 3].imshow(pred_slice > 0, alpha=0.35, cmap="Reds")
        axes[row_idx, 3].set_title(f"Overlay\nDice={tumor_dice:.4f}")
        axes[row_idx, 3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ============================================================
# 5. Main testing function
# ============================================================

def main():
    print("=" * 80)
    print("Script 12: Quick clean test of 3D U-Net")
    print("=" * 80)

    print("\nChecking files...")

    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    print(f"Test CSV: {TEST_CSV}")
    print(f"Model:    {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading model...")
    model = load_model(device)
    print("Model loaded successfully.")

    test_df = pd.read_csv(TEST_CSV)

    required_cols = ["patient_id", "flair", "seg"]
    for col in required_cols:
        if col not in test_df.columns:
            raise ValueError(
                f"Column '{col}' not found in test CSV. "
                f"Available columns: {list(test_df.columns)}"
            )

    print(f"\nTotal test patients in CSV: {len(test_df)}")

    test_df = test_df.head(NUM_TEST_PATIENTS).copy()

    print(f"Patients used in this quick test: {len(test_df)}")
    print(f"Patches per patient: {PATCHES_PER_PATIENT}")

    metrics_rows = []
    preview_items = []

    for local_patient_idx, (_, row) in enumerate(test_df.iterrows(), start=1):
        patient_id = row["patient_id"]
        flair_path = row["flair"]
        seg_path = row["seg"]

        print(f"\nTesting patient {local_patient_idx}/{len(test_df)}: {patient_id}")

        if not os.path.exists(flair_path):
            warnings.warn(f"FLAIR file missing for {patient_id}: {flair_path}")
            continue

        if not os.path.exists(seg_path):
            warnings.warn(f"SEG file missing for {patient_id}: {seg_path}")
            continue

        flair = load_nifti(flair_path)
        seg = load_nifti(seg_path)

        seg = remap_segmentation_labels(seg)

        original_shape = flair.shape

        flair_crop, seg_crop = crop_to_brain(flair, seg)

        flair_crop = normalize_flair(flair_crop)

        flair_crop = pad_if_needed(flair_crop, PATCH_SIZE, pad_value=0)
        seg_crop = pad_if_needed(seg_crop, PATCH_SIZE, pad_value=0)

        cropped_shape = flair_crop.shape

        for patch_idx in range(PATCHES_PER_PATIENT):
            force_tumor = patch_idx == 0

            flair_patch, seg_patch, patch_start = sample_patch(
                flair_crop,
                seg_crop,
                PATCH_SIZE,
                force_tumor=force_tumor
            )

            pred_patch = predict_patch(model, flair_patch, device)

            gt_tumor = seg_patch > 0
            pred_tumor = pred_patch > 0

            tumor_dice = dice_score_binary(pred_tumor, gt_tumor)
            tumor_iou = iou_score_binary(pred_tumor, gt_tumor)

            dice_class_1 = dice_score_for_class(pred_patch, seg_patch, class_id=1)
            dice_class_2 = dice_score_for_class(pred_patch, seg_patch, class_id=2)
            dice_class_3 = dice_score_for_class(pred_patch, seg_patch, class_id=3)

            gt_tumor_voxels = int(gt_tumor.sum())
            pred_tumor_voxels = int(pred_tumor.sum())

            metrics_rows.append({
                "patient_id": patient_id,
                "patch_index": patch_idx,
                "force_tumor_patch": force_tumor,
                "original_shape": str(original_shape),
                "cropped_padded_shape": str(cropped_shape),
                "patch_size": str(PATCH_SIZE),
                "patch_start_x": patch_start[0],
                "patch_start_y": patch_start[1],
                "patch_start_z": patch_start[2],
                "gt_tumor_voxels": gt_tumor_voxels,
                "pred_tumor_voxels": pred_tumor_voxels,
                "tumor_dice": tumor_dice,
                "tumor_iou": tumor_iou,
                "dice_class_1": dice_class_1,
                "dice_class_2": dice_class_2,
                "dice_class_3": dice_class_3,
            })

            print(
                f"  Patch {patch_idx}: "
                f"GT tumor voxels={gt_tumor_voxels}, "
                f"Pred tumor voxels={pred_tumor_voxels}, "
                f"Tumor Dice={tumor_dice:.4f}, "
                f"Tumor IoU={tumor_iou:.4f}"
            )

            if len(preview_items) < 4 and gt_tumor_voxels > 0:
                preview_items.append({
                    "patient_id": patient_id,
                    "flair_patch": flair_patch,
                    "seg_patch": seg_patch,
                    "pred_patch": pred_patch,
                    "tumor_dice": tumor_dice,
                })

    print("\nSaving results...")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(METRICS_CSV, index=False)

    if len(metrics_df) > 0:
        summary = {
            "model_path": MODEL_PATH,
            "test_csv": TEST_CSV,
            "num_test_patients_used": len(test_df),
            "patches_per_patient": PATCHES_PER_PATIENT,
            "total_patches_tested": len(metrics_df),
            "patch_size": str(PATCH_SIZE),
            "mean_tumor_dice": np.nanmean(metrics_df["tumor_dice"]),
            "std_tumor_dice": np.nanstd(metrics_df["tumor_dice"]),
            "mean_tumor_iou": np.nanmean(metrics_df["tumor_iou"]),
            "std_tumor_iou": np.nanstd(metrics_df["tumor_iou"]),
            "mean_dice_class_1": np.nanmean(metrics_df["dice_class_1"]),
            "mean_dice_class_2": np.nanmean(metrics_df["dice_class_2"]),
            "mean_dice_class_3": np.nanmean(metrics_df["dice_class_3"]),
        }
    else:
        summary = {
            "model_path": MODEL_PATH,
            "test_csv": TEST_CSV,
            "num_test_patients_used": len(test_df),
            "patches_per_patient": PATCHES_PER_PATIENT,
            "total_patches_tested": 0,
            "patch_size": str(PATCH_SIZE),
            "mean_tumor_dice": np.nan,
            "std_tumor_dice": np.nan,
            "mean_tumor_iou": np.nan,
            "std_tumor_iou": np.nan,
            "mean_dice_class_1": np.nan,
            "mean_dice_class_2": np.nan,
            "mean_dice_class_3": np.nan,
        }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(SUMMARY_CSV, index=False)

    create_prediction_preview(preview_items, PREVIEW_PNG)

    print(f"Detailed metrics saved to: {METRICS_CSV}")
    print(f"Summary saved to:          {SUMMARY_CSV}")
    print(f"Preview image saved to:    {PREVIEW_PNG}")

    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\nDone.")
    print("=" * 80)


if __name__ == "__main__":
    main()