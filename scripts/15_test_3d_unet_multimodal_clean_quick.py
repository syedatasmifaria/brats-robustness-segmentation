"""
Script 15: Quick clean test of 4-modal 3D U-Net

Goal:
- Load the quick 4-modal 3D U-Net model from Script 14.
- Test it on clean held-out BraTS2020 test patients.
- Inputs: FLAIR + T1 + T1ce + T2
- Output: predicted 4-class segmentation mask.
- Compute Dice and IoU.
- Save metrics CSV, summary CSV, and prediction preview PNG.

Important:
- This script does NOT train.
- This script does NOT apply degradation.
- This is clean testing only.
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
# 1. Settings
# ============================================================

PROJECT_ROOT = "/home/xfh25/brats_segmentation_project"

TEST_CSV = os.path.join(PROJECT_ROOT, "data/csvs/test_paths.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models/3d_unet_multimodal_clean_quick_best.pth")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

METRICS_CSV = os.path.join(RESULTS_DIR, "15_quick_multimodal_3d_clean_test_metrics.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "15_quick_multimodal_3d_clean_test_summary.csv")
PREVIEW_PNG = os.path.join(RESULTS_DIR, "15_quick_multimodal_3d_clean_test_predictions.png")

MODALITIES = ["flair", "t1", "t1ce", "t2"]

PATCH_SIZE = (96, 96, 96)

IN_CHANNELS = 4
OUT_CLASSES = 4
BASE_CHANNELS = 16

NUM_TEST_PATIENTS = 20
PATCHES_PER_PATIENT = 2

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
# 3. 3D U-Net model
# ============================================================

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
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_classes=4, base_channels=16):
        super().__init__()

        self.input_conv = DoubleConv3D(in_channels, base_channels)

        self.down1 = Down3D(base_channels, base_channels * 2)
        self.down2 = Down3D(base_channels * 2, base_channels * 4)
        self.down3 = Down3D(base_channels * 4, base_channels * 8)

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

        return self.output_conv(x)


# ============================================================
# 4. Data functions
# ============================================================

def load_nifti(path, dtype=np.float32):
    return np.asanyarray(nib.load(path).dataobj).astype(dtype)


def remap_segmentation_labels(seg):
    """
    BraTS labels:
    Original: 0, 1, 2, 4
    Model:    0, 1, 2, 3
    """
    seg = seg.astype(np.int64)
    seg[seg == 4] = 3
    return seg


def normalize_modality(volume):
    """
    Normalize one modality using nonzero brain voxels.
    Background remains zero.
    """
    volume = volume.astype(np.float32)

    brain_mask = volume > 0

    if brain_mask.sum() == 0:
        return volume

    mean = volume[brain_mask].mean()
    std = volume[brain_mask].std()

    if std < 1e-8:
        std = 1.0

    volume_norm = (volume - mean) / std
    volume_norm[~brain_mask] = 0

    return volume_norm.astype(np.float32)


def get_union_brain_bbox(modality_volumes):
    """
    Get one shared crop box using all modalities.
    This keeps FLAIR, T1, T1ce, T2, and SEG aligned.
    """
    brain_mask = np.zeros_like(modality_volumes[0], dtype=bool)

    for volume in modality_volumes:
        brain_mask = brain_mask | (volume > 0)

    coords = np.where(brain_mask)

    if len(coords[0]) == 0:
        return None

    x_min, x_max = coords[0].min(), coords[0].max() + 1
    y_min, y_max = coords[1].min(), coords[1].max() + 1
    z_min, z_max = coords[2].min(), coords[2].max() + 1

    return x_min, x_max, y_min, y_max, z_min, z_max


def crop_modalities_and_seg(modality_volumes, seg):
    bbox = get_union_brain_bbox(modality_volumes)

    if bbox is None:
        return modality_volumes, seg

    x_min, x_max, y_min, y_max, z_min, z_max = bbox

    cropped_modalities = [
        volume[x_min:x_max, y_min:y_max, z_min:z_max]
        for volume in modality_volumes
    ]

    seg_crop = seg[x_min:x_max, y_min:y_max, z_min:z_max]

    return cropped_modalities, seg_crop


def pad_3d_if_needed(volume, patch_size, pad_value=0):
    pad_width = []

    for dim_size, patch_dim in zip(volume.shape, patch_size):
        if dim_size >= patch_dim:
            pad_width.append((0, 0))
        else:
            total_pad = patch_dim - dim_size
            before = total_pad // 2
            after = total_pad - before
            pad_width.append((before, after))

    return np.pad(
        volume,
        pad_width=pad_width,
        mode="constant",
        constant_values=pad_value
    )


def sample_multimodal_patch(modality_volumes, seg, patch_size, force_tumor=True):
    """
    Sample the same 3D location from all four modalities and the segmentation mask.

    Image patch shape:
        (4, 96, 96, 96)

    Seg patch shape:
        (96, 96, 96)
    """
    px, py, pz = patch_size
    sx, sy, sz = seg.shape

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

    image_patch = np.stack(
        [
            volume[x_start:x_end, y_start:y_end, z_start:z_end]
            for volume in modality_volumes
        ],
        axis=0
    ).astype(np.float32)

    seg_patch = seg[x_start:x_end, y_start:y_end, z_start:z_end].astype(np.int64)

    return image_patch, seg_patch, (x_start, y_start, z_start)


# ============================================================
# 5. Metrics
# ============================================================

def dice_score_binary(pred, target, eps=1e-8):
    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(pred, target).sum()
    total = pred.sum() + target.sum()

    if total == 0:
        return np.nan

    return (2.0 * intersection + eps) / (total + eps)


def iou_score_binary(pred, target, eps=1e-8):
    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()

    if union == 0:
        return np.nan

    return (intersection + eps) / (union + eps)


def dice_score_for_class(pred, target, class_id, eps=1e-8):
    pred_class = pred == class_id
    target_class = target == class_id

    intersection = np.logical_and(pred_class, target_class).sum()
    total = pred_class.sum() + target_class.sum()

    if total == 0:
        return np.nan

    return (2.0 * intersection + eps) / (total + eps)


# ============================================================
# 6. Model loading and prediction
# ============================================================

def load_model(device):
    model = UNet3D(
        in_channels=IN_CHANNELS,
        out_classes=OUT_CLASSES,
        base_channels=BASE_CHANNELS
    ).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        saved_epoch = checkpoint.get("epoch", "unknown")
        best_val_loss = checkpoint.get("best_val_loss", "unknown")
    else:
        state_dict = checkpoint
        saved_epoch = "unknown"
        best_val_loss = "unknown"

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded checkpoint epoch: {saved_epoch}")
    print(f"Checkpoint best val loss: {best_val_loss}")

    return model


def predict_patch(model, image_patch, device):
    """
    image_patch shape:
        (4, 96, 96, 96)

    model input shape:
        (1, 4, 96, 96, 96)
    """
    x = torch.from_numpy(image_patch).float().unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1)

    pred_np = pred.squeeze(0).cpu().numpy().astype(np.int64)

    return pred_np


# ============================================================
# 7. Preview image
# ============================================================

def create_prediction_preview(preview_items, save_path):
    if len(preview_items) == 0:
        print("No preview items available.")
        return

    num_rows = min(len(preview_items), 4)

    fig, axes = plt.subplots(num_rows, 5, figsize=(18, 3.8 * num_rows))

    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx in range(num_rows):
        item = preview_items[row_idx]

        image_patch = item["image_patch"]
        seg_patch = item["seg_patch"]
        pred_patch = item["pred_patch"]
        patient_id = item["patient_id"]
        tumor_dice = item["tumor_dice"]

        tumor_per_slice = (seg_patch > 0).sum(axis=(0, 1))

        if tumor_per_slice.max() > 0:
            z = int(np.argmax(tumor_per_slice))
        else:
            z = seg_patch.shape[2] // 2

        flair_slice = image_patch[0, :, :, z]
        t1ce_slice = image_patch[2, :, :, z]
        seg_slice = seg_patch[:, :, z]
        pred_slice = pred_patch[:, :, z]

        axes[row_idx, 0].imshow(flair_slice, cmap="gray")
        axes[row_idx, 0].set_title(f"{patient_id}\nFLAIR")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(t1ce_slice, cmap="gray")
        axes[row_idx, 1].set_title("T1ce")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(seg_slice, cmap="viridis", vmin=0, vmax=3)
        axes[row_idx, 2].set_title("Ground truth")
        axes[row_idx, 2].axis("off")

        axes[row_idx, 3].imshow(pred_slice, cmap="viridis", vmin=0, vmax=3)
        axes[row_idx, 3].set_title("Prediction")
        axes[row_idx, 3].axis("off")

        axes[row_idx, 4].imshow(flair_slice, cmap="gray")
        axes[row_idx, 4].imshow(seg_slice > 0, alpha=0.35, cmap="Greens")
        axes[row_idx, 4].imshow(pred_slice > 0, alpha=0.35, cmap="Reds")
        axes[row_idx, 4].set_title(f"Overlay\nDice={tumor_dice:.4f}")
        axes[row_idx, 4].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ============================================================
# 8. Main
# ============================================================

def main():
    print("=" * 80)
    print("Script 15: Quick clean test of 4-modal 3D U-Net")
    print("=" * 80)

    print("\nChecking files...")

    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    print(f"Test CSV: {TEST_CSV}")
    print(f"Model:    {MODEL_PATH}")

    test_df = pd.read_csv(TEST_CSV)

    required_cols = ["patient_id", "flair", "t1", "t1ce", "t2", "seg"]

    for col in required_cols:
        if col not in test_df.columns:
            raise ValueError(
                f"Missing required column: {col}. "
                f"Available columns: {list(test_df.columns)}"
            )

    print(f"\nTotal test patients in CSV: {len(test_df)}")

    test_df = test_df.head(NUM_TEST_PATIENTS).copy()

    print(f"Patients used in quick clean test: {len(test_df)}")
    print(f"Patches per patient: {PATCHES_PER_PATIENT}")
    print(f"Modalities: {MODALITIES}")
    print(f"Patch size: {PATCH_SIZE}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nUsing device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading model...")
    model = load_model(device)
    print("Model loaded successfully.")

    metrics_rows = []
    preview_items = []

    for local_idx, (_, row) in enumerate(test_df.iterrows(), start=1):
        patient_id = row["patient_id"]

        print(f"\nTesting patient {local_idx}/{len(test_df)}: {patient_id}")

        paths = {
            "flair": row["flair"],
            "t1": row["t1"],
            "t1ce": row["t1ce"],
            "t2": row["t2"],
            "seg": row["seg"],
        }

        missing_files = []

        for key, path in paths.items():
            if not os.path.exists(path):
                missing_files.append(f"{key}: {path}")

        if len(missing_files) > 0:
            warnings.warn(
                f"Skipping {patient_id}; missing files:\n" +
                "\n".join(missing_files)
            )
            continue

        raw_modalities = [
            load_nifti(paths[modality], dtype=np.float32)
            for modality in MODALITIES
        ]

        seg = load_nifti(paths["seg"], dtype=np.int16)
        seg = remap_segmentation_labels(seg)

        original_shape = raw_modalities[0].shape

        cropped_modalities, seg_crop = crop_modalities_and_seg(raw_modalities, seg)

        normalized_modalities = [
            normalize_modality(volume)
            for volume in cropped_modalities
        ]

        padded_modalities = [
            pad_3d_if_needed(volume, PATCH_SIZE, pad_value=0)
            for volume in normalized_modalities
        ]

        seg_padded = pad_3d_if_needed(seg_crop, PATCH_SIZE, pad_value=0)

        cropped_padded_shape = seg_padded.shape

        for patch_idx in range(PATCHES_PER_PATIENT):
            # First patch is tumor-centered.
            # Second patch is random.
            force_tumor = patch_idx == 0

            image_patch, seg_patch, patch_start = sample_multimodal_patch(
                padded_modalities,
                seg_padded,
                PATCH_SIZE,
                force_tumor=force_tumor
            )

            pred_patch = predict_patch(model, image_patch, device)

            gt_tumor = seg_patch > 0
            pred_tumor = pred_patch > 0

            tumor_dice = dice_score_binary(pred_tumor, gt_tumor)
            tumor_iou = iou_score_binary(pred_tumor, gt_tumor)

            dice_class_1 = dice_score_for_class(pred_patch, seg_patch, class_id=1)
            dice_class_2 = dice_score_for_class(pred_patch, seg_patch, class_id=2)
            dice_class_3 = dice_score_for_class(pred_patch, seg_patch, class_id=3)

            gt_tumor_voxels = int(gt_tumor.sum())
            pred_tumor_voxels = int(pred_tumor.sum())

            unique_pred_labels = sorted(np.unique(pred_patch).astype(int).tolist())
            unique_gt_labels = sorted(np.unique(seg_patch).astype(int).tolist())

            metrics_rows.append({
                "patient_id": patient_id,
                "patch_index": patch_idx,
                "force_tumor_patch": force_tumor,
                "modalities": "+".join(MODALITIES),
                "original_shape": str(original_shape),
                "cropped_padded_shape": str(cropped_padded_shape),
                "patch_size": str(PATCH_SIZE),
                "patch_start_x": patch_start[0],
                "patch_start_y": patch_start[1],
                "patch_start_z": patch_start[2],
                "gt_unique_labels": str(unique_gt_labels),
                "pred_unique_labels": str(unique_pred_labels),
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
                f"GT voxels={gt_tumor_voxels}, "
                f"Pred voxels={pred_tumor_voxels}, "
                f"Dice={tumor_dice:.4f}, "
                f"IoU={tumor_iou:.4f}, "
                f"Pred labels={unique_pred_labels}"
            )

            if len(preview_items) < 4 and gt_tumor_voxels > 0:
                preview_items.append({
                    "patient_id": patient_id,
                    "image_patch": image_patch,
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
            "modalities": "+".join(MODALITIES),
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
            "modalities": "+".join(MODALITIES),
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
