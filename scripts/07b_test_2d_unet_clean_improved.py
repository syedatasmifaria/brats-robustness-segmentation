# ============================================================
# Script 07b: Test improved clean 2D U-Net on clean test slices
# Project: Robustness of Medical Image Segmentation Models
# ============================================================

from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ------------------------------------------------------------
# 1. Basic settings
# ------------------------------------------------------------

PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

TEST_CSV = PROJECT_ROOT / "data/csvs/test_2d_slices_flair.csv"

MODEL_PATH = PROJECT_ROOT / "models/2d_unet_flair_clean_improved.pth"

METRICS_SAVE_PATH = PROJECT_ROOT / "results/07b_clean_test_metrics_improved.csv"
SUMMARY_SAVE_PATH = PROJECT_ROOT / "results/07b_clean_test_summary_improved.csv"
PREDICTION_FIG_SAVE_PATH = PROJECT_ROOT / "results/07b_clean_test_predictions_improved.png"
COMPARE_SAVE_PATH = PROJECT_ROOT / "results/07b_old_vs_improved_clean_summary.csv"

OLD_SUMMARY_PATH = PROJECT_ROOT / "results/07_clean_test_summary.csv"

NUM_CLASSES = 4
INPUT_CHANNELS = 1
BATCH_SIZE = 16
NUM_WORKERS = 2
MAX_CACHE_PATIENTS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# 2. Helper functions
# ------------------------------------------------------------

def pick_column(df, candidates, column_purpose):
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not find column for {column_purpose}. "
        f"Tried: {candidates}. "
        f"Available columns are: {list(df.columns)}"
    )


def resolve_path(path_value):
    path_value = str(path_value)
    p = Path(path_value)

    if p.is_absolute():
        return str(p)

    return str(PROJECT_ROOT / p)


def normalize_slice(slice_2d):
    slice_2d = slice_2d.astype(np.float32)

    min_val = np.min(slice_2d)
    max_val = np.max(slice_2d)

    if max_val - min_val < 1e-8:
        return np.zeros_like(slice_2d, dtype=np.float32)

    return (slice_2d - min_val) / (max_val - min_val)


def remap_segmentation_labels(seg_slice):
    """
    Original BraTS labels:
        0, 1, 2, 4

    Model labels:
        0, 1, 2, 3

    So original label 4 becomes model label 3.
    """
    remapped = np.zeros_like(seg_slice, dtype=np.int64)

    remapped[seg_slice == 1] = 1
    remapped[seg_slice == 2] = 2
    remapped[seg_slice == 4] = 3

    return remapped


def dice_score_binary(pred_mask, true_mask, eps=1e-6):
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)

    intersection = np.logical_and(pred_mask, true_mask).sum()
    denominator = pred_mask.sum() + true_mask.sum()

    if denominator == 0:
        return np.nan

    return (2.0 * intersection + eps) / (denominator + eps)


def iou_score_binary(pred_mask, true_mask, eps=1e-6):
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)

    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()

    if union == 0:
        return np.nan

    return (intersection + eps) / (union + eps)


def has_tumor(mask):
    return np.any(mask > 0)


# ------------------------------------------------------------
# 3. Dataset class
# ------------------------------------------------------------

class BraTS2DTestDataset(Dataset):
    def __init__(self, csv_path, max_cache_patients=4):
        self.df = pd.read_csv(csv_path)

        self.flair_col = pick_column(
            self.df,
            candidates=["flair_path", "flair", "image_path", "image"],
            column_purpose="FLAIR image path"
        )

        self.seg_col = pick_column(
            self.df,
            candidates=["seg_path", "seg", "mask_path", "mask"],
            column_purpose="segmentation mask path"
        )

        self.slice_col = pick_column(
            self.df,
            candidates=["slice_idx", "slice_index", "slice", "z_index"],
            column_purpose="slice index"
        )

        self.patient_col = None
        for candidate in ["patient_id", "patient", "case_id", "case"]:
            if candidate in self.df.columns:
                self.patient_col = candidate
                break

        self.df[self.flair_col] = self.df[self.flair_col].apply(resolve_path)
        self.df[self.seg_col] = self.df[self.seg_col].apply(resolve_path)
        self.df[self.slice_col] = self.df[self.slice_col].astype(int)

        self.max_cache_patients = max_cache_patients
        self.cache = OrderedDict()

        print("Test dataset loaded.")
        print(f"CSV path: {csv_path}")
        print(f"Total test slices: {len(self.df)}")
        print(f"Using FLAIR column: {self.flair_col}")
        print(f"Using SEG column: {self.seg_col}")
        print(f"Using slice column: {self.slice_col}")

    def __len__(self):
        return len(self.df)

    def _load_patient_volumes(self, flair_path, seg_path):
        key = (flair_path, seg_path)

        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]

        flair_vol = nib.load(flair_path).get_fdata(dtype=np.float32)
        seg_vol = np.asanyarray(nib.load(seg_path).dataobj).astype(np.int16)

        self.cache[key] = (flair_vol, seg_vol)

        if len(self.cache) > self.max_cache_patients:
            self.cache.popitem(last=False)

        return flair_vol, seg_vol

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        flair_path = row[self.flair_col]
        seg_path = row[self.seg_col]
        slice_idx = int(row[self.slice_col])

        if self.patient_col is not None:
            patient_id = str(row[self.patient_col])
        else:
            patient_id = Path(flair_path).parent.name

        flair_vol, seg_vol = self._load_patient_volumes(flair_path, seg_path)

        flair_slice = flair_vol[:, :, slice_idx]
        seg_slice = seg_vol[:, :, slice_idx]

        flair_slice = normalize_slice(flair_slice)
        seg_slice = remap_segmentation_labels(seg_slice)

        flair_tensor = torch.from_numpy(flair_slice).float().unsqueeze(0)
        seg_tensor = torch.from_numpy(seg_slice).long()

        return flair_tensor, seg_tensor, patient_id, slice_idx


# ------------------------------------------------------------
# 4. 2D U-Net model
# Must match Script 06b exactly.
# ------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_from_decoder, x_from_encoder):
        x_from_decoder = self.up(x_from_decoder)

        diff_y = x_from_encoder.size(2) - x_from_decoder.size(2)
        diff_x = x_from_encoder.size(3) - x_from_decoder.size(3)

        x_from_decoder = F.pad(
            x_from_decoder,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2
            ]
        )

        x = torch.cat([x_from_encoder, x_from_decoder], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, base_channels=32):
        super().__init__()

        self.input_conv = DoubleConv(in_channels, base_channels)

        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)

        self.up1 = UpBlock(base_channels * 8, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2)
        self.up3 = UpBlock(base_channels * 2, base_channels)

        self.output_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

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


# ------------------------------------------------------------
# 5. Visualization helper
# ------------------------------------------------------------

def save_prediction_figure(saved_examples, save_path):
    """
    Save a simple figure:
        row 1: FLAIR
        row 2: ground truth mask
        row 3: predicted mask
    """
    if len(saved_examples) == 0:
        print("No examples were saved for visualization.")
        return

    n = len(saved_examples)

    fig, axes = plt.subplots(3, n, figsize=(4 * n, 10))

    if n == 1:
        axes = np.expand_dims(axes, axis=1)

    for col, example in enumerate(saved_examples):
        flair = example["flair"]
        true_mask = example["true_mask"]
        pred_mask = example["pred_mask"]
        patient_id = example["patient_id"]
        slice_idx = example["slice_idx"]

        axes[0, col].imshow(flair, cmap="gray")
        axes[0, col].set_title(f"{patient_id}\nSlice {slice_idx}\nFLAIR")
        axes[0, col].axis("off")

        axes[1, col].imshow(true_mask, vmin=0, vmax=3)
        axes[1, col].set_title("Ground truth")
        axes[1, col].axis("off")

        axes[2, col].imshow(pred_mask, vmin=0, vmax=3)
        axes[2, col].set_title("Prediction")
        axes[2, col].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved prediction figure: {save_path}")


# ------------------------------------------------------------
# 6. Main evaluation
# ------------------------------------------------------------

def main():
    print("=" * 80)
    print("Script 07b: Test improved clean 2D U-Net on clean test data")
    print("=" * 80)

    METRICS_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Could not find test CSV: {TEST_CSV}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Could not find improved model: {MODEL_PATH}")

    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    dataset = BraTS2DTestDataset(
        csv_path=TEST_CSV,
        max_cache_patients=MAX_CACHE_PATIENTS
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    base_channels = checkpoint.get("base_channels", 32)

    model = UNet2D(
        in_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        base_channels=base_channels
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model: {MODEL_PATH}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'unknown')}")
    print(f"Base channels: {base_channels}")

    rows = []
    saved_examples = []

    total_slices = 0
    tumor_slices = 0
    background_slices = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images, masks, patient_ids, slice_indices = batch

            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            images_np = images.cpu().numpy()
            masks_np = masks.cpu().numpy()
            preds_np = preds.cpu().numpy()

            batch_size = images_np.shape[0]

            for i in range(batch_size):
                true_mask = masks_np[i]
                pred_mask = preds_np[i]
                flair = images_np[i, 0]

                patient_id = str(patient_ids[i])
                slice_idx = int(slice_indices[i])

                total_slices += 1

                slice_has_tumor = has_tumor(true_mask)

                if slice_has_tumor:
                    tumor_slices += 1
                else:
                    background_slices += 1

                row = {
                    "patient_id": patient_id,
                    "slice_idx": slice_idx,
                    "has_tumor": slice_has_tumor
                }

                for cls in range(NUM_CLASSES):
                    true_cls = true_mask == cls
                    pred_cls = pred_mask == cls

                    row[f"class_{cls}_dice"] = dice_score_binary(pred_cls, true_cls)
                    row[f"class_{cls}_iou"] = iou_score_binary(pred_cls, true_cls)

                tumor_dice_values = [
                    row["class_1_dice"],
                    row["class_2_dice"],
                    row["class_3_dice"]
                ]

                tumor_iou_values = [
                    row["class_1_iou"],
                    row["class_2_iou"],
                    row["class_3_iou"]
                ]

                row["mean_tumor_dice"] = np.nanmean(tumor_dice_values)
                row["mean_tumor_iou"] = np.nanmean(tumor_iou_values)

                rows.append(row)

                # Save a few tumor examples for visual checking.
                if slice_has_tumor and len(saved_examples) < 5:
                    saved_examples.append(
                        {
                            "patient_id": patient_id,
                            "slice_idx": slice_idx,
                            "flair": flair,
                            "true_mask": true_mask,
                            "pred_mask": pred_mask
                        }
                    )

            if (batch_idx + 1) % 100 == 0:
                print(f"Processed batches: {batch_idx + 1}/{len(loader)}")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(METRICS_SAVE_PATH, index=False)

    print("-" * 80)
    print(f"Total evaluated test slices: {total_slices}")
    print(f"Tumor-containing test slices: {tumor_slices}")
    print(f"Background test slices: {background_slices}")
    print(f"Saved detailed metrics: {METRICS_SAVE_PATH}")

    summary_rows = []

    for cls in range(NUM_CLASSES):
        summary_rows.append(
            {
                "metric_group": f"class_{cls}",
                "mean_dice": metrics_df[f"class_{cls}_dice"].mean(skipna=True),
                "mean_iou": metrics_df[f"class_{cls}_iou"].mean(skipna=True)
            }
        )

    summary_rows.append(
        {
            "metric_group": "tumor_classes_1_2_3_all_test_slices",
            "mean_dice": metrics_df["mean_tumor_dice"].mean(skipna=True),
            "mean_iou": metrics_df["mean_tumor_iou"].mean(skipna=True)
        }
    )

    tumor_only_df = metrics_df[metrics_df["has_tumor"] == True]

    summary_rows.append(
        {
            "metric_group": "tumor_classes_1_2_3_tumor_slices_only",
            "mean_dice": tumor_only_df["mean_tumor_dice"].mean(skipna=True),
            "mean_iou": tumor_only_df["mean_tumor_iou"].mean(skipna=True)
        }
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_SAVE_PATH, index=False)

    print(f"Saved summary metrics: {SUMMARY_SAVE_PATH}")
    print("-" * 80)
    print("Improved clean test summary:")
    print(summary_df.to_string(index=False))

    save_prediction_figure(saved_examples, PREDICTION_FIG_SAVE_PATH)

    # --------------------------------------------------------
    # Compare with old baseline if old summary file exists
    # --------------------------------------------------------

    if OLD_SUMMARY_PATH.exists():
        old_df = pd.read_csv(OLD_SUMMARY_PATH)
        improved_df = summary_df.copy()

        old_df = old_df.rename(
            columns={
                "mean_dice": "old_mean_dice",
                "mean_iou": "old_mean_iou"
            }
        )

        improved_df = improved_df.rename(
            columns={
                "mean_dice": "improved_mean_dice",
                "mean_iou": "improved_mean_iou"
            }
        )

        comparison_df = old_df.merge(
            improved_df,
            on="metric_group",
            how="inner"
        )

        comparison_df["dice_change"] = (
            comparison_df["improved_mean_dice"] - comparison_df["old_mean_dice"]
        )

        comparison_df["iou_change"] = (
            comparison_df["improved_mean_iou"] - comparison_df["old_mean_iou"]
        )

        comparison_df.to_csv(COMPARE_SAVE_PATH, index=False)

        print("-" * 80)
        print(f"Saved old-vs-improved comparison: {COMPARE_SAVE_PATH}")
        print("Old vs improved clean test comparison:")
        print(comparison_df.to_string(index=False))
    else:
        print("-" * 80)
        print(f"Old summary file not found, so comparison was skipped: {OLD_SUMMARY_PATH}")

    print("=" * 80)
    print("Script 07b finished.")


if __name__ == "__main__":
    main()