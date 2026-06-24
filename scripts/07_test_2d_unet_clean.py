"""
Script 07: Test clean 2D U-Net baseline on clean FLAIR test slices.

Goal:
- Load trained clean 2D U-Net model
- Load selected 2D clean test slices
- Predict segmentation masks
- Compute Dice and IoU metrics
- Save metrics CSV
- Save visual prediction examples

Important:
- This script does NOT train a model.
- This script evaluates the clean model on clean test images.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Paths
# -----------------------------
PROJECT_DIR = Path("/home/xfh25/brats_segmentation_project")

TEST_SLICE_CSV = PROJECT_DIR / "data" / "csvs" / "test_2d_slices_flair.csv"

MODEL_PATH = PROJECT_DIR / "models" / "2d_unet_flair_clean.pth"

RESULTS_DIR = PROJECT_DIR / "results"
METRICS_CSV = RESULTS_DIR / "07_clean_test_metrics.csv"
SUMMARY_CSV = RESULTS_DIR / "07_clean_test_summary.csv"
PREDICTION_FIG = RESULTS_DIR / "07_clean_test_predictions.png"


# -----------------------------
# Settings
# -----------------------------
RANDOM_SEED = 42

BATCH_SIZE = 16
NUM_WORKERS = 2
NUM_CLASSES = 4

# For visual examples
NUM_VISUAL_EXAMPLES = 6


# -----------------------------
# Helper functions
# -----------------------------
def normalize_image(image):
    """
    Normalize one 2D FLAIR slice to 0-1.
    """
    image = image.astype(np.float32)

    min_val = np.min(image)
    max_val = np.max(image)

    if max_val - min_val == 0:
        return np.zeros_like(image, dtype=np.float32)

    return (image - min_val) / (max_val - min_val)


def remap_mask_labels(mask):
    """
    Convert BraTS segmentation labels from [0, 1, 2, 4] to [0, 1, 2, 3].
    """
    mask = mask.astype(np.int64)

    remapped = np.zeros_like(mask, dtype=np.int64)
    remapped[mask == 1] = 1
    remapped[mask == 2] = 2
    remapped[mask == 4] = 3

    return remapped


def dice_score(pred, target, class_id, eps=1e-7):
    """
    Compute Dice score for one class.
    """
    pred_class = (pred == class_id)
    target_class = (target == class_id)

    intersection = np.logical_and(pred_class, target_class).sum()
    pred_sum = pred_class.sum()
    target_sum = target_class.sum()

    if pred_sum == 0 and target_sum == 0:
        return np.nan

    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    return dice


def iou_score(pred, target, class_id, eps=1e-7):
    """
    Compute IoU/Jaccard score for one class.
    """
    pred_class = (pred == class_id)
    target_class = (target == class_id)

    intersection = np.logical_and(pred_class, target_class).sum()
    union = np.logical_or(pred_class, target_class).sum()

    if union == 0:
        return np.nan

    iou = (intersection + eps) / (union + eps)
    return iou


# -----------------------------
# Dataset
# -----------------------------
class BraTS2DTestDataset(Dataset):
    """
    Dataset for loading clean 2D FLAIR test slices and masks.
    """

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        flair_path = row["flair_path"]
        seg_path = row["seg_path"]
        slice_idx = int(row["slice_idx"])

        flair_volume = nib.load(str(flair_path)).get_fdata()
        seg_volume = nib.load(str(seg_path)).get_fdata()

        flair_slice = flair_volume[:, :, slice_idx]
        seg_slice = seg_volume[:, :, slice_idx]

        flair_slice = normalize_image(flair_slice)
        seg_slice = remap_mask_labels(seg_slice)

        image_tensor = torch.tensor(flair_slice, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(seg_slice, dtype=torch.long)

        metadata = {
            "patient_id": row["patient_id"],
            "slice_idx": slice_idx,
            "has_tumor": int(row["has_tumor"]),
            "slice_type": row["slice_type"],
        }

        return image_tensor, mask_tensor, metadata


# -----------------------------
# Model architecture
# Must match Script 06 exactly
# -----------------------------
class DoubleConv(nn.Module):
    """
    Two convolution layers with ReLU.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet2D(nn.Module):
    """
    Small 2D U-Net for baseline segmentation.
    """

    def __init__(self, in_channels=1, num_classes=4):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)

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


def save_prediction_figure(visual_examples, output_path):
    """
    Save visual examples of image, ground truth, and prediction.
    """
    if len(visual_examples) == 0:
        print("No visual examples available to save.")
        return

    num_examples = len(visual_examples)

    fig, axes = plt.subplots(num_examples, 3, figsize=(10, 3 * num_examples))

    if num_examples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, example in enumerate(visual_examples):
        image = example["image"]
        gt = example["gt"]
        pred = example["pred"]
        patient_id = example["patient_id"]
        slice_idx = example["slice_idx"]

        axes[i, 0].imshow(image, cmap="gray")
        axes[i, 0].set_title(f"{patient_id}\nSlice {slice_idx}\nFLAIR")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(image, cmap="gray")
        gt_masked = np.ma.masked_where(gt == 0, gt)
        axes[i, 1].imshow(gt_masked, alpha=0.45)
        axes[i, 1].set_title("Ground truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(image, cmap="gray")
        pred_masked = np.ma.masked_where(pred == 0, pred)
        axes[i, 2].imshow(pred_masked, alpha=0.45)
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 80)
    print("Script 07: Test clean 2D U-Net on clean FLAIR test slices")
    print("=" * 80)

    if not TEST_SLICE_CSV.exists():
        raise FileNotFoundError(f"Could not find test slice CSV: {TEST_SLICE_CSV}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Could not find trained model: {MODEL_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    print(f"\nLoading test slices from:\n{TEST_SLICE_CSV}")
    dataset = BraTS2DTestDataset(TEST_SLICE_CSV)

    print(f"Total test slices: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print(f"\nLoading model from:\n{MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = UNet2D(
        in_channels=1,
        num_classes=NUM_CLASSES
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model loaded successfully.")
    print(f"Saved epoch: {checkpoint['epoch']}")
    print(f"Best validation loss during training: {checkpoint['best_val_loss']}")

    all_metric_rows = []
    visual_examples = []

    print("\nStarting clean test evaluation...")

    with torch.no_grad():
        for images, masks, metadata in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            images_np = images.cpu().numpy()
            masks_np = masks.cpu().numpy()
            preds_np = predictions.cpu().numpy()

            batch_size = images_np.shape[0]

            for i in range(batch_size):
                pred = preds_np[i]
                target = masks_np[i]
                image = images_np[i, 0]

                patient_id = metadata["patient_id"][i]
                slice_idx = int(metadata["slice_idx"][i])
                has_tumor = int(metadata["has_tumor"][i])
                slice_type = metadata["slice_type"][i]

                row = {
                    "patient_id": patient_id,
                    "slice_idx": slice_idx,
                    "has_tumor": has_tumor,
                    "slice_type": slice_type,
                }

                # Per-class metrics
                for class_id in range(NUM_CLASSES):
                    row[f"dice_class_{class_id}"] = dice_score(pred, target, class_id)
                    row[f"iou_class_{class_id}"] = iou_score(pred, target, class_id)

                # Tumor-only average across classes 1, 2, 3
                tumor_dice_values = [
                    row["dice_class_1"],
                    row["dice_class_2"],
                    row["dice_class_3"],
                ]

                tumor_iou_values = [
                    row["iou_class_1"],
                    row["iou_class_2"],
                    row["iou_class_3"],
                ]

                row["mean_tumor_dice"] = np.nanmean(tumor_dice_values)
                row["mean_tumor_iou"] = np.nanmean(tumor_iou_values)

                all_metric_rows.append(row)

                # Save tumor-containing examples for visual inspection
                if has_tumor == 1 and len(visual_examples) < NUM_VISUAL_EXAMPLES:
                    visual_examples.append({
                        "image": image,
                        "gt": target,
                        "pred": pred,
                        "patient_id": patient_id,
                        "slice_idx": slice_idx,
                    })

    metrics_df = pd.DataFrame(all_metric_rows)
    metrics_df.to_csv(METRICS_CSV, index=False)

    # Summary metrics
    summary_rows = []

    for class_id in range(NUM_CLASSES):
        summary_rows.append({
            "metric_group": f"class_{class_id}",
            "mean_dice": metrics_df[f"dice_class_{class_id}"].mean(skipna=True),
            "mean_iou": metrics_df[f"iou_class_{class_id}"].mean(skipna=True),
        })

    tumor_slices_df = metrics_df[metrics_df["has_tumor"] == 1]

    summary_rows.append({
        "metric_group": "tumor_classes_1_2_3_all_test_slices",
        "mean_dice": metrics_df["mean_tumor_dice"].mean(skipna=True),
        "mean_iou": metrics_df["mean_tumor_iou"].mean(skipna=True),
    })

    summary_rows.append({
        "metric_group": "tumor_classes_1_2_3_tumor_slices_only",
        "mean_dice": tumor_slices_df["mean_tumor_dice"].mean(skipna=True),
        "mean_iou": tumor_slices_df["mean_tumor_iou"].mean(skipna=True),
    })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    save_prediction_figure(visual_examples, PREDICTION_FIG)

    print("\n" + "=" * 80)
    print("Clean Test Evaluation Summary")
    print("=" * 80)

    print(f"Total evaluated test slices: {len(metrics_df)}")
    print(f"Tumor-containing test slices: {(metrics_df['has_tumor'] == 1).sum()}")
    print(f"Background test slices: {(metrics_df['has_tumor'] == 0).sum()}")

    print("\nPer-class summary:")
    print(summary_df)

    print(f"\nSaved detailed metrics CSV to:\n{METRICS_CSV}")
    print(f"\nSaved summary CSV to:\n{SUMMARY_CSV}")
    print(f"\nSaved prediction visual figure to:\n{PREDICTION_FIG}")

    print("\nDone. Script 07 completed successfully.")


if __name__ == "__main__":
    main()