# ============================================================
# Script 08: Test improved clean 2D U-Net on degraded test data
# Project: Robustness of Medical Image Segmentation Models
# ============================================================

from pathlib import Path
from collections import OrderedDict
import time
import json

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
CLEAN_SUMMARY_PATH = PROJECT_ROOT / "results/07b_clean_test_summary_improved.csv"

# IMPORTANT:
# Keep this as "quick" for the first run.
# Later, after this works, we will change it to "full".
RUN_MODE = "quick"

if RUN_MODE == "quick":
    OUTPUT_PREFIX = "08_quick"
    MAX_SLICES = 200
    DEGRADATIONS_TO_RUN = ["blur"]
    LEVELS_TO_RUN = [1, 3, 5]
else:
    OUTPUT_PREFIX = "08_full"
    MAX_SLICES = None
    DEGRADATIONS_TO_RUN = ["blur", "noise", "contrast", "ringing", "ghosting"]
    LEVELS_TO_RUN = [1, 2, 3, 4, 5]

METRICS_SAVE_PATH = PROJECT_ROOT / f"results/{OUTPUT_PREFIX}_degraded_test_metrics.csv"
SUMMARY_SAVE_PATH = PROJECT_ROOT / f"results/{OUTPUT_PREFIX}_degraded_test_summary.csv"
PLOT_SAVE_PATH = PROJECT_ROOT / f"results/{OUTPUT_PREFIX}_mean_tumor_dice_curve.png"

NUM_CLASSES = 4
INPUT_CHANNELS = 1

BATCH_SIZE = 16
NUM_WORKERS = 2
MAX_CACHE_PATIENTS = 4
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# 2. Degradation severity levels
# ------------------------------------------------------------

BLUR_LEVELS = {
    1: 0.5,
    2: 1.0,
    3: 1.5,
    4: 2.0,
    5: 2.5
}

NOISE_LEVELS = {
    1: 0.02,
    2: 0.04,
    3: 0.06,
    4: 0.08,
    5: 0.10
}

CONTRAST_LEVELS = {
    1: 0.90,
    2: 0.75,
    3: 0.60,
    4: 0.45,
    5: 0.30
}

RINGING_LEVELS = {
    1: 0.75,
    2: 0.60,
    3: 0.45,
    4: 0.30,
    5: 0.20
}

GHOSTING_LEVELS = {
    1: {"shift": 4, "intensity": 0.15},
    2: {"shift": 8, "intensity": 0.25},
    3: {"shift": 12, "intensity": 0.35},
    4: {"shift": 16, "intensity": 0.45},
    5: {"shift": 20, "intensity": 0.55}
}


# ------------------------------------------------------------
# 3. Helper functions
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
    remapped = np.zeros_like(seg_slice, dtype=np.int64)

    remapped[seg_slice == 1] = 1
    remapped[seg_slice == 2] = 2
    remapped[seg_slice == 4] = 3

    return remapped


def safe_nanmean(values):
    values = np.array(values, dtype=np.float32)

    if np.all(np.isnan(values)):
        return np.nan

    return float(np.nanmean(values))


def dice_score_binary(pred_mask, true_mask, eps=1e-6):
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)

    intersection = np.logical_and(pred_mask, true_mask).sum()
    denominator = pred_mask.sum() + true_mask.sum()

    if denominator == 0:
        return np.nan

    return float((2.0 * intersection + eps) / (denominator + eps))


def iou_score_binary(pred_mask, true_mask, eps=1e-6):
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)

    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()

    if union == 0:
        return np.nan

    return float((intersection + eps) / (union + eps))


def has_tumor(mask):
    return bool(np.any(mask > 0))


def severity_to_string(value):
    if isinstance(value, dict):
        return json.dumps(value)

    return str(value)


# ------------------------------------------------------------
# 4. Degradation functions
# ------------------------------------------------------------

def apply_gaussian_blur(image, sigma):
    """
    Simple Gaussian blur.
    Larger sigma means more blur.
    """
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        raise ImportError(
            "scipy is required for Gaussian blur. "
            "Install it in the brats3d environment if missing."
        )

    degraded = gaussian_filter(image, sigma=sigma)
    degraded = np.clip(degraded, 0.0, 1.0)

    return degraded.astype(np.float32)


def apply_gaussian_noise(image, noise_std, seed):
    """
    Add random Gaussian noise.
    Larger std means stronger noise.

    We use a fixed seed pattern so the experiment is repeatable.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=noise_std, size=image.shape)

    degraded = image + noise
    degraded = np.clip(degraded, 0.0, 1.0)

    return degraded.astype(np.float32)


def apply_low_contrast(image, factor):
    """
    Reduce contrast around the middle intensity.

    factor close to 1 = mild contrast reduction
    factor close to 0 = severe contrast reduction
    """
    degraded = (image - 0.5) * factor + 0.5
    degraded = np.clip(degraded, 0.0, 1.0)

    return degraded.astype(np.float32)


def apply_ringing_artifact(image, keep_fraction):
    """
    Simulate ringing by removing high-frequency information in k-space.

    Smaller keep_fraction means more information is removed,
    so ringing/blur-like artifacts become stronger.
    """
    h, w = image.shape

    kspace = np.fft.fftshift(np.fft.fft2(image))

    keep_h = int(h * keep_fraction)
    keep_w = int(w * keep_fraction)

    start_h = (h - keep_h) // 2
    start_w = (w - keep_w) // 2

    mask = np.zeros_like(kspace, dtype=np.float32)
    mask[start_h:start_h + keep_h, start_w:start_w + keep_w] = 1.0

    filtered_kspace = kspace * mask

    degraded = np.fft.ifft2(np.fft.ifftshift(filtered_kspace))
    degraded = np.real(degraded)
    degraded = np.clip(degraded, 0.0, 1.0)

    return degraded.astype(np.float32)


def apply_ghosting_artifact(image, shift, intensity):
    """
    Simulate ghosting by adding a shifted copy of the image.

    Larger shift/intensity means stronger ghost artifact.
    """
    shifted = np.roll(image, shift=shift, axis=1)

    degraded = image + intensity * shifted
    degraded = degraded / (1.0 + intensity)
    degraded = np.clip(degraded, 0.0, 1.0)

    return degraded.astype(np.float32)


def apply_degradation(image, degradation_type, severity_level, sample_index):
    """
    Apply one degradation type at one severity level.
    """
    if degradation_type == "blur":
        sigma = BLUR_LEVELS[severity_level]
        return apply_gaussian_blur(image, sigma), sigma

    if degradation_type == "noise":
        noise_std = NOISE_LEVELS[severity_level]
        seed = RANDOM_SEED + sample_index * 100 + severity_level
        return apply_gaussian_noise(image, noise_std, seed), noise_std

    if degradation_type == "contrast":
        factor = CONTRAST_LEVELS[severity_level]
        return apply_low_contrast(image, factor), factor

    if degradation_type == "ringing":
        keep_fraction = RINGING_LEVELS[severity_level]
        return apply_ringing_artifact(image, keep_fraction), keep_fraction

    if degradation_type == "ghosting":
        params = GHOSTING_LEVELS[severity_level]
        return apply_ghosting_artifact(
            image,
            shift=params["shift"],
            intensity=params["intensity"]
        ), params

    raise ValueError(f"Unknown degradation type: {degradation_type}")


# ------------------------------------------------------------
# 5. Dataset class
# ------------------------------------------------------------

class BraTS2DDegradedDataset(Dataset):
    def __init__(
        self,
        csv_path,
        degradation_type,
        severity_level,
        max_slices=None,
        max_cache_patients=4
    ):
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

        if max_slices is not None and max_slices < len(self.df):
            rng = np.random.default_rng(RANDOM_SEED)
            selected_indices = rng.choice(
                len(self.df),
                size=max_slices,
                replace=False
            )
            selected_indices = np.sort(selected_indices)
            self.df = self.df.iloc[selected_indices].reset_index(drop=True)

        self.degradation_type = degradation_type
        self.severity_level = severity_level

        self.max_cache_patients = max_cache_patients
        self.cache = OrderedDict()

        print(
            f"Dataset ready | degradation={degradation_type} | "
            f"level={severity_level} | slices={len(self.df)}"
        )

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

        clean_flair_slice = flair_vol[:, :, slice_idx]
        seg_slice = seg_vol[:, :, slice_idx]

        clean_flair_slice = normalize_slice(clean_flair_slice)

        degraded_flair_slice, severity_value = apply_degradation(
            image=clean_flair_slice,
            degradation_type=self.degradation_type,
            severity_level=self.severity_level,
            sample_index=idx
        )

        seg_slice = remap_segmentation_labels(seg_slice)

        image_tensor = torch.from_numpy(degraded_flair_slice).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(seg_slice).long()

        return image_tensor, mask_tensor, patient_id, slice_idx


# ------------------------------------------------------------
# 6. 2D U-Net model
# Must match Script 06b/07b.
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
# 7. Evaluation function
# ------------------------------------------------------------

def evaluate_condition(model, degradation_type, severity_level):
    dataset = BraTS2DDegradedDataset(
        csv_path=TEST_CSV,
        degradation_type=degradation_type,
        severity_level=severity_level,
        max_slices=MAX_SLICES,
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

    severity_value = None

    if degradation_type == "blur":
        severity_value = BLUR_LEVELS[severity_level]
    elif degradation_type == "noise":
        severity_value = NOISE_LEVELS[severity_level]
    elif degradation_type == "contrast":
        severity_value = CONTRAST_LEVELS[severity_level]
    elif degradation_type == "ringing":
        severity_value = RINGING_LEVELS[severity_level]
    elif degradation_type == "ghosting":
        severity_value = GHOSTING_LEVELS[severity_level]

    rows = []

    total_slices = 0
    tumor_slices = 0
    background_slices = 0

    start_time = time.time()

    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images, masks, patient_ids, slice_indices = batch

            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            masks_np = masks.cpu().numpy()
            preds_np = preds.cpu().numpy()

            batch_size = masks_np.shape[0]

            for i in range(batch_size):
                true_mask = masks_np[i]
                pred_mask = preds_np[i]

                patient_id = str(patient_ids[i])
                slice_idx = int(slice_indices[i])

                total_slices += 1

                slice_has_tumor = has_tumor(true_mask)

                if slice_has_tumor:
                    tumor_slices += 1
                else:
                    background_slices += 1

                row = {
                    "degradation_type": degradation_type,
                    "severity_level": severity_level,
                    "severity_value": severity_to_string(severity_value),
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

                row["mean_tumor_dice"] = safe_nanmean(tumor_dice_values)
                row["mean_tumor_iou"] = safe_nanmean(tumor_iou_values)

                rows.append(row)

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"  {degradation_type} level {severity_level}: "
                    f"processed {batch_idx + 1}/{len(loader)} batches"
                )

    elapsed = time.time() - start_time

    metrics_df = pd.DataFrame(rows)

    tumor_only_df = metrics_df[metrics_df["has_tumor"] == True]

    summary = {
        "degradation_type": degradation_type,
        "severity_level": severity_level,
        "severity_value": severity_to_string(severity_value),
        "total_slices": total_slices,
        "tumor_slices": tumor_slices,
        "background_slices": background_slices,
        "elapsed_seconds": elapsed,
        "class_0_dice": metrics_df["class_0_dice"].mean(skipna=True),
        "class_0_iou": metrics_df["class_0_iou"].mean(skipna=True),
        "class_1_dice": metrics_df["class_1_dice"].mean(skipna=True),
        "class_1_iou": metrics_df["class_1_iou"].mean(skipna=True),
        "class_2_dice": metrics_df["class_2_dice"].mean(skipna=True),
        "class_2_iou": metrics_df["class_2_iou"].mean(skipna=True),
        "class_3_dice": metrics_df["class_3_dice"].mean(skipna=True),
        "class_3_iou": metrics_df["class_3_iou"].mean(skipna=True),
        "mean_tumor_dice_all_slices": metrics_df["mean_tumor_dice"].mean(skipna=True),
        "mean_tumor_iou_all_slices": metrics_df["mean_tumor_iou"].mean(skipna=True),
        "mean_tumor_dice_tumor_slices_only": tumor_only_df["mean_tumor_dice"].mean(skipna=True),
        "mean_tumor_iou_tumor_slices_only": tumor_only_df["mean_tumor_iou"].mean(skipna=True),
    }

    print(
        f"Finished {degradation_type} level {severity_level} | "
        f"mean tumor Dice on tumor slices: "
        f"{summary['mean_tumor_dice_tumor_slices_only']:.4f} | "
        f"time: {elapsed:.1f}s"
    )

    return metrics_df, summary


# ------------------------------------------------------------
# 8. Plot helper
# ------------------------------------------------------------

def save_degradation_plot(summary_df, save_path):
    plt.figure(figsize=(8, 5))

    for degradation_type in summary_df["degradation_type"].unique():
        temp = summary_df[summary_df["degradation_type"] == degradation_type]
        temp = temp.sort_values("severity_level")

        plt.plot(
            temp["severity_level"],
            temp["mean_tumor_dice_tumor_slices_only"],
            marker="o",
            label=degradation_type
        )

    plt.xlabel("Severity level")
    plt.ylabel("Mean tumor Dice, tumor slices only")
    plt.title("Segmentation Robustness Under Image Degradation")
    plt.xticks([1, 2, 3, 4, 5])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved degradation plot: {save_path}")


# ------------------------------------------------------------
# 9. Main
# ------------------------------------------------------------

def main():
    print("=" * 80)
    print("Script 08: Degraded robustness testing")
    print("=" * 80)

    print(f"RUN_MODE: {RUN_MODE}")

    if RUN_MODE == "quick":
        print("This is a QUICK SMOKE TEST, not the final robustness experiment.")
        print(f"Max slices: {MAX_SLICES}")
        print(f"Degradations: {DEGRADATIONS_TO_RUN}")
        print(f"Levels: {LEVELS_TO_RUN}")
    else:
        print("This is the FULL robustness experiment.")
        print("This may take a while.")

    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Could not find test CSV: {TEST_CSV}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Could not find model: {MODEL_PATH}")

    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

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

    clean_reference_dice = None
    clean_reference_iou = None

    if CLEAN_SUMMARY_PATH.exists():
        clean_summary = pd.read_csv(CLEAN_SUMMARY_PATH)
        clean_row = clean_summary[
            clean_summary["metric_group"] == "tumor_classes_1_2_3_tumor_slices_only"
        ]

        if len(clean_row) > 0:
            clean_reference_dice = float(clean_row["mean_dice"].iloc[0])
            clean_reference_iou = float(clean_row["mean_iou"].iloc[0])

            print(
                f"Clean reference tumor Dice: {clean_reference_dice:.4f} | "
                f"Clean reference tumor IoU: {clean_reference_iou:.4f}"
            )

    all_metrics = []
    all_summaries = []

    overall_start = time.time()

    for degradation_type in DEGRADATIONS_TO_RUN:
        for severity_level in LEVELS_TO_RUN:
            print("-" * 80)
            print(f"Evaluating: {degradation_type}, severity level {severity_level}")

            metrics_df, summary = evaluate_condition(
                model=model,
                degradation_type=degradation_type,
                severity_level=severity_level
            )

            if clean_reference_dice is not None:
                summary["clean_reference_tumor_dice"] = clean_reference_dice
                summary["dice_drop_from_clean"] = (
                    clean_reference_dice
                    - summary["mean_tumor_dice_tumor_slices_only"]
                )

            if clean_reference_iou is not None:
                summary["clean_reference_tumor_iou"] = clean_reference_iou
                summary["iou_drop_from_clean"] = (
                    clean_reference_iou
                    - summary["mean_tumor_iou_tumor_slices_only"]
                )

            all_metrics.append(metrics_df)
            all_summaries.append(summary)

    final_metrics_df = pd.concat(all_metrics, ignore_index=True)
    final_summary_df = pd.DataFrame(all_summaries)

    final_metrics_df.to_csv(METRICS_SAVE_PATH, index=False)
    final_summary_df.to_csv(SUMMARY_SAVE_PATH, index=False)

    save_degradation_plot(final_summary_df, PLOT_SAVE_PATH)

    total_elapsed = time.time() - overall_start

    print("=" * 80)
    print("Degraded robustness summary:")
    print(final_summary_df.to_string(index=False))
    print("-" * 80)
    print(f"Saved detailed metrics: {METRICS_SAVE_PATH}")
    print(f"Saved summary: {SUMMARY_SAVE_PATH}")
    print(f"Saved plot: {PLOT_SAVE_PATH}")
    print(f"Total elapsed time: {total_elapsed / 60:.2f} minutes")
    print("=" * 80)
    print("Script 08 finished.")


if __name__ == "__main__":
    main()