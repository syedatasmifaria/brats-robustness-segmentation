# ============================================================
# Script 06b: Train improved clean 2D U-Net on FLAIR slices
# Project: Robustness of Medical Image Segmentation Models
# ============================================================

from pathlib import Path
from collections import OrderedDict
import time
import random

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# ------------------------------------------------------------
# 1. Basic settings
# ------------------------------------------------------------

PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

TRAIN_CSV = PROJECT_ROOT / "data/csvs/train_2d_slices_flair.csv"

MODEL_SAVE_PATH = PROJECT_ROOT / "models/2d_unet_flair_clean_improved.pth"
LOG_SAVE_PATH = PROJECT_ROOT / "results/06b_training_log.csv"
PLOT_SAVE_PATH = PROJECT_ROOT / "results/06b_training_loss_curve.png"

NUM_CLASSES = 4
INPUT_CHANNELS = 1

EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

NUM_WORKERS = 2
MAX_CACHE_PATIENTS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# 2. Reproducibility
# ------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)


# ------------------------------------------------------------
# 3. Helper functions
# ------------------------------------------------------------

def pick_column(df, candidates, column_purpose):
    """
    Finds the correct column name from several possible choices.
    This makes the script safer if CSV column names vary slightly.
    """
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not find column for {column_purpose}. "
        f"Tried: {candidates}. "
        f"Available columns are: {list(df.columns)}"
    )


def resolve_path(path_value):
    """
    Converts a CSV path into an absolute path.
    If it is already absolute, keep it.
    If it is relative, attach PROJECT_ROOT.
    """
    path_value = str(path_value)
    p = Path(path_value)

    if p.is_absolute():
        return str(p)

    return str(PROJECT_ROOT / p)


def normalize_slice(slice_2d):
    """
    Normalize one FLAIR slice to 0-1.

    MRI intensities do not naturally live in a fixed range like normal images.
    Normalizing helps the neural network learn more stably.
    """
    slice_2d = slice_2d.astype(np.float32)

    min_val = np.min(slice_2d)
    max_val = np.max(slice_2d)

    if max_val - min_val < 1e-8:
        return np.zeros_like(slice_2d, dtype=np.float32)

    return (slice_2d - min_val) / (max_val - min_val)


def remap_segmentation_labels(seg_slice):
    """
    BraTS labels are originally:
        0 = background
        1 = tumor label 1
        2 = edema
        4 = enhancing tumor

    PyTorch multiclass loss expects labels to be consecutive:
        0, 1, 2, 3

    So we remap:
        original 0 -> 0
        original 1 -> 1
        original 2 -> 2
        original 4 -> 3
    """
    remapped = np.zeros_like(seg_slice, dtype=np.int64)

    remapped[seg_slice == 1] = 1
    remapped[seg_slice == 2] = 2
    remapped[seg_slice == 4] = 3

    return remapped


# ------------------------------------------------------------
# 4. Dataset class
# ------------------------------------------------------------

class BraTS2DSliceDataset(Dataset):
    """
    This dataset does not save images as separate PNG files.

    Instead, the CSV tells us:
        - which patient volume to open
        - which 2D slice index to use

    For each item:
        input  = one normalized FLAIR slice
        target = one segmentation mask slice
    """

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

        self.df[self.flair_col] = self.df[self.flair_col].apply(resolve_path)
        self.df[self.seg_col] = self.df[self.seg_col].apply(resolve_path)
        self.df[self.slice_col] = self.df[self.slice_col].astype(int)

        self.max_cache_patients = max_cache_patients
        self.cache = OrderedDict()

        print("Dataset loaded.")
        print(f"CSV path: {csv_path}")
        print(f"Total slices: {len(self.df)}")
        print(f"Using FLAIR column: {self.flair_col}")
        print(f"Using SEG column: {self.seg_col}")
        print(f"Using slice column: {self.slice_col}")

    def __len__(self):
        return len(self.df)

    def _load_patient_volumes(self, flair_path, seg_path):
        """
        Load FLAIR and SEG volumes.

        Small cache:
        Opening NIfTI files repeatedly can be slow.
        So we keep a few recently used patients in memory.
        Not the whole dataset, because that can waste RAM.
        """
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

        flair_vol, seg_vol = self._load_patient_volumes(flair_path, seg_path)

        flair_slice = flair_vol[:, :, slice_idx]
        seg_slice = seg_vol[:, :, slice_idx]

        flair_slice = normalize_slice(flair_slice)
        seg_slice = remap_segmentation_labels(seg_slice)

        flair_tensor = torch.from_numpy(flair_slice).float().unsqueeze(0)
        seg_tensor = torch.from_numpy(seg_slice).long()

        return flair_tensor, seg_tensor


# ------------------------------------------------------------
# 5. 2D U-Net model
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

        # In case shapes differ by 1 pixel due to pooling/upsampling
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
# 6. Loss function: CrossEntropy + Dice
# ------------------------------------------------------------

def dice_loss_multiclass(logits, targets, num_classes=4, include_background=False, eps=1e-6):
    """
    Dice loss compares predicted masks with true masks.

    We exclude background by default because background is too easy.
    If background dominates the loss, the model can look good while still
    doing badly on tumor regions.
    """
    probs = torch.softmax(logits, dim=1)

    targets_one_hot = F.one_hot(targets, num_classes=num_classes)
    targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

    if not include_background:
        probs = probs[:, 1:, :, :]
        targets_one_hot = targets_one_hot[:, 1:, :, :]

    dims = (0, 2, 3)

    intersection = torch.sum(probs * targets_one_hot, dims)
    denominator = torch.sum(probs + targets_one_hot, dims)

    dice_score = (2.0 * intersection + eps) / (denominator + eps)
    loss = 1.0 - dice_score.mean()

    return loss


def combined_loss(logits, targets):
    ce = F.cross_entropy(logits, targets)
    dice = dice_loss_multiclass(
        logits,
        targets,
        num_classes=NUM_CLASSES,
        include_background=False
    )

    total = ce + dice
    return total, ce.item(), dice.item()


# ------------------------------------------------------------
# 7. Training and validation loops
# ------------------------------------------------------------

def train_one_epoch(model, loader, optimizer):
    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_dice = 0.0

    for images, masks in loader:
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        logits = model(images)

        loss, ce_value, dice_value = combined_loss(logits, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce += ce_value
        total_dice += dice_value

    avg_loss = total_loss / len(loader)
    avg_ce = total_ce / len(loader)
    avg_dice = total_dice / len(loader)

    return avg_loss, avg_ce, avg_dice


def validate_one_epoch(model, loader):
    model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            logits = model(images)

            loss, ce_value, dice_value = combined_loss(logits, masks)

            total_loss += loss.item()
            total_ce += ce_value
            total_dice += dice_value

    avg_loss = total_loss / len(loader)
    avg_ce = total_ce / len(loader)
    avg_dice = total_dice / len(loader)

    return avg_loss, avg_ce, avg_dice


# ------------------------------------------------------------
# 8. Main function
# ------------------------------------------------------------

def main():
    print("=" * 80)
    print("Script 06b: Improved clean 2D U-Net training")
    print("=" * 80)

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Could not find training CSV: {TRAIN_CSV}")

    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")

    dataset = BraTS2DSliceDataset(
        csv_path=TRAIN_CSV,
        max_cache_patients=MAX_CACHE_PATIENTS
    )

    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    print("-" * 80)
    print(f"Training slices: {train_size}")
    print(f"Validation slices: {val_size}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("Loss: CrossEntropy + Dice loss")
    print("-" * 80)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )

    model = UNet2D(
        in_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        base_channels=32
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    training_log = []

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_ce, train_dice = train_one_epoch(
            model,
            train_loader,
            optimizer
        )

        val_loss, val_ce, val_dice = validate_one_epoch(
            model,
            val_loader
        )

        epoch_time = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_crossentropy": train_ce,
            "train_dice_loss": train_dice,
            "val_loss": val_loss,
            "val_crossentropy": val_ce,
            "val_dice_loss": val_dice,
            "epoch_time_seconds": epoch_time
        }

        training_log.append(row)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Train CE: {train_ce:.4f} | "
            f"Train Dice loss: {train_dice:.4f} | "
            f"Val CE: {val_ce:.4f} | "
            f"Val Dice loss: {val_dice:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "num_classes": NUM_CLASSES,
                "input_channels": INPUT_CHANNELS,
                "input_modality": "FLAIR",
                "training_type": "clean_improved",
                "loss_function": "CrossEntropy + Dice loss",
                "label_mapping": {
                    "0": "background",
                    "1": "BraTS original label 1",
                    "2": "BraTS original label 2 / edema",
                    "3": "BraTS original label 4"
                },
                "base_channels": 32
            }

            torch.save(checkpoint, MODEL_SAVE_PATH)

            print(
                f"  Saved new best model at epoch {epoch} "
                f"with val loss {best_val_loss:.4f}"
            )

    total_time = time.time() - start_time

    log_df = pd.DataFrame(training_log)
    log_df.to_csv(LOG_SAVE_PATH, index=False)

    print("-" * 80)
    print(f"Training finished in {total_time / 60:.2f} minutes.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Saved model: {MODEL_SAVE_PATH}")
    print(f"Saved training log: {LOG_SAVE_PATH}")

    plt.figure(figsize=(8, 5))
    plt.plot(log_df["epoch"], log_df["train_loss"], marker="o", label="Training loss")
    plt.plot(log_df["epoch"], log_df["val_loss"], marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Improved 2D U-Net Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH, dpi=300)
    plt.close()

    print(f"Saved loss curve: {PLOT_SAVE_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()