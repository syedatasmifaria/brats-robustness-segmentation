
# ============================================================
# Script 11: Train clean FLAIR-only 3D U-Net - quick test
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
from torch.utils.data import Dataset, DataLoader


# ------------------------------------------------------------
# 1. Basic settings
# ------------------------------------------------------------

PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

TRAIN_CSV = PROJECT_ROOT / "data/csvs/train_paths.csv"

MODEL_SAVE_PATH = PROJECT_ROOT / "models/3d_unet_flair_clean_quick.pth"
LOG_SAVE_PATH = PROJECT_ROOT / "results/11_quick_3d_training_log.csv"
PLOT_SAVE_PATH = PROJECT_ROOT / "results/11_quick_3d_training_loss_curve.png"

# This is only a quick test run.
RUN_MODE = "quick"

PATCH_SIZE = (96, 96, 96)

NUM_CLASSES = 4
INPUT_CHANNELS = 1
BASE_CHANNELS = 16

EPOCHS = 3
BATCH_SIZE = 1
LEARNING_RATE = 1e-4

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

MAX_TRAIN_PATIENTS_QUICK = 40
MAX_VAL_PATIENTS_QUICK = 10

PATCHES_PER_PATIENT_TRAIN = 2
PATCHES_PER_PATIENT_VAL = 1

BRAIN_THRESHOLD = 0.01
CROP_MARGIN = 8
TUMOR_PATCH_PROBABILITY = 0.75

# Keep this 0 for now. Safer for NIfTI loading and beginner debugging.
NUM_WORKERS = 0

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
# 3. Data helper functions
# ------------------------------------------------------------

def load_flair(path):
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def load_seg(path):
    return np.asanyarray(nib.load(str(path)).dataobj).astype(np.int16)


def normalize_volume(volume):
    volume = volume.astype(np.float32)

    min_val = np.min(volume)
    max_val = np.max(volume)

    if max_val - min_val < 1e-8:
        return np.zeros_like(volume, dtype=np.float32)

    return (volume - min_val) / (max_val - min_val)


def remap_segmentation_labels(seg):
    """
    Original BraTS labels:
        0, 1, 2, 4

    Model labels:
        0, 1, 2, 3

    So original label 4 becomes class 3.
    """
    remapped = np.zeros_like(seg, dtype=np.int64)

    remapped[seg == 1] = 1
    remapped[seg == 2] = 2
    remapped[seg == 4] = 3

    return remapped


def get_brain_bbox(volume, threshold=0.01, margin=8):
    """
    Find the bounding box around visible brain tissue.
    This removes black empty space and saves memory.
    """
    mask = volume > threshold
    coords = np.argwhere(mask)

    if coords.size == 0:
        return (0, volume.shape[0], 0, volume.shape[1], 0, volume.shape[2])

    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0) + 1

    x_min = max(int(x_min) - margin, 0)
    y_min = max(int(y_min) - margin, 0)
    z_min = max(int(z_min) - margin, 0)

    x_max = min(int(x_max) + margin, volume.shape[0])
    y_max = min(int(y_max) + margin, volume.shape[1])
    z_max = min(int(z_max) + margin, volume.shape[2])

    return (x_min, x_max, y_min, y_max, z_min, z_max)


def crop_with_bbox(array, bbox):
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    return array[x_min:x_max, y_min:y_max, z_min:z_max]


def pad_if_needed(volume, seg, patch_size):
    """
    If a cropped volume is smaller than 96x96x96, pad it.
    Usually BraTS crops are already big enough, but this protects us.
    """
    sx, sy, sz = patch_size
    x, y, z = volume.shape

    pad_x = max(sx - x, 0)
    pad_y = max(sy - y, 0)
    pad_z = max(sz - z, 0)

    if pad_x == 0 and pad_y == 0 and pad_z == 0:
        return volume, seg

    pad_width = (
        (pad_x // 2, pad_x - pad_x // 2),
        (pad_y // 2, pad_y - pad_y // 2),
        (pad_z // 2, pad_z - pad_z // 2),
    )

    volume = np.pad(volume, pad_width, mode="constant", constant_values=0)
    seg = np.pad(seg, pad_width, mode="constant", constant_values=0)

    return volume, seg


def sample_patch(volume, seg, patch_size, tumor_patch_probability=0.75):
    """
    Sample one 3D patch.

    Most patches are sampled near tumor because tumor voxels are rare.
    Some patches are random so the model also learns background.
    """
    sx, sy, sz = patch_size
    x_dim, y_dim, z_dim = volume.shape

    tumor_coords = np.argwhere(seg > 0)

    use_tumor_patch = (
        len(tumor_coords) > 0 and
        np.random.rand() < tumor_patch_probability
    )

    if use_tumor_patch:
        center = tumor_coords[np.random.randint(len(tumor_coords))]
    else:
        center = np.array([
            np.random.randint(0, x_dim),
            np.random.randint(0, y_dim),
            np.random.randint(0, z_dim),
        ])

    cx, cy, cz = center

    x_start = int(cx - sx // 2)
    y_start = int(cy - sy // 2)
    z_start = int(cz - sz // 2)

    x_start = max(0, min(x_start, x_dim - sx))
    y_start = max(0, min(y_start, y_dim - sy))
    z_start = max(0, min(z_start, z_dim - sz))

    x_end = x_start + sx
    y_end = y_start + sy
    z_end = z_start + sz

    image_patch = volume[x_start:x_end, y_start:y_end, z_start:z_end]
    seg_patch = seg[x_start:x_end, y_start:y_end, z_start:z_end]

    return image_patch, seg_patch


# ------------------------------------------------------------
# 4. Dataset
# ------------------------------------------------------------

class BraTS3DPatchDataset(Dataset):
    def __init__(
        self,
        dataframe,
        patches_per_patient,
        patch_size,
        mode,
        max_cache_patients=2
    ):
        self.df = dataframe.reset_index(drop=True)
        self.patches_per_patient = patches_per_patient
        self.patch_size = patch_size
        self.mode = mode
        self.max_cache_patients = max_cache_patients
        self.cache = OrderedDict()

        print(
            f"{mode} dataset ready | patients={len(self.df)} | "
            f"patches_per_patient={patches_per_patient} | "
            f"total patches per epoch={len(self)}"
        )

    def __len__(self):
        return len(self.df) * self.patches_per_patient

    def _load_patient(self, patient_index):
        row = self.df.iloc[patient_index]

        patient_id = row["patient_id"]
        flair_path = row["flair"]
        seg_path = row["seg"]

        key = patient_id

        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]

        flair = load_flair(flair_path)
        seg_original = load_seg(seg_path)

        flair = normalize_volume(flair)
        seg = remap_segmentation_labels(seg_original)

        bbox = get_brain_bbox(
            flair,
            threshold=BRAIN_THRESHOLD,
            margin=CROP_MARGIN
        )

        flair = crop_with_bbox(flair, bbox)
        seg = crop_with_bbox(seg, bbox)

        flair, seg = pad_if_needed(flair, seg, self.patch_size)

        self.cache[key] = (flair, seg, patient_id)

        if len(self.cache) > self.max_cache_patients:
            self.cache.popitem(last=False)

        return flair, seg, patient_id

    def __getitem__(self, idx):
        patient_index = idx // self.patches_per_patient

        flair, seg, patient_id = self._load_patient(patient_index)

        if self.mode == "train":
            tumor_prob = TUMOR_PATCH_PROBABILITY
        else:
            tumor_prob = 1.0

        image_patch, seg_patch = sample_patch(
            flair,
            seg,
            self.patch_size,
            tumor_patch_probability=tumor_prob
        )

        image_tensor = torch.from_numpy(image_patch).float().unsqueeze(0)
        seg_tensor = torch.from_numpy(seg_patch).long()

        return image_tensor, seg_tensor


# ------------------------------------------------------------
# 5. 3D U-Net model
# ------------------------------------------------------------

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


class DownBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)


class UpBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose3d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x_from_decoder, x_from_encoder):
        x_from_decoder = self.up(x_from_decoder)

        diff_x = x_from_encoder.size(2) - x_from_decoder.size(2)
        diff_y = x_from_encoder.size(3) - x_from_decoder.size(3)
        diff_z = x_from_encoder.size(4) - x_from_decoder.size(4)

        x_from_decoder = F.pad(
            x_from_decoder,
            [
                diff_z // 2,
                diff_z - diff_z // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
                diff_x // 2,
                diff_x - diff_x // 2,
            ]
        )

        x = torch.cat([x_from_encoder, x_from_decoder], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, base_channels=16):
        super().__init__()

        self.input_conv = DoubleConv3D(in_channels, base_channels)

        self.down1 = DownBlock3D(base_channels, base_channels * 2)
        self.down2 = DownBlock3D(base_channels * 2, base_channels * 4)
        self.down3 = DownBlock3D(base_channels * 4, base_channels * 8)

        self.up1 = UpBlock3D(base_channels * 8, base_channels * 4)
        self.up2 = UpBlock3D(base_channels * 4, base_channels * 2)
        self.up3 = UpBlock3D(base_channels * 2, base_channels)

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


# ------------------------------------------------------------
# 6. Loss function
# ------------------------------------------------------------

def dice_loss_multiclass_3d(logits, targets, num_classes=4, include_background=False, eps=1e-6):
    probs = torch.softmax(logits, dim=1)

    targets_one_hot = F.one_hot(targets, num_classes=num_classes)
    targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()

    if not include_background:
        probs = probs[:, 1:, :, :, :]
        targets_one_hot = targets_one_hot[:, 1:, :, :, :]

    dims = (0, 2, 3, 4)

    intersection = torch.sum(probs * targets_one_hot, dims)
    denominator = torch.sum(probs + targets_one_hot, dims)

    dice_score = (2.0 * intersection + eps) / (denominator + eps)
    loss = 1.0 - dice_score.mean()

    return loss


def combined_loss(logits, targets):
    ce = F.cross_entropy(logits, targets)
    dice = dice_loss_multiclass_3d(
        logits,
        targets,
        num_classes=NUM_CLASSES,
        include_background=False
    )

    total = ce + dice
    return total, ce.item(), dice.item()


# ------------------------------------------------------------
# 7. Training helpers
# ------------------------------------------------------------

def train_one_epoch(model, loader, optimizer):
    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_dice = 0.0

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        logits = model(images)

        loss, ce_value, dice_value = combined_loss(logits, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce += ce_value
        total_dice += dice_value

        if (batch_idx + 1) % 20 == 0:
            print(
                f"    batch {batch_idx + 1}/{len(loader)} | "
                f"loss={loss.item():.4f}"
            )

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
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

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
# 8. Main
# ------------------------------------------------------------

def main():
    print("=" * 80)
    print("Script 11: Train clean FLAIR-only 3D U-Net")
    print("=" * 80)

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Missing training CSV: {TRAIN_CSV}")

    df = pd.read_csv(TRAIN_CSV)

    required_cols = ["patient_id", "flair", "seg"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    print(f"Total available training patients: {len(df)}")

    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    split_idx = int(TRAIN_RATIO * len(df))

    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)

    if RUN_MODE == "quick":
        train_df = train_df.iloc[:MAX_TRAIN_PATIENTS_QUICK].reset_index(drop=True)
        val_df = val_df.iloc[:MAX_VAL_PATIENTS_QUICK].reset_index(drop=True)

    print(f"RUN_MODE: {RUN_MODE}")
    print(f"Train patients used: {len(train_df)}")
    print(f"Validation patients used: {len(val_df)}")
    print(f"Patch size: {PATCH_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Base channels: {BASE_CHANNELS}")
    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(f"GPU free memory: {free_bytes / (1024 ** 3):.2f} GB")
        print(f"GPU total memory: {total_bytes / (1024 ** 3):.2f} GB")

    train_dataset = BraTS3DPatchDataset(
        dataframe=train_df,
        patches_per_patient=PATCHES_PER_PATIENT_TRAIN,
        patch_size=PATCH_SIZE,
        mode="train"
    )

    val_dataset = BraTS3DPatchDataset(
        dataframe=val_df,
        patches_per_patient=PATCHES_PER_PATIENT_VAL,
        patch_size=PATCH_SIZE,
        mode="val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = UNet3D(
        in_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        base_channels=BASE_CHANNELS
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    training_log = []

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        print("-" * 80)
        print(f"Epoch {epoch}/{EPOCHS}")

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
            "epoch_time_seconds": epoch_time,
            "run_mode": RUN_MODE,
            "patch_size": str(PATCH_SIZE),
            "base_channels": BASE_CHANNELS,
            "batch_size": BATCH_SIZE,
            "train_patients": len(train_df),
            "val_patients": len(val_df)
        }

        training_log.append(row)

        print(
            f"Epoch {epoch}/{EPOCHS} complete | "
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
                "training_type": "clean_3d_unet_quick",
                "patch_size": PATCH_SIZE,
                "base_channels": BASE_CHANNELS,
                "batch_size": BATCH_SIZE,
                "label_mapping": {
                    "0": "background",
                    "1": "BraTS original label 1",
                    "2": "BraTS original label 2 / edema",
                    "3": "BraTS original label 4"
                },
            }

            torch.save(checkpoint, MODEL_SAVE_PATH)

            print(
                f"Saved new best model at epoch {epoch} "
                f"with val loss {best_val_loss:.4f}"
            )

    total_time = time.time() - start_time

    log_df = pd.DataFrame(training_log)
    log_df.to_csv(LOG_SAVE_PATH, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(log_df["epoch"], log_df["train_loss"], marker="o", label="Training loss")
    plt.plot(log_df["epoch"], log_df["val_loss"], marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Quick Clean FLAIR-only 3D U-Net Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH, dpi=300)
    plt.close()

    print("=" * 80)
    print(f"Training finished in {total_time / 60:.2f} minutes.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Saved model: {MODEL_SAVE_PATH}")
    print(f"Saved training log: {LOG_SAVE_PATH}")
    print(f"Saved loss curve: {PLOT_SAVE_PATH}")
    print("=" * 80)
    print("Script 11 finished.")


if __name__ == "__main__":
    main()
