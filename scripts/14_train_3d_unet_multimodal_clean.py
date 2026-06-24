"""
Script 14: Quick clean training of 4-modal 3D U-Net

Purpose:
- Train a quick 4-modal 3D U-Net sanity-check model.
- Inputs: FLAIR + T1 + T1ce + T2
- Output: 4 segmentation classes [0, 1, 2, 3]
- Training data: clean only
- No degradation is applied.

This is NOT final training yet.
This checks that the 4-modal training pipeline works before full training.

Outputs:
- models/3d_unet_multimodal_clean_quick_best.pth
- models/3d_unet_multimodal_clean_quick_last.pth
- results/14_quick_multimodal_3d_training_log.csv
- results/14_quick_multimodal_3d_training_loss_curve.png
"""

import os
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


# ============================================================
# 1. Settings
# ============================================================

PROJECT_ROOT = "/home/xfh25/brats_segmentation_project"

TRAIN_CSV = os.path.join(PROJECT_ROOT, "data/csvs/train_paths.csv")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "3d_unet_multimodal_clean_quick_best.pth")
LAST_MODEL_PATH = os.path.join(MODELS_DIR, "3d_unet_multimodal_clean_quick_last.pth")

LOG_CSV = os.path.join(RESULTS_DIR, "14_quick_multimodal_3d_training_log.csv")
LOSS_CURVE_PNG = os.path.join(RESULTS_DIR, "14_quick_multimodal_3d_training_loss_curve.png")

MODALITIES = ["flair", "t1", "t1ce", "t2"]

PATCH_SIZE = (96, 96, 96)

IN_CHANNELS = 4
OUT_CLASSES = 4
BASE_CHANNELS = 16

EPOCHS = 2
BATCH_SIZE = 1
LEARNING_RATE = 1e-4

QUICK_TRAIN_PATIENTS = 40
QUICK_VAL_PATIENTS = 10

TRAIN_TUMOR_PATCH_PROB = 0.80
VAL_FORCE_TUMOR = True

NUM_WORKERS = 0
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
    seg = seg.astype(np.int64)
    seg[seg == 4] = 3
    return seg


def normalize_modality(volume):
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


def sample_multimodal_patch(modality_volumes, seg, patch_size, force_tumor=True, rng=None):
    if rng is None:
        rng = np.random

    px, py, pz = patch_size
    sx, sy, sz = seg.shape

    if force_tumor and np.any(seg > 0):
        tumor_coords = np.array(np.where(seg > 0)).T
        center = tumor_coords[rng.choice(len(tumor_coords))]
        cx, cy, cz = center

        x_start = int(cx - px // 2)
        y_start = int(cy - py // 2)
        z_start = int(cz - pz // 2)
    else:
        x_start = rng.randint(0, sx - px + 1)
        y_start = rng.randint(0, sy - py + 1)
        z_start = rng.randint(0, sz - pz + 1)

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

    return image_patch, seg_patch


# ============================================================
# 5. Dataset
# ============================================================

class BraTSMultimodalPatchDataset(Dataset):
    def __init__(
        self,
        dataframe,
        modalities,
        patch_size,
        mode,
        tumor_patch_probability=0.80,
        deterministic=False,
        base_seed=42
    ):
        self.df = dataframe.reset_index(drop=True)
        self.modalities = modalities
        self.patch_size = patch_size
        self.mode = mode
        self.tumor_patch_probability = tumor_patch_probability
        self.deterministic = deterministic
        self.base_seed = base_seed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.deterministic:
            rng = np.random.RandomState(self.base_seed + idx)
        else:
            rng = np.random

        raw_modalities = [
            load_nifti(row[modality], dtype=np.float32)
            for modality in self.modalities
        ]

        seg = load_nifti(row["seg"], dtype=np.int16)
        seg = remap_segmentation_labels(seg)

        cropped_modalities, seg_crop = crop_modalities_and_seg(raw_modalities, seg)

        normalized_modalities = [
            normalize_modality(volume)
            for volume in cropped_modalities
        ]

        padded_modalities = [
            pad_3d_if_needed(volume, self.patch_size, pad_value=0)
            for volume in normalized_modalities
        ]

        seg_padded = pad_3d_if_needed(seg_crop, self.patch_size, pad_value=0)

        if self.mode == "train":
            force_tumor = rng.rand() < self.tumor_patch_probability
        else:
            force_tumor = VAL_FORCE_TUMOR

        image_patch, seg_patch = sample_multimodal_patch(
            padded_modalities,
            seg_padded,
            self.patch_size,
            force_tumor=force_tumor,
            rng=rng
        )

        image_tensor = torch.from_numpy(image_patch).float()
        seg_tensor = torch.from_numpy(seg_patch).long()

        return image_tensor, seg_tensor


# ============================================================
# 6. Loss
# ============================================================

def multiclass_dice_loss(logits, targets, eps=1e-6):
    probs = torch.softmax(logits, dim=1)

    targets_one_hot = F.one_hot(
        targets,
        num_classes=OUT_CLASSES
    ).permute(0, 4, 1, 2, 3).float()

    probs_fg = probs[:, 1:, :, :, :]
    targets_fg = targets_one_hot[:, 1:, :, :, :]

    dims = (0, 2, 3, 4)

    intersection = torch.sum(probs_fg * targets_fg, dim=dims)
    denominator = torch.sum(probs_fg + targets_fg, dim=dims)

    dice = (2.0 * intersection + eps) / (denominator + eps)

    return 1.0 - dice.mean()


def combined_loss(logits, targets, ce_loss_fn):
    ce = ce_loss_fn(logits, targets)
    dice = multiclass_dice_loss(logits, targets)
    total = ce + dice

    return total, ce, dice


# ============================================================
# 7. Training helpers
# ============================================================

def run_train_epoch(model, loader, optimizer, ce_loss_fn, device, epoch):
    model.train()

    total_sum = 0.0
    ce_sum = 0.0
    dice_sum = 0.0

    for batch_idx, (images, masks) in enumerate(loader, start=1):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss, ce, dice = combined_loss(logits, masks, ce_loss_fn)

        loss.backward()
        optimizer.step()

        total_sum += loss.item()
        ce_sum += ce.item()
        dice_sum += dice.item()

        if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == len(loader):
            print(
                f"  Train batch {batch_idx}/{len(loader)} | "
                f"loss={loss.item():.4f}, CE={ce.item():.4f}, DiceLoss={dice.item():.4f}"
            )

            if batch_idx == 1:
                print(f"    image shape:  {tuple(images.shape)}")
                print(f"    mask shape:   {tuple(masks.shape)}")
                print(f"    logits shape: {tuple(logits.shape)}")

    n = len(loader)
    return total_sum / n, ce_sum / n, dice_sum / n


def run_val_epoch(model, loader, ce_loss_fn, device):
    model.eval()

    total_sum = 0.0
    ce_sum = 0.0
    dice_sum = 0.0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loader, start=1):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)

            loss, ce, dice = combined_loss(logits, masks, ce_loss_fn)

            total_sum += loss.item()
            ce_sum += ce.item()
            dice_sum += dice.item()

            if batch_idx == 1 or batch_idx == len(loader):
                print(
                    f"  Val batch {batch_idx}/{len(loader)} | "
                    f"loss={loss.item():.4f}, CE={ce.item():.4f}, DiceLoss={dice.item():.4f}"
                )

    n = len(loader)
    return total_sum / n, ce_sum / n, dice_sum / n


def save_checkpoint(path, model, optimizer, epoch, best_val_loss):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "settings": {
            "modalities": MODALITIES,
            "patch_size": PATCH_SIZE,
            "in_channels": IN_CHANNELS,
            "out_classes": OUT_CLASSES,
            "base_channels": BASE_CHANNELS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "quick_train_patients": QUICK_TRAIN_PATIENTS,
            "quick_val_patients": QUICK_VAL_PATIENTS,
            "training_data": "clean only",
            "degradation_used_in_training": False,
        }
    }

    torch.save(checkpoint, path)


def save_loss_curve(log_df):
    plt.figure(figsize=(8, 5))
    plt.plot(log_df["epoch"], log_df["train_loss"], marker="o", label="Train loss")
    plt.plot(log_df["epoch"], log_df["val_loss"], marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Quick 4-modal 3D U-Net clean training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PNG, dpi=200)
    plt.close()


# ============================================================
# 8. Main
# ============================================================

def main():
    print("=" * 80)
    print("Script 14: Quick clean training of 4-modal 3D U-Net")
    print("=" * 80)

    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"Training CSV not found: {TRAIN_CSV}")

    df = pd.read_csv(TRAIN_CSV)

    required_cols = ["patient_id", "flair", "t1", "t1ce", "t2", "seg"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    print(f"Total patients in train CSV: {len(df)}")

    shuffled_df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    train_df = shuffled_df.iloc[:QUICK_TRAIN_PATIENTS].copy()
    val_df = shuffled_df.iloc[
        QUICK_TRAIN_PATIENTS:QUICK_TRAIN_PATIENTS + QUICK_VAL_PATIENTS
    ].copy()

    print(f"Quick train patients: {len(train_df)}")
    print(f"Quick val patients:   {len(val_df)}")
    print(f"Modalities:           {MODALITIES}")
    print(f"Patch size:           {PATCH_SIZE}")
    print(f"Epochs:               {EPOCHS}")
    print(f"Batch size:           {BATCH_SIZE}")
    print("Clean training only:  True")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(f"GPU free memory MB:  {free_bytes / (1024 ** 2):.1f}")
        print(f"GPU total memory MB: {total_bytes / (1024 ** 2):.1f}")

    train_dataset = BraTSMultimodalPatchDataset(
        dataframe=train_df,
        modalities=MODALITIES,
        patch_size=PATCH_SIZE,
        mode="train",
        tumor_patch_probability=TRAIN_TUMOR_PATCH_PROB,
        deterministic=False,
        base_seed=RANDOM_SEED
    )

    val_dataset = BraTSMultimodalPatchDataset(
        dataframe=val_df,
        modalities=MODALITIES,
        patch_size=PATCH_SIZE,
        mode="val",
        tumor_patch_probability=1.0,
        deterministic=True,
        base_seed=RANDOM_SEED + 1000
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    print(f"\nTrain batches per epoch: {len(train_loader)}")
    print(f"Val batches per epoch:   {len(val_loader)}")

    model = UNet3D(
        in_channels=IN_CHANNELS,
        out_classes=OUT_CLASSES,
        base_channels=BASE_CHANNELS
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    ce_loss_fn = nn.CrossEntropyLoss()

    training_log = []
    best_val_loss = float("inf")

    print("\nStarting quick training...")
    print("-" * 80)

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 80)

        train_loss, train_ce, train_dice = run_train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            ce_loss_fn=ce_loss_fn,
            device=device,
            epoch=epoch
        )

        val_loss, val_ce, val_dice = run_val_epoch(
            model=model,
            loader=val_loader,
            ce_loss_fn=ce_loss_fn,
            device=device
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
            "run_mode": "quick",
            "modalities": "+".join(MODALITIES),
            "patch_size": str(PATCH_SIZE),
            "base_channels": BASE_CHANNELS,
            "batch_size": BATCH_SIZE,
            "train_patients": len(train_df),
            "val_patients": len(val_df),
            "clean_training_only": True,
        }

        training_log.append(row)

        log_df = pd.DataFrame(training_log)
        log_df.to_csv(LOG_CSV, index=False)
        save_loss_curve(log_df)

        save_checkpoint(
            path=LAST_MODEL_PATH,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_loss=best_val_loss
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                path=BEST_MODEL_PATH,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss
            )
            print(f"  New best model saved: {BEST_MODEL_PATH}")

        print("\nEpoch summary:")
        print(f"  train_loss:       {train_loss:.4f}")
        print(f"  val_loss:         {val_loss:.4f}")
        print(f"  train_CE:         {train_ce:.4f}")
        print(f"  val_CE:           {val_ce:.4f}")
        print(f"  train_DiceLoss:   {train_dice:.4f}")
        print(f"  val_DiceLoss:     {val_dice:.4f}")
        print(f"  epoch_time_sec:   {epoch_time:.1f}")
        print(f"  Last checkpoint:  {LAST_MODEL_PATH}")
        print(f"  Log CSV:          {LOG_CSV}")
        print(f"  Loss curve:       {LOSS_CURVE_PNG}")

    print("\nQuick training complete.")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model: {BEST_MODEL_PATH}")
    print(f"Last model: {LAST_MODEL_PATH}")
    print(f"Log CSV:    {LOG_CSV}")
    print(f"Curve PNG:  {LOSS_CURVE_PNG}")
    print("=" * 80)


if __name__ == "__main__":
    main()
