"""
Script 06: Train a clean 2D U-Net baseline using FLAIR slices.

Goal:
- Load selected 2D FLAIR slices from train_2d_slices_flair.csv
- Load matching segmentation masks
- Train a simple 2D U-Net
- Save the trained model

Important:
- This trains only on clean images.
- This does NOT use degraded images.
- This is the first baseline model.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# -----------------------------
# Paths
# -----------------------------
PROJECT_DIR = Path("/home/xfh25/brats_segmentation_project")

SLICE_CSV = PROJECT_DIR / "data" / "csvs" / "train_2d_slices_flair.csv"

MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "2d_unet_flair_clean.pth"


# -----------------------------
# Training settings
# -----------------------------
RANDOM_SEED = 42

IMAGE_SIZE = 240
NUM_CLASSES = 4

BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4

TRAIN_FRACTION = 0.8
NUM_WORKERS = 2


# -----------------------------
# Reproducibility
# -----------------------------
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


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

    Original:
    0 = background
    1 = necrotic/non-enhancing tumor core
    2 = edema
    4 = enhancing tumor

    Remapped:
    0 = background
    1 = original label 1
    2 = original label 2
    3 = original label 4
    """
    mask = mask.astype(np.int64)

    remapped = np.zeros_like(mask, dtype=np.int64)
    remapped[mask == 1] = 1
    remapped[mask == 2] = 2
    remapped[mask == 4] = 3

    return remapped


# -----------------------------
# Dataset
# -----------------------------
class BraTS2DSliceDataset(Dataset):
    """
    Dataset for loading 2D FLAIR slices and segmentation masks.
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

        # Shape for PyTorch:
        # image: [channels, height, width]
        # mask:  [height, width]
        flair_tensor = torch.tensor(flair_slice, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(seg_slice, dtype=torch.long)

        return flair_tensor, mask_tensor


# -----------------------------
# Simple 2D U-Net
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


# -----------------------------
# Training and validation loops
# -----------------------------
def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()

    total_loss = 0.0

    for images, masks in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_one_epoch(model, dataloader, loss_fn, device):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    print("=" * 80)
    print("Script 06: Train clean 2D U-Net baseline")
    print("=" * 80)

    if not SLICE_CSV.exists():
        raise FileNotFoundError(f"Could not find slice CSV: {SLICE_CSV}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    print(f"\nLoading dataset from:\n{SLICE_CSV}")

    full_dataset = BraTS2DSliceDataset(SLICE_CSV)

    print(f"Total selected 2D slices: {len(full_dataset)}")

    train_size = int(TRAIN_FRACTION * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    print(f"Training slices: {len(train_dataset)}")
    print(f"Validation slices: {len(val_dataset)}")

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

    model = UNet2D(
        in_channels=1,
        num_classes=NUM_CLASSES
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("\nStarting training...")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        print("\n" + "-" * 80)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 80)

        train_loss = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device
        )

        val_loss = validate_one_epoch(
            model,
            val_loader,
            loss_fn,
            device
        )

        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss:   {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "num_classes": NUM_CLASSES,
                "input_modality": "FLAIR",
                "training_type": "clean"
            }, MODEL_PATH)

            print(f"Saved best model to:\n{MODEL_PATH}")

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved at:\n{MODEL_PATH}")


if __name__ == "__main__":
    main()