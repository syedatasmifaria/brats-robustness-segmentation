"""
Script 13: Check 3D multimodal BraTS data loading and GPU memory

Goal:
- Move from FLAIR-only input to 4-modal BraTS input:
    1. FLAIR
    2. T1
    3. T1ce
    4. T2

- Load a few training patients.
- Normalize each modality separately.
- Crop all modalities and segmentation together around the brain.
- Sample a 3D patch.
- Stack modalities into shape:
    (4, 96, 96, 96)

- Send one batch through a 4-channel 3D U-Net.
- Run one dummy loss/backward step to check training memory.
- Save:
    1. Summary CSV
    2. Multimodal patch preview image

Important:
- This script does NOT train a model.
- This script does NOT save a model.
- This script does NOT apply degradation.
- This is a safety check before full 4-modal 3D U-Net training.
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

TRAIN_CSV = os.path.join(PROJECT_ROOT, "data/csvs/train_paths.csv")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(RESULTS_DIR, "13_multimodal_3d_preflight_summary.csv")
PREVIEW_PNG = os.path.join(RESULTS_DIR, "13_multimodal_3d_patch_preview.png")

MODALITIES = ["flair", "t1", "t1ce", "t2"]

PATCH_SIZE = (96, 96, 96)

# We only check a few patients. This is a preflight, not training.
NUM_PATIENTS_TO_CHECK = 5

IN_CHANNELS = 4
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
# 3. 3D U-Net architecture for memory check
# ============================================================
# This follows the same naming style we recovered from Script 11:
# input_conv, down1, down2, down3, up1, up2, up3, output_conv.
# Difference:
# This version uses 4 input channels instead of 1.

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

        # Safety handling in case dimensions differ slightly
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
    def __init__(self, in_channels=4, out_classes=4, base_channels=16):
        super().__init__()

        self.input_conv = DoubleConv3D(in_channels, base_channels)

        self.down1 = Down3D(base_channels, base_channels * 2)
        self.down2 = Down3D(base_channels * 2, base_channels * 4)
        self.down3 = Down3D(base_channels * 4, base_channels * 8)

        # Deepest decoder block
        self.up1 = Up3D(
            in_channels=base_channels * 8,
            skip_channels=base_channels * 4,
            out_channels=base_channels * 4
        )

        # Middle decoder block
        self.up2 = Up3D(
            in_channels=base_channels * 4,
            skip_channels=base_channels * 2,
            out_channels=base_channels * 2
        )

        # Final shallow decoder block
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


def normalize_modality(volume):
    """
    Normalize one MRI modality using nonzero brain voxels.
    Background remains 0.

    We normalize each modality separately because FLAIR, T1, T1ce,
    and T2 have different intensity distributions.
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
    Find one shared brain bounding box using all modalities.

    This is important because all four modalities and the segmentation
    must stay spatially aligned.
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
    """
    Crop all modalities and segmentation using the same bounding box.
    """
    bbox = get_union_brain_bbox(modality_volumes)

    if bbox is None:
        return modality_volumes, seg

    x_min, x_max, y_min, y_max, z_min, z_max = bbox

    cropped_modalities = []

    for volume in modality_volumes:
        cropped = volume[x_min:x_max, y_min:y_max, z_min:z_max]
        cropped_modalities.append(cropped)

    seg_crop = seg[x_min:x_max, y_min:y_max, z_min:z_max]

    return cropped_modalities, seg_crop


def pad_3d_if_needed(volume, patch_size, pad_value=0):
    """
    If a 3D volume is smaller than patch size, pad it.
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


def sample_multimodal_patch(modality_volumes, seg, patch_size, force_tumor=True):
    """
    Sample one 3D patch from all four modalities and the segmentation.

    Output image patch shape:
        (4, 96, 96, 96)

    Output segmentation patch shape:
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

    modality_patches = []

    for volume in modality_volumes:
        patch = volume[x_start:x_end, y_start:y_end, z_start:z_end]
        modality_patches.append(patch)

    image_patch = np.stack(modality_patches, axis=0).astype(np.float32)
    seg_patch = seg[x_start:x_end, y_start:y_end, z_start:z_end].astype(np.int64)

    return image_patch, seg_patch, (x_start, y_start, z_start)


def get_gpu_memory_info():
    """
    Return current GPU memory information in MB.
    """
    if not torch.cuda.is_available():
        return {
            "gpu_available": False,
            "gpu_name": "none",
            "gpu_free_mb": np.nan,
            "gpu_total_mb": np.nan,
            "gpu_allocated_mb": np.nan,
            "gpu_reserved_mb": np.nan,
            "gpu_peak_allocated_mb": np.nan,
        }

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    allocated_bytes = torch.cuda.memory_allocated()
    reserved_bytes = torch.cuda.memory_reserved()
    peak_allocated_bytes = torch.cuda.max_memory_allocated()

    return {
        "gpu_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_free_mb": free_bytes / (1024 ** 2),
        "gpu_total_mb": total_bytes / (1024 ** 2),
        "gpu_allocated_mb": allocated_bytes / (1024 ** 2),
        "gpu_reserved_mb": reserved_bytes / (1024 ** 2),
        "gpu_peak_allocated_mb": peak_allocated_bytes / (1024 ** 2),
    }


def save_multimodal_preview(image_patch, seg_patch, patient_id, save_path):
    """
    Save one preview image showing the four modalities and segmentation mask.
    """
    tumor_per_slice = (seg_patch > 0).sum(axis=(0, 1))

    if tumor_per_slice.max() > 0:
        z = int(np.argmax(tumor_per_slice))
    else:
        z = seg_patch.shape[2] // 2

    modality_titles = ["FLAIR", "T1", "T1ce", "T2"]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    for i in range(4):
        axes[i].imshow(image_patch[i, :, :, z], cmap="gray")
        axes[i].set_title(modality_titles[i])
        axes[i].axis("off")

    axes[4].imshow(seg_patch[:, :, z], cmap="viridis", vmin=0, vmax=3)
    axes[4].set_title("SEG mask")
    axes[4].axis("off")

    fig.suptitle(
        f"4-modal 3D patch preview: {patient_id}, slice z={z}",
        fontsize=12
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def run_gpu_memory_check(image_patch, seg_patch, device):
    """
    Send one 4-channel patch through the 3D U-Net and run one backward pass.
    This checks whether the intended training setup fits in GPU memory.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping GPU memory check.")
        return {
            "memory_check_done": False,
            "forward_success": False,
            "backward_success": False,
            "dummy_loss": np.nan,
            "output_shape": "none",
        }

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = UNet3D(
        in_channels=IN_CHANNELS,
        out_classes=OUT_CLASSES,
        base_channels=BASE_CHANNELS
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    x = torch.from_numpy(image_patch).float().unsqueeze(0).to(device)
    y = torch.from_numpy(seg_patch).long().unsqueeze(0).to(device)

    model.train()

    optimizer.zero_grad()

    forward_success = False
    backward_success = False
    dummy_loss_value = np.nan
    output_shape = "none"

    try:
        logits = model(x)
        output_shape = str(tuple(logits.shape))
        forward_success = True

        loss = criterion(logits, y)
        dummy_loss_value = float(loss.item())

        loss.backward()
        optimizer.step()

        backward_success = True

    except RuntimeError as e:
        print("\nGPU memory/model check failed.")
        print("PyTorch RuntimeError:")
        print(e)

        # Clean up as much as possible
        del model
        torch.cuda.empty_cache()

        return {
            "memory_check_done": True,
            "forward_success": forward_success,
            "backward_success": backward_success,
            "dummy_loss": dummy_loss_value,
            "output_shape": output_shape,
        }

    del model
    torch.cuda.empty_cache()

    return {
        "memory_check_done": True,
        "forward_success": forward_success,
        "backward_success": backward_success,
        "dummy_loss": dummy_loss_value,
        "output_shape": output_shape,
    }


# ============================================================
# 5. Main function
# ============================================================

def main():
    print("=" * 80)
    print("Script 13: Check 3D multimodal BraTS data and GPU memory")
    print("=" * 80)

    print("\nChecking files...")

    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"Training CSV not found: {TRAIN_CSV}")

    print(f"Training CSV: {TRAIN_CSV}")

    train_df = pd.read_csv(TRAIN_CSV)

    required_cols = ["patient_id", "flair", "t1", "t1ce", "t2", "seg"]

    for col in required_cols:
        if col not in train_df.columns:
            raise ValueError(
                f"Column '{col}' not found in train CSV. "
                f"Available columns: {list(train_df.columns)}"
            )

    print(f"Total training patients in CSV: {len(train_df)}")
    print(f"Patients checked in this script: {NUM_PATIENTS_TO_CHECK}")
    print(f"Modalities: {MODALITIES}")
    print(f"Patch size: {PATCH_SIZE}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    gpu_before = get_gpu_memory_info()

    summary_rows = []
    preview_saved = False
    first_valid_image_patch = None
    first_valid_seg_patch = None

    subset_df = train_df.head(NUM_PATIENTS_TO_CHECK).copy()

    for local_idx, (_, row) in enumerate(subset_df.iterrows(), start=1):
        patient_id = row["patient_id"]

        print(f"\nChecking patient {local_idx}/{len(subset_df)}: {patient_id}")

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

        raw_modalities = []

        for modality in MODALITIES:
            volume = load_nifti(paths[modality])
            raw_modalities.append(volume)

        seg = load_nifti(paths["seg"])
        seg = remap_segmentation_labels(seg)

        original_shapes = [volume.shape for volume in raw_modalities]
        seg_shape = seg.shape

        print(f"  Original modality shapes: {original_shapes}")
        print(f"  Original SEG shape:       {seg_shape}")

        # Check shape alignment
        all_shapes = original_shapes + [seg_shape]
        shapes_match = len(set(all_shapes)) == 1

        if not shapes_match:
            warnings.warn(
                f"Shapes do not match for {patient_id}: {all_shapes}. Skipping."
            )
            continue

        # Crop before normalization using raw nonzero brain area
        cropped_modalities, seg_crop = crop_modalities_and_seg(raw_modalities, seg)

        cropped_shape = seg_crop.shape
        print(f"  Cropped shape:            {cropped_shape}")

        # Normalize each modality separately after cropping
        normalized_modalities = [
            normalize_modality(volume) for volume in cropped_modalities
        ]

        # Pad each modality and segmentation if needed
        padded_modalities = [
            pad_3d_if_needed(volume, PATCH_SIZE, pad_value=0)
            for volume in normalized_modalities
        ]

        seg_padded = pad_3d_if_needed(seg_crop, PATCH_SIZE, pad_value=0)

        padded_shape = seg_padded.shape
        print(f"  Padded shape:             {padded_shape}")

        image_patch, seg_patch, patch_start = sample_multimodal_patch(
            padded_modalities,
            seg_padded,
            PATCH_SIZE,
            force_tumor=True
        )

        unique_labels = sorted(np.unique(seg_patch).astype(int).tolist())
        tumor_voxels = int((seg_patch > 0).sum())

        print(f"  Image patch shape:        {image_patch.shape}")
        print(f"  SEG patch shape:          {seg_patch.shape}")
        print(f"  Patch start:              {patch_start}")
        print(f"  Patch labels:             {unique_labels}")
        print(f"  Tumor voxels in patch:    {tumor_voxels}")

        # Save first valid patch for preview and memory check
        if first_valid_image_patch is None:
            first_valid_image_patch = image_patch
            first_valid_seg_patch = seg_patch

        if not preview_saved:
            save_multimodal_preview(
                image_patch=image_patch,
                seg_patch=seg_patch,
                patient_id=patient_id,
                save_path=PREVIEW_PNG
            )
            preview_saved = True
            print(f"  Preview saved to:         {PREVIEW_PNG}")

        summary_rows.append({
            "patient_id": patient_id,
            "modalities": ",".join(MODALITIES),
            "original_shape": str(original_shapes[0]),
            "seg_shape": str(seg_shape),
            "shapes_match": shapes_match,
            "cropped_shape": str(cropped_shape),
            "padded_shape": str(padded_shape),
            "patch_size": str(PATCH_SIZE),
            "image_patch_shape": str(image_patch.shape),
            "seg_patch_shape": str(seg_patch.shape),
            "patch_start": str(patch_start),
            "patch_unique_labels": str(unique_labels),
            "patch_tumor_voxels": tumor_voxels,
        })

    if first_valid_image_patch is None:
        raise RuntimeError("No valid multimodal patch was created. Check paths/data.")

    print("\nRunning GPU memory check with one 4-channel patch...")
    memory_check = run_gpu_memory_check(
        image_patch=first_valid_image_patch,
        seg_patch=first_valid_seg_patch,
        device=device
    )

    gpu_after = get_gpu_memory_info()

    print("\nGPU memory check result:")
    for key, value in memory_check.items():
        print(f"  {key}: {value}")

    print("\nGPU before memory check:")
    for key, value in gpu_before.items():
        print(f"  before_{key}: {value}")

    print("\nGPU after memory check:")
    for key, value in gpu_after.items():
        print(f"  after_{key}: {value}")

    # Add memory check info to every row for easy record keeping
    for row in summary_rows:
        row.update({
            "in_channels": IN_CHANNELS,
            "out_classes": OUT_CLASSES,
            "base_channels": BASE_CHANNELS,
            "memory_check_done": memory_check["memory_check_done"],
            "forward_success": memory_check["forward_success"],
            "backward_success": memory_check["backward_success"],
            "dummy_loss": memory_check["dummy_loss"],
            "model_output_shape": memory_check["output_shape"],
            "gpu_name": gpu_after["gpu_name"],
            "gpu_total_mb": gpu_after["gpu_total_mb"],
            "gpu_free_mb_after": gpu_after["gpu_free_mb"],
            "gpu_peak_allocated_mb": gpu_after["gpu_peak_allocated_mb"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print("\nSaved outputs:")
    print(f"  Summary CSV: {SUMMARY_CSV}")
    print(f"  Preview PNG: {PREVIEW_PNG}")

    print("\nDone.")
    print("=" * 80)


if __name__ == "__main__":
    main()