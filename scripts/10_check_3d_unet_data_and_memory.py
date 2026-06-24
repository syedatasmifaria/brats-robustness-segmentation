# ============================================================
# Script 10: Check 3D U-Net data loading, memory, crop, and patch sampling
# Project: Robustness of Medical Image Segmentation Models
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

import torch


# ------------------------------------------------------------
# 1. Paths and settings
# ------------------------------------------------------------

PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

TRAIN_CSV = PROJECT_ROOT / "data/csvs/train_paths.csv"
TEST_CSV = PROJECT_ROOT / "data/csvs/test_paths.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_SAVE_PATH = RESULTS_DIR / "10_3d_preflight_summary.csv"
PREVIEW_SAVE_PATH = RESULTS_DIR / "10_3d_patch_preview.png"

NUM_CASES_TO_CHECK = 5

# This is only for checking. We are not training yet.
PATCH_SIZE = (96, 96, 96)

BRAIN_THRESHOLD = 0.01
CROP_MARGIN = 8

RANDOM_SEED = 42


# ------------------------------------------------------------
# 2. Helper functions
# ------------------------------------------------------------

def load_nifti(path):
    """
    Load a NIfTI file as a NumPy array.
    """
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def load_seg(path):
    """
    Load segmentation mask.

    We keep it as integer labels because masks are class labels,
    not continuous MRI intensities.
    """
    return np.asanyarray(nib.load(str(path)).dataobj).astype(np.int16)


def normalize_volume(volume):
    """
    Normalize MRI volume to 0-1.

    MRI intensities do not have a fixed range, so normalization makes
    the input easier for a neural network to learn from.
    """
    volume = volume.astype(np.float32)

    min_val = np.min(volume)
    max_val = np.max(volume)

    if max_val - min_val < 1e-8:
        return np.zeros_like(volume, dtype=np.float32)

    return (volume - min_val) / (max_val - min_val)


def remap_segmentation_labels(seg):
    """
    BraTS labels:
        0 = background
        1 = tumor label 1
        2 = edema
        4 = enhancing tumor

    PyTorch multiclass segmentation wants consecutive labels:
        0, 1, 2, 3

    So:
        0 -> 0
        1 -> 1
        2 -> 2
        4 -> 3
    """
    remapped = np.zeros_like(seg, dtype=np.int64)

    remapped[seg == 1] = 1
    remapped[seg == 2] = 2
    remapped[seg == 4] = 3

    return remapped


def estimate_array_size_mb(array):
    """
    Estimate how much RAM one array uses.
    """
    return array.nbytes / (1024 ** 2)


def get_brain_bbox(volume, threshold=0.01, margin=8):
    """
    Find the bounding box around visible brain tissue.

    This removes the large black empty area around the brain.
    That is important for 3D U-Net because empty space wastes GPU memory.
    """
    mask = volume > threshold

    coords = np.argwhere(mask)

    if coords.size == 0:
        return (0, volume.shape[0], 0, volume.shape[1], 0, volume.shape[2])

    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0) + 1

    x_min = max(x_min - margin, 0)
    y_min = max(y_min - margin, 0)
    z_min = max(z_min - margin, 0)

    x_max = min(x_max + margin, volume.shape[0])
    y_max = min(y_max + margin, volume.shape[1])
    z_max = min(z_max + margin, volume.shape[2])

    return (x_min, x_max, y_min, y_max, z_min, z_max)


def crop_with_bbox(array, bbox):
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    return array[x_min:x_max, y_min:y_max, z_min:z_max]


def get_tumor_center(seg):
    """
    Get the center of tumor voxels.

    If there is no tumor, return the center of the whole volume.
    For BraTS training data, there should be tumor.
    """
    tumor_coords = np.argwhere(seg > 0)

    if tumor_coords.size == 0:
        return np.array(seg.shape) // 2

    return tumor_coords.mean(axis=0).astype(int)


def sample_patch_around_center(volume, seg, center, patch_size):
    """
    Sample a fixed-size 3D patch around a chosen center.

    This is how 3D U-Net training usually becomes manageable:
    instead of feeding the full 240x240x155 volume, we feed smaller 3D patches.
    """
    sx, sy, sz = patch_size
    x, y, z = center

    x_start = x - sx // 2
    y_start = y - sy // 2
    z_start = z - sz // 2

    x_start = max(0, min(x_start, volume.shape[0] - sx))
    y_start = max(0, min(y_start, volume.shape[1] - sy))
    z_start = max(0, min(z_start, volume.shape[2] - sz))

    x_end = x_start + sx
    y_end = y_start + sy
    z_end = z_start + sz

    image_patch = volume[x_start:x_end, y_start:y_end, z_start:z_end]
    seg_patch = seg[x_start:x_end, y_start:y_end, z_start:z_end]

    return image_patch, seg_patch, (x_start, x_end, y_start, y_end, z_start, z_end)


def save_patch_preview(full_volume, full_seg, cropped_volume, cropped_seg, patch, patch_seg, save_path):
    """
    Save a visual preview.

    We show:
        1. Full volume middle tumor slice
        2. Cropped volume middle tumor slice
        3. 3D patch middle slice
    """
    full_center = get_tumor_center(full_seg)
    crop_center = get_tumor_center(cropped_seg)
    patch_center = get_tumor_center(patch_seg)

    full_z = int(full_center[2])
    crop_z = int(crop_center[2])
    patch_z = int(patch_center[2])

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].imshow(full_volume[:, :, full_z], cmap="gray")
    axes[0, 0].set_title(f"Full FLAIR\nz={full_z}")
    axes[0, 0].axis("off")

    axes[1, 0].imshow(full_seg[:, :, full_z], vmin=0, vmax=3)
    axes[1, 0].set_title("Full SEG")
    axes[1, 0].axis("off")

    axes[0, 1].imshow(cropped_volume[:, :, crop_z], cmap="gray")
    axes[0, 1].set_title(f"Cropped FLAIR\nz={crop_z}")
    axes[0, 1].axis("off")

    axes[1, 1].imshow(cropped_seg[:, :, crop_z], vmin=0, vmax=3)
    axes[1, 1].set_title("Cropped SEG")
    axes[1, 1].axis("off")

    axes[0, 2].imshow(patch[:, :, patch_z], cmap="gray")
    axes[0, 2].set_title(f"3D Patch FLAIR\nz={patch_z}")
    axes[0, 2].axis("off")

    axes[1, 2].imshow(patch_seg[:, :, patch_z], vmin=0, vmax=3)
    axes[1, 2].set_title("3D Patch SEG")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ------------------------------------------------------------
# 3. Main
# ------------------------------------------------------------

def main():
    print("=" * 80)
    print("Script 10: 3D U-Net data and memory preflight")
    print("=" * 80)

    print(f"Train CSV: {TRAIN_CSV}")
    print(f"Test CSV: {TEST_CSV}")

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Missing train CSV: {TRAIN_CSV}")

    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Missing test CSV: {TEST_CSV}")

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    print(f"Train patients: {len(train_df)}")
    print(f"Test patients: {len(test_df)}")
    print(f"CSV columns: {list(train_df.columns)}")

    required_cols = ["patient_id", "flair", "seg"]

    for col in required_cols:
        if col not in train_df.columns:
            raise ValueError(f"Required column missing from train CSV: {col}")

    print("-" * 80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU 0 name: {torch.cuda.get_device_name(0)}")
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(f"GPU 0 free memory: {free_bytes / (1024 ** 3):.2f} GB")
        print(f"GPU 0 total memory: {total_bytes / (1024 ** 3):.2f} GB")

    rng = np.random.default_rng(RANDOM_SEED)

    if len(train_df) < NUM_CASES_TO_CHECK:
        selected_indices = np.arange(len(train_df))
    else:
        selected_indices = rng.choice(len(train_df), size=NUM_CASES_TO_CHECK, replace=False)

    selected_indices = sorted(selected_indices)

    rows = []

    preview_saved = False

    print("-" * 80)
    print(f"Checking {len(selected_indices)} training cases...")

    for idx in selected_indices:
        row = train_df.iloc[idx]

        patient_id = row["patient_id"]
        flair_path = Path(row["flair"])
        seg_path = Path(row["seg"])

        print("-" * 80)
        print(f"Patient: {patient_id}")
        print(f"FLAIR: {flair_path}")
        print(f"SEG:   {seg_path}")

        if not flair_path.exists():
            raise FileNotFoundError(f"Missing FLAIR file: {flair_path}")

        if not seg_path.exists():
            raise FileNotFoundError(f"Missing SEG file: {seg_path}")

        flair = load_nifti(flair_path)
        seg_original = load_seg(seg_path)

        flair_norm = normalize_volume(flair)
        seg = remap_segmentation_labels(seg_original)

        original_labels = np.unique(seg_original)
        remapped_labels = np.unique(seg)

        bbox = get_brain_bbox(
            flair_norm,
            threshold=BRAIN_THRESHOLD,
            margin=CROP_MARGIN
        )

        cropped_flair = crop_with_bbox(flair_norm, bbox)
        cropped_seg = crop_with_bbox(seg, bbox)

        tumor_center = get_tumor_center(cropped_seg)

        patch, patch_seg, patch_bbox = sample_patch_around_center(
            cropped_flair,
            cropped_seg,
            tumor_center,
            PATCH_SIZE
        )

        tumor_voxels_full = int(np.sum(seg > 0))
        tumor_voxels_crop = int(np.sum(cropped_seg > 0))
        tumor_voxels_patch = int(np.sum(patch_seg > 0))

        print(f"Original FLAIR shape: {flair.shape}")
        print(f"Original SEG labels: {original_labels}")
        print(f"Remapped SEG labels: {remapped_labels}")
        print(f"Full FLAIR memory: {estimate_array_size_mb(flair):.2f} MB")
        print(f"Full SEG memory: {estimate_array_size_mb(seg):.2f} MB")
        print(f"Brain crop bbox: {bbox}")
        print(f"Cropped shape: {cropped_flair.shape}")
        print(f"Patch size: {patch.shape}")
        print(f"Tumor voxels full: {tumor_voxels_full}")
        print(f"Tumor voxels cropped: {tumor_voxels_crop}")
        print(f"Tumor voxels patch: {tumor_voxels_patch}")

        rows.append(
            {
                "patient_id": patient_id,
                "full_shape": str(flair.shape),
                "cropped_shape": str(cropped_flair.shape),
                "patch_shape": str(patch.shape),
                "original_labels": str(list(original_labels)),
                "remapped_labels": str(list(remapped_labels)),
                "flair_memory_mb": estimate_array_size_mb(flair),
                "seg_memory_mb": estimate_array_size_mb(seg),
                "tumor_voxels_full": tumor_voxels_full,
                "tumor_voxels_cropped": tumor_voxels_crop,
                "tumor_voxels_patch": tumor_voxels_patch,
                "brain_bbox": str(bbox),
                "patch_bbox": str(patch_bbox),
            }
        )

        if not preview_saved:
            save_patch_preview(
                full_volume=flair_norm,
                full_seg=seg,
                cropped_volume=cropped_flair,
                cropped_seg=cropped_seg,
                patch=patch,
                patch_seg=patch_seg,
                save_path=PREVIEW_SAVE_PATH
            )

            preview_saved = True

            print(f"Saved preview figure: {PREVIEW_SAVE_PATH}")

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(SUMMARY_SAVE_PATH, index=False)

    print("=" * 80)
    print(f"Saved preflight summary: {SUMMARY_SAVE_PATH}")
    print(f"Saved preview figure: {PREVIEW_SAVE_PATH}")
    print("-" * 80)
    print("Recommended next training strategy if this check looks good:")
    print(f"Use 3D patches of size {PATCH_SIZE}, sampled from cropped brain volumes.")
    print("Train on clean FLAIR first, then test degraded versions on-the-fly later.")
    print("=" * 80)
    print("Script 10 finished.")


if __name__ == "__main__":
    main()