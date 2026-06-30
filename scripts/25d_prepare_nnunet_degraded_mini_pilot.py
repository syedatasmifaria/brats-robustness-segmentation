#!/usr/bin/env python3
"""
Script 25D: Prepare nnU-Net degraded mini-pilot test inputs.

Purpose:
- Create degraded nnU-Net-style test inputs for 2 BraTS2020 test patients.
- Covers 5 artifacts x 5 severity levels.
- This is a mini pilot before the full 74-patient degraded nnU-Net evaluation.

Important:
- nnU-Net was trained on clean images only.
- Degraded images are created only for testing/evaluation.
- Temporary degraded .nii.gz files should NOT be pushed to GitHub.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import gaussian_filter


PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")
TEST_CSV = PROJECT_ROOT / "data/csvs/test_paths.csv"

OUT_ROOT = PROJECT_ROOT / "nnunet/temporary_degraded_tests/mini_pilot"

NUM_PATIENTS = 2
RNG_SEED = 2026

MODALITIES = [
    ("flair", "0000"),
    ("t1", "0001"),
    ("t1ce", "0002"),
    ("t2", "0003"),
]

DEGRADATION_LEVELS = {
    "blur": {
        1: {"sigma": 0.5},
        2: {"sigma": 1.0},
        3: {"sigma": 1.5},
        4: {"sigma": 2.0},
        5: {"sigma": 2.5},
    },
    "noise": {
        1: {"std": 0.02},
        2: {"std": 0.04},
        3: {"std": 0.06},
        4: {"std": 0.08},
        5: {"std": 0.10},
    },
    "contrast": {
        1: {"factor": 0.90},
        2: {"factor": 0.75},
        3: {"factor": 0.60},
        4: {"factor": 0.45},
        5: {"factor": 0.30},
    },
    "ringing": {
        1: {"keep_fraction": 0.85},
        2: {"keep_fraction": 0.75},
        3: {"keep_fraction": 0.65},
        4: {"keep_fraction": 0.55},
        5: {"keep_fraction": 0.45},
    },
    "ghosting": {
        1: {"num_repeats": 4, "intensity": 0.15},
        2: {"num_repeats": 8, "intensity": 0.25},
        3: {"num_repeats": 12, "intensity": 0.35},
        4: {"num_repeats": 16, "intensity": 0.45},
        5: {"num_repeats": 20, "intensity": 0.55},
    },
}


def load_nifti(path):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine, img.header


def save_image_nifti(data, affine, header, out_path):
    out_img = nib.Nifti1Image(data.astype(np.float32), affine, header)
    nib.save(out_img, str(out_path))


def save_label_nifti(data, affine, header, out_path):
    out_img = nib.Nifti1Image(data.astype(np.uint8), affine, header)
    nib.save(out_img, str(out_path))


def remap_seg_labels(seg):
    """
    BraTS original labels: [0, 1, 2, 4]
    Project/nnU-Net remapped labels: [0, 1, 2, 3]
    Original label 4 becomes class 3.
    """
    remapped = np.zeros_like(seg, dtype=np.uint8)
    remapped[seg == 1] = 1
    remapped[seg == 2] = 2
    remapped[seg == 4] = 3
    return remapped


def normalize_for_degradation(volume):
    """
    Normalize brain voxels to [0, 1] only while applying degradation.
    Later we restore to the original-ish intensity range before saving for nnU-Net.
    """
    brain_mask = volume > 0

    if brain_mask.sum() == 0:
        return volume.copy(), brain_mask, 0.0, 1.0

    brain_values = volume[brain_mask]
    v_min = float(np.percentile(brain_values, 1))
    v_max = float(np.percentile(brain_values, 99))

    if v_max <= v_min:
        return volume.copy(), brain_mask, v_min, v_max

    normalized = np.zeros_like(volume, dtype=np.float32)
    normalized[brain_mask] = (volume[brain_mask] - v_min) / (v_max - v_min)
    normalized = np.clip(normalized, 0.0, 1.0)

    return normalized, brain_mask, v_min, v_max


def restore_original_range(volume_01, brain_mask, v_min, v_max, original_volume):
    restored = np.zeros_like(original_volume, dtype=np.float32)

    if v_max <= v_min:
        restored[brain_mask] = original_volume[brain_mask]
        return restored

    restored[brain_mask] = volume_01[brain_mask] * (v_max - v_min) + v_min
    restored[~brain_mask] = 0.0
    return restored


def degrade_blur(volume_01, brain_mask, sigma):
    degraded = gaussian_filter(volume_01, sigma=sigma)
    degraded = np.clip(degraded, 0.0, 1.0)
    degraded[~brain_mask] = 0.0
    return degraded


def degrade_noise(volume_01, brain_mask, std, rng):
    degraded = volume_01.copy()
    noise = rng.normal(loc=0.0, scale=std, size=volume_01.shape).astype(np.float32)
    degraded[brain_mask] = degraded[brain_mask] + noise[brain_mask]
    degraded = np.clip(degraded, 0.0, 1.0)
    degraded[~brain_mask] = 0.0
    return degraded


def degrade_contrast(volume_01, brain_mask, factor):
    """
    Lower factor compresses intensities toward the midpoint.
    factor=1 means unchanged.
    factor=0.3 means strong contrast reduction.
    """
    degraded = volume_01.copy()
    midpoint = 0.5
    degraded[brain_mask] = midpoint + factor * (degraded[brain_mask] - midpoint)
    degraded = np.clip(degraded, 0.0, 1.0)
    degraded[~brain_mask] = 0.0
    return degraded


def degrade_ringing(volume_01, brain_mask, keep_fraction):
    """
    Simple Fourier low-pass truncation to create ringing/edge artifacts.
    Lower keep_fraction means stronger frequency truncation.
    """
    data = volume_01.copy().astype(np.float32)

    fft_data = np.fft.fftn(data)
    fft_shifted = np.fft.fftshift(fft_data)

    shape = np.array(data.shape)
    center = shape // 2
    keep = np.maximum((shape * keep_fraction / 2).astype(int), 1)

    mask = np.zeros(shape, dtype=bool)

    x0, x1 = center[0] - keep[0], center[0] + keep[0]
    y0, y1 = center[1] - keep[1], center[1] + keep[1]
    z0, z1 = center[2] - keep[2], center[2] + keep[2]

    mask[x0:x1, y0:y1, z0:z1] = True

    truncated = np.zeros_like(fft_shifted)
    truncated[mask] = fft_shifted[mask]

    restored = np.fft.ifftn(np.fft.ifftshift(truncated))
    degraded = np.real(restored).astype(np.float32)

    degraded = np.clip(degraded, 0.0, 1.0)
    degraded[~brain_mask] = 0.0
    return degraded


def degrade_ghosting(volume_01, brain_mask, num_repeats, intensity):
    degraded = volume_01.copy().astype(np.float32)
    axis = 1

    shape_size = degraded.shape[axis]
    shifts = np.linspace(4, max(5, shape_size // 3), num_repeats).astype(int)

    ghost = np.zeros_like(degraded, dtype=np.float32)

    for shift in shifts:
        ghost += np.roll(degraded, shift=shift, axis=axis)
        ghost += np.roll(degraded, shift=-shift, axis=axis)

    ghost = ghost / max(1, 2 * len(shifts))

    degraded = (1.0 - intensity) * degraded + intensity * ghost
    degraded = np.clip(degraded, 0.0, 1.0)
    degraded[~brain_mask] = 0.0
    return degraded


def apply_degradation(volume_01, brain_mask, artifact, params, rng):
    if artifact == "blur":
        return degrade_blur(volume_01, brain_mask, sigma=params["sigma"])

    if artifact == "noise":
        return degrade_noise(volume_01, brain_mask, std=params["std"], rng=rng)

    if artifact == "contrast":
        return degrade_contrast(volume_01, brain_mask, factor=params["factor"])

    if artifact == "ringing":
        return degrade_ringing(volume_01, brain_mask, keep_fraction=params["keep_fraction"])

    if artifact == "ghosting":
        return degrade_ghosting(
            volume_01,
            brain_mask,
            num_repeats=params["num_repeats"],
            intensity=params["intensity"],
        )

    raise ValueError(f"Unknown artifact: {artifact}")


def main():
    print("=" * 80)
    print("Script 25D: Prepare nnU-Net degraded mini-pilot inputs")
    print("=" * 80)

    rng = np.random.default_rng(RNG_SEED)

    df = pd.read_csv(TEST_CSV)
    mini_df = df.head(NUM_PATIENTS)

    print(f"Test CSV: {TEST_CSV}")
    print(f"Output root: {OUT_ROOT}")
    print(f"Patients: {len(mini_df)}")
    print("Artifacts:", list(DEGRADATION_LEVELS.keys()))
    print("Levels: 1-5")
    print()

    total_images = 0
    total_labels = 0
    condition_rows = []

    for artifact, level_dict in DEGRADATION_LEVELS.items():
        for level, params in level_dict.items():
            condition_name = f"{artifact}_L{level}"

            condition_root = OUT_ROOT / condition_name
            images_dir = condition_root / "imagesTs"
            labels_dir = condition_root / "labelsTs"
            predictions_dir = condition_root / "predictions"

            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            predictions_dir.mkdir(parents=True, exist_ok=True)

            print(f"Preparing condition: {condition_name} | params: {params}")

            for _, row in mini_df.iterrows():
                patient_id = row["patient_id"]

                for modality_name, channel_id in MODALITIES:
                    in_path = Path(row[modality_name])
                    volume, affine, header = load_nifti(in_path)

                    volume_01, brain_mask, v_min, v_max = normalize_for_degradation(volume)

                    degraded_01 = apply_degradation(
                        volume_01=volume_01,
                        brain_mask=brain_mask,
                        artifact=artifact,
                        params=params,
                        rng=rng,
                    )

                    degraded_volume = restore_original_range(
                        volume_01=degraded_01,
                        brain_mask=brain_mask,
                        v_min=v_min,
                        v_max=v_max,
                        original_volume=volume,
                    )

                    out_path = images_dir / f"{patient_id}_{channel_id}.nii.gz"
                    save_image_nifti(degraded_volume, affine, header, out_path)
                    total_images += 1

                seg_path = Path(row["seg"])
                seg, seg_affine, seg_header = load_nifti(seg_path)
                seg_remapped = remap_seg_labels(seg)

                label_out_path = labels_dir / f"{patient_id}.nii.gz"
                save_label_nifti(seg_remapped, seg_affine, seg_header, label_out_path)
                total_labels += 1

            condition_rows.append({
                "condition": condition_name,
                "artifact": artifact,
                "level": level,
                "params": str(params),
                "images_dir": str(images_dir),
                "labels_dir": str(labels_dir),
                "predictions_dir": str(predictions_dir),
                "num_patients": len(mini_df),
            })

    condition_df = pd.DataFrame(condition_rows)
    condition_csv = OUT_ROOT / "mini_pilot_conditions.csv"
    condition_df.to_csv(condition_csv, index=False)

    print()
    print("=" * 80)
    print("Mini-pilot degraded nnU-Net input preparation complete.")
    print(f"Conditions prepared: {len(condition_rows)}")
    print(f"Images written: {total_images}")
    print(f"Labels written: {total_labels}")
    print(f"Condition CSV: {condition_csv}")
    print("=" * 80)
    print()
    print("Expected:")
    print("25 conditions")
    print("2 patients x 25 conditions x 4 modalities = 200 image files")
    print("2 patients x 25 conditions = 50 label files")
    print()
    print("Reminder:")
    print("Do NOT commit nnunet/temporary_degraded_tests/ to GitHub.")


if __name__ == "__main__":
    main()
