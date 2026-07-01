#!/usr/bin/env python3
"""
Script 26A: Prepare final full nnU-Net degraded test inputs.

Purpose:
- Create degraded nnU-Net-style test inputs for all 74 held-out BraTS2020 test patients.
- Covers 5 artifacts x 5 severity levels.
- This prepares the final nnU-Net robustness evaluation.

Important:
- nnU-Net was trained on clean images only.
- Degraded images are created only for testing/evaluation.
- Temporary degraded .nii.gz files should NOT be pushed to GitHub.

Output structure:
nnunet/temporary_degraded_tests/final_full/
    labelsTs/
    blur_L1/imagesTs/
    blur_L1/predictions/
    ...
    ghosting_L5/imagesTs/
    ghosting_L5/predictions/
"""

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import gaussian_filter


PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")
TEST_CSV = PROJECT_ROOT / "data/csvs/test_paths.csv"

OUT_ROOT = PROJECT_ROOT / "nnunet/temporary_degraded_tests/final_full"
OUT_LABELS = OUT_ROOT / "labelsTs"

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


def stable_seed(*parts):
    """
    Create a deterministic seed from text parts.
    This makes noise degradation reproducible even if the script is resumed.
    """
    joined = "|".join(str(p) for p in parts)
    digest = hashlib.md5(joined.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + RNG_SEED) % (2**32)


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
    Project/nnU-Net labels: [0, 1, 2, 3]
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
    Then restore back to original-ish intensity range before saving.

    Why:
    - Degradation strength is easier to control on [0, 1].
    - nnU-Net was trained on BraTS-style image intensities, so we should not save
      permanently normalized images as final inputs.
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
    degraded = volume_01.copy()
    midpoint = 0.5
    degraded[brain_mask] = midpoint + factor * (degraded[brain_mask] - midpoint)
    degraded = np.clip(degraded, 0.0, 1.0)
    degraded[~brain_mask] = 0.0
    return degraded


def degrade_ringing(volume_01, brain_mask, keep_fraction):
    """
    Simple Fourier truncation degradation.
    Lower keep_fraction means more severe frequency removal.
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

    # Synthetic phase-like direction.
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


def prepare_shared_labels(df):
    OUT_LABELS.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    print("Preparing shared labelsTs...")

    for _, row in df.iterrows():
        patient_id = row["patient_id"]
        label_out_path = OUT_LABELS / f"{patient_id}.nii.gz"

        if label_out_path.exists():
            skipped += 1
            continue

        seg_path = Path(row["seg"])
        seg, seg_affine, seg_header = load_nifti(seg_path)
        seg_remapped = remap_seg_labels(seg)

        save_label_nifti(seg_remapped, seg_affine, seg_header, label_out_path)
        written += 1

    print(f"Shared labels written: {written}")
    print(f"Shared labels skipped: {skipped}")
    print()


def main():
    print("=" * 80)
    print("Script 26A: Prepare final full nnU-Net degraded test inputs")
    print("=" * 80)

    df = pd.read_csv(TEST_CSV)

    print(f"Test CSV: {TEST_CSV}")
    print(f"Output root: {OUT_ROOT}")
    print(f"Number of test patients: {len(df)}")
    print("Artifacts:", list(DEGRADATION_LEVELS.keys()))
    print("Levels: 1-5")
    print()

    prepare_shared_labels(df)

    total_images_written = 0
    total_images_skipped = 0
    condition_rows = []

    for artifact, level_dict in DEGRADATION_LEVELS.items():
        for level, params in level_dict.items():
            condition_name = f"{artifact}_L{level}"

            condition_root = OUT_ROOT / condition_name
            images_dir = condition_root / "imagesTs"
            predictions_dir = condition_root / "predictions"

            images_dir.mkdir(parents=True, exist_ok=True)
            predictions_dir.mkdir(parents=True, exist_ok=True)

            print("-" * 80)
            print(f"Preparing condition: {condition_name} | params: {params}")

            condition_written = 0
            condition_skipped = 0

            for _, row in df.iterrows():
                patient_id = row["patient_id"]

                for modality_name, channel_id in MODALITIES:
                    out_path = images_dir / f"{patient_id}_{channel_id}.nii.gz"

                    # Resume safety: do not regenerate files that already exist.
                    if out_path.exists():
                        condition_skipped += 1
                        total_images_skipped += 1
                        continue

                    in_path = Path(row[modality_name])
                    volume, affine, header = load_nifti(in_path)

                    volume_01, brain_mask, v_min, v_max = normalize_for_degradation(volume)

                    seed = stable_seed(patient_id, modality_name, artifact, level)
                    rng = np.random.default_rng(seed)

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

                    save_image_nifti(degraded_volume, affine, header, out_path)

                    condition_written += 1
                    total_images_written += 1

            print(f"Condition images written: {condition_written}")
            print(f"Condition images skipped: {condition_skipped}")

            condition_rows.append({
                "condition": condition_name,
                "artifact": artifact,
                "level": level,
                "params": str(params),
                "images_dir": str(images_dir),
                "predictions_dir": str(predictions_dir),
                "num_patients": len(df),
                "expected_image_files": len(df) * len(MODALITIES),
            })

    condition_df = pd.DataFrame(condition_rows)
    condition_csv = OUT_ROOT / "final_full_conditions.csv"
    condition_df.to_csv(condition_csv, index=False)

    print()
    print("=" * 80)
    print("Final full degraded nnU-Net input preparation complete.")
    print(f"Conditions prepared: {len(condition_rows)}")
    print(f"Images written this run: {total_images_written}")
    print(f"Images skipped this run: {total_images_skipped}")
    print(f"Expected total degraded image files: {len(df) * 25 * 4}")
    print(f"Condition CSV: {condition_csv}")
    print("=" * 80)
    print()
    print("Reminder:")
    print("Do NOT commit nnunet/temporary_degraded_tests/ to GitHub.")


if __name__ == "__main__":
    main()
