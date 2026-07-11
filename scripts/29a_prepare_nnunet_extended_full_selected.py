#!/usr/bin/env python3
"""
Script 29A: Prepare full 74-patient nnU-Net extended degradation inputs.

Purpose:
- Scale the extended L6-L10 pilot to the full 74-patient held-out test set.
- Artifacts:
  1. noise L6-L10
  2. contrast L6-L10
  3. ringing L6-L10

Important:
- This is testing/evaluation only.
- No model training.
- No validation fitting.
- Degradations are applied to all four MRI modalities.
- Do NOT commit generated .nii.gz files.

Note on ringing:
The ringing implementation here is the original frequency-domain truncation
version used in the earlier extended pilot. In the report, describe it carefully
as "frequency-domain ringing-like degradation" or
"Fourier truncation / low-pass frequency stress test," because visual inspection
showed it can behave like smoothing at high severity.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")
TEST_CSV = PROJECT_ROOT / "data/csvs/test_paths.csv"

OUT_ROOT = PROJECT_ROOT / "nnunet/temporary_degraded_tests/extended_full_selected"
LABELS_DIR = OUT_ROOT / "labelsTs"

OUT_ROOT.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = PROJECT_ROOT / "report_materials"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SUMMARY_TXT = REPORT_DIR / "29a_nnunet_extended_full_selected_preparation_summary.txt"


# ------------------------------------------------------------
# nnU-Net modality mapping
# ------------------------------------------------------------

MODALITIES = [
    ("flair", "0000"),
    ("t1", "0001"),
    ("t1ce", "0002"),
    ("t2", "0003"),
]


# ------------------------------------------------------------
# Extended artifact parameters
# Same parameter logic as the 5-patient extended pilot.
# ------------------------------------------------------------

CONDITIONS = {
    "noise_L6":    {"artifact": "noise",    "level": 6,  "std": 0.15},
    "noise_L7":    {"artifact": "noise",    "level": 7,  "std": 0.20},
    "noise_L8":    {"artifact": "noise",    "level": 8,  "std": 0.30},
    "noise_L9":    {"artifact": "noise",    "level": 9,  "std": 0.40},
    "noise_L10":   {"artifact": "noise",    "level": 10, "std": 0.50},

    "contrast_L6":  {"artifact": "contrast", "level": 6,  "factor": 0.20},
    "contrast_L7":  {"artifact": "contrast", "level": 7,  "factor": 0.15},
    "contrast_L8":  {"artifact": "contrast", "level": 8,  "factor": 0.10},
    "contrast_L9":  {"artifact": "contrast", "level": 9,  "factor": 0.05},
    "contrast_L10": {"artifact": "contrast", "level": 10, "factor": 0.02},

    "ringing_L6":  {"artifact": "ringing", "level": 6,  "keep_fraction": 0.35},
    "ringing_L7":  {"artifact": "ringing", "level": 7,  "keep_fraction": 0.25},
    "ringing_L8":  {"artifact": "ringing", "level": 8,  "keep_fraction": 0.15},
    "ringing_L9":  {"artifact": "ringing", "level": 9,  "keep_fraction": 0.10},
    "ringing_L10": {"artifact": "ringing", "level": 10, "keep_fraction": 0.05},
}


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def normalize_01(volume):
    """
    Normalize MRI volume to [0, 1] using nonzero brain voxels.
    This matches the degradation logic used in prior experiments.
    """
    volume = volume.astype(np.float32)
    brain_mask = volume > 0

    if brain_mask.sum() == 0:
        return np.zeros_like(volume, dtype=np.float32), brain_mask

    vals = volume[brain_mask]
    lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)

    volume_clip = np.clip(volume, lo, hi)
    volume_01 = (volume_clip - lo) / (hi - lo + 1e-8)
    volume_01[~brain_mask] = 0.0

    return volume_01.astype(np.float32), brain_mask


def degrade_noise(volume_01, brain_mask, std):
    """
    Add Gaussian noise inside the brain mask.
    """
    noise = np.random.normal(loc=0.0, scale=std, size=volume_01.shape).astype(np.float32)
    degraded = volume_01 + noise
    degraded = np.clip(degraded, 0.0, 1.0)
    degraded[~brain_mask] = 0.0
    return degraded.astype(np.float32)


def degrade_contrast(volume_01, brain_mask, factor):
    """
    Reduce contrast around the brain-intensity mean.

    factor < 1 compresses intensity differences.
    Very small factors make the image flatter.
    """
    degraded = volume_01.copy()
    mean_val = float(volume_01[brain_mask].mean()) if brain_mask.sum() > 0 else 0.5

    degraded[brain_mask] = (degraded[brain_mask] - mean_val) * factor + mean_val
    degraded = np.clip(degraded, 0.0, 1.0)
    degraded[~brain_mask] = 0.0
    return degraded.astype(np.float32)


def degrade_ringing(volume_01, brain_mask, keep_fraction):
    """
    Original frequency-domain truncation artifact.

    Code-level idea:
    - Apply FFT.
    - Keep only the central low-frequency block.
    - Remove outer high-frequency components.
    - Inverse FFT back to image space.

    This is why the artifact can behave like low-pass smoothing at high severity.
    Report it carefully as frequency-domain ringing-like / Fourier truncation.
    """
    vol = volume_01.astype(np.float32)

    fft_vol = np.fft.fftn(vol)
    fft_shifted = np.fft.fftshift(fft_vol)

    sx, sy, sz = vol.shape
    kx = max(1, int(sx * keep_fraction / 2))
    ky = max(1, int(sy * keep_fraction / 2))
    kz = max(1, int(sz * keep_fraction / 2))

    cx, cy, cz = sx // 2, sy // 2, sz // 2

    mask = np.zeros_like(vol, dtype=bool)
    mask[
        cx - kx: cx + kx,
        cy - ky: cy + ky,
        cz - kz: cz + kz
    ] = True

    truncated = np.zeros_like(fft_shifted)
    truncated[mask] = fft_shifted[mask]

    inv_shifted = np.fft.ifftshift(truncated)
    degraded = np.fft.ifftn(inv_shifted).real.astype(np.float32)

    degraded = np.clip(degraded, 0.0, 1.0)
    degraded[~brain_mask] = 0.0

    return degraded.astype(np.float32)


def apply_degradation(volume_01, brain_mask, condition_params):
    artifact = condition_params["artifact"]

    if artifact == "noise":
        return degrade_noise(volume_01, brain_mask, std=condition_params["std"])

    if artifact == "contrast":
        return degrade_contrast(volume_01, brain_mask, factor=condition_params["factor"])

    if artifact == "ringing":
        return degrade_ringing(volume_01, brain_mask, keep_fraction=condition_params["keep_fraction"])

    raise ValueError(f"Unknown artifact: {artifact}")


def remap_seg_to_nnunet(seg):
    """
    BraTS original labels:
    0, 1, 2, 4

    nnU-Net labels:
    0, 1, 2, 3

    Original label 4 becomes 3.
    """
    seg = seg.astype(np.int16)
    out = np.zeros_like(seg, dtype=np.uint8)
    out[seg == 1] = 1
    out[seg == 2] = 2
    out[seg == 4] = 3
    out[seg == 3] = 3
    return out


def save_nifti_like(data, reference_img, out_path):
    out_img = nib.Nifti1Image(data, reference_img.affine, reference_img.header)
    nib.save(out_img, str(out_path))


def case_id_from_patient_id(patient_id):
    """
    BraTS20_Training_001 -> 001
    """
    return str(patient_id).split("_")[-1]


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("=" * 80)
    print("Script 29A: Prepare full 74-patient extended nnU-Net inputs")
    print("=" * 80)
    print(f"Output root: {OUT_ROOT}")
    print("Artifacts: noise, contrast, ringing/frequency-domain")
    print("Levels: L6-L10")
    print("Testing only. No training.")
    print("=" * 80)

    # Make noise reproducible.
    np.random.seed(2029)

    test_df = pd.read_csv(TEST_CSV)
    n_patients = len(test_df)

    total_images = 0
    total_labels = 0

    for idx, row in test_df.iterrows():
        patient_id = row["patient_id"]
        case_num = case_id_from_patient_id(patient_id)

        print(f"\n[{idx + 1}/{n_patients}] Preparing patient: {patient_id}")

        # Save shared label once per patient.
        seg_img = nib.load(str(row["seg"]))
        seg = seg_img.get_fdata().astype(np.int16)
        seg_remap = remap_seg_to_nnunet(seg)

        label_out = LABELS_DIR / f"BraTS20_Training_{case_num}.nii.gz"
        save_nifti_like(seg_remap.astype(np.uint8), seg_img, label_out)
        total_labels += 1

        for condition_name, params in CONDITIONS.items():
            condition_images_dir = OUT_ROOT / condition_name / "imagesTs"
            condition_images_dir.mkdir(parents=True, exist_ok=True)

            for modality_col, channel_id in MODALITIES:
                img_path = Path(row[modality_col])
                img = nib.load(str(img_path))
                vol = img.get_fdata().astype(np.float32)

                vol_01, brain_mask = normalize_01(vol)

                degraded = apply_degradation(vol_01, brain_mask, params)

                out_name = f"BRATS_{case_num}_{channel_id}.nii.gz"
                out_path = condition_images_dir / out_name

                save_nifti_like(degraded.astype(np.float32), img, out_path)
                total_images += 1

    expected_images = n_patients * len(CONDITIONS) * len(MODALITIES)
    expected_labels = n_patients

    summary = []
    summary.append("=" * 80)
    summary.append("Script 29A: Full 74-patient extended nnU-Net input preparation summary")
    summary.append("=" * 80)
    summary.append(f"Output root: {OUT_ROOT}")
    summary.append(f"Patients: {n_patients}")
    summary.append(f"Conditions: {len(CONDITIONS)}")
    summary.append(f"Modalities per patient: {len(MODALITIES)}")
    summary.append(f"Total images written: {total_images}")
    summary.append(f"Expected images: {expected_images}")
    summary.append(f"Labels written: {total_labels}")
    summary.append(f"Expected labels: {expected_labels}")
    summary.append("")
    summary.append("Conditions prepared:")
    for condition_name, params in CONDITIONS.items():
        summary.append(f"- {condition_name}: {params}")
    summary.append("")
    summary.append("Important reporting note:")
    summary.append(
        "The ringing condition uses Fourier truncation. In the report, describe it as "
        "frequency-domain ringing-like degradation or Fourier truncation / low-pass "
        "frequency stress test, not pure classic Gibbs ringing."
    )
    summary.append("=" * 80)

    with open(OUT_SUMMARY_TXT, "w") as f:
        f.write("\n".join(summary) + "\n")

    print("\n" + "\n".join(summary))

    if total_images != expected_images:
        raise RuntimeError(f"Image count mismatch: got {total_images}, expected {expected_images}")

    if total_labels != expected_labels:
        raise RuntimeError(f"Label count mismatch: got {total_labels}, expected {expected_labels}")


if __name__ == "__main__":
    main()
