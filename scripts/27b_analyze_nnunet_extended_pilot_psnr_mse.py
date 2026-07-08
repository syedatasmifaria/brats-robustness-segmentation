#!/usr/bin/env python3
"""
Script 27B: Analyze MSE/PSNR for nnU-Net extended degradation pilot.

Purpose:
- Measure image-level degradation strength for extended levels L6-L10.
- Selected artifacts: noise, contrast, ringing.
- Compare clean images against degraded images for the 5-patient pilot.
- Compute MSE and PSNR inside the brain mask.
- Save report-ready CSV, TXT, and plots.

Why:
The professor suggested adding stronger levels to see where the model breaks.
Before running segmentation prediction, we should verify that the new levels
actually create progressively stronger image changes.

MSE:
Higher MSE = stronger difference from clean.

PSNR:
Lower PSNR = stronger difference from clean.
"""

from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt


PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

TEST_CSV = PROJECT_ROOT / "data/csvs/test_paths.csv"
EXTENDED_ROOT = PROJECT_ROOT / "nnunet/temporary_degraded_tests/extended_pilot"

REPORT_DIR = PROJECT_ROOT / "report_materials"

OUT_CSV = REPORT_DIR / "27b_nnunet_extended_pilot_psnr_mse.csv"
OUT_TXT = REPORT_DIR / "27b_nnunet_extended_pilot_psnr_mse_summary.txt"

MSE_CURVE = REPORT_DIR / "27b_nnunet_extended_pilot_mse_curve.png"
PSNR_CURVE = REPORT_DIR / "27b_nnunet_extended_pilot_psnr_curve.png"

MODALITIES = [
    ("flair", "0000"),
    ("t1", "0001"),
    ("t1ce", "0002"),
    ("t2", "0003"),
]

ARTIFACT_ORDER = ["noise", "contrast", "ringing"]
LEVEL_ORDER = [6, 7, 8, 9, 10]


def load_nifti(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def normalize_clean_volume(clean_volume: np.ndarray):
    """
    Normalize clean volume to [0,1] using nonzero brain voxels.

    Returns:
    clean_01, brain_mask, v_min, v_max
    """
    brain_mask = clean_volume != 0

    if brain_mask.sum() == 0:
        raise ValueError("Clean volume has no nonzero brain voxels.")

    brain_values = clean_volume[brain_mask]
    v_min = float(np.percentile(brain_values, 1))
    v_max = float(np.percentile(brain_values, 99))

    if v_max <= v_min:
        raise ValueError("Invalid normalization range.")

    clean_01 = np.zeros_like(clean_volume, dtype=np.float32)
    clean_01[brain_mask] = (clean_volume[brain_mask] - v_min) / (v_max - v_min)
    clean_01 = np.clip(clean_01, 0.0, 1.0)

    return clean_01, brain_mask, v_min, v_max


def normalize_degraded_with_clean_range(degraded_volume: np.ndarray, brain_mask: np.ndarray, v_min: float, v_max: float):
    """
    Normalize degraded volume using the clean volume's normalization range.

    This makes clean vs degraded comparison fair.
    """
    degraded_01 = np.zeros_like(degraded_volume, dtype=np.float32)
    degraded_01[brain_mask] = (degraded_volume[brain_mask] - v_min) / (v_max - v_min)
    degraded_01 = np.clip(degraded_01, 0.0, 1.0)
    return degraded_01


def mse_psnr(clean_01: np.ndarray, degraded_01: np.ndarray, brain_mask: np.ndarray):
    """
    Compute MSE and PSNR inside the brain mask.

    Since normalized images are clipped to [0,1], data_range = 1.0.
    """
    diff = clean_01[brain_mask] - degraded_01[brain_mask]
    mse = float(np.mean(diff ** 2))

    if mse == 0:
        psnr = float("inf")
    else:
        psnr = float(20.0 * math.log10(1.0 / math.sqrt(mse)))

    return mse, psnr


def parse_condition(condition: str):
    """
    Example:
    noise_L6 -> artifact=noise, level=6
    """
    match = re.match(r"(.+)_L(\d+)", condition)
    if match is None:
        raise ValueError(f"Could not parse condition: {condition}")
    return match.group(1), int(match.group(2))


def main():
    print("=" * 80)
    print("Script 27B: Analyze nnU-Net extended pilot MSE/PSNR")
    print("=" * 80)

    REPORT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(TEST_CSV).head(5).copy()

    print(f"Pilot patients: {len(df)}")
    print(f"Extended root: {EXTENDED_ROOT}")
    print(f"Output CSV: {OUT_CSV}")
    print()

    rows = []

    condition_dirs = sorted([
        p for p in EXTENDED_ROOT.iterdir()
        if p.is_dir() and "_L" in p.name and p.name != "labelsTs"
    ])

    if len(condition_dirs) == 0:
        raise RuntimeError(f"No extended condition directories found in {EXTENDED_ROOT}")

    print("Found conditions:")
    for p in condition_dirs:
        print(f"  {p.name}")
    print()

    for condition_dir in condition_dirs:
        condition = condition_dir.name
        artifact, level = parse_condition(condition)

        if artifact not in ARTIFACT_ORDER or level not in LEVEL_ORDER:
            print(f"Skipping unexpected condition: {condition}")
            continue

        images_dir = condition_dir / "imagesTs"

        print("-" * 80)
        print(f"Analyzing condition: {condition}")

        for _, row in df.iterrows():
            patient_id = row["patient_id"]

            for modality_name, channel_id in MODALITIES:
                clean_path = Path(row[modality_name])
                degraded_path = images_dir / f"{patient_id}_{channel_id}.nii.gz"

                if not degraded_path.exists():
                    raise FileNotFoundError(f"Missing degraded image: {degraded_path}")

                clean_volume = load_nifti(clean_path)
                degraded_volume = load_nifti(degraded_path)

                if clean_volume.shape != degraded_volume.shape:
                    raise RuntimeError(
                        f"Shape mismatch for {patient_id} {modality_name}: "
                        f"clean {clean_volume.shape}, degraded {degraded_volume.shape}"
                    )

                clean_01, brain_mask, v_min, v_max = normalize_clean_volume(clean_volume)
                degraded_01 = normalize_degraded_with_clean_range(
                    degraded_volume=degraded_volume,
                    brain_mask=brain_mask,
                    v_min=v_min,
                    v_max=v_max,
                )

                mse, psnr = mse_psnr(clean_01, degraded_01, brain_mask)

                rows.append({
                    "condition": condition,
                    "artifact": artifact,
                    "level": level,
                    "patient_id": patient_id,
                    "modality": modality_name,
                    "mse": mse,
                    "psnr": psnr,
                })

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUT_CSV, index=False)

    summary_df = (
        metrics_df
        .groupby(["artifact", "level", "condition"], as_index=False)
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            psnr_mean=("psnr", "mean"),
            psnr_std=("psnr", "std"),
            n=("mse", "count"),
        )
    )

    summary_df["artifact"] = pd.Categorical(
        summary_df["artifact"],
        categories=ARTIFACT_ORDER,
        ordered=True,
    )
    summary_df = summary_df.sort_values(["artifact", "level"])

    with open(OUT_TXT, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Script 27B: nnU-Net extended pilot MSE/PSNR summary\n")
        f.write("=" * 80 + "\n\n")
        f.write("Interpretation:\n")
        f.write("Higher MSE means stronger image change.\n")
        f.write("Lower PSNR means stronger image change.\n\n")

        for _, row in summary_df.iterrows():
            f.write(
                f"{row['condition']}: "
                f"MSE mean={row['mse_mean']:.6f}, "
                f"PSNR mean={row['psnr_mean']:.2f} dB, "
                f"n={int(row['n'])}\n"
            )

    # MSE curve
    plt.figure(figsize=(8, 5))
    for artifact in ARTIFACT_ORDER:
        sub = summary_df[summary_df["artifact"] == artifact].sort_values("level")
        plt.plot(sub["level"], sub["mse_mean"], marker="o", label=artifact)
    plt.xlabel("Severity level")
    plt.ylabel("Mean MSE")
    plt.title("Extended pilot degradation strength by MSE")
    plt.xticks(LEVEL_ORDER)
    plt.legend()
    plt.tight_layout()
    plt.savefig(MSE_CURVE, dpi=300)
    plt.close()

    # PSNR curve
    plt.figure(figsize=(8, 5))
    for artifact in ARTIFACT_ORDER:
        sub = summary_df[summary_df["artifact"] == artifact].sort_values("level")
        plt.plot(sub["level"], sub["psnr_mean"], marker="o", label=artifact)
    plt.xlabel("Severity level")
    plt.ylabel("Mean PSNR (dB)")
    plt.title("Extended pilot degradation strength by PSNR")
    plt.xticks(LEVEL_ORDER)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PSNR_CURVE, dpi=300)
    plt.close()

    print("=" * 80)
    print("Extended pilot MSE/PSNR analysis complete.")
    print(f"Saved CSV: {OUT_CSV}")
    print(f"Saved TXT: {OUT_TXT}")
    print(f"Saved MSE curve: {MSE_CURVE}")
    print(f"Saved PSNR curve: {PSNR_CURVE}")
    print("=" * 80)
    print()
    print("Summary preview:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
