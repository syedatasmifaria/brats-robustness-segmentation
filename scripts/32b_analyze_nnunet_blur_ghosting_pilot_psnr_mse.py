#!/usr/bin/env python3
"""
Script 32B: Analyze MSE and PSNR for the nnU-Net blur/ghosting pilot.

Purpose:
- Compare existing L5 against candidate L6-L10.
- Use the same five held-out patients and all four modalities.
- Compute MSE and PSNR inside the clean brain mask.
- Verify that candidate severity increases beyond L5.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt


PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")
TEST_CSV = PROJECT_ROOT / "data/csvs/test_paths.csv"

L5_ROOT = (
    PROJECT_ROOT
    / "nnunet/temporary_degraded_tests/final_full"
)

EXTENDED_ROOT = (
    PROJECT_ROOT
    / "nnunet/temporary_degraded_tests/blur_ghosting_extended_pilot"
)

REPORT_DIR = PROJECT_ROOT / "report_materials"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = (
    REPORT_DIR
    / "32b_nnunet_blur_ghosting_pilot_psnr_mse.csv"
)

OUT_SUMMARY_CSV = (
    REPORT_DIR
    / "32b_nnunet_blur_ghosting_pilot_psnr_mse_summary.csv"
)

OUT_TXT = (
    REPORT_DIR
    / "32b_nnunet_blur_ghosting_pilot_psnr_mse_summary.txt"
)

MSE_PLOT = (
    REPORT_DIR
    / "32b_nnunet_blur_ghosting_pilot_mse_curve.png"
)

PSNR_PLOT = (
    REPORT_DIR
    / "32b_nnunet_blur_ghosting_pilot_psnr_curve.png"
)

NUM_PATIENTS = 5

MODALITIES = [
    ("flair", "0000"),
    ("t1", "0001"),
    ("t1ce", "0002"),
    ("t2", "0003"),
]

ARTIFACTS = ["blur", "ghosting"]
LEVELS = [5, 6, 7, 8, 9, 10]


def load_nifti(path):
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def normalize_clean_volume(clean_volume):
    brain_mask = clean_volume != 0

    if brain_mask.sum() == 0:
        raise ValueError("Clean volume has no nonzero brain voxels.")

    brain_values = clean_volume[brain_mask]
    v_min = float(np.percentile(brain_values, 1))
    v_max = float(np.percentile(brain_values, 99))

    if v_max <= v_min:
        raise ValueError("Invalid clean normalization range.")

    clean_01 = np.zeros_like(clean_volume, dtype=np.float32)
    clean_01[brain_mask] = (
        clean_volume[brain_mask] - v_min
    ) / (v_max - v_min)

    clean_01 = np.clip(clean_01, 0.0, 1.0)

    return clean_01, brain_mask, v_min, v_max


def normalize_degraded_with_clean_range(
    degraded_volume,
    brain_mask,
    v_min,
    v_max,
):
    degraded_01 = np.zeros_like(
        degraded_volume,
        dtype=np.float32,
    )

    degraded_01[brain_mask] = (
        degraded_volume[brain_mask] - v_min
    ) / (v_max - v_min)

    degraded_01 = np.clip(degraded_01, 0.0, 1.0)

    return degraded_01


def compute_mse_psnr(clean_01, degraded_01, brain_mask):
    difference = (
        clean_01[brain_mask]
        - degraded_01[brain_mask]
    )

    mse = float(np.mean(difference ** 2))

    if mse == 0:
        psnr = float("inf")
    else:
        psnr = float(
            20.0 * math.log10(1.0 / math.sqrt(mse))
        )

    return mse, psnr


def condition_root(artifact, level):
    if level == 5:
        return L5_ROOT / f"{artifact}_L5"

    return EXTENDED_ROOT / f"{artifact}_L{level}"


def main():
    print("=" * 80)
    print("Script 32B: Blur and ghosting pilot MSE/PSNR analysis")
    print("=" * 80)

    test_df = pd.read_csv(TEST_CSV).head(NUM_PATIENTS).copy()

    print(f"Pilot patients: {len(test_df)}")
    print("Artifacts: blur, ghosting")
    print("Levels compared: L5-L10")
    print()

    rows = []

    for artifact in ARTIFACTS:
        for level in LEVELS:
            condition = f"{artifact}_L{level}"
            images_dir = condition_root(
                artifact,
                level,
            ) / "imagesTs"

            if not images_dir.exists():
                raise FileNotFoundError(
                    f"Missing condition directory: {images_dir}"
                )

            print(f"Analyzing {condition}")

            for _, row in test_df.iterrows():
                patient_id = row["patient_id"]

                for modality_name, channel_id in MODALITIES:
                    clean_path = Path(row[modality_name])

                    degraded_path = (
                        images_dir
                        / f"{patient_id}_{channel_id}.nii.gz"
                    )

                    if not degraded_path.exists():
                        raise FileNotFoundError(
                            f"Missing degraded image: {degraded_path}"
                        )

                    clean_volume = load_nifti(clean_path)
                    degraded_volume = load_nifti(degraded_path)

                    if clean_volume.shape != degraded_volume.shape:
                        raise RuntimeError(
                            f"Shape mismatch for {condition}, "
                            f"{patient_id}, {modality_name}: "
                            f"{clean_volume.shape} vs "
                            f"{degraded_volume.shape}"
                        )

                    (
                        clean_01,
                        brain_mask,
                        v_min,
                        v_max,
                    ) = normalize_clean_volume(clean_volume)

                    degraded_01 = (
                        normalize_degraded_with_clean_range(
                            degraded_volume=degraded_volume,
                            brain_mask=brain_mask,
                            v_min=v_min,
                            v_max=v_max,
                        )
                    )

                    mse, psnr = compute_mse_psnr(
                        clean_01,
                        degraded_01,
                        brain_mask,
                    )

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
        .groupby(
            ["artifact", "level", "condition"],
            as_index=False,
        )
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            psnr_mean=("psnr", "mean"),
            psnr_std=("psnr", "std"),
            n=("mse", "count"),
        )
        .sort_values(["artifact", "level"])
    )

    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)

    monotonic_rows = []

    for artifact in ARTIFACTS:
        artifact_df = (
            summary_df[
                summary_df["artifact"] == artifact
            ]
            .sort_values("level")
            .copy()
        )

        mse_values = artifact_df["mse_mean"].to_numpy()
        psnr_values = artifact_df["psnr_mean"].to_numpy()

        mse_monotonic = bool(
            np.all(np.diff(mse_values) > 0)
        )

        psnr_monotonic = bool(
            np.all(np.diff(psnr_values) < 0)
        )

        l5_mse = float(
            artifact_df.loc[
                artifact_df["level"] == 5,
                "mse_mean",
            ].iloc[0]
        )

        l6_mse = float(
            artifact_df.loc[
                artifact_df["level"] == 6,
                "mse_mean",
            ].iloc[0]
        )

        l6_stronger_than_l5 = l6_mse > l5_mse

        monotonic_rows.append({
            "artifact": artifact,
            "mse_strictly_increases_L5_L10": mse_monotonic,
            "psnr_strictly_decreases_L5_L10": psnr_monotonic,
            "L6_mse_greater_than_L5": l6_stronger_than_l5,
        })

    monotonic_df = pd.DataFrame(monotonic_rows)

    with open(OUT_TXT, "w", encoding="utf-8") as file:
        file.write("=" * 80 + "\n")
        file.write(
            "Script 32B: Blur and ghosting pilot "
            "MSE/PSNR summary\n"
        )
        file.write("=" * 80 + "\n\n")
        file.write(
            "Higher MSE indicates stronger image change.\n"
        )
        file.write(
            "Lower PSNR indicates stronger image change.\n\n"
        )

        for _, row in summary_df.iterrows():
            file.write(
                f"{row['condition']}: "
                f"MSE mean={row['mse_mean']:.6f}, "
                f"MSE SD={row['mse_std']:.6f}, "
                f"PSNR mean={row['psnr_mean']:.2f} dB, "
                f"PSNR SD={row['psnr_std']:.2f}, "
                f"n={int(row['n'])}\n"
            )

        file.write("\nMonotonicity checks:\n")

        for _, row in monotonic_df.iterrows():
            file.write(
                f"{row['artifact']}: "
                f"MSE strictly increases L5-L10="
                f"{row['mse_strictly_increases_L5_L10']}; "
                f"PSNR strictly decreases L5-L10="
                f"{row['psnr_strictly_decreases_L5_L10']}; "
                f"L6 stronger than L5="
                f"{row['L6_mse_greater_than_L5']}\n"
            )

    plt.figure(figsize=(8, 5))

    for artifact in ARTIFACTS:
        artifact_df = (
            summary_df[
                summary_df["artifact"] == artifact
            ]
            .sort_values("level")
        )

        plt.plot(
            artifact_df["level"],
            artifact_df["mse_mean"],
            marker="o",
            label=artifact,
        )

    plt.xlabel("Severity level")
    plt.ylabel("Mean MSE")
    plt.title("Blur and ghosting pilot degradation strength")
    plt.xticks(LEVELS)
    plt.legend()
    plt.tight_layout()
    plt.savefig(MSE_PLOT, dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))

    for artifact in ARTIFACTS:
        artifact_df = (
            summary_df[
                summary_df["artifact"] == artifact
            ]
            .sort_values("level")
        )

        plt.plot(
            artifact_df["level"],
            artifact_df["psnr_mean"],
            marker="o",
            label=artifact,
        )

    plt.xlabel("Severity level")
    plt.ylabel("Mean PSNR (dB)")
    plt.title("Blur and ghosting pilot degradation strength")
    plt.xticks(LEVELS)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PSNR_PLOT, dpi=300)
    plt.close()

    print()
    print("=" * 80)
    print("Blur and ghosting pilot MSE/PSNR analysis complete.")
    print(f"Detailed CSV: {OUT_CSV}")
    print(f"Summary CSV: {OUT_SUMMARY_CSV}")
    print(f"Summary TXT: {OUT_TXT}")
    print(f"MSE plot: {MSE_PLOT}")
    print(f"PSNR plot: {PSNR_PLOT}")
    print("=" * 80)
    print()
    print("Summary:")
    print(summary_df.to_string(index=False))
    print()
    print("Monotonicity checks:")
    print(monotonic_df.to_string(index=False))


if __name__ == "__main__":
    main()
