#!/usr/bin/env python3
"""
Script 33C: Compute full-cohort MSE and PSNR for blur/ghosting L6-L10.

Purpose:
- Analyze the 10 newly completed conditions:
  blur L6-L10 and ghosting L6-L10.
- Use all 74 held-out patients and all four MRI modalities.
- Normalize clean and degraded images with the clean MRI intensity range.
- Merge the 10 new rows with the existing 40-row Script 30A summary.
- Produce a final 50-condition image-quality summary.

Important:
The new blur/ghosting L6-L10 files were restored to the original MRI
intensity range before saving. Therefore, they must be normalized using
the corresponding clean MRI range, just like the original L1-L5 files.
"""

from pathlib import Path
import math

import nibabel as nib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

TEST_CSV = PROJECT_ROOT / "data/csvs/test_paths.csv"

NEW_ROOT = (
    PROJECT_ROOT
    / "nnunet/temporary_degraded_tests/blur_ghosting_extended_full"
)

EXISTING_SUMMARY = (
    PROJECT_ROOT
    / "report_materials/30a_nnunet_full_degradation_psnr_mse_summary.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_DIR = PROJECT_ROOT / "report_materials"

OUT_METRICS_CSV = (
    RESULTS_DIR
    / "33c_nnunet_blur_ghosting_L6_L10_psnr_mse_metrics.csv"
)

OUT_NEW_SUMMARY_CSV = (
    REPORT_DIR
    / "33c_nnunet_blur_ghosting_L6_L10_psnr_mse_summary.csv"
)

OUT_FINAL_SUMMARY_CSV = (
    REPORT_DIR
    / "33c_nnunet_all_artifacts_L1_L10_psnr_mse_summary.csv"
)

OUT_FINAL_SUMMARY_TXT = (
    REPORT_DIR
    / "33c_nnunet_all_artifacts_L1_L10_psnr_mse_summary.txt"
)

MODALITIES = [
    ("flair", "0000"),
    ("t1", "0001"),
    ("t1ce", "0002"),
    ("t2", "0003"),
]

ARTIFACTS = ["blur", "ghosting"]
LEVELS = [6, 7, 8, 9, 10]

DISPLAY_NAMES = {
    "blur": "Blur",
    "ghosting": "Ghosting",
    "noise": "Gaussian noise",
    "contrast": "Contrast reduction",
    "ringing": "Frequency-domain truncation",
}

ARTIFACT_ORDER = [
    "blur",
    "ghosting",
    "noise",
    "contrast",
    "ringing",
]


def load_nifti(path):
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def normalize_clean_volume(clean_volume):
    clean_volume = clean_volume.astype(np.float32)
    brain_mask = clean_volume != 0

    if brain_mask.sum() == 0:
        raise ValueError("Clean volume has no nonzero brain voxels.")

    brain_values = clean_volume[brain_mask]

    v_min = float(np.percentile(brain_values, 1))
    v_max = float(np.percentile(brain_values, 99))

    if v_max <= v_min:
        raise ValueError(
            f"Invalid clean normalization range: {v_min}, {v_max}"
        )

    clean_01 = np.zeros_like(clean_volume, dtype=np.float32)

    clean_01[brain_mask] = (
        clean_volume[brain_mask] - v_min
    ) / (v_max - v_min)

    clean_01 = np.clip(clean_01, 0.0, 1.0)
    clean_01[~brain_mask] = 0.0

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
    degraded_01[~brain_mask] = 0.0

    return degraded_01


def calculate_mse_psnr(clean_01, degraded_01, brain_mask):
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


def main():
    print("=" * 80)
    print("Script 33C: Blur/ghosting L6-L10 full MSE/PSNR analysis")
    print("=" * 80)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(TEST_CSV)

    if len(test_df) != 74:
        raise RuntimeError(
            f"Expected 74 test patients, found {len(test_df)}"
        )

    rows = []

    for artifact in ARTIFACTS:
        for level in LEVELS:
            condition = f"{artifact}_L{level}"
            images_dir = NEW_ROOT / condition / "imagesTs"

            if not images_dir.exists():
                raise FileNotFoundError(
                    f"Missing images directory: {images_dir}"
                )

            print(f"Analyzing {condition}")

            condition_count = 0

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
                            f"{patient_id}, {modality_name}"
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

                    mse, psnr = calculate_mse_psnr(
                        clean_01,
                        degraded_01,
                        brain_mask,
                    )

                    rows.append({
                        "experiment": "blur_ghosting_extended_L6_L10",
                        "artifact": artifact,
                        "artifact_display": DISPLAY_NAMES[artifact],
                        "level": level,
                        "condition": condition,
                        "patient_id": patient_id,
                        "modality": modality_name,
                        "mse": mse,
                        "psnr": psnr,
                    })

                    condition_count += 1

            if condition_count != 296:
                raise RuntimeError(
                    f"{condition}: expected 296 measurements, "
                    f"found {condition_count}"
                )

    metrics_df = pd.DataFrame(rows)

    if len(metrics_df) != 2960:
        raise RuntimeError(
            f"Expected 2960 measurements, found {len(metrics_df)}"
        )

    metrics_df.to_csv(OUT_METRICS_CSV, index=False)

    new_summary = (
        metrics_df
        .groupby(
            [
                "experiment",
                "artifact",
                "artifact_display",
                "level",
                "condition",
            ],
            as_index=False,
        )
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            mse_median=("mse", "median"),
            psnr_mean=("psnr", "mean"),
            psnr_std=("psnr", "std"),
            psnr_median=("psnr", "median"),
            n_measurements=("mse", "count"),
            n_patients=("patient_id", "nunique"),
        )
    )

    new_summary.to_csv(
        OUT_NEW_SUMMARY_CSV,
        index=False,
    )

    existing_summary = pd.read_csv(EXISTING_SUMMARY)

    if len(existing_summary) != 40:
        raise RuntimeError(
            f"Expected 40 existing rows, found {len(existing_summary)}"
        )

    final_summary = pd.concat(
        [existing_summary, new_summary],
        ignore_index=True,
    )

    duplicate_conditions = final_summary[
        final_summary["condition"].duplicated(keep=False)
    ]

    if not duplicate_conditions.empty:
        raise RuntimeError(
            "Duplicate conditions found:\n"
            + duplicate_conditions[
                ["condition", "artifact", "level"]
            ].to_string(index=False)
        )

    if len(final_summary) != 50:
        raise RuntimeError(
            f"Expected 50 final rows, found {len(final_summary)}"
        )

    expected_conditions = {
        f"{artifact}_L{level}"
        for artifact in ARTIFACT_ORDER
        for level in range(1, 11)
    }

    actual_conditions = set(final_summary["condition"])

    missing_conditions = sorted(
        expected_conditions - actual_conditions
    )

    if missing_conditions:
        raise RuntimeError(
            f"Missing final conditions: {missing_conditions}"
        )

    final_summary["artifact_display"] = (
        final_summary["artifact"].map(DISPLAY_NAMES)
    )

    final_summary["artifact"] = pd.Categorical(
        final_summary["artifact"],
        categories=ARTIFACT_ORDER,
        ordered=True,
    )

    final_summary = final_summary.sort_values(
        ["artifact", "level"]
    ).reset_index(drop=True)

    final_summary.to_csv(
        OUT_FINAL_SUMMARY_CSV,
        index=False,
    )

    with open(
        OUT_FINAL_SUMMARY_TXT,
        "w",
        encoding="utf-8",
    ) as file:
        file.write("=" * 80 + "\n")
        file.write(
            "Script 33C: Final nnU-Net image-quality "
            "summary, all artifacts L1-L10\n"
        )
        file.write("=" * 80 + "\n\n")
        file.write("Conditions: 50\n")
        file.write("Patients per condition: 74\n")
        file.write("Modalities per patient: 4\n")
        file.write("Measurements per condition: 296\n\n")
        file.write(
            "Higher MSE indicates stronger image change.\n"
        )
        file.write(
            "Lower PSNR indicates stronger image change.\n\n"
        )

        file.write("L10 summary:\n")

        for _, row in final_summary[
            final_summary["level"] == 10
        ].iterrows():
            file.write(
                f"{row['artifact_display']}: "
                f"MSE={row['mse_mean']:.6f}, "
                f"PSNR={row['psnr_mean']:.2f} dB\n"
            )

    print()
    print("=" * 80)
    print("Final MSE/PSNR merge complete.")
    print(f"Detailed new metrics: {OUT_METRICS_CSV}")
    print(f"New 10-row summary: {OUT_NEW_SUMMARY_CSV}")
    print(f"Final 50-row summary: {OUT_FINAL_SUMMARY_CSV}")
    print(f"Final summary TXT: {OUT_FINAL_SUMMARY_TXT}")
    print("=" * 80)
    print()
    print("L10 preview:")
    print(
        final_summary.loc[
            final_summary["level"] == 10,
            [
                "artifact_display",
                "mse_mean",
                "psnr_mean",
                "n_measurements",
                "n_patients",
            ],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
