#!/usr/bin/env python3
"""
Script 33A: Merge final nnU-Net robustness results across all artifacts.

Purpose:
- Combine original L1-L5 results for all five artifacts.
- Add noise, contrast, and frequency-domain truncation L6-L10.
- Add blur and ghosting L6-L10.
- Standardize columns and terminology.
- Produce one final 50-condition summary table.

Important terminology:
- The old condition named "ringing" is Fourier/frequency-domain truncation.
- Preserve the internal artifact key for compatibility, but provide a clear
  report-facing display name.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")
REPORT_DIR = PROJECT_ROOT / "report_materials"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_PATH = (
    REPORT_DIR / "26f_nnunet_robustness_drop_summary.csv"
)

EXTENDED_SELECTED_PATH = (
    REPORT_DIR
    / "29c_nnunet_extended_full_selected_drop_summary.csv"
)

BLUR_GHOSTING_PATH = (
    REPORT_DIR
    / "32h_nnunet_blur_ghosting_extended_full_drop_summary.csv"
)

OUT_CSV = (
    REPORT_DIR
    / "33a_nnunet_all_artifacts_L1_L10_summary.csv"
)

OUT_TXT = (
    REPORT_DIR
    / "33a_nnunet_all_artifacts_L1_L10_summary.txt"
)

ARTIFACT_ORDER = [
    "blur",
    "ghosting",
    "noise",
    "contrast",
    "ringing",
]

DISPLAY_NAMES = {
    "blur": "Blur",
    "ghosting": "Ghosting",
    "noise": "Gaussian noise",
    "contrast": "Contrast reduction",
    "ringing": "Frequency-domain truncation",
}

FINAL_COLUMNS = [
    "condition",
    "artifact",
    "artifact_display",
    "level",
    "num_test_patients",
    "clean_dice_WT",
    "degraded_dice_WT",
    "drop_dice_WT",
    "clean_iou_WT",
    "degraded_iou_WT",
    "drop_iou_WT",
    "clean_dice_TC",
    "degraded_dice_TC",
    "drop_dice_TC",
    "clean_iou_TC",
    "degraded_iou_TC",
    "drop_iou_TC",
    "clean_dice_ET",
    "degraded_dice_ET",
    "drop_dice_ET",
    "clean_iou_ET",
    "degraded_iou_ET",
    "drop_iou_ET",
    "clean_pred_WT_voxels",
    "degraded_pred_WT_voxels",
    "clean_true_WT_voxels",
    "degraded_true_WT_voxels",
    "break_threshold_crossed",
]


def standardize_original(df):
    """Standardize Script 26F and Script 32H-style tables."""
    out = df.copy()

    out["artifact_display"] = out["artifact"].map(DISPLAY_NAMES)

    return out


def standardize_extended_selected(df, clean_reference):
    """Convert Script 29C columns into the common final format."""
    out = pd.DataFrame()

    out["condition"] = df["condition"]
    out["artifact"] = df["artifact"]
    out["artifact_display"] = out["artifact"].map(DISPLAY_NAMES)
    out["level"] = df["level"].astype(int)
    out["num_test_patients"] = df["n_cases"].astype(int)

    metric_pairs = {
        "dice_WT": "dice_WT_mean",
        "iou_WT": "iou_WT_mean",
        "dice_TC": "dice_TC_mean",
        "iou_TC": "iou_TC_mean",
        "dice_ET": "dice_ET_mean",
        "iou_ET": "iou_ET_mean",
    }

    for metric, degraded_column in metric_pairs.items():
        clean_value = clean_reference[f"clean_{metric}"]

        out[f"clean_{metric}"] = clean_value
        out[f"degraded_{metric}"] = df[degraded_column]
        out[f"drop_{metric}"] = (
            clean_value - df[degraded_column]
        )

    out["clean_pred_WT_voxels"] = (
        clean_reference["clean_pred_WT_voxels"]
    )

    out["degraded_pred_WT_voxels"] = (
        df["pred_WT_voxels_mean"]
    )

    out["clean_true_WT_voxels"] = (
        clean_reference["clean_true_WT_voxels"]
    )

    out["degraded_true_WT_voxels"] = (
        df["true_WT_voxels_mean"]
    )

    return out


def main():
    print("=" * 80)
    print("Script 33A: Merge nnU-Net all-artifact L1-L10 results")
    print("=" * 80)

    original_df = pd.read_csv(ORIGINAL_PATH)
    selected_df = pd.read_csv(EXTENDED_SELECTED_PATH)
    blur_ghost_df = pd.read_csv(BLUR_GHOSTING_PATH)

    clean_reference = {
        "clean_dice_WT": float(original_df["clean_dice_WT"].iloc[0]),
        "clean_iou_WT": float(original_df["clean_iou_WT"].iloc[0]),
        "clean_dice_TC": float(original_df["clean_dice_TC"].iloc[0]),
        "clean_iou_TC": float(original_df["clean_iou_TC"].iloc[0]),
        "clean_dice_ET": float(original_df["clean_dice_ET"].iloc[0]),
        "clean_iou_ET": float(original_df["clean_iou_ET"].iloc[0]),
        "clean_pred_WT_voxels": float(
            original_df["clean_pred_WT_voxels"].iloc[0]
        ),
        "clean_true_WT_voxels": float(
            original_df["clean_true_WT_voxels"].iloc[0]
        ),
    }

    original_standard = standardize_original(original_df)
    blur_ghost_standard = standardize_original(blur_ghost_df)

    selected_standard = standardize_extended_selected(
        selected_df,
        clean_reference,
    )

    combined = pd.concat(
        [
            original_standard,
            selected_standard,
            blur_ghost_standard,
        ],
        ignore_index=True,
        sort=False,
    )

    combined["level"] = combined["level"].astype(int)

    combined["break_threshold_crossed"] = (
        (combined["drop_dice_WT"] >= 0.10)
        | (combined["drop_iou_WT"] >= 0.15)
    )

    combined["artifact"] = pd.Categorical(
        combined["artifact"],
        categories=ARTIFACT_ORDER,
        ordered=True,
    )

    combined = combined.sort_values(
        ["artifact", "level"]
    ).reset_index(drop=True)

    duplicate_conditions = combined[
        combined["condition"].duplicated(keep=False)
    ]

    if not duplicate_conditions.empty:
        raise RuntimeError(
            "Duplicate conditions found:\n"
            + duplicate_conditions[
                ["condition", "artifact", "level"]
            ].to_string(index=False)
        )

    expected_conditions = {
        f"{artifact}_L{level}"
        for artifact in ARTIFACT_ORDER
        for level in range(1, 11)
    }

    actual_conditions = set(
        combined["condition"].astype(str)
    )

    missing_conditions = sorted(
        expected_conditions - actual_conditions
    )

    unexpected_conditions = sorted(
        actual_conditions - expected_conditions
    )

    if missing_conditions:
        raise RuntimeError(
            f"Missing conditions: {missing_conditions}"
        )

    if unexpected_conditions:
        raise RuntimeError(
            f"Unexpected conditions: {unexpected_conditions}"
        )

    if len(combined) != 50:
        raise RuntimeError(
            f"Expected 50 rows, found {len(combined)}"
        )

    numeric_check_columns = [
        "clean_dice_WT",
        "degraded_dice_WT",
        "drop_dice_WT",
        "clean_iou_WT",
        "degraded_iou_WT",
        "drop_iou_WT",
        "clean_dice_TC",
        "degraded_dice_TC",
        "drop_dice_TC",
        "clean_dice_ET",
        "degraded_dice_ET",
        "drop_dice_ET",
    ]

    if combined[numeric_check_columns].isna().any().any():
        bad_columns = combined[
            numeric_check_columns
        ].columns[
            combined[numeric_check_columns]
            .isna()
            .any()
        ].tolist()

        raise RuntimeError(
            f"Missing metric values in columns: {bad_columns}"
        )

    combined = combined[FINAL_COLUMNS]
    combined.to_csv(OUT_CSV, index=False)

    first_break_rows = []

    for artifact in ARTIFACT_ORDER:
        artifact_df = combined[
            combined["artifact"].astype(str) == artifact
        ].copy()

        broken = artifact_df[
            artifact_df["break_threshold_crossed"]
        ]

        if broken.empty:
            first_break_level = "None"
        else:
            first_break_level = int(
                broken["level"].min()
            )

        first_break_rows.append({
            "artifact": artifact,
            "artifact_display": DISPLAY_NAMES[artifact],
            "first_break_level": first_break_level,
        })

    first_break_df = pd.DataFrame(first_break_rows)

    with open(OUT_TXT, "w", encoding="utf-8") as file:
        file.write("=" * 80 + "\n")
        file.write(
            "Script 33A: Final nnU-Net all-artifact "
            "L1-L10 summary\n"
        )
        file.write("=" * 80 + "\n\n")

        file.write("Rows: 50\n")
        file.write("Patients per condition: 74\n")
        file.write("Artifacts: 5\n")
        file.write("Levels per artifact: 10\n\n")

        file.write("Drop formulas:\n")
        file.write(
            "Dice drop = clean Dice - degraded Dice\n"
        )
        file.write(
            "IoU drop = clean IoU - degraded IoU\n\n"
        )

        file.write("Practical breaking threshold:\n")
        file.write(
            "WT Dice drop >= 0.10 or "
            "WT IoU drop >= 0.15\n\n"
        )

        file.write("First breaking level by artifact:\n")

        for _, row in first_break_df.iterrows():
            file.write(
                f"{row['artifact_display']}: "
                f"{row['first_break_level']}\n"
            )

        file.write("\nL10 summary:\n")

        l10_df = combined[
            combined["level"] == 10
        ]

        for _, row in l10_df.iterrows():
            file.write(
                f"{row['artifact_display']}: "
                f"WT Dice={row['degraded_dice_WT']:.6f}, "
                f"WT Dice drop={row['drop_dice_WT']:.6f}, "
                f"WT IoU={row['degraded_iou_WT']:.6f}, "
                f"WT IoU drop={row['drop_iou_WT']:.6f}, "
                f"threshold crossed="
                f"{row['break_threshold_crossed']}\n"
            )

        file.write("\nTerminology note:\n")
        file.write(
            "The internal artifact key 'ringing' refers to "
            "Fourier/frequency-domain truncation and should "
            "not be described as pure classic Gibbs ringing.\n"
        )

    print(f"Saved final CSV: {OUT_CSV}")
    print(f"Saved final TXT: {OUT_TXT}")
    print(f"Rows: {len(combined)}")
    print()

    print("First breaking level:")
    print(first_break_df.to_string(index=False))
    print()

    print("L10 preview:")
    print(
        combined.loc[
            combined["level"] == 10,
            [
                "artifact_display",
                "degraded_dice_WT",
                "drop_dice_WT",
                "degraded_iou_WT",
                "drop_iou_WT",
                "break_threshold_crossed",
            ],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
