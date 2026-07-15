#!/usr/bin/env python3
"""
Script 30B: Relate objective degradation strength to nnU-Net performance drop.

Purpose
-------
Combine:

1. Full-cohort image degradation measurements from Script 30A:
   - MSE
   - PSNR

2. Full-cohort nnU-Net segmentation robustness results:
   - WT Dice drop
   - WT IoU drop

The resulting figures show whether stronger image-level degradation produces
larger segmentation performance loss, and whether that relationship differs
across artifact types.

Important interpretation
------------------------
MSE and PSNR measure image-level distortion, but they do not directly measure
segmentation damage.

Two artifacts with similar MSE or PSNR may produce very different Dice or IoU
drops. Therefore, artifact type matters in addition to objective distortion
strength.

Terminology
-----------
The stored artifact name "ringing" refers to Fourier truncation / a
frequency-domain ringing-like stress test. It should not be described as pure
classic Gibbs ringing.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(
    "/home/xfh25/brats_segmentation_project"
)

REPORT_DIR = PROJECT_ROOT / "report_materials"

IMAGE_QUALITY_CSV = (
    REPORT_DIR
    / "30a_nnunet_full_degradation_psnr_mse_summary.csv"
)

ORIGINAL_DROP_CSV = (
    REPORT_DIR
    / "26f_nnunet_robustness_drop_summary.csv"
)

EXTENDED_DROP_CSV = (
    REPORT_DIR
    / "29c_nnunet_extended_full_selected_drop_summary.csv"
)

OUT_COMBINED_CSV = (
    REPORT_DIR
    / "30b_nnunet_degradation_strength_vs_drop_summary.csv"
)

OUT_SUMMARY_TXT = (
    REPORT_DIR
    / "30b_nnunet_degradation_strength_vs_drop_summary.txt"
)

MSE_VS_DICE_PLOT = (
    REPORT_DIR
    / "30b_mse_vs_wt_dice_drop.png"
)

PSNR_VS_DICE_PLOT = (
    REPORT_DIR
    / "30b_psnr_vs_wt_dice_drop.png"
)

MSE_VS_IOU_PLOT = (
    REPORT_DIR
    / "30b_mse_vs_wt_iou_drop.png"
)

PSNR_VS_IOU_PLOT = (
    REPORT_DIR
    / "30b_psnr_vs_wt_iou_drop.png"
)


# ---------------------------------------------------------------------
# Study definitions
# ---------------------------------------------------------------------

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
    "noise": "Noise",
    "contrast": "Contrast",
    "ringing": "Frequency-domain truncation",
}

EXPECTED_ORIGINAL_ROWS = 25
EXPECTED_EXTENDED_ROWS = 15
EXPECTED_TOTAL_ROWS = 40


# ---------------------------------------------------------------------
# Data loading and standardization
# ---------------------------------------------------------------------

def load_image_quality_summary():
    """
    Load the full-cohort MSE/PSNR summary created by Script 30A.
    """
    if not IMAGE_QUALITY_CSV.exists():
        raise FileNotFoundError(
            f"Missing Script 30A summary: {IMAGE_QUALITY_CSV}"
        )

    df = pd.read_csv(
        IMAGE_QUALITY_CSV
    )

    required_columns = {
        "condition",
        "artifact",
        "level",
        "mse_mean",
        "psnr_mean",
        "n_patients",
        "n_measurements",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise RuntimeError(
            "Script 30A summary is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(df) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_ROWS} image-quality rows, "
            f"found {len(df)}"
        )

    return df


def load_original_drop_summary():
    """
    Load and standardize the original L1-L5 nnU-Net robustness results.
    """
    if not ORIGINAL_DROP_CSV.exists():
        raise FileNotFoundError(
            f"Missing original drop summary: {ORIGINAL_DROP_CSV}"
        )

    df = pd.read_csv(
        ORIGINAL_DROP_CSV
    )

    required_columns = {
        "condition",
        "artifact",
        "level",
        "num_test_patients",
        "drop_dice_WT",
        "drop_iou_WT",
        "degraded_dice_WT",
        "degraded_iou_WT",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise RuntimeError(
            "Original L1-L5 summary is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(df) != EXPECTED_ORIGINAL_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ORIGINAL_ROWS} original rows, "
            f"found {len(df)}"
        )

    standardized = df[
        [
            "condition",
            "artifact",
            "level",
            "num_test_patients",
            "degraded_dice_WT",
            "degraded_iou_WT",
            "drop_dice_WT",
            "drop_iou_WT",
        ]
    ].copy()

    standardized = standardized.rename(
        columns={
            "num_test_patients": "n_cases",
            "degraded_dice_WT": "dice_WT_mean",
            "degraded_iou_WT": "iou_WT_mean",
            "drop_dice_WT": "dice_WT_drop",
            "drop_iou_WT": "iou_WT_drop",
        }
    )

    standardized["experiment"] = "original_L1_L5"

    return standardized


def load_extended_drop_summary():
    """
    Load and standardize the extended L6-L10 nnU-Net robustness results.
    """
    if not EXTENDED_DROP_CSV.exists():
        raise FileNotFoundError(
            f"Missing extended drop summary: {EXTENDED_DROP_CSV}"
        )

    df = pd.read_csv(
        EXTENDED_DROP_CSV
    )

    required_columns = {
        "condition",
        "artifact",
        "level",
        "n_cases",
        "dice_WT_mean",
        "iou_WT_mean",
        "dice_WT_drop",
        "iou_WT_drop",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise RuntimeError(
            "Extended L6-L10 summary is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(df) != EXPECTED_EXTENDED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_EXTENDED_ROWS} extended rows, "
            f"found {len(df)}"
        )

    standardized = df[
        [
            "condition",
            "artifact",
            "level",
            "n_cases",
            "dice_WT_mean",
            "iou_WT_mean",
            "dice_WT_drop",
            "iou_WT_drop",
        ]
    ].copy()

    standardized["experiment"] = "extended_L6_L10"

    return standardized


def combine_drop_summaries():
    """
    Combine original and extended segmentation-drop summaries.
    """
    original_df = load_original_drop_summary()
    extended_df = load_extended_drop_summary()

    combined_df = pd.concat(
        [
            original_df,
            extended_df,
        ],
        ignore_index=True,
    )

    if len(combined_df) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_ROWS} combined drop rows, "
            f"found {len(combined_df)}"
        )

    duplicate_count = int(
        combined_df.duplicated(
            subset=[
                "condition",
                "artifact",
                "level",
            ]
        ).sum()
    )

    if duplicate_count != 0:
        raise RuntimeError(
            f"Found {duplicate_count} duplicate degradation conditions."
        )

    return combined_df


# ---------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------

def merge_image_quality_and_performance(
    image_quality_df,
    performance_df,
):
    """
    Merge MSE/PSNR results with Dice/IoU drop results.
    """
    merged_df = image_quality_df.merge(
        performance_df,
        on=[
            "condition",
            "artifact",
            "level",
        ],
        how="inner",
        validate="one_to_one",
        suffixes=("_quality", "_performance"),
    )

    if len(merged_df) != EXPECTED_TOTAL_ROWS:
        image_conditions = set(
            image_quality_df["condition"]
        )

        performance_conditions = set(
            performance_df["condition"]
        )

        missing_from_quality = sorted(
            performance_conditions - image_conditions
        )

        missing_from_performance = sorted(
            image_conditions - performance_conditions
        )

        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_ROWS} merged rows, "
            f"found {len(merged_df)}.\n"
            f"Missing from image-quality summary: "
            f"{missing_from_quality}\n"
            f"Missing from performance summary: "
            f"{missing_from_performance}"
        )

    merged_df["artifact_display"] = (
        merged_df["artifact"].map(
            DISPLAY_NAMES
        )
    )

    merged_df["artifact"] = pd.Categorical(
        merged_df["artifact"],
        categories=ARTIFACT_ORDER,
        ordered=True,
    )

    merged_df = (
        merged_df
        .sort_values(
            [
                "artifact",
                "level",
            ]
        )
        .reset_index(drop=True)
    )

    return merged_df


# ---------------------------------------------------------------------
# Correlation summaries
# ---------------------------------------------------------------------

def calculate_artifact_correlations(
    merged_df,
):
    """
    Calculate Pearson and Spearman correlations within each artifact.

    These are descriptive only because each artifact has a small number of
    severity levels and the conditions are ordered rather than independent
    observations.
    """
    rows = []

    for artifact in ARTIFACT_ORDER:
        subset = merged_df[
            merged_df["artifact"] == artifact
        ].copy()

        if len(subset) < 3:
            continue

        rows.append({
            "artifact": artifact,
            "artifact_display": DISPLAY_NAMES[artifact],
            "n_levels": len(subset),
            "pearson_mse_dice_drop": (
                subset["mse_mean"]
                .corr(
                    subset["dice_WT_drop"],
                    method="pearson",
                )
            ),
            "spearman_mse_dice_drop": (
                subset["mse_mean"]
                .corr(
                    subset["dice_WT_drop"],
                    method="spearman",
                )
            ),
            "pearson_psnr_dice_drop": (
                subset["psnr_mean"]
                .corr(
                    subset["dice_WT_drop"],
                    method="pearson",
                )
            ),
            "spearman_psnr_dice_drop": (
                subset["psnr_mean"]
                .corr(
                    subset["dice_WT_drop"],
                    method="spearman",
                )
            ),
            "pearson_mse_iou_drop": (
                subset["mse_mean"]
                .corr(
                    subset["iou_WT_drop"],
                    method="pearson",
                )
            ),
            "spearman_mse_iou_drop": (
                subset["mse_mean"]
                .corr(
                    subset["iou_WT_drop"],
                    method="spearman",
                )
            ),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_relationship(
    merged_df,
    x_column,
    y_column,
    x_label,
    y_label,
    title,
    output_path,
    reverse_x=False,
):
    """
    Draw one artifact-specific degradation-response plot.

    Each point represents one artifact severity level.
    Points belonging to the same artifact are joined in level order.
    """
    plt.figure(
        figsize=(10, 6)
    )

    for artifact in ARTIFACT_ORDER:
        subset = merged_df[
            merged_df["artifact"] == artifact
        ].sort_values("level")

        if subset.empty:
            continue

        plt.plot(
            subset[x_column],
            subset[y_column],
            marker="o",
            label=DISPLAY_NAMES[artifact],
        )

        # Label the final severity level for each artifact.
        last_row = subset.iloc[-1]

        plt.annotate(
            last_row["condition"],
            (
                last_row[x_column],
                last_row[y_column],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.axhline(
        0,
        linewidth=1,
        linestyle="--",
    )

    plt.xlabel(
        x_label
    )

    plt.ylabel(
        y_label
    )

    plt.title(
        title
    )

    if reverse_x:
        plt.gca().invert_xaxis()

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()


def make_all_plots(
    merged_df,
):
    """
    Create four report-ready degradation-response figures.
    """
    plot_relationship(
        merged_df=merged_df,
        x_column="mse_mean",
        y_column="dice_WT_drop",
        x_label="Mean MSE",
        y_label="WT Dice drop",
        title=(
            "Image degradation strength versus "
            "nnU-Net WT Dice drop"
        ),
        output_path=MSE_VS_DICE_PLOT,
    )

    plot_relationship(
        merged_df=merged_df,
        x_column="psnr_mean",
        y_column="dice_WT_drop",
        x_label="Mean PSNR (dB)",
        y_label="WT Dice drop",
        title=(
            "Image quality versus "
            "nnU-Net WT Dice drop"
        ),
        output_path=PSNR_VS_DICE_PLOT,
        reverse_x=True,
    )

    plot_relationship(
        merged_df=merged_df,
        x_column="mse_mean",
        y_column="iou_WT_drop",
        x_label="Mean MSE",
        y_label="WT IoU drop",
        title=(
            "Image degradation strength versus "
            "nnU-Net WT IoU drop"
        ),
        output_path=MSE_VS_IOU_PLOT,
    )

    plot_relationship(
        merged_df=merged_df,
        x_column="psnr_mean",
        y_column="iou_WT_drop",
        x_label="Mean PSNR (dB)",
        y_label="WT IoU drop",
        title=(
            "Image quality versus "
            "nnU-Net WT IoU drop"
        ),
        output_path=PSNR_VS_IOU_PLOT,
        reverse_x=True,
    )


# ---------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------

def write_summary_text(
    merged_df,
    correlation_df,
):
    """
    Save a readable report of merged values and descriptive correlations.
    """
    with open(
        OUT_SUMMARY_TXT,
        "w",
        encoding="utf-8",
    ) as file:

        file.write("=" * 88 + "\n")
        file.write(
            "Script 30B: nnU-Net degradation strength "
            "versus performance-drop summary\n"
        )
        file.write("=" * 88 + "\n\n")

        file.write(
            f"Conditions analyzed: {len(merged_df)}\n"
        )
        file.write(
            "Original artifacts L1-L5: blur, ghosting, noise, "
            "contrast, ringing/frequency-domain truncation\n"
        )
        file.write(
            "Extended artifacts L6-L10: noise, contrast, "
            "ringing/frequency-domain truncation\n\n"
        )

        file.write("Interpretation guidance:\n")
        file.write(
            "- Higher MSE indicates greater image-level distortion.\n"
        )
        file.write(
            "- Lower PSNR indicates greater image-level distortion.\n"
        )
        file.write(
            "- Higher positive Dice or IoU drop indicates greater "
            "segmentation damage.\n"
        )
        file.write(
            "- Small negative drops should be interpreted as stability, "
            "not genuine performance improvement.\n"
        )
        file.write(
            "- MSE and PSNR do not fully predict segmentation damage; "
            "artifact type also matters.\n\n"
        )

        file.write("Terminology note:\n")
        file.write(
            "The stored condition name 'ringing' refers to "
            "frequency-domain truncation / a ringing-like frequency "
            "stress test, not pure classic Gibbs ringing.\n\n"
        )

        file.write("Condition-level results:\n")

        for _, row in merged_df.iterrows():
            file.write(
                f"{row['condition']}: "
                f"MSE={row['mse_mean']:.6f}, "
                f"PSNR={row['psnr_mean']:.2f} dB, "
                f"WT Dice drop={row['dice_WT_drop']:.6f}, "
                f"WT IoU drop={row['iou_WT_drop']:.6f}\n"
            )

        file.write("\n")
        file.write("Artifact-level descriptive correlations:\n")
        file.write(
            "These correlations are descriptive only because they use "
            "a small number of ordered severity levels.\n\n"
        )

        for _, row in correlation_df.iterrows():
            file.write(
                f"{row['artifact_display']} "
                f"(n={int(row['n_levels'])} levels): "
                f"Spearman MSE vs WT Dice drop="
                f"{row['spearman_mse_dice_drop']:.3f}; "
                f"Spearman MSE vs WT IoU drop="
                f"{row['spearman_mse_iou_drop']:.3f}\n"
            )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("=" * 88)
    print(
        "Script 30B: Plot nnU-Net degradation strength "
        "versus performance drop"
    )
    print("=" * 88)

    REPORT_DIR.mkdir(
        exist_ok=True
    )

    image_quality_df = (
        load_image_quality_summary()
    )

    performance_df = (
        combine_drop_summaries()
    )

    merged_df = (
        merge_image_quality_and_performance(
            image_quality_df=image_quality_df,
            performance_df=performance_df,
        )
    )

    correlation_df = (
        calculate_artifact_correlations(
            merged_df
        )
    )

    output_columns = [
        "experiment_quality",
        "condition",
        "artifact",
        "artifact_display",
        "level",
        "n_patients",
        "n_measurements",
        "mse_mean",
        "mse_std",
        "psnr_mean",
        "psnr_std",
        "n_cases",
        "dice_WT_mean",
        "iou_WT_mean",
        "dice_WT_drop",
        "iou_WT_drop",
    ]

    missing_output_columns = [
        column
        for column in output_columns
        if column not in merged_df.columns
    ]

    if missing_output_columns:
        raise RuntimeError(
            "Merged file is missing output columns: "
            f"{missing_output_columns}"
        )

    merged_df[
        output_columns
    ].to_csv(
        OUT_COMBINED_CSV,
        index=False,
    )

    make_all_plots(
        merged_df
    )

    write_summary_text(
        merged_df=merged_df,
        correlation_df=correlation_df,
    )

    print()
    print("=" * 88)
    print("Script 30B completed")
    print("=" * 88)

    print(
        f"Combined summary: {OUT_COMBINED_CSV}"
    )

    print(
        f"Text summary:     {OUT_SUMMARY_TXT}"
    )

    print(
        f"MSE vs Dice:      {MSE_VS_DICE_PLOT}"
    )

    print(
        f"PSNR vs Dice:     {PSNR_VS_DICE_PLOT}"
    )

    print(
        f"MSE vs IoU:       {MSE_VS_IOU_PLOT}"
    )

    print(
        f"PSNR vs IoU:      {PSNR_VS_IOU_PLOT}"
    )

    print()
    print("Quick preview:")

    print(
        merged_df[
            [
                "condition",
                "mse_mean",
                "psnr_mean",
                "dice_WT_drop",
                "iou_WT_drop",
            ]
        ].to_string(index=False)
    )

    print()
    print("Descriptive artifact correlations:")

    print(
        correlation_df[
            [
                "artifact_display",
                "n_levels",
                "spearman_mse_dice_drop",
                "spearman_mse_iou_drop",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
