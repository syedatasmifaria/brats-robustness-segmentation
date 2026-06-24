#!/usr/bin/env python3
"""
Script 20: Prepare report-ready robustness materials for full 4-modal 3D U-Net.

Purpose:
- Read Script 18 degraded testing summary.
- Create clean report-ready tables with Dice and IoU.
- Create separate Dice and IoU robustness curves.
- Create level-5 Dice and IoU drop bar charts.
- Save all outputs to report_materials.

This script does NOT train or test the model.
It only organizes and visualizes the completed robustness results.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

INPUT_SUMMARY_CSV = PROJECT_ROOT / "results" / "18_full_multimodal_3d_degraded_test_summary.csv"

REPORT_DIR = PROJECT_ROOT / "report_materials"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_TABLE_CSV = REPORT_DIR / "20_robustness_summary_dice_iou.csv"
OUTPUT_TABLE_TXT = REPORT_DIR / "20_robustness_summary_dice_iou.txt"

DICE_CURVE_PNG = REPORT_DIR / "20_whole_tumor_dice_curve.png"
IOU_CURVE_PNG = REPORT_DIR / "20_whole_tumor_iou_curve.png"

LEVEL5_DICE_DROP_PNG = REPORT_DIR / "20_level5_dice_drop_bar.png"
LEVEL5_IOU_DROP_PNG = REPORT_DIR / "20_level5_iou_drop_bar.png"

CLASSWISE_TABLE_CSV = REPORT_DIR / "20_classwise_dice_iou_summary.csv"
CLASSWISE_TABLE_TXT = REPORT_DIR / "20_classwise_dice_iou_summary.txt"


# =============================================================================
# Settings
# =============================================================================

ARTIFACT_ORDER = ["clean", "blur", "noise", "contrast", "ringing", "ghosting"]

PLOT_ARTIFACTS = ["blur", "noise", "contrast", "ringing", "ghosting"]


# =============================================================================
# Helpers
# =============================================================================

def artifact_sort_key(artifact):
    try:
        return ARTIFACT_ORDER.index(artifact)
    except ValueError:
        return 999


def save_main_table(df):
    """
    Save a compact report-ready table with whole tumor Dice and IoU.
    """
    table = df.copy()

    table["artifact_order"] = table["artifact"].apply(artifact_sort_key)
    table = table.sort_values(["artifact_order", "level"]).reset_index(drop=True)

    keep_cols = [
        "artifact",
        "level",
        "mean_whole_tumor_dice",
        "whole_tumor_dice_drop",
        "mean_whole_tumor_iou",
        "whole_tumor_iou_drop",
    ]

    table = table[keep_cols]

    # Rounded version for easy reading
    rounded = table.copy()
    numeric_cols = [
        "mean_whole_tumor_dice",
        "whole_tumor_dice_drop",
        "mean_whole_tumor_iou",
        "whole_tumor_iou_drop",
    ]

    for col in numeric_cols:
        rounded[col] = rounded[col].round(4)

    rounded.to_csv(OUTPUT_TABLE_CSV, index=False)

    with open(OUTPUT_TABLE_TXT, "w") as f:
        f.write("4-modal 3D U-Net robustness summary: whole tumor Dice and IoU\n")
        f.write("=" * 80 + "\n\n")
        f.write(rounded.to_string(index=False))
        f.write("\n")

    print(f"Saved main table CSV: {OUTPUT_TABLE_CSV}")
    print(f"Saved main table TXT: {OUTPUT_TABLE_TXT}")


def save_classwise_table(df):
    """
    Save class-wise Dice/IoU table.
    Useful because whole tumor combines labels 1, 2, and 3.
    """
    table = df.copy()

    table["artifact_order"] = table["artifact"].apply(artifact_sort_key)
    table = table.sort_values(["artifact_order", "level"]).reset_index(drop=True)

    keep_cols = [
        "artifact",
        "level",
        "mean_dice_class_1",
        "mean_dice_class_2",
        "mean_dice_class_3",
        "mean_iou_class_1",
        "mean_iou_class_2",
        "mean_iou_class_3",
    ]

    table = table[keep_cols]

    numeric_cols = [c for c in table.columns if c not in ["artifact", "level"]]
    for col in numeric_cols:
        table[col] = table[col].round(4)

    table.to_csv(CLASSWISE_TABLE_CSV, index=False)

    with open(CLASSWISE_TABLE_TXT, "w") as f:
        f.write("4-modal 3D U-Net robustness summary: class-wise Dice and IoU\n")
        f.write("=" * 80 + "\n\n")
        f.write("Class labels:\n")
        f.write("class 1 = BraTS original label 1\n")
        f.write("class 2 = BraTS original label 2\n")
        f.write("class 3 = BraTS original label 4 remapped to 3\n\n")
        f.write(table.to_string(index=False))
        f.write("\n")

    print(f"Saved class-wise table CSV: {CLASSWISE_TABLE_CSV}")
    print(f"Saved class-wise table TXT: {CLASSWISE_TABLE_TXT}")


def save_metric_curve(df, metric_col, y_label, title, output_path):
    """
    Save line curve across degradation levels for Dice or IoU.
    """
    clean_value = df[df["artifact"] == "clean"][metric_col].iloc[0]

    plt.figure(figsize=(9, 6))

    for artifact in PLOT_ARTIFACTS:
        sub = df[df["artifact"] == artifact].sort_values("level")
        plt.plot(
            sub["level"],
            sub[metric_col],
            marker="o",
            label=artifact,
        )

    plt.axhline(
        clean_value,
        linestyle="--",
        label=f"clean baseline ({clean_value:.4f})",
    )

    plt.xlabel("Degradation level")
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks([1, 2, 3, 4, 5])
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved curve: {output_path}")


def save_level5_drop_bar(df, drop_col, y_label, title, output_path):
    """
    Save level-5 drop bar chart for Dice or IoU.
    """
    level5 = df[
        (df["artifact"] != "clean") &
        (df["level"] == 5)
    ].copy()

    level5 = level5.sort_values(drop_col, ascending=False)

    plt.figure(figsize=(8, 5))
    plt.bar(level5["artifact"], level5[drop_col])
    plt.xlabel("Degradation type")
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved level-5 bar chart: {output_path}")


def print_interpretation(df):
    """
    Print short report-ready interpretation.
    """
    clean = df[df["artifact"] == "clean"].iloc[0]

    clean_dice = clean["mean_whole_tumor_dice"]
    clean_iou = clean["mean_whole_tumor_iou"]

    level5 = df[
        (df["artifact"] != "clean") &
        (df["level"] == 5)
    ].copy()

    level5_dice = level5.sort_values("whole_tumor_dice_drop", ascending=False).iloc[0]
    level5_iou = level5.sort_values("whole_tumor_iou_drop", ascending=False).iloc[0]

    print("\n" + "=" * 80)
    print("Report-ready interpretation")
    print("=" * 80)

    print(f"Clean whole-tumor Dice: {clean_dice:.4f}")
    print(f"Clean whole-tumor IoU:  {clean_iou:.4f}")

    print("\nLargest level-5 Dice drop:")
    print(
        f"{level5_dice['artifact']} | "
        f"Dice={level5_dice['mean_whole_tumor_dice']:.4f}, "
        f"drop={level5_dice['whole_tumor_dice_drop']:.4f}"
    )

    print("\nLargest level-5 IoU drop:")
    print(
        f"{level5_iou['artifact']} | "
        f"IoU={level5_iou['mean_whole_tumor_iou']:.4f}, "
        f"drop={level5_iou['whole_tumor_iou_drop']:.4f}"
    )

    print("\nSuggested wording:")
    print(
        "The full 4-modal 3D U-Net achieved a clean patch-based whole-tumor "
        f"Dice of {clean_dice:.4f} and IoU of {clean_iou:.4f}. "
        "Under all-modality degradation, ghosting produced the largest "
        "level-5 performance reduction, followed by blur. Contrast, noise, "
        "and ringing showed minimal impact under the current degradation settings."
    )


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("Script 20: Prepare report-ready 4-modal 3D U-Net robustness materials")
    print("=" * 80)

    if not INPUT_SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing Script 18 summary CSV: {INPUT_SUMMARY_CSV}")

    df = pd.read_csv(INPUT_SUMMARY_CSV)

    print(f"Loaded summary CSV: {INPUT_SUMMARY_CSV}")
    print(f"Rows: {len(df)}")
    print("Columns:")
    for col in df.columns:
        print(f"  - {col}")

    required_cols = [
        "artifact",
        "level",
        "mean_whole_tumor_dice",
        "whole_tumor_dice_drop",
        "mean_whole_tumor_iou",
        "whole_tumor_iou_drop",
        "mean_dice_class_1",
        "mean_dice_class_2",
        "mean_dice_class_3",
        "mean_iou_class_1",
        "mean_iou_class_2",
        "mean_iou_class_3",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    save_main_table(df)
    save_classwise_table(df)

    save_metric_curve(
        df=df,
        metric_col="mean_whole_tumor_dice",
        y_label="Mean whole-tumor Dice",
        title="4-modal 3D U-Net: Dice under all-modality degradation",
        output_path=DICE_CURVE_PNG,
    )

    save_metric_curve(
        df=df,
        metric_col="mean_whole_tumor_iou",
        y_label="Mean whole-tumor IoU",
        title="4-modal 3D U-Net: IoU under all-modality degradation",
        output_path=IOU_CURVE_PNG,
    )

    save_level5_drop_bar(
        df=df,
        drop_col="whole_tumor_dice_drop",
        y_label="Dice drop from clean baseline",
        title="Level-5 degradation impact: Dice drop",
        output_path=LEVEL5_DICE_DROP_PNG,
    )

    save_level5_drop_bar(
        df=df,
        drop_col="whole_tumor_iou_drop",
        y_label="IoU drop from clean baseline",
        title="Level-5 degradation impact: IoU drop",
        output_path=LEVEL5_IOU_DROP_PNG,
    )

    print_interpretation(df)

    print("\n" + "=" * 80)
    print("Script 20 complete.")
    print("=" * 80)
    print("Saved report materials:")
    print(f"  {OUTPUT_TABLE_CSV}")
    print(f"  {OUTPUT_TABLE_TXT}")
    print(f"  {CLASSWISE_TABLE_CSV}")
    print(f"  {CLASSWISE_TABLE_TXT}")
    print(f"  {DICE_CURVE_PNG}")
    print(f"  {IOU_CURVE_PNG}")
    print(f"  {LEVEL5_DICE_DROP_PNG}")
    print(f"  {LEVEL5_IOU_DROP_PNG}")


if __name__ == "__main__":
    main()
