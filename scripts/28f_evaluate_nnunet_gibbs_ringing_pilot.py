#!/usr/bin/env python3
"""
Script 28F: Evaluate nnU-Net Gibbs-like ringing pilot predictions.

Purpose:
- Evaluate the 5-patient gibbs_ringing pilot predictions.
- Compare predictions against ground-truth labels.
- Compute WT, TC, ET Dice and IoU.
- Compute metric drops from clean predictions on the same 5 patients.
- Save report-ready CSV/TXT summaries and plots.

Important:
This is a pilot evaluation only.
It is not the full 74-patient final evaluation.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

PILOT_ROOT = PROJECT_ROOT / "nnunet/temporary_degraded_tests/gibbs_ringing_pilot"
LABELS_DIR = PILOT_ROOT / "labelsTs"

# Existing clean nnU-Net predictions from Script 26E
CLEAN_PRED_DIR = PROJECT_ROOT / "nnunet/temporary_degraded_tests/final_full/clean/predictions"

RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_DIR = PROJECT_ROOT / "report_materials"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_METRICS = RESULTS_DIR / "28f_nnunet_gibbs_ringing_pilot_metrics.csv"

OUT_SUMMARY_CSV = REPORT_DIR / "28f_nnunet_gibbs_ringing_pilot_drop_summary.csv"
OUT_SUMMARY_TXT = REPORT_DIR / "28f_nnunet_gibbs_ringing_pilot_drop_summary.txt"

OUT_WT_DICE_CURVE = REPORT_DIR / "28f_gibbs_ringing_pilot_wt_dice_curve.png"
OUT_WT_IOU_CURVE = REPORT_DIR / "28f_gibbs_ringing_pilot_wt_iou_curve.png"
OUT_WT_DICE_DROP_CURVE = REPORT_DIR / "28f_gibbs_ringing_pilot_wt_dice_drop_curve.png"
OUT_WT_IOU_DROP_CURVE = REPORT_DIR / "28f_gibbs_ringing_pilot_wt_iou_drop_curve.png"

LEVELS = ["gibbs_L1", "gibbs_L2", "gibbs_L3", "gibbs_L4", "gibbs_L5"]


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def dice_score(pred, true):
    pred = pred.astype(bool)
    true = true.astype(bool)

    pred_sum = pred.sum()
    true_sum = true.sum()

    if pred_sum == 0 and true_sum == 0:
        return 1.0

    denom = pred_sum + true_sum

    if denom == 0:
        return 0.0

    intersection = np.logical_and(pred, true).sum()
    return float((2.0 * intersection) / denom)


def iou_score(pred, true):
    pred = pred.astype(bool)
    true = true.astype(bool)

    union = np.logical_or(pred, true).sum()

    if union == 0:
        return 1.0

    intersection = np.logical_and(pred, true).sum()
    return float(intersection / union)


def region_masks(seg):
    """
    Labels:
    0 background
    1 original BraTS label 1
    2 original BraTS label 2
    3 original BraTS label 4 remapped

    WT = 1 + 2 + 3
    TC = 1 + 3
    ET = 3
    """
    wt = seg > 0
    tc = np.logical_or(seg == 1, seg == 3)
    et = seg == 3

    return {
        "WT": wt,
        "TC": tc,
        "ET": et,
    }


def load_seg(path):
    return nib.load(str(path)).get_fdata().astype(np.uint8)


def case_id_from_label_path(path):
    """
    BraTS20_Training_001.nii.gz -> 001
    """
    match = re.search(r"(\d{3})", path.name)
    if match is None:
        raise ValueError(f"Could not extract case id from: {path}")
    return match.group(1)


def clean_pred_path_for_case(case_id):
    """
    Clean predictions are named like:
    BRATS_001.nii.gz
    """
    return CLEAN_PRED_DIR / f"BRATS_{case_id}.nii.gz"


def degraded_pred_path_for_case(level, case_id):
    """
    Degraded predictions are named like:
    BRATS_001.nii.gz
    """
    return PILOT_ROOT / level / "predictions" / f"BRATS_{case_id}.nii.gz"


def evaluate_one_prediction(pred_path, label_path, condition, case_id):
    pred = load_seg(pred_path)
    true = load_seg(label_path)

    pred_regions = region_masks(pred)
    true_regions = region_masks(true)

    row = {
        "condition": condition,
        "case_id": case_id,
        "pred_path": str(pred_path),
        "label_path": str(label_path),
        "pred_WT_voxels": int(pred_regions["WT"].sum()),
        "true_WT_voxels": int(true_regions["WT"].sum()),
    }

    for region in ["WT", "TC", "ET"]:
        row[f"dice_{region}"] = dice_score(pred_regions[region], true_regions[region])
        row[f"iou_{region}"] = iou_score(pred_regions[region], true_regions[region])

    # Classwise metrics
    for cls in [1, 2, 3]:
        pred_cls = pred == cls
        true_cls = true == cls

        row[f"dice_class_{cls}"] = dice_score(pred_cls, true_cls)
        row[f"iou_class_{cls}"] = iou_score(pred_cls, true_cls)

    return row


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("=" * 80)
    print("Script 28F: Evaluate nnU-Net Gibbs-like ringing pilot")
    print("=" * 80)

    label_paths = sorted(LABELS_DIR.glob("*.nii.gz"))

    if len(label_paths) == 0:
        raise FileNotFoundError(f"No labels found in: {LABELS_DIR}")

    print(f"Labels found: {len(label_paths)}")

    rows = []

    # Clean baseline on same 5 patients
    for label_path in label_paths:
        case_id = case_id_from_label_path(label_path)
        pred_path = clean_pred_path_for_case(case_id)

        if not pred_path.exists():
            raise FileNotFoundError(f"Missing clean prediction for case {case_id}: {pred_path}")

        rows.append(
            evaluate_one_prediction(
                pred_path=pred_path,
                label_path=label_path,
                condition="clean",
                case_id=case_id,
            )
        )

    # Gibbs predictions
    for level in LEVELS:
        for label_path in label_paths:
            case_id = case_id_from_label_path(label_path)
            pred_path = degraded_pred_path_for_case(level, case_id)

            if not pred_path.exists():
                raise FileNotFoundError(f"Missing prediction for {level}, case {case_id}: {pred_path}")

            rows.append(
                evaluate_one_prediction(
                    pred_path=pred_path,
                    label_path=label_path,
                    condition=level,
                    case_id=case_id,
                )
            )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUT_METRICS, index=False)

    print(f"Saved full metrics CSV: {OUT_METRICS}")
    print(f"Rows evaluated: {len(metrics_df)}")

    # Summary means
    summary = (
        metrics_df
        .groupby("condition")
        .agg(
            n_cases=("case_id", "nunique"),
            dice_WT_mean=("dice_WT", "mean"),
            iou_WT_mean=("iou_WT", "mean"),
            dice_TC_mean=("dice_TC", "mean"),
            iou_TC_mean=("iou_TC", "mean"),
            dice_ET_mean=("dice_ET", "mean"),
            iou_ET_mean=("iou_ET", "mean"),
            pred_WT_voxels_mean=("pred_WT_voxels", "mean"),
            true_WT_voxels_mean=("true_WT_voxels", "mean"),
            dice_class_1_mean=("dice_class_1", "mean"),
            iou_class_1_mean=("iou_class_1", "mean"),
            dice_class_2_mean=("dice_class_2", "mean"),
            iou_class_2_mean=("iou_class_2", "mean"),
            dice_class_3_mean=("dice_class_3", "mean"),
            iou_class_3_mean=("iou_class_3", "mean"),
        )
        .reset_index()
    )

    order = ["clean"] + LEVELS
    summary["condition"] = pd.Categorical(summary["condition"], categories=order, ordered=True)
    summary = summary.sort_values("condition").reset_index(drop=True)
    summary["level_num"] = summary["condition"].astype(str).str.extract(r"L(\d+)").astype(float)

    clean_row = summary[summary["condition"].astype(str) == "clean"].iloc[0]

    for region in ["WT", "TC", "ET"]:
        summary[f"dice_{region}_drop"] = clean_row[f"dice_{region}_mean"] - summary[f"dice_{region}_mean"]
        summary[f"iou_{region}_drop"] = clean_row[f"iou_{region}_mean"] - summary[f"iou_{region}_mean"]

    summary.to_csv(OUT_SUMMARY_CSV, index=False)

    with open(OUT_SUMMARY_TXT, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Script 28F: nnU-Net Gibbs-like ringing pilot drop summary\n")
        f.write("=" * 80 + "\n\n")
        f.write("Drop formula:\n")
        f.write("Drop = clean pilot metric - degraded pilot metric\n\n")

        f.write("Clean pilot baseline, same 5 patients:\n")
        f.write(f"WT Dice: {clean_row['dice_WT_mean']:.6f}\n")
        f.write(f"WT IoU:  {clean_row['iou_WT_mean']:.6f}\n")
        f.write(f"TC Dice: {clean_row['dice_TC_mean']:.6f}\n")
        f.write(f"TC IoU:  {clean_row['iou_TC_mean']:.6f}\n")
        f.write(f"ET Dice: {clean_row['dice_ET_mean']:.6f}\n")
        f.write(f"ET IoU:  {clean_row['iou_ET_mean']:.6f}\n\n")

        f.write("Gibbs ringing pilot results:\n")
        for _, r in summary.iterrows():
            condition = str(r["condition"])
            if condition == "clean":
                continue

            f.write(
                f"{condition}: "
                f"WT Dice {r['dice_WT_mean']:.6f}, "
                f"WT Dice drop {r['dice_WT_drop']:.6f}, "
                f"WT IoU {r['iou_WT_mean']:.6f}, "
                f"WT IoU drop {r['iou_WT_drop']:.6f}, "
                f"Pred WT voxels {r['pred_WT_voxels_mean']:.1f}\n"
            )

        f.write("\nInterpretation guide:\n")
        f.write("- This is a 5-patient pilot only.\n")
        f.write("- Positive drop means degraded performance is worse than clean.\n")
        f.write("- Negative tiny drops should not be interpreted as true improvement.\n")
        f.write("- If WT Dice drop approaches or exceeds 0.10, the artifact is causing a substantial robustness failure.\n")
        f.write("- If WT IoU drop approaches or exceeds 0.15, the artifact is causing a substantial robustness failure.\n")

    print(f"Saved summary CSV: {OUT_SUMMARY_CSV}")
    print(f"Saved summary TXT: {OUT_SUMMARY_TXT}")

    # Plot helper
    plot_df = summary[summary["condition"].astype(str) != "clean"].copy()

    def save_curve(y_col, out_path, ylabel, title):
        plt.figure(figsize=(7, 5))
        plt.plot(plot_df["level_num"], plot_df[y_col], marker="o")
        plt.xlabel("Gibbs ringing severity level")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(plot_df["level_num"])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

    save_curve(
        "dice_WT_mean",
        OUT_WT_DICE_CURVE,
        "WT Dice",
        "nnU-Net Gibbs-like ringing pilot: WT Dice",
    )

    save_curve(
        "iou_WT_mean",
        OUT_WT_IOU_CURVE,
        "WT IoU",
        "nnU-Net Gibbs-like ringing pilot: WT IoU",
    )

    save_curve(
        "dice_WT_drop",
        OUT_WT_DICE_DROP_CURVE,
        "WT Dice drop",
        "nnU-Net Gibbs-like ringing pilot: WT Dice drop",
    )

    save_curve(
        "iou_WT_drop",
        OUT_WT_IOU_DROP_CURVE,
        "WT IoU drop",
        "nnU-Net Gibbs-like ringing pilot: WT IoU drop",
    )

    print(f"Saved plot: {OUT_WT_DICE_CURVE}")
    print(f"Saved plot: {OUT_WT_IOU_CURVE}")
    print(f"Saved plot: {OUT_WT_DICE_DROP_CURVE}")
    print(f"Saved plot: {OUT_WT_IOU_DROP_CURVE}")

    print("\nQuick preview:")
    preview_cols = [
        "condition",
        "dice_WT_mean",
        "iou_WT_mean",
        "dice_WT_drop",
        "iou_WT_drop",
        "pred_WT_voxels_mean",
        "true_WT_voxels_mean",
    ]
    print(summary[preview_cols].to_string(index=False))

    print("=" * 80)


if __name__ == "__main__":
    main()
