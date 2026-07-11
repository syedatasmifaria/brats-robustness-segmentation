#!/usr/bin/env python3
"""
Script 29C: Evaluate full 74-patient nnU-Net extended selected degradation test.

Purpose:
- Evaluate full extended L6-L10 predictions for:
  noise, contrast, ringing/frequency-domain truncation.
- Compare against labels for the same 74 held-out test patients.
- Compute WT, TC, ET Dice and IoU.
- Compute drops from the clean same-test nnU-Net baseline.
- Save report-ready summaries and curves.

Important:
The ringing condition here uses the original Fourier truncation implementation.
In the report, describe it as:
"frequency-domain ringing-like degradation" or
"Fourier truncation / low-pass frequency stress test,"
not pure classic Gibbs ringing.
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

EXT_ROOT = PROJECT_ROOT / "nnunet/temporary_degraded_tests/extended_full_selected"
LABELS_DIR = EXT_ROOT / "labelsTs"

CLEAN_METRICS_CSV = PROJECT_ROOT / "results/26e_nnunet_clean_same_test_metrics.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_DIR = PROJECT_ROOT / "report_materials"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_METRICS = RESULTS_DIR / "29c_nnunet_extended_full_selected_metrics.csv"

OUT_SUMMARY_CSV = REPORT_DIR / "29c_nnunet_extended_full_selected_drop_summary.csv"
OUT_SUMMARY_TXT = REPORT_DIR / "29c_nnunet_extended_full_selected_drop_summary.txt"

OUT_WT_DICE_CURVE = REPORT_DIR / "29c_extended_full_wt_dice_curve.png"
OUT_WT_IOU_CURVE = REPORT_DIR / "29c_extended_full_wt_iou_curve.png"
OUT_WT_DICE_DROP_CURVE = REPORT_DIR / "29c_extended_full_wt_dice_drop_curve.png"
OUT_WT_IOU_DROP_CURVE = REPORT_DIR / "29c_extended_full_wt_iou_drop_curve.png"

OUT_LEVEL10_DICE_DROP_BAR = REPORT_DIR / "29c_extended_full_level10_wt_dice_drop_bar.png"
OUT_LEVEL10_IOU_DROP_BAR = REPORT_DIR / "29c_extended_full_level10_wt_iou_drop_bar.png"

CONDITIONS = [
    "noise_L6", "noise_L7", "noise_L8", "noise_L9", "noise_L10",
    "contrast_L6", "contrast_L7", "contrast_L8", "contrast_L9", "contrast_L10",
    "ringing_L6", "ringing_L7", "ringing_L8", "ringing_L9", "ringing_L10",
]


# ------------------------------------------------------------
# Metric helpers
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

    inter = np.logical_and(pred, true).sum()
    return float((2.0 * inter) / denom)


def iou_score(pred, true):
    pred = pred.astype(bool)
    true = true.astype(bool)

    union = np.logical_or(pred, true).sum()

    if union == 0:
        return 1.0

    inter = np.logical_and(pred, true).sum()
    return float(inter / union)


def region_masks(seg):
    wt = seg > 0
    tc = np.logical_or(seg == 1, seg == 3)
    et = seg == 3
    return {"WT": wt, "TC": tc, "ET": et}


def load_seg(path):
    return nib.load(str(path)).get_fdata().astype(np.uint8)


def case_id_from_label_path(path):
    match = re.search(r"(\d{3})", path.name)
    if match is None:
        raise ValueError(f"Could not extract case id from: {path}")
    return match.group(1)


def parse_condition(condition):
    """
    noise_L10 -> artifact=noise, level=10
    """
    artifact, level_str = condition.split("_L")
    level = int(level_str)
    return artifact, level


def evaluate_one_prediction(pred_path, label_path, condition, artifact, level, case_id):
    pred = load_seg(pred_path)
    true = load_seg(label_path)

    pred_regions = region_masks(pred)
    true_regions = region_masks(true)

    row = {
        "condition": condition,
        "artifact": artifact,
        "level": level,
        "case_id": case_id,
        "pred_path": str(pred_path),
        "label_path": str(label_path),
        "pred_WT_voxels": int(pred_regions["WT"].sum()),
        "true_WT_voxels": int(true_regions["WT"].sum()),
    }

    for region in ["WT", "TC", "ET"]:
        row[f"dice_{region}"] = dice_score(pred_regions[region], true_regions[region])
        row[f"iou_{region}"] = iou_score(pred_regions[region], true_regions[region])

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
    print("Script 29C: Evaluate full 74-patient extended selected nnU-Net test")
    print("=" * 80)

    label_paths = sorted(LABELS_DIR.glob("*.nii.gz"))

    if len(label_paths) != 74:
        raise RuntimeError(f"Expected 74 labels, found {len(label_paths)} in {LABELS_DIR}")

    print(f"Labels found: {len(label_paths)}")

    rows = []

    for condition in CONDITIONS:
        artifact, level = parse_condition(condition)

        pred_dir = EXT_ROOT / condition / "predictions"
        pred_paths = sorted(pred_dir.glob("*.nii.gz"))

        print(f"Evaluating {condition}: predictions found = {len(pred_paths)}")

        if len(pred_paths) != 74:
            raise RuntimeError(f"Expected 74 predictions for {condition}, found {len(pred_paths)}")

        for label_path in label_paths:
            case_id = case_id_from_label_path(label_path)
            pred_path = pred_dir / f"BRATS_{case_id}.nii.gz"

            if not pred_path.exists():
                raise FileNotFoundError(f"Missing prediction: {pred_path}")

            rows.append(
                evaluate_one_prediction(
                    pred_path=pred_path,
                    label_path=label_path,
                    condition=condition,
                    artifact=artifact,
                    level=level,
                    case_id=case_id,
                )
            )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUT_METRICS, index=False)

    print(f"Saved full metrics CSV: {OUT_METRICS}")
    print(f"Rows evaluated: {len(metrics_df)}")

    # Extended summary
    summary = (
        metrics_df
        .groupby(["artifact", "level", "condition"])
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
        .sort_values(["artifact", "level"])
    )

    # Clean baseline from already-computed full 74-patient clean nnU-Net metrics.
    clean_df = pd.read_csv(CLEAN_METRICS_CSV)

    clean_baseline = {
        "dice_WT_mean": clean_df["dice_WT"].mean(),
        "iou_WT_mean": clean_df["iou_WT"].mean(),
        "dice_TC_mean": clean_df["dice_TC"].mean(),
        "iou_TC_mean": clean_df["iou_TC"].mean(),
        "dice_ET_mean": clean_df["dice_ET"].mean(),
        "iou_ET_mean": clean_df["iou_ET"].mean(),
    }

    for region in ["WT", "TC", "ET"]:
        summary[f"dice_{region}_drop"] = clean_baseline[f"dice_{region}_mean"] - summary[f"dice_{region}_mean"]
        summary[f"iou_{region}_drop"] = clean_baseline[f"iou_{region}_mean"] - summary[f"iou_{region}_mean"]

    summary.to_csv(OUT_SUMMARY_CSV, index=False)

    with open(OUT_SUMMARY_TXT, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Script 29C: Full 74-patient extended selected nnU-Net drop summary\n")
        f.write("=" * 80 + "\n\n")

        f.write("Drop formula:\n")
        f.write("Drop = clean same-test baseline metric - degraded metric\n\n")

        f.write("Clean full 74-patient baseline:\n")
        f.write(f"WT Dice: {clean_baseline['dice_WT_mean']:.6f}\n")
        f.write(f"WT IoU:  {clean_baseline['iou_WT_mean']:.6f}\n")
        f.write(f"TC Dice: {clean_baseline['dice_TC_mean']:.6f}\n")
        f.write(f"TC IoU:  {clean_baseline['iou_TC_mean']:.6f}\n")
        f.write(f"ET Dice: {clean_baseline['dice_ET_mean']:.6f}\n")
        f.write(f"ET IoU:  {clean_baseline['iou_ET_mean']:.6f}\n\n")

        f.write("Extended full-cohort results:\n")
        for _, r in summary.iterrows():
            f.write(
                f"{r['condition']}: "
                f"WT Dice {r['dice_WT_mean']:.6f}, "
                f"WT Dice drop {r['dice_WT_drop']:.6f}, "
                f"WT IoU {r['iou_WT_mean']:.6f}, "
                f"WT IoU drop {r['iou_WT_drop']:.6f}, "
                f"Pred WT voxels {r['pred_WT_voxels_mean']:.1f}\n"
            )

        f.write("\nBreaking threshold used for interpretation:\n")
        f.write("- WT Dice drop >= 0.10 or WT IoU drop >= 0.15\n\n")

        f.write("Important reporting note:\n")
        f.write(
            "The ringing condition uses Fourier truncation and should be described as "
            "frequency-domain ringing-like degradation or Fourier truncation / low-pass "
            "frequency stress test, not pure classic Gibbs ringing.\n"
        )

    print(f"Saved summary CSV: {OUT_SUMMARY_CSV}")
    print(f"Saved summary TXT: {OUT_SUMMARY_TXT}")

    # Plots
    def save_metric_curve(y_col, out_path, ylabel, title):
        plt.figure(figsize=(8, 5))

        for artifact in ["noise", "contrast", "ringing"]:
            sub = summary[summary["artifact"] == artifact].sort_values("level")
            plt.plot(sub["level"], sub[y_col], marker="o", label=artifact)

        plt.xlabel("Extended severity level")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks([6, 7, 8, 9, 10])
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

    save_metric_curve(
        "dice_WT_mean",
        OUT_WT_DICE_CURVE,
        "WT Dice",
        "Full 74-patient extended nnU-Net: WT Dice",
    )

    save_metric_curve(
        "iou_WT_mean",
        OUT_WT_IOU_CURVE,
        "WT IoU",
        "Full 74-patient extended nnU-Net: WT IoU",
    )

    save_metric_curve(
        "dice_WT_drop",
        OUT_WT_DICE_DROP_CURVE,
        "WT Dice drop from clean",
        "Full 74-patient extended nnU-Net: WT Dice drop",
    )

    save_metric_curve(
        "iou_WT_drop",
        OUT_WT_IOU_DROP_CURVE,
        "WT IoU drop from clean",
        "Full 74-patient extended nnU-Net: WT IoU drop",
    )

    # Level 10 bar plots
    level10 = summary[summary["level"] == 10].copy().sort_values("artifact")

    plt.figure(figsize=(7, 5))
    plt.bar(level10["artifact"], level10["dice_WT_drop"])
    plt.xlabel("Artifact")
    plt.ylabel("WT Dice drop from clean")
    plt.title("Full 74-patient extended nnU-Net: Level 10 WT Dice drop")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_LEVEL10_DICE_DROP_BAR, dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.bar(level10["artifact"], level10["iou_WT_drop"])
    plt.xlabel("Artifact")
    plt.ylabel("WT IoU drop from clean")
    plt.title("Full 74-patient extended nnU-Net: Level 10 WT IoU drop")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_LEVEL10_IOU_DROP_BAR, dpi=200)
    plt.close()

    print(f"Saved plot: {OUT_WT_DICE_CURVE}")
    print(f"Saved plot: {OUT_WT_IOU_CURVE}")
    print(f"Saved plot: {OUT_WT_DICE_DROP_CURVE}")
    print(f"Saved plot: {OUT_WT_IOU_DROP_CURVE}")
    print(f"Saved plot: {OUT_LEVEL10_DICE_DROP_BAR}")
    print(f"Saved plot: {OUT_LEVEL10_IOU_DROP_BAR}")

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
