#!/usr/bin/env python3
"""
Step 26E-B: Evaluate clean nnU-Net predictions on the same 74 test patients.

Purpose:
- Evaluate clean nnU-Net test predictions against the same labels used for the
  final degraded nnU-Net evaluation.
- Compute WT, TC, ET Dice and IoU.
- Compute class-wise Dice and IoU for classes 1, 2, 3.
- Save per-patient metrics and clean summary files.

Important:
- This is testing/evaluation only.
- No training happens here.
- Clean predictions are named like:
    BRATS_026.nii.gz
- Labels are named like:
    BraTS20_Training_026.nii.gz
- We match by the numeric case ID.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import nibabel as nib


PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

PRED_DIR = PROJECT_ROOT / "nnunet/temporary_degraded_tests/final_full/clean/predictions"
LABEL_DIR = PROJECT_ROOT / "nnunet/temporary_degraded_tests/final_full/labelsTs"

RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_DIR = PROJECT_ROOT / "report_materials"

METRICS_CSV = RESULTS_DIR / "26e_nnunet_clean_same_test_metrics.csv"
SUMMARY_CSV = REPORT_DIR / "26e_nnunet_clean_same_test_summary.csv"
SUMMARY_TXT = REPORT_DIR / "26e_nnunet_clean_same_test_summary.txt"


def extract_case_number(path: Path) -> str:
    """
    Extract the 3-digit BraTS case number from filenames.

    Examples:
    BRATS_026.nii.gz -> 026
    BraTS20_Training_026.nii.gz -> 026
    """
    match = re.search(r"(\d{3})", path.name)
    if match is None:
        raise ValueError(f"Could not extract case number from filename: {path.name}")
    return match.group(1)


def load_nii(path: Path) -> np.ndarray:
    """Load NIfTI file and return integer label array."""
    arr = nib.load(str(path)).get_fdata()
    return arr.astype(np.int16)


def dice_score(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """
    Dice = 2 * intersection / (prediction + truth)

    If both prediction and truth are empty, return 1.0.
    This means the model correctly predicted absence of that region.
    """
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)

    pred_sum = pred_mask.sum()
    true_sum = true_mask.sum()

    if pred_sum == 0 and true_sum == 0:
        return 1.0

    denom = pred_sum + true_sum
    if denom == 0:
        return 0.0

    intersection = np.logical_and(pred_mask, true_mask).sum()
    return float((2.0 * intersection) / denom)


def iou_score(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """
    IoU = intersection / union

    If both prediction and truth are empty, return 1.0.
    """
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)

    union = np.logical_or(pred_mask, true_mask).sum()

    if union == 0:
        return 1.0

    intersection = np.logical_and(pred_mask, true_mask).sum()
    return float(intersection / union)


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """
    Compute BraTS region metrics and class-wise metrics.

    Label convention:
    0 = background
    1 = BraTS label 1
    2 = BraTS label 2
    3 = original BraTS label 4 remapped to 3

    Regions:
    WT = labels 1 + 2 + 3
    TC = labels 1 + 3
    ET = label 3
    """
    pred_wt = pred > 0
    true_wt = true > 0

    pred_tc = np.logical_or(pred == 1, pred == 3)
    true_tc = np.logical_or(true == 1, true == 3)

    pred_et = pred == 3
    true_et = true == 3

    metrics = {
        "dice_WT": dice_score(pred_wt, true_wt),
        "iou_WT": iou_score(pred_wt, true_wt),

        "dice_TC": dice_score(pred_tc, true_tc),
        "iou_TC": iou_score(pred_tc, true_tc),

        "dice_ET": dice_score(pred_et, true_et),
        "iou_ET": iou_score(pred_et, true_et),

        "pred_WT_voxels": int(pred_wt.sum()),
        "true_WT_voxels": int(true_wt.sum()),
    }

    for cls in [1, 2, 3]:
        pred_cls = pred == cls
        true_cls = true == cls

        metrics[f"dice_class_{cls}"] = dice_score(pred_cls, true_cls)
        metrics[f"iou_class_{cls}"] = iou_score(pred_cls, true_cls)

    return metrics


def main():
    print("=" * 80)
    print("Step 26E-B: Evaluate clean nnU-Net predictions on same 74 test patients")
    print("=" * 80)

    RESULTS_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    pred_paths = sorted(PRED_DIR.glob("*.nii.gz"))
    label_paths = sorted(LABEL_DIR.glob("*.nii.gz"))

    print(f"Prediction directory: {PRED_DIR}")
    print(f"Label directory:      {LABEL_DIR}")
    print(f"Predictions found:    {len(pred_paths)}")
    print(f"Labels found:         {len(label_paths)}")

    if len(pred_paths) == 0:
        raise RuntimeError("No prediction files found. Clean nnU-Net prediction step did not produce outputs.")

    if len(label_paths) == 0:
        raise RuntimeError("No label files found.")

    pred_by_case = {extract_case_number(p): p for p in pred_paths}
    label_by_case = {extract_case_number(p): p for p in label_paths}

    common_cases = sorted(set(pred_by_case.keys()) & set(label_by_case.keys()))

    missing_labels = sorted(set(pred_by_case.keys()) - set(label_by_case.keys()))
    missing_preds = sorted(set(label_by_case.keys()) - set(pred_by_case.keys()))

    print(f"Matched cases:        {len(common_cases)}")

    if missing_labels:
        print(f"WARNING: predictions without labels: {missing_labels}")

    if missing_preds:
        print(f"WARNING: labels without predictions: {missing_preds}")

    if len(common_cases) != 74:
        raise RuntimeError(f"Expected 74 matched cases, but found {len(common_cases)}.")

    rows = []

    for i, case_id in enumerate(common_cases, start=1):
        pred_path = pred_by_case[case_id]
        label_path = label_by_case[case_id]

        print(f"[{i:02d}/74] Evaluating case {case_id}")

        pred = load_nii(pred_path)
        true = load_nii(label_path)

        if pred.shape != true.shape:
            raise RuntimeError(
                f"Shape mismatch for case {case_id}: "
                f"pred shape {pred.shape}, label shape {true.shape}"
            )

        metrics = compute_metrics(pred, true)

        row = {
            "case_id": case_id,
            "prediction_file": pred_path.name,
            "label_file": label_path.name,
            **metrics,
        }

        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(METRICS_CSV, index=False)

    metric_cols = [
        col for col in metrics_df.columns
        if col.startswith("dice_")
        or col.startswith("iou_")
        or col in ["pred_WT_voxels", "true_WT_voxels"]
    ]

    summary_rows = []

    for col in metric_cols:
        summary_rows.append({
            "metric": col,
            "mean": metrics_df[col].mean(),
            "std": metrics_df[col].std(),
            "min": metrics_df[col].min(),
            "max": metrics_df[col].max(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    clean_summary = {
        "num_test_patients": len(metrics_df),
        "dice_WT_mean": metrics_df["dice_WT"].mean(),
        "iou_WT_mean": metrics_df["iou_WT"].mean(),
        "dice_TC_mean": metrics_df["dice_TC"].mean(),
        "iou_TC_mean": metrics_df["iou_TC"].mean(),
        "dice_ET_mean": metrics_df["dice_ET"].mean(),
        "iou_ET_mean": metrics_df["iou_ET"].mean(),
        "pred_WT_voxels_mean": metrics_df["pred_WT_voxels"].mean(),
        "true_WT_voxels_mean": metrics_df["true_WT_voxels"].mean(),
        "dice_class_1_mean": metrics_df["dice_class_1"].mean(),
        "iou_class_1_mean": metrics_df["iou_class_1"].mean(),
        "dice_class_2_mean": metrics_df["dice_class_2"].mean(),
        "iou_class_2_mean": metrics_df["iou_class_2"].mean(),
        "dice_class_3_mean": metrics_df["dice_class_3"].mean(),
        "iou_class_3_mean": metrics_df["iou_class_3"].mean(),
    }

    with open(SUMMARY_TXT, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Step 26E-B: Clean nnU-Net same-test evaluation summary\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Number of test patients: {clean_summary['num_test_patients']}\n\n")

        f.write("BraTS region metrics:\n")
        f.write(f"WT Dice mean: {clean_summary['dice_WT_mean']:.6f}\n")
        f.write(f"WT IoU mean:  {clean_summary['iou_WT_mean']:.6f}\n")
        f.write(f"TC Dice mean: {clean_summary['dice_TC_mean']:.6f}\n")
        f.write(f"TC IoU mean:  {clean_summary['iou_TC_mean']:.6f}\n")
        f.write(f"ET Dice mean: {clean_summary['dice_ET_mean']:.6f}\n")
        f.write(f"ET IoU mean:  {clean_summary['iou_ET_mean']:.6f}\n\n")

        f.write("Class-wise metrics:\n")
        f.write(f"Class 1 Dice mean: {clean_summary['dice_class_1_mean']:.6f}\n")
        f.write(f"Class 1 IoU mean:  {clean_summary['iou_class_1_mean']:.6f}\n")
        f.write(f"Class 2 Dice mean: {clean_summary['dice_class_2_mean']:.6f}\n")
        f.write(f"Class 2 IoU mean:  {clean_summary['iou_class_2_mean']:.6f}\n")
        f.write(f"Class 3 Dice mean: {clean_summary['dice_class_3_mean']:.6f}\n")
        f.write(f"Class 3 IoU mean:  {clean_summary['iou_class_3_mean']:.6f}\n\n")

        f.write("Voxel diagnostics:\n")
        f.write(f"Predicted WT voxels mean: {clean_summary['pred_WT_voxels_mean']:.1f}\n")
        f.write(f"True WT voxels mean:      {clean_summary['true_WT_voxels_mean']:.1f}\n\n")

        f.write("Interpretation note:\n")
        f.write(
            "This is the clean nnU-Net baseline on the same 74 test patients used "
            "for final degraded nnU-Net evaluation. These clean values should be "
            "used as the anchor for Dice and IoU drop calculations.\n"
        )

    print("=" * 80)
    print("Clean nnU-Net same-test evaluation complete.")
    print(f"Saved per-patient metrics CSV: {METRICS_CSV}")
    print(f"Saved summary CSV: {SUMMARY_CSV}")
    print(f"Saved summary TXT: {SUMMARY_TXT}")
    print("=" * 80)
    print()
    print("Quick preview:")
    print(f"WT Dice mean: {clean_summary['dice_WT_mean']:.6f}")
    print(f"WT IoU mean:  {clean_summary['iou_WT_mean']:.6f}")
    print(f"TC Dice mean: {clean_summary['dice_TC_mean']:.6f}")
    print(f"TC IoU mean:  {clean_summary['iou_TC_mean']:.6f}")
    print(f"ET Dice mean: {clean_summary['dice_ET_mean']:.6f}")
    print(f"ET IoU mean:  {clean_summary['iou_ET_mean']:.6f}")


if __name__ == "__main__":
    main()
