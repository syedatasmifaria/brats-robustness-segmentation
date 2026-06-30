#!/usr/bin/env python3
"""
Script 25C: Evaluate nnU-Net degraded smoke-test predictions.

Purpose:
- Compare nnU-Net predictions on degraded ghosting L5 inputs against ground-truth labels.
- Compute class-wise metrics and BraTS-style region metrics:
  WT = labels 1 + 2 + 3
  TC = labels 1 + 3
  ET = label 3

Important:
- This evaluates a clean-trained nnU-Net model on degraded test images.
- No training happens here.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import nibabel as nib


PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

TEST_ROOT = PROJECT_ROOT / "nnunet/temporary_degraded_tests/ghosting_L5"
PRED_DIR = TEST_ROOT / "predictions"
LABEL_DIR = TEST_ROOT / "labelsTs"

RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_DIR = PROJECT_ROOT / "report_materials"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_METRICS_CSV = RESULTS_DIR / "25c_nnunet_degraded_smoke_test_metrics.csv"
OUT_SUMMARY_CSV = REPORT_DIR / "25c_nnunet_degraded_smoke_test_summary.csv"
OUT_SUMMARY_TXT = REPORT_DIR / "25c_nnunet_degraded_smoke_test_summary.txt"


def load_label(path):
    """Load a segmentation/prediction NIfTI as integer labels."""
    img = nib.load(str(path))
    data = img.get_fdata().astype(np.uint8)
    return data


def dice_score(pred_mask, true_mask, eps=1e-8):
    """Dice = 2 * overlap / total predicted and true voxels."""
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)

    pred_sum = pred_mask.sum()
    true_sum = true_mask.sum()

    if pred_sum == 0 and true_sum == 0:
        return 1.0

    intersection = np.logical_and(pred_mask, true_mask).sum()
    return float((2.0 * intersection + eps) / (pred_sum + true_sum + eps))


def iou_score(pred_mask, true_mask, eps=1e-8):
    """IoU = overlap / union."""
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)

    union = np.logical_or(pred_mask, true_mask).sum()

    if union == 0:
        return 1.0

    intersection = np.logical_and(pred_mask, true_mask).sum()
    return float((intersection + eps) / (union + eps))


def region_masks(label_volume):
    """
    Create BraTS-style region masks from remapped labels.

    Remapped labels:
    0 = background
    1 = original BraTS label 1
    2 = original BraTS label 2
    3 = original BraTS label 4

    Regions:
    WT = whole tumor = labels 1, 2, 3
    TC = tumor core = labels 1, 3
    ET = enhancing tumor = label 3
    """
    return {
        "WT": np.isin(label_volume, [1, 2, 3]),
        "TC": np.isin(label_volume, [1, 3]),
        "ET": label_volume == 3,
    }


def main():
    print("=" * 80)
    print("Script 25C: Evaluate nnU-Net degraded smoke-test predictions")
    print("=" * 80)

    prediction_files = sorted([
        p for p in PRED_DIR.glob("*.nii.gz")
        if p.name.startswith("BraTS20_Training_")
    ])

    print(f"Prediction directory: {PRED_DIR}")
    print(f"Label directory: {LABEL_DIR}")
    print(f"Prediction files found: {len(prediction_files)}")
    print()

    if len(prediction_files) == 0:
        raise FileNotFoundError("No prediction .nii.gz files found.")

    rows = []

    for pred_path in prediction_files:
        patient_id = pred_path.name.replace(".nii.gz", "")
        label_path = LABEL_DIR / pred_path.name

        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {patient_id}: {label_path}")

        print(f"Evaluating patient: {patient_id}")

        pred = load_label(pred_path)
        true = load_label(label_path)

        if pred.shape != true.shape:
            raise ValueError(
                f"Shape mismatch for {patient_id}: pred {pred.shape}, true {true.shape}"
            )

        row = {
            "patient_id": patient_id,
            "artifact": "ghosting",
            "level": 5,
            "condition": "ghosting_L5",
            "pred_shape": str(pred.shape),
            "true_tumor_voxels_WT": int(np.isin(true, [1, 2, 3]).sum()),
            "pred_tumor_voxels_WT": int(np.isin(pred, [1, 2, 3]).sum()),
        }

        # Class-wise metrics
        for cls in [1, 2, 3]:
            pred_mask = pred == cls
            true_mask = true == cls

            row[f"dice_class_{cls}"] = dice_score(pred_mask, true_mask)
            row[f"iou_class_{cls}"] = iou_score(pred_mask, true_mask)
            row[f"true_voxels_class_{cls}"] = int(true_mask.sum())
            row[f"pred_voxels_class_{cls}"] = int(pred_mask.sum())

        # Region-wise metrics
        pred_regions = region_masks(pred)
        true_regions = region_masks(true)

        for region_name in ["WT", "TC", "ET"]:
            row[f"dice_{region_name}"] = dice_score(
                pred_regions[region_name], true_regions[region_name]
            )
            row[f"iou_{region_name}"] = iou_score(
                pred_regions[region_name], true_regions[region_name]
            )
            row[f"true_voxels_{region_name}"] = int(true_regions[region_name].sum())
            row[f"pred_voxels_{region_name}"] = int(pred_regions[region_name].sum())

        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUT_METRICS_CSV, index=False)

    # Summary means/stds for numeric columns
    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
    summary_rows = []

    for col in numeric_cols:
        if col == "level":
            continue
        summary_rows.append({
            "metric": col,
            "mean": metrics_df[col].mean(),
            "std": metrics_df[col].std(ddof=0),
            "min": metrics_df[col].min(),
            "max": metrics_df[col].max(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)

    key_metrics = {
        "condition": "ghosting_L5",
        "num_patients": int(len(metrics_df)),
        "mean_dice_WT": float(metrics_df["dice_WT"].mean()),
        "mean_iou_WT": float(metrics_df["iou_WT"].mean()),
        "mean_dice_TC": float(metrics_df["dice_TC"].mean()),
        "mean_iou_TC": float(metrics_df["iou_TC"].mean()),
        "mean_dice_ET": float(metrics_df["dice_ET"].mean()),
        "mean_iou_ET": float(metrics_df["iou_ET"].mean()),
        "mean_true_voxels_WT": float(metrics_df["true_voxels_WT"].mean()),
        "mean_pred_voxels_WT": float(metrics_df["pred_voxels_WT"].mean()),
    }

    with open(OUT_SUMMARY_TXT, "w") as f:
        f.write("nnU-Net degraded smoke-test summary\n")
        f.write("=" * 80 + "\n\n")
        f.write("Condition: ghosting level 5\n")
        f.write("Model: clean-trained nnU-Net v2 3d_fullres fold 0\n")
        f.write("Evaluation: degraded images, no retraining\n\n")

        for key, value in key_metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.6f}\n")
            else:
                f.write(f"{key}: {value}\n")

        f.write("\nBraTS-style region definitions:\n")
        f.write("WT = labels 1 + 2 + 3\n")
        f.write("TC = labels 1 + 3\n")
        f.write("ET = label 3\n")

    print()
    print("=" * 80)
    print("Evaluation complete.")
    print(f"Saved metrics CSV: {OUT_METRICS_CSV}")
    print(f"Saved summary CSV: {OUT_SUMMARY_CSV}")
    print(f"Saved summary TXT: {OUT_SUMMARY_TXT}")
    print("=" * 80)
    print()
    print("Key smoke-test results:")
    print(json.dumps(key_metrics, indent=2))


if __name__ == "__main__":
    main()
