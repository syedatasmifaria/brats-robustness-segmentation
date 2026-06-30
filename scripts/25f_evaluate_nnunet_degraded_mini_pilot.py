#!/usr/bin/env python3
"""
Script 25F: Evaluate nnU-Net degraded mini-pilot predictions.

Purpose:
- Evaluate clean-trained nnU-Net on degraded mini-pilot predictions.
- Mini pilot = 2 patients x 5 artifacts x 5 severity levels.
- Computes class-wise metrics and BraTS-style region metrics:
  WT = labels 1 + 2 + 3
  TC = labels 1 + 3
  ET = label 3

Important:
- No training happens here.
- This evaluates degraded test images only.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import nibabel as nib


PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

MINI_ROOT = PROJECT_ROOT / "nnunet/temporary_degraded_tests/mini_pilot"

RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_DIR = PROJECT_ROOT / "report_materials"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_METRICS_CSV = RESULTS_DIR / "25f_nnunet_degraded_mini_pilot_metrics.csv"
OUT_SUMMARY_CSV = REPORT_DIR / "25f_nnunet_degraded_mini_pilot_summary.csv"
OUT_SUMMARY_TXT = REPORT_DIR / "25f_nnunet_degraded_mini_pilot_summary.txt"


def load_label(path):
    img = nib.load(str(path))
    return img.get_fdata().astype(np.uint8)


def dice_score(pred_mask, true_mask, eps=1e-8):
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)

    pred_sum = pred_mask.sum()
    true_sum = true_mask.sum()

    if pred_sum == 0 and true_sum == 0:
        return 1.0

    intersection = np.logical_and(pred_mask, true_mask).sum()
    return float((2.0 * intersection + eps) / (pred_sum + true_sum + eps))


def iou_score(pred_mask, true_mask, eps=1e-8):
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)

    union = np.logical_or(pred_mask, true_mask).sum()

    if union == 0:
        return 1.0

    intersection = np.logical_and(pred_mask, true_mask).sum()
    return float((intersection + eps) / (union + eps))


def make_region_masks(label_volume):
    """
    Remapped labels:
    0 = background
    1 = original BraTS label 1
    2 = original BraTS label 2
    3 = original BraTS label 4

    BraTS-style regions:
    WT = whole tumor = 1 + 2 + 3
    TC = tumor core = 1 + 3
    ET = enhancing tumor = 3
    """
    return {
        "WT": np.isin(label_volume, [1, 2, 3]),
        "TC": np.isin(label_volume, [1, 3]),
        "ET": label_volume == 3,
    }


def parse_condition_name(condition_name):
    """
    Example:
    blur_L1 -> artifact=blur, level=1
    ghosting_L5 -> artifact=ghosting, level=5
    """
    match = re.match(r"(.+)_L([1-5])$", condition_name)
    if match is None:
        raise ValueError(f"Could not parse condition name: {condition_name}")

    artifact = match.group(1)
    level = int(match.group(2))
    return artifact, level


def evaluate_one_prediction(pred_path, label_path, artifact, level, condition):
    patient_id = pred_path.name.replace(".nii.gz", "")

    pred = load_label(pred_path)
    true = load_label(label_path)

    if pred.shape != true.shape:
        raise ValueError(
            f"Shape mismatch for {patient_id}: pred {pred.shape}, true {true.shape}"
        )

    row = {
        "patient_id": patient_id,
        "artifact": artifact,
        "level": level,
        "condition": condition,
        "pred_shape": str(pred.shape),
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
    pred_regions = make_region_masks(pred)
    true_regions = make_region_masks(true)

    for region in ["WT", "TC", "ET"]:
        row[f"dice_{region}"] = dice_score(pred_regions[region], true_regions[region])
        row[f"iou_{region}"] = iou_score(pred_regions[region], true_regions[region])
        row[f"true_voxels_{region}"] = int(true_regions[region].sum())
        row[f"pred_voxels_{region}"] = int(pred_regions[region].sum())

    return row


def main():
    print("=" * 80)
    print("Script 25F: Evaluate nnU-Net degraded mini-pilot predictions")
    print("=" * 80)
    print(f"Mini-pilot root: {MINI_ROOT}")
    print()

    condition_dirs = sorted([
        p for p in MINI_ROOT.glob("*_L*")
        if p.is_dir()
    ])

    print(f"Condition folders found: {len(condition_dirs)}")

    all_rows = []

    for condition_dir in condition_dirs:
        condition = condition_dir.name
        artifact, level = parse_condition_name(condition)

        pred_dir = condition_dir / "predictions"
        label_dir = condition_dir / "labelsTs"

        pred_files = sorted(pred_dir.glob("BraTS20_Training_*.nii.gz"))

        print(f"Evaluating {condition}: {len(pred_files)} predictions")

        if len(pred_files) == 0:
            raise FileNotFoundError(f"No predictions found for condition: {condition}")

        for pred_path in pred_files:
            label_path = label_dir / pred_path.name

            if not label_path.exists():
                raise FileNotFoundError(f"Missing label: {label_path}")

            row = evaluate_one_prediction(
                pred_path=pred_path,
                label_path=label_path,
                artifact=artifact,
                level=level,
                condition=condition,
            )
            all_rows.append(row)

    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(OUT_METRICS_CSV, index=False)

    # Summary by artifact and level
    summary_cols = [
        "dice_WT", "iou_WT",
        "dice_TC", "iou_TC",
        "dice_ET", "iou_ET",
        "dice_class_1", "dice_class_2", "dice_class_3",
        "pred_voxels_WT", "true_voxels_WT",
    ]

    summary_df = (
        metrics_df
        .groupby(["artifact", "level", "condition"], as_index=False)[summary_cols]
        .agg(["mean", "std"])
    )

    # Flatten multi-index column names
    summary_df.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary_df.columns
    ]

    summary_df = summary_df.sort_values(["artifact", "level"])
    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)

    # Text summary with key numbers
    with open(OUT_SUMMARY_TXT, "w") as f:
        f.write("nnU-Net degraded mini-pilot summary\n")
        f.write("=" * 80 + "\n\n")
        f.write("Model: clean-trained nnU-Net v2 3d_fullres fold 0\n")
        f.write("Evaluation: degraded images only, no retraining\n")
        f.write("Patients: 2\n")
        f.write("Artifacts: blur, noise, contrast, ringing, ghosting\n")
        f.write("Levels: 1-5\n\n")

        f.write("BraTS-style region definitions:\n")
        f.write("WT = labels 1 + 2 + 3\n")
        f.write("TC = labels 1 + 3\n")
        f.write("ET = label 3\n\n")

        f.write("Mean WT Dice by condition:\n")
        f.write("-" * 80 + "\n")

        wt_table = summary_df[[
            "artifact",
            "level",
            "condition",
            "dice_WT_mean",
            "iou_WT_mean",
            "dice_TC_mean",
            "dice_ET_mean",
        ]].copy()

        for _, row in wt_table.iterrows():
            f.write(
                f"{row['condition']}: "
                f"WT Dice={row['dice_WT_mean']:.6f}, "
                f"WT IoU={row['iou_WT_mean']:.6f}, "
                f"TC Dice={row['dice_TC_mean']:.6f}, "
                f"ET Dice={row['dice_ET_mean']:.6f}\n"
            )

    print()
    print("=" * 80)
    print("Mini-pilot evaluation complete.")
    print(f"Rows evaluated: {len(metrics_df)}")
    print(f"Saved full metrics CSV: {OUT_METRICS_CSV}")
    print(f"Saved summary CSV: {OUT_SUMMARY_CSV}")
    print(f"Saved summary TXT: {OUT_SUMMARY_TXT}")
    print("=" * 80)

    print()
    print("Quick preview:")
    preview = summary_df[[
        "condition",
        "dice_WT_mean",
        "iou_WT_mean",
        "dice_TC_mean",
        "dice_ET_mean",
    ]].head(10)
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
