#!/usr/bin/env python3
"""
Script 23: Extract nnU-Net validation summary.

Purpose:
- Read nnU-Net v2 validation summary.json.
- Extract foreground mean Dice/IoU.
- Extract class-wise Dice/IoU.
- Save clean report-ready CSV and TXT files.

Important:
- This does NOT train anything.
- This does NOT modify nnU-Net outputs.
- This only reads summary.json and creates small report files.
"""

import json
from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

SUMMARY_JSON = (
    PROJECT_ROOT
    / "nnunet/nnUNet_results/Dataset501_BraTS2020Multimodal/"
    / "nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation/summary.json"
)

REPORT_DIR = PROJECT_ROOT / "report_materials"
REPORT_DIR.mkdir(exist_ok=True)

OUT_CSV = REPORT_DIR / "23_nnunet_validation_summary.csv"
OUT_TXT = REPORT_DIR / "23_nnunet_validation_summary.txt"


# ============================================================
# Class descriptions
# ============================================================

CLASS_NAMES = {
    "1": "Class 1 / BraTS original label 1",
    "2": "Class 2 / BraTS original label 2",
    "3": "Class 3 / BraTS original label 4 remapped to 3",
}


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("Script 23: Extract nnU-Net validation summary")
    print("=" * 80)

    if not SUMMARY_JSON.exists():
        raise FileNotFoundError(f"Could not find summary.json at:\n{SUMMARY_JSON}")

    print(f"Reading summary file:\n{SUMMARY_JSON}")

    with open(SUMMARY_JSON, "r") as f:
        data = json.load(f)

    foreground_mean = data.get("foreground_mean", {})
    mean_by_class = data.get("mean", {})
    metric_per_case = data.get("metric_per_case", [])

    num_validation_cases = len(metric_per_case)

    rows = []

    # --------------------------------------------------------
    # Foreground mean metrics
    # --------------------------------------------------------

    if foreground_mean:
        rows.append({
            "section": "foreground_mean",
            "class_id": "foreground",
            "class_description": "Mean across foreground classes/regions reported by nnU-Net",
            "dice": foreground_mean.get("Dice"),
            "iou": foreground_mean.get("IoU"),
            "tp": foreground_mean.get("TP"),
            "fp": foreground_mean.get("FP"),
            "fn": foreground_mean.get("FN"),
            "tn": foreground_mean.get("TN"),
            "n_pred": foreground_mean.get("n_pred"),
            "n_ref": foreground_mean.get("n_ref"),
        })

    # --------------------------------------------------------
    # Class-wise metrics
    # --------------------------------------------------------

    for class_id, metrics in mean_by_class.items():
        class_id = str(class_id)

        rows.append({
            "section": "class_mean",
            "class_id": class_id,
            "class_description": CLASS_NAMES.get(class_id, f"Class {class_id}"),
            "dice": metrics.get("Dice"),
            "iou": metrics.get("IoU"),
            "tp": metrics.get("TP"),
            "fp": metrics.get("FP"),
            "fn": metrics.get("FN"),
            "tn": metrics.get("TN"),
            "n_pred": metrics.get("n_pred"),
            "n_ref": metrics.get("n_ref"),
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUT_CSV, index=False)

    # --------------------------------------------------------
    # Write readable TXT summary
    # --------------------------------------------------------

    with open(OUT_TXT, "w") as f:
        f.write("Script 23: nnU-Net validation summary\n")
        f.write("=" * 80 + "\n\n")

        f.write("Model:\n")
        f.write("nnU-Net v2 standard 3d_fullres, fold 0\n\n")

        f.write("Dataset:\n")
        f.write("Dataset501_BraTS2020Multimodal\n")
        f.write("Input modalities: FLAIR, T1, T1ce, T2\n")
        f.write("Training data: clean BraTS2020 images only\n\n")

        f.write("Source summary.json:\n")
        f.write(str(SUMMARY_JSON) + "\n\n")

        f.write(f"Number of validation cases in summary.json: {num_validation_cases}\n\n")

        f.write("Main validation metrics:\n")
        f.write("-" * 80 + "\n")

        if foreground_mean:
            f.write(f"Foreground mean Dice: {foreground_mean.get('Dice'):.6f}\n")
            f.write(f"Foreground mean IoU:  {foreground_mean.get('IoU'):.6f}\n\n")
        else:
            f.write("Foreground mean metrics not found.\n\n")

        f.write("Class-wise validation metrics:\n")
        f.write("-" * 80 + "\n")

        if mean_by_class:
            for class_id, metrics in mean_by_class.items():
                class_id = str(class_id)
                dice = metrics.get("Dice")
                iou = metrics.get("IoU")
                class_desc = CLASS_NAMES.get(class_id, f"Class {class_id}")

                f.write(f"{class_desc}\n")
                f.write(f"  Dice: {dice:.6f}\n")
                f.write(f"  IoU:  {iou:.6f}\n\n")
        else:
            f.write("Class-wise metrics not found.\n\n")

        f.write("Interpretation note:\n")
        f.write(
            "This nnU-Net result comes from the nnU-Net v2 fold-0 validation "
            "pipeline. It should not be directly over-compared against the custom "
            "3D U-Net patch-based held-out test result because the evaluation "
            "setups differ. The custom 3D U-Net result is patch-based and "
            "tumor-centered, while nnU-Net validation follows nnU-Net's own "
            "validation pipeline. A fair direct comparison would require matched "
            "test data, matched preprocessing, and matched metrics.\n"
        )

    print()
    print("Saved report files:")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_TXT}")

    print()
    print("Key nnU-Net validation result:")

    if foreground_mean:
        print(f"  Foreground mean Dice: {foreground_mean.get('Dice'):.6f}")
        print(f"  Foreground mean IoU:  {foreground_mean.get('IoU'):.6f}")

    print()
    print("Class-wise Dice/IoU:")

    for class_id, metrics in mean_by_class.items():
        class_id = str(class_id)
        class_desc = CLASS_NAMES.get(class_id, f"Class {class_id}")
        print(
            f"  {class_desc}: "
            f"Dice={metrics.get('Dice'):.6f}, "
            f"IoU={metrics.get('IoU'):.6f}"
        )

    print()
    print("=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()
