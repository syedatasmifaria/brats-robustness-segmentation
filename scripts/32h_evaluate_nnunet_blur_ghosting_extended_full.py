#!/usr/bin/env python3
"""
Script 32H: Evaluate nnU-Net blur and ghosting extended full predictions.

Purpose:
- Evaluate nnU-Net predictions for extended degradation levels L6-L10.
- Selected artifacts: blur and ghosting.
- Use 5 pilot patients.
- Compute WT, TC, ET Dice and IoU.
- Compute class-wise Dice and IoU.
- Compute drops from clean baseline using the same 5 patients.

Important:
Drop = clean same-pilot metric - degraded same-pilot metric

Positive drop means performance worsened.
Negative drop means degraded performance was slightly higher than clean.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt


PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

EXTENDED_ROOT = PROJECT_ROOT / "nnunet/temporary_degraded_tests/blur_ghosting_extended_full"
LABELS_DIR = EXTENDED_ROOT / "labelsTs"

CLEAN_METRICS_CSV = PROJECT_ROOT / "results/26e_nnunet_clean_same_test_metrics.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_DIR = PROJECT_ROOT / "report_materials"

OUT_METRICS_CSV = RESULTS_DIR / "32h_nnunet_blur_ghosting_extended_full_metrics.csv"
OUT_SUMMARY_CSV = REPORT_DIR / "32h_nnunet_blur_ghosting_extended_full_drop_summary.csv"
OUT_SUMMARY_TXT = REPORT_DIR / "32h_nnunet_blur_ghosting_extended_full_drop_summary.txt"

WT_DICE_CURVE = REPORT_DIR / "32h_nnunet_blur_ghosting_extended_full_wt_dice_curve.png"
WT_IOU_CURVE = REPORT_DIR / "32h_nnunet_blur_ghosting_extended_full_wt_iou_curve.png"
WT_DICE_DROP_CURVE = REPORT_DIR / "32h_nnunet_blur_ghosting_extended_full_wt_dice_drop_curve.png"
WT_IOU_DROP_CURVE = REPORT_DIR / "32h_nnunet_blur_ghosting_extended_full_wt_iou_drop_curve.png"

ARTIFACT_ORDER = ["blur", "ghosting"]
LEVEL_ORDER = [6, 7, 8, 9, 10]

METRIC_COLUMNS = [
    "dice_WT", "iou_WT",
    "dice_TC", "iou_TC",
    "dice_ET", "iou_ET",
    "dice_class_1", "iou_class_1",
    "dice_class_2", "iou_class_2",
    "dice_class_3", "iou_class_3",
]


def extract_case_number(path_or_name) -> str:
    """
    Extract 3-digit case number.

    Examples:
    BraTS20_Training_016.nii.gz -> 016
    BRATS_016.nii.gz -> 016
    """
    name = Path(path_or_name).name
    match = re.search(r"(\d{3})", name)
    if match is None:
        raise ValueError(f"Could not extract case number from: {name}")
    return match.group(1)


def parse_condition(condition: str):
    """
    Example:
    noise_L6 -> artifact=noise, level=6
    """
    artifact, level_text = condition.split("_L")
    return artifact, int(level_text)


def load_nii(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata().astype(np.int16)


def dice_score(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
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
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)

    union = np.logical_or(pred_mask, true_mask).sum()

    if union == 0:
        return 1.0

    intersection = np.logical_and(pred_mask, true_mask).sum()
    return float(intersection / union)


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """
    Region definitions:
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


def make_curve(summary_df, metric_col, clean_col, out_path, title, ylabel):
    plt.figure(figsize=(8, 5))

    for artifact in ARTIFACT_ORDER:
        sub = summary_df[summary_df["artifact"] == artifact].sort_values("level")
        plt.plot(sub["level"], sub[metric_col], marker="o", label=artifact)

    clean_value = summary_df[clean_col].iloc[0]
    plt.axhline(clean_value, linestyle="--", label="clean pilot baseline")

    plt.xlabel("Extended severity level")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(LEVEL_ORDER)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def make_drop_curve(summary_df, drop_col, out_path, title, ylabel):
    plt.figure(figsize=(8, 5))

    for artifact in ARTIFACT_ORDER:
        sub = summary_df[summary_df["artifact"] == artifact].sort_values("level")
        plt.plot(sub["level"], sub[drop_col], marker="o", label=artifact)

    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Extended severity level")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(LEVEL_ORDER)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    print("=" * 80)
    print("Script 32H: Evaluate nnU-Net blur and ghosting extended full evaluation")
    print("=" * 80)

    RESULTS_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    clean_df = pd.read_csv(CLEAN_METRICS_CSV)
    clean_df["case_id"] = clean_df["case_id"].astype(str).str.zfill(3)

    label_paths = sorted(LABELS_DIR.glob("*.nii.gz"))
    label_by_case = {extract_case_number(p): p for p in label_paths}
    pilot_case_ids = sorted(label_by_case.keys())

    print(f"Pilot labels found: {len(label_paths)}")
    print(f"Pilot case IDs: {pilot_case_ids}")

    if len(pilot_case_ids) != 74:
        raise RuntimeError(f"Expected 74 full-test labels, found {len(pilot_case_ids)}")

    clean_pilot_df = clean_df[clean_df["case_id"].isin(pilot_case_ids)].copy()

    if len(clean_pilot_df) != 74:
        raise RuntimeError(
            f"Expected 74 clean baseline rows for full-test patients, found {len(clean_pilot_df)}"
        )

    clean_means = {}
    for metric in METRIC_COLUMNS:
        clean_means[metric] = clean_pilot_df[metric].mean()

    clean_means["pred_WT_voxels"] = clean_pilot_df["pred_WT_voxels"].mean()
    clean_means["true_WT_voxels"] = clean_pilot_df["true_WT_voxels"].mean()

    rows = []

    for artifact in ARTIFACT_ORDER:
        for level in LEVEL_ORDER:
            condition = f"{artifact}_L{level}"
            pred_dir = EXTENDED_ROOT / condition / "predictions"

            pred_paths = sorted(pred_dir.glob("*.nii.gz"))
            pred_by_case = {extract_case_number(p): p for p in pred_paths}

            common_cases = sorted(set(pred_by_case.keys()) & set(label_by_case.keys()))

            print("-" * 80)
            print(f"Evaluating {condition}")
            print(f"Prediction files: {len(pred_paths)}")
            print(f"Matched cases:    {len(common_cases)}")

            if len(common_cases) != 74:
                raise RuntimeError(f"Expected 74 matched cases for {condition}, found {len(common_cases)}")

            for case_id in common_cases:
                pred_path = pred_by_case[case_id]
                label_path = label_by_case[case_id]

                pred = load_nii(pred_path)
                true = load_nii(label_path)

                if pred.shape != true.shape:
                    raise RuntimeError(
                        f"Shape mismatch for {condition} case {case_id}: "
                        f"pred {pred.shape}, true {true.shape}"
                    )

                metrics = compute_metrics(pred, true)

                rows.append({
                    "condition": condition,
                    "artifact": artifact,
                    "level": level,
                    "case_id": case_id,
                    "prediction_file": pred_path.name,
                    "label_file": label_path.name,
                    **metrics,
                })

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUT_METRICS_CSV, index=False)

    summary_rows = []

    for artifact in ARTIFACT_ORDER:
        for level in LEVEL_ORDER:
            condition = f"{artifact}_L{level}"
            sub = metrics_df[metrics_df["condition"] == condition]

            row = {
                "condition": condition,
                "artifact": artifact,
                "level": level,
                "num_test_patients": len(sub),
            }

            for metric in METRIC_COLUMNS:
                clean_mean = clean_means[metric]
                degraded_mean = sub[metric].mean()
                drop = clean_mean - degraded_mean

                row[f"clean_{metric}"] = clean_mean
                row[f"degraded_{metric}"] = degraded_mean
                row[f"drop_{metric}"] = drop

            row["clean_pred_WT_voxels"] = clean_means["pred_WT_voxels"]
            row["degraded_pred_WT_voxels"] = sub["pred_WT_voxels"].mean()
            row["clean_true_WT_voxels"] = clean_means["true_WT_voxels"]
            row["degraded_true_WT_voxels"] = sub["true_WT_voxels"].mean()

            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)

    with open(OUT_SUMMARY_TXT, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Script 32H: nnU-Net extended full evaluation drop summary\n")
        f.write("=" * 80 + "\n\n")

        f.write("Drop formula:\n")
        f.write("Drop = clean full-test metric - degraded pilot metric\n\n")

        f.write("Clean full-test baseline, same 74 patients:\n")
        f.write(f"WT Dice: {clean_means['dice_WT']:.6f}\n")
        f.write(f"WT IoU:  {clean_means['iou_WT']:.6f}\n")
        f.write(f"TC Dice: {clean_means['dice_TC']:.6f}\n")
        f.write(f"TC IoU:  {clean_means['iou_TC']:.6f}\n")
        f.write(f"ET Dice: {clean_means['dice_ET']:.6f}\n")
        f.write(f"ET IoU:  {clean_means['iou_ET']:.6f}\n\n")

        f.write("Extended condition summaries:\n")

        for _, row in summary_df.iterrows():
            f.write(
                f"{row['condition']}: "
                f"WT Dice {row['clean_dice_WT']:.6f} -> {row['degraded_dice_WT']:.6f}, "
                f"drop={row['drop_dice_WT']:.6f}; "
                f"WT IoU {row['clean_iou_WT']:.6f} -> {row['degraded_iou_WT']:.6f}, "
                f"drop={row['drop_iou_WT']:.6f}; "
                f"TC Dice drop={row['drop_dice_TC']:.6f}; "
                f"ET Dice drop={row['drop_dice_ET']:.6f}; "
                f"Pred WT voxels {row['clean_pred_WT_voxels']:.1f} -> {row['degraded_pred_WT_voxels']:.1f}\n"
            )

        f.write("\nPotential breaking threshold check:\n")
        f.write("Suggested break threshold: WT Dice drop >= 0.10 or WT IoU drop >= 0.15\n\n")

        broken = summary_df[
            (summary_df["drop_dice_WT"] >= 0.10) |
            (summary_df["drop_iou_WT"] >= 0.15)
        ].copy()

        if len(broken) == 0:
            f.write("No extended full evaluation condition reached the suggested WT breaking threshold.\n")
        else:
            for _, row in broken.iterrows():
                f.write(
                    f"{row['condition']}: "
                    f"WT Dice drop={row['drop_dice_WT']:.6f}, "
                    f"WT IoU drop={row['drop_iou_WT']:.6f}\n"
                )

        f.write("\nInterpretation note:\n")
        f.write(
            "This is a 74-patient full stress test. It should be used to decide "
            "whether L6-L10 are useful before scaling to all 74 test patients.\n"
        )

    make_curve(
        summary_df=summary_df,
        metric_col="degraded_dice_WT",
        clean_col="clean_dice_WT",
        out_path=WT_DICE_CURVE,
        title="nnU-Net extended full evaluation WT Dice",
        ylabel="WT Dice",
    )

    make_curve(
        summary_df=summary_df,
        metric_col="degraded_iou_WT",
        clean_col="clean_iou_WT",
        out_path=WT_IOU_CURVE,
        title="nnU-Net extended full evaluation WT IoU",
        ylabel="WT IoU",
    )

    make_drop_curve(
        summary_df=summary_df,
        drop_col="drop_dice_WT",
        out_path=WT_DICE_DROP_CURVE,
        title="nnU-Net extended full evaluation WT Dice drop",
        ylabel="WT Dice drop from clean pilot",
    )

    make_drop_curve(
        summary_df=summary_df,
        drop_col="drop_iou_WT",
        out_path=WT_IOU_DROP_CURVE,
        title="nnU-Net extended full evaluation WT IoU drop",
        ylabel="WT IoU drop from clean pilot",
    )

    print("=" * 80)
    print("Extended full evaluation evaluation complete.")
    print(f"Saved metrics CSV: {OUT_METRICS_CSV}")
    print(f"Saved summary CSV: {OUT_SUMMARY_CSV}")
    print(f"Saved summary TXT: {OUT_SUMMARY_TXT}")
    print(f"Saved plots:")
    print(f"  {WT_DICE_CURVE}")
    print(f"  {WT_IOU_CURVE}")
    print(f"  {WT_DICE_DROP_CURVE}")
    print(f"  {WT_IOU_DROP_CURVE}")
    print("=" * 80)
    print()
    print("Quick preview:")
    print(summary_df[[
        "condition",
        "clean_dice_WT",
        "degraded_dice_WT",
        "drop_dice_WT",
        "clean_iou_WT",
        "degraded_iou_WT",
        "drop_iou_WT",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
