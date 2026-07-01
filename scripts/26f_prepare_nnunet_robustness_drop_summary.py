#!/usr/bin/env python3
"""
Step 26F: Prepare nnU-Net robustness drop summary.

Purpose:
- Combine clean same-test nnU-Net results with final degraded nnU-Net results.
- Compute Dice and IoU drops from clean for WT, TC, ET, and classes 1, 2, 3.
- Save report-ready CSV, TXT, and plots.

Important:
Drop = clean metric - degraded metric

Positive drop means performance worsened under degradation.
Negative drop means degraded performance was slightly higher than clean.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

CLEAN_METRICS_CSV = PROJECT_ROOT / "results/26e_nnunet_clean_same_test_metrics.csv"
DEGRADED_METRICS_CSV = PROJECT_ROOT / "results/26d_nnunet_degraded_final_full_metrics.csv"

REPORT_DIR = PROJECT_ROOT / "report_materials"

DROP_SUMMARY_CSV = REPORT_DIR / "26f_nnunet_robustness_drop_summary.csv"
DROP_SUMMARY_TXT = REPORT_DIR / "26f_nnunet_robustness_drop_summary.txt"

WT_DICE_CURVE = REPORT_DIR / "26f_nnunet_wt_dice_curve.png"
WT_IOU_CURVE = REPORT_DIR / "26f_nnunet_wt_iou_curve.png"
WT_DICE_DROP_BAR = REPORT_DIR / "26f_nnunet_level5_wt_dice_drop_bar.png"
WT_IOU_DROP_BAR = REPORT_DIR / "26f_nnunet_level5_wt_iou_drop_bar.png"

TC_DICE_CURVE = REPORT_DIR / "26f_nnunet_tc_dice_curve.png"
ET_DICE_CURVE = REPORT_DIR / "26f_nnunet_et_dice_curve.png"
TC_DICE_DROP_BAR = REPORT_DIR / "26f_nnunet_level5_tc_dice_drop_bar.png"
ET_DICE_DROP_BAR = REPORT_DIR / "26f_nnunet_level5_et_dice_drop_bar.png"


ARTIFACT_ORDER = ["blur", "contrast", "ghosting", "noise", "ringing"]
LEVEL_ORDER = [1, 2, 3, 4, 5]


METRIC_COLUMNS = [
    "dice_WT", "iou_WT",
    "dice_TC", "iou_TC",
    "dice_ET", "iou_ET",
    "dice_class_1", "iou_class_1",
    "dice_class_2", "iou_class_2",
    "dice_class_3", "iou_class_3",
]


def parse_condition(condition: str):
    """
    Convert condition string like blur_L5 into artifact='blur', level=5.
    """
    artifact, level_text = condition.split("_L")
    level = int(level_text)
    return artifact, level


def get_col(df: pd.DataFrame, candidates):
    """
    Return the first matching column name from a list of possible names.
    This protects us from small naming differences like pred_WT_voxels vs pred_wt_voxels.
    """
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"None of these columns were found: {candidates}. Available columns: {df.columns.tolist()}")


def clean_mean_dict(clean_df: pd.DataFrame) -> dict:
    """
    Compute clean mean for each metric across the same 74 test patients.
    """
    means = {}
    for col in METRIC_COLUMNS:
        means[col] = clean_df[col].mean()

    pred_wt_col = get_col(clean_df, ["pred_WT_voxels", "pred_wt_voxels", "pred_tumor_voxels", "pred_voxels_WT"])
    true_wt_col = get_col(clean_df, ["true_WT_voxels", "true_wt_voxels", "true_tumor_voxels", "true_voxels_WT"])

    means["pred_WT_voxels"] = clean_df[pred_wt_col].mean()
    means["true_WT_voxels"] = clean_df[true_wt_col].mean()
    return means


def make_curve(summary_df: pd.DataFrame, metric_col: str, clean_col: str, out_path: Path, title: str, ylabel: str):
    """
    Plot degraded metric across severity levels.
    Adds a horizontal clean baseline line.
    """
    plt.figure(figsize=(8, 5))

    for artifact in ARTIFACT_ORDER:
        sub = summary_df[summary_df["artifact"] == artifact].sort_values("level")
        plt.plot(sub["level"], sub[metric_col], marker="o", label=artifact)

    clean_value = summary_df[clean_col].iloc[0]
    plt.axhline(clean_value, linestyle="--", label="clean baseline")

    plt.xlabel("Severity level")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(LEVEL_ORDER)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def make_level5_bar(summary_df: pd.DataFrame, drop_col: str, out_path: Path, title: str, ylabel: str):
    """
    Bar plot of level-5 drops by artifact.
    """
    level5 = summary_df[summary_df["level"] == 5].copy()
    level5["artifact"] = pd.Categorical(level5["artifact"], categories=ARTIFACT_ORDER, ordered=True)
    level5 = level5.sort_values("artifact")

    plt.figure(figsize=(8, 5))
    plt.bar(level5["artifact"], level5[drop_col])
    plt.xlabel("Artifact")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    print("=" * 80)
    print("Step 26F: Prepare nnU-Net robustness drop summary")
    print("=" * 80)

    REPORT_DIR.mkdir(exist_ok=True)

    print(f"Reading clean metrics:    {CLEAN_METRICS_CSV}")
    print(f"Reading degraded metrics: {DEGRADED_METRICS_CSV}")

    clean_df = pd.read_csv(CLEAN_METRICS_CSV)
    degraded_df = pd.read_csv(DEGRADED_METRICS_CSV)

    print(f"Clean rows:    {len(clean_df)}")
    print(f"Degraded rows: {len(degraded_df)}")

    if len(clean_df) != 74:
        raise RuntimeError(f"Expected 74 clean rows, found {len(clean_df)}")

    if len(degraded_df) != 1850:
        raise RuntimeError(f"Expected 1850 degraded rows, found {len(degraded_df)}")

    clean_means = clean_mean_dict(clean_df)

    degraded_df[["artifact", "level"]] = degraded_df["condition"].apply(
        lambda x: pd.Series(parse_condition(x))
    )

    summary_rows = []

    grouped = degraded_df.groupby(["artifact", "level"], as_index=False)

    for _, group_info in grouped:
        pass

    for artifact in ARTIFACT_ORDER:
        for level in LEVEL_ORDER:
            condition = f"{artifact}_L{level}"
            sub = degraded_df[degraded_df["condition"] == condition]

            if len(sub) != 74:
                raise RuntimeError(f"Expected 74 rows for {condition}, found {len(sub)}")

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

            degraded_pred_wt_col = get_col(sub, ["pred_WT_voxels", "pred_wt_voxels", "pred_tumor_voxels", "pred_voxels_WT"])
            degraded_true_wt_col = get_col(sub, ["true_WT_voxels", "true_wt_voxels", "true_tumor_voxels", "true_voxels_WT"])

            row["clean_pred_WT_voxels"] = clean_means["pred_WT_voxels"]
            row["degraded_pred_WT_voxels"] = sub[degraded_pred_wt_col].mean()
            row["clean_true_WT_voxels"] = clean_means["true_WT_voxels"]
            row["degraded_true_WT_voxels"] = sub[degraded_true_wt_col].mean()

            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(DROP_SUMMARY_CSV, index=False)

    with open(DROP_SUMMARY_TXT, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Step 26F: nnU-Net robustness drop summary\n")
        f.write("=" * 80 + "\n\n")

        f.write("Drop formula:\n")
        f.write("Drop = clean metric - degraded metric\n\n")

        f.write("Clean same-test baseline:\n")
        f.write(f"WT Dice: {clean_means['dice_WT']:.6f}\n")
        f.write(f"WT IoU:  {clean_means['iou_WT']:.6f}\n")
        f.write(f"TC Dice: {clean_means['dice_TC']:.6f}\n")
        f.write(f"TC IoU:  {clean_means['iou_TC']:.6f}\n")
        f.write(f"ET Dice: {clean_means['dice_ET']:.6f}\n")
        f.write(f"ET IoU:  {clean_means['iou_ET']:.6f}\n\n")

        f.write("Condition summaries:\n")

        for _, row in summary_df.iterrows():
            f.write(
                f"{row['condition']}: "
                f"WT Dice {row['clean_dice_WT']:.6f} -> {row['degraded_dice_WT']:.6f}, "
                f"drop={row['drop_dice_WT']:.6f}; "
                f"WT IoU {row['clean_iou_WT']:.6f} -> {row['degraded_iou_WT']:.6f}, "
                f"drop={row['drop_iou_WT']:.6f}; "
                f"TC Dice drop={row['drop_dice_TC']:.6f}; "
                f"TC IoU drop={row['drop_iou_TC']:.6f}; "
                f"ET Dice drop={row['drop_dice_ET']:.6f}; "
                f"ET IoU drop={row['drop_iou_ET']:.6f}; "
                f"Pred WT voxels {row['clean_pred_WT_voxels']:.1f} -> {row['degraded_pred_WT_voxels']:.1f}\n"
            )

        f.write("\nLevel-5 ranking by WT Dice drop:\n")
        level5 = summary_df[summary_df["level"] == 5].copy()
        level5 = level5.sort_values("drop_dice_WT", ascending=False)

        for _, row in level5.iterrows():
            f.write(
                f"{row['artifact']}: "
                f"WT Dice drop={row['drop_dice_WT']:.6f}, "
                f"WT IoU drop={row['drop_iou_WT']:.6f}\n"
            )

        f.write("\nInterpretation note:\n")
        f.write(
            "These results compare degraded nnU-Net performance against the clean "
            "same-patient full-volume test baseline. This is the appropriate basis "
            "for robustness claims because the test patients are held constant.\n"
        )

    make_curve(
        summary_df=summary_df,
        metric_col="degraded_dice_WT",
        clean_col="clean_dice_WT",
        out_path=WT_DICE_CURVE,
        title="nnU-Net WT Dice under degradation",
        ylabel="WT Dice",
    )

    make_curve(
        summary_df=summary_df,
        metric_col="degraded_iou_WT",
        clean_col="clean_iou_WT",
        out_path=WT_IOU_CURVE,
        title="nnU-Net WT IoU under degradation",
        ylabel="WT IoU",
    )

    make_level5_bar(
        summary_df=summary_df,
        drop_col="drop_dice_WT",
        out_path=WT_DICE_DROP_BAR,
        title="nnU-Net level-5 WT Dice drop",
        ylabel="WT Dice drop from clean",
    )

    make_level5_bar(
        summary_df=summary_df,
        drop_col="drop_iou_WT",
        out_path=WT_IOU_DROP_BAR,
        title="nnU-Net level-5 WT IoU drop",
        ylabel="WT IoU drop from clean",
    )

    make_curve(
        summary_df=summary_df,
        metric_col="degraded_dice_TC",
        clean_col="clean_dice_TC",
        out_path=TC_DICE_CURVE,
        title="nnU-Net TC Dice under degradation",
        ylabel="TC Dice",
    )

    make_curve(
        summary_df=summary_df,
        metric_col="degraded_dice_ET",
        clean_col="clean_dice_ET",
        out_path=ET_DICE_CURVE,
        title="nnU-Net ET Dice under degradation",
        ylabel="ET Dice",
    )

    make_level5_bar(
        summary_df=summary_df,
        drop_col="drop_dice_TC",
        out_path=TC_DICE_DROP_BAR,
        title="nnU-Net level-5 TC Dice drop",
        ylabel="TC Dice drop from clean",
    )

    make_level5_bar(
        summary_df=summary_df,
        drop_col="drop_dice_ET",
        out_path=ET_DICE_DROP_BAR,
        title="nnU-Net level-5 ET Dice drop",
        ylabel="ET Dice drop from clean",
    )

    print("=" * 80)
    print("nnU-Net robustness drop summary complete.")
    print(f"Saved CSV: {DROP_SUMMARY_CSV}")
    print(f"Saved TXT: {DROP_SUMMARY_TXT}")
    print("Saved plots:")
    print(f"  {WT_DICE_CURVE}")
    print(f"  {WT_IOU_CURVE}")
    print(f"  {WT_DICE_DROP_BAR}")
    print(f"  {WT_IOU_DROP_BAR}")
    print(f"  {TC_DICE_CURVE}")
    print(f"  {ET_DICE_CURVE}")
    print(f"  {TC_DICE_DROP_BAR}")
    print(f"  {ET_DICE_DROP_BAR}")
    print("=" * 80)

    print()
    print("Quick level-5 WT drop ranking:")
    level5 = summary_df[summary_df["level"] == 5].copy()
    level5 = level5.sort_values("drop_dice_WT", ascending=False)
    print(level5[[
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
