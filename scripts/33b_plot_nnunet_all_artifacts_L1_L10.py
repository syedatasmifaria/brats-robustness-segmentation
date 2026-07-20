#!/usr/bin/env python3
"""
Script 33B: Plot final nnU-Net robustness curves across all artifacts.

Creates report-ready plots for:
- Absolute WT Dice
- WT Dice drop from clean
- Absolute WT IoU
- WT IoU drop from clean
- WT, TC, and ET Dice drop
- Predicted WT voxel ratio relative to true WT volume

Interpretation:
- Absolute Dice/IoU decrease as performance worsens.
- Dice/IoU drop increases as performance worsens.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")
REPORT_DIR = PROJECT_ROOT / "report_materials"

INPUT_CSV = (
    REPORT_DIR
    / "33a_nnunet_all_artifacts_L1_L10_summary.csv"
)

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
    "noise": "Gaussian noise",
    "contrast": "Contrast reduction",
    "ringing": "Frequency-domain truncation",
}

LEVELS = list(range(1, 11))

OUTPUTS = {
    "wt_dice": REPORT_DIR / "33b_nnunet_all_artifacts_wt_dice_L1_L10.png",
    "wt_dice_pdf": REPORT_DIR / "33b_nnunet_all_artifacts_wt_dice_L1_L10.pdf",

    "wt_dice_drop": REPORT_DIR / "33b_nnunet_all_artifacts_wt_dice_drop_L1_L10.png",
    "wt_dice_drop_pdf": REPORT_DIR / "33b_nnunet_all_artifacts_wt_dice_drop_L1_L10.pdf",

    "wt_iou": REPORT_DIR / "33b_nnunet_all_artifacts_wt_iou_L1_L10.png",
    "wt_iou_pdf": REPORT_DIR / "33b_nnunet_all_artifacts_wt_iou_L1_L10.pdf",

    "wt_iou_drop": REPORT_DIR / "33b_nnunet_all_artifacts_wt_iou_drop_L1_L10.png",
    "wt_iou_drop_pdf": REPORT_DIR / "33b_nnunet_all_artifacts_wt_iou_drop_L1_L10.pdf",

    "region_dice_drop": REPORT_DIR / "33b_nnunet_region_dice_drop_L1_L10.png",
    "region_dice_drop_pdf": REPORT_DIR / "33b_nnunet_region_dice_drop_L1_L10.pdf",

    "wt_voxel_ratio": REPORT_DIR / "33b_nnunet_predicted_true_wt_ratio_L1_L10.png",
    "wt_voxel_ratio_pdf": REPORT_DIR / "33b_nnunet_predicted_true_wt_ratio_L1_L10.pdf",
}


def save_figure(png_path, pdf_path):
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()


def plot_artifact_curves(
    df,
    metric,
    ylabel,
    title,
    png_path,
    pdf_path,
    clean_reference=None,
    threshold=None,
):
    plt.figure(figsize=(9, 6))

    for artifact in ARTIFACT_ORDER:
        sub = (
            df[df["artifact"] == artifact]
            .sort_values("level")
        )

        plt.plot(
            sub["level"],
            sub[metric],
            marker="o",
            linewidth=2,
            label=DISPLAY_NAMES[artifact],
        )

    if clean_reference is not None:
        plt.axhline(
            clean_reference,
            linestyle="--",
            linewidth=1.5,
            label="Clean baseline",
        )

    if threshold is not None:
        plt.axhline(
            threshold,
            linestyle=":",
            linewidth=1.5,
            label=f"Breaking threshold ({threshold:.2f})",
        )

    plt.xlabel("Degradation severity level")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(LEVELS)
    plt.grid(alpha=0.25)
    plt.legend()
    save_figure(png_path, pdf_path)


def main():
    print("=" * 80)
    print("Script 33B: Plot final nnU-Net L1-L10 robustness curves")
    print("=" * 80)

    df = pd.read_csv(INPUT_CSV)

    if len(df) != 50:
        raise RuntimeError(
            f"Expected 50 rows, found {len(df)}"
        )

    clean_dice_wt = float(df["clean_dice_WT"].iloc[0])
    clean_iou_wt = float(df["clean_iou_WT"].iloc[0])

    plot_artifact_curves(
        df=df,
        metric="degraded_dice_WT",
        ylabel="WT Dice",
        title="nnU-Net WT Dice across degradation severity",
        png_path=OUTPUTS["wt_dice"],
        pdf_path=OUTPUTS["wt_dice_pdf"],
        clean_reference=clean_dice_wt,
    )

    plot_artifact_curves(
        df=df,
        metric="drop_dice_WT",
        ylabel="WT Dice drop from clean",
        title="nnU-Net WT Dice loss across degradation severity",
        png_path=OUTPUTS["wt_dice_drop"],
        pdf_path=OUTPUTS["wt_dice_drop_pdf"],
        threshold=0.10,
    )

    plot_artifact_curves(
        df=df,
        metric="degraded_iou_WT",
        ylabel="WT IoU",
        title="nnU-Net WT IoU across degradation severity",
        png_path=OUTPUTS["wt_iou"],
        pdf_path=OUTPUTS["wt_iou_pdf"],
        clean_reference=clean_iou_wt,
    )

    plot_artifact_curves(
        df=df,
        metric="drop_iou_WT",
        ylabel="WT IoU drop from clean",
        title="nnU-Net WT IoU loss across degradation severity",
        png_path=OUTPUTS["wt_iou_drop"],
        pdf_path=OUTPUTS["wt_iou_drop_pdf"],
        threshold=0.15,
    )

    plt.figure(figsize=(10, 6))

    region_columns = {
        "WT": "drop_dice_WT",
        "TC": "drop_dice_TC",
        "ET": "drop_dice_ET",
    }

    line_styles = {
        "WT": "-",
        "TC": "--",
        "ET": ":",
    }

    for artifact in ARTIFACT_ORDER:
        sub = (
            df[df["artifact"] == artifact]
            .sort_values("level")
        )

        for region, metric in region_columns.items():
            plt.plot(
                sub["level"],
                sub[metric],
                linestyle=line_styles[region],
                linewidth=1.8,
                marker="o",
                markersize=4,
                label=f"{DISPLAY_NAMES[artifact]} — {region}",
            )

    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Degradation severity level")
    plt.ylabel("Dice drop from clean")
    plt.title("nnU-Net regional Dice loss: WT, TC, and ET")
    plt.xticks(LEVELS)
    plt.grid(alpha=0.25)
    plt.legend(
        fontsize=8,
        ncol=2,
    )
    save_figure(
        OUTPUTS["region_dice_drop"],
        OUTPUTS["region_dice_drop_pdf"],
    )

    df["predicted_true_wt_ratio"] = (
        df["degraded_pred_WT_voxels"]
        / df["degraded_true_WT_voxels"]
    )

    plt.figure(figsize=(9, 6))

    for artifact in ARTIFACT_ORDER:
        sub = (
            df[df["artifact"] == artifact]
            .sort_values("level")
        )

        plt.plot(
            sub["level"],
            sub["predicted_true_wt_ratio"],
            marker="o",
            linewidth=2,
            label=DISPLAY_NAMES[artifact],
        )

    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=1.5,
        label="Predicted = true WT volume",
    )

    plt.xlabel("Degradation severity level")
    plt.ylabel("Predicted WT voxels / true WT voxels")
    plt.title("nnU-Net predicted WT volume relative to ground truth")
    plt.xticks(LEVELS)
    plt.grid(alpha=0.25)
    plt.legend()
    save_figure(
        OUTPUTS["wt_voxel_ratio"],
        OUTPUTS["wt_voxel_ratio_pdf"],
    )

    print("Saved figures:")

    for path in OUTPUTS.values():
        print(f"  {path}")

    print("=" * 80)


if __name__ == "__main__":
    main()
