#!/usr/bin/env python3
"""
Script 33E: Create cleaner report-ready nnU-Net robustness figures.

Creates:
- Separate WT, TC, and ET Dice-drop curves
- Simplified MSE versus WT Dice-drop figure
- Simplified PSNR versus WT Dice-drop figure

Scatter labels are limited to:
- the first WT breaking-threshold level for each artifact
- L10

The PSNR x-axis is reversed so worsening image quality progresses visually
from left to right.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")
REPORT_DIR = PROJECT_ROOT / "report_materials"

SEGMENTATION_CSV = (
    REPORT_DIR
    / "33a_nnunet_all_artifacts_L1_L10_summary.csv"
)

QUALITY_CSV = (
    REPORT_DIR
    / "33c_nnunet_all_artifacts_L1_L10_psnr_mse_summary.csv"
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

REGION_OUTPUTS = {
    "WT": (
        REPORT_DIR / "33e_nnunet_wt_dice_drop_L1_L10.png",
        REPORT_DIR / "33e_nnunet_wt_dice_drop_L1_L10.pdf",
    ),
    "TC": (
        REPORT_DIR / "33e_nnunet_tc_dice_drop_L1_L10.png",
        REPORT_DIR / "33e_nnunet_tc_dice_drop_L1_L10.pdf",
    ),
    "ET": (
        REPORT_DIR / "33e_nnunet_et_dice_drop_L1_L10.png",
        REPORT_DIR / "33e_nnunet_et_dice_drop_L1_L10.pdf",
    ),
}

MSE_SCATTER_PNG = (
    REPORT_DIR
    / "33e_nnunet_mse_vs_wt_dice_drop_report_ready.png"
)
MSE_SCATTER_PDF = (
    REPORT_DIR
    / "33e_nnunet_mse_vs_wt_dice_drop_report_ready.pdf"
)

PSNR_SCATTER_PNG = (
    REPORT_DIR
    / "33e_nnunet_psnr_vs_wt_dice_drop_report_ready.png"
)
PSNR_SCATTER_PDF = (
    REPORT_DIR
    / "33e_nnunet_psnr_vs_wt_dice_drop_report_ready.pdf"
)


def save_figure(png_path, pdf_path):
    plt.tight_layout()
    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        pdf_path,
        bbox_inches="tight",
    )
    plt.close()


def plot_region_curve(
    dataframe,
    region,
    metric_column,
    png_path,
    pdf_path,
):
    plt.figure(figsize=(9, 6))

    for artifact in ARTIFACT_ORDER:
        subset = (
            dataframe[dataframe["artifact"] == artifact]
            .sort_values("level")
        )

        plt.plot(
            subset["level"],
            subset[metric_column],
            marker="o",
            linewidth=2,
            label=DISPLAY_NAMES[artifact],
        )

    plt.axhline(
        0.0,
        linewidth=1,
    )

    if region == "WT":
        plt.axhline(
            0.10,
            linestyle=":",
            linewidth=1.5,
            label="WT Dice-drop threshold (0.10)",
        )

    plt.xlabel("Degradation severity level")
    plt.ylabel(f"{region} Dice drop from clean")
    plt.title(
        f"nnU-Net {region} Dice drop across degradation severity"
    )
    plt.xticks(LEVELS)
    plt.grid(alpha=0.25)
    plt.legend()

    save_figure(
        png_path,
        pdf_path,
    )


def levels_to_label(artifact_dataframe):
    levels = {10}

    broken = artifact_dataframe[
        artifact_dataframe["drop_dice_WT"] >= 0.10
    ]

    if not broken.empty:
        levels.add(int(broken["level"].min()))

    return levels


def plot_report_scatter(
    dataframe,
    x_column,
    x_label,
    title,
    png_path,
    pdf_path,
    reverse_x=False,
):
    plt.figure(figsize=(9, 6))

    for artifact in ARTIFACT_ORDER:
        subset = (
            dataframe[dataframe["artifact"] == artifact]
            .sort_values("level")
        )

        plt.plot(
            subset[x_column],
            subset["drop_dice_WT"],
            marker="o",
            linewidth=1.8,
            label=DISPLAY_NAMES[artifact],
        )

        label_levels = levels_to_label(subset)

        for _, row in subset.iterrows():
            level = int(row["level"])

            if level not in label_levels:
                continue

            plt.annotate(
                f"L{level}",
                (
                    row[x_column],
                    row["drop_dice_WT"],
                ),
                fontsize=8,
                xytext=(5, 5),
                textcoords="offset points",
            )

    plt.axhline(
        0.10,
        linestyle=":",
        linewidth=1.5,
        label="WT Dice-drop threshold (0.10)",
    )

    plt.xlabel(x_label)
    plt.ylabel("WT Dice drop from clean")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()

    if reverse_x:
        plt.gca().invert_xaxis()

    save_figure(
        png_path,
        pdf_path,
    )


def main():
    print("=" * 80)
    print("Script 33E: Create report-ready nnU-Net curves")
    print("=" * 80)

    segmentation_df = pd.read_csv(SEGMENTATION_CSV)
    quality_df = pd.read_csv(QUALITY_CSV)

    if len(segmentation_df) != 50:
        raise RuntimeError(
            f"Expected 50 segmentation rows, "
            f"found {len(segmentation_df)}"
        )

    if len(quality_df) != 50:
        raise RuntimeError(
            f"Expected 50 image-quality rows, "
            f"found {len(quality_df)}"
        )

    merged = quality_df.merge(
        segmentation_df[
            [
                "condition",
                "drop_dice_WT",
            ]
        ],
        on="condition",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != 50:
        raise RuntimeError(
            f"Expected 50 merged rows, found {len(merged)}"
        )

    region_metrics = {
        "WT": "drop_dice_WT",
        "TC": "drop_dice_TC",
        "ET": "drop_dice_ET",
    }

    for region, metric_column in region_metrics.items():
        png_path, pdf_path = REGION_OUTPUTS[region]

        plot_region_curve(
            dataframe=segmentation_df,
            region=region,
            metric_column=metric_column,
            png_path=png_path,
            pdf_path=pdf_path,
        )

    plot_report_scatter(
        dataframe=merged,
        x_column="mse_mean",
        x_label="Mean MSE",
        title="Image distortion versus nnU-Net WT Dice drop",
        png_path=MSE_SCATTER_PNG,
        pdf_path=MSE_SCATTER_PDF,
        reverse_x=False,
    )

    plot_report_scatter(
        dataframe=merged,
        x_column="psnr_mean",
        x_label="Mean PSNR (dB)",
        title="PSNR versus nnU-Net WT Dice drop",
        png_path=PSNR_SCATTER_PNG,
        pdf_path=PSNR_SCATTER_PDF,
        reverse_x=True,
    )

    print("Saved report-ready figures:")

    for png_path, pdf_path in REGION_OUTPUTS.values():
        print(f"  {png_path}")
        print(f"  {pdf_path}")

    for path in [
        MSE_SCATTER_PNG,
        MSE_SCATTER_PDF,
        PSNR_SCATTER_PNG,
        PSNR_SCATTER_PDF,
    ]:
        print(f"  {path}")

    print("=" * 80)


if __name__ == "__main__":
    main()
