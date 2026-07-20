#!/usr/bin/env python3
"""
Script 33D: Plot final nnU-Net image quality and segmentation relationships.

Creates:
1. MSE vs severity for all five artifacts, L1-L10
2. PSNR vs severity for all five artifacts, L1-L10
3. MSE vs WT Dice drop
4. PSNR vs WT Dice drop

Interpretation:
- Higher MSE means greater image-level change.
- Lower PSNR means greater image-level change.
- Higher WT Dice drop means greater segmentation damage.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")
REPORT_DIR = PROJECT_ROOT / "report_materials"

QUALITY_CSV = (
    REPORT_DIR
    / "33c_nnunet_all_artifacts_L1_L10_psnr_mse_summary.csv"
)

SEGMENTATION_CSV = (
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

MSE_CURVE_PNG = (
    REPORT_DIR
    / "33d_nnunet_all_artifacts_mse_L1_L10.png"
)
MSE_CURVE_PDF = (
    REPORT_DIR
    / "33d_nnunet_all_artifacts_mse_L1_L10.pdf"
)

PSNR_CURVE_PNG = (
    REPORT_DIR
    / "33d_nnunet_all_artifacts_psnr_L1_L10.png"
)
PSNR_CURVE_PDF = (
    REPORT_DIR
    / "33d_nnunet_all_artifacts_psnr_L1_L10.pdf"
)

MSE_DICE_PNG = (
    REPORT_DIR
    / "33d_nnunet_mse_vs_wt_dice_drop.png"
)
MSE_DICE_PDF = (
    REPORT_DIR
    / "33d_nnunet_mse_vs_wt_dice_drop.pdf"
)

PSNR_DICE_PNG = (
    REPORT_DIR
    / "33d_nnunet_psnr_vs_wt_dice_drop.png"
)
PSNR_DICE_PDF = (
    REPORT_DIR
    / "33d_nnunet_psnr_vs_wt_dice_drop.pdf"
)


def save_figure(png_path, pdf_path):
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()


def plot_severity_curve(
    dataframe,
    metric,
    ylabel,
    title,
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
            subset[metric],
            marker="o",
            linewidth=2,
            label=DISPLAY_NAMES[artifact],
        )

    plt.xlabel("Degradation severity level")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(LEVELS)
    plt.grid(alpha=0.25)
    plt.legend()

    save_figure(png_path, pdf_path)


def plot_scatter(
    dataframe,
    x_column,
    x_label,
    title,
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
            subset[x_column],
            subset["drop_dice_WT"],
            marker="o",
            linewidth=1.5,
            label=DISPLAY_NAMES[artifact],
        )

        for _, row in subset.iterrows():
            plt.annotate(
                f"L{int(row['level'])}",
                (
                    row[x_column],
                    row["drop_dice_WT"],
                ),
                fontsize=7,
                xytext=(3, 3),
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

    save_figure(png_path, pdf_path)


def main():
    print("=" * 80)
    print("Script 33D: Plot image quality vs segmentation damage")
    print("=" * 80)

    quality_df = pd.read_csv(QUALITY_CSV)
    segmentation_df = pd.read_csv(SEGMENTATION_CSV)

    if len(quality_df) != 50:
        raise RuntimeError(
            f"Expected 50 image-quality rows, found {len(quality_df)}"
        )

    if len(segmentation_df) != 50:
        raise RuntimeError(
            f"Expected 50 segmentation rows, found {len(segmentation_df)}"
        )

    merged = quality_df.merge(
        segmentation_df[
            [
                "condition",
                "drop_dice_WT",
                "drop_iou_WT",
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

    plot_severity_curve(
        dataframe=quality_df,
        metric="mse_mean",
        ylabel="Mean MSE",
        title="Image degradation strength across severity levels",
        png_path=MSE_CURVE_PNG,
        pdf_path=MSE_CURVE_PDF,
    )

    plot_severity_curve(
        dataframe=quality_df,
        metric="psnr_mean",
        ylabel="Mean PSNR (dB)",
        title="Image quality across degradation severity levels",
        png_path=PSNR_CURVE_PNG,
        pdf_path=PSNR_CURVE_PDF,
    )

    plot_scatter(
        dataframe=merged,
        x_column="mse_mean",
        x_label="Mean MSE",
        title="Image distortion versus nnU-Net WT Dice loss",
        png_path=MSE_DICE_PNG,
        pdf_path=MSE_DICE_PDF,
    )

    plot_scatter(
        dataframe=merged,
        x_column="psnr_mean",
        x_label="Mean PSNR (dB)",
        title="PSNR versus nnU-Net WT Dice loss",
        png_path=PSNR_DICE_PNG,
        pdf_path=PSNR_DICE_PDF,
    )

    print("Saved figures:")
    for path in [
        MSE_CURVE_PNG,
        MSE_CURVE_PDF,
        PSNR_CURVE_PNG,
        PSNR_CURVE_PDF,
        MSE_DICE_PNG,
        MSE_DICE_PDF,
        PSNR_DICE_PNG,
        PSNR_DICE_PDF,
    ]:
        print(f"  {path}")

    print("=" * 80)


if __name__ == "__main__":
    main()
