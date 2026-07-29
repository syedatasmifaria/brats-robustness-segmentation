#!/usr/bin/env python3
"""
Script 37C: Create report-ready nnU-Net XAI summary tables and figures
from the finalized Script 37B metrics and condition figures.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(
    "/home/xfh25/brats_segmentation_project"
)

INPUT_DIR = (
    PROJECT_ROOT
    / "report_materials"
    / "nnunet_xai_37b"
    / "final"
)

METRICS_PATH = (
    INPUT_DIR
    / "37b_nnunet_xai_condition_metrics.csv"
)

FIGURE_DIR = INPUT_DIR / "figures"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "report_materials"
    / "nnunet_xai_37c_report"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


ARTIFACT_CONDITIONS = {
    "blur": [
        ("clean", 0),
        ("blur_L3", 3),
        ("blur_L4", 4),
        ("blur_L10", 10),
    ],
    "ghosting": [
        ("clean", 0),
        ("ghosting_L4", 4),
        ("ghosting_L5", 5),
        ("ghosting_L10", 10),
    ],
    "noise": [
        ("clean", 0),
        ("noise_L6", 6),
        ("noise_L7", 7),
        ("noise_L10", 10),
    ],
    "ringing": [
        ("clean", 0),
        ("ringing_L7", 7),
        ("ringing_L8", 8),
        ("ringing_L10", 10),
    ],
    "contrast": [
        ("clean", 0),
        ("contrast_L10", 10),
    ],
}

REPORTING_LABELS = {
    "blur": "Blur",
    "ghosting": "Ghosting",
    "noise": "Gaussian noise",
    "ringing": "Fourier truncation",
    "contrast": "Contrast reduction",
}

CONDITION_ORDER = [
    "clean",
    "blur_L3",
    "blur_L4",
    "blur_L10",
    "ghosting_L4",
    "ghosting_L5",
    "ghosting_L10",
    "noise_L6",
    "noise_L7",
    "noise_L10",
    "ringing_L7",
    "ringing_L8",
    "ringing_L10",
    "contrast_L10",
]


def save_condition_summary(
    df: pd.DataFrame,
) -> None:
    summary = (
        df.groupby(
            "condition",
            sort=False,
        )
        .agg(
            n=(
                "patient_id",
                "count",
            ),
            patch_dice_WT_mean=(
                "patch_dice_WT",
                "mean",
            ),
            patch_dice_WT_std=(
                "patch_dice_WT",
                "std",
            ),
            heatmap_correlation_mean=(
                "clean_to_condition_heatmap_correlation",
                "mean",
            ),
            heatmap_correlation_std=(
                "clean_to_condition_heatmap_correlation",
                "std",
            ),
            centroid_shift_mean=(
                "attribution_centroid_shift_voxels",
                "mean",
            ),
            centroid_shift_std=(
                "attribution_centroid_shift_voxels",
                "std",
            ),
            high_saliency_WT_iou_mean=(
                "high_saliency_WT_iou",
                "mean",
            ),
            high_saliency_WT_iou_std=(
                "high_saliency_WT_iou",
                "std",
            ),
            inside_outside_ratio_mean=(
                "inside_outside_ratio",
                "mean",
            ),
            available_gradcam=(
                "gradcam_status",
                lambda values: int(
                    (values == "available").sum()
                ),
            ),
        )
        .reindex(CONDITION_ORDER)
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR
        / "37c_nnunet_xai_condition_summary.csv",
        index=False,
    )


def save_metric_curve(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_name: str,
) -> None:
    figure, axis = plt.subplots(
        figsize=(9, 6),
    )

    for artifact, condition_levels in (
        ARTIFACT_CONDITIONS.items()
    ):
        conditions = [
            condition
            for condition, _ in condition_levels
        ]

        levels = [
            level
            for _, level in condition_levels
        ]

        means = (
            df[df["condition"].isin(conditions)]
            .groupby("condition")[metric]
            .mean()
            .reindex(conditions)
        )

        axis.plot(
            levels,
            means.values,
            marker="o",
            linewidth=2,
            label=REPORTING_LABELS[artifact],
        )

    axis.set_xticks(
        [0, 3, 4, 5, 6, 7, 8, 10]
    )

    axis.set_xticklabels(
        [
            "Clean",
            "L3",
            "L4",
            "L5",
            "L6",
            "L7",
            "L8",
            "L10",
        ]
    )

    axis.set_xlabel(
        "Selected degradation level"
    )

    axis.set_ylabel(
        ylabel
    )

    axis.set_title(
        ylabel
        + " across selected nnU-Net XAI conditions"
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend(
        frameon=False
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIR / output_name,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def find_condition_figure(
    patient_id: str,
    condition: str,
) -> Path:
    matches = sorted(
        FIGURE_DIR.glob(
            f"*{patient_id}*{condition}*.png"
        )
    )

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one figure for "
            f"{patient_id} {condition}, "
            f"found {len(matches)}."
        )

    return matches[0]


def save_visual_comparison(
    patient_id: str,
    conditions: list[str],
    title: str,
    output_name: str,
) -> None:
    panel_ranges = [
        (20, 689),
        (814, 1483),
        (1608, 2277),
        (3196, 3865),
    ]

    panel_titles = [
        "FLAIR",
        "Ground-truth WT",
        "Predicted WT",
        "Grad-CAM overlay",
    ]

    vertical_crop = (
        104,
        773,
    )

    figure, axes = plt.subplots(
        len(conditions),
        4,
        figsize=(
            14,
            3.45 * len(conditions),
        ),
    )

    if len(conditions) == 1:
        axes = [axes]

    for row_axes, condition in zip(
        axes,
        conditions,
    ):
        image_path = find_condition_figure(
            patient_id,
            condition,
        )

        image = plt.imread(
            image_path
        )

        y0, y1 = vertical_crop

        for axis, (x0, x1), panel_title in zip(
            row_axes,
            panel_ranges,
            panel_titles,
        ):
            panel = image[
                y0:y1 + 1,
                x0:x1 + 1,
            ]

            axis.imshow(
                panel
            )

            axis.axis(
                "off"
            )

            if condition == conditions[0]:
                axis.set_title(
                    panel_title,
                    fontsize=13,
                    pad=7,
                )

        row_axes[0].text(
            -0.08,
            0.5,
            (
                "Fourier truncation L10"
                if condition == "ringing_L10"
                else condition.replace(
                    "_",
                    " ",
                ).title()
            ),
            transform=row_axes[0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=13,
            fontweight="bold",
        )

    figure.suptitle(
        f"nnU-Net: {title}",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )

    figure.subplots_adjust(
        left=0.055,
        right=0.995,
        bottom=0.02,
        top=0.94,
        wspace=0.035,
        hspace=0.16,
    )

    figure.savefig(
        OUTPUT_DIR / output_name,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.close(
        figure
    )


def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            METRICS_PATH
        )

    df = pd.read_csv(
        METRICS_PATH
    )

    required_columns = {
        "patient_id",
        "condition",
        "patch_dice_WT",
        "inside_outside_ratio",
        "high_saliency_WT_iou",
        "clean_to_condition_heatmap_correlation",
        "attribution_centroid_shift_voxels",
        "gradcam_status",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    if len(df) != 70:
        raise RuntimeError(
            f"Expected 70 rows, found {len(df)}."
        )

    if df["patient_id"].nunique() != 5:
        raise RuntimeError(
            "Expected five patients."
        )

    if df["condition"].nunique() != 14:
        raise RuntimeError(
            "Expected fourteen conditions."
        )

    if df.duplicated(
        ["patient_id", "condition"]
    ).any():
        raise RuntimeError(
            "Duplicate patient-condition rows found."
        )

    save_condition_summary(
        df
    )

    save_metric_curve(
        df=df,
        metric="patch_dice_WT",
        ylabel="Mean patch WT Dice",
        output_name=(
            "37c_nnunet_xai_patch_wt_dice.png"
        ),
    )

    save_metric_curve(
        df=df,
        metric=(
            "clean_to_condition_heatmap_correlation"
        ),
        ylabel=(
            "Mean clean-to-condition "
            "heatmap correlation"
        ),
        output_name=(
            "37c_nnunet_xai_heatmap_correlation.png"
        ),
    )

    save_metric_curve(
        df=df,
        metric=(
            "attribution_centroid_shift_voxels"
        ),
        ylabel=(
            "Mean attribution centroid "
            "shift (voxels)"
        ),
        output_name=(
            "37c_nnunet_xai_centroid_shift.png"
        ),
    )

    save_visual_comparison(
        patient_id="BraTS20_Training_178",
        conditions=[
            "clean",
            "ghosting_L5",
            "ghosting_L10",
            "contrast_L10",
        ],
        title=(
            "High-performing case: attribution under "
            "ghosting and contrast reduction"
        ),
        output_name=(
            "37c_strong_case_ghosting_contrast.png"
        ),
    )

    save_visual_comparison(
        patient_id="BraTS20_Training_046",
        conditions=[
            "clean",
            "ghosting_L10",
            "noise_L10",
            "ringing_L10",
        ],
        title=(
            "Difficult case: severe "
            "degradation and attribution drift"
        ),
        output_name=(
            "37c_difficult_case_severe_degradations.png"
        ),
    )

    save_visual_comparison(
        patient_id="BraTS20_Training_094",
        conditions=[
            "clean",
            "blur_L10",
            "ghosting_L10",
            "ringing_L10",
        ],
        title=(
            "Degradation-sensitive case: "
            "blur, ghosting, and Fourier truncation"
        ),
        output_name=(
            "37c_sensitive_case_degradation_comparison.png"
        ),
    )

    print(
        "Script 37C report-ready nnU-Net "
        "XAI outputs created."
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
