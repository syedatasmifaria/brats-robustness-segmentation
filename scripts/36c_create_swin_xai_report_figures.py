#!/usr/bin/env python3
"""
Script 36C: Create report-ready Swin-UNETR XAI summary figures
from the finalized Script 36B metrics and selected condition figures.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

INPUT_DIR = (
    PROJECT_ROOT
    / "report_materials"
    / "swin_xai_36b"
    / "final"
)

METRICS_PATH = (
    INPUT_DIR
    / "36b_swin_xai_condition_metrics.csv"
)

FIGURE_DIR = INPUT_DIR / "figures"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "report_materials"
    / "swin_xai_36c_report"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


ARTIFACT_LEVEL_ORDER = {
    "blur": ["clean", "blur_L3", "blur_L4", "blur_L10"],
    "ghosting": [
        "clean",
        "ghosting_L4",
        "ghosting_L5",
        "ghosting_L10",
    ],
    "noise": [
        "clean",
        "noise_L6",
        "noise_L7",
        "noise_L10",
    ],
    "ringing": [
        "clean",
        "ringing_L7",
        "ringing_L8",
        "ringing_L10",
    ],
    "contrast": ["clean", "contrast_L10"],
}

REPORTING_LABELS = {
    "blur": "Blur",
    "ghosting": "Ghosting",
    "noise": "Gaussian noise",
    "ringing": "Fourier truncation",
    "contrast": "Contrast reduction",
}


def save_metric_curve(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_name: str,
) -> None:
    figure, axis = plt.subplots(
        figsize=(9, 6),
    )

    stage_positions = {
        "blur": [0, 1, 2, 3],
        "ghosting": [0, 1, 2, 3],
        "noise": [0, 1, 2, 3],
        "ringing": [0, 1, 2, 3],
        "contrast": [0, 3],
    }

    for artifact, conditions in ARTIFACT_LEVEL_ORDER.items():
        subset = (
            df[df["condition"].isin(conditions)]
            .groupby(
                "condition",
                sort=False,
            )[metric]
            .mean()
            .reindex(conditions)
        )

        axis.plot(
            stage_positions[artifact],
            subset.values,
            marker="o",
            linewidth=2,
            label=REPORTING_LABELS[artifact],
        )

    axis.set_xticks(
        [0, 1, 2, 3],
        [
            "Clean",
            "Pre-breaking",
            "First breaking",
            "Severe",
        ],
    )

    axis.set_xlabel(
        "Selected degradation stage"
    )
    axis.set_ylabel(
        ylabel
    )
    axis.set_title(
        ylabel + " across selected XAI conditions"
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


def save_condition_summary(
    df: pd.DataFrame,
) -> None:
    condition_order = [
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

    summary = (
        df.groupby(
            "condition",
            sort=False,
        )
        .agg(
            patch_dice_WT=(
                "patch_dice_WT",
                "mean",
            ),
            heatmap_correlation=(
                "clean_to_condition_heatmap_correlation",
                "mean",
            ),
            centroid_shift=(
                "attribution_centroid_shift_voxels",
                "mean",
            ),
            high_saliency_WT_iou=(
                "high_saliency_WT_iou",
                "mean",
            ),
            available_gradcam=(
                "gradcam_status",
                lambda values: int(
                    (values == "available").sum()
                ),
            ),
        )
        .reindex(condition_order)
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR
        / "36c_swin_xai_condition_summary.csv",
        index=False,
    )


def save_visual_comparison(
    patient_id: str,
    conditions: list[str],
    title: str,
    output_name: str,
) -> None:
    figure, axes = plt.subplots(
        len(conditions),
        1,
        figsize=(
            17,
            4.5 * len(conditions),
        ),
    )

    if len(conditions) == 1:
        axes = [axes]

    for axis, condition in zip(
        axes,
        conditions,
    ):
        image_path = (
            FIGURE_DIR
            / (
                f"36b_{patient_id}_"
                f"{condition}_gradcam.png"
            )
        )

        if not image_path.exists():
            raise FileNotFoundError(
                image_path
            )

        image = plt.imread(
            image_path
        )

        axis.imshow(
            image
        )
        axis.axis(
            "off"
        )
        axis.set_title(
            condition.replace(
                "_",
                " ",
            ),
            fontsize=13,
        )

    figure.suptitle(
        title,
        fontsize=16,
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.98]
    )

    figure.savefig(
        OUTPUT_DIR / output_name,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    df = pd.read_csv(
        METRICS_PATH
    )

    if len(df) != 70:
        raise RuntimeError(
            f"Expected 70 rows, found {len(df)}."
        )

    save_condition_summary(
        df
    )

    save_metric_curve(
        df=df,
        metric="patch_dice_WT",
        ylabel="Mean patch WT Dice",
        output_name=(
            "36c_swin_xai_patch_wt_dice.png"
        ),
    )

    save_metric_curve(
        df=df,
        metric=(
            "clean_to_condition_heatmap_correlation"
        ),
        ylabel=(
            "Mean clean-to-condition heatmap correlation"
        ),
        output_name=(
            "36c_swin_xai_heatmap_correlation.png"
        ),
    )

    save_metric_curve(
        df=df,
        metric=(
            "attribution_centroid_shift_voxels"
        ),
        ylabel=(
            "Mean attribution centroid shift (voxels)"
        ),
        output_name=(
            "36c_swin_xai_centroid_shift.png"
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
            "Strong clean case: attribution under "
            "ghosting and contrast reduction"
        ),
        output_name=(
            "36c_strong_case_clean_ghosting_contrast.png"
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
            "Difficult case: severe degradation "
            "and attribution drift"
        ),
        output_name=(
            "36c_difficult_case_severe_degradations.png"
        ),
    )

    save_visual_comparison(
        patient_id="BraTS20_Training_094",
        conditions=[
            "clean",
            "noise_L6",
            "noise_L7",
            "noise_L10",
        ],
        title=(
            "Small-ET case: progression to zero "
            "predicted WT under severe noise"
        ),
        output_name=(
            "36c_small_et_noise_progression.png"
        ),
    )

    print(
        "Script 36C report-ready XAI outputs created."
    )
    print(
        f"Output directory: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
