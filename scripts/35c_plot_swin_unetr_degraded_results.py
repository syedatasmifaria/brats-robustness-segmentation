#!/usr/bin/env python3
"""
Script 35C: Create report-ready figures for Swin-UNETR degradation results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/xfh25/brats_segmentation_project")
REPORT = ROOT / "report_materials"

SUMMARY_FILE = (
    REPORT
    / "35b_swin_unetr_degraded_condition_summary.csv"
)

CLEAN_FILE = (
    ROOT
    / "results"
    / "34b_swin_unetr_clean_test_metrics.csv"
)

ARTIFACTS = [
    "blur",
    "ghosting",
    "noise",
    "contrast",
    "ringing",
]

LABELS = {
    "blur": "Blur",
    "ghosting": "Ghosting",
    "noise": "Gaussian noise",
    "contrast": "Contrast reduction",
    "ringing": "Fourier truncation",
}

OUTPUTS = {
    "wt_dice": REPORT / "35c_swin_unetr_wt_dice_by_level",
    "wt_drop": REPORT / "35c_swin_unetr_wt_dice_drop_by_level",
    "regional_drop": REPORT / "35c_swin_unetr_regional_dice_drop",
    "heatmap": REPORT / "35c_swin_unetr_wt_dice_drop_heatmap",
    "voxel_change": REPORT / "35c_swin_unetr_predicted_wt_voxel_change",
}


def save_figure(base_path: Path) -> None:
    plt.tight_layout()
    plt.savefig(
        base_path.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        base_path.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    plt.close()


def prepare_data() -> tuple[pd.DataFrame, float]:
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"Missing summary file: {SUMMARY_FILE}"
        )

    if not CLEAN_FILE.exists():
        raise FileNotFoundError(
            f"Missing clean metrics file: {CLEAN_FILE}"
        )

    summary = pd.read_csv(SUMMARY_FILE)
    clean = pd.read_csv(CLEAN_FILE)

    if len(summary) != 50:
        raise RuntimeError(
            f"Expected 50 summary rows, found {len(summary)}."
        )

    if summary["condition"].nunique() != 50:
        raise RuntimeError(
            "Expected 50 unique conditions."
        )

    clean_wt = float(
        clean["dice_WT"].mean()
    )

    summary["artifact"] = pd.Categorical(
        summary["artifact"],
        categories=ARTIFACTS,
        ordered=True,
    )

    summary = summary.sort_values(
        ["artifact", "level"]
    ).reset_index(drop=True)

    return summary, clean_wt


def plot_wt_dice(
    summary: pd.DataFrame,
    clean_wt: float,
) -> None:
    plt.figure(figsize=(9, 6))

    for artifact in ARTIFACTS:
        data = summary[
            summary["artifact"] == artifact
        ]

        plt.plot(
            data["level"],
            data["dice_WT_mean"],
            marker="o",
            linewidth=2,
            label=LABELS[artifact],
        )

    plt.axhline(
        clean_wt,
        linestyle="--",
        linewidth=1.5,
        label=f"Clean baseline ({clean_wt:.3f})",
    )

    plt.xticks(range(1, 11))
    plt.ylim(0, 1)
    plt.xlabel("Degradation level")
    plt.ylabel("Mean WT Dice")
    plt.title(
        "Swin-UNETR Whole-Tumor Dice Across Degradation Levels"
    )
    plt.grid(
        axis="y",
        alpha=0.25,
    )
    plt.legend(
        frameon=False,
        ncol=2,
    )

    save_figure(
        OUTPUTS["wt_dice"]
    )


def plot_wt_drop(
    summary: pd.DataFrame,
) -> None:
    plt.figure(figsize=(9, 6))

    for artifact in ARTIFACTS:
        data = summary[
            summary["artifact"] == artifact
        ]

        plt.plot(
            data["level"],
            data["dice_WT_drop_mean"],
            marker="o",
            linewidth=2,
            label=LABELS[artifact],
        )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    plt.axhline(
        0.10,
        linestyle=":",
        linewidth=1.5,
        label="Breaking threshold (0.10)",
    )

    plt.xticks(range(1, 11))
    plt.xlabel("Degradation level")
    plt.ylabel(
        "Mean paired WT Dice drop\n"
        "(clean minus degraded)"
    )
    plt.title(
        "Swin-UNETR Whole-Tumor Dice Decrease"
    )
    plt.grid(
        axis="y",
        alpha=0.25,
    )
    plt.legend(
        frameon=False,
        ncol=2,
    )

    save_figure(
        OUTPUTS["wt_drop"]
    )


def plot_regional_drop(
    summary: pd.DataFrame,
) -> None:
    region_columns = {
        "WT": "dice_WT_drop_mean",
        "TC": "dice_TC_drop_mean",
        "ET": "dice_ET_drop_mean",
    }

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(9, 13),
        sharex=True,
    )

    for axis, (
        region,
        column,
    ) in zip(
        axes,
        region_columns.items(),
    ):
        for artifact in ARTIFACTS:
            data = summary[
                summary["artifact"] == artifact
            ]

            axis.plot(
                data["level"],
                data[column],
                marker="o",
                linewidth=2,
                label=LABELS[artifact],
            )

        axis.axhline(
            0,
            linestyle="--",
            linewidth=1,
        )

        axis.set_ylabel(
            f"{region} Dice drop"
        )
        axis.set_title(
            f"{region}: Clean Minus Degraded Dice"
        )
        axis.grid(
            axis="y",
            alpha=0.25,
        )

    axes[-1].set_xlabel(
        "Degradation level"
    )
    axes[-1].set_xticks(
        range(1, 11)
    )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.suptitle(
        "Regional Swin-UNETR Dice Decrease Across Degradations",
        y=0.995,
    )

    fig.tight_layout(
        rect=[0, 0.05, 1, 0.98]
    )

    fig.savefig(
        OUTPUTS["regional_drop"].with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUTS["regional_drop"].with_suffix(".pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_heatmap(
    summary: pd.DataFrame,
) -> None:
    matrix = (
        summary.pivot(
            index="artifact",
            columns="level",
            values="dice_WT_drop_mean",
        )
        .reindex(ARTIFACTS)
    )

    values = matrix.to_numpy()

    plt.figure(figsize=(10, 5))

    image = plt.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
    )

    plt.colorbar(
        image,
        label="Mean paired WT Dice drop",
    )

    plt.xticks(
        np.arange(10),
        [f"L{level}" for level in range(1, 11)],
    )

    plt.yticks(
        np.arange(len(ARTIFACTS)),
        [LABELS[a] for a in ARTIFACTS],
    )

    plt.xlabel("Degradation level")
    plt.ylabel("Degradation type")
    plt.title(
        "Swin-UNETR Whole-Tumor Dice-Drop Heatmap"
    )

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            plt.text(
                column,
                row,
                f"{values[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    save_figure(
        OUTPUTS["heatmap"]
    )


def plot_voxel_change(
    summary: pd.DataFrame,
) -> None:
    plt.figure(figsize=(9, 6))

    for artifact in ARTIFACTS:
        data = summary[
            summary["artifact"] == artifact
        ]

        plt.plot(
            data["level"],
            data["pred_WT_voxels_change_mean"],
            marker="o",
            linewidth=2,
            label=LABELS[artifact],
        )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    plt.xticks(range(1, 11))
    plt.xlabel("Degradation level")
    plt.ylabel(
        "Mean change in predicted WT voxels\n"
        "(degraded minus clean)"
    )
    plt.title(
        "Change in Swin-UNETR Whole-Tumor Prediction Volume"
    )
    plt.grid(
        axis="y",
        alpha=0.25,
    )
    plt.legend(
        frameon=False,
        ncol=2,
    )

    save_figure(
        OUTPUTS["voxel_change"]
    )


def main() -> None:
    print("=" * 80)
    print("Script 35C: Plot Swin-UNETR degraded results")
    print("=" * 80)

    REPORT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for base_path in OUTPUTS.values():
        for suffix in [".png", ".pdf"]:
            output = base_path.with_suffix(suffix)

            if output.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing figure: {output}"
                )

    summary, clean_wt = prepare_data()

    plot_wt_dice(
        summary,
        clean_wt,
    )

    plot_wt_drop(
        summary
    )

    plot_regional_drop(
        summary
    )

    plot_heatmap(
        summary
    )

    plot_voxel_change(
        summary
    )

    print("Figures created:")
    for base_path in OUTPUTS.values():
        print(f"- {base_path}.png")
        print(f"- {base_path}.pdf")

    print("=" * 80)


if __name__ == "__main__":
    main()
