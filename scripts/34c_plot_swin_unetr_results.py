#!/usr/bin/env python3
"""Create report-ready Swin-UNETR training and clean-test figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")
REPORT_DIR = PROJECT_ROOT / "report_materials"
RESULTS_DIR = PROJECT_ROOT / "results"

HISTORY_PATH = (
    REPORT_DIR
    / "34a_swin_unetr_full_timing_20260719_training_history.csv"
)

TEST_METRICS_PATH = (
    RESULTS_DIR
    / "34b_swin_unetr_clean_test_metrics.csv"
)

LOSS_OUTPUT = (
    REPORT_DIR
    / "34c_swin_unetr_training_loss_curve"
)

VALIDATION_OUTPUT = (
    REPORT_DIR
    / "34c_swin_unetr_validation_dice_curve"
)

TEST_OUTPUT = (
    REPORT_DIR
    / "34c_swin_unetr_clean_test_region_dice"
)

SUMMARY_OUTPUT = (
    REPORT_DIR
    / "34c_swin_unetr_clean_test_region_summary.csv"
)


def save_figure(fig, output_base: Path) -> None:
    fig.savefig(
        output_base.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        output_base.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(HISTORY_PATH)

    if not TEST_METRICS_PATH.exists():
        raise FileNotFoundError(TEST_METRICS_PATH)

    history = pd.read_csv(HISTORY_PATH)
    test_metrics = pd.read_csv(TEST_METRICS_PATH)

    # ------------------------------------------------------------------
    # Figure 1: Training loss
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        history["epoch"],
        history["train_loss"],
        marker="o",
        markersize=3,
        linewidth=1.5,
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("Swin-UNETR Training Loss")
    ax.grid(alpha=0.25)

    save_figure(
        fig,
        LOSS_OUTPUT,
    )

    # ------------------------------------------------------------------
    # Figure 2: Full-volume validation Dice
    # ------------------------------------------------------------------
    validation = history.loc[
        history["validation_performed"].astype(str).str.lower()
        == "true"
    ].copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        validation["epoch"],
        validation["val_dice_WT"],
        marker="o",
        label="WT",
    )

    ax.plot(
        validation["epoch"],
        validation["val_dice_TC"],
        marker="o",
        label="TC",
    )

    ax.plot(
        validation["epoch"],
        validation["val_dice_ET"],
        marker="o",
        label="ET",
    )

    ax.plot(
        validation["epoch"],
        validation["val_macro_dice"],
        marker="o",
        linestyle="--",
        label="Macro Dice",
    )

    best_row = validation.loc[
        validation["val_macro_dice"].idxmax()
    ]

    ax.annotate(
        (
            f"Best epoch {int(best_row['epoch'])}\n"
            f"Macro Dice = {best_row['val_macro_dice']:.3f}"
        ),
        xy=(
            best_row["epoch"],
            best_row["val_macro_dice"],
        ),
        xytext=(8, -38),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Dice")
    ax.set_title("Swin-UNETR Full-Volume Validation Performance")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.25)

    save_figure(
        fig,
        VALIDATION_OUTPUT,
    )

    # ------------------------------------------------------------------
    # Figure 3: Clean-test regional Dice
    # ------------------------------------------------------------------
    regions = ["WT", "TC", "ET"]

    columns = [
        "dice_WT",
        "dice_TC",
        "dice_ET",
    ]

    means = np.array([
        test_metrics[column].mean()
        for column in columns
    ])

    standard_deviations = np.array([
        test_metrics[column].std(ddof=1)
        for column in columns
    ])

    summary = pd.DataFrame({
        "region": regions,
        "mean_dice": means,
        "sd_dice": standard_deviations,
        "n": len(test_metrics),
    })

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(
        regions,
        means,
        yerr=standard_deviations,
        capsize=6,
    )

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + 0.02,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
        )

    ax.set_xlabel("Tumor region")
    ax.set_ylabel("Dice score")
    ax.set_title(
        "Swin-UNETR Clean Test Performance (n=74)"
    )
    ax.set_ylim(0, 1)
    ax.grid(
        axis="y",
        alpha=0.25,
    )

    save_figure(
        fig,
        TEST_OUTPUT,
    )

    print("Created:")
    for path in [
        LOSS_OUTPUT.with_suffix(".png"),
        LOSS_OUTPUT.with_suffix(".pdf"),
        VALIDATION_OUTPUT.with_suffix(".png"),
        VALIDATION_OUTPUT.with_suffix(".pdf"),
        TEST_OUTPUT.with_suffix(".png"),
        TEST_OUTPUT.with_suffix(".pdf"),
        SUMMARY_OUTPUT,
    ]:
        print(path)


if __name__ == "__main__":
    main()
