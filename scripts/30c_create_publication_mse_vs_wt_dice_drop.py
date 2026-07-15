#!/usr/bin/env python3
"""
Script 30C: Create a simplified publication-style MSE versus WT Dice decrease figure.

Purpose
-------
Show how objective image distortion, measured by mean squared error, relates
to nnU-Net whole-tumor Dice loss across degradation types.

Design
------
- One color per degradation type.
- One circle marker for every severity level.
- One solid line per degradation type.
- Only the final available severity level is labeled.
- Blur and ghosting end at L5.
- Noise, contrast, and frequency-domain truncation continue through L10.

Interpretation
--------------
- Moving right means greater image distortion.
- Moving upward means greater segmentation performance loss.
- A line that stays near zero indicates robustness despite image distortion.
- A sharply rising line indicates sensitivity to that artifact.

Terminology
-----------
The stored artifact name "ringing" is displayed as
"Frequency-domain truncation."

This condition uses Fourier truncation and should not be described as pure
classic Gibbs ringing.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(
    "/home/xfh25/brats_segmentation_project"
)

REPORT_DIR = PROJECT_ROOT / "report_materials"

INPUT_CSV = (
    REPORT_DIR
    / "30b_nnunet_degradation_strength_vs_drop_summary.csv"
)

OUTPUT_PNG = (
    REPORT_DIR
    / "30c_publication_mse_vs_wt_dice_decrease.png"
)

OUTPUT_PDF = (
    REPORT_DIR
    / "30c_publication_mse_vs_wt_dice_decrease.pdf"
)

OUTPUT_TXT = (
    REPORT_DIR
    / "30c_publication_mse_vs_wt_dice_decrease_note.txt"
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
    "noise": "Noise",
    "contrast": "Contrast",
    "ringing": "Frequency-domain truncation",
}

ARTIFACT_COLORS = {
    "blur": "tab:blue",
    "ghosting": "tab:orange",
    "noise": "tab:green",
    "contrast": "tab:red",
    "ringing": "tab:purple",
}

FINAL_LABELS = {
    "blur": "Blur L5",
    "ghosting": "Ghosting L5",
    "noise": "Noise L10",
    "contrast": "Contrast L10",
    "ringing": "FFT truncation L10",
}

LABEL_OFFSETS = {
    "blur": (7, 5),
    "ghosting": (7, 5),
    "noise": (7, 4),
    "contrast": (7, -12),
    "ringing": (7, 7),
}

EXPECTED_ROWS = 40


def load_data():
    """Load and validate the Script 30B combined summary."""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Missing Script 30B summary: {INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    required_columns = {
        "condition",
        "artifact",
        "level",
        "mse_mean",
        "dice_WT_drop",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise RuntimeError(
            "Input summary is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows, found {len(df)}"
        )

    unexpected_artifacts = sorted(
        set(df["artifact"]) - set(ARTIFACT_ORDER)
    )

    if unexpected_artifacts:
        raise RuntimeError(
            f"Unexpected artifacts found: {unexpected_artifacts}"
        )

    df["level"] = pd.to_numeric(
        df["level"],
        errors="raise",
    ).astype(int)

    df["mse_mean"] = pd.to_numeric(
        df["mse_mean"],
        errors="raise",
    )

    df["dice_WT_drop"] = pd.to_numeric(
        df["dice_WT_drop"],
        errors="raise",
    )

    return df


def add_final_label(axis, artifact_df, artifact):
    """Label the final available severity point for one artifact."""
    final_row = artifact_df.sort_values(
        "level"
    ).iloc[-1]

    x_offset, y_offset = LABEL_OFFSETS[
        artifact
    ]

    axis.annotate(
        FINAL_LABELS[artifact],
        xy=(
            final_row["mse_mean"],
            final_row["dice_WT_drop"],
        ),
        xytext=(
            x_offset,
            y_offset,
        ),
        textcoords="offset points",
        fontsize=9,
    )


def create_figure(df):
    """Create the simplified publication figure."""
    figure, axis = plt.subplots(
        figsize=(10, 6.5)
    )

    for artifact in ARTIFACT_ORDER:
        artifact_df = (
            df[
                df["artifact"] == artifact
            ]
            .sort_values("level")
            .copy()
        )

        if artifact_df.empty:
            continue

        axis.plot(
            artifact_df["mse_mean"],
            artifact_df["dice_WT_drop"],
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=7,
            color=ARTIFACT_COLORS[artifact],
            label=DISPLAY_NAMES[artifact],
        )

        add_final_label(
            axis=axis,
            artifact_df=artifact_df,
            artifact=artifact,
        )

    axis.axhline(
        y=0,
        linestyle="--",
        linewidth=1,
        color="gray",
    )

    axis.set_xlabel(
        "Mean squared error (MSE)",
        fontsize=11,
    )

    axis.set_ylabel(
        "Whole-tumor Dice decrease",
        fontsize=11,
    )

    axis.set_title(
        "Objective Image Distortion Versus nnU-Net Whole-Tumor Dice Loss",
        fontsize=13,
    )

    axis.legend(
        title="Degradation type",
        frameon=True,
        fontsize=9,
        title_fontsize=9,
        loc="upper right",
    )

    axis.tick_params(
        axis="both",
        labelsize=10,
    )

    axis.margins(
        x=0.05,
        y=0.08,
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
    )

    plt.close(figure)


def write_figure_note():
    """Save a report-ready explanation of the figure."""
    note = (
        "Figure note:\n"
        "Mean squared error was calculated between clean and degraded MRI "
        "volumes within the brain mask. Moving to the right indicates greater "
        "image distortion. Moving upward indicates a larger whole-tumor Dice "
        "decrease relative to the clean same-patient baseline. Blur and "
        "ghosting were evaluated from L1 to L5. Noise, contrast, and "
        "frequency-domain truncation were evaluated from L1 to L10. The "
        "original and extended parameter schedules were developed separately, "
        "so severity levels should be interpreted within each artifact rather "
        "than as perfectly equivalent across artifacts. Small negative values "
        "indicate practical stability rather than genuine performance "
        "improvement. The condition labeled frequency-domain truncation uses "
        "Fourier truncation and is not pure classic Gibbs ringing.\n"
    )

    with open(
        OUTPUT_TXT,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(note)


def main():
    print("=" * 88)
    print(
        "Script 30C: Create simplified publication MSE versus WT Dice figure"
    )
    print("=" * 88)

    REPORT_DIR.mkdir(exist_ok=True)

    df = load_data()

    create_figure(df)
    write_figure_note()

    print()
    print("Script 30C completed")
    print(f"PNG figure: {OUTPUT_PNG}")
    print(f"PDF figure: {OUTPUT_PDF}")
    print(f"Figure note: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
