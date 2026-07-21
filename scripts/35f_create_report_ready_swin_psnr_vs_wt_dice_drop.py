#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/home/xfh25/brats_segmentation_project")
REPORT_DIR = ROOT / "report_materials"

INPUT_CSV = (
    REPORT_DIR
    / "35d_swin_unetr_image_quality_vs_segmentation.csv"
)

OUT_PNG = (
    REPORT_DIR
    / "35f_swin_unetr_psnr_vs_wt_dice_drop_report_ready.png"
)

OUT_PDF = (
    REPORT_DIR
    / "35f_swin_unetr_psnr_vs_wt_dice_drop_report_ready.pdf"
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
    "ringing": "Fourier truncation",
}


def main() -> None:
    print("=" * 80)
    print("Script 35F: Report-ready Swin PSNR versus WT Dice drop")
    print("=" * 80)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Missing input: {INPUT_CSV}"
        )

    for output in [OUT_PNG, OUT_PDF]:
        if output.exists():
            raise FileExistsError(
                f"Refusing to overwrite: {output}"
            )

    df = pd.read_csv(INPUT_CSV)

    if len(df) != 50:
        raise RuntimeError(
            f"Expected 50 rows, found {len(df)}."
        )

    plt.figure(figsize=(9, 6))

    for artifact in ARTIFACT_ORDER:
        subset = (
            df[df["artifact"] == artifact]
            .sort_values("level")
            .copy()
        )

        plt.plot(
            subset["psnr_mean"],
            subset["dice_WT_drop_mean"],
            marker="o",
            linewidth=2,
            label=DISPLAY_NAMES[artifact],
        )

        crossed = subset[
            subset["dice_WT_drop_mean"] >= 0.10
        ]

        rows_to_label = [subset.iloc[-1]]

        if not crossed.empty:
            rows_to_label.append(crossed.iloc[0])

        labelled_levels = set()

        for row in rows_to_label:
            level = int(row["level"])

            if level in labelled_levels:
                continue

            labelled_levels.add(level)

            plt.annotate(
                f"L{level}",
                (
                    row["psnr_mean"],
                    row["dice_WT_drop_mean"],
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1,
        label="No decrease",
    )

    plt.axhline(
        0.10,
        linestyle=":",
        linewidth=1.5,
        label="Breaking threshold (0.10)",
    )

    plt.xlabel("Peak signal-to-noise ratio (dB)")
    plt.ylabel("Mean paired WT Dice drop")
    plt.title(
        "Swin-UNETR: Image Degradation versus "
        "Whole-Tumor Dice Decrease"
    )
    plt.gca().invert_xaxis()
    plt.grid(alpha=0.25)
    plt.legend(
        frameon=False,
        ncol=2,
    )
    plt.tight_layout()

    plt.savefig(
        OUT_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        OUT_PDF,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")
    print("=" * 80)


if __name__ == "__main__":
    main()
