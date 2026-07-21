#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/home/xfh25/brats_segmentation_project")
REPORT_DIR = ROOT / "report_materials"

SWIN_CSV = (
    REPORT_DIR
    / "35b_swin_unetr_degraded_condition_summary.csv"
)

QUALITY_CSV = (
    REPORT_DIR
    / "33c_nnunet_all_artifacts_L1_L10_psnr_mse_summary.csv"
)

OUT_CSV = (
    REPORT_DIR
    / "35d_swin_unetr_image_quality_vs_segmentation.csv"
)

OUT_TXT = (
    REPORT_DIR
    / "35d_swin_unetr_image_quality_vs_segmentation_summary.txt"
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

PLOTS = [
    {
        "x": "mse_mean",
        "y": "dice_WT_drop_mean",
        "xlabel": "Mean squared error (MSE)",
        "ylabel": "Mean paired WT Dice drop",
        "title": "Swin-UNETR: MSE versus WT Dice decrease",
        "threshold": 0.10,
        "filename": "35d_swin_unetr_mse_vs_wt_dice_drop",
        "invert_x": False,
    },
    {
        "x": "psnr_mean",
        "y": "dice_WT_drop_mean",
        "xlabel": "Peak signal-to-noise ratio (dB)",
        "ylabel": "Mean paired WT Dice drop",
        "title": "Swin-UNETR: PSNR versus WT Dice decrease",
        "threshold": 0.10,
        "filename": "35d_swin_unetr_psnr_vs_wt_dice_drop",
        "invert_x": True,
    },
    {
        "x": "mse_mean",
        "y": "iou_WT_drop_mean",
        "xlabel": "Mean squared error (MSE)",
        "ylabel": "Mean paired WT IoU drop",
        "title": "Swin-UNETR: MSE versus WT IoU decrease",
        "threshold": 0.15,
        "filename": "35d_swin_unetr_mse_vs_wt_iou_drop",
        "invert_x": False,
    },
    {
        "x": "psnr_mean",
        "y": "iou_WT_drop_mean",
        "xlabel": "Peak signal-to-noise ratio (dB)",
        "ylabel": "Mean paired WT IoU drop",
        "title": "Swin-UNETR: PSNR versus WT IoU decrease",
        "threshold": 0.15,
        "filename": "35d_swin_unetr_psnr_vs_wt_iou_drop",
        "invert_x": True,
    },
]


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


def validate_conditions(df: pd.DataFrame, name: str) -> None:
    if len(df) != 50:
        raise RuntimeError(
            f"{name}: expected 50 rows, found {len(df)}."
        )

    if df["condition"].duplicated().any():
        duplicates = df.loc[
            df["condition"].duplicated(keep=False),
            "condition",
        ].tolist()
        raise RuntimeError(
            f"{name}: duplicate conditions: {duplicates}"
        )

    expected = {
        f"{artifact}_L{level}"
        for artifact in ARTIFACT_ORDER
        for level in range(1, 11)
    }

    actual = set(df["condition"].astype(str))

    if actual != expected:
        raise RuntimeError(
            f"{name}: condition mismatch. "
            f"Missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def main() -> None:
    print("=" * 80)
    print("Script 35D: Swin image quality versus segmentation")
    print("=" * 80)

    for required in [SWIN_CSV, QUALITY_CSV]:
        if not required.exists():
            raise FileNotFoundError(
                f"Missing required input: {required}"
            )

    output_paths = [OUT_CSV, OUT_TXT]

    for plot in PLOTS:
        base = REPORT_DIR / plot["filename"]
        output_paths.extend([
            base.with_suffix(".png"),
            base.with_suffix(".pdf"),
        ])

    existing = [
        path for path in output_paths
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing outputs:\n"
            + "\n".join(str(path) for path in existing)
        )

    swin = pd.read_csv(SWIN_CSV)
    quality = pd.read_csv(QUALITY_CSV)

    validate_conditions(swin, "Swin summary")
    validate_conditions(quality, "Image-quality summary")

    quality_columns = [
        "artifact",
        "level",
        "condition",
        "mse_mean",
        "mse_std",
        "mse_median",
        "psnr_mean",
        "psnr_std",
        "psnr_median",
        "n_measurements",
        "n_patients",
    ]

    merged = swin.merge(
        quality[quality_columns],
        on=["artifact", "level", "condition"],
        how="inner",
        validate="one_to_one",
        suffixes=("_swin", "_quality"),
    )

    if len(merged) != 50:
        raise RuntimeError(
            f"Expected 50 merged rows, found {len(merged)}."
        )

    required_numeric = [
        "mse_mean",
        "psnr_mean",
        "dice_WT_drop_mean",
        "iou_WT_drop_mean",
    ]

    if merged[required_numeric].isna().any().any():
        bad = merged[
            merged[required_numeric].isna().any(axis=1)
        ]
        raise RuntimeError(
            "Missing required values:\n"
            + bad[
                ["condition"] + required_numeric
            ].to_string(index=False)
        )

    merged["dice_threshold_crossed"] = (
        merged["dice_WT_drop_mean"] >= 0.10
    )

    merged["iou_threshold_crossed"] = (
        merged["iou_WT_drop_mean"] >= 0.15
    )

    merged["break_threshold_crossed"] = (
        merged["dice_threshold_crossed"]
        | merged["iou_threshold_crossed"]
    )

    merged["artifact_order"] = pd.Categorical(
        merged["artifact"],
        categories=ARTIFACT_ORDER,
        ordered=True,
    )

    merged = (
        merged
        .sort_values(["artifact_order", "level"])
        .drop(columns="artifact_order")
        .reset_index(drop=True)
    )

    merged.to_csv(
        OUT_CSV,
        index=False,
    )

    for plot in PLOTS:
        plt.figure(figsize=(9, 6))

        for artifact in ARTIFACT_ORDER:
            subset = (
                merged[
                    merged["artifact"] == artifact
                ]
                .sort_values("level")
            )

            plt.plot(
                subset[plot["x"]],
                subset[plot["y"]],
                marker="o",
                linewidth=1.8,
                label=DISPLAY_NAMES[artifact],
            )

            for _, row in subset.iterrows():
                plt.annotate(
                    f"L{int(row['level'])}",
                    (
                        row[plot["x"]],
                        row[plot["y"]],
                    ),
                    fontsize=7,
                    xytext=(3, 3),
                    textcoords="offset points",
                )

        plt.axhline(
            0,
            linestyle="--",
            linewidth=1,
            label="No decrease",
        )

        plt.axhline(
            plot["threshold"],
            linestyle=":",
            linewidth=1.5,
            label=(
                "Breaking threshold "
                f"({plot['threshold']:.2f})"
            ),
        )

        if plot["invert_x"]:
            plt.gca().invert_xaxis()

        plt.xlabel(plot["xlabel"])
        plt.ylabel(plot["ylabel"])
        plt.title(plot["title"])
        plt.grid(alpha=0.25)
        plt.legend(frameon=False)

        save_figure(
            REPORT_DIR / plot["filename"]
        )

    first_break_rows = []

    for artifact in ARTIFACT_ORDER:
        subset = (
            merged[
                merged["artifact"] == artifact
            ]
            .sort_values("level")
        )

        broken = subset[
            subset["break_threshold_crossed"]
        ]

        if broken.empty:
            first_break_rows.append({
                "artifact": DISPLAY_NAMES[artifact],
                "first_break_level": "None",
                "mse_mean": "",
                "psnr_mean": "",
                "dice_WT_drop_mean": "",
                "iou_WT_drop_mean": "",
            })
        else:
            row = broken.iloc[0]

            first_break_rows.append({
                "artifact": DISPLAY_NAMES[artifact],
                "first_break_level": int(row["level"]),
                "mse_mean": float(row["mse_mean"]),
                "psnr_mean": float(row["psnr_mean"]),
                "dice_WT_drop_mean": float(
                    row["dice_WT_drop_mean"]
                ),
                "iou_WT_drop_mean": float(
                    row["iou_WT_drop_mean"]
                ),
            })

    first_break_df = pd.DataFrame(first_break_rows)

    with open(OUT_TXT, "w", encoding="utf-8") as file:
        file.write("=" * 80 + "\n")
        file.write(
            "Swin-UNETR image quality versus segmentation summary\n"
        )
        file.write("=" * 80 + "\n\n")
        file.write("Conditions: 50\n")
        file.write("Patients per condition: 74\n")
        file.write(
            "Breaking rule: WT Dice drop >= 0.10 "
            "or WT IoU drop >= 0.15\n\n"
        )
        file.write("First breaking condition by artifact:\n")
        file.write(
            first_break_df.to_string(index=False)
        )
        file.write("\n\n")
        file.write(
            "Terminology: internal key 'ringing' represents "
            "Fourier/frequency-domain truncation, not pure "
            "classic Gibbs ringing.\n"
        )

    print(f"Merged rows: {len(merged)}")
    print(f"Saved CSV: {OUT_CSV}")
    print(f"Saved summary: {OUT_TXT}")
    print()
    print("First breaking condition:")
    print(first_break_df.to_string(index=False))
    print()
    print("Figures created:")

    for plot in PLOTS:
        base = REPORT_DIR / plot["filename"]
        print(f"- {base}.png")
        print(f"- {base}.pdf")

    print("=" * 80)


if __name__ == "__main__":
    main()
