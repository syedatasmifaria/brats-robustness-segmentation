from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")
REPORT_DIR = PROJECT_ROOT / "report_materials"

CUSTOM_FILE = (
    PROJECT_ROOT
    / "results/25b_3d_unet_full_volume_robustness_summary_paired.csv"
)
NNUNET_FILE = (
    REPORT_DIR
    / "33a_nnunet_all_artifacts_L1_L10_summary.csv"
)
SWIN_FILE = (
    REPORT_DIR
    / "35b_swin_unetr_degraded_condition_summary.csv"
)

OUTPUT_PNG = REPORT_DIR / "39a_cross_model_wt_dice_by_level.png"
OUTPUT_PDF = REPORT_DIR / "39a_cross_model_wt_dice_by_level.pdf"

ARTIFACT_ORDER = [
    "Gaussian blur",
    "Ghosting",
    "Gaussian noise",
    "Contrast reduction",
    "Frequency-domain truncation",
]


def standardize_artifact(value):
    """Convert different internal artifact names to report-ready names."""
    text = str(value).lower().replace("_", " ").replace("-", " ")

    if "blur" in text:
        return "Gaussian blur"
    if "ghost" in text:
        return "Ghosting"
    if "noise" in text:
        return "Gaussian noise"
    if "contrast" in text:
        return "Contrast reduction"
    if any(term in text for term in ["ring", "fourier", "frequency"]):
        return "Frequency-domain truncation"

    raise ValueError(f"Unrecognized artifact name: {value}")


def load_custom():
    df = pd.read_csv(CUSTOM_FILE)

    tidy = pd.DataFrame({
        "artifact": df["artifact"].map(standardize_artifact),
        "level": pd.to_numeric(df["level"]),
        "wt_dice": pd.to_numeric(df["mean_wt_dice"]),
    })

    clean_baseline = float(df["mean_clean_wt_dice"].median())
    return tidy, clean_baseline


def load_nnunet():
    df = pd.read_csv(NNUNET_FILE)

    tidy = pd.DataFrame({
        "artifact": df["artifact"].map(standardize_artifact),
        "level": pd.to_numeric(df["level"]),
        "wt_dice": pd.to_numeric(df["degraded_dice_WT"]),
    })

    clean_baseline = float(df["clean_dice_WT"].median())
    return tidy, clean_baseline


def load_swin():
    df = pd.read_csv(SWIN_FILE)

    tidy = pd.DataFrame({
        "artifact": df["artifact"].map(standardize_artifact),
        "level": pd.to_numeric(df["level"]),
        "wt_dice": pd.to_numeric(df["dice_WT_mean"]),
    })

    # Clean Dice = degraded Dice + Dice decrease from clean.
    clean_values = (
        pd.to_numeric(df["dice_WT_mean"])
        + pd.to_numeric(df["dice_WT_drop_mean"])
    )
    clean_baseline = float(clean_values.median())

    return tidy, clean_baseline


def validate_data(model_name, df):
    found = set(df["artifact"].unique())
    missing = set(ARTIFACT_ORDER) - found

    if missing:
        raise ValueError(
            f"{model_name} is missing these artifacts: {sorted(missing)}"
        )

    for artifact in ARTIFACT_ORDER:
        levels = sorted(
            df.loc[df["artifact"] == artifact, "level"]
            .astype(int)
            .unique()
            .tolist()
        )

        if levels != list(range(1, 11)):
            raise ValueError(
                f"{model_name}, {artifact}: expected levels 1–10, "
                f"but found {levels}"
            )


def main():
    model_data = {
        "Custom 3D U-Net": load_custom(),
        "nnU-Net v2": load_nnunet(),
        "Swin-UNETR": load_swin(),
    }

    for model_name, (df, _) in model_data.items():
        validate_data(model_name, df)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(7.2, 9.4),
        sharex=True,
        sharey=True,
    )

    for ax, (model_name, (df, clean_baseline)) in zip(
        axes, model_data.items()
    ):
        for artifact in ARTIFACT_ORDER:
            plot_df = (
                df[df["artifact"] == artifact]
                .sort_values("level")
            )

            ax.plot(
                plot_df["level"],
                plot_df["wt_dice"],
                marker="o",
                markersize=4,
                linewidth=1.8,
                label=artifact,
            )

        ax.axhline(
            clean_baseline,
            linestyle="--",
            linewidth=1.4,
            color="black",
            label="Clean baseline",
        )

        ax.set_title(model_name, fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean WT Dice")
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(np.arange(0.0, 1.01, 0.1))
        ax.grid(True, alpha=0.25)

    axes[-1].set_xticks(range(1, 11))
    axes[-1].set_xlabel("Degradation severity level")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.tight_layout(rect=(0, 0.09, 1, 1))

    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print("Cross-model WT Dice figure created successfully.")
    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")
    print()
    print("Clean baselines used:")
    for model_name, (_, clean_baseline) in model_data.items():
        print(f"  {model_name}: {clean_baseline:.6f}")


if __name__ == "__main__":
    main()
