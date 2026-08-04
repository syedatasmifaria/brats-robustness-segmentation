from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")
REPORT_DIR = PROJECT_ROOT / "report_materials"

QUALITY_FILE = (
    REPORT_DIR
    / "33c_nnunet_all_artifacts_L1_L10_psnr_mse_summary.csv"
)

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

OUTPUT_PNG = REPORT_DIR / "39b_cross_model_mse_vs_wt_dice_drop.png"
OUTPUT_PDF = REPORT_DIR / "39b_cross_model_mse_vs_wt_dice_drop.pdf"

ARTIFACT_ORDER = [
    "Gaussian blur",
    "Ghosting",
    "Gaussian noise",
    "Contrast reduction",
    "Frequency-domain truncation",
]


def standardize_artifact(value):
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

    raise ValueError(f"Unrecognized artifact: {value}")


def load_quality():
    df = pd.read_csv(QUALITY_FILE)

    return pd.DataFrame({
        "artifact": df["artifact"].map(standardize_artifact),
        "level": pd.to_numeric(df["level"]),
        "mse": pd.to_numeric(df["mse_mean"]),
    })


def load_custom():
    df = pd.read_csv(CUSTOM_FILE)

    return pd.DataFrame({
        "artifact": df["artifact"].map(standardize_artifact),
        "level": pd.to_numeric(df["level"]),
        "wt_dice_drop": pd.to_numeric(df["mean_wt_dice_drop"]),
    })


def load_nnunet():
    df = pd.read_csv(NNUNET_FILE)

    return pd.DataFrame({
        "artifact": df["artifact"].map(standardize_artifact),
        "level": pd.to_numeric(df["level"]),
        "wt_dice_drop": pd.to_numeric(df["drop_dice_WT"]),
    })


def load_swin():
    df = pd.read_csv(SWIN_FILE)

    return pd.DataFrame({
        "artifact": df["artifact"].map(standardize_artifact),
        "level": pd.to_numeric(df["level"]),
        "wt_dice_drop": pd.to_numeric(df["dice_WT_drop_mean"]),
    })


def merge_with_quality(model_df, quality_df, model_name):
    merged = model_df.merge(
        quality_df,
        on=["artifact", "level"],
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != 50:
        raise ValueError(
            f"{model_name}: expected 50 merged conditions, "
            f"found {len(merged)}"
        )

    return merged


def validate_levels(df, model_name):
    for artifact in ARTIFACT_ORDER:
        levels = sorted(
            df.loc[df["artifact"] == artifact, "level"]
            .astype(int)
            .unique()
            .tolist()
        )

        if levels != list(range(1, 11)):
            raise ValueError(
                f"{model_name}, {artifact}: "
                f"expected L1-L10, found {levels}"
            )


def main():
    quality = load_quality()

    model_data = {
        "Custom 3D U-Net": merge_with_quality(
            load_custom(),
            quality,
            "Custom 3D U-Net",
        ),
        "nnU-Net v2": merge_with_quality(
            load_nnunet(),
            quality,
            "nnU-Net v2",
        ),
        "Swin-UNETR": merge_with_quality(
            load_swin(),
            quality,
            "Swin-UNETR",
        ),
    }

    for model_name, df in model_data.items():
        validate_levels(df, model_name)

    all_drops = pd.concat(
        [df["wt_dice_drop"] for df in model_data.values()],
        ignore_index=True,
    )

    y_min = min(-0.02, float(all_drops.min()) - 0.02)
    y_max = float(all_drops.max()) + 0.05

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(7.2, 9.4),
        sharex=True,
        sharey=True,
    )

    for ax, (model_name, df) in zip(axes, model_data.items()):
        for artifact in ARTIFACT_ORDER:
            plot_df = (
                df[df["artifact"] == artifact]
                .sort_values("level")
            )

            ax.plot(
                plot_df["mse"],
                plot_df["wt_dice_drop"],
                marker="o",
                markersize=4,
                linewidth=1.8,
                label=artifact,
            )

        ax.axhline(
            0.10,
            linestyle="--",
            linewidth=1.3,
            color="black",
            label="WT Dice-drop threshold",
        )

        ax.axhline(
            0.0,
            linewidth=0.8,
            color="gray",
        )

        ax.set_title(
            model_name,
            fontsize=11,
            fontweight="bold",
        )

        ax.set_ylabel("Mean WT Dice decrease")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel(
        "Mean squared error between clean and degraded MRI"
    )

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

    fig.savefig(
        OUTPUT_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Cross-model MSE versus WT Dice-decrease figure created.")
    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")
    print()

    for model_name, df in model_data.items():
        print(
            f"{model_name}: {len(df)} conditions, "
            f"maximum WT Dice decrease = "
            f"{df['wt_dice_drop'].max():.6f}"
        )


if __name__ == "__main__":
    main()