from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")
REPORT_DIR = PROJECT_ROOT / "report_materials"

CUSTOM_FILE = (
    REPORT_DIR
    / "unet3d_xai_38b/final/38b_3d_unet_xai_condition_metrics.csv"
)

NNUNET_FILE = (
    REPORT_DIR
    / "nnunet_xai_37c_report/37c_nnunet_xai_condition_summary.csv"
)

SWIN_FILE = (
    REPORT_DIR
    / "swin_xai_36c_report/36c_swin_xai_condition_summary.csv"
)

OUTPUT_CSV = REPORT_DIR / "39d_cross_model_gradcam_condition_summary.csv"
OUTPUT_PNG = REPORT_DIR / "39d_cross_model_gradcam_severe_summary.png"
OUTPUT_PDF = REPORT_DIR / "39d_cross_model_gradcam_severe_summary.pdf"

MODEL_ORDER = [
    "Custom 3D U-Net",
    "nnU-Net v2",
    "Swin-UNETR",
]

CONDITION_ORDER = [
    "clean",
    "blur_L10",
    "ghosting_L10",
    "noise_L10",
    "ringing_L10",
    "contrast_L10",
]

CONDITION_LABELS = {
    "clean": "Clean",
    "blur_L10": "Blur\nL10",
    "ghosting_L10": "Ghosting\nL10",
    "noise_L10": "Noise\nL10",
    "ringing_L10": "Frequency-domain\ntruncation L10",
    "contrast_L10": "Contrast\nL10",
}


def load_custom():
    df = pd.read_csv(CUSTOM_FILE)

    available = df[
        df["gradcam_status"].astype(str).str.lower() == "available"
    ].copy()

    summary = (
        available
        .groupby("condition", as_index=False)
        .agg(
            patch_dice_WT=("patch_dice_WT", "mean"),
            heatmap_correlation=(
                "clean_to_condition_heatmap_correlation",
                "mean",
            ),
            centroid_shift=(
                "attribution_centroid_shift_voxels",
                "mean",
            ),
            available_gradcam=("gradcam_status", "size"),
        )
    )

    summary["model"] = "Custom 3D U-Net"
    return summary


def load_nnunet():
    df = pd.read_csv(NNUNET_FILE)

    summary = pd.DataFrame({
        "condition": df["condition"],
        "patch_dice_WT": df["patch_dice_WT_mean"],
        "heatmap_correlation": df["heatmap_correlation_mean"],
        "centroid_shift": df["centroid_shift_mean"],
        "available_gradcam": df["available_gradcam"],
        "model": "nnU-Net v2",
    })

    return summary


def load_swin():
    df = pd.read_csv(SWIN_FILE)

    summary = pd.DataFrame({
        "condition": df["condition"],
        "patch_dice_WT": df["patch_dice_WT"],
        "heatmap_correlation": df["heatmap_correlation"],
        "centroid_shift": df["centroid_shift"],
        "available_gradcam": df["available_gradcam"],
        "model": "Swin-UNETR",
    })

    return summary


def validate(df):
    for model in MODEL_ORDER:
        model_df = df[df["model"] == model]

        missing = [
            condition
            for condition in CONDITION_ORDER
            if condition not in model_df["condition"].tolist()
        ]

        if missing:
            raise ValueError(
                f"{model}: missing required conditions {missing}"
            )


def main():
    combined = pd.concat(
        [
            load_custom(),
            load_nnunet(),
            load_swin(),
        ],
        ignore_index=True,
    )

    validate(combined)

    combined.to_csv(OUTPUT_CSV, index=False)

    plot_df = combined[
        combined["condition"].isin(CONDITION_ORDER)
    ].copy()

    plot_df["condition"] = pd.Categorical(
        plot_df["condition"],
        categories=CONDITION_ORDER,
        ordered=True,
    )

    plot_df["model"] = pd.Categorical(
        plot_df["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    plot_df = plot_df.sort_values(["condition", "model"])

    x = np.arange(len(CONDITION_ORDER))
    bar_width = 0.24

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(7.2, 9.2),
        sharex=True,
    )

    panels = [
        (
            "patch_dice_WT",
            "Mean patch-level WT Dice",
            "Patch-level segmentation performance",
        ),
        (
            "heatmap_correlation",
            "Mean heatmap correlation",
            "Clean-to-degraded attribution similarity",
        ),
        (
            "centroid_shift",
            "Mean centroid shift (voxels)",
            "Attribution centroid displacement",
        ),
    ]

    for ax, (metric, ylabel, title) in zip(axes, panels):
        for model_index, model in enumerate(MODEL_ORDER):
            model_df = (
                plot_df[plot_df["model"] == model]
                .set_index("condition")
                .reindex(CONDITION_ORDER)
            )

            positions = (
                x
                + (model_index - 1) * bar_width
            )

            ax.bar(
                positions,
                model_df[metric].to_numpy(),
                width=bar_width,
                label=model,
            )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylim(0.0, 1.0)
    axes[1].set_ylim(-0.10, 1.05)
    axes[1].axhline(0.0, linewidth=0.8, color="black")

    maximum_shift = plot_df["centroid_shift"].max()
    axes[2].set_ylim(0.0, maximum_shift * 1.15)

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(
        [CONDITION_LABELS[c] for c in CONDITION_ORDER],
        fontsize=9,
    )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.025),
    )

    fig.text(
        0.5,
        0.008,
        (
            "Swin-UNETR noise L10 includes four available maps; "
            "all other model-condition cells include five."
        ),
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(rect=(0, 0.075, 1, 1))

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

    selected = plot_df[
        [
            "model",
            "condition",
            "patch_dice_WT",
            "heatmap_correlation",
            "centroid_shift",
            "available_gradcam",
        ]
    ]

    print("Cross-model Grad-CAM summary created successfully.")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")
    print()
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
