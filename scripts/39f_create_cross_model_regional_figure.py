from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/xfh25/brats_segmentation_project")
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "report_materials"

CUSTOM_TAG = "regional_metrics_full_20260802"

NNUNET_FILE = (
    REPORT_DIR
    / "33a_nnunet_all_artifacts_L1_L10_summary.csv"
)

SWIN_FILE = (
    REPORT_DIR
    / "35b_swin_unetr_degraded_condition_summary.csv"
)

OUTPUT_CSV = (
    REPORT_DIR
    / "39f_cross_model_regional_condition_summary.csv"
)

OUTPUT_L10_CSV = (
    REPORT_DIR
    / "39f_cross_model_regional_L10_summary.csv"
)

OUTPUT_PNG = (
    REPORT_DIR
    / "39f_cross_model_regional_dice_drop_L10.png"
)

OUTPUT_PDF = (
    REPORT_DIR
    / "39f_cross_model_regional_dice_drop_L10.pdf"
)


MODEL_ORDER = [
    "Custom 3D U-Net",
    "nnU-Net v2",
    "Swin-UNETR",
]

ARTIFACT_ORDER = [
    "blur",
    "ghosting",
    "noise",
    "contrast",
    "frequency_truncation",
]

ARTIFACT_LABELS = {
    "blur": "Gaussian\nblur",
    "ghosting": "Ghosting",
    "noise": "Gaussian\nnoise",
    "contrast": "Contrast\nreduction",
    "frequency_truncation": "Frequency-domain\ntruncation",
}

REGIONS = {
    "WT": "Whole tumor",
    "TC": "Tumor core",
    "ET": "Enhancing tumor",
}

CUSTOM_CLEAN = {
    "WT": 0.762805,
    "TC": 0.653063,
    "ET": 0.589437,
}


def canonical_artifact(value):
    value = str(value).strip().lower()

    if "blur" in value:
        return "blur"

    if "ghost" in value:
        return "ghosting"

    if "noise" in value:
        return "noise"

    if "contrast" in value:
        return "contrast"

    if (
        "ring" in value
        or "fourier" in value
        or "frequency" in value
        or "trunc" in value
    ):
        return "frequency_truncation"

    raise ValueError(f"Unrecognized artifact name: {value}")


def load_custom():
    frames = []

    custom_files = {
        "blur": "blur",
        "ghosting": "ghosting",
        "noise": "noise",
        "contrast": "contrast",
        "frequency_truncation": "ringing",
    }

    for canonical_name, filename_name in custom_files.items():
        path = (
            RESULTS_DIR
            / (
                f"25b_3d_unet_{filename_name}_"
                f"full_volume_metrics_{CUSTOM_TAG}.csv"
            )
        )

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        summary = (
            df.groupby("level", as_index=False)
            .agg(
                num_patients=("patient_id", "nunique"),
                dice_WT=("whole_tumor_dice", "mean"),
                dice_TC=("tumor_core_dice", "mean"),
                dice_ET=("enhancing_tumor_dice", "mean"),
            )
        )

        summary["model"] = "Custom 3D U-Net"
        summary["artifact"] = canonical_name

        for region in REGIONS:
            summary[f"drop_dice_{region}"] = (
                CUSTOM_CLEAN[region]
                - summary[f"dice_{region}"]
            )

        frames.append(summary)

    return pd.concat(frames, ignore_index=True)


def load_nnunet():
    df = pd.read_csv(NNUNET_FILE)

    summary = pd.DataFrame({
        "model": "nnU-Net v2",
        "artifact": df["artifact"].map(canonical_artifact),
        "level": df["level"],
        "num_patients": df["num_test_patients"],
        "dice_WT": df["degraded_dice_WT"],
        "drop_dice_WT": df["drop_dice_WT"],
        "dice_TC": df["degraded_dice_TC"],
        "drop_dice_TC": df["drop_dice_TC"],
        "dice_ET": df["degraded_dice_ET"],
        "drop_dice_ET": df["drop_dice_ET"],
    })

    return summary


def load_swin():
    df = pd.read_csv(SWIN_FILE)

    summary = pd.DataFrame({
        "model": "Swin-UNETR",
        "artifact": df["artifact"].map(canonical_artifact),
        "level": df["level"],
        "num_patients": df["num_patients"],
        "dice_WT": df["dice_WT_mean"],
        "drop_dice_WT": df["dice_WT_drop_mean"],
        "dice_TC": df["dice_TC_mean"],
        "drop_dice_TC": df["dice_TC_drop_mean"],
        "dice_ET": df["dice_ET_mean"],
        "drop_dice_ET": df["dice_ET_drop_mean"],
    })

    return summary


def validate(combined):
    expected_rows = 3 * 5 * 10

    if len(combined) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, found {len(combined)}."
        )

    for model in MODEL_ORDER:
        model_df = combined[combined["model"] == model]

        for artifact in ARTIFACT_ORDER:
            artifact_df = model_df[
                model_df["artifact"] == artifact
            ]

            levels = sorted(
                artifact_df["level"].astype(int).tolist()
            )

            if levels != list(range(1, 11)):
                raise ValueError(
                    f"{model}, {artifact}: invalid levels {levels}"
                )


def create_figure(l10):
    x = np.arange(len(ARTIFACT_ORDER))
    bar_width = 0.24

    all_drops = []

    for region in REGIONS:
        all_drops.extend(
            l10[f"drop_dice_{region}"].dropna().tolist()
        )

    minimum = min(-0.03, min(all_drops) - 0.03)
    maximum = max(all_drops) + 0.08

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(8.2, 9.4),
        sharex=True,
        sharey=True,
    )

    for axis, (region, region_title) in zip(
        axes,
        REGIONS.items(),
    ):
        for model_index, model in enumerate(MODEL_ORDER):
            model_df = (
                l10[l10["model"] == model]
                .set_index("artifact")
                .reindex(ARTIFACT_ORDER)
            )

            positions = (
                x
                + (model_index - 1) * bar_width
            )

            axis.bar(
                positions,
                model_df[f"drop_dice_{region}"],
                width=bar_width,
                label=model,
            )

        axis.axhline(
            0,
            linewidth=0.8,
            color="black",
        )

        axis.set_title(
            region_title,
            fontsize=11,
            fontweight="bold",
        )

        axis.set_ylabel(
            "Absolute Dice decrease"
        )

        axis.set_ylim(
            minimum,
            maximum,
        )

        axis.grid(
            axis="y",
            alpha=0.25,
        )

    axes[-1].set_xticks(x)

    axes[-1].set_xticklabels(
        [
            ARTIFACT_LABELS[artifact]
            for artifact in ARTIFACT_ORDER
        ],
        fontsize=9,
    )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
    )

    fig.suptitle(
        "Regional Segmentation Deterioration at Severity Level L10",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )

    fig.tight_layout(
        rect=(0, 0.065, 1, 0.97)
    )

    fig.savefig(
        OUTPUT_PNG,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


def main():
    combined = pd.concat(
        [
            load_custom(),
            load_nnunet(),
            load_swin(),
        ],
        ignore_index=True,
    )

    combined["level"] = combined["level"].astype(int)

    combined["model"] = pd.Categorical(
        combined["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    combined["artifact"] = pd.Categorical(
        combined["artifact"],
        categories=ARTIFACT_ORDER,
        ordered=True,
    )

    combined = combined.sort_values(
        ["model", "artifact", "level"]
    ).reset_index(drop=True)

    validate(combined)

    combined.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    l10 = combined[
        combined["level"] == 10
    ].copy()

    l10.to_csv(
        OUTPUT_L10_CSV,
        index=False,
    )

    create_figure(l10)

    display_columns = [
        "model",
        "artifact",
        "dice_WT",
        "drop_dice_WT",
        "dice_TC",
        "drop_dice_TC",
        "dice_ET",
        "drop_dice_ET",
    ]

    print("Regional summary and Figure 4 created successfully.")
    print(f"Full summary: {OUTPUT_CSV}")
    print(f"L10 summary:  {OUTPUT_L10_CSV}")
    print(f"PNG:          {OUTPUT_PNG}")
    print(f"PDF:          {OUTPUT_PDF}")
    print()
    print("L10 REGIONAL RESULTS")
    print(
        l10[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


if __name__ == "__main__":
    main()
