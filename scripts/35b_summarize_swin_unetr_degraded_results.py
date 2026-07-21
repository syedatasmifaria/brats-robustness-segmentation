#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/xfh25/brats_segmentation_project")
RESULTS = ROOT / "results"
REPORT = ROOT / "report_materials"

CLEAN_FILE = RESULTS / "34b_swin_unetr_clean_test_metrics.csv"

ARTIFACTS = [
    "blur",
    "ghosting",
    "noise",
    "contrast",
    "ringing",
]

INPUT_FILES = {
    artifact: RESULTS / f"35a_swin_unetr_{artifact}_patient_metrics.csv"
    for artifact in ARTIFACTS
}

OUTPUT_ALL = RESULTS / "35b_swin_unetr_all_degraded_patient_metrics.csv"
OUTPUT_SUMMARY = REPORT / "35b_swin_unetr_degraded_condition_summary.csv"
OUTPUT_TEXT = REPORT / "35b_swin_unetr_degraded_summary.txt"

METRICS = [
    "dice_WT",
    "iou_WT",
    "dice_TC",
    "iou_TC",
    "dice_ET",
    "iou_ET",
]

REPORT_NAMES = {
    "blur": "Blur",
    "ghosting": "Ghosting",
    "noise": "Gaussian noise",
    "contrast": "Contrast reduction",
    "ringing": "Fourier truncation / frequency-domain ringing-like degradation",
}


def main():
    print("=" * 80)
    print("Script 35B: Summarize Swin-UNETR degraded results")
    print("=" * 80)

    required_files = [CLEAN_FILE, *INPUT_FILES.values()]
    missing = [str(path) for path in required_files if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing input files:\n" + "\n".join(missing)
        )

    for output in [OUTPUT_ALL, OUTPUT_SUMMARY, OUTPUT_TEXT]:
        if output.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing output: {output}"
            )

    clean = pd.read_csv(CLEAN_FILE)

    if len(clean) != 74 or clean["patient_id"].nunique() != 74:
        raise RuntimeError(
            "Clean metrics must contain 74 unique patients."
        )

    clean_columns = [
        "patient_id",
        *METRICS,
        "pred_WT_voxels",
        "true_WT_voxels",
    ]

    missing_clean = [
        column
        for column in clean_columns
        if column not in clean.columns
    ]

    if missing_clean:
        raise ValueError(
            f"Clean metrics missing columns: {missing_clean}"
        )

    clean = clean[clean_columns].copy()

    clean = clean.rename(
        columns={
            column: f"clean_{column}"
            for column in clean_columns
            if column != "patient_id"
        }
    )

    degraded_frames = []

    for artifact in ARTIFACTS:
        frame = pd.read_csv(INPUT_FILES[artifact])

        if len(frame) != 740:
            raise RuntimeError(
                f"{artifact}: expected 740 rows, found {len(frame)}."
            )

        if frame["patient_id"].nunique() != 74:
            raise RuntimeError(
                f"{artifact}: expected 74 patients."
            )

        if frame["condition"].nunique() != 10:
            raise RuntimeError(
                f"{artifact}: expected 10 conditions."
            )

        if frame.duplicated(
            ["patient_id", "condition"]
        ).any():
            raise RuntimeError(
                f"{artifact}: duplicate patient-condition rows."
            )

        if frame[METRICS].isna().any().any():
            raise RuntimeError(
                f"{artifact}: missing Dice or IoU values."
            )

        degraded_frames.append(frame)

    degraded = pd.concat(
        degraded_frames,
        ignore_index=True,
    )

    if len(degraded) != 3700:
        raise RuntimeError(
            f"Expected 3,700 rows, found {len(degraded)}."
        )

    if degraded["condition"].nunique() != 50:
        raise RuntimeError(
            "Expected 50 unique conditions."
        )

    merged = degraded.merge(
        clean,
        on="patient_id",
        how="left",
        validate="many_to_one",
    )

    for metric in METRICS:
        merged[f"{metric}_drop"] = (
            merged[f"clean_{metric}"] - merged[metric]
        )

    merged["pred_WT_voxels_change"] = (
        merged["pred_WT_voxels"]
        - merged["clean_pred_WT_voxels"]
    )

    merged["report_artifact"] = merged["artifact"].map(
        REPORT_NAMES
    )

    merged["artifact_order"] = pd.Categorical(
        merged["artifact"],
        categories=ARTIFACTS,
        ordered=True,
    )

    merged = merged.sort_values(
        ["artifact_order", "level", "patient_id"]
    ).drop(
        columns="artifact_order"
    ).reset_index(drop=True)

    summary_rows = []

    grouping = [
        "artifact",
        "report_artifact",
        "level",
        "condition",
        "parameters",
        "degradation_pipeline",
    ]

    for keys, group in merged.groupby(
        grouping,
        sort=False,
    ):
        (
            artifact,
            report_artifact,
            level,
            condition,
            parameters,
            degradation_pipeline,
        ) = keys

        row = {
            "artifact": artifact,
            "report_artifact": report_artifact,
            "level": int(level),
            "condition": condition,
            "parameters": parameters,
            "degradation_pipeline": degradation_pipeline,
            "num_patients": len(group),
        }

        for metric in METRICS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_sd"] = group[metric].std(ddof=1)
            row[f"{metric}_drop_mean"] = group[
                f"{metric}_drop"
            ].mean()
            row[f"{metric}_drop_sd"] = group[
                f"{metric}_drop"
            ].std(ddof=1)

        row["pred_WT_voxels_mean"] = group[
            "pred_WT_voxels"
        ].mean()

        row["true_WT_voxels_mean"] = group[
            "true_WT_voxels"
        ].mean()

        row["pred_WT_voxels_change_mean"] = group[
            "pred_WT_voxels_change"
        ].mean()

        row["inference_seconds_mean"] = group[
            "inference_seconds"
        ].mean()

        row["degradation_seconds_mean"] = group[
            "degradation_seconds"
        ].mean()

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    summary["artifact_order"] = pd.Categorical(
        summary["artifact"],
        categories=ARTIFACTS,
        ordered=True,
    )

    summary = summary.sort_values(
        ["artifact_order", "level"]
    ).drop(
        columns="artifact_order"
    ).reset_index(drop=True)

    if len(summary) != 50:
        raise RuntimeError(
            f"Expected 50 summary rows, found {len(summary)}."
        )

    if not (summary["num_patients"] == 74).all():
        raise RuntimeError(
            "Not every condition contains 74 patients."
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)

    merged.to_csv(
        OUTPUT_ALL,
        index=False,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    text = [
        "=" * 88,
        "Script 35B: Swin-UNETR degraded evaluation summary",
        "=" * 88,
        "",
        f"Patient-condition rows: {len(merged)}",
        f"Patients: {merged['patient_id'].nunique()}",
        f"Conditions: {merged['condition'].nunique()}",
        (
            "Duplicate patient-condition rows: "
            f"{merged.duplicated(['patient_id', 'condition']).sum()}"
        ),
        (
            "Missing Dice/IoU values: "
            f"{merged[METRICS].isna().sum().sum()}"
        ),
        "",
        "Clean baseline means",
        "-" * 88,
    ]

    for metric in METRICS:
        text.append(
            f"{metric}: "
            f"{clean[f'clean_{metric}'].mean():.6f} "
            f"(SD {clean[f'clean_{metric}'].std(ddof=1):.6f})"
        )

    text.extend([
        "",
        "Condition means and paired drops",
        "Drop = clean patient metric - degraded patient metric",
        "-" * 88,
    ])

    for artifact in ARTIFACTS:
        text.extend([
            "",
            REPORT_NAMES[artifact],
            "." * 88,
        ])

        artifact_rows = summary[
            summary["artifact"] == artifact
        ]

        for _, row in artifact_rows.iterrows():
            text.append(
                f"{row['condition']} | "
                f"WT Dice {row['dice_WT_mean']:.6f} | "
                f"WT drop {row['dice_WT_drop_mean']:.6f} | "
                f"TC drop {row['dice_TC_drop_mean']:.6f} | "
                f"ET drop {row['dice_ET_drop_mean']:.6f}"
            )

    worst = summary.nlargest(
        10,
        "dice_WT_drop_mean",
    )

    text.extend([
        "",
        "Ten largest mean WT Dice drops",
        "-" * 88,
    ])

    for rank, (_, row) in enumerate(
        worst.iterrows(),
        start=1,
    ):
        text.append(
            f"{rank:02d}. {row['condition']} | "
            f"WT Dice {row['dice_WT_mean']:.6f} | "
            f"Mean paired drop {row['dice_WT_drop_mean']:.6f}"
        )

    zero_summary = (
        merged.assign(
            zero_WT=merged["dice_WT"].eq(0),
            zero_TC=merged["dice_TC"].eq(0),
            zero_ET=merged["dice_ET"].eq(0),
        )
        .groupby(
            ["artifact", "level", "condition"],
            sort=False,
        )[["zero_WT", "zero_TC", "zero_ET"]]
        .sum()
        .reset_index()
    )

    zero_summary = zero_summary[
        zero_summary[
            ["zero_WT", "zero_TC", "zero_ET"]
        ].sum(axis=1) > 0
    ]

    text.extend([
        "",
        "Conditions containing zero-Dice patients",
        "-" * 88,
    ])

    if zero_summary.empty:
        text.append("None")
    else:
        for _, row in zero_summary.iterrows():
            text.append(
                f"{row['condition']} | "
                f"WT zeros {int(row['zero_WT'])} | "
                f"TC zeros {int(row['zero_TC'])} | "
                f"ET zeros {int(row['zero_ET'])}"
            )

    text.extend([
        "",
        "Methodological notes",
        "-" * 88,
        "All models were trained on clean MRI only.",
        "Degradations were evaluation-only.",
        "Drops use paired clean and degraded patient metrics.",
        (
            "The internal label ringing represents Fourier truncation / "
            "frequency-domain ringing-like degradation."
        ),
        (
            "Contrast results must be interpreted alongside nonzero "
            "z-score normalization, which removes linear intensity scaling."
        ),
        "=" * 88,
    ])

    OUTPUT_TEXT.write_text(
        "\n".join(text) + "\n",
        encoding="utf-8",
    )

    print(f"Combined rows: {len(merged)}")
    print(f"Patients: {merged['patient_id'].nunique()}")
    print(f"Conditions: {merged['condition'].nunique()}")
    print(f"Condition summary rows: {len(summary)}")
    print(f"All-patient CSV: {OUTPUT_ALL}")
    print(f"Condition summary CSV: {OUTPUT_SUMMARY}")
    print(f"Summary TXT: {OUTPUT_TEXT}")
    print("=" * 80)


if __name__ == "__main__":
    main()
