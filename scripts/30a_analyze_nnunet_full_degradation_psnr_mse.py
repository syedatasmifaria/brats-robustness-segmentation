#!/usr/bin/env python3
"""
Script 30A: Full-cohort nnU-Net degradation strength analysis.

Purpose
-------
Compute image-level degradation strength using MSE and PSNR for the same
74 held-out BraTS2020 patients used in the full-volume nnU-Net evaluation.

Conditions
----------
Original full test:
- blur L1-L5
- ghosting L1-L5
- noise L1-L5
- contrast L1-L5
- ringing L1-L5

Extended full test:
- noise L6-L10
- contrast L6-L10
- ringing / frequency-domain truncation L6-L10

Important storage difference
----------------------------
The original L1-L5 degraded files are stored in a different intensity format
from the extended L6-L10 files.

For L1-L5:
- degraded files are normalized using the corresponding clean MRI range.

For L6-L10:
- Script 29A already saved degraded files directly in [0,1].
- These files must NOT be normalized a second time.

Interpretation
--------------
Higher MSE = stronger image-level change.
Lower PSNR = stronger image-level change.

Terminology
-----------
The condition named "ringing" uses Fourier truncation and behaves partly like
low-pass filtering at severe levels. Report it as frequency-domain truncation
or ringing-like frequency-domain degradation, not pure classic Gibbs ringing.
"""

from pathlib import Path
import math
import re

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

TEST_CSV = PROJECT_ROOT / "data/csvs/test_paths.csv"

ORIGINAL_ROOT = (
    PROJECT_ROOT
    / "nnunet/temporary_degraded_tests/final_full"
)

EXTENDED_ROOT = (
    PROJECT_ROOT
    / "nnunet/temporary_degraded_tests/extended_full_selected"
)

RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_DIR = PROJECT_ROOT / "report_materials"

OUT_METRICS_CSV = (
    RESULTS_DIR
    / "30a_nnunet_full_degradation_psnr_mse_metrics.csv"
)

OUT_SUMMARY_CSV = (
    REPORT_DIR
    / "30a_nnunet_full_degradation_psnr_mse_summary.csv"
)

OUT_SUMMARY_TXT = (
    REPORT_DIR
    / "30a_nnunet_full_degradation_psnr_mse_summary.txt"
)

MSE_CURVE = (
    REPORT_DIR
    / "30a_nnunet_full_degradation_mse_curve.png"
)

PSNR_CURVE = (
    REPORT_DIR
    / "30a_nnunet_full_degradation_psnr_curve.png"
)


# ---------------------------------------------------------------------
# Study definitions
# ---------------------------------------------------------------------

MODALITIES = [
    ("flair", "0000"),
    ("t1", "0001"),
    ("t1ce", "0002"),
    ("t2", "0003"),
]

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

EXPECTED_PATIENTS = 74
EXPECTED_CONDITIONS = 40
EXPECTED_ROWS_PER_CONDITION = EXPECTED_PATIENTS * len(MODALITIES)


# ---------------------------------------------------------------------
# NIfTI and normalization functions
# ---------------------------------------------------------------------

def load_nifti(path: Path) -> np.ndarray:
    """Load a NIfTI file as float32."""
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def normalize_clean_volume(clean_volume: np.ndarray):
    """
    Normalize a clean MRI to [0,1] using nonzero brain voxels.

    Returns
    -------
    clean_01
    brain_mask
    v_min
    v_max
    """
    clean_volume = clean_volume.astype(np.float32)

    brain_mask = clean_volume != 0

    if brain_mask.sum() == 0:
        raise ValueError(
            "Clean volume contains no nonzero brain voxels."
        )

    brain_values = clean_volume[brain_mask]

    v_min = float(
        np.percentile(brain_values, 1)
    )

    v_max = float(
        np.percentile(brain_values, 99)
    )

    if v_max <= v_min:
        raise ValueError(
            f"Invalid clean normalization range: "
            f"v_min={v_min}, v_max={v_max}"
        )

    clean_01 = np.zeros_like(
        clean_volume,
        dtype=np.float32,
    )

    clean_01[brain_mask] = (
        clean_volume[brain_mask] - v_min
    ) / (v_max - v_min)

    clean_01 = np.clip(
        clean_01,
        0.0,
        1.0,
    )

    clean_01[~brain_mask] = 0.0

    return clean_01, brain_mask, v_min, v_max


def normalize_original_degraded_volume(
    degraded_volume: np.ndarray,
    brain_mask: np.ndarray,
    v_min: float,
    v_max: float,
):
    """
    Normalize an original L1-L5 degraded image using the clean MRI range.

    This matches the comparison logic used for the original full degradation
    dataset.
    """
    degraded_volume = degraded_volume.astype(np.float32)

    degraded_01 = np.zeros_like(
        degraded_volume,
        dtype=np.float32,
    )

    degraded_01[brain_mask] = (
        degraded_volume[brain_mask] - v_min
    ) / (v_max - v_min)

    degraded_01 = np.clip(
        degraded_01,
        0.0,
        1.0,
    )

    degraded_01[~brain_mask] = 0.0

    return degraded_01


def prepare_extended_degraded_volume(
    degraded_volume: np.ndarray,
    brain_mask: np.ndarray,
):
    """
    Prepare an extended L6-L10 degraded image.

    Script 29A already saved these images directly in [0,1].
    Therefore, they must not be normalized again.
    """
    degraded_01 = degraded_volume.astype(np.float32)

    degraded_min = float(np.nanmin(degraded_01))
    degraded_max = float(np.nanmax(degraded_01))

    if degraded_min < -1e-4 or degraded_max > 1.0001:
        raise ValueError(
            "Extended degraded image is outside expected [0,1] range: "
            f"min={degraded_min}, max={degraded_max}"
        )

    degraded_01 = np.clip(
        degraded_01,
        0.0,
        1.0,
    )

    degraded_01[~brain_mask] = 0.0

    return degraded_01


def calculate_mse_psnr(
    clean_01: np.ndarray,
    degraded_01: np.ndarray,
    brain_mask: np.ndarray,
):
    """
    Compute MSE and PSNR inside the clean brain mask.

    The normalized intensity range is 1.0.
    """
    clean_values = clean_01[brain_mask]
    degraded_values = degraded_01[brain_mask]

    difference = clean_values - degraded_values

    mse = float(
        np.mean(difference ** 2)
    )

    if mse == 0:
        psnr = float("inf")
    else:
        psnr = float(
            20.0
            * math.log10(
                1.0 / math.sqrt(mse)
            )
        )

    return mse, psnr


# ---------------------------------------------------------------------
# Condition and filename utilities
# ---------------------------------------------------------------------

def parse_condition(condition: str):
    """
    Parse a condition such as noise_L7.

    Returns
    -------
    artifact
    level
    """
    match = re.fullmatch(
        r"(.+)_L(\d+)",
        condition,
    )

    if match is None:
        raise ValueError(
            f"Could not parse condition: {condition}"
        )

    artifact = match.group(1)
    level = int(match.group(2))

    return artifact, level


def extract_case_id(patient_id: str):
    """
    Extract the final three-digit BraTS case number.

    Example
    -------
    BraTS20_Training_001 -> 001
    """
    match = re.search(
        r"(\d{3})$",
        patient_id,
    )

    if match is None:
        raise ValueError(
            f"Could not extract case number from: {patient_id}"
        )

    return match.group(1)


def resolve_degraded_path(
    images_dir: Path,
    patient_id: str,
    channel_id: str,
):
    """
    Support both degraded filename conventions.

    Original files may use:
        BraTS20_Training_001_0000.nii.gz

    Extended files use:
        BRATS_001_0000.nii.gz
    """
    case_id = extract_case_id(patient_id)

    candidates = [
        images_dir
        / f"{patient_id}_{channel_id}.nii.gz",

        images_dir
        / f"BRATS_{case_id}_{channel_id}.nii.gz",

        images_dir
        / f"BraTS20_Training_{case_id}_{channel_id}.nii.gz",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried_paths = "\n".join(
        f"  {candidate}"
        for candidate in candidates
    )

    raise FileNotFoundError(
        "Could not locate degraded image.\n"
        f"Patient: {patient_id}\n"
        f"Channel: {channel_id}\n"
        f"Tried:\n{tried_paths}"
    )


def collect_condition_records():
    """
    Collect the 40 expected degradation condition directories.
    """
    root_specs = [
        ("original_L1_L5", ORIGINAL_ROOT),
        ("extended_L6_L10", EXTENDED_ROOT),
    ]

    records = []

    for experiment, root in root_specs:
        if not root.exists():
            raise FileNotFoundError(
                f"Missing degradation root: {root}"
            )

        for condition_dir in root.iterdir():
            if not condition_dir.is_dir():
                continue

            if condition_dir.name in {
                "labelsTs",
                "clean",
            }:
                continue

            if "_L" not in condition_dir.name:
                continue

            artifact, level = parse_condition(
                condition_dir.name
            )

            if artifact not in ARTIFACT_ORDER:
                print(
                    f"Skipping unexpected condition: "
                    f"{condition_dir.name}"
                )
                continue

            images_dir = condition_dir / "imagesTs"

            if not images_dir.exists():
                raise FileNotFoundError(
                    f"Missing imagesTs directory: {images_dir}"
                )

            if experiment == "original_L1_L5":
                if level < 1 or level > 5:
                    raise RuntimeError(
                        f"Unexpected original level: "
                        f"{condition_dir.name}"
                    )

            if experiment == "extended_L6_L10":
                if level < 6 or level > 10:
                    raise RuntimeError(
                        f"Unexpected extended level: "
                        f"{condition_dir.name}"
                    )

                if artifact not in {
                    "noise",
                    "contrast",
                    "ringing",
                }:
                    raise RuntimeError(
                        f"Unexpected extended artifact: "
                        f"{condition_dir.name}"
                    )

            records.append({
                "experiment": experiment,
                "condition": condition_dir.name,
                "artifact": artifact,
                "level": level,
                "images_dir": images_dir,
            })

    records = sorted(
        records,
        key=lambda item: (
            ARTIFACT_ORDER.index(
                item["artifact"]
            ),
            item["level"],
        ),
    )

    return records


# ---------------------------------------------------------------------
# Progress functions
# ---------------------------------------------------------------------

def load_existing_metrics():
    """
    Load saved progress if the script was interrupted previously.
    """
    if not OUT_METRICS_CSV.exists():
        return pd.DataFrame()

    existing_df = pd.read_csv(
        OUT_METRICS_CSV
    )

    required_columns = {
        "experiment",
        "condition",
        "artifact",
        "level",
        "patient_id",
        "modality",
        "mse",
        "psnr",
    }

    if not required_columns.issubset(
        existing_df.columns
    ):
        raise RuntimeError(
            "Existing Script 30A metrics file has an unexpected format. "
            "Delete it before rerunning."
        )

    return existing_df


def get_completed_conditions(
    existing_df: pd.DataFrame,
):
    """
    A condition is complete only when all 74 patients and 4 modalities exist.
    """
    if existing_df.empty:
        return set()

    counts = (
        existing_df
        .groupby("condition")
        .size()
    )

    return set(
        counts[
            counts == EXPECTED_ROWS_PER_CONDITION
        ].index
    )


def save_progress(
    metrics_df: pd.DataFrame,
):
    """Save detailed results after every completed condition."""
    metrics_df.to_csv(
        OUT_METRICS_CSV,
        index=False,
    )


# ---------------------------------------------------------------------
# Summary and plotting functions
# ---------------------------------------------------------------------

def create_summary(
    metrics_df: pd.DataFrame,
):
    """Create one summary row per degradation condition."""
    summary_df = (
        metrics_df
        .groupby(
            [
                "experiment",
                "artifact",
                "artifact_display",
                "level",
                "condition",
            ],
            as_index=False,
        )
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            mse_median=("mse", "median"),
            psnr_mean=("psnr", "mean"),
            psnr_std=("psnr", "std"),
            psnr_median=("psnr", "median"),
            n_measurements=("mse", "count"),
            n_patients=("patient_id", "nunique"),
        )
    )

    summary_df["artifact"] = pd.Categorical(
        summary_df["artifact"],
        categories=ARTIFACT_ORDER,
        ordered=True,
    )

    summary_df = (
        summary_df
        .sort_values(
            ["artifact", "level"]
        )
        .reset_index(drop=True)
    )

    return summary_df


def write_summary_text(
    summary_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
):
    """Write a readable text summary."""
    with open(
        OUT_SUMMARY_TXT,
        "w",
        encoding="utf-8",
    ) as file:

        file.write("=" * 88 + "\n")
        file.write(
            "Script 30A: Full-cohort nnU-Net degradation "
            "MSE/PSNR summary\n"
        )
        file.write("=" * 88 + "\n\n")

        file.write(
            f"Patients: {EXPECTED_PATIENTS}\n"
        )
        file.write(
            f"Modalities per patient: "
            f"{len(MODALITIES)}\n"
        )
        file.write(
            f"Conditions: {len(summary_df)}\n"
        )
        file.write(
            f"Detailed measurements: "
            f"{len(metrics_df)}\n\n"
        )

        file.write("Interpretation:\n")
        file.write(
            "Higher MSE indicates greater image-level change.\n"
        )
        file.write(
            "Lower PSNR indicates greater image-level change.\n"
        )
        file.write(
            "Metrics were calculated inside the clean brain mask.\n\n"
        )

        file.write("Intensity-handling note:\n")
        file.write(
            "Original L1-L5 degraded images were normalized using "
            "their corresponding clean MRI ranges.\n"
        )
        file.write(
            "Extended L6-L10 degraded images were already stored "
            "in [0,1] and were not normalized a second time.\n\n"
        )

        file.write("Terminology note:\n")
        file.write(
            "The stored artifact name 'ringing' refers to "
            "frequency-domain truncation and should not be described "
            "as pure classic Gibbs ringing.\n\n"
        )

        for _, row in summary_df.iterrows():
            file.write(
                f"{row['condition']}: "
                f"MSE mean={row['mse_mean']:.6f}, "
                f"MSE SD={row['mse_std']:.6f}, "
                f"PSNR mean={row['psnr_mean']:.2f} dB, "
                f"PSNR SD={row['psnr_std']:.2f}, "
                f"patients={int(row['n_patients'])}, "
                f"measurements={int(row['n_measurements'])}\n"
            )


def make_plots(
    summary_df: pd.DataFrame,
):
    """Create full L1-L10 MSE and PSNR severity plots."""

    plt.figure(figsize=(10, 6))

    for artifact in ARTIFACT_ORDER:
        subset = summary_df[
            summary_df["artifact"] == artifact
        ].sort_values("level")

        if subset.empty:
            continue

        plt.plot(
            subset["level"],
            subset["mse_mean"],
            marker="o",
            label=DISPLAY_NAMES[artifact],
        )

    plt.xlabel("Severity level")
    plt.ylabel("Mean MSE")
    plt.title(
        "Full-cohort degradation strength measured by MSE"
    )
    plt.xticks(range(1, 11))
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        MSE_CURVE,
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(10, 6))

    for artifact in ARTIFACT_ORDER:
        subset = summary_df[
            summary_df["artifact"] == artifact
        ].sort_values("level")

        if subset.empty:
            continue

        plt.plot(
            subset["level"],
            subset["psnr_mean"],
            marker="o",
            label=DISPLAY_NAMES[artifact],
        )

    plt.xlabel("Severity level")
    plt.ylabel("Mean PSNR (dB)")
    plt.title(
        "Full-cohort degradation strength measured by PSNR"
    )
    plt.xticks(range(1, 11))
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        PSNR_CURVE,
        dpi=300,
    )
    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("=" * 88)
    print(
        "Script 30A: Full-cohort nnU-Net "
        "degradation MSE/PSNR analysis"
    )
    print("=" * 88)

    RESULTS_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    test_df = pd.read_csv(
        TEST_CSV
    ).copy()

    if len(test_df) != EXPECTED_PATIENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_PATIENTS} test patients, "
            f"found {len(test_df)}"
        )

    condition_records = collect_condition_records()

    if len(condition_records) != EXPECTED_CONDITIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_CONDITIONS} degradation conditions, "
            f"found {len(condition_records)}"
        )

    existing_df = load_existing_metrics()

    completed_conditions = get_completed_conditions(
        existing_df
    )

    if existing_df.empty:
        all_rows = []
    else:
        all_rows = existing_df.to_dict(
            orient="records"
        )

    print(f"Test patients: {len(test_df)}")
    print(f"Conditions found: {len(condition_records)}")
    print(
        f"Expected rows per condition: "
        f"{EXPECTED_ROWS_PER_CONDITION}"
    )
    print(
        f"Previously completed conditions: "
        f"{len(completed_conditions)}"
    )
    print()

    for condition_index, record in enumerate(
        condition_records,
        start=1,
    ):
        experiment = record["experiment"]
        condition = record["condition"]
        artifact = record["artifact"]
        level = record["level"]
        images_dir = record["images_dir"]

        if condition in completed_conditions:
            print(
                f"[{condition_index:02d}/"
                f"{len(condition_records)}] "
                f"Skipping completed condition: {condition}"
            )
            continue

        print("-" * 88)
        print(
            f"[{condition_index:02d}/"
            f"{len(condition_records)}] "
            f"Analyzing {condition}"
        )

        condition_rows = []

        for patient_index, (_, row) in enumerate(
            test_df.iterrows(),
            start=1,
        ):
            patient_id = str(
                row["patient_id"]
            )

            for modality_name, channel_id in MODALITIES:
                clean_path = Path(
                    row[modality_name]
                )

                if not clean_path.exists():
                    raise FileNotFoundError(
                        f"Missing clean MRI: {clean_path}"
                    )

                degraded_path = resolve_degraded_path(
                    images_dir=images_dir,
                    patient_id=patient_id,
                    channel_id=channel_id,
                )

                clean_volume = load_nifti(
                    clean_path
                )

                degraded_volume = load_nifti(
                    degraded_path
                )

                if clean_volume.shape != degraded_volume.shape:
                    raise RuntimeError(
                        f"Shape mismatch for {condition}, "
                        f"{patient_id}, {modality_name}: "
                        f"clean={clean_volume.shape}, "
                        f"degraded={degraded_volume.shape}"
                    )

                (
                    clean_01,
                    brain_mask,
                    v_min,
                    v_max,
                ) = normalize_clean_volume(
                    clean_volume
                )

                if experiment == "extended_L6_L10":
                    degraded_01 = (
                        prepare_extended_degraded_volume(
                            degraded_volume=degraded_volume,
                            brain_mask=brain_mask,
                        )
                    )
                else:
                    degraded_01 = (
                        normalize_original_degraded_volume(
                            degraded_volume=degraded_volume,
                            brain_mask=brain_mask,
                            v_min=v_min,
                            v_max=v_max,
                        )
                    )

                mse, psnr = calculate_mse_psnr(
                    clean_01=clean_01,
                    degraded_01=degraded_01,
                    brain_mask=brain_mask,
                )

                condition_rows.append({
                    "experiment": experiment,
                    "condition": condition,
                    "artifact": artifact,
                    "artifact_display": DISPLAY_NAMES[
                        artifact
                    ],
                    "level": level,
                    "patient_id": patient_id,
                    "modality": modality_name,
                    "mse": mse,
                    "psnr": psnr,
                })

            if patient_index % 10 == 0:
                print(
                    f"  Processed "
                    f"{patient_index}/{len(test_df)} patients"
                )

        if len(condition_rows) != EXPECTED_ROWS_PER_CONDITION:
            raise RuntimeError(
                f"{condition} created "
                f"{len(condition_rows)} measurements, "
                f"expected {EXPECTED_ROWS_PER_CONDITION}"
            )

        all_rows.extend(
            condition_rows
        )

        progress_df = pd.DataFrame(
            all_rows
        )

        save_progress(
            progress_df
        )

        print(
            f"  Saved progress after {condition}"
        )

    metrics_df = pd.DataFrame(
        all_rows
    )

    expected_total_rows = (
        EXPECTED_CONDITIONS
        * EXPECTED_ROWS_PER_CONDITION
    )

    if len(metrics_df) != expected_total_rows:
        raise RuntimeError(
            f"Expected {expected_total_rows} detailed rows, "
            f"found {len(metrics_df)}"
        )

    duplicate_count = int(
        metrics_df.duplicated(
            subset=[
                "condition",
                "patient_id",
                "modality",
            ]
        ).sum()
    )

    if duplicate_count != 0:
        raise RuntimeError(
            f"Found {duplicate_count} duplicate measurements."
        )

    summary_df = create_summary(
        metrics_df
    )

    if len(summary_df) != EXPECTED_CONDITIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_CONDITIONS} summary rows, "
            f"found {len(summary_df)}"
        )

    summary_df.to_csv(
        OUT_SUMMARY_CSV,
        index=False,
    )

    write_summary_text(
        summary_df=summary_df,
        metrics_df=metrics_df,
    )

    make_plots(
        summary_df
    )

    print()
    print("=" * 88)
    print(
        "Full-cohort MSE/PSNR analysis completed"
    )
    print("=" * 88)

    print(
        f"Detailed metrics: {OUT_METRICS_CSV}"
    )
    print(
        f"Summary CSV:     {OUT_SUMMARY_CSV}"
    )
    print(
        f"Summary TXT:     {OUT_SUMMARY_TXT}"
    )
    print(
        f"MSE curve:       {MSE_CURVE}"
    )
    print(
        f"PSNR curve:      {PSNR_CURVE}"
    )
    print()
    print("Quick preview:")

    print(
        summary_df[
            [
                "condition",
                "mse_mean",
                "psnr_mean",
                "n_patients",
                "n_measurements",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
