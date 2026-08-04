#!/usr/bin/env python3
"""
Script 35A: Evaluate the fixed clean-trained Swin-UNETR checkpoint
on BraTS2020 degradation conditions generated on the fly.

Methodological rules
--------------------
- Use only the validation-selected epoch-45 checkpoint.
- Evaluate only the untouched 74-patient test cohort.
- Never retrain or select a checkpoint using degraded-test results.
- Apply degradation before Swin nonzero z-score normalization.
- Reproduce the historical nnU-Net degradation pipeline corresponding
  to each condition rather than silently harmonizing pipelines.

Historical degradation pipelines
--------------------------------
1. Script 26A pipeline:
   - All L1-L5 conditions
   - Blur L6-L10
   - Ghosting L6-L10
   - Normalize brain voxels to [0,1]
   - Apply degradation
   - Restore approximate original intensity range

2. Script 29A pipeline:
   - Noise L6-L10
   - Contrast L6-L10
   - Fourier truncation/ringing-like L6-L10
   - Normalize brain voxels to [0,1]
   - Apply degradation
   - Keep degraded volume in [0,1]
   - Contrast is compressed around the brain-intensity mean
   - Noise follows the original global np.random.seed(2029) sequence

The Fourier condition should be reported as frequency-domain
ringing-like degradation or Fourier truncation / low-pass frequency
stress test, not as pure classic Gibbs ringing.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR


PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "34a_train_swin_unetr_clean.py"
CLEAN_EVAL_SCRIPT = (
    PROJECT_ROOT / "scripts" / "34b_evaluate_swin_unetr_clean_test.py"
)
SCRIPT26A_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "26a_prepare_nnunet_degraded_final_full.py"
)
SCRIPT29A_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "29a_prepare_nnunet_extended_full_selected.py"
)

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "swin_unetr"
    / "swin_unetr_full_timing_20260719"
    / "best_checkpoint.pth"
)

TRAIN_CSV = PROJECT_ROOT / "data" / "csvs" / "swin_unetr_train_paths.csv"
VAL_CSV = PROJECT_ROOT / "data" / "csvs" / "swin_unetr_val_paths.csv"
TEST_CSV = PROJECT_ROOT / "data" / "csvs" / "test_paths.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_DIR = PROJECT_ROOT / "report_materials"

MODALITIES = ["flair", "t1", "t1ce", "t2"]
ARTIFACT_ORDER = ["blur", "ghosting", "noise", "contrast", "ringing"]

LEVELS_1_TO_5 = {
    "blur": {
        1: {"sigma": 0.5},
        2: {"sigma": 1.0},
        3: {"sigma": 1.5},
        4: {"sigma": 2.0},
        5: {"sigma": 2.5},
    },
    "ghosting": {
        1: {"num_repeats": 4, "intensity": 0.15},
        2: {"num_repeats": 8, "intensity": 0.25},
        3: {"num_repeats": 12, "intensity": 0.35},
        4: {"num_repeats": 16, "intensity": 0.45},
        5: {"num_repeats": 20, "intensity": 0.55},
    },
    "noise": {
        1: {"std": 0.02},
        2: {"std": 0.04},
        3: {"std": 0.06},
        4: {"std": 0.08},
        5: {"std": 0.10},
    },
    "contrast": {
        1: {"factor": 0.90},
        2: {"factor": 0.75},
        3: {"factor": 0.60},
        4: {"factor": 0.45},
        5: {"factor": 0.30},
    },
    "ringing": {
        1: {"keep_fraction": 0.85},
        2: {"keep_fraction": 0.75},
        3: {"keep_fraction": 0.65},
        4: {"keep_fraction": 0.55},
        5: {"keep_fraction": 0.45},
    },
}

LEVELS_6_TO_10 = {
    "blur": {
        6: {"sigma": 3.0},
        7: {"sigma": 3.5},
        8: {"sigma": 4.0},
        9: {"sigma": 4.5},
        10: {"sigma": 5.0},
    },
    "ghosting": {
        6: {"num_repeats": 24, "intensity": 0.65},
        7: {"num_repeats": 28, "intensity": 0.75},
        8: {"num_repeats": 32, "intensity": 0.85},
        9: {"num_repeats": 36, "intensity": 0.90},
        10: {"num_repeats": 40, "intensity": 0.95},
    },
    "noise": {
        6: {"std": 0.15},
        7: {"std": 0.20},
        8: {"std": 0.30},
        9: {"std": 0.40},
        10: {"std": 0.50},
    },
    "contrast": {
        6: {"factor": 0.20},
        7: {"factor": 0.15},
        8: {"factor": 0.10},
        9: {"factor": 0.05},
        10: {"factor": 0.02},
    },
    "ringing": {
        6: {"keep_fraction": 0.35},
        7: {"keep_fraction": 0.25},
        8: {"keep_fraction": 0.15},
        9: {"keep_fraction": 0.10},
        10: {"keep_fraction": 0.05},
    },
}

ALL_LEVELS = {
    artifact: {
        **LEVELS_1_TO_5[artifact],
        **LEVELS_6_TO_10[artifact],
    }
    for artifact in ARTIFACT_ORDER
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the fixed epoch-45 Swin-UNETR checkpoint "
            "on degradation conditions generated on the fly."
        )
    )

    parser.add_argument(
        "--artifact",
        choices=ARTIFACT_ORDER,
        required=True,
    )
    parser.add_argument(
        "--min-level",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-level",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--sw-batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--sw-overlap",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default=None,
        help=(
            "Optional suffix for smoke-test outputs. "
            "Do not use for finalized full runs."
        ),
    )

    args = parser.parse_args()

    if not 1 <= args.min_level <= 10:
        raise ValueError("--min-level must be between 1 and 10.")

    if not 1 <= args.max_level <= 10:
        raise ValueError("--max-level must be between 1 and 10.")

    if args.min_level > args.max_level:
        raise ValueError("--min-level cannot exceed --max-level.")

    if args.max_patients is not None and args.max_patients < 1:
        raise ValueError("--max-patients must be at least 1.")

    if args.sw_batch_size < 1:
        raise ValueError("--sw-batch-size must be at least 1.")

    if not 0.0 <= args.sw_overlap < 1.0:
        raise ValueError("--sw-overlap must be in [0,1).")

    return args


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import script: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def output_paths(
    artifact: str,
    output_tag: str | None,
) -> tuple[Path, Path, Path]:
    suffix = f"_{output_tag}" if output_tag else ""

    metrics_path = (
        RESULTS_DIR
        / f"35a_swin_unetr_{artifact}_patient_metrics{suffix}.csv"
    )
    summary_path = (
        REPORT_DIR
        / f"35a_swin_unetr_{artifact}_summary{suffix}.txt"
    )
    configuration_path = (
        REPORT_DIR
        / f"35a_swin_unetr_{artifact}_configuration{suffix}.json"
    )

    return metrics_path, summary_path, configuration_path


def load_volume(path: str | Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def pipeline_name(artifact: str, level: int) -> str:
    if level <= 5:
        return "script26a_restored_original_range"

    if artifact in {"blur", "ghosting"}:
        return "script26a_restored_original_range"

    return "script29a_saved_normalized_01"


def use_script29a_pipeline(
    artifact: str,
    level: int,
) -> bool:
    return (
        level >= 6
        and artifact in {"noise", "contrast", "ringing"}
    )


def degrade_script26a(
    original_volume: np.ndarray,
    patient_id: str,
    modality: str,
    artifact: str,
    level: int,
    params: dict[str, Any],
    script26a,
) -> np.ndarray:
    (
        volume_01,
        brain_mask,
        v_min,
        v_max,
    ) = script26a.normalize_for_degradation(
        original_volume
    )

    seed = script26a.stable_seed(
        patient_id,
        modality,
        artifact,
        level,
    )
    rng = np.random.default_rng(seed)

    degraded_01 = script26a.apply_degradation(
        volume_01=volume_01,
        brain_mask=brain_mask,
        artifact=artifact,
        params=params,
        rng=rng,
    )

    degraded = script26a.restore_original_range(
        volume_01=degraded_01,
        brain_mask=brain_mask,
        v_min=v_min,
        v_max=v_max,
        original_volume=original_volume,
    )

    return degraded.astype(np.float32)


def degrade_script29a_nonnoise(
    original_volume: np.ndarray,
    params: dict[str, Any],
    script29a,
) -> np.ndarray:
    volume_01, brain_mask = script29a.normalize_01(
        original_volume
    )

    degraded = script29a.apply_degradation(
        volume_01,
        brain_mask,
        params,
    )

    return degraded.astype(np.float32)


def prepare_script29a_noise_cache(
    full_test_df: pd.DataFrame,
    selected_patient_ids: set[str],
    selected_levels: set[int],
    script29a,
) -> dict[tuple[str, int, str], np.ndarray]:
    """
    Reproduce Script 29A's global RNG sequence exactly.

    Script 29A order:
        patient -> condition -> modality

    Contrast and ringing consume no random numbers. Therefore, exact noise
    reproduction requires processing all patients preceding a selected patient
    and all noise levels in their original order, even during a smoke test.
    Only requested arrays are retained in memory.
    """
    cache: dict[tuple[str, int, str], np.ndarray] = {}

    expected = (
        len(selected_patient_ids)
        * len(selected_levels)
        * len(MODALITIES)
    )

    np.random.seed(2029)

    original_noise_levels = [6, 7, 8, 9, 10]

    for _, row in full_test_df.iterrows():
        patient_id = str(row["patient_id"])

        for level in original_noise_levels:
            params = {
                "artifact": "noise",
                "level": level,
                **LEVELS_6_TO_10["noise"][level],
            }

            for modality in MODALITIES:
                original_volume = load_volume(row[modality])
                volume_01, brain_mask = script29a.normalize_01(
                    original_volume
                )

                degraded = script29a.apply_degradation(
                    volume_01,
                    brain_mask,
                    params,
                ).astype(np.float32)

                if (
                    patient_id in selected_patient_ids
                    and level in selected_levels
                ):
                    cache[
                        (patient_id, level, modality)
                    ] = degraded

                    if len(cache) == expected:
                        return cache

    if len(cache) != expected:
        raise RuntimeError(
            "Script 29A noise cache mismatch: "
            f"expected {expected}, found {len(cache)}."
        )

    return cache


def build_degraded_patient(
    row: pd.Series,
    artifact: str,
    level: int,
    params: dict[str, Any],
    training_module,
    script26a,
    script29a,
    script29a_noise_cache: dict[
        tuple[str, int, str],
        np.ndarray
    ] | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    patient_id = str(row["patient_id"])
    modalities: list[np.ndarray] = []

    degradation_start = time.perf_counter()

    for modality in MODALITIES:
        original_volume = load_volume(row[modality])

        if use_script29a_pipeline(artifact, level):
            if artifact == "noise":
                if script29a_noise_cache is None:
                    raise RuntimeError(
                        "Script 29A noise cache was not prepared."
                    )

                degraded = script29a_noise_cache[
                    (patient_id, level, modality)
                ]
            else:
                condition_params = {
                    "artifact": artifact,
                    "level": level,
                    **params,
                }

                degraded = degrade_script29a_nonnoise(
                    original_volume=original_volume,
                    params=condition_params,
                    script29a=script29a,
                )
        else:
            degraded = degrade_script26a(
                original_volume=original_volume,
                patient_id=patient_id,
                modality=modality,
                artifact=artifact,
                level=level,
                params=params,
                script26a=script26a,
            )

        normalized = training_module.normalize_nonzero(
            degraded
        )
        modalities.append(normalized)

    degradation_seconds = (
        time.perf_counter()
        - degradation_start
    )

    image = np.stack(
        modalities,
        axis=0,
    ).astype(np.float32)

    truth = training_module.remap_segmentation(
        training_module.load_nifti_float(
            row["seg"]
        )
    ).astype(np.int16)

    return image, truth, degradation_seconds


def write_summary(
    path: Path,
    metrics_df: pd.DataFrame,
    artifact: str,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    total_seconds: float,
    sw_batch_size: int,
    sw_overlap: float,
    amp_enabled: bool,
) -> None:
    lines = [
        "=" * 80,
        f"Script 35A: Swin-UNETR {artifact} degraded evaluation",
        "=" * 80,
        "",
        f"Artifact: {artifact}",
        (
            "Reporting label: frequency-domain ringing-like degradation"
            if artifact == "ringing"
            else f"Reporting label: {artifact}"
        ),
        f"Checkpoint: {checkpoint_path}",
        f"Checkpoint epoch: {checkpoint['epoch']}",
        f"Best validation epoch: {checkpoint['best_epoch']}",
        (
            "Best validation macro Dice: "
            f"{checkpoint['best_val_macro_dice']:.6f}"
        ),
        "Clean-trained model: Yes",
        "Degradation used for training: No",
        "Test cohort used for checkpoint selection: No",
        "Full-volume sliding-window inference: Yes",
        "ROI size: 96 x 96 x 96",
        f"Sliding-window batch size: {sw_batch_size}",
        f"Sliding-window overlap: {sw_overlap}",
        "Blend mode: gaussian",
        "Output stitching device: CPU",
        f"AMP enabled: {amp_enabled}",
        "",
        f"Rows: {len(metrics_df)}",
        f"Unique patients: {metrics_df['patient_id'].nunique()}",
        f"Conditions: {metrics_df['condition'].nunique()}",
        "",
        "Condition means",
        "-" * 80,
    ]

    for condition, group in metrics_df.groupby(
        "condition",
        sort=False,
    ):
        lines.extend([
            (
                f"{condition} | "
                f"WT Dice {group['dice_WT'].mean():.6f} | "
                f"TC Dice {group['dice_TC'].mean():.6f} | "
                f"ET Dice {group['dice_ET'].mean():.6f} | "
                f"WT IoU {group['iou_WT'].mean():.6f}"
            )
        ])

    lines.extend([
        "",
        f"Total evaluation time: {total_seconds / 60:.2f} minutes",
        (
            "Mean inference time per patient-condition: "
            f"{metrics_df['inference_seconds'].mean():.2f} seconds"
        ),
        (
            "Mean degradation/preprocessing time per patient-condition: "
            f"{metrics_df['degradation_seconds'].mean():.2f} seconds"
        ),
        "",
        "Pipeline note:",
        (
            "L1-L5 use Script 26A normalization, degradation, and restoration."
        ),
        (
            "Blur and ghosting L6-L10 also use the Script 26A restored-range "
            "pipeline."
        ),
        (
            "Noise, contrast, and ringing L6-L10 use Script 29A's saved-[0,1] "
            "pipeline."
        ),
        (
            "Script 29A contrast is centered on the brain-intensity mean."
        ),
        (
            "Script 29A noise uses its original global seed and loop sequence."
        ),
        "=" * 80,
    ])

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device(args.device)

    if device.type != "cuda":
        raise ValueError(
            "This evaluation requires a CUDA device."
        )

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    (
        output_metrics,
        output_summary,
        output_configuration,
    ) = output_paths(
        args.artifact,
        args.output_tag,
    )

    for output_path in [
        output_metrics,
        output_summary,
        output_configuration,
    ]:
        if output_path.exists():
            raise FileExistsError(
                "Output already exists. Refusing to overwrite:\n"
                f"{output_path}"
            )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_module = load_module(
        "swin_training_34a_for_35a",
        TRAIN_SCRIPT,
    )
    clean_eval_module = load_module(
        "swin_clean_eval_34b_for_35a",
        CLEAN_EVAL_SCRIPT,
    )
    script26a = load_module(
        "nnunet_degradation_26a_for_35a",
        SCRIPT26A_PATH,
    )
    script29a = load_module(
        "nnunet_degradation_29a_for_35a",
        SCRIPT29A_PATH,
    )

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    full_test_df = pd.read_csv(TEST_CSV)

    clean_eval_module.verify_split_integrity(
        train_df,
        val_df,
        full_test_df,
    )
    clean_eval_module.verify_test_paths(
        full_test_df,
        training_module.REQUIRED_COLUMNS,
    )

    test_df = full_test_df.copy()

    if args.max_patients is not None:
        test_df = test_df.head(
            args.max_patients
        ).copy()

    selected_levels = list(
        range(
            args.min_level,
            args.max_level + 1,
        )
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    required_checkpoint_keys = [
        "epoch",
        "best_epoch",
        "best_val_macro_dice",
        "model_state_dict",
        "model_configuration",
    ]

    missing_checkpoint_keys = [
        key
        for key in required_checkpoint_keys
        if key not in checkpoint
    ]

    if missing_checkpoint_keys:
        raise KeyError(
            "Checkpoint is missing keys: "
            f"{missing_checkpoint_keys}"
        )

    if int(checkpoint["epoch"]) != 45:
        raise RuntimeError(
            "Expected validation-selected epoch-45 checkpoint, "
            f"found epoch {checkpoint['epoch']}."
        )

    model_configuration = checkpoint[
        "model_configuration"
    ]

    expected_configuration = {
        "feature_size": 24,
        "in_channels": 4,
        "out_channels": 4,
        "patch_size": (96, 96, 96),
        "use_checkpoint": False,
    }

    for key, expected_value in expected_configuration.items():
        stored_value = model_configuration.get(key)

        if key == "patch_size":
            stored_value = tuple(stored_value)

        if stored_value != expected_value:
            raise RuntimeError(
                f"Checkpoint mismatch for {key}: "
                f"expected {expected_value}, found {stored_value}."
            )

    amp_enabled = not args.no_amp

    model = SwinUNETR(
        in_channels=int(
            model_configuration["in_channels"]
        ),
        out_channels=int(
            model_configuration["out_channels"]
        ),
        feature_size=int(
            model_configuration["feature_size"]
        ),
        use_checkpoint=bool(
            model_configuration["use_checkpoint"]
        ),
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    script29a_noise_cache = None

    exact_script29a_noise = (
        args.artifact == "noise"
        and any(level >= 6 for level in selected_levels)
    )

    if exact_script29a_noise:
        print(
            "Using patient-streamed Script 29A global-RNG noise generation."
        )
        np.random.seed(2029)

    configuration = {
        "artifact": args.artifact,
        "levels": selected_levels,
        "max_patients": args.max_patients,
        "checkpoint": str(
            args.checkpoint.resolve()
        ),
        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),
        "best_epoch": int(
            checkpoint["best_epoch"]
        ),
        "best_validation_macro_dice": float(
            checkpoint["best_val_macro_dice"]
        ),
        "test_csv": str(TEST_CSV),
        "full_test_patients": len(full_test_df),
        "evaluated_patients": len(test_df),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "amp_enabled": amp_enabled,
        "roi_size": [96, 96, 96],
        "sw_batch_size": args.sw_batch_size,
        "sw_overlap": args.sw_overlap,
        "blend_mode": "gaussian",
        "output_stitching_device": "cpu",
        "pipeline_by_level": {
            str(level): pipeline_name(
                args.artifact,
                level,
            )
            for level in selected_levels
        },
        "parameters": {
            str(level): ALL_LEVELS[
                args.artifact
            ][level]
            for level in selected_levels
        },
        "script29a_noise_sequence_exact": (
            args.artifact == "noise"
            and any(
                level >= 6
                for level in selected_levels
            )
        ),
    }

    output_configuration.write_text(
        json.dumps(
            configuration,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    def predictor(
        window: torch.Tensor,
    ) -> torch.Tensor:
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            return model(window)

    print("=" * 80)
    print("Script 35A: Swin-UNETR degraded evaluation")
    print("=" * 80)
    print(f"Artifact: {args.artifact}")
    print(f"Levels: {selected_levels}")
    print(f"Patients: {len(test_df)}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print("ROI size: (96, 96, 96)")
    print(f"Sliding-window overlap: {args.sw_overlap}")
    print(f"AMP enabled: {amp_enabled}")
    print("Output stitching: CPU")
    print("=" * 80)

    rows: list[dict[str, Any]] = []
    evaluation_start = time.perf_counter()

    for patient_number, (_, row) in enumerate(
        test_df.iterrows(),
        start=1,
    ):
        patient_id = str(row["patient_id"])

        truth = training_module.remap_segmentation(
            training_module.load_nifti_float(
                row["seg"]
            )
        ).astype(np.int16)

        patient_noise_images: dict[int, np.ndarray] = {}
        patient_noise_times: dict[int, float] = {}

        if exact_script29a_noise:
            # Reproduce Script 29A exactly:
            # patient -> noise L6-L10 -> modality.
            # All five levels must consume RNG values even when only a
            # subset is requested.
            for original_level in [6, 7, 8, 9, 10]:
                level_start = time.perf_counter()
                normalized_modalities: list[np.ndarray] = []

                condition_params = {
                    "artifact": "noise",
                    "level": original_level,
                    **LEVELS_6_TO_10["noise"][original_level],
                }

                for modality in MODALITIES:
                    original_volume = load_volume(
                        row[modality]
                    )

                    volume_01, brain_mask = (
                        script29a.normalize_01(
                            original_volume
                        )
                    )

                    degraded = script29a.apply_degradation(
                        volume_01,
                        brain_mask,
                        condition_params,
                    ).astype(np.float32)

                    if original_level in selected_levels:
                        normalized_modalities.append(
                            training_module.normalize_nonzero(
                                degraded
                            )
                        )

                if original_level in selected_levels:
                    patient_noise_images[original_level] = (
                        np.stack(
                            normalized_modalities,
                            axis=0,
                        ).astype(np.float32)
                    )

                    patient_noise_times[original_level] = (
                        time.perf_counter()
                        - level_start
                    )

        for level in selected_levels:
            params = ALL_LEVELS[
                args.artifact
            ][level]
            condition = f"{args.artifact}_L{level}"

            print("\n" + "-" * 80)
            print(
                f"Condition: {condition} | "
                f"params: {params} | "
                f"pipeline: {pipeline_name(args.artifact, level)}"
            )
            print("-" * 80)

            if (
                args.artifact == "noise"
                and level >= 6
            ):
                image = patient_noise_images[level]
                degradation_seconds = (
                    patient_noise_times[level]
                )
            else:
                (
                    image,
                    loaded_truth,
                    degradation_seconds,
                ) = build_degraded_patient(
                    row=row,
                    artifact=args.artifact,
                    level=level,
                    params=params,
                    training_module=training_module,
                    script26a=script26a,
                    script29a=script29a,
                    script29a_noise_cache=None,
                )

                if not np.array_equal(
                    loaded_truth,
                    truth,
                ):
                    raise RuntimeError(
                        f"Truth mismatch for {patient_id}."
                    )

            input_tensor = torch.from_numpy(
                image
            ).unsqueeze(0).float()

            inference_start = time.perf_counter()

            logits = sliding_window_inference(
                inputs=input_tensor,
                roi_size=(96, 96, 96),
                sw_batch_size=args.sw_batch_size,
                predictor=predictor,
                overlap=args.sw_overlap,
                mode="gaussian",
                sigma_scale=0.125,
                padding_mode="constant",
                cval=0.0,
                sw_device=device,
                device=torch.device("cpu"),
                progress=False,
            )

            inference_seconds = (
                time.perf_counter()
                - inference_start
            )

            prediction = torch.argmax(
                logits,
                dim=1,
            ).squeeze(0).numpy().astype(
                np.int16
            )

            if prediction.shape != truth.shape:
                raise RuntimeError(
                    f"Prediction shape {prediction.shape} "
                    f"does not match truth shape {truth.shape} "
                    f"for {patient_id}."
                )

            metrics = training_module.compute_metrics(
                prediction,
                truth,
            )

            rows.append({
                "patient_id": patient_id,
                "artifact": args.artifact,
                "level": level,
                "condition": condition,
                "parameters": json.dumps(
                    params,
                    sort_keys=True,
                ),
                "degradation_pipeline": pipeline_name(
                    args.artifact,
                    level,
                ),
                **metrics,
                "degradation_seconds": degradation_seconds,
                "inference_seconds": inference_seconds,
            })

            print(
                f"[{patient_number:02d}/{len(test_df):02d}] "
                f"{patient_id} | "
                f"{condition} | "
                f"WT={metrics['dice_WT']:.4f} | "
                f"TC={metrics['dice_TC']:.4f} | "
                f"ET={metrics['dice_ET']:.4f} | "
                f"degrade={degradation_seconds:.1f}s | "
                f"infer={inference_seconds:.1f}s"
            )

            del input_tensor
            del logits
            del prediction
            torch.cuda.empty_cache()

        patient_noise_images.clear()
        patient_noise_times.clear()

    metrics_df = pd.DataFrame(rows)

    expected_rows = (
        len(test_df)
        * len(selected_levels)
    )

    if len(metrics_df) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows, "
            f"found {len(metrics_df)}."
        )

    if metrics_df.duplicated(
        subset=["patient_id", "condition"]
    ).any():
        raise RuntimeError(
            "Duplicate patient-condition rows detected."
        )

    required_metric_columns = [
        "dice_WT",
        "iou_WT",
        "dice_TC",
        "iou_TC",
        "dice_ET",
        "iou_ET",
        "pred_WT_voxels",
        "true_WT_voxels",
    ]

    if metrics_df[
        required_metric_columns
    ].isna().any().any():
        raise RuntimeError(
            "Missing metric values detected."
        )

    metrics_df.to_csv(
        output_metrics,
        index=False,
    )

    total_seconds = (
        time.perf_counter()
        - evaluation_start
    )

    write_summary(
        path=output_summary,
        metrics_df=metrics_df,
        artifact=args.artifact,
        checkpoint_path=args.checkpoint,
        checkpoint=checkpoint,
        total_seconds=total_seconds,
        sw_batch_size=args.sw_batch_size,
        sw_overlap=args.sw_overlap,
        amp_enabled=amp_enabled,
    )

    print("\n" + "=" * 80)
    print("Degraded evaluation completed")
    print("=" * 80)
    print(f"Rows: {len(metrics_df)}")
    print(
        f"Conditions: {metrics_df['condition'].nunique()}"
    )
    print(
        f"Patients: {metrics_df['patient_id'].nunique()}"
    )
    print(
        f"Total time: {total_seconds / 60:.2f} minutes"
    )
    print(f"Metrics: {output_metrics}")
    print(f"Summary: {output_summary}")
    print(f"Configuration: {output_configuration}")
    print("=" * 80)


if __name__ == "__main__":
    main()
