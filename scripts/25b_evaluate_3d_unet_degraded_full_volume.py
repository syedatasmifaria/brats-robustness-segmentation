#!/usr/bin/env python3
"""
Script 25B: Full-volume degraded evaluation for the clean-trained custom 3D U-Net.

Method:
- Load the fixed clean-trained custom 3D U-Net checkpoint.
- Use the corrected Script 24B nonzero z-score preprocessing.
- Apply the same historical L1-L10 degradation pipelines used for Swin-UNETR.
- Run full-volume 96^3 sliding-window inference.
- Evaluate one artifact at a time across selected levels.
- Do not retrain.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

SCRIPT24B_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "24b_evaluate_3d_unet_full_volume_clean_corrected.py"
)

SCRIPT35A_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "35a_evaluate_swin_unetr_degraded_onthefly.py"
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

TEST_CSV = PROJECT_ROOT / "data/csvs/test_paths.csv"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "3d_unet_multimodal_clean_full_best.pth"
)

RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_DIR = PROJECT_ROOT / "report_materials"

ARTIFACT_ORDER = [
    "blur",
    "ghosting",
    "noise",
    "contrast",
    "ringing",
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import script: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate custom 3D U-Net under one degradation artifact."
    )

    parser.add_argument(
        "--artifact",
        required=True,
        choices=ARTIFACT_ORDER,
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
        "--device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--output-tag",
        type=str,
        default=None,
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

    return args


def output_paths(
    artifact: str,
    output_tag: str | None,
) -> tuple[Path, Path, Path]:
    suffix = f"_{output_tag}" if output_tag else ""

    metrics_path = (
        RESULTS_DIR
        / f"25b_3d_unet_{artifact}_full_volume_metrics{suffix}.csv"
    )

    summary_path = (
        REPORT_DIR
        / f"25b_3d_unet_{artifact}_full_volume_summary{suffix}.txt"
    )

    configuration_path = (
        REPORT_DIR
        / f"25b_3d_unet_{artifact}_configuration{suffix}.json"
    )

    return metrics_path, summary_path, configuration_path


def write_summary(
    path: Path,
    metrics_df: pd.DataFrame,
    artifact: str,
    total_seconds: float,
) -> None:
    lines = [
        "=" * 80,
        f"Script 25B: Custom 3D U-Net {artifact} full-volume evaluation",
        "=" * 80,
        "",
        "Clean-trained model: Yes",
        "Degradation used during training: No",
        "Full-volume sliding-window inference: Yes",
        "Patch size: 96 x 96 x 96",
        "Stride: 64 x 64 x 64",
        "Overlap combination: mean class probability",
        "Model preprocessing: nonzero-voxel z-score",
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
        lines.append(
            f"{condition} | "
            f"WT Dice {group['whole_tumor_dice'].mean():.6f} | "
            f"WT IoU {group['whole_tumor_iou'].mean():.6f} | "
            f"Predicted WT voxels {group['pred_tumor_voxels'].mean():.2f} | "
            f"True WT voxels {group['true_tumor_voxels'].mean():.2f}"
        )

    lines.extend([
        "",
        f"Total evaluation time: {total_seconds / 60:.2f} minutes",
        "",
        "Pipeline note:",
        "L1-L5 use the Script 26A restored-original-range degradation pipeline.",
        "Blur and ghosting L6-L10 also use the Script 26A pipeline.",
        (
            "Noise, contrast, and Fourier truncation L6-L10 use "
            "the historical Script 29A normalized-[0,1] pipeline."
        ),
        (
            "The ringing condition should be reported as Fourier truncation / "
            "low-pass frequency stress test."
        ),
        "=" * 80,
    ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device(args.device)

    if device.type != "cuda":
        raise ValueError("This evaluation requires a CUDA device.")

    for required_path in [
        SCRIPT24B_PATH,
        SCRIPT35A_PATH,
        SCRIPT26A_PATH,
        SCRIPT29A_PATH,
        TEST_CSV,
        MODEL_PATH,
    ]:
        if not required_path.exists():
            raise FileNotFoundError(required_path)

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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    script24b = load_module(
        "custom_3d_unet_clean_24b_for_25b",
        SCRIPT24B_PATH,
    )

    # Compatibility aliases expected by Script 35A's degradation helper.
    script24b.normalize_nonzero = script24b.normalize_modality
    script24b.load_nifti_float = script24b.load_nifti

    script35a = load_module(
        "swin_degradation_35a_for_25b",
        SCRIPT35A_PATH,
    )

    script26a = load_module(
        "nnunet_degradation_26a_for_25b",
        SCRIPT26A_PATH,
    )

    script29a = load_module(
        "nnunet_degradation_29a_for_25b",
        SCRIPT29A_PATH,
    )

    full_test_df = pd.read_csv(TEST_CSV)
    test_df = full_test_df.copy()

    if args.max_patients is not None:
        test_df = test_df.head(args.max_patients).copy()

    selected_levels = list(
        range(args.min_level, args.max_level + 1)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = script24b.UNet3D(
        in_channels=script24b.IN_CHANNELS,
        num_classes=script24b.NUM_CLASSES,
        base_channels=script24b.BASE_CHANNELS,
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    script24b.DEVICE = device

    selected_patient_ids = set(
        test_df["patient_id"].astype(str)
    )

    selected_noise_levels = {
        level
        for level in selected_levels
        if args.artifact == "noise" and level >= 6
    }

    script29a_noise_cache = None

    if selected_noise_levels:
        print(
            "Preparing exact Script 29A L6-L10 noise cache "
            "for selected patients and levels."
        )

        script29a_noise_cache = (
            script35a.prepare_script29a_noise_cache(
                full_test_df=full_test_df,
                selected_patient_ids=selected_patient_ids,
                selected_levels=selected_noise_levels,
                script29a=script29a,
            )
        )

    configuration = {
        "script": "25b_evaluate_3d_unet_degraded_full_volume.py",
        "artifact": args.artifact,
        "levels": selected_levels,
        "max_patients": args.max_patients,
        "model_path": str(MODEL_PATH),
        "test_csv": str(TEST_CSV),
        "full_test_patients": len(full_test_df),
        "evaluated_patients": len(test_df),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "patch_size": list(script24b.PATCH_SIZE),
        "stride": list(script24b.STRIDE),
        "model_normalization": "nonzero_voxel_zscore",
        "pipeline_by_level": {
            str(level): script35a.pipeline_name(
                args.artifact,
                level,
            )
            for level in selected_levels
        },
        "parameters": {
            str(level): script35a.ALL_LEVELS[
                args.artifact
            ][level]
            for level in selected_levels
        },
    }

    output_configuration.write_text(
        json.dumps(configuration, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 80)
    print("Script 25B: Custom 3D U-Net degraded full-volume evaluation")
    print("=" * 80)
    print(f"Artifact: {args.artifact}")
    print(f"Levels: {selected_levels}")
    print(f"Patients: {len(test_df)}")
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Patch size: {script24b.PATCH_SIZE}")
    print(f"Stride: {script24b.STRIDE}")
    print("=" * 80)

    rows: list[dict[str, Any]] = []
    evaluation_start = time.perf_counter()

    for patient_number, (_, row) in enumerate(
        test_df.iterrows(),
        start=1,
    ):
        patient_id = str(row["patient_id"])

        truth = script24b.remap_segmentation(
            script24b.load_nifti(row["seg"])
        ).astype(np.int16)

        for level in selected_levels:
            params = script35a.ALL_LEVELS[
                args.artifact
            ][level]

            condition = f"{args.artifact}_L{level}"

            image, loaded_truth, degradation_seconds = (
                script35a.build_degraded_patient(
                    row=row,
                    artifact=args.artifact,
                    level=level,
                    params=params,
                    training_module=script24b,
                    script26a=script26a,
                    script29a=script29a,
                    script29a_noise_cache=script29a_noise_cache,
                )
            )

            if not np.array_equal(loaded_truth, truth):
                raise RuntimeError(
                    f"Truth mismatch for {patient_id}."
                )

            inference_start = time.perf_counter()

            prediction = script24b.sliding_window_predict(
                model,
                image,
            )

            inference_seconds = (
                time.perf_counter() - inference_start
            )

            metrics = script24b.compute_patient_metrics(
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
                "degradation_pipeline": (
                    script35a.pipeline_name(
                        args.artifact,
                        level,
                    )
                ),
                **metrics,
                "degradation_seconds": degradation_seconds,
                "inference_seconds": inference_seconds,
            })

            print(
                f"[{patient_number:02d}/{len(test_df):02d}] "
                f"{patient_id} | "
                f"{condition} | "
                f"WT Dice={metrics['whole_tumor_dice']:.4f} | "
                f"degrade={degradation_seconds:.2f}s | "
                f"infer={inference_seconds:.2f}s"
            )

    metrics_df = pd.DataFrame(rows)

    expected_rows = len(test_df) * len(selected_levels)

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

    required_columns = [
        "whole_tumor_dice",
        "whole_tumor_iou",
        "true_tumor_voxels",
        "pred_tumor_voxels",
    ]

    if metrics_df[required_columns].isna().any().any():
        raise RuntimeError(
            "Missing required metric values detected."
        )

    metrics_df.to_csv(output_metrics, index=False)

    total_seconds = (
        time.perf_counter() - evaluation_start
    )

    write_summary(
        path=output_summary,
        metrics_df=metrics_df,
        artifact=args.artifact,
        total_seconds=total_seconds,
    )

    print("\n" + "=" * 80)
    print("Degraded evaluation completed")
    print("=" * 80)
    print(f"Rows: {len(metrics_df)}")
    print(f"Conditions: {metrics_df['condition'].nunique()}")
    print(f"Metrics: {output_metrics}")
    print(f"Summary: {output_summary}")
    print(f"Configuration: {output_configuration}")
    print(f"Runtime minutes: {total_seconds / 60:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
