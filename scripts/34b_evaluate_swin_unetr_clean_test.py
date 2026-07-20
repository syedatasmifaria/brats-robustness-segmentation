#!/usr/bin/env python3
"""
Script 34B: Evaluate the best clean-trained Swin-UNETR checkpoint
on the untouched 74-patient BraTS2020 test cohort.

Methodological rules
--------------------
- Load only the checkpoint selected using the 59-patient validation set.
- Evaluate the 74-patient test set exactly once.
- Do not use test results for checkpoint or hyperparameter selection.
- Use the same normalization, label mapping, model architecture, and
  full-volume sliding-window inference used during validation.
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
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR


PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

TRAIN_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "34a_train_swin_unetr_clean.py"
)

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "swin_unetr"
    / "swin_unetr_full_timing_20260719"
    / "best_checkpoint.pth"
)

TRAIN_CSV = (
    PROJECT_ROOT
    / "data"
    / "csvs"
    / "swin_unetr_train_paths.csv"
)

VAL_CSV = (
    PROJECT_ROOT
    / "data"
    / "csvs"
    / "swin_unetr_val_paths.csv"
)

TEST_CSV = (
    PROJECT_ROOT
    / "data"
    / "csvs"
    / "test_paths.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_DIR = PROJECT_ROOT / "report_materials"

OUTPUT_METRICS = (
    RESULTS_DIR
    / "34b_swin_unetr_clean_test_metrics.csv"
)

OUTPUT_SUMMARY = (
    REPORT_DIR
    / "34b_swin_unetr_clean_test_summary.txt"
)

OUTPUT_CONFIGURATION = (
    REPORT_DIR
    / "34b_swin_unetr_clean_test_configuration.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the best clean-trained Swin-UNETR checkpoint "
            "on the untouched 74-patient test cohort."
        )
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

    args = parser.parse_args()

    if args.sw_batch_size < 1:
        raise ValueError(
            "--sw-batch-size must be at least 1."
        )

    if not 0.0 <= args.sw_overlap < 1.0:
        raise ValueError(
            "--sw-overlap must be in [0, 1)."
        )

    return args


def load_training_module():
    spec = importlib.util.spec_from_file_location(
        "swin_unetr_training_34a",
        TRAIN_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not import training script: {TRAIN_SCRIPT}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def verify_split_integrity(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    for name, dataframe in [
        ("training", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:
        if "patient_id" not in dataframe.columns:
            raise ValueError(
                f"{name} CSV is missing patient_id."
            )

        if dataframe["patient_id"].duplicated().any():
            duplicated = dataframe.loc[
                dataframe["patient_id"].duplicated(),
                "patient_id",
            ].tolist()

            raise ValueError(
                f"Duplicate patients in {name} CSV: {duplicated}"
            )

    train_ids = set(
        train_df["patient_id"].astype(str)
    )

    val_ids = set(
        val_df["patient_id"].astype(str)
    )

    test_ids = set(
        test_df["patient_id"].astype(str)
    )

    if len(train_ids) != 235:
        raise ValueError(
            f"Expected 235 training patients, found {len(train_ids)}."
        )

    if len(val_ids) != 59:
        raise ValueError(
            f"Expected 59 validation patients, found {len(val_ids)}."
        )

    if len(test_ids) != 74:
        raise ValueError(
            f"Expected 74 test patients, found {len(test_ids)}."
        )

    if train_ids & val_ids:
        raise RuntimeError(
            "Training and validation cohorts overlap."
        )

    if train_ids & test_ids:
        raise RuntimeError(
            "Training and test cohorts overlap."
        )

    if val_ids & test_ids:
        raise RuntimeError(
            "Validation and test cohorts overlap."
        )

    if len(train_ids | val_ids | test_ids) != 368:
        raise RuntimeError(
            "The three cohorts do not contain 368 unique patients."
        )


def verify_test_paths(
    test_df: pd.DataFrame,
    required_columns: list[str],
) -> None:
    missing_columns = [
        column
        for column in required_columns
        if column not in test_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Test CSV is missing columns: {missing_columns}"
        )

    missing_files: list[str] = []

    for _, row in test_df.iterrows():
        patient_id = str(row["patient_id"])

        for column in required_columns:
            if column == "patient_id":
                continue

            path = Path(str(row[column]))

            if not path.exists():
                missing_files.append(
                    f"{patient_id}: {column}: {path}"
                )

    if missing_files:
        preview = "\n".join(
            missing_files[:20]
        )

        raise FileNotFoundError(
            "Missing test files:\n"
            f"{preview}"
        )


def write_summary(
    path: Path,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    metrics_df: pd.DataFrame,
    total_seconds: float,
    sw_batch_size: int,
    sw_overlap: float,
    amp_enabled: bool,
) -> None:
    dice_wt_mean = float(
        metrics_df["dice_WT"].mean()
    )

    dice_tc_mean = float(
        metrics_df["dice_TC"].mean()
    )

    dice_et_mean = float(
        metrics_df["dice_ET"].mean()
    )

    macro_dice = float(
        np.mean([
            dice_wt_mean,
            dice_tc_mean,
            dice_et_mean,
        ])
    )

    lines = [
        "=" * 80,
        "Script 34B: Swin-UNETR clean test evaluation",
        "=" * 80,
        "",
        f"Checkpoint: {checkpoint_path}",
        f"Checkpoint epoch: {checkpoint['epoch']}",
        f"Best validation epoch: {checkpoint['best_epoch']}",
        (
            "Best validation macro Dice: "
            f"{checkpoint['best_val_macro_dice']:.6f}"
        ),
        "",
        f"Test patients: {len(metrics_df)}",
        "Test cohort used for checkpoint selection: No",
        "Clean-trained model: Yes",
        "Clean test images: Yes",
        "Full-volume sliding-window inference: Yes",
        "ROI size: 96 x 96 x 96",
        f"Sliding-window batch size: {sw_batch_size}",
        f"Sliding-window overlap: {sw_overlap}",
        "Blend mode: gaussian",
        f"AMP enabled: {amp_enabled}",
        "",
        "Mean test metrics",
        "-" * 80,
        (
            f"WT Dice: {dice_wt_mean:.6f} "
            f"(SD {metrics_df['dice_WT'].std(ddof=1):.6f})"
        ),
        (
            f"WT IoU:  {metrics_df['iou_WT'].mean():.6f} "
            f"(SD {metrics_df['iou_WT'].std(ddof=1):.6f})"
        ),
        (
            f"TC Dice: {dice_tc_mean:.6f} "
            f"(SD {metrics_df['dice_TC'].std(ddof=1):.6f})"
        ),
        (
            f"TC IoU:  {metrics_df['iou_TC'].mean():.6f} "
            f"(SD {metrics_df['iou_TC'].std(ddof=1):.6f})"
        ),
        (
            f"ET Dice: {dice_et_mean:.6f} "
            f"(SD {metrics_df['dice_ET'].std(ddof=1):.6f})"
        ),
        (
            f"ET IoU:  {metrics_df['iou_ET'].mean():.6f} "
            f"(SD {metrics_df['iou_ET'].std(ddof=1):.6f})"
        ),
        f"Macro regional Dice: {macro_dice:.6f}",
        "",
        "Whole-tumor voxel counts",
        "-" * 80,
        (
            "Mean predicted WT voxels: "
            f"{metrics_df['pred_WT_voxels'].mean():.2f}"
        ),
        (
            "Mean true WT voxels: "
            f"{metrics_df['true_WT_voxels'].mean():.2f}"
        ),
        "",
        f"Total evaluation time: {total_seconds / 60:.2f} minutes",
        (
            "Mean inference time per patient: "
            f"{metrics_df['inference_seconds'].mean():.2f} seconds"
        ),
        f"Patient metrics CSV: {OUTPUT_METRICS}",
    ]

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()

    print("=" * 80)
    print(
        "Script 34B: Evaluate Swin-UNETR on clean test cohort"
    )
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    device = torch.device(
        args.device
    )

    if device.type != "cuda":
        raise ValueError(
            "This evaluation requires a CUDA device."
        )

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    for output_path in [
        OUTPUT_METRICS,
        OUTPUT_SUMMARY,
        OUTPUT_CONFIGURATION,
    ]:
        if output_path.exists():
            raise FileExistsError(
                "Clean test outputs already exist. "
                "Do not rerun or overwrite the untouched-test "
                f"evaluation casually:\n{output_path}"
            )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_module = load_training_module()

    train_df = pd.read_csv(
        TRAIN_CSV
    )

    val_df = pd.read_csv(
        VAL_CSV
    )

    test_df = pd.read_csv(
        TEST_CSV
    )

    verify_split_integrity(
        train_df,
        val_df,
        test_df,
    )

    verify_test_paths(
        test_df,
        training_module.REQUIRED_COLUMNS,
    )

    print("\nCohort verification")
    print("-" * 80)
    print(f"Training patients: {len(train_df)}")
    print(f"Validation patients: {len(val_df)}")
    print(f"Test patients: {len(test_df)}")
    print("All pairwise overlaps: 0")
    print("Total unique patients: 368")

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
        "configuration",
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
        stored_value = model_configuration.get(
            key
        )

        if key == "patch_size":
            stored_value = tuple(
                stored_value
            )

        if stored_value != expected_value:
            raise RuntimeError(
                f"Checkpoint configuration mismatch for {key}: "
                f"expected {expected_value}, found {stored_value}"
            )

    if int(checkpoint["epoch"]) != 45:
        raise RuntimeError(
            "Expected the validation-selected epoch-45 checkpoint, "
            f"but checkpoint epoch is {checkpoint['epoch']}."
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

    print("\nCheckpoint verification")
    print("-" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Best epoch: {checkpoint['best_epoch']}")
    print(
        "Best validation macro Dice: "
        f"{checkpoint['best_val_macro_dice']:.6f}"
    )

    print("\nInference configuration")
    print("-" * 80)
    print(f"Device: {device}")
    print(
        f"GPU: {torch.cuda.get_device_name(device)}"
    )
    print("ROI size: (96, 96, 96)")
    print(
        f"Sliding-window batch size: {args.sw_batch_size}"
    )
    print(
        f"Sliding-window overlap: {args.sw_overlap}"
    )
    print("Blend mode: gaussian")
    print(f"AMP enabled: {amp_enabled}")
    print("Output stitching device: CPU")

    configuration = {
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
        "model_configuration": model_configuration,
        "test_csv": str(
            TEST_CSV
        ),
        "test_patients": len(
            test_df
        ),
        "device": str(
            device
        ),
        "gpu": torch.cuda.get_device_name(
            device
        ),
        "amp_enabled": amp_enabled,
        "roi_size": [96, 96, 96],
        "sw_batch_size": args.sw_batch_size,
        "sw_overlap": args.sw_overlap,
        "blend_mode": "gaussian",
        "output_stitching_device": "cpu",
    }

    OUTPUT_CONFIGURATION.write_text(
        json.dumps(
            configuration,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    evaluation_start = time.perf_counter()

    def predictor(
        window: torch.Tensor,
    ) -> torch.Tensor:
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            return model(window)

    print("\n" + "=" * 80)
    print("Full-volume clean test evaluation")
    print("=" * 80)

    for patient_number, (_, row) in enumerate(
        test_df.iterrows(),
        start=1,
    ):
        patient_start = time.perf_counter()

        patient_id = str(
            row["patient_id"]
        )

        image, truth = (
            training_module.load_full_patient(
                row
            )
        )

        input_tensor = torch.from_numpy(
            image
        ).unsqueeze(0).float()

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

        metrics = (
            training_module.compute_metrics(
                prediction,
                truth,
            )
        )

        patient_seconds = (
            time.perf_counter()
            - patient_start
        )

        rows.append({
            "patient_id": patient_id,
            **metrics,
            "inference_seconds": patient_seconds,
        })

        print(
            f"[{patient_number:02d}/74] "
            f"{patient_id} | "
            f"WT={metrics['dice_WT']:.4f} | "
            f"TC={metrics['dice_TC']:.4f} | "
            f"ET={metrics['dice_ET']:.4f} | "
            f"{patient_seconds:.1f}s"
        )

        del input_tensor
        del logits
        del prediction

        torch.cuda.empty_cache()

    metrics_df = pd.DataFrame(
        rows
    )

    if len(metrics_df) != 74:
        raise RuntimeError(
            f"Expected 74 test results, found {len(metrics_df)}."
        )

    if metrics_df["patient_id"].nunique() != 74:
        raise RuntimeError(
            "Test metrics do not contain 74 unique patients."
        )

    metrics_df.to_csv(
        OUTPUT_METRICS,
        index=False,
    )

    total_seconds = (
        time.perf_counter()
        - evaluation_start
    )

    write_summary(
        path=OUTPUT_SUMMARY,
        checkpoint_path=args.checkpoint,
        checkpoint=checkpoint,
        metrics_df=metrics_df,
        total_seconds=total_seconds,
        sw_batch_size=args.sw_batch_size,
        sw_overlap=args.sw_overlap,
        amp_enabled=amp_enabled,
    )

    mean_wt = float(
        metrics_df["dice_WT"].mean()
    )

    mean_tc = float(
        metrics_df["dice_TC"].mean()
    )

    mean_et = float(
        metrics_df["dice_ET"].mean()
    )

    macro_dice = float(
        np.mean([
            mean_wt,
            mean_tc,
            mean_et,
        ])
    )

    print("\n" + "=" * 80)
    print("Clean test evaluation completed")
    print("=" * 80)
    print(f"WT Dice:    {mean_wt:.6f}")
    print(f"TC Dice:    {mean_tc:.6f}")
    print(f"ET Dice:    {mean_et:.6f}")
    print(f"Macro Dice: {macro_dice:.6f}")
    print(
        f"Total time: {total_seconds / 60:.2f} minutes"
    )
    print(f"Patient metrics: {OUTPUT_METRICS}")
    print(f"Summary: {OUTPUT_SUMMARY}")
    print(
        f"Configuration: {OUTPUT_CONFIGURATION}"
    )


if __name__ == "__main__":
    main()
