#!/usr/bin/env python3
"""
Script 36B: Swin-UNETR clean-versus-degraded 3D Grad-CAM analysis.

Scope
-----
- Uses five fixed representative BraTS2020 test patients.
- Uses one deterministic 96x96x96 tumor-centered patch per patient.
- Reuses the exact historical degradation logic from Script 35A.
- Uses decoder3.conv_block as the finalized Grad-CAM target layer.
- Uses the mean differentiable WT tumor score inside the model-predicted
  WT region as the Grad-CAM target.
- Ground truth is used only after attribution generation for evaluation.
- Grad-CAM is spatial attribution, not a causal explanation.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR


PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

TRAIN_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "34a_train_swin_unetr_clean.py"
)

CLEAN_EVAL_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "34b_evaluate_swin_unetr_clean_test.py"
)

DEGRADED_EVAL_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "35a_evaluate_swin_unetr_degraded_onthefly.py"
)

CHECKPOINT_PATH = (
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

SELECTION_CSV = (
    PROJECT_ROOT
    / "report_materials"
    / "36b_swin_xai_selected_patients.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "report_materials"
    / "swin_xai_36b"
)

PATCH_SIZE = (96, 96, 96)
TARGET_LAYER_NAME = "decoder3.conv_block"

CONDITION_PLAN = {
    "blur": [3, 4, 10],
    "ghosting": [4, 5, 10],
    "noise": [6, 7, 10],
    "ringing": [7, 8, 10],
    "contrast": [10],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-patch Swin-UNETR Grad-CAM on clean and "
            "selected degraded conditions."
        )
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Optional smoke-test limit.",
    )

    parser.add_argument(
        "--only-artifact",
        choices=list(CONDITION_PLAN),
        default=None,
        help="Optional smoke-test restriction.",
    )

    parser.add_argument(
        "--output-tag",
        default=None,
        help="Optional smoke-test output suffix.",
    )

    args = parser.parse_args()

    if args.max_patients is not None and args.max_patients < 1:
        raise ValueError("--max-patients must be at least 1.")

    return args


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not import script: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def parse_coordinate_tuple(value: str) -> tuple[int, int, int]:
    parsed = ast.literal_eval(value)

    if (
        not isinstance(parsed, tuple)
        or len(parsed) != 3
    ):
        raise ValueError(
            f"Invalid patch coordinate: {value}"
        )

    return tuple(int(x) for x in parsed)


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = cam.astype(np.float32)
    cam = cam - float(cam.min())

    maximum = float(cam.max())

    if maximum > 0:
        cam = cam / maximum

    return cam


def dice_binary(
    prediction: np.ndarray,
    truth: np.ndarray,
) -> float:
    prediction = prediction.astype(bool)
    truth = truth.astype(bool)

    denominator = (
        int(prediction.sum())
        + int(truth.sum())
    )

    if denominator == 0:
        return 1.0

    intersection = int(
        np.logical_and(
            prediction,
            truth,
        ).sum()
    )

    return 2.0 * intersection / denominator


def safe_mean(
    values: np.ndarray,
    mask: np.ndarray,
) -> float:
    selected = values[mask]

    if selected.size == 0:
        return float("nan")

    return float(selected.mean())


def high_saliency_iou(
    cam: np.ndarray,
    truth_wt: np.ndarray,
) -> float:
    threshold = float(
        np.quantile(
            cam,
            0.80,
        )
    )

    high_mask = cam >= threshold

    union = np.logical_or(
        high_mask,
        truth_wt,
    ).sum()

    if union == 0:
        return float("nan")

    intersection = np.logical_and(
        high_mask,
        truth_wt,
    ).sum()

    return float(intersection / union)


def weighted_centroid(
    cam: np.ndarray,
) -> np.ndarray:
    total = float(cam.sum())

    if total <= 0:
        return np.array(
            [np.nan, np.nan, np.nan],
            dtype=np.float64,
        )

    coordinates = np.indices(
        cam.shape,
        dtype=np.float64,
    )

    return np.array(
        [
            float(
                (coordinates[axis] * cam).sum()
                / total
            )
            for axis in range(3)
        ],
        dtype=np.float64,
    )


def heatmap_similarity(
    clean_cam: np.ndarray,
    degraded_cam: np.ndarray,
) -> float:
    clean_flat = clean_cam.ravel()
    degraded_flat = degraded_cam.ravel()

    clean_std = float(clean_flat.std())
    degraded_std = float(degraded_flat.std())

    if clean_std == 0 or degraded_std == 0:
        return float("nan")

    return float(
        np.corrcoef(
            clean_flat,
            degraded_flat,
        )[0, 1]
    )


def create_model(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> SwinUNETR:
    configuration = checkpoint[
        "model_configuration"
    ]

    model = SwinUNETR(
        in_channels=int(
            configuration["in_channels"]
        ),
        out_channels=int(
            configuration["out_channels"]
        ),
        feature_size=int(
            configuration["feature_size"]
        ),
        use_checkpoint=bool(
            configuration["use_checkpoint"]
        ),
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    return model


def run_gradcam(
    model: SwinUNETR,
    image_patch: np.ndarray,
    truth_patch: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    activation_record: dict[str, torch.Tensor] = {}
    gradient_record: dict[str, torch.Tensor] = {}

    target_layer = model.decoder3.conv_block

    def forward_hook(
        module,
        inputs,
        output,
    ):
        activation_record["value"] = output

        def save_gradient(
            gradient: torch.Tensor,
        ):
            gradient_record["value"] = gradient

        output.register_hook(save_gradient)

    hook = target_layer.register_forward_hook(
        forward_hook
    )

    try:
        input_tensor = torch.from_numpy(
            image_patch
        ).unsqueeze(0).float().to(device)

        model.zero_grad(
            set_to_none=True
        )

        logits = model(input_tensor)

        tumor_logit = torch.logsumexp(
            logits[:, 1:4],
            dim=1,
        )

        predicted_labels = torch.argmax(
            logits.detach(),
            dim=1,
        )[0]

        predicted_wt_tensor = (
            predicted_labels > 0
        )

        if not torch.any(
            predicted_wt_tensor
        ):
            prediction = (
                predicted_labels
                .detach()
                .cpu()
                .numpy()
                .astype(np.int16)
            )

            truth_wt = truth_patch > 0

            zero_cam = np.zeros(
                PATCH_SIZE,
                dtype=np.float32,
            )

            return {
                "cam": zero_cam,
                "prediction": prediction,
                "gradcam_status": (
                    "unavailable_no_predicted_WT"
                ),
                "target_score": float("nan"),
                "activation_shape": tuple(
                    activation_record["value"].shape
                ),
                "gradient_shape": None,
                "gradient_abs_mean": float("nan"),
                "mean_saliency_inside_WT": float("nan"),
                "mean_saliency_outside_WT": float("nan"),
                "inside_outside_ratio": float("nan"),
                "high_saliency_WT_iou": float("nan"),
                "false_positive_saliency": float("nan"),
                "patch_dice_WT": 0.0,
                "predicted_WT_patch_voxels": 0,
                "true_WT_patch_voxels": int(
                    truth_wt.sum()
                ),
                "centroid": np.array(
                    [np.nan, np.nan, np.nan],
                    dtype=np.float64,
                ),
            }

        score = tumor_logit[0][
            predicted_wt_tensor
        ].mean()

        score.backward()

        activations = activation_record[
            "value"
        ]
        gradients = gradient_record[
            "value"
        ]

        weights = gradients.mean(
            dim=(2, 3, 4),
            keepdim=True,
        )

        cam = torch.sum(
            weights * activations,
            dim=1,
            keepdim=True,
        )

        cam = torch.relu(cam)

        cam = F.interpolate(
            cam,
            size=PATCH_SIZE,
            mode="trilinear",
            align_corners=False,
        )

        cam_np = normalize_cam(
            cam[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        prediction = (
            predicted_labels
            .detach()
            .cpu()
            .numpy()
            .astype(np.int16)
        )

        truth_wt = truth_patch > 0
        predicted_wt = prediction > 0
        false_positive = np.logical_and(
            predicted_wt,
            ~truth_wt,
        )

        inside_mean = safe_mean(
            cam_np,
            truth_wt,
        )

        outside_mean = safe_mean(
            cam_np,
            ~truth_wt,
        )

        if (
            np.isnan(outside_mean)
            or outside_mean == 0
        ):
            inside_outside_ratio = float("nan")
        else:
            inside_outside_ratio = (
                inside_mean / outside_mean
            )

        return {
            "cam": cam_np,
            "prediction": prediction,
            "gradcam_status": "available",
            "target_score": float(
                score.detach().cpu()
            ),
            "activation_shape": tuple(
                activations.shape
            ),
            "gradient_shape": tuple(
                gradients.shape
            ),
            "gradient_abs_mean": float(
                gradients.abs()
                .mean()
                .detach()
                .cpu()
            ),
            "mean_saliency_inside_WT": inside_mean,
            "mean_saliency_outside_WT": outside_mean,
            "inside_outside_ratio": (
                inside_outside_ratio
            ),
            "high_saliency_WT_iou": (
                high_saliency_iou(
                    cam_np,
                    truth_wt,
                )
            ),
            "false_positive_saliency": (
                safe_mean(
                    cam_np,
                    false_positive,
                )
            ),
            "patch_dice_WT": dice_binary(
                predicted_wt,
                truth_wt,
            ),
            "predicted_WT_patch_voxels": int(
                predicted_wt.sum()
            ),
            "true_WT_patch_voxels": int(
                truth_wt.sum()
            ),
            "centroid": weighted_centroid(
                cam_np
            ),
        }

    finally:
        hook.remove()
        model.zero_grad(
            set_to_none=True
        )
        torch.cuda.empty_cache()


def select_axial_slice(
    truth_patch: np.ndarray,
) -> int:
    wt_counts = (
        truth_patch > 0
    ).sum(axis=(0, 1))

    return int(
        np.argmax(wt_counts)
    )


def save_condition_figure(
    output_path: Path,
    patient_id: str,
    condition: str,
    global_slice: int,
    image_patch: np.ndarray,
    truth_patch: np.ndarray,
    prediction: np.ndarray,
    cam: np.ndarray,
    local_slice: int,
) -> None:
    flair_slice = image_patch[
        0, :, :, local_slice
    ]

    truth_slice = (
        truth_patch[
            :, :, local_slice
        ]
        > 0
    )

    prediction_slice = (
        prediction[
            :, :, local_slice
        ]
        > 0
    )

    cam_slice = cam[
        :, :, local_slice
    ]

    figure, axes = plt.subplots(
        1,
        5,
        figsize=(20, 4),
    )

    axes[0].imshow(
        flair_slice.T,
        cmap="gray",
        origin="lower",
    )
    axes[0].set_title(
        "Normalized FLAIR"
    )

    axes[1].imshow(
        flair_slice.T,
        cmap="gray",
        origin="lower",
    )
    axes[1].imshow(
        truth_slice.T,
        cmap="Reds",
        alpha=0.45,
        origin="lower",
    )
    axes[1].set_title(
        "Ground-truth WT"
    )

    axes[2].imshow(
        flair_slice.T,
        cmap="gray",
        origin="lower",
    )
    axes[2].imshow(
        prediction_slice.T,
        cmap="Blues",
        alpha=0.45,
        origin="lower",
    )
    axes[2].set_title(
        "Predicted WT"
    )

    axes[3].imshow(
        cam_slice.T,
        cmap="jet",
        origin="lower",
        vmin=0,
        vmax=1,
    )
    axes[3].set_title(
        "Grad-CAM"
    )

    axes[4].imshow(
        flair_slice.T,
        cmap="gray",
        origin="lower",
    )
    axes[4].imshow(
        cam_slice.T,
        cmap="jet",
        alpha=0.50,
        origin="lower",
        vmin=0,
        vmax=1,
    )
    axes[4].set_title(
        "Grad-CAM overlay"
    )

    for axis in axes:
        axis.axis("off")

    figure.suptitle(
        (
            f"{patient_id} | {condition} | "
            f"axial slice {global_slice}"
        ),
        fontsize=13,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    args = parse_args()

    torch.manual_seed(2026)
    np.random.seed(2026)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required."
        )

    device = torch.device(
        args.device
    )

    if device.type != "cuda":
        raise ValueError(
            "A CUDA device is required."
        )

    output_name = "final"

    if args.output_tag:
        output_name = args.output_tag

    output_dir = (
        OUTPUT_ROOT
        / output_name
    )

    if output_dir.exists():
        raise FileExistsError(
            "Output directory already exists. "
            "Refusing to overwrite:\n"
            f"{output_dir}"
        )

    arrays_dir = (
        output_dir
        / "arrays"
    )

    figures_dir = (
        output_dir
        / "figures"
    )

    arrays_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    training_module = load_module(
        "swin_training_34a_for_36b",
        TRAIN_SCRIPT,
    )

    clean_eval_module = load_module(
        "swin_clean_eval_34b_for_36b",
        CLEAN_EVAL_SCRIPT,
    )

    degraded_module = load_module(
        "swin_degraded_35a_for_36b",
        DEGRADED_EVAL_SCRIPT,
    )

    train_df = pd.read_csv(
        TRAIN_CSV
    )

    val_df = pd.read_csv(
        VAL_CSV
    )

    full_test_df = pd.read_csv(
        TEST_CSV
    )

    selection_df = pd.read_csv(
        SELECTION_CSV
    )

    clean_eval_module.verify_split_integrity(
        train_df,
        val_df,
        full_test_df,
    )

    clean_eval_module.verify_test_paths(
        full_test_df,
        training_module.REQUIRED_COLUMNS,
    )

    if args.max_patients is not None:
        selection_df = selection_df.head(
            args.max_patients
        ).copy()

    selected_patient_ids = (
        selection_df[
            "patient_id"
        ]
        .astype(str)
        .tolist()
    )

    missing_ids = sorted(
        set(selected_patient_ids)
        - set(
            full_test_df[
                "patient_id"
            ].astype(str)
        )
    )

    if missing_ids:
        raise RuntimeError(
            "Selected patient IDs missing from "
            f"test CSV: {missing_ids}"
        )

    artifact_plan = CONDITION_PLAN

    if args.only_artifact:
        artifact_plan = {
            args.only_artifact: (
                CONDITION_PLAN[
                    args.only_artifact
                ]
            )
        }

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if int(checkpoint["epoch"]) != 45:
        raise RuntimeError(
            "Expected epoch-45 checkpoint, "
            f"found {checkpoint['epoch']}."
        )

    model = create_model(
        checkpoint,
        device,
    )

    script26a = degraded_module.load_module(
        "nnunet_degradation_26a_for_36b",
        degraded_module.SCRIPT26A_PATH,
    )

    script29a = degraded_module.load_module(
        "nnunet_degradation_29a_for_36b",
        degraded_module.SCRIPT29A_PATH,
    )

    noise_cache = None

    if "noise" in artifact_plan:
        noise_cache = (
            degraded_module
            .prepare_script29a_noise_cache(
                full_test_df=full_test_df,
                selected_patient_ids=set(
                    selected_patient_ids
                ),
                selected_levels=set(
                    artifact_plan["noise"]
                ),
                script29a=script29a,
            )
        )

    configuration = {
        "script": (
            "36b_evaluate_swin_gradcam_clean_degraded.py"
        ),
        "checkpoint": str(
            CHECKPOINT_PATH
        ),
        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),
        "target_layer": (
            TARGET_LAYER_NAME
        ),
        "target_definition": (
            "Mean differentiable WT tumor score "
            "inside model-predicted WT region"
        ),
        "patch_size": list(
            PATCH_SIZE
        ),
        "patch_reference": (
            "Fixed clean ground-truth tumor-centered "
            "coordinates recorded in the selection CSV"
        ),
        "selected_patients": (
            selected_patient_ids
        ),
        "condition_plan": (
            artifact_plan
        ),
        "device": str(
            device
        ),
        "gpu": torch.cuda.get_device_name(
            device
        ),
        "seed": 2026,
        "script29a_noise_sequence_exact": (
            "noise" in artifact_plan
        ),
        "gradcam_interpretation": (
            "Spatial attribution, not causal explanation"
        ),
    }

    configuration_path = (
        output_dir
        / "36b_configuration.json"
    )

    configuration_path.write_text(
        json.dumps(
            configuration,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    metrics_rows: list[
        dict[str, Any]
    ] = []

    clean_cams: dict[
        str,
        np.ndarray,
    ] = {}

    start_time = time.perf_counter()

    print("=" * 80)
    print(
        "Script 36B: Swin clean-versus-degraded Grad-CAM"
    )
    print("=" * 80)
    print(
        f"Patients: {len(selected_patient_ids)}"
    )
    print(
        f"Artifacts: {list(artifact_plan)}"
    )
    print(
        f"Target layer: {TARGET_LAYER_NAME}"
    )
    print(
        f"Output: {output_dir}"
    )
    print("=" * 80)

    for patient_number, patient_id in enumerate(
        selected_patient_ids,
        start=1,
    ):
        patient_selection = selection_df[
            selection_df["patient_id"]
            == patient_id
        ].iloc[0]

        patient_row = full_test_df[
            full_test_df["patient_id"]
            == patient_id
        ].iloc[0]

        patch_start = parse_coordinate_tuple(
            patient_selection[
                "patch_start"
            ]
        )

        patch_end = parse_coordinate_tuple(
            patient_selection[
                "patch_end"
            ]
        )

        x0, y0, z0 = patch_start
        x1, y1, z1 = patch_end

        clean_image, clean_truth = (
            training_module
            .load_full_patient(
                patient_row
            )
        )

        truth_patch = clean_truth[
            x0:x1,
            y0:y1,
            z0:z1,
        ]

        clean_image_patch = clean_image[
            :,
            x0:x1,
            y0:y1,
            z0:z1,
        ]

        if clean_image_patch.shape != (
            4,
            *PATCH_SIZE,
        ):
            raise RuntimeError(
                f"Unexpected clean patch shape "
                f"for {patient_id}: "
                f"{clean_image_patch.shape}"
            )

        local_slice = select_axial_slice(
            truth_patch
        )

        global_slice = (
            z0 + local_slice
        )

        condition_inputs: list[
            tuple[
                str,
                str,
                int | None,
                np.ndarray,
            ]
        ] = [
            (
                "clean",
                "clean",
                None,
                clean_image_patch,
            )
        ]

        for artifact, levels in (
            artifact_plan.items()
        ):
            for level in levels:
                params = (
                    degraded_module
                    .ALL_LEVELS[
                        artifact
                    ][level]
                )

                (
                    degraded_image,
                    degraded_truth,
                    _,
                ) = (
                    degraded_module
                    .build_degraded_patient(
                        row=patient_row,
                        artifact=artifact,
                        level=level,
                        params=params,
                        training_module=training_module,
                        script26a=script26a,
                        script29a=script29a,
                        script29a_noise_cache=(
                            noise_cache
                        ),
                    )
                )

                if not np.array_equal(
                    degraded_truth,
                    clean_truth,
                ):
                    raise RuntimeError(
                        f"Truth mismatch for "
                        f"{patient_id}."
                    )

                degraded_patch = (
                    degraded_image[
                        :,
                        x0:x1,
                        y0:y1,
                        z0:z1,
                    ]
                )

                condition_inputs.append(
                    (
                        f"{artifact}_L{level}",
                        artifact,
                        level,
                        degraded_patch,
                    )
                )

        for (
            condition,
            artifact,
            level,
            image_patch,
        ) in condition_inputs:
            condition_start = (
                time.perf_counter()
            )

            result = run_gradcam(
                model=model,
                image_patch=image_patch,
                truth_patch=truth_patch,
                device=device,
            )

            cam = result.pop(
                "cam"
            )

            prediction = result.pop(
                "prediction"
            )

            centroid = result.pop(
                "centroid"
            )

            if condition == "clean":
                clean_cams[
                    patient_id
                ] = cam.copy()

                similarity = 1.0
                centroid_shift = 0.0
            else:
                similarity = (
                    heatmap_similarity(
                        clean_cams[
                            patient_id
                        ],
                        cam,
                    )
                )

                clean_centroid = (
                    weighted_centroid(
                        clean_cams[
                            patient_id
                        ]
                    )
                )

                if (
                    np.isnan(
                        clean_centroid
                    ).any()
                    or np.isnan(
                        centroid
                    ).any()
                ):
                    centroid_shift = (
                        float("nan")
                    )
                else:
                    centroid_shift = float(
                        np.linalg.norm(
                            centroid
                            - clean_centroid
                        )
                    )

            array_path = (
                arrays_dir
                / (
                    f"36b_{patient_id}_"
                    f"{condition}_gradcam.npy"
                )
            )

            np.save(
                array_path,
                cam,
            )

            figure_path = (
                figures_dir
                / (
                    f"36b_{patient_id}_"
                    f"{condition}_gradcam.png"
                )
            )

            save_condition_figure(
                output_path=figure_path,
                patient_id=patient_id,
                condition=condition,
                global_slice=global_slice,
                image_patch=image_patch,
                truth_patch=truth_patch,
                prediction=prediction,
                cam=cam,
                local_slice=local_slice,
            )

            elapsed = (
                time.perf_counter()
                - condition_start
            )

            metrics_rows.append(
                {
                    "patient_id": patient_id,
                    "selection_category": (
                        patient_selection[
                            "selection_category"
                        ]
                    ),
                    "condition": condition,
                    "artifact": artifact,
                    "level": level,
                    "patch_start": str(
                        patch_start
                    ),
                    "patch_end": str(
                        patch_end
                    ),
                    "local_axial_slice": (
                        local_slice
                    ),
                    "global_axial_slice": (
                        global_slice
                    ),
                    **result,
                    "clean_to_condition_heatmap_correlation": (
                        similarity
                    ),
                    "attribution_centroid_shift_voxels": (
                        centroid_shift
                    ),
                    "centroid_x": float(
                        centroid[0]
                    ),
                    "centroid_y": float(
                        centroid[1]
                    ),
                    "centroid_z": float(
                        centroid[2]
                    ),
                    "array_path": str(
                        array_path
                    ),
                    "figure_path": str(
                        figure_path
                    ),
                    "gradcam_seconds": elapsed,
                }
            )

            print(
                f"[{patient_number}/"
                f"{len(selected_patient_ids)}] "
                f"{patient_id} | "
                f"{condition} | "
                f"patch WT Dice="
                f"{result['patch_dice_WT']:.4f} | "
                f"inside/outside="
                f"{result['inside_outside_ratio']:.3f} | "
                f"similarity="
                f"{similarity:.3f}"
            )

    metrics_df = pd.DataFrame(
        metrics_rows
    )

    metrics_path = (
        output_dir
        / "36b_swin_xai_condition_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    selected_copy_path = (
        output_dir
        / "36b_swin_xai_selected_patients.csv"
    )

    selection_df.to_csv(
        selected_copy_path,
        index=False,
    )

    total_seconds = (
        time.perf_counter()
        - start_time
    )

    summary_lines = [
        "=" * 80,
        (
            "Script 36B: Swin clean-versus-degraded "
            "Grad-CAM summary"
        ),
        "=" * 80,
        "",
        f"Patients: {len(selected_patient_ids)}",
        f"Conditions per patient: "
        f"{metrics_df['condition'].nunique()}",
        f"Rows: {len(metrics_df)}",
        f"Checkpoint epoch: {checkpoint['epoch']}",
        f"Target layer: {TARGET_LAYER_NAME}",
        (
            "Target definition: mean differentiable WT "
            "tumor score inside model-predicted WT region"
        ),
        (
            "Ground truth guided patch coordinates only; "
            "it did not define the Grad-CAM target score."
        ),
        (
            "Patch-based XAI: Yes. Full-volume segmentation "
            "robustness metrics were evaluated separately."
        ),
        (
            "Grad-CAM interpretation: spatial attribution, "
            "not causal explanation."
        ),
        "",
        "Mean metrics by condition",
        "-" * 80,
    ]

    for condition, group in metrics_df.groupby(
        "condition",
        sort=False,
    ):
        summary_lines.append(
            (
                f"{condition} | "
                f"patch WT Dice "
                f"{group['patch_dice_WT'].mean():.4f} | "
                f"inside/outside "
                f"{group['inside_outside_ratio'].mean():.3f} | "
                f"high-saliency WT IoU "
                f"{group['high_saliency_WT_iou'].mean():.3f} | "
                f"clean correlation "
                f"{group['clean_to_condition_heatmap_correlation'].mean():.3f} | "
                f"centroid shift "
                f"{group['attribution_centroid_shift_voxels'].mean():.3f}"
            )
        )

    summary_lines.extend(
        [
            "",
            (
                f"Total runtime: "
                f"{total_seconds / 60:.2f} minutes"
            ),
            f"Metrics: {metrics_path}",
            f"Configuration: {configuration_path}",
            f"Figures: {figures_dir}",
            f"Arrays: {arrays_dir}",
            "=" * 80,
        ]
    )

    summary_path = (
        output_dir
        / "36b_swin_xai_summary.txt"
    )

    summary_path.write_text(
        "\n".join(
            summary_lines
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n" + "\n".join(
        summary_lines
    ))


if __name__ == "__main__":
    main()
