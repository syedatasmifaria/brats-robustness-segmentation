#!/usr/bin/env python3
"""
Script 37B: nnU-Net clean-versus-degraded 3D Grad-CAM analysis.

Uses the exact saved nnU-Net input NIfTI files from the completed
robustness evaluation, official nnU-Net preprocessing, fixed clean-derived
128 x 128 x 128 patches, and decoder.stages[3] as the target layer.

Grad-CAM provides a spatial attribution map associated with the selected
target score. It does not establish a causal explanation of model reasoning.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


PROJECT_ROOT = Path(
    "/home/xfh25/brats_segmentation_project"
)

MODEL_FOLDER = (
    PROJECT_ROOT
    / "nnunet"
    / "nnUNet_results"
    / "Dataset501_BraTS2020Multimodal"
    / "nnUNetTrainer__nnUNetPlans__3d_fullres"
)

CLEAN_IMAGES_DIR = (
    PROJECT_ROOT
    / "nnunet"
    / "nnUNet_raw"
    / "Dataset501_BraTS2020Multimodal"
    / "imagesTs"
)

DEGRADED_ROOT = (
    PROJECT_ROOT
    / "nnunet"
    / "temporary_degraded_tests"
)

ORIGINAL_DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "BraTS2020_TrainingData"
    / "MICCAI_BraTS2020_TrainingData"
)

SELECTION_CSV = (
    PROJECT_ROOT
    / "report_materials"
    / "37b_nnunet_xai_selected_patients.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "report_materials"
    / "nnunet_xai_37b"
)

PATCH_SIZE = (
    128,
    128,
    128,
)

TARGET_LAYER_NAME = (
    "decoder.stages[3]"
)

CONDITIONS: dict[str, dict[str, Any]] = {
    "clean": {
        "artifact": "clean",
        "level": None,
        "directory": CLEAN_IMAGES_DIR,
        "template": (
            "BRATS_{number}_{channel:04d}.nii.gz"
        ),
    },
    "blur_L3": {
        "artifact": "blur",
        "level": 3,
        "directory": (
            DEGRADED_ROOT
            / "final_full"
            / "blur_L3"
            / "imagesTs"
        ),
        "template": (
            "BraTS20_Training_"
            "{number}_{channel:04d}.nii.gz"
        ),
    },
    "blur_L4": {
        "artifact": "blur",
        "level": 4,
        "directory": (
            DEGRADED_ROOT
            / "final_full"
            / "blur_L4"
            / "imagesTs"
        ),
        "template": (
            "BraTS20_Training_"
            "{number}_{channel:04d}.nii.gz"
        ),
    },
    "blur_L10": {
        "artifact": "blur",
        "level": 10,
        "directory": (
            DEGRADED_ROOT
            / "blur_ghosting_extended_full"
            / "blur_L10"
            / "imagesTs"
        ),
        "template": (
            "BraTS20_Training_"
            "{number}_{channel:04d}.nii.gz"
        ),
    },
    "ghosting_L4": {
        "artifact": "ghosting",
        "level": 4,
        "directory": (
            DEGRADED_ROOT
            / "final_full"
            / "ghosting_L4"
            / "imagesTs"
        ),
        "template": (
            "BraTS20_Training_"
            "{number}_{channel:04d}.nii.gz"
        ),
    },
    "ghosting_L5": {
        "artifact": "ghosting",
        "level": 5,
        "directory": (
            DEGRADED_ROOT
            / "final_full"
            / "ghosting_L5"
            / "imagesTs"
        ),
        "template": (
            "BraTS20_Training_"
            "{number}_{channel:04d}.nii.gz"
        ),
    },
    "ghosting_L10": {
        "artifact": "ghosting",
        "level": 10,
        "directory": (
            DEGRADED_ROOT
            / "blur_ghosting_extended_full"
            / "ghosting_L10"
            / "imagesTs"
        ),
        "template": (
            "BraTS20_Training_"
            "{number}_{channel:04d}.nii.gz"
        ),
    },
}

CONDITIONS.update(
    {
        "noise_L6": {
            "artifact": "noise",
            "level": 6,
            "directory": (
                DEGRADED_ROOT
                / "extended_full_selected"
                / "noise_L6"
                / "imagesTs"
            ),
            "template": (
                "BRATS_{number}_{channel:04d}.nii.gz"
            ),
        },
        "noise_L7": {
            "artifact": "noise",
            "level": 7,
            "directory": (
                DEGRADED_ROOT
                / "extended_full_selected"
                / "noise_L7"
                / "imagesTs"
            ),
            "template": (
                "BRATS_{number}_{channel:04d}.nii.gz"
            ),
        },
        "noise_L10": {
            "artifact": "noise",
            "level": 10,
            "directory": (
                DEGRADED_ROOT
                / "extended_full_selected"
                / "noise_L10"
                / "imagesTs"
            ),
            "template": (
                "BRATS_{number}_{channel:04d}.nii.gz"
            ),
        },
        "ringing_L7": {
            "artifact": "ringing",
            "level": 7,
            "directory": (
                DEGRADED_ROOT
                / "extended_full_selected"
                / "ringing_L7"
                / "imagesTs"
            ),
            "template": (
                "BRATS_{number}_{channel:04d}.nii.gz"
            ),
        },
        "ringing_L8": {
            "artifact": "ringing",
            "level": 8,
            "directory": (
                DEGRADED_ROOT
                / "extended_full_selected"
                / "ringing_L8"
                / "imagesTs"
            ),
            "template": (
                "BRATS_{number}_{channel:04d}.nii.gz"
            ),
        },
        "ringing_L10": {
            "artifact": "ringing",
            "level": 10,
            "directory": (
                DEGRADED_ROOT
                / "extended_full_selected"
                / "ringing_L10"
                / "imagesTs"
            ),
            "template": (
                "BRATS_{number}_{channel:04d}.nii.gz"
            ),
        },
        "contrast_L10": {
            "artifact": "contrast",
            "level": 10,
            "directory": (
                DEGRADED_ROOT
                / "extended_full_selected"
                / "contrast_L10"
                / "imagesTs"
            ),
            "template": (
                "BRATS_{number}_{channel:04d}.nii.gz"
            ),
        },
    }
)

ARTIFACTS = [
    "blur",
    "ghosting",
    "noise",
    "ringing",
    "contrast",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-patch nnU-Net Grad-CAM on clean "
            "and selected degraded conditions."
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
    )

    parser.add_argument(
        "--only-artifact",
        choices=ARTIFACTS,
        default=None,
    )

    parser.add_argument(
        "--output-tag",
        default=None,
    )

    args = parser.parse_args()

    if (
        args.max_patients is not None
        and args.max_patients < 1
    ):
        raise ValueError(
            "--max-patients must be at least 1."
        )

    return args


def parse_tuple3(
    value: Any,
) -> tuple[int, int, int]:
    if isinstance(
        value,
        (tuple, list),
    ):
        parsed = value
    else:
        parsed = ast.literal_eval(
            str(value)
        )

    if (
        not isinstance(
            parsed,
            (tuple, list),
        )
        or len(parsed) != 3
    ):
        raise ValueError(
            f"Expected three values, found: {value}"
        )

    return tuple(
        int(item)
        for item in parsed
    )


def normalize_cam(
    cam: np.ndarray,
) -> np.ndarray:
    cam = cam.astype(
        np.float32
    )

    cam = cam - float(
        cam.min()
    )

    maximum = float(
        cam.max()
    )

    if maximum > 0:
        cam = cam / maximum

    return cam

CONDITIONS.update(
    {
        "noise_L6": {
            "artifact": "noise",
            "level": 6,
            "directory": (
                DEGRADED_ROOT
                / "extended_full_selected"
                / "noise_L6"
                / "imagesTs"
            ),
            "template": (
                "BRATS_{number}_{channel:04d}.nii.gz"
            ),
        },
        "noise_L7": {
            "artifact": "noise",
            "level": 7,
            "directory": (
                DEGRADED_ROOT
                / "extended_full_selected"
                / "noise_L7"
                / "imagesTs"
            ),
            "template": (
                "BRATS_{number}_{channel:04d}.nii.gz"
            ),
        },
        "noise_L10": {
            "artifact": "noise",
            "level": 10,
            "directory": (
                DEGRADED_ROOT
                / "extended_full_selected"
                / "noise_L10"
                / "imagesTs"
            ),
            "template": (
                "BRATS_{number}_{channel:04d}.nii.gz"
            ),
        },
        "ringing_L7": {
            "artifact": "ringing",
            "level": 7,
            "directory": (
                DEGRADED_ROOT
                / "extended_full_selected"
                / "ringing_L7"
                / "imagesTs"
            ),
            "template": (
                "BRATS_{number}_{channel:04d}.nii.gz"
            ),
        },
        "ringing_L8": {
            "artifact": "ringing",
            "level": 8,
            "directory": (
                DEGRADED_ROOT
                / "extended_full_selected"
                / "ringing_L8"
                / "imagesTs"
            ),
            "template": (
                "BRATS_{number}_{channel:04d}.nii.gz"
            ),
        },
        "ringing_L10": {
            "artifact": "ringing",
            "level": 10,
            "directory": (
                DEGRADED_ROOT
                / "extended_full_selected"
                / "ringing_L10"
                / "imagesTs"
            ),
            "template": (
                "BRATS_{number}_{channel:04d}.nii.gz"
            ),
        },
        "contrast_L10": {
            "artifact": "contrast",
            "level": 10,
            "directory": (
                DEGRADED_ROOT
                / "extended_full_selected"
                / "contrast_L10"
                / "imagesTs"
            ),
            "template": (
                "BRATS_{number}_{channel:04d}.nii.gz"
            ),
        },
    }
)

ARTIFACTS = [
    "blur",
    "ghosting",
    "noise",
    "ringing",
    "contrast",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-patch nnU-Net Grad-CAM on clean "
            "and selected degraded conditions."
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
    )

    parser.add_argument(
        "--only-artifact",
        choices=ARTIFACTS,
        default=None,
    )

    parser.add_argument(
        "--output-tag",
        default=None,
    )

    args = parser.parse_args()

    if (
        args.max_patients is not None
        and args.max_patients < 1
    ):
        raise ValueError(
            "--max-patients must be at least 1."
        )

    return args


def parse_tuple3(
    value: Any,
) -> tuple[int, int, int]:
    if isinstance(
        value,
        (tuple, list),
    ):
        parsed = value
    else:
        parsed = ast.literal_eval(
            str(value)
        )

    if (
        not isinstance(
            parsed,
            (tuple, list),
        )
        or len(parsed) != 3
    ):
        raise ValueError(
            f"Expected three values, found: {value}"
        )

    return tuple(
        int(item)
        for item in parsed
    )


def normalize_cam(
    cam: np.ndarray,
) -> np.ndarray:
    cam = cam.astype(
        np.float32
    )

    cam = cam - float(
        cam.min()
    )

    maximum = float(
        cam.max()
    )

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

    return (
        2.0
        * intersection
        / denominator
    )


def safe_mean(
    values: np.ndarray,
    mask: np.ndarray,
) -> float:
    selected = values[mask]

    if selected.size == 0:
        return float("nan")

    return float(
        selected.mean()
    )


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

    high_mask = (
        cam >= threshold
    )

    union = int(
        np.logical_or(
            high_mask,
            truth_wt,
        ).sum()
    )

    if union == 0:
        return float("nan")

    intersection = int(
        np.logical_and(
            high_mask,
            truth_wt,
        ).sum()
    )

    return float(
        intersection
        / union
    )


def weighted_centroid(
    cam: np.ndarray,
) -> np.ndarray:
    total = float(
        cam.sum()
    )

    if total <= 0:
        return np.array(
            [
                np.nan,
                np.nan,
                np.nan,
            ],
            dtype=np.float64,
        )

    coordinates = np.indices(
        cam.shape,
        dtype=np.float64,
    )

    return np.array(
        [
            float(
                (
                    coordinates[axis]
                    * cam
                ).sum()
                / total
            )
            for axis in range(3)
        ],
        dtype=np.float64,
    )


def heatmap_similarity(
    clean_cam: np.ndarray,
    condition_cam: np.ndarray,
) -> float:
    clean_flat = clean_cam.ravel()
    condition_flat = condition_cam.ravel()

    if (
        float(clean_flat.std()) == 0
        or float(condition_flat.std()) == 0
    ):
        return float("nan")

    return float(
        np.corrcoef(
            clean_flat,
            condition_flat,
        )[0, 1]
    )


def patient_number(
    patient_id: str,
) -> str:
    number = patient_id.rsplit(
        "_",
        1,
    )[-1]

    if not number.isdigit():
        raise ValueError(
            "Could not extract patient number "
            f"from {patient_id}."
        )

    return number.zfill(3)


def image_paths(
    patient_id: str,
    condition: str,
) -> list[Path]:
    specification = CONDITIONS[
        condition
    ]

    number = patient_number(
        patient_id
    )

    paths = [
        Path(
            specification["directory"]
        )
        / str(
            specification["template"]
        ).format(
            number=number,
            channel=channel,
        )
        for channel in range(4)
    ]

    missing = [
        path
        for path in paths
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing condition inputs:\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )

    return paths


def segmentation_path(
    patient_id: str,
) -> Path:
    path = (
        ORIGINAL_DATA_ROOT
        / patient_id
        / f"{patient_id}_seg.nii"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing ground truth: {path}"
        )

    return path


def plain_value(
    value: Any,
) -> Any:
    if isinstance(
        value,
        np.ndarray,
    ):
        return plain_value(
            value.tolist()
        )

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): plain_value(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (tuple, list),
    ):
        return [
            plain_value(item)
            for item in value
        ]

    return value


def geometry_signature(
    properties: dict[str, Any],
) -> dict[str, Any]:
    keys = [
        "bbox_used_for_cropping",
        (
            "shape_after_cropping_"
            "and_before_resampling"
        ),
        "shape_before_cropping",
        "sitk_stuff",
        "spacing",
    ]

    return {
        key: plain_value(
            properties.get(key)
        )
        for key in keys
    }


def preprocess_case(
    paths: list[Path],
    raw_segmentation: np.ndarray,
    preprocessor,
    predictor: nnUNetPredictor,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, Any],
    tuple[int, ...],
]:
    (
        raw_data,
        image_properties,
    ) = SimpleITKIO().read_images(
        [
            str(path)
            for path in paths
        ]
    )

    if (
        raw_data.ndim != 4
        or raw_data.shape[0] != 4
    ):
        raise RuntimeError(
            "Unexpected raw image shape: "
            f"{raw_data.shape}"
        )

    if (
        tuple(
            raw_data.shape[1:]
        )
        != tuple(
            raw_segmentation.shape[1:]
        )
    ):
        raise RuntimeError(
            "Raw image and segmentation shapes "
            "do not match: "
            f"{raw_data.shape} versus "
            f"{raw_segmentation.shape}"
        )

    (
        data,
        segmentation,
        properties,
    ) = preprocessor.run_case_npy(
        raw_data,
        raw_segmentation.copy(),
        image_properties,
        predictor.plans_manager,
        predictor.configuration_manager,
        predictor.dataset_json,
    )

    return (
        data.astype(
            np.float32,
            copy=False,
        ),
        segmentation.astype(
            np.int16,
            copy=False,
        ),
        properties,
        tuple(
            int(item)
            for item in raw_data.shape
        ),
    )


def pad_case(
    data: np.ndarray,
    segmentation: np.ndarray,
    pad_before: tuple[int, int, int],
    pad_after: tuple[int, int, int],
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    spatial_padding = tuple(
        (
            pad_before[axis],
            pad_after[axis],
        )
        for axis in range(3)
    )

    padded_data = np.pad(
        data,
        (
            (0, 0),
            *spatial_padding,
        ),
        mode="constant",
        constant_values=0,
    )

    padded_segmentation = np.pad(
        segmentation,
        (
            (0, 0),
            *spatial_padding,
        ),
        mode="constant",
        constant_values=-1,
    )

    return (
        padded_data,
        padded_segmentation,
    )


def select_axial_slice(
    truth_patch: np.ndarray,
) -> int:
    counts = (
        truth_patch > 0
    ).sum(
        axis=(1, 2)
    )

    return int(
        np.argmax(
            counts
        )
    )


def run_gradcam(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    image_patch: np.ndarray,
    truth_patch: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    activation_record: dict[
        str,
        torch.Tensor,
    ] = {}

    gradient_record: dict[
        str,
        torch.Tensor,
    ] = {}

    def forward_hook(
        module,
        inputs,
        output,
    ):
        activation_record[
            "value"
        ] = output

        def save_gradient(
            gradient: torch.Tensor,
        ):
            gradient_record[
                "value"
            ] = gradient

        output.register_hook(
            save_gradient
        )

    hook = target_layer.register_forward_hook(
        forward_hook
    )

    try:
        input_tensor = (
            torch.from_numpy(
                image_patch
            )
            .unsqueeze(0)
            .float()
            .to(device)
        )

        model.zero_grad(
            set_to_none=True
        )

        logits = model(
            input_tensor
        )

        tumor_logit = torch.logsumexp(
            logits[:, 1:4],
            dim=1,
        )

        predicted_labels = torch.argmax(
            logits.detach(),
            dim=1,
        )[0]

        prediction = (
            predicted_labels
            .cpu()
            .numpy()
            .astype(np.int16)
        )

        predicted_wt_tensor = (
            predicted_labels > 0
        )

        predicted_wt = (
            prediction > 0
        )

        truth_wt = (
            truth_patch > 0
        )

        if not torch.any(
            predicted_wt_tensor
        ):
            return {
                "cam": np.zeros(
                    PATCH_SIZE,
                    dtype=np.float32,
                ),
                "prediction": prediction,
                "centroid": np.array(
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    dtype=np.float64,
                ),
                "gradcam_status": (
                    "unavailable_no_predicted_WT"
                ),
                "target_score": float("nan"),
                "activation_shape": tuple(
                    activation_record[
                        "value"
                    ].shape
                ),
                "gradient_shape": None,
                "gradient_abs_mean": float("nan"),
                "mean_saliency_inside_WT": (
                    float("nan")
                ),
                "mean_saliency_outside_WT": (
                    float("nan")
                ),
                "inside_outside_ratio": (
                    float("nan")
                ),
                "high_saliency_WT_iou": (
                    float("nan")
                ),
                "false_positive_saliency": (
                    float("nan")
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
            }

        target_score = tumor_logit[
            0
        ][
            predicted_wt_tensor
        ].mean()

        target_score.backward()

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

        cam = torch.relu(
            cam
        )

        cam = F.interpolate(
            cam,
            size=PATCH_SIZE,
            mode="trilinear",
            align_corners=False,
        )

        cam_np = normalize_cam(
            cam[
                0,
                0,
            ]
            .detach()
            .cpu()
            .numpy()
        )

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
            np.isnan(
                inside_mean
            )
            or np.isnan(
                outside_mean
            )
            or outside_mean == 0
        ):
            ratio = float("nan")
        else:
            ratio = (
                inside_mean
                / outside_mean
            )

        return {
            "cam": cam_np,
            "prediction": prediction,
            "centroid": weighted_centroid(
                cam_np
            ),
            "gradcam_status": "available",
            "target_score": float(
                target_score
                .detach()
                .cpu()
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
            "mean_saliency_inside_WT": (
                inside_mean
            ),
            "mean_saliency_outside_WT": (
                outside_mean
            ),
            "inside_outside_ratio": ratio,
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
        }

    finally:
        hook.remove()

        model.zero_grad(
            set_to_none=True
        )

        torch.cuda.empty_cache()


def save_condition_figure(
    output_path: Path,
    patient_id: str,
    condition: str,
    gradcam_status: str,
    global_slice: int,
    image_patch: np.ndarray,
    truth_patch: np.ndarray,
    prediction: np.ndarray,
    cam: np.ndarray,
    local_slice: int,
) -> None:
    flair_slice = image_patch[
        0,
        local_slice,
        :,
        :,
    ]

    truth_slice = (
        truth_patch[
            local_slice,
            :,
            :,
        ]
        > 0
    )

    prediction_slice = (
        prediction[
            local_slice,
            :,
            :,
        ]
        > 0
    )

    cam_slice = cam[
        local_slice,
        :,
        :,
    ]

    figure, axes = plt.subplots(
        1,
        5,
        figsize=(20, 4),
    )

    axes[0].imshow(
        flair_slice,
        cmap="gray",
        origin="lower",
    )
    axes[0].set_title(
        "Preprocessed FLAIR"
    )

    axes[1].imshow(
        flair_slice,
        cmap="gray",
        origin="lower",
    )
    axes[1].imshow(
        truth_slice,
        cmap="Reds",
        alpha=0.45,
        origin="lower",
    )
    axes[1].set_title(
        "Ground-truth WT"
    )

    axes[2].imshow(
        flair_slice,
        cmap="gray",
        origin="lower",
    )
    axes[2].imshow(
        prediction_slice,
        cmap="Blues",
        alpha=0.45,
        origin="lower",
    )
    axes[2].set_title(
        "Predicted WT"
    )

    axes[3].imshow(
        cam_slice,
        cmap="jet",
        origin="lower",
        vmin=0,
        vmax=1,
    )
    axes[3].set_title(
        "Grad-CAM"
    )

    axes[4].imshow(
        flair_slice,
        cmap="gray",
        origin="lower",
    )
    axes[4].imshow(
        cam_slice,
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
            f"preprocessed axial z={global_slice} | "
            f"{gradcam_status}"
        ),
        fontsize=12,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def metric_text(
    value: float,
) -> str:
    if pd.isna(
        value
    ):
        return "NaN"

    return f"{value:.4f}"


def main() -> None:
    args = parse_args()

    torch.manual_seed(
        2026
    )

    np.random.seed(
        2026
    )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    os.environ[
        "nnUNet_raw"
    ] = str(
        PROJECT_ROOT
        / "nnunet"
        / "nnUNet_raw"
    )

    os.environ[
        "nnUNet_preprocessed"
    ] = str(
        PROJECT_ROOT
        / "nnunet"
        / "nnUNet_preprocessed"
    )

    os.environ[
        "nnUNet_results"
    ] = str(
        PROJECT_ROOT
        / "nnunet"
        / "nnUNet_results"
    )

    output_name = (
        args.output_tag
        or "final"
    )

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

    selection_df = pd.read_csv(
        SELECTION_CSV
    )

    required_columns = {
        "patient_id",
        "selection_category",
        "preprocessed_shape",
        "pad_before_zyx",
        "pad_after_zyx",
        "padded_shape",
        "patch_start_padded_zyx",
        "patch_end_padded_zyx",
    }

    missing_columns = sorted(
        required_columns
        - set(
            selection_df.columns
        )
    )

    if missing_columns:
        raise RuntimeError(
            "Selection CSV is missing columns: "
            f"{missing_columns}"
        )

    if args.max_patients is not None:
        selection_df = selection_df.head(
            args.max_patients
        ).copy()

    if selection_df.empty:
        raise RuntimeError(
            "No selected patients remain."
        )

    selected_conditions = [
        condition
        for condition, specification
        in CONDITIONS.items()
        if (
            condition == "clean"
            or args.only_artifact is None
            or specification["artifact"]
            == args.only_artifact
        )
    ]

    patient_ids = (
        selection_df[
            "patient_id"
        ]
        .astype(str)
        .tolist()
    )

    for patient_id in patient_ids:
        segmentation_path(
            patient_id
        )

        for condition in selected_conditions:
            image_paths(
                patient_id,
                condition,
            )

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )

    predictor.initialize_from_trained_model_folder(
        MODEL_FOLDER,
        use_folds=(0,),
        checkpoint_name="checkpoint_best.pth",
    )

    model = predictor.network.to(
        device
    )

    model.eval()

    target_layer = (
        model.decoder.stages[3]
    )

    preprocessor = (
        predictor
        .configuration_manager
        .preprocessor_class(
            verbose=False
        )
    )

    configuration = {
        "script": (
            "37b_evaluate_nnunet_"
            "gradcam_clean_degraded.py"
        ),
        "model_folder": str(
            MODEL_FOLDER
        ),
        "checkpoint": (
            "fold_0/checkpoint_best.pth"
        ),
        "model_class": (
            type(model).__name__
        ),
        "target_layer": (
            TARGET_LAYER_NAME
        ),
        "target_feature_resolution": [
            64,
            64,
            64,
        ],
        "target_definition": (
            "Mean differentiable tumor logit "
            "inside the model-predicted "
            "whole-tumor region"
        ),
        "patch_size_zyx": list(
            PATCH_SIZE
        ),
        "input_source": (
            "Exact saved clean and degraded "
            "nnU-Net input NIfTI files"
        ),
        "preprocessing": (
            "SimpleITKIO plus official "
            "nnU-Net run_case_npy"
        ),
        "selected_patients": (
            patient_ids
        ),
        "selected_conditions": (
            selected_conditions
        ),
        "device": str(
            device
        ),
        "gpu": torch.cuda.get_device_name(
            device
        ),
        "seed": 2026,
        "coordinate_order": (
            "z, y, x"
        ),
        "high_saliency_percentile": (
            80
        ),
        "global_slice_definition": (
            "Preprocessed-volume axial z, "
            "not raw MRI z"
        ),
        "zero_WT_handling": (
            "Unavailable status, NaN attribution "
            "metrics, zero CAM placeholder"
        ),
        "ringing_reporting_label": (
            "Fourier truncation or low-pass "
            "frequency stress test"
        ),
        "gradcam_interpretation": (
            "Spatial attribution associated with "
            "the target score, not a causal "
            "explanation of model reasoning"
        ),
    }

    configuration_path = (
        output_dir
        / "37b_configuration.json"
    )

    configuration_path.write_text(
        json.dumps(
            configuration,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    selected_copy_path = (
        output_dir
        / "37b_nnunet_xai_"
        "selected_patients.csv"
    )

    selection_df.to_csv(
        selected_copy_path,
        index=False,
    )

    metrics_path = (
        output_dir
        / "37b_nnunet_xai_"
        "condition_metrics.csv"
    )

    metrics_rows: list[
        dict[str, Any]
    ] = []

    start_time = time.perf_counter()

    print("=" * 80)
    print(
        "Script 37B: nnU-Net "
        "clean-versus-degraded Grad-CAM"
    )
    print("=" * 80)
    print(
        f"Patients: {len(patient_ids)}"
    )
    print(
        "Conditions per patient: "
        f"{len(selected_conditions)}"
    )
    print(
        f"Target layer: {TARGET_LAYER_NAME}"
    )
    print(
        f"Output: {output_dir}"
    )
    print("=" * 80)

    for (
        patient_index,
        selection,
    ) in selection_df.reset_index(
        drop=True
    ).iterrows():
        patient_id = str(
            selection[
                "patient_id"
            ]
        )

        pad_before = parse_tuple3(
            selection[
                "pad_before_zyx"
            ]
        )

        pad_after = parse_tuple3(
            selection[
                "pad_after_zyx"
            ]
        )

        expected_preprocessed_shape = (
            parse_tuple3(
                selection[
                    "preprocessed_shape"
                ]
            )
        )

        expected_padded_shape = (
            parse_tuple3(
                selection[
                    "padded_shape"
                ]
            )
        )

        patch_start = parse_tuple3(
            selection[
                "patch_start_padded_zyx"
            ]
        )

        patch_end = parse_tuple3(
            selection[
                "patch_end_padded_zyx"
            ]
        )

        saved_patch_size = tuple(
            patch_end[axis]
            - patch_start[axis]
            for axis in range(3)
        )

        if saved_patch_size != PATCH_SIZE:
            raise RuntimeError(
                "Saved patch is not 128 cubed "
                f"for {patient_id}: "
                f"{patch_start} to {patch_end}"
            )

        raw_segmentation = (
            sitk.GetArrayFromImage(
                sitk.ReadImage(
                    str(
                        segmentation_path(
                            patient_id
                        )
                    )
                )
            )
            .astype(np.int16)[None]
        )

        clean_raw_shape = None
        clean_preprocessed_shape = None
        clean_geometry = None
        clean_segmentation = None
        clean_truth_patch = None
        clean_cam = None
        clean_centroid = None
        clean_status = None
        local_slice = None
        global_slice = None

        for condition in selected_conditions:
            condition_start = (
                time.perf_counter()
            )

            specification = CONDITIONS[
                condition
            ]

            (
                data,
                segmentation,
                properties,
                raw_shape,
            ) = preprocess_case(
                image_paths(
                    patient_id,
                    condition,
                ),
                raw_segmentation,
                preprocessor,
                predictor,
            )

            preprocessed_shape = tuple(
                int(item)
                for item in data.shape[1:]
            )

            segmentation_shape = tuple(
                int(item)
                for item
                in segmentation.shape[1:]
            )

            if (
                segmentation_shape
                != preprocessed_shape
            ):
                raise RuntimeError(
                    "Data and segmentation "
                    "preprocessed shapes differ "
                    f"for {patient_id} "
                    f"{condition}: "
                    f"{data.shape} versus "
                    f"{segmentation.shape}"
                )

            geometry = geometry_signature(
                properties
            )

            if condition == "clean":
                clean_raw_shape = (
                    raw_shape
                )

                clean_preprocessed_shape = (
                    preprocessed_shape
                )

                clean_geometry = (
                    geometry
                )

                clean_segmentation = (
                    segmentation.copy()
                )

                if (
                    preprocessed_shape
                    != expected_preprocessed_shape
                ):
                    raise RuntimeError(
                        "Selection CSV shape mismatch "
                        f"for {patient_id}: "
                        "expected "
                        f"{expected_preprocessed_shape}, "
                        "found "
                        f"{preprocessed_shape}"
                    )
            else:
                if (
                    raw_shape
                    != clean_raw_shape
                ):
                    raise RuntimeError(
                        "Raw shape mismatch for "
                        f"{patient_id} "
                        f"{condition}: "
                        f"{clean_raw_shape} versus "
                        f"{raw_shape}"
                    )

                if (
                    preprocessed_shape
                    != clean_preprocessed_shape
                ):
                    raise RuntimeError(
                        "Preprocessed shape mismatch "
                        f"for {patient_id} "
                        f"{condition}: "
                        f"{clean_preprocessed_shape} "
                        "versus "
                        f"{preprocessed_shape}"
                    )

                if (
                    geometry
                    != clean_geometry
                ):
                    raise RuntimeError(
                        "Preprocessing geometry "
                        f"mismatch for {patient_id} "
                        f"{condition}.\n"
                        f"Clean: {clean_geometry}\n"
                        f"Condition: {geometry}"
                    )

                if not np.array_equal(
                    np.maximum(
                        segmentation,
                        0,
                    ),
                    np.maximum(
                        clean_segmentation,
                        0,
                    ),
                ):
                    raise RuntimeError(
                        "Ground-truth tumor-label "
                        "preprocessing mismatch for "
                        f"{patient_id} {condition}."
                    )

            (
                padded_data,
                padded_segmentation,
            ) = pad_case(
                data,
                segmentation,
                pad_before,
                pad_after,
            )

            padded_shape = tuple(
                int(item)
                for item
                in padded_data.shape[1:]
            )

            if (
                padded_shape
                != expected_padded_shape
            ):
                raise RuntimeError(
                    "Padded shape mismatch for "
                    f"{patient_id} {condition}: "
                    f"expected "
                    f"{expected_padded_shape}, "
                    f"found {padded_shape}"
                )

            if (
                tuple(
                    padded_segmentation.shape[1:]
                )
                != padded_shape
            ):
                raise RuntimeError(
                    "Padded data and segmentation "
                    f"mismatch for {patient_id} "
                    f"{condition}."
                )

            (
                z0,
                y0,
                x0,
            ) = patch_start

            (
                z1,
                y1,
                x1,
            ) = patch_end

            image_patch = padded_data[
                :,
                z0:z1,
                y0:y1,
                x0:x1,
            ]

            truth_patch = padded_segmentation[
                0,
                z0:z1,
                y0:y1,
                x0:x1,
            ]

            if image_patch.shape != (
                4,
                *PATCH_SIZE,
            ):
                raise RuntimeError(
                    "Unexpected image patch "
                    f"for {patient_id} "
                    f"{condition}: "
                    f"{image_patch.shape}"
                )

            if truth_patch.shape != PATCH_SIZE:
                raise RuntimeError(
                    "Unexpected truth patch "
                    f"for {patient_id} "
                    f"{condition}: "
                    f"{truth_patch.shape}"
                )

            if condition == "clean":
                clean_truth_patch = (
                    truth_patch.copy()
                )

                local_slice = (
                    select_axial_slice(
                        truth_patch
                    )
                )

                global_slice = (
                    patch_start[0]
                    + local_slice
                    - pad_before[0]
                )

                if not (
                    0
                    <= global_slice
                    < preprocessed_shape[0]
                ):
                    raise RuntimeError(
                        "Invalid preprocessed "
                        f"global slice for "
                        f"{patient_id}: "
                        f"{global_slice}"
                    )
            elif not np.array_equal(
                np.maximum(
                    truth_patch,
                    0,
                ),
                np.maximum(
                    clean_truth_patch,
                    0,
                ),
            ):
                raise RuntimeError(
                    "Fixed tumor-label patch "
                    f"mismatch for {patient_id} "
                    f"{condition}."
                )

            if (
                local_slice is None
                or global_slice is None
            ):
                raise RuntimeError(
                    "Clean representative slice "
                    "was not initialized."
                )

            result = run_gradcam(
                model=model,
                target_layer=target_layer,
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

            gradcam_status = str(
                result[
                    "gradcam_status"
                ]
            )

            if condition == "clean":
                clean_cam = cam.copy()
                clean_centroid = centroid.copy()
                clean_status = gradcam_status

                if gradcam_status == "available":
                    similarity = 1.0
                    centroid_shift = 0.0
                else:
                    similarity = float("nan")
                    centroid_shift = float("nan")

            elif (
                clean_status == "available"
                and gradcam_status == "available"
            ):
                similarity = heatmap_similarity(
                    clean_cam,
                    cam,
                )

                if (
                    np.isnan(
                        clean_centroid
                    ).any()
                    or np.isnan(
                        centroid
                    ).any()
                ):
                    centroid_shift = float("nan")
                else:
                    centroid_shift = float(
                        np.linalg.norm(
                            centroid
                            - clean_centroid
                        )
                    )

            else:
                similarity = float("nan")
                centroid_shift = float("nan")

            array_path = (
                arrays_dir
                / (
                    f"37b_{patient_id}_"
                    f"{condition}_gradcam.npy"
                )
            )

            figure_path = (
                figures_dir
                / (
                    f"37b_{patient_id}_"
                    f"{condition}_gradcam.png"
                )
            )

            np.save(
                array_path,
                cam,
            )

            save_condition_figure(
                output_path=figure_path,
                patient_id=patient_id,
                condition=condition,
                gradcam_status=gradcam_status,
                global_slice=global_slice,
                image_patch=image_patch,
                truth_patch=truth_patch,
                prediction=prediction,
                cam=cam,
                local_slice=local_slice,
            )

            elapsed_seconds = (
                time.perf_counter()
                - condition_start
            )

            metrics_rows.append(
                {
                    "patient_id": patient_id,
                    "selection_category": (
                        selection[
                            "selection_category"
                        ]
                    ),
                    "condition": condition,
                    "artifact": (
                        specification[
                            "artifact"
                        ]
                    ),
                    "level": (
                        specification[
                            "level"
                        ]
                    ),
                    "raw_shape_czyx": str(
                        raw_shape
                    ),
                    (
                        "original_preprocessed_"
                        "shape_zyx"
                    ): str(
                        preprocessed_shape
                    ),
                    "preprocessing_bbox_zyx": str(
                        geometry[
                            "bbox_used_for_cropping"
                        ]
                    ),
                    "pad_before_zyx": str(
                        pad_before
                    ),
                    "pad_after_zyx": str(
                        pad_after
                    ),
                    "padded_shape_zyx": str(
                        padded_shape
                    ),
                    "patch_start_padded_zyx": str(
                        patch_start
                    ),
                    "patch_end_padded_zyx": str(
                        patch_end
                    ),
                    (
                        "local_axial_slice_"
                        "in_patch"
                    ): local_slice,
                    (
                        "global_axial_slice_"
                        "preprocessed"
                    ): global_slice,
                    **result,
                    (
                        "clean_to_condition_"
                        "heatmap_correlation"
                    ): similarity,
                    (
                        "attribution_centroid_"
                        "shift_voxels"
                    ): centroid_shift,
                    "centroid_z": float(
                        centroid[0]
                    ),
                    "centroid_y": float(
                        centroid[1]
                    ),
                    "centroid_x": float(
                        centroid[2]
                    ),
                    "array_path": str(
                        array_path
                    ),
                    "figure_path": str(
                        figure_path
                    ),
                    "gradcam_seconds": (
                        elapsed_seconds
                    ),
                }
            )

            pd.DataFrame(
                metrics_rows
            ).to_csv(
                metrics_path,
                index=False,
            )

            print(
                f"[{patient_index + 1}/"
                f"{len(patient_ids)}] "
                f"{patient_id} | "
                f"{condition} | "
                f"status={gradcam_status} | "
                f"patch WT Dice="
                f"{metric_text(result['patch_dice_WT'])} | "
                f"inside/outside="
                f"{metric_text(result['inside_outside_ratio'])} | "
                f"similarity="
                f"{metric_text(similarity)}"
            )

    metrics_df = pd.DataFrame(
        metrics_rows
    )

    expected_rows = (
        len(patient_ids)
        * len(selected_conditions)
    )

    if (
        len(metrics_df)
        != expected_rows
    ):
        raise RuntimeError(
            f"Expected {expected_rows} rows, "
            f"found {len(metrics_df)}."
        )

    duplicate_count = int(
        metrics_df.duplicated(
            subset=[
                "patient_id",
                "condition",
            ]
        ).sum()
    )

    if duplicate_count:
        raise RuntimeError(
            "Duplicate patient-condition rows "
            f"found: {duplicate_count}"
        )

    array_count = len(
        list(
            arrays_dir.glob(
                "*.npy"
            )
        )
    )

    figure_count = len(
        list(
            figures_dir.glob(
                "*.png"
            )
        )
    )

    if (
        array_count != expected_rows
        or figure_count != expected_rows
    ):
        raise RuntimeError(
            "Unexpected output counts. "
            f"Expected {expected_rows}, found "
            f"{array_count} arrays and "
            f"{figure_count} figures."
        )

    unavailable_count = int(
        (
            metrics_df[
                "gradcam_status"
            ]
            != "available"
        ).sum()
    )

    total_seconds = (
        time.perf_counter()
        - start_time
    )

    summary_lines = [
        "=" * 80,
        (
            "Script 37B: nnU-Net "
            "clean-versus-degraded "
            "Grad-CAM summary"
        ),
        "=" * 80,
        "",
        (
            f"Patients: "
            f"{len(patient_ids)}"
        ),
        (
            "Conditions per patient: "
            f"{len(selected_conditions)}"
        ),
        (
            f"Rows: "
            f"{len(metrics_df)}"
        ),
        (
            f"Arrays: "
            f"{array_count}"
        ),
        (
            f"Figures: "
            f"{figure_count}"
        ),
        (
            "Unavailable attribution maps: "
            f"{unavailable_count}"
        ),
        (
            f"Model: "
            f"{type(model).__name__}"
        ),
        (
            "Checkpoint: "
            "fold_0/checkpoint_best.pth"
        ),
        (
            f"Target layer: "
            f"{TARGET_LAYER_NAME}"
        ),
        (
            "Target definition: mean "
            "differentiable tumor logit "
            "inside the model-predicted "
            "whole-tumor region"
        ),
        (
            "Input source: exact saved "
            "clean and degraded nnU-Net "
            "input NIfTI files"
        ),
        (
            "Preprocessing: SimpleITKIO "
            "plus official nnU-Net "
            "run_case_npy"
        ),
        (
            "Ground truth was used for "
            "fixed patch placement, "
            "representative slice selection, "
            "and descriptive evaluation only."
        ),
        (
            "Global axial slice values are "
            "preprocessed-volume coordinates, "
            "not raw MRI coordinates."
        ),
        (
            "Grad-CAM provides a spatial "
            "attribution map associated with "
            "the selected target score. It "
            "does not establish a causal "
            "explanation of model reasoning."
        ),
        (
            "The internal ringing condition "
            "should be reported as Fourier "
            "truncation or a low-pass "
            "frequency stress test."
        ),
        "",
        "Mean metrics by condition",
        "-" * 80,
    ]

    for (
        condition,
        group,
    ) in metrics_df.groupby(
        "condition",
        sort=False,
    ):
        summary_lines.append(
            (
                f"{condition} | "
                "patch WT Dice "
                f"{group['patch_dice_WT'].mean():.4f} | "
                "inside/outside "
                f"{group['inside_outside_ratio'].mean():.3f} | "
                "high-saliency WT IoU "
                f"{group['high_saliency_WT_iou'].mean():.3f} | "
                "clean correlation "
                f"{group['clean_to_condition_heatmap_correlation'].mean():.3f} | "
                "centroid shift "
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
            (
                f"Metrics: "
                f"{metrics_path}"
            ),
            (
                f"Configuration: "
                f"{configuration_path}"
            ),
            (
                f"Selected patients: "
                f"{selected_copy_path}"
            ),
            (
                f"Figures: "
                f"{figures_dir}"
            ),
            (
                f"Arrays: "
                f"{arrays_dir}"
            ),
            "=" * 80,
        ]
    )

    summary_path = (
        output_dir
        / "37b_nnunet_xai_summary.txt"
    )

    summary_path.write_text(
        "\n".join(
            summary_lines
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "\n"
        + "\n".join(
            summary_lines
        )
    )


if __name__ == "__main__":
    main()
