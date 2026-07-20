#!/usr/bin/env python3
"""
Script 34A: Train Swin-UNETR on clean BraTS2020 MRI volumes.

Methodological rules
--------------------
- Train only on the fixed 235-patient clean training cohort.
- Select checkpoints only with full-volume validation on the fixed
  59-patient clean validation cohort.
- Never use the 74-patient held-out test cohort for checkpoint selection.
- Training uses balanced 96x96x96 patches.
- Validation uses MONAI full-volume sliding-window inference.
- Report WT, TC, and ET Dice/IoU, plus WT voxel counts.

Checkpoint selection
--------------------
The best checkpoint is selected using macro regional Dice:

    mean(WT Dice, TC Dice, ET Dice)

This prevents good whole-tumor performance from hiding poor TC or ET
performance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import SwinUNETR
from torch.utils.data import DataLoader, Dataset, get_worker_info


# ============================================================================
# Project configuration
# ============================================================================

PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

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

TRAIN_IDS_FILE = (
    PROJECT_ROOT
    / "report_materials"
    / "31a_swin_unetr_train_patient_ids.csv"
)

VAL_IDS_FILE = (
    PROJECT_ROOT
    / "report_materials"
    / "31a_swin_unetr_validation_patient_ids.csv"
)

EXPECTED_TRAIN_ID_HASH = (
    "cb70809e06f62c422e83b6d73193117c"
    "589bec77c8a0f719a0998d2f5f34c655"
)

EXPECTED_VAL_ID_HASH = (
    "c5831f45d2c0873066a7c7846ad7723"
    "b6fa312f880858b589858015062787efd"
)

REPORT_DIR = PROJECT_ROOT / "report_materials"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_ROOT = PROJECT_ROOT / "models" / "swin_unetr"

MODALITIES = ["flair", "t1", "t1ce", "t2"]
REQUIRED_COLUMNS = ["patient_id", *MODALITIES, "seg"]

PATCH_SIZE = (96, 96, 96)
NUM_CLASSES = 4
IN_CHANNELS = 4

TUMOR_PROBABILITY = 0.50
MIXED_PROBABILITY = 0.25
TUMOR_FREE_PROBABILITY = 0.25
MAX_TUMOR_FREE_ATTEMPTS = 200


# ============================================================================
# Command-line arguments
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Swin-UNETR on clean BraTS2020 data with "
            "full-volume validation."
        )
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="swin_unetr_clean",
        help="Unique name used for checkpoints and logs.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help=(
            "Final epoch number. Default is deliberately 1 to prevent "
            "an accidental long run."
        ),
    )

    parser.add_argument(
        "--validate-every",
        type=int,
        default=1,
        help="Run full-volume validation every N epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Begin with 0 for stability; increase only after timing tests.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help=(
            "Device visible inside the process. When CUDA_VISIBLE_DEVICES "
            "is used, the selected physical GPU normally becomes cuda:0."
        ),
    )

    parser.add_argument(
        "--sw-batch-size",
        type=int,
        default=1,
        help="Number of sliding-window patches inferred together.",
    )

    parser.add_argument(
        "--sw-overlap",
        type=float,
        default=0.50,
        help="Fractional overlap for full-volume validation.",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help=(
            "Checkpoint path to resume from, or 'auto' to use the run's "
            "last checkpoint."
        ),
    )

    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help=(
            "Optional smoke-test limit. Zero means all training batches."
        ),
    )

    parser.add_argument(
        "--max-val-patients",
        type=int,
        default=0,
        help=(
            "Optional smoke-test limit. Zero means all 59 validation "
            "patients."
        ),
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision.",
    )

    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")

    if args.validate_every < 1:
        raise ValueError("--validate-every must be at least 1.")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")

    if args.sw_batch_size < 1:
        raise ValueError("--sw-batch-size must be at least 1.")

    if not 0.0 <= args.sw_overlap < 1.0:
        raise ValueError("--sw-overlap must be in [0, 1).")

    if args.max_train_batches < 0:
        raise ValueError("--max-train-batches cannot be negative.")

    if args.max_val_patients < 0:
        raise ValueError("--max-val-patients cannot be negative.")

    if not args.run_name.strip():
        raise ValueError("--run-name cannot be empty.")

    return args


# ============================================================================
# Reproducibility and file verification
# ============================================================================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    label: str,
) -> None:
    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}"
        )


def verify_data_paths(
    dataframe: pd.DataFrame,
    label: str,
) -> None:
    missing_paths: list[str] = []

    for column in [*MODALITIES, "seg"]:
        for value in dataframe[column]:
            path = Path(str(value))

            if not path.exists():
                missing_paths.append(str(path))

                if len(missing_paths) >= 20:
                    break

        if len(missing_paths) >= 20:
            break

    if missing_paths:
        formatted = "\n".join(missing_paths)
        raise FileNotFoundError(
            f"{label} contains missing image paths. "
            f"First missing paths:\n{formatted}"
        )


def verify_fixed_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    required_files = [
        TRAIN_CSV,
        VAL_CSV,
        TEST_CSV,
        TRAIN_IDS_FILE,
        VAL_IDS_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required split file not found: {path}"
            )

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    train_ids_saved = pd.read_csv(TRAIN_IDS_FILE)
    val_ids_saved = pd.read_csv(VAL_IDS_FILE)

    require_columns(train_df, REQUIRED_COLUMNS, "Training CSV")
    require_columns(val_df, REQUIRED_COLUMNS, "Validation CSV")
    require_columns(test_df, REQUIRED_COLUMNS, "Test CSV")

    require_columns(
        train_ids_saved,
        ["patient_id"],
        "Portable training-ID CSV",
    )

    require_columns(
        val_ids_saved,
        ["patient_id"],
        "Portable validation-ID CSV",
    )

    expected_counts = {
        "training": (len(train_df), 235),
        "validation": (len(val_df), 59),
        "test": (len(test_df), 74),
    }

    for label, (observed, expected) in expected_counts.items():
        if observed != expected:
            raise RuntimeError(
                f"Expected {expected} {label} patients, "
                f"found {observed}."
            )

    for label, dataframe in [
        ("training", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:
        if dataframe["patient_id"].duplicated().any():
            duplicates = dataframe.loc[
                dataframe["patient_id"].duplicated(keep=False),
                "patient_id",
            ].tolist()

            raise RuntimeError(
                f"Duplicate patient IDs in {label} data: {duplicates}"
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

    overlaps = {
        "train-validation": train_ids & val_ids,
        "train-test": train_ids & test_ids,
        "validation-test": val_ids & test_ids,
    }

    for label, overlap in overlaps.items():
        if overlap:
            raise RuntimeError(
                f"Patient overlap detected for {label}: "
                f"{sorted(overlap)}"
            )

    saved_train_ids = set(
        train_ids_saved["patient_id"].astype(str)
    )

    saved_val_ids = set(
        val_ids_saved["patient_id"].astype(str)
    )

    if train_ids != saved_train_ids:
        raise RuntimeError(
            "Training path CSV does not match the portable "
            "training patient-ID record."
        )

    if val_ids != saved_val_ids:
        raise RuntimeError(
            "Validation path CSV does not match the portable "
            "validation patient-ID record."
        )

    train_hash = sha256_file(TRAIN_IDS_FILE)
    val_hash = sha256_file(VAL_IDS_FILE)

    if train_hash != EXPECTED_TRAIN_ID_HASH:
        raise RuntimeError(
            "Training patient-ID hash does not match the "
            "approved fixed split."
        )

    if val_hash != EXPECTED_VAL_ID_HASH:
        raise RuntimeError(
            "Validation patient-ID hash does not match the "
            "approved fixed split."
        )

    verify_data_paths(train_df, "Training CSV")
    verify_data_paths(val_df, "Validation CSV")

    print("\nFixed split verification")
    print("-" * 80)
    print(f"Training patients:   {len(train_df)}")
    print(f"Validation patients: {len(val_df)}")
    print(f"Held-out test:       {len(test_df)}")
    print("Train-validation overlap: 0")
    print("Train-test overlap:       0")
    print("Validation-test overlap:  0")
    print(f"Training ID hash:   {train_hash}")
    print(f"Validation ID hash: {val_hash}")
    print("Training and validation image paths: verified")

    return train_df, val_df


# ============================================================================
# Image loading, normalization, and patch sampling
# ============================================================================

def load_nifti_float(path: str | Path) -> np.ndarray:
    return nib.load(
        str(path)
    ).get_fdata(
        dtype=np.float32
    )


def normalize_nonzero(image: np.ndarray) -> np.ndarray:
    """
    Normalize one modality using z-score normalization over nonzero voxels.

    Background remains exactly zero.
    """
    image = image.astype(np.float32)
    mask = image != 0

    if not np.any(mask):
        raise ValueError(
            "MRI modality contains no nonzero voxels."
        )

    mean = float(image[mask].mean())
    std = float(image[mask].std())

    if std < 1e-8:
        raise ValueError(
            "MRI modality standard deviation is too small."
        )

    output = np.zeros_like(
        image,
        dtype=np.float32,
    )

    output[mask] = (
        image[mask] - mean
    ) / std

    return output


def remap_segmentation(
    segmentation: np.ndarray,
) -> np.ndarray:
    segmentation = np.rint(
        segmentation
    ).astype(np.int64)

    unexpected = set(
        np.unique(segmentation).tolist()
    ) - {0, 1, 2, 4}

    if unexpected:
        raise ValueError(
            f"Unexpected BraTS labels: {sorted(unexpected)}"
        )

    segmentation[
        segmentation == 4
    ] = 3

    return segmentation


def crop_patch(
    array: np.ndarray,
    center: tuple[int, int, int],
    patch_size: tuple[int, int, int],
) -> np.ndarray:
    spatial_shape = array.shape[-3:]
    starts: list[int] = []

    for coordinate, size, full_size in zip(
        center,
        patch_size,
        spatial_shape,
    ):
        if full_size < size:
            raise ValueError(
                f"Volume dimension {full_size} is smaller "
                f"than patch dimension {size}."
            )

        start = int(
            coordinate - size // 2
        )

        start = max(
            0,
            min(
                start,
                full_size - size,
            ),
        )

        starts.append(start)

    slices = tuple(
        slice(
            start,
            start + size,
        )
        for start, size in zip(
            starts,
            patch_size,
        )
    )

    if array.ndim == 4:
        return array[
            (slice(None),) + slices
        ]

    if array.ndim == 3:
        return array[slices]

    raise ValueError(
        f"Unsupported array shape: {array.shape}"
    )


class BraTSMixedPatchDataset(Dataset):
    """
    One sampled patch per patient per epoch.

    Sampling probabilities:
    - 50% tumor-centered
    - 25% mixed/random brain
    - 25% guaranteed tumor-free brain
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        patch_size: tuple[int, int, int] = PATCH_SIZE,
        seed: int = 42,
    ) -> None:
        self.dataframe = dataframe.reset_index(
            drop=True
        )

        self.patch_size = patch_size
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.dataframe)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def make_rng(
        self,
        index: int,
    ) -> np.random.Generator:
        worker_info = get_worker_info()
        worker_id = (
            worker_info.id
            if worker_info is not None
            else 0
        )

        derived_seed = (
            self.seed
            + self.epoch * 1_000_003
            + int(index) * 9_176
            + worker_id * 37
        )

        return np.random.default_rng(
            derived_seed
        )

    @staticmethod
    def choose_center(
        coordinates: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[int, int, int]:
        if len(coordinates) == 0:
            raise RuntimeError(
                "No candidate coordinates were available."
            )

        index = int(
            rng.integers(
                0,
                len(coordinates),
            )
        )

        selected = coordinates[index]

        return (
            int(selected[0]),
            int(selected[1]),
            int(selected[2]),
        )

    def choose_tumor_free_center(
        self,
        segmentation: np.ndarray,
        brain_coordinates: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[tuple[int, int, int], int]:
        # Fast path: try randomly sampled brain centers first.
        for attempt in range(
            1,
            MAX_TUMOR_FREE_ATTEMPTS + 1,
        ):
            center = self.choose_center(
                brain_coordinates,
                rng,
            )

            target_patch = crop_patch(
                segmentation,
                center,
                self.patch_size,
            )

            if int(
                np.sum(target_patch > 0)
            ) == 0:
                return center, attempt

        # Exact fallback: find every valid tumor-free window using a
        # summed-volume table. This avoids failing merely because the
        # random search missed a relatively small valid region.
        tumor_mask = (
            segmentation > 0
        ).astype(np.int32)

        ph, pw, pd = self.patch_size

        integral = np.pad(
            tumor_mask,
            ((1, 0), (1, 0), (1, 0)),
        ).cumsum(0).cumsum(1).cumsum(2)

        window_tumor = (
            integral[ph:, pw:, pd:]
            - integral[:-ph, pw:, pd:]
            - integral[ph:, :-pw, pd:]
            - integral[ph:, pw:, :-pd]
            + integral[:-ph, :-pw, pd:]
            + integral[:-ph, pw:, :-pd]
            + integral[ph:, :-pw, :-pd]
            - integral[:-ph, :-pw, :-pd]
        )

        brain_mask = np.zeros_like(
            segmentation,
            dtype=bool,
        )

        brain_mask[
            tuple(
                brain_coordinates.T
            )
        ] = True

        center_brain = brain_mask[
            ph // 2 : ph // 2 + window_tumor.shape[0],
            pw // 2 : pw // 2 + window_tumor.shape[1],
            pd // 2 : pd // 2 + window_tumor.shape[2],
        ]

        valid_starts = np.argwhere(
            (window_tumor == 0)
            & center_brain
        )

        if len(valid_starts) == 0:
            raise RuntimeError(
                "This patient has no valid tumor-free "
                f"{self.patch_size} brain patch."
            )

        selected_start = valid_starts[
            int(
                rng.integers(
                    0,
                    len(valid_starts),
                )
            )
        ]

        center_array = (
            selected_start
            + np.asarray(
                self.patch_size,
                dtype=np.int64,
            ) // 2
        )

        center = tuple(
            int(value)
            for value in center_array
        )

        return (
            center,
            MAX_TUMOR_FREE_ATTEMPTS + 1,
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        row = self.dataframe.iloc[index]
        rng = self.make_rng(index)

        modalities: list[np.ndarray] = []
        raw_brain_masks: list[np.ndarray] = []

        for modality in MODALITIES:
            raw_image = load_nifti_float(
                row[modality]
            )

            raw_brain_masks.append(
                raw_image != 0
            )

            modalities.append(
                normalize_nonzero(
                    raw_image
                )
            )

        image_stack = np.stack(
            modalities,
            axis=0,
        ).astype(np.float32)

        brain_mask = np.any(
            np.stack(
                raw_brain_masks,
                axis=0,
            ),
            axis=0,
        )

        segmentation = remap_segmentation(
            load_nifti_float(
                row["seg"]
            )
        )

        tumor_coordinates = np.argwhere(
            segmentation > 0
        )

        brain_coordinates = np.argwhere(
            brain_mask
        )

        draw = float(
            rng.random()
        )

        if draw < TUMOR_PROBABILITY:
            patch_type = "tumor-centered"

            center = self.choose_center(
                tumor_coordinates,
                rng,
            )

            search_attempts = 0

        elif draw < (
            TUMOR_PROBABILITY
            + MIXED_PROBABILITY
        ):
            patch_type = "mixed-random"

            center = self.choose_center(
                brain_coordinates,
                rng,
            )

            search_attempts = 0

        else:
            patch_type = "tumor-free"

            try:
                center, search_attempts = (
                    self.choose_tumor_free_center(
                        segmentation,
                        brain_coordinates,
                        rng,
                    )
                )
            except RuntimeError as error:
                if (
                    "no valid tumor-free"
                    not in str(error)
                ):
                    raise

                patch_type = "mixed-random-fallback"

                center = self.choose_center(
                    brain_coordinates,
                    rng,
                )

                search_attempts = (
                    MAX_TUMOR_FREE_ATTEMPTS + 2
                )

        image_patch = crop_patch(
            image_stack,
            center,
            self.patch_size,
        )

        target_patch = crop_patch(
            segmentation,
            center,
            self.patch_size,
        )

        if image_patch.shape != (
            IN_CHANNELS,
            *self.patch_size,
        ):
            raise RuntimeError(
                f"Unexpected image patch shape: "
                f"{image_patch.shape}"
            )

        if target_patch.shape != self.patch_size:
            raise RuntimeError(
                f"Unexpected target patch shape: "
                f"{target_patch.shape}"
            )

        tumor_voxels = int(
            np.sum(target_patch > 0)
        )

        if (
            patch_type == "tumor-free"
            and tumor_voxels != 0
        ):
            raise RuntimeError(
                "A tumor-free patch contains tumor."
            )

        return {
            "image": torch.from_numpy(
                image_patch.copy()
            ).float(),

            "target": torch.from_numpy(
                target_patch.copy()
            ).long(),

            "patient_id": str(
                row["patient_id"]
            ),

            "patch_type": patch_type,

            "tumor_voxels": tumor_voxels,

            "search_attempts": int(
                search_attempts
            ),
        }


def seed_worker(worker_id: int) -> None:
    worker_seed = (
        torch.initial_seed()
        % (2 ** 32)
    )

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_train_loader(
    dataset: BraTSMixedPatchDataset,
    args: argparse.Namespace,
    epoch: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(
        args.seed + epoch
    )

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=False,
    )


def load_full_patient(
    row: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    modalities = []

    for modality in MODALITIES:
        modalities.append(
            normalize_nonzero(
                load_nifti_float(
                    row[modality]
                )
            )
        )

    image = np.stack(
        modalities,
        axis=0,
    ).astype(np.float32)

    segmentation = remap_segmentation(
        load_nifti_float(
            row["seg"]
        )
    ).astype(np.int16)

    return image, segmentation


# ============================================================================
# WT, TC, and ET metrics
# ============================================================================

def dice_score(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
) -> float:
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)

    pred_sum = int(
        pred_mask.sum()
    )

    true_sum = int(
        true_mask.sum()
    )

    if pred_sum == 0 and true_sum == 0:
        return 1.0

    denominator = pred_sum + true_sum

    if denominator == 0:
        return 0.0

    intersection = int(
        np.logical_and(
            pred_mask,
            true_mask,
        ).sum()
    )

    return float(
        (2.0 * intersection)
        / denominator
    )


def iou_score(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
) -> float:
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)

    union = int(
        np.logical_or(
            pred_mask,
            true_mask,
        ).sum()
    )

    if union == 0:
        return 1.0

    intersection = int(
        np.logical_and(
            pred_mask,
            true_mask,
        ).sum()
    )

    return float(
        intersection / union
    )


def compute_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
) -> dict[str, float | int]:
    """
    BraTS project regions:

    WT = labels 1 + 2 + 3
    TC = labels 1 + 3
    ET = label 3
    """
    pred_wt = prediction > 0
    true_wt = truth > 0

    pred_tc = np.logical_or(
        prediction == 1,
        prediction == 3,
    )

    true_tc = np.logical_or(
        truth == 1,
        truth == 3,
    )

    pred_et = prediction == 3
    true_et = truth == 3

    return {
        "dice_WT": dice_score(
            pred_wt,
            true_wt,
        ),
        "iou_WT": iou_score(
            pred_wt,
            true_wt,
        ),
        "dice_TC": dice_score(
            pred_tc,
            true_tc,
        ),
        "iou_TC": iou_score(
            pred_tc,
            true_tc,
        ),
        "dice_ET": dice_score(
            pred_et,
            true_et,
        ),
        "iou_ET": iou_score(
            pred_et,
            true_et,
        ),
        "pred_WT_voxels": int(
            pred_wt.sum()
        ),
        "true_WT_voxels": int(
            true_wt.sum()
        ),
    }


# ============================================================================
# Full-volume validation
# ============================================================================

@torch.no_grad()
def validate_full_volume(
    model: torch.nn.Module,
    val_df: pd.DataFrame,
    device: torch.device,
    amp_enabled: bool,
    sw_batch_size: int,
    sw_overlap: float,
    max_val_patients: int,
    epoch: int,
    run_name: str,
) -> tuple[dict[str, float], float, Path]:
    model.eval()

    if max_val_patients > 0:
        evaluation_df = val_df.iloc[
            :max_val_patients
        ].copy()
    else:
        evaluation_df = val_df.copy()

    rows: list[dict[str, Any]] = []
    validation_start = time.perf_counter()

    print("\n" + "=" * 80)
    print(
        f"Full-volume validation: epoch {epoch}"
    )
    print("=" * 80)
    print(
        f"Validation patients: {len(evaluation_df)}"
    )
    print(f"ROI size: {PATCH_SIZE}")
    print(f"Sliding-window batch size: {sw_batch_size}")
    print(f"Sliding-window overlap: {sw_overlap}")
    print("Blend mode: gaussian")
    print("Window inference device: GPU")
    print("Output stitching device: CPU")

    def predictor(
        window: torch.Tensor,
    ) -> torch.Tensor:
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            return model(window)

    for patient_number, (_, row) in enumerate(
        evaluation_df.iterrows(),
        start=1,
    ):
        patient_start = time.perf_counter()
        patient_id = str(
            row["patient_id"]
        )

        image, truth = load_full_patient(row)

        input_tensor = torch.from_numpy(
            image
        ).unsqueeze(0).float()

        logits = sliding_window_inference(
            inputs=input_tensor,
            roi_size=PATCH_SIZE,
            sw_batch_size=sw_batch_size,
            predictor=predictor,
            overlap=sw_overlap,
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

        metrics = compute_metrics(
            prediction,
            truth,
        )

        patient_seconds = (
            time.perf_counter()
            - patient_start
        )

        rows.append({
            "epoch": epoch,
            "patient_id": patient_id,
            **metrics,
            "inference_seconds": patient_seconds,
        })

        print(
            f"[{patient_number:02d}/"
            f"{len(evaluation_df):02d}] "
            f"{patient_id} | "
            f"WT={metrics['dice_WT']:.4f} | "
            f"TC={metrics['dice_TC']:.4f} | "
            f"ET={metrics['dice_ET']:.4f} | "
            f"{patient_seconds:.1f}s"
        )

        del input_tensor
        del logits
        del prediction

        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics_df = pd.DataFrame(rows)

    if metrics_df.empty:
        raise RuntimeError(
            "Validation produced no patient metrics."
        )

    validation_output = (
        RESULTS_DIR
        / (
            f"34a_{run_name}_"
            f"validation_epoch_{epoch:03d}_metrics.csv"
        )
    )

    metrics_df.to_csv(
        validation_output,
        index=False,
    )

    mean_wt_dice = float(
        metrics_df["dice_WT"].mean()
    )

    mean_tc_dice = float(
        metrics_df["dice_TC"].mean()
    )

    mean_et_dice = float(
        metrics_df["dice_ET"].mean()
    )

    summary = {
        "val_dice_WT": mean_wt_dice,
        "val_iou_WT": float(
            metrics_df["iou_WT"].mean()
        ),
        "val_dice_TC": mean_tc_dice,
        "val_iou_TC": float(
            metrics_df["iou_TC"].mean()
        ),
        "val_dice_ET": mean_et_dice,
        "val_iou_ET": float(
            metrics_df["iou_ET"].mean()
        ),
        "val_macro_dice": float(
            np.mean([
                mean_wt_dice,
                mean_tc_dice,
                mean_et_dice,
            ])
        ),
        "val_pred_WT_voxels_mean": float(
            metrics_df[
                "pred_WT_voxels"
            ].mean()
        ),
        "val_true_WT_voxels_mean": float(
            metrics_df[
                "true_WT_voxels"
            ].mean()
        ),
    }

    validation_seconds = (
        time.perf_counter()
        - validation_start
    )

    print("\nValidation summary")
    print("-" * 80)
    print(
        f"WT Dice:    {summary['val_dice_WT']:.6f}"
    )
    print(
        f"TC Dice:    {summary['val_dice_TC']:.6f}"
    )
    print(
        f"ET Dice:    {summary['val_dice_ET']:.6f}"
    )
    print(
        f"Macro Dice: {summary['val_macro_dice']:.6f}"
    )
    print(
        f"Validation time: "
        f"{validation_seconds / 60:.2f} minutes"
    )
    print(
        f"Patient metrics: {validation_output}"
    )

    model.train()

    return (
        summary,
        validation_seconds,
        validation_output,
    )


# ============================================================================
# Logging and checkpoint helpers
# ============================================================================

def append_history_row(
    path: Path,
    row: dict[str, Any],
) -> None:
    file_exists = path.exists()

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(row.keys()),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def atomic_torch_save(
    payload: dict[str, Any],
    path: Path,
) -> None:
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        payload,
        temporary_path,
    )

    temporary_path.replace(path)


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            default=str,
        )

    temporary_path.replace(path)


def write_summary(
    path: Path,
    args: argparse.Namespace,
    current_epoch: int,
    best_epoch: int,
    best_val_macro_dice: float,
    last_history_row: dict[str, Any],
    best_checkpoint: Path,
    last_checkpoint: Path,
) -> None:
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write("=" * 80 + "\n")
        file.write(
            "Script 34A: Swin-UNETR clean training summary\n"
        )
        file.write("=" * 80 + "\n\n")

        file.write(
            f"Run name: {args.run_name}\n"
        )
        file.write(
            f"Current epoch: {current_epoch}\n"
        )
        file.write(
            f"Requested final epoch: {args.epochs}\n"
        )
        file.write(
            f"Best epoch: {best_epoch}\n"
        )
        file.write(
            f"Best validation macro Dice: "
            f"{best_val_macro_dice:.6f}\n\n"
        )

        file.write("Architecture\n")
        file.write("-" * 80 + "\n")
        file.write("Model: MONAI SwinUNETR\n")
        file.write("Input channels: 4\n")
        file.write("Output classes: 4\n")
        file.write("Feature size: 24\n")
        file.write("Use checkpointing: False\n")
        file.write(
            f"Patch size: {PATCH_SIZE}\n\n"
        )

        file.write("Training sampler\n")
        file.write("-" * 80 + "\n")
        file.write("Tumor-centered: 50%\n")
        file.write("Mixed/random brain: 25%\n")
        file.write("Guaranteed tumor-free brain: 25%\n\n")

        file.write("Data split\n")
        file.write("-" * 80 + "\n")
        file.write("Training patients: 235\n")
        file.write("Validation patients: 59\n")
        file.write("Held-out test patients: 74\n")
        file.write(
            "The held-out test cohort is not used for "
            "checkpoint selection.\n\n"
        )

        file.write("Validation\n")
        file.write("-" * 80 + "\n")
        file.write(
            "Full-volume MONAI sliding-window inference\n"
        )
        file.write(
            f"ROI size: {PATCH_SIZE}\n"
        )
        file.write(
            f"Overlap: {args.sw_overlap}\n"
        )
        file.write("Blend mode: Gaussian\n")
        file.write(
            "Best checkpoint criterion: mean of "
            "WT, TC, and ET validation Dice\n\n"
        )

        file.write("Latest epoch record\n")
        file.write("-" * 80 + "\n")

        for key, value in last_history_row.items():
            file.write(
                f"{key}: {value}\n"
            )

        file.write("\nCheckpoints\n")
        file.write("-" * 80 + "\n")
        file.write(
            f"Best: {best_checkpoint}\n"
        )
        file.write(
            f"Last: {last_checkpoint}\n"
        )

    temporary_path.replace(path)


# ============================================================================
# Training
# ============================================================================

def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    print("=" * 80)
    print(
        "Script 34A: Train Swin-UNETR on clean BraTS2020"
    )
    print("=" * 80)

    probability_sum = (
        TUMOR_PROBABILITY
        + MIXED_PROBABILITY
        + TUMOR_FREE_PROBABILITY
    )

    if not np.isclose(
        probability_sum,
        1.0,
    ):
        raise RuntimeError(
            "Patch sampling probabilities do not sum to 1."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Do not run full "
            "Swin-UNETR training on CPU."
        )

    device = torch.device(
        args.device
    )

    if device.type != "cuda":
        raise ValueError(
            "This training script requires a CUDA device."
        )

    amp_enabled = not args.no_amp

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_model_dir = (
        MODEL_ROOT
        / args.run_name
    )

    run_model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_checkpoint = (
        run_model_dir
        / "best_checkpoint.pth"
    )

    last_checkpoint = (
        run_model_dir
        / "last_checkpoint.pth"
    )

    history_path = (
        REPORT_DIR
        / f"34a_{args.run_name}_training_history.csv"
    )

    summary_path = (
        REPORT_DIR
        / f"34a_{args.run_name}_training_summary.txt"
    )

    config_path = (
        REPORT_DIR
        / f"34a_{args.run_name}_configuration.json"
    )

    if args.resume:
        if args.resume == "auto":
            resume_path = last_checkpoint
        else:
            resume_path = Path(
                args.resume
            )

        if not resume_path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_path}"
            )
    else:
        resume_path = None

        existing_outputs = [
            path
            for path in [
                best_checkpoint,
                last_checkpoint,
                history_path,
                summary_path,
                config_path,
            ]
            if path.exists()
        ]

        if existing_outputs:
            formatted = "\n".join(
                str(path)
                for path in existing_outputs
            )

            raise FileExistsError(
                "This run name already has outputs. "
                "Use a new --run-name or --resume auto.\n"
                f"{formatted}"
            )

    train_df, val_df = verify_fixed_splits()

    print("\nRuntime configuration")
    print("-" * 80)
    print(f"Run name: {args.run_name}")
    print(f"Device: {device}")
    print(
        f"Visible GPU: "
        f"{torch.cuda.get_device_name(device)}"
    )
    print(
        f"PyTorch: {torch.__version__}"
    )
    print(
        f"Patch size: {PATCH_SIZE}"
    )
    print(
        f"Batch size: {args.batch_size}"
    )
    print(
        f"AMP enabled: {amp_enabled}"
    )
    print(
        f"Learning rate: {args.learning_rate}"
    )
    print(
        f"Weight decay: {args.weight_decay}"
    )
    print(
        f"Validate every: {args.validate_every} epoch(s)"
    )
    print(
        f"Maximum train batches: "
        f"{args.max_train_batches or 'all'}"
    )
    print(
        f"Maximum validation patients: "
        f"{args.max_val_patients or 'all 59'}"
    )

    configuration = {
        **vars(args),
        "project_root": str(PROJECT_ROOT),
        "train_csv": str(TRAIN_CSV),
        "val_csv": str(VAL_CSV),
        "test_csv": str(TEST_CSV),
        "patch_size": PATCH_SIZE,
        "tumor_probability": TUMOR_PROBABILITY,
        "mixed_probability": MIXED_PROBABILITY,
        "tumor_free_probability": TUMOR_FREE_PROBABILITY,
        "expected_train_id_hash": EXPECTED_TRAIN_ID_HASH,
        "expected_val_id_hash": EXPECTED_VAL_ID_HASH,
        "checkpoint_selection": (
            "mean validation Dice across WT, TC, and ET"
        ),
    }

    if not config_path.exists():
        write_json(
            config_path,
            configuration,
        )

    train_dataset = BraTSMixedPatchDataset(
        dataframe=train_df,
        patch_size=PATCH_SIZE,
        seed=args.seed,
    )

    model = SwinUNETR(
        in_channels=IN_CHANNELS,
        out_channels=NUM_CLASSES,
        feature_size=24,
        use_checkpoint=False,
    ).to(device)

    criterion = DiceCELoss(
        include_background=False,
        to_onehot_y=True,
        softmax=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    start_epoch = 1
    best_epoch = 0
    best_val_macro_dice = float("-inf")

    if resume_path is not None:
        print(
            f"\nLoading checkpoint: {resume_path}"
        )

        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

        start_epoch = int(
            checkpoint["epoch"]
        ) + 1

        best_epoch = int(
            checkpoint.get(
                "best_epoch",
                0,
            )
        )

        best_val_macro_dice = float(
            checkpoint.get(
                "best_val_macro_dice",
                float("-inf"),
            )
        )

        print(
            f"Resuming at epoch {start_epoch}"
        )
        print(
            f"Best validation macro Dice so far: "
            f"{best_val_macro_dice:.6f}"
        )

    if start_epoch > args.epochs:
        raise ValueError(
            f"Resume checkpoint starts at epoch {start_epoch}, "
            f"but --epochs is only {args.epochs}."
        )

    total_training_start = time.perf_counter()

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):
        print("\n" + "=" * 80)
        print(
            f"Epoch {epoch}/{args.epochs}"
        )
        print("=" * 80)

        train_dataset.set_epoch(epoch)

        train_loader = make_train_loader(
            train_dataset,
            args,
            epoch,
        )

        model.train()

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(
                device
            )

        epoch_start = time.perf_counter()
        running_loss = 0.0
        completed_batches = 0

        patch_counts = {
            "tumor-centered": 0,
            "mixed-random": 0,
            "tumor-free": 0,
            "mixed-random-fallback": 0,
        }

        tumor_free_attempts: list[int] = []

        expected_batches = len(
            train_loader
        )

        if args.max_train_batches > 0:
            expected_batches = min(
                expected_batches,
                args.max_train_batches,
            )

        for batch_index, batch in enumerate(
            train_loader,
            start=1,
        ):
            if (
                args.max_train_batches > 0
                and batch_index
                > args.max_train_batches
            ):
                break

            image = batch["image"].to(
                device,
                non_blocking=True,
            )

            target = batch["target"].to(
                device,
                non_blocking=True,
            )

            target_for_loss = target.unsqueeze(1)

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(image)

                loss = criterion(
                    logits,
                    target_for_loss,
                )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at "
                    f"epoch {epoch}, batch {batch_index}: "
                    f"{loss.item()}"
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            loss_value = float(
                loss.item()
            )

            running_loss += loss_value
            completed_batches += 1

            for patch_type in batch["patch_type"]:
                patch_counts[
                    str(patch_type)
                ] += 1

            for patch_type, attempts in zip(
                batch["patch_type"],
                batch["search_attempts"],
            ):
                if str(patch_type) == "tumor-free":
                    tumor_free_attempts.append(
                        int(attempts)
                    )

            if (
                batch_index == 1
                or batch_index % 10 == 0
                or batch_index == expected_batches
            ):
                current_mean_loss = (
                    running_loss
                    / completed_batches
                )

                print(
                    f"Batch "
                    f"{batch_index:03d}/"
                    f"{expected_batches:03d} | "
                    f"loss={loss_value:.6f} | "
                    f"mean={current_mean_loss:.6f}"
                )

        if completed_batches == 0:
            raise RuntimeError(
                "No training batches were completed."
            )

        train_seconds = (
            time.perf_counter()
            - epoch_start
        )

        mean_train_loss = (
            running_loss
            / completed_batches
        )

        if device.type == "cuda":
            peak_memory_gb = (
                torch.cuda.max_memory_allocated(
                    device
                )
                / (1024 ** 3)
            )
        else:
            peak_memory_gb = float("nan")

        print("\nTraining summary")
        print("-" * 80)
        print(
            f"Completed batches: {completed_batches}"
        )
        print(
            f"Mean DiceCE loss: {mean_train_loss:.6f}"
        )
        print(
            f"Training time: "
            f"{train_seconds / 60:.2f} minutes"
        )
        print(
            f"Peak allocated GPU memory: "
            f"{peak_memory_gb:.2f} GB"
        )
        print(
            "Patch counts: "
            f"{patch_counts}"
        )

        if tumor_free_attempts:
            print(
                "Tumor-free search attempts: "
                f"mean={np.mean(tumor_free_attempts):.1f}, "
                f"max={np.max(tumor_free_attempts)}"
            )

        should_validate = (
            epoch % args.validate_every == 0
            or epoch == args.epochs
        )

        validation_summary = {
            "val_dice_WT": float("nan"),
            "val_iou_WT": float("nan"),
            "val_dice_TC": float("nan"),
            "val_iou_TC": float("nan"),
            "val_dice_ET": float("nan"),
            "val_iou_ET": float("nan"),
            "val_macro_dice": float("nan"),
            "val_pred_WT_voxels_mean": float("nan"),
            "val_true_WT_voxels_mean": float("nan"),
        }

        validation_seconds = 0.0
        validation_metrics_path = ""

        if should_validate:
            (
                validation_summary,
                validation_seconds,
                validation_output,
            ) = validate_full_volume(
                model=model,
                val_df=val_df,
                device=device,
                amp_enabled=amp_enabled,
                sw_batch_size=args.sw_batch_size,
                sw_overlap=args.sw_overlap,
                max_val_patients=args.max_val_patients,
                epoch=epoch,
                run_name=args.run_name,
            )

            validation_metrics_path = str(
                validation_output
            )

        current_val_macro_dice = float(
            validation_summary[
                "val_macro_dice"
            ]
        )

        is_best = (
            should_validate
            and np.isfinite(
                current_val_macro_dice
            )
            and current_val_macro_dice
            > best_val_macro_dice
        )

        if is_best:
            best_val_macro_dice = (
                current_val_macro_dice
            )

            best_epoch = epoch

        history_row = {
            "epoch": epoch,
            "completed_train_batches": completed_batches,
            "train_loss": mean_train_loss,
            "train_minutes": train_seconds / 60,
            "peak_gpu_memory_gb": peak_memory_gb,
            "tumor_centered_patches": patch_counts[
                "tumor-centered"
            ],
            "mixed_random_patches": patch_counts[
                "mixed-random"
            ],
            "tumor_free_patches": patch_counts[
                "tumor-free"
            ],
            "mixed_random_fallback_patches": patch_counts[
                "mixed-random-fallback"
            ],
            "tumor_free_search_attempts_mean": (
                float(
                    np.mean(
                        tumor_free_attempts
                    )
                )
                if tumor_free_attempts
                else 0.0
            ),
            "validation_performed": should_validate,
            **validation_summary,
            "validation_minutes": (
                validation_seconds / 60
            ),
            "validation_metrics_path": (
                validation_metrics_path
            ),
            "best_epoch_so_far": best_epoch,
            "best_val_macro_dice_so_far": (
                best_val_macro_dice
            ),
            "learning_rate": optimizer.param_groups[
                0
            ]["lr"],
        }

        append_history_row(
            history_path,
            history_row,
        )

        checkpoint_payload = {
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_macro_dice": (
                best_val_macro_dice
            ),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "scaler_state_dict": (
                scaler.state_dict()
            ),
            "history_row": history_row,
            "configuration": configuration,
            "model_configuration": {
                "in_channels": IN_CHANNELS,
                "out_channels": NUM_CLASSES,
                "feature_size": 24,
                "use_checkpoint": False,
                "patch_size": PATCH_SIZE,
            },
        }

        atomic_torch_save(
            checkpoint_payload,
            last_checkpoint,
        )

        if is_best:
            atomic_torch_save(
                checkpoint_payload,
                best_checkpoint,
            )

            print(
                "\nNew best checkpoint saved"
            )
            print(
                f"Epoch: {best_epoch}"
            )
            print(
                f"Validation macro Dice: "
                f"{best_val_macro_dice:.6f}"
            )
            print(
                f"Path: {best_checkpoint}"
            )

        write_summary(
            path=summary_path,
            args=args,
            current_epoch=epoch,
            best_epoch=best_epoch,
            best_val_macro_dice=(
                best_val_macro_dice
            ),
            last_history_row=history_row,
            best_checkpoint=best_checkpoint,
            last_checkpoint=last_checkpoint,
        )

        print("\nSaved epoch outputs")
        print("-" * 80)
        print(
            f"Last checkpoint: {last_checkpoint}"
        )

        if best_checkpoint.exists():
            print(
                f"Best checkpoint: {best_checkpoint}"
            )

        print(
            f"History CSV: {history_path}"
        )
        print(
            f"Summary TXT: {summary_path}"
        )

    total_seconds = (
        time.perf_counter()
        - total_training_start
    )

    print("\n" + "=" * 80)
    print("Swin-UNETR training run completed")
    print("=" * 80)
    print(
        f"Final epoch: {args.epochs}"
    )
    print(
        f"Best epoch: {best_epoch}"
    )
    print(
        f"Best validation macro Dice: "
        f"{best_val_macro_dice:.6f}"
    )
    print(
        f"Total run time: "
        f"{total_seconds / 3600:.2f} hours"
    )
    print(
        f"Best checkpoint: {best_checkpoint}"
    )
    print(
        f"Last checkpoint: {last_checkpoint}"
    )


if __name__ == "__main__":
    main()
