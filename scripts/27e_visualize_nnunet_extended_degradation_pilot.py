#!/usr/bin/env python3
"""
Script 27E: Visualize nnU-Net extended degradation pilot images.

Purpose:
- Create visual grids for extended degradation levels L6-L10.
- Selected artifacts: noise, contrast, ringing.
- Show all four modalities: FLAIR, T1, T1ce, T2.
- Compare Clean, L6, L7, L8, L9, L10.
- Save report-ready PNG figures.

Important:
These are visual sanity checks for the extended stress-test levels.
They are not model predictions.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt


PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

TEST_CSV = PROJECT_ROOT / "data/csvs/test_paths.csv"
EXTENDED_ROOT = PROJECT_ROOT / "nnunet/temporary_degraded_tests/extended_pilot"
REPORT_DIR = PROJECT_ROOT / "report_materials"

ARTIFACTS = ["noise", "contrast", "ringing"]
LEVELS = [6, 7, 8, 9, 10]

MODALITIES = [
    ("flair", "0000", "FLAIR"),
    ("t1", "0001", "T1"),
    ("t1ce", "0002", "T1ce"),
    ("t2", "0003", "T2"),
]


def load_nifti(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def normalize_for_display(slice_2d: np.ndarray) -> np.ndarray:
    """
    Normalize a 2D slice for display using robust percentiles.
    This avoids one extreme voxel making the image too dark/bright.
    """
    values = slice_2d[np.isfinite(slice_2d)]

    if values.size == 0:
        return np.zeros_like(slice_2d, dtype=np.float32)

    nonzero = values[values != 0]

    if nonzero.size > 0:
        lo = np.percentile(nonzero, 1)
        hi = np.percentile(nonzero, 99)
    else:
        lo = np.percentile(values, 1)
        hi = np.percentile(values, 99)

    if hi <= lo:
        return np.zeros_like(slice_2d, dtype=np.float32)

    out = (slice_2d - lo) / (hi - lo)
    out = np.clip(out, 0, 1)
    return out


def choose_slice_index(flair_volume: np.ndarray) -> int:
    """
    Choose an axial slice near the middle of the brain.
    We use the clean FLAIR volume because it usually shows tumor/brain clearly.
    """
    brain_mask = flair_volume != 0
    z_has_brain = brain_mask.sum(axis=(0, 1)) > 0
    z_indices = np.where(z_has_brain)[0]

    if len(z_indices) == 0:
        return flair_volume.shape[2] // 2

    return int(z_indices[len(z_indices) // 2])


def main():
    print("=" * 80)
    print("Script 27E: Visualize nnU-Net extended degradation pilot")
    print("=" * 80)

    REPORT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(TEST_CSV).head(5).copy()

    # Use the first pilot patient for visualization.
    row = df.iloc[0]
    patient_id = row["patient_id"]

    print(f"Using patient: {patient_id}")

    clean_flair = load_nifti(Path(row["flair"]))
    z = choose_slice_index(clean_flair)

    print(f"Selected axial slice index: {z}")

    for artifact in ARTIFACTS:
        print("-" * 80)
        print(f"Creating figure for artifact: {artifact}")

        fig, axes = plt.subplots(
            nrows=len(MODALITIES),
            ncols=1 + len(LEVELS),
            figsize=(18, 10)
        )

        column_titles = ["Clean"] + [f"L{level}" for level in LEVELS]

        for col_idx, title in enumerate(column_titles):
            axes[0, col_idx].set_title(title, fontsize=12)

        for row_idx, (modality_name, channel_id, modality_label) in enumerate(MODALITIES):
            clean_path = Path(row[modality_name])
            clean_vol = load_nifti(clean_path)
            clean_slice = normalize_for_display(clean_vol[:, :, z])

            axes[row_idx, 0].imshow(clean_slice.T, cmap="gray", origin="lower")
            axes[row_idx, 0].set_ylabel(modality_label, fontsize=12)

            for level_idx, level in enumerate(LEVELS, start=1):
                degraded_path = (
                    EXTENDED_ROOT
                    / f"{artifact}_L{level}"
                    / "imagesTs"
                    / f"{patient_id}_{channel_id}.nii.gz"
                )

                if not degraded_path.exists():
                    raise FileNotFoundError(f"Missing degraded image: {degraded_path}")

                degraded_vol = load_nifti(degraded_path)
                degraded_slice = normalize_for_display(degraded_vol[:, :, z])

                axes[row_idx, level_idx].imshow(degraded_slice.T, cmap="gray", origin="lower")

            for col_idx in range(1 + len(LEVELS)):
                axes[row_idx, col_idx].set_xticks([])
                axes[row_idx, col_idx].set_yticks([])

        fig.suptitle(
            f"Extended degradation pilot: {artifact} | Patient {patient_id} | Axial slice {z}",
            fontsize=16
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        out_path = REPORT_DIR / f"27e_extended_pilot_{artifact}_visual_grid.png"
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"Saved: {out_path}")

    print("=" * 80)
    print("Extended degradation visualization complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
