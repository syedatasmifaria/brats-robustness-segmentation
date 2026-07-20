#!/usr/bin/env python3
"""
Script 32C: Visualize nnU-Net blur and ghosting severity levels.

Purpose:
- Show Clean and L1-L10 for blur and ghosting.
- Use one representative pilot patient.
- Select the axial slice with the largest whole-tumor area.
- Use a fixed clean-image display window across all severity levels.
- Save high-resolution PNG and PDF figures.

This is a visual quality-control and report figure.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt


PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")
TEST_CSV = PROJECT_ROOT / "data/csvs/test_paths.csv"

L1_L5_ROOT = (
    PROJECT_ROOT
    / "nnunet/temporary_degraded_tests/final_full"
)

L6_L10_ROOT = (
    PROJECT_ROOT
    / "nnunet/temporary_degraded_tests/blur_ghosting_extended_pilot"
)

REPORT_DIR = PROJECT_ROOT / "report_materials"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODALITY_NAME = "flair"
CHANNEL_ID = "0000"

ARTIFACTS = ["blur", "ghosting"]
LEVELS = list(range(1, 11))


def load_volume(path):
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def get_condition_path(artifact, level, patient_id):
    if level <= 5:
        root = L1_L5_ROOT
    else:
        root = L6_L10_ROOT

    return (
        root
        / f"{artifact}_L{level}"
        / "imagesTs"
        / f"{patient_id}_{CHANNEL_ID}.nii.gz"
    )


def find_largest_tumor_slice(segmentation):
    tumor_mask = segmentation > 0

    tumor_area_by_slice = tumor_mask.sum(axis=(0, 1))

    if tumor_area_by_slice.max() == 0:
        raise RuntimeError("No tumor voxels found in segmentation.")

    return int(np.argmax(tumor_area_by_slice))


def clean_display_window(clean_volume):
    brain_values = clean_volume[clean_volume > 0]

    if brain_values.size == 0:
        raise RuntimeError("Clean volume contains no nonzero brain voxels.")

    display_min = float(np.percentile(brain_values, 1))
    display_max = float(np.percentile(brain_values, 99))

    return display_min, display_max


def plot_artifact(
    artifact,
    patient_id,
    clean_volume,
    slice_index,
    display_min,
    display_max,
):
    images = [clean_volume]
    titles = ["Clean"]

    for level in LEVELS:
        degraded_path = get_condition_path(
            artifact=artifact,
            level=level,
            patient_id=patient_id,
        )

        if not degraded_path.exists():
            raise FileNotFoundError(
                f"Missing degraded image: {degraded_path}"
            )

        degraded_volume = load_volume(degraded_path)
        images.append(degraded_volume)
        titles.append(f"L{level}")

    figure, axes = plt.subplots(
        2,
        6,
        figsize=(18, 7),
    )

    axes = axes.flatten()

    for index, axis in enumerate(axes):
        if index < len(images):
            image_slice = images[index][:, :, slice_index]

            axis.imshow(
                np.rot90(image_slice),
                cmap="gray",
                vmin=display_min,
                vmax=display_max,
            )

            axis.set_title(titles[index], fontsize=12)
            axis.axis("off")
        else:
            axis.axis("off")

    figure.suptitle(
        (
            f"{artifact.capitalize()} severity progression: "
            f"Clean and L1-L10\n"
            f"{patient_id}, FLAIR, axial slice {slice_index}"
        ),
        fontsize=15,
    )

    figure.tight_layout(rect=[0, 0, 1, 0.92])

    png_path = (
        REPORT_DIR
        / f"32c_{artifact}_clean_L1_L10_fixed_window.png"
    )

    pdf_path = (
        REPORT_DIR
        / f"32c_{artifact}_clean_L1_L10_fixed_window.pdf"
    )

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")


def main():
    print("=" * 80)
    print("Script 32C: Visualize blur and ghosting L1-L10")
    print("=" * 80)

    test_df = pd.read_csv(TEST_CSV)
    patient_row = test_df.iloc[0]

    patient_id = patient_row["patient_id"]
    clean_path = Path(patient_row[MODALITY_NAME])
    segmentation_path = Path(patient_row["seg"])

    clean_volume = load_volume(clean_path)
    segmentation = load_volume(segmentation_path)

    if clean_volume.shape != segmentation.shape:
        raise RuntimeError(
            f"Shape mismatch: clean={clean_volume.shape}, "
            f"segmentation={segmentation.shape}"
        )

    slice_index = find_largest_tumor_slice(segmentation)

    display_min, display_max = clean_display_window(
        clean_volume
    )

    print(f"Patient: {patient_id}")
    print(f"Modality: {MODALITY_NAME}")
    print(f"Selected axial slice: {slice_index}")
    print(
        f"Fixed display window: "
        f"{display_min:.4f} to {display_max:.4f}"
    )
    print()

    for artifact in ARTIFACTS:
        plot_artifact(
            artifact=artifact,
            patient_id=patient_id,
            clean_volume=clean_volume,
            slice_index=slice_index,
            display_min=display_min,
            display_max=display_max,
        )

    print()
    print("=" * 80)
    print("Blur and ghosting visualization complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
