"""
Script 05b: Visual sanity check for selected 2D training slices.

Goal:
- Load train_2d_slices_flair.csv
- Randomly select tumor and background slices
- Visualize FLAIR slice and segmentation overlay
- Save figure to results folder

This script does NOT train a model.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_DIR = Path("/home/xfh25/brats_segmentation_project")

SLICE_CSV = PROJECT_DIR / "data" / "csvs" / "train_2d_slices_flair.csv"
RESULTS_DIR = PROJECT_DIR / "results"

OUTPUT_FIG = RESULTS_DIR / "05b_selected_training_slices_preview.png"

RANDOM_SEED = 42
NUM_TUMOR_EXAMPLES = 4
NUM_BACKGROUND_EXAMPLES = 4


def normalize_image(image):
    """
    Normalize image to 0-1 for display.
    """
    image = image.astype(np.float32)

    min_val = np.min(image)
    max_val = np.max(image)

    if max_val - min_val == 0:
        return np.zeros_like(image)

    return (image - min_val) / (max_val - min_val)


def load_slice(flair_path, seg_path, slice_idx):
    """
    Load one axial FLAIR slice and its matching segmentation slice.
    """
    flair_volume = nib.load(str(flair_path)).get_fdata()
    seg_volume = nib.load(str(seg_path)).get_fdata()

    flair_slice = flair_volume[:, :, slice_idx]
    seg_slice = seg_volume[:, :, slice_idx]

    flair_slice = normalize_image(flair_slice)

    return flair_slice, seg_slice


def main():
    print("=" * 80)
    print("Script 05b: Visualize selected 2D training slices")
    print("=" * 80)

    if not SLICE_CSV.exists():
        raise FileNotFoundError(f"Could not find slice CSV: {SLICE_CSV}")

    df = pd.read_csv(SLICE_CSV)

    print(f"Loaded slice CSV: {SLICE_CSV}")
    print(f"Total selected slices: {len(df)}")

    tumor_df = df[df["has_tumor"] == 1]
    background_df = df[df["has_tumor"] == 0]

    print(f"Tumor slices available: {len(tumor_df)}")
    print(f"Background slices available: {len(background_df)}")

    tumor_examples = tumor_df.sample(
        n=NUM_TUMOR_EXAMPLES,
        random_state=RANDOM_SEED
    )

    background_examples = background_df.sample(
        n=NUM_BACKGROUND_EXAMPLES,
        random_state=RANDOM_SEED
    )

    examples = pd.concat([tumor_examples, background_examples]).reset_index(drop=True)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, examples.iterrows()):
        flair_slice, seg_slice = load_slice(
            row["flair_path"],
            row["seg_path"],
            int(row["slice_idx"])
        )

        ax.imshow(flair_slice, cmap="gray")

        if row["has_tumor"] == 1:
            masked_seg = np.ma.masked_where(seg_slice == 0, seg_slice)
            ax.imshow(masked_seg, alpha=0.45)

        title = (
            f'{row["patient_id"]}\n'
            f'Slice {row["slice_idx"]}, tumor={row["has_tumor"]}'
        )
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FIG, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved preview figure to:\n{OUTPUT_FIG}")
    print("\nDone. Visual sanity check completed.")


if __name__ == "__main__":
    main()