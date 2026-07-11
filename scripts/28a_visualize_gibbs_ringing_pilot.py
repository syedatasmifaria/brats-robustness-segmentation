#!/usr/bin/env python3
"""
Script 28A revised again: Stronger visual-only Gibbs-like ringing pilot.

Purpose:
- Create a clearer Gibbs-like edge ringing artifact.
- Save full-slice, zoomed, and difference-map visual grids.
- Use one held-out test patient only.
- Do NOT run prediction.
- Do NOT overwrite old ringing results.

Key idea:
Gibbs ringing appears as alternating bright/dark oscillations near sharp edges.
This script creates edge-localized ripples and visualizes both the degraded image
and the degraded-clean difference.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import ndimage


PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")
TEST_CSV = PROJECT_ROOT / "data/csvs/test_paths.csv"

OUT_DIR = PROJECT_ROOT / "report_materials"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FULL = OUT_DIR / "28a_gibbs_ringing_visual_grid.png"
OUT_ZOOM = OUT_DIR / "28a_gibbs_ringing_zoom_grid.png"
OUT_DIFF = OUT_DIR / "28a_gibbs_ringing_difference_grid.png"


def normalize_01(volume):
    volume = volume.astype(np.float32)
    brain_mask = volume > 0

    if brain_mask.sum() == 0:
        return np.zeros_like(volume, dtype=np.float32), brain_mask

    vals = volume[brain_mask]
    lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)

    volume_clip = np.clip(volume, lo, hi)
    volume_01 = (volume_clip - lo) / (hi - lo + 1e-8)
    volume_01[~brain_mask] = 0.0

    return volume_01.astype(np.float32), brain_mask


def degrade_gibbs_ringing(volume_01, brain_mask, strength=0.10, wavelength=5.0):
    """
    Create stronger Gibbs-like edge ringing.

    Code-level logic:
    1. Detect strong boundaries using gradient magnitude.
    2. Compute distance from those boundaries.
    3. Add sine-wave oscillations as a function of distance from edges.
    4. Decay the oscillation away from edges.
    5. Add the artifact to the clean image.

    This should create visible bright/dark ripple halos near edges.
    """

    vol = volume_01.astype(np.float32).copy()

    # Slight smoothing only for stable edge detection, not for output.
    smooth = ndimage.gaussian_filter(vol, sigma=0.5)

    # Edge strength
    edge = ndimage.gaussian_gradient_magnitude(smooth, sigma=0.7)
    edge = edge * brain_mask

    if edge.max() > 0:
        edge = edge / (edge.max() + 1e-8)

    # Focus on stronger edges.
    edge_values = edge[brain_mask]
    threshold = np.percentile(edge_values, 82)
    strong_edges = edge > threshold

    # Distance to nearest strong edge.
    dist = ndimage.distance_transform_edt(~strong_edges)

    # Oscillation away from edge.
    ripple = np.sin(2.0 * np.pi * dist / wavelength)

    # Decay away from edge.
    decay = np.exp(-dist / 7.0)

    # Edge localization.
    edge_band = ndimage.gaussian_filter(strong_edges.astype(np.float32), sigma=2.0)
    if edge_band.max() > 0:
        edge_band = edge_band / (edge_band.max() + 1e-8)

    # Combine. The 0.45 term keeps ripples visible around edge neighborhoods.
    artifact = strength * ripple * decay * (0.45 + 0.55 * edge_band)
    artifact = artifact * brain_mask

    degraded = vol + artifact
    degraded = np.clip(degraded, 0.0, 1.0)
    degraded[~brain_mask] = 0.0

    return degraded.astype(np.float32)


def pick_slice_from_seg(seg):
    tumor = seg > 0
    scores = tumor.sum(axis=(0, 1))

    if scores.max() == 0:
        return int(seg.shape[2] // 2)

    return int(np.argmax(scores))


def get_zoom_crop(seg_slice, margin=45):
    tumor = seg_slice > 0

    if tumor.sum() == 0:
        h, w = seg_slice.shape
        return 0, h, 0, w

    ys, xs = np.where(tumor)

    y1 = max(0, ys.min() - margin)
    y2 = min(seg_slice.shape[0], ys.max() + margin)
    x1 = max(0, xs.min() - margin)
    x2 = min(seg_slice.shape[1], xs.max() + margin)

    return y1, y2, x1, x2


def save_image_grid(images, z, patient_id, out_path, title, crop=None):
    clean_slice = images["clean"][:, :, z]
    positive = clean_slice[clean_slice > 0]

    vmin = np.percentile(positive, 1)
    vmax = np.percentile(positive, 99)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    for ax, (name, vol) in zip(axes, images.items()):
        img = vol[:, :, z]

        if crop is not None:
            y1, y2, x1, x2 = crop
            img = img[y1:y2, x1:x2]

        ax.imshow(np.rot90(img), cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.axis("off")

    fig.suptitle(f"{title}\nPatient {patient_id}, FLAIR, axial slice {z}", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_difference_grid(images, z, patient_id, out_path, crop=None):
    clean = images["clean"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    for ax, (name, vol) in zip(axes, images.items()):
        if name == "clean":
            diff = np.zeros_like(clean[:, :, z])
        else:
            diff = vol[:, :, z] - clean[:, :, z]

        if crop is not None:
            y1, y2, x1, x2 = crop
            diff = diff[y1:y2, x1:x2]

        ax.imshow(np.rot90(diff), cmap="gray", vmin=-0.18, vmax=0.18)
        ax.set_title(f"{name} minus clean")
        ax.axis("off")

    fig.suptitle(
        f"Script 28A: Gibbs-like ringing difference map\nPatient {patient_id}, FLAIR, axial slice {z}",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main():
    test_df = pd.read_csv(TEST_CSV)

    row = test_df.iloc[0]
    patient_id = row["patient_id"]
    flair_path = Path(row["flair"])
    seg_path = Path(row["seg"])

    print("=" * 80)
    print("Script 28A: Stronger Gibbs-like ringing visual pilot")
    print("=" * 80)
    print(f"Patient: {patient_id}")
    print(f"FLAIR:   {flair_path}")
    print(f"SEG:     {seg_path}")

    flair = nib.load(str(flair_path)).get_fdata().astype(np.float32)
    seg = nib.load(str(seg_path)).get_fdata().astype(np.int16)

    flair_01, brain_mask = normalize_01(flair)

    z = pick_slice_from_seg(seg)
    print(f"Selected tumor-rich axial slice z = {z}")

    levels = {
        "clean": None,
        "gibbs_L1": {"strength": 0.045, "wavelength": 7.0},
        "gibbs_L2": {"strength": 0.075, "wavelength": 6.5},
        "gibbs_L3": {"strength": 0.110, "wavelength": 6.0},
        "gibbs_L4": {"strength": 0.155, "wavelength": 5.5},
        "gibbs_L5": {"strength": 0.210, "wavelength": 5.0},
    }

    images = {}

    for name, params in levels.items():
        if name == "clean":
            images[name] = flair_01
        else:
            images[name] = degrade_gibbs_ringing(
                flair_01,
                brain_mask,
                strength=params["strength"],
                wavelength=params["wavelength"],
            )

    seg_slice = seg[:, :, z]
    crop = get_zoom_crop(seg_slice, margin=50)

    save_image_grid(
        images=images,
        z=z,
        patient_id=patient_id,
        out_path=OUT_FULL,
        title="Script 28A: Stronger Gibbs-like ringing full-slice grid",
        crop=None,
    )

    save_image_grid(
        images=images,
        z=z,
        patient_id=patient_id,
        out_path=OUT_ZOOM,
        title="Script 28A: Stronger Gibbs-like ringing zoomed tumor/edge grid",
        crop=crop,
    )

    save_difference_grid(
        images=images,
        z=z,
        patient_id=patient_id,
        out_path=OUT_DIFF,
        crop=crop,
    )

    print(f"Saved full grid:       {OUT_FULL}")
    print(f"Saved zoom grid:       {OUT_ZOOM}")
    print(f"Saved difference grid: {OUT_DIFF}")
    print("")
    print("Inspect the zoom grid and the difference grid.")
    print("If difference map shows edge-localized ripple bands, we can proceed to PSNR/MSE.")
    print("If not, we revise again.")
    print("=" * 80)


if __name__ == "__main__":
    main()
