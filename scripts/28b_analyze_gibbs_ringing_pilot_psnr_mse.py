#!/usr/bin/env python3
"""
Script 28B: Compute MSE and PSNR for the Gibbs-like ringing visual pilot.

Purpose:
- Quantify image-level strength of the new gibbs_ringing artifact.
- Use the same single held-out test patient as Script 28A.
- Compare clean FLAIR vs gibbs_L1 ... gibbs_L5.
- Save CSV and TXT summary.

Important:
This is still a pilot sanity check.
MSE/PSNR measure image-level difference, not segmentation robustness.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")
TEST_CSV = PROJECT_ROOT / "data/csvs/test_paths.csv"

OUT_DIR = PROJECT_ROOT / "report_materials"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "28b_gibbs_ringing_pilot_psnr_mse.csv"
OUT_TXT = OUT_DIR / "28b_gibbs_ringing_pilot_psnr_mse_summary.txt"


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

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
    Same artifact logic as revised Script 28A.
    """
    vol = volume_01.astype(np.float32).copy()

    smooth = ndimage.gaussian_filter(vol, sigma=0.5)

    edge = ndimage.gaussian_gradient_magnitude(smooth, sigma=0.7)
    edge = edge * brain_mask

    if edge.max() > 0:
        edge = edge / (edge.max() + 1e-8)

    edge_values = edge[brain_mask]
    threshold = np.percentile(edge_values, 82)
    strong_edges = edge > threshold

    dist = ndimage.distance_transform_edt(~strong_edges)

    ripple = np.sin(2.0 * np.pi * dist / wavelength)
    decay = np.exp(-dist / 7.0)

    edge_band = ndimage.gaussian_filter(strong_edges.astype(np.float32), sigma=2.0)
    if edge_band.max() > 0:
        edge_band = edge_band / (edge_band.max() + 1e-8)

    artifact = strength * ripple * decay * (0.45 + 0.55 * edge_band)
    artifact = artifact * brain_mask

    degraded = vol + artifact
    degraded = np.clip(degraded, 0.0, 1.0)
    degraded[~brain_mask] = 0.0

    return degraded.astype(np.float32)


def compute_mse(clean, degraded, brain_mask=None):
    """
    Compute MSE. We use brain voxels only so background zeros do not dominate.
    """
    if brain_mask is None:
        diff = clean - degraded
    else:
        diff = clean[brain_mask] - degraded[brain_mask]

    return float(np.mean(diff ** 2))


def compute_psnr(mse, data_range=1.0):
    """
    PSNR = 20 * log10(data_range / sqrt(MSE))
    Since images are normalized to [0,1], data_range=1.0.
    """
    if mse <= 0:
        return float("inf")
    return float(20.0 * np.log10(data_range / np.sqrt(mse)))


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    test_df = pd.read_csv(TEST_CSV)

    row = test_df.iloc[0]
    patient_id = row["patient_id"]
    flair_path = Path(row["flair"])

    print("=" * 80)
    print("Script 28B: Gibbs-like ringing pilot PSNR/MSE")
    print("=" * 80)
    print(f"Patient: {patient_id}")
    print(f"FLAIR:   {flair_path}")

    flair = nib.load(str(flair_path)).get_fdata().astype(np.float32)
    clean, brain_mask = normalize_01(flair)

    levels = {
        "gibbs_L1": {"strength": 0.045, "wavelength": 7.0},
        "gibbs_L2": {"strength": 0.075, "wavelength": 6.5},
        "gibbs_L3": {"strength": 0.110, "wavelength": 6.0},
        "gibbs_L4": {"strength": 0.155, "wavelength": 5.5},
        "gibbs_L5": {"strength": 0.210, "wavelength": 5.0},
    }

    rows = []

    for level_name, params in levels.items():
        degraded = degrade_gibbs_ringing(
            clean,
            brain_mask,
            strength=params["strength"],
            wavelength=params["wavelength"],
        )

        mse = compute_mse(clean, degraded, brain_mask=brain_mask)
        psnr = compute_psnr(mse, data_range=1.0)

        rows.append({
            "patient_id": patient_id,
            "artifact": "gibbs_ringing",
            "level": level_name,
            "strength": params["strength"],
            "wavelength": params["wavelength"],
            "mse": mse,
            "psnr_db": psnr
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    with open(OUT_TXT, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Script 28B: Gibbs-like ringing pilot PSNR/MSE summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Patient: {patient_id}\n")
        f.write("Modality: FLAIR\n")
        f.write("Comparison: clean vs gibbs_L1 to gibbs_L5\n")
        f.write("Metric scope: brain voxels only\n")
        f.write("Normalization: [0,1], so PSNR data_range = 1.0\n\n")

        for _, r in df.iterrows():
            f.write(
                f"{r['level']}: "
                f"MSE = {r['mse']:.6f}, "
                f"PSNR = {r['psnr_db']:.2f} dB, "
                f"strength = {r['strength']:.3f}, "
                f"wavelength = {r['wavelength']:.1f}\n"
            )

        f.write("\nInterpretation guide:\n")
        f.write("- Higher MSE means stronger image-level change.\n")
        f.write("- Lower PSNR means stronger image-level change.\n")
        f.write("- A sensible pilot should show MSE increasing and PSNR decreasing from L1 to L5.\n")

    print(f"Saved CSV: {OUT_CSV}")
    print(f"Saved TXT: {OUT_TXT}")
    print("\nPreview:")
    print(df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
