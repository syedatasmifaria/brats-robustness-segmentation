#!/usr/bin/env python3
"""
Script 27A: Prepare nnU-Net extended degradation pilot inputs.

Purpose:
- Create stronger degradation levels L6-L10 for selected artifacts only.
- Selected artifacts: noise, contrast, ringing.
- Use only 5 test patients first as a pilot.
- Apply degradation to all four modalities: FLAIR, T1, T1ce, T2.
- Save nnU-Net-compatible imagesTs folders for prediction.
- Save shared labelsTs for evaluation.

Why this script exists:
The original L1-L5 degradation study showed that nnU-Net was stable under
noise, contrast, and ringing. The goal here is to test stronger levels to
identify whether/when these artifacts begin to break the model.

Important:
- This is testing/evaluation preparation only.
- No model training happens here.
- Do not commit generated .nii.gz files to GitHub.
"""

from pathlib import Path
import hashlib
import importlib.util

import numpy as np
import pandas as pd
import nibabel as nib


PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

TEST_CSV = PROJECT_ROOT / "data/csvs/test_paths.csv"

OUT_ROOT = PROJECT_ROOT / "nnunet/temporary_degraded_tests/extended_pilot"
LABELS_DIR = OUT_ROOT / "labelsTs"

SCRIPT26A_PATH = PROJECT_ROOT / "scripts/26a_prepare_nnunet_degraded_final_full.py"

RNG_SEED = 2026
NUM_PATIENTS = 5

MODALITIES = [
    ("flair", "0000"),
    ("t1", "0001"),
    ("t1ce", "0002"),
    ("t2", "0003"),
]

# Extended stronger levels only.
# These continue the same logic used in L1-L5.
EXTENDED_LEVELS = {
    "noise": {
        6: {"std": 0.15},
        7: {"std": 0.20},
        8: {"std": 0.30},
        9: {"std": 0.40},
        10: {"std": 0.50},
    },
    "contrast": {
        6: {"factor": 0.20},
        7: {"factor": 0.15},
        8: {"factor": 0.10},
        9: {"factor": 0.05},
        10: {"factor": 0.02},
    },
    "ringing": {
        6: {"keep_fraction": 0.35},
        7: {"keep_fraction": 0.25},
        8: {"keep_fraction": 0.15},
        9: {"keep_fraction": 0.10},
        10: {"keep_fraction": 0.05},
    },
}


def load_script26a():
    """
    Load functions from Script 26A.

    We reuse the exact degradation functions that already worked for nnU-Net
    L1-L5, so the extended pilot remains consistent with the final nnU-Net
    degraded evaluation.
    """
    spec = importlib.util.spec_from_file_location("script26a", SCRIPT26A_PATH)
    script26a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script26a)
    return script26a


def stable_seed(*parts):
    """
    Create deterministic seed from text parts.
    This keeps noise degradation reproducible.
    """
    joined = "|".join(str(p) for p in parts)
    digest = hashlib.md5(joined.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + RNG_SEED) % (2**32)


def load_nifti(path):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine, img.header


def save_image_nifti(data, affine, header, out_path):
    out_img = nib.Nifti1Image(data.astype(np.float32), affine, header)
    nib.save(out_img, str(out_path))


def save_label_nifti(data, affine, header, out_path):
    out_img = nib.Nifti1Image(data.astype(np.int16), affine, header)
    nib.save(out_img, str(out_path))


def remap_seg_labels(seg):
    """
    BraTS original labels are [0,1,2,4].
    This project remaps label 4 to 3.
    """
    seg = seg.astype(np.int16)
    seg_remap = seg.copy()
    seg_remap[seg_remap == 4] = 3
    return seg_remap


def prepare_shared_labels(df):
    """
    Save one label file per pilot patient.

    Labels are shared across all extended degradation conditions.
    """
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    for _, row in df.iterrows():
        patient_id = row["patient_id"]
        out_path = LABELS_DIR / f"{patient_id}.nii.gz"

        if out_path.exists():
            skipped += 1
            continue

        seg_path = Path(row["seg"])
        seg, affine, header = load_nifti(seg_path)
        seg = remap_seg_labels(seg)

        save_label_nifti(seg, affine, header, out_path)
        written += 1

    print(f"Labels written: {written}")
    print(f"Labels skipped: {skipped}")


def main():
    print("=" * 80)
    print("Script 27A: Prepare nnU-Net extended degradation pilot")
    print("=" * 80)

    script26a = load_script26a()

    df = pd.read_csv(TEST_CSV)

    required_cols = ["patient_id", "flair", "t1", "t1ce", "t2", "seg"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required CSV column: {col}")

    pilot_df = df.head(NUM_PATIENTS).copy()

    print(f"Test CSV: {TEST_CSV}")
    print(f"Output root: {OUT_ROOT}")
    print(f"Pilot patients: {len(pilot_df)}")
    print("Selected artifacts:", list(EXTENDED_LEVELS.keys()))
    print("Extended levels: L6-L10")
    print()

    print("Pilot patient IDs:")
    for patient_id in pilot_df["patient_id"].tolist():
        print(f"  {patient_id}")
    print()

    prepare_shared_labels(pilot_df)

    total_images_written = 0
    total_images_skipped = 0
    condition_rows = []

    for artifact, level_dict in EXTENDED_LEVELS.items():
        for level, params in level_dict.items():
            condition_name = f"{artifact}_L{level}"

            condition_root = OUT_ROOT / condition_name
            images_dir = condition_root / "imagesTs"
            predictions_dir = condition_root / "predictions"

            images_dir.mkdir(parents=True, exist_ok=True)
            predictions_dir.mkdir(parents=True, exist_ok=True)

            print("-" * 80)
            print(f"Preparing condition: {condition_name} | params: {params}")

            condition_written = 0
            condition_skipped = 0

            for _, row in pilot_df.iterrows():
                patient_id = row["patient_id"]

                for modality_name, channel_id in MODALITIES:
                    out_path = images_dir / f"{patient_id}_{channel_id}.nii.gz"

                    # Resume safety.
                    if out_path.exists():
                        condition_skipped += 1
                        total_images_skipped += 1
                        continue

                    in_path = Path(row[modality_name])
                    volume, affine, header = load_nifti(in_path)

                    volume_01, brain_mask, v_min, v_max = script26a.normalize_for_degradation(volume)

                    seed = stable_seed(patient_id, modality_name, artifact, level)
                    rng = np.random.default_rng(seed)

                    degraded_01 = script26a.apply_degradation(
                        volume_01=volume_01,
                        brain_mask=brain_mask,
                        artifact=artifact,
                        params=params,
                        rng=rng,
                    )

                    degraded_volume = script26a.restore_original_range(
                        volume_01=degraded_01,
                        brain_mask=brain_mask,
                        v_min=v_min,
                        v_max=v_max,
                        original_volume=volume,
                    )

                    save_image_nifti(degraded_volume, affine, header, out_path)

                    condition_written += 1
                    total_images_written += 1

            print(f"Condition images written: {condition_written}")
            print(f"Condition images skipped: {condition_skipped}")

            condition_rows.append({
                "condition": condition_name,
                "artifact": artifact,
                "level": level,
                "params": str(params),
                "num_patients": len(pilot_df),
                "expected_images": len(pilot_df) * len(MODALITIES),
                "images_written": condition_written,
                "images_skipped": condition_skipped,
            })

    summary_df = pd.DataFrame(condition_rows)

    summary_csv = OUT_ROOT / "27a_extended_pilot_preparation_summary.csv"
    summary_txt = OUT_ROOT / "27a_extended_pilot_preparation_summary.txt"

    summary_df.to_csv(summary_csv, index=False)

    with open(summary_txt, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Script 27A: nnU-Net extended degradation pilot preparation summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Pilot patients: {len(pilot_df)}\n")
        f.write("Selected artifacts: noise, contrast, ringing\n")
        f.write("Extended levels: L6-L10\n")
        f.write(f"Total images written: {total_images_written}\n")
        f.write(f"Total images skipped: {total_images_skipped}\n\n")
        f.write("Expected images:\n")
        f.write("5 patients × 3 artifacts × 5 levels × 4 modalities = 300 images\n\n")
        f.write("Conditions:\n")
        for _, row in summary_df.iterrows():
            f.write(
                f"{row['condition']}: params={row['params']}, "
                f"written={row['images_written']}, skipped={row['images_skipped']}\n"
            )

    print("=" * 80)
    print("Extended nnU-Net pilot preparation complete.")
    print(f"Total images written: {total_images_written}")
    print(f"Total images skipped: {total_images_skipped}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved summary TXT: {summary_txt}")
    print("=" * 80)


if __name__ == "__main__":
    main()
