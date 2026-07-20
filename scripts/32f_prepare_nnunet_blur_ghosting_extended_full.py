#!/usr/bin/env python3
"""
Script 32F: Prepare nnU-Net blur and ghosting extended full evaluation inputs.

Purpose:
- Extend blur and ghosting from L1-L5 to provisional L6-L10.
- Use all 74 held-out test patients.
- Apply degradation to all four MRI modalities.
- Reuse the exact production degradation functions from Script 26A.
- Preserve the original MRI intensity range before saving.

Important:
- This is testing/evaluation preparation only.
- No model training occurs here.
- The L6-L10 parameters were approved after the five-patient pilot.
- Do not commit generated NIfTI files to GitHub.
"""

from pathlib import Path
import hashlib
import importlib.util

import numpy as np
import pandas as pd
import nibabel as nib


PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")
TEST_CSV = PROJECT_ROOT / "data/csvs/test_paths.csv"

OUT_ROOT = (
    PROJECT_ROOT
    / "nnunet/temporary_degraded_tests/blur_ghosting_extended_full"
)
LABELS_DIR = OUT_ROOT / "labelsTs"

SCRIPT26A_PATH = (
    PROJECT_ROOT
    / "scripts/26a_prepare_nnunet_degraded_final_full.py"
)

RNG_SEED = 2026
NUM_PATIENTS = 74

MODALITIES = [
    ("flair", "0000"),
    ("t1", "0001"),
    ("t1ce", "0002"),
    ("t2", "0003"),
]

# Pilot-approved extended levels.
# These passed visual, MSE/PSNR, and five-patient segmentation pilot checks.
EXTENDED_LEVELS = {
    "blur": {
        6: {"sigma": 3.0},
        7: {"sigma": 3.5},
        8: {"sigma": 4.0},
        9: {"sigma": 4.5},
        10: {"sigma": 5.0},
    },
    "ghosting": {
        6: {"num_repeats": 24, "intensity": 0.65},
        7: {"num_repeats": 28, "intensity": 0.75},
        8: {"num_repeats": 32, "intensity": 0.85},
        9: {"num_repeats": 36, "intensity": 0.90},
        10: {"num_repeats": 40, "intensity": 0.95},
    },
}


def load_script26a():
    """Load the production degradation functions from Script 26A."""
    spec = importlib.util.spec_from_file_location(
        "script26a",
        SCRIPT26A_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Script 26A: {SCRIPT26A_PATH}")

    script26a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script26a)
    return script26a


def stable_seed(*parts):
    """Create a deterministic seed from text components."""
    joined = "|".join(str(part) for part in parts)
    digest = hashlib.md5(joined.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + RNG_SEED) % (2**32)


def load_nifti(path):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine, img.header


def save_image_nifti(data, affine, header, out_path):
    out_img = nib.Nifti1Image(
        data.astype(np.float32),
        affine,
        header,
    )
    nib.save(out_img, str(out_path))


def save_label_nifti(data, affine, header, out_path):
    out_img = nib.Nifti1Image(
        data.astype(np.uint8),
        affine,
        header,
    )
    nib.save(out_img, str(out_path))


def remap_seg_labels(seg):
    """Remap BraTS label 4 to project/nnU-Net label 3."""
    remapped = np.zeros_like(seg, dtype=np.uint8)
    remapped[seg == 1] = 1
    remapped[seg == 2] = 2
    remapped[seg == 4] = 3
    return remapped


def prepare_shared_labels(df):
    """Save one shared label file per full test patient."""
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    for _, row in df.iterrows():
        patient_id = row["patient_id"]
        out_path = LABELS_DIR / f"{patient_id}.nii.gz"

        if out_path.exists():
            skipped += 1
            continue

        seg, affine, header = load_nifti(Path(row["seg"]))
        seg_remapped = remap_seg_labels(seg)

        save_label_nifti(
            seg_remapped,
            affine,
            header,
            out_path,
        )
        written += 1

    print(f"Labels written: {written}")
    print(f"Labels skipped: {skipped}")


def main():
    print("=" * 80)
    print("Script 32F: Prepare blur and ghosting extended full evaluation")
    print("=" * 80)

    script26a = load_script26a()
    df = pd.read_csv(TEST_CSV)

    required_columns = [
        "patient_id",
        "flair",
        "t1",
        "t1ce",
        "t2",
        "seg",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required CSV columns: {missing_columns}"
        )

    full_df = df.head(NUM_PATIENTS).copy()

    print(f"Test CSV: {TEST_CSV}")
    print(f"Output root: {OUT_ROOT}")
    print(f"Full test patients: {len(full_df)}")
    print("Artifacts: blur, ghosting")
    print("Candidate levels: L6-L10")
    print()

    print("Full test patient IDs:")
    for patient_id in full_df["patient_id"]:
        print(f"  {patient_id}")
    print()

    prepare_shared_labels(full_df)

    total_written = 0
    total_skipped = 0
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
            print(
                f"Preparing condition: {condition_name} "
                f"| params: {params}"
            )

            condition_written = 0
            condition_skipped = 0

            for _, row in full_df.iterrows():
                patient_id = row["patient_id"]

                for modality_name, channel_id in MODALITIES:
                    out_path = (
                        images_dir
                        / f"{patient_id}_{channel_id}.nii.gz"
                    )

                    if out_path.exists():
                        condition_skipped += 1
                        total_skipped += 1
                        continue

                    volume, affine, header = load_nifti(
                        Path(row[modality_name])
                    )

                    (
                        volume_01,
                        brain_mask,
                        v_min,
                        v_max,
                    ) = script26a.normalize_for_degradation(volume)

                    seed = stable_seed(
                        patient_id,
                        modality_name,
                        artifact,
                        level,
                    )
                    rng = np.random.default_rng(seed)

                    degraded_01 = script26a.apply_degradation(
                        volume_01=volume_01,
                        brain_mask=brain_mask,
                        artifact=artifact,
                        params=params,
                        rng=rng,
                    )

                    degraded_volume = (
                        script26a.restore_original_range(
                            volume_01=degraded_01,
                            brain_mask=brain_mask,
                            v_min=v_min,
                            v_max=v_max,
                            original_volume=volume,
                        )
                    )

                    save_image_nifti(
                        degraded_volume,
                        affine,
                        header,
                        out_path,
                    )

                    condition_written += 1
                    total_written += 1

            print(
                f"Condition images written: {condition_written}"
            )
            print(
                f"Condition images skipped: {condition_skipped}"
            )

            condition_rows.append({
                "condition": condition_name,
                "artifact": artifact,
                "level": level,
                "params": str(params),
                "num_patients": len(full_df),
                "expected_images": (
                    len(full_df) * len(MODALITIES)
                ),
                "images_written": condition_written,
                "images_skipped": condition_skipped,
            })

    summary_df = pd.DataFrame(condition_rows)

    summary_csv = (
        OUT_ROOT
        / "32f_blur_ghosting_extended_full_summary.csv"
    )
    summary_txt = (
        OUT_ROOT
        / "32f_blur_ghosting_extended_full_summary.txt"
    )

    summary_df.to_csv(summary_csv, index=False)

    expected_total = (
        len(full_df)
        * len(EXTENDED_LEVELS)
        * 5
        * len(MODALITIES)
    )

    with open(summary_txt, "w", encoding="utf-8") as file:
        file.write("=" * 80 + "\n")
        file.write(
            "Script 32F: Blur and ghosting extended full evaluation summary\n"
        )
        file.write("=" * 80 + "\n\n")
        file.write(f"Full test patients: {len(full_df)}\n")
        file.write("Artifacts: blur, ghosting\n")
        file.write("Candidate levels: L6-L10\n")
        file.write(f"Expected images: {expected_total}\n")
        file.write(f"Images written: {total_written}\n")
        file.write(f"Images skipped: {total_skipped}\n\n")
        file.write("Candidate conditions:\n")

        for _, row in summary_df.iterrows():
            file.write(
                f"{row['condition']}: "
                f"params={row['params']}, "
                f"written={row['images_written']}, "
                f"skipped={row['images_skipped']}\n"
            )

    print()
    print("=" * 80)
    print("Blur and ghosting extended full evaluation preparation complete.")
    print(f"Expected image files: {expected_total}")
    print(f"Images written this run: {total_written}")
    print(f"Images skipped this run: {total_skipped}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary TXT: {summary_txt}")
    print("=" * 80)
    print()
    print("Do not run nnU-Net predictions yet.")
    print("Next: verify file counts, then run full nnU-Net predictions.")


if __name__ == "__main__":
    main()
