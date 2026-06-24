import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# -----------------------------
# 1. Define project paths
# -----------------------------
PROJECT_ROOT = Path.home() / "brats_segmentation_project"

DATA_ROOT = PROJECT_ROOT / "data"
TRAINING_ROOT = DATA_ROOT / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData"

OUTPUT_DIR = PROJECT_ROOT / "data" / "csvs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# 2. Find patient folders
# -----------------------------
patient_dirs = sorted([
    p for p in TRAINING_ROOT.iterdir()
    if p.is_dir() and p.name.startswith("BraTS20_Training_")
])

print("Training root:", TRAINING_ROOT)
print("Number of patient folders found:", len(patient_dirs))


# -----------------------------
# 3. Build file path table
# -----------------------------
rows = []

for patient_dir in patient_dirs:
    patient_id = patient_dir.name

    flair_path = patient_dir / f"{patient_id}_flair.nii"
    t1_path = patient_dir / f"{patient_id}_t1.nii"
    t1ce_path = patient_dir / f"{patient_id}_t1ce.nii"
    t2_path = patient_dir / f"{patient_id}_t2.nii"
    seg_path = patient_dir / f"{patient_id}_seg.nii"

    rows.append({
        "patient_id": patient_id,
        "patient_dir": str(patient_dir),
        "flair": str(flair_path),
        "t1": str(t1_path),
        "t1ce": str(t1ce_path),
        "t2": str(t2_path),
        "seg": str(seg_path),
        "flair_exists": flair_path.exists(),
        "t1_exists": t1_path.exists(),
        "t1ce_exists": t1ce_path.exists(),
        "t2_exists": t2_path.exists(),
        "seg_exists": seg_path.exists(),
    })

all_paths = pd.DataFrame(rows)

print("\nPreview:")
print(all_paths.head())


# -----------------------------
# 4. Check missing files
# -----------------------------
required_cols = ["flair_exists", "t1_exists", "t1ce_exists", "t2_exists", "seg_exists"]

missing_summary = all_paths[required_cols].sum()
print("\nExisting file counts:")
print(missing_summary)

usable_paths = all_paths[all_paths[required_cols].all(axis=1)].copy()

print("\nUsable patients:", len(usable_paths))


# -----------------------------
# 5. Create 80/20 patient-level split
# -----------------------------
train_df, test_df = train_test_split(
    usable_paths,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

train_df = train_df.sort_values("patient_id").reset_index(drop=True)
test_df = test_df.sort_values("patient_id").reset_index(drop=True)
usable_paths = usable_paths.sort_values("patient_id").reset_index(drop=True)

print("\nTrain patients:", len(train_df))
print("Test patients:", len(test_df))


# -----------------------------
# 6. Save CSV files
# -----------------------------
raw_csv = OUTPUT_DIR / "all_patient_paths_raw.csv"
all_csv = OUTPUT_DIR / "all_patient_paths.csv"
train_csv = OUTPUT_DIR / "train_paths.csv"
test_csv = OUTPUT_DIR / "test_paths.csv"

all_paths.to_csv(raw_csv, index=False)
usable_paths.to_csv(all_csv, index=False)
train_df.to_csv(train_csv, index=False)
test_df.to_csv(test_csv, index=False)

print("\nSaved CSV files:")
print(raw_csv)
print(all_csv)
print(train_csv)
print(test_csv)
