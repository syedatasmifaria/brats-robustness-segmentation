# %%
from pathlib import Path

import pandas as pd
import nibabel as nib
import torch

# %%
PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")
CSV_DIR = PROJECT_ROOT / "data" / "csvs"

train_csv = CSV_DIR / "train_paths.csv"
test_csv = CSV_DIR / "test_paths.csv"

print("Project root:", PROJECT_ROOT)
print("Train CSV exists:", train_csv.exists())
print("Test CSV exists:", test_csv.exists())

# %%
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

print("Train patients:", len(train_df))
print("Test patients:", len(test_df))

print("\nColumns:")
print(train_df.columns.tolist())

print("\nFirst patient:")
print(train_df.iloc[0])

# %%
first_flair_path = Path(train_df.iloc[0]["flair"])
first_seg_path = Path(train_df.iloc[0]["seg"])

flair_img = nib.load(first_flair_path)
seg_img = nib.load(first_seg_path)

print("First FLAIR path:", first_flair_path)
print("FLAIR shape:", flair_img.shape)
print("FLAIR dtype:", flair_img.get_data_dtype())

print("\nFirst SEG path:", first_seg_path)
print("SEG shape:", seg_img.shape)
print("SEG dtype:", seg_img.get_data_dtype())

# %%
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU 0:", torch.cuda.get_device_name(0))
# %%
