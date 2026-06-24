# %%
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%
# Paths
PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")
CSV_DIR = PROJECT_ROOT / "data" / "csvs"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

train_csv = CSV_DIR / "train_paths.csv"

print("Train CSV exists:", train_csv.exists())


# %%
# Load train CSV
train_df = pd.read_csv(train_csv)

print("Train shape:", train_df.shape)
print("Columns:", train_df.columns.tolist())


# %%
# Select one patient
patient = train_df.iloc[0]

patient_id = patient["patient_id"]
flair_path = Path(patient["flair"])
seg_path = Path(patient["seg"])

print("Patient ID:", patient_id)
print("FLAIR exists:", flair_path.exists())
print("SEG exists:", seg_path.exists())


# %%
# Load MRI and segmentation mask
flair_data = nib.load(str(flair_path)).get_fdata()
seg_data = nib.load(str(seg_path)).get_fdata()

print("FLAIR shape:", flair_data.shape)
print("SEG shape:", seg_data.shape)
print("SEG unique labels:", np.unique(seg_data))


# %%
# Find slice with largest tumor area
tumor_mask = seg_data > 0
tumor_area_per_slice = tumor_mask.sum(axis=(0, 1))

slice_idx = int(np.argmax(tumor_area_per_slice))

flair_slice = flair_data[:, :, slice_idx]
seg_slice = seg_data[:, :, slice_idx]

print("Best tumor slice index:", slice_idx)
print("Tumor pixels in this slice:", tumor_area_per_slice[slice_idx])
print("Unique labels in this slice:", np.unique(seg_slice))


# %%
# Save FLAIR + full segmentation overlay
save_path_overlay = RESULTS_DIR / f"{patient_id}_flair_seg_overlay_slice_{slice_idx}.png"

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(flair_slice, cmap="gray")
plt.title(f"{patient_id} - FLAIR slice {slice_idx}")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(flair_slice, cmap="gray")
plt.imshow(seg_slice, cmap="jet", alpha=0.4)
plt.title(f"Segmentation overlay - labels {np.unique(seg_slice).astype(int)}")
plt.axis("off")

plt.tight_layout()
plt.savefig(save_path_overlay, dpi=300, bbox_inches="tight")
plt.show()

print("Saved overlay figure to:", save_path_overlay)


# %%
# Save label-wise overlay
save_path_labels = RESULTS_DIR / f"{patient_id}_labelwise_overlay_slice_{slice_idx}.png"

label_names = {
    1: "Label 1: Necrotic / non-enhancing tumor core",
    2: "Label 2: Edema",
    4: "Label 4: Enhancing tumor",
}

plt.figure(figsize=(15, 5))

for i, label in enumerate([1, 2, 4], start=1):
    label_mask = seg_slice == label

    plt.subplot(1, 3, i)
    plt.imshow(flair_slice, cmap="gray")
    plt.imshow(label_mask, cmap="Reds", alpha=0.5)
    plt.title(label_names[label])
    plt.axis("off")

plt.tight_layout()
plt.savefig(save_path_labels, dpi=300, bbox_inches="tight")
plt.show()

print("Saved label-wise figure to:", save_path_labels)
# %%
