# %%
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# %%
# Paths
PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")
CSV_DIR = PROJECT_ROOT / "data" / "csvs"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

train_csv = CSV_DIR / "train_paths.csv"

print("Train CSV exists:", train_csv.exists())


# %%
# Load one patient from train CSV
train_df = pd.read_csv(train_csv)

patient = train_df.iloc[0]

patient_id = patient["patient_id"]
flair_path = Path(patient["flair"])
seg_path = Path(patient["seg"])

print("Patient ID:", patient_id)
print("FLAIR exists:", flair_path.exists())
print("SEG exists:", seg_path.exists())


# %%
# Load FLAIR image and segmentation mask
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
# Normalize image slice to 0-1 range
def normalize_image(image):
    image = image.astype(np.float32)

    min_val = image.min()
    max_val = image.max()

    if max_val - min_val == 0:
        return image

    return (image - min_val) / (max_val - min_val)


flair_norm = normalize_image(flair_slice)

print("Normalized FLAIR min:", flair_norm.min())
print("Normalized FLAIR max:", flair_norm.max())


# %%
# Create degraded versions
blurred = gaussian_filter(flair_norm, sigma=2)

np.random.seed(42)
noise = np.random.normal(loc=0, scale=0.08, size=flair_norm.shape)
noisy = flair_norm + noise
noisy = np.clip(noisy, 0, 1)

low_contrast = 0.5 + 0.5 * (flair_norm - 0.5)

print("Clean min/max:", flair_norm.min(), flair_norm.max())
print("Blurred min/max:", blurred.min(), blurred.max())
print("Noisy min/max:", noisy.min(), noisy.max())
print("Low contrast min/max:", low_contrast.min(), low_contrast.max())


# %%
# Display and save clean vs degraded comparison
save_path_degradation = RESULTS_DIR / f"{patient_id}_degradation_examples_slice_{slice_idx}.png"

images = {
    "Clean": flair_norm,
    "Gaussian blur\nsigma=2": blurred,
    "Gaussian noise\nstd=0.08": noisy,
    "Low contrast": low_contrast,
}

plt.figure(figsize=(16, 4))

for i, (title, image) in enumerate(images.items(), start=1):
    plt.subplot(1, 4, i)
    plt.imshow(image, cmap="gray", vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.savefig(save_path_degradation, dpi=300, bbox_inches="tight")
plt.show()

print("Saved degradation comparison to:", save_path_degradation)


# %%
