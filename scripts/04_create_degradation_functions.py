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
# Helper function: normalize image to 0-1 range

def normalize_image(image):
    image = image.astype(np.float32)

    min_val = image.min()
    max_val = image.max()

    if max_val - min_val == 0:
        return image

    return (image - min_val) / (max_val - min_val)


# %%
# Degradation function 1: Gaussian blur

def apply_gaussian_blur(image, sigma=2):
    """
    Apply Gaussian blur to simulate loss of sharpness.
    """
    blurred = gaussian_filter(image, sigma=sigma)
    blurred = np.clip(blurred, 0, 1)

    return blurred


# %%
# Degradation function 2: Gaussian noise

def apply_gaussian_noise(image, std=0.08, seed=42):
    """
    Add Gaussian noise to simulate noisy MRI acquisition.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0, scale=std, size=image.shape)

    noisy = image + noise
    noisy = np.clip(noisy, 0, 1)

    return noisy


# %%
# Degradation function 3: Low contrast

def apply_low_contrast(image, factor=0.5):
    """
    Reduce contrast around the middle intensity value 0.5.

    factor < 1 reduces contrast.
    factor = 1 keeps original contrast.
    """
    low_contrast = 0.5 + factor * (image - 0.5)
    low_contrast = np.clip(low_contrast, 0, 1)

    return low_contrast

# %%
# Degradation function 4: Ringing artifact

def apply_ringing_artifact(image, keep_fraction=0.65):
    """
    Simulate ringing artifact by removing high-frequency information
    in the Fourier domain.

    Smaller keep_fraction = stronger artifact.
    """
    image = image.astype(np.float32)

    # Fourier transform
    f = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f)

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # Create centered square mask
    mask = np.zeros_like(image, dtype=np.float32)

    keep_rows = int(rows * keep_fraction / 2)
    keep_cols = int(cols * keep_fraction / 2)

    mask[
        crow - keep_rows : crow + keep_rows,
        ccol - keep_cols : ccol + keep_cols
    ] = 1

    # Apply mask in frequency domain
    f_shift_filtered = f_shift * mask

    # Inverse Fourier transform
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    image_back = np.fft.ifft2(f_ishift)
    image_back = np.abs(image_back)

    # Normalize back to 0-1
    image_back = normalize_image(image_back)

    return image_back

# %%
# Degradation function 5: Ghosting artifact

def apply_ghosting_artifact(image, shift=12, intensity=0.35):
    """
    Simulate ghosting artifact by adding a faint shifted copy
    of the image back onto itself.

    shift controls how far the ghost copy is moved.
    intensity controls how visible the ghost copy is.
    """
    image = image.astype(np.float32)

    # Shift image horizontally
    shifted = np.roll(image, shift=shift, axis=1)

    # Blend original image and shifted ghost image
    ghosted = image + intensity * shifted

    # Normalize back to 0-1
    ghosted = normalize_image(ghosted)

    return ghosted

# %%
# Test the degradation functions on one BraTS patient slice

train_df = pd.read_csv(train_csv)

patient = train_df.iloc[0]

patient_id = patient["patient_id"]
flair_path = Path(patient["flair"])
seg_path = Path(patient["seg"])

flair_data = nib.load(str(flair_path)).get_fdata()
seg_data = nib.load(str(seg_path)).get_fdata()

tumor_mask = seg_data > 0
tumor_area_per_slice = tumor_mask.sum(axis=(0, 1))
slice_idx = int(np.argmax(tumor_area_per_slice))

flair_slice = flair_data[:, :, slice_idx]
flair_norm = normalize_image(flair_slice)

blur_test = apply_gaussian_blur(flair_norm, sigma=2)
noise_test = apply_gaussian_noise(flair_norm, std=0.08, seed=42)
contrast_test = apply_low_contrast(flair_norm, factor=0.5)
ringing_test = apply_ringing_artifact(flair_norm, keep_fraction=0.65)
ghosting_test = apply_ghosting_artifact(flair_norm, shift=12, intensity=0.35)

print("Patient ID:", patient_id)
print("Best tumor slice index:", slice_idx)

print("Clean min/max:", flair_norm.min(), flair_norm.max())
print("Blur test min/max:", blur_test.min(), blur_test.max())
print("Noise test min/max:", noise_test.min(), noise_test.max())
print("Contrast test min/max:", contrast_test.min(), contrast_test.max())
print("Ringing test min/max:", ringing_test.min(), ringing_test.max())
print("Ghosting test min/max:", ghosting_test.min(), ghosting_test.max())

# %%
# Define 5 severity levels for each degradation type

BLUR_LEVELS = [0.5, 1.0, 1.5, 2.0, 2.5]

NOISE_LEVELS = [0.02, 0.04, 0.06, 0.08, 0.10]

CONTRAST_LEVELS = [0.9, 0.75, 0.6, 0.45, 0.3]

RINGING_LEVELS = [0.75, 0.60, 0.45, 0.30, 0.20]

GHOSTING_LEVELS = [
    {"shift": 4, "intensity": 0.15},
    {"shift": 8, "intensity": 0.25},
    {"shift": 12, "intensity": 0.35},
    {"shift": 16, "intensity": 0.45},
    {"shift": 20, "intensity": 0.55},
]

print("Blur levels:", BLUR_LEVELS)
print("Noise levels:", NOISE_LEVELS)
print("Contrast levels:", CONTRAST_LEVELS)
print("Ringing levels:", RINGING_LEVELS)
print("Ghosting levels:", GHOSTING_LEVELS)


# %%
# Visualize 5 levels of Gaussian blur

blur_images = {}

for sigma in BLUR_LEVELS:
    blur_images[f"Blur sigma={sigma}"] = apply_gaussian_blur(flair_norm, sigma=sigma)

plt.figure(figsize=(18, 4))

for i, (title, image) in enumerate(blur_images.items(), start=1):
    plt.subplot(1, 5, i)
    plt.imshow(image, cmap="gray", vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()

# %%
# Save 5 levels of Gaussian blur

save_path_blur_levels = RESULTS_DIR / f"{patient_id}_blur_5_levels_slice_{slice_idx}.png"

blur_images = {}

for sigma in BLUR_LEVELS:
    blur_images[f"Blur sigma={sigma}"] = apply_gaussian_blur(flair_norm, sigma=sigma)

plt.figure(figsize=(18, 4))

for i, (title, image) in enumerate(blur_images.items(), start=1):
    plt.subplot(1, 5, i)
    plt.imshow(image, cmap="gray", vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.savefig(save_path_blur_levels, dpi=300, bbox_inches="tight")
plt.show()

print("Saved blur levels figure to:", save_path_blur_levels)


# %%
# Visualize 5 levels of Gaussian noise

noise_images = {}

for std in NOISE_LEVELS:
    noise_images[f"Noise std={std}"] = apply_gaussian_noise(
        flair_norm,
        std=std,
        seed=42
    )

plt.figure(figsize=(18, 4))

for i, (title, image) in enumerate(noise_images.items(), start=1):
    plt.subplot(1, 5, i)
    plt.imshow(image, cmap="gray", vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()


# %%
# Save 5 levels of Gaussian noise

save_path_noise_levels = RESULTS_DIR / f"{patient_id}_noise_5_levels_slice_{slice_idx}.png"

noise_images = {}

for std in NOISE_LEVELS:
    noise_images[f"Noise std={std}"] = apply_gaussian_noise(
        flair_norm,
        std=std,
        seed=42
    )

plt.figure(figsize=(18, 4))

for i, (title, image) in enumerate(noise_images.items(), start=1):
    plt.subplot(1, 5, i)
    plt.imshow(image, cmap="gray", vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.savefig(save_path_noise_levels, dpi=300, bbox_inches="tight")
plt.show()

print("Saved noise levels figure to:", save_path_noise_levels)


# %%
# Visualize 5 levels of contrast degradation

contrast_images = {}

for factor in CONTRAST_LEVELS:
    contrast_images[f"Contrast factor={factor}"] = apply_low_contrast(
        flair_norm,
        factor=factor
    )

plt.figure(figsize=(18, 4))

for i, (title, image) in enumerate(contrast_images.items(), start=1):
    plt.subplot(1, 5, i)
    plt.imshow(image, cmap="gray", vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()

# %%
# Save 5 levels of contrast degradation

save_path_contrast_levels = RESULTS_DIR / f"{patient_id}_contrast_5_levels_slice_{slice_idx}.png"

contrast_images = {}

for factor in CONTRAST_LEVELS:
    contrast_images[f"Contrast factor={factor}"] = apply_low_contrast(
        flair_norm,
        factor=factor
    )

plt.figure(figsize=(18, 4))

for i, (title, image) in enumerate(contrast_images.items(), start=1):
    plt.subplot(1, 5, i)
    plt.imshow(image, cmap="gray", vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.savefig(save_path_contrast_levels, dpi=300, bbox_inches="tight")
plt.show()

print("Saved contrast levels figure to:", save_path_contrast_levels)


# %%
# Visualize 5 levels of ringing artifact

ringing_images = {}

for keep_fraction in RINGING_LEVELS:
    ringing_images[f"Ringing keep={keep_fraction}"] = apply_ringing_artifact(
        flair_norm,
        keep_fraction=keep_fraction
    )

plt.figure(figsize=(18, 4))

for i, (title, image) in enumerate(ringing_images.items(), start=1):
    plt.subplot(1, 5, i)
    plt.imshow(image, cmap="gray", vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()


# %%
# Save 5 levels of ringing artifact

save_path_ringing_levels = RESULTS_DIR / f"{patient_id}_ringing_5_levels_slice_{slice_idx}.png"

ringing_images = {}

for keep_fraction in RINGING_LEVELS:
    ringing_images[f"Ringing keep={keep_fraction}"] = apply_ringing_artifact(
        flair_norm,
        keep_fraction=keep_fraction
    )

plt.figure(figsize=(18, 4))

for i, (title, image) in enumerate(ringing_images.items(), start=1):
    plt.subplot(1, 5, i)
    plt.imshow(image, cmap="gray", vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.savefig(save_path_ringing_levels, dpi=300, bbox_inches="tight")
plt.show()

print("Saved ringing levels figure to:", save_path_ringing_levels)


# %%
# Visualize 5 levels of ghosting artifact

ghosting_images = {}

for params in GHOSTING_LEVELS:
    shift = params["shift"]
    intensity = params["intensity"]

    ghosting_images[f"Ghost shift={shift}\nintensity={intensity}"] = apply_ghosting_artifact(
        flair_norm,
        shift=shift,
        intensity=intensity
    )

plt.figure(figsize=(18, 4))

for i, (title, image) in enumerate(ghosting_images.items(), start=1):
    plt.subplot(1, 5, i)
    plt.imshow(image, cmap="gray", vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()


# %%
# Save 5 levels of ghosting artifact

save_path_ghosting_levels = RESULTS_DIR / f"{patient_id}_ghosting_5_levels_slice_{slice_idx}.png"

ghosting_images = {}

for params in GHOSTING_LEVELS:
    shift = params["shift"]
    intensity = params["intensity"]

    ghosting_images[f"Ghost shift={shift}\nintensity={intensity}"] = apply_ghosting_artifact(
        flair_norm,
        shift=shift,
        intensity=intensity
    )

plt.figure(figsize=(18, 4))

for i, (title, image) in enumerate(ghosting_images.items(), start=1):
    plt.subplot(1, 5, i)
    plt.imshow(image, cmap="gray", vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.savefig(save_path_ghosting_levels, dpi=300, bbox_inches="tight")
plt.show()

print("Saved ghosting levels figure to:", save_path_ghosting_levels)

# %%
