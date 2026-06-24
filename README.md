# Robustness of Medical Image Segmentation Models to Image Quality Degradation

Independent study project on robustness of brain tumor segmentation models under image quality degradation using BraTS2020 multimodal MRI.

## Project Goal

This project evaluates how medical image segmentation models trained on clean MRI data perform when test images are degraded by common image quality artifacts.

The models are trained only on clean BraTS2020 images. Degradations are applied only during testing/evaluation.

## Dataset

Dataset: BraTS2020 TrainingData

Modalities:
- FLAIR
- T1
- T1ce
- T2

Segmentation labels were remapped from original BraTS labels `[0, 1, 2, 4]` to `[0, 1, 2, 3]`.

Raw medical image data are not included in this repository.

## Completed Work

### 2D Pilot

A 2D FLAIR-only U-Net pilot was completed to validate the training, testing, and degradation evaluation pipeline.

### Custom 4-Modal 3D U-Net

A custom 3D U-Net was trained on clean 4-modal BraTS2020 patches.

Clean held-out patch-based test result:

- Mean whole tumor Dice: 0.8938
- Mean whole tumor IoU: 0.8150

Robustness testing applied degradation to all four modalities during testing only.

At severity level 5, ghosting caused the largest Dice drop, followed by blur. Contrast, noise, and ringing had smaller effects under the current degradation settings.

### nnU-Net

BraTS2020 data were converted into nnU-Net v2 format as Dataset501_BraTS2020Multimodal.

Standard nnU-Net v2 3d_fullres fold 0 training was started as the second model baseline.

## Important Implementation Rule

Models are trained on clean images only. Degraded images are generated dynamically during evaluation and are not saved as a full degraded dataset.

## Repository Contents

This repository includes:

- Python scripts
- Small summary CSV/TXT files
- Selected plots and visualization figures
- Documentation notes

This repository does not include:

- Raw BraTS MRI data
- NIfTI files
- Model checkpoints
- nnU-Net raw/preprocessed/results folders
- Large per-patch or per-slice metric CSV files

## Citation

If using nnU-Net, cite:

Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 18, 203–211. https://doi.org/10.1038/s41592-020-01008-z
