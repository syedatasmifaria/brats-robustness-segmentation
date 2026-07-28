from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


PROJECT_ROOT = Path(
    "/home/xfh25/brats_segmentation_project"
)

PATIENT_NUMBER = os.environ.get(
    "XAI_PATIENT_NUMBER",
    "178",
)

PATCH_SIZE = (128, 128, 128)

MODEL_FOLDER = (
    PROJECT_ROOT
    / "nnunet"
    / "nnUNet_results"
    / "Dataset501_BraTS2020Multimodal"
    / "nnUNetTrainer__nnUNetPlans__3d_fullres"
)

IMAGES_DIR = (
    PROJECT_ROOT
    / "nnunet"
    / "nnUNet_raw"
    / "Dataset501_BraTS2020Multimodal"
    / "imagesTs"
)

ORIGINAL_PATIENT_DIR = (
    PROJECT_ROOT
    / "data"
    / "BraTS2020_TrainingData"
    / "MICCAI_BraTS2020_TrainingData"
    / f"BraTS20_Training_{PATIENT_NUMBER}"
)

SEGMENTATION_PATH = (
    ORIGINAL_PATIENT_DIR
    / f"BraTS20_Training_{PATIENT_NUMBER}_seg.nii"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "report_materials"
    / "nnunet_xai_smoke"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def compute_patch_bounds(
    center,
    volume_shape,
    patch_size,
):
    starts = []
    ends = []

    for axis in range(3):
        start = int(
            center[axis]
            - patch_size[axis] // 2
        )

        start = max(start, 0)
        end = start + patch_size[axis]

        if end > volume_shape[axis]:
            end = volume_shape[axis]
            start = end - patch_size[axis]

        starts.append(start)
        ends.append(end)

    return tuple(starts), tuple(ends)


def normalize_cam(cam):
    cam = cam - cam.min()
    maximum = cam.max()

    if maximum > 0:
        cam = cam / maximum

    return cam


def dice_score(prediction, truth):
    intersection = np.logical_and(
        prediction,
        truth,
    ).sum()

    denominator = (
        prediction.sum()
        + truth.sum()
    )

    if denominator == 0:
        return 1.0

    return (
        2.0
        * float(intersection)
        / float(denominator)
    )


def main():
    torch.manual_seed(2026)
    np.random.seed(2026)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for this smoke test."
        )

    os.environ["nnUNet_raw"] = str(
        PROJECT_ROOT
        / "nnunet"
        / "nnUNet_raw"
    )

    os.environ["nnUNet_preprocessed"] = str(
        PROJECT_ROOT
        / "nnunet"
        / "nnUNet_preprocessed"
    )

    os.environ["nnUNet_results"] = str(
        PROJECT_ROOT
        / "nnunet"
        / "nnUNet_results"
    )

    device = torch.device("cuda:0")

    image_files = [
        str(
            IMAGES_DIR
            / (
                f"BRATS_{PATIENT_NUMBER}_"
                f"{channel:04d}.nii.gz"
            )
        )
        for channel in range(4)
    ]

    for image_file in image_files:
        if not Path(image_file).exists():
            raise FileNotFoundError(
                f"Missing MRI file: {image_file}"
            )

    if not SEGMENTATION_PATH.exists():
        raise FileNotFoundError(
            f"Missing ground truth: {SEGMENTATION_PATH}"
        )

    raw_data, image_properties = (
        SimpleITKIO().read_images(
            image_files
        )
    )

    raw_segmentation = (
        sitk.GetArrayFromImage(
            sitk.ReadImage(
                str(SEGMENTATION_PATH)
            )
        )
        .astype(np.int16)[None]
    )

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )

    predictor.initialize_from_trained_model_folder(
        MODEL_FOLDER,
        use_folds=(0,),
        checkpoint_name="checkpoint_best.pth",
    )

    model = predictor.network.to(device)
    model.eval()

    preprocessor = (
        predictor
        .configuration_manager
        .preprocessor_class(
            verbose=False
        )
    )

    data, segmentation, properties = (
        preprocessor.run_case_npy(
            raw_data,
            raw_segmentation,
            image_properties,
            predictor.plans_manager,
            predictor.configuration_manager,
            predictor.dataset_json,
        )
    )

    whole_tumor = segmentation[0] > 0
    coordinates = np.argwhere(
        whole_tumor
    )

    if coordinates.size == 0:
        raise RuntimeError(
            "The selected patient has no WT voxels."
        )

    tumor_center = np.rint(
        coordinates.mean(axis=0)
    ).astype(int)

    patch_start, patch_end = (
        compute_patch_bounds(
            center=tumor_center,
            volume_shape=data.shape[1:],
            patch_size=PATCH_SIZE,
        )
    )

    z0, y0, x0 = patch_start
    z1, y1, x1 = patch_end

    image_patch = data[
        :,
        z0:z1,
        y0:y1,
        x0:x1,
    ]

    truth_patch = segmentation[
        0,
        z0:z1,
        y0:y1,
        x0:x1,
    ]

    if image_patch.shape != (
        4,
        *PATCH_SIZE,
    ):
        raise RuntimeError(
            "Unexpected image patch shape: "
            f"{image_patch.shape}"
        )

    input_tensor = (
        torch.from_numpy(
            image_patch
        )
        .unsqueeze(0)
        .float()
        .to(device)
    )

    candidate_layers = {
        "decoder_stage_2": (
            model.decoder.stages[2]
        ),
        "decoder_stage_3": (
            model.decoder.stages[3]
        ),
    }

    activation_records = {}
    gradient_records = {}
    hooks = []

    def forward_hook(name):
        def hook(
            module,
            inputs,
            output,
        ):
            activation_records[name] = output

            def save_gradient(gradient):
                gradient_records[name] = (
                    gradient
                )

            output.register_hook(
                save_gradient
            )

        return hook

    for layer_name, layer in (
        candidate_layers.items()
    ):
        hooks.append(
            layer.register_forward_hook(
                forward_hook(layer_name)
            )
        )

    model.zero_grad(
        set_to_none=True
    )

    logits = model(
        input_tensor
    )

    tumor_logit = torch.logsumexp(
        logits[:, 1:4],
        dim=1,
    )

    predicted_labels = torch.argmax(
        logits.detach(),
        dim=1,
    )[0]

    target_mask = predicted_labels > 0

    if not torch.any(target_mask):
        raise RuntimeError(
            "The model predicted no WT voxels "
            "in the selected patch."
        )

    target_score = tumor_logit[0][
        target_mask
    ].mean()

    target_score.backward()

    prediction_patch = (
        predicted_labels
        .cpu()
        .numpy()
        .astype(np.int16)
    )

    truth_wt = truth_patch > 0
    prediction_wt = prediction_patch > 0

    wt_counts_by_slice = (
        truth_wt.sum(
            axis=(1, 2)
        )
    )

    local_slice = int(
        np.argmax(
            wt_counts_by_slice
        )
    )

    global_slice = (
        z0 + local_slice
    )

    flair_slice = image_patch[
        0,
        local_slice,
        :,
        :,
    ]

    truth_slice = truth_wt[
        local_slice,
        :,
        :,
    ]

    prediction_slice = prediction_wt[
        local_slice,
        :,
        :,
    ]

    captured_wt = int(
        truth_wt.sum()
    )

    total_wt = int(
        whole_tumor.sum()
    )

    summary_lines = [
        "nnU-Net Grad-CAM patch smoke test",
        f"Patient: BRATS_{PATIENT_NUMBER}",
        "Model: PlainConvUNet",
        "Checkpoint: checkpoint_best.pth",
        f"Patch size: {PATCH_SIZE}",
        f"Preprocessed shape: {data.shape}",
        f"Patch start (z, y, x): {patch_start}",
        f"Patch end (z, y, x): {patch_end}",
        (
            "Tumor center (z, y, x): "
            f"{tumor_center.tolist()}"
        ),
        f"Local axial slice: {local_slice}",
        f"Global axial slice: {global_slice}",
        (
            "Target definition: mean differentiable "
            "tumor logit inside the model-predicted "
            "WT region"
        ),
        (
            "Ground truth use: patch placement and "
            "descriptive localization checks only"
        ),
        (
            f"Target score: "
            f"{float(target_score.detach().cpu()):.8f}"
        ),
        (
            f"True WT patch voxels: "
            f"{captured_wt}"
        ),
        (
            f"Total preprocessed WT voxels: "
            f"{total_wt}"
        ),
        (
            "WT captured by patch: "
            f"{100 * captured_wt / total_wt:.2f}%"
        ),
        (
            f"Predicted WT patch voxels: "
            f"{int(prediction_wt.sum())}"
        ),
        (
            f"Patch WT Dice: "
            f"{dice_score(prediction_wt, truth_wt):.6f}"
        ),
    ]

    for layer_name in candidate_layers:
        activations = (
            activation_records[
                layer_name
            ]
        )

        gradients = (
            gradient_records[
                layer_name
            ]
        )

        weights = gradients.mean(
            dim=(2, 3, 4),
            keepdim=True,
        )

        cam = torch.sum(
            weights * activations,
            dim=1,
            keepdim=True,
        )

        cam = torch.relu(
            cam
        )

        cam = F.interpolate(
            cam,
            size=PATCH_SIZE,
            mode="trilinear",
            align_corners=False,
        )

        cam_np = (
            cam[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        cam_maximum_before_normalization = (
            float(
                cam_np.max()
            )
        )

        cam_np = normalize_cam(
            cam_np
        )

        np.save(
            OUTPUT_DIR
            / (
                f"37a_BRATS_{PATIENT_NUMBER}_"
                f"{layer_name}_gradcam.npy"
            ),
            cam_np,
        )

        inside_mean = float(
            cam_np[truth_wt].mean()
        )

        outside_mean = float(
            cam_np[~truth_wt].mean()
        )

        inside_outside_ratio = (
            inside_mean
            / max(
                outside_mean,
                1e-12,
            )
        )

        threshold = np.quantile(
            cam_np,
            0.90,
        )

        top_10_mask = (
            cam_np >= threshold
        )

        top_10_inside_fraction = (
            np.logical_and(
                top_10_mask,
                truth_wt,
            ).sum()
            / max(
                top_10_mask.sum(),
                1,
            )
        )

        wt_covered_by_top_10 = (
            np.logical_and(
                top_10_mask,
                truth_wt,
            ).sum()
            / max(
                truth_wt.sum(),
                1,
            )
        )

        cam_slice = cam_np[
            local_slice,
            :,
            :,
        ]

        figure, axes = plt.subplots(
            1,
            5,
            figsize=(20, 4),
        )

        axes[0].imshow(
            flair_slice,
            cmap="gray",
            origin="lower",
        )
        axes[0].set_title(
            "Normalized FLAIR"
        )

        axes[1].imshow(
            flair_slice,
            cmap="gray",
            origin="lower",
        )
        axes[1].imshow(
            truth_slice,
            cmap="Reds",
            alpha=0.45,
            origin="lower",
        )
        axes[1].set_title(
            "Ground-truth WT"
        )

        axes[2].imshow(
            flair_slice,
            cmap="gray",
            origin="lower",
        )
        axes[2].imshow(
            prediction_slice,
            cmap="Blues",
            alpha=0.45,
            origin="lower",
        )
        axes[2].set_title(
            "Predicted WT"
        )

        axes[3].imshow(
            cam_slice,
            cmap="jet",
            origin="lower",
            vmin=0,
            vmax=1,
        )
        axes[3].set_title(
            f"{layer_name} Grad-CAM"
        )

        axes[4].imshow(
            flair_slice,
            cmap="gray",
            origin="lower",
        )
        axes[4].imshow(
            cam_slice,
            cmap="jet",
            alpha=0.50,
            origin="lower",
            vmin=0,
            vmax=1,
        )
        axes[4].set_title(
            "Grad-CAM overlay"
        )

        for axis in axes:
            axis.axis("off")

        figure.suptitle(
            (
                f"BRATS_{PATIENT_NUMBER} | "
                f"axial slice {global_slice} | "
                f"{layer_name}"
            ),
            fontsize=13,
        )

        figure.tight_layout()

        figure_path = (
            OUTPUT_DIR
            / (
                f"37a_BRATS_{PATIENT_NUMBER}_"
                f"{layer_name}_gradcam.png"
            )
        )

        figure.savefig(
            figure_path,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(
            figure
        )

        summary_lines.extend(
            [
                "",
                f"Layer: {layer_name}",
                (
                    "Activation shape: "
                    f"{tuple(activations.shape)}"
                ),
                (
                    "Gradient shape: "
                    f"{tuple(gradients.shape)}"
                ),
                (
                    "Gradient absolute mean: "
                    f"{float(gradients.abs().mean().detach().cpu()):.10e}"
                ),
                (
                    "CAM maximum before normalization: "
                    f"{cam_maximum_before_normalization:.10e}"
                ),
                (
                    "Mean CAM inside WT: "
                    f"{inside_mean:.6f}"
                ),
                (
                    "Mean CAM outside WT: "
                    f"{outside_mean:.6f}"
                ),
                (
                    "Inside/outside CAM ratio: "
                    f"{inside_outside_ratio:.4f}"
                ),
                (
                    "Top-10% CAM voxels inside WT: "
                    f"{100 * top_10_inside_fraction:.2f}%"
                ),
                (
                    "WT covered by top-10% CAM: "
                    f"{100 * wt_covered_by_top_10:.2f}%"
                ),
                f"Saved figure: {figure_path}",
            ]
        )

    summary_lines.extend(
        [
            "",
            "Selected target layer: decoder.stages[3]",
            (
                "Reason: better tumor localization at "
                "64 x 64 x 64 resolution than "
                "decoder.stages[2]."
            ),
            (
                "Interpretation: Grad-CAM is a spatial "
                "attribution map, not a causal account "
                "of model reasoning."
            ),
        ]
    )

    for hook in hooks:
        hook.remove()

    summary_path = (
        OUTPUT_DIR
        / (
            f"37a_BRATS_{PATIENT_NUMBER}_"
            "gradcam_smoke_summary.txt"
        )
    )

    summary_path.write_text(
        "\n".join(
            summary_lines
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "\n".join(
            summary_lines
        )
    )

    print(
        f"\nSummary saved to: {summary_path}"
    )

    print(
        "\nnnU-Net Grad-CAM smoke test "
        "completed successfully."
    )


if __name__ == "__main__":
    main()
