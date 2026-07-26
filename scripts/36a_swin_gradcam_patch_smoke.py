from pathlib import Path
import os
import importlib.util

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR


PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")
PATIENT_ID = os.environ.get(
    "XAI_PATIENT_ID",
    "BraTS20_Training_178",
)
PATCH_SIZE = (96, 96, 96)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "swin_unetr"
    / "swin_unetr_full_timing_20260719"
    / "best_checkpoint.pth"
)

TEST_CSV = PROJECT_ROOT / "data" / "csvs" / "test_paths.csv"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "34a_train_swin_unetr_clean.py"

OUTPUT_DIR = PROJECT_ROOT / "report_materials" / "xai_smoke"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_training_module():
    spec = importlib.util.spec_from_file_location(
        "swin_training_module",
        TRAIN_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the Swin training script.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_patch_bounds(center, volume_shape, patch_size):
    starts = []
    ends = []

    for axis in range(3):
        start = int(center[axis] - patch_size[axis] // 2)
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


def main():
    torch.manual_seed(2026)
    np.random.seed(2026)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this smoke test.")

    device = torch.device("cuda:0")
    training_module = load_training_module()

    test_df = pd.read_csv(TEST_CSV)
    patient_rows = test_df[
        test_df["patient_id"] == PATIENT_ID
    ]

    if patient_rows.empty:
        raise RuntimeError(f"{PATIENT_ID} was not found in {TEST_CSV}.")

    row = patient_rows.iloc[0]

    image, truth = training_module.load_full_patient(row)

    wt_mask = truth > 0
    coordinates = np.argwhere(wt_mask)

    if coordinates.size == 0:
        raise RuntimeError("The selected patient has no WT voxels.")

    tumor_center = np.rint(
        coordinates.mean(axis=0)
    ).astype(int)

    patch_start, patch_end = compute_patch_bounds(
        center=tumor_center,
        volume_shape=truth.shape,
        patch_size=PATCH_SIZE,
    )

    x0, y0, z0 = patch_start
    x1, y1, z1 = patch_end

    image_patch = image[
        :,
        x0:x1,
        y0:y1,
        z0:z1,
    ]

    truth_patch = truth[
        x0:x1,
        y0:y1,
        z0:z1,
    ]

    if image_patch.shape != (4, *PATCH_SIZE):
        raise RuntimeError(
            f"Unexpected image patch shape: {image_patch.shape}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    model_configuration = checkpoint["model_configuration"]

    model = SwinUNETR(
        in_channels=int(model_configuration["in_channels"]),
        out_channels=int(model_configuration["out_channels"]),
        feature_size=int(model_configuration["feature_size"]),
        use_checkpoint=bool(model_configuration["use_checkpoint"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    candidate_layers = {
        "decoder3": model.decoder3.conv_block,
        "decoder2": model.decoder2.conv_block,
    }

    activation_records = {}
    gradient_records = {}
    hooks = []

    def forward_hook(name):
        def hook(module, inputs, output):
            activation_records[name] = output

            def save_gradient(gradient):
                gradient_records[name] = gradient

            output.register_hook(save_gradient)

        return hook

    for layer_name, layer in candidate_layers.items():
        hooks.append(
            layer.register_forward_hook(
                forward_hook(layer_name)
            )
        )

    input_tensor = torch.from_numpy(
        image_patch
    ).unsqueeze(0).float().to(device)

    truth_tensor = torch.from_numpy(
        truth_patch
    ).to(device)

    model.zero_grad(set_to_none=True)

    logits = model(input_tensor)

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
            "The model predicted no WT voxels in the selected patch."
        )

    score = tumor_logit[0][target_mask].mean()
    score.backward()

    prediction_patch = predicted_labels.cpu().numpy().astype(
        np.int16
    )

    wt_counts_by_slice = (
        truth_patch > 0
    ).sum(axis=(0, 1))

    local_slice = int(np.argmax(wt_counts_by_slice))
    global_slice = z0 + local_slice

    flair_slice = image_patch[0, :, :, local_slice]
    truth_slice = truth_patch[:, :, local_slice] > 0
    prediction_slice = prediction_patch[:, :, local_slice] > 0

    summary_lines = [
        "Swin-UNETR Grad-CAM patch smoke test",
        f"Patient: {PATIENT_ID}",
        f"Checkpoint epoch: {checkpoint['epoch']}",
        f"Patch start: {patch_start}",
        f"Patch end: {patch_end}",
        f"Tumor center: {tumor_center.tolist()}",
        f"Local axial slice: {local_slice}",
        f"Global axial slice: {global_slice}",
        f"Target score: {float(score.detach().cpu()):.8f}",
        f"True WT patch voxels: {int((truth_patch > 0).sum())}",
        f"Predicted WT patch voxels: {int((prediction_patch > 0).sum())}",
    ]

    for layer_name in candidate_layers:
        activations = activation_records[layer_name]
        gradients = gradient_records[layer_name]

        weights = gradients.mean(
            dim=(2, 3, 4),
            keepdim=True,
        )

        cam = torch.sum(
            weights * activations,
            dim=1,
            keepdim=True,
        )

        cam = torch.relu(cam)

        cam = F.interpolate(
            cam,
            size=PATCH_SIZE,
            mode="trilinear",
            align_corners=False,
        )

        cam_np = cam[
            0, 0
        ].detach().cpu().numpy()

        cam_np = normalize_cam(cam_np)

        np.save(
            OUTPUT_DIR
            / f"36a_{PATIENT_ID}_{layer_name}_gradcam.npy",
            cam_np,
        )

        cam_slice = cam_np[:, :, local_slice]

        figure, axes = plt.subplots(
            1,
            5,
            figsize=(20, 4),
        )

        axes[0].imshow(
            flair_slice.T,
            cmap="gray",
            origin="lower",
        )
        axes[0].set_title("Normalized FLAIR")

        axes[1].imshow(
            flair_slice.T,
            cmap="gray",
            origin="lower",
        )
        axes[1].imshow(
            truth_slice.T,
            cmap="Reds",
            alpha=0.45,
            origin="lower",
        )
        axes[1].set_title("Ground-truth WT")

        axes[2].imshow(
            flair_slice.T,
            cmap="gray",
            origin="lower",
        )
        axes[2].imshow(
            prediction_slice.T,
            cmap="Blues",
            alpha=0.45,
            origin="lower",
        )
        axes[2].set_title("Predicted WT")

        axes[3].imshow(
            cam_slice.T,
            cmap="jet",
            origin="lower",
            vmin=0,
            vmax=1,
        )
        axes[3].set_title(f"{layer_name} Grad-CAM")

        axes[4].imshow(
            flair_slice.T,
            cmap="gray",
            origin="lower",
        )
        axes[4].imshow(
            cam_slice.T,
            cmap="jet",
            alpha=0.50,
            origin="lower",
            vmin=0,
            vmax=1,
        )
        axes[4].set_title("Grad-CAM overlay")

        for axis in axes:
            axis.axis("off")

        figure.suptitle(
            f"{PATIENT_ID} | axial slice {global_slice} | {layer_name}",
            fontsize=13,
        )

        figure.tight_layout()

        figure_path = (
            OUTPUT_DIR
            / f"36a_{PATIENT_ID}_{layer_name}_gradcam.png"
        )

        figure.savefig(
            figure_path,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(figure)

        summary_lines.extend(
            [
                "",
                f"Layer: {layer_name}",
                f"Activation shape: {tuple(activations.shape)}",
                f"Gradient shape: {tuple(gradients.shape)}",
                f"Gradient absolute mean: "
                f"{float(gradients.abs().mean().detach().cpu()):.10f}",
                f"CAM maximum before normalization: "
                f"{float(cam.detach().max().cpu()):.10f}",
                f"Saved figure: {figure_path}",
            ]
        )

    for hook in hooks:
        hook.remove()

    summary_path = (
        OUTPUT_DIR
        / f"36a_{PATIENT_ID}_gradcam_smoke_summary.txt"
    )

    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(summary_lines))
    print(f"\nSummary saved to: {summary_path}")
    print("\nGrad-CAM smoke test completed successfully.")


if __name__ == "__main__":
    main()
