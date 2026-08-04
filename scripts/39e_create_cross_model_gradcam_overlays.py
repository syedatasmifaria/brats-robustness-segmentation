from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageChops


ROOT = Path("/home/xfh25/brats_segmentation_project")
REPORT_DIR = ROOT / "report_materials"

OUTPUT_PNG = REPORT_DIR / "39e_cross_model_gradcam_overlays_final.png"
OUTPUT_PDF = REPORT_DIR / "39e_cross_model_gradcam_overlays_final.pdf"

PATIENT = "BraTS20_Training_178"

MODEL_INFORMATION = {
    "Custom 3D U-Net": {
        "folder": REPORT_DIR / "unet3d_xai_38b/final/figures",
        "prefix": "38b",
    },
    "nnU-Net v2": {
        "folder": REPORT_DIR / "nnunet_xai_37b/final/figures",
        "prefix": "37b",
    },
    "Swin-UNETR": {
        "folder": REPORT_DIR / "swin_xai_36b/final/figures",
        "prefix": "36b",
    },
}

CONDITIONS = {
    "Clean": "clean",
    "Ghosting L5": "ghosting_L5",
    "Noise L7": "noise_L7",
    "Contrast L10": "contrast_L10",
}

MODEL_ORDER = [
    "Custom 3D U-Net",
    "nnU-Net v2",
    "Swin-UNETR",
]


def trim_white_space(
    image: Image.Image,
    padding: int = 4,
) -> Image.Image:
    image = image.convert("RGB")

    background = Image.new(
        "RGB",
        image.size,
        (255, 255, 255),
    )

    difference = ImageChops.difference(
        image,
        background,
    ).convert("L")

    mask = difference.point(
        lambda value: 255 if value > 8 else 0
    )

    bounding_box = mask.getbbox()

    if bounding_box is None:
        return image

    left, top, right, bottom = bounding_box

    return image.crop(
        (
            max(0, left - padding),
            max(0, top - padding),
            min(image.width, right + padding),
            min(image.height, bottom + padding),
        )
    )


def extract_gradcam_overlay(path: Path) -> Image.Image:
    """
    The original image contains five panels:

    1. MRI
    2. Ground-truth WT
    3. Predicted WT
    4. Grad-CAM heatmap
    5. Grad-CAM overlay

    Retain only the fifth, rightmost panel.
    """
    image = trim_white_space(Image.open(path))

    width, height = image.size

    overlay = image.crop(
        (
            int(width * 0.80),
            int(height * 0.10),
            int(width * 0.995),
            int(height * 0.94),
        )
    )

    return trim_white_space(
        overlay,
        padding=2,
    )


def source_path(
    model: str,
    condition: str,
) -> Path:
    information = MODEL_INFORMATION[model]

    return (
        information["folder"]
        / (
            f'{information["prefix"]}_'
            f"{PATIENT}_"
            f"{condition}_gradcam.png"
        )
    )


def main():
    for model in MODEL_ORDER:
        for condition in CONDITIONS.values():
            path = source_path(model, condition)

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing source image: {path}"
                )

    fig, axes = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=(11.5, 8.3),
    )

    for row_index, model in enumerate(MODEL_ORDER):
        for column_index, (
            condition_label,
            condition_filename,
        ) in enumerate(CONDITIONS.items()):

            axis = axes[row_index, column_index]

            path = source_path(
                model,
                condition_filename,
            )

            overlay = extract_gradcam_overlay(path)

            axis.imshow(overlay)
            axis.axis("off")

            if row_index == 0:
                axis.set_title(
                    condition_label,
                    fontsize=12,
                    fontweight="bold",
                    pad=8,
                )

    row_positions = [
        0.775,
        0.500,
        0.225,
    ]

    for position, model in zip(
        row_positions,
        MODEL_ORDER,
    ):
        fig.text(
            0.027,
            position,
            model,
            rotation=90,
            va="center",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    fig.suptitle(
        (
            "Grad-CAM Attribution Under Selected MRI "
            "Degradation Conditions"
        ),
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )

    fig.text(
        0.5,
        0.018,
        (
            "Representative case: "
            "BraTS20_Training_178"
        ),
        ha="center",
        fontsize=9,
    )

    plt.subplots_adjust(
        left=0.065,
        right=0.995,
        top=0.925,
        bottom=0.055,
        wspace=0.025,
        hspace=0.055,
    )

    fig.savefig(
        OUTPUT_PNG,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print("Final Grad-CAM overlay figure created.")
    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
