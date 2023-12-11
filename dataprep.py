from pathlib import Path

import numpy as np
from patchify import patchify
from PIL import Image


def create_folders():
    FOLDERS = ["train", "val", "test"]

    for folder in FOLDERS:
        Path(f"training-data/{folder}/images").mkdir(parents=True, exist_ok=True)
        Path(f"training-data/{folder}/masks").mkdir(parents=True, exist_ok=True)


def create_patches(src: Path, dest: Path):
    image = Image.open(src)
    image = np.asarray(image)

    patches = patchify(image, (512, 512, 3), step=448)

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch = patches[i, j, 0]

            patch_image = Image.fromarray(patch)
            patch_image.save(dest / f"{src.stem}_{i}_{j}.png")


create_folders()

images = list(Path("data/counted").glob("*.jpg"))
total_images = len(images)

for i in range(total_images):
    image = images[i]

    if i < total_images * 0.6:
        dest = Path("training-data/train")
    elif i < total_images * 0.8:
        dest = Path("training-data/val")
    else:
        dest = Path("training-data/test")

    mask_path = image.parent / "masks" / image.name.replace(".jpg", ".png")

    create_patches(image, dest / "images")
    create_patches(mask_path, dest / "masks")
