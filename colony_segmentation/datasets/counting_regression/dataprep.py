import os
from pathlib import Path
from shutil import copy2


def create_folders():
    FOLDERS = ["train", "val", "test"]

    for folder in FOLDERS:
        Path(f"training_data/{folder}/images").mkdir(parents=True, exist_ok=True)
        Path(f"training_data/{folder}/masks").mkdir(parents=True, exist_ok=True)


create_folders()

image_paths = list(Path("../../../data/labeled/images").glob("*.jpg"))
mask_paths = list(Path("../../../data/labeled/masks").glob("*.png"))

pairs = []

for image_path in image_paths:
    mask_path = None

    for mask in mask_paths:
        if mask.stem.startswith(image_path.stem):
            mask_path = mask
            break

    if mask_path is not None:
        pairs.append((image_path, mask_path))

for i in range(len(pairs)):
    image_mask_pair = pairs[i]

    if i < len(pairs) * 0.6:
        dest = Path("training_data/train")
    elif i < len(pairs) * 0.8:
        dest = Path("training_data/val")
    else:
        dest = Path("training_data/test")

    copy2(image_mask_pair[0], dest / "images")
    copy2(image_mask_pair[1], dest / "masks")

print("Finished copying images and masks to training_data folder")
