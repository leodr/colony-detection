import os
from pathlib import Path
from shutil import copy2

Path(f"training_data/images").mkdir(parents=True, exist_ok=True)
Path(f"training_data/masks").mkdir(parents=True, exist_ok=True)

image_paths = list(sorted(Path("data/labeled/images").glob("*.jpg")))
mask_paths = list(sorted(Path("data/labeled/masks").glob("*.png")))

pairs = []

for image_path in image_paths:
    mask_path = None

    for mask in mask_paths:
        if mask.stem.startswith(image_path.stem):
            mask_path = mask
            break

    if mask_path is not None:
        pairs.append((image_path, mask_path))

for pair in pairs:
    copy2(pair[0], Path("training_data/images"))
    copy2(pair[1], Path("training_data/masks"))

print("Finished copying images and masks to training_data folder")
