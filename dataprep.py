import csv
from pathlib import Path
from shutil import copy2

import cv2
import numpy as np
import torch
from torchvision.io import read_image

images_path = Path("training_data/images")
masks_path = Path("training_data/masks")
binary_masks_path = Path("training_data/binary_masks")

# Ensure the directories exist
images_path.mkdir(parents=True, exist_ok=True)
masks_path.mkdir(parents=True, exist_ok=True)
binary_masks_path.mkdir(parents=True, exist_ok=True)

# Paths to the source data
image_paths = list(sorted(Path("data/labeled/images").glob("*.jpg")))
mask_paths = list(sorted(Path("data/labeled/masks").glob("*.png")))

pairs = []

# Match each image with its corresponding mask
for image_path in image_paths:
    mask_path = None
    for mask in mask_paths:
        if mask.stem.startswith(image_path.stem):
            mask_path = mask
            break

    if mask_path is not None:
        pairs.append((image_path, mask_path))

metadata = [["filename", "instance_count"]]

for image_path, mask_path in pairs:
    # Copy original images and masks
    copy2(image_path, images_path)
    copy2(mask_path, masks_path)

    # Read mask for thresholding
    mask = cv2.imread(str(mask_path))
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, thresholded_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)

    # Save the thresholded mask
    thresholded_mask_path = binary_masks_path / mask_path.name
    cv2.imwrite(str(thresholded_mask_path), thresholded_mask)

    mask_tensor = read_image(str(mask_path))

    int_mask = (
        (mask_tensor[0].int() << 16)
        + (mask_tensor[1].int() << 8)
        + mask_tensor[2].int()
    )

    colors = torch.unique(int_mask)

    # Remove 0 (black) from the instance IDs
    colors = colors[colors != 0]
    unique_color_count = len(colors)

    metadata.append([mask_path.name, unique_color_count])

# Write metadata to CSV file
with open("training_data/metadata.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(metadata)

print(
    "Finished copying and processing images and masks to training_data folder, metadata saved to CSV."
)
