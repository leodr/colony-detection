import csv
import random
import time
from contextlib import contextmanager
from pathlib import Path
from shutil import copy2

import cv2
import numpy as np


@contextmanager
def timer(label="Execution Time"):
    """
    A context manager to measure the execution time of a block of code.
    """
    start = time.time()
    yield
    end = time.time()
    duration = end - start
    print(f"{label}: {duration:1.3f}s")


CLASSES = ["0", "1", "2", "3", "4", "5"]

# Color codes for colony classes (1 - 5)
COLORS = [
    (51, 51, 51),
    (102, 102, 102),
    (153, 153, 153),
    (204, 204, 204),
    (255, 255, 255),
]


def is_acceptable_overlap(existing_boxes, new_box, max_overlap_fraction):
    """
    Checks if a new box overlaps with existing boxes more than allowed.
    """
    for box in existing_boxes:
        # Compute the intersection area
        dx = min(box[0] + box[2], new_box[0] + new_box[2]) - max(box[0], new_box[0])
        dy = min(box[1] + box[3], new_box[1] + new_box[3]) - max(box[1], new_box[1])
        if dx >= 0 and dy >= 0:
            intersection_area = dx * dy
            smallest_area = min(box[2] * box[3], new_box[2] * new_box[3])
            # If overlap is more than allowed, return False
            if intersection_area / smallest_area > max_overlap_fraction:
                return False
    return True


def generate_random_boxes(
    image_shape, component_boxes, num_random_boxes=10, max_attempts=100
):
    """
    Generates random bounding boxes within the image boundaries. Avoids overlap with
    existing component boxes.
    """
    # Determine the size range of the component bounding boxes
    widths = [box[2] for box in component_boxes]
    heights = [box[3] for box in component_boxes]
    min_width, max_width = min(widths), max(widths)
    min_height, max_height = min(heights), max(heights)

    random_boxes = []
    attempts = 0
    while len(random_boxes) < num_random_boxes and attempts < max_attempts:
        attempts += 1
        # Choose random dimensions within the min and max sizes of components
        width = random.randint(min_width, max_width)
        height = random.randint(min_height, max_height)

        # Choose a random position within the image boundaries
        x = random.randint(0, image_shape[1] - width)
        y = random.randint(0, image_shape[0] - height)

        new_box = (x, y, width, height)

        # Check overlap with existing component boxes
        if is_acceptable_overlap(component_boxes, new_box, 0.1):
            random_boxes.append(new_box)

    return random_boxes


images_path = Path("training_data/images")
masks_path = Path("training_data/masks")
binary_masks_path = Path("training_data/binary_masks")
class_masks_path = Path("training_data/class_masks")
patches_path = Path("training_data/patches")

# Ensure the directories exist
images_path.mkdir(parents=True, exist_ok=True)
masks_path.mkdir(parents=True, exist_ok=True)
binary_masks_path.mkdir(parents=True, exist_ok=True)
class_masks_path.mkdir(parents=True, exist_ok=True)

for class_name in CLASSES:
    (patches_path / class_name).mkdir(parents=True, exist_ok=True)

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

print(f"Found {len(pairs)} pairs of images and masks.")

metadata = [["filename", "instance_count"]]

for i in range(len(pairs)):
    print("\nProcessing pair", i + 1, "of", len(pairs))
    image_path, mask_path = pairs[i]

    with timer(" - Copying image and mask"):
        copy2(image_path, images_path)
        copy2(mask_path, masks_path)

    with timer(" - Saving binary mask"):
        mask = cv2.imread(str(mask_path))
        grayscale_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(grayscale_mask, 1, 255, cv2.THRESH_BINARY)

        cv2.imwrite(str(binary_masks_path / mask_path.name), binary_mask)

    with timer(" - Calculating colony count for metadata"):
        int_mask = mask.copy().astype(np.uint32)
        int_mask = (
            (int_mask[:, :, 2] << 16) + (int_mask[:, :, 1] << 8) + int_mask[:, :, 0]
        )

        unique_colors = np.unique(int_mask[int_mask != 0])
        unique_colors_count = len(unique_colors)

        metadata.append([mask_path.name, unique_colors_count])

    print("   Found", unique_colors_count, "instances in the mask")

    with timer(" - Saving class mask and patches"):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask
        )

        output = np.zeros_like(mask)

        image = cv2.imread(str(image_path))

        for i in range(1, num_labels):
            component_mask = labels == i

            x, y, width, height = (
                stats[i, cv2.CC_STAT_LEFT],
                stats[i, cv2.CC_STAT_TOP],
                stats[i, cv2.CC_STAT_WIDTH],
                stats[i, cv2.CC_STAT_HEIGHT],
            )

            x_start = max(x - 10, 0)
            y_start = max(y - 10, 0)
            x_end = min(x + width + 10, image.shape[1])
            y_end = min(y + height + 10, image.shape[0])

            patch = image[y_start:y_end, x_start:x_end]

            new_int_mask = np.where(component_mask, int_mask, 0)

            unique_colors = np.unique(new_int_mask[new_int_mask != 0])
            unique_colors_count = len(unique_colors)

            patch_class = min(max(unique_colors_count, 1), 5)

            cv2.imwrite(
                str(patches_path / f"{patch_class}/{mask_path.stem}_{i}.png"),
                patch,
            )

            output[component_mask] = COLORS[min(unique_colors_count, len(COLORS)) - 1]

        cv2.imwrite(str(class_masks_path / mask_path.name), output)

    with timer(" - Extracting empty patches"):
        # Extract component boxes, skip the first entry (background)
        component_boxes = stats[1:]

        if len(component_boxes) > 0:
            # Generate random boxes
            random_boxes = generate_random_boxes(image.shape, component_boxes, 10)

            # Extract and save the random patches
            for i, box in enumerate(random_boxes, start=1):
                x, y, width, height = box
                patch = image[y : y + height, x : x + width]
                filename = str(patches_path / f"0/{mask_path.stem}_{i}.png")
                cv2.imwrite(filename, patch)


# Write metadata to CSV file
with open("training_data/metadata.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(metadata)

print("\nMetadata saved to metadata.csv")
