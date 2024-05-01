import csv
import json
import random
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from shutil import copy2

import cv2
import numpy as np


@contextmanager
def timer(label="Execution Time"):
    start = time.time()
    yield
    duration = time.time() - start
    print(f"{label}: {duration:1.3f}s")


CLASSES = ["0", "1", "2", "3", "4", "5"]

# Grayscale colors for colony classes (1 - 5)
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


def setup_directories(base_path: Path):
    images_path = base_path / "images"
    masks_path = base_path / "masks"
    binary_masks_path = base_path / "binary_masks"
    class_masks_path = base_path / "class_masks"
    patches_path = base_path / "patches"
    boxes_path = base_path / "boxes"

    images_path.mkdir(parents=True, exist_ok=True)
    masks_path.mkdir(parents=True, exist_ok=True)
    binary_masks_path.mkdir(parents=True, exist_ok=True)
    class_masks_path.mkdir(parents=True, exist_ok=True)
    boxes_path.mkdir(parents=True, exist_ok=True)

    for class_name in CLASSES:
        (patches_path / class_name).mkdir(parents=True, exist_ok=True)

    return (
        images_path,
        masks_path,
        binary_masks_path,
        class_masks_path,
        patches_path,
        boxes_path,
    )


def process_item(
    image_path,
    mask_path,
    images_path,
    masks_path,
    binary_masks_path,
    class_masks_path,
    patches_path,
    boxes_path,
):
    metadata = None
    box_data = []

    with timer(" - Copying image and mask"):
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path))

        cv2.imwrite(str(images_path / image_path.name), image)
        cv2.imwrite(str(masks_path / mask_path.name), mask)

    with timer(" - Saving binary mask"):
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

        metadata = [mask_path.name, unique_colors_count]

    print("   Found", unique_colors_count, "instances in the mask")

    with timer(" - Saving bounding boxes"):
        bounding_boxes = []

        for i, color in enumerate(unique_colors):
            rows, cols = np.where(int_mask == color)

            min_row = np.min(rows)
            max_row = np.max(rows)
            min_col = np.min(cols)
            max_col = np.max(cols)

            x = float(min_col)
            y = float(min_row)
            width = float(max_col - min_col + 1)
            height = float(max_row - min_row + 1)

            bounding_boxes.append({"x": x, "y": y, "width": width, "height": height})
            box_data.append([mask_path.name, i, width * height])

        assert len(bounding_boxes) == unique_colors_count, "Mismatch in box count"

        with open(boxes_path / f"{mask_path.stem}.json", "w") as f:
            json.dump({"boxes": bounding_boxes}, f)

    with timer(" - Saving class mask and patches"):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)

        output = np.zeros_like(mask)

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

    return metadata, box_data


def main():
    (
        train_images_path,
        train_masks_path,
        train_binary_masks_path,
        train_class_masks_path,
        train_patches_path,
        train_boxes_path,
    ) = setup_directories(Path("training_data"))
    (
        test_images_path,
        test_masks_path,
        test_binary_masks_path,
        test_class_masks_path,
        test_patches_path,
        test_boxes_path,
    ) = setup_directories(Path("test_data"))

    mask_paths = list(sorted(Path("data/labeled/masks").glob("*.png")))

    pairs = [
        (Path(f"data/labeled/images/{mask_path.stem}.jpg"), mask_path)
        for mask_path in mask_paths
    ]

    random.seed(42)
    random.shuffle(pairs)
    num_test = int(len(pairs) * 0.2)
    test_pairs = pairs[:num_test]
    train_pairs = pairs[num_test:]

    print(f"Found {len(pairs)} pairs of images and masks.")
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Test pairs: {len(test_pairs)}")

    # Training data processing
    training_metadata = [["filename", "instance_count"]]
    box_data = [["filename", "id", "area"]]

    for i, (image_path, mask_path) in enumerate(train_pairs):
        print("\nProcessing training pair", i + 1, "of", len(train_pairs))

        metadata_chunk, box_data_chunk = process_item(
            image_path,
            mask_path,
            train_images_path,
            train_masks_path,
            train_binary_masks_path,
            train_class_masks_path,
            train_patches_path,
            train_boxes_path,
        )
        training_metadata.append(metadata_chunk)
        box_data.extend(box_data_chunk)

    with open("training_data/metadata.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(training_metadata)

    with open("training_data/box_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(box_data)

    print("\nMetadata saved to metadata.csv and box_data.csv")

    # Test data processing
    test_metadata = [["filename", "instance_count"]]
    for i, (image_path, mask_path) in enumerate(test_pairs):
        print("\nProcessing test pair", i + 1, "of", len(test_pairs))

        metadata_chunk = process_item(
            image_path,
            mask_path,
            test_images_path,
            test_masks_path,
            test_binary_masks_path,
            test_class_masks_path,
            test_patches_path,
            test_boxes_path,
        )
        test_metadata.append(metadata_chunk)

    with open("test_data/metadata.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(test_metadata)

    with open("test_data/box_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(box_data)

    print("\nMetadata saved to metadata.csv and box_data.csv")


if __name__ == "__main__":
    main()
