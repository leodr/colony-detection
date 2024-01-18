import cv2
import numpy as np
from scipy.spatial import KDTree

from random_colors import get_nth_random_color


def merge_masks_with_color(existing_mask, new_mask, coordinates, new_coordinate):
    existing = cv2.imread(existing_mask)
    mask = cv2.imread(new_mask, 0)

    binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

    result_mask = existing.copy()
    result_mask[binary_mask == 255] = [255, 255, 255]

    # Convert to grayscale if it's a color image
    gray_result_mask = cv2.cvtColor(result_mask, cv2.COLOR_BGR2GRAY)

    # Create a mask used for flood filling
    # Note: the size needs to be 2 pixels more than the image
    h, w = gray_result_mask.shape[:2]
    empty_mask = np.zeros((h + 2, w + 2), np.uint8)

    # Flood fill parameters
    floodfill_flags = 8  # 8-connected
    floodfill_flags |= cv2.FLOODFILL_MASK_ONLY
    floodfill_flags |= 255 << 8

    # Perform flood fill
    num, im, mask, rect = cv2.floodFill(
        gray_result_mask, empty_mask, new_coordinate, 255, flags=floodfill_flags
    )

    # Convert mask to binary (0 or 255)

    empty_mask = empty_mask[1 : h + 1, 1 : w + 1]  # Remove the extra border
    binary_mask = np.where(empty_mask == 1, 255, 0).astype(np.uint8)

    # List to hold coordinates inside the mask

    inside_mask_coords = []

    # Iterate through each coordinate
    for coord in coordinates:
        x, y = coord
        # Check if the coordinate is inside the mask
        if binary_mask[y, x] == 255:
            inside_mask_coords.append(coord)

    tree = KDTree(inside_mask_coords)

    # Iterate through all pixels in the mask
    for y in range(binary_mask.shape[0]):
        for x in range(binary_mask.shape[1]):
            if binary_mask[y, x] == 1:
                _, index = tree.query((x, y))
                color = get_nth_random_color(index + 1)
                result_mask[y, x] = color

    # Add new_center to the list of coordinates
    coordinates.append(new_coordinate)

    return result_mask, coordinates
