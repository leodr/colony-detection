from typing import List, Tuple

import cv2
import numpy as np
from random_colors import get_n_random_colors
from scipy.spatial import KDTree

FLOODFILL_FLAGS = 4  # 4-connected (only direct neightbours, no diagonal)
FLOODFILL_FLAGS |= cv2.FLOODFILL_MASK_ONLY
FLOODFILL_FLAGS |= 255 << 8


def merge_masks(
    existing_mask: np.ndarray,
    new_mask: np.ndarray,
    center_points: List[List[Tuple[int, int]]],
) -> np.ndarray:
    """
    Merges an existing mask with a new mask, using the center point of the new mask to determine the color of each pixel.

    Args:
        existing_mask (np.ndarray): The existing (colored) mask
        new_mask (np.ndarray): The new mask (binary, True and False)
        center_points (List[List[Tuple[int, int]]]): The list of center points for each region

    Returns:
        np.ndarray: The resulting colored mask
    """

    binary_mask = cv2.cvtColor(existing_mask, cv2.COLOR_RGB2GRAY)
    _, binary_mask = cv2.threshold(binary_mask, 1, 255, cv2.THRESH_BINARY)
    binary_mask[new_mask] = 255

    # Create a mask used for flood filling
    # Note: the size needs to be 2 pixels more than the image
    height, width = binary_mask.shape[:2]
    floodfill_mask = np.zeros((height + 2, width + 2), np.uint8)

    new_center = center_points[-1][-1]

    cv2.floodFill(binary_mask, floodfill_mask, new_center, 255, flags=FLOODFILL_FLAGS)

    # Remove the extra border
    floodfill_mask = floodfill_mask[1 : height + 1, 1 : width + 1]

    colors = get_n_random_colors(len(center_points))

    point_count = sum(len(points) for points in center_points)

    center_points_with_colors = np.zeros((point_count, 5), dtype=np.uint32)

    for i, center_point in enumerate(center_points):
        for j, point in enumerate(center_point):
            center_points_with_colors[i + j] = [*point, *colors[i]]

    x_coords = center_points_with_colors[:, 0]
    y_coords = center_points_with_colors[:, 1]

    inside_mask = floodfill_mask[y_coords, x_coords] == 255

    center_points_with_colors_within_fill = center_points_with_colors[inside_mask]

    tree = KDTree([point[:2] for point in center_points_with_colors_within_fill])

    floodfill_indices = np.argwhere(floodfill_mask == 255)

    result_mask = existing_mask.copy()

    floodfill_indices_columns_swapped = floodfill_indices[:, [1, 0]]

    _, indices = tree.query(floodfill_indices_columns_swapped)

    x_coords, y_coords = np.hsplit(floodfill_indices_columns_swapped, 2)
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()

    # Update result_mask using advanced indexing
    result_mask[y_coords, x_coords] = center_points_with_colors_within_fill[indices, 2:]

    return result_mask
