import csv
import json
import sys
import time

import cv2
import numpy as np
from circle_detection import CircleDetection, CircleDetectionHyperparameters

# Read the JSON file
with open("../data/counted/metadata.json", "r") as f:
    data = json.load(f)


def create_masks():
    algorithm = CircleDetection()

    # Iterate through each element in the JSON array
    for i in range(len(data)):
        print("Processing image " + str(i + 1) + " of " + str(len(data)))
        entry = data[i]
        file_name = entry.get("fileName")
        actual_count = entry.get("countedColonies")

        if file_name is not None and actual_count is not None:
            image = cv2.imread(f"../data/counted/{file_name}")
            mask = algorithm.create_mask(image)

            mask_filename = file_name.replace(".jpg", ".png")
            cv2.imwrite(mask_filename, mask)


create_masks()


def rel_diff(x, y):
    if y == 0 and x == 0:
        return 0
    elif y == 0:
        return 1
    else:
        return abs(x - y) / y


# start_time = time.time()

# result = process_with_hyperparams(CircleDetectionHyperparameters())

# end_time = time.time()

# elapsed_time_ms = round((end_time - start_time) * 1000)

# absolute_differences = [x[1] - x[2] for x in result]
# absolute_differences.sort()

# mean_abs_diff = np.mean(absolute_differences)
# median_abs_diff = np.median(absolute_differences)
# percentile_5th_abs_diff = np.percentile(absolute_differences, 5)
# percentile_95th_abs_diff = np.percentile(absolute_differences, 95)

# relative_differences = [rel_diff(x[1], x[2]) for x in result]
# relative_differences.sort()

# mean_rel_diff = np.mean(relative_differences)
# median_rel_diff = np.median(relative_differences)
# percentile_5th_rel_diff = np.percentile(relative_differences, 5)
# percentile_95th_rel_diff = np.percentile(relative_differences, 95)

# print(f"Elapsed time: {elapsed_time_ms} ms")
# print(f"Mean absolute difference: {mean_abs_diff}")
# print(f"Median absolute difference: {median_abs_diff}")

# print(f"Mean relative difference: {mean_rel_diff}")
# print(f"Median relative difference: {median_rel_diff}")
