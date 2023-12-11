import csv
import json
import time

import numpy as np
from circle_detection import CircleDetection, CircleDetectionHyperparameters


def generate_configurations():
    """Generate different configurations of hyperparameters for grid search."""
    configurations = []

    for blur_kernel_size in [9, 15, 21]:
        for blur_sigmax in [0]:
            for outer_circle_param1 in [150]:
                for outer_circle_param2 in [10]:
                    for colony_minDist in [10]:
                        for colony_param1 in [30, 50, 70]:
                            for colony_param2 in [15, 30, 65]:
                                for colony_minRadius in [3, 6]:
                                    for colony_maxRadius in [50]:
                                        configurations.append(
                                            CircleDetectionHyperparameters(
                                                blur_kernel_size,
                                                blur_sigmax,
                                                outer_circle_param1,
                                                outer_circle_param2,
                                                colony_minDist,
                                                colony_param1,
                                                colony_param2,
                                                colony_minRadius,
                                                colony_maxRadius,
                                            )
                                        )


configurations = generate_configurations()

# Read the JSON file
with open("../data/counted/metadata.json", "r") as f:
    data = json.load(f)


def process_with_hyperparams(hyperparameters):
    results = []
    algorithm = CircleDetection(hyperparameters)

    # Iterate through each element in the JSON array
    for i in range(len(data)):
        print("Processing image " + str(i + 1) + " of " + str(len(data)))
        entry = data[i]
        file_name = entry.get("fileName")
        actual_count = entry.get("countedColonies")

        if file_name is not None and actual_count is not None:
            image = cv2.imread(f"../data/counted/{file_name}")
            estimated_count = algorithm.count_colonies(image)

            results.append((file_name, estimated_count, actual_count))

    return results


def rel_diff(x, y):
    if y == 0 and x == 0:
        return 0
    elif y == 0:
        return 1
    else:
        return abs(x - y) / y


# Open a CSV file for writing
with open("timing_results.csv", "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)

    # Write the header row
    csvwriter.writerow(
        [
            "blur_kernel_size",
            "blur_sigmax",
            "outer_circle_param1",
            "outer_circle_param2",
            "colony_minDist",
            "colony_param1",
            "colony_param2",
            "colony_minRadius",
            "colony_maxRadius",
            "mean_abs_diff",
            "median_abs_diff",
            "5th_percentile_abs_diff",
            "95th_percentile_abs_diff",
            "mean_rel_diff",
            "median_rel_diff",
            "5th_percentile_rel_diff",
            "95th_percentile_rel_diff",
        ]
    )

    for i in range(len(configurations)):
        configuration = configurations[i]

        start_time = time.time()

        result = process_with_hyperparams(configuration)

        end_time = time.time()

        elapsed_time_ms = round((end_time - start_time) * 1000)

        absolute_differences = [x[1] - x[2] for x in result]
        absolute_differences.sort()

        mean_abs_diff = np.mean(absolute_differences)
        median_abs_diff = np.median(absolute_differences)
        percentile_5th_abs_diff = np.percentile(absolute_differences, 5)
        percentile_95th_abs_diff = np.percentile(absolute_differences, 95)

        relative_differences = [rel_diff(x[1], x[2]) for x in result]
        relative_differences.sort()

        mean_rel_diff = np.mean(relative_differences)
        median_rel_diff = np.median(relative_differences)
        percentile_5th_rel_diff = np.percentile(relative_differences, 5)
        percentile_95th_rel_diff = np.percentile(relative_differences, 95)

        csvwriter.writerow(
            [
                configuration.blur_kernel_size,
                configuration.blur_sigmax,
                configuration.outer_circle_param1,
                configuration.outer_circle_param2,
                configuration.colony_minDist,
                configuration.colony_param1,
                configuration.colony_param2,
                configuration.colony_minRadius,
                configuration.colony_maxRadius,
                mean_abs_diff,
                median_abs_diff,
                percentile_5th_abs_diff,
                percentile_95th_abs_diff,
                mean_rel_diff,
                median_rel_diff,
                percentile_5th_rel_diff,
                percentile_95th_rel_diff,
            ]
        )

        update_progress((i + 1) / len(configurations))
