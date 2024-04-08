import os
import re

# Set the directory you want to rename files in
directory = "data/labeled/masks"

# Regular expression to match files ending with "_mask_<number>.png"
pattern = r"(.+)_mask_\d+(\.png)$"

# Iterate through all the files in the directory
for filename in os.listdir(directory):
    # Check if the filename matches the pattern
    match = re.match(pattern, filename)
    if match:
        # Construct the new filename
        new_filename = match.group(1) + match.group(2)
        # Full path for current and new file
        current_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        # Rename the file
        os.rename(current_file, new_file)
        print(f"Renamed '{filename}' to '{new_filename}'")

print("Renaming process completed.")
