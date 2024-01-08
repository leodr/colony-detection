# Checks if the files listed in the JSON file are present in the directory and
# vice versa
import csv
import os

# Define the path to the directory and the JSON file
dir_path = "data/counted"
json_file_path = os.path.join(dir_path, "metadata.csv")

# Load the JSON file
with open(json_file_path, "r") as f:
    reader = csv.DictReader(f)

    data = []
    for row in reader:
        data.append(row)


# Get the list of file names from the JSON data
json_file_names = [item["filename"] for item in data]

print("Number of files in JSON:", len(json_file_names))

# Get the list of actual file names in the directory
actual_file_names = os.listdir(dir_path)
# Only keep the file names that end with ".jpg"
actual_file_names = [file for file in actual_file_names if file.endswith(".jpg")]

print("Number of files in directory:", len(actual_file_names))

# Find and print the file names in the JSON data but not in the directory
missing_files = [file for file in json_file_names if file not in actual_file_names]
if missing_files:
    print("Files listed in JSON but not found in directory:")
    for file in missing_files:
        print(file)

# Find and print the file names in the directory but not in the JSON data
extra_files = [file for file in actual_file_names if file not in json_file_names]
if extra_files:
    print("Files found in directory but not listed in JSON:")
    for file in extra_files:
        print(file)
