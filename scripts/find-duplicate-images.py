import glob
import hashlib
import sys


def file_hash(filepath):
    """Compute the MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def find_duplicates(pattern):
    """Find duplicate images based on the given glob pattern."""
    hashes = {}
    duplicates = []

    for filepath in glob.glob(pattern):
        if filepath.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
            filehash = file_hash(filepath)

            if filehash in hashes:
                duplicates.append((filepath, hashes[filehash]))
            else:
                hashes[filehash] = filepath

    return duplicates


# Usage
if len(sys.argv) < 2:
    print("Please provide a glob pattern as a command line argument.")
    sys.exit(1)

pattern = sys.argv[1]
duplicates = find_duplicates(pattern)

if duplicates:
    for dup in duplicates:
        print(f"Duplicate found: {dup[0]} and {dup[1]}")
else:
    print("No duplicate images found.")
