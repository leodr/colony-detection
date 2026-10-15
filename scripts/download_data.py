"""Restore the data/ folder from the photobox dataset on the Hugging Face Hub.

Usage: python scripts/download_data.py [target directory, default: data]
Requires the datasets package and access to leodriesch/photobox-colonies.
"""
import os
import sys

from datasets import Image, load_dataset

# Dataset column -> folder under data/labeled
LABELED_FOLDERS = {
    "image": "images",
    "instance_mask": "masks",
    "plate_mask": "circle_masks",
    "pen_annotated": "pen_annotated",
}


def save(image, folder):
    """Write an undecoded image to folder under its original file name."""
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, image["path"]), "wb") as f:
        f.write(image["bytes"])


def download(target):
    ds = load_dataset("leodriesch/photobox-colonies", "photobox")
    for split, rows in ds.items():
        for column in LABELED_FOLDERS:
            rows = rows.cast_column(column, Image(decode=False))
        print(f"Writing {len(rows)} images from split {split}")
        for row in rows:
            # Photos with a plate mask were counted and live in data/labeled, the rest in data/unlabeled
            if row["plate_mask"] is None:
                save(row["image"], os.path.join(target, "unlabeled"))
                continue
            for column, folder in LABELED_FOLDERS.items():
                if row[column] is not None:
                    save(row[column], os.path.join(target, "labeled", folder))


if __name__ == "__main__":
    download(sys.argv[1] if len(sys.argv) > 1 else "data")
