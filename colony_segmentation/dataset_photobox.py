import json
import os
import re

import cv2
import numpy as np
import torch
from dataset_transforms import get_transforms
from PIL import Image
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as F


def get_filename_components(filename: str) -> tuple[str, int, int, int]:
    """Extracts the components of a filename in the format id_x_y_r.ext"""

    match = re.search(r"(.+?)_(\d+)_(\d+)_(\d+)\.\w+", filename)

    if match:
        id = match.group(1)
        x, y, r = match.group(2), match.group(3), match.group(4)

        return id, int(x), int(y), int(r)
    else:
        raise ValueError(f"Filename {filename} does not match the expected pattern")


class PhotoboxBBDataset(torch.utils.data.Dataset):
    num_classes = 2  # background and colony

    def __init__(
        self,
        root: str,
        is_train: bool,
        rotate_augmentation: bool = True,
        patch_augmentation: bool = False,
    ):
        self.root = root
        self.transforms = get_transforms(is_train, rotate=rotate_augmentation)

        self.patch_augmentation = patch_augmentation

        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.boxes = list(sorted(os.listdir(os.path.join(root, "boxes"))))

        if self.patch_augmentation:
            self.patch_imgs = list(
                sorted(
                    os.listdir("../inpaint_data_augmentation/patched_images/correct")
                )
            )
            self.patch_imgs_args = [
                get_filename_components(filename) for filename in self.patch_imgs
            ]

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        boxes_path = os.path.join(self.root, "boxes", self.boxes[idx])

        img = cv2.imread(img_path)
        with open(boxes_path, "r") as f:
            boxes_json = json.load(f)

        labels = boxes_json["boxes"]

        if self.patch_augmentation:
            labels = [
                {
                    "x": int(box["x"]),
                    "y": int(box["y"]),
                    "width": int(box["width"]),
                    "height": int(box["height"]),
                    "center_x": int(box["x"]) + round(box["width"] / 2),
                    "center_y": int(box["y"]) + round(box["height"] / 2),
                }
                for box in labels
            ]

            id = self.imgs[idx].split(".")[0]

            patches = [patch for patch in self.patch_imgs_args if patch[0] == id]
            used_patches = [
                patch for patch in patches if np.random.choice([True, False])
            ]

            replacement_mask = np.zeros(img.shape[:2], dtype=np.uint8)

            for used_patch in used_patches:
                id, x, y, radius = used_patch

                patch_path = os.path.join(
                    "../inpaint_data_augmentation/patched_images/correct",
                    f"{id}_{x}_{y}_{radius}.jpg",
                )

                patch_img = cv2.imread(patch_path)

                mask = np.zeros(patch_img.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), radius, (255, 255, 255), -1)
                cv2.circle(replacement_mask, (x, y), radius, 255, -1)

                np.copyto(img, patch_img, where=(mask == (255, 255, 255)))

            # Remove all labels that are in the replacement mask
            labels = [
                box
                for box in labels
                if (replacement_mask[box["center_x"], box["center_y"]] == 0)
            ]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(img)

        img = tv_tensors.Image(pil_image)

        boxes = np.zeros((len(labels), 4), dtype=np.float32)
        area = np.zeros((len(labels)), dtype=np.float32)

        for i, box in enumerate(labels):
            boxes[i] = [
                box["x"],
                box["y"],
                box["x"] + box["width"],
                box["y"] + box["height"],
            ]
            area[i] = box["width"] * box["height"]

        num_boxes = len(boxes)

        target = {
            "boxes": tv_tensors.BoundingBoxes(
                boxes, format="XYXY", canvas_size=F.get_size(img)
            ),
            # there is only one class (colony)
            "labels": torch.ones(num_boxes, dtype=torch.int64),
            "image_id": idx,
            "iscrowd": torch.zeros(num_boxes, dtype=torch.int64),
            "area": area,
        }

        img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
