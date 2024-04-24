import json
import os

import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as F


class BoundingBoxDataset(torch.utils.data.Dataset):
    num_classes = 2  # background and colony

    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms

        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.boxes = list(sorted(os.listdir(os.path.join(root, "boxes"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        boxes_path = os.path.join(self.root, "boxes", self.boxes[idx])
        img = read_image(img_path)
        with open(boxes_path, "r") as f:
            boxes_json = json.load(f)

        boxes = np.array(
            [
                (box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"])
                for box in boxes_json["boxes"]
            ],
            dtype=np.float32,
        )

        num_boxes = len(boxes)

        img = tv_tensors.Image(img)

        target = {
            "boxes": tv_tensors.BoundingBoxes(
                boxes, format="XYXY", canvas_size=F.get_size(img)
            ),
            # there is only one class (colony)
            "labels": torch.ones(num_boxes, dtype=torch.int64),
            "image_id": idx,
            "iscrowd": torch.zeros(num_boxes, dtype=torch.int64),
            "area": (
                (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                if num_boxes > 0
                else torch.tensor([])
            ),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
