import json
import os
from typing import List

import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as F


class AgarBoundingBoxDataset(torch.utils.data.Dataset):
    num_classes = 2

    def __init__(self, root: str, ids: List[str], transforms: callable):
        self.root = root
        self.transforms = transforms

        # load all image and boxes files, sorting them to ensure that they are aligned
        self.imgs = list(sorted([f"{id}.jpg" for id in ids]))
        self.boxes = list(sorted([f"{id}.json" for id in ids]))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        boxes_path = os.path.join(self.root, self.boxes[idx])
        img = read_image(img_path)
        with open(boxes_path, "r") as f:
            boxes_json = json.load(f)

        labels = boxes_json["labels"]

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

        img = tv_tensors.Image(img)

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

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
