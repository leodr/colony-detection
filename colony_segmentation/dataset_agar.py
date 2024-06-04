# %%
import json
import os
from typing import List

import numpy as np
import torch
from dataset_transforms import get_transforms
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as F


class AgarBBDataset(torch.utils.data.Dataset):
    num_classes = 2  # background and colony

    def __init__(self, root: str, ids: List[str], is_train: bool):
        self.root = root
        self.transforms = get_transforms(is_train, rotate=False)

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

        img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


# %% Preview of dataset
if __name__ == "__main__":
    import json

    import cv2
    import matplotlib.pyplot as plt
    import torchvision.utils as utils
    from torchvision import transforms

    with open("../AGAR_dataset/training_lists/higher_resolution_train.txt", "r") as f:
        ids = json.load(f)

    dataset = AgarBBDataset(
        root="../AGAR_dataset/dataset",
        ids=ids,
        is_train=True,
    )

    random_ids = np.random.choice(len(dataset), 4)

    for idx in random_ids:
        img, target = dataset[idx]

        img = img * 255
        img = img.to(torch.uint8)

        boxes_drawn = utils.draw_bounding_boxes(
            img, target["boxes"], width=3, colors="red"
        )
        boxes_drawn = F.to_pil_image(boxes_drawn)

        plt.figure(figsize=(10, 10))
        plt.imshow(boxes_drawn)
        plt.axis("off")
        plt.show()

# %%
