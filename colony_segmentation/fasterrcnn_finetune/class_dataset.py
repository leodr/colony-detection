import os

import cv2
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as F

COLOR_MAP = {
    51: 1,
    102: 2,
    153: 3,
    204: 4,
    255: 5,
}

# our dataset has two classes only - background and colony
num_classes = len(COLOR_MAP) + 1


class SegmentationClassDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms

        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "class_masks"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "class_masks", self.masks[idx])
        img = read_image(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        class_labels = np.zeros(num_labels - 1, dtype=np.uint8)

        bboxes = np.zeros((num_labels - 1, 4), dtype=np.float32)
        for i, stat in enumerate(stats[1:]):
            x, y, w, h = stat[:4]
            bboxes[i] = [x, y, x + w, y + h]

        for i in range(1, num_labels):
            # extract the mask for the current connected component
            region_mask = labels == i

            color = np.median(mask[region_mask])
            class_labels[i - 1] = COLOR_MAP[color]

        bboxes_torch = torch.tensor(bboxes, dtype=torch.float32)

        img = tv_tensors.Image(img)

        target = {
            "boxes": tv_tensors.BoundingBoxes(
                bboxes_torch, format="XYXY", canvas_size=F.get_size(img)
            ),
            # there is only one class (colony)
            "labels": torch.tensor(class_labels, dtype=torch.int64),
            "image_id": idx,
            "iscrowd": torch.zeros(len(class_labels), dtype=torch.int64),
            "area": (bboxes_torch[:, 3] - bboxes_torch[:, 1])
            * (bboxes_torch[:, 2] - bboxes_torch[:, 0]),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
