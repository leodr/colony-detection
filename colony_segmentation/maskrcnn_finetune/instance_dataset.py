import os
from typing import List

import torch
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision.transforms.v2 import functional as F


def colormask_to_binary_masks(mask: torch.Tensor) -> List[torch.Tensor]:
    """Converts a color-encoded mask to a set of binary masks.

    Args:
        mask (torch.Tensor): A color-encoded mask tensor of shape [C, H, W].

    Returns:
        List[torch.Tensor]: A list of binary masks, each of shape [H, W].
    """

    # turn mask from [C, H, W] to [H, W] by combining the color channels into a single integer
    int_mask = (mask[0].int() << 16) + (mask[1].int() << 8) + mask[2].int()

    instance_ids = torch.unique(int_mask)
    # first id is the background (black), so remove it
    instance_ids = instance_ids[1:]

    # split the color-encoded mask into a set of binary masks
    masks = (int_mask == instance_ids[:, None, None]).to(dtype=torch.uint8)

    return masks


class SegmentationInstanceDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms

        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = read_image(img_path)
        mask = read_image(mask_path)

        masks = colormask_to_binary_masks(mask)
        boxes = masks_to_boxes(masks)

        num_masks = len(masks)

        img = tv_tensors.Image(img)

        target = {
            "boxes": tv_tensors.BoundingBoxes(
                boxes, format="XYXY", canvas_size=F.get_size(img)
            ),
            "masks": tv_tensors.Mask(masks),
            # there is only one class (colony)
            "labels": torch.ones(num_masks, dtype=torch.int64),
            "image_id": idx,
            "iscrowd": torch.zeros(num_masks, dtype=torch.int64),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
