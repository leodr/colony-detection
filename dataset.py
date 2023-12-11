import os
from pathlib import Path

import cv2
from torch.utils.data import Dataset


class ColonySegmentationDataset(Dataset):
    def __init__(self, path: Path) -> None:
        super().__init__()

        images_path = path / "images"
        self.masks_path = path / "masks"

        self.image_paths = sorted(list(images_path.glob("*.png")))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __convert_mask(self, mask):
        """
        Converts the mask to a binary mask, white (255) is 1, black (0) is 0
        """
        mask[mask == 255] = 1
        return mask

    def __getitem__(self, index: int) -> tuple:
        image_path = self.image_paths[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(self.masks_path / image_path.name), cv2.IMREAD_GRAYSCALE)
        mask = self.__convert_mask(mask)

        return image, mask
