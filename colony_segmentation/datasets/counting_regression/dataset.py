import re
from pathlib import Path
from typing import List, Tuple

import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

random_even_rotation = v2.RandomChoice(
    [
        v2.RandomRotation((0, 0)),
        v2.RandomRotation((90, 90)),
        v2.RandomRotation((180, 180)),
        v2.RandomRotation((270, 270)),
    ]
)

transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        random_even_rotation,
        v2.ColorJitter(brightness=0.1, contrast=0.1),
        v2.ToDtype(torch.float32, scale=True),
    ]
)


class CountingRegressionDataset(Dataset):
    """Dataset that provides images and their respective counting results."""

    pairs: List[Tuple[Path, Path]]

    def __init__(self, path: Path) -> None:
        super().__init__()

        images_path = path / "images"
        masks_path = path / "masks"

        image_paths = sorted(list(images_path.glob("*.jpg")))
        mask_paths = sorted(list(masks_path.glob("*.png")))

        self.pairs = []

        for image_path in image_paths:
            mask_path = None

            for mask in mask_paths:
                if mask.stem.startswith(image_path.stem):
                    mask_path = mask
                    break

            if mask_path is not None:
                self.pairs.append((image_path, mask_path))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_transformed = transform(image)

        # Extract the count from the mask filename and convert it to an integer
        count = int(re.findall(r"\d+", mask_path.stem)[-1])

        count_tensor = torch.tensor([count], dtype=torch.float32)

        return image_transformed, count_tensor
