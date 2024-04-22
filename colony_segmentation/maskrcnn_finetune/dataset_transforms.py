import torch
from torchvision.transforms import v2 as T


def get_transforms(is_train: bool):
    """Get the transforms for the dataset.

    Args:
        is_train (bool): Whether the transforms are for training or testing.

    Returns:
        func: A function that takes in an image and target and applies the transforms.
    """

    transforms = []

    if is_train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        # transforms.append(T.RandomVerticalFlip(0.5))
        # Rotation by 0, 90, 180, or 270 degrees
        # transforms.append(
        #     T.RandomChoice(
        #         [
        #             T.RandomRotation((0, 0)),
        #             T.RandomRotation((90, 90)),
        #             T.RandomRotation((180, 180)),
        #             T.RandomRotation((270, 270)),
        #         ]
        #     )
        # )
        # transforms.append(T.ColorJitter(brightness=0.1, contrast=0.1))

    transforms.append(T.Resize((3040, 3040)))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)
