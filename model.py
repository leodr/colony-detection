from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ColonySegmentationDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 50
BATCH_SIZE = 4

train_dataset = ColonySegmentationDataset(Path("training-data/train"))
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = ColonySegmentationDataset(Path("training-data/val"))
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
