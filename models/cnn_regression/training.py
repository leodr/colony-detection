# %%
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from dataset import CNNRegressionDataset
from model import CountingCNN
from torch.utils.data import DataLoader

device = "cpu"

# Check for accelerator availability
if torch.backends.mps.is_available():
    device = "mps"
    print("Using MPS for training.")
elif torch.cuda.is_available():
    device = "cuda"
    print("Using CUDA for training.")
else:
    print("Using CPU for training.")

# %%
train_dataset = CNNRegressionDataset(path=Path("training_data/train"))
val_dataset = CNNRegressionDataset(path=Path("training_data/val"))
test_dataset = CNNRegressionDataset(path=Path("training_data/test"))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

model = CountingCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %% Training loop
NUM_EPOCHS = 10

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, counts in train_loader:
        inputs, counts = inputs.to(device), counts.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        print(outputs.shape, counts.shape)
        loss = criterion(outputs, counts)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch}, Training Loss: {running_loss / len(train_loader)}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, counts in val_loader:
            inputs, counts = inputs.to(device), counts.to(device)

            outputs = model(inputs)
            print(outputs.shape, counts.shape)
            loss = criterion(outputs, counts)
            val_loss += loss.item()
    print(f"Epoch {epoch}, Validation Loss: {val_loss / len(val_loader)}")


# %% Test loop
model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, counts in test_loader:
        inputs, counts = inputs.to(device), counts.to(device)

        outputs = model(inputs)
        for output in outputs:
            print(f"Output: {output.item()}, Expected: {counts.item()}")
        loss = criterion(outputs, counts)
        test_loss += loss.item()
print(f"Test Loss: {test_loss / len(test_loader)}")
