import torch
from ultralytics import YOLO

model = YOLO("yolov8x.pt")

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
print(f"Using {device}")

model.train(data="./agar_data.yaml", epochs=100, imgsz=1024, device=device, batch=-1)
