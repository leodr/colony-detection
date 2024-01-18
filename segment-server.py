import base64
import io
import random
from typing import Tuple

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from scipy.ndimage import find_objects, label
from scipy.spatial import KDTree, cKDTree
from segment_anything import SamPredictor, sam_model_registry
from skimage.measure import label

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
if torch.backends.mps.is_available():
    sam.to(device="mps")
elif torch.cuda.is_available():
    sam.to(device="cuda")

predictor = SamPredictor(sam)


app = FastAPI()


class ImageData(BaseModel):
    base64_image: str


@app.put("/set-image")
def set_image(image_data: ImageData):
    try:
        # Strip the base64 header
        base64image = image_data.base64_image.split(",")[1]

        # Decode the base64 string
        image_bytes = base64.b64decode(base64image)

        # Open the image using PIL and convert to ndarray
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)

        predictor.set_image(image_array)

        return {"message": "Image received and processed", "shape": image_array.shape}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/predict-from-coords")
def predict_from_coords(x: int, y: int):
    masks, _, _ = predictor.predict(
        point_coords=np.array([[x, y]]), point_labels=np.array([1])
    )

    # Convert mask to image and then to base64 string
    buffered = io.BytesIO()
    img = Image.fromarray(masks[0])
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return {"mask_image": img_str}


def generate_unique_colors(n):
    colors = set()
    while len(colors) < n:
        color = (
            random.randint(64, 255),
            random.randint(64, 255),
            random.randint(64, 255),
        )  # Avoid dark colors
        colors.add(color)
    return list(colors)


class ColorizeData(BaseModel):
    base64_image: str
    coordinates: list[Tuple[int, int]]  # List of (x, y) tuples


def generate_unique_colors(n):
    colors = set()
    while len(colors) < n:
        color = (
            random.randint(64, 255),
            random.randint(64, 255),
            random.randint(64, 255),
        )  # Avoid dark colors
        colors.add(color)
    return list(colors)


class ColorizeData(BaseModel):
    base64_image: str
    coordinates: list[Tuple[int, int]]  # List of (x, y) tuples


def preprocess_image(image_array):
    # Convert the image to grayscale and apply threshold
    grayscale = np.mean(image_array, axis=2)
    threshold = 127
    binary_image = np.where(grayscale >= threshold, 1, 0)
    return binary_image


@app.post("/colorize-from-coordinates")
def colorize_from_coordinates(data: ColorizeData):
    try:
        base64image = data.base64_image.split(",")[1]
        image_bytes = base64.b64decode(base64image)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(image)

        # Preprocess the image to a binary mask
        binary_mask = preprocess_image(image_array)

        # Segment the image into connected components
        labeled_array, num_features = label(binary_mask)

        # Assign random colors to each coordinate
        colors = generate_unique_colors(num_features)
        color_mapping = {i + 1: colors[i] for i in range(num_features)}

        # Assign coordinates to regions and create color mapping
        coord_regions = [labeled_array[y, x] for x, y in data.coordinates]
        coord_colors = {
            coord: color_mapping[region]
            for coord, region in zip(data.coordinates, coord_regions)
            if region > 0
        }

        # Create KDTree for efficient nearest neighbor search
        tree = cKDTree(data.coordinates)

        # Colorize the image
        for y in range(labeled_array.shape[0]):
            for x in range(labeled_array.shape[1]):
                region = labeled_array[y, x]
                if region > 0:
                    distance, idx = tree.query((x, y))  # Corrected this line
                    nearest_coord = data.coordinates[idx]
                    if labeled_array[nearest_coord[1], nearest_coord[0]] == region:
                        image_array[y, x] = coord_colors[nearest_coord]

        colored_img = Image.fromarray(image_array)
        buffered = io.BytesIO()
        colored_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        colored_img.save("colored.png")
        return {"colorized_image": img_str}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
