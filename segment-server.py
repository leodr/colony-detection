import base64
import io
import random
from typing import Tuple

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from scipy.spatial import KDTree
from segment_anything import SamPredictor, sam_model_registry

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


def create_color_map(kd_tree, shape, colors):
    # Create a grid of coordinates (y, x)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    coords = np.c_[yy.ravel(), xx.ravel()]

    # Query the KDTree for nearest coordinate index for each pixel in the grid
    _, indices = kd_tree.query(coords)

    # Map the indices to colors
    color_map = np.array([colors[idx] for idx in indices])
    return color_map.reshape(shape[0], shape[1], 3)


def preprocess_image(image_array):
    # Convert the image to grayscale
    grayscale = np.mean(image_array, axis=2)

    # Apply threshold - pixels closer to 255 become white, closer to 0 become black
    threshold = 127  # Midpoint of 0 and 255
    image_array[grayscale >= threshold] = [255, 255, 255]
    image_array[grayscale < threshold] = [0, 0, 0]
    return image_array


@app.post("/colorize-from-coordinates")
def colorize_from_coordinates(data: ColorizeData):
    try:
        # Decode the image
        base64image = data.base64_image.split(",")[1]
        image_bytes = base64.b64decode(base64image)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(image)

        # Preprocess the image to make each pixel either black or white
        image_array = preprocess_image(image_array)

        # Check if image is black and white
        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            raise ValueError("Image must be in RGB format")

        # Generate unique colors
        unique_colors = generate_unique_colors(len(data.coordinates))
        colors = dict(zip(range(len(data.coordinates)), unique_colors))

        # Adjust coordinates for the KDTree and create the KDTree
        adjusted_coordinates = [(y, x) for x, y in data.coordinates]
        kd_tree = KDTree(adjusted_coordinates)

        # Create a color map for the entire image
        color_map = create_color_map(kd_tree, image_array.shape, colors)

        # Apply the color map to non-black pixels in the image
        non_black_pixels = ~(np.all(image_array == [0, 0, 0], axis=-1))
        image_array[non_black_pixels] = color_map[non_black_pixels]

        # Convert the array back to an image and then to a base64 string
        colored_img = Image.fromarray(image_array)
        buffered = io.BytesIO()
        colored_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Save image to file
        colored_img.save("colored.png")

        return {"colorized_image": img_str}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
