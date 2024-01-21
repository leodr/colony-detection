import io
import logging
import shutil
from typing import List, Tuple

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from merge_masks import merge_masks
from PIL import Image
from pydantic import BaseModel
from segment_anything import SamPredictor, sam_model_registry

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")

if torch.backends.mps.is_available():
    sam.to(device="mps")
elif torch.cuda.is_available():
    sam.to(device="cuda")

predictor = SamPredictor(sam)


app = FastAPI()


class State:
    """Stores the state of the app for a certain image."""

    image: np.ndarray
    masks_stack: List[np.ndarray]
    clicks: List[Tuple[int, int]]

    def __init__(self, image: np.ndarray):
        self.image = image
        self.masks_stack = [np.zeros_like(image)]
        self.clicks = []

    @property
    def current_mask(self):
        return self.masks_stack[-1]

    def undo(self):
        if len(self.masks_stack) > 1:
            self.masks_stack.pop()
        else:
            self.masks_stack = [np.zeros_like(self.image)]

        if len(self.clicks) > 0:
            self.clicks.pop()

    def clear_mask(self):
        self.masks_stack = [np.zeros_like(self.image)]
        self.clicks = []


state: State = None


def image_array_to_response(image_array: np.ndarray):
    image = Image.fromarray(image_array)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")

    img_byte_array = buffered.getvalue()

    return StreamingResponse(io.BytesIO(img_byte_array), media_type="image/png")


@app.put("/set-image")
def set_image(file: UploadFile = File(...)):
    """Sets the image to segment."""
    global state

    with io.BytesIO() as buffer:
        shutil.copyfileobj(file.file, buffer)
        buffer.seek(0)

        image_array = np.array(Image.open(buffer))

        state = State(image=image_array)

        predictor.set_image(image_array)

        return image_array_to_response(state.current_mask)


class ClickData(BaseModel):
    x: int
    y: int


@app.post("/register-click")
def register_click(click_data: ClickData):
    """Returns the new mask image after registering a click."""
    global state

    state.clicks.append((click_data.x, click_data.y))

    masks, _, _ = predictor.predict(
        point_coords=np.array([[click_data.x, click_data.y]]),
        point_labels=np.array([1]),
    )

    new_mask = merge_masks(
        existing_mask=state.current_mask,
        new_mask=masks[0],
        center_points=state.clicks,
    )

    state.masks_stack.append(new_mask)

    return image_array_to_response(state.current_mask)


@app.get("/click-count")
def click_count():
    """Returns the number of clicks registered."""
    global state

    return len(state.clicks)


@app.post("/undo")
def undo():
    """Returns the new mask image after undoing the last click."""
    global state

    state.undo()

    return image_array_to_response(state.current_mask)


@app.post("/clear-mask")
def clear_mask():
    """Returns the new mask image after clearing the mask."""
    global state

    state.clear_mask()

    return image_array_to_response(state.current_mask)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    uvicorn.run(app, port=8000, log_level="info")
