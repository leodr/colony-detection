import io
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from limited_map import LimitedMap
from merge_masks import merge_masks
from PIL import Image
from pydantic import BaseModel
from segment_anything import SamPredictor, sam_model_registry
from uvicorn.config import LOGGING_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s:    %(message)s",
)
logger = logging.getLogger(__name__)

logging.info(f"Loading SAM model.")
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
logging.info(f"Loaded SAM model.")

if torch.backends.mps.is_available():
    logging.info(f"Using MPS for SAM.")
    sam.to(device="mps")
    logging.info(f"Moved SAM to MPS device.")
elif torch.cuda.is_available():
    logging.info(f"Using CUDA for SAM.")
    sam.to(device="cuda")
    logging.info(f"Moved SAM to CUDA device.")
else:
    logging.info(f"Using CPU for SAM.")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists("./interface/dist"):
    app.mount("/static", StaticFiles(directory="./interface/dist"), name="static")

    # Custom route to serve `index.html` at root ("/") and "/index.html"
    @app.get("/")
    @app.get("/index.html")
    async def root():
        return FileResponse("./interface/dist/index.html")

    logging.info(f"Serving static files from './interface/dist'.")
else:
    logging.info(f"Could not find './interface/dist'. Not serving static files.")


class ImageState:
    """Stores the state for a certain image."""

    image_shape: Tuple[int, int]
    predictor: SamPredictor
    masks_stack: List[np.ndarray]
    clicks: List[List[Tuple[int, int]]]

    def __init__(self, image: np.ndarray):
        self.image_shape = image.shape

        self.predictor = SamPredictor(sam)
        self.predictor.set_image(image)

        self.masks_stack = [np.zeros(self.image_shape, dtype=np.uint8)]
        logging.info(f"Initial mask shape: {self.masks_stack[-1].shape}.")
        self.clicks = []

    @property
    def current_mask(self):
        return self.masks_stack[-1]

    def undo(self):
        if len(self.masks_stack) > 1:
            self.masks_stack.pop()
        else:
            self.masks_stack = [np.zeros(self.image_shape, dtype=np.uint8)]

        if len(self.clicks) > 0:
            self.clicks[-1].pop()

            if len(self.clicks[-1]) == 0:
                self.clicks.pop()

    def clear_mask(self):
        self.masks_stack = [np.zeros(self.image_shape, dtype=np.uint8)]
        self.clicks = []


# Global state, holds the current image state for up to 20 images.
state_map = LimitedMap(20)


def get_image_state(image_id: str) -> ImageState:
    """Returns the image state for the given image ID.

    Args:
        image_id (str): The ID of the image to get the state for.

    Raises:
        ValueError: If the image with the given ID is not found.

    Returns:
        ImageState: The image state for the given image ID.
    """
    global state_map

    if image_id in state_map:
        return state_map[image_id]
    else:
        raise HTTPException(
            status_code=400, detail=f"Image with ID {image_id} not found."
        )


def image_array_to_response(image_array: np.ndarray) -> StreamingResponse:
    """Prepares a numpy array for streaming as an image.

    Args:
        image_array (np.ndarray): The image as a numpy array.

    Returns:
        StreamingResponse: The image as a PNG streaming response.
    """
    image = Image.fromarray(image_array)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")

    img_byte_array = buffered.getvalue()

    return StreamingResponse(io.BytesIO(img_byte_array), media_type="image/png")


@app.post("/images")
def set_image(file: UploadFile = File(...)) -> str:
    """Uploads an image and returns an ID to reference it. The image has to be
    supplied as a form-data file.

    Args:
        file (UploadFile, optional): The image to upload.

    Returns:
        str: The ID of the uploaded image.
    """
    global state_map

    with io.BytesIO() as buffer:
        shutil.copyfileobj(file.file, buffer)
        buffer.seek(0)

        image_array = np.array(Image.open(buffer))

    logging.info(f"Received image with shape {image_array.shape}.")

    state = ImageState(image=image_array)

    id = str(uuid.uuid4())

    logging.info(f"Registered image with ID {id}.")

    state_map[id] = state

    return id


@app.get("/images/{image_id}/mask")
def get_mask(image_id: str) -> StreamingResponse:
    """Returns the current mask for the image with the given ID.

    Args:
        image_id (str): The ID of the image to get the mask for.

    Raises:
        ValueError: If the image with the given ID is not found.

    Returns:
        StreamingResponse: The current mask as a PNG image.
    """
    state = get_image_state(image_id)

    return image_array_to_response(state.current_mask)


class ClickData(BaseModel):
    x: int
    y: int


@app.post("/images/{image_id}/clicks")
def register_click(
    image_id: str, click_data: ClickData, expand: bool = False
) -> StreamingResponse:
    """Registers a click on the image with the given ID and returns the new mask.

    Args:
        image_id (str): The ID of the image to register the click on.
        click_data (ClickData): The coordinates of the click.

    Raises:
        ValueError: If the image with the given ID is not found.

    Returns:
        StreamingResponse: The new mask as a PNG image.
    """
    state = get_image_state(image_id)

    if expand and len(state.clicks) > 0:
        state.clicks[-1].append((click_data.x, click_data.y))
    else:
        state.clicks.append([(click_data.x, click_data.y)])

    masks, _, _ = state.predictor.predict(
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


@app.get("/images/{image_id}/click-count")
def click_count(image_id: str) -> int:
    """Returns the number of clicks on the image with the given ID.

    Args:
        image_id (str): The ID of the image to get the click count for.

    Raises:
        ValueError: If the image with the given ID is not found.

    Returns:
        int: The number of clicks on the image.
    """
    state = get_image_state(image_id)

    return len(state.clicks)


@app.post("/images/{image_id}/undo")
def undo(image_id: str) -> StreamingResponse:
    """Undoes the last click on the image with the given ID and returns the new (old) mask.

    Args:
        image_id (str): The ID of the image to undo the last click on.

    Raises:
        ValueError: If the image with the given ID is not found.

    Returns:
        StreamingResponse: The new (old) mask as a PNG image.
    """
    state = get_image_state(image_id)

    state.undo()

    return image_array_to_response(state.current_mask)


@app.post("/images/{image_id}/clear-mask")
def clear_mask(image_id: str) -> StreamingResponse:
    """Clears the mask for the image with the given ID and returns a clear mask.

    Args:
        image_id (str): The ID of the image to clear the mask for.

    Raises:
        ValueError: If the image with the given ID is not found.

    Returns:
        StreamingResponse: The clear mask as a PNG image.
    """
    state = get_image_state(image_id)

    state.clear_mask()

    return image_array_to_response(state.current_mask)


output_path = Path(__file__).parent / "output"


class FinishData(BaseModel):
    filename: str


@app.post("/images/{image_id}/finish")
def finish_image(image_id: str, data: FinishData) -> None:
    """Finishes the image with the given ID.

    Args:
        image_id (str): The ID of the image to finish.
    """
    state = get_image_state(image_id)

    output_path.mkdir(parents=True, exist_ok=True)

    mask = state.current_mask

    mask_image = Image.fromarray(mask)
    mask_image.save(output_path / data.filename)

    del state_map[image_id]


@app.delete("/images/{image_id}")
def delete_image(image_id: str) -> None:
    """Deletes the image with the given ID.

    Args:
        image_id (str): The ID of the image to delete.
    """
    global state_map

    del state_map[image_id]


if __name__ == "__main__":
    # Add timestamp to logs
    LOGGING_CONFIG["formatters"]["default"][
        "fmt"
    ] = "%(asctime)s [%(name)s] %(levelprefix)s %(message)s"
    LOGGING_CONFIG["formatters"]["access"][
        "fmt"
    ] = '%(asctime)s [%(name)s] %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'

    uvicorn.run(
        app,
        port=80 if os.getenv("IS_DOCKER_CONTAINER") else 8000,
        host="0.0.0.0" if os.getenv("IS_DOCKER_CONTAINER") else "127.0.0.1",
        log_level="info",
    )
