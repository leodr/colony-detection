import base64
import io
from typing import Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from segment_anything import SamPredictor, sam_model_registry

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
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
