import os
from io import BytesIO
from typing import Any

from fastapi import FastAPI, File, UploadFile
from PIL import Image
from starlette.responses import RedirectResponse

from .inference import TchPredictor

stage = os.environ.get("STAGE", None)
openapi_prefix = f"/{stage}" if stage else "/"

app = FastAPI(title="dl4tb", openapi_prefix=openapi_prefix)


@app.get("/", tags=["redirect"])
def redirect_to_docs() -> Any:
    return RedirectResponse(url="redoc")


# load the model
predictor = TchPredictor(
    os.path.join(os.getcwd(), "./saved/exports/model_best_checkpoint.onnx"), onnx=True
)


def read_image_from_upload(upload_file: UploadFile):
    img_stream = BytesIO(upload_file.file.read())
    return Image.open(img_stream).convert("RGB")


@app.post("/")
def get_predictions(file: UploadFile = File(...)) -> Any:
    """
    Upload image and get prediction
    """
    img = read_image_from_upload(file)
    pred = predictor.onnx_predict(img)
    return pred
