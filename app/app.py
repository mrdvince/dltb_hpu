import os
from io import BytesIO
from typing import Any

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.responses import RedirectResponse

from .inference import load_model, predict

app = FastAPI(title="dl4tb")
base_path = os.path.dirname(os.path.abspath(__file__))

model = load_model(os.path.join(base_path, "checkpoint_4_hpu.pth"))

static_dir = os.path.join(base_path, "..", "static/")

app.mount("/media", StaticFiles(directory=static_dir, html=True), name="runs")


@app.get("/", tags=["redirect"])
def redirect_to_docs() -> Any:
    return RedirectResponse(url="redoc")


def read_image_from_upload(upload_file: UploadFile):
    img_stream = BytesIO(upload_file.file.read())
    return Image.open(img_stream).convert("RGB")


@app.post("/")
def get_predictions(request: Request, file: UploadFile = File(...)) -> Any:
    """
    Upload image and get prediction
    """
    url = request.client.host
    if url in ["localhost", "127.0.0.1"]:
        url = f"http://{url}:8000"
    img = read_image_from_upload(file)
    filename = predict(model, img, filename=file.filename, static_dir=static_dir)

    return {
        "mask_url": url + "/media/mask_" + filename,
        "segmentation_url": url + "/media/weighted_" + filename,
    }
