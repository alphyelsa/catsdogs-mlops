import io
import logging
import time

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from prometheus_fastapi_instrumentator import Instrumentator

from config import CLASS_NAMES, MODEL_PATH
from model import build_model
from pipelines import preprocess_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("catsdogs-api")

app = FastAPI(title="Cats vs Dogs Image Classification Service")

# Prometheus metrics: request count, latency histograms, etc. at /metrics
Instrumentator().instrument(app).expose(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    logger.info(
        "Request completed | method=%s path=%s status=%s duration=%.4fs",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )
    return response


def _load_model():
    try:
        model = build_model(num_classes=len(CLASS_NAMES))
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        logger.info("Model loaded successfully from %s", MODEL_PATH)
        return model
    except Exception as exc:  # noqa: BLE001 - want to log any load failure
        logger.error("Model loading failed: %s", exc)
        return None


model = _load_model()


@app.get("/health")
def health():
    """
    Health check endpoint for readiness/liveness probes and smoke
    tests. Reports degraded (503) if the model failed to load so the
    orchestrator / CD smoke test can catch a broken deployment before
    routing real traffic to it.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        logger.error("Prediction request failed: model unavailable")
        raise HTTPException(status_code=503, detail="Model artifact unavailable.")

    request_start = time.time()

    try:
        raw_bytes = await file.read()
        image = Image.open(io.BytesIO(raw_bytes))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    try:
        preprocessing_start = time.time()
        input_tensor = preprocess_image(image).to(device)
        preprocessing_time = time.time() - preprocessing_start

        inference_start = time.time()
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())
        inference_time = time.time() - inference_start

        total_time = time.time() - request_start

        logger.info(
            "Prediction completed | label=%s confidence=%.4f "
            "preprocessing_time=%.4fs inference_time=%.4fs total_time=%.4fs",
            CLASS_NAMES[pred_idx],
            float(probs[pred_idx]),
            preprocessing_time,
            inference_time,
            total_time,
        )

        return {
            "label": CLASS_NAMES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
            },
        }

    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed | error=%s", str(exc))
        raise HTTPException(status_code=500, detail="Prediction failed.")
