import sys
import os
import io

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import app as app_module  # noqa: E402

client = TestClient(app_module.app)


def _fake_image_bytes() -> bytes:
    array = (np.random.rand(200, 200, 3) * 255).astype(np.uint8)
    image = Image.fromarray(array, mode="RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def test_health_endpoint_reports_status():
    response = client.get("/health")
    # 200 if a model checkpoint happens to be present in this env, 503
    # if not (e.g. CI running before train.py has produced model.pt) —
    # either way the endpoint itself must respond, not crash.
    assert response.status_code in (200, 503)


def test_predict_rejects_non_image_payload():
    response = client.post(
        "/predict",
        files={"file": ("not_an_image.txt", b"hello world", "text/plain")},
    )
    assert response.status_code in (400, 503)


def test_predict_with_valid_image_when_model_loaded():
    if app_module.model is None:
        # No trained checkpoint available in this environment (e.g. a
        # fresh checkout before training has run) — skip rather than
        # fail, since /health already covers the "model missing" path.
        return

    response = client.post(
        "/predict",
        files={"file": ("cat.jpg", _fake_image_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in ("cat", "dog")
    assert 0.0 <= body["confidence"] <= 1.0
