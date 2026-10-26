"""AWS Lambda: BRIA RMBG 2.0 on one 1024x1024 image (api/bria.py's predict).

Called through the function URL by api/bria.py:
    POST {"image": <base64 PNG, 1024x1024 RGB>}   header X-Rembg-Key: <REMBG_KEY>
    ->   {"mask":  <base64 PNG, 1024x1024 L>}

The model is loaded on the first call of each Lambda instance and kept for
the warm calls after it.
"""

import base64
import json
import os

import numpy as np

import bria

MODEL = "/opt/bria-rmbg.onnx"
_session = None


def _reply(status: int, payload: dict) -> dict:
    return {"statusCode": status, "headers": {"Content-Type": "application/json"}, "body": json.dumps(payload)}


def handler(event, context):
    global _session
    want = os.environ.get("REMBG_KEY", "")
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    if not want or headers.get("x-rembg-key") != want:
        return _reply(401, {"error": "missing or wrong X-Rembg-Key"})
    body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode()
    try:
        pixels = bria.from_png_b64(json.loads(body)["image"])
    except (ValueError, KeyError, TypeError) as exc:
        return _reply(400, {"error": f"bad request: {exc}"})
    if pixels.shape != (bria.SIDE, bria.SIDE, 3) or pixels.dtype != np.uint8:
        return _reply(400, {"error": f"expected {bria.SIDE}x{bria.SIDE} RGB, got {pixels.shape} {pixels.dtype}"})
    if _session is None:
        _session = bria.session(MODEL)
    mask = bria.predict(_session, pixels)
    return _reply(200, {"mask": bria.png_b64(mask)})
