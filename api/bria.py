"""The animal's mask from BRIA RMBG 2.0, here or on the Lambda behind REMBG_URL.

One 1024x1024 pass of rembg's model needs about 8 GB of memory on CPU (measured
2026-10-27: 9.5 GB through rembg, 8.6 GB with the CPU memory arena off), more
than the server should hold for the ten seconds it runs. With REMBG_URL set
the pass runs on AWS Lambda (lambda/rembg/), which only bills while it runs,
on the same model rewritten to peak at ~1.9 GB (lambda/rembg/slim_model.py;
masks within 1/255). The split is rembg's own computation cut in two: the
caller resizes the photo to 1024x1024 and the mask back to the photo's size,
the model step in between is predict() on either side.

This module is copied into the Lambda image as it is: it imports nothing from
api/, and rembg (model download) and httpx (the call) only where used.

Environment:
    REMBG_URL   the Lambda function URL; unset: run the model here
    REMBG_KEY   shared secret, sent as X-Rembg-Key and checked by the Lambda
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from functools import lru_cache

import numpy as np
from PIL import Image

SIDE = 1024                                   # the exported graph only runs at 1024x1024
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
# a cold Lambda loads the 1 GB model first; the first call after a new image
# ran past 180 s once. Keep this at the function's timeout (5 min): giving up
# sooner retries while the first call still runs, on a second cold instance.
TIMEOUT_S = 300
ATTEMPTS = 3


def model_input(rgb: Image.Image) -> np.ndarray:
    """The photo as the model takes it: 1024x1024 RGB uint8 (rembg's resize)."""
    return np.asarray(rgb.convert("RGB").resize((SIDE, SIDE), Image.Resampling.LANCZOS))


def predict(session, pixels: np.ndarray) -> np.ndarray:
    """1024x1024 uint8 mask for model_input's pixels (rembg's BriaRmBgSession:
    scaled by the brightest value, normalised, min-max of the output)."""
    a = pixels / max(np.max(pixels), 1e-6)
    x = np.zeros((a.shape[0], a.shape[1], 3))
    for c in range(3):
        x[:, :, c] = (a[:, :, c] - MEAN[c]) / STD[c]
    x = np.expand_dims(x.transpose((2, 0, 1)), 0).astype(np.float32)
    pred = session.run(None, {session.get_inputs()[0].name: x})[0][:, 0, :, :]
    pred = np.squeeze((pred - np.min(pred)) / (np.max(pred) - np.min(pred)))
    return (pred * 255).astype("uint8")


def to_alpha(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """The 1024x1024 mask at the photo's size, 0..1 float32."""
    resized = Image.fromarray(mask, mode="L").resize(size, Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.float32) / 255.0


def session(model_path: str):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    # same masks, ~1 GB less at the peak (the arena keeps freed blocks)
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    # onnxruntime starts a thread per core of the machine, also in a container
    # held to fewer CPUs: 6 threads on 2 CPUs ran 3x slower
    if hasattr(os, "sched_getaffinity"):
        opts.intra_op_num_threads = len(os.sched_getaffinity(0))
    return ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])


@lru_cache(maxsize=1)
def _local_session():
    from rembg.sessions.bria_rmbg import BriaRmBgSession

    return session(BriaRmBgSession.download_models())   # rembg's model dir, downloaded once


def png_b64(array: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def from_png_b64(text: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(text))))


def _remote_mask(url: str, pixels: np.ndarray) -> np.ndarray:
    import httpx

    body = json.dumps({"image": png_b64(pixels)})
    headers = {"Content-Type": "application/json", "X-Rembg-Key": os.environ.get("REMBG_KEY", "")}
    # the Lambda runtime emulator (testing the image locally) takes the event itself
    emulator = "/2015-03-31/functions/" in url
    if emulator:
        body = json.dumps({"body": body, "headers": {k.lower(): v for k, v in headers.items()},
                           "isBase64Encoded": False})
    error: Exception | None = None
    for attempt in range(ATTEMPTS):
        try:
            r = httpx.post(url, content=body, headers=headers, timeout=TIMEOUT_S)
            r.raise_for_status()
            reply = r.json()
            if emulator:
                if reply.get("statusCode", 200) >= 300:
                    raise RuntimeError(f"rembg lambda {reply.get('statusCode')}: {reply.get('body')}")
                reply = json.loads(reply["body"])
            return from_png_b64(reply["mask"])
        except Exception as exc:  # noqa: BLE001 - retried, then raised
            error = exc
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (401, 403, 413):
                break
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"background removal on {url} failed: {error}")


def alpha(rgb: Image.Image) -> np.ndarray:
    """The animal's alpha at the photo's size, 0..1."""
    pixels = model_input(rgb)
    url = os.environ.get("REMBG_URL")
    mask = _remote_mask(url, pixels) if url else predict(_local_session(), pixels)
    return to_alpha(mask, rgb.size)


__all__ = ["SIDE", "alpha", "from_png_b64", "model_input", "png_b64", "predict", "session", "to_alpha"]
