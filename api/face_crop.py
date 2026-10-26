"""Stage 2-2 - the face, cut out of the photograph for the face call.

WHY A CROP
----------
On a full-body shelter photo the face is a few hundred pixels at best, and
the five face regions - brows, muzzle bridge, cheeks, inner ears, skull with the outer ears -
are each a sliver of that. Sent the whole photo, the region agent reads the
face at the same scale as the flank. Sent the face alone, at the model's
full input resolution, it reads the brows and the muzzle as the large
areas they are.

HOW
---
Three steps, each narrowing the last:
  1. rembg cuts the animal out of the background, and its alpha's bounding
     box is the subject. A photo with no cut-out (rembg missing, or nothing
     found) falls back to the whole frame.
  2. a vision call locates the face box inside the subject box, in pixel
     coordinates of the crop it was shown, and says whether the face is
     seen frontally - which the face-photo step (face_photo.py) reads later.
  3. the box is grown by 20% on every side so the ears and the throat,
     which sit on its edge, are inside it, then clipped to the image.

The crop is returned as a PIL image and its box in ORIGINAL photo pixels,
so a landmark read on the crop can be mapped back to the photograph.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .agents import Transport

PAD_SHARE = 0.10
# The crop the model sees is resized so its longer side is this many pixels;
# the box it returns is scaled back.
FACE_QUERY_SIDE = 1024
MIN_FACE_SIDE = 48
# The crop is sent at least this large on its longer side. A shelter photo
# is a few hundred pixels and its head a hundred; sent as-is the model reads
# the brows and the muzzle as a handful of pixels each.
FACE_SEND_SIDE = 1024

# The head is asked for as grid cells, not pixels: the pixel box came back
# off the head in 6 of 17 eval photos (an ear alone for e12, the collar for
# e14, half a face for e05, e16, e20 - eval v3), and the face colours were
# then read off whatever was in it. The same lettered grid the colour
# measurement points with (face_photo.draw_grid) is read far more reliably.
FACE_BOX_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "face_box": {
            "type": "object",
            "description": "The grid cells of the smallest rectangle holding the whole head.",
            "properties": {
                "top_left": {"type": "string", "description": "Cell label, e.g. C2: the top-left corner cell."},
                "bottom_right": {"type": "string", "description": "Cell label, e.g. F6: the bottom-right corner cell."},
            },
            "required": ["top_left", "bottom_right"],
            "additionalProperties": False,
        },
        "visible": {"type": "boolean", "description": "false if no head is in the image."},
        "frontal": {
            "type": "boolean",
            "description": "true if both eyes and the nose are clearly visible and the head faces the camera within about 30 degrees.",
        },
    },
    "required": ["face_box", "visible", "frontal"],
    "additionalProperties": False,
}

FACE_BOX_SYSTEM = """You locate a dog's head in a photograph.

The image has a lettered grid: columns A, B, C... left to right, rows 1, 2,
3... top to bottom, each cell's label in its corner. Name the top-left and
the bottom-right cell of the smallest rectangle that holds the WHOLE head:
the tips of both ears, the tip of the nose and the underside of the jaw
must all be inside it. When in doubt take the larger rectangle - a cell too
many is harmless, a cut-off muzzle is not.

Set frontal to true only when the head faces the camera - both eyes and the
nose leather are clearly visible and the muzzle points at the viewer within
about thirty degrees. A three-quarter or profile view is not frontal."""


@dataclass
class FaceCrop:
    image: Image.Image
    box: tuple[int, int, int, int]        # x0, y0, x1, y1 in original photo pixels
    subject_box: tuple[int, int, int, int]
    frontal: bool
    visible: bool
    raw_box: dict[str, str]

    def to_photo(self, x: float, y: float) -> tuple[float, float]:
        """A point in crop pixels -> original photo pixels."""
        return x + self.box[0], y + self.box[1]

    def report(self) -> dict[str, Any]:
        return {
            "box": list(self.box),
            "subject_box": list(self.subject_box),
            "frontal": self.frontal,
            "visible": self.visible,
            "size": list(self.image.size),
        }


def _image_block(image: Image.Image) -> dict[str, Any]:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png",
                   "data": base64.standard_b64encode(buffer.getvalue()).decode("ascii")},
    }


def subject_box(image: Image.Image) -> tuple[int, int, int, int]:
    """Bounding box of the animal from rembg, or the whole frame."""
    try:
        from .imagegen import _rembg_alpha
        alpha = _rembg_alpha(image.convert("RGB"))
    except Exception:  # noqa: BLE001 - rembg is optional
        alpha = None
    width, height = image.size
    if alpha is None or (alpha > 0.5).sum() < 0.01 * width * height:
        return 0, 0, width, height
    rows, cols = np.nonzero(alpha > 0.5)
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def _pad(box: tuple[int, int, int, int], bounds: tuple[int, int], share: float) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    px, py = w * share, h * share
    width, height = bounds
    return (
        int(max(0, round(x0 - px))), int(max(0, round(y0 - py))),
        int(min(width, round(x1 + px))), int(min(height, round(y1 + py))),
    )


# Two answers, their rectangles joined: one answer in 17 still cut half a
# face off (e18, the grid run); a second read rarely misses the same side.
BOX_SAMPLES = 2


async def crop_face(photo_path: str | Path, transport: Transport, samples: int = BOX_SAMPLES,
                    subject: tuple[int, int, int, int] | None = None) -> FaceCrop:
    """The face crop of one photograph. When no head is found the crop is the
    subject box itself, marked visible=False, so the face call still has
    something to look at and the log says why. `subject`: the animal's box
    when the caller already has it (prepare_photo's cut), else rembg finds it."""
    image = Image.open(photo_path).convert("RGB")
    width, height = image.size
    sx0, sy0, sx1, sy1 = subject or subject_box(image)
    subject = image.crop((sx0, sy0, sx1, sy1))

    scale = FACE_QUERY_SIDE / max(subject.size)
    shown = subject if scale >= 1.0 else subject.resize(
        (max(1, int(subject.size[0] * scale)), max(1, int(subject.size[1] * scale))), Image.LANCZOS)
    scale = shown.size[0] / subject.size[0]

    from .face_photo import GRID_COLS, GRID_ROWS, _COLS, draw_grid

    import asyncio

    blocks = [_image_block(draw_grid(shown)), {"type": "text", "text": (
        f"The image is {shown.size[0]} pixels wide and {shown.size[1]} pixels tall, under a grid of "
        f"{GRID_COLS} columns ({_COLS[0]}-{_COLS[-1]}) by {GRID_ROWS} rows. Give the head's corner cells.")}]
    payloads = await asyncio.gather(*(transport.complete(system=FACE_BOX_SYSTEM, blocks=blocks, schema=FACE_BOX_SCHEMA)
                                      for _ in range(max(samples, 1))), return_exceptions=True)
    payloads = [p for p in payloads if isinstance(p, dict)]
    if not payloads:
        raise RuntimeError("the head box call failed")
    raw = payloads[0].get("face_box") or {}
    visible = any(bool(p.get("visible", True)) for p in payloads)
    frontal = sum(bool(p.get("frontal", False)) for p in payloads) * 2 > len(payloads)

    def cell(label: str) -> tuple[int, int]:
        label = str(label).strip().upper()
        col, row = _COLS.index(label[0]), int(label[1:]) - 1
        if not (0 <= col < GRID_COLS and 0 <= row < GRID_ROWS):
            raise ValueError(label)
        return col, row

    cw, ch = subject.size[0] / GRID_COLS, subject.size[1] / GRID_ROWS
    rects = []
    for p in payloads:
        answer = p.get("face_box") or {}
        try:
            c0, r0 = cell(answer["top_left"])
            c1, r1 = cell(answer["bottom_right"])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        rects.append((min(c0, c1), min(r0, r1), max(c0, c1), max(r0, r1)))
    if rects:
        c0, r0 = min(r[0] for r in rects), min(r[1] for r in rects)
        c1, r1 = max(r[2] for r in rects), max(r[3] for r in rects)
        bx0, by0 = c0 * cw + sx0, r0 * ch + sy0
        bx1, by1 = (c1 + 1) * cw + sx0, (r1 + 1) * ch + sy0
        raw = {"cells": [f"{_COLS[r[0]]}{r[1] + 1}-{_COLS[r[2]]}{r[3] + 1}" for r in rects]}
    else:
        visible = False
        bx0, by0, bx1, by1 = sx0, sy0, sx1, sy1

    box = (int(np.clip(min(bx0, bx1), 0, width)), int(np.clip(min(by0, by1), 0, height)),
           int(np.clip(max(bx0, bx1), 0, width)), int(np.clip(max(by0, by1), 0, height)))
    if not visible or box[2] - box[0] < MIN_FACE_SIDE or box[3] - box[1] < MIN_FACE_SIDE:
        visible = False
        frontal = False
        box = (sx0, sy0, sx1, sy1)
    else:
        box = _pad(box, (width, height), PAD_SHARE)

    return FaceCrop(
        image=image.crop(box),
        box=box,
        subject_box=(sx0, sy0, sx1, sy1),
        frontal=frontal,
        visible=visible,
        raw_box={k: str(v) for k, v in raw.items()} if raw else {},
    )


def crop_block(crop: FaceCrop) -> dict[str, Any]:
    image = crop.image
    longest = max(image.size)
    if longest < FACE_SEND_SIDE:
        scale = FACE_SEND_SIDE / longest
        image = image.resize((max(1, round(image.size[0] * scale)), max(1, round(image.size[1] * scale))),
                             Image.LANCZOS)
    return _image_block(image)


__all__ = ["FaceCrop", "crop_block", "crop_face", "subject_box"]
