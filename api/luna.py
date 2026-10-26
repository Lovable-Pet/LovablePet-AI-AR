"""Template-conditioned view generation through a Responses-API model driving the image tool.

WHAT "LUNA" IS HERE
-------------------
The driver was GPT-5.6 Luna (`gpt-5.6-luna`); it is now gpt-4o by default
(LUNA_MODEL in the environment picks another). Either is a text-and-image-in, text-out
model. It does not draw by itself: OpenAI's model page lists the Images
endpoints as not supported for it, and lists the `image_generation` tool of
the Responses API as supported. So a Luna image generation is a Responses
call with the guide and the photo as `input_image` content, and the
`image_generation` tool attached; Luna reads both pictures and the prompt and
drives the image model (`gpt-image-*`) through the tool. The picture comes
back as an `image_generation_call` output item whose `result` is base64 PNG.

Nothing below is guessed: the request shape and the parameter names are the
ones in OpenAI's image-generation guide (Responses API section) and the
model page. The tool's own model defaults to gpt-image-1, the image model
this project has used from the start; LUNA_IMAGE_MODEL overrides it.

THE CONDITIONING
----------------
Each view is one call with two pictures: the structure guide rendered from
the template with the very camera the projection will use, and the shelter
photo. The prompt binds pose, proportions and silhouette to the guide and
the coat to the photo, asks for flat light and a pure white ground. Four
calls, one per canonical camera; the result is cut out against the guide's
silhouette and saved at the guide's resolution so the projection sees one
pixel grid.

Credentials come from LUNA_API_KEY (OPENAI_API_KEY is accepted as a fallback
because it is the same platform); nothing is hard-coded.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from .imagegen import _rembg_alpha, cut_out

# The Responses-API model that drives the image tool. gpt-4o by default (it
# reads the guide and the photo and calls image_generation the same way);
# LUNA_MODEL in the environment switches it, e.g. back to gpt-5.6-luna.
LUNA_MODEL = os.environ.get("LUNA_MODEL", "gpt-4o")
DEFAULT_IMAGE_TOOL_MODEL = "gpt-image-1"
DEFAULT_SIZE = "1024x1024"
VIEW_NAMES = ("front", "left", "right", "back")

# Retries: exponential back-off, three attempts in all.
MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 2.0
BACKOFF_JITTER_SECONDS = 0.5

# FRAMING
# -------
# The model keeps the guide's pose and proportions but not its place in the
# frame: the first live run drew the side view a third larger than the guide
# and the front view as a head close-up. The camera is the same, so a picture
# that is not where the guide is projects onto the wrong part of the mesh.
# Two answers. The generated dog is fitted to the guide's silhouette by one
# similarity transform - uniform scale and translation from the bounding
# boxes, no warping - and the fit is scored by silhouette overlap (IoU). A
# score below the floor is a framing failure (a close-up, a cropped body) and
# the view is generated again with the framing spelled out harder, up to
# FRAMING_ATTEMPTS pictures per view.
FRAMING_IOU_FLOOR = 0.6
FRAMING_ATTEMPTS = 3
FRAMING_EMPHASIS = (
    "IMPORTANT: the previous attempt did not match the guide's framing. Show the ENTIRE dog, "
    "full body, at exactly the same size and position in the frame as the guide - nothing "
    "cropped, no close-up, no zoom. The dog's outline must fall on the guide's outline."
)

ANGLE_WORDS = {
    "front": "seen from directly in front",
    "left": "seen from its left side, in profile",
    "right": "seen from its right side, in profile",
    "back": "seen from directly behind",
}


class LunaTransport(Protocol):
    async def generate(self, *, prompt: str, images: list[bytes], size: str) -> bytes: ...


def _data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


class OpenAILunaTransport:
    """The driver model (gpt-4o by default) on the Responses API with the image_generation tool."""

    def __init__(self, client=None, model: str = LUNA_MODEL,
                 image_model: str | None = None, quality: str = "high") -> None:
        import openai

        api_key = os.environ.get("LUNA_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if client is None and not api_key:
            raise RuntimeError("LUNA_API_KEY is not set")
        self._client = client or openai.AsyncOpenAI(api_key=api_key)
        self._model = model
        self._image_model = image_model or os.environ.get("LUNA_IMAGE_MODEL") or DEFAULT_IMAGE_TOOL_MODEL
        self._quality = quality

    async def generate(self, *, prompt: str, images: list[bytes], size: str) -> bytes:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        content += [{"type": "input_image", "image_url": _data_url(png)} for png in images]
        response = await self._client.responses.create(
            model=self._model,
            input=[{"role": "user", "content": content}],
            tools=[{
                "type": "image_generation",
                "model": self._image_model,
                "size": size,
                "quality": self._quality,
                "background": "opaque",
            }],
        )
        results = [item.result for item in response.output
                   if getattr(item, "type", "") == "image_generation_call" and getattr(item, "result", None)]
        if not results:
            text = "".join(getattr(item, "text", "") or "" for item in response.output
                           if getattr(item, "type", "") == "message")
            raise RuntimeError("Luna returned no image_generation_call result"
                               + (f"; message: {text[:200]}" if text else ""))
        return base64.b64decode(results[0])


def build_prompt(angle: str, appearance_text: str | None = None) -> str:
    """The four conditions the request lists, plus the angle and, when known,
    the coat in words."""
    lines = [
        f"Draw the dog in the second image {ANGLE_WORDS.get(angle, angle)}, as a single clean "
        "reference picture.",
        "The first image is a structure guide: a flat grey render of the exact 3D model this "
        "picture will be applied to. Keep exactly the same pose, proportions and silhouette as "
        "the guide - the same head shape, muzzle length, eye positions, ear shape and leg "
        "placement, in the same place in the frame. Do not restyle the anatomy.",
        "Take the coat from the second image, the actual pet photograph: reproduce its colours "
        "and markings exactly as they are - the same patches in the same places, with the same "
        "soft or hard edges. Do not invent markings the photo does not show.",
        "Flat lighting: perfectly even illumination with no shadows, no highlights, no shading "
        "gradients and no ambient occlusion. Every part of the coat shows its own colour only.",
        "Pure white background (#FFFFFF), nothing else in the picture, no ground shadow.",
        "Keep the eyes, nose and mouth where the guide has them.",
        "Show the entire dog exactly as framed in the guide: the same size and the same "
        "position in the picture, whole body visible, nothing cropped or zoomed.",
    ]
    if appearance_text:
        lines.append(f"For reference, the coat reads as: {appearance_text}")
    return "\n".join(lines)


async def generate_view(
    guide_image: Image.Image | str | Path,
    pet_photo: Image.Image | str | Path,
    angle: str,
    transport: LunaTransport,
    size: str = DEFAULT_SIZE,
    appearance_text: str | None = None,
    extra: str | None = None,
) -> Image.Image:
    """One view of the pet drawn to the template's structure guide.

    Retries the call up to MAX_ATTEMPTS times with exponential back-off, and
    returns the picture as a PIL image."""
    guide_png = _png_bytes(guide_image)
    photo_png = _png_bytes(pet_photo)
    prompt = build_prompt(angle, appearance_text)
    if extra:
        prompt = prompt + "\n" + extra
    last: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            data = await transport.generate(prompt=prompt, images=[guide_png, photo_png], size=size)
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - any transport failure is retried
            last = exc
            if attempt == MAX_ATTEMPTS - 1:
                break
            delay = BACKOFF_BASE_SECONDS * (2 ** attempt) + random.uniform(0, BACKOFF_JITTER_SECONDS)
            print(f"[luna] {angle}: attempt {attempt + 1} failed ({type(exc).__name__}: {exc}); "
                  f"retrying in {delay:.1f}s", flush=True)
            await asyncio.sleep(delay)
    raise RuntimeError(f"Luna generation failed for {angle} after {MAX_ATTEMPTS} attempts") from last


def _png_bytes(image: Image.Image | str | Path) -> bytes:
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def subject_alpha(picture: Image.Image) -> np.ndarray:
    """Where the dog is in a generated picture: rembg, else not-white."""
    alpha = _rembg_alpha(picture.convert("RGB"))
    if alpha is None:
        rgb = np.asarray(picture.convert("RGB"), dtype=np.float32) / 255.0
        alpha = (rgb.min(axis=2) < 0.97).astype(np.float32)
    return np.clip(alpha, 0.0, 1.0)


def align_to_guide(picture: Image.Image, guide_mask: Image.Image) -> tuple[Image.Image, float]:
    """The picture scaled and moved so that its dog's bounding box sits on
    the guide silhouette's - one similarity transform, no warping - on a
    white canvas at the guide's size. Returns the aligned picture and the
    silhouette IoU after alignment."""
    target = guide_mask.size
    mask = np.asarray(guide_mask.convert("L").resize(target, Image.NEAREST)) > 127
    alpha = subject_alpha(picture) > 0.5
    src, dst = _bbox(alpha), _bbox(mask)
    if src is None or dst is None:
        return picture.resize(target, Image.LANCZOS), 0.0
    sw, sh = src[2] - src[0], src[3] - src[1]
    dw, dh = dst[2] - dst[0], dst[3] - dst[1]
    # one scale for both axes, from the box heights (the model draws the
    # body's height more reliably than its width), centred on the boxes
    scale = dh / max(sh, 1)
    new_size = (max(1, int(round(picture.size[0] * scale))), max(1, int(round(picture.size[1] * scale))))
    scaled = picture.convert("RGB").resize(new_size, Image.LANCZOS)
    scaled_alpha = Image.fromarray((alpha * 255).astype(np.uint8)).resize(new_size, Image.NEAREST)
    src_cx, src_cy = (src[0] + src[2]) / 2 * scale, (src[1] + src[3]) / 2 * scale
    dst_cx, dst_cy = (dst[0] + dst[2]) / 2, (dst[1] + dst[3]) / 2
    offset = (int(round(dst_cx - src_cx)), int(round(dst_cy - src_cy)))
    canvas = Image.new("RGB", target, (255, 255, 255))
    canvas.paste(scaled, offset)
    placed = Image.new("L", target, 0)
    placed.paste(scaled_alpha, offset)
    moved = np.asarray(placed) > 127
    union = (moved | mask).sum()
    iou = float((moved & mask).sum() / union) if union else 0.0
    return canvas, iou


@dataclass(frozen=True)
class GuidedViews:
    views: dict[str, Path]       # cut-out RGBA at the guide resolution
    raw: dict[str, Path]         # the pictures as generated
    prompt: str
    cameras: Path


def guide_paths(guides_dir: str | Path, template_stem: str) -> tuple[dict[str, Path], dict[str, Path], Path]:
    guides_dir = Path(guides_dir)
    guides = {name: guides_dir / f"{template_stem}_{name}.png" for name in VIEW_NAMES}
    masks = {name: guides_dir / f"{template_stem}_{name}_mask.png" for name in VIEW_NAMES}
    cameras = guides_dir / f"{template_stem}_cameras.json"
    missing = [str(p) for p in [*guides.values(), *masks.values(), cameras] if not p.exists()]
    if missing:
        raise FileNotFoundError("structure guides missing; run ar_pipeline/render_guides.py first: "
                                + ", ".join(missing[:3]))
    return guides, masks, cameras


async def generate_views(
    photo_path: str | Path,
    guides_dir: str | Path,
    template_stem: str,
    transport: LunaTransport,
    out_dir: str | Path,
    appearance_text: str | None = None,
    size: str = DEFAULT_SIZE,
) -> GuidedViews:
    """The four views, one call each, cut out against the guide silhouettes
    and written beside a views.json the projection step reads."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    guides, masks, cameras = guide_paths(guides_dir, template_stem)
    resolution = tuple(json.loads(cameras.read_text(encoding="utf-8"))["resolution"])
    photo = Image.open(photo_path).convert("RGB")

    prompt = build_prompt("front", appearance_text)
    (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    views: dict[str, Path] = {}
    raw: dict[str, Path] = {}
    framing: dict[str, dict[str, Any]] = {}
    for name in VIEW_NAMES:
        best: tuple[float, Image.Image, Image.Image] | None = None
        extra = None
        for attempt in range(FRAMING_ATTEMPTS):
            print(f"[luna] generating the {name} view (attempt {attempt + 1})...", flush=True)
            picture = await generate_view(guides[name], photo, name, transport, size, appearance_text, extra)
            aligned, iou = align_to_guide(picture, Image.open(masks[name]))
            print(f"[luna] {name}: silhouette IoU after alignment {iou:.2f}", flush=True)
            if best is None or iou > best[0]:
                best = (iou, picture, aligned)
            if iou >= FRAMING_IOU_FLOOR:
                break
            extra = FRAMING_EMPHASIS
        iou, picture, aligned = best
        raw_path = out_dir / f"{name}_raw.png"
        picture.save(raw_path)
        raw[name] = raw_path
        aligned.save(out_dir / f"{name}_aligned.png")
        rgba, how = cut_out(aligned, Image.open(masks[name]))
        path = out_dir / f"{name}.png"
        rgba.save(path)
        views[name] = path
        framing[name] = {"iou": round(iou, 3), "ok": iou >= FRAMING_IOU_FLOOR}
        print(f"[luna] {name}: {how}", flush=True)

    manifest = {
        "source": "luna",
        "model": LUNA_MODEL,
        "guides": {name: str(path) for name, path in guides.items()},
        "cameras": str(cameras),
        "size_generated": size,
        "size_views": list(resolution),
        "views": {name: path.name for name, path in views.items()},
        "raw": {name: path.name for name, path in raw.items()},
        "framing": framing,
    }
    (out_dir / "views.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return GuidedViews(views=views, raw=raw, prompt=prompt, cameras=cameras)


def realign_views(guides_dir: str | Path, template_stem: str, out_dir: str | Path) -> dict[str, float]:
    """Re-derive the cut-out views from the raw pictures already on disk,
    with the alignment - no new calls."""
    out_dir = Path(out_dir)
    guides, masks, cameras = guide_paths(guides_dir, template_stem)
    scores: dict[str, float] = {}
    for name in VIEW_NAMES:
        raw_path = out_dir / f"{name}_raw.png"
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        aligned, iou = align_to_guide(Image.open(raw_path), Image.open(masks[name]))
        aligned.save(out_dir / f"{name}_aligned.png")
        rgba, how = cut_out(aligned, Image.open(masks[name]))
        rgba.save(out_dir / f"{name}.png")
        scores[name] = round(iou, 3)
        print(f"[luna] {name}: realigned, IoU {iou:.2f}; {how}", flush=True)
    manifest_path = out_dir / "views.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["framing"] = {name: {"iou": iou, "ok": iou >= FRAMING_IOU_FLOOR} for name, iou in scores.items()}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return scores


__all__ = [
    "DEFAULT_SIZE", "LUNA_MODEL", "GuidedViews", "LunaTransport", "OpenAILunaTransport",
    "align_to_guide", "build_prompt", "generate_view", "generate_views", "guide_paths",
    "realign_views", "subject_alpha",
]
