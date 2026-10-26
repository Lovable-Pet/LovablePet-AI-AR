"""Stage 4-2 - generate the four canonical views of the photographed dog.

WHAT IS ASKED OF THE IMAGE MODEL
--------------------------------
Two references go in: the shelter photo, and the 2x2 silhouette sheet that
canonical_views.py rendered from the template (front | left / right | back,
orthographic, one scale, rest pose). The model is asked for the same animal
drawn into those four cells with exactly that pose, angle and framing, on a
white ground, under flat even light with no cast shadows and no highlights -
as close to albedo as a painting can be. The coat is described in words from
the stage-2 appearance so the model does not have to guess it from one photo,
and it is told to infer the unseen side from symmetry and the coat pattern
rather than invent markings.

ONE SHEET, NOT FOUR CALLS
-------------------------
The four views are generated as a single image and cut into quarters. Four
separate generations would give four slightly different dogs - a different
tan, a different edge to the white - and the projection step would stitch
those differences into the coat as seams. One sheet is one animal.

BACKGROUND
----------
The white ground is removed with rembg when it is installed. The cut is then
intersected with the template's own silhouette, dilated a little, because the
projection only ever samples where the mesh is and a stray blob rembg kept
would otherwise land on the model. Without rembg the silhouette alone is the
alpha, which is a coarser cut but never wrong about where the animal is.
"""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageFilter

from .regions import REGIONS

# The order the cells are read back from the sheet: must match the order
# canonical_views wrote the silhouette grid in.
SHEET_ORDER = (("front", "left"), ("right", "back"))

# gpt-image-1 retires on 2026-10-23; gpt-image-2 is its documented successor
# and takes the same edit call (several references), but NOT input_fidelity:
# it answers 400 invalid_input_fidelity_model. The parameter is sent only to
# the models that take it, and a 400 naming it drops it and retries once, so
# a model added later that also refuses it still works.
DEFAULT_IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "gpt-image-2")
INPUT_FIDELITY_MODELS = ("gpt-image-1", "gpt-image-1-mini", "gpt-image-1.5", "chatgpt-image-latest")
# medium is a quarter of high's price and the atlas is painted at 1024 and
# upsampled to 2048 anyway; IMAGE_QUALITY=high to compare.
DEFAULT_IMAGE_QUALITY = os.environ.get("IMAGE_QUALITY", "medium")
DEFAULT_SIZE = "1024x1024"

# How far past the template silhouette the alpha may reach, as a fraction of
# the cell. The model draws into the silhouette but not perfectly; a hair of
# slack keeps an ear tip that overshoots.
MASK_DILATE_FRACTION = 0.012

# A pixel this close to white, inside the silhouette, is taken to be ground
# the model failed to fill rather than white fur - only used without rembg.
WHITE_CUTOFF = 0.985

# The alpha is pulled in by this fraction of the cell after the cut, so a dark
# outline the model drew along the silhouette is not projected as a seam. Two
# pixels at 1024 is the width of that stroke; six took 12% of the front view,
# because a foreleg is only a few dozen pixels across.
EDGE_ERODE_FRACTION = 0.002

PATTERN_NAMES = {
    "solid": "a solid coat",
    "bicolor": "a two-colour coat",
    "tricolor": "a black-and-tan coat with urajiro (cream on the underside)",
    "tan_point": "a black-and-tan coat with urajiro (cream on the underside)",
    "spotted": "a spotted coat",
    "brindle": "a brindle coat",
    "merle": "a merle coat",
    "sable": "a sable coat (dark-tipped hairs shading over the back, no hard edge)",
    "piebald": "a white coat with a coloured head and ears",
}


class ImageTransport(Protocol):
    async def edit(self, *, prompt: str, references: list[bytes], size: str,
                   mask: bytes | None = None) -> bytes: ...


class OpenAIImageTransport:
    """The edit endpoint, which accepts several reference images at once.
    input_fidelity (which keeps the photographed animal's identity) is sent
    only to the models that support it - see INPUT_FIDELITY_MODELS."""

    def __init__(self, client=None, model: str | None = None, quality: str | None = None) -> None:
        import openai

        self._client = client or openai.AsyncOpenAI()
        self._model = model or DEFAULT_IMAGE_MODEL
        self._quality = quality or DEFAULT_IMAGE_QUALITY
        self._fidelity = any(self._model.startswith(name) for name in INPUT_FIDELITY_MODELS)

    async def edit(self, *, prompt: str, references: list[bytes], size: str,
                   mask: bytes | None = None) -> bytes:
        files = [(f"reference_{index}.png", data, "image/png") for index, data in enumerate(references)]
        extra = {"mask": ("mask.png", mask, "image/png")} if mask is not None else {}
        if self._fidelity:
            extra["input_fidelity"] = "high"

        async def call(**kwargs: Any):
            return await self._client.images.edit(
                model=self._model, image=files, prompt=prompt, size=size,
                quality=self._quality, **kwargs,
            )

        try:
            response = await call(**extra)
        except Exception as exc:  # noqa: BLE001 - only the one 400 is handled
            if "input_fidelity" not in extra or "input_fidelity" not in str(exc):
                raise
            print(f"[imagegen] {self._model} rejected input_fidelity; retrying without it")
            self._fidelity = False
            extra.pop("input_fidelity")
            response = await call(**extra)

        usage = getattr(response, "usage", None)
        if usage is not None:
            from . import usage as ledger
            details = getattr(usage, "input_tokens_details", None)
            ledger.record("image_edit", self._model,
                          int(getattr(usage, "input_tokens", 0) or 0),
                          int(getattr(usage, "output_tokens", 0) or 0),
                          image_tokens=int(getattr(details, "image_tokens", 0) or 0) if details is not None else None)

        item = response.data[0]
        if getattr(item, "b64_json", None):
            return base64.b64decode(item.b64_json)
        raise RuntimeError("image model returned no b64_json (a URL response is not supported here)")


def appearance_to_text(appearance: dict[str, Any]) -> str:
    """The coat in words, grouped by colour, for the image prompt."""
    by_colour: dict[str, list[str]] = {}
    for entry in appearance.get("region_colors") or []:
        region, colour = entry.get("region"), str(entry.get("color", "")).lower()
        if region in REGIONS and colour:
            by_colour.setdefault(colour, []).append(region.replace("_", " "))
    base = str(appearance.get("base_color", "")).lower()
    parts = [f"base coat {base}"] if base else []
    for colour, regions in by_colour.items():
        if colour != base:
            parts.append(f"{colour} on the {', '.join(regions)}")
    return "; ".join(parts) + "."


def build_prompt(appearance: dict[str, Any], mismatches: list[str] | None = None) -> str:
    pattern = PATTERN_NAMES.get(str(appearance.get("pattern", "")).lower(), "the coat pattern seen in the photo")
    lines = [
        "Reference 1 is a shelter photograph of a dog. Reference 2 is a 2x2 sheet of "
        "white silhouettes of a 3D dog model on black: top-left is the front view, "
        "top-right the left side, bottom-left the right side, bottom-right the back "
        "view - all orthographic, at one scale, standing in a neutral rest pose.",
        "Draw the SAME dog from reference 1 as a 2x2 sheet, filling each cell with "
        "exactly the pose, angle, framing and outline of the corresponding silhouette.",
        "Pure white background. Flat, even studio lighting: no cast shadows, no "
        "specular highlights, no rim light - matte, albedo-like colour throughout. "
        "Keep the coat colours distinct and clearly separated; do not draw an outline "
        "stroke around the animal.",
        f"Coat colours: {appearance_to_text(appearance)}",
        f"For any side not visible in the photograph, infer it from left-right symmetry "
        f"and from {pattern}. Do not add any marking that is not on the photographed dog.",
    ]
    if mismatches:
        # A long list of corrections turns the task into a colour checklist and
        # the model paints blocks - the eyes and nose were the first things to
        # go. Three at most, and the face is named as something to keep.
        top = list(mismatches)[:3]
        lines.append(
            "A previous attempt got these wrong; fix only these: " + "; ".join(top) + ". "
            "Keep everything else as it was, especially the eyes, nose, mouth and the "
            "shape of the face - a dog with no visible eyes is wrong regardless of its colours."
        )
    return "\n".join(lines)


def _dilate(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask
    image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    return np.asarray(image.filter(ImageFilter.MaxFilter(2 * pixels + 1)), dtype=np.float32) / 255.0


def _erode(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask
    image = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8), mode="L")
    return np.asarray(image.filter(ImageFilter.MinFilter(2 * pixels + 1)), dtype=np.float32) / 255.0


def _rembg_alpha(rgb: Image.Image) -> np.ndarray | None:
    """The animal's alpha (0..1) from BRIA RMBG 2.0, the model rembg defaults
    to - on the Lambda when REMBG_URL is set (api/bria.py) - or None when the
    model cannot run here."""
    from . import bria

    try:
        return bria.alpha(rgb)
    except ImportError:
        return None


def cut_out(cell: Image.Image, silhouette: Image.Image) -> tuple[Image.Image, str]:
    """Return the cell as RGBA and a note saying how the alpha was made."""
    size = cell.size
    template = np.asarray(silhouette.convert("L").resize(size, Image.LANCZOS), dtype=np.float32) / 255.0
    slack = _dilate(template, max(1, int(size[0] * MASK_DILATE_FRACTION)))

    alpha = _rembg_alpha(cell.convert("RGB"))
    if alpha is not None:
        alpha = np.minimum(alpha, slack)
        how = "rembg intersected with the template silhouette"
    else:
        rgb = np.asarray(cell.convert("RGB"), dtype=np.float32) / 255.0
        not_white = (rgb.min(axis=2) < WHITE_CUTOFF).astype(np.float32)
        # keep the whole silhouette, and only drop white inside it where the
        # template says there is animal - white fur inside the outline stays
        alpha = np.where(template > 0.5, 1.0, slack * not_white)
        how = "template silhouette (rembg not installed)"

    alpha = _erode(alpha, max(1, int(size[0] * EDGE_ERODE_FRACTION)))
    rgba = np.dstack([np.asarray(cell.convert("RGB"), dtype=np.float32) / 255.0, alpha])
    return Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), mode="RGBA"), how


@dataclass(frozen=True)
class GeneratedViews:
    sheet_path: Path
    views: dict[str, Path]
    prompt: str
    cutout: str


async def generate_views(
    photo_path: str | Path,
    canonical_dir: str | Path,
    appearance: dict[str, Any],
    transport: ImageTransport,
    out_dir: str | Path,
    mismatches: list[str] | None = None,
    size: str = DEFAULT_SIZE,
) -> GeneratedViews:
    """One generated sheet, split into the four views, backgrounds removed."""
    canonical_dir, out_dir = Path(canonical_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_path = canonical_dir / "silhouette_grid.png"
    cameras = json.loads((canonical_dir / "cameras.json").read_text(encoding="utf-8"))
    target = tuple(cameras["resolution"])

    prompt = build_prompt(appearance, mismatches)
    references = [Path(photo_path).read_bytes(), grid_path.read_bytes()]
    print(f"[views] asking the image model for one {size} sheet (this takes a minute)...", flush=True)
    sheet_bytes = await transport.edit(prompt=prompt, references=references, size=size)
    print("[views] sheet received; cutting out the four views", flush=True)

    sheet = Image.open(io.BytesIO(sheet_bytes)).convert("RGB")
    sheet_path = out_dir / "sheet.png"
    if sheet_path.exists():
        # never overwrite a generation: the previous sheet is numbered and kept,
        # so a worse re-run can be undone by recutting from the older file
        index = 1
        while (out_dir / f"sheet_{index}.png").exists():
            index += 1
        sheet_path.replace(out_dir / f"sheet_{index}.png")
        print(f"[views] previous sheet kept as sheet_{index}.png", flush=True)
    sheet.save(sheet_path)
    (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    half_w, half_h = sheet.size[0] // 2, sheet.size[1] // 2
    views: dict[str, Path] = {}
    how = ""
    for row, names in enumerate(SHEET_ORDER):
        for col, name in enumerate(names):
            cell = sheet.crop((col * half_w, row * half_h, (col + 1) * half_w, (row + 1) * half_h))
            # generated at the model's size; brought up to the canonical
            # resolution so the projection step sees one pixel grid
            cell = cell.resize(target, Image.LANCZOS)
            silhouette = Image.open(canonical_dir / f"{name}_mask.png")
            rgba, how = cut_out(cell, silhouette)
            path = out_dir / f"{name}.png"
            rgba.save(path)
            views[name] = path
            print(f"[views] {name}: {how}", flush=True)

    manifest = {
        "sheet": sheet_path.name,
        "size_generated": list(sheet.size),
        "size_views": list(target),
        "cutout": how,
        "views": {name: path.name for name, path in views.items()},
    }
    (out_dir / "views.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return GeneratedViews(sheet_path=sheet_path, views=views, prompt=prompt, cutout=how)


def recut_views(canonical_dir: str | Path, out_dir: str | Path) -> dict[str, Path]:
    """Re-derive the four views from the sheet already on disk.

    The generated sheet is kept precisely so the cut-out can be redone - a
    different erosion, a better matte - without paying for another generation.
    """
    canonical_dir, out_dir = Path(canonical_dir), Path(out_dir)
    sheet = Image.open(out_dir / "sheet.png").convert("RGB")
    cameras = json.loads((canonical_dir / "cameras.json").read_text(encoding="utf-8"))
    target = tuple(cameras["resolution"])
    half_w, half_h = sheet.size[0] // 2, sheet.size[1] // 2
    views: dict[str, Path] = {}
    for row, names in enumerate(SHEET_ORDER):
        for col, name in enumerate(names):
            cell = sheet.crop((col * half_w, row * half_h, (col + 1) * half_w, (row + 1) * half_h))
            cell = cell.resize(target, Image.LANCZOS)
            rgba, how = cut_out(cell, Image.open(canonical_dir / f"{name}_mask.png"))
            path = out_dir / f"{name}.png"
            rgba.save(path)
            views[name] = path
            print(f"[views] {name}: recut, {how}", flush=True)
    return views


__all__ = [
    "GeneratedViews", "ImageTransport", "OpenAIImageTransport", "SHEET_ORDER",
    "appearance_to_text", "build_prompt", "cut_out", "generate_views", "recut_views",
]
