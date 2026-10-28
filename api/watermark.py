"""Shelter logos stamped over the photograph, painted out before the gate.

The Korea Animal Rescue & Management Association (한국동물구조관리협회, 50 of
1,000 dogs in the public shelter feed on 2026-10-28) puts its logo on every
photograph it publishes, always in the same place: bottom centre of a
1600x1200 frame. The logo is navy (92 of 100 photographs) or white. When it
lies over the dog's paws or chest, it counts as text on the animal and the
gate refuses the photo (prepare_photo.REJECT_RULES): 2 of the 3 such dogs in
a 44-dog sample. The cut would also keep it as part of the coat.

The glyphs are known exactly. They were taken from those 100 photographs:
every logo pixel reads the same in all of them, and every other pixel varies.
So the logo is found by a plain comparison, with no model call. Its glyphs
are filled from their surroundings before the gate and the cut see the
photograph. Only that exact frame size is checked; a resized copy is left
alone and gated as before. What lay under the logo is then a smudge, so a
note tells the reading prompts not to read colours from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

DATA = Path(__file__).resolve().parent / "data" / "watermarks"


@dataclass(frozen=True)
class Watermark:
    name: str
    files: tuple[str, ...]      # RGBA in DATA, one per colouring; alpha = the glyphs (the same in each)
    frame: tuple[int, int]      # the photograph size the logo was cut from
    origin: tuple[int, int]     # where the templates sit in that frame (left, top)


WATERMARKS = (
    Watermark("karma", ("karma_navy.png", "karma_white.png"), (1600, 1200), (520, 950)),
)
# Median |difference| over the glyphs (0..255, mean of RGB) that counts as the logo. Its own 100 photographs
# read at most 1.7; 90 photographs from other shelters, resized to the frame, read 42 or more.
MATCH_DIFF = 8.0
# Pixels around the glyphs filled as well: the anti-aliased edges and the JPEG ringing around them. At 4 the
# ringing left a coloured fringe on photo 34; 6 leaves none.
FILL_GROW = 6
# Context around the logo's box that the fill reads from.
FILL_PAD = 12

WATERMARK_NOTE = ("A shelter logo near the bottom of the photograph was painted out; the smudged patch it left "
                  "is not coat - read no colours from it.")


@lru_cache(maxsize=None)
def _templates(mark: Watermark) -> tuple[np.ndarray, list[np.ndarray]]:
    """(glyph mask, one float RGB template per colouring)."""
    images = [np.asarray(Image.open(DATA / name).convert("RGBA")) for name in mark.files]
    glyph = images[0][..., 3] > 127
    return glyph, [image[..., :3].astype(np.float32) for image in images]


def find_watermarks(image: Image.Image) -> list[tuple[Watermark, float]]:
    """The known logos on this photograph, with how far each matched."""
    rgb = None
    found = []
    for mark in WATERMARKS:
        if image.size != mark.frame:
            continue
        if rgb is None:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        glyph, colourings = _templates(mark)
        left, top = mark.origin
        region = rgb[top:top + glyph.shape[0], left:left + glyph.shape[1]][glyph]
        diff = min(float(np.median(np.abs(region - colours[glyph]).mean(axis=1))) for colours in colourings)
        if diff <= MATCH_DIFF:
            found.append((mark, diff))
    return found


def remove_watermarks(image: Image.Image) -> tuple[Image.Image, list[str]]:
    """The photograph with every known logo filled in from around it, and
    the names of the logos removed (the photograph unchanged when none)."""
    found = find_watermarks(image)
    if not found:
        return image, []
    from scipy import ndimage
    from skimage.restoration import inpaint_biharmonic

    rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    for mark, _ in found:
        glyph, _ = _templates(mark)
        left, top = mark.origin
        y0, x0 = max(top - FILL_PAD, 0), max(left - FILL_PAD, 0)
        y1 = min(top + glyph.shape[0] + FILL_PAD, rgb.shape[0])
        x1 = min(left + glyph.shape[1] + FILL_PAD, rgb.shape[1])
        hole = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        hole[top - y0:top - y0 + glyph.shape[0], left - x0:left - x0 + glyph.shape[1]] = glyph
        hole = ndimage.binary_dilation(hole, iterations=FILL_GROW)
        rgb[y0:y1, x0:x1] = inpaint_biharmonic(rgb[y0:y1, x0:x1], hole, channel_axis=-1)
    cleaned = Image.fromarray(np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="RGB")
    return cleaned, [mark.name for mark, _ in found]


__all__ = ["WATERMARK_NOTE", "WATERMARKS", "Watermark", "find_watermarks", "remove_watermarks"]
