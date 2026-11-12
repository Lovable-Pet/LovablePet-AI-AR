"""Each dog's whisker colour, from the coat round the whisker roots.

The whisker strands of a split template (ar_pipeline/split_whiskers.py)
have their own material, and following the muzzle they rendered as orange
wires on a tan muzzle (shelter puppies 9 and 10). Real whiskers are dark
on almost every coat and white on a white or cream muzzle, so the colour
is one of two: WHISKER_LIGHT_RGB when the finished coat over the whisker
pads (the template's whisker_pad.png) is light and nearly grey, else
WHISKER_DARK_RGB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

WHISKER_DARK_RGB = (34, 30, 28)
WHISKER_LIGHT_RGB = (236, 233, 226)
# nearly as matte as the coat: at 0.45 the faceted tubes caught a
# highlight spiralling round them and read as twisted wire
WHISKER_ROUGHNESS = 0.8
# the pads count as light from this L* up, with chroma at most this
WHISKER_LIGHT_L = 65.0
WHISKER_LIGHT_MAX_CHROMA = 20.0


def whisker_colour(coat: Image.Image, template_dir: str | Path) -> tuple[tuple[int, int, int] | None, dict[str, Any]]:
    """The whisker colour for this finished coat, and what it was read
    from; (None, ...) for a template without whisker_pad.png."""
    pad_path = Path(template_dir) / "whisker_pad.png"
    if not pad_path.exists():
        return None, {"skipped": "template has no whisker_pad.png"}
    pad = np.asarray(Image.open(pad_path).convert("L").resize(coat.size, Image.NEAREST)) > 127
    if not pad.any():
        return None, {"skipped": "empty whisker pad"}
    from .fur import to_linear
    from .projection import linear_to_lab

    rgb = np.asarray(coat.convert("RGB"), dtype=np.float32)[pad]
    lab = linear_to_lab(np.clip(to_linear(rgb), 1e-4, 1.0))
    lightness = float(np.median(lab[:, 0]))
    chroma = float(np.hypot(np.median(lab[:, 1]), np.median(lab[:, 2])))
    light = lightness >= WHISKER_LIGHT_L and chroma <= WHISKER_LIGHT_MAX_CHROMA
    colour = WHISKER_LIGHT_RGB if light else WHISKER_DARK_RGB
    return colour, {"pad_L": round(lightness, 1), "pad_chroma": round(chroma, 1),
                    "colour": "light" if light else "dark"}
