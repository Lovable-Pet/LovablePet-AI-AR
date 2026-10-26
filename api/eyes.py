"""The eyes: the template's eyeball, in this dog's eye colour.

The eyeball is its own mesh with its own UV island: a white sclera, a
dark limbus ring, a textured iris and a black pupil, painted by the
template's artist. protect.png keeps that island out of every coat pass,
which is right for the shape and the highlight and wrong for the colour -
every dog was getting the same amber eyes.

The iris is found inside the eyeball island by what it is: the coloured
ring (chroma above the sclera's, lightness between the pupil's and the
sclera's). Those texels are moved to the target eye colour in LAB, keeping
the iris's own light-dark texture around the new colour, so a blue eye has
the same fibres as the brown one had. The pupil, limbus and sclera are left
alone.

PAINT MODE'S PROTECT
--------------------
The mask build needs protect.png to include the fur right around the eyes
(the lid bones drag template fur into the features mask, and the mask
build paints over eyes otherwise). A repainted atlas already has natural
fur around the eyes, so paint mode protects only the parts that are not
fur: the whole eyeball and mouth islands, the nose leather and the pads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

# Target eye colours as LAB (L, a, b), the iris's mid tone.
EYE_COLOURS: dict[str, tuple[float, float, float]] = {
    "dark_brown": (22.0, 12.0, 16.0),
    "brown": (36.0, 18.0, 28.0),
    "amber": (55.0, 22.0, 50.0),
    "hazel": (46.0, 6.0, 30.0),
    "blue": (55.0, -6.0, -28.0),
    "gray": (58.0, 0.0, -2.0),
}
EYE_COLOUR_NAMES = tuple(EYE_COLOURS)

IRIS_MIN_CHROMA = 10.0
IRIS_L_RANGE = (12.0, 80.0)
# A feature island this share protected is a whole protected island (an
# eyeball, the tongue, the mouth interior) rather than an eye socket.
WHOLE_SHARE = 0.9


def feature_islands(masks_dir: str | Path, size: tuple[int, int]):
    features = np.asarray(Image.open(Path(masks_dir) / "_features.png").convert("L").resize(size)) > 127
    labels, count = ndimage.label(features)
    return features, labels, count


def eyeball_island(masks_dir: str | Path, texture_linear: np.ndarray, protect: np.ndarray) -> np.ndarray | None:
    """The eyeball island: the largest whole-protected feature island that
    holds a saturated ring - the iris."""
    from .projection import linear_to_lab

    size = (texture_linear.shape[1], texture_linear.shape[0])
    _, labels, count = feature_islands(masks_dir, size)
    lab = linear_to_lab(np.clip(texture_linear, 1e-4, 1.0))
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    best, best_area = None, 0
    for index in range(1, count + 1):
        island = labels == index
        area = int(island.sum())
        if area < 500 or protect[island].mean() < WHOLE_SHARE:
            continue
        if float(np.percentile(chroma[island], 97)) < IRIS_MIN_CHROMA:
            continue
        if area > best_area:
            best, best_area = island, area
    return best


def recolour_iris(texture_linear: np.ndarray, masks_dir: str | Path, protect: np.ndarray,
                  eye_colour: str, keep_variation: float = 0.8) -> tuple[np.ndarray, dict]:
    """Move the iris texels of the template's eyeball to `eye_colour`."""
    from .projection import lab_to_linear, linear_to_lab

    if eye_colour not in EYE_COLOURS:
        return texture_linear, {"eye_colour": eye_colour, "applied": False, "why": "unknown colour"}
    island = eyeball_island(masks_dir, texture_linear, protect)
    if island is None:
        return texture_linear, {"eye_colour": eye_colour, "applied": False, "why": "no eyeball island found"}
    lab = linear_to_lab(np.clip(texture_linear, 1e-4, 1.0))
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    iris = island & (chroma >= IRIS_MIN_CHROMA) & (lab[..., 0] >= IRIS_L_RANGE[0]) & (lab[..., 0] <= IRIS_L_RANGE[1])
    iris = ndimage.binary_closing(iris, iterations=2) & island
    if iris.sum() < 50:
        return texture_linear, {"eye_colour": eye_colour, "applied": False, "why": "no iris texels"}
    target = np.asarray(EYE_COLOURS[eye_colour], dtype=np.float32)
    l_mid = float(np.median(lab[..., 0][iris]))
    c_mid = float(np.median(chroma[iris]))
    out_lab = lab.copy()
    # lightness: the iris's own variation around the new mid tone
    out_lab[..., 0][iris] = target[0] + (lab[..., 0][iris] - l_mid) * keep_variation
    # chroma: the target direction, scaled by how strong each texel's colour was
    scale = np.clip(chroma[iris] / max(c_mid, 1e-3), 0.4, 1.4)
    out_lab[..., 1][iris] = target[1] * scale
    out_lab[..., 2][iris] = target[2] * scale
    # soften the edge of the recolour by one texel
    weight = ndimage.gaussian_filter(iris.astype(np.float32), 1.0)[..., None]
    out = lab_to_linear(out_lab)
    result = texture_linear * (1 - weight) + out * weight
    return result.astype(np.float32), {"eye_colour": eye_colour, "applied": True, "iris_texels": int(iris.sum())}


def protect_for_paint(masks_dir: str | Path, protect: np.ndarray) -> np.ndarray:
    """The parts of protect.png that are not fur: whole feature islands
    (eyeballs, tongue, mouth) and protect texels outside the features
    mask (pads, nose leather)."""
    size = (protect.shape[1], protect.shape[0])
    features, labels, count = feature_islands(masks_dir, size)
    keep = protect & ~features
    for index in range(1, count + 1):
        island = labels == index
        if protect[island].mean() >= WHOLE_SHARE:
            keep |= island
    # the nose leather: the feature island that is nearly all dark and protected
    return keep


__all__ = ["EYE_COLOURS", "EYE_COLOUR_NAMES", "eyeball_island", "protect_for_paint", "recolour_iris"]
