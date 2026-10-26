"""The template's atlas with its colour taken out: shape and fur grain only.

The template GLB was painted as a red shiba - tan back, cream chest, tan
inside the ears, cream pips over the eyes. The template is meant to lend
its SHAPE (mesh, rig, UV layout, the hand-painted fur strokes), not that
colour layout, yet the repaint step handed the painted atlas to the image
model as its first reference, and a colour the model was not told to change
stayed (a beige blot inside a white dog's ears is the tan of the template's
ear).

This module makes the reference the image model should see instead: every
fur texel becomes a neutral grey that keeps only the fine, hair-scale
luminance detail of the original strokes. Anything broader than a hair -
the tan-versus-cream layout, the shading painted into ear cups and leg
folds - is removed by subtracting a blurred copy of the luminance
(a high-pass), so nothing about where the template's colours were can
leak into the result. The eyes, nose leather, mouth and paw pads are not
fur and are kept as painted; the padding between islands keeps the
template's near-white, because that contrast is what tells the model where
one piece ends.

The colour guide sent alongside is built on this neutral atlas too, so the
three references the model sees (atlas, photograph, guide) contain no
colour that did not come from the photograph or the reading.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .fur import to_linear, to_srgb
from .projection import lab_to_linear, linear_to_lab

# Luminance of the neutral fur (LAB L, 0-100). Mid-light so the model can
# go both darker and lighter from it.
BASE_L = 72.0
# Detail broader than this (texels of blur sigma, on the 2048 atlas) is
# treated as colour layout or painted shading and removed; finer detail is
# fur grain and kept.
GRAIN_SIGMA = 10.0
# The grain's contrast, relative to the original strokes.
GRAIN_GAIN = 1.0
# The grain never pushes the neutral further than this from BASE_L.
GRAIN_CLIP = 18.0
# Texels just outside a protected feature keep the template's own colour,
# fading to neutral over this many texels. Handing the image model a bare
# grey field around the eyes and nose cost it the alignment: the feature
# SSIM of a repaint fell from about 0.9 to 0.57 once the surroundings went
# flat. This ring is the context it needs to place them, and it is inside
# the band that finish_painted restores from the template anyway, so it
# cannot leak the template's coat colour into the result.
FEATURE_CONTEXT_PX = 8.0
# The tone of the space between the islands. The template's padding is a
# near-white L 93 against fur at L 60, so every piece reads as a shape. The
# first neutral atlas filled the padding with the fur's own grey, and with
# no boundary to see the image model drew its own: that repaint came back
# with the padding at L 72.6 and the fur at L 83.6, a DARKER rim around
# every island, which renders as a dark seam line along each piece edge.
# Painted from the template the same model left the padding at L 94.3 and
# the fur at L 88.4, no line. A single constant carries the boundary and
# nothing about where the template's colours were.
PADDING_L = 93.0


def neutral_atlas(base: Image.Image, valid: np.ndarray, protect: np.ndarray | None,
                  base_l: float = BASE_L, sigma: float = GRAIN_SIGMA, gain: float = GRAIN_GAIN,
                  context_px: float = FEATURE_CONTEXT_PX) -> Image.Image:
    """The template atlas as neutral grey fur with its hair-scale grain.
    valid: where the mesh is on the atlas; protect: non-fur texels kept as
    painted (eyeballs, nose, mouth, pads), with a short feathered ring of
    the original around them."""
    rgb = to_linear(np.asarray(base.convert("RGB").resize(valid.shape[::-1] if valid.shape != base.size[::-1]
                                                          else base.size), dtype=np.float32))
    lab = linear_to_lab(rgb)
    lum = lab[..., 0]
    fur = valid.astype(np.float32)
    if protect is not None:
        fur = fur * (1.0 - np.clip(protect, 0.0, 1.0))
    # normalised blur: islands do not bleed into the padding or each other
    blurred = ndimage.gaussian_filter(lum * fur, sigma) / np.maximum(ndimage.gaussian_filter(fur, sigma), 1e-3)
    grain = np.clip((lum - blurred) * gain, -GRAIN_CLIP, GRAIN_CLIP)
    neutral = np.zeros_like(lab)
    neutral[..., 0] = base_l + grain
    out = np.clip(lab_to_linear(neutral), 0.0, 1.0)
    # 1 on protected features, falling to 0 over context_px of fur around them
    original = np.clip(protect, 0.0, 1.0) if protect is not None else np.zeros_like(fur)
    if context_px > 0 and original.max() > 0:
        reach = ndimage.gaussian_filter(original, context_px / 2.0)
        original = np.clip(np.maximum(original, reach / max(float(reach.max()), 1e-6)), 0.0, 1.0)
    original = (original[..., None] * (valid > 0.5)[..., None]).astype(np.float32)
    out = out * (1.0 - original) + rgb * original        # features, and a ring, stay painted
    # the padding is the template's near-white, so an island reads as a shape
    # against it and the model has no reason to outline one
    padding = ~(valid > 0.5)
    out[padding] = np.clip(lab_to_linear(np.array([[PADDING_L, 0.0, 0.0]], dtype=np.float32)), 0, 1)[0]
    return Image.fromarray((to_srgb(out) * 255).astype(np.uint8), mode="RGB")


def neutral_atlas_for(base: Image.Image, valid: np.ndarray, masks_dir: str | Path) -> Image.Image:
    """neutral_atlas with the template's protect mask read from masks_dir."""
    protect_path = Path(masks_dir).parent / "protect.png"
    protect = None
    if protect_path.exists():
        protect = np.asarray(Image.open(protect_path).convert("L").resize(base.size), dtype=np.float32) / 255.0
    return neutral_atlas(base, valid, protect)


__all__ = ["neutral_atlas", "neutral_atlas_for", "BASE_L", "GRAIN_SIGMA", "PADDING_L"]
