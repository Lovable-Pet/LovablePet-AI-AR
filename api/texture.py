"""Stage 3-1 - compose a pet's coat texture from the stage-2 appearance.

HOW ACCURATE THIS IS MEANT TO BE
--------------------------------
The target is "which body part is what colour, and where on it", not a
likeness. A region is one colour, or one colour fading into another away
from a landmark; a marking lands beside the landmark it was described
against, at the size it was given. That is the level stage 2 delivers, so it
is the level this can honestly reach. Where the photograph itself can do
better - the face, seen frontally - face_photo.py hands in a layer and it is
composited here.

THE PASSES, IN ORDER
--------------------
  1. base coat       the brightness map through the base colour's ramp
  2. regions         each region's colour, or its gradient (landmarks.py
                     distance fields), observed regions before inferred
  2b. shapes         where generated views were read (views_prior.py):
                     inside regions with a marking or a partial coverage,
                     the views' colour labels say which of the coat's
                     colours sits on each texel, painted with our ramps
  3. markings        placed against a landmark when one was named, grown
                     out of a same-coloured neighbour otherwise
  4. feathering      the two boundaries a hard edge shows worst on:
                     cheeks|chest and skull|back, 8-10 px
  5. overlay         brindle stripes or merle mottling, when flagged
  6. face layer      the photograph's face, if one was warped in
  7. fur detail      hair-direction noise, +-5% brightness, and the
                     normal map it implies over the template's own
  8. protect         the template's eyes, nose, pads and mouth put back
  9. seams           the UV-seam post-process (seams.py)

WHY A GRADIENT MAP AND NOT A MULTIPLY
-------------------------------------
A multiply blend can only darken, so every coat came out below the colour it
was given and a black coat collapsed to a silhouette. The template's brightness
is mapped through a three-stop fur ramp instead, which can lighten as well as
darken and gives the undercoat and the guard hairs different colours rather
than one colour at two exposures. See api/fur.py.

WHY THE PATTERNS ARE SEEDED FROM petId
--------------------------------------
The noise shaping a partial region or a marking is drawn from a generator
seeded with the pet's 관리번호, so re-running gives that pet the same coat every
time. A pet whose texture changed between runs would look like a different
animal to whoever adopted it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

from .fur import (
    FUR_COLORS,
    anisotropic_noise,
    fur_multiplier,
    fur_normal_map,
    gradient_map,
    hex_to_linear,
    normalize_brightness,
    ramp_for,
    read_direction,
    to_linear,
    to_srgb,
)
from .patterns import color_family
from .regions import FACE_REGIONS as _FACE_REGIONS
from .regions import REGIONS

# The face is what a viewer reads first. Everything else is the body.
FACE_REGIONS = frozenset(_FACE_REGIONS)

# How much of a region a "partial" coverage paints.
PARTIAL_KEEP = 0.55

# How much of its region a marking covers, by reported size (no landmark).
MARKING_AREA = {"small": 0.15, "medium": 0.35, "large": 0.60}
# Radius of a landmark-placed marking, in eye distances.
MARKING_RADIUS_EYES = {"small": 0.45, "medium": 0.7, "large": 1.0}

# Width of the ramp from painted to unpainted, in units of the noise field.
EDGE_SOFTNESS = 0.08

# Marking types that say where they sit even without a landmark.
PLACEABLE_MARKINGS = frozenset({"socks", "blaze", "stripe", "mask"})
# A blaze is the strip up the middle of the face: half its width in eye
# distances. With geometry it is placed on the world midline (see
# _blaze_strip); without, _restrict takes the middle columns of the mask.
BLAZE_HALF_WIDTH_EYES = {"small": 0.13, "medium": 0.2, "large": 0.3}

# How far the proximity field reaches, in pixels of the 2048 map, when growing
# an unplaceable marking out of a neighbouring region of the same colour.
ADJACENCY_REACH = 80

# The boundaries feathered, and how wide.
FEATHER_PAIRS = (("cheeks", "chest"), ("skull", "back"))
FEATHER_PX = 9
# Simple mode feathers EVERY boundary between two differently coloured
# regions this wide, so the layout reads as a coat and not as a mask set.
# (Off by default now: a coat's colour regions have clear edges. What blurs
# them in a photograph is lighting, not the coat.)
FEATHER_ALL_PX = 14

# STYLE - "hard boundary, fur-level blending, a touch of illustration"
# ----------------------------------------------------------------------
# The colour regions keep their edges. Along each edge the mask is jittered
# by the hair noise a few texels either way, so the boundary interlocks at
# the scale of individual hairs instead of running as a geometric line.
EDGE_JITTER_PX = 1.5
# ... and the jittered edge is softened over this many texels, so it reads
# as hairs crossing the line rather than a torn edge.
EDGE_SOFT_PX = 1.0
# The template's own hair detail (the brightness map) is compressed toward
# flat by this factor: 1.0 is the template's full hair contrast, 0 is flat
# paint. Around half reads as fur that has been drawn rather than
# photographed.
HAIR_CONTRAST = 0.45
# Illustrated style: the brightness map is NOT taken from the template's
# painted coat (its painted edges survive the flattening as thin lines and
# its shading as grey blotches) but synthesised from the hair-direction
# noise: even fur everywhere, the same on every region.
SYNTHETIC_HAIR = True
SYNTHETIC_HAIR_SPREAD = 0.10
# A light belly on a dark coat reaches about this far up the flank, as a
# share of the height from the belly line to the top of the back. The
# underside mask runs higher than that (it holds the lower flank), so a
# white belly painted to the mask's edge reads as a cow. Above the line the
# region keeps the base coat, with the same fur-level edge as everywhere.
BELLY_REACH = 0.22
# ... and reaches down only this far below the belly centre (share of the
# same height), which keeps it off the legs.
BELLY_BELOW = 0.12
# ... and only on surfaces facing down at least this much (world -z).
BELLY_FACING = 0.25
# The rule above was written for the bone-seeded underside mask, which ran up
# the lower flank and down the inside of every leg. The geometric mask
# (refine_masks.py) is the belly alone, so the rule now only cuts a straight
# line across it - the horizontal edge on the flank and the white block in
# the armpit that the side render showed. Off; the mask's own edge is used.
BELLY_REACH_RULE = False
# The chest mask on this template runs round the shoulders and leaves a
# piece on the hip. A chest colour is painted only where the surface faces
# forward (world -y): the breast. The rest of the mask keeps the base coat.
CHEST_FACING = 0.45

# The inside of a leg. A region entry for legs_lower may carry
# "inner_color": the colour of the leg's surfaces that face the body's
# midline or forward - on a dog with a light chest and paws (the shiba's
# urajiro, most tan-points) the inside and front of every leg is that
# light colour and only the outer face keeps the points colour. The ten
# regions cannot split a leg's inside from its outside, so the split is by
# the surface normal: inward = the normal's lateral component toward x=0,
# forward = -y as for the chest. Measured on the shiba template 62% of
# legs_lower faces inward (> 0.2) or forward (> CHEST_FACING).
LEG_INNER_FACING = 0.2

# The protect mask's edge is softened over this many texels.
PROTECT_FEATHER_PX = 3

# Where a gradient's fade begins and ends, as percentiles of the distance
# field inside the region; so a gradient always spans the region.
GRADIENT_LOW, GRADIENT_HIGH = 10.0, 90.0
# ...except from these anchors, where the fade is done well before the far
# side. On the cheek the points colour is a patch under and beside the eye,
# not half the cheek: spanning the region put the fade's midpoint at the
# cheek's median distance, so half the cheek came out tan-ish (b* > 25 on
# 49% of the texels) on a dog whose cheek is white with a patch under the
# eye, and the user read the cheek as tan. Inside the ear the warm colour
# sits at the base and the cup above it is pale - but the fade spans the
# whole cup (the default reach). It used to finish at the 45th percentile,
# which by construction painted the far half of whatever the rim rule left
# in solid cream: on the jindo template's small cup (5k texels, 40% kept
# after the rim) that was the entire centre of the ear, a hard-edged
# near-white patch inside a black ear on every dark dog. Measured on the
# black-and-tan shiba's photo the cup is L 57..78, between tan and cream,
# which the midpoint of a full-span fade gives.
GRADIENT_REACH = {"eye": (5.0, 35.0)}
# How far a fade is allowed to go toward gradient_to, per anchor. The ear
# fade's target is cream because the cup is paler than the points colour;
# but cream's top stop is near-white (#F6F0E4), and the far end of the cup
# rendered as a white patch inside a black ear - no dark dog has that. The
# measured cup tops out at L 78, roughly 60% of the way from tan to cream.
GRADIENT_CAP = {"ear_base": 0.6}

# Overlays.
BRINDLE_PERIOD_PX = 36
BRINDLE_COVER = 0.4
MERLE_COVER = 0.35

LUMA = (0.2126, 0.7152, 0.0722)

# How far the blur reaches when separating the template's painted coat from its
# fur. Wide enough to swallow a marking, narrow enough to leave the hairs.
FLATTEN_RADIUS = 48
FLATTEN_FLOOR = 0.80
FLATTEN_CEILING = 1.25
RESIDUAL_RADIUS = 24

NOISE_OCTAVES = 4
NOISE_BASE_PERIOD = 6


def seed_for(pet_id: str) -> int:
    """A stable 64-bit seed for this pet, independent of platform hash salting."""
    return int.from_bytes(hashlib.sha256(pet_id.encode("utf-8")).digest()[:8], "big")


# --------------------------------------------------------------------------
# the brightness map
# --------------------------------------------------------------------------

def brightness_map(image: Image.Image) -> np.ndarray:
    """Strip the template's own coat from its texture, keeping only the fur.

    Dividing by a heavily blurred copy removes everything at the scale of a
    marking and keeps what is smaller: individual hairs. A second, tighter
    high pass removes the plateaus the clamp leaves. Flattening is correct
    here rather than a compromise, because the material carries its own
    normal and occlusion maps: shading is applied at render time from those,
    so the base colour should be flat albedo.
    """
    linear = to_linear(np.asarray(image, dtype=np.float32))
    blurred = to_linear(
        np.asarray(image.filter(ImageFilter.GaussianBlur(radius=FLATTEN_RADIUS)),
                   dtype=np.float32)
    )
    ratio = linear / np.maximum(blurred, 1e-4)
    gray = ratio @ np.asarray(LUMA, dtype=np.float32)
    gray = np.clip(gray, FLATTEN_FLOOR, FLATTEN_CEILING)

    residual = Image.fromarray((np.clip(gray, 0, 1) * 255).astype(np.uint8), mode="L")
    residual = np.asarray(
        residual.filter(ImageFilter.GaussianBlur(radius=RESIDUAL_RADIUS)), dtype=np.float32
    ) / 255.0
    gray = gray / np.maximum(residual, 1e-3) * float(gray.mean())

    return normalize_brightness(gray).astype(np.float32)


# --------------------------------------------------------------------------
# noise, for partial coverage and markings
# --------------------------------------------------------------------------

def _perlin(shape: tuple[int, int], period: tuple[int, int], rng) -> np.ndarray:
    height, width = shape
    rows, cols = period
    angles = 2 * np.pi * rng.random((rows + 1, cols + 1))
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    ys = np.linspace(0, rows, height, endpoint=False)
    xs = np.linspace(0, cols, width, endpoint=False)
    y0 = ys.astype(int)[:, None]
    x0 = xs.astype(int)[None, :]
    fy = (ys - ys.astype(int))[:, None]
    fx = (xs - xs.astype(int))[None, :]

    def corner(dy: int, dx: int) -> np.ndarray:
        gradient = gradients[y0 + dy, x0 + dx]
        return gradient[..., 0] * (fy - dy) + gradient[..., 1] * (fx - dx)

    u = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
    v = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
    top = corner(0, 0) * (1 - u) + corner(0, 1) * u
    bottom = corner(1, 0) * (1 - u) + corner(1, 1) * u
    return top * (1 - v) + bottom * v


def noise_field(shape: tuple[int, int], seed: int) -> np.ndarray:
    """Fractal noise in 0..1, the shape source for patches and markings."""
    rng = np.random.default_rng(seed)
    total = np.zeros(shape, dtype=np.float32)
    amplitude, weight = 1.0, 0.0
    for octave in range(NOISE_OCTAVES):
        period = NOISE_BASE_PERIOD * 2 ** octave
        total += amplitude * _perlin(shape, (period, period), rng)
        weight += amplitude
        amplitude *= 0.5
    total /= weight
    spread = float(total.max() - total.min())
    return ((total - total.min()) / max(spread, 1e-6)).astype(np.float32)


def _select(mask: np.ndarray, field: np.ndarray, keep: float) -> np.ndarray:
    """Cut a soft-edged patch covering roughly `keep` of the mask."""
    inside = mask > 0.5
    if not inside.any():
        return np.zeros_like(mask)
    threshold = float(np.quantile(field[inside], 1.0 - float(np.clip(keep, 0.01, 0.99))))
    soft = np.clip((field - threshold) / EDGE_SOFTNESS + 0.5, 0.0, 1.0)
    return (mask * soft).astype(np.float32)


def _proximity(neighbour: np.ndarray, reach: int = ADJACENCY_REACH) -> np.ndarray:
    image = Image.fromarray((np.clip(neighbour, 0, 1) * 255).astype(np.uint8), mode="L")
    blurred = image.filter(ImageFilter.GaussianBlur(radius=reach))
    return np.asarray(blurred, dtype=np.float32) / 255.0


def _restrict(mask: np.ndarray, marking_type: str, min_pixels: int = 0) -> np.ndarray:
    """Narrow a region mask to where a marking of this type sits, on
    percentiles of the mask's own pixels rather than its bounding box."""
    rows, cols = np.nonzero(mask > 0.5)
    if rows.size == 0:
        return mask

    if marking_type == "socks":
        keep = rows >= np.percentile(rows, 55)
    elif marking_type == "blaze":
        low, high = np.percentile(cols, [32, 68])
        keep = (cols >= low) & (cols <= high)
    elif marking_type == "stripe":
        low, high = np.percentile(rows, [35, 65])
        keep = (rows >= low) & (rows <= high)
    else:
        return mask

    limited = np.zeros_like(mask)
    limited[rows[keep], cols[keep]] = mask[rows[keep], cols[keep]]
    if int((limited > 0.5).sum()) < min_pixels:
        return mask
    return limited


def _blaze_strip(landmarks: Any, mask: np.ndarray, size: str, noise: np.ndarray) -> np.ndarray | None:
    """The blaze as a band along the face's midline in WORLD space, from
    the nose up between the eyes to the ear bases. The skull mask holds the
    ears' islands too, so the mask's own column percentiles (_restrict) are
    not the middle of the face; the geometry's x = 0 is."""
    ears = landmarks.anchor_points("ear_base")
    if not ears:
        return None
    y_back = float(np.mean([landmarks.world(name)[1] for name in ears]))
    half = BLAZE_HALF_WIDTH_EYES.get(size, BLAZE_HALF_WIDTH_EYES["medium"]) * landmarks.eye_distance
    strip = landmarks.midline_strip(mask, half, y_max=y_back)
    if strip is None or int((strip > 0.5).sum()) < 30:
        return None
    # a natural outline: solid inside, the edge broken by the noise
    return np.clip((strip * (0.7 + 0.6 * noise) - 0.15) * 2.0, 0.0, 1.0) * (mask > 0.5)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _blur(array: np.ndarray, sigma: float) -> np.ndarray:
    if array.ndim == 2:
        return ndimage.gaussian_filter(array, sigma)
    return np.stack([ndimage.gaussian_filter(array[..., i], sigma) for i in range(array.shape[-1])], -1)


# --------------------------------------------------------------------------
# painting
# --------------------------------------------------------------------------

def paint_region(
    gray: np.ndarray,
    color: str,
    region: np.ndarray | None = None,
    palette: dict[str, tuple[str, str, str]] | None = None,
) -> np.ndarray:
    """Fur of one colour: the brightness map through that colour's ramp."""
    key = (color or "").strip().lower()
    if palette and key in palette:
        dark, mid, light = (hex_to_linear(stop) for stop in palette[key])
    else:
        dark, mid, light = ramp_for(key)
    painted = gradient_map(gray, dark, mid, light, region)
    # a touch of illustration: the ramp's spread compressed toward its mid stop
    return (mid + (painted - mid) * HAIR_CONTRAST).astype(np.float32)


def _blend(composed: np.ndarray, painted: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weight = mask[..., None]
    return composed * (1 - weight) + painted * weight


# The ear's rim. Stage 1 tags the whole concave face of each ear shell as
# ears_inner (seed_regions.ear_shell_split: a normals test, so the front face
# is the inner face out to its very edge). On the dog the cup's lighter hair
# stops short of the edge and a band of head colour runs round the rim; the
# tan-point shiba's photo shows a black rim about a fifth of the ear's width
# on each side, and painting the whole front tan made the ears read as
# orange leaves. So the inner ear is pulled in from its boundary with the
# other regions by this many texels and the band goes to skull (the head
# colour, which already carries the ear's back). Measured to the nearest
# texel of another REGION, not to the island edge, so a UV seam through the
# cup does not put a dark line down its middle. On the shiba template (two
# ~13.8k-texel cups, ~54 px half-width) 14 px keeps 60% of the cup.
EAR_RIM_PX = 14
# The rim's inner edge is softened over this share of the rim's width (at
# least 1 px). A 1 px step left the kept cup as a hard-edged, jagged
# island of pale colour inside the dark ear; on the jindo (rim ~8.6 px)
# this is a 4 px feather, on the shiba ~7 px.
EAR_RIM_RAMP_SHARE = 0.5
EAR_RIM_RAMP_PX = 1
# ... on a cup of this many texels. Another template's cup is a different
# size on its atlas (the jindo's is 4.9k texels, half the width), and the
# same 14 px ate most of it, leaving a blob in the middle of the ear; the
# rim scales with the cup's width (the square root of its texel count).
EAR_RIM_REF_TEXELS = 13800


class MaskLibrary:
    """The region masks for one template, loaded once and kept as float arrays.

    With `jitter` set (a zero-mean noise field), every mask's edge is pushed
    in and out by EDGE_JITTER_PX texels of that noise - the fur-level
    interlock between two colour regions - while the region itself stays
    put. The same field jitters every mask, so two neighbouring regions'
    edges move together and leave no gap.

    ears_inner comes back pulled in by EAR_RIM_PX and skull gains the band
    that was taken (see EAR_RIM_PX); every consumer of the masks - the
    build, the refinement's per-region restore, the projection - sees the
    same shapes."""

    def __init__(self, masks_dir: str | Path, size: tuple[int, int],
                 jitter: np.ndarray | None = None) -> None:
        self.dir = Path(masks_dir)
        self.size = size
        self.missing: list[str] = []
        self._cache: dict[str, np.ndarray] = {}
        self._raw: dict[str, np.ndarray | None] = {}
        self.jitter = jitter
        self.valid: np.ndarray | None = None
        self.ear_rim_texels = 0

    def _raw_mask(self, region: str) -> np.ndarray | None:
        """The mask as stage 1 baked it, resized, before the rim rule and
        the jitter."""
        if region not in self._raw:
            path = self.dir / f"{region}.png"
            if not path.exists():
                self._raw[region] = None
            else:
                image = Image.open(path).convert("L")
                if image.size != self.size:
                    image = image.resize(self.size, Image.LANCZOS)
                self._raw[region] = np.asarray(image, dtype=np.float32) / 255.0
        return self._raw[region]

    def _ear_rim(self) -> np.ndarray | None:
        """The band of ears_inner within EAR_RIM_PX of another region, as a
        soft 0..1 weight; None when there is no inner-ear mask."""
        if "ear_rim" in self._cache:
            return self._cache["ear_rim"]
        inner = self._raw_mask("ears_inner")
        if inner is None or EAR_RIM_PX <= 0:
            self._cache["ear_rim"] = None
            return None
        # a texel is "another region's" where that region's mask outweighs
        # the inner ear's own: the baked masks overlap by a few texels at
        # every edge (skull reaches 0.76 inside ears_inner), and a plain
        # > 0.5 test on the neighbours started the rim from inside the cup
        other = np.zeros(inner.shape, dtype=bool)
        for name in REGIONS:
            if name == "ears_inner":
                continue
            mask = self._raw_mask(name)
            if mask is not None:
                other |= (mask > 0.5) & (mask > inner)
        distance = ndimage.distance_transform_edt(~other)
        labels, count = ndimage.label(inner > 0.5)
        cup = int(np.bincount(labels.ravel())[1:].max()) if count else EAR_RIM_REF_TEXELS
        rim_px = EAR_RIM_PX * float(np.sqrt(cup / EAR_RIM_REF_TEXELS))
        ramp_px = max(EAR_RIM_RAMP_PX, rim_px * EAR_RIM_RAMP_SHARE)
        inside = np.clip((distance - rim_px) / max(ramp_px, 1e-6), 0.0, 1.0).astype(np.float32)
        rim = inner * (1.0 - inside)
        self.ear_rim_texels = int((rim > 0.5).sum())
        self._cache["ear_rim"] = rim
        return rim

    def get(self, region: str) -> np.ndarray | None:
        if region not in REGIONS:
            self.missing.append(f"{region} (not in the region enum)")
            return None
        if region not in self._cache:
            mask = self._raw_mask(region)
            if mask is None:
                self.missing.append(f"{region} (no {region}.png)")
                return None
            rim = self._ear_rim() if region in ("ears_inner", "skull") else None
            if rim is not None:
                mask = np.clip(mask - rim, 0.0, 1.0) if region == "ears_inner" else np.maximum(mask, rim)
            if self.jitter is not None:
                mask = _jitter_edge(mask, self.jitter)
            self._cache[region] = mask
        return self._cache[region]

    def extra(self, name: str) -> np.ndarray | None:
        """A mask that is not a coat region - the features mask, protect -
        loaded the same way but without the enum check. Looked for in the
        masks folder and then beside it (the template folder)."""
        for path in (self.dir / f"{name}.png", self.dir.parent / f"{name}.png"):
            if path.exists():
                image = Image.open(path).convert("L")
                if image.size != self.size:
                    image = image.resize(self.size, Image.LANCZOS)
                return np.asarray(image, dtype=np.float32) / 255.0
        return None

    def union(self, regions) -> np.ndarray:
        total = np.zeros((self.size[1], self.size[0]), dtype=np.float32)
        for region in regions:
            mask = self.get(region)
            if mask is not None:
                total = np.maximum(total, mask)
        return total


def _belly_reach(landmarks, mask: np.ndarray, jitter: np.ndarray | None,
                 back_mask: np.ndarray | None = None) -> np.ndarray:
    """1 where the underside is low enough to be belly, 0 up the flank, with
    a fur-level edge. The line is BELLY_REACH of the way from the belly
    centre's height to the height of the back (the back mask's median, so
    the ears and the head do not count as "top")."""
    z = landmarks.position[..., 2]
    inside = mask > 0.5
    if not inside.any():
        return np.ones_like(mask)
    belly_z = float(landmarks.world("belly_center")[2]) if "belly_center" in landmarks.points else float(np.percentile(z[inside], 10))
    if back_mask is not None and (back_mask > 0.5).any():
        top_z = float(np.median(z[(back_mask > 0.5) & (landmarks.valid if landmarks.valid is not None else True)]))
    else:
        top_z = float(np.percentile(z[inside], 95))
    line = belly_z + BELLY_REACH * (top_z - belly_z)
    texel = max(landmarks.texel_metres(), 1e-4)
    signed = (line - z) / texel            # texels below the line, positive
    # ... and above the legs: the underside mask runs down the inside and
    # back of every leg, which on a standard coat stay the base colour
    floor = belly_z - BELLY_BELOW * (top_z - belly_z)
    below = (z - floor) / texel
    signed = np.minimum(signed, below)
    # ... and facing down: the flank beside the armpit and the hip sits at
    # belly height but faces sideways, and stays the base coat
    if landmarks.normal is not None:
        down = (-landmarks.normal[..., 2] - BELLY_FACING) / 0.08
        signed = np.minimum(signed, down)
    if jitter is not None:
        signed = signed + EDGE_JITTER_PX * jitter
    return np.clip(signed / max(EDGE_SOFT_PX, 1e-3) + 0.5, 0.0, 1.0).astype(np.float32)


def _leg_inner(landmarks, jitter: np.ndarray | None) -> np.ndarray:
    """1 where the surface faces the body's midline or forward, 0 where it
    faces outward or back, with a fur-level edge; the inside of a leg."""
    normal = landmarks.normal
    side = np.sign(landmarks.position[..., 0]) if landmarks.position is not None else 1.0
    inward = -side * normal[..., 0]
    forward = -normal[..., 1]
    signed = np.maximum((inward - LEG_INNER_FACING) / 0.08, (forward - CHEST_FACING) / 0.08)
    if jitter is not None:
        signed = signed + 0.6 * jitter
    return np.clip(signed + 0.5, 0.0, 1.0).astype(np.float32)


def _chest_front(landmarks, jitter: np.ndarray | None) -> np.ndarray:
    """1 where the surface faces forward, 0 where it faces sideways or back,
    with a fur-level edge; the part of the chest mask that is a chest."""
    forward = -landmarks.normal[..., 1]          # -y is forward on this template
    signed = (forward - CHEST_FACING) / 0.08     # ~ one texel per 0.08 of facing
    if jitter is not None:
        signed = signed + 0.6 * jitter
    facing = np.clip(signed + 0.5, 0.0, 1.0)
    # ... and in the front of the animal: the thigh faces forward too, and
    # the chest mask holds a piece of it
    if landmarks.position is not None and "chest_center" in landmarks.points and "belly_center" in landmarks.points:
        y = landmarks.position[..., 1]
        chest_y = float(landmarks.world("chest_center")[1])
        belly_y = float(landmarks.world("belly_center")[1])
        limit = chest_y + 0.5 * (belly_y - chest_y)
        texel = max(landmarks.texel_metres(), 1e-4)
        front = np.clip((limit - y) / texel + 0.5 + (0.6 * jitter if jitter is not None else 0.0), 0.0, 1.0)
        facing = facing * front
        # ... but the throat, which is part of the chest region, faces down
        # rather than forward: above the breastbone the whole region counts
        z = landmarks.position[..., 2]
        chest_z = float(landmarks.world("chest_center")[2])
        throat = np.clip((z - chest_z) / texel + 0.5, 0.0, 1.0)
        facing = np.maximum(facing, throat)
    return facing.astype(np.float32)


# A doberman-type chest: dark, with two small points-coloured spots on the
# breastbone either side of the midline (body_colours "dark_tan_spots"). Centres in
# eye distances from chest_center: out to each side, and down.
CHEST_SPOT_SIDE_EYES = 0.55
CHEST_SPOT_DOWN_EYES = 0.5
CHEST_SPOT_RADIUS_EYES = 0.55
# Points-coloured cheeks on a dark head (a doberman's, a rottweiler's) are
# the sides of the muzzle and the jaw, not the whole cheek up beside the eye
# to the ear the mask holds (it was cut for a shiba's cream cheek). Such a
# cheek stops this many eye distances below the eyes.
CHEEK_POINT_BELOW_EYE = 0.35


def _chest_spots(landmarks, mask: np.ndarray, noise: np.ndarray) -> np.ndarray | None:
    """Two soft spots on the front of the chest mask, broken by the noise."""
    if landmarks.position is None or landmarks.normal is None or "chest_center" not in landmarks.points:
        return None
    inside = (mask > 0.5) & (-landmarks.normal[..., 1] > 0.3)
    if not inside.any():
        return None
    rows, cols = np.nonzero(inside)
    pos = landmarks.position[rows, cols]
    centre = landmarks.world("chest_center")
    eye = landmarks.eye_distance
    out = np.zeros(mask.shape, dtype=np.float32)
    for side in (-1.0, 1.0):
        tx, tz = centre[0] + side * CHEST_SPOT_SIDE_EYES * eye, centre[2] - CHEST_SPOT_DOWN_EYES * eye
        nearest = int(np.argmin((pos[:, 0] - tx) ** 2 + (pos[:, 2] - tz) ** 2))
        disc = landmarks.blob(pos[nearest], mask, CHEST_SPOT_RADIUS_EYES * eye)
        out = np.maximum(out, np.clip((disc * (0.55 + 0.9 * noise) - 0.25) * 2.5, 0.0, 1.0))
    out *= mask > 0.5
    return out if float((out > 0.5).sum()) >= 30 else None


def _below_eyes(landmarks, jitter: np.ndarray | None) -> np.ndarray:
    """1 below the line CHEEK_POINT_BELOW_EYE under the eyes, 0 above, with a
    fur-level edge."""
    eye_z = float(np.mean([landmarks.world(name)[2] for name in landmarks.anchor_points("eye")]))
    limit = eye_z - CHEEK_POINT_BELOW_EYE * landmarks.eye_distance
    texel = max(landmarks.texel_metres(), 1e-4)
    signed = (limit - landmarks.position[..., 2]) / texel + 0.5
    if jitter is not None:
        signed = signed + 3.0 * jitter
    return np.clip(signed, 0.0, 1.0).astype(np.float32)


def _jitter_edge(mask: np.ndarray, jitter: np.ndarray, px: float = EDGE_JITTER_PX) -> np.ndarray:
    """Push a mask's edge in and out by the noise: the signed distance to the
    edge is offset by `px * jitter` and re-thresholded with a one-texel
    soft edge. Interior and exterior far from the edge are untouched."""
    inside = mask > 0.5
    if not inside.any() or inside.all():
        return mask
    distance = ndimage.distance_transform_edt(inside) - ndimage.distance_transform_edt(~inside)
    offset = distance + px * jitter
    return np.clip(offset / max(EDGE_SOFT_PX, 1e-3) + 0.5, 0.0, 1.0).astype(np.float32)


def _blur_inside(array: np.ndarray, sigma: float, valid: np.ndarray) -> np.ndarray:
    """A blur that never reads outside `valid` (the UV islands), so a
    boundary near an island edge does not pick up the padding's colour."""
    weight = ndimage.gaussian_filter(valid.astype(np.float32), sigma)
    if array.ndim == 2:
        return ndimage.gaussian_filter(array * valid, sigma) / np.maximum(weight, 1e-6)
    out = np.stack([ndimage.gaussian_filter(array[..., i] * valid, sigma) for i in range(array.shape[-1])], -1)
    return out / np.maximum(weight, 1e-6)[..., None]


def _save_debug(directory: Path, name: str, array: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if array.ndim == 2:
        data = (np.clip(array, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(data, mode="L").save(directory / f"{name}.png")
    else:
        data = (to_srgb(array) * 255).astype(np.uint8)
        Image.fromarray(data, mode="RGB").save(directory / f"{name}.png")


def _gradient_weight(landmarks, entry: dict[str, Any], mask: np.ndarray) -> np.ndarray | None:
    """0 where `color` is strongest (at the anchor), 1 where it has faded
    fully into gradient_to; None when the entry has no usable gradient."""
    to = str(entry.get("gradient_to") or "none").lower()
    anchor = str(entry.get("gradient_anchor") or "none").lower()
    if landmarks is None or to == "none" or anchor == "none" or to not in FUR_COLORS:
        return None
    distance = landmarks.distance_field(anchor, mask)
    if distance is None:
        return None
    inside = (mask > 0.5) & np.isfinite(distance)
    if inside.sum() < 50:
        return None
    reach = entry.get("gradient_reach") or GRADIENT_REACH.get(anchor, (GRADIENT_LOW, GRADIENT_HIGH))
    low, high = np.percentile(distance[inside], reach)
    if high - low < 1e-6:
        return None
    t = np.where(inside, (np.nan_to_num(distance, nan=high) - low) / (high - low), 1.0)
    cap = float(entry.get("gradient_cap") or GRADIENT_CAP.get(anchor, 1.0))
    return (_smoothstep(t) * cap).astype(np.float32)


def compose_body(
    gray: np.ndarray,
    masks: MaskLibrary,
    appearance: dict[str, Any],
    noise: np.ndarray,
    palette: dict[str, tuple[str, str, str]] | None = None,
    debug: Path | None = None,
    landmarks=None,
    report: dict[str, Any] | None = None,
    label_layer: tuple[np.ndarray, np.ndarray, list[str]] | None = None,
    feather_all: bool = False,
) -> np.ndarray:
    """Passes 1-4: base, regions (with gradients), markings, feathering. Linear RGB."""
    base_color = appearance.get("base_color") or "fawn"
    report = report if report is not None else {}

    entries = list(appearance.get("region_colors") or [])
    inferred_names = set(appearance.get("inferred_regions") or [])
    observed = [e for e in entries if e.get("region") not in inferred_names
                and not str(e.get("source", "")).startswith(("pattern", "llm_completion"))]
    inferred = [e for e in entries if e not in observed]

    claimed = np.zeros_like(gray)
    for entry in entries:
        mask = masks.get(entry.get("region"))
        if mask is not None:
            claimed = np.maximum(claimed, mask)
    unclaimed = 1.0 - claimed

    composed = paint_region(
        gray, base_color,
        unclaimed if float((unclaimed > 0.5).mean()) > 0.05 else None,
        palette,
    )
    if debug:
        _save_debug(debug, "00_brightness", gray)
        _save_debug(debug, f"01_base_{base_color}", composed)

    adjacency: dict[str, np.ndarray] = {}

    def remember(entry: dict[str, Any], region_mask: np.ndarray) -> None:
        family = color_family(entry.get("color", ""))
        adjacency[family] = (
            region_mask if family not in adjacency
            else np.maximum(adjacency[family], region_mask)
        )

    gradients_drawn: list[str] = []
    order = 0
    for entry in observed + inferred:
        mask = masks.get(entry.get("region"))
        if mask is None:
            continue
        order += 1
        region, color_name = entry.get("region"), entry.get("color", "")

        keep = PARTIAL_KEEP if entry.get("coverage") == "partial" else 1.0
        keep *= float(np.clip(entry.get("weight", 1.0), 0.0, 1.0))

        painted = paint_region(gray, color_name, mask, palette)
        fade = _gradient_weight(landmarks, entry, mask)
        if fade is not None:
            painted = _blend(painted, paint_region(gray, entry["gradient_to"], mask, palette), fade)
            gradients_drawn.append(f"{region}:{color_name}->{entry['gradient_to']}@{entry['gradient_anchor']}")
            # a gradient already says where the colour thins out; a partial
            # coverage on top of it would cut holes in the fade
            keep = max(keep, 1.0 if fade is not None else keep)

        if keep >= 1.0:
            applied = mask
        else:
            neighbour = adjacency.get(color_family(color_name))
            field = _proximity(neighbour) if neighbour is not None else noise
            applied = _select(mask, field, keep)

        if (region == "chest" and landmarks is not None and landmarks.normal is not None
                and color_name != base_color):
            applied = applied * _chest_front(landmarks, masks.jitter)
            if debug:
                _save_debug(debug, f"02_{order:02d}_chest_front", applied)
        if (region == "cheeks" and landmarks is not None and landmarks.position is not None
                and color_family(color_name) == "point"
                and color_family(next((str(e.get("color", "")) for e in entries if e.get("region") == "skull"),
                                      base_color)) == "dark"):
            applied = applied * _below_eyes(landmarks, masks.jitter)
            if debug:
                _save_debug(debug, f"02_{order:02d}_cheeks_below_eyes", applied)
        if (BELLY_REACH_RULE and region == "underside" and landmarks is not None
                and landmarks.position is not None
                and color_family(color_name) == "light" and color_family(base_color) != "light"):
            applied = applied * _belly_reach(landmarks, mask, masks.jitter, masks.get("back"))
            if debug:
                _save_debug(debug, f"02_{order:02d}_underside_reach", applied)

        composed = _blend(composed, painted, applied)
        inner_color = str(entry.get("inner_color") or "").lower()
        if (region == "legs_lower" and inner_color in FUR_COLORS and inner_color != color_name
                and landmarks is not None and landmarks.normal is not None):
            inner = applied * _leg_inner(landmarks, masks.jitter)
            composed = _blend(composed, paint_region(gray, inner_color, mask, palette), inner)
            gradients_drawn.append(f"legs_lower inner:{inner_color}")
            if debug:
                _save_debug(debug, f"02_{order:02d}_legs_inner", inner)
        remember(entry, mask)
        if debug:
            tag = "inferred" if entry in inferred else "seen"
            _save_debug(debug, f"02_{order:02d}_{region}_{color_name}_{tag}_mask", applied)
            _save_debug(debug, f"02_{order:02d}_{region}_{color_name}_{tag}", composed)
    # the band round each eye (ar_pipeline/open_face_masks.py): a
    # light-cheeked dog's cheek colour runs up round the eyes to the brows -
    # a black-and-tan's cream spectacles (eval demo e08). A piebald keeps its
    # dark cap over the eyes.
    spectacle = masks.extra("_spectacle")
    if spectacle is not None:
        colours = {str(e.get("region")): str(e.get("color", "")) for e in entries}
        cheek, skull_colour = colours.get("cheeks", ""), colours.get("skull", base_color)
        # ... under a DARK head only: on a tan head (urajiro, sable) the cream
        # stops below the eyes, and spectacles whitened e09's face
        if (cheek in FUR_COLORS and color_family(cheek) == "light" and cheek != skull_colour
                and color_family(skull_colour) == "dark"
                and str(appearance.get("pattern", "")) != "piebald"):
            composed = _blend(composed, paint_region(gray, cheek, spectacle, palette), np.clip(spectacle, 0, 1))
            gradients_drawn.append(f"spectacle:{cheek}")
    report["gradients"] = gradients_drawn

    # the masks do not tile the surface exactly, and a jittered edge opens
    # thin gaps between neighbours where the base coat shows as a line; every
    # unclaimed texel inside the body takes the colour of the nearest claimed one
    claimed_now = np.zeros_like(gray)
    for entry in entries:
        mask = masks.get(entry.get("region"))
        if mask is not None:
            claimed_now = np.maximum(claimed_now, mask)
    body_valid = masks.valid if masks.valid is not None else claimed_now > 0
    gap = (claimed_now < 0.5) & body_valid
    if gap.any() and (claimed_now >= 0.5).any():
        indices = ndimage.distance_transform_edt(~(claimed_now >= 0.5), return_distances=False, return_indices=True)
        composed = np.where(gap[..., None], composed[indices[0], indices[1]], composed)
        report["gap_texels_filled"] = int(gap.sum())

    # 2b. the views' shapes, only where a single colour is known to be too coarse
    shaped: list[str] = []
    if label_layer is not None:
        label_map, label_weight, colours = label_layer
        marked = {m.get("region") for m in appearance.get("markings") or []}
        for entry in entries:
            region = entry.get("region")
            if region not in marked and entry.get("coverage") != "partial":
                continue
            mask = masks.get(region)
            if mask is None:
                continue
            own = str(entry.get("color", "")).lower()
            inside = (mask > 0.5) & (label_map >= 0)
            changed = 0
            for index, name in enumerate(colours):
                if name == own or name not in FUR_COLORS:
                    continue
                where = inside & (label_map == index)
                if where.sum() < 40:
                    continue
                soft = _blur(where.astype(np.float32), 2.0) * label_weight * mask
                composed = _blend(composed, paint_region(gray, name, mask, palette), np.clip(soft, 0, 1))
                changed += int(where.sum())
            if changed:
                shaped.append(f"{region}:{changed}px")
                if debug:
                    _save_debug(debug, f"02b_shapes_{region}", composed)
    report["shapes_applied"] = shaped
    shaped_regions = {s.split(":")[0] for s in shaped}

    placed: list[str] = []
    for order, marking in enumerate(appearance.get("markings") or [], start=1):
        mask = masks.get(marking.get("region"))
        if mask is None:
            continue
        region = marking.get("region")
        marking_type = str(marking.get("type", "other"))
        size = str(marking.get("size", "medium"))
        area = MARKING_AREA.get(size, MARKING_AREA["medium"])
        anchor = str(marking.get("anchor") or "none").lower()

        blob = None
        if anchor != "none" and landmarks is not None:
            target = landmarks.target_point(anchor, str(marking.get("offset") or "on"),
                                            str(marking.get("distance") or "near"))
            if target is not None:
                radius = MARKING_RADIUS_EYES.get(size, 0.7) * landmarks.eye_distance
                disc = landmarks.blob(target, mask, radius)
                # a natural outline: the disc broken by the noise, clipped to the mask
                blob = np.clip((disc * (0.55 + 0.9 * noise) - 0.25) * 2.5, 0.0, 1.0) * (mask > 0.5)
                if float((blob > 0.5).sum()) < 30:
                    blob = None
                else:
                    placed.append(f"{marking_type}:{region}@{anchor}/{marking.get('offset')}/{marking.get('distance')}")
        if blob is None and marking_type == "chest_spots" and landmarks is not None:
            blob = _chest_spots(landmarks, mask, noise)
            if blob is not None:
                placed.append(f"chest_spots:{region}@chest_center")
            else:
                continue    # no geometry: a dark chest is closer than a random patch
        if blob is None and marking_type == "patches":
            # piebald spots: large blobs anywhere on the region. The shape is
            # a random field smoothed over the dog's SURFACE: from the UV
            # noise the patches broke at every seam and came out with torn,
            # jagged edges (user on eval demo e13)
            field = noise
            if landmarks is not None and landmarks.position is not None:
                rng = np.random.default_rng(PATCH_SEED + order)
                surface = landmarks.valid if landmarks.valid is not None else mask > -1
                smooth = smooth_on_surface(rng.random(mask.shape).astype(np.float32), landmarks.position,
                                           surface, PATCH_SIGMA_M)
                inside = (mask > 0.5) & surface
                if inside.any():
                    spread = float(smooth[inside].std()) or 1.0
                    field = 0.5 + (smooth - float(smooth[inside].mean())) / spread * PATCH_FIELD_SPREAD
            blob = _select(mask, field, area)
            placed.append(f"patches:{region}")
        if blob is None and marking_type == "blaze" and landmarks is not None:
            blob = _blaze_strip(landmarks, mask, size, noise)
            if blob is not None:
                placed.append(f"blaze:{region}@midline")
        if blob is None and region in shaped_regions:
            # the views' labels already drew this region's second colour
            continue
        if blob is None:
            if marking_type in PLACEABLE_MARKINGS:
                needed = int(area * int((mask > 0.5).sum()))
                blob = _select(_restrict(mask, marking_type, needed), noise, area)
            else:
                neighbour = adjacency.get(color_family(marking.get("color", "")))
                field = _proximity(neighbour) if neighbour is not None else noise
                blob = _select(mask, field, area)

        composed = _blend(
            composed, paint_region(gray, marking.get("color", ""), mask, palette), blob
        )
        if debug:
            _save_debug(debug, f"03_{order:02d}_{marking_type}_{region}_mask", blob)
            _save_debug(debug, f"03_{order:02d}_{marking_type}_{region}", composed)
    report["markings_placed"] = placed

    if feather_all:
        # every boundary between two regions of different colour: a colour
        # index per texel from the masks, then the band where it changes
        index = np.full(gray.shape, -1, dtype=np.int16)
        names: list[str] = []
        for entry in entries:
            mask = masks.get(entry.get("region"))
            if mask is None:
                continue
            colour = str(entry.get("color", "")).lower()
            if colour not in names:
                names.append(colour)
            index[mask > 0.5] = names.index(colour)
        reach = FEATHER_ALL_PX // 2
        boundary = np.zeros(gray.shape, dtype=bool)
        for shift in (1, reach):
            boundary |= (index != np.roll(index, shift, axis=0)) & (index >= 0) & (np.roll(index, shift, axis=0) >= 0)
            boundary |= (index != np.roll(index, shift, axis=1)) & (index >= 0) & (np.roll(index, shift, axis=1) >= 0)
        band = ndimage.binary_dilation(boundary, iterations=reach)
        valid_here = index >= 0
        soft = _blur(band.astype(np.float32), FEATHER_ALL_PX / 3.0) * valid_here
        composed = _blend(composed, _blur_inside(composed, FEATHER_ALL_PX / 2.0, valid_here), np.clip(soft * 1.5, 0, 1))
        report["feather_all_texels"] = int(band.sum())
        if debug:
            _save_debug(debug, "04_feather_all", np.clip(soft * 1.5, 0, 1))

    # feathering at the two boundaries a hard edge shows worst on
    for a, b in FEATHER_PAIRS:
        mask_a, mask_b = masks.get(a), masks.get(b)
        if mask_a is None or mask_b is None:
            continue
        reach = FEATHER_PX // 2 + 1
        band = ndimage.binary_dilation(mask_a > 0.5, iterations=reach) & \
            ndimage.binary_dilation(mask_b > 0.5, iterations=reach)
        if not band.any():
            continue
        valid_here = (mask_a > 0.5) | (mask_b > 0.5)
        soft = _blur(band.astype(np.float32), FEATHER_PX / 3.0) * valid_here
        composed = _blend(composed, _blur_inside(composed, FEATHER_PX / 2.0, valid_here), np.clip(soft * 1.5, 0, 1))
        if debug:
            _save_debug(debug, f"04_feather_{a}_{b}", np.clip(soft * 1.5, 0, 1))

    return composed


# Patches (piebald body spots) come from a random field smoothed over the
# surface with this sigma: blobs about 10-15 cm across on the jindo. The
# field is rescaled so EDGE_SOFTNESS gives a fur-soft, not a blurred, edge.
PATCH_SIGMA_M = 0.035
PATCH_FIELD_SPREAD = 0.25
PATCH_SEED = 4242


# Sable: the second colour laid over the top of the dog as dark-tipped hair.
# How much of it per texel = how high up the dog it sits (the larger of: how
# much the surface faces up, full at SABLE_UP_FULL; and its height between
# the belly line and the back's top, from SABLE_HEIGHT_FROM of the way up,
# full at the top) x the region's share below x a streak field along the
# hair, so the tipping reads as hairs and no edge exists anywhere. Facing
# alone left the flanks bare: most of the back region is the trunk's sides.
SABLE_UP_FULL = 0.7
SABLE_HEIGHT_FROM = 0.15
# the 95th percentile is the neck's crest (z 0.62+ on the jindo), which left
# the back itself at 0.6 weight - a thin brown line on the spine
SABLE_HEIGHT_TOP_PCT = 75
# 0.6 left the real sable (eval e09) with no dark back at all; the dog the
# user read as wrong (e11) is a red dog with a dark mask, not a lighter
# sable - that is the classifier's call (pattern_classify SABLE rule)
SABLE_REGION_SHARE = {"back": 1.0, "skull": 0.4}
SABLE_STREAK_PX = 24
# 0.35 came back from the refine as leopard blotches on the neck (eval e05,
# e11, e12); the tipping reads as a shade, the hairs come from the fur pass.
# 0 since the weight is smoothed on the surface: the streak is UV noise and
# does not continue across a seam
SABLE_STREAK_MIX = 0.0
SABLE_BLUR_PX = 8.0
# The weight is smoothed over the dog's SURFACE (3D, gaussian sigma in
# metres), not the UV atlas. In UV the region shares (back 1.0, skull 0.4),
# the per-island smoothed normals and the mask edges all break at seams and
# region borders: a pale stripe along the spine seam, a pale ring at the tail
# root and blotches where the skull meets the neck (user on eval v4 e09,
# e11: "lines on the back, the head join and the tail"). Two texels either
# side of a seam are one point on the surface, so they get one value.
SABLE_SURFACE_SIGMA_M = 0.025
# surface samples: every Nth texel in each direction
SURFACE_STRIDE = 5


def smooth_on_surface(field: np.ndarray, position: np.ndarray, valid: np.ndarray,
                      sigma_m: float, stride: int = SURFACE_STRIDE) -> np.ndarray:
    """`field` (h, w) averaged over the surface with a gaussian of `sigma_m`
    metres, through the position map: continuous across UV seams and region
    borders by construction. Texels outside `valid` come back 0."""
    from scipy.spatial import cKDTree

    height, width = field.shape
    rows, cols = np.nonzero(valid[::stride, ::stride])
    rows, cols = rows * stride, cols * stride
    points = position[rows, cols].astype(np.float64)
    values = field[rows, cols].astype(np.float64)
    tree = cKDTree(points)
    neighbours = tree.query_ball_point(points, r=2.5 * sigma_m, workers=-1)
    smoothed = np.empty(len(points))
    for index, near in enumerate(neighbours):
        near = np.asarray(near)
        d2 = np.sum((points[near] - points[index]) ** 2, axis=1)
        w = np.exp(-d2 / (2.0 * sigma_m * sigma_m))
        smoothed[index] = float(np.dot(w, values[near]) / w.sum())
    # every texel from its nearest samples on the surface
    out = np.zeros((height, width), dtype=np.float32)
    all_rows, all_cols = np.nonzero(valid)
    distance, nearest = tree.query(position[all_rows, all_cols].astype(np.float64), k=4, workers=-1)
    w = 1.0 / np.maximum(distance, 1e-5)
    out[all_rows, all_cols] = (np.sum(w * smoothed[nearest], axis=1) / w.sum(axis=1)).astype(np.float32)
    return out


def _sable_weight(gray: np.ndarray, masks: "MaskLibrary", landmarks, direction: np.ndarray | None,
                  valid: np.ndarray, seed: int, extra: dict[str, float] | None = None) -> np.ndarray | None:
    if landmarks is None or landmarks.normal is None:
        return None
    up = np.clip(landmarks.normal[..., 2] / SABLE_UP_FULL, 0.0, 1.0)
    # regions tipped whichever way they face (the muzzle top): full facing
    for region in (extra or {}):
        mask = masks.get(region)
        if mask is not None:
            up = np.maximum(up, mask)
    back = masks.get("back")
    if landmarks.position is not None and back is not None and "belly_center" in landmarks.points:
        z = landmarks.position[..., 2]
        inside = back > 0.5
        if landmarks.valid is not None:
            inside &= landmarks.valid
        bottom = float(landmarks.world("belly_center")[2])
        top = float(np.percentile(z[inside], SABLE_HEIGHT_TOP_PCT)) if inside.any() else bottom + 0.3
        start = bottom + SABLE_HEIGHT_FROM * (top - bottom)
        height = np.clip((z - start) / max(top - start, 1e-6), 0.0, 1.0)
        up = np.maximum(up, height * back)
    share = np.zeros(gray.shape, dtype=np.float32)
    for region, amount in {**SABLE_REGION_SHARE, **(extra or {})}.items():
        mask = masks.get(region)
        if mask is not None:
            share = np.maximum(share, mask * amount)
    if SABLE_STREAK_MIX <= 0.0:
        streak = np.ones(gray.shape, dtype=np.float32)
    elif direction is not None:
        streak = anisotropic_noise(gray.shape, direction, seed + 13, streak=SABLE_STREAK_PX, grain=2.0)
        streak = (streak - streak.min()) / max(float(streak.max() - streak.min()), 1e-6)
    else:
        streak = noise_field(gray.shape, seed + 13)
    weight = up * share * ((1.0 - SABLE_STREAK_MIX) + SABLE_STREAK_MIX * streak)
    if landmarks.position is not None:
        # every texel with geometry, not only those with a fur direction: the
        # tail root has 56k texels without one, and they stayed a bare tan
        # patch on the dark tail (eval v5 e09)
        surface = landmarks.valid if landmarks.valid is not None else valid
        return np.clip(smooth_on_surface(weight, landmarks.position, surface, SABLE_SURFACE_SIGMA_M), 0.0, 1.0) * surface
    return np.clip(_blur_inside(weight, SABLE_BLUR_PX, valid), 0.0, 1.0) * valid


def apply_overlay(composed: np.ndarray, gray: np.ndarray, appearance: dict[str, Any],
                  direction: np.ndarray | None, valid: np.ndarray, seed: int,
                  palette=None, masks: "MaskLibrary | None" = None,
                  landmarks=None) -> tuple[np.ndarray, str | None]:
    """Pass 5: brindle stripes across the fur, merle mottling, or sable tipping."""
    overlay = str(appearance.get("overlay") or "").lower()
    if overlay == "sable":
        second = appearance.get("second_color") or "black"
        if second not in FUR_COLORS or second == "none":
            second = "black"
        weight = (_sable_weight(gray, masks, landmarks, direction, valid, seed, appearance.get("sable_extra"))
                  if masks is not None else None)
        if weight is None:
            return composed, None
        weight = weight * float(appearance.get("sable_strength") or 1.0)
        return _blend(composed, paint_region(gray, second, None, palette), weight), overlay
    if overlay not in ("brindle", "merle"):
        return composed, None
    second = appearance.get("second_color") or ("black" if overlay == "brindle" else "chocolate")
    if second not in FUR_COLORS:
        second = "black"
    shape = gray.shape
    if overlay == "brindle" and direction is not None:
        # stripes run across the hair: smear noise along the perpendicular
        across = np.stack([-direction[..., 1], direction[..., 0]], axis=-1)
        stripes = anisotropic_noise(shape, across, seed + 7, streak=BRINDLE_PERIOD_PX, grain=BRINDLE_PERIOD_PX / 4)
        field = (stripes - stripes.min()) / max(float(stripes.max() - stripes.min()), 1e-6)
        cover = BRINDLE_COVER
    else:
        field = noise_field(shape, seed + 11)
        cover = MERLE_COVER
    threshold = float(np.quantile(field[valid], 1.0 - cover))
    weight = np.clip((field - threshold) / 0.06 + 0.5, 0.0, 1.0) * valid
    return _blend(composed, paint_region(gray, second, None, palette), weight), overlay


@dataclass
class TextureResult:
    image: Image.Image
    normal: Image.Image | None
    report: dict[str, Any] = field(default_factory=dict)


def compose_texture(
    base_texture_path: str | Path,
    masks_dir: str | Path,
    appearance: dict[str, Any],
    debug_dir: str | Path | None = None,
    palette: dict[str, tuple[str, str, str]] | None = None,
    template_dir: str | Path | None = None,
    geometry_dir: str | Path | None = None,
    mesh_glb: str | Path | None = None,
    face_layer: tuple[np.ndarray, np.ndarray] | None = None,
    base_normal: Image.Image | None = None,
    fur_detail: bool = True,
    seam_smoothing: bool = True,
    face_triangle_ratio: float | None = None,
    label_layer: tuple[np.ndarray, np.ndarray, list[str]] | None = None,
    feather_all: bool = False,
) -> TextureResult:
    """All nine passes. `face_layer` is (linear rgb, weight) from
    face_photo.py; `base_normal` the template's own normal map."""
    debug = Path(debug_dir) if debug_dir else None
    report: dict[str, Any] = {}

    base_image = Image.open(base_texture_path).convert("RGB")
    template_dir = Path(template_dir) if template_dir else Path(masks_dir).parent
    if geometry_dir is None and (template_dir / "geometry" / "geometry.json").exists():
        geometry_dir = template_dir / "geometry"

    gray = brightness_map(base_image)
    pet_id = str(appearance.get("pet_id") or appearance.get("petId") or "")
    seed = seed_for(pet_id)
    noise = noise_field(gray.shape, seed)

    # the fur-level edge jitter: hair noise along the fur direction when the
    # template has one, isotropic otherwise
    direction0 = None
    if (template_dir / "fur_direction.png").exists():
        direction0, _ = read_direction(template_dir / "fur_direction.png")
        if direction0.shape[:2] != gray.shape:
            direction0 = None
    if direction0 is not None:
        jitter = anisotropic_noise(gray.shape, direction0, seed + 5, streak=4, grain=1.0)
    else:
        jitter = (noise_field(gray.shape, seed + 5) - 0.5) * 4.0

    if SYNTHETIC_HAIR:
        # even, drawn fur: hair streaks of one amplitude everywhere, no
        # painted edges, no shading blotches
        hair0 = (anisotropic_noise(gray.shape, direction0, seed + 9, streak=8, grain=1.2)
                 if direction0 is not None else jitter)
        gray = normalize_brightness(
            np.clip(1.0 + SYNTHETIC_HAIR_SPREAD * np.clip(hair0, -2.5, 2.5) / 2.5, 0.0, 2.0)
        ).astype(np.float32)
    else:
        gray = (float(gray.mean()) + (gray - float(gray.mean())) * HAIR_CONTRAST).astype(np.float32)
    masks = MaskLibrary(masks_dir, base_image.size, jitter=np.clip(jitter, -2.5, 2.5).astype(np.float32))
    if geometry_dir is None and Path("output/geometry/geometry.json").exists():
        geometry_dir = Path("output/geometry")
    if geometry_dir is not None and (Path(geometry_dir) / "valid.npy").exists():
        masks.valid = np.load(Path(geometry_dir) / "valid.npy") > 0.5

    landmarks = None
    if geometry_dir is None and Path("output/geometry/geometry.json").exists():
        geometry_dir = Path("output/geometry")
    if (template_dir / "landmarks.json").exists():
        from .landmarks import Landmarks
        landmarks = Landmarks.load(template_dir, geometry_dir)
        if landmarks.position is None:
            print("[texture] landmarks.json found but no geometry; gradients and placed markings off")
            landmarks = None
    else:
        print("[texture] no landmarks.json; gradients and placed markings off")

    direction = defined = None
    if (template_dir / "fur_direction.png").exists():
        direction, defined = read_direction(template_dir / "fur_direction.png")
        if direction.shape[:2] != gray.shape:
            direction = defined = None
    valid = defined if defined is not None else np.ones(gray.shape, dtype=bool)

    # 1-4
    composed = compose_body(gray, masks, appearance, noise, palette, debug, landmarks, report, label_layer,
                            feather_all)

    # 5
    composed, overlay = apply_overlay(composed, gray, appearance, direction, valid, seed, palette,
                                      masks=masks, landmarks=landmarks)
    report["overlay"] = overlay

    # 6 - and the check that the photo actually landed: in the ring around
    # the protected eyes and nose the composite with the face must still look
    # like the composite without it, or the warp put a photo eye on fur
    protect = masks.extra("protect")
    report["face_layer"] = None
    if face_layer is not None:
        from .face_photo import dark_intrusion, face_gate, ring_ssim

        rgb, weight = face_layer
        weight = np.clip(weight, 0, 1)
        with_face = _blend(composed, rgb, weight)
        template_linear = to_linear(np.asarray(base_image, dtype=np.float32))
        score = intrusion = None
        if protect is not None:
            score = ring_ssim(with_face, composed, protect, weight > 0.05, template_linear)
            intrusion = dark_intrusion(with_face, protect, weight)
        accepted, why = face_gate(score, intrusion, face_triangle_ratio)
        report["face_layer"] = {
            "share": round(float((weight > 0.5).mean()), 4),
            "ring_ssim": None if score is None else round(score, 3),
            "dark_intrusion": None if intrusion is None else round(intrusion, 4),
            "triangle_ratio": face_triangle_ratio,
            "accepted": accepted,
            "why": why,
        }
        if accepted:
            composed = with_face
            print(f"[texture] face layer accepted: {why}")
        else:
            print(f"[texture] face layer rejected: {why}; keeping the mask build")
        if debug:
            _save_debug(debug, "05a_face_layer_weight", weight)
            _save_debug(debug, "05b_with_face", with_face)
    if debug:
        _save_debug(debug, "05_before_fur", composed)

    # 7
    normal_image = None
    if fur_detail and direction is not None:
        hair = anisotropic_noise(gray.shape, direction, seed + 3)
        composed = composed * fur_multiplier(hair)[..., None]
        base_normal_array = (np.asarray(base_normal.convert("RGB"), dtype=np.float32) / 255.0
                             if base_normal is not None else None)
        if base_normal_array is not None and base_normal_array.shape[:2] != gray.shape:
            base_normal_array = None
        normal = fur_normal_map(hair * (defined if defined is not None else 1.0), base_normal=base_normal_array)
        normal_image = Image.fromarray((normal * 255).astype(np.uint8), mode="RGB")
        report["fur_detail"] = True
    else:
        report["fur_detail"] = False

    # 8
    template = to_linear(np.asarray(base_image, dtype=np.float32))
    if protect is not None:
        weight = _blur(protect, PROTECT_FEATHER_PX / 2.0)
        weight = np.maximum(weight, protect)
        composed = _blend(composed, template, np.clip(weight, 0, 1))
        report["protect_texels"] = int((protect > 0.5).sum())
    else:
        composed = keep_features(composed, base_image, masks)
        report["protect_texels"] = None

    # 9
    if seam_smoothing and mesh_glb is not None and Path(mesh_glb).exists():
        from .mesh import TemplateMesh
        from .seams import smooth_seams
        mesh = TemplateMesh.load(mesh_glb, base_image.size)
        composed, seam_report = smooth_seams(np.clip(composed, 0, 1), mesh)
        report["seams"] = seam_report
    else:
        report["seams"] = None

    if debug:
        _save_debug(debug, "06_final", composed)
        if normal_image is not None:
            normal_image.save(debug / "07_normal.png")

    if masks.missing:
        print(f"[texture] skipped {len(masks.missing)} entries with no mask: {masks.missing}")
    print(f"[texture] gradients {report.get('gradients')}, placed markings {report.get('markings_placed')}, "
          f"overlay {overlay}, fur {report['fur_detail']}, seams {'on' if report['seams'] else 'off'}")

    image = Image.fromarray((to_srgb(np.clip(composed, 0, 1)) * 255).astype(np.uint8), mode="RGB")
    return TextureResult(image, normal_image, report)


# --------------------------------------------------------------------------
# legacy helpers
# --------------------------------------------------------------------------

FUR_TEXTURE_SD = 0.015
FUR_TEXTURE_PX = 3
FUR_MIN_LUMA = 0.15
FEATURE_CORE_GROW_PX = 3


def feature_core(features: np.ndarray, base_image: Image.Image) -> np.ndarray:
    """The features mask without the template fur the lid bones dragged in:
    fur is told by its texture (local luminance spread), the eyes and nose
    are flat or dark."""
    from scipy.ndimage import uniform_filter

    template = to_linear(np.asarray(base_image, dtype=np.float32))
    luma = template @ np.asarray(LUMA, dtype=np.float32)
    size = 2 * FUR_TEXTURE_PX + 1
    mean = uniform_filter(luma, size, mode="nearest")
    spread = np.sqrt(np.maximum(uniform_filter(luma * luma, size, mode="nearest") - mean * mean, 0.0))
    fur = (luma > FUR_MIN_LUMA) & (spread > FUR_TEXTURE_SD)
    core = (features > 0.5) & ~fur
    image = Image.fromarray((core * 255).astype(np.uint8), mode="L")
    image = image.filter(ImageFilter.MaxFilter(2 * FEATURE_CORE_GROW_PX + 1))
    image = image.filter(ImageFilter.GaussianBlur(radius=1.0))
    grown = np.asarray(image, dtype=np.float32) / 255.0
    return np.minimum(grown, features)


def keep_features(final: np.ndarray, base_image: Image.Image, masks: MaskLibrary) -> np.ndarray:
    """Put the template's own eyes, nose and mouth back, from the _features
    mask - the fallback when a template has no protect.png."""
    features = masks.extra("_features")
    if features is None:
        return final
    template = to_linear(np.asarray(base_image, dtype=np.float32))
    weight = feature_core(features, base_image)[..., None]
    return final * (1.0 - weight) + template * weight


def synthesize_texture(
    base_texture_path: str | Path,
    masks_dir: str | Path,
    appearance: dict[str, Any],
    projected_texture_path: str | Path | None = None,
    debug_dir: str | Path | None = None,
    palette: dict[str, tuple[str, str, str]] | None = None,
    **kwargs: Any,
) -> Image.Image:
    """The coat texture as an image: compose_texture without the normal map
    and the report, for callers that only want the picture.

    `projected_texture_path` is the experimental path's projected face: when
    given, its face regions are composited as the face layer."""
    face_layer = None
    if projected_texture_path is not None:
        base_image = Image.open(base_texture_path).convert("RGB")
        masks = MaskLibrary(masks_dir, base_image.size)
        projected = Image.open(projected_texture_path).convert("RGB")
        if projected.size != base_image.size:
            projected = projected.resize(base_image.size, Image.LANCZOS)
        weight = _blur(masks.union(FACE_REGIONS), 5.0)
        face_layer = (to_linear(np.asarray(projected, dtype=np.float32)), weight)
    return compose_texture(
        base_texture_path, masks_dir, appearance, debug_dir=debug_dir, palette=palette,
        face_layer=face_layer, **kwargs,
    ).image


__all__ = [
    "FACE_REGIONS", "FUR_COLORS", "MaskLibrary", "TextureResult", "brightness_map",
    "compose_body", "compose_texture", "feature_core", "keep_features", "noise_field",
    "paint_region", "seed_for", "synthesize_texture",
]
