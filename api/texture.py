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
  1. base coat       the base colour times the template's own shading
                     (TEMPLATE_SHADING), or the brightness map through
                     the colour's ramp
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

The generated templates changed that: their base colour holds all their
shading (no normal or occlusion map ships with them), and a coat that
dropped it read as paint next to the template. TEMPLATE_SHADING multiplies
each region's mid stop by the template's brightness over its coat's median -
1 at the median, so it lightens lit fur as well as darkening the hollows.

WHY THE PATTERNS ARE SEEDED FROM petId
--------------------------------------
The noise shaping a partial region or a marking is drawn from a generator
seeded with the pet's 관리번호, so re-running gives that pet the same coat every
time. A pet whose texture changed between runs would look like a different
animal to whoever adopted it.
"""

from __future__ import annotations

import hashlib
import json
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
# compose_body: below this summed claim a texel is left to the base coat
COVER_FLOOR = 0.1

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
# (half widths in face sizes; a shelter puppy's blaze is a thin line - medium
# 0.2 painted a broad white bar, user)
BLAZE_HALF_WIDTH_EYES = {"small": 0.08, "medium": 0.12, "large": 0.25, "full": 0.3}
# ... BLAZE_MUZZLE_WIDEN times as wide a face size under the eyes, narrowing
# above them to nothing BLAZE_ABOVE_EYES face sizes up (a large blaze - a
# white face's - runs on up the forehead; a full one - the crown mostly
# light, dark only round the eyes and on the ears - on over the head to the
# ears' line without narrowing)
BLAZE_MUZZLE_WIDEN = 1.8
BLAZE_ABOVE_EYES = {"small": 0.45, "medium": 0.55, "large": 1.1, "full": 99.0}
# ... and a full one widens above the eyes instead, to this many times its
# width a face size up: the white between grey eye patches spreads over the
# crown (shelter puppy 14)
BLAZE_FULL_WIDEN = 2.2
# the share of the blaze's width that is solid; the rest fades to the coat
BLAZE_SOLID_SHARE = 0.4
# The face's size for the blaze: the eye radius (90th percentile of the eye
# feature texels' distance from the eye centre, times EYE_RIM_TO_RADIUS) over
# EYE_RADIUS - ar_pipeline/refine_face_masks.eye_radius, which gives the
# jindo's own eye distance back on the jindo
FACE_EYE_RIM_TO_RADIUS = 0.88
FACE_EYE_RADIUS = 0.2

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
# ... between texels this close on the surface per texel apart in the atlas
# (a texel is ~0.8 mm on the templates), and fading out where the feather's
# window drifts this far off the texel's own place (m): the window crossed a
# chart edge onto another part of the dog
FEATHER_ALL_SEAM_M = 0.003
FEATHER_ALL_DRIFT_M = (0.003, 0.008)
# ... and where it drifts, the feather is drawn on the surface: a gaussian of
# this many metres (about the atlas blur's), sampled every STRIDE texels
FEATHER_ALL_SURFACE_SIGMA_M = 0.0056
FEATHER_ALL_SURFACE_STRIDE = 4

# STYLE - "hard boundary, fur-level blending, a touch of illustration"
# ----------------------------------------------------------------------
# The colour regions keep their edges. Along each edge the mask is jittered
# by the hair noise a few texels either way, so the boundary interlocks at
# the scale of individual hairs instead of running as a geometric line.
EDGE_JITTER_PX = 1.5
# ... and the jittered edge is softened over this many texels, so it reads
# as hairs crossing the line rather than a torn edge.
EDGE_SOFT_PX = 1.0
# ... and an edge is one only where the texel across it is within this
# many metres per texel of reach on the dog (see _jitter_edge)
JITTER_SEAM_M = 0.004
JITTER_RIM_FADE_PX = 4.0
# ... except the body's regions, whose masks are smoothed over the surface
# (ar_pipeline/smooth_masks.py): those edges are steepened from the mask by
# SURFACE_EDGE_GAIN instead of jittered. The jitter is noise in the atlas,
# not on the dog, so wherever an edge met a seam, a chart folded back on
# itself or a fan of sliver charts, it broke differently on each side:
# lightning bolts down a black-and-white or black-and-tan coat's shoulders,
# flames of tan and white up its thighs (user, 11-05: "06 09 아직 번개무늬
# 남아있어 색깔 경계면에"). A mask smoothed at 1.8 cm crosses 0.5 over ~4.6
# cm; times 3, over ~1.5 cm.
SURFACE_EDGE_REGIONS = ("back", "chest", "underside", "legs_lower", "paws")
SURFACE_EDGE_GAIN = 3.0
# ... except where a light underside meets a darker coat. A dog's cream or
# white chest, belly, legs (inner or whole) and feet shade into its coat over a few
# centimetres of mixed hairs, and drawn at the fur-level edge above they
# read as pieces cut out and laid on: a straight cream band along the
# belly, a cream block on the front of each stifle, a stripe down the
# inside of every foreleg (shelter adult jindos, user 11-03: "다리 경계가
# 근원적으로 부자연스러운 느낌"). Those regions, when light on a coat that is
# not, are painted last with their weight averaged over the surface
# (SOFT_LIGHT_SIGMA_M, through the position map, so seams do not show),
# steepened by SOFT_LIGHT_GAIN and broken by the hair noise
# (SOFT_LIGHT_JITTER per unit of it).
SOFT_LIGHT_REGIONS = ("underside", "chest", "legs_lower", "paws")
SOFT_LIGHT_SIGMA_M = 0.012
SOFT_LIGHT_GAIN = 1.4
# (0.12 drew the atlas's streak noise across the soft edge: flames of white up
# a black-and-white dog's thighs, broken at every seam - see SURFACE_EDGE_GAIN)
SOFT_LIGHT_JITTER = 0.0
SOFT_LIGHT_STRIDE = 7
# ... and a light underside stays off the outer face of the thighs: the
# underside mask takes in a fifth of the jindo's outer thigh, the stifle's
# front, where the cream sat as a block. Where the surface faces away from
# the midline more than this (the normal's lateral share), the base coat -
# on the legs only, within this share of the feet's spread along the body
# from the front or hind feet: dropped all along, the belly's light line
# went from the side view too.
UNDERSIDE_OUTWARD = 0.3
UNDERSIDE_LEG_ZONE = 0.25
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
# ... with the template's own fine detail laid over it: strands, curls and
# the dark crevices between them, at the scale of TEMPLATE_DETAIL_SIGMA
# texels and smaller. Without it every generated template came out as flat
# paint with even streaks (the maltese's and the poodle's curls, the
# retriever's feathering gone; user: "덮어씌워진 느낌"). The template's own
# colour edges (the chihuahua's white blaze and bib) are the "painted edges"
# the synthetic hair was brought in to avoid; where the template's
# brightness changes this much across TEMPLATE_EDGE_SIGMA texels, the detail
# fades out (TEMPLATE_EDGE_RANGE, relative change per texel).
TEMPLATE_DETAIL = 1.0
TEMPLATE_DETAIL_SIGMA = 8.0
TEMPLATE_DETAIL_CLIP = (0.55, 1.5)
TEMPLATE_EDGE_SIGMA = 8.0
TEMPLATE_EDGE_RANGE = (0.010, 0.025)
# Template style (on by default; it replaces the two above): each region's
# colour times the template's own brightness relative to its coat - the
# template recoloured. The generated templates carry their shading in the
# base colour (no normal or occlusion map ships), so the ear hollows, the
# shadow round the eyes, the curls and the whisker dots are all in it; the
# synthetic hair threw them away and next to the template it read as paint
# (user, 10-31: "기본 템플릿 텍스처는 어색함이 없는데 입히니 이상해진다").
# The ratio is to the coat's median, so the median texel gets the region's
# mid stop and lit fur goes above it - not the darkening-only multiply.
TEMPLATE_SHADING = True
# The template's own colour pattern is not shading: the chihuahua's white
# blaze, bib and feet on fawn would come out as light patches. When the
# template's fur saturation (1 - min/max, which shading leaves alone) spreads
# wider than this between its 5th and 95th percentiles, the brightness is
# taken relative to the median of texels of the same saturation instead.
# A white template (spread 0.10 on the jindo) is one colour.
TEMPLATE_PATTERN_SPREAD = 0.2
TEMPLATE_SHADING_CLIP = (0.15, 1.3)
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

# The protect mask's edge is softened inward over this many texels.
PROTECT_FEATHER_PX = 1.2
# The template's eyelid line is dark grey: black against its white coat,
# lighter than a black one - a grey ring round each eye (user). In a band
# this wide along the inside of the protected eye (within EYE_LID_REACH_M of
# an eye landmark) the darker of the template and the coat is kept.
EYE_LID_BAND_PX = 3
EYE_LID_REACH_M = 0.03

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
# A brindle's streaks: noise BRINDLE_GRAIN_PX wide smeared BRINDLE_STREAK_PX
# across the hair, the top BRINDLE_COVER of it painted, its edge soft over
# BRINDLE_SOFT of the field's spread and at most BRINDLE_OPACITY strong.
# (Grain 5 px, smear 20 px and a hard edge drew leopard blotches - shelter
# puppy 8, user: "표범처럼 보임, 개 같지 않음"; the photo's streaks are thin,
# long and low in contrast.)
BRINDLE_GRAIN_PX = 2.0
BRINDLE_STREAK_PX = 40
BRINDLE_COVER = 0.4
BRINDLE_SOFT = 0.6
BRINDLE_OPACITY = 0.65
# A brindle on a DARK coat - the shelters' dark brindle puppies - is not
# stripes at all but lighter brown hairs mixed into a dark grey-brown coat
# in small soft clouds; drawn as stripes, fine or coarse, it came out a
# tiger or wood grain (shelter puppy 8, user: "진짜 개같지 않다"). A random
# field smoothed over the SURFACE at BRINDLE_MOTTLE_SIGMA_M (no seams),
# streaked across the hair for BRINDLE_MOTTLE_STREAK of it, at most
# BRINDLE_MOTTLE_OPACITY strong, in BRINDLE_MOTTLE_COLOUR.
BRINDLE_MOTTLE_SIGMA_M = 0.0025
BRINDLE_MOTTLE_OPACITY = 0.28
BRINDLE_MOTTLE_STREAK = 0.7
BRINDLE_MOTTLE_COLOUR = "tan"
# ... and every texel gets at least this share of it. From nothing to the full
# opacity the clouds stood out of a black coat as blotches - a leopard again on
# the adult (shelter adult jindo 7), and on a dog whose chest alone was black a
# brown-patched chest under a grey body (hard photo 178, user 2026-11-11: "과연
# 실제로 가능한 강아지인가"). With a floor the brown is mostly IN the coat's
# tone, a dark brown-black, and the clouds only vary it.
BRINDLE_MOTTLE_FLOOR = 0.5
# A coat already painted in the grey-brown of its own fur (lite's
# gray_for_black, palette.gray_from_face) has its brown in that grey: the
# mottle is this much of itself there, which leaves the coat the tan it had
# before the floor (0.11 of the tone; shelter puppy 8, judged right, keeps
# its dark grey). On black the mottle IS the coat's brown and stays whole:
# a dark chocolate at 0.28 (0.35 read a mid brown).
BRINDLE_MOTTLE_ON_GREY = 0.62
# A steel-grey "blue and tan" saddle (lite.steel_saddle) is grizzled: the
# base's tan hairs through the grey, the dark brindle's mottle at this
# opacity. Flat grey read as a grey blanket over a tan dog (shelter
# maltese 10, its back grey and tan hair in the photo).
GRIZZLE_OPACITY = 0.45
MERLE_COVER = 0.35
# ... and fur painted lighter than this (CIE L*, faded over the range) is left
# without them: the white markings (cream mid L 88, white 93; fawn 68, tan 56)
OVERLAY_SKIP_L = (76.0, 82.0)

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

def template_detail(image: Image.Image, valid: np.ndarray) -> np.ndarray:
    """The template's fine brightness detail as a ratio around 1: its luma
    over its own blur at TEMPLATE_DETAIL_SIGMA (inside the islands), faded
    to 1 across the template's own colour edges."""
    luma = to_linear(np.asarray(image, dtype=np.float32)) @ np.asarray(LUMA, dtype=np.float32)
    ratio = luma / np.maximum(_blur_inside(luma, TEMPLATE_DETAIL_SIGMA, valid), 1e-4)
    ratio = np.clip(ratio, *TEMPLATE_DETAIL_CLIP)
    broad = _blur_inside(luma, TEMPLATE_EDGE_SIGMA, valid)
    gy, gx = np.gradient(broad)
    edge = np.hypot(gx, gy) / np.maximum(broad, 1e-3)
    low, high = TEMPLATE_EDGE_RANGE
    keep = np.clip((high - edge) / (high - low), 0.0, 1.0)
    return np.where(valid, 1.0 + (ratio - 1.0) * keep, 1.0).astype(np.float32)


def template_shading(image: Image.Image, valid: np.ndarray, protect: np.ndarray | None) -> np.ndarray:
    """The template's brightness over its coat's (TEMPLATE_SHADING): 1 at
    the coat's median, below in its hollows, above on lit fur; 1 outside
    the islands."""
    linear = to_linear(np.asarray(image, dtype=np.float32))
    luma = linear @ np.asarray(LUMA, dtype=np.float32)
    fur = valid & (protect < 0.1 if protect is not None else True)
    if not fur.any():
        return np.ones(luma.shape, dtype=np.float32)
    saturation = _blur_inside(1.0 - linear.min(-1) / np.maximum(linear.max(-1), 1e-4), 2.0, fur)
    low, high = np.percentile(saturation[fur], [5, 95])
    if high - low > TEMPLATE_PATTERN_SPREAD:
        # the coat's brightness per saturation: 50 bins, each the median of
        # its texels, the sparse ones interpolated
        edges = np.linspace(0.0, 1.0, 51)
        index = np.clip(np.digitize(saturation, edges) - 1, 0, 49)
        level = np.full(50, np.nan)
        for b in range(50):
            members = fur & (index == b)
            if members.sum() > 400:
                level[b] = np.median(luma[members])
        known = ~np.isnan(level)
        centres = (edges[:-1] + edges[1:]) / 2
        level = ndimage.gaussian_filter1d(np.interp(centres, centres[known], level[known]), 1.5)
        coat = level[index]
    else:
        coat = np.full(luma.shape, float(np.median(luma[fur])), dtype=np.float32)
    shading = np.clip(luma / np.maximum(coat, 1e-4), *TEMPLATE_SHADING_CLIP)
    return np.where(valid, shading, 1.0).astype(np.float32)


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


def _select(mask: np.ndarray, field: np.ndarray, keep: float, domain: np.ndarray | None = None) -> np.ndarray:
    """Cut a soft-edged patch covering roughly `keep` of the mask (of the
    domain's texels, when given)."""
    inside = mask > 0.5
    if domain is not None and (domain & inside).any():
        inside = domain & inside
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


def _face_scale(landmarks: Any, template_dir: Path) -> float | None:
    """The eye distance a jindo with this template's eyes would have
    (FACE_EYE_*); None without the eye feature mask."""
    path = template_dir / "masks" / "_features.png"
    eyes = landmarks.anchor_points("eye")
    if not path.exists() or len(eyes) != 2 or landmarks.valid is None:
        return None
    image = Image.open(path).convert("L")
    if image.size != (landmarks.position.shape[1], landmarks.position.shape[0]):
        return None
    rim = landmarks.position[(np.asarray(image) > 127) & landmarks.valid]
    radii = []
    for name in eyes:
        distance = np.linalg.norm(rim - landmarks.world(name), axis=1)
        near = distance[distance < 0.4 * landmarks.eye_distance]
        if near.size >= 100:
            radii.append(float(np.percentile(near, 90)))
    if len(radii) != 2:
        return None
    return FACE_EYE_RIM_TO_RADIUS * float(np.mean(radii)) / FACE_EYE_RADIUS


def _blaze_strip(landmarks: Any, mask: np.ndarray, size: str, noise: np.ndarray) -> np.ndarray | None:
    """The blaze as a band along the face's midline in WORLD space, from the
    nose up between the eyes onto the forehead: widest on the muzzle,
    narrowing to a point above the eyes. The skull mask holds the ears'
    islands too, so the mask's own column percentiles (_restrict) are not
    the middle of the face; the geometry's x = 0 is."""
    ears = landmarks.anchor_points("ear_base")
    eyes = landmarks.anchor_points("eye")
    if not ears or not eyes or landmarks.position is None:
        return None
    y_back = float(np.mean([landmarks.world(name)[1] for name in ears]))
    z_eyes = float(np.mean([landmarks.world(name)[2] for name in eyes]))
    # in the face's own size (face_scale: the eye distance a jindo with this
    # template's eyes would have): the puppy 2's eyes sit 11 cm apart against
    # the jindo's 6.4, and its blaze came out 4.5 cm wide
    scale = float(getattr(landmarks, "face_scale", None) or landmarks.eye_distance)
    half = BLAZE_HALF_WIDTH_EYES.get(size, BLAZE_HALF_WIDTH_EYES["medium"]) * scale
    inside = (mask > 0.5) & (landmarks.position[..., 1] < y_back)
    rows, cols = np.nonzero(inside)
    if rows.size == 0:
        return None
    pos = landmarks.position[rows, cols]
    # the width by height: BLAZE_MUZZLE_WIDEN times `half` a face size under
    # the eyes, `half` at the eyes, narrowing to nothing BLAZE_ABOVE_EYES face
    # sizes above them. Cut by how far back it ran, the band ran up the
    # puppy's near-vertical forehead to the top of the head and read as a
    # ruler-straight white line (user: "머리에 흰줄 자꾸 왜")
    rise = (pos[:, 2] - z_eyes) / scale
    above = BLAZE_ABOVE_EYES.get(size, BLAZE_ABOVE_EYES["medium"])
    up = (1.0 + (BLAZE_FULL_WIDEN - 1.0) * np.clip(rise, 0.0, 1.0) if size == "full"
          else np.clip(1.0 - rise / above, 0.0, 1.0))
    width = half * np.where(rise < 0.0, 1.0 + (BLAZE_MUZZLE_WIDEN - 1.0) * np.clip(-rise, 0.0, 1.0), up)
    lateral = np.abs(pos[:, 0])
    # solid in the middle 0.4 of the width, fading over the rest: with the
    # fade over the outer 0.4 and the noise doubled in contrast it was a
    # band with ruled edges (shelter puppy 14's white head stripe)
    weight = np.clip(1.0 - (lateral - BLAZE_SOLID_SHARE * width) / np.maximum((1.0 - BLAZE_SOLID_SHARE) * width, 1e-6),
                     0.0, 1.0) * (width > 1e-6)
    strip = np.zeros(mask.shape, dtype=np.float32)
    strip[rows, cols] = weight
    if int((strip > 0.5).sum()) < 30:
        return None
    # a natural outline: solid inside, the edge broken by the noise
    return np.clip((strip * (0.8 + 0.4 * noise) - 0.1) * 1.4, 0.0, 1.0) * (mask > 0.5)


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
    if TEMPLATE_SHADING:
        # gray is template_shading(): the mid stop at the coat's median
        return (mid * gray[..., None]).astype(np.float32)
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
# other regions and the band goes to skull (the head colour, which already
# carries the ear's back). Measured to the nearest texel of another REGION on
# the cup's own UV island, not to the island edge, so a UV seam through the
# cup does not put a dark line down its middle. The band is this share of the
# cup's depth (the 95th percentile of that distance over the cup, so it
# scales with the cup on every template's atlas).
EAR_RIM_SHARE = 0.2
# ... and the inner colour fades in over a further share of the depth. The
# earlier rim (0.34 of the depth, faded over 0.17) left the inner colour as a
# hard-edged patch in the middle of the cup on every template (user: "귀쪽
# 털영역도 잘 안되있다").
EAR_RIM_RAMP_SHARE = 0.25


class MaskLibrary:
    """The region masks for one template, loaded once and kept as float arrays.

    With `jitter` set (a zero-mean noise field), every mask's edge is pushed
    in and out by EDGE_JITTER_PX texels of that noise - the fur-level
    interlock between two colour regions - while the region itself stays
    put. The same field jitters every mask, so two neighbouring regions'
    edges move together and leave no gap.

    ears_inner comes back pulled in by its rim and skull gains the band
    that was taken (see EAR_RIM_SHARE); every consumer of the masks - the
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
        # ar_pipeline/tidy_regions.py already cut the chest mask to the part
        # that faces forward (the _chest_front test, without the jitter)
        settings = self.dir.parent / "template.json"
        self.chest_trimmed = bool(settings.exists()
                                  and json.loads(settings.read_text(encoding="utf-8")).get("regions_tidied"))

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

    def _surface(self) -> tuple[np.ndarray, np.ndarray] | None:
        """(position map, valid) from the geometry beside the masks, None
        without one or at another size."""
        folder = self.dir.parent / "geometry"
        if not (folder / "position.npy").exists() or not (folder / "valid.npy").exists():
            return None
        position = np.load(folder / "position.npy")
        valid = self.valid if self.valid is not None else np.load(folder / "valid.npy") > 0.5
        if position.shape[:2] != (self.size[1], self.size[0]) or valid.shape != position.shape[:2]:
            return None
        return position, valid

    def _islands(self) -> np.ndarray | None:
        """The atlas's UV islands, labelled; from the geometry's valid map
        (set by the caller, or beside the masks), None without one."""
        valid = self.valid
        path = self.dir.parent / "geometry" / "valid.npy"
        if valid is None and path.exists():
            valid = np.load(path) > 0.5
        if valid is None or valid.shape != (self.size[1], self.size[0]):
            return None
        return ndimage.label(valid)[0]

    def _ear_rim(self) -> np.ndarray | None:
        """The band of ears_inner near another region (EAR_RIM_SHARE), as a
        soft 0..1 weight; None when there is no inner-ear mask."""
        if "ear_rim" in self._cache:
            return self._cache["ear_rim"]
        inner = self._raw_mask("ears_inner")
        if inner is None or EAR_RIM_SHARE <= 0:
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
        distance = ndimage.distance_transform_edt(~other).astype(np.float32)
        islands = self._islands()
        if islands is not None:
            # ... on the cup's own island: the nearest other-region texel in
            # the atlas was often across a gutter, on another island, and bit
            # circles out of the cup - a scalloped pale blob in the ear
            # (user: "귀쪽 털영역"). A splinter island with no other region
            # on it (a fringe piece at the ear's edge) keeps the atlas distance.
            for index in np.unique(islands[(inner > 0.5) & (islands > 0)]):
                rows, cols = np.nonzero(islands == index)
                box = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
                sources = other[box] & (islands[box] == index)
                if not sources.any():
                    continue
                own = islands[box] == index
                distance[box][own] = ndimage.distance_transform_edt(~sources)[own]
        depths = distance[(inner > 0.5) & np.isfinite(distance)]
        depth = float(np.percentile(depths, 95)) if depths.size else 0.0
        rim_px, ramp_px = EAR_RIM_SHARE * depth, max(EAR_RIM_RAMP_SHARE * depth, 1.0)
        inside = np.clip((distance - rim_px) / ramp_px, 0.0, 1.0).astype(np.float32)
        rim = inner * (1.0 - inside)
        self.ear_rim_texels = int((rim > 0.5).sum())
        self._cache["ear_rim"] = rim
        return rim

    def ear_cup(self) -> np.ndarray | None:
        """The whole inner ear, rim included, jittered as the regions are:
        what skull paints under ears_inner (compose_body), so the rim's soft
        split between the two leaves no gap for the base coat."""
        if "ear_cup" not in self._cache:
            mask = self._raw_mask("ears_inner")
            if mask is not None and self.jitter is not None:
                mask = _jitter_edge(mask, self.jitter)
            self._cache["ear_cup"] = mask
        return self._cache["ear_cup"]

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
                if "surface" not in self._cache:
                    self._cache["surface"] = self._surface()
                if region in SURFACE_EDGE_REGIONS and self._cache["surface"] is not None:
                    # smoothed on the surface already (smooth_masks): an edge
                    # made sharper from the mask alone, the same on either
                    # side of every seam, no noise (SURFACE_EDGE_GAIN)
                    mask = np.clip((mask - 0.5) * SURFACE_EDGE_GAIN + 0.5, 0.0, 1.0).astype(np.float32)
                else:
                    mask = _jitter_edge(mask, self.jitter, surface=self._cache["surface"])
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


def _leg_inner(landmarks, jitter: np.ndarray | None, forward_too: bool = True) -> np.ndarray:
    """1 where the surface faces the body's midline or (`forward_too`)
    forward, 0 where it faces outward or back, with a fur-level edge; the
    inside of a leg."""
    normal = landmarks.normal
    side = np.sign(landmarks.position[..., 0]) if landmarks.position is not None else 1.0
    inward = -side * normal[..., 0]
    signed = (inward - LEG_INNER_FACING) / 0.08
    if forward_too:
        signed = np.maximum(signed, (-normal[..., 1] - CHEST_FACING) / 0.08)
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


# The chest mask runs down the front of each foreleg to the legs_lower band
# (the retriever's from the brisket at 0.37 m to 0.30 m), and a light chest
# painted there drew cream stockings from elbow to toes on golden and yellow
# dogs whose photos show coat-coloured forelegs (shelter retriever-types 1,
# 2, 9). Where the legs are the coat's colour a light chest stops at the
# brisket - the lowest point of the chest mask's forward-facing midline
# (BRISKET_MIDLINE_M either side of chest_center, its BRISKET_PERCENTILE) -
# over BRISKET_RAMP_M; below it the mask keeps the base coat, the legs'
# colour. Not where the legs are another colour (a black-and-tan's tan
# legs under a white chest): the base coat there is black.
BRISKET_MIDLINE_M = 0.015
BRISKET_PERCENTILE = 5.0
BRISKET_RAMP_M = 0.012


def _above_brisket(landmarks, chest: np.ndarray, jitter: np.ndarray | None) -> np.ndarray | None:
    """1 above the brisket, 0 below it on the forelegs (None without the
    landmarks it needs)."""
    if landmarks.position is None or landmarks.normal is None or "chest_center" not in landmarks.points:
        return None
    centre = landmarks.world("chest_center")
    position = landmarks.position
    midline = ((chest > 0.5) & landmarks.valid & (np.abs(position[..., 0] - centre[0]) < BRISKET_MIDLINE_M)
               & (-landmarks.normal[..., 1] > CHEST_FACING) & (position[..., 1] < centre[1] + 0.1))
    if int(midline.sum()) < 20:
        return None
    brisket = float(np.percentile(position[..., 2][midline], BRISKET_PERCENTILE))
    signed = (position[..., 2] - brisket) / BRISKET_RAMP_M
    if jitter is not None:
        signed = signed + 0.3 * jitter
    return np.clip(signed + 0.5, 0.0, 1.0).astype(np.float32)


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


def _jitter_edge(mask: np.ndarray, jitter: np.ndarray, px: float = EDGE_JITTER_PX,
                 surface: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """Push a mask's edge in and out by the noise: the signed distance to the
    edge is offset by `px * jitter` and re-thresholded with a one-texel
    soft edge. Interior and exterior far from the edge are untouched.

    With `surface` (position map, valid) an edge counts only where the
    nearest texel across it is on the islands and near on the dog
    (JITTER_SEAM_M per texel of reach): across the padding, or where a chart
    touches itself in the atlas, it is no edge. (Measured across the atlas,
    the rim of a black shoulder one texel from the white lower leg - the
    same chart, folded back on itself, 10 cm away on the dog - turned white
    wherever the noise pushed, and the shoulder's seam ran as a light
    lightning bolt down a black-and-white or black-and-tan coat's upper
    arm; user, 11-05: "팔 경계 번개 모양 경계 자국".)"""
    inside = mask > 0.5
    if not inside.any() or inside.all():
        return mask
    plain = ndimage.distance_transform_edt(inside) - ndimage.distance_transform_edt(~inside)
    if surface is None or surface[1].shape != mask.shape:
        distance = plain
    else:
        position, on = surface
        far = 4.0 * px + 2.0
        # outside texels: to the nearest inside texel on the islands, if it is near on the dog
        d_out, (ri, ci) = ndimage.distance_transform_edt(~(inside & on), return_indices=True)
        near_out = np.linalg.norm(position[ri, ci] - position, axis=-1) <= JITTER_SEAM_M * (d_out + 1.0)
        # inside texels: likewise to the nearest outside one
        d_in, (ro, co) = ndimage.distance_transform_edt(~(~inside & on), return_indices=True)
        near_in = np.linalg.norm(position[ro, co] - position, axis=-1) <= JITTER_SEAM_M * (d_in + 1.0)
        distance = np.where(inside, np.where(near_in, np.minimum(d_in, far), far),
                            -np.where(near_out, np.minimum(d_out, far), far))
        # the padding keeps the plain measure (it is refilled from the islands later)
        distance = np.where(on, distance, plain)
        jittered = np.clip((distance + px * jitter) / max(EDGE_SOFT_PX, 1e-3) + 0.5, 0.0, 1.0)
        # near an island's rim the edge is left soft (the mask as it is,
        # continuous over the seam): jittered on each side of a seam on its
        # own, an edge running along one - the elbow, where the leg's chart
        # ends - came out as a dotted line of the other colour on both
        # sides (JITTER_RIM_FADE_PX)
        rim = ndimage.distance_transform_edt(on)
        fade = np.clip((rim - 1.0) / JITTER_RIM_FADE_PX, 0.0, 1.0)
        return (fade * jittered + (1.0 - fade) * mask).astype(np.float32)
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
    t = (np.nan_to_num(distance, nan=high) - low) / (high - low)
    # beyond the region the fade the nearest texel of it has (it was the
    # fade's far end: a light region's soft edge reaches past its mask, and a
    # white leg fading into black above its paw put a black band round the
    # paw where the leg's edge met it - user, 11-05)
    nearest = ndimage.distance_transform_edt(~inside, return_distances=False, return_indices=True)
    t = t[nearest[0], nearest[1]]
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
    cover = np.zeros_like(gray)
    for entry in entries:
        mask = masks.get(entry.get("region"))
        if mask is not None:
            claimed = np.maximum(claimed, mask)
            cover = cover + mask
    unclaimed = 1.0 - claimed
    # A texel the regions claim only in part (their masks summing under 1)
    # had the base coat under the rest: at every island's edge, where the
    # masks fade into the empty gutter, and on the small shards of fur
    # whose masks are faint all over. On a white dog with a black head that
    # was a grey rim round every black island and white shards at the ear
    # roots, read across each seam as white streaks in the black ears
    # (shelter chihuahua 10: the ear island's edge texels luma 84 against
    # 25 inside, skull mask 0.86 there). Such texels take the regions' own
    # colours, the masks scaled up to sum to one; where they claim under
    # COVER_FLOOR the texel is the base's (nothing painted it).
    lift = np.where(cover > COVER_FLOOR, 1.0 / np.clip(cover, COVER_FLOOR, 1.0), 1.0).astype(np.float32)

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
    # The ear's rim is a soft split of the cup between ears_inner and skull
    # (MaskLibrary._ear_rim): a texel half each, painted one over the other,
    # kept a quarter of the base coat - on a white dog with a black head a
    # dotted grey-white line round the inside of each black ear (shelter
    # chihuahua 10), unseen wherever the base is the head's colour, and no
    # feathering runs between two regions of one colour. So skull goes first
    # and covers the whole cup, and ears_inner is laid over it.
    painting = observed + inferred
    names = [e.get("region") for e in painting]
    cup = masks.ear_cup() if "skull" in names and "ears_inner" in names else None
    if cup is not None and names.index("ears_inner") < names.index("skull"):
        painting.insert(names.index("ears_inner"), painting.pop(names.index("skull")))
    # light underside regions, painted after the rest and soft (SOFT_LIGHT_*)
    soft_light: list[tuple[str, np.ndarray, np.ndarray]] = []
    soft_ready = (landmarks is not None and landmarks.position is not None and landmarks.normal is not None
                  and landmarks.position.shape[:2] == gray.shape)
    for entry in painting:
        mask = masks.get(entry.get("region"))
        if mask is None:
            continue
        mask = np.clip(mask * lift, 0.0, 1.0)
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

        # ... and on a tidied template the mask is that already: the test
        # again, with its jitter, only scattered base-coat specks where the
        # fur's normals hover at the threshold (the retriever's bib)
        if (region == "chest" and landmarks is not None and landmarks.normal is not None
                and color_name != base_color and not masks.chest_trimmed):
            applied = applied * _chest_front(landmarks, masks.jitter)
            if debug:
                _save_debug(debug, f"02_{order:02d}_chest_front", applied)
        if (region == "chest" and landmarks is not None and color_family(color_name) == "light"
                and color_family(base_color) != "light"
                and next((str(e.get("color", "")) for e in entries
                          if e.get("region") == "legs_lower"), base_color) == base_color):
            above = _above_brisket(landmarks, mask, masks.jitter)
            if above is not None:
                applied = applied * above
                if debug:
                    _save_debug(debug, f"02_{order:02d}_chest_above_brisket", applied)
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
        if region == "skull" and cup is not None:
            applied = np.maximum(applied, cup)
        if (soft_ready and region == "underside" and color_family(color_name) == "light"
                and color_family(base_color) != "light"):
            applied = applied * _not_outward(landmarks, masks.jitter)

        if (soft_ready and region in SOFT_LIGHT_REGIONS and color_family(color_name) == "light"
                and color_family(base_color) != "light"):
            soft_light.append((f"{region}:{color_name}", painted, applied))
        else:
            composed = _blend(composed, painted, applied)
        inner_color = str(entry.get("inner_color") or "").lower()
        if (region == "legs_lower" and inner_color in FUR_COLORS and inner_color != color_name
                and landmarks is not None and landmarks.normal is not None):
            inner = applied * _leg_inner(landmarks, masks.jitter,
                                         forward_too=str(entry.get("inner_faces") or "") != "inward")
            if soft_ready and color_family(inner_color) == "light" and color_family(color_name) != "light":
                soft_light.append((f"legs_lower inner:{inner_color}",
                                   paint_region(gray, inner_color, mask, palette), inner))
            else:
                composed = _blend(composed, paint_region(gray, inner_color, mask, palette), inner)
            gradients_drawn.append(f"legs_lower inner:{inner_color}")
            if debug:
                _save_debug(debug, f"02_{order:02d}_legs_inner", inner)
        remember(entry, mask)
        if debug:
            tag = "inferred" if entry in inferred else "seen"
            _save_debug(debug, f"02_{order:02d}_{region}_{color_name}_{tag}_mask", applied)
            _save_debug(debug, f"02_{order:02d}_{region}_{color_name}_{tag}", composed)
    # the light underside, last and soft (SOFT_LIGHT_*)
    if soft_light:
        valid_here = landmarks.valid if landmarks.valid is not None else (gray > -1)
        smoothed = smooth_fields_on_surface([np.asarray(w, dtype=np.float32) for _, _, w in soft_light],
                                            landmarks.position, valid_here, SOFT_LIGHT_SIGMA_M)
        # painted together: how light the coat is here is the regions'
        # weights summed, steepened once at its edge against the coat, and
        # each light colour has its share of it. (Laid on one after another,
        # two light regions meeting - a white leg and its white paw, the
        # chest and the legs - each came to half there and let a quarter of
        # the dark coat through: a dark broken ring round each wrist and
        # dark lines inside the white along every edge between them; user,
        # 11-05: "흑백 경계면에 알 수 없는 선들")
        total = np.clip(np.sum(smoothed, axis=0), 0.0, 1.0)
        weight = (total - 0.5) * SOFT_LIGHT_GAIN + 0.5
        if masks.jitter is not None:
            # the hair noise in the transition only: out in the coat it
            # scattered white specks over a black flank
            bell = np.clip(1.0 - np.abs(total - 0.5) / 0.4, 0.0, 1.0)
            weight = weight + SOFT_LIGHT_JITTER * masks.jitter * bell
        weight = np.clip(weight, 0.0, 1.0) * valid_here
        light = sum(soft[..., None] * painted for (_, painted, _), soft in zip(soft_light, smoothed))
        light = light / np.maximum(np.sum(smoothed, axis=0), 1e-6)[..., None]
        composed = _blend(composed, light, weight)
        if debug:
            _save_debug(debug, "02_soft_light", weight)
        report["soft_light"] = [name for name, _, _ in soft_light]
    # the band round each eye (ar_pipeline/open_face_masks.py): a
    # light-cheeked dog's cheek colour runs up round the eyes to the brows -
    # a black-and-tan's cream spectacles (eval demo e08). A piebald keeps its
    # dark cap over the eyes.
    spectacle = masks.extra("_spectacle")
    if spectacle is not None:
        colours = {str(e.get("region")): str(e.get("color", "")) for e in entries}
        cheek, skull_colour = colours.get("cheeks", ""), colours.get("skull", base_color)
        # ... under a DARK head only: on a tan head (urajiro, sable) the cream
        # stops below the eyes, and spectacles whitened e09's face. And not
        # where the head close-up saw the eyes in the head colour
        # (face_colours eye_ring): a black shiba is black round the eyes and
        # cream below them, the same reading as e08 everywhere else.
        if (cheek in FUR_COLORS and color_family(cheek) == "light" and cheek != skull_colour
                and color_family(skull_colour) == "dark"
                and str(appearance.get("pattern", "")) != "piebald"
                and appearance.get("eye_ring") != "dark"):
            composed = _blend(composed, paint_region(gray, cheek, spectacle, palette), np.clip(spectacle, 0, 1))
            gradients_drawn.append(f"spectacle:{cheek}")
    # ... and for that dark-eyed dog the cheeks' top, which reaches eye level
    # on the templates, round and just under each eye takes the head colour
    # (open_face_masks EYE_RING_*): the cream starts below the eye
    eye_ring = masks.extra("_eye_ring") if appearance.get("eye_ring") == "dark" else None
    if eye_ring is not None:
        colours = {str(e.get("region")): str(e.get("color", "")) for e in entries}
        skull_colour = colours.get("skull", base_color)
        if skull_colour in FUR_COLORS and colours.get("cheeks", skull_colour) != skull_colour:
            composed = _blend(composed, paint_region(gray, skull_colour, eye_ring, palette), np.clip(eye_ring, 0, 1))
            gradients_drawn.append(f"eye_ring:{skull_colour}")
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
    # where the markings went, for the finish to leave alone (lite._clean_head
    # took a white blaze on a dark crown for a shard and painted it over)
    marking_mask = np.zeros(gray.shape, dtype=np.float32)
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
            trunk = None
            if landmarks is not None and landmarks.position is not None:
                rng = np.random.default_rng(PATCH_SEED + order)
                surface = landmarks.valid if landmarks.valid is not None else mask > -1
                # sampled as coarsely as the blobs are wide: the cost grows
                # with the samples times the neighbours within 2.5 sigma
                smooth = smooth_on_surface(rng.random(mask.shape).astype(np.float32), landmarks.position,
                                           surface, PATCH_SIGMA_M,
                                           stride=max(SURFACE_STRIDE, round(SURFACE_STRIDE * PATCH_SIGMA_M / 0.035)))
                inside = (mask > 0.5) & surface
                if inside.any():
                    spread = float(smooth[inside].std()) or 1.0
                    field = 0.5 + (smooth - float(smooth[inside].mean())) / spread * PATCH_FIELD_SPREAD
                    # on the trunk only: the back region runs onto the neck
                    # and the tail, and the neck crest is its highest place -
                    # the patches ran from a black head down the neck as one
                    # strip, and a whole tail came out coloured, on dogs with
                    # a white neck and a white tail (eval e13, shelter
                    # maltese 6 and 7, e15 - all five patched dogs)
                    hard = _patch_off_trunk(masks, landmarks, mask.shape)
                    off = smooth_on_surface(hard, landmarks.position, surface, PATCH_OFF_RAMP_M,
                                            stride=max(SURFACE_STRIDE,
                                                       round(SURFACE_STRIDE * PATCH_OFF_RAMP_M / 0.035)))
                    # ... the fade on the trunk's side only: a tail curled
                    # over the back lies within the fade of the back itself
                    # and came out black on a white-tailed piebald (shelter
                    # chihuahua 10, its tail at 0.4 of the full lowering)
                    off = np.maximum(off, hard)
                    field = field - PATCH_OFF_TRUNK * off
                    trunk = inside & (off < 0.5)
                    # on the top of the body: the back region runs down the
                    # flanks onto the upper legs, and patches scattered over
                    # all of it came out a cow's spots down to the knees where
                    # the dog had a saddle and a rump patch over white legs
                    # (shelter maltese 7)
                    z = landmarks.position[..., 2]
                    low, high = np.percentile(z[trunk if trunk.any() else inside], [10, 90])
                    top = np.clip((z - low) / max(high - low, 1e-6), 0.0, 1.0)
                    # ... and drawn to where the reading put them (body_colours
                    # patch_*): on the side, the top bias would pull a flank
                    # patch onto the spine (shelter maltese 6)
                    places = [p for p in marking.get("places") or () if p in PATCH_PLACE_SHAPE]
                    if "side" not in places:
                        field = field + PATCH_TOP_BIAS * (top - 0.5)
                    # one flank: a flank patch is one dog's side, and the
                    # photograph does not say which
                    side = float(rng.choice((-1.0, 1.0)))
                    place = _patch_place_weight(landmarks, trunk, places, side)
                    if place is not None:
                        field = field + PATCH_PLACE_BIAS * place
                        placed.append(f"patches@{'+'.join(places)}")
            # the area is the trunk's share: off it the field is too low to
            # be picked, and the region's share came out a third more of the
            # trunk once the neck and tail were taken out (eval e13, a
            # blanket from the withers to the rump)
            blob = _select(mask, field, PATCH_AREA.get(size, PATCH_AREA["medium"]), domain=trunk)
            if trunk is not None:
                # no lone round spot beside the patches: its area goes back
                # to them (PATCH_MIN_PART)
                for _ in range(PATCH_SPOT_PASSES):
                    spots = _patch_spots(blob, landmarks.position, surface)
                    if spots is None:
                        break
                    # lowered round it as well, smoothly on the surface: cut
                    # out alone, its shoulders came back as a grey ring round
                    # a white hole on the next cut
                    drop = smooth_on_surface(spots.astype(np.float32), landmarks.position, surface, PATCH_SIGMA_M,
                                             stride=max(SURFACE_STRIDE,
                                                        round(SURFACE_STRIDE * PATCH_SIGMA_M / 0.035)))
                    field = field - PATCH_SPOT_DROP * np.clip(np.maximum(drop * 4.0, spots), 0.0, 1.0)
                    blob = _select(mask, field, PATCH_AREA.get(size, PATCH_AREA["medium"]), domain=trunk)
                    placed.append(f"patches: dropped a spot of {int(spots.sum())} texels")
                # no pale half-painted spots: where the field rose near the
                # cut but not past it, a faint grey or tan smudge (e13's
                # shoulder, e15's back). Only the edge band of a patch with a
                # painted core keeps its fade, measured on the surface so a
                # seam does not cut it
                core = smooth_on_surface((blob > PATCH_CORE).astype(np.float32), landmarks.position, surface,
                                         PATCH_CORE_REACH_M, stride=PATCH_CORE_STRIDE)
                blob = blob * np.clip(core / PATCH_CORE_NEAR, 0.0, 1.0)
            placed.append(f"patches:{region}")
        if blob is None and marking_type == "ears":
            # the backs of the ears (lite.dark_ears_on_light): the template's
            # ears_outer source mask inside the skull, softened on the
            # surface so the ear root is not a cut line
            path = masks.dir.parent / "masks_source" / "ears_outer.png"
            if not path.exists():
                continue
            ear = (np.asarray(Image.open(path).convert("L").resize((mask.shape[1], mask.shape[0]), Image.NEAREST))
                   > 127).astype(np.float32)
            if (landmarks is not None and landmarks.position is not None
                    and landmarks.position.shape[:2] == mask.shape[:2]):
                surface = landmarks.valid if landmarks.valid is not None else mask > -1
                ear = smooth_on_surface(ear, landmarks.position, surface, EARS_SMOOTH_M,
                                        stride=max(SURFACE_STRIDE, round(SURFACE_STRIDE * EARS_SMOOTH_M / 0.035)))
                ear = np.clip((ear - 0.5) * 2.5 + 0.5, 0.0, 1.0)
            blob = ear * (mask > 0.5)
            if float((blob > 0.5).sum()) < 30:
                continue
            placed.append("ears:skull")
        if blob is None and marking_type == "nape" and landmarks is not None:
            weight = _nape_weight(masks, landmarks, mask.shape)
            if weight is not None and float((weight > 0.5).sum()) >= 30:
                # smoothed over the surface so a UV seam does not cut it, and
                # broken a little by the noise like any coat edge
                surface = landmarks.valid if landmarks.valid is not None else mask > -1
                weight = smooth_on_surface(weight, landmarks.position, surface, NAPE_SMOOTH_M,
                                           stride=max(SURFACE_STRIDE, round(SURFACE_STRIDE * NAPE_SMOOTH_M / 0.035)))
                blob = np.clip((weight - 0.5) * 2.5 + 0.5 + (noise - 0.5) * 0.4, 0.0, 1.0) * (mask > 0.5)
                placed.append("nape:back")
            else:
                continue
        if blob is None and marking_type == "blaze" and landmarks is not None:
            # down the top of the muzzle as well: a blaze runs from the nose
            # up between the eyes, and on the skull alone it was a white bar
            # on the forehead above a dark bridge (shelter puppy 8)
            bridge = masks.get("muzzle_bridge") if region == "skull" else None
            blob = _blaze_strip(landmarks, np.maximum(mask, bridge) if bridge is not None else mask, size, noise)
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
        marking_mask = np.maximum(marking_mask, np.asarray(blob, dtype=np.float32))
        if debug:
            _save_debug(debug, f"03_{order:02d}_{marking_type}_{region}_mask", blob)
            _save_debug(debug, f"03_{order:02d}_{marking_type}_{region}", composed)
    report["markings_placed"] = placed
    report["marking_mask"] = marking_mask

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
        # on the islands, region or not: where two soft masks are both near
        # 0.5 the jitter can leave a texel in neither (index -1), and left
        # out of the feather it kept the base coat - black specks in a line
        # through the tan below a black-and-tan's elbow (user, 11-05). It
        # counts as a colour of its own, so the feather takes it in.
        if masks.valid is not None and masks.valid.shape == gray.shape:
            valid_here = masks.valid.copy()
        else:
            valid_here = index >= 0
        position = (landmarks.position if landmarks is not None and landmarks.position is not None
                    and landmarks.position.shape[:2] == gray.shape else None)
        boundary = np.zeros(gray.shape, dtype=bool)
        for shift in (1, reach):
            for axis in (0, 1):
                other = np.roll(index, shift, axis=axis)
                change = (index != other) & valid_here & np.roll(valid_here, shift, axis=axis)
                if position is not None:
                    # ... between texels that are neighbours on the dog too
                    # (FEATHER_ALL_SEAM_M per texel of shift): two charts packed
                    # edge to edge in the atlas are not
                    apart = np.linalg.norm(position - np.roll(position, shift, axis=axis), axis=-1)
                    change &= apart < FEATHER_ALL_SEAM_M * shift
                boundary |= change
        band = ndimage.binary_dilation(boundary, iterations=reach)
        soft = _blur(band.astype(np.float32), FEATHER_ALL_PX / 3.0) * valid_here
        blurred = _blur_inside(composed, FEATHER_ALL_PX / 2.0, valid_here)
        if position is not None:
            # ... and only where the blur stayed on this part of the surface:
            # its window crossed a chart edge, or a gutter, onto a place 15 cm
            # away - a white bib chart packed against the black shoulder chart
            # blurred into it as a white zigzag (the jindo puppy 2, user:
            # "앞다리 바깥쪽 번개무늬"). The blurred position tells: it drifts
            # off the texel's own where the window took in far surfaces.
            drift = np.linalg.norm(_blur_inside(position, FEATHER_ALL_PX / 2.0, valid_here) - position, axis=-1)
            low, high = FEATHER_ALL_DRIFT_M
            gate = np.clip((high - drift) / (high - low), 0.0, 1.0)
            # there the feather is drawn on the surface instead (left out, the
            # jittered edge stayed hard where a chart touches itself in the
            # atlas: a dotted line of black hair-tips in the tan below a
            # black-and-tan's elbow; user, 11-05). (Drawn on the surface
            # everywhere it left the fine seam lines it was meant to take
            # away and added a few light specks.)
            off = (gate < 0.999) & (soft > 0.01)
            if off.any():
                smoothed = smooth_fields_on_surface([composed[..., i] for i in range(composed.shape[-1])],
                                                    position, valid_here, FEATHER_ALL_SURFACE_SIGMA_M,
                                                    stride=FEATHER_ALL_SURFACE_STRIDE)
                surface_blurred = np.stack(smoothed, axis=-1)
                blurred = np.where(off[..., None], gate[..., None] * blurred + (1.0 - gate[..., None]) * surface_blurred,
                                   blurred)
                report["feather_all_on_surface"] = int(off.sum())
        composed = _blend(composed, blurred, np.clip(soft * 1.5, 0, 1))
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
# surface with this sigma. The field is rescaled so EDGE_SOFTNESS gives a
# fur-soft, not a blurred, edge. At 3.5 cm (blobs 10-15 cm across) a
# parti-coloured dog with one saddle and a rump patch came out spotted all
# over like a cow (shelter maltese 7); a part-coloured coat is a few large
# patches.
PATCH_SIGMA_M = 0.07
PATCH_FIELD_SPREAD = 0.25
PATCH_SEED = 4242
# the field's lift from the bottom tenth of the trunk's height to the top
# tenth (in the field's units: its spread is PATCH_FIELD_SPREAD)
PATCH_TOP_BIAS = 0.4
# Off the trunk, where no patch goes: the template's own collar and tail
# masks (masks_source) and what lies PATCH_TAIL_BEHIND_M behind the hind feet
# (the maltese's tail root is tagged back), lowered in the field by
# PATCH_OFF_TRUNK, faded in over PATCH_OFF_RAMP_M on the surface so a patch
# does not end on the masks' line
PATCH_OFF_SOURCES = ("collar", "tail_top", "tail_tip")
PATCH_TAIL_BEHIND_M = 0.05
PATCH_OFF_TRUNK = 1.0
# as wide as the patches' own noise: over 2.5 cm the fade outweighed the
# noise and a patch ended on a straight line at the tail root and the neck
PATCH_OFF_RAMP_M = 0.06
# a patch's painted core (blob above PATCH_CORE) and how near to one, on the
# surface, its fading edge must lie: the core smoothed with this sigma
# reaches PATCH_CORE_NEAR at about one sigma off its edge
PATCH_CORE = 0.6
PATCH_CORE_REACH_M = 0.02
PATCH_CORE_NEAR = 0.15
# sampled every 7 texels: 2.9 s on the jindo against 11.5 s at 5, the reach
# within 0.06 at the 99th percentile
PATCH_CORE_STRIDE = 7
# The share of the TRUNK (the back region off the neck and the tail) the
# patches cover, by body_colours patch_amount (lite.PATCH_SIZE: small ->
# small, large -> medium, most -> large; "large" means patches bigger than
# the head, not more of the body). medium is also for no answer. As shares
# of the whole region (neck and tail in) a quarter sat between a white puppy
# with one flank patch (shelter maltese 6, ~15 %) and a saddle-and-rump parti
# (maltese 7, ~35 %); of the trunk, 0.25 draws maltese 7 as a saddle and a
# separate rump patch and eval e13 with a white back before its rump patch.
PATCH_AREA = {"small": 0.12, "medium": 0.25, "large": 0.45}
# A place the reading put patches on, drawn there on top of the top-of-body
# bias: along the trunk, 0 at the front feet and 1 at the hind feet, as
# (centre, gaussian sigma), times the height on the trunk (0 at its lowest
# tenth, 1 at its highest) as a ramp (from, to) or a band (centre, -sigma);
# the most of the places read, added to the field at PATCH_PLACE_BIAS per
# unit (its noise spread is PATCH_FIELD_SPREAD). A rump weight alone moves a
# few percent of the texels: the top bias already leans the patches back
# (e13 5.6 %). An over-the-spine patch is the top bias itself. Weighting a
# saddle, a flank and a rump at once covered the whole trunk - one blanket
# where the dog has a white back and a rump patch (eval e13).
PATCH_PLACE_SHAPE = {
    "rump": ((0.95, 0.2), (0.15, 0.6)),
    "side": ((0.55, 0.18), (0.45, -0.18)),
}
PATCH_PLACE_BIAS = 0.5
# a side patch: on one flank, faded across the midline over this width
PATCH_SIDE_FADE_M = 0.03


# A part of the patches standing alone on the surface (its texels joined in
# PATCH_SPOT_CELL_M cells, neighbouring cells one part, so a seam does not
# split a patch) with under PATCH_MIN_PART of all their texels is a spot:
# the smoothed field's lesser peaks came out as round polka dots beside the
# real patches - a black disc on a white flank beside the rump patch (shelter
# chihuahua 10, 13 % of the patches; user 11-07 "이상한데"). Its field is
# lowered by PATCH_SPOT_DROP and the area chosen again, so the patches keep
# their size; up to PATCH_SPOT_PASSES times.
PATCH_MIN_PART = 0.2
PATCH_SPOT_CELL_M = 0.015
PATCH_SPOT_DROP = 10.0
PATCH_SPOT_PASSES = 3


def _patch_spots(blob: np.ndarray, position: np.ndarray, valid: np.ndarray) -> np.ndarray | None:
    """The texels of the patches' small lone parts, or None when there are none."""
    on = (blob > 0.5) & valid
    if on.sum() < 100:
        return None
    points = position[on]
    cells = np.floor((points - points.min(axis=0)) / PATCH_SPOT_CELL_M).astype(np.int64)
    grid = np.zeros(tuple(cells.max(axis=0) + 1), dtype=bool)
    grid[tuple(cells.T)] = True
    labels, count = ndimage.label(ndimage.binary_dilation(grid))
    if count < 2:
        return None
    part = labels[tuple(cells.T)]
    sizes = np.bincount(part, minlength=count + 1)
    small = sizes[part] < PATCH_MIN_PART * len(part)
    if not small.any():
        return None
    spots = np.zeros(blob.shape, dtype=bool)
    rows, cols = np.nonzero(on)
    spots[rows[small], cols[small]] = True
    return spots


def _patch_off_trunk(masks: "MaskLibrary", landmarks: Any, shape: tuple[int, ...]) -> np.ndarray:
    """1 on the neck and the tail (PATCH_OFF_SOURCES and behind the hind
    feet), 0 on the trunk."""
    off = np.zeros(shape[:2], dtype=np.float32)
    for name in PATCH_OFF_SOURCES:
        path = masks.dir.parent / "masks_source" / f"{name}.png"
        if path.exists():
            image = Image.open(path).convert("L").resize((shape[1], shape[0]), Image.NEAREST)
            off = np.maximum(off, (np.asarray(image) > 127).astype(np.float32))
    hind = [landmarks.world(name)[1] for name in ("paw_lb", "paw_rb") if name in landmarks.points]
    if hind:
        off[landmarks.position[..., 1] > float(np.mean(hind)) + PATCH_TAIL_BEHIND_M] = 1.0
    return off


# The nape (body_colours.NAPE_ANSWERS head): the head's colour down the back
# and upper sides of the neck, from behind the skull to the withers.
# Measured round the neck's own axis (the collar ring's ends): a
# texel's direction from the axis against the axis's upward perpendicular -
# 1 on the crest, 0 at the middle of the side, -1 under the throat - full
# from NAPE_FROM_COS + NAPE_FADE_COS, none below NAPE_FROM_COS, and fading
# out over the last NAPE_END_FADE of the way to the withers, within
# NAPE_RADIUS_M of the axis. Sliced by height along the body instead, the
# near-upright neck's crest came out white between two brown side stripes.
# A brown head running down the neck to a white band at the shoulders
# (shelter maltese 7).
NAPE_FROM_COS = -0.35
NAPE_FADE_COS = 0.3
NAPE_END_FADE = 0.3
# 0.14 left a white ring round the back of the skull at the ear roots: the
# collar's head end slants, and the crown's back stands 14-16 cm off the
# axis's end (shelter maltese 7); the end fade keeps it off the trunk
NAPE_RADIUS_M = 0.2
NAPE_SMOOTH_M = 0.015
# the ears' backs (marking "ears"): their edge at the ear root softened over
# about this much of the surface
EARS_SMOOTH_M = 0.008
# the share of the collar's length at each end averaged for the axis' ends
NAPE_RING_SHARE = 0.15


def _nape_weight(masks: "MaskLibrary", landmarks: Any, shape: tuple[int, ...]) -> np.ndarray | None:
    """0..1 per texel on the back and upper sides of the neck, or None when
    the template lacks its geometry or a collar mask."""
    if landmarks.position is None:
        return None
    position = landmarks.position
    if position.shape[:2] != tuple(shape[:2]):
        return None

    def source(name: str) -> np.ndarray:
        path = masks.dir.parent / "masks_source" / f"{name}.png"
        if not path.exists():
            return np.zeros(shape[:2], dtype=bool)
        return np.asarray(Image.open(path).convert("L").resize((shape[1], shape[0]), Image.NEAREST)) > 127

    def region(name: str) -> np.ndarray:
        mask = masks.get(name)
        return mask > 0.5 if mask is not None else np.zeros(shape[:2], dtype=bool)

    valid = landmarks.valid if landmarks.valid is not None else np.ones(shape[:2], dtype=bool)
    neck = (source("collar") | region("back")) & valid & ~source("throat") & ~region("chest")
    if not neck.any():
        return None
    # the neck's own centre line: from the middle of the collar ring's head
    # end to the middle of its body end (the ear bases and the withers are
    # both on the crest - an axis through them lay along the crest itself)
    collar = source("collar") & valid
    if collar.sum() < 100:
        return None
    y = position[..., 1]
    near, far = np.percentile(y[collar], [5, 95])
    reach = NAPE_RING_SHARE * (far - near)
    head = position[collar & (y < near + reach)].mean(axis=0)
    withers = position[collar & (y > far - reach)].mean(axis=0)
    head[0] = withers[0] = 0.0
    axis = withers - head
    length2 = float(axis @ axis)
    t = np.clip(((position - head) @ axis) / length2, 0.0, 1.0)
    radial = position - (head + t[..., None] * axis)
    up = np.array([0.0, -axis[2], axis[1]], dtype=np.float32)
    up = up if up[2] > 0 else -up
    up /= float(np.linalg.norm(up))
    distance = np.linalg.norm(radial, axis=-1)
    cos = (radial @ up) / np.maximum(distance, 1e-6)
    weight = (neck * (distance < NAPE_RADIUS_M)
              * np.clip((cos - NAPE_FROM_COS) / NAPE_FADE_COS, 0.0, 1.0)
              * np.clip((1.0 - t) / NAPE_END_FADE, 0.0, 1.0))
    return weight.astype(np.float32)


def _patch_place_weight(landmarks: Any, inside: np.ndarray, places: list[str],
                        side: float = 1.0) -> np.ndarray | None:
    """0..1 per texel: how well it lies on one of the places (side: on the
    flank whose x has this sign), or None when no place was read or the
    template has no feet to measure the trunk by."""
    feet = {name: landmarks.world(name) for name in ("paw_lf", "paw_rf", "paw_lb", "paw_rb")
            if name in landmarks.points}
    if not places or len(feet) < 4:
        return None
    position = landmarks.position
    front = float(np.mean([feet["paw_lf"][1], feet["paw_rf"][1]]))
    back = float(np.mean([feet["paw_lb"][1], feet["paw_rb"][1]]))
    along = (position[..., 1] - front) / (back - front)
    trunk = inside & (along >= 0.0) & (along <= 1.0)
    if not trunk.any():
        return None
    z = position[..., 2]
    low, high = np.percentile(z[trunk], [10, 90])
    height = np.clip((z - low) / max(high - low, 1e-6), 0.0, 1.0)
    weight = np.zeros(inside.shape, dtype=np.float32)
    for place in places:
        (centre, sigma), (a, b) = PATCH_PLACE_SHAPE[place]
        lengthwise = np.exp(-0.5 * ((along - centre) / sigma) ** 2)
        upright = np.exp(-0.5 * ((height - a) / -b) ** 2) if b < 0 else np.clip((height - a) / (b - a), 0.0, 1.0)
        if place == "side":
            upright = upright * np.clip(0.5 + side * position[..., 0] / PATCH_SIDE_FADE_M, 0.0, 1.0)
        weight = np.maximum(weight, (lengthwise * upright).astype(np.float32))
    return weight


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


def smooth_fields_on_surface(fields: list[np.ndarray], position: np.ndarray, valid: np.ndarray,
                             sigma_m: float, stride: int = SOFT_LIGHT_STRIDE) -> list[np.ndarray]:
    """smooth_on_surface for several fields over one set of samples: the
    gaussian weights are built once as a sparse matrix and applied to each."""
    from scipy import sparse
    from scipy.spatial import cKDTree

    if not fields:
        return []
    height, width = fields[0].shape
    rows, cols = np.nonzero(valid[::stride, ::stride])
    rows, cols = rows * stride, cols * stride
    points = position[rows, cols].astype(np.float64)
    tree = cKDTree(points)
    pairs = tree.sparse_distance_matrix(tree, 2.5 * sigma_m, output_type="coo_matrix")
    weights = sparse.coo_matrix((np.exp(-pairs.data ** 2 / (2.0 * sigma_m * sigma_m)), (pairs.row, pairs.col)),
                                shape=pairs.shape).tocsr()
    weights = weights + sparse.identity(len(points), format="csr")   # the zero distances
    norm = np.asarray(weights.sum(axis=1)).ravel()
    all_rows, all_cols = np.nonzero(valid)
    distance, nearest = tree.query(position[all_rows, all_cols].astype(np.float64), k=4, workers=-1)
    near_w = 1.0 / np.maximum(distance, 1e-5)
    near_w /= near_w.sum(axis=1, keepdims=True)
    out = []
    for field in fields:
        smoothed = (weights @ field[rows, cols].astype(np.float64)) / norm
        full = np.zeros((height, width), dtype=np.float32)
        full[all_rows, all_cols] = np.sum(near_w * smoothed[nearest], axis=1).astype(np.float32)
        out.append(full)
    return out


def _not_outward(landmarks, jitter: np.ndarray | None) -> np.ndarray:
    """0 on the legs' outer faces (facing away from the midline more than
    UNDERSIDE_OUTWARD, within UNDERSIDE_LEG_ZONE of the body's length from
    the front or the hind feet), 1 elsewhere - the belly between the legs
    keeps its sides, the light line a side view shows."""
    side = np.sign(landmarks.position[..., 0])
    outward = side * landmarks.normal[..., 0]
    signed = (UNDERSIDE_OUTWARD - outward) / 0.15
    if jitter is not None:
        signed = signed + 0.3 * jitter
    keep = np.clip(signed + 0.5, 0.0, 1.0)
    feet = [landmarks.points.get(k) for k in ("paw_lf", "paw_rf", "paw_lb", "paw_rb")]
    if any(p is None for p in feet):
        return keep.astype(np.float32)
    front = float(np.mean([landmarks.world("paw_lf")[1], landmarks.world("paw_rf")[1]]))
    hind = float(np.mean([landmarks.world("paw_lb")[1], landmarks.world("paw_rb")[1]]))
    reach = UNDERSIDE_LEG_ZONE * (hind - front)
    y = landmarks.position[..., 1]
    ramp = max(0.1 * (hind - front), 1e-3)
    legs = np.maximum(np.clip((front + reach - y) / ramp + 0.5, 0.0, 1.0),
                      np.clip((y - (hind - reach)) / ramp + 0.5, 0.0, 1.0))
    return (1.0 - legs * (1.0 - keep)).astype(np.float32)


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
    if overlay == "grizzle":
        # a steel-grey saddle's tan hairs (lite.steel_saddle): the dark
        # brindle's mottle, in the base colour, on the grey regions only
        if masks is None or landmarks is None or landmarks.position is None:
            return composed, None
        grey = np.zeros(gray.shape, dtype=np.float32)
        for entry in appearance.get("region_colors") or []:
            mask = masks.get(entry.get("region")) if str(entry.get("color", "")).lower() == "gray" else None
            if mask is not None:
                grey = np.maximum(grey, mask)
        base = str(appearance.get("base_color") or "tan").lower()
        if not grey.any() or base not in FUR_COLORS:
            return composed, None
        weight = (_brindle_mottle(gray.shape, direction, valid, seed, landmarks)
                  * (GRIZZLE_OPACITY / BRINDLE_MOTTLE_OPACITY) * grey)
        return _blend(composed, paint_region(gray, base, None, palette), weight), overlay
    if overlay not in ("brindle", "merle"):
        return composed, None
    second = appearance.get("second_color") or ("black" if overlay == "brindle" else "chocolate")
    if second not in FUR_COLORS:
        second = "black"
    base = str(appearance.get("base_color") or "").lower()
    if overlay == "brindle" and color_family(base) == "dark" and color_family(second) == "dark":
        # a dark brindle is black over brown: black stripes on a black base
        # drew nothing, and the brindle puppy came out plain black (shelter
        # puppy 8); its brown shows as the stripes instead - dark brown, tan
        # stripes made a tiger
        second = "chocolate"
    shape = gray.shape
    if (overlay == "brindle" and color_family(base) == "dark" and landmarks is not None
            and landmarks.position is not None):
        weight = _brindle_mottle(shape, direction, valid, seed, landmarks, floor=BRINDLE_MOTTLE_FLOOR)
        if appearance.get("gray_lab") is not None or any(
                str(entry.get("color", "")).lower() == "gray" for entry in appearance.get("region_colors") or []):
            weight = weight * BRINDLE_MOTTLE_ON_GREY
        second = BRINDLE_MOTTLE_COLOUR
    elif overlay == "brindle" and direction is not None:
        # stripes run across the hair: smear noise along the perpendicular
        across = np.stack([-direction[..., 1], direction[..., 0]], axis=-1)
        stripes = anisotropic_noise(shape, across, seed + 7, streak=BRINDLE_STREAK_PX, grain=BRINDLE_GRAIN_PX)
        field = (stripes - stripes.min()) / max(float(stripes.max() - stripes.min()), 1e-6)
        cover, soft, opacity = BRINDLE_COVER, BRINDLE_SOFT * float(field[valid].std()), BRINDLE_OPACITY
    else:
        field = noise_field(shape, seed + 11)
        cover, soft, opacity = MERLE_COVER, 0.06, 1.0
    if overlay != "brindle" or color_family(base) != "dark" or landmarks is None or landmarks.position is None:
        threshold = float(np.quantile(field[valid], 1.0 - cover))
        weight = np.clip((field - threshold) / max(soft, 1e-6) + 0.5, 0.0, 1.0) * valid * opacity
    # a brindle's or a merle's white markings carry no stripes: drawn over
    # everything, the stripes showed only on the white chest and legs of a
    # dark brindle and made zebra legs (shelter puppy, user). By the painted
    # lightness, so a white blaze placed as a marking is spared too.
    lightness = 116.0 * np.cbrt(np.clip(composed @ np.asarray(LUMA, dtype=np.float32), 0.0, 1.0)) - 16.0
    low, high = OVERLAY_SKIP_L
    weight = weight * np.clip((high - lightness) / (high - low), 0.0, 1.0)
    return _blend(composed, paint_region(gray, second, None, palette), weight), overlay


def _brindle_mottle(shape: tuple[int, int], direction: np.ndarray | None, valid: np.ndarray, seed: int,
                    landmarks, floor: float = 0.0) -> np.ndarray:
    """The dark brindle's lighter hairs as a 0..1 weight (BRINDLE_MOTTLE_*):
    soft clouds on the surface, streaked across the hair, over `floor` of
    the opacity everywhere."""
    rng = np.random.default_rng(seed + 7)
    surface = landmarks.valid if landmarks.valid is not None else valid
    smooth = smooth_on_surface(rng.random(shape).astype(np.float32), landmarks.position, surface,
                               BRINDLE_MOTTLE_SIGMA_M)
    inside = surface & valid
    field = (smooth - float(smooth[inside].mean())) / max(float(smooth[inside].std()), 1e-6)
    weight = np.clip(0.5 + field * 0.6, 0.0, 1.0)
    if direction is not None:
        across = np.stack([-direction[..., 1], direction[..., 0]], axis=-1)
        hair = np.clip(0.5 + anisotropic_noise(shape, across, seed + 9, streak=12, grain=1.0) * 0.35, 0.0, 1.0)
        weight = weight * ((1.0 - BRINDLE_MOTTLE_STREAK) + BRINDLE_MOTTLE_STREAK * hair)
    weight = floor + (1.0 - floor) * weight
    return (weight * BRINDLE_MOTTLE_OPACITY * valid).astype(np.float32)


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

    if TEMPLATE_SHADING:
        pass  # template_shading(), below, once the islands are loaded
    elif SYNTHETIC_HAIR:
        # even, drawn fur: hair streaks of one amplitude everywhere, no
        # painted edges, no shading blotches
        hair0 = (anisotropic_noise(gray.shape, direction0, seed + 9, streak=8, grain=1.2)
                 if direction0 is not None else jitter)
        hair = np.clip(1.0 + SYNTHETIC_HAIR_SPREAD * np.clip(hair0, -2.5, 2.5) / 2.5, 0.0, 2.0)
        if TEMPLATE_DETAIL > 0:
            # ... and the template's own strands and curls on top (TEMPLATE_DETAIL)
            valid_path = (Path(geometry_dir) if geometry_dir is not None else template_dir / "geometry") / "valid.npy"
            island = (np.load(valid_path) > 0.5 if valid_path.exists()
                      else np.ones(gray.shape, dtype=bool))
            if island.shape == gray.shape:
                hair = hair * template_detail(base_image, island) ** TEMPLATE_DETAIL
        gray = normalize_brightness(np.clip(hair, 0.0, 2.0)).astype(np.float32)
    else:
        gray = (float(gray.mean()) + (gray - float(gray.mean())) * HAIR_CONTRAST).astype(np.float32)
    masks = MaskLibrary(masks_dir, base_image.size, jitter=np.clip(jitter, -2.5, 2.5).astype(np.float32))
    if geometry_dir is None and Path("output/geometry/geometry.json").exists():
        geometry_dir = Path("output/geometry")
    if geometry_dir is not None and (Path(geometry_dir) / "valid.npy").exists():
        masks.valid = np.load(Path(geometry_dir) / "valid.npy") > 0.5
    if TEMPLATE_SHADING:
        island = (masks.valid if masks.valid is not None and masks.valid.shape == gray.shape
                  else np.ones(gray.shape, dtype=bool))
        gray = template_shading(base_image, island, masks.extra("protect"))

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
            landmarks.face_scale = _face_scale(landmarks, template_dir)
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
        # the mask's own soft fringe cut to its middle, and feathered inward:
        # at half weight several texels out, and with the blur's outer half,
        # the template's fur came back round every protected edge - a light
        # ring round a black dog's nose (the chihuahua, user) and the grey eye
        # shadow round its eyes
        core = np.clip((protect - 0.35) / 0.3, 0.0, 1.0)
        core = core * core * (3.0 - 2.0 * core)
        if landmarks is not None:
            core = _clear_lid_band(core, composed, template, landmarks)
        weight = np.minimum(_blur(core, PROTECT_FEATHER_PX / 2.0), core)
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


def _clear_lid_band(core: np.ndarray, composed: np.ndarray, template: np.ndarray, landmarks) -> np.ndarray:
    """The protect weight with the band along the inside of each eye
    (EYE_LID_BAND_PX) cleared where the coat is darker than the template."""
    if landmarks.position is None or landmarks.position.shape[:2] != core.shape or EYE_LID_BAND_PX <= 0:
        return core
    inside = core > 0.5
    band = inside & ~ndimage.binary_erosion(inside, iterations=EYE_LID_BAND_PX)
    near = np.zeros_like(inside)
    for name in landmarks.anchor_points("eye"):
        near |= np.linalg.norm(landmarks.position - landmarks.world(name), axis=-1) < EYE_LID_REACH_M
    luma = np.asarray(LUMA, dtype=np.float32)
    darker = (composed @ luma) < (template @ luma)
    return np.where(band & near & darker, 0.0, core).astype(np.float32)


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
    report_out: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Image.Image:
    """The coat texture as an image: compose_texture without the normal map
    and the report, for callers that only want the picture (`report_out`,
    when given, is filled with the report).

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
    result = compose_texture(
        base_texture_path, masks_dir, appearance, debug_dir=debug_dir, palette=palette,
        face_layer=face_layer, **kwargs,
    )
    if report_out is not None:
        report_out.update(result.report)
    return result.image


__all__ = [
    "FACE_REGIONS", "FUR_COLORS", "MaskLibrary", "TextureResult", "brightness_map",
    "compose_body", "compose_texture", "feature_core", "keep_features", "noise_field",
    "paint_region", "seed_for", "synthesize_texture",
]
