"""The lite path: read little, build the coat from the masks, refine once.

    0. prepare        photo gate + cut-out (prepare_photo.py)              gpt-4o 2
    1. record         size class, base colour (metadata.py)                0
    1b. model         ONE call x3 with the grey model sheet: the outline
                      (morphs) and the closest model (body_shape.py), with
                      the DINOv2 looks (template_pick.py, local)            gpt-4o 3
    2. reading        ONE call x3 samples over the photo: pattern, two
                      colours, eye colour, sock height, up to four stand-out
                      features (pattern_classify); then the head box
                      (face_crop.py) and ONE call x3 over the close-up alone
                      for the cheek, brow, muzzle-top and inner-ear colours
                      (face_colours.py); ONE call x3 over the photo for
                      the chest, belly and feet colours (body_colours.py)   gpt-4o 3+1+3+3
    3. layout         pattern defaults + features -> ten region colours     0
    4. mask build     synthesize_texture: the template's fur under the
                      region colours, hard mask edges (texture.py)          0
    5. refine         one image-model edit asked to blend the boundaries
                      like fur. REFINE_MODE "model": its colours and hairs
                      are kept, each region's mean put back on the build's
                      inside that region, and a region it repainted takes
                      the build's colour under its hairs (restore_in_region);
                      "build": the build's colours with only the edit's
                      blend shape and hairs. Dropped when the edit moves
                      the atlas or the eyes                                  gpt-image 1
    6. finish         template eyes/nose/pads back, iris recoloured,
                      seams smoothed (uv_paint.finish_painted)              0
    7. output         GLB with the base colour replaced, renders, log       0

WHY
---
The whole-atlas repaint (api.run --mode paint) asked the image model to
COLOUR a grey atlas from a guide, and the model left grey wherever it did not
paint - a stain that took a second roll, a combine over cached rolls and a
critic loop to hide. The mask build gets every colour where it belongs but
its edges read as stickers. Here the mask build goes first and the model's
one job is to make an already-correct coat look like fur: soften the
boundaries into interleaving hairs, add the shading real fur has. Refining
is a job image models do reliably; colouring from grey was not.

Per animal: five vision calls and one image edit, against twenty-two and
one to three on the paint path.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from .assets import BLENDER, BuiltAsset, template_paths
from .fur import to_linear, to_srgb
from .glb import animation_names, extract_base_texture, replace_base_texture
from .metadata import process_metadata
from .palette import FACE_CUT_MIN_SHARE, FACE_DARK_MIN_SHARE, FACE_GRAY_MIN_L
from .behavior import resolve_behavior
from .pipeline import PipelineResult, _json, default_transport, load_dotenv
from .regions import REGIONS
from .schemas import PetMetadata
from .texture import MaskLibrary, synthesize_texture
from .uv_paint import (
    COLOUR_WORDS,
    EDIT_SIZE,
    FEATURE_SSIM_FLOOR,
    LAYOUT_IOU_FLOOR,
    ScriptedImageTransport,
    feature_alignment,
    finish_painted,
    layout_iou,
)

CACHE_DIR = Path("output/cache/lite")
# how much of the sable tipping a sable's "dark" muzzle top gets
DARK_BACK_SABLE = 0.55
SABLE_MUZZLE_SHARE = 0.5

# What the model's edit is used for: the SHAPE of the blending at each
# boundary, and nothing else.
#
# Measured on the black-and-tan: the model returns the atlas with a global
# tone compression (whites at ~77 instead of 92), and inside a large light
# region it paints "shading" that is really a repaint (54% of the chest's
# interior came back dark). Two attempts to keep the model's colours and
# correct them failed in ways the user could see: per-region mean offsets
# through a blurred field bled a white region's lift 30 texels into the
# black beside it (a grey band along every mask edge) and lifted the
# repainted dark into grey blotches; a global tone curve posterised the
# greys because it cannot tell a darkened white from a lightened black.
#
# So the colours are the mask build's, everywhere, with its fur grain. In a
# band of BAND_PX texels along each mask edge, the model's texel is read as
# a MIX of the two colours that meet there - how far it lies from this
# side's colour toward the other side's, in the model's own colour space -
# and the build's texel is moved toward the other side's build colour by
# that share. Outside the band the build stands as it is. Nothing here can
# produce a colour that is not one of the two the build put at that edge.
BAND_PX = 40
# a mix below this is this side's colour outright; the model's fur strokes
# are never exactly the region mean, and without a floor every texel
# would move a little
MIX_FLOOR = 0.05
# the mix field is spread this much (texels, gaussian sigma) so a single
# hair the model drew is a soft stroke on the build's grain, not a hard line
MIX_SPREAD_PX = 2.5
MIX_GAMMA = 1.0
# The band, floor and gamma were 24 / 0.15 / 1.8 at first, which kept the
# build's edges nearly as crisp as a mask: the user judged the earlier
# version that kept the model's colours (and its 30-texel soft transitions)
# more natural. Wider and fuller, plus the build's own all-edge feather
# (FEATHER_ALL, below), the transitions match that softness while every
# colour is still one of the two the build put at the edge.
# The mask build feathers EVERY colour edge over texture.FEATHER_ALL_PX
# (14) before the blend: a guaranteed soft transition where the model drew
# nothing, on top of which its hairs are laid.
FEATHER_ALL = False
# Which of the two refine results is used:
#   "model"  the model's colours and hairs everywhere (what the user judged
#            most natural), with its two defects repaired: each region's
#            mean colour is put back on the build's INSIDE that region only
#            (restore_in_region - no field blurred across edges, so no grey
#            band), and a region whose interior the model repainted (its
#            share over REPAINT_SHARE beyond KEEP_LAB from the build - the
#            white chest came back 68% dark) takes the build's colour there
#            under the model's hairs. The model's own edge blends stay.
#   "build"  the build's colours with the model's blend shape and hairs
#            (blend_from_edit + transfer_detail): guaranteed colours, less
#            of the model's character.
REFINE_MODE = "model"
KEEP_LAB = 18.0
REPAINT_SHARE = 0.25
REGION_BAND_PX = 20
OFFSET_BLUR_PX = 24
# below DARK_L (build L, 6px blurred) the confined offset is used; the
# smeared one takes over across DARK_RAMP above it
DARK_L = 25.0
DARK_RAMP = 20.0
# The model's HAIRS are kept even though its colours are not: its texture
# above DETAIL_SIGMA texels (individual strokes) is laid over the build's
# colour field, whose own grain below that scale it replaces. Everything the
# model got wrong - the darkened whites, the shading, the moved boundary -
# lives below that scale and is discarded with the low frequencies. The L
# detail is clamped to +-DETAIL_CLAMP so a stray stroke cannot punch a hole
# in a white; a and b detail is halved (colour noise reads as dirt).
DETAIL_SIGMA = 4.0
DETAIL_CLAMP = 14.0
DETAIL_CHROMA = 0.5
# A bad roll of the edit is rolled again. The model's edits vary: one roll
# blends the chest's edge into interleaving hairs, the next darkens the
# whole chest by 32 L and draws a grey halo round it, and the restore then
# keeps that halo (the user: "the fur went strange at the chest's edge and
# on the face"). A roll is bad when a region had to be given back (its
# interior repainted, see REPAINT_SHARE) or a region's mean moved more than
# EDIT_MAX_OFFSET_L. Up to EDIT_ROLLS rolls, each cached in its own slot so
# a re-run costs nothing; the roll with the lowest score is kept.
EDIT_ROLLS = 2
EDIT_MAX_OFFSET_L = 20.0

DETAIL_RIM_PX = 14.0
# The build holds the whole DETAIL_RIM_PX band and then fades into the
# model's texture over this many more texels. A linear ramp from the rim
# (build at 0 px, model at 14 px) left the model's outline half in at 7 px:
# measured on the chest of one edit the outline ran -10..-15 L over the
# first 17 px and faded by 25 px, so the rim band came out 12 L below the
# chest interior - a light stripe along every seam of the chest.
DETAIL_RIM_FEATHER_PX = 12.0


def _rim_weight(valid: np.ndarray) -> np.ndarray:
    """0 on an island's rim (the build), 1 past DETAIL_RIM_PX +
    DETAIL_RIM_FEATHER_PX from it (the model)."""
    distance = ndimage.distance_transform_edt(valid)
    return np.clip((distance - DETAIL_RIM_PX) / DETAIL_RIM_FEATHER_PX, 0.0, 1.0).astype(np.float32)
def _now() -> float:
    return time.perf_counter()


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# --------------------------------------------------------------------------
# the refinement prompt
# --------------------------------------------------------------------------

# how wide the model is told to interleave two colours' hairs. "normal" is
# the shiba's finger's width; "narrow" is for a generated template whose
# atlas is many small islands, where a finger-wide interleave drawn island
# by island read as torn edges where the white chest meets the black neck
BLEND_WIDTH_TEXT = {
    "normal": "over about a finger's width the hairs of the two colours interleave",
    "narrow": "over only a few hairs' width - a pencil line, no wider - the hairs of the two colours interleave",
}


def template_settings(masks_dir: str | Path) -> dict[str, Any]:
    """output/templates/<slug>/template.json, written by new_template.py for
    a generated model; an empty dict for the shiba."""
    path = Path(masks_dir).parent / "template.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def refine_prompt(appearance: dict[str, Any], notes: list[str] | None = None,
                  blend_width: str = "normal") -> str:
    by_colour: dict[str, list[str]] = {}
    for entry in appearance.get("region_colors") or []:
        region, colour = entry.get("region"), str(entry.get("color", "")).lower()
        if region in REGIONS and colour:
            by_colour.setdefault(colour, []).append(region.replace("_", " "))
    placement = "; ".join(f"{COLOUR_WORDS.get(c, c)} on the {', '.join(r)}" for c, r in by_colour.items())
    lines = [
        "Image 1 is the UV texture atlas of a 3D dog model, already painted with the correct coat colours "
        "in the correct places. Image 2 is a photograph of the real dog.",
        "Repaint image 1 so the coat looks like real fur as in the photograph, and output the whole atlas.",
        f"Keep the colour placement of image 1 exactly: {placement}.",
        f"Where two colours meet, blend them the way fur does: {BLEND_WIDTH_TEXT.get(blend_width, BLEND_WIDTH_TEXT['normal'])}, "
        "with a soft irregular edge that follows the fur direction, never a hard line or a stencil edge.",
        "Inside each area draw individual hairs following the grain already visible in image 1, with only "
        "the slight hair-to-hair variation of real fur. This is an UNLIT colour map: no shadows, no shading, "
        "no darkening toward edges or folds - every white area stays fully white to its edge, every black "
        "area fully black. The lighting is added later by the 3D renderer, not here.",
        "Do not add markings, stripes, spots or patches that image 1 does not have. Do not change which "
        "colour any area is, and do not move any boundary further than the blend width.",
        "Every island must stay exactly where it is, at the same size and shape, with the same padding "
        "between islands. Do not draw the dog; draw only the atlas. Leave the eyes, nose, mouth and paw "
        "pads exactly as they are.",
    ]
    if notes:
        lines.append("About the photograph: " + " ".join(notes))
    return " ".join(lines)


# --------------------------------------------------------------------------
# refine: one edit, three gates
# --------------------------------------------------------------------------

def _lowpass_on_islands(x: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian low-pass of `x` (h, w, 3) that only averages island texels:
    a plain blur pulls the padding past every island rim into the rim's
    colour, which showed on the mesh as a dark line along every UV seam
    (the chest's cross). Normalised convolution: blur(x * valid) / blur(valid)."""
    from scipy import ndimage

    weight = ndimage.gaussian_filter(valid.astype(np.float32), sigma)
    out = np.stack([ndimage.gaussian_filter(x[..., i] * valid, sigma) for i in range(3)], axis=-1)
    return out / np.maximum(weight, 1e-3)[..., None]


# a UV island under this many texels (~100 faces) is confetti (a claw, an
# eyelid shell, a seam fringe): the build's colour is kept there whatever the
# model returned. The finish's 5 px seam ring is wider than such an island,
# so it was blended whole with whatever lay across its seam (a white paw
# beside a black shard) and rimmed in grey
TINY_ISLAND_TEXELS = 6000
# ... and no thicker than this anywhere (texels from its rim): a piece of coat
# a few centimetres across (20 such on the jindo, up to 4,300 texels) is no
# confetti, the seam ring reaches only its rim, and filled from the surface
# round it - the nearest texels of other charts, cell by cell - one at the
# throat's black-and-white edge came out as crumpled shards of white in the
# black (user, 11-05: "번개무늬 남아있어 색깔 경계면에")
TINY_ISLAND_THICK_PX = 8.0

# A thin strip of one UV chart packed inside another: fewer than
# SLIVER_SAME_SHARE of the island texels on a ring SLIVER_RADIUS_PX round it
# lie on the same part of the dog (baked positions within SLIVER_NEAR_M).
# Every step that works in the atlas reads the other chart into it - the
# masks' edge jitter, the refine model, the seam pass's vertex samples. The
# jindo puppy 2's shoulder seam is a 2-6 texel strip wedged into the white
# bib, and it drew a grey zigzag down the black shoulder (shelter puppy 11,
# user: "앞다리 바깥에 번개무늬"). Such texels take the colour of the surface
# round them (_fill_from_surface) before the finish and again after it.
SLIVER_RADIUS_PX = 3
SLIVER_NEAR_M = 0.01
SLIVER_SAME_SHARE = 0.5


# an island whose mean L moved further than this from the build (or whose
# a/b moved further than ISLAND_DRIFT_AB) was recoloured, not refined
ISLAND_DRIFT_L = 18.0
ISLAND_DRIFT_AB = 12.0


# Dark fur the edit turned light: the build's colour back where its L* was
# under FLIP_DARK_L and the edit lifted it by more than FLIP_LIFT_L (grown by
# FLIP_GROW_PX and faded in, so no edge is left). The image model drew a pale
# ring round each black foreleg at a seam (shelter chihuahua 7, 6,000
# texels; user 11-07); on the other rebuilt chihuahuas it flipped 14-200
# texels, and 9,800 on chihuahua 8 - the edge of its white bib pushed a few
# texels into the black.
# ... and light fur it turned dark: under FLIP_LIGHT_L in the build, dropped
# by more than FLIP_DROP_L - a black blotch painted on chihuahua 8's white
# chest (two patches of 290 and 250 texels at L* 92 -> 29; user 11-07 "08
# 가슴에 검은거뭐냐"; the other rebuilt chihuahuas 6-200 texels).
FLIP_DARK_L = 30.0
# 15: the hairs the edit draws over black fur lift it by under 9 L* at the
# 99th percentile on five rebuilt chihuahuas; at 30 a line of pale
# strokes along chihuahua 8's black shoulder (lift 15-30) stayed
FLIP_LIFT_L = 15.0
FLIP_LIGHT_L = 70.0
# 25: past the 99.9th percentile of the edit's darkening of white fur (13-23)
FLIP_DROP_L = 25.0
FLIP_GROW_PX = 2


def _revert_dark_flips(build_lab: np.ndarray, merged_lab: np.ndarray,
                       valid: np.ndarray) -> tuple[np.ndarray, int]:
    """(merged LAB with the flips reverted, how many texels flipped)."""
    build_l, merged_l = build_lab[..., 0], merged_lab[..., 0]
    flip = valid & (((build_l < FLIP_DARK_L) & (merged_l - build_l > FLIP_LIFT_L))
                    | ((build_l > FLIP_LIGHT_L) & (build_l - merged_l > FLIP_DROP_L)))
    count = int(flip.sum())
    if not count:
        return merged_lab, 0
    grown = ndimage.binary_dilation(flip, iterations=FLIP_GROW_PX) & valid
    weight = np.clip(ndimage.gaussian_filter(grown.astype(np.float32), 1.0) * 1.5, 0.0, 1.0)
    weight = np.maximum(weight, grown.astype(np.float32))[..., None]
    return merged_lab * (1.0 - weight) + build_lab * weight, count


def _revert_drifted_islands(build_lab: np.ndarray, merged_lab: np.ndarray,
                            valid: np.ndarray, masks: MaskLibrary | None = None) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Cells are island x region: a hind leg painted dark sits on the same
    island as a body quarter, and the island's mean hides it, but the
    legs_lower cell of that island is 70 L off."""
    from scipy import ndimage

    labels, count = ndimage.label(valid)
    if count == 0:
        return merged_lab, []
    if masks is not None:
        order = [r for r in REGIONS if masks.get(r) is not None]
        region_id = np.zeros(valid.shape, dtype=np.int64)
        strength = np.zeros(valid.shape, dtype=np.float32)
        for index, region in enumerate(order, start=1):
            mask = masks.get(region)
            take = (mask > strength) & (mask > 0.5)
            region_id[take] = index
            strength = np.maximum(strength, mask)
        labels = labels * (len(order) + 1) + region_id
        labels[~valid] = 0
        labels = np.unique(labels, return_inverse=True)[1].reshape(valid.shape)
        count = int(labels.max())
    flat = labels.ravel()
    area = np.bincount(flat, minlength=count + 1).astype(np.float64)
    area[area == 0] = 1.0
    deltas = []
    for c in range(3):
        b = np.bincount(flat, weights=build_lab[..., c].ravel(), minlength=count + 1) / area
        m = np.bincount(flat, weights=merged_lab[..., c].ravel(), minlength=count + 1) / area
        deltas.append(m - b)
    d_l, d_a, d_b = deltas
    bad = ((np.abs(d_l) > ISLAND_DRIFT_L) | (np.hypot(d_a, d_b) > ISLAND_DRIFT_AB)) & (area >= 300)
    bad[0] = False
    if not bad.any():
        return merged_lab, []
    revert = bad[labels]
    out = np.where(revert[..., None], build_lab, merged_lab)
    report = [{"island": int(i), "texels": int(area[i]), "dL": round(float(d_l[i]), 1),
               "dab": round(float(np.hypot(d_a[i], d_b[i])), 1)} for i in np.nonzero(bad)[0]]
    report.sort(key=lambda r: -r["texels"])
    return out, report[:12]


# Shards at the UV islands' edges: a texel in an island's outer SPIKE_PX
# whose colour hardly any of its SPIKE_K nearest texels ON THE SURFACE share
# (under SPIKE_SHARE of them within SPIKE_LUMA sRGB luma) takes their median.
# The chihuahua template's fur cards at the back of the thigh are tagged
# half underside and half back, and on a black dog with a white belly they
# came out as white strokes on the black thigh and tail (shelter chihuahua
# 8, user 11-07: "팔이랑 꼬리쪽에 하얀색 조각"). Judged against the texel 3
# px in, in the atlas, instead, it copied the black across a chest/back
# boundary onto the white chest's edge and the white onto the black
# shoulder's - a black notch in the bib and white zigzags on the foreleg
# (user 11-08: "가슴에 검은색 부분이랑 오른쪽팔에 흰색 번개무늬"); at a colour
# boundary half the surface around a texel has its colour. The eyes, nose
# and pads (protect.png) and the placed markings are left alone.
SPIKE_PX = 3
SPIKE_LUMA = 60.0
SPIKE_K = 96
SPIKE_SHARE = 0.3


def _clean_spikes(image: Image.Image, masks_dir, valid: np.ndarray, position: np.ndarray | None,
                  keep: np.ndarray | None = None) -> tuple[Image.Image, int]:
    from scipy import ndimage
    from scipy.spatial import cKDTree

    tex = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    if position is None or valid.shape != tex.shape[:2] or position.shape[:2] != valid.shape:
        return image, 0
    edge = valid & ~ndimage.binary_erosion(valid, iterations=SPIKE_PX)
    protect_path = Path(masks_dir).parent / "protect.png"
    if protect_path.exists():
        protect = np.asarray(Image.open(protect_path).convert("L").resize(image.size)) > 127
        edge &= ~ndimage.binary_dilation(protect, iterations=2)
    if keep is not None and keep.shape == valid.shape:
        edge &= ~ndimage.binary_dilation(keep, iterations=2)
    if not edge.any():
        return image, 0
    luma = tex @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    grid = np.zeros_like(valid)
    grid[::2, ::2] = True
    rows, cols = np.nonzero(valid & grid)
    er, ec = np.nonzero(edge)
    _, near = cKDTree(position[rows, cols]).query(position[er, ec], k=SPIKE_K, workers=-1)
    around = luma[rows[near], cols[near]]                                   # (N, K)
    share = (np.abs(around - luma[er, ec][:, None]) < SPIKE_LUMA).mean(axis=1)
    spike = share < SPIKE_SHARE
    if not spike.any():
        return image, 0
    tex[er[spike], ec[spike]] = np.median(tex[rows[near[spike]], cols[near[spike]]], axis=1)
    # ... and the padding from the islands again, for the texels the
    # renderer reads just past each edge
    nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    tex = tex[nearest[0], nearest[1]]
    return Image.fromarray(np.clip(tex, 0, 255).astype(np.uint8)), int(spike.sum())


# The unbaked gutter texels the mesh's triangles sample - every generated
# template has triangles whose corners lie on two UV charts, or whose inside
# crosses the gap between a chart's own spikes (~50,000 texels a template) -
# take the colour of the surface each one shows: the mean of GUTTER_NEAREST
# baked texels nearest that point in 3D. Padded from the nearest chart in
# the atlas instead, the chihuahua's thigh and tail triangles read the white
# belly's chart packed beside them, white strokes on a black thigh (shelter
# chihuahua 8, 176 texels off the surface by over 60 luma; user 11-07: "진도
# 처럼 기본템플릿 문제 같아"). The same fix ar_pipeline/fill_gutters made in
# the jindo puppy's base texture (11-06), here on every dog's coat.
GUTTER_NEAREST = 8
GUTTER_SAMPLES_PER_TEXEL = 2.0
GUTTER_MIN_SAMPLES = 64
_GUTTER_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def _gutter_samples(mesh, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(texels (N, 2) row/col, 3D points (N, 3)): the unbaked texels the
    triangles sample, each with the point it shows (the first sample)."""
    height, width = valid.shape
    empty = ~valid
    px = mesh.uvs[mesh.triangles].astype(np.float64)           # (T, 3, 2) col, row
    pts = mesh.positions[mesh.triangles].astype(np.float64)    # (T, 3, 3)
    counts = np.clip((np.ptp(px, axis=1).max(1) * GUTTER_SAMPLES_PER_TEXEL) ** 2,
                     GUTTER_MIN_SAMPLES, 4096).astype(int)
    hits, where = [], []
    for count in np.unique(counts):
        sel = np.nonzero(counts == count)[0]
        n = int(np.ceil(np.sqrt(2 * count)))
        a, b = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        keep = a + b <= 1.0
        bary = np.stack([1 - a[keep] - b[keep], a[keep], b[keep]], -1)
        p_uv = np.einsum("sk,tkc->tsc", bary, px[sel])
        col = np.clip(np.round(p_uv[..., 0]).astype(int), 0, width - 1)
        row = np.clip(np.round(p_uv[..., 1]).astype(int), 0, height - 1)
        on = empty[row, col]
        if on.any():
            hits.append(np.stack([row[on], col[on]], -1))
            where.append(np.einsum("sk,tkc->tsc", bary, pts[sel])[on])
    if not hits:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 3))
    hits, where = np.concatenate(hits), np.concatenate(where)
    _, first = np.unique(hits[:, 0] * width + hits[:, 1], return_index=True)
    return hits[first], where[first]


def _gutters_from_surface(image: Image.Image, mesh, position: np.ndarray, valid: np.ndarray,
                          key: str) -> tuple[Image.Image, int]:
    from scipy.spatial import cKDTree

    if valid.shape != (image.size[1], image.size[0]) or position.shape[:2] != valid.shape:
        return image, 0
    cache_key = (key, valid.shape)
    if cache_key not in _GUTTER_CACHE:
        _GUTTER_CACHE[cache_key] = _gutter_samples(mesh, valid)
    texels, where = _GUTTER_CACHE[cache_key]
    if not len(texels):
        return image, 0
    from scipy import ndimage

    tex = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    rows, cols = np.nonzero(valid)
    _, near = cKDTree(position[rows, cols]).query(where, k=GUTTER_NEAREST, workers=-1)
    tex[texels[:, 0], texels[:, 1]] = tex[rows[near], cols[near]].mean(axis=1)
    # ... and the rest of the gutter from its nearest coloured texel, these
    # included: the filter reads a 2x2 block round each sample, and the ring
    # next to a recoloured texel kept the belly chart's white (as
    # fill_gutters pads the template)
    rest = ~valid
    rest[texels[:, 0], texels[:, 1]] = False
    idx = ndimage.distance_transform_edt(rest, return_distances=False, return_indices=True)
    tex[rest] = tex[idx[0][rest], idx[1][rest]]
    return Image.fromarray(np.clip(tex + 0.5, 0, 255).astype(np.uint8)), int(len(texels))


# The head's final colours made consistent over the SURFACE: its many small
# UV islands (lids, whisker pads, nose fringe, the ear cups) were painted
# island by island and came out as white shards and bluish streaks along
# the muzzle (user on demo e08: "why is the face broken"). The low
# frequencies are averaged in 3D (sigma HEAD_LOW_SIGMA_M) with the hair kept,
# and texels far off their 3D surroundings (HEAD_SHARD_L) take them. The
# eyes, nose and pads (protect.png) are left alone.
HEAD_RADIUS_M = 0.14
HEAD_LOW_SIGMA_M = 0.004
HEAD_WIDE_SIGMA_M = 0.006
HEAD_SHARD_L = 45.0


def _surface_channels(fields: np.ndarray, position: np.ndarray, use: np.ndarray, sigma: float,
                      stride: int = 2) -> np.ndarray:
    from scipy.spatial import cKDTree

    rows, cols = np.nonzero(use[::stride, ::stride])
    rows, cols = rows * stride, cols * stride
    points = position[rows, cols].astype(np.float64)
    values = fields[rows, cols].astype(np.float64)
    tree = cKDTree(points)
    smoothed = np.empty_like(values)
    for index, near in enumerate(tree.query_ball_point(points, r=2.5 * sigma, workers=-1)):
        near = np.asarray(near)
        w = np.exp(-np.sum((points[near] - points[index]) ** 2, axis=1) / (2.0 * sigma * sigma))
        smoothed[index] = w @ values[near] / w.sum()
    out = fields.astype(np.float32).copy()
    ur, uc = np.nonzero(use)
    distance, nearest = tree.query(position[ur, uc].astype(np.float64), k=4, workers=-1)
    w = 1.0 / np.maximum(distance, 1e-5)
    out[ur, uc] = np.einsum("nk,nkc->nc", w, smoothed[nearest]) / w.sum(axis=1)[:, None]
    return out


def _clean_head(image: Image.Image, masks_dir, valid: np.ndarray,
                keep: np.ndarray | None = None) -> tuple[Image.Image, int]:
    from scipy import ndimage

    position = _surface_position(masks_dir)
    points_path = Path(masks_dir).parent / "landmarks.json"
    if position is None or not points_path.exists():
        return image, 0
    points = json.loads(points_path.read_text(encoding="utf-8"))["points"]
    if "left_eye" not in points or "right_eye" not in points:
        return image, 0
    eye = (np.asarray(points["left_eye"]["world"]) + np.asarray(points["right_eye"]["world"])) / 2.0
    protect_path = Path(masks_dir).parent / "protect.png"
    protect = np.zeros(valid.shape, dtype=bool)
    if protect_path.exists():
        protect = np.asarray(Image.open(protect_path).convert("L").resize(image.size)) > 127
        protect = ndimage.binary_dilation(protect, iterations=2)
    # the build's placed markings (a white blaze on a dark crown) are not
    # shards: the shard test painted shelter puppy 13's blaze brown
    if keep is not None and keep.shape == valid.shape:
        protect |= ndimage.binary_dilation(keep, iterations=2)
    use = valid & ~protect & (np.linalg.norm(position - eye[None, None, :], axis=-1) < HEAD_RADIUS_M)
    if use.sum() < 1000:
        return image, 0
    tex = np.asarray(image.convert("RGB"), dtype=np.float32)
    weight = ndimage.gaussian_filter(valid.astype(np.float32), 2.0)
    low = np.stack([ndimage.gaussian_filter(tex[..., c] * valid, 2.0) for c in range(3)], -1) / np.maximum(weight, 1e-4)[..., None]
    even = _surface_channels(low, position, use, HEAD_LOW_SIGMA_M)
    wide = _surface_channels(tex, position, use, HEAD_WIDE_SIGMA_M)
    out = tex + (even - low) * use[..., None]
    luma = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    shard = use & (np.abs(out @ luma - wide @ luma) > HEAD_SHARD_L)
    shard = ndimage.binary_dilation(shard, iterations=1) & use
    w = (np.clip(ndimage.gaussian_filter(shard.astype(np.float32), 1.0) * 2.0, 0.0, 1.0) * use)[..., None]
    out = out * (1.0 - w) + wide * w
    nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    out = np.clip(out, 0, 255)[nearest[0], nearest[1]]
    return Image.fromarray(out.astype(np.uint8)), int(shard.sum())


def _surface_position(masks_dir) -> np.ndarray | None:
    from .assets import geometry_dir_for
    path = geometry_dir_for(masks_dir) / "position.npy"
    return np.load(path) if path.exists() else None


def _fill_from_surface(image: np.ndarray, fill: np.ndarray, valid: np.ndarray,
                       position: np.ndarray | None, exclude: np.ndarray | None = None) -> np.ndarray:
    """`image` (h, w, c) with the `fill` texels taken from the nearest other
    texels on the dog's SURFACE (3D, inverse-distance over the 6 nearest).
    The confetti islands (eyelid shells, claws, fur-clump fringes) used to
    take the build's colour, which since the build is feathered and the
    refine blends differs from the finished colour around them: they came
    out as white shards on the face (user on demo e08: "the face is
    broken"). None position falls back to the image as it is. `exclude`:
    texels that give no colour (the protected pads, eyes and nose - the
    pads beside a filled toe turned a white paw grey)."""
    if position is None or not fill.any():
        return image
    from scipy.spatial import cKDTree

    source = valid & ~fill
    if exclude is not None:
        source &= ~exclude
    rows, cols = np.nonzero(source[::2, ::2])
    rows, cols = rows * 2, cols * 2
    tree = cKDTree(position[rows, cols].astype(np.float64))
    fr, fc = np.nonzero(fill)
    distance, nearest = tree.query(position[fr, fc].astype(np.float64), k=6, workers=-1)
    weight = 1.0 / np.maximum(distance, 1e-5)
    values = image[rows, cols].astype(np.float64)
    out = image.copy()
    out[fr, fc] = (np.einsum("nk,nkc->nc", weight, values[nearest]) / weight.sum(axis=1)[:, None]).astype(image.dtype)
    return out


# The fine detail a filled island keeps: its own lightness against its own
# blur at this sigma (texels), clipped. A paw cut into 13 small islands
# (the maltese's left forepaw, half its texels under TINY_ISLAND_TEXELS) took
# one flat colour from around it and lost its toes - a smooth white block
# beside three toed paws (user on shelter maltese 7 in 3D). A wrongly
# coloured shard is flat in itself, so it still takes its surroundings'
# colour; its own crevices and claws come back.
ISLAND_DETAIL_SIGMA = 4.0
ISLAND_DETAIL_CLIP = (0.5, 1.5)


def _keep_island_detail(filled: np.ndarray, before: np.ndarray, fill: np.ndarray, valid: np.ndarray,
                        lab: bool = False) -> np.ndarray:
    """`filled` with the `fill` texels' fine lightness from `before`: L*
    added back (lab) or the RGB scaled by the luma ratio."""
    from scipy import ndimage

    if not fill.any():
        return filled
    light = before[..., 0].astype(np.float64) if lab else before.astype(np.float64).mean(-1)
    weight = ndimage.gaussian_filter(valid.astype(np.float64), ISLAND_DETAIL_SIGMA)
    broad = ndimage.gaussian_filter(light * valid, ISLAND_DETAIL_SIGMA) / np.maximum(weight, 1e-3)
    out = filled.copy()
    if lab:
        out[..., 0][fill] = (filled[..., 0][fill] + (light - broad)[fill]).astype(filled.dtype)
        return out
    ratio = np.clip(light / np.maximum(broad, 1.0), *ISLAND_DETAIL_CLIP)
    out[fill] = np.clip(filled[fill].astype(np.float64) * ratio[fill][:, None], 0, 255).astype(filled.dtype)
    return out


def _atlas_slivers(valid: np.ndarray, position: np.ndarray | None) -> np.ndarray:
    """The island texels on a thin strip packed inside another chart
    (SLIVER_*); none without a position map."""
    if position is None or position.shape[:2] != valid.shape:
        return np.zeros(valid.shape, dtype=bool)
    same = np.zeros(valid.shape, dtype=np.float32)
    total = np.zeros(valid.shape, dtype=np.float32)
    r = SLIVER_RADIUS_PX
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if max(abs(dy), abs(dx)) != r:
                continue
            there = np.roll(valid, (dy, dx), (0, 1))
            apart = np.linalg.norm(position - np.roll(position, (dy, dx), (0, 1)), axis=-1)
            same += there & (apart < SLIVER_NEAR_M)
            total += there
    return valid & (total > 0) & (same < SLIVER_SAME_SHARE * total)


def _tiny_islands(valid: np.ndarray, max_texels: int, max_thick: float | None = None) -> np.ndarray:
    """Islands under `max_texels` and no more than `max_thick` texels from
    their rim anywhere (TINY_ISLAND_THICK_PX)."""
    from scipy import ndimage

    labels, count = ndimage.label(valid)
    if count == 0:
        return np.zeros(valid.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    small = sizes < max_texels
    small[0] = False
    thick = TINY_ISLAND_THICK_PX if max_thick is None else max_thick
    if thick is not None:
        depth = ndimage.maximum(ndimage.distance_transform_edt(valid), labels, np.arange(count + 1))
        small &= np.asarray(depth) <= thick
    return small[labels]


def _lab_of(image: Image.Image) -> np.ndarray:
    from .projection import linear_to_lab
    return linear_to_lab(np.clip(to_linear(np.asarray(image.convert("RGB"), dtype=np.float32)), 1e-4, 1.0))


def _region_drift(a: np.ndarray, b: np.ndarray, masks: MaskLibrary, valid: np.ndarray) -> dict[str, float]:
    """How far each region's mean LAB moved from `a` to `b`."""
    out: dict[str, float] = {}
    for region in REGIONS:
        mask = masks.get(region)
        if mask is None:
            continue
        inside = (mask > 0.7) & valid
        if inside.sum() < 200:
            continue
        out[region] = round(float(np.linalg.norm(b[inside].mean(axis=0) - a[inside].mean(axis=0))), 1)
    return out


def blend_from_edit(build_lab: np.ndarray, painted_lab: np.ndarray, masks: MaskLibrary,
                    valid: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """The mask build, its edges blended the way the model's edit blended
    them. Returns the merged LAB and a small report.

    For every texel in the band along a mask edge:
      here   the region the texel belongs to; other: the nearest texel of a
             different region (a distance transform over the region map)
      m_here, m_other   the model's mean colour over the interior of each
             (its own tone, so its darkened white is still "its white")
      mix    where the model's texel lies on the line from m_here to
             m_other, clipped to [0, 1]; below MIX_FLOOR it is 0
      out    build(here texel) + mix * (build colour of the other texel -
             build(here texel))
    The band's outer edge is feathered so the blend fades into the build."""
    from scipy import ndimage

    h, w = valid.shape
    region_id = np.full((h, w), -1, dtype=np.int16)
    order = [r for r in REGIONS if masks.get(r) is not None]
    strength = np.zeros((h, w), dtype=np.float32)
    for index, region in enumerate(order):
        mask = masks.get(region)
        take = (mask > strength) & (mask > 0.5)
        region_id[take] = index
        strength = np.maximum(strength, mask)
    known = (region_id >= 0) & valid

    # the colour edges: a texel whose 4-neighbour is another region ON THE
    # SAME ISLAND. An island rim (region against padding) is not a colour
    # edge - the region across the padding is not adjacent on the mesh, and
    # reaching for it painted white spokes fanning into the black
    edge = np.zeros((h, w), dtype=bool)
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        shifted = np.roll(region_id, (dr, dc), axis=(0, 1))
        shifted_known = np.roll(known, (dr, dc), axis=(0, 1))
        edge |= known & shifted_known & (shifted != region_id)
    to_edge = ndimage.distance_transform_edt(~edge)
    band = known & (to_edge <= BAND_PX)
    interior = known & (to_edge > BAND_PX)
    means = np.zeros((len(order), 3), dtype=np.float32)
    for index in range(len(order)):
        sel = interior & (region_id == index)
        if sel.sum() < 200:
            sel = known & (region_id == index)
        means[index] = painted_lab[sel].mean(axis=0) if sel.any() else 0.0

    # for each band texel, the nearest texel of a DIFFERENT region
    rows, cols = np.nonzero(band)
    other_r = np.empty_like(rows)
    other_c = np.empty_like(cols)
    other_r[:] = rows
    other_c[:] = cols
    for index in range(len(order)):
        sel = band & (region_id == index)
        if not sel.any():
            continue
        foreign = known & (region_id != index)
        idx = ndimage.distance_transform_edt(~foreign, return_distances=False, return_indices=True)
        pick = sel[rows, cols]
        other_r[pick] = idx[0][rows[pick], cols[pick]]
        other_c[pick] = idx[1][rows[pick], cols[pick]]

    here_id = region_id[rows, cols]
    other_id = region_id[other_r, other_c]
    m_here = means[here_id]
    m_other = means[other_id]
    axis = m_other - m_here
    length = np.maximum((axis * axis).sum(axis=1), 1e-3)
    mix = ((painted_lab[rows, cols] - m_here) * axis).sum(axis=1) / length
    mix = np.clip((mix - MIX_FLOOR) / (1.0 - MIX_FLOOR), 0.0, 1.0)
    # same region on both ends, or the other texel further away than the
    # band is wide (it lies across padding, not across this edge): no blend
    reach = np.hypot(other_r - rows, other_c - cols)
    mix[(here_id == other_id) | (reach > 1.5 * BAND_PX)] = 0.0
    # taper with distance from the edge: the model draws long single hairs
    # across a boundary, and carried over at full strength onto the build's
    # fur they came out as hard one-texel lines fanning into the other
    # colour. Tapered and slightly spread they read as hairs again.
    fade = np.clip(1.0 - to_edge[rows, cols] / BAND_PX, 0.0, 1.0)
    # a partial mix is compressed (mix^MIX_GAMMA): the model's single grey
    # hair in the black (mix ~0.6) should be a faint hair here, not a 60%
    # white line; a texel it painted fully the other colour stays 1
    mix = (np.power(mix, MIX_GAMMA) * fade).astype(np.float32)
    field = np.zeros((h, w), dtype=np.float32)
    field[rows, cols] = mix
    field = ndimage.gaussian_filter(field, MIX_SPREAD_PX)
    mix = field[rows, cols]

    merged = build_lab.copy()
    target = build_lab[other_r, other_c]
    merged[rows, cols] = build_lab[rows, cols] + mix[:, None] * (target - build_lab[rows, cols])
    report = {"band_texels": int(band.sum()), "band_mean_mix": round(float(mix.mean()), 3),
              "band_blended_share": round(float((mix > 0.05).mean()), 3),
              "model_region_means_l": {order[i]: round(float(means[i][0]), 1) for i in range(len(order))}}
    return merged, report


def transfer_detail(base_lab: np.ndarray, painted_lab: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """`base_lab`'s colours (its content below DETAIL_SIGMA) with `painted_lab`'s
    hair detail (its content above it)."""
    from scipy import ndimage

    def lowpass(x: np.ndarray) -> np.ndarray:
        return _lowpass_on_islands(x, valid, DETAIL_SIGMA)

    detail = painted_lab - lowpass(painted_lab)
    detail[..., 0] = np.clip(detail[..., 0], -DETAIL_CLAMP, DETAIL_CLAMP)
    detail[..., 1:] *= DETAIL_CHROMA
    # along an island's rim the model draws its own outline, and carried
    # over as detail that outline showed on the mesh as a thin line at every
    # UV seam. Within DETAIL_RIM_PX of a rim the base's own grain is kept.
    rim = _rim_weight(valid)[..., None]
    base_low = lowpass(base_lab)
    out = base_low + rim * detail + (1.0 - rim) * (base_lab - base_low)
    # off the islands the base is kept as it was (padding is filled later)
    return np.where(valid[..., None], out, base_lab)


# restore_in_region's local guard: a blot this much darker than a light build
BLOT_LIGHT_L = 70.0
BLOT_DARKER_L = 35.0
BLOT_MIN_TEXELS = 150
BLOT_EDGE_MARGIN_PX = 20
# ... and on a mid-tone build (tan, fawn: L* over BLOT_MID_L) a blot at less
# than BLOT_MID_SHARE of the build's lightness, away from the build's dark
# parts: the model drew a black band round both wrists of a black-and-tan
# puppy's tan legs (L* 16 on 45, shelter puppy 9), which the light-only test
# let through.
BLOT_MID_L = 35.0
BLOT_MID_SHARE = 0.5


def restore_in_region(build_lab: np.ndarray, painted_lab: np.ndarray, masks: MaskLibrary,
                      valid: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """The model's atlas with each region's mean colour put back on the
    build's, applied inside that region only; regions the model repainted
    take the build's colour (low frequencies) under the model's hairs in
    their interior. Returns the LAB and a report."""
    from scipy import ndimage

    h, w = valid.shape
    order = [r for r in REGIONS if masks.get(r) is not None]
    region_id = np.full((h, w), -1, dtype=np.int16)
    strength = np.zeros((h, w), dtype=np.float32)
    for index, region in enumerate(order):
        mask = masks.get(region)
        take = (mask > strength) & (mask > 0.5)
        region_id[take] = index
        strength = np.maximum(strength, mask)
    known = (region_id >= 0) & valid
    edge = np.zeros((h, w), dtype=bool)
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        edge |= known & np.roll(known, (dr, dc), axis=(0, 1)) & (np.roll(region_id, (dr, dc), axis=(0, 1)) != region_id)
    to_edge = ndimage.distance_transform_edt(~edge)

    def lowpass(x: np.ndarray) -> np.ndarray:
        return _lowpass_on_islands(x, valid, DETAIL_SIGMA)

    # two offset fields from the same per-region deltas: one smeared across
    # mask edges (OFFSET_BLUR_PX, the look the user preferred: a light
    # region's lift brightens the tan beside it the way fur does) and one
    # confined to each region. Where the build is dark the confined one is
    # used - the smeared lift of a white region turned the black beside it
    # into a grey band; between light regions the smeared one is used.
    field_hard = np.zeros((h, w, 3), dtype=np.float64)
    field_num = np.zeros((h, w, 3), dtype=np.float64)
    field_den = np.zeros((h, w), dtype=np.float64)
    offsets: dict[str, list[float]] = {}
    repainted: dict[str, float] = {}
    for index, region in enumerate(order):
        own = region_id == index
        inside = own & valid
        if inside.sum() < 200:
            continue
        interior = inside & (to_edge > REGION_BAND_PX)
        ref = interior if interior.sum() > 200 else inside
        delta = (build_lab[ref].mean(axis=0) - painted_lab[ref].mean(axis=0)).astype(np.float64)
        offsets[region] = [round(float(d), 1) for d in delta]
        if interior.sum() > 200:
            share = float((np.linalg.norm(painted_lab[interior] - build_lab[interior], axis=-1) > KEEP_LAB).mean())
        else:
            share = 0.0
        repainted[region] = round(share, 3)
        field_hard += (ndimage.gaussian_filter(own.astype(np.float32), 1.5) * own)[..., None] * delta
        mask = masks.get(region).astype(np.float64)
        field_num += mask[..., None] * delta
        field_den += mask
    field_blur = field_num / np.maximum(field_den, 1e-3)[..., None]
    # blurred over the islands only: a plain gaussian averaged the padding's
    # zero offset into the last ~2 blur radii of every island, so a white
    # region's lift faded to nothing toward each UV seam while the seam's
    # own DETAIL_RIM_PX band kept the build's full brightness - a dark
    # trough beside every seam (the chest rim came out 12 L below the
    # chest interior) that showed on the mesh as a line along the seam
    island = valid.astype(np.float64)
    island_blur = np.maximum(ndimage.gaussian_filter(island, OFFSET_BLUR_PX), 1e-3)
    field_blur = np.stack([ndimage.gaussian_filter(field_blur[..., i] * island, OFFSET_BLUR_PX) / island_blur
                           for i in range(3)], axis=-1)
    light = np.clip((ndimage.gaussian_filter(build_lab[..., 0], 6.0) - DARK_L) / DARK_RAMP, 0.0, 1.0)[..., None]
    out = painted_lab.astype(np.float64) + light * field_blur + (1.0 - light) * field_hard

    # a region the model repainted: the build's colour under the model's
    # hairs in its interior; the band along its edges stays as painted
    fixed = None
    given_back: list[str] = []
    for index, region in enumerate(order):
        if repainted.get(region, 0.0) <= REPAINT_SHARE:
            continue
        given_back.append(region)
        own = region_id == index
        if fixed is None:
            detail = np.clip(painted_lab - lowpass(painted_lab), -DETAIL_CLAMP, DETAIL_CLAMP)
            detail[..., 1:] *= DETAIL_CHROMA
            fixed = (lowpass(build_lab) + detail).astype(np.float64)
        core = (np.clip((to_edge - REGION_BAND_PX * 0.5) / (REGION_BAND_PX * 0.5), 0.0, 1.0) * own)[..., None]
        out = out * (1 - core) + fixed * core

    # a dark blot the model drew on a light part - too small for the region
    # share above (a black patch on one white hind foot, eval demo e08):
    # the build's colour under the model's hairs there. Dark on the build
    # (markings, patches) is left alone.
    # Only away from the build's dark parts: along a dark/light edge the
    # model's darker fur is its blend of the edge, and restoring it put the
    # build's hard edge back on every boundary (user: "the colour edges look
    # unnatural, they didn't before" - e08 had 86 such restores)
    dark_build = ndimage.binary_dilation(build_lab[..., 0] < BLOT_LIGHT_L - 20.0, iterations=BLOT_EDGE_MARGIN_PX)
    blot = (valid & ~dark_build & (build_lab[..., 0] > BLOT_LIGHT_L)
            & (out[..., 0] < build_lab[..., 0] - BLOT_DARKER_L))
    near_dark = ndimage.binary_dilation(build_lab[..., 0] < BLOT_MID_L - 10.0, iterations=BLOT_EDGE_MARGIN_PX)
    blot |= (valid & ~near_dark & (build_lab[..., 0] > BLOT_MID_L) & (build_lab[..., 0] <= BLOT_LIGHT_L)
             & (out[..., 0] < build_lab[..., 0] * (1.0 - BLOT_MID_SHARE)))
    labels, count = ndimage.label(blot)
    blots = 0
    if count:
        sizes = ndimage.sum(blot, labels, index=np.arange(1, count + 1))
        big = np.isin(labels, 1 + np.flatnonzero(sizes >= BLOT_MIN_TEXELS))
        if big.any():
            blots = int((sizes >= BLOT_MIN_TEXELS).sum())
            if fixed is None:
                detail = np.clip(painted_lab - lowpass(painted_lab), -DETAIL_CLAMP, DETAIL_CLAMP)
                detail[..., 1:] *= DETAIL_CHROMA
                fixed = (lowpass(build_lab) + detail).astype(np.float64)
            grow = ndimage.binary_dilation(big, iterations=3)
            weight = np.clip(ndimage.gaussian_filter(grow.astype(np.float32), 2.0) * 1.5, 0.0, 1.0)[..., None]
            out = out * (1 - weight) + fixed * weight

    # along island rims the model draws its own outline - a dark band 10-15
    # texels wide (the chest rim came back at L 29 against 87) - which
    # showed on the mesh as a line along every UV seam, the chest's cross.
    # Within DETAIL_RIM_PX of a rim the texel is the build's, then feathered
    # over DETAIL_RIM_FEATHER_PX.
    rim = _rim_weight(valid)[..., None]
    out = rim * out + (1.0 - rim) * build_lab
    out = np.where(valid[..., None], out, painted_lab)
    return out.astype(np.float32), {"region_offsets": offsets, "interior_repainted": repainted,
                                    "regions_given_back": given_back, "dark_blots_restored": blots}


async def refine_coat(
    transport, coat: Image.Image, photo_path: str | Path, appearance: dict[str, Any],
    valid: np.ndarray, masks_dir: str | Path, notes: list[str] | None = None,
    use_cache: bool = True, cache_dir: str | Path = CACHE_DIR,
) -> tuple[Image.Image, bool, dict[str, Any]]:
    """The mask build refined by the image model. Returns (texture, refined,
    report): `refined` False means a gate failed and the texture is the mask
    build itself."""
    settings = template_settings(masks_dir)
    prompt = refine_prompt(appearance, notes, blend_width=str(settings.get("blend_width", "normal")))
    pet_id = str(appearance.get("pet_id") or "pet")
    small = coat.convert("RGB").resize((1024, 1024), Image.LANCZOS)
    small_bytes = _png(small)
    key = hashlib.sha256(prompt.encode("utf-8") + hashlib.sha256(small_bytes).digest()).hexdigest()[:16]
    cache_dir = Path(cache_dir)
    masks = MaskLibrary(masks_dir, coat.size)
    build_lab = _lab_of(coat)
    surface_position = _surface_position(masks_dir)
    photo_bytes: bytes | None = None

    async def one_roll(slot: int) -> tuple[Image.Image, dict[str, Any]]:
        nonlocal photo_bytes
        cached = cache_dir / (f"{pet_id}-{key}.png" if slot == 0 else f"{pet_id}-{key}-r{slot + 1}.png")
        roll: dict[str, Any] = {"cache": str(cached)}
        if use_cache and cached.exists():
            roll["source"] = "cache"
            return Image.open(cached).convert("RGB"), roll
        if photo_bytes is None:
            photo = Image.open(photo_path).convert("RGB")
            if max(photo.size) > 1024:
                scale = 1024 / max(photo.size)
                photo = photo.resize((int(photo.size[0] * scale), int(photo.size[1] * scale)), Image.LANCZOS)
            photo_bytes = _png(photo)
        data = await transport.edit(prompt=prompt, references=[small_bytes, photo_bytes], size=EDIT_SIZE)
        painted = Image.open(io.BytesIO(data)).convert("RGB")
        roll["source"] = "model"
        if use_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            painted.save(cached)
        return painted, roll

    def judge(painted: Image.Image, roll: dict[str, Any]) -> tuple[np.ndarray | None, float]:
        """The restored LAB and the roll's score (lower is better); None
        when a gate rejects the roll outright."""
        iou = layout_iou(painted, valid, coat)
        roll["layout_iou"] = round(iou, 3)
        if iou < LAYOUT_IOU_FLOOR:
            roll["rejected"] = f"layout IoU {iou:.2f} < {LAYOUT_IOU_FLOOR}: the model did not keep the atlas"
            return None, float("inf")
        alignment = feature_alignment(painted, coat, masks_dir)
        roll["feature_ssim"] = None if alignment is None else round(alignment, 3)
        if alignment is not None and alignment < FEATURE_SSIM_FLOOR:
            roll["rejected"] = f"feature SSIM {alignment:.2f} < {FEATURE_SSIM_FLOOR}: eyes/nose moved"
            return None, float("inf")
        painted_lab = _lab_of(painted.resize(coat.size, Image.LANCZOS))
        roll["drift_raw"] = _region_drift(build_lab, painted_lab, masks, valid)
        if REFINE_MODE == "model":
            merged_lab, info = restore_in_region(build_lab, painted_lab, masks, valid)
        else:
            merged_lab, info = blend_from_edit(build_lab, painted_lab, masks, valid)
            merged_lab = transfer_detail(merged_lab, painted_lab, valid)
        roll["mode"] = REFINE_MODE
        roll.update(info)
        # islands too small to carry a picture take the build as they are:
        # the image model paints a confetti island (a few texels of paw,
        # eyelid or seam fringe) as a grey shard, and on the mesh those
        # shards read as bright angular flecks all over a dark coat
        tiny = _tiny_islands(valid, TINY_ISLAND_TEXELS)
        if tiny.any():
            merged_lab = _keep_island_detail(_fill_from_surface(merged_lab, tiny, valid, surface_position),
                                             merged_lab, tiny, valid, lab=True)
            roll["tiny_islands_from_surface"] = int(tiny.sum())
        # an island the model recoloured wholesale (one hind leg painted
        # dark grey while the other three stayed white) hides inside the
        # per-region mean, which spans all four legs: islands whose mean
        # moved further than ISLAND_DRIFT_L from the build take the build's
        # colour back, hairs and all
        merged_lab, drifted = _revert_drifted_islands(build_lab, merged_lab, valid, masks)
        if drifted:
            roll["islands_reverted"] = drifted
        merged_lab, flipped = _revert_dark_flips(build_lab, merged_lab, valid)
        if flipped:
            roll["dark_flips_reverted"] = flipped
        roll["colour_drift"] = _region_drift(build_lab, merged_lab, masks, valid)
        given_back = list(info.get("regions_given_back") or [])
        worst = max((abs(v[0]) for v in (info.get("region_offsets") or {}).values()), default=0.0)
        roll["score"] = round(100.0 * len(given_back) + worst, 1)
        roll["bad"] = bool(given_back) or worst > EDIT_MAX_OFFSET_L
        return merged_lab, roll["score"]

    report: dict[str, Any] = {"prompt": prompt, "rolls": []}
    best: tuple[float, np.ndarray, dict[str, Any]] | None = None
    for slot in range(max(EDIT_ROLLS, 1)):
        painted, roll = await one_roll(slot)
        merged_lab, score = judge(painted, roll)
        report["rolls"].append(roll)
        if merged_lab is not None and (best is None or score < best[0]):
            best = (score, merged_lab, roll)
        if merged_lab is not None and not roll["bad"]:
            break
    if best is None:
        report["rejected"] = report["rolls"][-1].get("rejected", "every roll rejected")
        return coat, False, report
    _, merged_lab, chosen = best
    report.update({k: v for k, v in chosen.items()})
    report["chosen_roll"] = report["rolls"].index(chosen) + 1

    from .projection import lab_to_linear
    restored = to_srgb(np.clip(lab_to_linear(merged_lab), 0, 1))
    return Image.fromarray((restored * 255).astype(np.uint8), mode="RGB"), True, report


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

def crown_on_skull(region_colors: list[dict[str, Any]], crown: str | None, base: str | None,
                   second: str | None) -> str | None:
    """The skull painted the crown's colour when the head close-up saw a
    coloured crown and the layout left the skull light: a white-and-tan irish
    puppy with a brown head came out all white (shelter puppy 13, user) - the
    layout puts the base on the skull, and the face call's crown answer was
    used only to demote a piebald. Returns the colour written, or None."""
    from .patterns import color_family

    words = {"tan": ("tan", "red", "fawn", "chocolate"), "dark": ("black", "chocolate"), "gray": ("gray",)}
    if crown not in words:
        return None
    skull = next((e for e in region_colors if e.get("region") == "skull"), None)
    if skull is None or color_family(str(skull.get("color", ""))) != "light":
        return None
    colour = next((c for c in (second, base) if c in words[crown]), words[crown][0])
    skull.update({"color": colour, "coverage": "full", "source": f"face:crown {crown}",
                  "note": f"the crown seen {crown} on the head close-up"})
    # ... and the brows and ear cups the layout or no_brows left light: a
    # piebald's cap is all three, and a red papillon's head came out with
    # white brow spots and white ear cups (shelter maltese 7, 11-07; the
    # shih tzu 5, the grey-headed poodle 8 and puppy 13 alike). A light
    # answer from the face call (white_brows, cream_ears ...) stays.
    for entry in region_colors:
        source = str(entry.get("source", ""))
        if (entry.get("region") in ("eyebrows", "ears_inner")
                and color_family(str(entry.get("color", ""))) == "light"
                and (source.startswith("pattern") or source == "feature:no_brows")):
            entry.update({"color": colour, "source": f"face:crown {crown}",
                          "note": f"like the crown seen {crown}"})
    return colour


def crown_is_light(answer: str | None, votes: dict[str, int] | None) -> bool:
    """The head close-up saw a light crown: answered white or cream, or
    every vote white or cream - split 2 / 2 the answer was "unknown", and a
    white maltese with tear-stained cheeks kept a tan cap (shelter maltese 2,
    11-07)."""
    votes = {k: int(v) for k, v in (votes or {}).items()}
    return answer in ("white", "cream") or (sum(votes.values()) >= 3
                                            and all(k in ("white", "cream") for k in votes))


CAP_REGIONS = ("skull", "eyebrows", "ears_inner")


def cap_off(layout: dict[str, str], base: str | None) -> list[str]:
    """The base colour, in place, on the regions of a coloured cap - the
    skull and the eyebrows and ear cups painted like it - of a layout whose
    crown the head close-up saw light. A dirty white maltese's nine layout
    samples split six ways (piebald, saddle, other, solid) and agreed region
    by region on a tan skull, brows and ears; the close-up saw the crown
    cream 2 / white 2 and the crown measured L* 93 (shelter maltese 2, 11-07;
    its first run read piebald 3/3 and was demoted). Returns the regions."""
    from .patterns import color_family

    cap = layout.get("skull")
    if not cap or not base or color_family(cap) == "light":
        return []
    regions = [r for r in CAP_REGIONS if layout.get(r) == cap]
    for region in regions:
        layout[region] = base
    return regions


def coloured_cheeks(region_colors: list[dict[str, Any]], answer: str | None) -> bool:
    """The face call saw tan or dark cheeks (neither names a feature: the
    layouts that have them paint them) under a coloured skull the layout
    gave light cheeks: a red papillon's cheeks read dark 3 / tan 2 came out
    white below its brown crown (shelter maltese 7, 11-07). cheeks_to_head
    then paints them the skull's colour."""
    from .patterns import color_family

    skull = next((e for e in region_colors if e.get("region") == "skull"), None)
    cheeks = next((e for e in region_colors if e.get("region") == "cheeks"), None)
    return (answer in ("tan", "dark") and skull is not None and cheeks is not None
            and color_family(str(skull.get("color", ""))) != "light"
            and color_family(str(cheeks.get("color", ""))) == "light")


def irish_back(appearance: dict[str, Any], measured: dict[str, Any], read_irish: bool) -> str | None:
    """The back painted the second colour on a piebald read as irish
    (irish_on_light) when the photo's back measures nearer that colour's
    ramp than the base's: "irish white/tan" 3/3 is both a white dog with a
    brown head and a brown dog with white trim, the samples' colour order
    does not say which. The brown shih tzu's back measured (42, 8, 22) and
    it came out white below a brown head, its patches dropped for a front
    view (shelter maltese 5, 11-07); the white puppy the rule was made for
    measures (71, 0, 1) (shelter puppy 13). Returns the colour written."""
    from .fur import hex_to_linear
    from .palette import FUR_PALETTE
    from .projection import linear_to_lab

    base, second = appearance.get("base_color"), appearance.get("second_color")
    lab = ((measured or {}).get("back") or {}).get("lab")
    entry = next((e for e in appearance.get("region_colors") or [] if e.get("region") == "back"), None)
    if (not read_irish or appearance.get("pattern") != "piebald" or not lab or entry is None
            or base not in FUR_PALETTE or second not in FUR_PALETTE or entry.get("color") != base):
        return None

    def mid(name: str) -> np.ndarray:
        return linear_to_lab(hex_to_linear(FUR_PALETTE[name][1])[None, None, :])[0, 0]
    back = np.asarray(lab, dtype=np.float32)
    if np.linalg.norm(back - mid(second)) >= np.linalg.norm(back - mid(base)):
        return None
    entry.update({"color": second, "source": "measured:irish_back",
                  "note": f"the back measured nearer {second} than {base}"})
    return str(second)


def saddle_back(appearance: dict[str, Any], measured: dict[str, Any]) -> str | None:
    """The back painted the base colour on a saddle whose back measures nearer
    the base's ramp than the saddle's: a white poodle with a grey head and
    ears, read "saddle white/gray" 3 of 6, came out with a grey blanket
    over a white body (shelter toy poodle 8, back measured (77, -4, 2): 17
    from white, 24 from grey). The saddles on file whose backs are their
    second colour measured nearer it (a steel-grey yorkie mix: 23 from grey,
    46 from fawn). Returns the colour written."""
    from .fur import hex_to_linear
    from .palette import FUR_PALETTE
    from .projection import linear_to_lab

    base, second = appearance.get("base_color"), appearance.get("second_color")
    lab = ((measured or {}).get("back") or {}).get("lab")
    entry = next((e for e in appearance.get("region_colors") or [] if e.get("region") == "back"), None)
    if (appearance.get("pattern") != "saddle" or not lab or entry is None or base not in FUR_PALETTE
            or second not in FUR_PALETTE or entry.get("color") != second):
        return None

    def mid(name: str) -> np.ndarray:
        return linear_to_lab(hex_to_linear(FUR_PALETTE[name][1])[None, None, :])[0, 0]
    back = np.asarray(lab, dtype=np.float32)
    if np.linalg.norm(back - mid(base)) >= np.linalg.norm(back - mid(second)):
        return None
    entry.update({"color": base, "source": "measured:saddle_back",
                  "note": f"the back measured nearer {base} than {second}"})
    return str(base)


def nape_marking(region_colors: list[dict[str, Any]], body_votes: dict[str, Any] | None) -> dict[str, Any] | None:
    """The head's colour down the back of the neck (texture._nape_weight)
    when every sample of the body call saw it there (body_colours.
    NAPE_ANSWERS head) on a dog whose skull is coloured over a light back -
    a parti whose brown head ran down its neck to the shoulders came out
    with a white neck (shelter maltese 7, user 11-07). Unanimous, and on a
    body the patches cover most of (patch_amount most): two white-necked
    dogs with coloured hanging ears came back head 2 / light 1 (eval e13,
    e15), two more head 3/3 - a black head over a white neck and collar, a
    brittany's orange ears over a white neck (shelter retrievers 10 and 7,
    patches large); the brown-necked ones were head 3/3 with patches most
    (maltese 7 twice, puppy 13). A coat coloured all over has nothing to
    add."""
    from .patterns import color_family

    votes = body_votes or {}
    samples = int(votes.get("samples") or 0)
    unanimous = (samples > 0 and int((votes.get("nape_votes") or {}).get("head", 0)) == samples
                 and votes.get("patch_amount") == "most")
    skull = next((e for e in region_colors if e.get("region") == "skull"), None)
    back = next((e for e in region_colors if e.get("region") == "back"), None)
    if (not unanimous or skull is None or back is None
            or color_family(str(skull.get("color", ""))) == "light"
            or color_family(str(back.get("color", ""))) != "light"):
        return None
    return {"type": "nape", "region": "back", "color": skull["color"], "size": "medium", "anchor": "none",
            "note": "the head's colour down the back of the neck"}


# Dark ears on a light coat. dark_ears (simple.FEATURES) puts the base colour
# in the cup, "the same as the head", which on a white or cream base is the
# white or cream the face call did not see: a cream chihuahua's brown ears
# came out cream, cup and back (shelter chihuahua 1), and a piebald's cups
# white in its black head (shelter retriever-type 10). Under a coloured head
# the cups take the head's colour; under a light head the ears themselves
# are coloured, the cup and the back of the ear (texture: the ears_outer
# source mask, a marking on the skull), in EARS_ON_LIGHT - the face call's
# "dark" is black, chocolate or grey, and the one such ear on file is a warm
# brown (L* 22, chroma 17 in the cup's shade, its rim lighter).
EARS_ON_LIGHT = "tan"


def dark_ears_on_light(region_colors: list[dict[str, Any]], features: list[str],
                       base: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """(marking for the backs of the ears or None, what changed or None);
    the cup's entry is recoloured in place."""
    from .patterns import color_family

    if "dark_ears" not in features or color_family(str(base or "")) != "light":
        return None, None
    entries = {e.get("region"): e for e in region_colors}
    skull, cup = entries.get("skull"), entries.get("ears_inner")
    if skull is None or cup is None:
        return None, None
    if color_family(str(skull.get("color", ""))) != "light":
        if cup.get("color") == skull.get("color"):
            return None, None
        cup.update({"color": skull["color"], "note": "dark ears: the head's colour",
                    "source": "feature:dark_ears under a coloured head"})
        return None, f"cups {skull['color']}"
    cup.update({"color": EARS_ON_LIGHT, "note": "dark ears on a light head",
                "source": "feature:dark_ears on a light head"})
    return ({"type": "ears", "region": "skull", "color": EARS_ON_LIGHT, "size": "medium", "anchor": "none",
             "note": "the backs of the dark ears"}, f"ears {EARS_ON_LIGHT}")


def irish_on_light(pattern: str, base: str | None, second: str | None) -> str:
    """The pattern an Irish reading stands for. Irish is a coloured dog with
    a white collar, chest and feet; read with a LIGHT base and a coloured
    second colour it is a white dog whose colour is on the head - a piebald.
    Laid out as Irish, the second colour went to the chest, belly and legs:
    a white puppy with a brown head came out with a brown chest and belly
    (shelter puppy 13, "irish white/tan" 2 of 3, photo chest white)."""
    from .patterns import color_family

    if (pattern == "irish" and color_family(str(base or "")) == "light"
            and second and color_family(str(second)) != "light"):
        return "piebald"
    return pattern


def cheeks_to_head(region_colors: list[dict[str, Any]]) -> str | None:
    """The cheeks painted the skull's colour (the face crop saw a dark face
    under a dark crown, palette.cheeks_from_photo). On a piebald the base is
    the white body, and "the base colour" painted a black-headed puppy's
    face white."""
    skull = next((e for e in region_colors if e.get("region") == "skull"), None)
    cheeks = next((e for e in region_colors if e.get("region") == "cheeks"), None)
    if skull is None or cheeks is None:
        return None
    cheeks.update({"color": skull.get("color"), "source": "face:crop dark",
                   "note": "a dark face in the head close-up"})
    for key in ("gradient_to", "gradient_anchor"):
        cheeks.pop(key, None)
    return str(skull.get("color"))


# dark fur on this share of the crown in the face crop (palette.face_dark)
# makes a brindle a dark one
DARK_BRINDLE_SHARE = 0.6
# ... and under this share a white blaze covers the head (texture "full")
BLAZE_FULL_DARK_SHARE = 0.6
# body_colours patch_amount -> the patches' size (texture.PATCH_AREA). large
# is patches bigger than the head, not more of the body: the medium area
PATCH_SIZE = {"small": "small", "large": "medium", "most": "large"}

# One colour named twice: a solid apricot poodle read cream 3/3 by the
# layout call and tan by the face (crown 5/5) and body (chest 2/3) calls came
# out a near-white body with an orange head and chest. Its back, chest,
# crown and legs all measured one light apricot (L* 46-69, b* 11-18), nearer
# the tan ramp than cream's (shelter toy poodle 5). On a solid light coat, a
# warm colour the face or body call gave some regions is the whole coat's
# when the back measures within SAME_COAT_DE of them and nearer that colour's
# ramp than the base's. A white dog whose shaded chest was called tan keeps
# its white: its back measures white. The other way round too: a near-white
# yellow Labrador read fawn by the layout call and cream by the face (crown,
# cheeks 5/5) and body (chest, belly, feet) calls came out a tan dog; its
# back measured (81, 4, 14), cream's (shelter retriever 5). A golden with a
# cream front keeps its gold: its back measures 25 dE off the cream parts.
# With no measurement (a dog seen through bars is not measured) the face
# call's crown stands witness instead, when every sample named it that one
# colour: a cream jindo behind cage bars, layout fawn 3/3, crown, cheeks
# and muzzle cream 5/5, chest, belly and feet cream 3/3, came out a yellow
# dog (shelter adult jindo 1). On file a solid coat's crown was unanimous
# against its layout base three times - the two measured (toy poodle 5,
# retriever 5) both one coat.
SAME_COAT_DE = 12.0


def one_coat_colour(appearance: dict[str, Any], measured: dict[str, Any],
                    crown_votes: dict[str, int] | None = None) -> list[str]:
    """Recolour, in place, the base-coloured layout regions of a solid coat
    to the one colour the calls gave its other regions - warm on a light
    base, light on a warm one - when the measurements say it is one colour
    (or, with no measurement, the face call's crown votes are unanimous for
    it); returns what changed."""
    from .fur import hex_to_linear
    from .palette import FUR_PALETTE
    from .patterns import color_family
    from .projection import linear_to_lab

    base = str(appearance.get("base_color") or "").lower()
    entries = appearance.get("region_colors") or []
    other = {"light": "point", "point": "light"}.get(color_family(base))
    if appearance.get("pattern") != "solid" or other is None:
        return []
    measured = measured if isinstance(measured, dict) else {}
    back = (measured.get("back") or {}).get("lab")
    called = {}
    for entry in entries:
        colour = str(entry.get("color", "")).lower()
        source = str(entry.get("source", ""))
        if color_family(colour) == other and source.startswith(("face:", "body:")):
            called.setdefault(colour, []).append(entry.get("region"))
    if len(called) != 1:
        return []
    (warm, regions), = called.items()
    votes = {k: int(v) for k, v in (crown_votes or {}).items()}
    total = sum(votes.values())
    top = max(votes, key=votes.get) if votes else ""
    if back:
        back = np.asarray(back, dtype=np.float32)

        def mid(name: str) -> np.ndarray:
            return linear_to_lab(hex_to_linear(FUR_PALETTE[name][1])[None, None, :])[0, 0]
        if np.linalg.norm(back - mid(warm)) >= np.linalg.norm(back - mid(base)):
            return []
        labs = [measured[r]["lab"] for r in regions if (measured.get(r) or {}).get("lab")]
        same = bool(labs) and float(np.linalg.norm(np.median(np.asarray(labs, dtype=np.float32), axis=0)
                                                   - back)) < SAME_COAT_DE
        # ... or, the other parts' readings off (a muddy chest, the nose
        # in the muzzle's cells), the crown: a cream maltese with golden
        # ears read tan / fawn / cream, the tie broken by its record's
        # 갈색, came out brown; its back measured (87, 4, 10), its cream
        # cheeks, muzzle and chest 54, 32 and 20, and the crown cream 4/5
        # (shelter maltese 4, 11-07)
        if not same and not (top == warm and votes[top] * 2 > total):
            return []
        witness = "measured" if same else "measured back, crown"
    else:
        if total < 3 or votes.get(top) != total or color_family(top) != other:
            return []
        witness = "crown"
    # ... the layout's base-coloured regions, the features that take the base
    # (no_brows' eyebrows, dark_ears' cups), and the base itself: kept fawn,
    # it was the coat under every edge - fawn brow spots on a cream jindo,
    # fawn where its light belly leaves the outer thigh, fawn at the top of
    # its legs - and the cream was mixed toward it (relative_cream), a beige
    # dog (shelter adult jindo 1, user: "01번 백구 다시 봐봐")
    from .simple import FEATURES
    base_features = {f"feature:{name}" for name, (_, roles) in FEATURES.items() if "base" in roles.values()}
    changed = []
    for entry in entries:
        source = str(entry.get("source", ""))
        if str(entry.get("color", "")).lower() == base and (source.startswith("pattern") or source in base_features):
            entry.update({"color": warm, "note": f"one coat: {warm} like {'/'.join(regions)}",
                          "source": f"{witness}:one_coat"})
            changed.append(f"{entry.get('region')}:{base}->{warm}")
    if changed:
        appearance["base_color"] = warm
        changed.append(f"base:{base}->{warm}")
    return changed



# A skull painted the crown's warm colour (crown_on_skull), or a piebald's
# cap, that the photo measures darker than this (CIE L*) is chocolate: a
# dark brown head read "tan" came out orange (shelter puppy 13, measured L 26).
CHOCOLATE_CROWN_MAX_L = 35.0
CHOCOLATE_CAP_SOURCES = ("face:crown", "pattern_default:piebald")


def chocolate_crown(region_colors: list[dict[str, Any]], measured: dict[str, Any],
                    markings: list[dict[str, Any]] | None = None) -> str | None:
    skull = next((e for e in region_colors if e.get("region") == "skull"), None)
    reading = (measured or {}).get("skull")
    if (skull is None or not str(skull.get("source", "")).startswith(CHOCOLATE_CAP_SOURCES)
            or str(skull.get("color")) not in ("tan", "red", "fawn") or not isinstance(reading, dict)
            or not reading.get("lab") or float(reading["lab"][0]) >= CHOCOLATE_CROWN_MAX_L):
        return None
    warm = skull["color"]
    skull["color"] = "chocolate"
    skull["note"] = f"{skull.get('note', '')}; measured L {reading['lab'][0]:.0f}".lstrip("; ")
    # ... and the rest of the cap read the same warm colour - the cheeks
    # (cheeks_vs_brows), the brows and the ear cups: an orange face, orange
    # brow spots and orange ears under a brown crown otherwise
    for entry in region_colors:
        if entry.get("region") in ("cheeks", "eyebrows", "ears_inner") and entry.get("color") == warm:
            entry["color"] = "chocolate"
    # ... and the body's patches and nape the same warm colour: one brown on
    # the dog, a dark brown head over orange patches otherwise (shelter
    # maltese 7, its saddle measured (35, 9, 15) beside its crown's (30, 4, 11))
    for marking in markings or []:
        if marking.get("type") in ("patches", "nape") and marking.get("color") == warm:
            marking["color"] = "chocolate"
    return "chocolate"


# A black saddle over a tan coat that the body call saw as dark-tipped hair
# (back_tipping clear, a majority) and the photo measures lighter than this
# (CIE L* of the back) is a steel-grey "blue and tan": a yorkshire-type mix
# with a grizzled grey back measured L* 32 and 35 in two runs came out with
# a jet-black blanket and a black muzzle (shelter maltese 10, user 11-07:
# "새까만 몸통, 검은 주둥이"). Black backs on file measured L* 0-17 on the
# black-and-tan saddles and solid blacks; the only saddle read with clear
# tipping and a black second colour is this dog.
STEEL_MIN_L = 25.0


def steel_saddle(appearance: dict[str, Any], measured: dict[str, Any],
                 tipping_votes: dict[str, int] | None) -> str | None:
    """Every black region and marking grey (gray_for_black) at the back's
    measured lightness (palette.gray_target, no lift), for a saddle read
    black over a lighter coat whose back was seen tipped and measures
    lighter than STEEL_MIN_L. Returns what changed, or None."""
    from .palette import gray_target

    votes = {k: int(v) for k, v in (tipping_votes or {}).items()}
    lab = ((measured or {}).get("back") or {}).get("lab")
    back = next((e for e in appearance.get("region_colors") or [] if e.get("region") == "back"), None)
    if (appearance.get("pattern") != "saddle" or appearance.get("second_color") != "black"
            or back is None or back.get("color") != "black" or not lab or float(lab[0]) < STEEL_MIN_L
            or votes.get("clear", 0) * 2 <= sum(votes.values())):
        return None
    changed = gray_for_black(appearance.get("region_colors") or [], appearance.get("markings") or [])
    appearance["second_color"] = "gray"
    appearance["overlay"] = "grizzle"     # the base's hairs through the grey (texture.GRIZZLE_*)
    appearance["gray_lab"] = gray_target(float(lab[0]), float(lab[1]), float(lab[2]), lift=0.0)
    return f"{changed} black entries grey at {appearance['gray_lab']} (back measured {[round(x, 1) for x in lab]})"


# Points the agreeing samples split evenly between tan and cream are a pale
# tan: the vote kept tan, and a black-and-tan chihuahua whose forelegs are a
# pale buff (L* 61, chroma 11 in its photo; the table's tan is L* 56,
# chroma 30) came out with orange boots (shelter chihuahua 8, tan 1 / cream
# 1; user 11-07 "8번 다리부분"). palette.pale_points mixes the points' ramp
# toward the table's cream. The only such split on file.
def pale_points(pattern: str | None, second: str | None, samples: list[dict[str, Any]] | None) -> bool:
    from collections import Counter

    if second != "tan":
        return False
    seconds = Counter(s.get("second_color") for s in samples or [] if s.get("pattern") == pattern)
    return bool(seconds.get("tan")) and seconds.get("tan") == seconds.get("cream")


def paws_split(body_votes: dict[str, Any] | None) -> bool:
    """Some sample of the body call gave the feet a colour that is not light."""
    said = (body_votes or {}).get("paws") or {}
    return any(int(n) > 0 for answer, n in said.items() if answer in ("dark", "tan"))


# A black-and-tan's cream front that measures warm is a pale tan. The body
# call's "cream" on a black coat with tan points is the table's near-white
# cream (palette.relative_cream mixes toward the base only when the base is
# warm), and a chihuahua whose chest and belly are a pale buff came out with
# a white bib (shelter chihuahua 6: chest L* 46, chroma 21, belly chroma 31,
# its tan cheeks L* 40, chroma 17). A cream chest or belly measured at least
# TAN_CREAM_MIN_CHROMA in a warm hue takes its cream from the points' colour
# instead (relative_cream, appearance["cream_from"]). The light chests and
# bellies of the other dark-coated dogs on file measured chroma 0.3-7.6,
# shaded or not.
TAN_CREAM_MIN_CHROMA = 15.0
TAN_CREAM_HUE = (30.0, 100.0)


def tan_cream(appearance: dict[str, Any], measured: dict[str, Any]) -> str | None:
    """Sets appearance["cream_from"] to the warm second colour and returns
    what was seen, or None."""
    from .palette import WARM_BASES
    from .patterns import color_family

    appearance.pop("cream_from", None)   # a reused appearance's, decided again here
    second = appearance.get("second_color")
    if color_family(str(appearance.get("base_color") or "")) != "dark" or second not in WARM_BASES:
        return None
    warm = []
    for entry in appearance.get("region_colors") or []:
        if entry.get("color") != "cream" or entry.get("region") not in ("chest", "underside"):
            continue
        lab = ((measured or {}).get(entry["region"]) or {}).get("lab")
        if not lab:
            continue
        chroma = float(np.hypot(lab[1], lab[2]))
        hue = float(np.degrees(np.arctan2(lab[2], lab[1])))
        if chroma >= TAN_CREAM_MIN_CHROMA and TAN_CREAM_HUE[0] <= hue <= TAN_CREAM_HUE[1]:
            warm.append(f"{entry['region']} C{chroma:.0f} h{hue:.0f}")
    if not warm:
        return None
    appearance["cream_from"] = second
    return f"cream from {second}: {', '.join(warm)}"


# A solid coat read black whose body measures charcoal: two of the skull,
# back and lower legs at least STEEL_MIN_L. A faded black poodle came out
# jet black where its photo is a dark charcoal grey (shelter toy poodle 10:
# L* 29, 28.5 and 31; its smoky crown was kept black by the black-coat rule
# above, made for a black puppy whose back measured L* 12). The black solids
# on file measured L* 0.3-15 on their backs. Painted grey at the back's
# lightness (palette.gray_target, no lift).
def faded_black(appearance: dict[str, Any], measured: dict[str, Any]) -> str | None:
    """Every black entry grey (gray_for_black) at the measured back's
    lightness; returns what changed, or None."""
    from .palette import gray_target

    if (appearance.get("pattern") != "solid" or appearance.get("base_color") != "black"
            or appearance.get("overlay")):
        return None
    lab = {k: ((measured or {}).get(k) or {}).get("lab") for k in ("skull", "back", "legs_lower")}
    light = [k for k, v in lab.items() if v and float(v[0]) >= STEEL_MIN_L]
    if len(light) < 2 or not lab["back"]:
        return None
    changed = gray_for_black(appearance.get("region_colors") or [], appearance.get("markings") or [])
    back = lab["back"]
    appearance["gray_lab"] = gray_target(float(back[0]), float(back[1]), float(back[2]), lift=0.0)
    return f"{changed} black entries grey at {appearance['gray_lab']} ({', '.join(light)} at L* {STEEL_MIN_L:.0f}+)"


def gray_for_black(region_colors: list[dict[str, Any]], markings: list[dict[str, Any]]) -> int:
    """Every black region and marking painted gray: the head close-up saw a
    grey crown, and the layout call has no word between black and white for
    a smoky coat (shelter puppy 14). Returns how many entries changed."""
    changed = 0
    for entry in [*region_colors, *markings]:
        if str(entry.get("color", "")).lower() == "black":
            entry["color"] = "gray"
            entry["note"] = f"{entry.get('note', '')}; gray crown on the close-up".lstrip("; ")
            changed += 1
    return changed


def upright_photo(path: Path, out_dir: Path) -> Path:
    """The photo with its EXIF rotation applied. Phone photos store the
    pixels sideways plus an orientation tag; rembg honours the tag and PIL's
    Image.open does not, so the cut-out's alpha came back transposed against
    the pixels (eval e02). Upright photos are returned as they are."""
    from PIL import ImageOps

    with Image.open(path) as image:
        if image.getexif().get(0x0112, 1) == 1:
            return path
        upright = ImageOps.exif_transpose(image).convert("RGB")
    target = out_dir / f"{path.stem}_upright.png"
    upright.save(target)
    return target


async def run_lite(
    record: dict[str, Any],
    photos: list[str | Path],
    out_dir: str | Path,
    transport=None,
    image_transport=None,
    refine: bool = True,
    render: bool = True,
    samples: int = 3,
    blender: str = BLENDER,
    prepare: bool = True,
    strict_photo: bool = True,
    reuse_appearance: str | Path | None = None,
    face_crop: bool = True,
    measure: bool = True,
    body_call: bool = True,
    shape_call: bool = True,
    read_shape: bool = False,
    fit_stop: bool = False,
    real_size: bool = True,
    pick_template: bool = True,
    gate_readings: list[dict[str, Any]] | None = None,
) -> PipelineResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    photos = [upright_photo(Path(p), out_dir) for p in photos]
    # the ORIGINAL frame for the colour measurement: its border is the grey
    # card for the white balance, and the cut-out replaces it with flat grey
    original_photo = photos[0]
    log: dict[str, Any] = {"mode": "lite", "steps": {}, "timings_s": {}}
    started = _now()
    from . import usage as ledger
    ledger.reset()

    # 0. prepare
    photo_notes: list[str] = []
    photo_quality: dict[str, Any] | None = None
    photo_bars = False
    photo_subject: tuple[int, int, int, int] | None = None   # the animal's box in the cut-out
    photo_animal: np.ndarray | None = None   # the animal in the original photo's pixels
    if prepare:
        from .prepare_photo import prepare_photo

        t = _now()
        gate_transport = None if reuse_appearance else (transport or default_transport())
        # gate_readings: /v1/fit already gated this photo (api/server.py), no second call
        prepared = await prepare_photo(gate_transport, photos[0], out_dir, species="dog", strict=strict_photo,
                                       gate_readings=gate_readings)
        photos = [prepared.path] + photos[1:]
        photo_notes = prepared.notes
        photo_quality = prepared.quality
        photo_bars = prepared.bars
        photo_subject = prepared.subject_box
        photo_animal = prepared.animal_mask
        log["steps"]["prepare"] = prepared.report()
        # a saturated backdrop's light in the fur: the calls read white as tan
        from .photo_colours import remove_backdrop_cast
        cast = remove_backdrop_cast(original_photo, photos[0], out_dir / "prepared_cast.png")
        if cast:
            photos = [out_dir / "prepared_cast.png"] + photos[1:]
            log["steps"]["prepare"]["backdrop_cast"] = cast
        log["timings_s"]["prepare"] = round(_now() - t, 2)
        q = prepared.quality or {}
        print(f"[lite] photo prepared: cut={prepared.cut}, coverage {prepared.coverage:.2f}, "
              + ("through bars, " if prepared.bars else "") +
              f"notes {prepared.notes or 'none'}, quality {q.get('level', '?')} "
              f"(short side {q.get('short_side_px', '?')}px, sharpness {q.get('sharpness', '?')}, "
              f"clipped {q.get('clipped_share', '?')})"
              + (f", REJECTED but running on: {prepared.rejected}" if prepared.rejected else ""))

    # 1. record
    meta = PetMetadata.model_validate(record)
    resolved = process_metadata(meta.pet_id, meta.species, meta.weight_kg, meta.coat_color)
    # 1b. which model: the outline call with the grey model sheet
    #     (body_shape.py, models=True, one call x3 - the same call the jindo's
    #     morphs read, made here for every dog) and the DINOv2 looks
    #     (template_pick.py, local), combined (template_pick.combine).
    #     --template (PET_TEMPLATE) skips both and keeps the old order.
    base_template = resolved.base_template
    pick: dict[str, Any] | None = None
    early_shape: tuple[dict[str, str], dict[str, Any]] | None = None
    if pick_template and not os.environ.get("PET_TEMPLATE"):
        from .template_pick import combine, template_file
        from .template_pick import pick_template as pick_by_looks
        t = _now()
        looks = None
        try:
            looks = pick_by_looks(original_photo, meta.weight_kg)
        except Exception as exc:  # noqa: BLE001 - no weights or no backbone: the vision call alone
            log["steps"]["template_looks_error"] = f"{type(exc).__name__}: {exc}"
        if shape_call and (not reuse_appearance or read_shape):
            from .body_shape import read_body_shape
            from .extract import photo_blocks
            try:
                early_shape = await read_body_shape(transport or default_transport(), photo_blocks(photos),
                                                    samples=samples, models=True)
            except Exception as exc:  # noqa: BLE001 - the looks alone pick
                log["steps"]["shape"] = {"error": f"{type(exc).__name__}: {exc}"}
        pick = combine(looks, early_shape[0] if early_shape else None, meta.weight_kg)
        base_template = template_file(pick["template"], resolved.size_class)
        log["steps"]["template_pick"] = {**pick, "looks_pick": looks}
        log["timings_s"]["template_pick"] = round(_now() - t, 2)
        print(f"[lite] template: {pick['template']} by {pick['by']}"
              + (f", {pick['match']} {pick['differences']}" if pick["match"] else "")
              + (", looks: none (no template)" if pick["no_template"] else "")
              + (", heavy puppy -> adult" if pick["heavy_puppy"] else "") + f" -> {base_template}")
    template_glb, masks_dir = template_paths(base_template)
    template_dir = masks_dir.parent
    log["steps"]["record"] = {"pet_id": resolved.pet_id, "base_color": resolved.base_color,
                              "size_class": resolved.size_class, "base_template": base_template}

    # 2. reading: one call, three samples
    t = _now()
    if reuse_appearance:
        from .regions import canonicalize_appearance
        appearance = canonicalize_appearance(json.loads(Path(reuse_appearance).read_text(encoding="utf-8")))
        log["steps"]["reading"] = {"reused_from": str(reuse_appearance)}
        # a reused colour reading with only the outline read fresh: the shape
        # call is the one call made (read_shape), ~$0.01 a dog
        from .shape import target_names
        if early_shape:
            # read above with the model sheet
            from .body_shape import shape_weights
            answers, shape_votes = early_shape
            appearance["shape"] = {"answers": answers, "weights": shape_weights(answers, shape_votes)}
            log["steps"]["shape"] = {**appearance["shape"], "votes": shape_votes}
        elif read_shape and target_names(template_glb):
            from .body_shape import read_body_shape, shape_weights
            from .extract import photo_blocks
            answers, shape_votes = await read_body_shape(transport or default_transport(), photo_blocks(photos),
                                                         samples=samples)
            appearance["shape"] = {"answers": answers, "weights": shape_weights(answers, shape_votes)}
            log["steps"]["shape"] = {**appearance["shape"], "votes": shape_votes}
    else:
        from .extract import photo_blocks
        from .pattern_classify import classify_pattern
        from .pattern_classify import describe as describe_decision
        from .simple import apply_features

        transport = transport or default_transport()
        blocks = photo_blocks(photos)
        if photo_notes:
            blocks = blocks + [{"type": "text", "text": "About the photograph: " + " ".join(photo_notes)}]
        decision = await classify_pattern(transport, blocks, template_dir / "anchors", resolved.base_color,
                                          samples=samples)
        features = list(decision.features)
        markings: list[dict[str, Any]] = []
        # the face's colours from the head close-up ALONE (no anchors, no
        # layout words): asked in the layout call they came back as whatever
        # the layout implies. One small call for the head box, then the
        # colour call x samples.
        face_log: dict[str, Any] = {"asked": False}
        # the colour of body patches survives a demotion from piebald: the
        # cap colour goes (e01), the patches keep theirs (eval v3 e15)
        patch_colour: str | None = None
        if face_crop:
            from .face_colours import SAMPLES as FACE_SAMPLES
            from .face_colours import read_face_colours
            from .face_crop import crop_block, crop_face
            from .pattern_classify import BROW_FEATURES, CHEEK_FEATURES, EAR_FEATURES, MUZZLE_FEATURES
            try:
                crop = await crop_face(photos[0], transport, subject=photo_subject)
                face_log["box"], face_log["frontal"], face_log["visible"] = list(crop.box), crop.frontal, crop.visible
                if crop.visible:
                    crop.image.save(out_dir / "face_crop.png")
                    cheeks, brows, muzzle, ears, blaze, votes = await read_face_colours(
                        transport, crop_block(crop), samples=max(samples, FACE_SAMPLES))
                    face_log.update({"asked": True, "cheeks": cheeks, "brows": brows, "muzzle": muzzle, "ears": ears,
                                     "blaze": blaze, "crown": votes.get("crown_answer"),
                                     "eye_ring": votes.get("eye_ring_answer"), "mask": votes.get("mask_answer"),
                                     "votes": votes})
                    # cheeks the colour of tan brow spots are the tan points,
                    # whatever the cheek question said: beside a black crown
                    # it read a puppy's tan cheeks cream and painted a white
                    # mask (shelter puppies 9, 10; face_colours CHEEK_BROW_ANSWERS)
                    if votes.get("cheeks_vs_brows_answer") == "same" and brows == "tan" and cheeks in ("cream", "white"):
                        face_log["cheeks_from_brows"] = f"cheeks {cheeks} -> tan (as deep as the brow spots)"
                        cheeks = face_log["cheeks"] = "tan"
                    # light cheeks under a dark crown, checked against this
                    # face's own crop (palette.cheeks_from_photo): a dark face,
                    # tan cheeks, or cream taking the light fur's colour
                    # (palette.cream_from_face)
                    if votes.get("crown_answer") == "dark":
                        from .palette import face_dark
                        face_log["face_dark"] = list(face_dark(crop.image))
                    # ... on a cut-out crop only, as the grey test: in an uncut
                    # crop the "light fur" was the white clothes and gloves round
                    # a black-and-tan puppy (shelter puppy 10: L* 69, a* b* 0)
                    crop_cut = (face_log.get("face_dark") or [None, None, 0.0])[2]
                    # ... and under a tan crown, by the light fur's share alone:
                    # a side-on head bowed to the floor read "white cheeks" 4/5
                    # off the white neck beside it (shelter maltese 7, light
                    # fur 6 % of the lower face; shelter puppy 13, brown cheeks
                    # by a white muzzle, 10-16 %; light cheeks on file 47-94 %).
                    # The light fur's colour is not asked: an urajiro's cream
                    # cheeks sit near the tan ramp
                    tan_crown = votes.get("crown_answer") == "tan" and cheeks in ("cream", "white")
                    if tan_crown:
                        from .palette import face_dark
                        crop_cut = face_dark(crop.image)[2]
                    light_cheeks = (votes.get("crown_answer") == "dark" or tan_crown) and cheeks in ("cream", "white")
                    if light_cheeks and crop_cut < FACE_CUT_MIN_SHARE:
                        face_log["face_light"] = {"skipped": f"uncut crop (cut {crop_cut})"}
                    elif light_cheeks:
                        from .palette import cheeks_from_photo, face_light
                        share, light_lab = face_light(crop.image)
                        verdict = cheeks_from_photo(cheeks, share, light_lab)
                        if tan_crown and verdict != "dark":
                            verdict = cheeks
                        face_log["face_light"] = {"share": share, "lab": light_lab, "cheeks": verdict}
                        if verdict != cheeks:
                            face_log["cheeks_from_photo"] = f"cheeks {cheeks} -> {verdict}"
                            cheeks = face_log["cheeks"] = verdict
                            if verdict == "tan":
                                features.append("tan_cheeks")
                            if verdict == "gray" and not face_log.get("gray_lab"):
                                from .palette import frost_lab
                                face_log["gray_lab"] = frost_lab(light_lab)
                        elif cheeks == "cream" and not tan_crown:
                            face_log["face_light_lab"] = light_lab
                    # a dark muzzle top on a coat that is not dark, from every
                    # sample only: the dark lips and nose of a pale face read
                    # as one 3-4 of 5 times (shelter chihuahuas 1 and 3, a
                    # black mask round a cream face; eval e04 4/5, e18 3/5),
                    # the real black masks of fawn dogs 5 of 5 (eval e10,
                    # e20, shelter puppy 6 in three runs). Short of that, the
                    # next answer
                    from .patterns import color_family
                    muzzle_votes = votes.get("muzzle") or {}
                    if (muzzle == "dark" and color_family(decision.base_color) != "dark"
                            and muzzle_votes.get("dark", 0) < votes.get("samples", 0)):
                        others = sorted(((n, a) for a, n in muzzle_votes.items() if a not in ("dark", "unknown")),
                                        reverse=True)
                        face_log["muzzle_not_unanimous"] = f"dark {muzzle_votes} on {decision.base_color}"
                        muzzle = face_log["muzzle"] = others[0][1] if others else "unknown"
                    # ... and a light muzzle on a warm coat split between white
                    # and cream is the softer cream: a golden's paler muzzle
                    # read white 3 / cream 2 drew a white stripe up its gold
                    # face (shelter retriever 1; the white blazes of the fawn
                    # chihuahuas had no cream vote)
                    if (muzzle == "white" and color_family(decision.base_color) == "point"
                            and muzzle_votes.get("cream", 0) > 0):
                        face_log["muzzle_softened"] = f"white {muzzle_votes} on {decision.base_color}"
                        muzzle = face_log["muzzle"] = "cream"
                    for name in (*CHEEK_FEATURES.get(cheeks, ()), *BROW_FEATURES.get(brows, ()),
                                 *MUZZLE_FEATURES.get(muzzle, ()), *EAR_FEATURES.get(ears, ())):
                        if name not in features:
                            features.append(name)
                    # a blaze up the forehead is not a region: the skull is one
                    # piece. It is a placed marking - a band on the face's
                    # midline from the nose up the forehead (texture._blaze_strip).
                    # Only with the muzzle top read the same colour: a blaze IS
                    # the light muzzle top running on up between the eyes, and
                    # the samples said "white blaze" for a puppy's paler bridge
                    # under a dark or cream muzzle answer (shelter puppies 6-8,
                    # no blaze in the photo; user: "머리에 흰줄 자꾸 왜")
                    if blaze in ("white", "cream") and muzzle != blaze:
                        face_log["blaze_dropped"] = f"blaze {blaze} but muzzle {muzzle}"
                    elif blaze in ("white", "cream"):
                        # wide on a white face (cheeks and muzzle light): a
                        # piebald puppy's white face between dark eye patches;
                        # over the whole head when the crop's crown is mostly
                        # light - a white puppy's grey eye patches and ears
                        # with the forehead white to the top (shelter puppy 14,
                        # dark fur 45 % of the crown; dark heads 89-96 %)
                        dark_crown = (face_log.get("face_dark") or [None, None, 0.0])
                        mostly_light = (dark_crown[0] is not None and dark_crown[2] >= FACE_CUT_MIN_SHARE
                                        and dark_crown[0] < BLAZE_FULL_DARK_SHARE)
                        size = "full" if mostly_light else "large" if cheeks in ("white", "cream") else "medium"
                        markings.append({"type": "blaze", "region": "skull", "color": blaze,
                                         "size": size,
                                         "anchor": "none",
                                         "note": f"a {blaze} blaze up the forehead between the eyes"})
            except Exception as exc:  # noqa: BLE001 - the face then keeps the layout's colours
                face_log["error"] = f"{type(exc).__name__}: {exc}"
        log["steps"]["face"] = face_log
        as_read = decision.pattern
        decision.pattern = irish_on_light(decision.pattern, decision.base_color, decision.second_color)
        if decision.pattern != as_read:
            face_log["irish_to_piebald"] = f"{as_read} {decision.base_color}/{decision.second_color}"
        # a piebald's cap must be seen on the crown: a white or cream crown
        # means the layout call took pale ear tips for a cap (eval e01, e02)
        light_crown = crown_is_light(face_log.get("crown"), (face_log.get("votes") or {}).get("crown"))
        capped_away = False
        if decision.pattern == "other" and decision.layout and light_crown:
            # ... and so must the cap of a layout the samples agreed on
            # region by region (pattern_classify.consensus_layout)
            capped = cap_off(decision.layout, decision.base_color)
            if capped:
                face_log["cap_dropped"] = capped
                capped_away = all(c == decision.base_color for c in decision.layout.values())
        if (decision.pattern == "piebald" or capped_away) and light_crown:
            face_log["piebald_demoted"] = True
            decision.pattern = "solid"
            decision.layout = None
            # the cap's colour goes with it: a solid coat's leftover fawn
            # counted as a points colour and faded the cheeks tan from the
            # eyes (eval v2 e01)
            patch_colour = decision.second_color
            decision.second_color = None
        # ears the face call could not see (folded, hanging) on a dark coat:
        # the layout's tan-to-cream cup reads as a white blob at the tip
        from .patterns import color_family
        if (face_log.get("ears") == "unknown" and color_family(decision.base_color) == "dark"
                and not any(f.endswith("_ears") for f in features)):
            features.append("dark_ears")
        # the chest, belly and feet from the photograph alone (body_colours.py):
        # the layout's tan front is a rottweiler's, not every tan-point's
        body_log: dict[str, Any] = {"asked": False}
        body_answers: tuple[str, str, str] | None = None
        cream_share: float | None = None
        if body_call:
            from .body_colours import filter_leg_features, light_by_family, read_body_colours
            try:
                chest, belly, paws, body_votes = await read_body_colours(transport, blocks, samples=samples)
                split = (chest, belly, paws)
                chest, belly, paws = (light_by_family(answer, body_votes.get(key), body_votes.get("samples"),
                                                      decision.base_color)
                                      for answer, key in zip(split, ("chest", "belly", "paws")))
                body_log.update({"asked": True, "chest": chest, "belly": belly, "paws": paws, "votes": body_votes})
                if (chest, belly, paws) != split:
                    body_log["light_by_family"] = {k: f"{a} -> {b}" for k, a, b in
                                                   zip(("chest", "belly", "paws"), split, (chest, belly, paws)) if a != b}
                features = filter_leg_features(features, paws)
                body_answers = (chest, belly, paws)
            except Exception as exc:  # noqa: BLE001 - the layout's chest, belly and legs stay
                body_log["error"] = f"{type(exc).__name__}: {exc}"
        # a sable's dark muzzle is dark-tipped hair, not a black bar: drawn by
        # the sable overlay at half strength over the base (fixture_sable3)
        # the sable overlay only where the back itself was seen tipped
        # (body_colours back_tipping): none turns a sable call into a solid
        # coat, slight draws the tipping at part strength
        # Only a unanimous none drops it: the real sable of e09 got none 2/3
        # (eval v4) while the masked red of e11 got none 3/3.
        votes = body_log.get("votes") or {}
        tipping = votes.get("back_tipping") or "unknown"
        tipping_votes = votes.get("back_tipping_votes") or {}
        plain_back = tipping == "none" and tipping_votes.get("none", 0) == votes.get("samples", -1)
        # ... or a majority none when the layout call itself was split with a
        # solid sample: every real sable on file was sable in all three
        # layout samples (e09 with none 2/3, e10), while a plain red jindo
        # read sable 2 / solid 1 and none 2 / slight 1 came out with a black
        # back and tail (shelter adult jindo 2); its back measured as tan as
        # e10's sable, so the measurement cannot tell them apart
        layout_solid = any(s.get("pattern") == "solid" for s in decision.samples or [])
        if (decision.overlay == "sable" and not plain_back and layout_solid and tipping == "none"
                and not tipping_votes.get("clear")):
            plain_back = True
            body_log["sable_split"] = {"layout": [s.get("pattern") for s in decision.samples or []],
                                       "back_tipping": tipping_votes}
        sable_by_back = 1.0
        if decision.overlay == "sable" and plain_back:
            body_log["sable_dropped"] = ("back_tipping none from every sample" if "sable_split" not in body_log
                                         else "back_tipping none, layout split with solid")
            decision.overlay = None
            decision.pattern = "solid"
            decision.second_color = None
        elif decision.overlay == "sable" and tipping == "slight":
            # a split none keeps the layout's full sable: e09's dark back at
            # part strength was paler than the photo (eval v4)
            sable_by_back = DARK_BACK_SABLE
        if decision.pattern == "saddle" and tipping == "none":
            # a saddle is a black back over a lighter flank; the back question
            # saw the back the flank's colour (a majority): a fawn puppy with
            # a dark muzzle came out with a black saddle (shelter puppy 6,
            # user). The dark muzzle is its own feature and stays.
            body_log["saddle_dropped"] = f"back_tipping none {tipping_votes}"
            decision.pattern = "solid"
            decision.second_color = None
        if plain_back and "dark_back" in features:
            # the layout call's dark back, which every sample of the back
            # question denied (e20: a sable shade on a plain fawn back)
            features = [f for f in features if f != "dark_back"]
            body_log["dark_back_dropped"] = True
        sable_extra: dict[str, float] = {}
        # a darker back on a warm one-colour coat is shading along the top,
        # not a black blanket: dark_back painted e12's whole back region
        # black (eval v3). It becomes the sable tipping at part strength.
        sable_strength = sable_by_back
        if ("dark_back" in features and decision.overlay is None and decision.pattern == "solid"
                and str(decision.base_color) in ("tan", "red", "fawn")):
            features = [f for f in features if f != "dark_back"]
            decision.overlay = "sable"
            sable_strength = DARK_BACK_SABLE
        # ... unless the face call was unanimous: a black mask every sample
        # saw is a mask, not tipping - at half strength e12's all but
        # vanished (eval demo). It stays a region, faded from the nose.
        muzzle_votes = (face_log.get("votes") or {}).get("muzzle") or {}
        unanimous_mask = muzzle_votes.get("dark", 0) >= max((face_log.get("votes") or {}).get("samples", 99), 1)
        if (decision.overlay == "sable" and "dark_muzzle" in features and sable_by_back >= 1.0
                and not unanimous_mask):
            features = [f for f in features if f != "dark_muzzle"]
            sable_extra["muzzle_bridge"] = SABLE_MUZZLE_SHARE
        # a brindle under a mostly dark crown is a dark brindle: a dark coat
        # with brown hairs mixed in (texture._brindle_mottle). Read on a fawn
        # base it came out a leopard-spotted puppy (shelter puppy 8). Its base
        # goes through the grey test below with the plain coats: the grey-
        # brown of its own fur, not black. Read on a black base too: the same
        # puppy's next reading said black/tan, skipped this, and the refine
        # lifted the black-and-tan mottle to a chestnut coat
        dark_share = (face_log.get("face_dark") or [None])[0]
        if (decision.overlay == "brindle" and face_log.get("crown") == "dark" and dark_share is not None
                and dark_share >= DARK_BRINDLE_SHARE and decision.base_color not in ("chocolate", "gray")):
            face_log["dark_brindle"] = f"base {decision.base_color} -> black (dark fur {dark_share})"
            decision.base_color, decision.second_color = "black", None
        # a plain coat read white by a split vote, one sample saying cream,
        # on a dog whose face reads cream: cream (shelter maltese 3, a cream
        # puppy read white 2-1 with a cream face 5/5). The face call says
        # cream for most white dogs too, so it decides only when the pattern
        # vote itself was not unanimous (no white dog on file had a cream
        # sample). Cream ahead of every other crown answer: tied 2 / 2 with
        # white, max() took cream for coming first and a dirty white maltese
        # would have gone cream (shelter maltese 2, 11-07)
        crown_votes = (face_log.get("votes") or {}).get("crown") or {}
        solid_bases = [s.get("base_color") for s in decision.samples if s.get("pattern") == "solid"]
        if (decision.pattern == "solid" and decision.base_color == "white" and "cream" in solid_bases
                and crown_votes.get("cream", 0) > max((v for k, v in crown_votes.items() if k != "cream"),
                                                      default=0)):
            face_log["white_to_cream"] = f"pattern bases {solid_bases}, crown {crown_votes}"
            decision.base_color = "cream"
        region_colors = apply_features(decision.pattern, decision.base_color, decision.second_color, features,
                                       consensus=decision.layout)
        face_log["crown_applied"] = crown_on_skull(region_colors, face_log.get("crown"),
                                                   decision.base_color, decision.second_color)
        if (face_log.get("face_light") or {}).get("cheeks") == "dark" or coloured_cheeks(region_colors,
                                                                                          face_log.get("cheeks")):
            face_log["cheeks_to_head"] = cheeks_to_head(region_colors)
        # a crown read dark whose dark fur is lighter than black in the crop
        # (palette.face_dark): a smoky grey cap, not a black one - on a coat
        # read light or coloured. A coat read black keeps its black: in
        # daylight a black puppy's crown measured L* 28.9, the grey cap it
        # was meant for 27.2, and the rule turned all ten of its black
        # regions grey (shelter maltese 8; its back measured L* 12)
        dark_share, dark_l, cut, tint_a, tint_b = (list(face_log.get("face_dark") or [])
                                                   + [None, None, 0.0, None, None])[:5]
        black_coat = decision.base_color == "black" and decision.second_color != "gray"
        smoky = (face_log.get("crown") == "dark" and dark_l is not None
                 and cut >= FACE_CUT_MIN_SHARE and dark_share >= FACE_DARK_MIN_SHARE and dark_l >= FACE_GRAY_MIN_L)
        if smoky and black_coat and "dark_brindle" not in face_log:
            face_log["gray_skipped"] = f"coat read black (dark L {dark_l})"
        if face_log.get("crown") == "gray" or (
                smoky and ("dark_brindle" in face_log or (decision.overlay != "brindle" and not black_coat))):
            face_log["gray_for_black"] = gray_for_black(region_colors, markings)
            # ... in the grey of its own fur (palette.gray_from_face), measured
            # on a cut-out crop only
            if cut >= FACE_CUT_MIN_SHARE:
                from .palette import gray_target
                face_log["gray_lab"] = (gray_target(dark_l, tint_a, tint_b, lift=0.0) if "dark_brindle" in face_log
                                        else gray_target(dark_l, tint_a, tint_b))
        # patches are placed at random over the body, so only on a body the
        # photograph shows from the side: seen from the front the flank is
        # a guess, and a plain coat is the usual one (eval v2 e02: black
        # patches from a front view with one small spot on a leg)
        views = [g.get("view") for g in (log["steps"].get("prepare") or {}).get("gate") or []]
        flank_seen = not views or "side" in views or "mixed" in views
        if "body_patches" in features and not flank_seen:
            log["steps"]["patches_skipped"] = f"body_patches but the photo view is {views}"
        patches_in = decision.second_color if decision.second_color not in (None, "none") else patch_colour
        # how much and where, from the body call (body_colours patch_*): the
        # amount sizes them, the places draw them there. A body every sample
        # called patch-free keeps the coat plain.
        body_votes = body_log.get("votes") or {}
        patch_amount = body_votes.get("patch_amount") or "unknown"
        no_patches = ((body_votes.get("patch_votes") or {}).get("amount") or {}).get("none", 0) == \
            body_votes.get("samples", -1)
        # a piebald is a white body with patches: the body call seeing them
        # is enough when the layout samples did not name them (shelter
        # chihuahua 10: body_patches 1 of 3 layout samples, patches large
        # 3/3 in the body call - a white body under a black head). Only on a
        # piebald: a fawn dog with a white bib read "large patches" too
        # (chihuahua 2)
        if (decision.pattern == "piebald" and "body_patches" not in features
                and patch_amount in ("small", "large", "most")):
            features.append("body_patches")
            log["steps"]["patches_from_body_call"] = f"piebald, body call {patch_amount}"
        if "body_patches" in features and flank_seen and patches_in and no_patches:
            log["steps"]["patches_skipped"] = "body call: no patches on the body"
        elif "body_patches" in features and flank_seen and patches_in:
            markings.append({"type": "patches", "region": "back", "color": patches_in,
                             "size": PATCH_SIZE.get(patch_amount, "medium"),
                             "places": list(body_votes.get("patch_places") or []),
                             "anchor": "none", "note": "piebald patches on the body"})
        if body_answers:
            from .body_colours import apply_body_colours
            chest, belly, paws = body_answers
            body_log["changed"] = apply_body_colours(region_colors, markings, chest, belly,
                                                     decision.base_color, decision.second_color, paws=paws)
            if (body_log.get("votes") or {}).get("cream_share"):
                cream_share = body_log["votes"]["cream_share"]
        nape = nape_marking(region_colors, body_log.get("votes"))
        if nape:
            markings.append(nape)
            body_log["nape_marking"] = nape["color"]
        ears_marking, ears_change = dark_ears_on_light(region_colors, features, decision.base_color)
        if ears_marking:
            markings.append(ears_marking)
        if ears_change:
            log["steps"]["dark_ears_on_light"] = ears_change
        log["steps"]["body"] = body_log
        # the outline (body_shape.py), only when the template carries the
        # morph targets to show it (ar_pipeline/shape_keys.py)
        shape: dict[str, Any] = {}
        from .shape import target_names
        if early_shape:
            # read above with the model sheet; the morphs take what the
            # template has (set_shape skips unknown targets)
            from .body_shape import shape_weights
            answers, shape_votes = early_shape
            shape = {"answers": answers, "weights": shape_weights(answers, shape_votes)}
            log["steps"]["shape"] = {**shape, "votes": shape_votes}
        elif shape_call and target_names(template_glb):
            from .body_shape import read_body_shape, shape_weights
            try:
                answers, shape_votes = await read_body_shape(transport, blocks, samples=samples)
                shape = {"answers": answers, "weights": shape_weights(answers, shape_votes)}
                log["steps"]["shape"] = {**shape, "votes": shape_votes}
            except Exception as exc:  # noqa: BLE001 - the template's own outline stays
                log["steps"]["shape"] = {"error": f"{type(exc).__name__}: {exc}"}
        appearance = {
            "pattern": decision.pattern, "base_color": decision.base_color, "second_color": decision.second_color,
            "eye_color": decision.eye_color, "overlay": decision.overlay, "sable_extra": sable_extra,
            "sable_strength": sable_strength,
            "features": features, "feature_votes": decision.feature_votes,
            "region_colors": region_colors, "markings": markings, "shape": shape,
            "cream_share": cream_share,
            "points_pale": pale_points(decision.pattern, decision.second_color, decision.samples),
            # the body call's feet not light in every sample (simple.default_gradients' inner legs)
            "paws_split": paws_split(body_log.get("votes")),
            # light or dark round the eyes (face_colours.EYE_RING_ANSWERS);
            # on dark texture.compose_body leaves out the cream spectacles
            "eye_ring": face_log.get("eye_ring") if face_log.get("eye_ring") in ("light", "dark") else None,
            # how far a dark muzzle runs up the face (face_colours MASK_ANSWERS,
            # simple.MASK_REACH): only for a muzzle that passed as dark above
            "mask_reach": (face_log.get("mask") if face_log.get("muzzle") == "dark"
                           and face_log.get("mask") in ("muzzle", "eyes") else None),
            # the light face fur's LAB on a dark-crowned, cream-cheeked dog
            "face_light_lab": face_log.get("face_light_lab"),
            # the grey of a smoky crown's own fur (palette.gray_from_face)
            "gray_lab": face_log.get("gray_lab"),
            "inferred_regions": [e["region"] for e in region_colors
                                 if str(e.get("source", "")).startswith("pattern")],
        }
        log["steps"]["reading"] = {"pattern": decision.as_dict(), "summary": describe_decision(decision),
                                   "features": features, "feature_votes": decision.feature_votes}
    appearance.update({"pet_id": resolved.pet_id, "base_template": base_template,
                       "size_class": resolved.size_class})
    _json(out_dir / "appearance.json", appearance)
    log["timings_s"]["reading"] = round(_now() - t, 2)
    print(f"[lite] reading: {appearance.get('pattern')} {appearance.get('base_color')}/{appearance.get('second_color')}, "
          f"eyes {appearance.get('eye_color')}, features {appearance.get('features') or []}"
          + (f", markings {[m['type'] + ':' + m['color'] for m in appearance.get('markings') or []]}"
             if appearance.get("markings") else ""))

    # 2a. does the picked template suit this dog (fit.py): looks no template
    #     covers, long coat, short legs, a toy's head, a flat face, a layout
    #     outside the pattern set. Logged
    #     always; with fit_stop a poor verdict ends the run here, before the
    #     measurement call and the image edit.
    from .fit import FitRejected, assess_fit
    from .fit import describe as describe_fit
    from .prepare_photo import seen_from_above
    fit = assess_fit(meta.species, meta.weight_kg, meta.age_text,
                     (appearance.get("shape") or {}).get("answers"), appearance.get("pattern"),
                     votes=(log["steps"].get("shape") or {}).get("votes"), photo_quality=photo_quality,
                     pick=pick, photo_bars=photo_bars,
                     from_above=seen_from_above((log["steps"].get("prepare") or {}).get("gate") or []))
    log["steps"]["fit"] = fit
    _json(out_dir / "fit.json", fit)
    print(f"[lite] fit: {describe_fit(fit)}")
    if fit_stop and fit["level"] == "poor":
        log["timings_s"]["total"] = round(_now() - started, 2)
        log["usage"] = ledger.summary()
        log["final"] = {"stopped": "fit", "fit": fit["label"]}
        _json(out_dir / "pipeline_log.json", log)
        raise FitRejected(fit)

    # 2b. the photograph's own colour per region. A name says which ramp; the
    #     measurement says how light and how warm this dog's version of it
    #     is, and the hued ramps move toward it (palette.ramps_from_measurement).
    #     Carried in the appearance so a --reuse run builds the same coat.
    measured: dict[str, Any] = appearance.get("measured_colors") or {}
    palette: dict[str, tuple[str, str, str]] = {}
    palette_report: dict[str, Any] = {}
    # a photograph whose fur is clipped to black or white has no lightness
    # to measure: the table ramps are closer than a number read off L 0 or
    # L 255, and the measurement call is saved
    if measure and not measured and (photo_quality or {}).get("levels", {}).get("exposure") == "poor":
        print(f"[lite] colour measurement skipped: {photo_quality['clipped_share']:.0%} of the fur is clipped; "
              "using the table ramps")
        log["steps"]["measure_skipped"] = {"why": "exposure poor", "clipped_share": photo_quality["clipped_share"]}
        measure = False
    # through bars the measurement's pixels land on bars as often as on fur
    if measure and not measured and photo_bars:
        print("[lite] colour measurement skipped: the dog is seen through bars; using the table ramps")
        log["steps"]["measure_skipped"] = {"why": "bars"}
        measure = False
    if measure and not measured:
        from .photo_colours import measure_region_colours

        t = _now()
        try:
            measured = await measure_region_colours(original_photo, transport or default_transport(),
                                                    animal_mask=photo_animal)
        except Exception as exc:  # noqa: BLE001 - the table ramp is the fallback
            print(f"[lite] colour measurement failed ({type(exc).__name__}: {exc}); using the table ramps")
            measured = {}
        appearance["measured_colors"] = measured
        _json(out_dir / "appearance.json", appearance)
        log["timings_s"]["measure"] = round(_now() - t, 2)
    if measured:
        from .palette import ramps_from_measurement
        # the border's kind is the photo's, not the reading's: a --reuse of
        # an appearance saved before it was recorded gets it stamped here
        if any(isinstance(r, dict) and "border" not in r for r in measured.values()):
            from .photo_colours import border_kind_of
            kind = border_kind_of(original_photo)
            for reading in measured.values():
                if isinstance(reading, dict):
                    reading.setdefault("border", kind)
    back = irish_back(appearance, measured or {}, bool((log["steps"].get("face") or {}).get("irish_to_piebald")))
    if back:
        log["steps"]["irish_back"] = back
    steel = steel_saddle(appearance, measured or {},
                         ((log["steps"].get("body") or {}).get("votes") or {}).get("back_tipping_votes"))
    if steel:
        log["steps"]["steel_saddle"] = steel
    saddle = saddle_back(appearance, measured or {})
    if saddle:
        log["steps"]["saddle_back"] = saddle
    faded = faded_black(appearance, measured or {})
    if faded:
        log["steps"]["faded_black"] = faded
    one_coat = one_coat_colour(appearance, measured or {},
                               ((log["steps"].get("face") or {}).get("votes") or {}).get("crown"))
    if one_coat:
        log["steps"]["one_coat_colour"] = one_coat
    # a light coat in low sun (photo_colours.sun_cast): the photo's hue and
    # chroma are the light's, so the ramps take lightness only
    sunlit = False
    if photo_animal is not None:
        from .photo_colours import sun_cast
        try:
            sun = sun_cast(Image.open(original_photo).convert("RGB"), photo_animal)
        except Exception as exc:  # noqa: BLE001 - read as not sunlit
            sun = None
            log["steps"]["sun_error"] = f"{type(exc).__name__}: {exc}"
        if sun:
            log["steps"]["sun"] = sun
            sunlit = bool(sun["sunlit"])
            if sunlit:
                print(f"[lite] sunlit: lit fur chroma {sun['lit_chroma']} against {sun['mid_chroma']} in the shade; "
                      "the coat's hue and chroma stay the table's")
    if measured:
        palette, palette_report = ramps_from_measurement(appearance, measured, sunlit=sunlit)
    # ... and their chroma from the whole coat's pixels (palette.chroma_from_coat)
    if photo_animal is not None:
        from .palette import chroma_from_coat
        from .photo_colours import coat_chroma
        try:
            coat = None if sunlit else coat_chroma(Image.open(original_photo).convert("RGB"), photo_animal)
        except Exception as exc:  # noqa: BLE001 - the cells' ramps stand
            coat = None
            log["steps"]["coat_chroma_error"] = f"{type(exc).__name__}: {exc}"
        palette, coat_report = chroma_from_coat(appearance, palette or {}, coat)
        log["steps"]["coat_chroma"] = coat_report or {"skipped": "sunlit" if sunlit else coat}
        # ... and a cream dog's cream from its lit fur (palette.cream_from_coat)
        if str(appearance.get("base_color", "")).lower() == "cream":
            from .palette import cream_from_coat
            from .photo_colours import light_coat
            try:
                light = light_coat(Image.open(original_photo).convert("RGB"), photo_animal)
            except Exception as exc:  # noqa: BLE001 - the table's cream stands
                light = None
                log["steps"]["cream_coat_error"] = f"{type(exc).__name__}: {exc}"
            palette, cream_report = cream_from_coat(appearance, palette or {}, light, sunlit=sunlit)
            log["steps"]["cream_coat"] = cream_report or {"skipped": light}
    chocolate = chocolate_crown(appearance.get("region_colors") or [], measured, appearance.get("markings"))
    if chocolate:
        log["steps"].setdefault("face", {})["crown_chocolate"] = chocolate
    from .palette import cream_from_face, gray_from_face, relative_cream
    from .palette import pale_points as pale_points_ramp
    palette, pale = pale_points_ramp(appearance, palette or {})
    if pale:
        log["steps"]["pale_points"] = pale
    warm_front = tan_cream(appearance, measured or {})
    if warm_front:
        log["steps"]["tan_cream"] = warm_front
    palette = relative_cream(appearance, palette or {})
    palette = cream_from_face(appearance, palette)
    palette = gray_from_face(appearance, palette)
    if "cream" in palette:
        palette_report["cream_relative"] = list(palette["cream"])
    log["steps"]["measure"] = {"regions": {r: v["lab"] for r, v in measured.items()}, "ramps": palette_report}
    print(f"[lite] measured {len(measured)} region(s): "
          + (", ".join(f"{n} L{r['table_mid'][0]:.0f}->{r['table_mid'][0] + r['shift'][0]:.0f}"
                       for n, r in palette_report.items() if isinstance(r, dict)) or "no ramp moved")
          + (f", skipped {palette_report['skipped']}" if palette_report.get("skipped") else "")
          + (f" ({palette_report['hue_kept']})" if palette_report.get("hue_kept") else ""))

    # 3-4. the mask build, with the default gradients where colours blend
    t = _now()
    from .simple import default_gradients
    build_appearance, gradients = default_gradients(appearance)
    log["steps"]["gradients"] = gradients
    base = extract_base_texture(template_glb)
    base_path = out_dir / "template_base.png"
    base.save(base_path)
    from .assets import geometry_dir_for
    valid = np.load(geometry_dir_for(masks_dir) / "valid.npy") > 0.5
    # Every colour edge is feathered in the build. With no refine the build
    # IS the coat (the white bib met the tan as a cut line, eval v3 e05); with
    # a refine the model was meant to blend them, but it traced the build's
    # cut edges and kept them hard (user on demo e08: "the edges where the
    # colour changes look unnatural, they didn't before").
    build_report: dict[str, Any] = {}
    coat = synthesize_texture(base_path, masks_dir, build_appearance, feather_all=True,
                              palette=palette or None, report_out=build_report)
    coat.save(out_dir / "mask_build.png")
    log["timings_s"]["mask_build"] = round(_now() - t, 2)

    # 5. refine
    t = _now()
    refine_log: dict[str, Any] = {"ran": False}
    refined = False
    texture = coat
    if refine:
        if image_transport is None:
            if os.environ.get("OPENAI_API_KEY") or os.environ.get("LUNA_API_KEY"):
                from .imagegen import OpenAIImageTransport
                image_transport = OpenAIImageTransport()
        if image_transport is None:
            refine_log["skipped"] = "no OPENAI_API_KEY for the image model; the mask build is the coat"
        else:
            use_cache = not isinstance(image_transport, ScriptedImageTransport)
            texture, refined, report = await refine_coat(image_transport, coat, photos[0], appearance, valid,
                                                          masks_dir, notes=photo_notes, use_cache=use_cache)
            refine_log = {"ran": True, "accepted": refined, **{k: v for k, v in report.items() if k != "prompt"}}
            if refined:
                texture.save(out_dir / "refined.png")
    else:
        refine_log["skipped"] = "--no-refine"
    log["steps"]["refine"] = refine_log
    log["timings_s"]["refine"] = round(_now() - t, 2)
    print(f"[lite] refine: {'accepted' if refined else refine_log.get('rejected') or refine_log.get('skipped') or 'not run'}")

    # 6. finish, 7. output
    t = _now()
    final_dir = out_dir / "final"
    final_dir.mkdir(exist_ok=True)
    # flatten_rims off: the rims are already the build's (restore_in_region),
    # and the finish step's rim measurement - the outer three texels against
    # 4-10 px in - read the model's outline at 4-10 px as the reference and
    # darkened the correct rim toward it by 4.6 L, a line along every seam.
    # Off without a refine too: the texture is then the build itself, with
    # no model outline to take out, and the correction - one offset per
    # texel of rim, the median over every colour of coat at once - lifted
    # the rims of a black-and-tan and a black-and-white coat's black from
    # sRGB 27 to 43, a light zigzag along every seam of the black (user,
    # 11-05: "경계면쪽 선 보이는 문제")
    # thin strips packed inside another chart take the surface's colour, so
    # the seam pass samples the right one (SLIVER_*)
    surface = _surface_position(masks_dir)
    slivers = _atlas_slivers(valid, surface)
    if (Path(masks_dir).parent / "protect.png").exists():
        # not the template's eyes, nose and pads
        slivers &= ~(np.asarray(Image.open(Path(masks_dir).parent / "protect.png").convert("L")
                                .resize(texture.size)) > 127)
    if slivers.any():
        texture = Image.fromarray(_fill_from_surface(np.asarray(texture.convert("RGB")).copy(), slivers, valid, surface))
    finished, finish_report = finish_painted(texture, base, masks_dir, template_glb, valid,
                                             eye_colour=appearance.get("eye_color"),
                                             flatten_rims=False)
    finish_report["slivers_from_surface"] = int(slivers.sum())
    # confetti islands (claws, eyelid shells, seam fringe on a generated
    # mesh) come out of the finish as grey shards: the seam ring is wider
    # than the island, so the whole island is blended with whatever lies
    # across its seam. They take the build's colour back, except where the
    # template's own pixels were restored (eyes, nose, pads)
    # after a refine the eyes, nose and pads come back from the build (which
    # took them from the template, all of protect.png): the finish restores
    # only protect's WHOLE feature islands, and the jindo's eye islands
    # carry lid fur - on a black head the model painted the eyes over with
    # fur and they were gone (eval demo e13)
    if refined:
        protect_path = Path(masks_dir).parent / "protect.png"
        if protect_path.exists():
            from scipy import ndimage as _nd
            keep = np.asarray(Image.open(protect_path).convert("L").resize(finished.size), dtype=np.float32) / 255.0
            weight = np.maximum(_nd.gaussian_filter(keep, 1.0), keep)[..., None]
            out = np.asarray(finished.convert("RGB"), dtype=np.float32)
            src = np.asarray(coat.convert("RGB").resize(finished.size), dtype=np.float32)
            finished = Image.fromarray(np.clip(out * (1 - weight) + src * weight, 0, 255).astype(np.uint8))
            finish_report["protect_from_build"] = int((keep > 0.5).sum())
    tiny = _tiny_islands(valid, TINY_ISLAND_TEXELS)
    if tiny.any() or slivers.any():
        protect_path = Path(masks_dir).parent / "protect.png"
        if protect_path.exists():
            tiny &= ~(np.asarray(Image.open(protect_path).convert("L").resize(finished.size)) > 127)
        # ... and the slivers again: the seam ring repainted their edges
        before = np.asarray(finished.convert("RGB")).copy()
        protected = (np.asarray(Image.open(protect_path).convert("L").resize(finished.size)) > 127
                     if protect_path.exists() else None)
        out = _fill_from_surface(before.copy(), tiny | slivers, valid, surface, exclude=protected)
        out = _keep_island_detail(out, before, tiny & ~slivers, valid)
        # ... and the padding is refilled from the islands, or the grey rim
        # the finish left around a small island stays in the texels the
        # renderer samples just past its edge
        from scipy import ndimage
        idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        out = out[idx[0], idx[1]]
        finished = Image.fromarray(out)
        finish_report["tiny_islands_from_surface"] = int(tiny.sum())
    marking_mask = build_report.get("marking_mask")
    finished, finish_report["head_shards_cleaned"] = _clean_head(
        finished, masks_dir, valid, keep=marking_mask > 0.3 if marking_mask is not None else None)
    finished, finish_report["spikes_cleaned"] = _clean_spikes(
        finished, masks_dir, valid, surface, keep=marking_mask > 0.3 if marking_mask is not None else None)
    # the gutter texels the triangles sample take the surface they show
    # (_gutters_from_surface), then the stray triangles once more, last
    # (seams.fill_stray_triangles): the padding refill above took back what
    # the finish painted under them, and a black-and-tan's tan foreleg kept
    # its white triangle
    if template_glb and Path(template_glb).exists() and surface is not None:
        from .mesh import TemplateMesh
        from .seams import fill_stray_triangles
        stray_mesh = TemplateMesh.load(template_glb, finished.size)
        finished, finish_report["gutters_from_surface"] = _gutters_from_surface(
            finished, stray_mesh, surface, valid, str(template_glb))
        out, finish_report["stray_triangles_last"] = fill_stray_triangles(
            np.asarray(finished.convert("RGB"), dtype=np.float32), stray_mesh, surface, valid)
        finished = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    finished.save(final_dir / f"{resolved.pet_id}.png")
    from .whiskers import whisker_colour
    whisker_rgb, finish_report["whiskers"] = whisker_colour(finished, template_dir)
    glb = replace_base_texture(template_glb, finished, final_dir / f"{resolved.pet_id}.glb", whisker_rgb=whisker_rgb)
    weights = (appearance.get("shape") or {}).get("weights") or {}
    if weights:
        from .shape import fit_to_template, set_shape, target_names
        applied = fit_to_template(weights, set(target_names(glb)))
        if applied:
            set_shape(glb, applied)
        log["steps"]["shape_applied"] = applied
    log["steps"]["finish"] = finish_report
    log["timings_s"]["finish"] = round(_now() - t, 2)

    renders: dict[str, Any] = {}
    if render and Path(blender).exists():
        from .critic import render_views

        t = _now()
        body, face = render_views(glb, out_dir / "renders", blender, "sitting")
        renders = {"body": str(body), "face": str(face), "side": str(Path(body).parent / "render_side.png")}
        log["timings_s"]["render"] = round(_now() - t, 2)
    log["steps"]["renders"] = renders

    # real-world size (size.py): the record's weight -> a withers height ->
    # one uniform scale on the scene root. After the renders, whose cameras
    # are set for the template's own size.
    size_info: dict[str, Any] = {"skipped": True}
    if real_size:
        from .size import apply_scale, scale_for
        try:
            size_info = scale_for(meta.weight_kg, template_glb)
        except KeyError as exc:
            # a mesh without the joints to measure its shape: the model
            # stays at the template's own size rather than a guessed one
            size_info = {"skipped": True, "reason": str(exc)}
            print(f"[lite] size: skipped, {exc}")
        else:
            apply_scale(glb, size_info["scale"])
            print(f"[lite] size: {meta.weight_kg:g} kg -> withers {size_info['withers_target_m']:.2f} m "
                  f"(template {size_info['withers_template_m']:.2f} m, scale {size_info['scale']:.3f})")
    log["steps"]["size"] = size_info

    behavior = resolve_behavior(animation_names(glb))
    log["final"] = {"glb": str(glb), "texture": str(final_dir / f"{resolved.pet_id}.png"),
                    "served_path": resolved.asset_path, "behavior": behavior, "refined": refined,
                    "features": appearance.get("features") or [], "eye_color": appearance.get("eye_color"),
                    "fit": {"level": fit["level"], "label": fit["label"], "score": fit["score"]},
                    "size": size_info}
    log["timings_s"]["total"] = round(_now() - started, 2)
    log["usage"] = ledger.summary()
    _json(out_dir / "pipeline_log.json", log)
    u = log["usage"]["total"]
    print(f"[lite] usage: {u['calls']} calls, {u['prompt_tokens']} in ({u['cached_tokens']} cached), "
          f"{u['completion_tokens']} out, ~${u['usd']:.3f}"
          + (f" (unpriced: {log['usage']['unpriced_models']})" if log["usage"]["unpriced_models"] else ""))
    print(f"[lite] done in {log['timings_s']['total']}s -> {glb}")
    asset = BuiltAsset(pet_id=resolved.pet_id, glb_path=glb, texture_path=final_dir / f"{resolved.pet_id}.png",
                       served_path=resolved.asset_path, behavior=behavior,
                       report={"path": "lite", "refine": refine_log})
    return PipelineResult(resolved.pet_id, appearance, asset, log, out_dir)


__all__ = ["run_lite", "refine_coat", "refine_prompt", "blend_from_edit", "CACHE_DIR", "BAND_PX", "MIX_FLOOR",
           "MIX_SPREAD_PX", "MIX_GAMMA", "transfer_detail", "DETAIL_SIGMA", "restore_in_region", "REFINE_MODE"]
