"""Stage 4-4 - put the generated views onto the mesh, take the light out, and
fall back to the mask build wherever no view reached.

THE THREE PARTS
---------------
Projection. For every texel the baked geometry says where it is in the world
and which way it faces. Each canonical camera is rebuilt from cameras.json,
the texel is dropped into that camera, and if the view's depth map agrees the
texel is the nearest surface there - and the view actually painted that pixel -
the view's colour is taken with weight max(0, n·v)^2. Facing weights squared,
so a texel seen edge-on by two cameras and face-on by a third takes almost all
of its colour from the third instead of averaging three smeared samples.

Delighting. Whatever light the image model painted in is in the luminance.
The projected colour is split in LAB: chroma (a, b) and fine luminance detail
are kept, and the broad luminance - the part a shadow lives in - is replaced,
region by region, with the luminance of that region's calibrated fur ramp.
Because the generated views are flat art with no hair, the fine detail is
taken from the template's own brightness map, which is exactly the fur that
synthesize_texture has always used.

Fallback. The mask build is still made, and the two are blended by coverage:
where the projection saw the surface the projection wins; where it did not,
the mask build stands in. If the consistency check scored the views poorly the
projection's weight is lowered by that score, so a sheet the vision agents
could not recognise does not get to overrule the record.

post_process() at the very end is an identity hook: the seat for an img2img
pass over the finished texture, should one ever be worth adding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from .fur import FUR_PALETTE, hex_to_linear, to_linear, to_srgb
from .texture import (
    FACE_REGIONS, LUMA, MaskLibrary, brightness_map, feature_core, keep_features,
    synthesize_texture,
)

VIEW_NAMES = ("front", "left", "right", "back")

# A texel is taken to be the surface a camera sees when its distance along the
# view axis is within this much of the depth map, as a fraction of the model's
# span. Depth was rendered at 16 bits over the clip range, so this is well
# above the quantisation.
DEPTH_TOLERANCE = 0.006

# The AA edge of the depth render holds distances blended toward the clip
# start; anything closer than this fraction of the camera distance is noise.
DEPTH_MIN_FRACTION = 0.5

# Weight = max(0, n·v) ** FACING_POWER. Squared made each view drop out
# abruptly where its facing fell off, and the handover from one view to the
# next showed as a seam; 1.5 keeps the face-on view dominant with a softer edge.
FACING_POWER = 1.5

# A texel a view sees at a grazing angle - n·v below this - takes nothing
# from it. At a grazing angle the sample is one pixel smeared over many
# texels and, on the inside of a leg, the wrong leg's pixel; the inner
# thighs came out a patchwork of whatever each camera glimpsed. Such texels
# count as unseen, and the neighbour fill or the mask build decides them.
FACING_FLOOR = 0.25

# Each view's weight map is blurred over this many texels before the views are
# normalised against each other, so where two views meet they cross-fade over
# a band instead of switching at a line. The blur is masked back to the view's
# own visibility, so no texel takes colour from a view that could not see it.
WEIGHT_SMOOTH_PX = 20

# FACE RULES
# ----------
# The generated views draw a rounder, shorter head than the mesh has, so on the
# face the projection lands the markings a little off: the tan the drawing put
# beside the eyes fell on the stop and the inner eye corners, and the eyebrow
# pips missed the brow entirely. The body can be trusted to the projection; the
# face takes three rules from the record and the mesh instead.
#
# 1. Eyebrow pips are two ovals directly above the eyes, each about as wide
#    as the eye (that is the region's own definition). The eyebrow mask is
#    baked from the brow bones and spans the whole forehead, five centimetres
#    tall on this template; painting all of it put a tan slab across the stop,
#    and painting its lateral part put a block above each eye. So the pip is
#    drawn on the mesh: an ellipse per eye, sized from the eye's own radius,
#    and only where the eyebrow mask agrees. Sizes are in eye radii.
PIP_HALF_WIDTH = 1.0
PIP_HALF_HEIGHT = 0.7
PIP_RISE = 1.3

# 2. A face region the record calls "partial" is coloured in the part nearest
#    the eyes: tan-point and urajiro cheeks are tan under the eye and white
#    along the muzzle, never the other way round. This share of the region,
#    by distance to the nearest eye on the mesh and no higher than the eye's
#    centre, takes the declared colour. Less than the mask build's
#    PARTIAL_KEEP: on the photograph the tan is a patch under each eye, not
#    half the cheek.
FACE_PARTIAL_KEEP = 0.35

# A rule's patch is closed in UV space by this many texels: an ellipse on the
# mesh crosses the eyelid crease, where the baked geometry dips, and came out
# with a notch.
RULE_CLOSE_PX = 4
#
# 3. A face region the record gives no colour for wears the base coat or
#    urajiro white; a hued colour that the projection put there is the
#    drawing's misplacement, and is snapped to the nearest of those instead.
UNDECLARED_FACE_ALLOWED = frozenset({"white", "cream"})

# 5. The inside of an ear takes the record's colour outright when the record
#    gives it one in full. In a 512-pixel cell the ear's cup is a few pixels,
#    and the drawing paints it whatever it painted the head; the vision read
#    of the photograph is the better witness at that size.
EAR_RECORD_REGIONS = frozenset({"ears_inner"})

# 4. Close around the eyes the drawing is least reliable of all - its muzzle
#    white landed between the eye and the brow - so within this many eye radii
#    of an eye a texel takes only its own region's declared colour or the base
#    coat. On a tan-point that is black around the eye with the pips and the
#    cheek patch from rules 1 and 2; on a dog whose skull is declared white it
#    stays white.
EYE_VICINITY = 2.5

# Texels this close to the eyes, nose and tongue take no projected colour of
# their own. The generated views draw the eyes a little off the mesh's eyes,
# and projecting them put a second pair beside the template's. Two earlier
# answers were wrong: handing the ring to the mask build put a ring of a
# different texture around every eye, and painting it the record's colour put
# an orange mask on a cream-cheeked dog, because the record said the cheeks
# were tan and the drawing - correctly - said cream. So the ring is filled in
# from the projected colour just outside it, and keep_features then lays the
# template's eyes over the top. Fourteen texels is enough to swallow a
# misplaced painted eye without reaching the eyebrows.
FEATURE_EXCLUSION_PX = 10

# Coverage is feathered over this many texels where the projection hands over
# to the mask build, so the seam does not read as a cut.
COVERAGE_FEATHER_PX = 12

# Below this consistency score the projection's weight is scaled by the score.
CONSISTENCY_FLOOR = 0.6

# Broad luminance is what lives at this scale and above: about 2% of the
# texture width, wide enough to hold a painted shadow, narrow enough to leave
# the edge between two colour areas alone.
DELIGHT_SIGMA_FRACTION = 0.02

# How strongly the template's hair detail is added back, in L units per unit of
# the brightness map's own variation. 60 made a black back read as grey streaks;
# 15 keeps the hair visible without lifting the coat.
TEMPLATE_DETAIL_GAIN = 15.0

# When snapping a texel to the nearest known fur colour, L counts this much
# relative to a and b. Painted shading moves L a great deal and the coat colour
# hardly at all, so a black texel the model shaded lighter must still be black:
# the decision is made on hue and the luminance is only a tiebreaker.
SNAP_L_WEIGHT = 0.2

# The colour the appearance declares for a texel's region - the base coat where
# no region claims it - has its snap distance scaled by this. The masks are a
# prior, the projection is the evidence: the projection still wins where it
# disagrees clearly, but a small region the model painted a shade off (dark
# eyebrow tan that fell nearer black) is pulled back to what the record says
# it is.
SNAP_PRIOR_BIAS = 0.35

# The snap assignment is smoothed over this many texels by majority, so a lone
# misclassified texel does not become a grey speck in a black coat. Six, not
# eight: the eyebrow pips are only a few dozen texels across.
SNAP_SMOOTH_PX = 6


# --------------------------------------------------------------------------
# colour space
# --------------------------------------------------------------------------

_M_RGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                          [0.2126729, 0.7151522, 0.0721750],
                          [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
_M_XYZ_TO_RGB = np.linalg.inv(_M_RGB_TO_XYZ).astype(np.float32)
_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)


def _f(t: np.ndarray) -> np.ndarray:
    delta = 6.0 / 29.0
    return np.where(t > delta ** 3, np.cbrt(t), t / (3 * delta ** 2) + 4.0 / 29.0)


def _f_inv(t: np.ndarray) -> np.ndarray:
    delta = 6.0 / 29.0
    return np.where(t > delta, t ** 3, 3 * delta ** 2 * (t - 4.0 / 29.0))


def linear_to_lab(rgb: np.ndarray) -> np.ndarray:
    xyz = rgb @ _M_RGB_TO_XYZ.T / _D65
    fx, fy, fz = (_f(xyz[..., i]) for i in range(3))
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)


def lab_to_linear(lab: np.ndarray) -> np.ndarray:
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    xyz = np.stack([_f_inv(fx), _f_inv(fy), _f_inv(fz)], axis=-1) * _D65
    return np.clip(xyz @ _M_XYZ_TO_RGB.T, 0.0, 1.0)


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

def _array(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)


@dataclass
class Geometry:
    position: np.ndarray      # (h, w, 3) world metres
    normal: np.ndarray        # (h, w, 3) unit
    valid: np.ndarray         # (h, w) 0/1
    span: float

    @classmethod
    def load(cls, geometry_dir: str | Path) -> "Geometry":
        geometry_dir = Path(geometry_dir)
        manifest = json.loads((geometry_dir / "geometry.json").read_text(encoding="utf-8"))
        lo = np.asarray(manifest["position"]["min"], dtype=np.float32)
        hi = np.asarray(manifest["position"]["max"], dtype=np.float32)
        position = _array(geometry_dir / manifest["position"]["file"])
        normal = _array(geometry_dir / manifest["normal"]["file"])
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6)
        valid = _array(geometry_dir / manifest["valid"]["file"])
        return cls(position, normal, (valid > 0.5).astype(np.float32), float((hi - lo).max()))


@dataclass
class View:
    name: str
    matrix_world: np.ndarray  # (4, 4)
    ortho_scale: float
    resolution: tuple[int, int]
    depth: np.ndarray         # (H, W) metres
    depth_valid: np.ndarray   # (H, W) bool
    rgb: np.ndarray           # (H, W, 3) linear
    alpha: np.ndarray         # (H, W) 0..1

    @property
    def toward_camera(self) -> np.ndarray:
        # a Blender camera looks down its local -Z, so +Z points back at it
        return self.matrix_world[:3, 2]


def load_views(canonical_dir: str | Path, geometry_dir: str | Path, views_dir: str | Path,
               names=VIEW_NAMES) -> list[View]:
    canonical_dir, geometry_dir, views_dir = Path(canonical_dir), Path(geometry_dir), Path(views_dir)
    # `canonical_dir` is the folder holding cameras.json, or the cameras file
    # itself - the structure guides write theirs as {template}_cameras.json
    cameras_path = canonical_dir if canonical_dir.suffix == ".json" else canonical_dir / "cameras.json"
    cameras = json.loads(cameras_path.read_text(encoding="utf-8"))
    manifest = json.loads((geometry_dir / "geometry.json").read_text(encoding="utf-8"))
    views: list[View] = []
    for name in names:
        cam = cameras["views"][name]
        depth = _array(geometry_dir / manifest["depth"]["files"][name])
        location = np.asarray(cam["location"], dtype=np.float32)
        centre = np.asarray(cameras["model_centre"], dtype=np.float32)
        distance = float(np.linalg.norm(location - centre))
        depth_valid = depth > distance * DEPTH_MIN_FRACTION

        image = Image.open(views_dir / f"{name}.png").convert("RGBA")
        if image.size != tuple(cameras["resolution"]):
            image = image.resize(tuple(cameras["resolution"]), Image.LANCZOS)
        rgba = np.asarray(image, dtype=np.float32) / 255.0
        views.append(View(
            name=name,
            matrix_world=np.asarray(cam["matrix_world"], dtype=np.float32),
            ortho_scale=float(cameras["ortho_scale"]),
            resolution=tuple(cameras["resolution"]),
            depth=depth.astype(np.float32),
            depth_valid=depth_valid,
            rgb=to_linear(rgba[..., :3]),
            alpha=rgba[..., 3],
        ))
    return views


# --------------------------------------------------------------------------
# exposure
# --------------------------------------------------------------------------

# The most a view may be shifted toward the front view, in LAB units. The cells
# come from one generated sheet, so any real difference between them is small;
# a large measured difference is the animal, not the lighting - a side view is
# mostly black back, the front view mostly white chest - and matching it would
# repaint the coat. Unbounded matching once lifted the sides by 22 L.
EXPOSURE_MAX_L = 6.0
EXPOSURE_MAX_AB = 4.0


def match_exposure(views: list[View], reference: str = "front") -> dict[str, float]:
    """Nudge every view's colour toward the reference view, within bounds.

    A view whose mean LAB inside the silhouette differs slightly from the
    front's is shifted by that difference; one that differs a lot is shifted
    only by the bound, because a large difference is real coat, not exposure.
    No contrast scaling, for the same reason.
    """
    by_name = {view.name: view for view in views}
    ref = by_name[reference]
    ref_lab = linear_to_lab(np.clip(ref.rgb, 1e-4, 1.0))[ref.alpha > 0.5]
    ref_mean = ref_lab.mean(axis=0)
    bounds = np.array([EXPOSURE_MAX_L, EXPOSURE_MAX_AB, EXPOSURE_MAX_AB], dtype=np.float32)
    shifts: dict[str, float] = {}
    for view in views:
        if view.name == reference:
            shifts[view.name] = 0.0
            continue
        inside = view.alpha > 0.5
        if inside.sum() < 100 or ref_lab.shape[0] < 100:
            shifts[view.name] = 0.0
            continue
        lab = linear_to_lab(np.clip(view.rgb, 1e-4, 1.0))
        delta = np.clip(ref_mean - lab[inside].mean(axis=0), -bounds, bounds)
        adjusted = lab_to_linear(lab + delta)
        view.rgb = np.where(inside[..., None], adjusted, view.rgb).astype(np.float32)
        shifts[view.name] = float(delta[0])
    return shifts


# --------------------------------------------------------------------------
# a shared palette for the four views
# --------------------------------------------------------------------------

# How many k-means passes; the centroids are seeded from the fur ramps and
# settle in a handful.
KMEANS_ITERATIONS = 20

# At most this many pixels feed the clustering, drawn evenly from all views.
KMEANS_SAMPLE = 120_000

# Colour is pushed this far outward from the UV islands into the gutter, so a
# texture filter sampling across an island edge sees fur, not gutter grey.
ISLAND_PAD_PX = 16

# A texel no view saw takes the colour of the nearest seen texel on the mesh
# when one lies within this fraction of the model's span. The four cameras
# are all at the animal's own height, so the ridge of the back and the top of
# the neck face none of them; those are thin strips with seen coat on both
# sides, and the mask build that used to fill them painted the withers white
# where the chest mask reaches over the shoulder. Only a gap wider than this
# - the belly, the inside of a leg - is the mask build's to fill. Distance
# alone cannot tell the two apart (the inside of a thigh is as close to the
# seen outer thigh as the ridge is to the flank), so the neighbour's colour
# is taken only where it is the colour the record gives that texel's region:
# black beside the black ridge, yes; black onto a white inner thigh, no.
NEAR_FILL_FRACTION = 0.05


def shared_palette(
    views: list[View], stops: list[tuple[str, np.ndarray]]
) -> list[tuple[str, np.ndarray]]:
    """One palette for all four views: k-means over every cell's pixels,
    seeded from the labelled ramp stops so each cluster keeps a name.

    The four cells are the same animal but not to the pixel - the tan on the
    front is a shade off the tan on the side - and projected side by side that
    shade is a seam. Clustering all cells together and rewriting each in the
    shared colours removes the difference at the source.
    """
    pixels = []
    for view in views:
        inside = view.alpha > 0.5
        pixels.append(linear_to_lab(np.clip(view.rgb, 1e-4, 1.0))[inside])
    data = np.concatenate(pixels)
    if data.shape[0] > KMEANS_SAMPLE:
        data = data[np.random.default_rng(0).choice(data.shape[0], KMEANS_SAMPLE, replace=False)]

    names = [name for name, _ in stops]
    centroids = np.stack([lab for _, lab in stops]).astype(np.float32)
    weights = np.array([SNAP_L_WEIGHT, 1.0, 1.0], dtype=np.float32)
    for _ in range(KMEANS_ITERATIONS):
        distance = (((data[:, None, :] - centroids[None, :, :]) * weights) ** 2).sum(axis=-1)
        nearest = distance.argmin(axis=1)
        for k in range(len(names)):
            members = data[nearest == k]
            if members.shape[0] > 20:
                centroids[k] = members.mean(axis=0)
    return [(name, centroids[k]) for k, name in enumerate(names)]


def flat_colours(
    labelled: list[tuple[str, np.ndarray]], ramp_l: dict[str, float]
) -> dict[str, np.ndarray]:
    """One flat LAB per name: chroma is the mean of that name's clusters,
    luminance is the calibrated ramp's mid - the painted L carries the
    model's shading and the ramp's does not."""
    out: dict[str, np.ndarray] = {}
    for name in dict.fromkeys(n for n, _ in labelled):
        members = np.stack([lab for n, lab in labelled if n == name])
        colour = members.mean(axis=0).astype(np.float32)
        if name in ramp_l:
            colour[0] = ramp_l[name]
        out[name] = colour
    return out


def quantize_views(
    views: list[View], labelled: list[tuple[str, np.ndarray]], flat: dict[str, np.ndarray]
) -> None:
    """Rewrite every view's pixels, inside the silhouette, as the flat colour
    of whichever labelled cluster they fall nearest - identical across views."""
    centroids = np.stack([lab for _, lab in labelled]).astype(np.float32)
    names = [name for name, _ in labelled]
    flat_rgb = np.stack([lab_to_linear(flat[name][None, None, :])[0, 0] for name in names])
    weights = np.array([SNAP_L_WEIGHT, 1.0, 1.0], dtype=np.float32)
    for view in views:
        inside = view.alpha > 0.5
        lab = linear_to_lab(np.clip(view.rgb, 1e-4, 1.0))
        distance = (((lab[..., None, :] - centroids[None, None, :, :]) * weights) ** 2).sum(axis=-1)
        nearest = distance.argmin(axis=-1)
        view.rgb = np.where(inside[..., None], flat_rgb[nearest], view.rgb).astype(np.float32)


def restyle_flat(
    rgb: np.ndarray, valid: np.ndarray, candidates: dict[str, np.ndarray],
    ramp_l: dict[str, float], hair: np.ndarray,
) -> np.ndarray:
    """A texture rewritten texel by texel as the nearest palette colour, flat,
    with the template's hair on top - the style the delighting gives the
    projection. No blur: the mask build's boundaries are already where they
    should be, and blurring them first ate its thin regions."""
    names = list(candidates)
    mids = np.stack([candidates[n] for n in names]).astype(np.float32)
    for k, name in enumerate(names):
        if name in ramp_l:
            mids[k, 0] = ramp_l[name]
    lab = linear_to_lab(np.clip(rgb, 1e-4, 1.0))
    weights = np.array([SNAP_L_WEIGHT, 1.0, 1.0], dtype=np.float32)
    flat = lab.reshape(-1, 3)
    distance = (((flat[:, None, :] - mids[None, :, :]) * weights) ** 2).sum(axis=-1)
    nearest = distance.argmin(axis=1)
    out = mids[nearest].reshape(lab.shape).copy()
    out[..., 0] = np.clip(out[..., 0] + hair * TEMPLATE_DETAIL_GAIN, 0.0, 100.0)
    restyled = lab_to_linear(out)
    return np.where((valid > 0.5)[..., None], restyled, rgb)


def fill_unseen_from_neighbours(
    geometry: Geometry, coverage: np.ndarray, snap_map: np.ndarray,
    prior: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Give unseen texels near seen ones the snapped colour of their nearest
    seen neighbour on the mesh, where that colour agrees with the record's
    prior for the texel. Returns the new snap map, the new coverage and the
    share of unseen texels that were filled."""
    from scipy.spatial import cKDTree

    valid = geometry.valid > 0.5
    seen = valid & (coverage > 0.5)
    unseen = valid & ~seen
    if seen.sum() < 100 or unseen.sum() == 0:
        return snap_map, coverage, 0.0
    tree = cKDTree(geometry.position[seen])
    distance, index = tree.query(geometry.position[unseen],
                                 distance_upper_bound=NEAR_FILL_FRACTION * geometry.span)
    hit = np.isfinite(distance)
    if prior is not None:
        neighbour = snap_map[seen][np.where(hit, index, 0)]
        wanted = prior[unseen]
        hit &= (wanted < 0) | (neighbour == wanted)
    new_snap = snap_map.copy()
    new_cov = coverage.copy()
    flat_unseen = np.flatnonzero(unseen.reshape(-1))[hit]
    new_snap.reshape(-1)[flat_unseen] = snap_map[seen][index[hit]]
    new_cov.reshape(-1)[flat_unseen] = 1.0
    return new_snap, new_cov, float(hit.mean())


def fill_outside(image: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Every texel outside the UV islands takes the colour of the nearest
    texel inside one - the whole map, not a rim of it.

    A 16-texel rim was not enough. Beyond it the template's own background
    remained, cream fur on this template, and a viewer's mipmaps average far
    past sixteen texels: at every UV seam a thin light line showed where the
    filtered lookup reached the cream. With the outside filled to the
    horizon there is nothing at any distance but the coat itself."""
    from scipy.ndimage import distance_transform_edt

    inside = valid > 0.5
    if inside.all() or not inside.any():
        return image
    _, (rows, cols) = distance_transform_edt(~inside, return_indices=True)
    return image[rows, cols]


def pad_islands(texture: np.ndarray, valid: np.ndarray, pixels: int = ISLAND_PAD_PX) -> np.ndarray:
    """Push colour outward from the UV islands into the gutter.

    A texture filter sampling near an island edge mixes in whatever lies
    beyond it; left as gutter, that shows as a thin dark line along every UV
    seam once mipmaps kick in. Each pass fills an empty texel with the mean of
    its filled neighbours, so the gutter takes on the nearest fur colour.
    """
    filled = texture.copy()
    known = valid.copy().astype(np.float32)
    for _ in range(pixels):
        summed = np.zeros_like(filled)
        count = np.zeros_like(known)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            shifted_known = np.roll(known, (dy, dx), axis=(0, 1))
            summed += np.roll(filled, (dy, dx), axis=(0, 1)) * shifted_known[..., None]
            count += shifted_known
        grow = (known < 0.5) & (count > 0)
        filled[grow] = (summed[grow] / count[grow][:, None])
        known = np.where(grow, 1.0, known)
    return filled


# --------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------

def _bilinear(image: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    r0 = np.clip(np.floor(rows).astype(int), 0, h - 2)
    c0 = np.clip(np.floor(cols).astype(int), 0, w - 2)
    fr = np.clip(rows - r0, 0, 1)
    fc = np.clip(cols - c0, 0, 1)
    if image.ndim == 2:
        fr_, fc_ = fr, fc
    else:
        fr_, fc_ = fr[..., None], fc[..., None]
    top = image[r0, c0] * (1 - fc_) + image[r0, c0 + 1] * fc_
    bottom = image[r0 + 1, c0] * (1 - fc_) + image[r0 + 1, c0 + 1] * fc_
    return top * (1 - fr_) + bottom * fr_


def project_view(geometry: Geometry, view: View) -> tuple[np.ndarray, np.ndarray]:
    """Colour and weight this view contributes to every texel."""
    h, w, _ = geometry.position.shape
    flat_pos = geometry.position.reshape(-1, 3)
    flat_nrm = geometry.normal.reshape(-1, 3)

    inverse = np.linalg.inv(view.matrix_world)
    homogeneous = np.concatenate([flat_pos, np.ones((flat_pos.shape[0], 1), dtype=np.float32)], axis=1)
    cam = homogeneous @ inverse.T
    half = view.ortho_scale / 2.0
    width, height = view.resolution
    cols = (cam[:, 0] / half + 1.0) / 2.0 * width - 0.5
    rows = (1.0 - (cam[:, 1] / half + 1.0) / 2.0) * height - 0.5
    depth = -cam[:, 2]

    in_frame = (cols >= 0) & (cols <= width - 1) & (rows >= 0) & (rows <= height - 1)
    facing = flat_nrm @ view.toward_camera
    weight = np.where(facing > FACING_FLOOR, facing, 0.0) ** FACING_POWER

    sample_rows = np.where(in_frame, rows, 0.0)
    sample_cols = np.where(in_frame, cols, 0.0)
    seen_depth = _bilinear(view.depth, sample_rows, sample_cols)
    depth_ok = _bilinear(view.depth_valid.astype(np.float32), sample_rows, sample_cols) > 0.5
    visible = in_frame & depth_ok & (depth <= seen_depth + DEPTH_TOLERANCE * geometry.span)

    alpha = _bilinear(view.alpha, sample_rows, sample_cols)
    colour = _bilinear(view.rgb, sample_rows, sample_cols)

    weight = weight * visible * (alpha > 0.5) * geometry.valid.reshape(-1)
    return colour.reshape(h, w, 3), weight.reshape(h, w).astype(np.float32)


@dataclass
class Projected:
    rgb: np.ndarray           # (h, w, 3) linear, weighted average
    coverage: np.ndarray      # (h, w) 0..1 after normalisation and feathering
    raw_weight: np.ndarray    # (h, w) summed facing weights before feathering
    per_view: dict[str, float] = field(default_factory=dict)


def project(geometry: Geometry, views: list[View]) -> Projected:
    h, w, _ = geometry.position.shape
    total_colour = np.zeros((h, w, 3), dtype=np.float32)
    total_weight = np.zeros((h, w), dtype=np.float32)
    per_view: dict[str, float] = {}
    for view in views:
        colour, weight = project_view(geometry, view)
        # blur the weight, then confine it to where the view really saw the
        # surface: a gradual cross-fade at the seams, never colour from a
        # view that could not see the texel
        support = (weight > 0).astype(np.float32)
        weight = _gaussian(weight, WEIGHT_SMOOTH_PX) * support
        total_colour += colour * weight[..., None]
        total_weight += weight
        per_view[view.name] = float((support > 0).mean())
    rgb = total_colour / np.maximum(total_weight, 1e-6)[..., None]

    covered = (total_weight > 1e-4).astype(np.float32)
    blurred = Image.fromarray((covered * 255).astype(np.uint8), mode="L").filter(
        ImageFilter.GaussianBlur(radius=COVERAGE_FEATHER_PX)
    )
    coverage = np.asarray(blurred, dtype=np.float32) / 255.0
    # feathering must never claim texels the projection did not reach
    coverage = np.minimum(coverage, covered)
    coverage *= geometry.valid
    return Projected(rgb, coverage, total_weight, per_view)


# --------------------------------------------------------------------------
# delighting
# --------------------------------------------------------------------------

def _box_pass(array: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """One box blur along `axis` via a cumulative sum, so the cost does not
    grow with the radius."""
    pad = [(0, 0), (0, 0)]
    pad[axis] = (radius, radius)
    padded = np.pad(array, pad, mode="edge")
    shape = list(padded.shape)
    shape[axis] += 1
    cumulative = np.zeros(shape, dtype=np.float64)
    index = [slice(None), slice(None)]
    index[axis] = slice(1, None)
    cumulative[tuple(index)] = np.cumsum(padded, axis=axis)
    width = 2 * radius + 1
    hi, lo = [slice(None), slice(None)], [slice(None), slice(None)]
    hi[axis] = slice(width, None)
    lo[axis] = slice(0, -width)
    return ((cumulative[tuple(hi)] - cumulative[tuple(lo)]) / width).astype(np.float32)


def _gaussian(array: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of a float array, kept in float.

    Pillow will not blur float or 16-bit images, and an 8-bit round trip would
    quantise L to 0.4 units - coarser than the fur detail we are trying to keep.
    Three box passes approximate a Gaussian closely; the box half-width follows
    from the target sigma.
    """
    radius = max(1, int(round(np.sqrt(sigma * sigma * 12.0 / 3.0 + 1.0) / 2.0)))
    out = np.asarray(array, dtype=np.float32)
    for _ in range(3):
        out = _box_pass(out, radius, 0)
        out = _box_pass(out, radius, 1)
    return out


def region_target_luminance(
    masks: MaskLibrary, appearance: dict[str, Any], palette: dict[str, tuple[str, str, str]] | None
) -> np.ndarray:
    """The L each texel should average to: the mid stop of its region's fur
    ramp, or the base coat's where no region claims it."""
    def mid_l(colour: str) -> float:
        key = (colour or "").strip().lower()
        stops = (palette or {}).get(key) or FUR_PALETTE.get(key) or FUR_PALETTE["fawn"]
        return float(linear_to_lab(hex_to_linear(stops[1])[None, None, :])[0, 0, 0])

    h, w = masks.size[1], masks.size[0]
    numerator = np.zeros((h, w), dtype=np.float32)
    denominator = np.zeros((h, w), dtype=np.float32)
    for entry in appearance.get("region_colors") or []:
        mask = masks.get(entry.get("region"))
        if mask is None:
            continue
        numerator += mask * mid_l(entry.get("color", ""))
        denominator += mask
    base = mid_l(appearance.get("base_color", ""))
    target = np.where(denominator > 1e-3, numerator / np.maximum(denominator, 1e-3), base)
    return target.astype(np.float32)


def candidate_colours(
    appearance: dict[str, Any], palette: dict[str, tuple[str, str, str]] | None
) -> dict[str, np.ndarray]:
    """The fur colours this animal is known to have, as LAB of each ramp's mid
    stop. Only colours the appearance names are candidates, so a texel is never
    snapped to a colour the dog does not wear."""
    names = {str(appearance.get("base_color", "")).lower()}
    names |= {str(entry.get("color", "")).lower() for entry in appearance.get("region_colors") or []}
    out: dict[str, np.ndarray] = {}
    for name in names:
        stops = (palette or {}).get(name) or FUR_PALETTE.get(name)
        if stops is None:
            continue
        out[name] = linear_to_lab(hex_to_linear(stops[1])[None, None, :])[0, 0]
    return out


def candidate_stops(
    appearance: dict[str, Any], palette: dict[str, tuple[str, str, str]] | None
) -> list[tuple[str, np.ndarray]]:
    """Every stop of every known colour's ramp, each labelled with its name.

    One centroid per colour was not enough: the model paints the eyebrow tan
    darker than the leg tan, and with a single "tan" sitting at the light end
    the dark brows fell nearer to black and were snapped away. Matching against
    dark, mid and light of each ramp lets a dark tan still be tan; what is
    painted afterwards is the name's one flat colour.
    """
    names = {str(appearance.get("base_color", "")).lower()}
    names |= {str(entry.get("color", "")).lower() for entry in appearance.get("region_colors") or []}
    out: list[tuple[str, np.ndarray]] = []
    for name in sorted(names):
        stops = (palette or {}).get(name) or FUR_PALETTE.get(name)
        if stops is None:
            continue
        for stop in stops:
            out.append((name, linear_to_lab(hex_to_linear(stop)[None, None, :])[0, 0]))
    return out


def region_prior(
    masks: MaskLibrary, appearance: dict[str, Any], names: list[str]
) -> np.ndarray:
    """For every texel, the index (into `names`) of the colour the appearance
    declares for it: its region's colour, or the base coat's; -1 if unknown."""
    h, w = masks.size[1], masks.size[0]
    index = {name: k for k, name in enumerate(names)}
    base = str(appearance.get("base_color", "")).lower()
    prior = np.full((h, w), index.get(base, -1), dtype=np.int64)
    strength = np.zeros((h, w), dtype=np.float32)
    for entry in appearance.get("region_colors") or []:
        mask = masks.get(entry.get("region"))
        colour = str(entry.get("color", "")).lower()
        if mask is None or colour not in index:
            continue
        take = mask > strength
        prior[take] = index[colour]
        strength = np.maximum(strength, mask)
    return prior


def template_hair(template_detail: np.ndarray, sigma: float) -> np.ndarray:
    """The template's fine hair detail, as a signed relative luminance map."""
    return (template_detail - _gaussian(template_detail, sigma)) / max(float(template_detail.mean()), 1e-3)


def delight(
    projected_rgb: np.ndarray,
    coverage: np.ndarray,
    candidates: dict[str, np.ndarray],
    template_detail: np.ndarray,
    prior: np.ndarray | None = None,
    ramp_l: dict[str, float] | None = None,
    force_colour: np.ndarray | None = None,
    paint: dict[str, np.ndarray] | None = None,
    restrict_mask: np.ndarray | None = None,
    restrict_to: frozenset[str] | set[str] | None = None,
    prior_only_mask: np.ndarray | None = None,
    base_name: str | None = None,
    quantize_strength: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Take the painted light out of the projection without moving its colours.

    `candidates` decide which name a texel is nearest to - they are the views'
    own colours, since the views are what is being classified. `paint` is what
    that name is then painted as: the calibrated palette, whose hue comes from
    the photograph when there is one, so a dog whose white is neutral is not
    painted the generator's ivory.

    An earlier version replaced the broad luminance region by region from the
    masks. That put the mask boundaries straight back into a texture whose
    whole point was to be free of them: where the projection painted the chest
    white a little past the chest mask, the white got black's luminance and
    came out grey. So the target is chosen per texel from the projection's own
    colour instead - each texel's broad LAB is snapped to whichever of the
    animal's known fur colours it is nearest to, and that colour's ramp mid
    supplies the luminance. Every texel the model painted black gets black's
    L whatever shading it painted; placement stays the model's.

    Chroma is kept from the projection. Fine luminance detail is kept too, and
    the template's hair is added on top because the generated views are flat
    art with none of their own.
    """
    h, w, _ = projected_rgb.shape
    sigma = DELIGHT_SIGMA_FRACTION * w
    lab = linear_to_lab(np.clip(projected_rgb, 1e-4, 1.0))
    inside = coverage > 0.05
    fill = lab[inside].mean(axis=0) if inside.any() else np.array([50.0, 0.0, 0.0], dtype=np.float32)
    filled = np.where(inside[..., None], lab, fill)
    low = np.stack([_gaussian(filled[..., i], sigma) for i in range(3)], axis=-1)
    high_l = lab[..., 0] - low[..., 0]

    names = list(candidates)
    mids = np.stack([candidates[n] for n in names]).astype(np.float32)   # (k, 3)
    if ramp_l:
        # chroma from the views' own palette, luminance from the calibrated
        # fur ramp: the painted L carries the model's shading, the ramp's does not
        mids = mids.copy()
        for k, name in enumerate(names):
            if name in ramp_l:
                mids[k, 0] = ramp_l[name]
    weights = np.array([SNAP_L_WEIGHT, 1.0, 1.0], dtype=np.float32)
    flat = low.reshape(-1, 3)
    distance = (((flat[:, None, :] - mids[None, :, :]) * weights) ** 2).sum(axis=-1)
    if prior is not None:
        rows = np.arange(distance.shape[0])
        flat_prior = prior.reshape(-1)
        has_prior = flat_prior >= 0
        distance[rows[has_prior], flat_prior[has_prior]] *= SNAP_PRIOR_BIAS
    if restrict_mask is not None and restrict_to:
        blocked = np.array([name not in restrict_to for name in names])
        if blocked.any():
            restricted = restrict_mask.reshape(-1) > 0.5
            distance[np.ix_(restricted, blocked)] += 1e9
    if prior_only_mask is not None and prior is not None:
        # only the texel's own declared colour or the base coat
        rows = prior_only_mask.reshape(-1) > 0.5
        index = np.arange(len(names))
        allowed = index[None, :] == prior.reshape(-1)[rows][:, None]
        if base_name in names:
            allowed |= index[None, :] == names.index(base_name)
        distance[rows] = np.where(allowed, distance[rows], distance[rows] + 1e9)
    # a majority vote over a neighbourhood: one-hot the choice, blur, re-pick
    onehot = np.zeros((len(names), h, w), dtype=np.float32)
    first = distance.argmin(axis=1).reshape(h, w)
    for k in range(len(names)):
        onehot[k] = _gaussian((first == k).astype(np.float32), SNAP_SMOOTH_PX)
    nearest = onehot.argmax(axis=0).reshape(-1)
    if force_colour is not None:
        # Where the face rules paint outright the projected colour is not
        # trusted at all, but the texture must stay in one style: the texel
        # keeps the flat-plus-hair treatment of its neighbours and simply takes
        # the colour the rule gives. Handing those texels to the mask build
        # instead put a ring of a different texture around every eye, which
        # read as mottling.
        flat_force = force_colour.reshape(-1)
        nearest = np.where(flat_force >= 0, flat_force, nearest)
    target_l = mids[nearest, 0].reshape(h, w)
    assigned = np.bincount(nearest[inside.reshape(-1)], minlength=len(names)) / max(int(inside.sum()), 1)

    hair = template_hair(template_detail, sigma)
    # the colour is the snapped colour outright - a and b as well as L - so the
    # texture is as flat as the views it came from; the only fine detail left
    # is the template's hair. The projection's own high frequency was the
    # cartoon's edges and shading, which is what read as mottling.
    painted = mids
    if paint:
        painted = np.stack([paint.get(n, candidates[n]) for n in names]).astype(np.float32)
        painted[:, 0] = mids[:, 0]
    # quantize_strength 1.0 is the flat palette colour outright, the cartoon
    # views' own style; lower keeps that much of the projection's broad colour
    # under the palette, for views drawn in a photographic style whose soft
    # gradients are worth keeping
    q = float(np.clip(quantize_strength, 0.0, 1.0))
    out = np.empty_like(lab)
    flat_l = low[..., 0] + (target_l - low[..., 0]) * q
    out[..., 0] = np.clip(flat_l + hair * TEMPLATE_DETAIL_GAIN, 0.0, 100.0)
    out[..., 1] = low[..., 1] + (painted[nearest, 1].reshape(h, w) - low[..., 1]) * q
    out[..., 2] = low[..., 2] + (painted[nearest, 2].reshape(h, w) - low[..., 2]) * q
    stats = {
        "projected_l_low_std": float(low[..., 0][inside].std()) if inside.any() else 0.0,
        "high_freq_l_std": float(high_l[inside].std()) if inside.any() else 0.0,
        "snapped_to": {name: round(float(share), 3) for name, share in zip(names, assigned)},
        "snap_map": nearest.reshape(h, w),
        "snap_names": names,
    }
    return lab_to_linear(out), stats


def _close(mask: np.ndarray) -> np.ndarray:
    """Morphological closing by RULE_CLOSE_PX."""
    image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    size = 2 * RULE_CLOSE_PX + 1
    image = image.filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.MinFilter(size))
    return np.asarray(image) > 127


def face_rules(
    masks: MaskLibrary, geometry: Geometry, appearance: dict[str, Any], names: list[str],
    photo_face: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Where on the face the record overrides the projection (see FACE RULES).

    With `photo_face` the photograph paints the brows and the cheek patch,
    and rules 1 and 2 - the record's stand-ins for them - are not drawn:
    an ellipse showing through the photo's feathered edge was a second brow.

    Returns, per texel, the index into `names` of a colour painted outright
    (-1 where the projection decides), the texels where a hued colour is not
    allowed, the texels close to an eye, and a few numbers for the report."""
    h, w = masks.size[1], masks.size[0]
    force = np.full((h, w), -1, dtype=np.int64)
    neutral = np.zeros((h, w), dtype=np.float32)
    vicinity = np.zeros((h, w), dtype=np.float32)
    index = {name: k for k, name in enumerate(names)}

    def colour_of(entry: dict[str, Any]) -> int:
        return index.get(str(entry.get("color", "")).lower(), -1)
    valid = geometry.valid > 0.5
    x = geometry.position[..., 0]
    z = geometry.position[..., 2]
    declared = {
        str(entry.get("region")): entry for entry in appearance.get("region_colors") or []
    }
    report: dict[str, Any] = {}

    # the eyes on the mesh: the highest of the features (the nose and tongue
    # sit lower), one centre and radius per side of the midline
    eyes: list[tuple[np.ndarray, float]] = []
    features = masks.extra("_features")
    if features is not None:
        feat = (features > 0.5) & valid
        if feat.any():
            high = feat & (z >= np.percentile(z[feat], 40))
            for side in (x < 0, x > 0):
                sel = high & side
                if sel.sum() < 50:
                    continue
                centre = geometry.position[sel].mean(axis=0)
                radius = float(np.linalg.norm(geometry.position[sel] - centre, axis=1).mean())
                if radius > 0:
                    eyes.append((centre, radius))

    # 2. partial face regions: the declared colour nearest the eyes
    for name, entry in declared.items():
        if photo_face or name not in FACE_REGIONS or name == "eyebrows" or entry.get("coverage") != "partial":
            continue
        mask = masks.get(name)
        if mask is None or not eyes or colour_of(entry) < 0:
            continue
        inside = (mask > 0.5) & valid
        if inside.sum() < 100:
            continue
        per_eye = np.stack([np.linalg.norm(geometry.position - centre, axis=-1) for centre, _ in eyes])
        distance = per_eye.min(axis=0)
        eye_z = np.array([centre[2] for centre, _ in eyes])[per_eye.argmin(axis=0)]
        cut = np.percentile(distance[inside], 100 * FACE_PARTIAL_KEEP)
        near = _close(inside & (distance <= cut) & (z <= eye_z))
        force[near] = colour_of(entry)
        report[f"{name}_near_eye_share"] = round(float(near.sum() / max(inside.sum(), 1)), 3)

    # 1. pips above each eye, painted last so they win over a cheek patch.
    #    The ellipse is cut against the mesh alone, not the eyebrow mask - that
    #    mask's baked edge is a staircase, and the pip took its shape - and not
    #    the face masks' union either, which has gaps at the eyelids.
    if not photo_face and "eyebrows" in declared and eyes and colour_of(declared["eyebrows"]) >= 0:
        pips = np.zeros((h, w), dtype=bool)
        for centre, radius in eyes:
            dx = (x - centre[0]) / (PIP_HALF_WIDTH * radius)
            dz = (z - (centre[2] + PIP_RISE * radius)) / (PIP_HALF_HEIGHT * radius)
            pips |= (dx * dx + dz * dz <= 1.0)
        pips = _close(pips & valid)
        force[pips] = colour_of(declared["eyebrows"])
        report["eyebrow_pips_texels"] = int(pips.sum())

    # 5. inner ears from the record
    for name in EAR_RECORD_REGIONS:
        entry = declared.get(name)
        mask = masks.get(name)
        if entry is None or mask is None or entry.get("coverage") == "partial" or colour_of(entry) < 0:
            continue
        inside = (mask > 0.5) & valid
        force[inside] = colour_of(entry)
        report[f"{name}_from_record"] = int(inside.sum())

    # 4. the eye vicinity
    for centre, radius in eyes:
        near = np.linalg.norm(geometry.position - centre, axis=-1) <= EYE_VICINITY * radius
        vicinity = np.maximum(vicinity, (near & valid).astype(np.float32))

    # 3. undeclared face regions: no hued colour
    for name in sorted(FACE_REGIONS - set(declared)):
        mask = masks.get(name)
        if mask is not None:
            neutral = np.maximum(neutral, mask)

    return force, neutral, vicinity, report


# --------------------------------------------------------------------------
# the whole step
# --------------------------------------------------------------------------

def post_process(image: Image.Image) -> Image.Image:
    """Identity for now. The seat for an img2img refinement of the finished
    texture; keep the signature so a caller never has to change."""
    return image


def _save(debug: Path, name: str, array: np.ndarray, linear: bool = True) -> None:
    debug.mkdir(parents=True, exist_ok=True)
    if array.ndim == 2:
        data = (np.clip(array, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(data, mode="L").save(debug / f"{name}.png")
    else:
        data = (to_srgb(array) if linear else np.clip(array, 0, 1)) * 255
        Image.fromarray(data.astype(np.uint8), mode="RGB").save(debug / f"{name}.png")


def compose_projected(
    base_texture_path: str | Path,
    masks_dir: str | Path,
    appearance: dict[str, Any],
    canonical_dir: str | Path,
    geometry_dir: str | Path,
    views_dir: str | Path,
    palette: dict[str, tuple[str, str, str]] | None = None,
    consistency_score: float | None = None,
    debug_dir: str | Path | None = None,
    face_photo: "FacePhoto | None" = None,
    exclude_head_region: bool = False,
    quantize_strength: float = 1.0,
    apply_face_rules: bool = True,
) -> tuple[Image.Image, dict[str, Any]]:
    """Projection, delighting and the fallback composite in one call.

    `canonical_dir` is the folder with cameras.json or the cameras file
    itself. `exclude_head_region` keeps every projected colour off the head
    (the face regions and the ears), so the mask build paints the head from
    the record - the face-preset fallback for when generated views do not
    line up with the mesh. `quantize_strength` is how far the projected
    colour is flattened to the palette (see delight). `apply_face_rules`
    False leaves the brows and cheeks to the views: for views drawn to the
    template's own structure guides the record's ellipses would only double
    what the drawing already put in the right place.

    With `face_photo` - the photograph and its four landmarks - the face
    regions take the photograph's own pixels over the delit projection (see
    face_photo.py); the face rules above then only matter where the photo did
    not reach."""
    debug = Path(debug_dir) if debug_dir else None
    base_image = Image.open(base_texture_path).convert("RGB")

    geometry = Geometry.load(geometry_dir)
    views = load_views(canonical_dir, geometry_dir, views_dir)
    shifts = match_exposure(views)

    ramp_candidates = candidate_colours(appearance, palette)
    ramp_l = {name: float(lab[0]) for name, lab in ramp_candidates.items()}
    labelled = shared_palette(views, [(name, lab) for name, lab in ramp_candidates.items()])
    view_palette = flat_colours(labelled, ramp_l)
    # the views are left continuous: flattening them here fixed the dark
    # eyebrow tan as black before the mask prior could argue for it

    projected = project(geometry, views)

    masks = MaskLibrary(masks_dir, base_image.size)
    head_excluded_share = 0.0
    if exclude_head_region:
        head = masks.union(FACE_REGIONS)
        head_excluded_share = float(((head > 0.5) & (projected.coverage > 0.5)).sum()
                                    / max((projected.coverage > 0.5).sum(), 1))
        projected.coverage = projected.coverage * (1.0 - np.clip(head, 0.0, 1.0))
    # on the face the record and the mesh override the drawing where the
    # drawing is known to land its markings off (FACE RULES above); the
    # overridden texels are painted in the same flat style as their neighbours
    force_colour, neutral_only, eye_vicinity, face_rules_report = face_rules(
        masks, geometry, appearance, list(view_palette),
        photo_face=(face_photo is not None and face_photo.complete) or not apply_face_rules)
    base_name = str(appearance.get("base_color", "")).lower()
    restrict_to = UNDECLARED_FACE_ALLOWED | {base_name}
    features = masks.extra("_features")
    eye_ring = None
    if features is not None:
        # the ring is drawn around the features themselves, not the fur the
        # lid bones move: that fur is coat, and the rules above decide it
        features = feature_core(features, base_image)
        grown = Image.fromarray((np.clip(features, 0, 1) * 255).astype(np.uint8), mode="L")
        grown = grown.filter(ImageFilter.MaxFilter(2 * FEATURE_EXCLUSION_PX + 1))
        eye_ring = np.asarray(grown, dtype=np.float32) / 255.0

    detail = brightness_map(base_image)
    prior = region_prior(masks, appearance, list(view_palette))
    delit, stats = delight(projected.rgb, projected.coverage, view_palette,
                           detail, prior, ramp_l, force_colour,
                           paint=ramp_candidates, restrict_mask=neutral_only,
                           restrict_to=restrict_to, prior_only_mask=eye_vicinity,
                           base_name=base_name, quantize_strength=quantize_strength)
    if eye_ring is not None:
        # inpaint the ring from its own surroundings: the texels just outside
        # it carry the projected face colour, and that colour - not the
        # record's, not the mask build's - is what belongs beside the eye
        delit = pad_islands(delit, (eye_ring < 0.5).astype(np.float32), FEATURE_EXCLUSION_PX + 4)
    snap_map, snap_names = stats.pop("snap_map"), stats.pop("snap_names")

    # thin unseen strips - the ridge of the back, the top of the neck - take
    # the coat beside them, in the same flat-plus-hair style
    snap_map, filled_cov, near_share = fill_unseen_from_neighbours(
        geometry, projected.coverage, snap_map, prior)
    newly = (filled_cov > 0.5) & (projected.coverage <= 0.5)
    if newly.any():
        paint_lab = np.stack([ramp_candidates[n] for n in snap_names]).astype(np.float32)
        hair = template_hair(detail, DELIGHT_SIGMA_FRACTION * detail.shape[1]) * TEMPLATE_DETAIL_GAIN
        lab = paint_lab[snap_map]
        lab[..., 0] = np.clip(lab[..., 0] + hair, 0.0, 100.0)
        filled_rgb = lab_to_linear(lab)
        delit = np.where(newly[..., None], filled_rgb, delit)
        projected.coverage = filled_cov
    stats["near_filled_share_of_unseen"] = round(near_share, 3)

    face_report: dict[str, Any] = {}
    face_projection = None
    if face_photo is not None and face_photo.complete:
        from .face_photo import project_face

        front = next(view for view in views if view.name == "front")
        hair = template_hair(detail, DELIGHT_SIGMA_FRACTION * detail.shape[1]) * TEMPLATE_DETAIL_GAIN
        face_prior = region_prior(masks, appearance, list(ramp_candidates))
        face_projection = project_face(geometry, front, face_photo, masks, ramp_candidates, hair, face_prior)
        w = face_projection.weight[..., None]
        delit = delit * (1.0 - w) + face_projection.rgb * w
        projected.coverage = np.maximum(projected.coverage, face_projection.weight)
        face_report = face_projection.report

    fallback = synthesize_texture(base_texture_path, masks_dir, appearance, palette=palette)
    fallback_lin = to_linear(np.asarray(fallback, dtype=np.float32))
    # the mask build in the projection's own style - flat palette colour plus
    # the template's hair - so where the two meet under the feather nothing
    # but the colour can differ; the mask build's own noise and brightness
    # map made the boundary a visible line
    fallback_lin = restyle_flat(fallback_lin, geometry.valid, ramp_candidates, ramp_l,
                                template_hair(detail, DELIGHT_SIGMA_FRACTION * detail.shape[1]))

    weight = projected.coverage.copy()
    score = 1.0 if consistency_score is None else float(consistency_score)
    if score < CONSISTENCY_FLOOR:
        weight *= score
    final = delit * weight[..., None] + fallback_lin * (1.0 - weight[..., None])
    final = keep_features(final, base_image, masks)
    final = fill_outside(final, geometry.valid)

    image = post_process(Image.fromarray((to_srgb(final) * 255).astype(np.uint8), mode="RGB"))

    if debug:
        _save(debug, "01_projected_raw", projected.rgb)
        _save(debug, "02_coverage", projected.coverage)
        _save(debug, "03_delight_before", projected.rgb * (projected.coverage > 0.05)[..., None])
        _save(debug, "04_delight_after", delit * (projected.coverage > 0.05)[..., None])
        _save(debug, "05_fallback", fallback_lin)
        _save(debug, "06_final", final)
        _save(debug, "07_weight", weight)
        # which fur colour each texel was snapped to, for checking the blotches
        palette_rgb = {"black": (0.1, 0.1, 0.1), "white": (0.95, 0.95, 0.9), "tan": (0.7, 0.45, 0.2),
                       "cream": (0.9, 0.85, 0.7), "fawn": (0.75, 0.6, 0.4), "red": (0.7, 0.35, 0.15),
                       "chocolate": (0.3, 0.18, 0.1), "gray": (0.5, 0.5, 0.5)}
        snap_rgb = np.zeros((*snap_map.shape, 3), dtype=np.float32)
        for k, name in enumerate(snap_names):
            snap_rgb[snap_map == k] = palette_rgb.get(name, (1, 0, 1))
        _save(debug, "08_snap_map", snap_rgb * (projected.coverage > 0.05)[..., None], linear=False)
        if face_projection is not None:
            face_projection.warped.save(debug / "09_face_warp.png")
            _save(debug, "10_face_weight", face_projection.weight)

    report = {
        "view_palette": {name: [round(float(v), 1) for v in lab] for name, lab in view_palette.items()},
        "coverage": float((projected.coverage > 0.5).mean()),
        "coverage_of_valid": float((projected.coverage > 0.5).sum() / max(geometry.valid.sum(), 1)),
        "per_view_hit": projected.per_view,
        "exposure_shifts": shifts,
        "consistency_score": score,
        "projection_weight_scaled": score < CONSISTENCY_FLOOR,
        "paint_palette": {name: [round(float(v), 1) for v in lab] for name, lab in ramp_candidates.items()},
        "face_rules": face_rules_report,
        "face_photo": face_report,
        "head_excluded_share": round(head_excluded_share, 4),
        "quantize_strength": float(quantize_strength),
        **stats,
    }
    return image, report


__all__ = [
    "Geometry", "Projected", "View", "compose_projected", "delight", "lab_to_linear",
    "linear_to_lab", "load_views", "match_exposure", "post_process", "project",
    "project_view", "region_target_luminance",
]
