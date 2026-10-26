"""The face from the photograph itself.

WHY
---
The generated sheet gives the face about a hundred pixels, and the drawing's
head is rounder than the mesh's, so on the face the projection lands every
marking a little off. The rules in projection.py repair that with ellipses -
a fair likeness of a tan-point pattern, not of this dog. The photograph has
the dog's actual face: the shape of the brow pips, how far the tan runs under
the eye, where the mask fades. This module puts those pixels on the mesh.

HOW
---
Four landmarks - the two eyes, the nose, the top of the head - are read off
the photograph (by a vision call, or given by hand) and located on the mesh
from the features mask and the skull mask. An affine map taking the photo's
landmarks onto the mesh's, as seen by the canonical front camera, warps the
photograph into the front view's frame, and the ordinary projection then
carries it onto the face regions. Nothing else changes: the body keeps the
generated views, the eyes are the template's, the coat colours stay the
calibrated palette's - the photograph's lightness is stretched to the
palette's range so the face and body meet.

A photograph is one angle. The nose's offset from the midpoint of the eyes
says which way the head is turned; the far half of the face, which the photo
shows foreshortened or not at all, takes the mirror of the near half.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from .agents import Transport
from .palette import _background
from .texture import MaskLibrary

LANDMARKS = ("left_eye", "right_eye", "nose", "head_top")

# The regions the photograph is allowed to paint. Ears are left out: their
# angle in a photo has nothing to do with their angle on the mesh.
FACE_PHOTO_REGIONS = frozenset({"skull", "eyebrows", "cheeks", "muzzle_bridge"})

# The photograph's face is blended into the projected coat over this many
# texels at the edge of the face regions.
FACE_BLEND_PX = 12

# The nose sits this far from the eyes' midpoint, in eye distances, before the
# head counts as turned and the far half of the face is mirrored.
TURN_THRESHOLD = 0.12

# Two texels are mirror images if their reflected positions are within this
# fraction of the model's span.
MIRROR_TOLERANCE = 0.01

# A texel facing the front camera less than this takes no photo colour: at a
# grazing angle the warp's error is largest.
MIN_FACING = 0.35

# The photo paints the face only up to this many eye radii above the eyes'
# centre - just over the brows. Higher is the crown and the ear bases, and
# the photo's ears sit lower and wider than the mesh's, which an affine map
# cannot know: warped, their inner tan landed beside the brows.
ABOVE_EYES = 2.5

# Above the eyes the photo paints only this many eye radii outward of the
# outer eye: further out is the ear base, and on a three-quarter photo the
# near ear's inner tan warps onto the side of the forehead.
BESIDE_EYES = 1.0

# (An earlier version also limited the colours allowed above the eyes to the
# base coat and the eyebrows' colour, against the near ear, which on a
# three-quarter photo stands straight above the near eye. It turned this
# dog's cream brows tan. The lateral cut above and the record prior below
# do that job without deciding the brows' colour.)

# The photograph's lightness is stretched so these percentiles of the face's
# pixels land on the palette's darkest and lightest mid stops. The low end is
# the tenth percentile, not the second: the second is the pupils, and pinning
# them to black's lightness lifted the black fur above it, where the snap
# read the muzzle's dark brown as tan.
LIGHTNESS_LOW, LIGHTNESS_HIGH = 10.0, 98.0

# The record's colour for a texel's region has its snap distance scaled by
# this, as in the projection. It settles the ambiguous pixels only - a dark
# brown on the nose bridge, which the record calls black - and not the clear
# ones: cream brows stay cream against a record that calls them tan.
FACE_PRIOR_BIAS = 0.6

# How decided a pixel is: the margin between its nearest palette colour and
# the next, as a share of their sum. At or above this the pixel snaps with
# the full strength; below, in proportion - so the soft fade where the black
# of the muzzle runs into the white of its sides stays a fade.
SNAP_FULL_MARGIN = 0.4

# On a turned head the far half takes the mirror of the near half - except
# where the far half faces the camera squarely enough that its own pixels
# are the better witness: the nose bridge and the inner corner of the eye.
# Further round, the far cheek is foreshortened and an affine map smears it;
# at 0.5 the far cheek's own tan patch landed beside the eye.
MIRROR_OWN_FACING = 0.75

# Then each texel is pulled this far toward the palette colour it is nearest
# to. The body is flat palette colour plus the template's hair; a photograph
# is shading, noise and compression on top of the same colours, and left as
# it is the face reads as a grey haze against the body. Pulled all the way it
# would be the flat cartoon again; this far, the photo's soft marking edges
# survive and its greys do not.
FACE_SNAP_STRENGTH = 0.85
FACE_SNAP_L_WEIGHT = 0.5

# The template's hair is added to the face at this share of the body's gain,
# so the two meet in one texture.
FACE_HAIR_SHARE = 0.6

# The features on the mesh: the eyes are the highest features, the nose tip the
# most forward of the rest, the head top the highest skull texels.
EYE_Z_PERCENTILE = 40
NOSE_FORWARD_PERCENTILE = 3
HEAD_TOP_PERCENTILE = 98


@dataclass(frozen=True)
class FacePhoto:
    photo_path: Path
    landmarks: dict[str, tuple[float, float]]   # pixel x, y in the photo

    @classmethod
    def load(cls, photo_path: str | Path, landmarks_path: str | Path) -> "FacePhoto":
        data = json.loads(Path(landmarks_path).read_text(encoding="utf-8"))
        points = {name: (float(data[name]["x"]), float(data[name]["y"]))
                  for name in LANDMARKS if name in data and data[name] is not None}
        return cls(Path(photo_path), points)

    @property
    def complete(self) -> bool:
        return all(name in self.landmarks for name in LANDMARKS)


# --------------------------------------------------------------------------
# landmarks on the photograph
# --------------------------------------------------------------------------

LANDMARK_SYSTEM = """You locate four points on a photograph of a dog. Coordinates are pixels \
in the image exactly as given, origin at the top-left corner, x to the right, y down.

- left_eye: the centre of the eye that appears on the LEFT side of the image.
- right_eye: the centre of the eye that appears on the RIGHT side of the image.
- nose: the centre of the nose leather (the black tip of the muzzle).
- head_top: the top of the skull on the midline between the ears, where the \
fur of the forehead ends - not the tip of an ear.

Mark a point not visible in the photo as visible=false and still give your \
best estimate of where it would be. Be precise: the points align the \
photograph to a 3D model, and an error of a few pixels is visible."""

LANDMARK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        name: {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "visible": {"type": "boolean"},
            },
            "required": ["x", "y", "visible"],
            "additionalProperties": False,
        }
        for name in LANDMARKS
    },
    "required": list(LANDMARKS),
    "additionalProperties": False,
}


def _image_block(image: Image.Image) -> dict[str, Any]:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png",
                   "data": base64.standard_b64encode(buffer.getvalue()).decode("ascii")},
    }


async def read_landmarks(photo_path: str | Path, transport: Transport) -> dict[str, Any]:
    """The four landmarks off the photograph, via the vision transport."""
    image = Image.open(photo_path).convert("RGB")
    width, height = image.size
    blocks = [
        _image_block(image),
        {"type": "text", "text": f"The image is {width} pixels wide and {height} pixels tall. "
                                 "Give the four points."},
    ]
    payload = await transport.complete(system=LANDMARK_SYSTEM, blocks=blocks, schema=LANDMARK_SCHEMA)
    out: dict[str, Any] = {}
    for name in LANDMARKS:
        point = payload.get(name) or {}
        out[name] = {
            "x": float(np.clip(float(point.get("x", 0.0)), 0, width - 1)),
            "y": float(np.clip(float(point.get("y", 0.0)), 0, height - 1)),
            "visible": bool(point.get("visible", True)),
        }
    return out


def draw_landmarks(photo_path: str | Path, landmarks: dict[str, Any], out_path: str | Path) -> None:
    """The landmarks marked on the photograph, for a person to check."""
    from PIL import ImageDraw

    image = Image.open(photo_path).convert("RGB")
    scale = max(1, 800 // max(image.size))
    image = image.resize((image.size[0] * scale, image.size[1] * scale), Image.LANCZOS)
    draw = ImageDraw.Draw(image)
    for name, point in landmarks.items():
        x, y = (point["x"] if isinstance(point, dict) else point[0]) * scale, \
               (point["y"] if isinstance(point, dict) else point[1]) * scale
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(255, 40, 40), width=2)
        draw.text((x + 8, y - 6), name, fill=(255, 40, 40))
    image.save(out_path)


# --------------------------------------------------------------------------
# landmarks on the mesh
# --------------------------------------------------------------------------

def mesh_landmarks(geometry, masks: MaskLibrary) -> dict[str, np.ndarray]:
    """World positions of the four landmarks, from the features and skull
    masks. Keys are 'eye_a', 'eye_b' (either side), 'nose', 'head_top'."""
    valid = geometry.valid > 0.5
    position = geometry.position
    x, y, z = position[..., 0], position[..., 1], position[..., 2]
    features = masks.extra("_features")
    skull = masks.get("skull")
    if features is None or skull is None:
        raise ValueError("mesh landmarks need the _features and skull masks")

    feat = (features > 0.5) & valid
    high = feat & (z >= np.percentile(z[feat], EYE_Z_PERCENTILE))
    out: dict[str, np.ndarray] = {}
    for key, side in (("eye_a", x < 0), ("eye_b", x > 0)):
        sel = high & side
        if sel.sum() < 50:
            raise ValueError(f"too few eye texels on one side ({int(sel.sum())})")
        out[key] = position[sel].mean(axis=0)

    low = feat & ~high
    forward = low & (y <= np.percentile(y[low], NOSE_FORWARD_PERCENTILE))
    out["nose"] = position[forward].mean(axis=0)

    top = (skull > 0.5) & valid
    top &= z >= np.percentile(z[top], HEAD_TOP_PERCENTILE)
    out["head_top"] = position[top].mean(axis=0)
    return out


def to_front_pixels(view, points: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    """Where each world point lands in the front view's image."""
    inverse = np.linalg.inv(view.matrix_world)
    half = view.ortho_scale / 2.0
    width, height = view.resolution
    out: dict[str, tuple[float, float]] = {}
    for name, point in points.items():
        cam = inverse @ np.array([*point, 1.0], dtype=np.float32)
        u = (cam[0] / half + 1.0) / 2.0 * width - 0.5
        v = (1.0 - (cam[1] / half + 1.0) / 2.0) * height - 0.5
        out[name] = (float(u), float(v))
    return out


def front_landmarks(view, mesh_points: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    """The mesh landmarks named as the photo's are: left_eye is whichever eye
    the front camera sees on the left of its image."""
    pixels = to_front_pixels(view, mesh_points)
    a, b = pixels["eye_a"], pixels["eye_b"]
    left, right = (a, b) if a[0] <= b[0] else (b, a)
    return {"left_eye": left, "right_eye": right, "nose": pixels["nose"], "head_top": pixels["head_top"]}


# --------------------------------------------------------------------------
# the warp
# --------------------------------------------------------------------------

def affine_fit(src: list[tuple[float, float]], dst: list[tuple[float, float]]) -> np.ndarray:
    """Least-squares affine taking src points onto dst: a (2, 3) matrix."""
    s = np.asarray(src, dtype=np.float64)
    d = np.asarray(dst, dtype=np.float64)
    design = np.concatenate([s, np.ones((s.shape[0], 1))], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, d, rcond=None)
    return coefficients.T   # (2, 3)


def warp_photo(photo: Image.Image, photo_points: dict[str, tuple[float, float]],
               front_points: dict[str, tuple[float, float]], size: tuple[int, int]) -> Image.Image:
    """The photograph laid into the front view's frame so that its landmarks
    sit on the mesh's."""
    names = [name for name in LANDMARKS if name in photo_points and name in front_points]
    forward = affine_fit([photo_points[n] for n in names], [front_points[n] for n in names])
    full = np.vstack([forward, [0.0, 0.0, 1.0]])
    inverse = np.linalg.inv(full)[:2]     # output pixel -> photo pixel
    return photo.transform(size, Image.AFFINE, tuple(inverse.reshape(-1)), resample=Image.BICUBIC)


def photo_rgba(photo_path: str | Path) -> Image.Image:
    """The photograph white-balanced by its border, with the animal cut out."""
    from .imagegen import _rembg_alpha

    image = Image.open(photo_path).convert("RGB")
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    background = _background(rgb)
    illuminant = background / max(float(background.mean()), 1e-4)
    balanced = np.clip(rgb / np.maximum(illuminant, 1e-3), 0.0, 1.0)
    alpha = _rembg_alpha(image)
    if alpha is None:
        alpha = np.ones(rgb.shape[:2], dtype=np.float32)
    rgba = np.concatenate([balanced, np.clip(alpha, 0, 1)[..., None]], axis=-1)
    return Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA")


def head_turn(points: dict[str, tuple[float, float]]) -> float:
    """How far the nose sits from the eyes' midpoint, in eye distances, signed
    toward the image's right."""
    left, right, nose = points["left_eye"], points["right_eye"], points["nose"]
    eye_distance = max(float(np.hypot(right[0] - left[0], right[1] - left[1])), 1e-3)
    midpoint = ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
    return float((nose[0] - midpoint[0]) / eye_distance)


# --------------------------------------------------------------------------
# onto the mesh
# --------------------------------------------------------------------------

@dataclass
class FaceProjection:
    rgb: np.ndarray        # (h, w, 3) linear
    weight: np.ndarray     # (h, w) 0..1, feathered
    report: dict[str, Any]
    warped: Image.Image


def project_face(geometry, front_view, face: FacePhoto, masks: MaskLibrary,
                 palette_lab: dict[str, np.ndarray] | None = None,
                 hair: np.ndarray | None = None,
                 prior: np.ndarray | None = None) -> FaceProjection:
    """The photograph's face on the face regions of the mesh.

    `palette_lab` is the calibrated palette's mid stop per colour name, in
    LAB; `hair` the template's hair detail already scaled by the body's gain;
    `prior` the record's colour per texel as an index into `palette_lab`'s
    names (-1 where the record says nothing)."""
    from dataclasses import replace

    from .fur import to_linear
    from .projection import lab_to_linear, linear_to_lab, project_view

    mesh_points = mesh_landmarks(geometry, masks)
    front_points = front_landmarks(front_view, mesh_points)
    photo_points = {name: face.landmarks[name] for name in LANDMARKS if name in face.landmarks}
    if len(photo_points) < 3:
        raise ValueError("at least three photo landmarks are needed")

    photo = photo_rgba(face.photo_path)
    warped = warp_photo(photo, photo_points, front_points, front_view.resolution)
    rgba = np.asarray(warped, dtype=np.float32) / 255.0
    view = replace(front_view, rgb=to_linear(rgba[..., :3]), alpha=rgba[..., 3])

    colour, weight = project_view(geometry, view)
    facing = geometry.normal @ front_view.toward_camera
    weight = np.where(facing >= MIN_FACING, weight, 0.0).astype(np.float32)
    face_mask = masks.union(FACE_PHOTO_REGIONS)
    weight *= (face_mask > 0.5)
    eye_z = (mesh_points["eye_a"][2] + mesh_points["eye_b"][2]) / 2.0
    eye_x = (abs(mesh_points["eye_a"][0]) + abs(mesh_points["eye_b"][0])) / 2.0
    eye_r = float(np.linalg.norm(mesh_points["eye_a"] - mesh_points["eye_b"])) / 4.0
    z = geometry.position[..., 2]
    x = np.abs(geometry.position[..., 0])
    weight *= (z <= eye_z + ABOVE_EYES * eye_r)
    weight *= ~((z > eye_z) & (x > eye_x + BESIDE_EYES * eye_r))

    # the far half of a turned head takes the mirror of the near half
    turn = head_turn(photo_points)
    mirrored = 0.0
    if abs(turn) >= TURN_THRESHOLD:
        near_side = _near_side(front_view, turn)
        x = geometry.position[..., 0]
        far = (x * near_side < 0) & (geometry.valid > 0.5) & (face_mask > 0.5)
        far &= ~((facing >= MIRROR_OWN_FACING) & (weight > 0))
        colour, weight, mirrored = _mirror_fill(geometry, colour, weight, far)

    if palette_lab:
        names = list(palette_lab)
        mids = np.stack([palette_lab[n] for n in names]).astype(np.float32)
        lightness_range = (float(mids[:, 0].min()), float(mids[:, 0].max()))
        colour = _stretch_lightness(colour, weight > 0, lightness_range, linear_to_lab, lab_to_linear)
        colour = _soft_snap(colour, weight > 0, mids, hair, linear_to_lab, lab_to_linear, prior)

    blurred = Image.fromarray((np.clip(weight, 0, 1) * 255).astype(np.uint8), mode="L").filter(
        ImageFilter.GaussianBlur(radius=FACE_BLEND_PX))
    feathered = np.minimum(np.asarray(blurred, dtype=np.float32) / 255.0, (weight > 0).astype(np.float32) * 1.0)
    # the feather only fades in from the outside: inside the support the photo
    # is used in full
    inside = Image.fromarray(((weight > 0) * 255).astype(np.uint8), mode="L").filter(
        ImageFilter.MinFilter(2 * FACE_BLEND_PX + 1))
    core = np.asarray(inside, dtype=np.float32) / 255.0
    final_weight = np.maximum(core, feathered)

    report = {
        "photo_landmarks": {k: [round(v[0], 1), round(v[1], 1)] for k, v in photo_points.items()},
        "front_landmarks": {k: [round(v[0], 1), round(v[1], 1)] for k, v in front_points.items()},
        "head_turn": round(turn, 3),
        "mirrored_share": round(float(mirrored), 3),
        "face_share_of_valid": round(float((final_weight > 0.5).sum() / max(geometry.valid.sum(), 1)), 4),
    }
    return FaceProjection(colour, final_weight, report, warped)


def _near_side(front_view, turn: float) -> float:
    """The sign of world x on the side of the face the photo shows best: the
    head turned toward the image's right shows its image-right half, and the
    camera's right vector says which world x that is."""
    right = front_view.matrix_world[:3, 0]
    return float(np.sign(right[0]) * np.sign(turn)) or 1.0


def _mirror_fill(geometry, colour: np.ndarray, weight: np.ndarray, far: np.ndarray):
    """Give every far-side texel the colour of its mirror image on the near side."""
    from scipy.spatial import cKDTree

    near = (weight > 0) & ~far
    if near.sum() < 100 or far.sum() == 0:
        return colour, weight, 0.0
    near_positions = geometry.position[near]
    tree = cKDTree(near_positions)
    far_positions = geometry.position[far] * np.array([-1.0, 1.0, 1.0], dtype=np.float32)
    distance, index = tree.query(far_positions, distance_upper_bound=MIRROR_TOLERANCE * geometry.span)
    hit = np.isfinite(distance)
    near_colour = colour[near]
    near_weight = weight[near]
    out_colour = colour.copy()
    out_weight = weight.copy()
    far_index = np.flatnonzero(far.reshape(-1))
    taken = far_index[hit]
    out_colour.reshape(-1, 3)[taken] = near_colour[index[hit]]
    out_weight.reshape(-1)[taken] = near_weight[index[hit]]
    # far texels the mirror could not reach keep nothing
    out_weight.reshape(-1)[far_index[~hit]] = 0.0
    return out_colour, out_weight, float(hit.mean())


def _soft_snap(colour, inside, mids, hair, linear_to_lab, lab_to_linear, prior=None):
    lab = linear_to_lab(np.clip(colour, 1e-4, 1.0))
    flat = lab.reshape(-1, 3)
    weights = np.array([FACE_SNAP_L_WEIGHT, 1.0, 1.0], dtype=np.float32)
    distance = (((flat[:, None, :] - mids[None, :, :]) * weights) ** 2).sum(axis=-1)
    if prior is not None:
        flat_prior = prior.reshape(-1)
        rows = np.flatnonzero(flat_prior >= 0)
        distance[rows, flat_prior[rows]] *= FACE_PRIOR_BIAS
    order = np.argsort(distance, axis=1)
    nearest = order[:, 0]
    d1 = distance[np.arange(distance.shape[0]), nearest]
    d2 = distance[np.arange(distance.shape[0]), order[:, 1]] if distance.shape[1] > 1 else d1 + 1.0
    margin = (d2 - d1) / np.maximum(d2 + d1, 1e-6)
    strength = FACE_SNAP_STRENGTH * np.clip(margin / SNAP_FULL_MARGIN, 0.0, 1.0)
    target = mids[nearest].reshape(lab.shape)
    snapped = lab + strength.reshape(lab.shape[:2])[..., None] * (target - lab)
    if hair is not None:
        snapped[..., 0] = np.clip(snapped[..., 0] + hair * FACE_HAIR_SHARE, 0.0, 100.0)
    out = np.where(inside[..., None], snapped, lab)
    return lab_to_linear(out)


def _stretch_lightness(colour, inside, lightness_range, linear_to_lab, lab_to_linear):
    lab = linear_to_lab(np.clip(colour, 1e-4, 1.0))
    if inside.sum() < 100:
        return colour
    low, high = np.percentile(lab[..., 0][inside], [LIGHTNESS_LOW, LIGHTNESS_HIGH])
    target_low, target_high = lightness_range
    if high - low < 1e-3:
        return colour
    scale = (target_high - target_low) / (high - low)
    lab[..., 0] = np.clip(target_low + (lab[..., 0] - low) * scale, 0.0, 100.0)
    return lab_to_linear(lab)


# ==========================================================================
# DEFAULT PATH - the photograph's face warped straight into UV space
# ==========================================================================
#
# Everything above projects the photo through the canonical front camera,
# which needs the experimental path's geometry and cameras. The default path
# is simpler and needs neither: the face UV island IS a front view of the
# face, so the photo's eyes and nose are matched to the island's (from
# landmarks.json) and a thin-plate-spline warp lays the photo onto the
# island directly. It only runs when the face-crop call said the head is
# frontal, and only paints the six face regions on that island. Whether the
# warp actually landed is checked afterwards, in the ring around the
# protected eyes and nose (see texture.compose_texture): a misaligned face
# puts photo eyes where the template's fur is, and the structural
# similarity to the mask build drops.

FACE_LANDMARKS = ("left_eye", "right_eye", "nose", "left_ear_base", "right_ear_base")
# GRID REFERENCING
# ----------------
# Asked for pixel coordinates, gpt-4o put this dog's eyes in the background
# beside its ear - twice, in agreement. Vision models read a picture well and
# count pixels badly. So the crop is sent with a lettered grid drawn over it
# (columns A.., rows 1..) and the model names the CELL each landmark sits in,
# which is a reading task; the cell centre is then refined to the dark blob
# nearest it (the eye, the nose leather), which is what makes a cell
# precise enough to align a face.
GRID_COLS = 10
GRID_ROWS = 10
_COLS = "ABCDEFGHIJKLMNOP"[:GRID_COLS]

FACE_LANDMARK_SYSTEM = f"""You locate five points on a photograph of a dog's head, using the grid drawn over it.

The image has a labelled grid: columns {_COLS[0]} to {_COLS[-1]} from left to right, rows 1 to \
{GRID_ROWS} from top to bottom, with the label of each cell in its corner. For every point name \
the cell it falls in (column letter and row number) and where in that cell it sits.

- left_eye: the centre of the eye that appears on the LEFT side of the image.
- right_eye: the centre of the eye that appears on the RIGHT side of the image.
- nose: the centre of the nose leather (the black tip of the muzzle).
- left_ear_base: where the LEFT ear (left side of the image) meets the skull, at the middle of its base.
- right_ear_base: the same for the ear on the right side of the image.

Look at the eyes themselves - the dark, glossy ovals - not the fur markings around them. Mark a \
point that is hidden or cut off as visible=false and still give your best cell."""

FACE_LANDMARK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        name: {
            "type": "object",
            "properties": {
                "col": {"type": "string", "enum": list(_COLS), "description": "Column letter of the cell."},
                "row": {"type": "integer", "minimum": 1, "maximum": GRID_ROWS, "description": "Row number of the cell."},
                "within": {
                    "type": "string",
                    "enum": ["center", "top", "bottom", "left", "right", "top-left", "top-right",
                             "bottom-left", "bottom-right"],
                    "description": "Where inside that cell the point sits.",
                },
                "visible": {"type": "boolean"},
            },
            "required": ["col", "row", "within", "visible"],
            "additionalProperties": False,
        }
        for name in FACE_LANDMARKS
    },
    "required": list(FACE_LANDMARKS),
    "additionalProperties": False,
}

_WITHIN = {"center": (0.5, 0.5), "top": (0.5, 0.2), "bottom": (0.5, 0.8), "left": (0.2, 0.5), "right": (0.8, 0.5),
           "top-left": (0.2, 0.2), "top-right": (0.8, 0.2), "bottom-left": (0.2, 0.8), "bottom-right": (0.8, 0.8)}


def draw_grid(image: Image.Image) -> Image.Image:
    """The image with the lettered grid drawn over it."""
    from PIL import ImageDraw, ImageFont

    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out, "RGBA")
    width, height = out.size
    cw, ch = width / GRID_COLS, height / GRID_ROWS
    try:
        font = ImageFont.truetype("arial.ttf", max(12, int(min(cw, ch) * 0.28)))
    except OSError:
        font = ImageFont.load_default()
    for c in range(GRID_COLS + 1):
        x = round(c * cw)
        draw.line([(x, 0), (x, height)], fill=(255, 255, 0, 160), width=1)
    for r in range(GRID_ROWS + 1):
        y = round(r * ch)
        draw.line([(0, y), (width, y)], fill=(255, 255, 0, 160), width=1)
    for c in range(GRID_COLS):
        for r in range(GRID_ROWS):
            label = f"{_COLS[c]}{r + 1}"
            x, y = c * cw + 2, r * ch + 1
            box = draw.textbbox((x, y), label, font=font)
            draw.rectangle([box[0] - 1, box[1] - 1, box[2] + 1, box[3] + 1], fill=(0, 0, 0, 150))
            draw.text((x, y), label, fill=(255, 255, 0, 255), font=font)
    return out


def cell_to_pixel(col: str, row: int, within: str, size: tuple[int, int]) -> tuple[float, float]:
    width, height = size
    cw, ch = width / GRID_COLS, height / GRID_ROWS
    c = _COLS.index(col.upper()) if col and col.upper() in _COLS else 0
    r = int(np.clip(int(row) - 1, 0, GRID_ROWS - 1))
    fx, fy = _WITHIN.get(within, (0.5, 0.5))
    return (c + fx) * cw, (r + fy) * ch


LANDMARK_QUERY_SIDE = 1024
# Ring around the protected features, in texels, that the alignment check reads.
CHECK_RING_PX = 5
SSIM_FLOOR = 0.8
# The photo's face is blended into the composed coat over this many texels.
FACE_LAYER_FEATHER_PX = 10

# A landmark is refined to the darkest compact blob near it - the eye and the
# nose leather are the darkest things on any face - within this radius (in
# eye distances) of the point given, and may move at most this far.
REFINE_RADIUS_EYES = 0.35
REFINE_DARK_SHARE = 0.12
# ... and only when that blob is this much darker than the rest of the
# window: on a black face nothing stands out and the point stays put.
REFINE_MIN_CONTRAST = 0.12

# Only the eyes and the nose drive the warp. Ear bases are read and logged
# but not used: an ear's position in a photo depends on the animal's mood,
# and a wrong ear point bends the whole face.
WARP_LANDMARKS = ("left_eye", "right_eye", "nose")


def refine_dark_points(image: Image.Image, points: dict[str, dict[str, Any]],
                       names=WARP_LANDMARKS) -> dict[str, dict[str, Any]]:
    """Snap each named point to the centroid of the darkest pixels near it.

    A photo of a small dog is a few hundred pixels across and the face UV
    island is two thousand, so a three-pixel slip in a landmark is a
    twelve-texel slip on the face. The eyes and the nose leather are the
    darkest compact things on a face, so the local darkness centroid is a
    steadier centre than a point anyone placed by hand or by eye."""
    from scipy import ndimage

    if not all(n in points for n in ("left_eye", "right_eye")):
        return points
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    height, width = gray.shape
    left, right = points["left_eye"], points["right_eye"]
    eye_distance = max(float(np.hypot(right["x"] - left["x"], right["y"] - left["y"])), 4.0)
    radius = max(3.0, REFINE_RADIUS_EYES * eye_distance)
    out = dict(points)
    for name in names:
        if name not in points:
            continue
        cx, cy = points[name]["x"], points[name]["y"]
        x0, x1 = int(max(0, cx - radius)), int(min(width, cx + radius + 1))
        y0, y1 = int(max(0, cy - radius)), int(min(height, cy + radius + 1))
        window = gray[y0:y1, x0:x1]
        if window.size < 16:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        inside = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2
        threshold = np.quantile(window[inside], REFINE_DARK_SHARE)
        dark = inside & (window <= threshold)
        # the compact blob nearest the given point, not every dark pixel
        labels, count = ndimage.label(dark)
        if count == 0:
            continue
        best, best_score = None, None
        for index in range(1, count + 1):
            blob = labels == index
            size = int(blob.sum())
            if size < 3:
                continue
            by, bx = ys[blob].mean(), xs[blob].mean()
            score = np.hypot(bx - cx, by - cy) - 0.5 * np.sqrt(size)
            if best_score is None or score < best_score:
                best, best_score = (bx, by), score
        if best is None:
            continue
        chosen = dark & (np.hypot(xs - best[0], ys - best[1]) <= max(2.0, np.sqrt(dark.sum())))
        rest = inside & ~chosen
        if rest.sum() < 8 or window[rest].mean() - window[chosen].mean() < REFINE_MIN_CONTRAST:
            continue
        out[name] = {**points[name], "x": float(best[0]), "y": float(best[1]),
                     "refined_from": [round(cx, 1), round(cy, 1)]}
    return out


async def read_face_landmarks(crop_image: Image.Image, transport: Transport,
                              samples: int = 2) -> dict[str, dict[str, Any]]:
    """The five landmarks on a face crop, in the crop's own pixels.

    Read `samples` times. A reading whose eye-nose triangle could not be a
    dog's face (the nose on the eye line, or below the chin) is dropped; of
    the plausible readings the eyes and nose are averaged, which halves a
    single reading's slip. All implausible: the first is returned and the
    warp's own gate decides."""
    import asyncio

    readings = await asyncio.gather(*(_read_face_landmarks_once(crop_image, transport) for _ in range(samples)),
                                    return_exceptions=True)
    readings = [r for r in readings if isinstance(r, dict) and r]
    if not readings:
        return {}
    plausible = [r for r in readings if (triangle_ratio(r) or 0.0) >= TRIANGLE_RATIO[0]
                 and (triangle_ratio(r) or 9.0) <= TRIANGLE_RATIO[1]]
    if not plausible:
        return readings[0]
    out: dict[str, dict[str, Any]] = {}
    for name in FACE_LANDMARKS:
        points = [r[name] for r in plausible if name in r]
        if not points:
            continue
        out[name] = {"x": float(np.mean([p["x"] for p in points])), "y": float(np.mean([p["y"] for p in points])),
                     "visible": any(p.get("visible", True) for p in points),
                     "samples": len(points)}
    return out


async def _read_face_landmarks_once(crop_image: Image.Image, transport: Transport) -> dict[str, dict[str, Any]]:
    width, height = crop_image.size
    scale = LANDMARK_QUERY_SIDE / max(width, height)
    shown = crop_image if scale >= 1.0 else crop_image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))), Image.LANCZOS)
    payload = await transport.complete(
        system=FACE_LANDMARK_SYSTEM,
        blocks=[_image_block(draw_grid(shown)), {"type": "text", "text": (
            f"The grid is {GRID_COLS} columns ({_COLS[0]}-{_COLS[-1]}) by {GRID_ROWS} rows (1-{GRID_ROWS}). "
            "Give the cell of each of the five points.")}],
        schema=FACE_LANDMARK_SCHEMA,
    )
    out: dict[str, dict[str, Any]] = {}
    for name in FACE_LANDMARKS:
        point = payload.get(name) or {}
        try:
            x, y = cell_to_pixel(str(point.get("col", "A")), int(point.get("row", 1)),
                                 str(point.get("within", "center")), (width, height))
        except (TypeError, ValueError):
            continue
        out[name] = {"x": float(np.clip(x, 0, width - 1)), "y": float(np.clip(y, 0, height - 1)),
                     "visible": bool(point.get("visible", True)),
                     "cell": f"{point.get('col')}{point.get('row')}/{point.get('within')}"}
    return out


def tps_fit(src: np.ndarray, dst: np.ndarray, regularisation: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Thin-plate spline taking src (n, 2) onto dst (n, 2). Returns (weights
    (n, 2), affine (3, 2)). Three points give a plain affine map."""
    n = src.shape[0]
    d = np.linalg.norm(src[:, None, :] - src[None, :, :], axis=-1)
    kernel = np.where(d > 0, d * d * np.log(np.maximum(d, 1e-12)), 0.0)
    kernel += regularisation * np.eye(n)
    p = np.concatenate([np.ones((n, 1)), src], axis=1)
    system = np.zeros((n + 3, n + 3))
    system[:n, :n] = kernel
    system[:n, n:] = p
    system[n:, :n] = p.T
    rhs = np.zeros((n + 3, 2))
    rhs[:n] = dst
    solution = np.linalg.lstsq(system, rhs, rcond=None)[0]
    return solution[:n], solution[n:]


def tps_apply(points: np.ndarray, src: np.ndarray, weights: np.ndarray, affine: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(points[:, None, :] - src[None, :, :], axis=-1)
    kernel = np.where(d > 0, d * d * np.log(np.maximum(d, 1e-12)), 0.0)
    return affine[0] + points @ affine[1:] + kernel @ weights


@dataclass
class FaceLayer:
    rgb: np.ndarray        # (h, w, 3) linear
    weight: np.ndarray     # (h, w) 0..1
    report: dict[str, Any]


def build_face_layer(
    photo_path: str | Path,
    crop_box: tuple[int, int, int, int],
    crop_landmarks: dict[str, dict[str, Any]],
    template_dir: str | Path,
    masks: MaskLibrary,
    palette: dict[str, tuple[str, str, str]] | None = None,
    hair: np.ndarray | None = None,
    trust_all: bool = False,
) -> FaceLayer | None:
    """Warp the photograph's face onto the face UV island.

    `crop_landmarks` are in crop pixels; `crop_box` puts them back in the
    photo. Returns None when fewer than the two eyes and the nose are
    usable. `trust_all` uses a point the model flagged as not visible: on a
    head the face-box call already called frontal, "not visible" has meant
    "partly in shadow", and the dark-blob refinement settles it anyway.
    """
    from scipy import ndimage
    from scipy.ndimage import map_coordinates

    from .fur import hex_to_linear, to_linear
    from .landmarks import Landmarks
    from .projection import lab_to_linear, linear_to_lab
    from .regions import REGIONS

    template = Landmarks.load(template_dir)
    photo_image = Image.open(photo_path).convert("RGB")
    in_photo = {n: {**p, "x": p["x"] + crop_box[0], "y": p["y"] + crop_box[1]}
                for n, p in crop_landmarks.items()}
    in_photo = refine_dark_points(photo_image, in_photo)
    names = [n for n in WARP_LANDMARKS if n in in_photo and n in template.points
             and (trust_all or in_photo[n].get("visible", True))]
    if len(names) < 3:
        return None

    src = np.array([[in_photo[n]["x"], in_photo[n]["y"]] for n in names])
    dst = np.array([template.uv(n) for n in names], dtype=np.float64)

    # inverse map: for every texel of the face island, where in the photo
    weights, affine = tps_fit(dst, src)

    face_union = masks.union(FACE_PHOTO_REGIONS | {"ears_inner"}) > 0.5   # skull carries the outer ears
    islands, _ = ndimage.label(masks.union(REGIONS) > 0.5)
    nose_uv = template.uv("nose")
    face_island = islands == islands[nose_uv[1], nose_uv[0]]
    target = face_union & face_island
    if target.sum() < 500:
        return None

    photo = photo_rgba(photo_path)
    rgba = np.asarray(photo, dtype=np.float32) / 255.0
    height, width = rgba.shape[:2]

    rows, cols = np.nonzero(target)
    mapped = tps_apply(np.stack([cols, rows], axis=1).astype(np.float64), dst, weights, affine)
    px, py = mapped[:, 0], mapped[:, 1]
    inside = (px >= 0) & (px <= width - 1) & (py >= 0) & (py <= height - 1)
    layer = np.zeros(target.shape + (3,), dtype=np.float32)
    alpha = np.zeros(target.shape, dtype=np.float32)
    for channel in range(3):
        layer[rows, cols, channel] = map_coordinates(rgba[..., channel], [py, px], order=1, mode="nearest")
    alpha[rows, cols] = map_coordinates(rgba[..., 3], [py, px], order=1, mode="constant") * inside
    layer = to_linear(layer)

    # the photo's lightness and hue pulled toward the coat's palette, as in
    # the experimental path, so the face and the body meet in one texture
    support = alpha > 0.5
    if palette:
        mids = {name: linear_to_lab(hex_to_linear(stops[1])[None, None, :])[0, 0] for name, stops in palette.items()}
        mids_array = np.stack(list(mids.values())).astype(np.float32)
        lightness_range = (float(mids_array[:, 0].min()), float(mids_array[:, 0].max()))
        layer = _stretch_lightness(layer, support, lightness_range, linear_to_lab, lab_to_linear)
        layer = _soft_snap(layer, support, mids_array, hair, linear_to_lab, lab_to_linear, None)

    weight = ndimage.gaussian_filter(support.astype(np.float32), FACE_LAYER_FEATHER_PX / 2.0)
    weight = np.minimum(weight, ndimage.gaussian_filter(target.astype(np.float32), 1.0)) * (alpha > 0.05)
    core = ndimage.binary_erosion(support, iterations=FACE_LAYER_FEATHER_PX)
    weight = np.maximum(weight, core.astype(np.float32))

    report = {
        "landmarks_used": names,
        "triangle_ratio": triangle_ratio(in_photo),
        "refined": {n: in_photo[n].get("refined_from") for n in names if in_photo[n].get("refined_from")},
        "photo_points": {n: [round(float(v), 1) for v in p] for n, p in zip(names, src)},
        "uv_points": {n: [int(v) for v in p] for n, p in zip(names, dst)},
        "island_texels": int(target.sum()),
        "covered_share": round(float(support.sum() / max(target.sum(), 1)), 3),
    }
    return FaceLayer(layer, weight.astype(np.float32), report)


# THE GATE, AND WHAT WAS MEASURED
# --------------------------------
# The specification gates the face layer on the structural similarity
# between the composite with the face and the one without it, in the 5 px
# ring around the protected eyes and nose, with a floor of 0.8. Measured on
# the black shiba fixture with the face visibly aligned (eyes on the
# sockets, nose on the leather), that score is about 0.48 (blur sigma 6,
# protected cores restored in both images first so only the ring's own
# content counts); shifting the landmarks by 8-12 photo pixels moves it to
# 0.44-0.49. The mask build's ring is flat fur and the photo's ring is a real
# eye's surroundings, so the two never look alike whether the warp landed
# or not, and at 0.8 the literal rule rejects every face. The score is still
# computed and logged (`ring_ssim`) and the floor still applies when
# RELAXED_GATE is False. With it True (the default) the layer is accepted
# when the two checks that DID separate the aligned case from the shifted
# ones both pass:
#   - dark intrusion: the share of ring texels as dark as the photo's own
#     eyes and nose (0.010 aligned against 0.031-0.034 shifted);
#   - triangle sanity: the nose's distance from the eye line, in eye
#     distances, is within the range a frontal head can produce.
RELAXED_GATE = True
INTRUSION_MAX = 0.02
TRIANGLE_RATIO = (0.30, 0.95)
RING_BLUR_SIGMA = 6.0


def ring_ssim(with_face: np.ndarray, without: np.ndarray, protect: np.ndarray,
              where: np.ndarray, template: np.ndarray | None = None,
              ring_px: int = CHECK_RING_PX) -> float | None:
    """Structural similarity between the two composites in the ring just
    outside the protected features, on blurred luminance. When `template`
    (the protected pixels' source) is given, it is restored into both
    images first, so the eyes and nose themselves do not enter the score."""
    from scipy import ndimage
    from skimage.metrics import structural_similarity

    core = protect > 0.5
    ring = ndimage.binary_dilation(core, iterations=ring_px) & ~core & where
    if ring.sum() < 200:
        return None
    a, b = np.clip(with_face, 0, 1), np.clip(without, 0, 1)
    if template is not None:
        w = np.maximum(ndimage.gaussian_filter(protect, 1.5), protect)[..., None]
        a = a * (1 - w) + template * w
        b = b * (1 - w) + template * w
    luma = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    a = ndimage.gaussian_filter(a @ luma, RING_BLUR_SIGMA)
    b = ndimage.gaussian_filter(b @ luma, RING_BLUR_SIGMA)
    rows, cols = np.nonzero(ring)
    y0, y1, x0, x1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    _, full = structural_similarity(a[y0:y1, x0:x1], b[y0:y1, x0:x1], data_range=1.0, full=True)
    return float(full[ring[y0:y1, x0:x1]].mean())


def dark_intrusion(with_face: np.ndarray, protect: np.ndarray, weight: np.ndarray,
                   ring_px: int = CHECK_RING_PX) -> float | None:
    """Share of the ring's texels as dark as the photo face's darkest few
    percent - its eyes and nose leather. Aligned, those sit inside the
    protected core and the ring holds fur."""
    from scipy import ndimage

    core = protect > 0.5
    ring = ndimage.binary_dilation(core, iterations=ring_px) & ~core & (weight > 0.05)
    face = weight > 0.5
    if ring.sum() < 200 or face.sum() < 500:
        return None
    luma = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    lum = np.clip(with_face, 0, 1) @ luma
    threshold = float(np.quantile(lum[face], 0.08))
    return float((lum[ring] < threshold).mean())


def triangle_ratio(points: dict[str, dict[str, Any]]) -> float | None:
    """The nose's distance from the line between the eyes, in eye distances."""
    try:
        left = np.array([points["left_eye"]["x"], points["left_eye"]["y"]])
        right = np.array([points["right_eye"]["x"], points["right_eye"]["y"]])
        nose = np.array([points["nose"]["x"], points["nose"]["y"]])
    except KeyError:
        return None
    eye = right - left
    length = float(np.linalg.norm(eye))
    if length < 1e-3:
        return None
    normal = np.array([-eye[1], eye[0]]) / length
    return float(abs(np.dot(nose - (left + right) / 2, normal)) / length)


def face_gate(ring_score: float | None, intrusion: float | None, ratio: float | None) -> tuple[bool, str]:
    """Whether the face layer is used, and why."""
    if ring_score is not None and ring_score >= SSIM_FLOOR:
        return True, f"ring SSIM {ring_score:.3f} >= {SSIM_FLOOR}"
    if not RELAXED_GATE:
        return False, f"ring SSIM {ring_score} < {SSIM_FLOOR}"
    problems = []
    if intrusion is None or intrusion > INTRUSION_MAX:
        problems.append(f"dark intrusion {intrusion} > {INTRUSION_MAX}")
    if ratio is None or not (TRIANGLE_RATIO[0] <= ratio <= TRIANGLE_RATIO[1]):
        problems.append(f"eye-nose ratio {ratio} outside {TRIANGLE_RATIO}")
    if problems:
        return False, "; ".join(problems) + f" (ring SSIM {ring_score})"
    return True, (f"relaxed gate: dark intrusion {intrusion:.3f} <= {INTRUSION_MAX}, "
                  f"eye-nose ratio {ratio:.2f} (ring SSIM {ring_score} below {SSIM_FLOOR})")


__all__ = [
    "FACE_LANDMARKS", "FACE_PHOTO_REGIONS", "LANDMARKS", "FaceLayer", "FacePhoto",
    "FaceProjection", "SSIM_FLOOR", "affine_fit", "build_face_layer", "draw_landmarks",
    "front_landmarks", "head_turn", "mesh_landmarks", "project_face", "read_face_landmarks",
    "read_landmarks", "refine_dark_points", "ring_ssim", "dark_intrusion", "face_gate", "draw_grid",
    "cell_to_pixel",
    "triangle_ratio", "tps_apply", "tps_fit", "warp_photo",
]
