"""Body-shape morph targets on a template GLB: ears, tail, build, muzzle.

WHY
---
Every dog was the template's jindo: pricked ears, a thick tail carried
level, a stocky trunk. A black-and-tan with folded ears and a thin tail came
out right in colour and still read as "a jindo painted like him" - people
recognise a dog by its outline first. Texture cannot fix an outline.

HOW
---
glTF morph targets (Blender shape keys), computed here in numpy from the
template's own vertices - no Blender run, per template or per dog. A dog's
shape is then a handful of default weights written into its GLB's mesh
(api/shape.py), which every glTF viewer applies before skinning, so the
idle and tail-wag clips keep working on the reshaped mesh.

Morph targets ADD. A rotation (ear folding over) and a scale (tail
thinning) do not compose by addition, so mutually exclusive shapes are
separate targets and a dog uses at most one of each group:

  ears   ear_fold        the top half of each ear folds forward and down
         ear_drop        the whole ear hangs down the side of the head
  tail   tail_thin       a slim tail carried as the template carries it
         tail_down       the bushy tail hanging down and back
         tail_down_thin  a slim tail hanging down and back (a pointer's)
  build  body_slim       the trunk narrowed (negative weight = stockier)
  muzzle muzzle_long     the muzzle lengthened (negative = shorter)

The ear and tail regions are chosen by GEOMETRY, not skin weights: the
auto-rig gave the ear bones 70 vertices out of the ~1500 in the ears
(skull-weighted), and the tail bones only the top of the tail's fur.
Geometry is measured on the jindo (mesh space: y up, z forward, x to the
dog's left); the constants below are in metres in that space.

The ear constants below are the jindo's; another template's ears are
measured on its own mesh (measure_ears, from its ear-base joint), which
gives back the jindo's constants on the jindo. --only ears writes the ear
targets alone: the tail, build and muzzle constants are the jindo's body and
another template's would need measuring too (the jindo puppy 2's ears stand,
and ear_drop / ear_fold give one model both kinds of puppy).

Usage:
    python -m ar_pipeline.shape_keys output/skeleton_jindo_small.glb output/skeleton_jindo_small_shaped.glb
    python -m ar_pipeline.shape_keys SRC.glb DST.glb --only ears
    python -m ar_pipeline.shape_keys SRC.glb DST.glb --only ear_fold --template output/templates/jindo_puppy_small
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pygltflib import GLTF2, Accessor, BufferView

# --- ears: the crown between them is at y 0.782 (the left and right ear
# cross-sections join below it); the ears stand at z 0.42..0.50, |x| > 0.03
EAR_ROOT_Y = 0.776
EAR_RAMP = 0.012
EAR_Z = (0.40, 0.51)
EAR_MIN_X = 0.025
# fold: the part above EAR_FOLD_Y turns about the lateral axis, forward-down.
# 125 deg folded the tip back onto the ear (morphs blend linearly, so a big
# turn collapses the flap into a lump); 80 reads as a tipped ear
EAR_FOLD_Y = 0.803
EAR_FOLD_DEG = 80.0
# drop: the whole ear turns outward about the forward axis at its root, and
# a little forward, so it hangs beside the cheek
EAR_DROP_DEG = 165.0      # 115 left the ears sticking out sideways; 165 lies along the cheek
EAR_DROP_FORWARD_DEG = 15.0
# the hinge sits on the skull's outer edge at the root, not mid-ear: turned
# about the ear's own middle the flap swung INTO the head (x 0.05..0.10 is
# inside the skull there). A hanging ear is longer than a pricked one.
EAR_DROP_HINGE_X = 0.095
EAR_DROP_LENGTHEN = 1.3
# ...and its root is gathered toward that hinge first (lateral offset x this):
# turned whole, the ear's inner root (x 0.03) crossed the crown and stretched
# a flat shelf over the top of the head
EAR_DROP_GATHER = 0.35
# ears picked by the template's masks take the fur fringe within this of them
EAR_FRINGE_M = 0.012

# --- tail: behind the rump at z -0.29 and above the hocks (y 0.44)
TAIL_ROOT_Z = -0.285
TAIL_RAMP = 0.035
TAIL_MIN_Y = 0.44
TAIL_THIN = 0.42          # radial scale toward the tail's centre line
TAIL_DOWN_DEG = 62.0      # the tip ends this far below level
TAIL_DOWN_ROOT_SHARE = 0.55   # the root turns this share of it, the tip all

# --- build: the trunk between the shoulders and the rump
BODY_Z = (-0.33, 0.33)
BODY_RAMP = 0.08
BODY_SLIM = 0.82          # x scale at weight 1

# --- muzzle: in front of the eyes
MUZZLE_FROM_Z = 0.53
MUZZLE_MAX_Y = 0.72
MUZZLE_LONG = 0.40        # extra length per metre in front of the eyes


def _smooth(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _rot_x(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def _rot_z(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def _rotate(pos: np.ndarray, weight: np.ndarray, pivot: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Each vertex moved `weight` of the way to its rotated position."""
    turned = (pos - pivot) @ rotation.T + pivot
    return pos + weight[:, None] * (turned - pos)


@dataclass
class Ears:
    root_y: float = EAR_ROOT_Y
    fold_y: float = EAR_FOLD_Y
    z: tuple[float, float] = EAR_Z
    min_x: float = EAR_MIN_X
    hinge_x: float = EAR_DROP_HINGE_X


# measured: the root sits this far under the crown (the skull's top on the
# midline at the ears), the fold this share of the way from the crown to the
# ears' top, the forward band this much past the ears either way, the hinge
# this much outside the ears' outer edge - the jindo's constants above
EAR_ROOT_BELOW_CROWN = 0.006
EAR_FOLD_SHARE = 0.44
EAR_Z_PAD = 0.03
EAR_HINGE_OUT = 0.003


def measure_ears(pos: np.ndarray, base: np.ndarray) -> Ears:
    """The ears of this mesh (y up, z forward), found round the left ear-base
    joint `base`: what rises above the crown beside it."""
    beside = (pos[:, 0] > 0.02) & (np.abs(pos[:, 2] - base[2]) < 0.07)
    midline = (np.abs(pos[:, 0]) < 0.012) & (np.abs(pos[:, 2] - base[2]) < 0.05)
    crown = float(pos[midline, 1].max())
    ear = beside & (pos[:, 1] > crown)
    top = float(pos[ear, 1].max())
    return Ears(root_y=crown - EAR_ROOT_BELOW_CROWN, fold_y=crown + EAR_FOLD_SHARE * (top - crown),
                z=(float(pos[ear, 2].min()) - EAR_Z_PAD, float(pos[ear, 2].max()) + EAR_Z_PAD),
                min_x=float(pos[ear, 0].min()) - 0.001, hinge_x=float(pos[ear, 0].max()) + EAR_HINGE_OUT)


def ear_weight(pos: np.ndarray, side: float, from_y: float, ears: Ears = Ears()) -> np.ndarray:
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    inside = (side * x > ears.min_x) & (z > ears.z[0]) & (z < ears.z[1])
    return np.where(inside, _smooth((y - from_y) / EAR_RAMP), 0.0)


def tail_weight(pos: np.ndarray) -> np.ndarray:
    return np.where(pos[:, 1] > TAIL_MIN_Y, _smooth((TAIL_ROOT_Z - pos[:, 2]) / TAIL_RAMP), 0.0)


def _tail_centre(pos: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Per vertex, the centre of the tail's cross-section at its z (slices
    of 1 cm, smoothed along the tail)."""
    inside = weight > 0.01
    z = pos[inside, 2]
    edges = np.arange(z.min() - 0.01, z.max() + 0.02, 0.01)
    which = np.clip(np.digitize(pos[:, 2], edges) - 1, 0, len(edges) - 2)
    centre = np.zeros((len(edges) - 1, 3))
    counts = np.zeros(len(edges) - 1)
    np.add.at(centre, which[inside], pos[inside])
    np.add.at(counts, which[inside], 1)
    filled = counts > 0
    centre[filled] /= counts[filled, None]
    for axis in range(3):   # empty slices take their neighbours' centre
        centre[~filled, axis] = np.interp(np.flatnonzero(~filled), np.flatnonzero(filled), centre[filled, axis])
    kernel = np.array([1, 2, 3, 2, 1], dtype=float) / 9.0
    for axis in range(3):
        centre[:, axis] = np.convolve(np.pad(centre[:, axis], 2, mode="edge"), kernel, mode="valid")
    return centre[which]


def ear_targets(pos: np.ndarray, ears: Ears = Ears(), mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """ear_fold and ear_drop. `mask` is each vertex's ear membership read
    from the template's ear masks (ear_mask); without it the ears are the
    geometric band above the root (the jindo)."""
    fold = pos.copy()
    drop = pos.copy()
    for side in (1.0, -1.0):
        if mask is not None:
            # the ear moves whole, faded in from its root by height only: the
            # masks' soft edge moved neighbouring vertices of one flap by
            # different amounts and shredded it
            inside = (side * pos[:, 0] > 0) & (mask > 0.5)
            # ... with the fur fringe round its edge, which the masks leave
            # to the skull: left behind it stood up in spikes where the ear
            # had been
            from scipy.spatial import cKDTree
            near, _ = cKDTree(pos[inside]).query(pos, distance_upper_bound=EAR_FRINGE_M)
            root_y = float(np.percentile(pos[inside, 1], 3))
            inside = (side * pos[:, 0] > 0) & np.isfinite(near) & (pos[:, 1] > root_y - EAR_RAMP)
            ear = pos[inside]
            fold_y = root_y + EAR_FOLD_SHARE * (float(ear[:, 1].max()) - root_y)
            hinge_x = float((side * ear[:, 0]).max()) + EAR_HINGE_OUT
            whole = np.where(inside, _smooth((pos[:, 1] - root_y + EAR_RAMP) / EAR_RAMP), 0.0)
            upper = np.where(inside, _smooth((pos[:, 1] - fold_y) / EAR_RAMP), 0.0)
        else:
            ear = pos[ear_weight(pos, side, ears.root_y, ears) > 0]
            root_y, fold_y, hinge_x = ears.root_y, ears.fold_y, ears.hinge_x
            whole = ear_weight(pos, side, root_y, ears)
            upper = ear_weight(pos, side, fold_y, ears)
        root = np.array([side * np.percentile(side * ear[:, 0], 50), root_y, np.median(ear[:, 2])])
        hinge = np.array([root[0], fold_y, root[2]])
        # +x rotation turns +y toward +z: up toward forward, then down
        fold = _rotate(fold, upper, hinge, _rot_x(EAR_FOLD_DEG))
        hinge = np.array([side * hinge_x, root_y, root[2]])
        longer = pos.copy()
        longer[:, 1] = root_y + (pos[:, 1] - root_y) * EAR_DROP_LENGTHEN
        longer[:, 0] = hinge[0] + (pos[:, 0] - hinge[0]) * EAR_DROP_GATHER
        # about +z, a positive angle turns +y toward -x: outward on the +x side
        # is a negative angle there
        outward = _rot_z(-side * EAR_DROP_DEG)
        turned = (longer - hinge) @ (_rot_x(EAR_DROP_FORWARD_DEG) @ outward).T + hinge
        drop = drop + whole[:, None] * (turned - pos)
    return {"ear_fold": fold, "ear_drop": drop}


def targets(pos: np.ndarray, ears: Ears = Ears()) -> dict[str, np.ndarray]:
    """name -> deformed positions (same order as pos)."""
    out: dict[str, np.ndarray] = ear_targets(pos, ears)

    # tail
    tail = tail_weight(pos)
    centre = _tail_centre(pos, tail)
    thin = pos + tail[:, None] * (centre + (pos - centre) * TAIL_THIN - pos)
    out["tail_thin"] = thin
    root = np.array([0.0, np.median(pos[(tail > 0.5) & (pos[:, 2] > TAIL_ROOT_Z - 0.05), 1]), TAIL_ROOT_Z])
    length = max(TAIL_ROOT_Z - float(pos[tail > 0.5, 2].min()), 1e-3)

    def hang(source: np.ndarray) -> np.ndarray:
        along = np.clip((TAIL_ROOT_Z - pos[:, 2]) / length, 0.0, 1.0)
        # the tail points to -z; a rotation about +x by -a turns -z toward -y,
        # more at the tip than at the root so the tail hangs in a curve
        angle = np.radians(-TAIL_DOWN_DEG * (TAIL_DOWN_ROOT_SHARE + (1.0 - TAIL_DOWN_ROOT_SHARE) * along))
        rel = source - root
        cos, sin = np.cos(angle), np.sin(angle)
        turned = np.stack([rel[:, 0], cos * rel[:, 1] - sin * rel[:, 2], sin * rel[:, 1] + cos * rel[:, 2]], 1) + root
        return source + tail[:, None] * (turned - source)

    out["tail_down"] = hang(pos)
    out["tail_down_thin"] = hang(thin)

    # build
    z = pos[:, 2]
    trunk = _smooth((z - BODY_Z[0] + BODY_RAMP) / BODY_RAMP) * _smooth((BODY_Z[1] + BODY_RAMP - z) / BODY_RAMP)
    slim = pos.copy()
    slim[:, 0] = pos[:, 0] * (1.0 - trunk * (1.0 - BODY_SLIM))
    out["body_slim"] = slim

    # muzzle
    front = np.clip(pos[:, 2] - MUZZLE_FROM_Z, 0.0, None) * _smooth((MUZZLE_MAX_Y - pos[:, 1]) / 0.03)
    front *= _smooth((pos[:, 1] - 0.5) / 0.05)          # the head, not the chest below it
    longer = pos.copy()
    longer[:, 2] = pos[:, 2] + front * MUZZLE_LONG
    out["muzzle_long"] = longer
    return out


# --- outline, second set (2026-09-24): the proportions one template cannot
# hold. Mesh space: y up from the ground, z forward, height 0.83.
LEG_TOP_Y = 0.34          # the legs end and the chest begins
LEG_FOOT_Y = 0.04         # the paws keep their shape
# Weight 1 is the most that still reads as a dog of this build: at 0.10 m /
# 0.18 m / 55% the demo's long legs, long trunk and big ears looked stretched
# (user, 2026-09-24); these are 60% of those.
LEG_LONG = 0.06           # metres added to the leg at weight 1
TRUNK_Z = (-0.20, 0.22)   # between the hips and the shoulders: the part that stretches
BODY_LONG = 0.11          # metres added to the trunk at weight 1
HEAD_PIVOT = np.array([0.0, 0.60, 0.40])   # the top of the neck
HEAD_FROM_Z, HEAD_FROM_Y = 0.36, 0.48
# A puppy is not a big head alone - scaled by itself the head read as an
# adult with a big head. The puppy target is the whole proportion at once:
# a larger head, a shorter muzzle, shorter legs and a rounder trunk.
PUPPY_HEAD = 0.18
PUPPY_MUZZLE = 0.35       # share of the muzzle in front of the eyes taken off
PUPPY_LEG = 0.035         # metres off the legs
PUPPY_ROUND = 0.08        # the trunk this much wider
EAR_BIG = 0.33            # the ears a third larger about their roots


def more_targets(pos: np.ndarray, tri: np.ndarray) -> dict[str, np.ndarray]:
    """The second set: leg_long, body_long, ear_big, puppy. Every one is a
    function of the position alone, so the copies of a vertex split along a
    UV seam move together. (A uniform "fluff" push along the normals was
    tried and dropped: it read as a swollen dog, not a long coat.)"""
    out: dict[str, np.ndarray] = {}
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

    leg = pos.copy()
    leg[:, 1] = y + LEG_LONG * _smooth((y - LEG_FOOT_Y) / (LEG_TOP_Y - LEG_FOOT_Y))
    out["leg_long"] = leg

    body = pos.copy()
    body[:, 2] = z + BODY_LONG * (_smooth((z - TRUNK_Z[0]) / (TRUNK_Z[1] - TRUNK_Z[0])) - 0.5)
    out["body_long"] = body

    ears = pos.copy()
    for side in (1.0, -1.0):
        w = ear_weight(pos, side, EAR_ROOT_Y)
        inside = w > 0
        if not inside.any():
            continue
        root = np.array([side * np.percentile(side * x[inside], 20), EAR_ROOT_Y, np.median(z[inside])])
        ears = ears + (w * EAR_BIG)[:, None] * (pos - root)
    out["ear_big"] = ears

    # puppy: the legs first (the body drops with them), then the head about
    # the top of the neck, the muzzle pulled back toward the eyes, the trunk
    # a little wider. Weights are taken on the rest positions.
    puppy = pos.copy()
    puppy[:, 1] = y - PUPPY_LEG * _smooth((y - LEG_FOOT_Y) / (LEG_TOP_Y - LEG_FOOT_Y))
    head = _smooth((z - HEAD_FROM_Z) / 0.08) * _smooth((y - HEAD_FROM_Y) / 0.06)
    pivot = HEAD_PIVOT - np.array([0.0, PUPPY_LEG, 0.0])
    puppy = puppy + (head * PUPPY_HEAD)[:, None] * (puppy - pivot)
    front = np.clip(z - MUZZLE_FROM_Z, 0.0, None) * _smooth((MUZZLE_MAX_Y - y) / 0.03) * _smooth((y - 0.5) / 0.05)
    puppy[:, 2] -= PUPPY_MUZZLE * front * (1.0 + PUPPY_HEAD)
    trunk = _smooth((z - BODY_Z[0] + BODY_RAMP) / BODY_RAMP) * _smooth((BODY_Z[1] + BODY_RAMP - z) / BODY_RAMP)
    trunk *= _smooth((y - LEG_TOP_Y + 0.04) / 0.06)
    puppy[:, 0] = puppy[:, 0] * (1.0 + PUPPY_ROUND * trunk)
    out["puppy"] = puppy
    return out


def vertex_normals(pos: np.ndarray, tri: np.ndarray) -> np.ndarray:
    face = np.cross(pos[tri[:, 1]] - pos[tri[:, 0]], pos[tri[:, 2]] - pos[tri[:, 0]])
    normals = np.zeros_like(pos)
    for k in range(3):
        np.add.at(normals, tri[:, k], face)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def _read(gltf: GLTF2, blob: bytes, index: int) -> np.ndarray:
    accessor = gltf.accessors[index]
    view = gltf.bufferViews[accessor.bufferView]
    dtype = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}[accessor.componentType]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[accessor.type]
    offset = (view.byteOffset or 0) + (accessor.byteOffset or 0)
    return np.frombuffer(blob, dtype=dtype, count=accessor.count * width, offset=offset).reshape(accessor.count, width)


def ear_mask(template: str | Path, uv: np.ndarray) -> np.ndarray | None:
    """Each vertex's ear membership (0..1): the template's ear source masks
    (ears_outer, ears_inner) at its UV. The puppy 2's ears sit on the
    corners of its skull and reach 2 cm below the crown there, so no height
    band picks them without taking the skull's corners too; the masks are
    the auto-rig's own ear part, cut along its seams."""
    from PIL import Image
    paths = [Path(template) / "masks_source" / f"{name}.png" for name in ("ears_outer", "ears_inner")]
    if not all(p.exists() for p in paths):
        return None
    masks = [np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0 for p in paths]
    h, w = masks[0].shape
    col = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
    row = np.clip((uv[:, 1] * h).astype(int), 0, h - 1)
    return np.maximum(masks[0][row, col], masks[1][row, col])


def template_ears(src: str | Path, pos: np.ndarray) -> Ears:
    """The template's ears measured round its left ear-base joint (Blender
    space x, -z, y -> mesh space x, y, z); the jindo's constants without one."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from api.mesh import TemplateMesh
        joint = TemplateMesh.load(src, (2048, 2048)).joints.get("L_ear_base_jnt")
    except Exception:  # noqa: BLE001 - a GLB the loader cannot read keeps the constants
        joint = None
    if joint is None:
        return Ears()
    ears = measure_ears(pos, np.array([joint[0], joint[2], -joint[1]]))
    print(f"[shape] ears: root y {ears.root_y:.3f}, fold y {ears.fold_y:.3f}, z {ears.z[0]:.3f}..{ears.z[1]:.3f}, "
          f"x > {ears.min_x:.3f}, hinge x {ears.hinge_x:.3f}")
    return ears


def add_targets(src: str | Path, dst: str | Path, append_only: bool = False, only: str | None = None,
                template: str | Path | None = None) -> list[str]:
    gltf = GLTF2().load(str(src))
    blob = bytearray(gltf.binary_blob())
    if len(gltf.meshes) != 1 or len(gltf.meshes[0].primitives) != 1:
        raise ValueError("expected one mesh with one primitive (the template body)")
    primitive = gltf.meshes[0].primitives[0]
    existing = list(((gltf.meshes[0].extras or {}).get("targetNames")) or [])
    if primitive.targets and not append_only:
        raise ValueError(f"{src} already has morph targets (use --append for the second set)")
    pos = _read(gltf, bytes(blob), primitive.attributes.POSITION).astype(np.float64)
    normal = _read(gltf, bytes(blob), primitive.attributes.NORMAL).astype(np.float64)
    tri = _read(gltf, bytes(blob), primitive.indices).reshape(-1, 3).astype(np.int64)
    rest_normals = vertex_normals(pos, tri)

    def append(array: np.ndarray, with_bounds: bool) -> int:
        while len(blob) % 4:
            blob.append(0)
        data = array.astype(np.float32).tobytes()
        gltf.bufferViews.append(BufferView(buffer=0, byteOffset=len(blob), byteLength=len(data)))
        blob.extend(data)
        accessor = Accessor(bufferView=len(gltf.bufferViews) - 1, componentType=5126, count=len(array), type="VEC3")
        if with_bounds:
            accessor.min = array.min(axis=0).astype(float).tolist()
            accessor.max = array.max(axis=0).astype(float).tolist()
        gltf.accessors.append(accessor)
        return len(gltf.accessors) - 1

    names: list[str] = list(existing) if append_only else []
    if not append_only:
        primitive.targets = []
    ears = template_ears(src, pos)
    mask = None
    if template is not None:
        mask = ear_mask(template, _read(gltf, bytes(blob), primitive.attributes.TEXCOORD_0).astype(np.float64))
        print(f"[shape] ears from the template's masks: {int((mask > 0.5).sum()) if mask is not None else 0} vertices")
    if only in ("ears", "ear_fold"):
        made = ear_targets(pos, ears, mask)
        if only == "ear_fold":
            made = {"ear_fold": made["ear_fold"]}
    else:
        made = more_targets(pos, tri) if append_only else {**targets(pos, ears), **more_targets(pos, tri)}
    for name, moved in made.items():
        if name in names:
            raise ValueError(f"{src} already has a target {name}")
        delta = moved - pos
        # the normal's change as the change of the recomputed normals, laid on
        # the template's own (which keep its smoothing and seams)
        new_normal = normal + (vertex_normals(moved, tri) - rest_normals)
        new_normal /= np.maximum(np.linalg.norm(new_normal, axis=1, keepdims=True), 1e-12)
        primitive.targets.append({"POSITION": append(delta, True), "NORMAL": append(new_normal - normal, False)})
        names.append(name)
        print(f"[shape] {name}: {int((np.linalg.norm(delta, axis=1) > 1e-5).sum())} vertices move, "
              f"max {np.linalg.norm(delta, axis=1).max() * 100:.1f} cm")
    mesh = gltf.meshes[0]
    mesh.weights = list(mesh.weights or [])[:len(existing)] + [0.0] * (len(names) - len(existing)) if append_only         else [0.0] * len(names)
    mesh.extras = {**(mesh.extras or {}), "targetNames": names}
    gltf.buffers[0].byteLength = len(blob)
    gltf.set_binary_blob(bytes(blob))
    gltf.save_binary(str(dst))
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--append", action="store_true",
                        help="add only the second set to a GLB that already has the first")
    parser.add_argument("--only", choices=("ears", "ear_fold"), default=None,
                        help="write these targets alone (ear_fold: a template whose ear_drop does not read; "
                             "api/shape.STAND_INS shows a dropped ear as the fold)")
    parser.add_argument("--template", default=None,
                        help="the template folder: its ear masks pick the ear vertices (ear_mask)")
    args = parser.parse_args()
    names = add_targets(args.src, args.dst, append_only=args.append, only=args.only, template=args.template)
    print(f"[shape] wrote {args.dst} with {len(names)} targets: {names}")


if __name__ == "__main__":
    main()
