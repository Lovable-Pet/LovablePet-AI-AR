"""Stage 1 extras: landmarks.json, fur_direction.png and protect.png.

Runs without Blender. Everything it needs was baked already - the region
masks, the position map from bake_geometry.py, the template's own base
colour texture - plus the joint positions read straight out of the GLB.

landmarks.json
    Named points on the template, each as a UV texel and a world position:
    the two eyes, the nose, both ear bases, the chest centre, the belly
    centre and the four paws. The eyes and the nose come from the _features
    mask, whose islands on the face are exactly those; the rest come from
    the rig's joints snapped to the nearest texel of the right region.
    "left" and "right" are the sides of a FRONT-VIEW IMAGE (world -x is
    left), the same convention the photo landmarks use, so the two can be
    paired without thinking about which is the animal's own left.

fur_direction.png
    Which way the hair lies, per texel, in UV space. Fur is not painted by
    hand here: it is a world-space rule - back along the body, down the
    legs, out along the tail and the ears, back from the nose over the head
    - carried into UV space through the baked position map. The finite
    differences of position across a texel say how world directions land in
    UV, so a 2x2 solve per texel turns the world flow into a UV vector.
    R,G encode the unit vector (x right, y down) as (v+1)/2; B is 255 where
    the direction is defined.

protect.png
    Texels the coat must never paint over: the eyes, the nose leather, the
    mouth interior and the paw pads. The features mask holds the first
    three but also drags in the fur around each eye (the lid bones); the
    non-fur core of each island is kept, and an island that is mostly
    non-fur - an eyeball, the tongue - is kept whole. Pads are the dark,
    flat texels inside the paws mask.

Usage:
    python ar_pipeline/build_landmarks.py \
        output/skeleton_shiba_small.glb output/templates/shiba_small output/geometry
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.mesh import TemplateMesh  # noqa: E402

# How the joints in the rig map onto landmark names. Left/right are image
# sides: the joint with world x < 0 is the "left" one, whatever the rig
# calls it.
EAR_BASE_JOINTS = ("L_ear_base_jnt", "R_ear_base_jnt")
EAR_TIP_JOINTS = ("L_ear_tip_jnt", "R_ear_tip_jnt")
PAW_JOINTS = {"paw_lf": "LF_toes_jnt", "paw_rf": "RF_toes_jnt",
              "paw_lb": "LB_toes_jnt", "paw_rb": "RB_toes_jnt"}
TAIL_JOINTS = ("tail_1_jnt", "tail_7_jnt")

# A feature island whose non-fur core is at least this share of it is an
# eyeball or the tongue, and is protected whole.
WHOLE_ISLAND_SHARE = 0.6
# ...unless it is the nose: a feature island whose texels sit within this
# many metres of the nose landmark (the eyes are 0.09+ away on the jindo; a
# midline seam splits the nose into two islands, both within 0.03). The
# nose island carries the template's own muzzle fur around the leather -
# white on the jindo, and smooth enough that the fur test in feature_core
# calls it non-fur, so the whole-island rule put it back on every dog as a
# white ring round the nose of a black-muzzled one. On the nose only the
# dark leather is protected: texels darker than this (linear luma), grown
# a little so the nostril edges come with it.
NOSE_ISLAND_REACH_M = 0.04
# 0.35 let the grey fur round the jindo's leather through (a quarter of the
# island was above 0.1) and it rendered as a white lip ring on a black dog;
# the leather is below 0.1, and closing + hole filling keeps the nostrils.
NOSE_LEATHER_MAX_LUMA = 0.1
NOSE_LEATHER_GROW_PX = 1
# The eye sockets: on a template whose eyes are painted on the face (the
# Meshy jindo) the eye's feature island is whole-protected and carries a
# ring of the template's own white fur round the lids, which came back on
# every dog as white spectacles. On an island within this reach of an eye
# landmark only the eye opening is protected: the dark lid line (linear
# luma below EYE_LID_MAX_LUMA, its largest piece) with the iris and the
# highlight it encloses, grown a little.
EYE_ISLAND_REACH_M = 0.03
# 0.2 plus a 2 px grow kept the lid's grey fade into the white fur, a grey
# rim round every eye; the lid itself is below 0.1.
EYE_LID_MAX_LUMA = 0.1
EYE_OPENING_GROW_PX = 0
# Paw pads: darker than this (linear luma) inside the paws mask.
PAD_MAX_LUMA = 0.12
PAD_MIN_PIXELS = 40

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def load_mask(masks_dir: Path, name: str, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(masks_dir / f"{name}.png").convert("L")
    if image.size != size:
        image = image.resize(size, Image.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def nearest_texel(position: np.ndarray, valid: np.ndarray, mask: np.ndarray,
                  point: np.ndarray) -> tuple[int, int]:
    """The (x, y) texel of `mask` whose world position is closest to `point`."""
    select = (mask > 0.5) & valid
    rows, cols = np.nonzero(select)
    distance = np.linalg.norm(position[rows, cols] - point, axis=1)
    best = int(np.argmin(distance))
    return int(cols[best]), int(rows[best])


def feature_islands(features: np.ndarray, valid: np.ndarray, position: np.ndarray):
    labels, count = ndimage.label(features > 0.5)
    islands = []
    for index in range(1, count + 1):
        select = (labels == index) & valid
        if select.sum() < 30:
            continue
        rows, cols = np.nonzero(select)
        islands.append({
            "label": index,
            "texels": int(select.sum()),
            "uv": (float(cols.mean()), float(rows.mean())),
            "world": position[select].mean(axis=0),
            "z_span": float(position[select][:, 2].max() - position[select][:, 2].min()),
        })
    return labels, islands


def eyes_and_nose(islands, position, valid, labels, joints=None):
    """The two eye sockets and the nose leather among the feature islands.

    With eye and nose joints in the rig (the shiba source rig has them, and
    auto_rig.py builds them), each is simply the feature island nearest that
    joint - the face-island rule below assumes one face island, and a
    template unwrapped with a midline seam has two.

    The eyeballs are separate meshes with their own UV islands that sit at
    the same world position as the sockets; the sockets are the ones the
    face photo is warped onto, so they are told apart by lying on the same
    connected UV island as the nose - the face island - rather than by
    position. Left is world -x, one socket each side.
    """
    if joints and all(name in joints for name in ("L_eye_jnt", "R_eye_jnt", "nose_jnt")) and islands:
        def nearest(name):
            return min(islands, key=lambda island: float(np.linalg.norm(island["world"] - joints[name])))
        eyes = [nearest("L_eye_jnt"), nearest("R_eye_jnt")]
        if eyes[0] is not eyes[1]:
            left = min(eyes, key=lambda island: island["world"][0])
            right = max(eyes, key=lambda island: island["world"][0])
            print("[landmarks] eyes and nose from the rig's joints")
            return left, right, nearest("nose_jnt")
    uv_islands, _ = ndimage.label(valid)
    nose = min(islands, key=lambda island: island["world"][1])   # most forward (-y)
    face_island = uv_islands[int(nose["uv"][1]), int(nose["uv"][0])]
    on_face = [island for island in islands if island is not nose
               and uv_islands[int(island["uv"][1]), int(island["uv"][0])] == face_island]
    left = [i for i in on_face if i["world"][0] < -0.01]
    right = [i for i in on_face if i["world"][0] > 0.01]
    if not left or not right:
        raise ValueError("could not find an eye socket on each side of the nose on the face island")
    # the socket is the largest feature island on its side of the face
    return max(left, key=lambda i: i["texels"]), max(right, key=lambda i: i["texels"]), nose


def midline_point(position, valid, mask, y_range, z_take_low: bool, share: float = 0.1):
    """The average of the lowest (or frontmost) `share` of the mask's texels
    that sit on the midline within the y range."""
    x, y, z = position[..., 0], position[..., 1], position[..., 2]
    inside = (mask > 0.5) & valid
    # "midline" is relative to the mask: a mask that never reaches x=0 (the
    # chest mask is the two shoulder fronts) still has a nearest-to-centre band
    band = max(0.03, float(np.quantile(np.abs(x[inside]), 0.2)))
    select = inside & (np.abs(x) <= band) & (y >= y_range[0]) & (y <= y_range[1])
    if select.sum() < 20:
        select = inside & (np.abs(x) <= band)
    values = z if z_take_low else y
    threshold = np.quantile(values[select], share)
    chosen = select & (values <= threshold)
    return position[chosen].mean(axis=0)


# --------------------------------------------------------------------------
# fur direction
# --------------------------------------------------------------------------

def world_flow(position: np.ndarray, joints: dict[str, np.ndarray],
               masks: dict[str, np.ndarray]) -> np.ndarray:
    """Which way the hair lies at each texel, in world space (unnormalised)."""
    x, y, z = position[..., 0], position[..., 1], position[..., 2]
    flow = np.zeros_like(position)

    # body: back along the spine
    flow[..., 1] = 1.0

    # head: away from the nose, which is +y over the skull and outward round
    # the muzzle and cheeks
    nose = joints["nose_jnt"]
    head_mask = np.maximum.reduce([masks[n] for n in
                                   ("skull", "eyebrows", "muzzle_bridge", "cheeks", "throat")])
    away = position - nose
    away[..., 2] *= 0.5
    head_weight = np.clip(head_mask, 0, 1)[..., None]
    flow = flow * (1 - head_weight) + away / np.maximum(np.linalg.norm(away, axis=-1, keepdims=True), 1e-4) * head_weight

    # ears: base to tip
    for base_name, tip_name in zip(EAR_BASE_JOINTS, EAR_TIP_JOINTS):
        base, tip = joints[base_name], joints[tip_name]
        direction = (tip - base) / max(np.linalg.norm(tip - base), 1e-4)
        side = (np.sign(x) == np.sign(base[0])) if abs(base[0]) > 1e-3 else np.ones_like(x, bool)
        ear = np.maximum(masks["ears_outer"], masks["ears_inner"]) * side
        flow = flow * (1 - ear[..., None]) + direction * ear[..., None]

    # tail: root to tip
    root, tip = joints[TAIL_JOINTS[0]], joints[TAIL_JOINTS[1]]
    direction = (tip - root) / max(np.linalg.norm(tip - root), 1e-4)
    tail = np.maximum(masks["tail_top"], masks["tail_tip"])
    flow = flow * (1 - tail[..., None]) + direction * tail[..., None]

    # legs: down. legs_lower and paws are always legs; legs_upper and the
    # underside are legs below the elbow/knee height, body above it
    knee_z = float(np.mean([joints[name][2] for name in PAW_JOINTS.values()])) + 0.18
    leg_mask = np.maximum(masks["legs_lower"], masks["paws"])
    below = np.clip((knee_z + 0.06 - z) / 0.12, 0, 1)
    leg_mask = np.maximum(leg_mask, np.maximum(masks["legs_upper"], masks["underside"]) * below)
    down = np.array([0.0, 0.0, -1.0])
    flow = flow * (1 - leg_mask[..., None]) + down * leg_mask[..., None]

    return flow


def uv_direction(position: np.ndarray, valid: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Carry a world direction into UV space through the position map.

    Px = dP/dx and Py = dP/dy (texel steps) span the tangent plane at each
    texel. The UV vector (a, b) with a*Px + b*Py closest to the world flow
    solves the 2x2 normal equations. Differences are only taken between
    valid neighbours so nothing leaks across an island edge.
    """
    interior = ndimage.binary_erosion(valid, iterations=1)
    px = (np.roll(position, -1, axis=1) - np.roll(position, 1, axis=1)) / 2.0
    py = (np.roll(position, -1, axis=0) - np.roll(position, 1, axis=0)) / 2.0

    gxx = (px * px).sum(-1)
    gxy = (px * py).sum(-1)
    gyy = (py * py).sum(-1)
    rx = (px * flow).sum(-1)
    ry = (py * flow).sum(-1)
    det = gxx * gyy - gxy * gxy
    ok = interior & (det > 1e-14)
    a = np.where(ok, (gyy * rx - gxy * ry) / np.where(ok, det, 1.0), 0.0)
    b = np.where(ok, (gxx * ry - gxy * rx) / np.where(ok, det, 1.0), 0.0)
    length = np.hypot(a, b)
    ok &= length > 1e-9
    a = np.where(ok, a / np.maximum(length, 1e-9), 0.0)
    b = np.where(ok, b / np.maximum(length, 1e-9), 0.0)
    return np.stack([a, b], axis=-1).astype(np.float32), ok


def smooth_and_pad(direction: np.ndarray, ok: np.ndarray, valid: np.ndarray,
                   sigma: float = 3.0, pad: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Smooth the vector field inside the islands (normalised average) and
    spread it a few texels past the island edge so bilinear sampling never
    reads an undefined texel."""
    weight = ok.astype(np.float32)
    smoothed = np.stack([ndimage.gaussian_filter(direction[..., i] * weight, sigma) for i in range(2)], -1)
    norm = ndimage.gaussian_filter(weight, sigma)
    smoothed = smoothed / np.maximum(norm, 1e-6)[..., None]
    defined = norm > 1e-3
    # pad outward by nearest defined texel
    indices = ndimage.distance_transform_edt(~defined, return_distances=False, return_indices=True)
    padded = smoothed[indices[0], indices[1]]
    reach = ndimage.binary_dilation(defined, iterations=pad)
    padded = np.where(reach[..., None], padded, 0.0)
    length = np.linalg.norm(padded, axis=-1, keepdims=True)
    padded = np.where(length > 1e-6, padded / np.maximum(length, 1e-6), 0.0)
    return padded.astype(np.float32), reach


# --------------------------------------------------------------------------
# protect
# --------------------------------------------------------------------------

def eye_opening(island: np.ndarray, luma: np.ndarray) -> np.ndarray | None:
    """The eye opening inside an eye-socket island: the lid line's largest
    dark piece, holes filled (iris, highlight), grown; soft-edged. None when
    the island has no dark lid."""
    dark = ndimage.binary_closing(island & (luma < EYE_LID_MAX_LUMA), iterations=2)
    labels, count = ndimage.label(dark)
    if count == 0:
        return None
    sizes = ndimage.sum(dark, labels, range(1, count + 1))
    lid = labels == 1 + int(np.argmax(sizes))
    opening = ndimage.binary_fill_holes(lid)
    if EYE_OPENING_GROW_PX:
        opening = ndimage.binary_dilation(opening, iterations=EYE_OPENING_GROW_PX)
    opening &= island
    return np.clip(ndimage.gaussian_filter(opening.astype(np.float32), 0.7), 0, 1)


def nose_leather(island: np.ndarray, luma: np.ndarray) -> np.ndarray:
    """The dark leather of a nose island, nostrils included, soft-edged."""
    leather = ndimage.binary_closing(island & (luma < NOSE_LEATHER_MAX_LUMA), iterations=2)
    leather = ndimage.binary_fill_holes(leather)
    leather = ndimage.binary_dilation(leather, iterations=NOSE_LEATHER_GROW_PX) & island
    return np.clip(ndimage.gaussian_filter(leather.astype(np.float32), 0.7), 0, 1)


def trim_eye_sockets(protect: np.ndarray, luma: np.ndarray, valid: np.ndarray, position: np.ndarray,
                     eyes_world: list[np.ndarray]) -> tuple[np.ndarray, int]:
    """protect with every island near an eye cut to its eye opening."""
    protect = protect.copy()
    labels, count = ndimage.label(protect > 0.5)
    trimmed = 0
    for index in range(1, count + 1):
        island = labels == index
        texels = island & valid
        if not texels.any():
            continue
        reach = min(float(np.median(np.linalg.norm(position[texels] - eye, axis=1))) for eye in eyes_world)
        if reach >= EYE_ISLAND_REACH_M:
            continue
        opening = eye_opening(island, luma)
        if opening is not None:
            protect[island] = opening[island]
            trimmed += 1
    return protect, trimmed


def protect_mask(features: np.ndarray, paws: np.ndarray, base: Image.Image,
                 valid: np.ndarray, position: np.ndarray | None = None,
                 nose_world: np.ndarray | None = None,
                 eyes_world: list[np.ndarray] | None = None) -> np.ndarray:
    from api.texture import feature_core

    core = feature_core(features, base)
    linear = (np.asarray(base, dtype=np.float32) / 255.0) ** 2.2
    luma = linear @ LUMA
    labels, count = ndimage.label(features > 0.5)
    protect = core.copy()
    nose_islands = 0
    for index in range(1, count + 1):
        island = labels == index
        if island.sum() < 30:
            continue
        if position is not None and nose_world is not None:
            texels = island & valid
            if texels.any():
                distance = np.linalg.norm(position[texels] - nose_world, axis=1)
                if float(np.median(distance)) < NOSE_ISLAND_REACH_M:
                    protect[island] = nose_leather(island, luma)[island]
                    nose_islands += 1
                    continue
        share = float((core[island] > 0.5).mean())
        if share >= WHOLE_ISLAND_SHARE:
            protect[island] = 1.0
    if position is not None and nose_world is not None:
        print(f"[landmarks] protect: {nose_islands} nose island(s) kept to the leather only")
    if position is not None and eyes_world:
        protect, sockets = trim_eye_sockets(protect, luma, valid, position, eyes_world)
        print(f"[landmarks] protect: {sockets} eye socket(s) kept to the eye opening only")

    pads = (paws > 0.5) & (luma < PAD_MAX_LUMA) & valid
    pads = ndimage.binary_opening(pads, iterations=2)
    labels, count = ndimage.label(pads)
    sizes = ndimage.sum(pads, labels, range(1, count + 1))
    keep = np.isin(labels, [i + 1 for i, s in enumerate(sizes) if s >= PAD_MIN_PIXELS])
    protect = np.maximum(protect, keep.astype(np.float32))
    return np.clip(protect, 0, 1)


# --------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    glb_path, template_dir, geometry_dir = (Path(p) for p in sys.argv[1:])
    # the sixteen source masks: the landmarks need the ears, the tail and the
    # throat on their own, which the ten region masks no longer separate
    masks_dir = template_dir / "masks_source"
    features_dir = template_dir / "masks"

    manifest = json.loads((geometry_dir / "geometry.json").read_text(encoding="utf-8"))
    size = tuple(manifest["size"])
    position = np.load(geometry_dir / manifest["position"]["file"]).astype(np.float32)
    valid = np.load(geometry_dir / manifest["valid"]["file"]) > 0.5

    names = ["skull", "eyebrows", "muzzle_bridge", "cheeks", "throat", "ears_outer",
             "ears_inner", "chest", "underside", "legs_upper", "legs_lower", "paws",
             "tail_top", "tail_tip", "collar", "back"]
    masks = {name: load_mask(masks_dir, name, size) for name in names}
    features = load_mask(features_dir, "_features", size)
    base = Image.open(template_dir / "base_color.png").convert("RGB")

    mesh = TemplateMesh.load(glb_path, size)
    joints = mesh.joints

    # ---- landmarks -------------------------------------------------------
    labels, islands = feature_islands(features, valid, position)
    left_eye, right_eye, nose = eyes_and_nose(islands, position, valid, labels, joints)

    points: dict[str, dict] = {}

    def put(name: str, uv, world, source: str) -> None:
        points[name] = {
            "uv": [int(round(uv[0])), int(round(uv[1]))],
            "world": [round(float(v), 5) for v in world],
            "source": source,
        }

    put("left_eye", left_eye["uv"], left_eye["world"], "features island (eye socket)")
    put("right_eye", right_eye["uv"], right_eye["world"], "features island (eye socket)")
    put("nose", nose["uv"], nose["world"], "features island (nose leather)")

    head = np.maximum.reduce([masks["ears_outer"], masks["ears_inner"], masks["skull"]])
    ear_bases = sorted((joints[name] for name in EAR_BASE_JOINTS), key=lambda p: p[0])
    for label, joint in zip(("left_ear_base", "right_ear_base"), ear_bases):
        uv = nearest_texel(position, valid, head, joint)
        put(label, uv, position[uv[1], uv[0]], "joint ear_base snapped to the head")

    # the chest mask is the two shoulder fronts, so its centre is the
    # breastbone between them: the frontmost band mirrored onto x = 0, then
    # snapped to whichever region's texel is nearest
    chest_world = midline_point(position, valid, masks["chest"],
                                (joints["chest_jnt"][1] - 0.3, joints["chest_jnt"][1] + 0.1), False)
    chest_world = np.array([0.0, chest_world[1], chest_world[2]], dtype=np.float32)
    uv = nearest_texel(position, valid, valid.astype(np.float32), chest_world)
    put("chest_center", uv, position[uv[1], uv[0]], "breastbone: chest mask front mirrored to x=0")

    belly_world = midline_point(position, valid, masks["underside"],
                                (joints["spine_1_jnt"][1] - 0.05, joints["L_hip_jnt"][1] - 0.05), True)
    uv = nearest_texel(position, valid, masks["underside"], belly_world)
    put("belly_center", uv, position[uv[1], uv[0]], "lowest midline of the underside mask")

    for label, joint_name in PAW_JOINTS.items():
        uv = nearest_texel(position, valid, masks["paws"], joints[joint_name])
        put(label, uv, position[uv[1], uv[0]], f"joint {joint_name} snapped to the paws")

    landmarks = {
        "template": glb_path.name,
        "size": list(size),
        "space": {
            "uv": "texel coordinates, x right, y down, origin top-left",
            "world": "Blender convention: z up, -y forward, metres; matches geometry/position.npy",
            "left_right": "sides of a front-view image: left is world -x",
        },
        "points": points,
        "anchors": {
            "eye": ["left_eye", "right_eye"],
            "nose": ["nose"],
            "ear_base": ["left_ear_base", "right_ear_base"],
            "chest_center": ["chest_center"],
            "belly_center": ["belly_center"],
            "paw": list(PAW_JOINTS),
        },
        "eye_distance_world": round(float(np.linalg.norm(
            np.array(left_eye["world"]) - np.array(right_eye["world"]))), 5),
    }
    (template_dir / "landmarks.json").write_text(
        json.dumps(landmarks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[landmarks] wrote {template_dir / 'landmarks.json'}")
    for name, point in points.items():
        print(f"  {name:16s} uv={point['uv']} world={point['world']}")

    # ---- fur direction ---------------------------------------------------
    flow = world_flow(position, joints, masks)
    direction, ok = uv_direction(position, valid, flow)
    direction, defined = smooth_and_pad(direction, ok, valid)
    encoded = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    encoded[..., 0] = np.round((direction[..., 0] + 1) / 2 * 255).astype(np.uint8)
    encoded[..., 1] = np.round((direction[..., 1] + 1) / 2 * 255).astype(np.uint8)
    encoded[..., 2] = np.where(defined, 255, 0).astype(np.uint8)
    Image.fromarray(encoded, mode="RGB").save(template_dir / "fur_direction.png")
    print(f"[landmarks] wrote fur_direction.png ({defined.mean():.1%} of texels defined)")

    # ---- protect ---------------------------------------------------------
    protect = protect_mask(features, masks["paws"], base, valid, position,
                           np.asarray(nose["world"], dtype=np.float32),
                           [np.asarray(eye["world"], dtype=np.float32) for eye in (left_eye, right_eye)])
    Image.fromarray((protect * 255).astype(np.uint8), mode="L").save(template_dir / "protect.png")
    print(f"[landmarks] wrote protect.png ({int((protect > 0.5).sum())} texels)")


if __name__ == "__main__":
    main()
