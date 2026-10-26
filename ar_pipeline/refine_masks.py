"""Redefine the body region masks from the mesh's geometry, not its bones.

WHY
---
The template's region masks were seeded from bone weights (seed_regions.py).
Bones follow the rig, not the coat: "chest" came out as the two shoulder
fronts plus a piece of the hip, "underside" ran down the inside and back
of every leg. Every stage that reads, scores, combines or protects by
region was measuring with that ruler, and the errors it produced (white
shoulders, black bellies, orange hips) were the masks' errors.

WHAT THIS DOES
--------------
The baked position and normal maps say, for every texel, where it is on
the animal and which way it faces; the rig says where the limbs are. Each
texel is first assigned to the nearest bone chain (trunk, neck, a leg, the
tail) and then to a coat region by anatomy:

  neck    throat      the front and underside of the neck (facing forward/down)
          collar      the rest of the neck
  trunk   chest       the breast: forward-facing, between and above the forelegs
          underside   the belly: facing down, below the trunk's midline
          back        everything else on the trunk (top and flanks)
  legs    legs_upper  above the elbow / knee
          legs_lower  elbow-to-wrist / knee-to-ankle
          paws        below the wrist / ankle

The head regions (eyebrows, muzzle_bridge, cheeks, skull, ears) and the
tail regions were tagged by hand around real anatomy and are kept. The old
masks are moved to masks_bones/ so the change can be undone.

Works at the sixteen SOURCE regions in masks_source/ (the kept ones are read
from there, the body ones written there) and then writes masks/ as the
ten-region union, the same way build_template does.

Usage:
    python ar_pipeline/refine_masks.py [--template shiba_small] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.mesh import TemplateMesh  # noqa: E402
from api.regions import MERGE, REGIONS, SOURCE_REGIONS, SOURCES  # noqa: E402

KEEP = ("eyebrows", "muzzle_bridge", "cheeks", "skull", "ears_outer", "ears_inner", "tail_top", "tail_tip")
BODY = ("throat", "collar", "back", "chest", "underside", "legs_upper", "legs_lower", "paws")

# Bone chains as lists of joint names; a texel belongs to the chain whose
# segments it is nearest, weighted by the chain's typical radius so a thin
# leg does not steal the flank next to it.
CHAINS = {
    "trunk": (["chest_jnt", "spine_2_jnt", "spine_1_jnt", "pelvis_jnt", "tail_1_jnt"], 0.17),
    "neck": (["neck_base_jnt", "neck_mid_jnt", "head_jnt"], 0.11),
    "head": (["head_jnt", "nose_jnt"], 0.12),
    "leg_LF": (["L_shoulder_jnt", "L_elbow_jnt", "L_wrist_jnt", "L_hand_jnt", "LF_toes_jnt"], 0.05),
    "leg_RF": (["R_shoulder_jnt", "R_elbow_jnt", "R_wrist_jnt", "R_hand_jnt", "RF_toes_jnt"], 0.05),
    "leg_LB": (["L_hip_jnt", "L_knee_jnt", "L_ankle_jnt", "L_foot_jnt", "LB_toes_jnt"], 0.06),
    "leg_RB": (["R_hip_jnt", "R_knee_jnt", "R_ankle_jnt", "R_foot_jnt", "RB_toes_jnt"], 0.06),
    "tail": (["tail_1_jnt", "tail_3_jnt", "tail_5_jnt", "tail_7_jnt"], 0.04),
}
# Facing thresholds (dot with world axes; -y is forward, -z is down).
THROAT_FACING = 0.25       # normal·(forward+down)/√2 above this on the neck: throat
CHEST_FORWARD = 0.45       # normal·forward above this on the trunk front: chest
BELLY_DOWN = 0.35          # normal·down above this on the trunk: underside
BELLY_BEHIND_TAIL = 0.02   # trunk texels this far behind the tail root are the tail's underside, not belly
# The leg chains run close to the sternum and the groin, so the trunk texels
# BETWEEN a pair of legs (inside this share of the shoulder / hip half-width,
# above the elbow / knee) came out as legs_upper and painted base-colour: a
# black band between the white bib and the white belly. They are trunk.
BETWEEN_LEGS = 0.75
# the belly and chest edges are decided texel by texel from the facing and
# come out ragged; a closing then an opening this many texels wide takes
# the spurs off before the feather
EDGE_CLEAN_PX = 4
FEATHER_PX = 2


def segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((points - a) @ ab) / max(float(ab @ ab), 1e-9), 0.0, 1.0)
    closest = a + t[:, None] * ab
    return np.linalg.norm(points - closest, axis=1)


def chain_distance(points: np.ndarray, joints: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    best = np.full(points.shape[0], np.inf, dtype=np.float32)
    for a, b in zip(names[:-1], names[1:]):
        best = np.minimum(best, segment_distance(points, joints[a], joints[b]))
    return best


def smooth_normals(normal: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smoothed unit normals, each UV island averaged with ITSELF
    only. A blur over the whole map, even weighted by validity, mixes in
    the islands packed a few texels away: a narrow leg island took its
    neighbour's normals along both rims and the whole leg tested as
    "facing inward", which painted one foreleg in the chest colour."""
    labels, count = ndimage.label(valid)
    out = normal.astype(np.float32).copy()
    objects = ndimage.find_objects(labels)
    pad = int(3 * sigma) + 1
    h, w = valid.shape
    for index, box in enumerate(objects, start=1):
        if box is None:
            continue
        r0, r1 = max(0, box[0].start - pad), min(h, box[0].stop + pad)
        c0, c1 = max(0, box[1].start - pad), min(w, box[1].stop + pad)
        own = (labels[r0:r1, c0:c1] == index)
        if own.sum() < 4:
            continue
        weight = ndimage.gaussian_filter(own.astype(np.float32), sigma)
        blurred = np.stack([ndimage.gaussian_filter(normal[r0:r1, c0:c1, i] * own, sigma) for i in range(3)], axis=-1)
        blurred /= np.maximum(weight, 1e-3)[..., None]
        length = np.linalg.norm(blurred, axis=-1, keepdims=True)
        blurred = np.where(length > 1e-3, blurred / np.maximum(length, 1e-3), normal[r0:r1, c0:c1])
        out[r0:r1, c0:c1][own] = blurred[own]
    return out


def load_mask(masks_dir: Path, name: str, size) -> np.ndarray:
    return np.asarray(Image.open(masks_dir / f"{name}.png").convert("L").resize(size), dtype=np.float32) / 255.0


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8), mode="L").save(path)


def main() -> None:
    global EDGE_CLEAN_PX
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default="shiba_small")
    parser.add_argument("--glb", default="output/skeleton_shiba_small.glb")
    parser.add_argument("--geometry", default="output/geometry")
    parser.add_argument("--dry-run", action="store_true", help="write to masks_refined/ instead of replacing")
    parser.add_argument("--normal-blur", type=float, default=0.0,
                        help="blur the normal map this many texels (inside the islands) before the facing tests: "
                             "a generated mesh is bumpy at the vertex level and its raw normals cut the "
                             "chest/belly/leg boundaries into a ragged fringe")
    parser.add_argument("--label-smooth", type=float, default=0.0,
                        help="blur radius (texels) of a majority vote over the body labels before the edge clean")
    parser.add_argument("--edge-clean", type=int, default=EDGE_CLEAN_PX,
                        help="closing/opening radius that takes the spurs off the belly and chest edges")
    args = parser.parse_args()
    EDGE_CLEAN_PX = args.edge_clean

    template_dir = Path("output/templates") / args.template
    masks_dir = template_dir / "masks"
    source_dir = template_dir / "masks_source"
    manifest = json.loads((Path(args.geometry) / "geometry.json").read_text(encoding="utf-8"))
    size = tuple(manifest["size"])
    position = np.load(Path(args.geometry) / manifest["position"]["file"]).astype(np.float32)
    normal = np.load(Path(args.geometry) / manifest["normal"]["file"]).astype(np.float32)
    valid = np.load(Path(args.geometry) / manifest["valid"]["file"]) > 0.5
    if args.normal_blur > 0:
        normal = smooth_normals(normal, valid, args.normal_blur)
        print(f"[masks] normals smoothed over {args.normal_blur:.0f} texels")
    joints = TemplateMesh.load(args.glb, size).joints

    kept = {name: load_mask(source_dir, name, size) for name in KEEP}
    features = load_mask(masks_dir, "_features", size)
    taken = np.zeros(valid.shape, dtype=bool)
    for name in KEEP:
        taken |= kept[name] > 0.5
    # the eyeballs, tongue and mouth interior are not coat
    taken |= features > 0.5

    body = valid & ~taken
    rows, cols = np.nonzero(body)
    points = position[rows, cols]
    normals = normal[rows, cols]

    # nearest chain, radius-weighted
    chain_names = list(CHAINS)
    distances = np.stack([chain_distance(points, joints, CHAINS[c][0]) / CHAINS[c][1] for c in chain_names], axis=1)
    chain = np.argmin(distances, axis=1)
    # the sternum and the groin belong to the trunk, whichever chain is nearer
    trunk_index = chain_names.index("trunk")
    front_between = ((np.abs(points[:, 0]) < BETWEEN_LEGS * abs(joints["L_shoulder_jnt"][0]))
                     & (points[:, 2] > joints["L_elbow_jnt"][2])
                     & (points[:, 1] > joints["L_shoulder_jnt"][1] - 0.06) & (points[:, 1] < joints["spine_1_jnt"][1]))
    hind_between = ((np.abs(points[:, 0]) < BETWEEN_LEGS * abs(joints["L_hip_jnt"][0]))
                    & (points[:, 2] > joints["L_knee_jnt"][2])
                    & (points[:, 1] > joints["spine_1_jnt"][1]) & (points[:, 1] < joints["tail_1_jnt"][1]))
    is_leg = np.array([chain_names[c].startswith("leg_") for c in chain])
    chain = np.where(is_leg & (front_between | hind_between), trunk_index, chain)
    label = np.full(points.shape[0], "", dtype=object)

    forward = -normals[:, 1]
    down = -normals[:, 2]
    z = points[:, 2]
    y = points[:, 1]

    for index, name in enumerate(chain_names):
        sel = chain == index
        if name == "neck" or name == "head":
            throat = sel & (((forward + down) / np.sqrt(2.0)) > THROAT_FACING)
            label[throat] = "throat"
            label[sel & ~throat] = "collar"
        elif name == "trunk":
            chest = sel & (forward > CHEST_FORWARD) & (y < joints["chest_jnt"][1] + 0.02)
            # the trunk chain reaches the tail root, and the underside of the
            # tail base faces down too: it stays with the back (the tail's
            # colour), not the belly
            belly = sel & ~chest & (down > BELLY_DOWN) & (y < joints["tail_1_jnt"][1] - BELLY_BEHIND_TAIL)
            label[chest] = "chest"
            label[belly] = "underside"
            label[sel & ~chest & ~belly] = "back"
        elif name.startswith("leg_"):
            side = "L" if name[4] == "L" else "R"
            front = name.endswith("F")
            upper_z = joints[f"{side}_elbow_jnt"][2] if front else joints[f"{side}_knee_jnt"][2]
            lower_z = joints[f"{side}_wrist_jnt"][2] if front else joints[f"{side}_ankle_jnt"][2]
            label[sel & (z > upper_z)] = "legs_upper"
            label[sel & (z <= upper_z) & (z > lower_z)] = "legs_lower"
            label[sel & (z <= lower_z)] = "paws"
        elif name == "tail":
            # the tail masks are kept; tail texels the kept masks missed join tail_top
            label[sel] = "tail_top"

    masks = {name: kept[name].copy() for name in KEEP}
    for name in BODY:
        masks[name] = np.zeros(valid.shape, dtype=np.float32)
    for name in BODY + ("tail_top",):
        pick = label == name
        masks[name][rows[pick], cols[pick]] = 1.0
    if args.label_smooth > 0:
        # majority vote over a blurred neighbourhood: a boundary that a
        # facing threshold cuts into a fractal fringe (the chest on the
        # shoulder of a bumpy generated mesh) becomes one line, and single
        # texels of another region inside a mask vanish
        names = list(BODY) + ["tail_top"]
        weight = np.maximum(ndimage.gaussian_filter(body.astype(np.float32), args.label_smooth), 1e-3)
        scores = np.stack([ndimage.gaussian_filter(masks[n] * body, args.label_smooth) / weight for n in names])
        winner = np.argmax(scores, axis=0)
        for i, n in enumerate(names):
            masks[n] = ((winner == i) & body).astype(np.float32) if n != "tail_top"                 else np.maximum(kept["tail_top"], ((winner == i) & body).astype(np.float32))
        print(f"[masks] labels smoothed over {args.label_smooth:.0f} texels")
    # take the spurs off the belly and chest edges; what they lose goes to
    # the back (the base coat), what they gain comes from it
    disc = np.ones((EDGE_CLEAN_PX * 2 + 1, EDGE_CLEAN_PX * 2 + 1), dtype=bool)
    for name in ("underside", "chest"):
        hard = masks[name] > 0.5
        cleaned = ndimage.binary_opening(ndimage.binary_closing(hard, disc), disc) & body
        masks["back"][hard & ~cleaned] = 1.0
        masks["back"][cleaned & ~hard] = 0.0
        masks[name] = cleaned.astype(np.float32)

    # a few texels of feather at every edge, and the padding past the
    # islands filled from the nearest island texel, as build_template does
    indices = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    for name in BODY:
        mask = masks[name]
        mask = mask[indices[0], indices[1]]
        mask = ndimage.gaussian_filter(mask, FEATHER_PX / 2.0)
        masks[name] = np.clip(mask, 0, 1)

    if args.dry_run:
        source_out, out_dir = template_dir / "masks_refined_source", template_dir / "masks_refined"
    else:
        source_out, out_dir = source_dir, masks_dir
        backup = template_dir / "masks_bones"
        if not backup.exists():
            shutil.copytree(source_dir, backup)
            print(f"[masks] previous source masks backed up to {backup}")
    source_out.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in BODY:
        save_mask(source_out / f"{name}.png", masks[name])
    merged = {region: np.max(np.stack([masks[s] for s in SOURCES[region]]), axis=0) for region in REGIONS}
    for region in REGIONS:
        save_mask(out_dir / f"{region}.png", merged[region])
    counts = {name: int((merged[name] > 0.5).sum()) for name in REGIONS}
    unassigned = int((body & ~np.any(np.stack([masks[n] > 0.5 for n in BODY]), axis=0)).sum())
    print(f"[masks] wrote {len(BODY)} body source masks to {source_out} and {len(REGIONS)} region masks "
          f"to {out_dir}; texels per region: {counts}; unassigned body texels: {unassigned}")


if __name__ == "__main__":
    main()
