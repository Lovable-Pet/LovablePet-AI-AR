"""Re-weight a template's legs so they bend as legs: each leg's skin to its
own bone chain, the torso's to the torso, a narrow seam between.

WHY
---
auto_rig.skin weighs every vertex by a Gaussian of its distance to every
bone over the bone's radius - 14 cm for the pelvis, spine and COG, 8 for a
hip. On the jindo the thigh's outer face sits about as near (in radii) to
the pelvis bone, which runs through the hindquarters, as to the femur: 1100
thigh vertices had the hip at 0.28 on average and never above 0.5. While
the templates only stood still that did not show; with clips (retarget_clip,
11-03) the thigh followed its leg half way and melted into the belly when
the dog sat, and a raised foreleg dragged the chest (user: "팔이 섞이고
웨이트가 잘못된 느낌").

HOW
---
Leg membership per vertex: its distance to the leg's bone chain (hip or
shoulder down to the toes) over LEG_RADIUS against its distance to the
torso's core line (pelvis, COG, spine, chest) over CORE_RADIUS, through a
logistic of slope MEMBERSHIP_SLOPE - nearly all leg or all torso, with a
seam a centimetre or two wide. A leg vertex's weights are a Gaussian over
that chain's bones alone (CHAIN_RADIUS, the strongest MAX_INFLUENCES), a
torso vertex keeps its old weights on the torso's bones, and the seam
mixes the two. Vertices whose strongest old bone belongs to neither (head,
neck, ears, tail) are left as they were. Positions, UVs and morph targets
are not touched: re-running the clips (retarget_clip --install) after this
is all that is needed.

    python -m ar_pipeline.leg_weights --slug jindo             # report, write work/leg_weights_<slug>.glb
    python -m ar_pipeline.leg_weights --slug jindo --install   # into the template GLB

--install backs the GLB up to work/backup_weights_<slug>/ first (once) and
always works from that backup.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from pygltflib import GLTF2

from ar_pipeline.retarget_clip import Rig, read

ROOT = Path(__file__).resolve().parent.parent
CHAINS = {
    "LB": ("L_hip_jnt", "L_knee_jnt", "L_ankle_jnt", "L_foot_jnt", "LB_toes_jnt"),
    "RB": ("R_hip_jnt", "R_knee_jnt", "R_ankle_jnt", "R_foot_jnt", "RB_toes_jnt"),
    "LF": ("L_shoulder_jnt", "L_elbow_jnt", "L_wrist_jnt", "L_hand_jnt", "LF_toes_jnt"),
    "RF": ("R_shoulder_jnt", "R_elbow_jnt", "R_wrist_jnt", "R_hand_jnt", "RF_toes_jnt"),
}
CORE = ("pelvis_jnt", "COG_jnt", "spine_2_jnt", "chest_jnt")
TORSO_BONES = ("root", "COG_jnt", "pelvis_jnt", "central_corrective", "spine_1_jnt", "spine_2_jnt",
               "breathe_jnt", "chest_jnt", "L_clavicle_jnt", "R_clavicle_jnt")
# each leg bone's reach for membership, hip (shoulder) down: the thigh is
# some 19 cm deep front to back round its bone
LEG_RADII = {"B": (0.10, 0.06, 0.045, 0.04), "F": (0.09, 0.055, 0.04, 0.035)}
CORE_RADIUS = 0.12
# above the hip (shoulder) joint the skin is the torso's, fading over this (m)
TOP_FADE_M = 0.04
MEMBERSHIP_SLOPE = 10.0
LEG_SHARE_SLOPE = 25.0
CHAIN_RADIUS = 0.05
MAX_INFLUENCES = 4


def _segments(points: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(zip(points, points[1:]))


def _distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def new_weights(rig: Rig, pos: np.ndarray, joints: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, dict]:
    """(dense weights over the skin's joints, report)."""
    names = [(rig.gltf.nodes[j].name or "").split(".")[0] for j in rig.joints]
    index = {n: k for k, n in enumerate(names)}
    world = rig.world()
    at = {n: world[rig.joints[k]][:3, 3] for n, k in index.items()}
    n_joints = len(names)
    old = np.zeros((len(pos), n_joints))
    np.add.at(old, (np.repeat(np.arange(len(pos)), joints.shape[1]), joints.ravel()), weights.ravel())
    old /= np.maximum(old.sum(axis=1, keepdims=True), 1e-9)
    dominant = old.argmax(axis=1)
    leg_bones = {index[n] for chain in CHAINS.values() for n in chain if n in index}
    torso_bones = [index[n] for n in TORSO_BONES if n in index]
    editable = np.isin(dominant, list(leg_bones) + torso_bones)

    core_pts = [at[n] for n in CORE if n in at]
    d_core = np.min([_distance(pos, a, b) for a, b in _segments(core_pts)], axis=0)
    out = old.copy()
    torso_part = np.zeros_like(old)
    torso_part[:, torso_bones] = old[:, torso_bones]
    has_torso = torso_part.sum(axis=1, keepdims=True)
    report = {}
    legs = [leg for leg, chain in CHAINS.items() if all(n in at for n in chain)]
    seg_ds, d_norm, tops = {}, {}, {}
    for leg in legs:
        pts = [at[n] for n in CHAINS[leg]]
        seg_ds[leg] = np.stack([_distance(pos, a, b) for a, b in _segments(pts)], axis=1)
        d_norm[leg] = (seg_ds[leg] / np.array(LEG_RADII[leg[1]])).min(axis=1)
        tops[leg] = np.clip((pts[0][1] - pos[:, 1]) / TOP_FADE_M + 0.5, 0.0, 1.0)
    # the legs compete among themselves first - the nearest takes nearly all
    # (LEG_SHARE_SLOPE): judged each against the torso alone, a lower
    # foreleg far from the core line went a tenth to the other foreleg too,
    # and the raised paw of "shake" pulled a ghost of the other leg up with it
    stack = np.stack([d_norm[leg] for leg in legs], axis=1)
    nearest = stack.min(axis=1)
    share = np.exp(-LEG_SHARE_SLOPE * (stack - nearest[:, None]))
    share /= share.sum(axis=1, keepdims=True)
    to_legs = 1.0 / (1.0 + np.exp(-MEMBERSHIP_SLOPE * (d_core / CORE_RADIUS - nearest)))
    member_total = np.zeros(len(pos))
    leg_dense = np.zeros_like(old)
    for k_leg, leg in enumerate(legs):
        member = to_legs * share[:, k_leg] * tops[leg]
        member_total += member
        g = np.exp(-(seg_ds[leg] / CHAIN_RADIUS) ** 2)
        g /= np.maximum(g.sum(axis=1, keepdims=True), 1e-12)
        dense = np.zeros_like(old)
        for k, n in enumerate(CHAINS[leg][:-1]):
            dense[:, index[n]] = g[:, k]
        leg_dense += member[:, None] * dense
        report[leg] = int((editable & (member > 0.5)).sum())
    member_total = np.minimum(member_total, 1.0)
    # torso: the old torso weights, renormalised (all old if it had none)
    torso_norm = np.where(has_torso > 1e-6, torso_part / np.maximum(has_torso, 1e-9), old)
    mixed = leg_dense + (1.0 - member_total)[:, None] * torso_norm
    out[editable] = mixed[editable]
    # the strongest MAX_INFLUENCES, normalised
    order = np.argsort(-out, axis=1)[:, :MAX_INFLUENCES]
    kept = np.zeros_like(out)
    rows = np.arange(len(pos))[:, None]
    kept[rows, order] = out[rows, order]
    kept /= np.maximum(kept.sum(axis=1, keepdims=True), 1e-9)
    report["edited"] = int(editable.sum())
    return kept, report


def _write(gltf: GLTF2, blob: bytearray, accessor: int, data: np.ndarray) -> None:
    """Overwrite an accessor's data in place (same count, type and component type)."""
    acc = gltf.accessors[accessor]
    view = gltf.bufferViews[acc.bufferView]
    dtype = {5126: np.float32, 5121: np.uint8, 5123: np.uint16}[acc.componentType]
    width = data.shape[1]
    item = np.dtype(dtype).itemsize * width
    stride = view.byteStride or item
    start = (view.byteOffset or 0) + (acc.byteOffset or 0)
    raw = np.ascontiguousarray(data.astype(dtype))
    if stride == item:
        blob[start:start + raw.nbytes] = raw.tobytes()
    else:
        for i in range(len(raw)):
            blob[start + i * stride:start + i * stride + item] = raw[i].tobytes()


def apply(glb: Path, out: Path) -> dict:
    rig = Rig(glb)
    gltf = rig.gltf
    blob = bytearray(rig.blob)
    report = {}
    for mesh_node in [n for n in gltf.nodes if n.mesh is not None and n.skin is not None]:
        for prim in gltf.meshes[mesh_node.mesh].primitives:
            pos = read(gltf, rig.blob, prim.attributes.POSITION)
            joints = read(gltf, rig.blob, prim.attributes.JOINTS_0).astype(np.int64)
            weights = read(gltf, rig.blob, prim.attributes.WEIGHTS_0)
            w_acc = gltf.accessors[prim.attributes.WEIGHTS_0]
            normalised_int = w_acc.componentType in (5121, 5123)
            if normalised_int:
                weights = weights / (255.0 if w_acc.componentType == 5121 else 65535.0)
            dense, rep = new_weights(rig, pos, joints, weights)
            k = joints.shape[1]
            order = np.argsort(-dense, axis=1)[:, :k]
            new_w = np.take_along_axis(dense, order, axis=1)
            new_w /= np.maximum(new_w.sum(axis=1, keepdims=True), 1e-9)
            if normalised_int:
                top = 255.0 if w_acc.componentType == 5121 else 65535.0
                q = np.round(new_w * top)
                q[:, 0] += top - q.sum(axis=1)          # exact sum after rounding
                new_w = q
            _write(gltf, blob, prim.attributes.JOINTS_0, order)
            _write(gltf, blob, prim.attributes.WEIGHTS_0, new_w)
            report = rep
    gltf.set_binary_blob(bytes(blob))
    out.parent.mkdir(parents=True, exist_ok=True)
    gltf.save(str(out))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slug", required=True)
    parser.add_argument("--size", default="small")
    parser.add_argument("--glb", default=None, help="re-weight this GLB instead of the template")
    parser.add_argument("--out", default=None)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    template = ROOT / "output" / f"skeleton_{args.slug}_{args.size}.glb"
    glb = Path(args.glb) if args.glb else template
    if args.install:
        backup = ROOT / "work" / f"backup_weights_{args.slug}"
        if not backup.exists():
            backup.mkdir(parents=True)
            shutil.copy(template, backup / template.name)
        source, out = backup / template.name, template
    else:
        source, out = glb, Path(args.out) if args.out else ROOT / "work" / f"leg_weights_{args.slug}.glb"
    report = apply(source, out)
    print(f"[leg_weights] {report} -> {out}")


if __name__ == "__main__":
    main()
