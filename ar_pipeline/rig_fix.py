"""Re-rig a template so it bends like a dog: the joints where the mesh
bends, and each part of the skin following the bones along its own body
chain.

WHY
---
auto_rig put the joints at fixed shares of the height and weighed every
vertex by a Gaussian of its distance to every bone. That held while the
templates only stood; with clips (pose_clips, 11-03) it showed everywhere:

  * legs: the jindo's hip sat on its back line (0.53 m) and its knee in the
    middle of the thigh (0.38; the stifle is at the chest floor, 0.30), and
    the thigh's outer face went half to the pelvis - folded to sit, the
    haunch crumpled into a hollow shell and the thigh melted into the
    belly ("팔이 섞이고 웨이트가 잘못된 느낌");
  * neck and head: the neck joints ran up the back of the neck, at its
    surface, and the head turned about a point at the top of the skull
    behind the eyes; the lower face and throat went partly to the neck
    joints. Turned, the face sheared and a ring of creases tore round the
    neck; nodded, the face flattened into the neck (user, 11-03: "고개를
    돌릴 때마다 얼굴이 찌부", "얼굴이 목에 살짝 파묻히긴 해");
  * torso and tail: weights by distance to whichever bone was nearest, so
    a twist of the chest broke along a seam at the withers and the body
    crumpled when the dog lay on its side ("죽은 척도 왜 몸이 자꾸
    찌그러지는"), and the tail's root strands stayed with the rump as the
    tail swung.

The old shiba (archive/shiba, an artist's rig) does none of this: its head
turns as one piece on a neck that blends smoothly, its body twists evenly
from hips to shoulders.

HOW
---
Joints (move_joints) - only positions change; the nodes' translations, the
inverse bind matrices and any clip's translation keys on the moved joints
and their children follow, so the rest pose is the same mesh:
  * legs: the knee and elbow to the chest floor, the hock and wrist to
    their shares of it (refine_masks.leg_levels, measured on the mesh), the
    hip and shoulder HIP_SHARE / SHOULDER_SHARE of the way from there up to
    where auto_rig had them; a hind leg's knee and hock also to the middle
    of the leg's cross-section at their height (auto_rig ran them down the
    back of the thigh);
  * neck and head (neck_positions): cross-sections of the neck taken along
    its old axis, the joints at their centres - neck_base at the old
    base's section, the head at the first section that is mostly head (the
    junction of skull and neck, where a dog's head pivots), neck_mid half
    way.

Weights (new_weights):
  * each leg: its own bone chain, the torso beside it, a narrow seam
    between - membership by distance to the leg's chain (LEG_RADII)
    against distance to the torso's core line (CORE_RADIUS) through a
    logistic, the nearest leg taking nearly all, nothing above the hip /
    shoulder joint (a hind leg's seam with the belly and the rump much
    wider, HIND_MEMBERSHIP_SLOPE and HIND_TOP_FADE_M: sitting, the thigh
    turns some 65 deg against the belly); within a foreleg a Gaussian over
    its bones, within a hind leg each bone its own stretch of height,
    blended at the joints (HIND_BLEND_M);
  * the body (torso, neck, head): the chain pelvis - COG - spine - chest -
    neck - head, each vertex shared between the two joints either side of
    its nearest point on the chain, by how far along it lies; everything
    beyond the head joint's cross-section (the whole skull and face, with a
    HEAD_BLEND_M seam) on the head alone; the breathing bone keeps its old
    share (the idle clip scales it);
  * the tail: the chain pelvis - tail_1 ... tail_7 the same way;
  * the ears keep their weight on the ear bones (ear_flaps reads them),
    the rest of it on the head.

    python -m ar_pipeline.rig_fix --slug jindo             # report, write work/rig_fix_<slug>.glb
    python -m ar_pipeline.rig_fix --slug jindo --install   # into the template GLB

--install backs the GLB up to work/backup_weights_<slug>/ first (once) and
always works from that backup (auto_rig's joints and weights). Positions,
UVs and morph targets are not touched; rebuild the clips after it
(pose_clips --install).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from pygltflib import GLTF2, AnimationSampler

from ar_pipeline.gltf_rig import Rig, append_accessor, read
from ar_pipeline.refine_masks import leg_levels

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
LEG_RADII = {"B": (0.10, 0.06, 0.045, 0.04), "F": (0.055, 0.05, 0.04, 0.035)}
CORE_RADIUS = 0.12
# above the hip (shoulder) joint the skin is the torso's, fading over this (m)
TOP_FADE_M = 0.04
MEMBERSHIP_SLOPE = 10.0
LEG_SHARE_SLOPE = 25.0
CHAIN_RADIUS = 0.05
MAX_INFLUENCES = 4
# the hip / shoulder joint this share of the way from the knee / elbow level
# up to where auto_rig put it (on the back line): inside the haunch and the
# point of the shoulder (jindo: hip 0.53 -> 0.43 m, shoulder 0.53 -> 0.44)
HIP_SHARE = 0.55
SHOULDER_SHARE = 0.6
# a hind leg's knee and hock at the middle of the leg's cross-section at
# their height, taken this thick (auto_rig ran the hind leg's bones straight
# down the back of the thigh: the jindo's knee 1.5 cm inside the leg's back
# and 13 cm behind its front, and folded to sit, the front of the stifle
# went half with the shin and swung out sideways like a frog's thigh -
# user, 11-03: "뒷다리도 매우 어색하고")
LEG_SECTION_HALF_M = 0.008
# the cross-section is taken within 15 cm of the joint front to back, then
# again round the middle it found, up to this many times: round the
# retriever's old knee, 4.6 cm behind its thigh, the window cut off the
# front of the thigh and the knee came out 5.4 cm behind the middle
LEG_CENTRE_PASSES = 4
# within a hind leg each bone takes the skin between its joint's height and
# the next one's, blended over HIND_BLEND_M either side of a line
# HIND_SHIFT_M below the joint, so the outside of the fold (the front of the
# stifle, the point of the hock) stays rounded with the bone above. (A
# Gaussian over the bones' distances, CHAIN_RADIUS, gave the thigh's surface
# 10-50 % of the shin, and the sitting haunch crumpled.)
HIND_BLEND_M = (0.055, 0.03, 0.02)             # knee, hock, paw
HIND_SHIFT_M = (0.03, 0.015, 0.0)
# a sitting dog's thigh turns some 65 deg against its belly (its knee comes
# up by the flank): the hind leg's skin fades into the torso's over a wide
# band - at the hip over HIND_TOP_FADE_M, toward the belly with
# HIND_MEMBERSHIP_SLOPE in place of MEMBERSHIP_SLOPE. With the narrow seams
# of the forelegs the fold of skin from the stifle to the belly stood up
# over the knee as a fin, or tore along the seam; much wider (slope 1.6),
# the belly sagged in front of the thigh.
HIND_TOP_FADE_M = 0.10
HIND_MEMBERSHIP_SLOPE = 2.5
# a foreleg likewise (elbow, wrist, paw; None keeps a Gaussian over its
# bones), its seam with the breast and the chest wall at
# FRONT_MEMBERSHIP_SLOPE, and FRONT_TOP_FADE_M at the shoulder: with the
# upper arm reaching 9 cm (LEG_RADII) and a narrow seam it carried 70 % of
# the breast beside it and 83 % of the chest wall behind it, and a paw
# raised to shake took the breast with it (user, 11-04: "앞다리랑 가슴이랑
# 붙어 움직이는 느낌"; at 7 cm and slope 3, still half, "가슴이 너무
# 끌려가는 느낌"; now some 37 %)
FRONT_BLEND_M: tuple | None = (0.035, 0.02, 0.02)
FRONT_SHIFT_M = (0.02, 0.01, 0.0)
FRONT_TOP_FADE_M = 0.08
FRONT_MEMBERSHIP_SLOPE = 2.5
# the body's chain, rump to head, and the tail's; a node where two joints
# sit (COG and spine_1, both at the COG) is shared between them on both
# sides of it (given to one behind it and the other in front, the skin
# tore along it when only spine_1 turned)
BODY_CHAIN = (("pelvis_jnt",), ("COG_jnt", "spine_1_jnt"), ("spine_2_jnt",), ("chest_jnt",),
              ("neck_base_jnt",), ("neck_mid_jnt",), ("head_jnt",))
TAIL_CHAIN = ("pelvis_jnt",) + tuple(f"tail_{k}_jnt" for k in range(1, 8))
HEAD_PARTS = ("head_jnt", "jaw_jnt", "L_eye_jnt", "R_eye_jnt", "nose_jnt")
EAR_PARTS = ("L_ear_base_jnt", "L_ear_tip_jnt", "R_ear_base_jnt", "R_ear_tip_jnt")
NECK_PARTS = ("neck_base_jnt", "neck_mid_jnt")
BREATHE = "breathe_jnt"
# beyond the head joint's cross-section the skin is the head's alone,
# blended over this either side of it
HEAD_BLEND_M = 0.015
# the seam between the tail's skin and the body's, by their distances to the
# two chains
TAIL_SEAM_M = 0.01
# a hanging tail's free length (free_tail): the mesh cut level from this far
# above the tail's root joint down, in steps of FREE_TAIL_STEP_M
FREE_TAIL_ABOVE_M = 0.05
FREE_TAIL_STEP_M = 0.005
# ... and its root, fused into the croup above that, only as wide as the tail
# (tail_column): by the distances to the chains alone tail_1 took the whole
# top of the retriever's rump, 24 cm across for a 7 cm tail, and the croup
# turned with the tail - so its wag had to start at the free length, half
# way down the buttocks (user, 11-08: "꼬리흔들어지는 시작점이 맞지 않는
# 느낌임. 좀더 위여야할 것 같아"). The width is the free tail's within
# TAIL_ROOT_TOP_M of its top; the skin beside it goes to the body, blended
# over TAIL_ROOT_BLEND_M
TAIL_ROOT_TOP_M = 0.03
TAIL_ROOT_BLEND_M = 0.008
# ... and the tube above the free length the tail's whole (tail_tube): the
# retriever's tail is a tube of its own up to 0.48 m and a notch in the
# croup's outline above, but the front half of the tube, a centimetre or two
# behind the pelvis joint, went with the rump, and the tail swung with its
# front face left on the buttocks (user, 11-08: "꼬리 이거 연결부위 보면
# 좀더 위까지 꼬리로하고 엉덩이랑 떨어지게 해도될것같은데"). The tube's front
# face is told from the rump just in front of it by facing: forward (normal
# along +z past TAIL_TUBE_FACING) against back
TAIL_TUBE_MARGIN_M = 0.005
TAIL_TUBE_BELOW_M = 0.01
TAIL_TUBE_FACING = 0.3
# the rump's skin within TAIL_SOFT_M of the tail's (free length and tube,
# behind the hips) goes partly with it, up to TAIL_SOFT_SHARE right at it:
# one surface with the tail's front there, held to the pelvis it stretched
# into a web a triangle wide as the tail lifted from its root
TAIL_SOFT_M = 0.025
TAIL_SOFT_SHARE = 0.8
# a hanging tail's share of the skin by distance fades out from this far in
# front of its root joint to this far (m): none of the back
TAIL_ROOT_FRONT_M = (0.01, 0.03)
# neck cross-sections: this thick, within this of the old neck axis
SECTION_HALF_M = 0.01
SECTION_RADIUS_M = 0.18


def _segments(points: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(zip(points, points[1:]))


def _distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def polyline_distance(pos: np.ndarray, points: list[np.ndarray]) -> np.ndarray:
    return np.min([_distance(pos, a, b) for a, b in zip(points, points[1:])], axis=0)


def chain_weights(pos: np.ndarray, points: list[np.ndarray], owners: list[tuple[int, ...]], n_joints: int) -> np.ndarray:
    """Each vertex shared between the two nodes either side of its nearest
    point on the polyline `points`, by how far along that segment it lies
    (clamped at the ends); node k's share split evenly among owners[k]."""
    best_d = np.full(len(pos), np.inf)
    seg = np.zeros(len(pos), dtype=int)
    frac = np.zeros(len(pos))
    for k, (a, b) in enumerate(zip(points, points[1:])):
        ab = b - a
        t = np.clip(((pos - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        d = np.linalg.norm(pos - (a + t[:, None] * ab), axis=1)
        closer = d < best_d
        best_d[closer], seg[closer], frac[closer] = d[closer], k, t[closer]
    node = np.zeros((len(points), n_joints))
    for k, joints in enumerate(owners):
        node[k, list(joints)] = 1.0 / len(joints)
    return node[seg] * (1.0 - frac)[:, None] + node[seg + 1] * frac[:, None]


def hind_shares(pos: np.ndarray, at: dict[str, np.ndarray], chain: tuple[str, ...],
                blends: tuple | None = None, shifts: tuple | None = None) -> np.ndarray:
    """(N, 4) shares of a leg's four bones (a hind leg's femur, shin,
    metatarsus and paw) by height (see HIND_BLEND_M)."""
    downs = []
    for joint, shift, blend in zip(chain[1:4], shifts or HIND_SHIFT_M, blends or HIND_BLEND_M):
        u = np.clip((at[joint][1] - shift - pos[:, 1]) / (2 * blend) + 0.5, 0.0, 1.0)
        downs.append(u * u * (3 - 2 * u))
    knee, hock, paw = downs
    return np.stack([1.0 - knee, knee - hock, hock - paw, paw], axis=1)


def hanging_tail(at: dict[str, np.ndarray]) -> list[np.ndarray]:
    """The tail's joints from the last one above the hips down: the part of
    a tail hanging beside the hind legs (the retriever's, to 12 cm off the
    floor). [] for a tail carried over the back (the jindo's), so its rig
    is as before."""
    tail = [at[n] for n in TAIL_CHAIN[1:] if n in at]
    hips = [at[n][1] for n in ("L_hip_jnt", "R_hip_jnt") if n in at]
    if len(tail) < 2 or not hips:
        return []
    below = [k for k, p in enumerate(tail) if p[1] < min(hips)]
    if not below:
        return []
    return tail[max(below[0] - 1, 0):]


def free_tail(pos: np.ndarray, tri: np.ndarray | None, at: dict[str, np.ndarray]) -> np.ndarray:
    """The skin of a hanging tail's free length: the mesh cut level at
    heights from just above the tail's root down (FREE_TAIL_STEP_M), the
    piece holding the tail's tip at the first height where no hind paw is
    in it. The retriever's tail is a tube hanging free of the thighs below
    0.45 m, its front face 1-5 cm behind them; that face was nearer the
    legs' chains than the tail's and went with the legs, so the tail swung
    with its front glued to the thighs (user, 11-08: "꼬리흔드는것도
    엉덩이에 붙어서 흔들리는 느낌"). All False without a hanging tail, the
    triangles, or such a height."""
    none = np.zeros(len(pos), dtype=bool)
    tail = hanging_tail(at)
    paws = [at[n] for n in ("LB_toes_jnt", "RB_toes_jnt") if n in at]
    if len(tail) < 2 or tri is None or not paws:
        return none
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    # one vertex per position: UV seams split them
    _, weld = np.unique(np.round(pos / 1e-5).astype(np.int64), axis=0, return_inverse=True)
    weld = weld.ravel()
    n = int(weld.max()) + 1
    welded = np.zeros((n, 3))
    welded[weld] = pos
    edges = weld[np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])]
    tip = int(np.argmin(np.linalg.norm(welded - tail[-1], axis=1)))
    paw_ids = [int(np.argmin(np.linalg.norm(welded - p, axis=1))) for p in paws]
    for y in np.arange(tail[0][1] + FREE_TAIL_ABOVE_M, tail[-1][1], -FREE_TAIL_STEP_M):
        below = welded[:, 1] < y
        if not below[tip]:
            break
        keep = below[edges[:, 0]] & below[edges[:, 1]]
        graph = coo_matrix((np.ones(int(keep.sum())), (edges[keep, 0], edges[keep, 1])), shape=(n, n))
        _, label = connected_components(graph, directed=False)
        if all(label[p] != label[tip] for p in paw_ids):
            return (label == label[tip])[weld]
    return none


def vertex_normals(pos: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals of the triangles."""
    face = np.cross(pos[tri[:, 1]] - pos[tri[:, 0]], pos[tri[:, 2]] - pos[tri[:, 0]])
    n = np.zeros_like(pos)
    for c in range(3):
        np.add.at(n, tri[:, c], face)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def tail_tube(pos: np.ndarray, tri: np.ndarray | None, at: dict[str, np.ndarray], free: np.ndarray) -> np.ndarray:
    """The rest of a hanging tail's tube above its free length (`free`,
    free_tail), up to its root (see TAIL_TUBE_FACING): within the free
    tail's width of its midline and its depth of the tube's middle - the
    chain carried along by the free top's offset from it - (each and
    TAIL_TUBE_MARGIN_M), each vertex behind that middle or facing forward,
    the tube's front face toward the body (the rump in front of it faces
    back). (Without the depth, the breast's midline, facing forward too,
    went to the tail.) All False without a free tail."""
    none = np.zeros(len(pos), dtype=bool)
    if tri is None or not free.any():
        return none
    q = pos[free]
    mid, top = float(np.median(q[:, 0])), float(q[:, 1].max())
    ring = q[q[:, 1] > top - TAIL_ROOT_TOP_M]
    half = float(np.percentile(np.abs(ring[:, 0] - mid), 95))
    chain = np.array([at[n] for n in TAIL_CHAIN[1:] if n in at])
    order = np.argsort(chain[:, 1])
    chain_z = lambda y: np.interp(y, chain[order, 1], chain[order, 2])
    middle = chain_z(pos[:, 1]) + float(np.median(ring[:, 2]) - chain_z(top - TAIL_ROOT_TOP_M / 2))
    reach = float(np.percentile(np.abs(ring[:, 2] - np.median(ring[:, 2])), 95)) + TAIL_TUBE_MARGIN_M
    inside = ((np.abs(pos[:, 0] - mid) < half + TAIL_TUBE_MARGIN_M) & (pos[:, 1] > top - TAIL_TUBE_BELOW_M)
              & (pos[:, 1] < chain[:, 1].max()) & (np.abs(pos[:, 2] - middle) < reach) & ~free)
    return inside & ((pos[:, 2] < middle) | (vertex_normals(pos, tri)[:, 2] > TAIL_TUBE_FACING))


def tail_column(pos: np.ndarray, own_tail: np.ndarray) -> np.ndarray:
    """(N,) how far each vertex lies within a hanging tail's own width of
    its midline (see TAIL_ROOT_TOP_M): 1 inside, 0 beyond, blended; all 1
    without a free tail (`own_tail`, free_tail)."""
    if not own_tail.any():
        return np.ones(len(pos))
    q = pos[own_tail]
    mid = float(np.median(q[:, 0]))
    top = q[q[:, 1] > q[:, 1].max() - TAIL_ROOT_TOP_M]
    half = float(np.percentile(np.abs(top[:, 0] - mid), 95))
    return 1.0 / (1.0 + np.exp(-(half - np.abs(pos[:, 0] - mid)) / TAIL_ROOT_BLEND_M))


def tail_skin(pos: np.ndarray, at: dict[str, np.ndarray], leg_chain: tuple[str, ...]) -> np.ndarray:
    """The skin nearer a hanging tail's chain than the leg's (all False
    without one)."""
    tail = hanging_tail(at)
    leg = [at[n] for n in leg_chain if n in at]
    if len(tail) < 2 or len(leg) < 2:
        return np.zeros(len(pos), dtype=bool)
    return polyline_distance(pos, tail) < polyline_distance(pos, leg)


def leg_centre(pos: np.ndarray, at: dict[str, np.ndarray], joint: str, y: float,
               exclude: np.ndarray | None = None) -> tuple[float, float] | None:
    """(x, z) of the middle of the leg's cross-section at height `y` round
    `joint` (its side of the body, within 15 cm of it front to back),
    leaving out the `exclude` skin (the tail's)."""
    side = np.sign(at[joint][0])
    around, found = at[joint][2], None
    for _ in range(LEG_CENTRE_PASSES):
        hit = (side * pos[:, 0] > 0.015) & (np.abs(pos[:, 2] - around) < 0.15) & \
            (np.abs(pos[:, 1] - y) < LEG_SECTION_HALF_M)
        if exclude is not None:
            hit &= ~exclude
        if hit.sum() < 10:
            return found
        q = pos[hit]
        centre = (float(q[:, 0].min() + q[:, 0].max()) / 2, float(q[:, 2].min() + q[:, 2].max()) / 2)
        if found is not None and abs(centre[1] - found[1]) < 1e-4:
            break
        found, around = centre, centre[1]
    return found


def body_weights(names: list[str], index: dict[str, int], at: dict[str, np.ndarray], pos: np.ndarray,
                 dominant_name: np.ndarray, old: np.ndarray, own_tail: np.ndarray | None = None,
                 tube: np.ndarray | None = None) -> np.ndarray:
    """The torso, neck, head, tail and ears' weights (see HOW), dense;
    `own_tail` (free_tail) and `tube` (tail_tube) the tail's whatever the
    distances say."""
    n_joints = len(names)
    body = [node for node in BODY_CHAIN if all(n in index for n in node)]
    weights = chain_weights(pos, [at[node[0]] for node in body], [tuple(index[n] for n in node) for node in body],
                            n_joints)
    # beyond the head's cross-section: the head alone
    if "head_jnt" in at and "neck_mid_jnt" in at:
        normal = at["head_jnt"] - at["neck_mid_jnt"]
        normal = normal / np.linalg.norm(normal)
        beyond = (pos - at["head_jnt"]) @ normal
        share = np.clip((beyond + HEAD_BLEND_M) / (2 * HEAD_BLEND_M), 0.0, 1.0)
        share = share * share * (3 - 2 * share)
        head = np.zeros(n_joints)
        head[index["head_jnt"]] = 1.0
        weights = weights * (1.0 - share[:, None]) + share[:, None] * head
    # the tail along its own chain, wherever the skin is nearer the tail's
    # chain than the body's (the strands hanging under the tail's root were
    # auto_rig's pelvis and hips, and stayed with the rump as the tail swung)
    tail_names = [n for n in TAIL_CHAIN if n in index]
    tail = np.zeros(len(pos), dtype=bool)
    if len(tail_names) > 2:
        tail_pts = [at[n] for n in tail_names]
        d_tail = polyline_distance(pos, tail_pts[1:])
        d_body = polyline_distance(pos, [at[node[0]] for node in body])
        share = 1.0 / (1.0 + np.exp(-(d_body - d_tail) / TAIL_SEAM_M))
        whole = None
        if own_tail is not None:
            whole = own_tail | tube if tube is not None else own_tail
            share = share * tail_column(pos, own_tail)
            if own_tail.any():
                # (and none of the back in front of the tail's root: up on the
                # croup the back's skin lies as far from the body's chain as
                # from the tail's, and a strip of it 17 cm long went half to
                # tail_1 and swayed with the wag)
                root_z = at[tail_names[1]][2]
                share = share * np.clip((root_z + TAIL_ROOT_FRONT_M[1] - pos[:, 2])
                                        / (TAIL_ROOT_FRONT_M[1] - TAIL_ROOT_FRONT_M[0]), 0.0, 1.0)
            share = np.where(whole, 1.0, share)
            if whole.any() and TAIL_SOFT_M > 0:
                # (see TAIL_SOFT_M)
                from scipy.spatial import cKDTree
                # (only round its top: lower down the free length hangs
                # 1-5 cm behind the thighs, and they would swing with it)
                top = float(pos[own_tail, 1].max()) if own_tail.any() else float(pos[whole, 1].min())
                # (the rump's skin by the tail's front face lies up to 2 cm
                # in front of the hip joints: kept to behind them, it stayed
                # the pelvis's and tore 1 cm into 8 sitting)
                behind = (~whole) & (pos[:, 2] < max(at[n][2] for n in ("L_hip_jnt", "R_hip_jnt") if n in at)
                                     + TAIL_SOFT_M * 2)
                near = behind & (pos[:, 1] > top - TAIL_SOFT_M)
                if near.any():
                    d = cKDTree(pos[whole]).query(pos[near])[0]
                    u = np.clip(1.0 - d / TAIL_SOFT_M, 0.0, 1.0)
                    soft = np.zeros(len(pos))
                    soft[near] = TAIL_SOFT_SHARE * u * u * (3 - 2 * u)
                    share = np.maximum(share, soft)
                    whole = whole | (soft > 0)
        tail = share > 0.5
        along_tail = chain_weights(pos, tail_pts, [(index[n],) for n in tail_names], n_joints)
        if whole is not None and whole.any():
            # the tail's own skin (and the rump's by it, TAIL_SOFT_M) on the
            # tail's joints alone: along the chain from the pelvis, the front
            # face of the retriever's tail - the joints run down its back,
            # 6-7 cm behind it - lay as near the pelvis joint, just in front
            # of it, and 608 of its vertices went wholly to the pelvis: the
            # tail swung with its front face left on the rump, torn 1 cm into
            # 11 sitting (user, 11-08: "꼬리부분 여전히 문제있네")
            own = chain_weights(pos, tail_pts[1:], [(index[n],) for n in tail_names[1:]], n_joints)
            along_tail = np.where(whole[:, None], own, along_tail)
        weights = weights * (1.0 - share[:, None]) + along_tail * share[:, None]
    # the ears keep their ear bones' share, the rest on the head
    ears = np.isin(dominant_name, EAR_PARTS)
    if ears.any() and "head_jnt" in index:
        ear_cols = [index[n] for n in EAR_PARTS if n in index]
        kept = np.zeros((int(ears.sum()), n_joints))
        kept[:, ear_cols] = old[ears][:, ear_cols]
        kept[:, index["head_jnt"]] += 1.0 - kept.sum(axis=1)
        weights[ears] = kept
    # the breathing bone keeps its old share
    if BREATHE in index:
        b = old[:, index[BREATHE]].copy()
        b[tail | ears] = 0.0
        weights *= (1.0 - b)[:, None]
        weights[:, index[BREATHE]] += b
    return weights


def new_weights(rig: Rig, pos: np.ndarray, joints: np.ndarray, weights: np.ndarray,
                tri: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
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
    body_bones = [index[n] for n in names if n in TORSO_BONES or n in NECK_PARTS or n in HEAD_PARTS
                  or n in EAR_PARTS or n.startswith("tail_")]
    editable = np.isin(dominant, list(leg_bones) + body_bones)
    dominant_name = np.array(names)[dominant]

    core_pts = [at[n] for n in CORE if n in at]
    d_core = np.min([_distance(pos, a, b) for a, b in _segments(core_pts)], axis=0)
    out = old.copy()
    report = {}
    legs = [leg for leg, chain in CHAINS.items() if all(n in at for n in chain)]
    seg_ds, d_norm, tops = {}, {}, {}
    for leg in legs:
        pts = [at[n] for n in CHAINS[leg]]
        seg_ds[leg] = np.stack([_distance(pos, a, b) for a, b in _segments(pts)], axis=1)
        d_norm[leg] = (seg_ds[leg] / np.array(LEG_RADII[leg[1]])).min(axis=1)
        fade = HIND_TOP_FADE_M if leg[1] == "B" else FRONT_TOP_FADE_M
        tops[leg] = np.clip((pts[0][1] - pos[:, 1]) / fade + 0.5, 0.0, 1.0)
    # the legs compete among themselves first - the nearest takes nearly all
    # (LEG_SHARE_SLOPE): judged each against the torso alone, a lower
    # foreleg far from the core line went a tenth to the other foreleg too,
    # and the raised paw of "shake" pulled a ghost of the other leg up with it
    stack = np.stack([d_norm[leg] for leg in legs], axis=1)
    nearest = stack.min(axis=1)
    share = np.exp(-LEG_SHARE_SLOPE * (stack - nearest[:, None]))
    share /= share.sum(axis=1, keepdims=True)
    slope = np.where(np.array([leg[1] == "B" for leg in legs])[stack.argmin(axis=1)],
                     HIND_MEMBERSHIP_SLOPE, FRONT_MEMBERSHIP_SLOPE)
    to_legs = 1.0 / (1.0 + np.exp(-slope * (d_core / CORE_RADIUS - nearest)))
    # ... and none of a tail hanging between the hocks: where the tail's
    # chain is nearer than the leg's, the skin goes on to the body's and the
    # tail's chains (body_weights). The retriever's tail, down to 12 cm off
    # the floor, was half its hind legs' (some 8,000 vertices each against
    # the jindo's 1,450) and would have folded with the thighs
    tail_pts = hanging_tail(at)
    if len(tail_pts) > 1:
        d_tail = polyline_distance(pos, tail_pts)
        d_leg = np.min([seg_ds[leg].min(axis=1) for leg in legs], axis=0)
        to_legs = to_legs / (1.0 + np.exp(-(d_tail - d_leg) / TAIL_SEAM_M))
    # and none of its free length or the tube above it, which go to the
    # tail's chain whole
    own_tail = free_tail(pos, tri, at)
    tube = tail_tube(pos, tri, at, own_tail)
    to_legs = np.where(own_tail | tube, 0.0, to_legs)
    member_total = np.zeros(len(pos))
    leg_dense = np.zeros_like(old)
    for k_leg, leg in enumerate(legs):
        member = to_legs * share[:, k_leg] * tops[leg]
        member_total += member
        if leg[1] == "B":
            g = hind_shares(pos, at, CHAINS[leg])
        elif FRONT_BLEND_M is not None:
            g = hind_shares(pos, at, CHAINS[leg], FRONT_BLEND_M, FRONT_SHIFT_M)
        else:
            g = np.exp(-(seg_ds[leg] / CHAIN_RADIUS) ** 2)
            g /= np.maximum(g.sum(axis=1, keepdims=True), 1e-12)
        dense = np.zeros_like(old)
        for k, n in enumerate(CHAINS[leg][:-1]):
            dense[:, index[n]] = g[:, k]
        leg_dense += member[:, None] * dense
        report[leg] = int((editable & (member > 0.5)).sum())
    member_total = np.minimum(member_total, 1.0)
    # the rest of the skin along the body's and the tail's chains
    body = body_weights(names, index, at, pos, dominant_name, old, own_tail, tube)
    mixed = leg_dense + (1.0 - member_total)[:, None] * body
    out[editable] = mixed[editable]
    # the strongest MAX_INFLUENCES, normalised
    order = np.argsort(-out, axis=1)[:, :MAX_INFLUENCES]
    kept = np.zeros_like(out)
    rows = np.arange(len(pos))[:, None]
    kept[rows, order] = out[rows, order]
    kept /= np.maximum(kept.sum(axis=1, keepdims=True), 1e-9)
    report["edited"] = int(editable.sum())
    if own_tail.any():
        report["free_tail"] = int(own_tail.sum())
        report["tail_tube"] = int(tube.sum())
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


def neck_positions(pos: np.ndarray, dominant: np.ndarray, at: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """neck_base, neck_mid and head at the centres of the neck's cross-
    sections (see HOW); {} if the rig has no such joints."""
    if not all(n in at for n in ("neck_base_jnt", "neck_mid_jnt", "head_jnt")):
        return {}
    axis = at["head_jnt"] - at["neck_base_jnt"]
    axis = axis / np.linalg.norm(axis)
    body = ~np.isin(dominant, [n for c in CHAINS.values() for n in c] + list(EAR_PARTS)) & \
        ~np.char.startswith(dominant.astype(str), "tail_")
    pts, kind = pos[body], dominant[body]
    span = float((at["head_jnt"] - at["neck_base_jnt"]) @ axis)

    def section(t: float) -> tuple[np.ndarray, float] | None:
        origin = at["neck_base_jnt"] + axis * t
        rel = pts - origin
        along = rel @ axis
        across = np.linalg.norm(rel - np.outer(along, axis), axis=1)
        hit = (np.abs(along) < SECTION_HALF_M) & (across < SECTION_RADIUS_M)
        if hit.sum() < 20:
            return None
        lo, hi = pts[hit].min(axis=0), pts[hit].max(axis=0)
        centre = (lo + hi) / 2
        centre[0] = at["neck_base_jnt"][0]
        return centre, float(np.isin(kind[hit], HEAD_PARTS).mean())

    base = section(0.0)
    head_t = None
    for t in np.arange(0.0, 2.0 * span, 0.005):
        s = section(float(t))
        if s is not None and s[1] >= 0.5:
            head_t = float(t)
            break
    if base is None or head_t is None:
        return {}
    head = section(head_t)[0]
    mid = section(head_t / 2)
    return {"neck_base_jnt": base[0], "neck_mid_jnt": mid[0] if mid else (base[0] + head) / 2, "head_jnt": head}


def move_joints(rig: Rig, blob: bytearray) -> dict[str, tuple]:
    """The leg, neck and head joints to where the mesh bends (see HOW), in
    place on `rig` (nodes, inverse bind matrices, clips' translation keys)
    and `blob`; returns {joint: (old position, new position)}."""
    gltf = rig.gltf
    prim = max((p for n in gltf.nodes if n.mesh is not None and n.skin is not None
                for p in gltf.meshes[n.mesh].primitives),
               key=lambda p: gltf.accessors[p.attributes.POSITION].count)
    pos = read(gltf, rig.blob, prim.attributes.POSITION)
    joints = read(gltf, rig.blob, prim.attributes.JOINTS_0).astype(np.int64)
    weights = read(gltf, rig.blob, prim.attributes.WEIGHTS_0)
    tri = read(gltf, rig.blob, prim.indices).reshape(-1, 3).astype(np.int64) if prim.indices is not None else None
    names = [(gltf.nodes[j].name or "").split(".")[0] for j in rig.joints]
    dominant = np.array(names)[joints[np.arange(len(joints)), weights.argmax(axis=1)]]
    world = rig.world()
    at = {n: world[i][:3, 3].copy() for n, i in rig.by_name.items()}
    hanging_free = free_tail(pos, tri, at)
    z_up = lambda p: np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)   # leg_levels works in Blender's axes
    levels = leg_levels(z_up(pos), np.isin(dominant, TORSO_BONES), {n: z_up(p) for n, p in at.items()})
    new = {n: p.copy() for n, p in at.items()}
    for side in "LR":
        hip, knee, ankle = f"{side}_hip_jnt", f"{side}_knee_jnt", f"{side}_ankle_jnt"
        shoulder, elbow, wrist = f"{side}_shoulder_jnt", f"{side}_elbow_jnt", f"{side}_wrist_jnt"
        if not all(n in at for n in (hip, knee, ankle, shoulder, elbow, wrist)):
            continue
        new[knee][1], new[ankle][1] = levels["knee"], levels["ankle"]
        new[elbow][1], new[wrist][1] = levels["elbow"], levels["wrist"]
        new[hip][1] = levels["knee"] + HIP_SHARE * (at[hip][1] - levels["knee"])
        new[shoulder][1] = levels["elbow"] + SHOULDER_SHARE * (at[shoulder][1] - levels["elbow"])
        # a tail hanging behind the leg is not in its cross-section: with
        # it the retriever's knee went to the hip's line and its hock 6 cm
        # behind the leg
        tail = tail_skin(pos, at, CHAINS[f"{side}B"]) | hanging_free
        for joint in (knee, ankle):
            centre = leg_centre(pos, at, joint, new[joint][1], exclude=tail)
            if centre is not None:
                new[joint][0], new[joint][2] = centre
    new.update(neck_positions(pos, dominant, at))
    moved = {n for n in new if np.linalg.norm(new[n] - at[n]) > 1e-7}
    # each moved joint's and each moved joint's child's local translation
    # (world rotations unchanged)
    shifts = {}
    for name, i in rig.by_name.items():
        parent = rig.parent.get(i)
        pname = (gltf.nodes[parent].name or "").split(".")[0] if parent is not None else None
        if name not in moved and pname not in moved:
            continue
        pw = world[parent].copy() if parent is not None else np.eye(4)
        if pname in new:
            pw[:3, 3] = new[pname]
        local = (np.linalg.inv(pw) @ np.r_[new[name], 1.0])[:3]
        shifts[i] = local - rig.rest_trs[i][0]
        node = gltf.nodes[i]
        if node.matrix:
            m = np.array(node.matrix, dtype=np.float64).reshape(4, 4).T
            m[:3, 3] = local
            node.matrix = [float(v) for v in m.T.ravel()]
        else:
            node.translation = [float(v) for v in local]
    rig.refresh()
    world = rig.world()
    ibm = np.stack([np.linalg.inv(world[j]).T.ravel() for j in rig.joints])
    _write(gltf, blob, gltf.skins[0].inverseBindMatrices, ibm)
    # a clip keying a moved node's translation keeps its motion about the new
    # place - on keys of its own (the exporter shares one constant key array
    # among joints and clips)
    for anim in gltf.animations or []:
        for channel in anim.channels:
            if channel.target.path == "translation" and channel.target.node in shifts:
                sampler = anim.samplers[channel.sampler]
                values = read(gltf, rig.blob, sampler.output) + shifts[channel.target.node]
                anim.samplers.append(AnimationSampler(input=sampler.input, interpolation=sampler.interpolation,
                                                      output=append_accessor(gltf, blob, values, "VEC3")))
                channel.sampler = len(anim.samplers) - 1
    return {n: ([round(float(v), 3) for v in at[n]], [round(float(v), 3) for v in new[n]]) for n in sorted(moved)}


def apply(glb: Path, out: Path) -> dict:
    rig = Rig(glb)
    gltf = rig.gltf
    blob = bytearray(rig.blob)
    report = {"joints": move_joints(rig, blob)}
    for mesh_node in [n for n in gltf.nodes if n.mesh is not None and n.skin is not None]:
        for prim in gltf.meshes[mesh_node.mesh].primitives:
            pos = read(gltf, rig.blob, prim.attributes.POSITION)
            joints = read(gltf, rig.blob, prim.attributes.JOINTS_0).astype(np.int64)
            weights = read(gltf, rig.blob, prim.attributes.WEIGHTS_0)
            w_acc = gltf.accessors[prim.attributes.WEIGHTS_0]
            normalised_int = w_acc.componentType in (5121, 5123)
            if normalised_int:
                weights = weights / (255.0 if w_acc.componentType == 5121 else 65535.0)
            tri = read(gltf, rig.blob, prim.indices).reshape(-1, 3).astype(np.int64) if prim.indices is not None else None
            dense, rep = new_weights(rig, pos, joints, weights, tri)
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
            report.update(rep)
    gltf.buffers[0].byteLength = len(blob)
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
        source, out = glb, Path(args.out) if args.out else ROOT / "work" / f"rig_fix_{args.slug}.glb"
    report = apply(source, out)
    joints = report.pop("joints")
    for name, (old, new) in joints.items():
        print(f"[rig_fix] {name:16s} {old} -> {new}")
    print(f"[rig_fix] {report} -> {out}")


if __name__ == "__main__":
    main()
