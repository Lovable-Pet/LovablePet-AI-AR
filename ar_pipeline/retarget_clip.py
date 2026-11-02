"""Copy an animation clip from one rigged dog onto a template's skeleton.

WHY
---
The templates carry two clips made by script (build_template.py: idle,
tail_wag). The retired shiba source model (archive/shiba, "baby dog")
came with five hand-animated ones - standing, sitting, shake, rollover,
play_dead - and auto_rig.py gave every template the same joint names as
that rig (the shiba's carry a ".NN" suffix; 48 of its 121 joints), so its
motion can be carried over by name.

HOW
---
Pure numbers on the glTF, no re-export (a Blender round trip would touch
the templates' morph targets, split primitives and fixed normals):

  1. sample the source clip at --fps, and evaluate both skeletons' world
     transforms by forward kinematics through the node tree, at rest (the
     nodes' own TRS, which is the bind pose on both) and posed;
  2. each joint's motion as a WORLD rotation away from its rest
     (D = R_world(t) R_world_rest^T): independent of how each rig's bone
     axes are rolled;
  3. the template joint of the same name turned by the same D from its own
     rest, its local rotation taken against its parent's posed world
     rotation (parents first); a joint the source lacks keeps its rest
     local rotation;
  4. the body's translation (the topmost joint whose world position the
     clip moves) carried over as the source's world displacement scaled by
     the hip-height ratio;
  5. written into the template GLB as a new animation (LINEAR samplers,
     every joint's rotation, the moving joint's translation), the mesh,
     skin and other clips untouched.

    python -m ar_pipeline.retarget_clip --clip sitting --slug jindo --out work/anim/jindo_sitting.glb
    python -m ar_pipeline.retarget_clip --clip sitting --slug jindo --install   # into the template GLB

--install backs the GLB up to work/backup_anim_<slug>/ first (once), and
replaces a clip of the same name.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from pygltflib import GLTF2, Accessor, Animation, AnimationChannel, AnimationChannelTarget, AnimationSampler, \
    BufferView
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "archive" / "shiba" / "output" / "skeleton_shiba_small.glb"
COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
DTYPES = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}
# bones whose direction follows the source's absolute direction (a chain bone
# with one child). None now: copied onto the jindo's legs, whose rest
# stance differs from the puppy's, they crossed its front legs and splayed
# its stance even standing still (user, 11-03: "다리가 요염한 자세") - the
# change from rest (the world delta) keeps the template's own stance, and
# the ground pass takes care of a leg through the floor
CHAIN_PREFIXES: tuple[str, ...] = ()
# GROUND CONTACT. The source is a short-legged puppy: given its joint
# angles, the jindo's rump sat 31 cm under the floor and its legs went
# through it lying down; lifted until the rump touched, its front paws hung
# 18 cm in the air. Each frame (ground_pass): what the source has on the
# floor - its torso (vertices whose dominant joint starts with TORSO, under
# CONTACT_M, easing off by CONTACT_FADE_M) and each paw (its toe joint
# within TOE_CONTACT_M of its rest height) - is put on the floor on the
# template too: the body raised or lowered to it, the body pitched about
# the rump's contact (within PITCH_LIMIT_DEG) when the front paws must reach
# the floor too, and each planted paw, or one under the floor, brought to
# its rest height by CCD on its leg (IK_ITERATIONS passes, IK_STEP_DEG a
# joint a pass); last, no paw nearer the midline than LATERAL_KEEP of its
# rest stance (the puppy's inward-angled legs crossed on the jindo's).
TORSO = ("pelvis", "spine", "chest", "breathe", "COG", "central")
CONTACT_M = 0.03
CONTACT_FADE_M = 0.08
TOE_CONTACT_M = 0.025
PITCH_LIMIT_DEG = 35.0
IK_ITERATIONS = 12
IK_STEP_DEG = 25.0
# the share of a front leg's slack (its straight length less its reach) the
# pitch leaves for the leg to close by straightening
REACH_SHARE = 0.85
# a paw comes no nearer the midline than this share of its rest stance
LATERAL_KEEP = 0.6
# the share of a front leg's straight length it stands at when the dog sits
SIT_LEG_SHARE = 0.96
LEGS = {"LB": ("L_hip_jnt", "L_knee_jnt", "L_ankle_jnt", "L_foot_jnt", "LB_toes_jnt"),
        "RB": ("R_hip_jnt", "R_knee_jnt", "R_ankle_jnt", "R_foot_jnt", "RB_toes_jnt"),
        "LF": ("L_shoulder_jnt", "L_elbow_jnt", "L_wrist_jnt", "L_hand_jnt", "LF_toes_jnt"),
        "RF": ("R_shoulder_jnt", "R_elbow_jnt", "R_wrist_jnt", "R_hand_jnt", "RF_toes_jnt")}


def read(gltf: GLTF2, blob: bytes, index: int) -> np.ndarray:
    acc = gltf.accessors[index]
    view = gltf.bufferViews[acc.bufferView]
    dtype, width = DTYPES[acc.componentType], COMPONENTS[acc.type]
    item = np.dtype(dtype).itemsize * width
    stride = view.byteStride or item
    start = (view.byteOffset or 0) + (acc.byteOffset or 0)
    if stride == item:
        return np.frombuffer(blob, dtype, acc.count * width, start).reshape(acc.count, width).astype(np.float64)
    return np.stack([np.frombuffer(blob, dtype, width, start + i * stride) for i in range(acc.count)]).astype(np.float64)


class Rig:
    """A glTF's node tree, its skin's joints by bare name, and forward
    kinematics with some nodes' local TRS overridden."""

    def __init__(self, path: Path):
        self.gltf = GLTF2().load(str(path))
        self.blob = self.gltf.binary_blob()
        nodes = self.gltf.nodes
        self.parent = {c: i for i, n in enumerate(nodes) for c in (n.children or [])}
        self.joints = list(self.gltf.skins[0].joints)
        self.by_name = {(nodes[j].name or "").split(".")[0]: j for j in self.joints}
        self.rest_trs = {i: self._trs(i) for i in range(len(nodes))}
        # parents before children
        depth = {i: self._depth(i) for i in range(len(nodes))}
        self.order = sorted(range(len(nodes)), key=depth.get)

    def _depth(self, i: int) -> int:
        d = 0
        while i in self.parent:
            i, d = self.parent[i], d + 1
        return d

    def _trs(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.gltf.nodes[i]
        if n.matrix:
            m = np.array(n.matrix, dtype=np.float64).reshape(4, 4).T
            scale = np.linalg.norm(m[:3, :3], axis=0)
            return m[:3, 3].copy(), Rotation.from_matrix(m[:3, :3] / scale).as_quat(), scale
        return (np.array(n.translation or [0, 0, 0], dtype=np.float64),
                np.array(n.rotation or [0, 0, 0, 1], dtype=np.float64),
                np.array(n.scale or [1, 1, 1], dtype=np.float64))

    def lowest(self, world: dict[int, np.ndarray], step: int = 6, torso: bool = True) -> float:
        """The lowest point (glTF y) of the skinned mesh in pose `world`, over
        every `step`-th vertex of its largest skinned primitive - of the
        torso's vertices (dominant joint in TORSO) by default."""
        if not hasattr(self, "_skin"):
            gltf, blob = self.gltf, self.blob
            prim = max((p for n in gltf.nodes if n.mesh is not None and n.skin is not None
                        for p in gltf.meshes[n.mesh].primitives),
                       key=lambda p: gltf.accessors[p.attributes.POSITION].count)
            pos = read(gltf, blob, prim.attributes.POSITION)[::step]
            joints = read(gltf, blob, prim.attributes.JOINTS_0)[::step].astype(np.int64)
            weights = read(gltf, blob, prim.attributes.WEIGHTS_0)[::step]
            if weights.max() > 1.5:
                weights /= 255.0 if weights.max() <= 255 else 65535.0
            weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-9)
            ibm = read(gltf, blob, gltf.skins[0].inverseBindMatrices).reshape(-1, 4, 4).transpose(0, 2, 1)
            names = [(gltf.nodes[j].name or "").split(".")[0] for j in self.joints]
            dominant = joints[np.arange(len(joints)), weights.argmax(axis=1)]
            in_torso = np.array([names[d].startswith(TORSO) for d in dominant])
            self._skin = (np.c_[pos, np.ones(len(pos))], joints, weights, ibm, in_torso)
        verts, joints, weights, ibm, in_torso = self._skin
        mats = np.stack([world[j] @ ibm[k] for k, j in enumerate(self.joints)])
        y = np.zeros(len(verts))
        for k in range(joints.shape[1]):
            y += weights[:, k] * np.einsum("nj,nj->n", mats[joints[:, k], 1, :], verts)
        return float((y[in_torso] if torso and in_torso.any() else y).min())

    def body_low(self, world: dict[int, np.ndarray]) -> tuple[float, np.ndarray]:
        """(height, position) of the lowest torso vertex in pose `world`."""
        self.lowest(world)
        verts, joints, weights, ibm, in_torso = self._skin
        mats = np.stack([world[j] @ ibm[k] for k, j in enumerate(self.joints)])
        v = verts[in_torso]
        p = np.zeros((len(v), 4))
        for k in range(joints.shape[1]):
            p += weights[in_torso, k, None] * np.einsum("nij,nj->ni", mats[joints[in_torso, k]], v)
        low = int(np.argmin(p[:, 1]))
        return float(p[low, 1]), p[low, :3]

    def world(self, overrides: dict[int, tuple] | None = None) -> dict[int, np.ndarray]:
        """Every node's world matrix, with `overrides` {node: (t, q, s)}."""
        out: dict[int, np.ndarray] = {}
        for i in self.order:
            t, q, s = (overrides or {}).get(i, self.rest_trs[i])
            m = np.eye(4)
            m[:3, :3] = Rotation.from_quat(q).as_matrix() * s
            m[:3, 3] = t
            out[i] = out[self.parent[i]] @ m if i in self.parent else m
        return out


def _swing(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotation taking direction `a` onto `b`."""
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    axis = np.cross(a, b)
    sin, cos = np.linalg.norm(axis), float(np.dot(a, b))
    if sin < 1e-9:
        return np.eye(3) if cos > 0 else Rotation.from_rotvec(np.pi * _perpendicular(a)).as_matrix()
    return Rotation.from_rotvec(axis / sin * np.arctan2(sin, cos)).as_matrix()


def _perpendicular(a: np.ndarray) -> np.ndarray:
    other = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v = np.cross(a, other)
    return v / np.linalg.norm(v)


def _rot(m: np.ndarray) -> np.ndarray:
    """The rotation of a 4x4 (scale divided out)."""
    r = m[:3, :3]
    return r / np.linalg.norm(r, axis=0)


def sample_clip(rig: Rig, name: str, fps: float) -> tuple[np.ndarray, list[dict[int, tuple]]]:
    """(times, per frame {node: (t, q, s)}) of clip `name` at `fps`."""
    anims = [a for a in rig.gltf.animations or [] if a.name == name]
    if not anims:
        raise SystemExit(f"no clip {name!r}; have {[a.name for a in rig.gltf.animations or []]}")
    anim = anims[0]
    end = max(float(read(rig.gltf, rig.blob, s.input).max()) for s in anim.samplers)
    times = np.arange(0.0, end + 1e-6, 1.0 / fps)
    frames = [{} for _ in times]
    for channel in anim.channels:
        sampler = anim.samplers[channel.sampler]
        node, path = channel.target.node, channel.target.path
        if path not in ("translation", "rotation", "scale"):
            continue
        keys = read(rig.gltf, rig.blob, sampler.input).ravel()
        values = read(rig.gltf, rig.blob, sampler.output)
        if sampler.interpolation == "CUBICSPLINE":
            values = values.reshape(len(keys), 3, -1)[:, 1]      # the value, without the tangents
        t = np.clip(times, keys[0], keys[-1])
        if path == "rotation":
            if len(keys) == 1:
                sampled = np.repeat(values[:1], len(t), axis=0)
            elif sampler.interpolation == "STEP":
                sampled = values[np.clip(np.searchsorted(keys, t, side="right") - 1, 0, len(keys) - 1)]
            else:
                sampled = Slerp(keys, Rotation.from_quat(values))(t).as_quat()
        else:
            if sampler.interpolation == "STEP":
                sampled = values[np.clip(np.searchsorted(keys, t, side="right") - 1, 0, len(keys) - 1)]
            else:
                sampled = np.stack([np.interp(t, keys, values[:, k]) for k in range(values.shape[1])], axis=1)
        slot = {"translation": 0, "rotation": 1, "scale": 2}[path]
        for f, value in enumerate(sampled):
            trs = list(frames[f].get(node, rig.rest_trs[node]))
            trs[slot] = value
            frames[f][node] = tuple(trs)
    return times, frames


def _hip_height(rig: Rig, world: dict[int, np.ndarray]) -> float:
    hips = [world[rig.by_name[n]][:3, 3] for n in ("L_hip_jnt", "R_hip_jnt") if n in rig.by_name]
    toes = [world[rig.by_name[n]][:3, 3] for n in ("LB_toes_jnt", "RB_toes_jnt") if n in rig.by_name]
    up = 1  # glTF y-up
    return float(np.mean([h[up] for h in hips]) - np.mean([t[up] for t in toes]))


def retarget(source: Rig, target: Rig, clip: str, fps: float) -> dict:
    times, frames = sample_clip(source, clip, fps)
    src_rest = source.world()
    tgt_rest = target.world()
    scale = _hip_height(target, tgt_rest) / max(_hip_height(source, src_rest), 1e-6)
    # the joint that carries the body's travel: the topmost target joint
    # whose source counterpart the clip moves
    moving = None
    for i in target.order:
        if i not in target.joints:
            continue
        name = (target.gltf.nodes[i].name or "").split(".")[0]
        j = source.by_name.get(name)
        if j is not None and any(j in f and np.linalg.norm(f[j][0] - source.rest_trs[j][0]) > 1e-6 for f in frames):
            moving = (i, j)
            break
    # each chain bone's one joint child of the same name on both rigs, for
    # matching its direction (CHAIN_PREFIXES)
    child_of: dict[int, tuple[int, int]] = {}
    for i in target.joints:
        name = (target.gltf.nodes[i].name or "").split(".")[0]
        j = source.by_name.get(name)
        kids = [c for c in target.gltf.nodes[i].children or [] if c in target.joints]
        if j is None or len(kids) != 1 or not CHAIN_PREFIXES or not name.startswith(CHAIN_PREFIXES):
            continue
        kid_name = (target.gltf.nodes[kids[0]].name or "").split(".")[0]
        if kid_name in source.by_name:
            child_of[i] = (kids[0], source.by_name[kid_name])
    rotations = {i: [] for i in target.joints}
    translations = []
    for frame in frames:
        src_world = source.world(frame)
        posed_rot: dict[int, np.ndarray] = {}
        out_q: dict[int, np.ndarray] = {}
        for i in target.order:
            parent = target.parent.get(i)
            parent_rot = posed_rot.get(parent, _rot(tgt_rest[parent])) if parent is not None else np.eye(3)
            rest_local = Rotation.from_quat(target.rest_trs[i][1]).as_matrix()
            name = (target.gltf.nodes[i].name or "").split(".")[0]
            j = source.by_name.get(name) if i in target.joints else None
            if j is None:
                world_rot = parent_rot @ rest_local
            else:
                delta = _rot(src_world[j]) @ _rot(src_rest[j]).T
                world_rot = delta @ _rot(tgt_rest[i])
                if i in child_of:
                    # the bone pointed where the source's points: the rest
                    # legs of the two rigs stand at different angles, and the
                    # delta alone swung the jindo's legs down through the
                    # floor when the puppy's lay flat
                    c, jc = child_of[i]
                    want = src_world[jc][:3, 3] - src_world[j][:3, 3]
                    have = delta @ (tgt_rest[c][:3, 3] - tgt_rest[i][:3, 3])
                    world_rot = _swing(have, want) @ world_rot
            posed_rot[i] = world_rot
            if i in target.joints:
                local = parent_rot.T @ world_rot
                out_q[i] = Rotation.from_matrix(local).as_quat()
        for i in target.joints:
            rotations[i].append(out_q[i])
        if moving is not None:
            i, j = moving
            shift_world = (src_world[j][:3, 3] - src_rest[j][:3, 3]) * scale
            parent = target.parent.get(i)
            parent_world = tgt_rest[parent] if parent is not None else np.eye(4)
            # the parent is not animated (the moving joint is the topmost): its rest frame
            shift_local = np.linalg.inv(parent_world[:3, :3]) @ shift_world
            translations.append(target.rest_trs[i][0] + shift_local)
    # keep quaternions on one hemisphere frame to frame (no flips in LINEAR)
    for i, qs in rotations.items():
        for k in range(1, len(qs)):
            if np.dot(qs[k], qs[k - 1]) < 0:
                qs[k] = -qs[k]
    ground = []
    if moving is not None:
        ground = ground_pass(source, target, frames, rotations, translations, moving[0], scale)
    return {"times": times, "rotations": rotations, "moving": moving[0] if moving else None,
            "translations": translations, "scale": scale, "ground": ground}


def ground_pass(source: Rig, target: Rig, frames: list[dict[int, tuple]], rotations: dict[int, list],
                translations: list[np.ndarray], mover: int, scale: float) -> list[float]:
    """Put on the floor what the source has there (see GROUND CONTACT), in
    place on `rotations` and `translations`; returns each frame's lift."""
    src_rest, tgt_rest = source.world(), target.world()
    parent = target.parent.get(mover)
    parent_world = tgt_rest[parent] if parent is not None else np.eye(4)
    to_local = np.linalg.inv(parent_world[:3, :3])
    parent_rot = _rot(parent_world)
    legs = {leg: [target.by_name[n] for n in names] for leg, names in LEGS.items()
            if all(n in target.by_name for n in names) and names[-1] in source.by_name}
    toe_rest = {leg: tgt_rest[chain[-1]][1, 3] for leg, chain in legs.items()}
    toe_x = {leg: tgt_rest[chain[-1]][0, 3] for leg, chain in legs.items()}
    src_toe_rest = {leg: src_rest[source.by_name[LEGS[leg][-1]]][1, 3] for leg in legs}
    leg_length = {leg: sum(np.linalg.norm(tgt_rest[b][:3, 3] - tgt_rest[a][:3, 3]) for a, b in zip(chain, chain[1:]))
                  for leg, chain in legs.items()}
    lifts = []
    for k, frame in enumerate(frames):
        q = {i: np.asarray(rotations[i][k], dtype=np.float64) for i in target.joints}
        state = {"t": np.asarray(translations[k], dtype=np.float64)}

        def pose() -> dict[int, np.ndarray]:
            over = {i: (target.rest_trs[i][0], q[i], target.rest_trs[i][2]) for i in target.joints}
            over[mover] = (state["t"], q[mover], target.rest_trs[mover][2])
            return target.world(over)

        src_world = source.world(frame)
        src_body, _ = source.body_low(src_world)
        planted = {leg: src_world[source.by_name[LEGS[leg][-1]]][1, 3] - src_toe_rest[leg] < TOE_CONTACT_M
                   for leg in legs}
        on_floor = float(np.clip((CONTACT_FADE_M - src_body) / (CONTACT_FADE_M - CONTACT_M), 0.0, 1.0))
        world = pose()
        # 1. the height: the torso onto the source's, or the lowest planted paw onto the floor
        body_y, _ = target.body_low(world)
        feet = [world[legs[leg][-1]][1, 3] - toe_rest[leg] for leg in legs if planted[leg]]
        foot_lift = -min(feet) if feet else 0.0
        lift = on_floor * (src_body * scale - body_y) + (1.0 - on_floor) * foot_lift
        state["t"] = state["t"] + to_local @ np.array([0.0, lift, 0.0])
        world = pose()
        # 2. the pitch about the rump's contact, for front paws that must reach the floor
        front = [leg for leg in ("LF", "RF") if leg in legs and planted[leg]]
        sit_front = front if on_floor > 0.5 else []
        if on_floor > 0.5 and front:
            _, pivot = target.body_low(world)
            cog_rot, cog_pos = _rot(world[mover]), world[mover][:3, 3].copy()
            keep_q, keep_t = q[mover].copy(), state["t"].copy()

            def set_pitch(angle: float) -> None:
                turn = Rotation.from_rotvec([np.radians(angle), 0.0, 0.0]).as_matrix()
                q[mover] = Rotation.from_matrix(parent_rot.T @ turn @ cog_rot).as_quat()
                state["t"] = to_local @ (pivot + turn @ (cog_pos - pivot) - parent_world[:3, 3])

            def front_gap(angle: float) -> float:
                # the shoulders at the height a near-straight front leg
                # reaches (SIT_LEG_SHARE of its length) above its rest toe:
                # the jindo's torso is 1.9 times the puppy's against a front
                # leg 1.2 times, so pitched up as far as the puppy's the
                # paws hung 19 cm in the air; pitched down until the paws
                # touched where they were, it crouched on legs slanting back
                # (user, 11-03: "자연스럽게 안 앉고")
                set_pitch(angle)
                w = pose()
                return float(np.mean([w[legs[leg][0]][1, 3] - SIT_LEG_SHARE * leg_length[leg] - toe_rest[leg]
                                      for leg in front]))

            angles = np.linspace(-PITCH_LIMIT_DEG, PITCH_LIMIT_DEG, 29)
            gaps = np.array([front_gap(a) for a in angles])
            best = float(angles[np.argmin(np.abs(gaps) + 0.002 * np.abs(angles))])
            q[mover], state["t"] = keep_q, keep_t
            if abs(best) > 1e-6:
                set_pitch(best)
            world = pose()
        def reach(chain: list[int], goal: np.ndarray) -> dict[int, np.ndarray]:
            """CCD of the chain's toe onto `goal`; returns the pose."""
            toe = chain[-1]
            w = pose()
            for _ in range(IK_ITERATIONS):
                for joint in reversed(chain[:-1]):
                    w = pose()
                    here = w[joint][:3, 3]
                    vec = Rotation.from_matrix(_swing(w[toe][:3, 3] - here, goal - here)).as_rotvec()
                    angle = float(np.linalg.norm(vec))
                    if angle > np.radians(IK_STEP_DEG):
                        vec = vec * np.radians(IK_STEP_DEG) / angle
                    up = target.parent[joint]
                    q[joint] = Rotation.from_matrix(
                        _rot(w[up]).T @ Rotation.from_rotvec(vec).as_matrix() @ _rot(w[joint])).as_quat()
                w = pose()
                if np.linalg.norm(w[toe][:3, 3] - goal) < 0.003:
                    break
            return w

        # 3. each planted paw, and any under the floor, to its rest height (CCD)
        for leg, chain in legs.items():
            toe = chain[-1]
            if not planted[leg] and world[toe][1, 3] >= toe_rest[leg] - 0.005:
                continue
            goal = world[toe][:3, 3].copy()
            goal[1] = toe_rest[leg]
            if leg in sit_front:
                # a sitting dog's front legs stand straight down from the shoulders
                goal[2] = world[chain[0]][2, 3]
            world = reach(chain, goal)
        # 4. no paw nearer the midline than LATERAL_KEEP of its rest stance:
        # the puppy's legs angled in, on the jindo's longer ones, crossed
        for leg, chain in legs.items():
            toe = chain[-1]
            side = np.sign(toe_x[leg])
            keep = LATERAL_KEEP * abs(toe_x[leg])
            if world[toe][0, 3] * side < keep:
                goal = world[toe][:3, 3].copy()
                goal[0] = side * keep
                world = reach(chain, goal)
        for i in target.joints:
            rotations[i][k] = q[i]
        translations[k] = state["t"]
        lifts.append(lift)
    return lifts


def _add_accessor(gltf: GLTF2, blob: bytearray, data: np.ndarray, kind: str, with_bounds: bool = False) -> int:
    while len(blob) % 4:
        blob.append(0)
    raw = np.ascontiguousarray(data, dtype=np.float32).tobytes()
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=len(blob), byteLength=len(raw)))
    blob.extend(raw)
    acc = Accessor(bufferView=len(gltf.bufferViews) - 1, componentType=5126, count=len(data), type=kind)
    if with_bounds:
        acc.min = [float(v) for v in np.atleast_1d(data.min(axis=0))]
        acc.max = [float(v) for v in np.atleast_1d(data.max(axis=0))]
    gltf.accessors.append(acc)
    return len(gltf.accessors) - 1


def write_clip(target: Rig, result: dict, name: str, out: Path) -> None:
    gltf = target.gltf
    gltf.animations = [a for a in gltf.animations or [] if a.name != name]
    blob = bytearray(target.blob)
    times = np.asarray(result["times"], dtype=np.float32)
    time_acc = _add_accessor(gltf, blob, times, "SCALAR", with_bounds=True)
    anim = Animation(name=name, samplers=[], channels=[])
    for node, quats in result["rotations"].items():
        out_acc = _add_accessor(gltf, blob, np.asarray(quats, dtype=np.float32), "VEC4")
        anim.samplers.append(AnimationSampler(input=time_acc, output=out_acc, interpolation="LINEAR"))
        anim.channels.append(AnimationChannel(sampler=len(anim.samplers) - 1,
                                              target=AnimationChannelTarget(node=node, path="rotation")))
    if result["moving"] is not None:
        out_acc = _add_accessor(gltf, blob, np.asarray(result["translations"], dtype=np.float32), "VEC3")
        anim.samplers.append(AnimationSampler(input=time_acc, output=out_acc, interpolation="LINEAR"))
        anim.channels.append(AnimationChannel(sampler=len(anim.samplers) - 1,
                                              target=AnimationChannelTarget(node=result["moving"],
                                                                            path="translation")))
    gltf.animations.append(anim)
    gltf.buffers[0].byteLength = len(blob)
    gltf.set_binary_blob(bytes(blob))
    out.parent.mkdir(parents=True, exist_ok=True)
    gltf.save(str(out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clip", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--size", default="small")
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--name", default=None, help="the clip's name in the template (default: --clip)")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--glb", default=None, help="retarget onto this GLB (a built dog) instead of the template")
    args = parser.parse_args()
    glb = Path(args.glb) if args.glb else ROOT / "output" / f"skeleton_{args.slug}_{args.size}.glb"
    source, target = Rig(Path(args.source)), Rig(glb)
    missing = [n for n in ((target.gltf.nodes[j].name or "").split(".")[0] for j in target.joints)
               if n not in source.by_name]
    result = retarget(source, target, args.clip, args.fps)
    print(f"[retarget] {args.clip}: {len(result['times'])} frames at {args.fps:g} fps, "
          f"{len(result['rotations'])} joints ({len(missing)} not in the source: {missing}), "
          f"travel on {target.gltf.nodes[result['moving']].name if result['moving'] is not None else None}, "
          f"scale {result['scale']:.3f}")
    if result["ground"]:
        lift = np.asarray(result["ground"]) * 100
        print(f"[retarget] ground contact: body moved {lift.min():+.1f} .. {lift.max():+.1f} cm "
              f"(median {np.median(lift):+.1f})")
    name = args.name or args.clip
    if args.install:
        backup = ROOT / "work" / f"backup_anim_{args.slug}"
        if not backup.exists():
            backup.mkdir(parents=True)
            shutil.copy(glb, backup / glb.name)
        out = glb
    else:
        out = Path(args.out) if args.out else ROOT / "work" / "anim" / f"{args.slug}_{name}.glb"
    write_clip(target, result, name, out)
    print(f"[retarget] wrote {out}")


if __name__ == "__main__":
    main()
