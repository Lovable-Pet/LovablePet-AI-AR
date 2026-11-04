"""Real-dog poses for the templates' clips, from motion capture.

WHY
---
pose_clips solved its key poses by hand from the template's bones, and
the lying ones did not look like a dog (user, 11-03: "진짜 개 같지가
않아"): legs stuck out stiff, hind legs splayed like sticks. A real dog's
lying down and lying on its side are full of small bends no rule of thumb
gives. The SIGGRAPH 2018 quadruped capture (Zhang, Starke et al., "Mode-
Adaptive Neural Networks for Quadruped Motion Control"; a medium-large dog,
hips 47 cm off the floor standing, the jindo's 43) has them: lying down,
lying on its side with its head on the floor, standing, as plain BVH.

Licence: the capture is CC BY-NC 4.0 (AI4Animation,
https://github.com/sebastianstarke/AI4Animation) - attribution, no
commercial use.

The poses alone, blended from one to the next, still moved like a
machine (user, 11-03: "걍 이상하잖아"): a dog does not slide from sitting
to lying on its side, it shifts its weight, folds its legs one at a time,
flops. So the clips that change pose are the capture's own stretches of
motion (CLIPS): standing still; lying down, flopping onto its side and
getting up.

HOW
---
Only the poses and stretches the clips use are kept: POSES ->
ar_pipeline/data/dog_poses.json (a few KB) and CLIPS ->
ar_pipeline/data/dog_clips.npz, each as the world rotations of the
capture's joints, turned to face +z. A clip's pieces are joined end to
start, the next turned and moved to carry on from where the last one is
CROSSFADE_S before its end and the two blended over that; the whole is then
turned and moved back a little each frame so it ends where it started,
and its last LOOP_S blended onto its first frame, so it loops in place.
Each paw is marked planted where it is near the floor and still. A pose goes onto a template as each joint's
turn away from the capture's own standing pose (POSES["stand"]) put on the
template's rest pose, which also stands: the capture's legs are angled
(femur 35 deg forward, tibia 40 back) and the templates' nearly straight,
so matching bone directions instead bent the template's legs into the
capture dog's shape and its head on its own axis.

    python -m ar_pipeline.dog_mocap --bvh <dir of the capture's .bvh>   # rewrite the json and npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
POSES_JSON = ROOT / "ar_pipeline" / "data" / "dog_poses.json"
CLIPS_NPZ = ROOT / "ar_pipeline" / "data" / "dog_clips.npz"
CLIP_FPS = 24.0
# clip -> pieces (file, start s, end s), read off the capture's timelines
# (stand / sit / down / on its side by the hips' height, the body's pitch
# and roll): standing still; standing, lying down (by way of a sit),
# flopping onto the left side, and - from another take, lying on the same
# side - getting up again. (A plain lie-down-and-get-up, D1_061 41-54 s,
# came out hunched on the template, with a walk in it, and play_dead holds
# the same lying down.)
CLIPS = {"standing": [("D1_001_KAN01_001.bvh", 0.0, 11.5)],
         "play_dead": [("D1_071_KAN02_002.bvh", 7.5, 11.0), ("D1_071_KAN02_002.bvh", 13.0, 17.5),
                       ("D1_006_KAN01_002.bvh", 11.5, 17.5)]}
CROSSFADE_S = 0.4
LOOP_S = 0.6
FEET = ("LeftHand_end", "RightHand_end", "LeftFoot_end", "RightFoot_end")
# a paw within this of the floor and moving slower than this is planted
PLANT_HEIGHT_M = 0.03
PLANT_SPEED_MS = 0.15
SPIKE_DEG = 25.0
SMOOTH_FRAMES = 1.5
# (file, frame at 60 fps): standing still; lying down with the head up,
# forelegs forward and hind legs folded under (sphinx); lying flat on the
# left side with the head down
POSES = {"stand": ("D1_001_KAN01_001.bvh", 300),
         "down": ("D1_071_KAN02_002.bvh", 798),
         "side": ("D1_006_KAN01_002.bvh", 300)}
# template joint <- capture joint whose turn it takes (the rest of the
# template's joints keep their place relative to their parent)
MAP = {"COG_jnt": "Hips", "spine_1_jnt": "Spine", "spine_2_jnt": "Spine1", "neck_base_jnt": "Neck",
       "head_jnt": "Head"}
for _s, _m in (("L", "Left"), ("R", "Right")):
    MAP.update({f"{_s}_clavicle_jnt": f"{_m}Shoulder", f"{_s}_shoulder_jnt": f"{_m}Arm",
                f"{_s}_elbow_jnt": f"{_m}ForeArm", f"{_s}_wrist_jnt": f"{_m}Hand",
                f"{_s}_hip_jnt": f"{_m}UpLeg", f"{_s}_knee_jnt": f"{_m}Leg", f"{_s}_ankle_jnt": f"{_m}Foot"})


class BVH:
    """A BVH file's hierarchy and channels, with forward kinematics."""

    def __init__(self, path: Path):
        text = Path(path).read_text().split()
        self.names: list[str] = []
        self.parent: list[int] = []
        self.offset: list[np.ndarray] = []
        self.channels: list[list[str]] = []
        i, stack, last = 0, [], -1
        while text[i] != "MOTION":
            tok = text[i]
            if tok in ("ROOT", "JOINT", "End"):
                name = text[i + 1] if tok != "End" else self.names[stack[-1]] + "_end"
                self.names.append(name)
                self.parent.append(stack[-1] if stack else -1)
                self.offset.append(np.zeros(3))
                self.channels.append([])
                last, i = len(self.names) - 1, i + 2
            elif tok == "{":
                stack.append(last)
                i += 1
            elif tok == "}":
                stack.pop()
                i += 1
            elif tok == "OFFSET":
                self.offset[last] = np.array([float(x) for x in text[i + 1:i + 4]])
                i += 4
            elif tok == "CHANNELS":
                n = int(text[i + 1])
                self.channels[last] = text[i + 2:i + 2 + n]
                i += 2 + n
            else:
                i += 1
        frames = int(text[i + 2])
        width = sum(len(c) for c in self.channels)
        start = i + 6                                      # MOTION Frames: n Frame Time: t
        self.data = np.array(text[start:start + width * frames], dtype=float).reshape(frames, width)
        self.index = {n: k for k, n in enumerate(self.names)}

    def world(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        """(positions (J, 3), rotations (J, 3, 3)) at `frame`."""
        row, col = self.data[frame], 0
        pos, rot = np.zeros((len(self.names), 3)), np.zeros((len(self.names), 3, 3))
        for j, channels in enumerate(self.channels):
            t, axes, angles = self.offset[j].copy(), "", []
            for c in channels:
                if c.endswith("position"):
                    t["XYZ".index(c[0])] = row[col]
                else:
                    axes += c[0]
                    angles.append(row[col])
                col += 1
            r = Rotation.from_euler(axes, angles, degrees=True).as_matrix() if axes else np.eye(3)
            p = self.parent[j]
            rot[j] = r if p < 0 else rot[p] @ r
            pos[j] = t if p < 0 else pos[p] + rot[p] @ t
        return pos, rot


    def worlds(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(positions (F, J, 3), rotations (F, J, 3, 3)) at `frames`, in cm."""
        d = self.data[frames]
        F, J = len(d), len(self.names)
        pos, rot = np.zeros((F, J, 3)), np.zeros((F, J, 3, 3))
        col = 0
        for j, channels in enumerate(self.channels):
            t = np.repeat(self.offset[j][None], F, 0)
            axes, angles = "", []
            for c in channels:
                if c.endswith("position"):
                    t[:, "XYZ".index(c[0])] = d[:, col]
                else:
                    axes += c[0]
                    angles.append(d[:, col])
                col += 1
            r = (Rotation.from_euler(axes, np.stack(angles, 1), degrees=True).as_matrix() if axes
                 else np.repeat(np.eye(3)[None], F, 0))
            p = self.parent[j]
            rot[:, j] = r if p < 0 else rot[:, p] @ r
            pos[:, j] = t if p < 0 else pos[:, p] + np.einsum("fij,fj->fi", rot[:, p], t)
        return pos, rot


def _heading(hips: np.ndarray, spine: np.ndarray) -> np.ndarray:
    ahead = spine - hips
    return np.arctan2(ahead[..., 0], ahead[..., 2])


def _yaw(angle) -> np.ndarray:
    return Rotation.from_rotvec(np.outer(np.atleast_1d(angle), [0.0, 1.0, 0.0])).as_matrix()


def _piece(bvh: BVH, floor: float, t0: float, t1: float, joints: list[str]) -> dict:
    """A stretch of a take at CLIP_FPS, in metres above its floor, turned to
    face +z and moved to start at the origin."""
    frames = np.clip(np.round(np.arange(t0, t1, 1.0 / CLIP_FPS) * 60).astype(int), 0, len(bvh.data) - 1)
    pos, rot = bvh.worlds(frames)
    pos = pos * 0.01
    pos[..., 1] -= floor
    I = bvh.index
    face = _yaw(-_heading(pos[0, I["Hips"]], pos[0, I["Spine1"]]))[0]
    origin = pos[0, I["Hips"]] * np.array([1.0, 0.0, 1.0])
    pos = np.einsum("ij,fkj->fki", face, pos - origin)
    rot = np.einsum("ij,fkjl->fkil", face, rot)
    return {"rot": rot[:, [I[j] for j in joints]], "hips": pos[:, I["Hips"]], "spine": pos[:, I["Spine1"]],
            "feet": pos[:, [I[f] for f in FEET]]}


def _moved(piece: dict, turn: float, to: np.ndarray) -> dict:
    """`piece` turned by `turn` about the vertical and moved by `to` (xz)."""
    R = _yaw(turn)[0]
    shift = to * np.array([1.0, 0.0, 1.0])
    out = {k: np.einsum("ij,f...j->f...i", R, v) + shift for k, v in piece.items() if k != "rot"}
    out["rot"] = np.einsum("ij,fkjl->fkil", R, piece["rot"])
    return out


def _blend(a: dict, b: dict, w: np.ndarray) -> dict:
    """Frame by frame, `w` of the way from a to b (rotations by slerp)."""
    out = {k: a[k] + (b[k] - a[k]) * w.reshape((-1,) + (1,) * (a[k].ndim - 1)) for k in a if k != "rot"}
    ra = Rotation.from_matrix(a["rot"].reshape(-1, 3, 3))
    rb = Rotation.from_matrix(b["rot"].reshape(-1, 3, 3))
    ww = np.repeat(w, a["rot"].shape[1])
    delta = (rb * ra.inv()).as_rotvec() * ww[:, None]
    out["rot"] = (Rotation.from_rotvec(delta) * ra).as_matrix().reshape(a["rot"].shape)
    return out


def _smooth(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def clean_rotations(rot: np.ndarray) -> np.ndarray:
    """(F, J, 3, 3): one to three frames that turn a joint SPIKE_DEG or more
    away from the frames either side while those agree replaced by their
    blend (the capture has the odd stretch where a forearm or paw jumps
    50-65 deg and back - a marker swap; a paw does not turn 1400 deg/s),
    then each joint smoothed over SMOOTH_FRAMES."""
    from scipy.ndimage import gaussian_filter1d
    out = rot.copy()
    F, J = rot.shape[:2]
    for j in range(J):
        for span in (1, 2, 3):                         # glitches one to three frames long
            r = Rotation.from_matrix(out[:, j])
            for k in range(1, F - span):
                a, c = r[k - 1], r[k + span]
                if (c * a.inv()).magnitude() > np.radians(SPIKE_DEG):
                    continue
                inner = [r[k + m] for m in range(span)]
                if all((x * a.inv()).magnitude() > np.radians(SPIKE_DEG) for x in inner):
                    for m in range(span):
                        u = (m + 1) / (span + 1)
                        out[k + m, j] = (Rotation.from_rotvec((c * a.inv()).as_rotvec() * u) * a).as_matrix()
        q = Rotation.from_matrix(out[:, j]).as_quat()
        for k in range(1, F):
            if np.dot(q[k], q[k - 1]) < 0:
                q[k] = -q[k]
        q = gaussian_filter1d(q, SMOOTH_FRAMES, axis=0, mode="nearest")
        out[:, j] = Rotation.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True)).as_matrix()
    return out


def extract_clip(bvh_dir: Path, pieces: list, joints: list[str], cache: dict) -> dict:
    """The clip's pieces joined, looped in place, with planted paws marked."""
    out = None
    k = int(round(CROSSFADE_S * CLIP_FPS))
    for file, t0, t1 in pieces:
        if file not in cache:
            b = BVH(bvh_dir / file)
            pos, _ = b.worlds(np.arange(0, len(b.data), 6))
            cache[file] = (b, float(np.percentile(pos[:, [b.index[f] for f in FEET], 1], 2)) * 0.01)
        bvh, floor = cache[file]
        piece = _piece(bvh, floor, t0, t1, joints)
        if out is None:
            out = piece
            continue
        # carry on from where the clip is k frames before its end
        at = len(out["hips"]) - k
        piece = _moved(piece, float(_heading(out["hips"][at], out["spine"][at])), out["hips"][at])
        w = _smooth((np.arange(k) + 1) / (k + 1))
        tail = {key: v[at:] for key, v in out.items()}
        head = {key: v[:k] for key, v in piece.items()}
        mixed = _blend(tail, head, w)
        out = {key: np.concatenate([out[key][:at], mixed[key], piece[key][k:]]) for key in out}
    # back to where it started, a little each frame the dog moves (by how far
    # its hips go: spread evenly over the frames, the turn and the shift went
    # on while it lay still, and lying "dead" it slid 25 cm and turned)
    n = len(out["hips"])
    step = np.r_[0.0, np.linalg.norm(np.diff(out["hips"][:, [0, 2]], axis=0), axis=1)]
    u = np.cumsum(step) / step.sum() if step.sum() > 1e-9 else np.arange(n) / (n - 1)
    turn = float(_heading(out["hips"][-1], out["spine"][-1]))
    turns = _yaw(-turn * u)
    end = _yaw(-turn)[0] @ (out["hips"][-1] * np.array([1.0, 0.0, 1.0]))
    for key in ("hips", "spine"):
        out[key] = np.einsum("fij,fj->fi", turns, out[key]) - u[:, None] * end
    out["feet"] = np.einsum("fij,fkj->fki", turns, out["feet"]) - u[:, None, None] * end
    out["rot"] = np.einsum("fij,fkjl->fkil", turns, out["rot"])
    # the last LOOP_S onto the first frame
    m = int(round(LOOP_S * CLIP_FPS))
    first = {key: np.repeat(v[:1], m, 0) for key, v in out.items()}
    last = {key: v[-m:] for key, v in out.items()}
    mixed = _blend(last, first, _smooth((np.arange(m) + 1) / m))
    out = {key: np.concatenate([out[key][:-m], mixed[key]]) for key in out}
    out["rot"] = clean_rotations(out["rot"])
    speed = np.linalg.norm(np.gradient(out["feet"][..., [0, 2]], axis=0), axis=-1) * CLIP_FPS
    out["planted"] = (out["feet"][..., 1] < PLANT_HEIGHT_M) & (speed < PLANT_SPEED_MS)
    return out


def extract(bvh_dir: Path) -> dict:
    """POSES as {name: {capture joint: world rotation (row-major 9)}}, each
    turned about the vertical to face +z (by its lumbar, Hips -> Spine1)."""
    out = {}
    for name, (file, frame) in POSES.items():
        b = BVH(bvh_dir / file)
        pos, rot = b.world(frame)
        ahead = pos[b.index["Spine1"]] - pos[b.index["Hips"]]
        face = Rotation.from_rotvec([0.0, -np.arctan2(ahead[0], ahead[2]), 0.0]).as_matrix()
        out[name] = {j: [round(float(v), 6) for v in (face @ rot[b.index[j]]).ravel()]
                     for j in sorted(set(MAP.values()))}
    return out


def extract_clips(bvh_dir: Path) -> dict[str, np.ndarray]:
    """CLIPS, flattened for an npz: <clip>/rot as quaternions (F, J, 4),
    <clip>/hips, <clip>/feet (metres, y above the floor), <clip>/planted."""
    joints = sorted(set(MAP.values()))
    cache: dict = {}
    out: dict[str, np.ndarray] = {"joints": np.array(joints)}
    stand_file, stand_frame = POSES["stand"]
    b = BVH(bvh_dir / stand_file)
    pos, _ = b.worlds(np.array([stand_frame]))
    floor = float(np.min(pos[0, [b.index[f] for f in FEET], 1]))
    out["stand_hips_height"] = np.array((pos[0, b.index["Hips"], 1] - floor) * 0.01)
    for name, pieces in CLIPS.items():
        clip = extract_clip(bvh_dir, pieces, joints, cache)
        shape = clip["rot"].shape
        out[f"{name}/rot"] = Rotation.from_matrix(clip["rot"].reshape(-1, 3, 3)).as_quat().reshape(
            shape[0], shape[1], 4).astype(np.float32)
        out[f"{name}/hips"] = clip["hips"].astype(np.float32)
        out[f"{name}/feet"] = clip["feet"].astype(np.float32)
        out[f"{name}/planted"] = clip["planted"]
        print(f"[dog_mocap] {name}: {shape[0]} frames ({shape[0] / CLIP_FPS:.1f} s), "
              f"paws planted {clip['planted'].mean() * 100:.0f} %")
    return out


def load_clips() -> dict:
    """{clip: {"rot": {capture joint: (F, 3, 3)}, "hips", "feet", "planted"}, "stand_hips_height": m}"""
    data = np.load(CLIPS_NPZ)
    joints = [str(j) for j in data["joints"]]
    out: dict = {"stand_hips_height": float(data["stand_hips_height"])}
    for name in CLIPS:
        if f"{name}/rot" not in data:
            continue
        q = data[f"{name}/rot"]
        mats = Rotation.from_quat(q.reshape(-1, 4)).as_matrix().reshape(q.shape[0], q.shape[1], 3, 3)
        out[name] = {"rot": {j: mats[:, k] for k, j in enumerate(joints)}, "hips": data[f"{name}/hips"],
                     "feet": data[f"{name}/feet"], "planted": data[f"{name}/planted"]}
    return out


def load() -> dict[str, dict[str, np.ndarray]]:
    data = json.loads(POSES_JSON.read_text(encoding="utf-8"))
    return {name: {j: np.array(v).reshape(3, 3) for j, v in pose.items()} for name, pose in data["poses"].items()}


def turns(poses: dict, name: str) -> dict[str, np.ndarray]:
    """Each mapped template joint's world turn away from standing in pose `name`."""
    stand, pose = poses["stand"], poses[name]
    return {ours: pose[theirs] @ stand[theirs].T for ours, theirs in MAP.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bvh", required=True, help="directory of the capture's .bvh files (MotionCapture.zip)")
    args = parser.parse_args()
    POSES_JSON.parent.mkdir(parents=True, exist_ok=True)
    POSES_JSON.write_text(json.dumps({
        "source": "AI4Animation SIGGRAPH 2018 quadruped motion capture (Zhang, Starke, Komura, Saito: "
                  "Mode-Adaptive Neural Networks for Quadruped Motion Control), "
                  "https://github.com/sebastianstarke/AI4Animation",
        "license": "CC BY-NC 4.0",
        "frames": {k: list(v) for k, v in POSES.items()},
        "poses": extract(Path(args.bvh))}, indent=1), encoding="utf-8")
    print(f"[dog_mocap] wrote {POSES_JSON}")
    np.savez_compressed(CLIPS_NPZ, **extract_clips(Path(args.bvh)))
    print(f"[dog_mocap] wrote {CLIPS_NPZ} ({CLIPS_NPZ.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
