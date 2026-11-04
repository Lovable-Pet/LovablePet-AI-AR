"""Dog motion read off ordinary video, for the clips no capture has.

WHY
---
The real-dog capture (dog_mocap) has standing, sitting and lying, but no
tricks: no paw given, no roll. Built by hand the shake looked like no dog
(user, 11-03: "진짜 개 같지가 않아"). MoCapAnything (Gong et al., CVPR 2026,
https://github.com/animotionlab26/MocapAnything, MIT) reads a 3D motion
for a given rigged skeleton out of one video; run (locally, in WSL) on a
free stock clip of a dog doing the trick with the Truebones "Dog" skeleton
as the target, it gives that skeleton's joint rotations frame by frame.

Videos used (Pexels licence: free to use, no attribution needed):
  shake - https://www.pexels.com/video/a-woman-shaking-her-dog-s-paw-6825840/
          (a shiba sitting side-on gives its right paw), cropped to the dog

HOW
---
Only the stretch a clip needs is kept (VIDEO_CLIPS ->
ar_pipeline/data/video_clips.npz): the world rotations of the joints
pose_clips maps, turned to face +z, cleaned of the odd glitch and smoothed
(dog_mocap.clean_rotations), at the clips' 24 fps, and its last LOOP_S
blended onto its first frame so it loops. pose_clips puts each joint's turn
since the stretch's first frame on top of the template's own key pose (the
sit, for the shake): the trick's motion, the template's own pose.

    python -m ar_pipeline.video_motion --clip shake --bvh Dog_Dog_rot6d_pred.bvh
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ar_pipeline.dog_mocap import BVH, clean_rotations

ROOT = Path(__file__).resolve().parent.parent
VIDEO_NPZ = ROOT / "ar_pipeline" / "data" / "video_clips.npz"
FPS = 24.0
LOOP_S = 0.5
# clip -> (seconds of the source video, frames per second of its BVH)
VIDEO_CLIPS = {"shake": (0.0, 5.8, 30.0)}
# template joint <- Truebones "Dog" joint (the MoCapAnything target skeleton)
MAP = {"COG_jnt": "Bip01_Spine0", "spine_1_jnt": "Bip01_Spine1", "spine_2_jnt": "Bip01_Spine2",
       "chest_jnt": "Bip01_Spine3", "neck_base_jnt": "Bip01_Neck1", "neck_mid_jnt": "Bip01_Neck2",
       "head_jnt": "Bip01_Head", "L_ear_base_jnt": "Bip01_Ponytail1", "R_ear_base_jnt": "Bip01_Ponytail2",
       "tail_1_jnt": "Bip01_Spine0_Tail", "tail_2_jnt": "Bip01_Spine0_Tail1", "tail_3_jnt": "Bip01_Spine0_Tail2",
       "tail_4_jnt": "Bip01_Spine0_Tail3", "tail_5_jnt": "Bip01_Spine0_Tail4"}
for _s in "LR":
    MAP.update({f"{_s}_clavicle_jnt": f"Bip01_{_s}_Clavicle", f"{_s}_shoulder_jnt": f"Bip01_{_s}_UpperArm",
                f"{_s}_elbow_jnt": f"Bip01_{_s}_Forearm", f"{_s}_wrist_jnt": f"Bip01_{_s}_Hand",
                f"{_s}_hand_jnt": f"Bip01_{_s}_Finger0", f"{_s}_hip_jnt": f"Bip01_{_s}_Thigh",
                f"{_s}_knee_jnt": f"Bip01_{_s}_Calf", f"{_s}_ankle_jnt": f"Bip01_{_s}_Foot",
                f"{_s}_foot_jnt": f"Bip01_{_s}_Toe0"})


def _smooth(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def extract(bvh_path: Path, t0: float, t1: float, fps_in: float) -> dict[str, np.ndarray]:
    """{Truebones joint: (F, 3, 3) world rotations}, facing +z, cleaned,
    at FPS, looped."""
    b = BVH(bvh_path)
    joints = sorted(set(MAP.values()))
    frames = np.clip(np.round(np.arange(t0, t1, 1.0 / FPS) * fps_in).astype(int), 0, len(b.data) - 1)
    pos, rot = b.worlds(frames)
    I = b.index
    ahead = pos[0, I["Bip01_Spine3"]] - pos[0, I["Bip01_Spine0"]]
    face = Rotation.from_rotvec([0.0, -np.arctan2(ahead[0], ahead[2]), 0.0]).as_matrix()
    rot = np.einsum("ij,fkjl->fkil", face, rot[:, [I[j] for j in joints]])
    rot = clean_rotations(rot)
    m = int(round(LOOP_S * FPS))
    first = Rotation.from_matrix(rot[0].reshape(-1, 3, 3))
    for k in range(m):
        w = float(_smooth(np.array((k + 1) / m)))
        f = len(rot) - m + k
        cur = Rotation.from_matrix(rot[f])
        delta = (first * cur.inv()).as_rotvec() * w
        rot[f] = (Rotation.from_rotvec(delta) * cur).as_matrix()
    return {j: rot[:, n] for n, j in enumerate(joints)}


def load() -> dict[str, dict[str, np.ndarray]]:
    """{clip: {Truebones joint: (F, 3, 3)}}"""
    if not VIDEO_NPZ.exists():
        return {}
    data = np.load(VIDEO_NPZ)
    out: dict[str, dict[str, np.ndarray]] = {}
    for key in data.files:
        clip, joint = key.split("/", 1)
        q = data[key]
        out.setdefault(clip, {})[joint] = Rotation.from_quat(q).as_matrix()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clip", required=True, choices=sorted(VIDEO_CLIPS))
    parser.add_argument("--bvh", required=True, help="MoCapAnything's *_rot6d_pred.bvh for the clip's video")
    args = parser.parse_args()
    t0, t1, fps_in = VIDEO_CLIPS[args.clip]
    clip = extract(Path(args.bvh), t0, t1, fps_in)
    old = dict(np.load(VIDEO_NPZ)) if VIDEO_NPZ.exists() else {}
    old = {k: v for k, v in old.items() if not k.startswith(args.clip + "/")}
    for joint, R in clip.items():
        old[f"{args.clip}/{joint}"] = Rotation.from_matrix(R).as_quat().astype(np.float32)
    VIDEO_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(VIDEO_NPZ, **old)
    n = len(next(iter(clip.values())))
    print(f"[video_motion] {args.clip}: {n} frames ({n / FPS:.1f} s) -> {VIDEO_NPZ} "
          f"({VIDEO_NPZ.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
