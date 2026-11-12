"""Open the mouth in the clips a template already has.

pose_clips.py opens the mouth as it builds a clip (open_mouth, for MOUTH_SLUGS). A
template whose clips were built elsewhere keeps them: the maltese, toy poodle,
chihuahua and jindo puppy got theirs from a teammate's copy on 2026-11-12, the
mouth shut in every clip but the bark (user: "기존 진도, 리트리버처럼 모션중간중간
입 움직이게 안되있찌?"). This lays the same jaw, nod and panting chest over the
keys those clips have - three channels - and leaves the body's motion as it is.

    python -m ar_pipeline.mouth_clips --slug maltese             -> work/anim/maltese_mouth.glb
    python -m ar_pipeline.mouth_clips --slug maltese --install   (the template; backup work/backup_anim_<slug>_mouth/)

The spans are pose_clips.MOUTH's. The roll and the fall are timed by the clip
itself, since they differ in length from template to template: by when the body
(COG_jnt) is turned BODY_TURNED_DEG from the clip's first frame. The mouth opens
ROLL_LEAD_S before that in a rollover and shuts ROLL_LAG_S after it; playing dead
it opens DEAD_AFTER_S after the body went down and shuts DEAD_BEFORE_S before it
rises. Those are the jindo's own spans measured the same way: 0.8-2.5 s about a
roll at 1.17-1.79 s, 5.2-9.8 s inside a fall at 4.12-10.17 s.

A clip whose jaw already opens is left alone (--force opens it further).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from pygltflib import GLTF2

from ar_pipeline.gltf_rig import append_accessor, prune, read
from ar_pipeline.pose_clips import MOUTH, MOUTH_EASE_S, ROOT, Skeleton, open_mouth

BODY_TURNED_DEG = 45.0
ROLL_LEAD_S, ROLL_LAG_S = 0.37, 0.71
DEAD_AFTER_S, DEAD_BEFORE_S = 1.08, 0.37
# a jaw that turns this far from the clip's first frame is already open
OPEN_DEG = 2.0


def turned_from_first(quats: np.ndarray) -> np.ndarray:
    """(F,) degrees each key's rotation is from the first key's."""
    q = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    return np.degrees(2.0 * np.arccos(np.clip(np.abs(q @ q[0]), -1.0, 1.0)))


def spans_for(name: str, times: np.ndarray, body: np.ndarray) -> tuple:
    """MOUTH's spans for the clip, the rollover's and play_dead's moved to
    where this clip's body is turned."""
    spans = MOUTH[name]
    if name not in ("rollover", "play_dead"):
        return spans
    turned = np.nonzero(turned_from_first(body) > BODY_TURNED_DEG)[0]
    if not len(turned):
        raise ValueError(f"{name}: the body never turns {BODY_TURNED_DEG:.0f} deg from its first frame")
    t0, t1 = float(times[turned[0]]), float(times[turned[-1]])
    span = (t0 - ROLL_LEAD_S, t1 + ROLL_LAG_S) if name == "rollover" else (t0 + DEAD_AFTER_S, t1 - DEAD_BEFORE_S)
    return ((round(span[0], 2), round(span[1], 2), spans[0][2]),)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slug", required=True)
    parser.add_argument("--size", default="small")
    parser.add_argument("--glb", default=None, help="a built dog's GLB instead of the template")
    parser.add_argument("--clip", action="append", help="only these clips (default: every clip MOUTH names)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--force", action="store_true", help="also a clip whose jaw already opens")
    args = parser.parse_args()
    glb = Path(args.glb) if args.glb else ROOT / "output" / f"skeleton_{args.slug}_{args.size}.glb"
    sk = Skeleton(glb)
    if "jaw_jnt" not in sk.j or not sk.mask([sk.j["jaw_jnt"]]).any():
        raise SystemExit(f"{glb}: the jaw carries no skin (ar_pipeline/mouth.py first)")
    jaw, head, cog, breathe = sk.j["jaw_jnt"], sk.j["head_jnt"], sk.cog, sk.j.get("breathe_jnt")
    gltf = GLTF2().load(str(glb))
    blob = bytearray(gltf.binary_blob())
    done = 0
    for anim in gltf.animations or []:
        if anim.name not in MOUTH or (args.clip and anim.name not in args.clip):
            continue
        sampler = {(c.target.node, c.target.path): anim.samplers[c.sampler] for c in anim.channels}
        missing = [n for n, key in (("jaw", (jaw, "rotation")), ("head", (head, "rotation")), ("body", (cog, "rotation")))
                   if key not in sampler]
        if missing:
            print(f"[mouth_clips] {anim.name}: skipped - no rotation keys for the {', '.join(missing)}")
            continue
        own = [sampler[(jaw, "rotation")], sampler[(head, "rotation")]]
        if (breathe, "scale") in sampler:
            own.append(sampler[(breathe, "scale")])
        shared = [s for s in own if sum(s is other for other in anim.samplers) != 1 or
                  sum(c.sampler == anim.samplers.index(s) for c in anim.channels) != 1]
        if shared or any(s.interpolation not in (None, "LINEAR") for s in own) or len({s.input for s in own}) != 1:
            print(f"[mouth_clips] {anim.name}: skipped - its jaw, head and breath keys are not three LINEAR tracks on one clock")
            continue
        # (the times as pose_clips.write stores them)
        times = read(gltf, blob, own[0].input).ravel().astype(np.float32).astype(np.float64)
        jaw_q, head_q = read(gltf, blob, own[0].output), read(gltf, blob, own[1].output)
        opens = float(turned_from_first(jaw_q).max())
        if opens > OPEN_DEG and not args.force:
            print(f"[mouth_clips] {anim.name}: left - the jaw already opens {opens:.0f} deg")
            continue
        spans = spans_for(anim.name, times, read(gltf, blob, sampler[(cog, "rotation")].output))
        if spans[0][0] < MOUTH_EASE_S / 2 or spans[-1][1] > times[-1] - MOUTH_EASE_S / 2:
            raise SystemExit(f"{anim.name}: the mouth's spans {[(a, b) for a, b, _ in spans]} do not fit a clip of {times[-1]:.2f} s")
        result = {"times": times, "rotations": {jaw: jaw_q.copy(), head: head_q.copy()}}
        if len(own) == 3:
            result["breathe"] = read(gltf, blob, own[2].output).copy()
        open_mouth(sk, anim.name, result, spans)
        own[0].output = append_accessor(gltf, blob, np.asarray(result["rotations"][jaw], dtype=np.float32), "VEC4")
        own[1].output = append_accessor(gltf, blob, np.asarray(result["rotations"][head], dtype=np.float32), "VEC4")
        if len(own) == 3:
            own[2].output = append_accessor(gltf, blob, np.asarray(result["breathe"], dtype=np.float32), "VEC3")
        done += 1
        after = turned_from_first(np.asarray(result["rotations"][jaw]))
        print(f"[mouth_clips] {anim.name}: open {[(a, b) for a, b, _ in spans]} s, the jaw to {after.max():.0f} deg, "
              f"shut at both ends ({after[0]:.1f}, {after[-1]:.1f})" + ("" if len(own) == 3 else "; no breath track in this clip"))
    if not done:
        print("[mouth_clips] nothing to do")
        return
    blob = prune(gltf, bytes(blob))
    gltf.buffers[0].byteLength = len(blob)
    gltf.set_binary_blob(bytes(blob))
    if args.install:
        backup = ROOT / "work" / f"backup_anim_{args.slug}_mouth"
        if not backup.exists():
            backup.mkdir(parents=True)
            shutil.copy(glb, backup / glb.name)
        out = glb
    else:
        out = Path(args.out) if args.out else ROOT / "work" / "anim" / f"{args.slug}_mouth.glb"
    out.parent.mkdir(parents=True, exist_ok=True)
    gltf.save(str(out))
    print(f"[mouth_clips] wrote {out} ({out.stat().st_size / 1e6:.1f} MB), {done} clips")


if __name__ == "__main__":
    main()
