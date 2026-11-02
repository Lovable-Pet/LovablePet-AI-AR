"""Give a drop-eared template's ear flaps to the skull mask, out of the face
regions the seeding spread over them.

WHY
---
On the retriever template the cheeks mask runs onto the front of each
hanging ear, from eye level to the tip - the seeding went by distance on
the surface, and a drop ear lies against the cheek. A dog read with cream
cheeks got cream ears (shelter retriever-type 8, a fawn with a black mask
whose photo shows dark ears, 11-02). An ear flap is the head's coat:
ears_outer, painted as the skull.

HOW
---
The flap is where the mesh is skinned to the ear joints (any joint named
"*ear*"): the vertex weights, interpolated over each triangle into the UV
atlas, a ramp from EAR_FLAP_MIN over EAR_FLAP_RAMP of ear weight. Its front
edge is skinned to the head alone, and that left a white strip down the
front of each tan ear on a white-cheeked dog (shelter retriever-type 7), so
the flap also takes the texels within EAR_EDGE_M (3D, fading over
EAR_EDGE_RAMP_M) of its core (ear weight over EAR_CORE) that lie behind the
eyes - the head's side under the ear goes with it, unseen. Each face region
(FACE_REGIONS) gives up its weight on the flap - its mask times the flap
share - to the skull. The masks are taken from the backup
(work/backup_earflaps_<slug>/, written on the first --install) when it
exists, so a run repeats exactly.

    python -m ar_pipeline.ear_flaps --slug retriever             # report + preview
    python -m ar_pipeline.ear_flaps --slug retriever --install   # write masks/

Done for retriever on 2026-11-02. The template assets go into the next
release (assets-v2).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from pygltflib import GLTF2

from ar_pipeline.fix_uv_overlaps import read

FACE_REGIONS = ("cheeks", "eyebrows", "muzzle_bridge")
# ear weight from which a texel is flap, and over which it becomes all flap:
# the ear joints' weights fall off toward the flap's root, 0.01 still on
# the flap; above, the flap's front half stayed cheeks
EAR_FLAP_MIN = 0.01
EAR_FLAP_RAMP = 0.08
EAR_CORE = 0.03
EAR_EDGE_M = 0.012
EAR_EDGE_RAMP_M = 0.006
# "behind the eyes": this far back of the eye landmarks
EAR_BEHIND_EYES_M = 0.02


def edge_of(flap_core: np.ndarray, candidates: np.ndarray, position: np.ndarray) -> np.ndarray:
    """The share (0..1) of each candidate texel that is the flap's edge, by
    its 3D distance to the flap's core."""
    from scipy.spatial import cKDTree

    out = np.zeros(candidates.shape, dtype=np.float32)
    if not flap_core.any() or not candidates.any():
        return out
    reach = EAR_EDGE_M + EAR_EDGE_RAMP_M
    distance, _ = cKDTree(position[flap_core]).query(position[candidates], distance_upper_bound=reach)
    out[candidates] = np.clip((reach - distance) / EAR_EDGE_RAMP_M, 0.0, 1.0)
    return out


def ear_weight(glb: Path, size: int) -> np.ndarray:
    """Each texel's skin weight on the ear joints, interpolated over its
    triangle (0 off the ears and outside the atlas)."""
    gltf = GLTF2().load(str(glb))
    blob = gltf.binary_blob()
    names = [gltf.nodes[j].name or "" for j in gltf.skins[0].joints]
    ears = [i for i, name in enumerate(names) if "ear" in name.lower()]
    if not ears:
        raise SystemExit(f"{glb.name}: no ear joints")
    prim = max((p for m in gltf.meshes for p in m.primitives), key=lambda p: gltf.accessors[p.attributes.POSITION].count)
    joints = read(gltf, blob, prim.attributes.JOINTS_0).astype(np.int64)
    weights = read(gltf, blob, prim.attributes.WEIGHTS_0).astype(np.float64)
    if weights.max() > 1.5:   # normalised integers
        weights /= 255.0 if weights.max() <= 255 else 65535.0
    uv = read(gltf, blob, prim.attributes.TEXCOORD_0).astype(np.float64)
    tri = read(gltf, blob, prim.indices).reshape(-1, 3).astype(np.int64)
    per_vertex = sum(((joints == j) * weights).sum(axis=1) for j in ears)
    out = np.zeros((size, size))
    for t in np.nonzero(per_vertex[tri].max(axis=1) > 0)[0]:
        p = uv[tri[t]] * size - 0.5
        w = per_vertex[tri[t]]
        x0, y0 = np.maximum(np.floor(p.min(axis=0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(p.max(axis=0)).astype(int), size - 1)
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        a, b, c = p
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(den) < 1e-12:
            continue
        l1 = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / den
        l2 = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / den
        l3 = 1.0 - l1 - l2
        inside = (l1 >= -0.02) & (l2 >= -0.02) & (l3 >= -0.02)
        value = l1 * w[0] + l2 * w[1] + l3 * w[2]
        out[ys[inside], xs[inside]] = np.maximum(out[ys[inside], xs[inside]], value[inside])
    return out.astype(np.float32)


def _load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _save(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slug", required=True)
    parser.add_argument("--size", default="small")
    parser.add_argument("--install", action="store_true", help="write the template's masks/")
    args = parser.parse_args()
    tpl = Path(f"output/templates/{args.slug}_{args.size}")
    masks = tpl / "masks"
    glb = Path(f"output/skeleton_{args.slug}_{args.size}.glb")
    backup = Path(f"work/backup_earflaps_{args.slug}")
    source = backup if backup.exists() else masks
    names = [n for n in (*FACE_REGIONS, "skull") if (source / f"{n}.png").exists()]
    loaded = {n: _load(source / f"{n}.png") for n in names}
    size = next(iter(loaded.values())).shape[0]
    weight = ear_weight(glb, size)
    flap = np.clip((weight - EAR_FLAP_MIN) / EAR_FLAP_RAMP, 0.0, 1.0)
    from api.landmarks import Landmarks

    lm = Landmarks.load(tpl, tpl / "geometry")
    position, valid = lm.position, lm.valid
    eyes_y = max(lm.world("left_eye")[1], lm.world("right_eye")[1])
    face = np.zeros_like(valid)
    for name in FACE_REGIONS:
        if name in loaded:
            face |= loaded[name] > 0.02
    behind = valid & face & (position[..., 1] > eyes_y + EAR_BEHIND_EYES_M)
    edge = edge_of(valid & (weight > EAR_CORE), behind, position)
    print(f"[ear_flaps] flap {int((flap > 0.5).sum())} texels, its front edge {int((edge > 0.5).sum())} more")
    flap = np.maximum(flap, edge)
    out = dict(loaded)
    for name in FACE_REGIONS:
        if name not in out:
            continue
        moved = out[name] * flap
        out[name] = out[name] - moved
        out["skull"] = np.clip(out["skull"] + moved, 0.0, 1.0)
        print(f"[ear_flaps] {name}: {int((moved > 0.5).sum())} texels to the skull")
    if not args.install:
        preview = Path(f"work/ear_flaps_{args.slug}")
        preview.mkdir(parents=True, exist_ok=True)
        for name, mask in out.items():
            _save(mask, preview / f"{name}.png")
        print(f"[ear_flaps] preview masks in {preview}")
        return
    if not backup.exists():
        backup.mkdir(parents=True)
        for name in names:
            shutil.copy(masks / f"{name}.png", backup / f"{name}.png")
    for name, mask in out.items():
        _save(mask, masks / f"{name}.png")
    print(f"[ear_flaps] wrote {masks} (backup {backup})")


if __name__ == "__main__":
    main()
