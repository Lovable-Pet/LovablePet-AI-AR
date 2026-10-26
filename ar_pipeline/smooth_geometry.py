"""Smooth the sculpted fur clumps out of a template's chest, in place.

WHY
---
The Meshy jindo has its chest and throat ruff modelled as lumps of fur,
1-2 cm of relief the rest of the body does not have. The GLB carries no
normal map, so the vertex normals are the shading: on a cream chest the
lumps read as a cotton-wool patch whose fur looks unlike the flank's (user
on eval v8 e05: "the front of the neck has different fur").

HOW
---
A weight per vertex from the region masks sampled at its UV (the chest by
default), spread a few rings over the mesh so it fades out instead of
stopping at the mask's edge. Taubin smoothing (a shrink step and an inflate
step each round, so the chest keeps its volume) moves each vertex by that
weight; vertices split along a UV seam are one point and move together.
The normals of the moved vertices are recomputed from the new faces and
blended into the old ones by the same weight. Morph targets, skin and UVs
are untouched.

Usage:
    python -m ar_pipeline.smooth_geometry --slug jindo --size small [--regions chest] [--rounds 20]
Back up output/skeleton_<slug>_<size>.glb first; this edits it in place.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pygltflib import GLTF2
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LAMBDA, MU = 0.5, -0.53
SPREAD_RINGS = 6


def accessor_view(gltf: GLTF2, blob: bytearray, index: int, dtype, width: int) -> tuple[np.ndarray, int]:
    acc = gltf.accessors[index]
    view = gltf.bufferViews[acc.bufferView]
    offset = (view.byteOffset or 0) + (acc.byteOffset or 0)
    stride = view.byteStride or np.dtype(dtype).itemsize * width
    if stride != np.dtype(dtype).itemsize * width:
        raise ValueError("interleaved accessors are not handled")
    count = acc.count * width
    return np.frombuffer(bytes(blob[offset:offset + count * np.dtype(dtype).itemsize]), dtype).reshape(-1, width).copy(), offset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", required=True)
    parser.add_argument("--size", default="small")
    parser.add_argument("--regions", default="chest")
    parser.add_argument("--rounds", type=int, default=20)
    args = parser.parse_args()
    glb = ROOT / "output" / f"skeleton_{args.slug}_{args.size}.glb"
    masks_dir = ROOT / "output" / "templates" / f"{args.slug}_{args.size}" / "masks"

    gltf = GLTF2().load(str(glb))
    blob = bytearray(gltf.binary_blob())
    prim = max((p for m in gltf.meshes for p in m.primitives),
               key=lambda p: gltf.accessors[p.attributes.POSITION].count)
    pos, pos_off = accessor_view(gltf, blob, prim.attributes.POSITION, np.float32, 3)
    nrm, nrm_off = accessor_view(gltf, blob, prim.attributes.NORMAL, np.float32, 3)
    uv, _ = accessor_view(gltf, blob, prim.attributes.TEXCOORD_0, np.float32, 2)
    idx_acc = gltf.accessors[prim.indices]
    idx_dtype = {5125: np.uint32, 5123: np.uint16}[idx_acc.componentType]
    tri, _ = accessor_view(gltf, blob, prim.indices, idx_dtype, 1)
    tri = tri.reshape(-1, 3).astype(np.int64)

    # one node per position (seam copies move together)
    _, node = np.unique(np.round(pos / 1e-5).astype(np.int64), axis=0, return_inverse=True)
    node = node.reshape(-1)
    n = int(node.max()) + 1
    count = np.bincount(node, minlength=n).astype(np.float64)
    p = np.zeros((n, 3))
    np.add.at(p, node, pos.astype(np.float64))
    p /= count[:, None]

    # the weight: the regions' masks at each copy's UV, the largest per node
    size = None
    weight = np.zeros(n)
    for region in args.regions.split(","):
        mask = np.asarray(Image.open(masks_dir / f"{region}.png").convert("L"), dtype=np.float32) / 255.0
        size = mask.shape[0]
        x = np.clip(np.round(uv[:, 0] * (size - 1)).astype(int), 0, size - 1)
        y = np.clip(np.round(uv[:, 1] * (size - 1)).astype(int), 0, size - 1)
        np.maximum.at(weight, node, mask[y, x])

    t = node[tri]
    edges = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    edges = edges[edges[:, 0] != edges[:, 1]]
    edges = np.concatenate([edges, edges[:, ::-1]])
    adj = sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    adj.data[:] = 1.0
    degree = np.asarray(adj.sum(axis=1)).ravel()
    avg = sparse.diags(1.0 / np.maximum(degree, 1)) @ adj
    # fade out over a few rings instead of stopping at the mask edge
    for _ in range(SPREAD_RINGS):
        weight = np.maximum(weight, 0.8 * (avg @ weight))
    weight = np.clip(weight, 0.0, 1.0)

    before = p.copy()
    for _ in range(args.rounds):
        for step in (LAMBDA, MU):
            p = p + (step * weight)[:, None] * (avg @ p - p)
    moved = np.linalg.norm(p - before, axis=1)
    touched = weight > 0.01
    print(f"[smooth_geometry] {int(touched.sum())} of {n} points weighted; moved mean "
          f"{moved[touched].mean() * 1000:.1f} mm, p95 {np.percentile(moved[touched], 95) * 1000:.1f} mm, "
          f"max {moved.max() * 1000:.1f} mm")

    # normals from the new faces (area-weighted, per node), blended in by weight
    fn = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
    vn = np.zeros((n, 3))
    for k in range(3):
        np.add.at(vn, t[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    blend = np.clip(weight * 1.5, 0.0, 1.0)[node][:, None]
    old = nrm.astype(np.float64)
    # keep the file's orientation convention: flip the new normal where it disagrees
    new = vn[node]
    new = np.where((np.sum(new * old, axis=1) < 0)[:, None], -new, new)
    out_n = (1 - blend) * old + blend * new
    out_n /= np.maximum(np.linalg.norm(out_n, axis=1, keepdims=True), 1e-12)

    new_pos = p[node].astype(np.float32)
    new_pos = np.where(touched[node][:, None], new_pos, pos)
    blob[pos_off:pos_off + new_pos.nbytes] = new_pos.tobytes()
    blob[nrm_off:nrm_off + out_n.astype(np.float32).nbytes] = out_n.astype(np.float32).tobytes()
    acc = gltf.accessors[prim.attributes.POSITION]
    acc.min, acc.max = new_pos.min(0).tolist(), new_pos.max(0).tolist()
    gltf.set_binary_blob(bytes(blob))
    gltf.save(str(glb))
    print(f"[smooth_geometry] wrote {glb.name}")


if __name__ == "__main__":
    main()
