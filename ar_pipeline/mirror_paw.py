"""Reshape one paw of a template after its mirrored twin, in place.

WHY
---
The Meshy maltese (2026-10-12) has its right front paw sculpted as a few
large flat facets with a flat polygonal sole, while the other three paws
are rounded; with a plain coat colour on it the flat paw reads as broken
(user: "한쪽 발바닥 깨진 것 같은데"). Texture cannot fix a shape.

HOW
---
The good paw's faces are mirrored across the midline (x -> -x) and slid so
their centroid sits on the bad paw's centroid. Every point of the bad paw
(the paws mask at its UV, faded out over a few rings like
smooth_geometry.py) is pulled to the nearest point on that mirrored
surface by its weight. Normals of the moved points are recomputed from the
new faces and blended in. Skin weights, UVs and morph targets are left
alone, so the rig and the texture atlas still fit.

Usage:
    python -m ar_pipeline.mirror_paw --slug maltese --good LF --bad RF
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ar_pipeline"))
from smooth_geometry import SPREAD_RINGS, accessor_view  # noqa: E402

PAW_SIGN = {  # glTF frame: +x is the dog's left, +z is forward
    "LF": (1, 1), "RF": (-1, 1), "LB": (1, -1), "RB": (-1, -1),
}
PAW_MASK_MIN = 0.5
PAW_MAX_HEIGHT = 0.16   # m above the ground: the paw and the pastern, not the leg
MAX_PULL = 0.05         # m: a point further than this from the mirrored surface is left alone


def closest_points(points: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Nearest point on a triangle soup for every point (Ericson, Real-Time
    Collision Detection 5.1.5), vectorised over the triangles."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac = b - a, c - a
    out = np.empty_like(points)
    for i, p in enumerate(points):
        ap = p - a
        d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
        bp = p - b
        d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
        cp = p - c
        d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
        vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
        q = np.empty_like(tri[:, 0])
        # vertex regions
        q[:] = a
        q = np.where(((d3 >= 0) & (d4 <= d3))[:, None], b, q)
        q = np.where(((d6 >= 0) & (d5 <= d6))[:, None], c, q)
        # edge regions
        v = np.divide(d1, d1 - d3, out=np.zeros_like(d1), where=(d1 - d3) != 0)
        on_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        q = np.where(on_ab[:, None], a + v[:, None] * ab, q)
        w = np.divide(d2, d2 - d6, out=np.zeros_like(d2), where=(d2 - d6) != 0)
        on_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        q = np.where(on_ac[:, None], a + w[:, None] * ac, q)
        w2 = np.divide(d4 - d3, (d4 - d3) + (d5 - d6), out=np.zeros_like(d4), where=((d4 - d3) + (d5 - d6)) != 0)
        on_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        q = np.where(on_bc[:, None], b + w2[:, None] * (c - b), q)
        # face region
        inside = (va > 0) & (vb > 0) & (vc > 0)
        denom = va + vb + vc
        vv = np.divide(vb, denom, out=np.zeros_like(vb), where=denom != 0)
        ww = np.divide(vc, denom, out=np.zeros_like(vc), where=denom != 0)
        q = np.where(inside[:, None], a + vv[:, None] * ab + ww[:, None] * ac, q)
        d = np.einsum("ij,ij->i", q - p, q - p)
        out[i] = q[np.argmin(d)]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", required=True)
    parser.add_argument("--size", default="small")
    parser.add_argument("--good", default="LF", choices=sorted(PAW_SIGN))
    parser.add_argument("--bad", default="RF", choices=sorted(PAW_SIGN))
    args = parser.parse_args()
    glb = ROOT / "output" / f"skeleton_{args.slug}_{args.size}.glb"
    mask_path = ROOT / "output" / "templates" / f"{args.slug}_{args.size}" / "masks" / "paws.png"

    gltf = GLTF2().load(str(glb))
    blob = bytearray(gltf.binary_blob())
    prim = max((p for m in gltf.meshes for p in m.primitives),
               key=lambda p: gltf.accessors[p.attributes.POSITION].count)
    pos, pos_off = accessor_view(gltf, blob, prim.attributes.POSITION, np.float32, 3)
    nrm, nrm_off = accessor_view(gltf, blob, prim.attributes.NORMAL, np.float32, 3)
    uv, _ = accessor_view(gltf, blob, prim.attributes.TEXCOORD_0, np.float32, 2)
    idx_acc = gltf.accessors[prim.indices]
    tri, _ = accessor_view(gltf, blob, prim.indices, {5125: np.uint32, 5123: np.uint16}[idx_acc.componentType], 1)
    tri = tri.reshape(-1, 3).astype(np.int64)

    _, node = np.unique(np.round(pos / 1e-5).astype(np.int64), axis=0, return_inverse=True)
    node = node.reshape(-1)
    n = int(node.max()) + 1
    count = np.bincount(node, minlength=n).astype(np.float64)
    p = np.zeros((n, 3))
    np.add.at(p, node, pos.astype(np.float64))
    p /= count[:, None]

    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    size = mask.shape[0]
    x = np.clip(np.round(uv[:, 0] * (size - 1)).astype(int), 0, size - 1)
    y = np.clip(np.round(uv[:, 1] * (size - 1)).astype(int), 0, size - 1)
    paw = np.zeros(n)
    np.maximum.at(paw, node, mask[y, x])

    ground = p[:, 1].min()

    def paw_nodes(name: str) -> np.ndarray:
        sx, sz = PAW_SIGN[name]
        return ((paw > PAW_MASK_MIN) & (np.sign(p[:, 0]) == sx) & (np.sign(p[:, 2]) == sz)
                & (p[:, 1] < ground + PAW_MAX_HEIGHT))

    good, bad = paw_nodes(args.good), paw_nodes(args.bad)
    t = node[tri]
    good_faces = t[good[t].all(axis=1)]
    surface = p[good_faces].copy()
    if PAW_SIGN[args.good][0] != PAW_SIGN[args.bad][0]:
        surface[:, :, 0] *= -1
    if PAW_SIGN[args.good][1] != PAW_SIGN[args.bad][1]:
        surface[:, :, 2] *= -1
    shift = p[bad].mean(axis=0) - surface.reshape(-1, 3).mean(axis=0)
    shift[1] = 0.0  # both paws stand on the ground
    surface += shift
    print(f"[mirror_paw] {args.good} -> {args.bad}: {int(good.sum())} good points / {len(good_faces)} faces, "
          f"{int(bad.sum())} bad points, slid {np.linalg.norm(shift) * 1000:.0f} mm")

    # weight: the bad paw's mask faded over a few rings, so the ankle blends
    edges = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    edges = edges[edges[:, 0] != edges[:, 1]]
    edges = np.concatenate([edges, edges[:, ::-1]])
    adj = sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    adj.data[:] = 1.0
    avg = sparse.diags(1.0 / np.maximum(np.asarray(adj.sum(axis=1)).ravel(), 1)) @ adj
    weight = np.where(bad, paw, 0.0)
    sx, sz = PAW_SIGN[args.bad]
    same_leg = (np.sign(p[:, 0]) == sx) & (np.sign(p[:, 2]) == sz) & (p[:, 1] < ground + PAW_MAX_HEIGHT * 1.5)
    for _ in range(SPREAD_RINGS):
        weight = np.maximum(weight, 0.8 * (avg @ weight)) * same_leg
    weight = np.clip(weight, 0.0, 1.0)

    touched = weight > 0.01
    target = closest_points(p[touched], surface)
    pull = np.linalg.norm(target - p[touched], axis=1)
    weight[np.flatnonzero(touched)[pull > MAX_PULL]] = 0.0
    touched = weight > 0.01
    target = target[pull <= MAX_PULL]
    before = p.copy()
    p[touched] += (weight[touched])[:, None] * (target - p[touched])
    moved = np.linalg.norm(p - before, axis=1)
    print(f"[mirror_paw] moved {int(touched.sum())} points: mean {moved[touched].mean() * 1000:.1f} mm, "
          f"p95 {np.percentile(moved[touched], 95) * 1000:.1f} mm, max {moved.max() * 1000:.1f} mm")

    fn = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
    vn = np.zeros((n, 3))
    for k in range(3):
        np.add.at(vn, t[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    blend = np.clip(weight * 1.5, 0.0, 1.0)[node][:, None]
    old = nrm.astype(np.float64)
    new = vn[node]
    new = np.where((np.sum(new * old, axis=1) < 0)[:, None], -new, new)
    out_n = (1 - blend) * old + blend * new
    out_n /= np.maximum(np.linalg.norm(out_n, axis=1, keepdims=True), 1e-12)

    new_pos = np.where(touched[node][:, None], p[node], pos.astype(np.float64)).astype(np.float32)
    blob[pos_off:pos_off + new_pos.nbytes] = new_pos.tobytes()
    blob[nrm_off:nrm_off + out_n.astype(np.float32).nbytes] = out_n.astype(np.float32).tobytes()
    acc = gltf.accessors[prim.attributes.POSITION]
    acc.min, acc.max = new_pos.min(0).tolist(), new_pos.max(0).tolist()
    gltf.set_binary_blob(bytes(blob))
    gltf.save(str(glb))
    print(f"[mirror_paw] wrote {glb.name}")


if __name__ == "__main__":
    main()
