"""Give every triangle that samples another part of the atlas the texels of
the surface it sits on.

WHY
---
On the generated templates a few hundred triangles have UVs that land on
an island belonging to somewhere else on the dog: the baked position map
(geometry/position.npy) says the texels under their UVs are 2 cm or more
away from the triangle. Whatever colour the coat gives that other place
shows on them: the black blot in the middle of the chihuahua's bib (88
triangles; its own texture has the same mark), the dots over the maltese's
head and ears (494), the poodle's ears (720), the retriever's tail fringe
(1,662), the jindo's tail (125). The masks cannot fix it - the texels are
shared with the surface that owns them.

HOW
---
For each such triangle, every corner takes the UV of the valid texel
nearest to it in 3D - the surface it lies on, so it shows that surface's
coat. A corner shared with a correct triangle is copied first (all its
attributes and morph-target deltas) so the correct one keeps its UV. A
triangle whose corners would land on different islands takes its centre's
texel for all three (a flat patch of the right colour). The skin, clips and
morph targets are kept; the rewritten accessors go at the end of the buffer.

--reference <glb> takes the positions from another GLB with the same
topology: the jindo puppy's head was shrunk (shrink_head.py) after its
position map was baked, so the shrunk head would test as misplaced
everywhere; its pre-shrink GLB matches the map.

A triangle whose centre is right but whose corner reaches over another
place's texels (a partial overlap, PULL_KEEP) has those corners pulled in
toward its centre instead. --partial-only does that alone, for a template
this already fixed.

Usage:
    python -m ar_pipeline.fix_uv_overlaps --slug chihuahua [--size small] [--reference work/.../x.glb]
    python -m ar_pipeline.fix_uv_overlaps --slug jindo_puppy --partial-only [--out x.glb]
Back up output/skeleton_<slug>_<size>.glb first; without --out this edits it in place.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from pygltflib import GLTF2, Accessor, BufferView
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MISMATCH_M = 0.02
# Partial overlaps: a triangle whose centre is on its own surface while a
# corner's wedge lies over another place's texels. On the jindo puppy 2 the
# shoulder seam's corners lay on the bib, 12-18 cm away, and rendered a white
# zigzag down the black shoulder (shelter puppy 11, user: "앞다리 바깥에
# 번개무늬"); the centre test caught 88 of the puppy's 183 such triangles and
# 1 of the jindo's 218. The corners whose wedge is wrong are pulled toward the
# triangle's UV centre by the first of these shares that leaves no covered
# texel baked MISMATCH_M or more from the triangle (the least wrong one if
# none does).
PULL_KEEP = (0.85, 0.7, 0.55, 0.4, 0.25, 0.1)
COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
DTYPES = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}


def read(gltf: GLTF2, blob: bytes, index: int) -> np.ndarray:
    acc = gltf.accessors[index]
    view = gltf.bufferViews[acc.bufferView]
    dtype, width = DTYPES[acc.componentType], COMPONENTS[acc.type]
    item = np.dtype(dtype).itemsize * width
    stride = view.byteStride or item
    start = (view.byteOffset or 0) + (acc.byteOffset or 0)
    if stride == item:
        return np.frombuffer(blob, dtype, acc.count * width, start).reshape(acc.count, width).copy()
    rows = [np.frombuffer(blob, dtype, width, start + i * stride) for i in range(acc.count)]
    return np.stack(rows).copy()


def append(gltf: GLTF2, blob: bytearray, index: int, data: np.ndarray, target: int | None) -> None:
    """Point accessor `index` at `data`, written to the end of the buffer."""
    while len(blob) % 4:
        blob.append(0)
    offset = len(blob)
    raw = np.ascontiguousarray(data).tobytes()
    blob.extend(raw)
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(raw), target=target))
    acc: Accessor = gltf.accessors[index]
    acc.bufferView = len(gltf.bufferViews) - 1
    acc.byteOffset = 0
    acc.count = int(data.shape[0]) if data.ndim > 1 else int(data.size)
    if acc.min is not None or acc.max is not None:
        flat = data.reshape(acc.count, -1)
        acc.min, acc.max = flat.min(0).tolist(), flat.max(0).tolist()


def repack(gltf: GLTF2, blob: bytearray) -> bytearray:
    """The buffer with only the views still referenced, in order: the
    accessors rewritten by append() leave their old data behind."""
    used = sorted({a.bufferView for a in gltf.accessors if a.bufferView is not None}
                  | {i.bufferView for i in gltf.images or [] if i.bufferView is not None})
    packed = bytearray()
    views, remap = [], {}
    for old in used:
        view = gltf.bufferViews[old]
        while len(packed) % 4:
            packed.append(0)
        start = view.byteOffset or 0
        chunk = blob[start:start + view.byteLength]
        view.byteOffset = len(packed)
        packed.extend(chunk)
        remap[old] = len(views)
        views.append(view)
    for item in list(gltf.accessors) + list(gltf.images or []):
        if item.bufferView is not None:
            item.bufferView = remap[item.bufferView]
    gltf.bufferViews = views
    return packed


def to_blender(p: np.ndarray) -> np.ndarray:
    return np.stack([p[:, 0], -p[:, 2], p[:, 1]], axis=1)


def wrong_wedges(corner_uv: np.ndarray, corner_xyz: np.ndarray, position: np.ndarray,
                 valid: np.ndarray) -> np.ndarray:
    """How many texels the triangle (UV 0..1, its three corners' positions)
    covers whose baked position is MISMATCH_M or more from the triangle's own
    point there, counted per corner (the corner nearest each such texel in
    barycentric terms) - and each corner's own bilinear taps too: a corner
    on the edge between two texels reads both, though neither centre lies
    inside the triangle."""
    h, w = valid.shape
    xs, ys = corner_uv[:, 0] * w - 0.5, corner_uv[:, 1] * h - 0.5
    taps = np.zeros(3, dtype=int)
    for k in range(3):
        cx, cy = int(np.floor(xs[k])), int(np.floor(ys[k]))
        for ty in (cy, cy + 1):
            for tx in (cx, cx + 1):
                if (0 <= ty < h and 0 <= tx < w and valid[ty, tx]
                        and np.linalg.norm(position[ty, tx] - corner_xyz[k]) > MISMATCH_M):
                    taps[k] += 1
    x0, x1 = max(int(np.floor(xs.min())), 0), min(int(np.ceil(xs.max())), w - 1)
    y0, y1 = max(int(np.floor(ys.min())), 0), min(int(np.ceil(ys.max())), h - 1)
    denom = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
    if x1 < x0 or y1 < y0 or abs(denom) < 1e-12:
        return np.zeros(3, dtype=int)
    gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
    l0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / denom
    l1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / denom
    bary = np.stack([l0, l1, 1.0 - l0 - l1], axis=-1)
    inside = (bary >= -1e-3).all(axis=-1) & valid[y0:y1 + 1, x0:x1 + 1]
    if not inside.any():
        return taps
    here = bary @ corner_xyz
    off = np.linalg.norm(position[y0:y1 + 1, x0:x1 + 1] - here, axis=-1)
    wrong = inside & (off > MISMATCH_M)
    return taps + np.bincount(np.argmax(bary[wrong], axis=-1), minlength=3)


def pull_corners(corner_uv: np.ndarray, corner_xyz: np.ndarray, position: np.ndarray,
                 valid: np.ndarray) -> tuple[np.ndarray, int, int] | None:
    """(new corner UVs, wrong texels before, after) for a partial overlap
    (PULL_KEEP); None when the triangle covers no wrong texel."""
    before = wrong_wedges(corner_uv, corner_xyz, position, valid)
    if not before.any():
        return None
    move = before > 0
    centre = corner_uv.mean(axis=0)
    best = (corner_uv, int(before.sum()))
    for keep in PULL_KEEP:
        trial = corner_uv.copy()
        trial[move] = centre + keep * (corner_uv[move] - centre)
        left = int(wrong_wedges(trial, corner_xyz, position, valid).sum())
        if left < best[1]:
            best = (trial, left)
        if left == 0:
            break
    return best[0], int(before.sum()), best[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--size", default="small")
    parser.add_argument("--reference", default=None, help="GLB whose positions match the baked position map")
    parser.add_argument("--out", default=None, help="write here instead of over the template GLB")
    parser.add_argument("--partial-only", action="store_true",
                        help="pull in the partial overlaps only (a template this already fixed: what the centre "
                             "test still finds are fur tufts off the surface, already on their nearest texel)")
    args = parser.parse_args()
    glb = ROOT / "output" / f"skeleton_{args.slug}_{args.size}.glb"
    geometry = ROOT / "output" / "templates" / f"{args.slug}_{args.size}" / "geometry"
    position = np.load(geometry / "position.npy").astype(np.float32)
    valid = np.load(geometry / "valid.npy") > 0.5
    h, w = valid.shape

    gltf = GLTF2().load(str(glb))
    blob = bytearray(gltf.binary_blob())
    prim = max((p for m in gltf.meshes for p in m.primitives),
               key=lambda p: gltf.accessors[p.attributes.POSITION].count)
    attributes = {name: idx for name, idx in vars(prim.attributes).items() if isinstance(idx, int)}
    data = {name: read(gltf, bytes(blob), idx) for name, idx in attributes.items()}
    targets = [{name: read(gltf, bytes(blob), idx) for name, idx in vars(t).items() if isinstance(idx, int)}
               if not isinstance(t, dict) else {name: read(gltf, bytes(blob), idx) for name, idx in t.items()}
               for t in (prim.targets or [])]
    tri = read(gltf, bytes(blob), prim.indices).reshape(-1, 3).astype(np.int64)
    uv = data["TEXCOORD_0"].astype(np.float32)

    if args.reference:
        ref = GLTF2().load(args.reference)
        ref_prim = max((p for m in ref.meshes for p in m.primitives),
                       key=lambda p: ref.accessors[p.attributes.POSITION].count)
        points = to_blender(read(ref, ref.binary_blob(), ref_prim.attributes.POSITION))
        if points.shape[0] != uv.shape[0]:
            raise SystemExit("[uv] the reference GLB has a different vertex count")
    else:
        points = to_blender(data["POSITION"])

    def texel(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        col = np.clip((coords[:, 0] * w).astype(int), 0, w - 1)
        row = np.clip((coords[:, 1] * h).astype(int), 0, h - 1)
        return row, col

    centre = points[tri].mean(axis=1)
    row, col = texel(uv[tri].mean(axis=1))
    bad = valid[row, col] & (np.linalg.norm(position[row, col] - centre, axis=1) > MISMATCH_M)
    print(f"[uv] {int(bad.sum())} of {len(tri)} triangles sample texels baked 2 cm or more away")
    centre_bad = bad.copy()
    if args.partial_only:
        print("[uv] --partial-only: those are left as they are")
        bad[:] = False
    # partial overlaps among the rest (PULL_KEEP), on the UVs as loaded
    pulls: dict[int, tuple[np.ndarray, int, int]] = {}
    for t in np.nonzero(~centre_bad)[0]:
        pulled = pull_corners(uv[tri[t]].astype(np.float64), points[tri[t]].astype(np.float64), position, valid)
        if pulled is not None:
            pulls[int(t)] = pulled
    if pulls:
        print(f"[uv] {len(pulls)} more triangles reach over other places' texels at a corner: "
              f"{sum(p[1] for p in pulls.values())} texels, {sum(p[2] for p in pulls.values())} left once pulled in")
    if not bad.any() and not pulls:
        return

    count = uv.shape[0]
    copies: list[int] = []        # source vertex of each appended vertex
    copy_uv: list[np.ndarray] = []
    flat = 0
    if bad.any():
        flat = fix_whole(tri, uv, points, centre, bad, position, valid, copies, copy_uv)
    print(f"[uv] {len(copies)} vertices copied off correct triangles, {flat} triangles flattened to one texel")

    # the pulled corners: a vertex another triangle also uses is copied
    usage = np.bincount(tri.ravel(), minlength=count + len(copies))
    moved = 0
    for t, (new_corners, _, _) in pulls.items():
        for j in range(3):
            v = int(tri[t, j])
            if np.allclose(new_corners[j], uv[v]):
                continue
            moved += 1
            if usage[v] > 1:
                copies.append(v); copy_uv.append(new_corners[j].astype(np.float32))
                tri[t, j] = count + len(copies) - 1
            else:
                uv[v] = new_corners[j]
    if pulls:
        print(f"[uv] {moved} corners pulled in ({len(copies)} vertices copied in all)")

    source = np.asarray(copies, dtype=np.int64)
    for name, idx in attributes.items():
        values = uv if name == "TEXCOORD_0" else data[name]
        extra = values[source]
        if name == "TEXCOORD_0" and len(copies):
            extra = np.stack(copy_uv).astype(values.dtype)
        append(gltf, blob, idx, np.concatenate([values, extra]).astype(values.dtype), 34962)
    for target, target_data in zip(prim.targets or [], targets):
        items = target.items() if isinstance(target, dict) else vars(target).items()
        for name, idx in items:
            if not isinstance(idx, int):
                continue
            values = target_data[name]
            append(gltf, blob, idx, np.concatenate([values, values[source]]).astype(values.dtype), 34962)
    index_dtype = np.uint32 if count + len(copies) > 65535 else np.uint16
    gltf.accessors[prim.indices].componentType = 5125 if index_dtype is np.uint32 else 5123
    append(gltf, blob, prim.indices, tri.reshape(-1).astype(index_dtype), 34963)
    blob = repack(gltf, blob)
    gltf.buffers[0].byteLength = len(blob)
    gltf.set_binary_blob(bytes(blob))
    out = Path(args.out) if args.out else glb
    gltf.save(str(out))
    print(f"[uv] wrote {out}")


def fix_whole(tri: np.ndarray, uv: np.ndarray, points: np.ndarray, centre: np.ndarray, bad: np.ndarray,
              position: np.ndarray, valid: np.ndarray, copies: list[int], copy_uv: list[np.ndarray]) -> int:
    """The triangles whose centre samples another place (`bad`) moved onto
    the surface they lie on; edits `tri` and `uv`, appends to `copies` and
    `copy_uv`. Returns how many were flattened to one texel."""
    h, w = valid.shape
    count = uv.shape[0]
    rows, cols = np.nonzero(valid)
    tree = cKDTree(position[rows, cols].astype(np.float64))
    islands, _ = ndimage.label(valid)
    corners = np.unique(tri[bad])
    _, nearest = tree.query(points[corners].astype(np.float64), workers=-1)
    corner_uv = np.stack([(cols[nearest] + 0.5) / w, (rows[nearest] + 0.5) / h], axis=1).astype(np.float32)
    corner_island = islands[rows[nearest], cols[nearest]]
    new_uv = dict(zip(corners.tolist(), corner_uv))
    island_of = dict(zip(corners.tolist(), corner_island.tolist()))
    _, centre_nearest = tree.query(centre[bad].astype(np.float64), workers=-1)

    used_by_good = np.zeros(uv.shape[0], dtype=bool)
    used_by_good[tri[~bad].ravel()] = True
    remap: dict[int, int] = {}
    flat = 0
    for k, t in enumerate(np.nonzero(bad)[0]):
        corners_t = tri[t]
        if len({island_of[int(v)] for v in corners_t}) > 1:
            # the corners would span islands: one texel for the whole triangle
            r, c = rows[centre_nearest[k]], cols[centre_nearest[k]]
            flat_uv = np.array([(c + 0.5) / w, (r + 0.5) / h], dtype=np.float32)
            for j, v in enumerate(corners_t):
                copies.append(int(v)); copy_uv.append(flat_uv)
                tri[t, j] = count + len(copies) - 1
            flat += 1
            continue
        for j, v in enumerate(corners_t):
            v = int(v)
            if used_by_good[v]:
                if v not in remap:
                    copies.append(v); copy_uv.append(new_uv[v])
                    remap[v] = count + len(copies) - 1
                tri[t, j] = remap[v]
            else:
                uv[v] = new_uv[v]
    return flat


if __name__ == "__main__":
    main()
