"""Give the triangles whose colour a browser's mip levels change room of
their own in the atlas, on a dog built from a template.

WHY
---
model-viewer (three.js) samples a shrunk copy of the texture - a mip level,
each texel the mean of 2^k x 2^k texels - once the dog is small on screen.
The generated templates' atlases wedge thin strips of one place into the
chart of another: the jindo puppy's black shoulder is a strip 2-6 texels
wide inside the white bib's chart. Full size it is black; at mip 2-3 its
texels are averaged with the bib's and the shoulder shows a white zigzag
(shelter puppy 11, user: "오른쪽 앞발에 번개무늬"; chihuahua 08's tail and
thighs the same, NOTES 11-08). The texture is right; only the mips are not,
and a finish step cannot fix that - the bib's texels are the bib's.

HOW
---
Triangles whose mip-k value (k in LEVELS) differs from both their full-size
value and the mean of their 3D neighbourhood at that scale by more than
LIFT are moved, in patches of triangles sharing corners, into free atlas
space (FREE_MARGIN_PX off every island) at SCALE times their UV size; their
corners are copied first (all attributes and morph-target deltas), so the
triangles around keep theirs. The new texels are painted with what each
point showed full size - the old texture at the old UV, its bilinear taps
limited to texels baked on that surface (position map, MISMATCH_M) - and
padded PAD_PX round with the patch's own colours, so every mip block up to
2^3 over the patch is the patch. Template-protected texels (eyes, nose,
pads, claws) and triangles flattened to one texel (fix_uv_overlaps) stay.

Usage:
    python -m ar_pipeline.mip_islands --glb <built dog .glb> --slug jindo_puppy [--size small] [--out x.glb]
Without --out the GLB is rewritten in place; back it up first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pygltflib import GLTF2
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ar_pipeline.fix_uv_overlaps import MISMATCH_M, append, read, repack, to_blender  # noqa: E402

# luma (0-255) a mip level may move a sample before it counts, and how many
# of a triangle's samples must move: 40 and 2 moved 295 of puppy 11's
# triangles and left the zigzag faint but there (its seam edges change by
# 25-40); 25 and 1 moved 882 and it was gone, nothing new elsewhere
LIFT = 25.0
LEVELS = (1, 2, 3)   # mip levels checked: 2x2 .. 8x8 texel means
MIN_SAMPLES = 1
SCALE = 2.0          # moved patches at this many times their UV size
PAD_PX = 10          # own-colour padding round a moved patch (>= 2^max(LEVELS))
FREE_MARGIN_PX = 6   # kept between a moved patch's padding and any other texels in use
GRID = 5             # barycentric sample grid per triangle


def mip_pyramid(luma: np.ndarray, levels: int) -> list[np.ndarray]:
    out = [luma]
    for _ in range(levels):
        a = out[-1]
        h, w = a.shape
        out.append(a[: h // 2 * 2, : w // 2 * 2].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3)))
    return out


def bilinear(a: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = a.shape[:2]
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - x0, y - y0
    if a.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    xa, xb = np.clip(x0, 0, w - 1), np.clip(x0 + 1, 0, w - 1)
    ya, yb = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    return (a[ya, xa] * (1 - fx) * (1 - fy) + a[ya, xb] * fx * (1 - fy)
            + a[yb, xa] * (1 - fx) * fy + a[yb, xb] * fx * fy)


def lifted_triangles(tri: np.ndarray, uv: np.ndarray, points: np.ndarray, luma: np.ndarray) -> np.ndarray:
    """Per triangle: do MIN_SAMPLES or more of its samples change by more
    than LIFT at some mip level in LEVELS, away from their own 3D
    neighbourhood's mean at that scale too."""
    w = luma.shape[1]
    # the edges too: along a seam the mip levels average the triangle's edge
    # texels with the chart across the seam, a line the inside never shows
    s = np.linspace(0.0, 1.0, GRID)
    a, b = np.meshgrid(s, s)
    keep = a + b <= 1.0 + 1e-9
    bary = np.stack([1.0 - a[keep] - b[keep], a[keep], b[keep]], axis=-1)
    puv = np.einsum("sk,tkc->tsc", bary, uv[tri])
    p3 = np.einsum("sk,tkc->tsc", bary, points[tri])
    pyramid = mip_pyramid(luma, max(LEVELS))
    full = bilinear(pyramid[0], puv[..., 0] * w - 0.5, puv[..., 1] * w - 0.5)
    flat_p3, flat_full = p3.reshape(-1, 3), full.reshape(-1)
    tree = cKDTree(flat_p3)
    # one texel in metres, from the triangles' 3D and UV edge lengths
    e3 = np.linalg.norm(points[tri[:, 0]] - points[tri[:, 1]], axis=1)
    eu = np.linalg.norm(uv[tri[:, 0]] - uv[tri[:, 1]], axis=1) * w
    texel_m = float(np.median(e3[eu > 0.5] / eu[eu > 0.5]))
    hit = np.zeros(len(tri), dtype=bool)
    for k in LEVELS:
        wk = pyramid[k].shape[1]
        at_k = bilinear(pyramid[k], puv[..., 0] * wk - 0.5, puv[..., 1] * wk - 0.5)
        near = tree.query_ball_point(flat_p3, (2 ** k) * texel_m, workers=-1)
        local = np.fromiter((flat_full[i].mean() for i in near), dtype=np.float64, count=len(near)).reshape(full.shape)
        moved = (np.abs(at_k - full) > LIFT) & (np.abs(at_k - local) > LIFT)
        hit |= moved.sum(axis=1) >= MIN_SAMPLES
    return hit


def raster(corner_px: np.ndarray, shape: tuple[int, int], slack: float = 0.0):
    """(rows, cols, barycentric weights) of the texel centres inside a
    triangle (pixel coordinates, texel centres at +0.5), within `slack`."""
    h, w = shape
    xs, ys = corner_px[:, 0] - 0.5, corner_px[:, 1] - 0.5
    x0, x1 = max(int(np.floor(xs.min())) - 1, 0), min(int(np.ceil(xs.max())) + 1, w - 1)
    y0, y1 = max(int(np.floor(ys.min())) - 1, 0), min(int(np.ceil(ys.max())) + 1, h - 1)
    if x1 < x0 or y1 < y0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros((0, 3))
    gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
    denom = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
    if abs(denom) < 1e-12:
        return np.zeros(0, int), np.zeros(0, int), np.zeros((0, 3))
    l0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / denom
    l1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / denom
    bary = np.stack([l0, l1, 1.0 - l0 - l1], axis=-1)
    inside = (bary >= -slack).all(axis=-1)
    return gy[inside], gx[inside], bary[inside]


def first_fit(free: np.ndarray, rh: int, rw: int) -> tuple[int, int] | None:
    """Top-left of the first rh x rw block of `free` that is free throughout."""
    h, w = free.shape
    if rh > h or rw > w:
        return None
    s = np.zeros((h + 1, w + 1), dtype=np.int32)
    s[1:, 1:] = np.cumsum(np.cumsum(free, axis=0, dtype=np.int32), axis=1)
    block = s[rh:, rw:] - s[:-rh, rw:] - s[rh:, :-rw] + s[:-rh, :-rw]
    fits = np.argwhere(block == rh * rw)
    if not len(fits):
        return None
    return int(fits[0][0]), int(fits[0][1])


def surface_colour(texture: np.ndarray, old_px: np.ndarray, at: np.ndarray, position: np.ndarray | None,
                   valid: np.ndarray | None) -> np.ndarray:
    """The old texture at old pixel points (N, 2), its bilinear taps limited
    to texels baked within MISMATCH_M of the 3D points `at` (N, 3) - plain
    bilinear where none is."""
    h, w = texture.shape[:2]
    x, y = old_px[:, 0] - 0.5, old_px[:, 1] - 0.5
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - x0, y - y0
    total = np.zeros((len(x), texture.shape[2]))
    weight_sum = np.zeros(len(x))
    plain = np.zeros((len(x), texture.shape[2]))
    for dy, dx, wt in ((0, 0, (1 - fx) * (1 - fy)), (0, 1, fx * (1 - fy)), (1, 0, (1 - fx) * fy), (1, 1, fx * fy)):
        ty, tx = np.clip(y0 + dy, 0, h - 1), np.clip(x0 + dx, 0, w - 1)
        tap = texture[ty, tx]
        plain += tap * wt[:, None]
        ok = np.ones(len(x), dtype=bool)
        if position is not None:
            ok = valid[ty, tx] & (np.linalg.norm(position[ty, tx] - at, axis=1) < MISMATCH_M)
        total += tap * (wt * ok)[:, None]
        weight_sum += wt * ok
    out = plain.copy()
    some = weight_sum > 1e-6
    out[some] = total[some] / weight_sum[some, None]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--glb", required=True, help="a dog built from the template")
    parser.add_argument("--slug", required=True, help="its template, for the position map and protected texels")
    parser.add_argument("--size", default="small")
    parser.add_argument("--out", default=None, help="write here instead of over --glb")
    args = parser.parse_args()

    from api.glb import extract_base_texture, replace_base_texture

    glb = Path(args.glb)
    template = ROOT / "output" / "templates" / f"{args.slug}_{args.size}"
    position = np.load(template / "geometry" / "position.npy").astype(np.float32)
    valid = np.load(template / "geometry" / "valid.npy") > 0.5

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
    uv = data["TEXCOORD_0"].astype(np.float64)
    points = to_blender(data["POSITION"]).astype(np.float64)

    texture = np.asarray(extract_base_texture(glb).convert("RGB"), dtype=np.float32)
    h, w = texture.shape[:2]
    if position.shape[:2] != (h, w):
        raise SystemExit(f"[mip] texture {w}x{h} but the position map is {position.shape[1]}x{position.shape[0]}")
    luma = texture @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    hit = lifted_triangles(tri, uv, points, luma)
    span = (uv[tri].max(axis=1) - uv[tri].min(axis=1)).max(axis=1) * w
    protect_path = template / "protect.png"
    protected = np.zeros(len(tri), dtype=bool)
    if protect_path.exists():
        keep = np.asarray(Image.open(protect_path).convert("L").resize((w, h))) > 127
        cu = uv[tri].mean(axis=1)
        protected = keep[np.clip((cu[:, 1] * h).astype(int), 0, h - 1), np.clip((cu[:, 0] * w).astype(int), 0, w - 1)]
        protected |= keep[np.clip((uv[tri][..., 1] * h).astype(int), 0, h - 1),
                          np.clip((uv[tri][..., 0] * w).astype(int), 0, w - 1)].any(axis=1)
    chosen = np.nonzero(hit & ~protected & (span >= 0.5))[0]
    print(f"[mip] {int(hit.sum())} triangles change at mip {LEVELS}; {len(chosen)} to move "
          f"({int((hit & protected).sum())} protected, {int((hit & (span < 0.5)).sum())} flattened kept)")
    if not len(chosen):
        return

    # patches: chosen triangles that share a corner
    n = len(chosen)
    corner_of = tri[chosen]
    rows_i = np.repeat(np.arange(n), 3)
    inc = coo_matrix((np.ones(3 * n), (rows_i, corner_of.ravel())), shape=(n, uv.shape[0])).tocsr()
    _, label = connected_components(inc @ inc.T, directed=False)

    # free space: off every triangle's texels and the template's islands
    used = valid.copy()
    for t in range(len(tri)):
        r, c, _ = raster(uv[tri[t]] * [w, h], (h, w), slack=0.5)
        used[r, c] = True
    free = ndimage.distance_transform_edt(~used) > FREE_MARGIN_PX

    count = uv.shape[0]
    copies: list[int] = []
    copy_uv: list[np.ndarray] = []
    out_tex = texture.copy()
    moved_tris = skipped = 0
    patches = [chosen[label == k] for k in range(label.max() + 1)]
    patches.sort(key=lambda p: -np.ptp(uv[tri[p]].reshape(-1, 2), axis=0).prod())
    for patch in patches:
        verts = np.unique(tri[patch])
        old_px = uv[verts] * [w, h]
        lo = old_px.min(axis=0)
        size = np.ceil((old_px.max(axis=0) - lo) * SCALE).astype(int) + 1
        rh, rw = int(size[1]) + 2 * PAD_PX, int(size[0]) + 2 * PAD_PX
        spot = first_fit(free, rh, rw)
        if spot is None:
            skipped += len(patch)
            continue
        top, left = spot
        origin = np.array([left + PAD_PX, top + PAD_PX], dtype=np.float64)
        new_px = (old_px - lo) * SCALE + origin
        index_of = {}
        for v, p in zip(verts.tolist(), new_px):
            copies.append(v)
            copy_uv.append((p / [w, h]).astype(np.float32))
            index_of[v] = count + len(copies) - 1
        new_of = dict(zip(verts.tolist(), new_px))
        old_of = dict(zip(verts.tolist(), old_px))
        painted = np.zeros((rh, rw), dtype=bool)
        block = out_tex[top:top + rh, left:left + rw]
        for t in patch:
            corners = tri[t].tolist()
            r, c, bary = raster(np.stack([new_of[v] for v in corners]), (h, w), slack=0.02)
            inside = (r >= top) & (r < top + rh) & (c >= left) & (c < left + rw)
            r, c, bary = r[inside], c[inside], bary[inside]
            if not len(r):
                # too small to cover a texel centre: its centre's texel
                centre = np.stack([new_of[v] for v in corners]).mean(axis=0)
                r = np.array([min(max(int(centre[1]), top), top + rh - 1)])
                c = np.array([min(max(int(centre[0]), left), left + rw - 1)])
                bary = np.full((1, 3), 1.0 / 3.0)
            bary = np.clip(bary, 0.0, None)
            bary /= bary.sum(axis=1, keepdims=True)
            old_points = bary @ np.stack([old_of[v] for v in corners])
            at = bary @ points[corners]
            block[r - top, c - left] = surface_colour(texture, old_points, at, position, valid)
            painted[r - top, c - left] = True
            for j, v in enumerate(corners):
                tri[t, j] = index_of[v]
        if painted.any():
            near = ndimage.distance_transform_edt(~painted, return_distances=False, return_indices=True)
            block[:] = block[near[0], near[1]]
        out_tex[top:top + rh, left:left + rw] = block
        taken = np.zeros_like(free)
        taken[max(top - FREE_MARGIN_PX, 0):top + rh + FREE_MARGIN_PX,
              max(left - FREE_MARGIN_PX, 0):left + rw + FREE_MARGIN_PX] = True
        free &= ~taken
        moved_tris += len(patch)
    print(f"[mip] moved {moved_tris} triangles in {len(patches)} patches at x{SCALE}"
          f"{f', {skipped} left (no room)' if skipped else ''}; {len(copies)} corners copied")
    if not copies:
        return

    source = np.asarray(copies, dtype=np.int64)
    for name, idx in attributes.items():
        values = data[name]
        extra = np.stack(copy_uv).astype(values.dtype) if name == "TEXCOORD_0" else values[source]
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
    replace_base_texture(out, Image.fromarray(np.clip(out_tex, 0, 255).astype(np.uint8)), out)
    print(f"[mip] wrote {out}")


if __name__ == "__main__":
    main()
