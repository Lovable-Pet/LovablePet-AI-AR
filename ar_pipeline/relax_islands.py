"""Relax UV islands that are squashed at one end, in place, and carry every
UV-space template asset along.

WHY
---
The jindo's front legs (islands 74 and 791 of the generated template) were
unwrapped with their lower third squashed: 0.6 mm of dog per texel at the
top, up to 21 mm at the bottom (the whole template sits at 0.8). The fur
drawn there is stretched and smeared, and where the island meets the paw
island below it the hair pattern changes size at once - a band round each
foreleg (user on eval v4 e05: "a line on the front legs"). Colour work on
the texture cannot fix it: the texels are not there.

HOW
---
1. Find islands whose own stretch varies: 95th percentile over median of
   (mm of surface per texel) above SQUASH_RATIO, at least MIN_TRIANGLES.
2. Take each island's largest connected piece to Blender and run
   uv.minimize_stretch from the EXISTING layout. That relaxation starts
   flip-free and stays close to the old shape, so the piece fits where it
   was. (Tried: SLIM unwrap - even, but folded 7.5k texels of one leg over
   itself; ABF - as squashed as before; conformal - worse; a numpy ARAP -
   23 flipped triangles.)
3. Align the result onto the old piece (winding, Procrustes rotation, the
   old UV area) and check it overlaps nothing else and stays in the atlas.
4. Rewrite the GLB's TEXCOORD_0 for those vertices and remap every UV-space
   asset: each texel of the new layout takes the value at the same surface
   point in the old layout (per-triangle barycentric map); fur directions
   are turned by the triangle's old->new Jacobian. Texels the piece used to
   cover and no longer does are cleared.

Usage:
    python -m ar_pipeline.relax_islands --slug jindo --size small [--dry-run]
Back up output/templates/<slug>_<size>/ and output/skeleton_<slug>_<size>.glb
first; this edits them in place.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from pygltflib import GLTF2
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.assets import BLENDER  # noqa: E402
from api.glb import replace_base_texture  # noqa: E402
from api.mesh import TemplateMesh  # noqa: E402

SQUASH_RATIO = 4.0
MIN_TRIANGLES = 200
RELAX_ROUNDS = 30          # x 100 minimize_stretch iterations
PAD_PX = 4

BLENDER_RELAX = r'''
import sys, numpy as np, bpy
argv = sys.argv[sys.argv.index("--") + 1:]
d = np.load(argv[0]); rounds = int(argv[2])
bpy.ops.wm.read_factory_settings(use_empty=True)
me = bpy.data.meshes.new("piece"); me.from_pydata(d["P"].tolist(), [], d["faces"].tolist()); me.update()
ob = bpy.data.objects.new("piece", me); bpy.context.collection.objects.link(ob)
bpy.context.view_layer.objects.active = ob; ob.select_set(True)
uvl = me.uv_layers.new(name="UV")
for poly in me.polygons:
    for li in poly.loop_indices:
        v = me.loops[li].vertex_index
        uvl.data[li].uv = (float(d["uv"][v, 0]) / 2048.0, 1.0 - float(d["uv"][v, 1]) / 2048.0)
bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.select_all(action="SELECT"); bpy.ops.uv.select_all(action="SELECT")
for _ in range(rounds):
    bpy.ops.uv.minimize_stretch(fill_holes=True, blend=0.0, iterations=100)
bpy.ops.object.mode_set(mode="OBJECT")
out = np.zeros((len(me.vertices), 2))
for poly in me.polygons:
    for li in poly.loop_indices:
        out[me.loops[li].vertex_index] = uvl.data[li].uv
np.save(argv[1], out)
'''


def signed_area(uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = uv[faces[:, 0]], uv[faces[:, 1]], uv[faces[:, 2]]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


def stretch_mm(mesh: TemplateMesh, uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    p = mesh.positions.astype(np.float64)
    area3 = 0.5 * np.linalg.norm(np.cross(p[faces[:, 1]] - p[faces[:, 0]], p[faces[:, 2]] - p[faces[:, 0]]), axis=1)
    return np.sqrt(area3 / np.maximum(np.abs(signed_area(uv, faces)), 1e-12)) * 1000.0


def coverage(uv: np.ndarray, faces: np.ndarray, size: int) -> np.ndarray:
    """texel -> how many of `faces` cover it (texel centres at integer coords,
    as TemplateMesh.rasterize)"""
    count = np.zeros((size, size), np.int16)
    for tri in faces:
        for_each_texel(uv, tri, size, lambda ys, xs, l: count.__setitem__((ys, xs), count[ys, xs] + 1))
    return count


def for_each_texel(uv, tri, size, fn):
    xs, ys = uv[tri, 0], uv[tri, 1]
    x0, x1 = max(int(np.floor(xs.min())), 0), min(int(np.ceil(xs.max())), size - 1)
    y0, y1 = max(int(np.floor(ys.min())), 0), min(int(np.ceil(ys.max())), size - 1)
    if x1 < x0 or y1 < y0:
        return
    den = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
    if abs(den) < 1e-12:
        return
    gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
    l0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / den
    l1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / den
    l2 = 1.0 - l0 - l1
    inside = (l0 >= -1e-3) & (l1 >= -1e-3) & (l2 >= -1e-3)
    if inside.any():
        fn(gy[inside], gx[inside], np.stack([l0[inside], l1[inside], l2[inside]], 1))


def find_squashed(mesh: TemplateMesh) -> tuple[list[np.ndarray], np.ndarray]:
    """(triangle index arrays: the largest connected piece of each squashed
    island, the island label of every triangle)"""
    size = mesh.size[0]
    covered = ~np.isnan(mesh.rasterize(np.ones(len(mesh.uvs), np.float32), fill=np.nan))
    label, _ = ndimage.label(covered)
    centre = mesh.uvs[mesh.triangles].mean(1).astype(int)
    island = label[np.clip(centre[:, 1], 0, size - 1), np.clip(centre[:, 0], 0, size - 1)]
    pieces = []
    for k in np.unique(island):
        tidx = np.flatnonzero(island == k)
        if k == 0 or len(tidx) < MIN_TRIANGLES:
            continue
        tri = mesh.triangles[tidx]
        n = len(mesh.uvs)
        adj = sparse.coo_matrix((np.ones(len(tri) * 3), (np.r_[tri[:, 0], tri[:, 1], tri[:, 2]],
                                                         np.r_[tri[:, 1], tri[:, 2], tri[:, 0]])), shape=(n, n))
        _, comp = connected_components(adj, directed=False)
        tc = comp[tri[:, 0]]
        main = tidx[tc == np.bincount(tc).argmax()]
        if len(main) < MIN_TRIANGLES:
            continue
        s = stretch_mm(mesh, mesh.uvs.astype(np.float64), mesh.triangles[main])
        ratio = np.percentile(s, 95) / np.median(s)
        if ratio > SQUASH_RATIO:
            print(f"[relax] island {k}: {len(main)} triangles, mm/texel median {np.median(s):.2f} "
                  f"p95 {np.percentile(s, 95):.2f} (x{ratio:.1f})")
            pieces.append(main)
    return pieces, island


def relax_piece(mesh: TemplateMesh, main: np.ndarray, work: Path) -> tuple[np.ndarray, np.ndarray]:
    tri = mesh.triangles[main]
    verts = np.unique(tri)
    remap = -np.ones(len(mesh.uvs), int)
    remap[verts] = np.arange(len(verts))
    faces = remap[tri]
    old = mesh.uvs[verts].astype(np.float64)
    src, dst, script = work / f"piece_{main[0]}.npz", work / f"relaxed_{main[0]}.npy", work / "relax.py"
    script.write_text(BLENDER_RELAX, encoding="utf-8")
    np.savez(src, P=mesh.positions[verts], faces=faces, uv=old)
    subprocess.run([str(BLENDER), "--background", "--python-exit-code", "1", "--python", str(script), "--",
                    str(src), str(dst), str(RELAX_ROUNDS)], check=True, capture_output=True)
    raw = np.load(dst)
    new = np.stack([raw[:, 0] * 2048.0, (1.0 - raw[:, 1]) * 2048.0], 1)
    # onto the old piece: winding, rotation, area, centre
    if np.sign(signed_area(new, faces).sum()) != np.sign(signed_area(old, faces).sum()):
        new[:, 0] *= -1
    a, b = new - new.mean(0), old - old.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    placed = a @ rot
    placed *= np.sqrt(np.abs(signed_area(old, faces)).sum() / np.abs(signed_area(placed, faces)).sum())
    return verts, placed + old.mean(0)


def remap_assets(template_dir: Path, mesh: TemplateMesh, faces: np.ndarray, old_uv: np.ndarray,
                 new_uv: np.ndarray) -> list[str]:
    size = mesh.size[0]
    # every texel of the new layout: its old (x, y), and the Jacobian old->new
    src_x = np.full((size, size), np.nan, np.float32)
    src_y = np.full((size, size), np.nan, np.float32)
    jac = np.zeros((size, size, 2, 2), np.float32)
    for tri in faces:
        o = old_uv[tri]
        n = new_uv[tri]
        do = np.stack([o[1] - o[0], o[2] - o[0]], 1)
        dn = np.stack([n[1] - n[0], n[2] - n[0]], 1)
        j = dn @ np.linalg.pinv(do) if abs(np.linalg.det(do)) > 1e-9 else np.eye(2)

        def put(ys, xs, lam, o=o, j=j):
            p = lam @ o
            src_x[ys, xs], src_y[ys, xs] = p[:, 0], p[:, 1]
            jac[ys, xs] = j
        for_each_texel(new_uv, tri, size, put)
    new_mask = ~np.isnan(src_x)
    old_mask = np.zeros((size, size), bool)
    for tri in faces:
        for_each_texel(old_uv, tri, size, lambda ys, xs, lam: old_mask.__setitem__((ys, xs), True))
    ys, xs = np.nonzero(new_mask)
    coords = np.stack([src_y[ys, xs], src_x[ys, xs]])
    # the padding ring just outside the new piece that no other island owns
    geometry = template_dir / "geometry"
    valid_old = np.load(geometry / "valid.npy") > 0.5
    others = valid_old & ~old_mask
    pad = ndimage.binary_dilation(new_mask, iterations=PAD_PX) & ~new_mask & ~others
    near = ndimage.distance_transform_edt(~new_mask, return_distances=False, return_indices=True)

    def remap_array(arr: np.ndarray, clear, order: int, fill_pad: bool) -> np.ndarray:
        out = arr.copy()
        out[old_mask & ~new_mask] = clear
        chans = arr.reshape(size, size, -1)
        vals = np.stack([ndimage.map_coordinates(chans[..., c].astype(np.float32), coords, order=order,
                                                 mode="nearest") for c in range(chans.shape[-1])], -1)
        flat = out.reshape(size, size, -1)
        flat[ys, xs] = vals.astype(arr.dtype)
        if fill_pad:
            flat[pad] = flat[near[0][pad], near[1][pad]]
        return flat.reshape(arr.shape)

    changed: list[str] = []
    for folder in ("masks", "masks_source", "masks_bones", "masks_face_hand"):
        for path in sorted((template_dir / folder).glob("*.png")):
            im = Image.open(path)
            arr = np.asarray(im)
            Image.fromarray(remap_array(arr, 0, 1, True), mode=im.mode).save(path)
            changed.append(str(path.relative_to(template_dir)))
    for name, clear in (("protect.png", 0), ("base_color.png", None), ("uv_layout.png", None)):
        path = template_dir / name
        if path.exists():
            im = Image.open(path)
            arr = np.asarray(im)
            c = arr[old_mask & ~new_mask] if clear is None else clear
            Image.fromarray(remap_array(arr, c, 1, True), mode=im.mode).save(path)
            changed.append(name)
    for name, order in (("position.npy", 1), ("normal.npy", 1), ("normal_raw.npy", 1), ("valid.npy", 0)):
        path = geometry / name
        arr = np.load(path)
        out = remap_array(arr, 0, order, False)
        if name.startswith("normal"):
            length = np.linalg.norm(out, axis=-1, keepdims=True)
            out = np.where(length > 1e-6, out / np.maximum(length, 1e-6), out)
        np.save(path, out.astype(arr.dtype))
        changed.append(f"geometry/{name}")
    # fur direction: vectors turned by the Jacobian, defined flag nearest
    path = template_dir / "fur_direction.png"
    data = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
    vec = np.stack([data[..., 0] / 255 * 2 - 1, data[..., 1] / 255 * 2 - 1], -1)
    out = data.copy()
    out[old_mask & ~new_mask] = (127.5, 127.5, 0)
    vx = ndimage.map_coordinates(vec[..., 0], coords, order=1, mode="nearest")
    vy = ndimage.map_coordinates(vec[..., 1], coords, order=1, mode="nearest")
    defined = ndimage.map_coordinates(data[..., 2], coords, order=0, mode="nearest")
    turned = np.einsum("nij,nj->ni", jac[ys, xs], np.stack([vx, vy], 1))
    length = np.linalg.norm(turned, axis=1, keepdims=True)
    turned = np.where(length > 1e-6, turned / np.maximum(length, 1e-6), 0.0)
    out[ys, xs, 0] = (turned[:, 0] + 1) / 2 * 255
    out[ys, xs, 1] = (turned[:, 1] + 1) / 2 * 255
    out[ys, xs, 2] = defined
    Image.fromarray(np.clip(np.round(out), 0, 255).astype(np.uint8), mode="RGB").save(path)
    changed.append("fur_direction.png")
    return changed


def write_glb_uvs(glb: Path, verts: np.ndarray, texel_uv: np.ndarray, size: int) -> None:
    gltf = GLTF2().load(str(glb))
    best = max(((gltf.accessors[p.attributes.POSITION].count, p) for m in gltf.meshes for p in m.primitives),
               key=lambda x: x[0])[1]
    acc = gltf.accessors[best.attributes.TEXCOORD_0]
    view = gltf.bufferViews[acc.bufferView]
    blob = bytearray(gltf.binary_blob())
    offset = (view.byteOffset or 0) + (acc.byteOffset or 0)
    stride = view.byteStride or 8
    if stride != 8 or acc.componentType != 5126:
        raise ValueError("expected tightly packed float32 TEXCOORD_0")
    uv = np.frombuffer(bytes(blob[offset:offset + acc.count * 8]), np.float32).reshape(-1, 2).copy()
    uv[verts] = (texel_uv / (size - 1)).astype(np.float32)
    blob[offset:offset + acc.count * 8] = uv.tobytes()
    acc.min, acc.max = uv.min(0).tolist(), uv.max(0).tolist()
    gltf.set_binary_blob(bytes(blob))
    gltf.save(str(glb))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", required=True)
    parser.add_argument("--size", default="small")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    glb = ROOT / "output" / f"skeleton_{args.slug}_{args.size}.glb"
    template_dir = ROOT / "output" / "templates" / f"{args.slug}_{args.size}"
    size = Image.open(template_dir / "base_color.png").size[0]
    mesh = TemplateMesh.load(glb, (size, size))
    pieces, island = find_squashed(mesh)
    if not pieces:
        print("[relax] no squashed island")
        return
    old_uv = mesh.uvs.astype(np.float64)
    new_uv = old_uv.copy()
    moved = np.zeros(len(mesh.uvs), bool)
    accepted = []
    with tempfile.TemporaryDirectory() as tmp:
        for main_tris in pieces:
            verts, placed = relax_piece(mesh, main_tris, Path(tmp))
            faces = mesh.triangles[main_tris]
            trial = new_uv.copy()
            trial[verts] = placed
            before, after = stretch_mm(mesh, old_uv, faces), stretch_mm(mesh, trial, faces)
            flips = int((np.sign(signed_area(trial, faces)) != np.sign(signed_area(old_uv, faces).sum())).sum())
            # one piece at a time against everything else as it now stands:
            # other islands must stay 2 px clear; the small loose shells that
            # share the piece's island (fur clumps beside the leg) sit within
            # 2 px of it by construction, so only real overlap with them counts,
            # and up to 1% of the piece: a shell under the leg takes the leg's
            # colour there, which is what it is on the dog
            rest = ~np.isin(mesh.triangles, verts).any(axis=1)
            label = island[main_tris[0]]
            others = coverage(new_uv, mesh.triangles[rest & (island != label)], size) > 0
            siblings = coverage(new_uv, mesh.triangles[rest & (island == label)], size) > 0
            landed = coverage(trial, faces, size)
            clash = int(((landed > 0) & ndimage.binary_dilation(others, iterations=2)).sum())
            shells_before = int(((coverage(old_uv, faces, size) > 0) & siblings).sum())
            shells_after = int(((landed > 0) & siblings).sum())
            self_before = int((coverage(old_uv, faces, size) > 1).sum())
            self_after = int((landed > 1).sum())
            inside = bool((placed >= 0).all() and (placed <= size - 1).all())
            ok = (flips == 0 and clash == 0 and inside and self_after <= max(2 * self_before, 500)
                  and shells_after <= shells_before + 0.01 * int((landed > 0).sum()))
            print(f"[relax]   {len(faces)} triangles: p95 {np.percentile(before, 95):.2f} -> "
                  f"{np.percentile(after, 95):.2f} mm/texel, max {before.max():.1f} -> {after.max():.1f}, "
                  f"flipped {flips}, on other islands {clash}, on own shells {shells_before} -> {shells_after}, "
                  f"self-overlap {self_before} -> {self_after}: {'ACCEPT' if ok else 'skip'}")
            if ok:
                new_uv = trial
                moved[verts] = True
                accepted.append(faces)
    if not accepted:
        print("[relax] nothing accepted; nothing written")
        return
    faces = np.concatenate(accepted)
    if args.dry_run:
        return
    changed = remap_assets(template_dir, mesh, faces, old_uv, new_uv)
    write_glb_uvs(glb, np.unique(faces), new_uv[np.unique(faces)], size)
    replace_base_texture(glb, Image.open(template_dir / "base_color.png"), glb)
    print(f"[relax] wrote {glb.name} and {len(changed)} assets")


if __name__ == "__main__":
    main()
