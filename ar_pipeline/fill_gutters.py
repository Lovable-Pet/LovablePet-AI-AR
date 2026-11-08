"""Fill a template texture's empty gutters, so no triangle samples black.

WHY
---
The generated jindo puppy's atlas leaves everything between its UV charts
black (1.26 M of 1.86 M unbaked texels), and some of its triangles do not
stay inside one chart: seven on the left of the chest have their corners on
two charts and their insides across the gap between them. And a few baked
texels in the white coat were left pure black (8 on the right thigh, 3 on
an ear). Rendered, they were black streaks and dots in the white coat
(user, 11-06: "진도강아지 가슴쪽이랑 앞발 옆쪽에 검은색 박혀있는데 이거
수정좀"); the coat's own texels have no black there.

HOW
---
Every coat triangle is sampled densely in UV; each unbaked texel a triangle
samples (black, or the dark grey a chart's edge faded into) gets the 3D
point it shows (barycentric on the rest mesh) and takes the colour of the
baked texels nearest that point on the surface (the eyes, nose and pads
too, where it lies on them: from the coat alone a white crack ran across
the nose). So do small islands of pure black baked texels in a light coat
(holes - not in the paws, which paint_pads does, nor a dark lip line).
Then the rest of the gutter takes its nearest chart's colour, for the mip
levels a far camera reads. The charts' own colours are otherwise
untouched, and a second run on the same charts changes nothing.

    python -m ar_pipeline.fill_gutters --slug jindo_puppy            # preview only
    python -m ar_pipeline.fill_gutters --slug jindo_puppy --install  # GLB and base_color.png

--install backs the GLB and base_color.png up to work/backup_gutters_<slug>/
first (once). The template assets go into the next release (assets-v2).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from pygltflib import GLTF2
from scipy import ndimage
from scipy.spatial import cKDTree

from ar_pipeline.gltf_rig import read
from api.glb import extract_base_texture, replace_base_texture

# a texel this dark (sRGB max) counts as empty
EMPTY_MAX = 24
# a baked empty texel is a hole in the coat when its island is this small
# and the coat within HOLE_REACH_M of it is this light (median L*)
HOLE_MAX_TEXELS = 50
HOLE_REACH_M = 0.008
HOLE_COAT_L = 70.0
# samples per texel of a triangle's UV extent (and at least this many)
SAMPLES_PER_TEXEL = 2.0
MIN_SAMPLES = 64
# surface colour: the mean of this many baked texels nearest the point
NEAREST = 12


def coat_triangles(glb: Path) -> tuple[np.ndarray, np.ndarray]:
    """(corner UVs (T, 3, 2), corner points in the bake's frame (T, 3, 3)) of
    the coat material's triangles: glTF's +z front, y up -> x, -z, y."""
    gltf = GLTF2().load(str(glb))
    blob = gltf.binary_blob()
    coat = next(i for i, m in enumerate(gltf.materials) if m.name == "coat")
    uvs, points = [], []
    for mesh in gltf.meshes:
        for prim in mesh.primitives:
            if prim.material != coat:
                continue
            pos = read(gltf, blob, prim.attributes.POSITION).astype(np.float64)
            uv = read(gltf, blob, prim.attributes.TEXCOORD_0).astype(np.float64)
            tri = read(gltf, blob, prim.indices).reshape(-1, 3).astype(np.int64)
            uvs.append(uv[tri])
            points.append(np.stack([pos[..., 0], -pos[..., 2], pos[..., 1]], -1)[tri])
    return np.concatenate(uvs), np.concatenate(points)


def sampled_gutter(uvs: np.ndarray, points: np.ndarray, empty: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The empty texels the triangles sample, and the 3D point each shows."""
    h, w = empty.shape
    px = uvs * np.array([w, h])
    extent = np.ptp(px, axis=1).max(1)
    hits, where = [], []
    for count in np.unique(np.clip((extent * SAMPLES_PER_TEXEL) ** 2, MIN_SAMPLES, 4096).astype(int)):
        sel = np.nonzero(np.clip((extent * SAMPLES_PER_TEXEL) ** 2, MIN_SAMPLES, 4096).astype(int) == count)[0]
        n = int(np.ceil(np.sqrt(2 * count)))
        a, b = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        keep = a + b <= 1.0
        bary = np.stack([1 - a[keep] - b[keep], a[keep], b[keep]], -1)          # (S, 3)
        p_uv = np.einsum("sk,tkc->tsc", bary, px[sel])
        col = np.clip(p_uv[..., 0].astype(int), 0, w - 1)
        row = np.clip(p_uv[..., 1].astype(int), 0, h - 1)
        on = empty[row, col]
        if on.any():
            p3 = np.einsum("sk,tkc->tsc", bary, points[sel])
            hits.append(np.stack([row[on], col[on]], -1))
            where.append(p3[on])
    if not hits:
        return np.zeros((0, 2), int), np.zeros((0, 3))
    hits, where = np.concatenate(hits), np.concatenate(where)
    # one point per texel (the first sample that landed there)
    _, first = np.unique(hits[:, 0] * w + hits[:, 1], return_index=True)
    return hits[first], where[first]


def fill(slug: str, source: Path | None = None) -> tuple[Image.Image, Image.Image, list[str]]:
    tpl = Path(f"output/templates/{slug}_small")
    glb = Path(f"output/skeleton_{slug}_small.glb")
    texture = np.asarray(extract_base_texture(source or glb).convert("RGB")).copy()
    h, w = texture.shape[:2]
    geometry = tpl / "geometry"
    position = np.load(geometry / "position.npy")
    valid = np.load(geometry / "valid.npy") > 0.5
    if valid.shape != (h, w):
        raise SystemExit(f"{slug}: bake {valid.shape} and texture {(h, w)} differ")
    mask = lambda name: np.asarray(Image.open(name).convert("L").resize((w, h))) > 127
    protect = mask(tpl / "protect.png")
    features, paws = mask(tpl / "masks" / "_features.png"), mask(tpl / "masks" / "paws.png")
    report = [f"{slug}: {int((~valid).sum())} unbaked texels, {int((~valid & (texture.max(-1) <= EMPTY_MAX)).sum())} "
              "of them empty (black)"]

    uvs, points = coat_triangles(glb)
    texels, where = sampled_gutter(uvs, points, ~valid)
    # holes: small black islands of baked texels in a light coat
    luma_l = 116.0 * np.cbrt(np.clip(((texture / 255.0) ** 2.2) @ np.array([0.2126, 0.7152, 0.0722]), 0.008856, None)) - 16.0
    black = valid & (texture.max(-1) <= EMPTY_MAX) & ~protect & ~features & ~paws
    labels, count = ndimage.label(black)
    sizes = ndimage.sum(black, labels, range(1, count + 1))
    coat = valid & ~protect & ~black
    tree = cKDTree(position[coat])
    holes = np.zeros_like(black)
    for index in np.nonzero(sizes <= HOLE_MAX_TEXELS)[0] + 1:
        island = labels == index
        near = tree.query_ball_point(position[island].mean(0), HOLE_REACH_M)
        if near and float(np.median(luma_l[coat][near])) > HOLE_COAT_L:
            holes |= island
    targets = np.concatenate([texels, np.argwhere(holes)])
    where = np.concatenate([where, position[holes].astype(np.float64)])
    # the colour of the surface there, the eyes, nose and pads included (from
    # the coat alone, the gutter texels the nose's triangles sample came out
    # white and a light crack ran across the nose)
    shown = valid & ~holes
    tree = cKDTree(position[shown])
    colours = texture[shown].astype(np.float64)
    out = texture.copy()
    if len(targets):
        _, near = tree.query(where, k=NEAREST, workers=-1)
        out[targets[:, 0], targets[:, 1]] = np.clip(colours[near].mean(1) + 0.5, 0, 255).astype(np.uint8)
    report.append(f"{len(texels)} unbaked texels sampled by triangles and {int(holes.sum())} in black holes "
                  "coloured from the surface they show")

    # the rest of the gutter: its nearest coloured texel's colour
    rest = ~valid
    rest[texels[:, 0], texels[:, 1]] = False
    idx = ndimage.distance_transform_edt(rest, return_distances=False, return_indices=True)
    out[rest] = out[idx[0][rest], idx[1][rest]]
    report.append(f"{int(rest.sum())} other unbaked texels padded from the nearest chart")
    filled = Image.fromarray(out)

    # preview: the sampled texels' neighbourhoods before and after
    cells = []
    if len(targets):
        order = np.argsort(targets[:, 0])
        for k in order[:: max(1, len(order) // 3)][:3]:
            r, c = targets[k]
            box = (slice(max(r - 24, 0), r + 24), slice(max(c - 24, 0), c + 24))
            for src in (texture, out):
                cell = Image.fromarray(src[box])
                cells.append(cell.resize((cell.width * 6, cell.height * 6), Image.NEAREST))
    preview = Image.new("RGB", (max(sum(c.width + 12 for c in cells), 1), max((c.height for c in cells), default=1)), "white")
    x = 0
    for cell in cells:
        preview.paste(cell, (x, 0))
        x += cell.width + 12
    return filled, preview, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slug", required=True)
    parser.add_argument("--out", default=None, help="where the preview goes (default work/gutters_<slug>/)")
    parser.add_argument("--install", action="store_true", help="write into the template GLB and base_color.png")
    args = parser.parse_args()
    out = Path(args.out or f"work/gutters_{args.slug}")
    out.mkdir(parents=True, exist_ok=True)
    filled, preview, report = fill(args.slug)
    filled.save(out / "base_color.png")
    preview.save(out / "gutters_preview.png")
    print("\n".join(report), f"\npreview {out / 'gutters_preview.png'}")
    if args.install:
        glb = Path(f"output/skeleton_{args.slug}_small.glb")
        png = Path(f"output/templates/{args.slug}_small/base_color.png")
        backup = Path(f"work/backup_gutters_{args.slug}")
        if not backup.exists():
            backup.mkdir(parents=True)
            shutil.copy(glb, backup / glb.name)
            shutil.copy(png, backup / png.name)
        replace_base_texture(glb, filled, out / glb.name)
        shutil.copy(out / glb.name, glb)
        filled.save(png)
        print(f"installed into {glb} and {png} (backup {backup})")


if __name__ == "__main__":
    main()
