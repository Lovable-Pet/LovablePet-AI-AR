"""Fill the template's baked surface maps where its UVs now cover texels the
bake left out.

WHY
---
The position, normal and valid maps (bake_geometry, September) were baked
before the UV fixes of 10-31 and 11-01 (fix_uv_overlaps, relax_islands) moved
some vertices. 1,483 texels the jindo's triangles now cover were never baked:
cracks a texel or two wide inside islands. Everything that reads the surface
skips them - the light underside is painted only on baked texels (texture,
SOFT_LIGHT_*) - so on a black-and-white coat a crack across a white leg
showed as a dark broken ring round the wrist (user, 11-05: "흑백 경계면에 알
수 없는 선들").

HOW
---
The current mesh is rasterised into UV space: a covered texel the bake missed
takes the position interpolated across its triangle, and the normal of the
nearest baked texel. Texels under a stray triangle (seams.fill_stray_triangles:
one whose UV lands on another part of the dog) stay unbaked - baked, they would
take that triangle's own position and the masks there, which belong to the
other part, would paint it. The maps are backed up to
work/backup_geometry_<slug>/ the first time.

    python -m ar_pipeline.patch_geometry --slug jindo
    python -m ar_pipeline.patch_geometry --slug jindo --rebake

--rebake: a texel the bake gave a place more than REBAKE_OFF_M from where
its one current triangle puts it (the UV fixes moved the triangle, the
bake still holds the place it used to map) takes the triangle's place, the
normal there, and every mask's value and the base colour's from the nearest
texel on the dog whose bake is right. Left as baked, a triangle straddling a
colour edge showed the colours a centimetre away: straight-edged white
wedges in a black-and-white dog's black upper arm (user, 11-05: "09의
오른쪽 팔에 살짝 남아있는 거"). 65,000 texels on the jindo (5,400
triangles over 5 mm). protect.png (eyes, nose, pads) is left as it is.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent


def stray_triangles(mesh, position: np.ndarray, valid: np.ndarray) -> np.ndarray:
    from api.seams import STRAY_TRIANGLE_M

    height, width = valid.shape
    tri = mesh.triangles
    centre = mesh.positions[tri].mean(axis=1)
    cuv = mesh.uvs[tri].mean(axis=1)
    col = np.clip(np.round(cuv[:, 0]).astype(int), 0, width - 1)
    row = np.clip(np.round(cuv[:, 1]).astype(int), 0, height - 1)
    nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    landed = position[nearest[0][row, col], nearest[1][row, col]]
    return np.flatnonzero(np.linalg.norm(landed - centre, axis=1) > STRAY_TRIANGLE_M)


def patch(template_glb: Path, geometry_dir: Path) -> dict:
    from api.mesh import TemplateMesh

    info = json.loads((geometry_dir / "geometry.json").read_text(encoding="utf-8"))
    size = tuple(info["size"])
    position = np.load(geometry_dir / "position.npy")
    valid_raw = np.load(geometry_dir / "valid.npy")
    valid = valid_raw > 0.5
    mesh = TemplateMesh.load(template_glb, size)
    stray = stray_triangles(mesh, position, valid)
    keep = np.ones(len(mesh.triangles), dtype=bool)
    keep[stray] = False
    # every covered texel's surface point, from the triangles that belong where they land
    mesh_kept = TemplateMesh(mesh.positions, mesh.uvs, mesh.triangles[keep], mesh.merged, mesh.joints, mesh.size)
    surface = mesh_kept.rasterize(mesh.positions, fill=np.nan)
    covered = ~np.isnan(surface[..., 0])
    missed = covered & ~valid
    report = {"covered": int(covered.sum()), "baked": int(valid.sum()), "missed": int(missed.sum()),
              "stray_triangles": int(len(stray))}
    if not missed.any():
        return report
    position = position.copy()
    position[missed] = surface[missed]
    nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    for name in ("normal.npy", "normal_raw.npy"):
        path = geometry_dir / name
        if path.exists():
            normal = np.load(path)
            normal[missed] = normal[nearest[0][missed], nearest[1][missed]]
            np.save(path, normal)
    valid_out = valid_raw.copy()
    valid_out[missed] = 1.0
    np.save(geometry_dir / "position.npy", position)
    np.save(geometry_dir / "valid.npy", valid_out)
    return report


REBAKE_OFF_M = 0.003
# texels two triangles cover count as one place when those agree this well
REBAKE_AGREE_M = 0.001
# a texel's bake counts as right within this
REBAKE_GOOD_M = 0.0015


def raster_owner(mesh, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per texel: the last triangle covering it (-1 none), how many do, its
    barycentric weights in that triangle, and the place the first one
    covering it gives (to tell two triangles meeting at an edge, which agree,
    from two charts overlapping, which do not)."""
    width, height = size
    owner = np.full((height, width), -1, dtype=np.int32)
    count = np.zeros((height, width), dtype=np.uint8)
    bary = np.zeros((height, width, 3), dtype=np.float32)
    first = np.full((height, width, 3), np.nan, dtype=np.float32)
    uv = mesh.uvs
    for k, tri in enumerate(mesh.triangles):
        xs, ys = uv[tri, 0], uv[tri, 1]
        x0, x1 = max(int(np.floor(xs.min())), 0), min(int(np.ceil(xs.max())), width - 1)
        y0, y1 = max(int(np.floor(ys.min())), 0), min(int(np.ceil(ys.max())), height - 1)
        denom = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if x1 < x0 or y1 < y0 or abs(denom) < 1e-12:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        l0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / denom
        l1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / denom
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -1e-3) & (l1 >= -1e-3) & (l2 >= -1e-3)
        if not inside.any():
            continue
        rows, cols = gy[inside], gx[inside]
        weights = np.stack([l0[inside], l1[inside], l2[inside]], axis=-1)
        fresh = owner[rows, cols] < 0
        first[rows[fresh], cols[fresh]] = weights[fresh] @ mesh.positions[tri]
        count[rows, cols] = np.minimum(count[rows, cols].astype(np.int32) + 1, 255).astype(np.uint8)
        owner[rows, cols] = k
        bary[rows, cols] = weights
    return owner, count, bary, first


def rebake(template_glb: Path, template_dir: Path) -> dict:
    from PIL import Image
    from scipy.spatial import cKDTree
    from api.mesh import TemplateMesh

    geometry_dir = template_dir / "geometry"
    info = json.loads((geometry_dir / "geometry.json").read_text(encoding="utf-8"))
    size = tuple(info["size"])
    position = np.load(geometry_dir / "position.npy")
    valid = np.load(geometry_dir / "valid.npy") > 0.5
    mesh = TemplateMesh.load(template_glb, size)
    owner, count, bary, first = raster_owner(mesh, size)
    covered = owner >= 0
    corners = mesh.positions[mesh.triangles[np.maximum(owner, 0)]]
    place = np.einsum("hwk,hwkc->hwc", bary, corners).astype(np.float32)
    off = np.where(covered & valid, np.linalg.norm(place - position, axis=-1), 0.0)
    # one place: one triangle, or triangles meeting along an edge (left out,
    # the texels on every edge kept the old bake - lines of tan and white
    # along the triangle edges, a fork where slivers fan out; user, 11-05:
    # "포크 모양으로 뭔가 생겼는데")
    agree = np.nan_to_num(np.linalg.norm(first - place, axis=-1), nan=1.0) <= REBAKE_AGREE_M
    single = (count == 1) | ((count > 1) & agree)
    fix = covered & valid & single & (off > REBAKE_OFF_M)
    good = covered & valid & (off <= REBAKE_GOOD_M)
    report = {"texels_moved": int(fix.sum()), "off_mm_median": round(float(np.median(off[fix]) * 1000), 1) if fix.any() else 0.0}
    if not fix.any():
        return report
    good_rows, good_cols = np.nonzero(good)
    tree = cKDTree(position[good_rows, good_cols])
    fix_rows, fix_cols = np.nonzero(fix)
    distance, hit = tree.query(place[fix_rows, fix_cols], workers=-1)
    src_rows, src_cols = good_rows[hit], good_cols[hit]
    report["source_distance_mm_p95"] = round(float(np.percentile(distance, 95) * 1000), 2)
    position = position.copy()
    position[fix_rows, fix_cols] = place[fix_rows, fix_cols]
    np.save(geometry_dir / "position.npy", position)
    for name in ("normal.npy", "normal_raw.npy"):
        path = geometry_dir / name
        if path.exists():
            normal = np.load(path)
            normal[fix_rows, fix_cols] = normal[src_rows, src_cols]
            np.save(path, normal)
    images = sorted((template_dir / "masks").glob("*.png")) + [template_dir / "base_color.png"]
    resampled = []
    for path in images:
        if not path.exists():
            continue
        image = Image.open(path)
        if image.size != (size[0], size[1]):
            continue
        data = np.asarray(image).copy()
        data[fix_rows, fix_cols] = data[src_rows, src_cols]
        Image.fromarray(data, mode=image.mode).save(path)
        resampled.append(path.name)
    report["images_resampled"] = len(resampled)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slug", required=True)
    parser.add_argument("--size", default="small")
    parser.add_argument("--rebake", action="store_true", help="also move texels whose bake is off (see --rebake)")
    args = parser.parse_args()
    template_glb = ROOT / "output" / f"skeleton_{args.slug}_{args.size}.glb"
    template_dir = ROOT / "output" / "templates" / f"{args.slug}_{args.size}"
    geometry_dir = template_dir / "geometry"
    backup = ROOT / "work" / f"backup_geometry_{args.slug}"
    if not backup.exists():
        backup.mkdir(parents=True)
        for name in ("position.npy", "normal.npy", "normal_raw.npy", "valid.npy"):
            if (geometry_dir / name).exists():
                shutil.copy(geometry_dir / name, backup / name)
    report = patch(template_glb, geometry_dir)
    print(f"[patch_geometry] {args.slug}: {report}")
    if args.rebake:
        backup = ROOT / "work" / f"backup_rebake_{args.slug}"
        if not backup.exists():
            (backup / "geometry").mkdir(parents=True)
            (backup / "masks").mkdir(parents=True)
            for name in ("position.npy", "normal.npy", "normal_raw.npy", "valid.npy"):
                if (geometry_dir / name).exists():
                    shutil.copy(geometry_dir / name, backup / "geometry" / name)
            for path in (template_dir / "masks").glob("*.png"):
                shutil.copy(path, backup / "masks" / path.name)
            if (template_dir / "base_color.png").exists():
                shutil.copy(template_dir / "base_color.png", backup / "base_color.png")
        print(f"[patch_geometry] {args.slug} rebake: {rebake(template_glb, template_dir)}")


if __name__ == "__main__":
    main()
