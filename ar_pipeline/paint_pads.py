"""Paint a template's soles as paw pads, the way another template has them.

WHY
---
The generated jindo puppy and maltese came with each sole one dark disc -
no toe pads, no main pad, no fur between them - where the jindo, chihuahua
and retriever have four toe pads and a main pad with fur around them. Seen
from below or as a paw lifts, the two looked shod in black rubber (user,
11-06: "진도강아지, 말티즈 기본모델 발바닥 다른개처럼 자연스럽게 수정바라고").
And protect.png keeps a sole's dark texels out of every coat pass, so every
dog built on them got the same black discs. The toy poodle's soles had no
pads at all, white fur all over (user, 11-06: "토이푸들발바닥도").

HOW
---
Each paw's sole (the paws mask's texels within SOLE_M of the floor, facing
down) gets a 2D frame: across the paw (out from the body) and along it
(toward the head), each scaled by the sole's own extent. The reference
template's pads (its dark sole texels, as build_landmarks finds pads) are
laid on that frame per paw, front and hind, left and right, and read back
on the target's sole: a pad takes the sole's own dark colour (its median)
with the reference's grain at that spot - one flat colour looked like a
sticker, and the jindo puppy's own texels kept the blotches the generator
had drawn in its disc, a main pad light above and dark below. Elsewhere
the sole takes the fur of its own paw just above it. A sole with no dark
of its own takes the template's other soles' colour, or with none (the toy
poodle) the reference's. Where a pad's surface turns up from the floor -
up the front and sides of a toe, seen standing - it fades out (PAD_TURN).
protect.png in the soles becomes the pads only, so the fur between them is
the coat's.

Each sole lies on two or more UV charts, and the unbaked gutter beside a
chart still held the old sole's colour: bilinear filtering and the mip
levels read it along every seam, dark lines across the fur between the new
pads and light ones across the pads. So the gutter texels the soles'
triangles sample get the colour of the surface they show (as fill_gutters
does), and the rest of the gutter beside a sole its nearest sole texel's.

    python -m ar_pipeline.paint_pads --slug jindo_puppy                 # preview only
    python -m ar_pipeline.paint_pads --slug jindo_puppy --install       # GLB, base_color.png, protect.png

Only the sole texels and the gutter beside them change, and the soles are
read from the backup when there is one, so the tool composes with others
that edit the same texture and a second run gives the same result.
--install backs the GLB, base_color.png and protect.png up to
work/backup_pads_<slug>/ first (once). The template assets go into the
next release (assets-v2).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from api.glb import extract_base_texture, replace_base_texture
from ar_pipeline.fill_gutters import NEAREST, coat_triangles, sampled_gutter

LUMA = np.array([0.2126, 0.7152, 0.0722])
# a sole: the paws mask's texels this close to the floor whose normal points
# at least this far down (the frame is fitted on the flat ones, FLAT_NZ)
SOLE_M = 0.02
SOLE_NZ = -0.2
FLAT_NZ = -0.6
# a pad texel, as build_landmarks.protect_mask tells pads (linear luminance)
PAD_MAX_LUMA = 0.12
# the pad map: cells over [-GRID_REACH, GRID_REACH] in sole units, smoothed
GRID = 96
GRID_REACH = 1.5
GRID_SMOOTH = 0.8
# a pad's edge: its weight ramps over this band
EDGE = (0.35, 0.65)
# the pads a sole has: four toes and the main pad (the reference's claw tips
# and dark specks at the sole's edge came over as dark crumbs). The jindo's
# dark claws touch its toe pads, and came over as tails on them and a blob
# past a toe: the map is opened first by a disc of this radius in cells (a
# toe pad is 20 or more across, a claw 3 to 5; at radius 3 bumps stayed)
PADS = 5
OPEN_CELLS = 5
# a sole with less of it dark than this has no pad colour of its own: it
# takes the template's other soles', or the reference's when none has one
# (the toy poodle's soles were 0-1 % dark, white fur all over)
OWN_PAD_SHARE = 0.3
# the fur between the pads: the paw's own just above the sole, a little
# lighter or darker over the surface (share, smooth noise in 3D on cells of
# FUR_CELL_M and a third of it; sines drew a fine diamond mesh on it)
FUR_BAND_M = 0.03
FUR_VARIATION = 0.05
FUR_CELL_M = 0.0012
# a pad's grain: the reference's pad texel at the same place over its pads'
# median, within these bounds
GRAIN = (0.7, 1.5)
# a pad fades out where its surface turns up from the pad's most down-facing
# part (-normal z, its PAD_CORE percentile): all of it within the first of
# that, none past the second. The toe pads ran up the toy poodle's and the
# jindo puppy's toes, dark patches on their fronts and sides seen standing;
# measured against the floor instead, the maltese's hind toes (they face
# forward) lost their pads. The normals are averaged over this many nearest
# sole texels first - the paw's shape, not its facets (the toy poodle's
# crumpled mesh notched its pads).
PAD_CORE = 90
PAD_TURN = (0.25, 0.45)
PAD_SMOOTH_TEXELS = 96
# a gutter texel a triangle samples belongs to a sole when the point it
# shows is this close to one
SEAM_REACH_M = 0.003


def load(slug: str, texture: Image.Image | None = None) -> dict:
    tpl = Path(f"output/templates/{slug}_small")
    geometry = tpl / "geometry"
    position = np.load(geometry / "position.npy")
    valid = np.load(geometry / "valid.npy") > 0.5
    normal = np.load(geometry / "normal.npy")
    h, w = valid.shape
    if texture is None:
        texture = Image.open(tpl / "base_color.png")
    rgb = np.asarray(texture.convert("RGB").resize((w, h))).astype(np.float32)
    paws = np.asarray(Image.open(tpl / "masks" / "paws.png").convert("L").resize((w, h))) > 127
    nose_y = json.loads((tpl / "landmarks.json").read_text(encoding="utf-8"))["points"]["nose"]["world"][1]
    floor = float(position[valid][:, 2].min())
    sole = valid & paws & (position[..., 2] < floor + SOLE_M) & (normal[..., 2] < SOLE_NZ)
    return {"tpl": tpl, "position": position, "valid": valid, "normal": normal, "rgb": rgb, "paws": paws,
            "head": float(np.sign(nose_y) or -1.0), "floor": floor, "sole": sole,
            "luma": ((rgb / 255.0) ** 2.2) @ LUMA}


def soles(t: dict) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per paw (LF, RF, LB, RB): its sole texels and their (across, along)
    coordinates in sole units (+across away from the body, +along toward the
    head), the frame fitted on its flat texels."""
    pos = t["position"]
    front = (pos[..., 1] * t["head"]) > 0
    out = {}
    for name, left, fore in (("LF", True, True), ("RF", False, True), ("LB", True, False), ("RB", False, False)):
        side = 1.0 if left else -1.0
        # (x is the dog's left in the bake when its head is at -y)
        sel = t["sole"] & ((pos[..., 0] * side * -t["head"]) > 0) & (front == fore)
        flat = sel & (t["normal"][..., 2] < FLAT_NZ)
        if flat.sum() < 50:
            raise SystemExit(f"{t['tpl'].name} {name}: no flat sole ({int(flat.sum())} texels)")
        xy = pos[flat][:, :2]
        centre = np.median(xy, 0)
        half = (np.percentile(xy, 97, 0) - np.percentile(xy, 3, 0)) / 2
        p = pos[sel][:, :2]
        across = (p[:, 0] - centre[0]) / half[0] * np.sign(centre[0])
        along = (p[:, 1] - centre[1]) * t["head"] / half[1]
        out[name] = (np.argwhere(sel), across, along)
    return out


def value_noise(points: np.ndarray, cell_m: float, seed: int) -> np.ndarray:
    """Smooth noise in [-1, 1] at 3D points (metres): random values on a grid
    of cell_m, trilinear between them (the grid repeats every 64 cells)."""
    grid = np.random.default_rng(seed).uniform(-1.0, 1.0, (64, 64, 64))
    return ndimage.map_coordinates(grid, (points / cell_m).T, order=1, mode="grid-wrap")


def cell(across: np.ndarray, along: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = lambda c: (np.clip(c, -GRID_REACH, GRID_REACH) + GRID_REACH) / (2 * GRID_REACH) * (GRID - 1)
    return f(across), f(along)


def pad_maps(ref: dict) -> dict[str, np.ndarray]:
    """Per paw, where the reference's pads are on the grid (1 on a pad, 0 off
    it, soft at the edges): its dark sole texels' share, opened, the PADS
    largest pads kept and filled."""
    out = {}
    yy, xx = np.mgrid[-OPEN_CELLS:OPEN_CELLS + 1, -OPEN_CELLS:OPEN_CELLS + 1]
    disc = xx * xx + yy * yy <= OPEN_CELLS * OPEN_CELLS
    for name, (texels, across, along) in soles(ref).items():
        pad = (ref["luma"][texels[:, 0], texels[:, 1]] < PAD_MAX_LUMA).astype(np.float64)
        ca, cl = cell(across, along)
        ia, il = np.round(ca).astype(int), np.round(cl).astype(int)
        total = np.zeros((GRID, GRID)); count = np.zeros((GRID, GRID))
        np.add.at(total, (il, ia), pad)
        np.add.at(count, (il, ia), 1.0)
        total = ndimage.gaussian_filter(total, GRID_SMOOTH)
        count = ndimage.gaussian_filter(count, GRID_SMOOTH)
        share = np.where(count > 1e-3, total / np.maximum(count, 1e-9), 0.0)
        pads = ndimage.binary_opening(share > 0.5, structure=disc)
        labels, n = ndimage.label(pads)
        sizes = ndimage.sum(pads, labels, range(1, n + 1))
        pads = ndimage.binary_fill_holes(np.isin(labels, np.argsort(sizes)[::-1][:PADS] + 1))
        out[name] = ndimage.gaussian_filter(pads.astype(np.float64), GRID_SMOOTH)
    return out


def pad_grain(ref: dict) -> dict[str, tuple[cKDTree, np.ndarray, np.ndarray]]:
    """Per paw: the reference's sole texels on their frame, each one's grain
    (its brightness over its pads' median, 1 off the pads) and that median."""
    out = {}
    for name, (texels, across, along) in soles(ref).items():
        rgb = ref["rgb"][texels[:, 0], texels[:, 1]]
        pad = ref["luma"][texels[:, 0], texels[:, 1]] < PAD_MAX_LUMA
        median = np.median(rgb[pad], 0)
        grain = np.where(pad, np.clip(rgb.mean(-1) / median.mean(), *GRAIN), 1.0)
        out[name] = (cKDTree(np.stack([across, along], -1)), grain, median)
    return out


def paint(slug: str, ref_slug: str) -> tuple[Image.Image, Image.Image, list[str], Image.Image]:
    """(painted texture, new protect.png, report, before/after soles) - the
    soles read from the backup when there is one, the rest of the texture
    and protect.png as they are now."""
    tpl = Path(f"output/templates/{slug}_small")
    glb = Path(f"output/skeleton_{slug}_small.glb")
    backup = Path(f"work/backup_pads_{slug}")
    current = extract_base_texture(glb).convert("RGB")
    t = load(slug, extract_base_texture(backup / glb.name) if (backup / glb.name).exists() else current)
    ref = load(ref_slug)
    maps, grains = pad_maps(ref), pad_grain(ref)
    # every cell of a map: the number of the pad nearest it
    pad_of = {}
    for name, share in maps.items():
        labels, _ = ndimage.label(share > 0.5)
        nearest = ndimage.distance_transform_edt(labels == 0, return_distances=False, return_indices=True)
        pad_of[name] = labels[nearest[0], nearest[1]]
    out = np.asarray(current).astype(np.float32).copy()
    h, w = t["valid"].shape
    protect_path = (backup if (backup / "protect.png").exists() else tpl) / "protect.png"
    protect = np.asarray(Image.open(protect_path).convert("L").resize((w, h))).copy()
    current_protect = np.asarray(Image.open(tpl / "protect.png").convert("L").resize((w, h))).copy()
    pos, rgb = t["position"], t["rgb"]
    all_soles = soles(t)
    dark_of = lambda tx: t["luma"][tx[:, 0], tx[:, 1]] < PAD_MAX_LUMA
    dark_soles = [rgb[tx[:, 0], tx[:, 1]][dark_of(tx)] for tx, _, _ in all_soles.values()
                  if dark_of(tx).mean() >= OWN_PAD_SHARE]
    report = []
    for name, (texels, across, along) in all_soles.items():
        r, c = texels[:, 0], texels[:, 1]
        ca, cl = cell(across, along)
        weight = ndimage.map_coordinates(maps[name], [cl, ca], order=1, mode="constant", cval=0.0)
        weight = np.clip((weight - EDGE[0]) / (EDGE[1] - EDGE[0]), 0.0, 1.0)
        weight = weight * weight * (3 - 2 * weight)
        weight[(np.abs(across) >= GRID_REACH) | (np.abs(along) >= GRID_REACH)] = 0.0
        # each texel belongs to the pad nearest it on the map, and fades as it
        # turns up from that pad's most down-facing part (PAD_TURN)
        k = pad_of[name][np.clip(np.round(cl).astype(int), 0, GRID - 1), np.clip(np.round(ca).astype(int), 0, GRID - 1)]
        _, near = cKDTree(pos[r, c]).query(pos[r, c], k=PAD_SMOOTH_TEXELS, workers=-1)
        down = -t["normal"][r, c][near][..., 2].mean(1)
        turn = np.zeros(len(r))
        for pad in np.unique(k):
            core = down[(k == pad) & (weight > 0.5)]
            if len(core):
                turn[k == pad] = np.percentile(core, PAD_CORE) - down[k == pad]
        keep = np.clip((PAD_TURN[1] - turn) / (PAD_TURN[1] - PAD_TURN[0]), 0.0, 1.0)
        weight = weight * keep * keep * (3 - 2 * keep)
        # the pads: the sole's own dark (or the template's other soles' if
        # this one has hardly any, or the reference's if none has) with the
        # reference's grain; the rest is the paw's fur from just above the sole
        dark = t["luma"][r, c] < PAD_MAX_LUMA
        own = dark.mean() >= OWN_PAD_SHARE
        tree, ref_grain, ref_median = grains[name]
        grain = ref_grain[tree.query(np.stack([across, along], -1), workers=-1)[1]]
        pad_colour = (np.median(rgb[r, c][dark], 0) if own else
                      np.median(np.concatenate(dark_soles), 0) if dark_soles else ref_median)
        pad_rgb = pad_colour[None, :] * grain[:, None]
        side = (pos[..., 0] * np.sign(pos[r, c][:, 0].mean())) > 0
        ahead = (pos[..., 1] * t["head"] > 0) == (name[1] == "F")
        band = (t["valid"] & t["paws"] & side & ahead & ~t["sole"]
                & (pos[..., 2] >= t["floor"] + SOLE_M) & (pos[..., 2] < t["floor"] + SOLE_M + FUR_BAND_M)
                & (t["luma"] > PAD_MAX_LUMA))
        fur = np.median(rgb[band], 0) if band.any() else np.array([235.0, 230.0, 225.0])
        p = pos[r, c].astype(np.float64)
        wobble = 0.7 * value_noise(p, FUR_CELL_M, 1) + 0.3 * value_noise(p, FUR_CELL_M / 3, 2)
        fur_rgb = np.clip(fur[None, :] * (1.0 + FUR_VARIATION * wobble[:, None]), 0, 255)
        out[r, c] = weight[:, None] * pad_rgb + (1.0 - weight[:, None]) * fur_rgb
        # the soles in protect.png: the pads only
        protect[r, c] = np.where(weight > 0.5, 255, 0)
        current_protect[r, c] = protect[r, c]
        report.append(f"{name}: {len(r)} sole texels, {int((weight > 0.5).sum())} pad, pad colour "
                      f"{'its own' if own else 'from its dark soles' if dark_soles else 'the reference pads'}, "
                      f"fur {fur.round().astype(int).tolist()}")

    # the gutter beside the soles (see the module doc): the texels the soles'
    # triangles sample take the surface they show, the rest beside a sole
    # its nearest sole texel's colour
    sole_all = np.zeros(t["valid"].shape, bool)
    for texels, _, _ in all_soles.values():
        sole_all[texels[:, 0], texels[:, 1]] = True
    gutter = ~t["valid"]
    sampled, where = sampled_gutter(*coat_triangles(glb), gutter)
    if len(sampled):
        on_sole = np.isfinite(cKDTree(pos[sole_all]).query(where, distance_upper_bound=SEAM_REACH_M)[0])
        sampled, where = sampled[on_sole], where[on_sole]
    seam = np.zeros_like(gutter)
    if len(sampled):
        _, near = cKDTree(pos[t["valid"]]).query(where, k=NEAREST, workers=-1)
        out[sampled[:, 0], sampled[:, 1]] = out[t["valid"]][near].mean(1)
        seam[sampled[:, 0], sampled[:, 1]] = True
    rest = gutter & ~seam
    idx = ndimage.distance_transform_edt(rest, return_distances=False, return_indices=True)
    beside = rest & (sole_all | seam)[idx[0], idx[1]]
    out[beside] = out[idx[0][beside], idx[1][beside]]
    report.append(f"gutter: {int(seam.sum())} texels the soles' triangles sample coloured from the surface, "
                  f"{int(beside.sum())} more nearest a sole padded from it")
    painted = Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))
    preview = sole_preview(t, np.asarray(painted).astype(np.float32))
    return painted, Image.fromarray(current_protect), report, preview


def sole_preview(t: dict, after: np.ndarray, size: int = 180) -> Image.Image:
    """The four soles seen from below, before (top) and after (bottom)."""
    sheet = Image.new("RGB", (4 * size, 2 * size), (128, 128, 128))
    for k, (name, (texels, across, along)) in enumerate(soles(t).items()):
        for row, src in enumerate((t["rgb"], after)):
            img = np.full((size, size, 3), 128, np.uint8)
            u = np.clip(((across / GRID_REACH) * 0.5 + 0.5) * (size - 1), 0, size - 1).astype(int)
            v = np.clip((0.5 - (along / GRID_REACH) * 0.5) * (size - 1), 0, size - 1).astype(int)
            img[v, u] = src[texels[:, 0], texels[:, 1]].astype(np.uint8)
            sheet.paste(Image.fromarray(img), (k * size, row * size))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slug", required=True)
    parser.add_argument("--ref", default="jindo", help="the template whose pads are copied (default jindo)")
    parser.add_argument("--out", default=None, help="where the preview goes (default work/pads_<slug>/)")
    parser.add_argument("--install", action="store_true", help="write into the GLB, base_color.png and protect.png")
    args = parser.parse_args()
    out = Path(args.out or f"work/pads_{args.slug}")
    out.mkdir(parents=True, exist_ok=True)
    painted, protect, report, preview = paint(args.slug, args.ref)
    painted.save(out / "base_color.png")
    protect.save(out / "protect.png")
    preview.save(out / "soles_preview.png")
    print("\n".join(report), f"\npreview {out / 'soles_preview.png'} (soles from below: before, after; toes up)")
    if args.install:
        tpl = Path(f"output/templates/{args.slug}_small")
        glb = Path(f"output/skeleton_{args.slug}_small.glb")
        backup = Path(f"work/backup_pads_{args.slug}")
        if not backup.exists():
            backup.mkdir(parents=True)
            for f in (glb, tpl / "base_color.png", tpl / "protect.png"):
                shutil.copy(f, backup / f.name)
        replace_base_texture(glb, painted, out / glb.name)
        shutil.copy(out / glb.name, glb)
        painted.save(tpl / "base_color.png")
        protect.save(tpl / "protect.png")
        print(f"installed into {glb}, {tpl / 'base_color.png'} and protect.png (backup {backup})")


if __name__ == "__main__":
    main()
