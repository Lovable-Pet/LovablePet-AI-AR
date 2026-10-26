"""Smooth a template's body region masks over the dog's surface.

WHY
---
The region masks were cut and cleaned per UV island (refine_masks.py:
normals, label smoothing, edge clean, all in texel space). On the generated
jindo the "back" mask came out as 281 pieces, 255 of them small shards
sitting inside the chest and the throat; with a cream chest those shards
stay tan and read as lightning bolts down the neck and the bib (user on
eval v7 e05, e18), and every region edge that meets a UV seam stops there
instead of running on.

HOW
---
Each body mask is averaged over the surface with a gaussian of SIGMA_M
metres through the position map (the same surface smoothing the sable
overlay uses): a shard smaller than about two sigma falls under 0.5 and
joins its surroundings, and two texels either side of a seam - one point on
the dog - get one value. The face regions are small (brows are 1.6k texels)
and are left as they are. All masks are then renormalised so they still
share each texel. Texels without geometry keep their old values.

Usage:
    python -m ar_pipeline.smooth_masks --slug jindo --size small
Back up output/templates/<slug>_<size>/masks first; this rewrites it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BODY_REGIONS = ("back", "chest", "underside", "legs_lower", "paws")
SIGMA_M = 0.018
STRIDE = 4


def smooth_channels(fields: np.ndarray, position: np.ndarray, valid: np.ndarray, sigma: float,
                    stride: int = STRIDE) -> np.ndarray:
    """(h, w, k) fields averaged over the surface; one neighbour search for all k."""
    rows, cols = np.nonzero(valid[::stride, ::stride])
    rows, cols = rows * stride, cols * stride
    points = position[rows, cols].astype(np.float64)
    values = fields[rows, cols].astype(np.float64)
    tree = cKDTree(points)
    smoothed = np.empty_like(values)
    for index, near in enumerate(tree.query_ball_point(points, r=2.5 * sigma, workers=-1)):
        near = np.asarray(near)
        w = np.exp(-np.sum((points[near] - points[index]) ** 2, axis=1) / (2.0 * sigma * sigma))
        smoothed[index] = w @ values[near] / w.sum()
    out = fields.astype(np.float32).copy()
    all_rows, all_cols = np.nonzero(valid)
    distance, nearest = tree.query(position[all_rows, all_cols].astype(np.float64), k=4, workers=-1)
    w = 1.0 / np.maximum(distance, 1e-5)
    out[all_rows, all_cols] = (np.einsum("nk,nkc->nc", w, smoothed[nearest]) / w.sum(axis=1)[:, None])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", required=True)
    parser.add_argument("--size", default="small")
    parser.add_argument("--sigma", type=float, default=SIGMA_M)
    args = parser.parse_args()
    template = ROOT / "output" / "templates" / f"{args.slug}_{args.size}"
    masks_dir = template / "masks"
    position = np.load(template / "geometry" / "position.npy")
    valid = np.load(template / "geometry" / "valid.npy") > 0.5

    names = sorted(p.stem for p in masks_dir.glob("*.png") if not p.stem.startswith("_"))
    masks = {n: np.asarray(Image.open(masks_dir / f"{n}.png").convert("L"), dtype=np.float32) / 255.0 for n in names}
    body = [n for n in BODY_REGIONS if n in masks]
    before = {n: int((masks[n] > 0.5)[valid].sum()) for n in body}

    smoothed = smooth_channels(np.stack([masks[n] for n in body], -1), position, valid, args.sigma)
    for k, n in enumerate(body):
        masks[n] = np.where(valid, smoothed[..., k], masks[n])
    total = sum(masks.values())
    for n in names:
        masks[n] = np.where(valid & (total > 1e-4), masks[n] / np.maximum(total, 1e-4), masks[n])
        Image.fromarray(np.clip(masks[n] * 255 + 0.5, 0, 255).astype(np.uint8), mode="L").save(masks_dir / f"{n}.png")

    from scipy import ndimage
    for n in body:
        b = masks[n] > 0.5
        _, pieces = ndimage.label(b)
        print(f"[smooth_masks] {n:11s} texels {before[n]:8d} -> {int(b[valid].sum()):8d}, pieces {pieces}")


if __name__ == "__main__":
    main()
