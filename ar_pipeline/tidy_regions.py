"""The last pass over a template's ten region masks: every texel gets one
region that holds it, the chest is the part the compositor paints, and no
region keeps stray pieces inside another.

WHY
---
The masks are cut in several passes (seed_regions, refine_masks,
refine_face_masks, open_face_masks, smooth_masks) and each leaves something
the region map (tools/viewer/region_map.py) shows and the coat shows too:

- After the 1.8 cm surface smoothing about 1.5% of the texels, where three
  regions meet, have no region at 0.5 or more. The compositor re-thresholds
  every mask at 0.5 (its fur-level edge jitter), so those texels are gaps,
  and the gap fill copies the nearest claimed texel IN THE ATLAS - often on
  the next UV island, another body part.
- The chest mask runs onto the sides of the upper forelegs. The compositor
  paints the chest colour only where the surface faces forward
  (texture._chest_front) and leaves the rest of the mask the base coat: the
  black patch at the elbow of every tan-point and Irish anchor.
- The cheeks run down the neck behind the jaw (the retriever, the maltese):
  refine_face_masks takes every head-bone texel below the eyes and in front
  of the ear base, and the head bone reaches the upper throat.
- Pieces of one region inside another: skull on a puppy's forepaw and on
  the maltese tail, back inside the retriever's bib, cheek spots on the
  maltese chin.

HOW
---
On the hard labels (the region with the largest mask value):
1. chest texels the compositor would leave the base coat become whichever
   other region is nearest on the surface; cheek texels behind and below the
   jaw become the throat (chest) where they face forward or down as
   refine_masks' throat does, the neck (back) elsewhere;
2. connected pieces of each region are found in the atlas and joined across
   UV seams by surface distance; a piece smaller than STRAY_SHARE of the
   region's largest piece and STRAY_MAX_TEXELS (ENCLOSED_MAX_TEXELS when one
   region surrounds it) becomes its surroundings;
3. the masks keep their soft values where the label did not change, a
   changed texel is all its new region, and every texel's region is raised
   to at least DOMINANT with the others scaled to share the rest.

Usage:
    python -m ar_pipeline.tidy_regions --slug maltese [--size small]
Run after smooth_masks. Rewrites output/templates/<slug>_<size>/masks and
sets "regions_tidied" in template.json, which tells the compositor the chest
mask is already cut to the forward-facing part.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.landmarks import Landmarks  # noqa: E402
from api.mesh import TemplateMesh  # noqa: E402
from api.regions import REGIONS  # noqa: E402
from api.texture import _chest_front  # noqa: E402
from ar_pipeline.smooth_masks import smooth_channels  # noqa: E402

STRAY_SHARE = 0.15
STRAY_MAX_TEXELS = 6000
# ... or up to this size when one other region holds ENCLOSED_SHARE of its
# edge in the atlas: a hole of back inside the retriever's bib (7-10k
# texels). The inside of a thigh (13-16k, underside between back and
# legs_lower) is not enclosed and stays.
ENCLOSED_MAX_TEXELS = 30000
ENCLOSED_SHARE = 0.9
# texels this many texel widths apart on the surface are one piece (across a UV seam)
SEAM_REACH_TEXELS = 6.0
DOMINANT = 0.52
CHEST_FACING_SIGMA_M = 0.015
# the cheek stops at the jaw: this many eye distances below the jaw joint
# (the hinge sits above the jaw's lower edge), behind it
JAW_DROP_EYES = 0.25
THROAT_FACING = 0.25   # as refine_masks: normal·(forward+down)/√2 above this is throat


def nearest_fill(label: np.ndarray, position: np.ndarray, todo: np.ndarray, exclude: int = -1) -> np.ndarray:
    """Each `todo` texel takes the label of the nearest labelled texel on the
    surface, never `exclude` (the region it is being taken out of)."""
    known = (label >= 0) & ~todo & (label != exclude)
    rows, cols = np.nonzero(known)
    tree = cKDTree(position[rows, cols].astype(np.float64))
    t_rows, t_cols = np.nonzero(todo)
    _, nearest = tree.query(position[t_rows, t_cols].astype(np.float64), workers=-1)
    out = label.copy()
    out[t_rows, t_cols] = label[rows[nearest], cols[nearest]]
    return out


def pieces(label: np.ndarray, position: np.ndarray, reach: float) -> tuple[np.ndarray, np.ndarray]:
    """Per texel, the id of its piece (connected in the atlas, joined across
    seams within `reach` metres); and each piece's texel count."""
    piece = np.full(label.shape, -1, dtype=np.int64)
    count = 0
    for index in range(len(REGIONS)):
        ids, n = ndimage.label(label == index, structure=np.ones((3, 3)))
        piece[ids > 0] = ids[ids > 0] - 1 + count
        count += n
    # the texels on a piece's edge, looked up in 3D: two pieces of one region
    # that meet across a seam are one piece on the dog
    padded = np.pad(label, 1, constant_values=-1)
    edge = (label >= 0) & ((padded[:-2, 1:-1] != label) | (padded[2:, 1:-1] != label)
                           | (padded[1:-1, :-2] != label) | (padded[1:-1, 2:] != label))
    rows, cols = np.nonzero(edge)
    pairs = cKDTree(position[rows, cols].astype(np.float64)).query_pairs(reach, output_type="ndarray")
    a, b = piece[rows[pairs[:, 0]], cols[pairs[:, 0]]], piece[rows[pairs[:, 1]], cols[pairs[:, 1]]]
    same = (label[rows[pairs[:, 0]], cols[pairs[:, 0]]] == label[rows[pairs[:, 1]], cols[pairs[:, 1]]]) & (a != b)
    graph = coo_matrix((np.ones(int(same.sum())), (a[same], b[same])), shape=(count, count))
    _, joined = connected_components(graph, directed=False)
    merged = np.where(piece >= 0, joined[np.maximum(piece, 0)], -1)
    sizes = np.bincount(merged[merged >= 0])
    return merged, sizes


def enclosed(label: np.ndarray, merged: np.ndarray, count: int) -> np.ndarray:
    """Per piece: whether a single other region holds ENCLOSED_SHARE of the
    texels across its edge in the atlas (texels across an island edge are
    not counted)."""
    padded = np.pad(label, 1, constant_values=-1)
    tally = np.zeros(count * len(REGIONS), dtype=np.int64)
    for neighbour in (padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]):
        across = (label >= 0) & (neighbour >= 0) & (neighbour != label)
        tally += np.bincount(merged[across] * len(REGIONS) + neighbour[across], minlength=tally.size)
    tally = tally.reshape(count, len(REGIONS))
    return tally.max(axis=1) >= ENCLOSED_SHARE * np.maximum(tally.sum(axis=1), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--size", default="small")
    args = parser.parse_args()
    template = ROOT / "output" / "templates" / f"{args.slug}_{args.size}"
    masks_dir = template / "masks"
    landmarks = Landmarks.load(template, template / "geometry")
    position, valid = landmarks.position, landmarks.valid
    masks = np.stack([np.asarray(Image.open(masks_dir / f"{r}.png").convert("L"), dtype=np.float32) / 255.0
                      for r in REGIONS])
    label = np.where(valid, np.argmax(masks, axis=0), -1)
    before = label.copy()
    print(f"[tidy] texels without a region at 0.5: {int((valid & (masks.max(axis=0) < 0.5)).sum())}")

    # 1. the chest the compositor paints
    chest = REGIONS.index("chest")
    # the facing smoothed on the surface first: a fur bump in the middle of
    # the bib that faces sideways is still bib (the puppy's came out as a
    # belly-and-back spot on its chest)
    facing = smooth_channels(_chest_front(landmarks, None)[..., None], position, valid, CHEST_FACING_SIGMA_M)[..., 0]
    off = (label == chest) & (facing < 0.5)
    if off.any():
        label = nearest_fill(np.where(off, -1, label), position, off, exclude=chest)
    print(f"[tidy] chest texels facing sideways, given to their neighbours: {int(off.sum())}")
    jaw = TemplateMesh.load(ROOT / "output" / f"skeleton_{args.slug}_{args.size}.glb", valid.shape[::-1]).joints["jaw_jnt"]
    neck = ((label == REGIONS.index("cheeks")) & (position[..., 1] > jaw[1])
            & (position[..., 2] < jaw[2] - JAW_DROP_EYES * landmarks.eye_distance))
    throat = (-landmarks.normal[..., 1] - landmarks.normal[..., 2]) / np.sqrt(2.0) > THROAT_FACING
    label = np.where(neck, np.where(throat, chest, REGIONS.index("back")), label)
    print(f"[tidy] cheek texels behind and below the jaw: {int((neck & throat).sum())} to the throat, "
          f"{int((neck & ~throat).sum())} to the neck")

    # 2. stray pieces
    reach = SEAM_REACH_TEXELS * landmarks.texel_metres()
    merged, sizes = pieces(label, position, reach)
    inside = enclosed(label, merged, sizes.size)
    stray = np.zeros(label.shape, dtype=bool)
    for index, region in enumerate(REGIONS):
        own = np.unique(merged[label == index])
        if own.size == 0:
            continue
        largest = sizes[own].max()
        small = own[(sizes[own] < STRAY_SHARE * largest)
                    & ((sizes[own] < STRAY_MAX_TEXELS) | (inside[own] & (sizes[own] < ENCLOSED_MAX_TEXELS)))]
        if small.size:
            stray |= np.isin(merged, small)
            print(f"[tidy] {region:13s} pieces {own.size:4d}, largest {int(largest):8d}; "
                  f"{small.size} stray ({int(sizes[small].sum())} texels, biggest {int(sizes[small].max())})")
    if stray.any():
        label = nearest_fill(np.where(stray, -1, label), position, stray)

    # 3. soft masks back: unchanged texels keep theirs, every texel's region dominant
    changed = valid & (label != before)
    onehot = (label[None] == np.arange(len(REGIONS))[:, None, None]).astype(np.float32)
    masks = np.where(changed[None], onehot, masks)
    own = np.take_along_axis(masks, np.maximum(label, 0)[None], axis=0)[0]
    raise_to = valid & (own < DOMINANT)
    rest = np.maximum(masks.sum(axis=0) - own, 1e-6)
    scale = np.where(raise_to, (1.0 - DOMINANT) / rest, 1.0)
    masks = np.where(raise_to[None], masks * scale[None], masks)
    masks = np.where(raise_to[None] & (onehot > 0), DOMINANT, masks)
    # the padding round every island takes its nearest island texel's values:
    # the renderer reads across an island's edge, and after the passes above
    # 11-17% of the padding next to an island held another region - on the
    # maltese's thousand fur-shard islands that read as light specks on black
    nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    masks = masks[:, nearest[0], nearest[1]]
    for index, region in enumerate(REGIONS):
        Image.fromarray((np.clip(masks[index], 0, 1) * 255 + 0.5).astype(np.uint8), mode="L").save(
            masks_dir / f"{region}.png")
    # the compositor reads this and skips its own chest facing test
    settings_path = template / "template.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
    settings["regions_tidied"] = True
    settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[tidy] {int(changed.sum())} texels changed region, {int(raise_to.sum())} raised to {DOMINANT}; "
          f"wrote {masks_dir}")


if __name__ == "__main__":
    main()
