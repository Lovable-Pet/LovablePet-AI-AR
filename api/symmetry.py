"""Mirror symmetry for the coat: the left and right of the animal agree.

The atlas holds each leg, each ear and each side of the body as its own
piece, and the image model paints every piece on its own - one front leg
came back tan and the other white. A dog's markings are not perfectly
symmetric, but they are far more symmetric than that, and a viewer reads a
left/right mismatch as a mistake before anything else.

Every texel has a world position (geometry/position.npy). Its mirror is the
texel whose position is closest to (-x, y, z); a KD-tree over the valid
texels finds it. Within the regions asked for, each texel is blended with
its mirror at `strength` (1.0 = both sides take the average). The face is
left out by default: a photograph's face is the one place an asymmetry may
be the animal's own.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .regions import REGIONS

BODY_SYMMETRIC = tuple(r for r in REGIONS if r not in ("eyebrows", "muzzle_bridge", "cheeks", "skull"))
# a mirror partner further than this share of the model's span is no partner
MIRROR_TOLERANCE = 0.01


def symmetrize(texture: np.ndarray, position: np.ndarray, valid: np.ndarray, masks,
               regions=BODY_SYMMETRIC, strength: float = 1.0) -> tuple[np.ndarray, dict]:
    """`texture` (h, w, 3) linear. Returns the symmetrised texture and a report."""
    span = float((position[valid].max(axis=0) - position[valid].min(axis=0)).max())
    where = np.zeros(valid.shape, dtype=bool)
    for region in regions:
        mask = masks.get(region)
        if mask is not None:
            where |= mask > 0.5
    where &= valid
    rows, cols = np.nonzero(valid)
    tree = cKDTree(position[rows, cols])
    sel_rows, sel_cols = np.nonzero(where)
    mirrored = position[sel_rows, sel_cols] * np.array([-1.0, 1.0, 1.0], dtype=np.float32)
    distance, index = tree.query(mirrored, distance_upper_bound=MIRROR_TOLERANCE * span)
    hit = np.isfinite(distance)
    out = texture.copy()
    partner_rows, partner_cols = rows[index[hit]], cols[index[hit]]
    mine = texture[sel_rows[hit], sel_cols[hit]]
    theirs = texture[partner_rows, partner_cols]
    out[sel_rows[hit], sel_cols[hit]] = mine * (1 - strength / 2) + theirs * (strength / 2)
    report = {"regions": list(regions), "texels": int(where.sum()), "paired_share": round(float(hit.mean()), 3),
              "strength": strength}
    return out, report


__all__ = ["BODY_SYMMETRIC", "symmetrize"]
