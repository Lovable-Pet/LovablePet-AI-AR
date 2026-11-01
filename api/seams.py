"""Seam post-process: make the texture agree with itself across UV seams.

A region mask ends where its UV island ends, and two islands that meet on
the mesh - the left and right halves of the back, the face and the skull -
are painted independently. Any difference between them, a feathered edge
that reached one island and not the other, noise that does not continue,
shows on the model as a line along the seam.

The mesh knows which texels are neighbours across a seam; the texture does
not. So the texture goes to the mesh and back:

  UV -> vertex colour   every vertex samples the texture at its own UV.
  seam smoothing        vertices split by a seam share a position; their
                        samples are averaged, so both sides of the seam
                        now hold one colour.
  rebake                the vertex colours are interpolated back across
                        the UV triangles: a smooth field that is, by
                        construction, continuous across every seam.

Only a ring of RING_PX texels along each island edge takes the rebaked
colour, and it keeps its own hair detail on top - the rebake is a vertex
field, far too smooth to stand in for the texture anywhere but at the seam.
The texels past the island edge (the padding bilinear filtering reads) are
filled from the rebake as well, so a filter tap that lands off the island
finds the seam's colour rather than whatever the padding held.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .mesh import TemplateMesh

RING_PX = 5
# The hair detail kept on top of the rebaked seam colour is the texture minus
# its blur at this sigma.
DETAIL_SIGMA = 2.0
# Texels past the island edge filled from the rebake.
PAD_PX = 8
# A vertex's bilinear sample leaves out the taps baked further than this from
# the vertex (the template's position map): a corner on the edge of a strip
# packed into another chart read that chart - on the jindo puppy 2 the
# shoulder seam's corners read the white bib 12-18 cm away, and the shared
# seam colour drew a grey zigzag down the black shoulder (shelter puppy 11).
# With no tap left, the vertex takes the nearest texel within TAP_WINDOW_PX
# that is that close, else the plain sample.
TAP_MAX_M = 0.02
TAP_WINDOW_PX = 3


def sample_on_surface(texture: np.ndarray, mesh: TemplateMesh, position: np.ndarray,
                      valid: np.ndarray) -> tuple[np.ndarray, dict]:
    """mesh.sample, from the taps on the vertex's own part of the dog
    (TAP_MAX_M)."""
    height, width = texture.shape[:2]
    points = mesh.positions
    x = np.clip(mesh.uvs[:, 0], 0, width - 1)
    y = np.clip(mesh.uvs[:, 1], 0, height - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    fx, fy = x - x0, y - y0

    def near(ty: np.ndarray, tx: np.ndarray) -> np.ndarray:
        return valid[ty, tx] & (np.linalg.norm(position[ty, tx] - points, axis=1) < TAP_MAX_M)

    total = np.zeros((len(x), texture.shape[2]), dtype=np.float64)
    weight = np.zeros(len(x), dtype=np.float64)
    dropped = np.zeros(len(x), dtype=bool)
    for ty, tx, w in ((y0, x0, (1 - fx) * (1 - fy)), (y0, x1, fx * (1 - fy)),
                      (y1, x0, (1 - fx) * fy), (y1, x1, fx * fy)):
        ok = near(ty, tx)
        dropped |= valid[ty, tx] & ~ok & (w > 1e-6)
        w = w * ok
        total += texture[ty, tx] * w[:, None]
        weight += w
    out = (total / np.maximum(weight, 1e-12)[:, None]).astype(np.float32)

    lost = weight <= 1e-9
    best = np.full(len(x), np.inf)
    by, bx = np.zeros(len(x), dtype=int), np.zeros(len(x), dtype=int)
    cx, cy = np.rint(x).astype(int), np.rint(y).astype(int)
    for dy in range(-TAP_WINDOW_PX, TAP_WINDOW_PX + 1):
        for dx in range(-TAP_WINDOW_PX, TAP_WINDOW_PX + 1):
            ty, tx = np.clip(cy + dy, 0, height - 1), np.clip(cx + dx, 0, width - 1)
            d = (tx - x) ** 2 + (ty - y) ** 2
            better = lost & near(ty, tx) & (d < best)
            best = np.where(better, d, best)
            by, bx = np.where(better, ty, by), np.where(better, tx, bx)
    found = lost & np.isfinite(best)
    out[found] = texture[by[found], bx[found]]
    plain = lost & ~found
    out[plain] = mesh.sample(texture)[plain]
    return out, {"far_taps_dropped": int((dropped & ~lost).sum()), "nearest_texel": int(found.sum()),
                 "plain_sample": int(plain.sum())}


def smooth_seams(texture: np.ndarray, mesh: TemplateMesh, ring_px: int = RING_PX,
                 pad_px: int = PAD_PX, position: np.ndarray | None = None,
                 valid: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """`texture` is (h, w, 3) linear; `position` and `valid` the template's
    position map, which keeps each vertex's sample on its own part of the dog
    (sample_on_surface). Returns the smoothed texture and a small report."""
    height, width = texture.shape[:2]

    # UV -> vertex colour, averaged over the vertices that share a position
    tap_report: dict = {}
    if position is not None and valid is not None and position.shape[:2] == texture.shape[:2]:
        sampled, tap_report = sample_on_surface(texture, mesh, position, valid)
    else:
        sampled = mesh.sample(texture)                   # (n, 3)
    groups = mesh.merged
    count = np.bincount(groups, minlength=groups.max() + 1).astype(np.float32)
    summed = np.zeros((count.shape[0], 3), dtype=np.float32)
    np.add.at(summed, groups, sampled)
    shared = summed / np.maximum(count, 1.0)[:, None]
    vertex_colour = shared[groups]                       # every split copy gets the shared colour

    # rebake
    field = mesh.rasterize(vertex_colour, fill=np.nan)
    covered = ~np.isnan(field[..., 0])
    # pad the field outward so the ring and the padding can read it
    indices = ndimage.distance_transform_edt(~covered, return_distances=False, return_indices=True)
    field = field[indices[0], indices[1]]

    # the ring along every island edge, inside the island
    eroded = ndimage.binary_erosion(covered, iterations=ring_px)
    ring = covered & ~eroded
    distance_in = ndimage.distance_transform_edt(covered)
    weight = np.clip(1.0 - (distance_in - 1.0) / max(ring_px, 1), 0.0, 1.0) * ring

    # keep the hair on the seam: the texture's own high frequencies
    blurred = np.stack([ndimage.gaussian_filter(texture[..., i], DETAIL_SIGMA) for i in range(3)], -1)
    detail = texture - blurred
    seam_colour = np.clip(field + detail, 0.0, 1.0)
    out = texture * (1.0 - weight[..., None]) + seam_colour * weight[..., None]

    # the padding outside the islands
    padding = ndimage.binary_dilation(covered, iterations=pad_px) & ~covered
    out = np.where(padding[..., None], np.clip(field + detail, 0.0, 1.0), out)

    seam_vertices = mesh.seam_vertices()
    before = np.abs(sampled - vertex_colour)[seam_vertices].mean() if seam_vertices.any() else 0.0
    report = {
        "ring_px": ring_px,
        "ring_texels": int(ring.sum()),
        "seam_vertices": int(seam_vertices.sum()),
        "mean_seam_gap_before": round(float(before), 4),
        "taps": tap_report,
    }
    return out.astype(np.float32), report


__all__ = ["smooth_seams", "RING_PX"]
