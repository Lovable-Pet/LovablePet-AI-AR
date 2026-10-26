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


def smooth_seams(texture: np.ndarray, mesh: TemplateMesh, ring_px: int = RING_PX,
                 pad_px: int = PAD_PX) -> tuple[np.ndarray, dict]:
    """`texture` is (h, w, 3) linear. Returns the smoothed texture and a
    small report."""
    height, width = texture.shape[:2]

    # UV -> vertex colour, averaged over the vertices that share a position
    sampled = mesh.sample(texture)                       # (n, 3)
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
    }
    return out.astype(np.float32), report


__all__ = ["smooth_seams", "RING_PX"]
