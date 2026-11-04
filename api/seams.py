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
# the ring takes the rebake where the copies of a seam vertex differ by more
# than SEAM_AGREE in lightness (0..1, gamma-encoded), fully at SEAM_DISAGREE
SEAM_AGREE = 0.02
SEAM_DISAGREE = 0.08
# A vertex's bilinear sample leaves out the taps baked further than this from
# the vertex (the template's position map): a corner on the edge of a strip
# packed into another chart read that chart - on the jindo puppy 2 the
# shoulder seam's corners read the white bib 12-18 cm away, and the shared
# seam colour drew a grey zigzag down the black shoulder (shelter puppy 11).
# With no tap left, the vertex takes the nearest texel within TAP_WINDOW_PX
# that is that close, else the plain sample.
TAP_MAX_M = 0.02
TAP_WINDOW_PX = 3


# Along each island's rim, SEAM_TEXEL_PX deep, a texel is blended toward the
# nearest texel of another island within SEAM_TEXEL_M on the dog (half-way at
# the rim, fading inward): both sides of a seam meet on one colour.
SEAM_TEXEL_PX = 3.0
SEAM_TEXEL_M = 0.003


def blend_across_seams(texture: np.ndarray, position: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict]:
    from scipy.spatial import cKDTree

    labels, _ = ndimage.label(valid)
    depth = ndimage.distance_transform_edt(valid)
    rim = valid & (depth <= SEAM_TEXEL_PX + 0.5)
    rows, cols = np.nonzero(rim)
    points = position[rows, cols].astype(np.float64)
    own = labels[rows, cols]
    tree = cKDTree(points)
    distance, index = tree.query(points, k=12, distance_upper_bound=SEAM_TEXEL_M, workers=-1)
    found = np.isfinite(distance)
    other = np.where(found, own[np.minimum(index, len(own) - 1)], -1) != own[:, None]
    other &= found
    first = np.argmax(other, axis=1)
    has = other[np.arange(len(own)), first]
    partner = index[np.arange(len(own)), first]
    out = texture.copy()
    r, c = rows[has], cols[has]
    pr, pc = rows[partner[has]], cols[partner[has]]
    weight = np.clip(0.5 * (1.0 - (depth[r, c] - 1.0) / SEAM_TEXEL_PX), 0.0, 0.5)[:, None]
    out[r, c] = texture[r, c] * (1.0 - weight) + texture[pr, pc] * weight
    return out, {"rim_texels": int(len(own)), "blended": int(has.sum())}


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

    # how far the copies of each vertex disagree (perceptual lightness): the
    # ring takes the rebake only that much (SEAM_AGREE / SEAM_DISAGREE).
    # Where the texture already agrees over a seam, the rebake - a field
    # interpolated across triangles a centimetre or two wide - only drew
    # their shape into the seam: a colour edge crossing a seam turned into
    # straight-edged white wedges in a black-and-white coat's black (user,
    # 11-05: "색깔 경계면에 번개무늬")
    light = np.power(np.clip(sampled @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), 0.0, 1.0), 1.0 / 2.2)
    hi = np.full(count.shape[0], -np.inf, dtype=np.float32)
    lo = np.full(count.shape[0], np.inf, dtype=np.float32)
    np.maximum.at(hi, groups, light)
    np.minimum.at(lo, groups, light)
    spread = (hi - lo)[groups]
    need = np.clip((spread - SEAM_AGREE) / (SEAM_DISAGREE - SEAM_AGREE), 0.0, 1.0).astype(np.float32)

    # rebake
    field = mesh.rasterize(vertex_colour, fill=np.nan)
    need_field = mesh.rasterize(need, fill=0.0)
    covered = ~np.isnan(field[..., 0])
    # pad the field outward so the ring and the padding can read it
    indices = ndimage.distance_transform_edt(~covered, return_distances=False, return_indices=True)
    field = field[indices[0], indices[1]]

    # the ring along every island edge, inside the island
    eroded = ndimage.binary_erosion(covered, iterations=ring_px)
    ring = covered & ~eroded
    distance_in = ndimage.distance_transform_edt(covered)
    weight = np.clip(1.0 - (distance_in - 1.0) / max(ring_px, 1), 0.0, 1.0) * ring * need_field

    # keep the hair on the seam: the texture's own high frequencies
    blurred = np.stack([ndimage.gaussian_filter(texture[..., i], DETAIL_SIGMA) for i in range(3)], -1)
    detail = texture - blurred
    seam_colour = np.clip(field + detail, 0.0, 1.0)
    across_report: dict = {}
    if position is not None and valid is not None and position.shape[:2] == texture.shape[:2]:
        # texel to texel instead (SEAM_TEXEL_*): the rebake is a field of
        # vertex colours drawn across triangles, and along a seam the
        # triangles are often slivers lying wholly in the ring - one vertex
        # that sampled the tan below a black upper arm fanned out as rays of
        # tan up the slivers, a fork 2 cm long in the black (user, 11-05:
        # "포크 모양으로 뭔가 생겼는데")
        out, across_report = blend_across_seams(texture, position, valid)
    else:
        out = texture * (1.0 - weight[..., None]) + seam_colour * weight[..., None]

    # the padding outside the islands - not the texels of an island that fall
    # in the hairline gaps between its triangles (two copies of a vertex a
    # fraction of a texel apart in UV): filled from the vertex field, they
    # ran as lines of the far vertex's colour along the triangle edges, a
    # fork of tan where slivers fan out in a black upper arm (user, 11-05:
    # "포크 모양으로 뭔가 생겼는데")
    padding = ndimage.binary_dilation(covered, iterations=pad_px) & ~covered
    if valid is not None and valid.shape == padding.shape:
        padding &= ~valid
    out = np.where(padding[..., None], np.clip(field + detail, 0.0, 1.0), out)

    seam_vertices = mesh.seam_vertices()
    before = np.abs(sampled - vertex_colour)[seam_vertices].mean() if seam_vertices.any() else 0.0
    report = {
        "ring_px": ring_px,
        "ring_share_rebaked": round(float(weight[ring].mean()) if ring.any() else 0.0, 3),
        "ring_texels": int(ring.sum()),
        "seam_vertices": int(seam_vertices.sum()),
        "mean_seam_gap_before": round(float(before), 4),
        "taps": tap_report,
        "across": across_report,
    }
    return out.astype(np.float32), report


__all__ = ["smooth_seams", "RING_PX"]


# A triangle whose UV lands this far (in 3D) from its own surface - one the
# template's UV fixes moved after the position map was baked, onto texels
# the bake left uncovered (a hole in an island, the padding between them) -
# reads whatever the padding holds: on a black-and-tan shiba's tan foreleg a
# white triangle (the padding beside the white front of the leg) and on a
# white one pale angular patches (user, 11-05: "왼쪽 앞다리에 알 수 없는
# 하얀색 네모"; 74 such triangles on the jindo template).
STRAY_TRIANGLE_M = 0.02


def fill_stray_triangles(texture: np.ndarray, mesh: TemplateMesh, position: np.ndarray,
                         valid: np.ndarray) -> tuple[np.ndarray, dict]:
    """The uncovered texels under each stray triangle (STRAY_TRIANGLE_M)
    painted with the colour of the surface it lies on: each corner takes
    the texture at the baked texel nearest it in 3D, interpolated across
    the triangle (and one texel round it, for the filter). Texels an island
    owns are left alone."""
    from scipy.spatial import cKDTree

    height, width = valid.shape
    tri = mesh.triangles
    uv = mesh.uvs
    centre = mesh.positions[tri].mean(axis=1)
    cuv = uv[tri].mean(axis=1)
    col = np.clip(np.round(cuv[:, 0]).astype(int), 0, width - 1)
    row = np.clip(np.round(cuv[:, 1]).astype(int), 0, height - 1)
    nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    landed = position[nearest[0][row, col], nearest[1][row, col]]
    stray = np.flatnonzero(np.linalg.norm(landed - centre, axis=1) > STRAY_TRIANGLE_M)
    report = {"stray_triangles": int(len(stray))}
    if not len(stray):
        return texture, report
    rows, cols = np.nonzero(valid)
    tree = cKDTree(position[rows, cols])
    corners = np.unique(tri[stray])
    _, hit = tree.query(mesh.positions[corners])
    colour = np.zeros((len(mesh.positions), texture.shape[2]), dtype=np.float32)
    colour[corners] = texture[rows[hit], cols[hit]]
    out = texture.copy()
    painted = 0
    for t in stray:
        xs, ys = uv[tri[t], 0], uv[tri[t], 1]
        x0, x1 = max(int(np.floor(xs.min())) - 1, 0), min(int(np.ceil(xs.max())) + 1, width - 1)
        y0, y1 = max(int(np.floor(ys.min())) - 1, 0), min(int(np.ceil(ys.max())) + 1, height - 1)
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        denom = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if abs(denom) < 1e-9:
            # a sliver: the corners' mean over its box
            weights = np.full(gx.shape + (3,), 1.0 / 3.0)
        else:
            l0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / denom
            l1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / denom
            weights = np.clip(np.stack([l0, l1, 1.0 - l0 - l1], axis=-1), 0.0, 1.0)
            weights = weights / np.maximum(weights.sum(axis=-1, keepdims=True), 1e-9)
        value = np.einsum("hwk,kc->hwc", weights, colour[tri[t]])
        free = ~valid[y0:y1 + 1, x0:x1 + 1]
        block = out[y0:y1 + 1, x0:x1 + 1]
        block[free] = value[free]
        painted += int(free.sum())
    report["texels_painted"] = painted
    return out, report
