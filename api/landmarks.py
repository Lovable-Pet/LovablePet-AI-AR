"""The template's landmarks, and the fields the compositor builds from them.

landmarks.json (ar_pipeline/build_landmarks.py) names a handful of points
on the template - eyes, nose, ear bases, chest and belly centres, paws - as
UV texels and world positions. Two things are built from them here:

DISTANCE FIELDS, for gradients. "Tan strongest at the eye, fading to cream"
needs, for every texel of the cheeks, how far it is from the eye. That is a
distance transform inside the region's mask: the eye's texels seed it and
the distance grows through the mask only, so it follows the cheek round
the muzzle rather than jumping across the UV sheet. A region's UV island
that holds no seed - the far side of a paw whose toes sit on another
island - takes the world-space distance instead, in the same units, so the
two halves of one region agree.

TARGET POINTS, for markings. "A chocolate patch above the left eye, near"
is a world point: the eye, moved up by a step scaled to the eye distance.
The compositor then paints the region's texels around that point.

Both work in world metres through geometry/position.npy, which is what
makes "above" mean up on the animal rather than up on the UV sheet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

# Offsets for marking placement, in world axes (Blender: z up, -y forward,
# -x is the left of a front-view image).
OFFSET_VECTORS = {
    "above": np.array([0.0, 0.0, 1.0]),
    "below": np.array([0.0, 0.0, -1.0]),
    "left": np.array([-1.0, 0.0, 0.0]),
    "right": np.array([1.0, 0.0, 0.0]),
    "on": np.array([0.0, 0.0, 0.0]),
}
# ... in eye distances
OFFSET_STEP = {"near": 0.7, "far": 1.8}

# The seed of a distance field: mask texels within this many eye distances
# of the anchor point.
SEED_RADIUS_EYES = 0.35


@dataclass
class Landmarks:
    points: dict[str, dict[str, Any]]
    anchors: dict[str, list[str]]
    eye_distance: float
    position: np.ndarray | None      # (h, w, 3) world, or None without geometry
    valid: np.ndarray | None
    size: tuple[int, int]
    normal: np.ndarray | None = None  # (h, w, 3) unit world normals

    @classmethod
    def load(cls, template_dir: str | Path, geometry_dir: str | Path | None = None) -> "Landmarks":
        template_dir = Path(template_dir)
        data = json.loads((template_dir / "landmarks.json").read_text(encoding="utf-8"))
        position = valid = normal = None
        if geometry_dir is not None and (Path(geometry_dir) / "geometry.json").exists():
            manifest = json.loads((Path(geometry_dir) / "geometry.json").read_text(encoding="utf-8"))
            position = np.load(Path(geometry_dir) / manifest["position"]["file"]).astype(np.float32)
            valid = np.load(Path(geometry_dir) / manifest["valid"]["file"]) > 0.5
            normal = np.load(Path(geometry_dir) / manifest["normal"]["file"]).astype(np.float32)
        return cls(
            points=data["points"],
            anchors=data["anchors"],
            eye_distance=float(data.get("eye_distance_world", 0.13)),
            position=position,
            valid=valid,
            size=tuple(data["size"]),
            normal=normal,
        )

    # ------------------------------------------------------------------
    def world(self, name: str) -> np.ndarray:
        return np.asarray(self.points[name]["world"], dtype=np.float32)

    def uv(self, name: str) -> tuple[int, int]:
        x, y = self.points[name]["uv"]
        return int(x), int(y)

    def anchor_points(self, anchor: str) -> list[str]:
        """The landmark names an anchor word stands for. Marking anchors name
        one eye; gradient anchors name both."""
        if anchor in self.points:
            return [anchor]
        return list(self.anchors.get(anchor, []))

    # ------------------------------------------------------------------
    def texel_metres(self) -> float:
        """How long a texel is on the surface, roughly - the median step of
        the position map between valid neighbours."""
        if self.position is None:
            return 1e-3
        interior = ndimage.binary_erosion(self.valid, iterations=1)
        step = np.linalg.norm(self.position[:, 1:] - self.position[:, :-1], axis=-1)[interior[:, 1:] & interior[:, :-1]]
        return float(np.median(step)) if step.size else 1e-3

    def distance_field(self, anchor: str, mask: np.ndarray) -> np.ndarray | None:
        """Distance from the anchor through the mask, in metres, for every
        texel of the mask; NaN outside it. None when the anchor is unknown
        or the geometry is missing."""
        names = self.anchor_points(anchor)
        if not names or self.position is None:
            return None
        inside = mask > 0.5
        if not inside.any():
            return None

        points = np.stack([self.world(name) for name in names])
        world_distance = np.full(mask.shape, np.inf, dtype=np.float32)
        rows, cols = np.nonzero(inside)
        pos = self.position[rows, cols]
        for point in points:
            world_distance[rows, cols] = np.minimum(
                world_distance[rows, cols], np.linalg.norm(pos - point, axis=1))

        seed = inside & (world_distance <= SEED_RADIUS_EYES * self.eye_distance)
        metres = self.texel_metres()
        out = np.full(mask.shape, np.nan, dtype=np.float32)

        labels, count = ndimage.label(inside)
        for index in range(1, count + 1):
            island = labels == index
            island_seed = seed & island
            if not island_seed.any():
                # no seed on this island: the straight world distance
                out[island] = world_distance[island]
                continue
            # geodesic distance through the island from the seed, in texels
            geodesic = _geodesic(island, island_seed)
            out[island] = geodesic[island] * metres + (SEED_RADIUS_EYES * self.eye_distance)
            out[island_seed] = world_distance[island_seed]
        return out

    def target_point(self, anchor: str, offset: str, distance: str) -> np.ndarray | None:
        names = self.anchor_points(anchor)
        if not names:
            return None
        base = np.stack([self.world(name) for name in names]).mean(axis=0)
        step = OFFSET_STEP.get(distance, OFFSET_STEP["near"]) * self.eye_distance
        return base + OFFSET_VECTORS.get(offset, OFFSET_VECTORS["on"]) * step

    def midline_strip(self, mask: np.ndarray, half_width: float,
                      y_max: float | None = None, y_min: float | None = None) -> np.ndarray | None:
        """A soft band of the mask's texels along the body's midline (world
        x = 0): 1 inside six tenths of `half_width` metres, falling to 0 at
        `half_width`. `y_max` / `y_min` cut the band in world y (-y is
        forward), so a blaze can stop at the ear bases instead of running
        down the neck. None without geometry."""
        if self.position is None:
            return None
        inside = mask > 0.5
        rows, cols = np.nonzero(inside)
        if rows.size == 0:
            return None
        pos = self.position[rows, cols]
        lateral = np.abs(pos[:, 0])
        weight = np.clip(1.0 - (lateral - 0.6 * half_width) / max(0.4 * half_width, 1e-6), 0.0, 1.0)
        if y_max is not None:
            weight[pos[:, 1] > y_max] = 0.0
        if y_min is not None:
            weight[pos[:, 1] < y_min] = 0.0
        out = np.zeros(mask.shape, dtype=np.float32)
        out[rows, cols] = weight
        return out

    def blob(self, point: np.ndarray, mask: np.ndarray, radius: float) -> np.ndarray:
        """A soft disc of the mask's texels within `radius` metres of `point`,
        1 at the centre falling to 0 at the edge."""
        if self.position is None:
            return np.zeros_like(mask)
        inside = mask > 0.5
        rows, cols = np.nonzero(inside)
        distance = np.linalg.norm(self.position[rows, cols] - point, axis=1)
        weight = np.clip(1.0 - distance / max(radius, 1e-6), 0.0, 1.0)
        out = np.zeros(mask.shape, dtype=np.float32)
        out[rows, cols] = weight
        return out


def _geodesic(island: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Distance in texels from `seed` through `island` only."""
    from skimage.graph import MCP_Geometric

    rows, cols = np.nonzero(island)
    y0, y1 = rows.min(), rows.max() + 1
    x0, x1 = cols.min(), cols.max() + 1
    cost = np.where(island[y0:y1, x0:x1], 1.0, np.inf).astype(np.float64)
    starts = np.argwhere(seed[y0:y1, x0:x1])
    mcp = MCP_Geometric(cost, fully_connected=True)
    costs, _ = mcp.find_costs(starts)
    out = np.full(island.shape, np.inf, dtype=np.float32)
    out[y0:y1, x0:x1] = costs
    return out


__all__ = ["Landmarks", "OFFSET_STEP", "OFFSET_VECTORS"]
