"""The template mesh without Blender: vertices, UVs, triangles and joints.

Stage 1 bakes what it can into UV space (masks, position, normal) so the
texture stages are plain array arithmetic. Two later steps still need the
mesh itself: the landmark builder wants the joint positions of the rig, and
the seam post-process needs to know which UV vertices are the same mesh
vertex, which only the index buffer can say. Both read the GLB directly with
pygltflib, so neither needs Blender.

SPACES
------
glTF is Y-up with -Z forward; Blender, and every baked array in output/, is
Z-up with -Y forward. Everything returned here is converted to the Blender
convention so it lines up with position.npy and cameras.json:

    blender (x, y, z) = (gltf x, -gltf z, gltf y)

The mesh is skinned, so its vertex positions are already in world space and
the mesh node's own transform is ignored, as the glTF specification says.
Joint positions come from composing the node tree's rest transforms.

UV space is texel coordinates: x to the right, y down, origin at the top-left
of the image, matching every mask PNG.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pygltflib import GLTF2

_COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
              5125: np.uint32, 5126: np.float32}
_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def gltf_to_blender(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return np.stack([points[..., 0], -points[..., 2], points[..., 1]], axis=-1)


def _accessor(gltf: GLTF2, blob: bytes, index: int) -> np.ndarray:
    accessor = gltf.accessors[index]
    view = gltf.bufferViews[accessor.bufferView]
    dtype = np.dtype(_COMPONENT[accessor.componentType])
    width = _COUNT[accessor.type]
    start = (view.byteOffset or 0) + (accessor.byteOffset or 0)
    stride = view.byteStride or dtype.itemsize * width
    if stride == dtype.itemsize * width:
        data = np.frombuffer(blob, dtype=dtype, count=accessor.count * width, offset=start)
        return data.reshape(accessor.count, width)
    rows = [np.frombuffer(blob, dtype=dtype, count=width, offset=start + i * stride)
            for i in range(accessor.count)]
    return np.stack(rows)


def _quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _local_matrix(node) -> np.ndarray:
    if node.matrix:
        return np.array(node.matrix, dtype=np.float64).reshape(4, 4).T
    matrix = np.eye(4)
    rotation = _quat_to_matrix(node.rotation or [0, 0, 0, 1])
    scale = np.array(node.scale or [1, 1, 1], dtype=np.float64)
    matrix[:3, :3] = rotation * scale
    matrix[:3, 3] = node.translation or [0, 0, 0]
    return matrix


@dataclass
class TemplateMesh:
    positions: np.ndarray      # (n, 3) blender space, metres
    uvs: np.ndarray            # (n, 2) texel coordinates for `size`
    triangles: np.ndarray      # (t, 3) indices into positions/uvs
    merged: np.ndarray         # (n,) id shared by vertices at the same position
    joints: dict[str, np.ndarray]   # bare joint name -> blender-space position
    size: tuple[int, int]

    @classmethod
    def load(cls, glb_path: str | Path, size: tuple[int, int] = (2048, 2048),
             merge_tolerance: float = 1e-5) -> "TemplateMesh":
        gltf = GLTF2().load(str(glb_path))
        blob = gltf.binary_blob()

        # the body: the primitive with the most vertices
        best = None
        for mesh in gltf.meshes or []:
            for primitive in mesh.primitives:
                count = gltf.accessors[primitive.attributes.POSITION].count
                if best is None or count > best[0]:
                    best = (count, primitive)
        if best is None:
            raise ValueError(f"{glb_path} has no mesh primitive")
        primitive = best[1]

        positions = gltf_to_blender(_accessor(gltf, blob, primitive.attributes.POSITION))
        uv = _accessor(gltf, blob, primitive.attributes.TEXCOORD_0).astype(np.float64)
        width, height = size
        # glTF UV origin is top-left with v down, the same as image space
        texel = np.stack([uv[:, 0] * (width - 1), uv[:, 1] * (height - 1)], axis=1)
        if primitive.indices is not None:
            triangles = _accessor(gltf, blob, primitive.indices).reshape(-1, 3).astype(np.int64)
        else:
            triangles = np.arange(len(positions), dtype=np.int64).reshape(-1, 3)

        # vertices split along UV seams share a position; give them one id
        quantised = np.round(positions / merge_tolerance).astype(np.int64)
        _, merged = np.unique(quantised, axis=0, return_inverse=True)

        joints = cls._joints(gltf)
        return cls(positions.astype(np.float32), texel.astype(np.float32), triangles,
                   merged.reshape(-1), joints, size)

    @staticmethod
    def _joints(gltf: GLTF2) -> dict[str, np.ndarray]:
        parent: dict[int, int] = {}
        for index, node in enumerate(gltf.nodes or []):
            for child in node.children or []:
                parent[child] = index
        world: dict[int, np.ndarray] = {}

        def resolve(index: int) -> np.ndarray:
            if index not in world:
                matrix = _local_matrix(gltf.nodes[index])
                if index in parent:
                    matrix = resolve(parent[index]) @ matrix
                world[index] = matrix
            return world[index]

        joints: dict[str, np.ndarray] = {}
        for index, node in enumerate(gltf.nodes or []):
            if not node.name:
                continue
            # Blender exports "name.123"; the bare name is what people use
            bare = node.name.rsplit(".", 1)[0] if node.name.rsplit(".", 1)[-1].isdigit() else node.name
            position = resolve(index)[:3, 3]
            joints.setdefault(bare, gltf_to_blender(position).astype(np.float32))
        return joints

    # ------------------------------------------------------------------
    # UV rasterisation
    # ------------------------------------------------------------------

    def rasterize(self, values: np.ndarray, fill: float = 0.0) -> np.ndarray:
        """Interpolate per-vertex values across every UV triangle.

        `values` is (n,) or (n, k). Returns (height, width) or (height,
        width, k). Texels no triangle covers hold `fill`. Triangles are
        drawn in order, later ones over earlier; a UV layout with no
        overlapping islands makes the order irrelevant.
        """
        width, height = self.size
        values = np.asarray(values, dtype=np.float32)
        scalar = values.ndim == 1
        if scalar:
            values = values[:, None]
        out = np.full((height, width, values.shape[1]), fill, dtype=np.float32)
        uv = self.uvs
        for tri in self.triangles:
            xs, ys = uv[tri, 0], uv[tri, 1]
            x0, x1 = max(int(np.floor(xs.min())), 0), min(int(np.ceil(xs.max())), width - 1)
            y0, y1 = max(int(np.floor(ys.min())), 0), min(int(np.ceil(ys.max())), height - 1)
            if x1 < x0 or y1 < y0:
                continue
            denom = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
            if abs(denom) < 1e-12:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
            l0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / denom
            l1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / denom
            l2 = 1.0 - l0 - l1
            eps = -1e-3
            inside = (l0 >= eps) & (l1 >= eps) & (l2 >= eps)
            if not inside.any():
                continue
            v = values[tri]
            interpolated = (l0[..., None] * v[0] + l1[..., None] * v[1] + l2[..., None] * v[2])
            block = out[y0:y1 + 1, x0:x1 + 1]
            block[inside] = interpolated[inside]
        return out[..., 0] if scalar else out

    def sample(self, image: np.ndarray) -> np.ndarray:
        """Bilinear sample of an (h, w[, k]) image at every vertex's UV."""
        height, width = image.shape[:2]
        x = np.clip(self.uvs[:, 0], 0, width - 1)
        y = np.clip(self.uvs[:, 1], 0, height - 1)
        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        x1 = np.minimum(x0 + 1, width - 1)
        y1 = np.minimum(y0 + 1, height - 1)
        fx = (x - x0)[:, None] if image.ndim == 3 else (x - x0)
        fy = (y - y0)[:, None] if image.ndim == 3 else (y - y0)
        top = image[y0, x0] * (1 - fx) + image[y0, x1] * fx
        bottom = image[y1, x0] * (1 - fx) + image[y1, x1] * fx
        return top * (1 - fy) + bottom * fy

    def seam_vertices(self) -> np.ndarray:
        """Vertices that share a position with at least one other vertex - the
        ones a UV seam runs through."""
        counts = np.bincount(self.merged, minlength=self.merged.max() + 1)
        return counts[self.merged] > 1


__all__ = ["TemplateMesh", "gltf_to_blender"]
