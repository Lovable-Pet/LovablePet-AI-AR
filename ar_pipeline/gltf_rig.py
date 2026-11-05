"""A rigged glTF's node tree and clips as numbers, for the scripts that pose,
re-weight or animate a template without a Blender round trip (pose_clips,
rig_fix): accessor reads, rest transforms by forward kinematics, and a
clip sampled at a fixed rate."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from pygltflib import GLTF2, Accessor, BufferView
from scipy.spatial.transform import Rotation, Slerp

COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
DTYPES = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}


def read(gltf: GLTF2, blob: bytes, index: int) -> np.ndarray:
    acc = gltf.accessors[index]
    view = gltf.bufferViews[acc.bufferView]
    dtype, width = DTYPES[acc.componentType], COMPONENTS[acc.type]
    item = np.dtype(dtype).itemsize * width
    stride = view.byteStride or item
    start = (view.byteOffset or 0) + (acc.byteOffset or 0)
    if stride == item:
        return np.frombuffer(blob, dtype, acc.count * width, start).reshape(acc.count, width).astype(np.float64)
    return np.stack([np.frombuffer(blob, dtype, width, start + i * stride) for i in range(acc.count)]).astype(np.float64)


def append_accessor(gltf: GLTF2, blob: bytearray, data: np.ndarray, kind: str, bounds: bool = False,
                    component: int = 5126) -> int:
    """A new accessor (and buffer view) for `data` at the end of `blob`,
    float unless `component` names another type (joints, indices)."""
    while len(blob) % 4:
        blob.append(0)
    raw = np.ascontiguousarray(data, dtype=DTYPES[component]).tobytes()
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=len(blob), byteLength=len(raw)))
    blob.extend(raw)
    acc = Accessor(bufferView=len(gltf.bufferViews) - 1, componentType=component, count=len(data), type=kind)
    if bounds:
        acc.min = [float(v) for v in np.atleast_1d(data.min(axis=0))]
        acc.max = [float(v) for v in np.atleast_1d(data.max(axis=0))]
    gltf.accessors.append(acc)
    return len(gltf.accessors) - 1


def prune(gltf: GLTF2, blob: bytes) -> bytearray:
    """Drop the accessors and buffer views nothing refers to any more (a
    replaced clip's, a re-cut mesh's), and pack the buffer."""
    used: set[int] = set()
    for mesh in gltf.meshes:
        for prim in mesh.primitives:
            used.update(v for v in vars(prim.attributes).values() if isinstance(v, int))
            if prim.indices is not None:
                used.add(prim.indices)
            for target in prim.targets or []:
                used.update(v for v in target.values() if isinstance(v, int))
    for skin in gltf.skins:
        if skin.inverseBindMatrices is not None:
            used.add(skin.inverseBindMatrices)
    for anim in gltf.animations or []:
        for s in anim.samplers:
            used.update((s.input, s.output))
    order = sorted(used)
    remap = {old: new for new, old in enumerate(order)}
    gltf.accessors = [gltf.accessors[i] for i in order]
    for mesh in gltf.meshes:
        for prim in mesh.primitives:
            for key, v in vars(prim.attributes).items():
                if isinstance(v, int):
                    setattr(prim.attributes, key, remap[v])
            if prim.indices is not None:
                prim.indices = remap[prim.indices]
            for target in prim.targets or []:
                for key, v in list(target.items()):
                    if isinstance(v, int):
                        target[key] = remap[v]
    for skin in gltf.skins:
        if skin.inverseBindMatrices is not None:
            skin.inverseBindMatrices = remap[skin.inverseBindMatrices]
    for anim in gltf.animations or []:
        for s in anim.samplers:
            s.input, s.output = remap[s.input], remap[s.output]
    views = sorted({a.bufferView for a in gltf.accessors if a.bufferView is not None}
                   | {im.bufferView for im in gltf.images or [] if im.bufferView is not None})
    packed, vmap, kept = bytearray(), {}, []
    for old in views:
        view = gltf.bufferViews[old]
        while len(packed) % 4:
            packed.append(0)
        start = view.byteOffset or 0
        chunk = blob[start:start + view.byteLength]
        view.byteOffset = len(packed)
        packed.extend(chunk)
        vmap[old] = len(kept)
        kept.append(view)
    for item in list(gltf.accessors) + list(gltf.images or []):
        if item.bufferView is not None:
            item.bufferView = vmap[item.bufferView]
    gltf.bufferViews = kept
    return packed


class Rig:
    """A glTF's node tree, its skin's joints by bare name (a ".NN" suffix
    dropped), and forward kinematics with some nodes' local TRS overridden."""

    def __init__(self, path: Path):
        self.gltf = GLTF2().load(str(path))
        self.blob = self.gltf.binary_blob()
        nodes = self.gltf.nodes
        self.parent = {c: i for i, n in enumerate(nodes) for c in (n.children or [])}
        self.joints = list(self.gltf.skins[0].joints)
        self.by_name = {(nodes[j].name or "").split(".")[0]: j for j in self.joints}
        self.refresh()
        depth = {i: self._depth(i) for i in range(len(nodes))}
        self.order = sorted(range(len(nodes)), key=depth.get)          # parents first

    def refresh(self) -> None:
        """Re-read the nodes' rest TRS (after editing them)."""
        self.rest_trs = {i: self._trs(i) for i in range(len(self.gltf.nodes))}

    def _depth(self, i: int) -> int:
        d = 0
        while i in self.parent:
            i, d = self.parent[i], d + 1
        return d

    def _trs(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.gltf.nodes[i]
        if n.matrix:
            m = np.array(n.matrix, dtype=np.float64).reshape(4, 4).T
            scale = np.linalg.norm(m[:3, :3], axis=0)
            return m[:3, 3].copy(), Rotation.from_matrix(m[:3, :3] / scale).as_quat(), scale
        return (np.array(n.translation or [0, 0, 0], dtype=np.float64),
                np.array(n.rotation or [0, 0, 0, 1], dtype=np.float64),
                np.array(n.scale or [1, 1, 1], dtype=np.float64))

    def world(self, overrides: dict[int, tuple] | None = None) -> dict[int, np.ndarray]:
        """Every node's world matrix, with `overrides` {node: (t, q, s)}."""
        out: dict[int, np.ndarray] = {}
        for i in self.order:
            t, q, s = (overrides or {}).get(i, self.rest_trs[i])
            m = np.eye(4)
            m[:3, :3] = Rotation.from_quat(q).as_matrix() * s
            m[:3, 3] = t
            out[i] = out[self.parent[i]] @ m if i in self.parent else m
        return out


def sample_clip(rig: Rig, name: str, fps: float) -> tuple[np.ndarray, list[dict[int, tuple]]]:
    """(times, per frame {node: (t, q, s)}) of clip `name` at `fps`."""
    anims = [a for a in rig.gltf.animations or [] if a.name == name]
    if not anims:
        raise SystemExit(f"no clip {name!r}; have {[a.name for a in rig.gltf.animations or []]}")
    anim = anims[0]
    end = max(float(read(rig.gltf, rig.blob, s.input).max()) for s in anim.samplers)
    times = np.arange(0.0, end + 1e-6, 1.0 / fps)
    frames = [{} for _ in times]
    for channel in anim.channels:
        sampler = anim.samplers[channel.sampler]
        node, path = channel.target.node, channel.target.path
        if path not in ("translation", "rotation", "scale"):
            continue
        keys = read(rig.gltf, rig.blob, sampler.input).ravel()
        values = read(rig.gltf, rig.blob, sampler.output)
        if sampler.interpolation == "CUBICSPLINE":
            values = values.reshape(len(keys), 3, -1)[:, 1]      # the value, without the tangents
        t = np.clip(times, keys[0], keys[-1])
        if sampler.interpolation == "STEP" or len(keys) == 1:
            sampled = values[np.clip(np.searchsorted(keys, t, side="right") - 1, 0, len(keys) - 1)]
        elif path == "rotation":
            sampled = Slerp(keys, Rotation.from_quat(values))(t).as_quat()
        else:
            sampled = np.stack([np.interp(t, keys, values[:, k]) for k in range(values.shape[1])], axis=1)
        slot = {"translation": 0, "rotation": 1, "scale": 2}[path]
        for f, value in enumerate(sampled):
            trs = list(frames[f].get(node, rig.rest_trs[node]))
            trs[slot] = value
            frames[f][node] = tuple(trs)
    return times, frames
