"""A pet GLB as a USDZ for iPhone AR (AR Quick Look, RealityKit).

iOS Safari has no WebXR, so on an iPhone the AR page hands the model to AR
Quick Look, which reads USDZ only (model-viewer `ios-src`, or a plain
`<a rel="ar" href="model.usdz">`). This writes that file from the GLB the
pipeline already makes, with the same mesh, skin, skeleton and texture, and
one of its clips:

- Quick Look plays one animation, the file's timeline, on a loop. So a USDZ
  carries one clip (default the loop the app plays at rest: the jindo's
  look-around `standing`, else `idle`); `--all` writes one file per clip.
- The skeleton keeps glTF's joints, rest pose and inverse bind matrices; the
  scene root above it (the real-size scale, api/size.py) becomes the
  SkelRoot's transform. Joint transforms are sampled at every key time of the
  clip, so a LINEAR clip comes through exactly (USD interpolates samples
  linearly, quaternions by slerp); STEP keys get a second sample just before
  the next key, CUBICSPLINE is resampled at 60 fps.
- Morph targets carry no clip in our GLBs (the breed shape is the mesh's
  default weights), so the weights are baked into the points.
- Vertex colours (the jindo's mouth) are ignored by Quick Look, so they are
  baked into a small texture: each triangle gets its own 8 px cell, filled
  with its three colours interpolated as the GPU would, and face-varying UVs.

Only what Quick Look renders goes in: Mesh, SkelRoot, Skeleton,
SkelAnimation, UsdPreviewSurface with UsdUVTexture and the st reader, PNG or
JPEG textures, Y up, metres.

    python -m api.usdz output/runs/<run>/final/<pet>.glb             # <pet>.usdz, rest loop
    python -m api.usdz <glb> --clip bark -o bark.usdz
    python -m api.usdz <glb> --all                                   # <pet>_<clip>.usdz each
"""

from __future__ import annotations

import argparse
import io
import math
import re
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pygltflib import GLTF2
from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdShade, UsdSkel, Vt
from scipy.spatial.transform import Rotation, Slerp

# the loop the app plays at rest, the first of these the GLB has (as in
# docs/API.md's three.js code): the jindo's look-around `standing`, else the
# breathing-only `idle` of the other templates. Quick Look shows only this
REST_CLIPS = ("standing", "idle")
# glTF accessor component type -> numpy dtype, and the divisor of a normalized one
_DTYPE = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NORM = {5120: 127.0, 5121: 255.0, 5122: 32767.0, 5123: 65535.0}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
_WRAP = {10497: "repeat", 33071: "clamp", 33648: "mirror"}
_CUBIC_FPS = 60.0
# a STEP key holds until this long (s) before the next one
_STEP_EPS = 1e-3
# vertex-colour bake: one triangle per cell this many px across, its legs
# this far in from the cell's edge so bilinear filtering stays in the cell
_CELL, _PAD = 8, 1.0


class UsdzError(RuntimeError):
    """The GLB uses something this converter does not carry over."""


# --------------------------------------------------------------------------- glTF


class _Glb:
    def __init__(self, path: Path):
        self.path = path
        self.gltf = GLTF2().load(str(path))
        self.blob = self.gltf.binary_blob() or b""
        nodes = self.gltf.nodes or []
        self.parent = {c: i for i, n in enumerate(nodes) for c in (n.children or [])}
        self.rest = [self._node_trs(i) for i in range(len(nodes))]
        depth = {i: self.depth(i) for i in range(len(nodes))}
        self.order = sorted(range(len(nodes)), key=depth.get)          # parents first

    def depth(self, i: int) -> int:
        d = 0
        while i in self.parent:
            i, d = self.parent[i], d + 1
        return d

    def read(self, index: int) -> np.ndarray:
        """Accessor `index` as (count, width) float64, normalized and sparse applied."""
        acc = self.gltf.accessors[index]
        dtype, width = _DTYPE[acc.componentType], _WIDTH[acc.type]
        if acc.bufferView is None:
            out = np.zeros((acc.count, width))
        else:
            view = self.gltf.bufferViews[acc.bufferView]
            item = np.dtype(dtype).itemsize * width
            stride = view.byteStride or item
            start = (view.byteOffset or 0) + (acc.byteOffset or 0)
            raw = np.frombuffer(self.blob, np.uint8, stride * (acc.count - 1) + item, start)
            rows = np.lib.stride_tricks.as_strided(raw, (acc.count, item), (stride, 1))
            out = np.ascontiguousarray(rows).view(dtype).reshape(acc.count, width).astype(np.float64)
        if acc.sparse is not None:
            sp = acc.sparse
            iv = self.gltf.bufferViews[sp.indices.bufferView]
            idx = np.frombuffer(self.blob, _DTYPE[sp.indices.componentType], sp.count,
                                (iv.byteOffset or 0) + (sp.indices.byteOffset or 0)).astype(np.int64)
            vv = self.gltf.bufferViews[sp.values.bufferView]
            out[idx] = np.frombuffer(self.blob, dtype, sp.count * width,
                                     (vv.byteOffset or 0) + (sp.values.byteOffset or 0)).reshape(-1, width)
        if acc.normalized and acc.componentType in _NORM:
            out = np.maximum(out / _NORM[acc.componentType], -1.0)
        return out

    def view_bytes(self, view_index: int) -> bytes:
        view = self.gltf.bufferViews[view_index]
        start = view.byteOffset or 0
        return self.blob[start:start + view.byteLength]

    def _node_trs(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.gltf.nodes[i]
        """Its local TRS as one frame: (1, 3), (1, 4), (1, 3)."""
        if n.matrix and any(abs(a - b) > 1e-12 for a, b in zip(n.matrix, np.eye(4).ravel())):
            return _decompose(np.array(n.matrix, dtype=np.float64).reshape(1, 4, 4).transpose(0, 2, 1))
        return (np.array([n.translation or [0, 0, 0]], dtype=np.float64),
                np.array([n.rotation or [0, 0, 0, 1]], dtype=np.float64),
                np.array([n.scale or [1, 1, 1]], dtype=np.float64))

    def world(self, trs: list) -> list[np.ndarray]:
        """Every node's world matrices (F, 4, 4), column vectors, from per-node
        local TRS over F frames."""
        out: list[np.ndarray] = [np.eye(4)[None]] * len(trs)
        for i in self.order:
            m = _compose(*trs[i])
            out[i] = out[self.parent[i]] @ m if i in self.parent else m
        return out


def _compose(t: np.ndarray, q: np.ndarray, s: np.ndarray) -> np.ndarray:
    """(F, 4, 4) from (F, 3) translations, (F, 4) xyzw quaternions, (F, 3) scales."""
    m = np.zeros((len(t), 4, 4))
    m[:, :3, :3] = Rotation.from_quat(q).as_matrix() * s[:, None, :]
    m[:, :3, 3] = t
    m[:, 3, 3] = 1.0
    return m


def _decompose(m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TRS of (..., 4, 4) matrices without shear (a negative determinant goes on x)."""
    s = np.linalg.norm(m[..., :3, :3], axis=-2)
    r = m[..., :3, :3] / np.where(s > 1e-12, s, 1.0)[..., None, :]
    flip = np.linalg.det(r) < 0
    s[flip, 0] *= -1
    r[flip, :, 0] *= -1
    q = Rotation.from_matrix(r.reshape(-1, 3, 3)).as_quat().reshape(*m.shape[:-2], 4)
    return m[..., :3, 3].copy(), q, s


def _gf(m: np.ndarray) -> Gf.Matrix4d:
    """A column-vector numpy matrix as USD's row-vector Gf.Matrix4d."""
    return Gf.Matrix4d(m.T.tolist())


def _name(text: str | None, fallback: str, taken: set[str] | None = None) -> str:
    """A valid, unique USD prim name."""
    name = re.sub(r"[^A-Za-z0-9_]", "_", text or "") or fallback
    if name[0].isdigit():
        name = "_" + name
    if taken is not None:
        base, k = name, 1
        while name in taken:
            name, k = f"{base}_{k}", k + 1
        taken.add(name)
    return name


# --------------------------------------------------------------------------- animation


def clip_names(glb_path: str | Path) -> list[str]:
    gltf = GLTF2().load(str(glb_path))
    return [a.name or f"clip_{i}" for i, a in enumerate(gltf.animations or [])]


def _sample_times(glb: _Glb, anim: Any) -> np.ndarray:
    """Every key time of the clip, plus what STEP and CUBICSPLINE need to come
    through USD's linear interpolation."""
    times = [np.array([0.0])]
    for channel in anim.channels:
        sampler = anim.samplers[channel.sampler]
        keys = glb.read(sampler.input).ravel()
        times.append(keys)
        interp = sampler.interpolation or "LINEAR"
        if interp == "STEP" and len(keys) > 1:
            values = glb.read(sampler.output)
            jumps = np.flatnonzero(np.abs(np.diff(values, axis=0)).max(axis=1) > 0) + 1
            times.append(np.maximum(keys[jumps] - _STEP_EPS, keys[jumps - 1]))
        elif interp == "CUBICSPLINE" and len(keys) > 1:
            times.append(np.arange(keys[0], keys[-1], 1.0 / _CUBIC_FPS))
    return np.unique(np.round(np.concatenate(times), 6))


def _hermite(keys: np.ndarray, values: np.ndarray, t: np.ndarray) -> np.ndarray:
    """glTF CUBICSPLINE: values hold (in-tangent, value, out-tangent) per key."""
    v = values.reshape(len(keys), 3, -1)
    i = np.clip(np.searchsorted(keys, t, side="right") - 1, 0, len(keys) - 2)
    dt = keys[i + 1] - keys[i]
    u = np.clip((t - keys[i]) / np.where(dt > 0, dt, 1.0), 0.0, 1.0)[:, None]
    u2, u3 = u * u, u * u * u
    return ((2 * u3 - 3 * u2 + 1) * v[i, 1] + (u3 - 2 * u2 + u) * dt[:, None] * v[i, 2]
            + (-2 * u3 + 3 * u2) * v[i + 1, 1] + (u3 - u2) * dt[:, None] * v[i + 1, 0])


def _sample_channel(glb: _Glb, sampler: Any, path: str, times: np.ndarray) -> np.ndarray:
    keys = glb.read(sampler.input).ravel()
    values = glb.read(sampler.output)
    interp = sampler.interpolation or "LINEAR"
    t = np.clip(times, keys[0], keys[-1])
    if interp == "CUBICSPLINE":
        if len(keys) == 1:
            return np.repeat(values.reshape(1, 3, -1)[:, 1], len(times), axis=0)
        out = _hermite(keys, values, t)
        if path == "rotation":
            out /= np.linalg.norm(out, axis=1, keepdims=True)
        return out
    if interp == "STEP" or len(keys) == 1:
        return values[np.clip(np.searchsorted(keys, t + 1e-9, side="right") - 1, 0, len(keys) - 1)]
    if path == "rotation":
        return Slerp(keys, Rotation.from_quat(values))(t).as_quat()
    return np.stack([np.interp(t, keys, values[:, k]) for k in range(values.shape[1])], axis=1)


def _clip_frames(glb: _Glb, anim: Any, times: np.ndarray) -> list[list[np.ndarray]]:
    """Every node's local [T, Q, S] over the sample times, the clip applied."""
    frames = [[np.repeat(x, len(times), axis=0) for x in trs] for trs in glb.rest]
    for channel in anim.channels:
        node, path = channel.target.node, channel.target.path
        if node is None or path not in ("translation", "rotation", "scale"):
            continue
        slot = ("translation", "rotation", "scale").index(path)
        frames[node][slot] = _sample_channel(glb, anim.samplers[channel.sampler], path, times)
    return frames


# --------------------------------------------------------------------------- materials


def _srgb_encode(linear: np.ndarray) -> np.ndarray:
    c = np.clip(linear, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def _bake_vertex_colours(colours: np.ndarray, faces: np.ndarray) -> tuple[Image.Image, np.ndarray]:
    """Linear vertex colours as an sRGB atlas, one cell per triangle, and the
    face-varying st (USD, origin bottom left) of each triangle's corners."""
    m = len(faces)
    grid = math.ceil(math.sqrt(m))
    size = max(64, 1 << math.ceil(math.log2(grid * _CELL)))
    leg = _CELL - 2 * _PAD
    # barycentric weights of every pixel centre of a cell, clamped onto the triangle
    px = (np.arange(_CELL) + 0.5 - _PAD) / leg
    s, t = np.meshgrid(px, px)                                 # s along x, t along y (image rows)
    s, t = np.clip(s, 0, None), np.clip(t, 0, None)
    over = np.clip((s + t - 1) / 2, 0, None)
    s, t = np.clip(s - over, 0, 1), np.clip(t - over, 0, 1)
    w = np.stack([1 - s - t, s, t], axis=-1)                   # (cell, cell, 3)
    cells = np.einsum("yxk,fkc->fyxc", w, colours[faces][..., :3])
    image = np.zeros((size, size, 3))
    col, row = np.arange(m) % grid, np.arange(m) // grid
    for f in range(m):
        y0, x0 = row[f] * _CELL, col[f] * _CELL
        image[y0:y0 + _CELL, x0:x0 + _CELL] = cells[f]
    img = Image.fromarray(np.round(_srgb_encode(image) * 255).astype(np.uint8), "RGB")
    # corners in pixels: vertex 0 at the right angle, 1 along x, 2 along y
    x0, y0 = col * _CELL + _PAD, row * _CELL + _PAD
    corners = np.stack([np.stack([x0, y0], 1), np.stack([x0 + leg, y0], 1), np.stack([x0, y0 + leg], 1)], 1)
    st = np.stack([corners[..., 0] / size, 1.0 - corners[..., 1] / size], -1).reshape(-1, 2)
    return img, st


class _Materials:
    """glTF materials as UsdPreviewSurface under /Root/Materials, textures
    written next to the layer in textures/."""

    def __init__(self, glb: _Glb, stage: Usd.Stage, root: Sdf.Path, work: Path):
        self.glb, self.stage, self.work = glb, stage, work
        self.scope = UsdGeom.Scope.Define(stage, root.AppendChild("Materials")).GetPath()
        self.taken: set[str] = set()
        self.images: dict[int, str] = {}
        self.made: dict[tuple, UsdShade.Material] = {}
        (work / "textures").mkdir(exist_ok=True)

    def _image_file(self, index: int) -> str:
        if index in self.images:
            return self.images[index]
        image = self.glb.gltf.images[index]
        if image.bufferView is not None:
            data = self.glb.view_bytes(image.bufferView)
        elif image.uri and image.uri.startswith("data:"):
            import base64
            data = base64.b64decode(image.uri.split(",", 1)[1])
        elif image.uri:
            data = (self.glb.path.parent / image.uri).read_bytes()
        else:
            raise UsdzError(f"image {index} has no data")
        name = _name(Path(image.name or f"image_{index}").stem, f"image_{index}", self.taken)
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            rel = f"textures/{name}.png"
            (self.work / rel).write_bytes(data)
        elif data[:3] == b"\xff\xd8\xff":
            rel = f"textures/{name}.jpg"
            (self.work / rel).write_bytes(data)
        else:                                       # webp, ktx...: Quick Look reads PNG/JPEG only
            rel = f"textures/{name}.png"
            Image.open(io.BytesIO(data)).save(self.work / rel)
        self.images[index] = rel
        return rel

    def _texture(self, mat: Sdf.Path, label: str, file: str, wrap: tuple[str, str], st: UsdShade.Output,
                 raw: bool, scale: tuple | None = None, bias: tuple | None = None) -> UsdShade.Shader:
        tex = UsdShade.Shader.Define(self.stage, mat.AppendChild(label))
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(file))
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st)
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set(wrap[0])
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set(wrap[1])
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw" if raw else "sRGB")
        if scale is not None:
            tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
        if bias is not None:
            tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*bias))
        return tex

    def _wrap(self, texture_index: int) -> tuple[str, str]:
        tex = self.glb.gltf.textures[texture_index]
        if tex.sampler is None:
            return "repeat", "repeat"
        sampler = self.glb.gltf.samplers[tex.sampler]
        return _WRAP.get(sampler.wrapS or 10497, "repeat"), _WRAP.get(sampler.wrapT or 10497, "repeat")

    def material(self, index: int | None, baked: Image.Image | None = None, tag: str = "") -> UsdShade.Material:
        """The USD material of glTF material `index` (None: glTF's default
        material), with `baked` as its base colour texture when given."""
        key = (index, tag)
        if key in self.made and baked is None:
            return self.made[key]
        src = self.glb.gltf.materials[index] if index is not None else None
        name = _name((src.name if src else "default") + (f"_{tag}" if tag else ""), "material", self.taken)
        path = self.scope.AppendChild(name)
        mat = UsdShade.Material.Define(self.stage, path)
        surf = UsdShade.Shader.Define(self.stage, path.AppendChild("PreviewSurface"))
        surf.CreateIdAttr("UsdPreviewSurface")
        surf.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
        mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
        reader = UsdShade.Shader.Define(self.stage, path.AppendChild("stReader"))
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
        st = reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        used_st = False

        pbr = src.pbrMetallicRoughness if src else None
        factor = list(pbr.baseColorFactor) if pbr and pbr.baseColorFactor else [1.0, 1.0, 1.0, 1.0]
        alpha_mode = (src.alphaMode if src else None) or "OPAQUE"
        diffuse = surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        opacity_tex = None
        scale = None if factor == [1.0, 1.0, 1.0, 1.0] else tuple(factor)
        if baked is not None:
            rel = f"textures/{name}.png"
            baked.save(self.work / rel)
            tex = self._texture(path, "baseColorTex", rel, ("clamp", "clamp"), st, raw=False, scale=scale)
            diffuse.ConnectToSource(tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
            used_st = True
        elif pbr and pbr.baseColorTexture is not None:
            info = pbr.baseColorTexture
            file = self._image_file(self.glb.gltf.textures[info.index].source)
            tex = self._texture(path, "baseColorTex", file, self._wrap(info.index), st, raw=False, scale=scale)
            diffuse.ConnectToSource(tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
            opacity_tex, used_st = tex, True
        else:
            diffuse.Set(Gf.Vec3f(*factor[:3]))
        if alpha_mode != "OPAQUE":
            opacity = surf.CreateInput("opacity", Sdf.ValueTypeNames.Float)
            if opacity_tex is not None:
                opacity.ConnectToSource(opacity_tex.CreateOutput("a", Sdf.ValueTypeNames.Float))
            else:
                opacity.Set(float(factor[3]))
            if alpha_mode == "MASK":
                cutoff = src.alphaCutoff if src.alphaCutoff is not None else 0.5
                surf.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(float(cutoff))

        metallic = float(pbr.metallicFactor) if pbr and pbr.metallicFactor is not None else 1.0
        roughness = float(pbr.roughnessFactor) if pbr and pbr.roughnessFactor is not None else 1.0
        if src is None:
            metallic, roughness = 0.0, 1.0           # glTF's default material is a matte grey...
            diffuse.Set(Gf.Vec3f(1.0, 1.0, 1.0))     # ...in fact white
        if pbr and pbr.metallicRoughnessTexture is not None:
            info = pbr.metallicRoughnessTexture
            file = self._image_file(self.glb.gltf.textures[info.index].source)
            tex = self._texture(path, "metallicRoughnessTex", file, self._wrap(info.index), st, raw=True,
                                scale=(1.0, roughness, metallic, 1.0))
            surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(
                tex.CreateOutput("b", Sdf.ValueTypeNames.Float))
            surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
                tex.CreateOutput("g", Sdf.ValueTypeNames.Float))
            used_st = True
        else:
            surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
            surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)

        if src and src.normalTexture is not None:
            info = src.normalTexture
            k = float(info.scale if info.scale is not None else 1.0)
            file = self._image_file(self.glb.gltf.textures[info.index].source)
            tex = self._texture(path, "normalTex", file, self._wrap(info.index), st, raw=True,
                                scale=(2 * k, 2 * k, 2.0, 1.0), bias=(-k, -k, -1.0, 0.0))
            surf.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
                tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
            used_st = True
        if src and src.occlusionTexture is not None:
            info = src.occlusionTexture
            file = self._image_file(self.glb.gltf.textures[info.index].source)
            tex = self._texture(path, "occlusionTex", file, self._wrap(info.index), st, raw=True)
            surf.CreateInput("occlusion", Sdf.ValueTypeNames.Float).ConnectToSource(
                tex.CreateOutput("r", Sdf.ValueTypeNames.Float))
            used_st = True
        emissive = list(src.emissiveFactor) if src and src.emissiveFactor else [0.0, 0.0, 0.0]
        if src and src.emissiveTexture is not None:
            info = src.emissiveTexture
            file = self._image_file(self.glb.gltf.textures[info.index].source)
            tex = self._texture(path, "emissiveTex", file, self._wrap(info.index), st, raw=False,
                                scale=(*emissive, 1.0))
            surf.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
            used_st = True
        elif any(emissive):
            surf.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
        if not used_st:
            self.stage.RemovePrim(reader.GetPath())
        if src and src.extensions:
            warnings.warn(f"material {src.name!r}: extensions {sorted(src.extensions)} not carried over")
        if baked is None:
            self.made[key] = mat
        return mat


# --------------------------------------------------------------------------- conversion


def _skinned_points(points: np.ndarray, joints: np.ndarray, weights: np.ndarray,
                    skin_mats: np.ndarray) -> np.ndarray:
    """glTF linear blend skinning: sum_i w_i (joint world_i @ inverse bind_i) p."""
    m = np.einsum("vk,vkab->vab", weights, skin_mats[joints])
    return np.einsum("vab,vb->va", m[:, :3, :3], points) + m[:, :3, 3]


def glb_to_usdz(glb_path: str | Path, usdz_path: str | Path, clip: str | None = None) -> dict[str, Any]:
    """Write `glb_path` with clip `clip` (default the rest loop, REST_CLIPS,
    else the first; "" for none) as the USDZ `usdz_path`. Returns what went in."""
    glb_path, usdz_path = Path(glb_path), Path(usdz_path)
    glb = _Glb(glb_path)
    gltf = glb.gltf
    if gltf.extensionsRequired:
        raise UsdzError(f"{glb_path}: requires {gltf.extensionsRequired}")
    anims = {a.name or f"clip_{i}": a for i, a in enumerate(gltf.animations or [])}
    if clip is None:
        clip = next((c for c in REST_CLIPS if c in anims), next(iter(anims), ""))
    if clip and clip not in anims:
        raise UsdzError(f"no clip {clip!r}; {glb_path.name} has {list(anims)}")
    anim = anims.get(clip) if clip else None

    scene = gltf.scenes[gltf.scene or 0]
    in_scene: set[int] = set()
    stack = list(scene.nodes or [])
    while stack:
        i = stack.pop()
        in_scene.add(i)
        stack.extend(gltf.nodes[i].children or [])
    mesh_nodes = [i for i in glb.order if i in in_scene and gltf.nodes[i].mesh is not None]
    skins = {gltf.nodes[i].skin for i in mesh_nodes if gltf.nodes[i].skin is not None}
    if len(skins) > 1:
        raise UsdzError(f"{glb_path}: {len(skins)} skins; one skeleton per file is supported")
    skin = gltf.skins[skins.pop()] if skins else None

    rest_world = [m[0] for m in glb.world(glb.rest)]
    times = _sample_times(glb, anim) if anim is not None else np.array([0.0])
    worlds = glb.world(_clip_frames(glb, anim, times) if anim is not None else glb.rest)   # node -> (F, 4, 4)
    if anim is not None:
        for channel in anim.channels:
            if channel.target.path == "weights":
                warnings.warn(f"clip {clip!r} animates morph weights; they stay at the mesh's defaults")
                break

    work = Path(tempfile.mkdtemp(prefix="usdz_"))
    try:
        layer = work / "model.usdc"
        stage = Usd.Stage.CreateNew(str(layer))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        root = UsdGeom.Xform.Define(stage, "/Root")
        stage.SetDefaultPrim(root.GetPrim())
        Usd.ModelAPI(root.GetPrim()).SetKind(Kind.Tokens.component)
        duration = float(times[-1])
        fps = 60.0
        if anim is not None and len(times) > 1:
            steps = np.diff(times)
            fps = float(min(120, max(24, round(1.0 / np.median(steps[steps > 2 * _STEP_EPS]))))) \
                if np.any(steps > 2 * _STEP_EPS) else 60.0
        stage.SetTimeCodesPerSecond(fps)
        stage.SetFramesPerSecond(fps)
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(round(duration * fps, 6))
        stage.GetRootLayer().customLayerData = {
            "source": glb_path.name, "clip": clip or "", "duration_s": duration,
            "generator": "LovablePet api/usdz.py"}
        materials = _Materials(glb, stage, root.GetPath(), work)
        taken: set[str] = set()
        info: dict[str, Any] = {"clip": clip, "duration_s": round(duration, 4), "fps": fps,
                                "samples": int(len(times)), "meshes": []}

        skel_root_path = skel_path = None
        order: list[int] = []
        if skin is not None:
            joints = list(skin.joints)
            jset = set(joints)
            order = sorted(joints, key=glb.depth)                           # parents first
            tops = [j for j in order if glb.parent.get(j) not in jset]
            # the space the skeleton lives in: the top joints' common parent (the scene root)
            holders = {glb.parent.get(j) for j in tops}
            holder = holders.pop() if len(holders) == 1 else None
            holder_rest = rest_world[holder] if holder is not None else np.eye(4)
            skel_root = UsdSkel.Root.Define(stage, root.GetPath().AppendChild(
                _name(gltf.nodes[holder].name if holder is not None else "Skeletal", "Skeletal", taken)))
            if holder is not None and not np.allclose(holder_rest, np.eye(4)):
                skel_root.MakeMatrixXform().Set(_gf(holder_rest))
            skel_root_path = skel_root.GetPath()
            skel = UsdSkel.Skeleton.Define(stage, skel_root_path.AppendChild("Skeleton"))
            skel_path = skel.GetPath()

            jnames: dict[int, str] = {}
            paths: dict[int, str] = {}
            siblings: dict[int | None, set[str]] = {}
            for j in order:
                up = glb.parent.get(j) if glb.parent.get(j) in jset else None
                jnames[j] = _name(gltf.nodes[j].name, f"joint_{j}", siblings.setdefault(up, set()))
                paths[j] = jnames[j] if up is None else f"{paths[up]}/{jnames[j]}"
            skel.CreateJointsAttr([paths[j] for j in order])
            skel.CreateJointNamesAttr([jnames[j] for j in order])
            ibm = (glb.read(skin.inverseBindMatrices).reshape(-1, 4, 4).transpose(0, 2, 1)
                   if skin.inverseBindMatrices is not None else np.tile(np.eye(4), (len(joints), 1, 1)))
            ibm_of = {j: ibm[k] for k, j in enumerate(joints)}
            skel.CreateBindTransformsAttr(Vt.Matrix4dArray([_gf(np.linalg.inv(ibm_of[j])) for j in order]))
            inv_holder = np.linalg.inv(holder_rest)

            def local(world: list[np.ndarray], j: int) -> np.ndarray:
                """Joint j relative to its parent joint, a top joint to the
                holder at rest (so an animated scene root still moves it)."""
                up = glb.parent.get(j)
                return (np.linalg.inv(world[up]) if up in jset else inv_holder) @ world[j]

            skel.CreateRestTransformsAttr(Vt.Matrix4dArray([_gf(local(rest_world, j)) for j in order]))
            tr, rot, sc = _decompose(np.stack([local(worlds, j) for j in order], axis=1))   # (F, J, ...)

            anim_prim = UsdSkel.Animation.Define(stage, skel_path.AppendChild(_name(clip or "rest", "clip")))
            anim_prim.CreateJointsAttr([paths[j] for j in order])
            t_attr, r_attr, s_attr = (anim_prim.CreateTranslationsAttr(), anim_prim.CreateRotationsAttr(),
                                      anim_prim.CreateScalesAttr())
            for f, time in enumerate(times):
                code = Usd.TimeCode(round(float(time) * fps, 6)) if anim is not None else Usd.TimeCode.Default()
                t_attr.Set(Vt.Vec3fArray.FromNumpy(tr[f].astype(np.float32)), code)
                r_attr.Set(Vt.QuatfArray.FromNumpy(rot[f].astype(np.float32)), code)     # xyzw: GfQuatf's layout
                s_attr.Set(Vt.Vec3hArray.FromNumpy(sc[f].astype(np.float16)), code)
            binding = UsdSkel.BindingAPI.Apply(skel.GetPrim())
            binding.CreateAnimationSourceRel().SetTargets([anim_prim.GetPath()])
            info["joints"] = len(order)

        # skinning matrices per sample, for the extent the animation sweeps
        sweep_lo, sweep_hi = np.full(3, np.inf), np.full(3, -np.inf)
        pick = np.unique(np.linspace(0, len(times) - 1, min(len(times), 48)).round().astype(int))
        for node_index in mesh_nodes:
            node = gltf.nodes[node_index]
            mesh = gltf.meshes[node.mesh]
            skinned = node.skin is not None
            parent_path = skel_root_path if skinned else root.GetPath()
            holder_path = UsdGeom.Xform.Define(stage, parent_path.AppendChild(
                _name(node.name or mesh.name, f"mesh_{node_index}", taken))).GetPath()
            if not skinned:
                UsdGeom.Xform(stage.GetPrimAtPath(holder_path)).MakeMatrixXform().Set(_gf(rest_world[node_index]))
            weights_default = list(getattr(node, "weights", None) or mesh.weights or [])
            local_names: set[str] = set()
            for p_index, prim in enumerate(mesh.primitives):
                if (prim.mode if prim.mode is not None else 4) != 4:
                    warnings.warn(f"mesh {mesh.name!r} primitive {p_index}: mode {prim.mode} is not triangles; skipped")
                    continue
                a = prim.attributes
                pts = glb.read(a.POSITION)
                nrm = glb.read(a.NORMAL) if a.NORMAL is not None else None
                for w, target in zip(weights_default, prim.targets or []):
                    if w:
                        pts = pts + w * glb.read(target["POSITION"]) if "POSITION" in target else pts
                        if nrm is not None and "NORMAL" in target:
                            nrm = nrm + w * glb.read(target["NORMAL"])
                if nrm is not None:
                    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
                faces = (glb.read(prim.indices).astype(np.int64).reshape(-1, 3) if prim.indices is not None
                         else np.arange(len(pts)).reshape(-1, 3))
                # a triangle on two of the same vertex draws nothing, but a
                # reader that drops it (Blender does) shifts every face-varying
                # value after it - the baked mouth came out black
                degenerate = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 0] == faces[:, 2])
                faces = faces[~degenerate]
                mat_src = gltf.materials[prim.material] if prim.material is not None else None
                label = _name(mat_src.name if mat_src else f"part_{p_index}", f"part_{p_index}", local_names)
                umesh = UsdGeom.Mesh.Define(stage, holder_path.AppendChild(label))
                umesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                umesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
                umesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
                umesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel().astype(np.int32)))
                umesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*pts.min(0)), Gf.Vec3f(*pts.max(0))]))
                umesh.CreateDoubleSidedAttr(bool(mat_src.doubleSided) if mat_src else False)
                if nrm is not None:
                    umesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(nrm.astype(np.float32)))
                    umesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
                pv = UsdGeom.PrimvarsAPI(umesh)
                textured = mat_src is not None and mat_src.pbrMetallicRoughness is not None and \
                    mat_src.pbrMetallicRoughness.baseColorTexture is not None
                baked = None
                if a.COLOR_0 is not None and not textured:
                    baked, st = _bake_vertex_colours(glb.read(a.COLOR_0), faces)
                    pv.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying).Set(
                        Vt.Vec2fArray.FromNumpy(st.astype(np.float32)))
                elif a.TEXCOORD_0 is not None:
                    if a.COLOR_0 is not None:
                        warnings.warn(f"{label}: vertex colours on a textured material are dropped")
                    uv = glb.read(a.TEXCOORD_0)
                    uv[:, 1] = 1.0 - uv[:, 1]                              # glTF's v runs down
                    pv.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
                        Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
                mat = materials.material(prim.material, baked=baked, tag=label if baked is not None else "")
                UsdShade.MaterialBindingAPI.Apply(umesh.GetPrim()).Bind(mat)

                entry = {"prim": str(umesh.GetPath()), "points": int(len(pts)), "triangles": int(len(faces)),
                         "degenerate_dropped": int(degenerate.sum()), "vertex_colours_baked": baked is not None}
                if skinned:
                    j_sets = [glb.read(getattr(a, f"JOINTS_{k}")).astype(np.int64)
                              for k in range(4) if getattr(a, f"JOINTS_{k}", None) is not None]
                    w_sets = [glb.read(getattr(a, f"WEIGHTS_{k}"))
                              for k in range(4) if getattr(a, f"WEIGHTS_{k}", None) is not None]
                    jidx, jw = np.concatenate(j_sets, 1), np.concatenate(w_sets, 1)
                    jw = np.where(np.isfinite(jw) & (jw > 0), jw, 0.0)
                    total = jw.sum(1, keepdims=True)
                    jw = np.where(total > 0, jw / np.where(total > 0, total, 1.0), 0.0)
                    jidx = np.where(jw > 0, jidx, 0)
                    # skin joint index -> the skeleton's (parents first) order
                    remap = np.array([order.index(j) for j in skin.joints])
                    usd_idx = remap[jidx]
                    width = jidx.shape[1]
                    sb = UsdSkel.BindingAPI.Apply(umesh.GetPrim())
                    sb.CreateSkeletonRel().SetTargets([skel_path])
                    sb.CreateJointIndicesPrimvar(False, width).Set(Vt.IntArray.FromNumpy(usd_idx.ravel().astype(np.int32)))
                    sb.CreateJointWeightsPrimvar(False, width).Set(Vt.FloatArray.FromNumpy(jw.ravel().astype(np.float32)))
                    sb.CreateGeomBindTransformAttr(Gf.Matrix4d(1.0))
                    for f in pick:
                        mats = np.stack([worlds[j][f] for j in skin.joints]) @ ibm
                        sk = _skinned_points(pts, jidx, jw, mats)
                        sweep_lo, sweep_hi = np.minimum(sweep_lo, sk.min(0)), np.maximum(sweep_hi, sk.max(0))
                else:
                    wp = pts @ rest_world[node_index][:3, :3].T + rest_world[node_index][:3, 3]
                    sweep_lo, sweep_hi = np.minimum(sweep_lo, wp.min(0)), np.maximum(sweep_hi, wp.max(0))
                info["meshes"].append(entry)

        if skel_root_path is not None and np.all(np.isfinite(sweep_lo)):
            # the SkelRoot's extent in its own space, covering the whole clip
            inv = np.linalg.inv(rest_world[holder]) if holder is not None else np.eye(4)
            corners = np.array([[x, y, z] for x in (sweep_lo[0], sweep_hi[0]) for y in (sweep_lo[1], sweep_hi[1])
                                for z in (sweep_lo[2], sweep_hi[2])])
            local_c = corners @ inv[:3, :3].T + inv[:3, 3]
            UsdSkel.Root(stage.GetPrimAtPath(skel_root_path)).CreateExtentAttr(
                Vt.Vec3fArray([Gf.Vec3f(*local_c.min(0)), Gf.Vec3f(*local_c.max(0))]))
        info["bounds_m"] = {"min": [round(float(v), 4) for v in sweep_lo], "max": [round(float(v), 4) for v in sweep_hi]}
        stage.GetRootLayer().Save()

        usdz_path.parent.mkdir(parents=True, exist_ok=True)
        # packed by hand: UsdUtils' packager moves the textures into 0/ and
        # warns about a path it made itself; this layer has nothing to resolve
        partial = usdz_path.with_suffix(".usdz.part")
        writer = Sdf.ZipFileWriter.CreateNew(str(partial))
        writer.AddFile(str(layer), "model.usdc")                       # the root layer comes first
        for tex in sorted((work / "textures").iterdir()):
            writer.AddFile(str(tex), f"textures/{tex.name}")
        writer.Save()
        partial.replace(usdz_path)
        info["usdz"] = str(usdz_path)
        info["bytes"] = usdz_path.stat().st_size
        return info
    finally:
        shutil.rmtree(work, ignore_errors=True)


def glb_to_usdz_all(glb_path: str | Path, out_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """One USDZ per clip, `<stem>_<clip>.usdz`."""
    glb_path = Path(glb_path)
    out_dir = Path(out_dir) if out_dir else glb_path.parent
    return [glb_to_usdz(glb_path, out_dir / f"{glb_path.stem}_{_name(c, 'clip')}.usdz", c)
            for c in clip_names(glb_path)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("glb", type=Path)
    ap.add_argument("-o", "--out", type=Path, help="output .usdz (default: next to the GLB); a folder with --all")
    ap.add_argument("--clip", help=f"clip to play (default the first of {REST_CLIPS}, else the first; "
                                   "'' for none)")
    ap.add_argument("--all", action="store_true", help="one USDZ per clip")
    args = ap.parse_args()
    if args.all:
        results = glb_to_usdz_all(args.glb, args.out)
    else:
        results = [glb_to_usdz(args.glb, args.out or args.glb.with_suffix(".usdz"), args.clip)]
    for r in results:
        print(f"{r['usdz']}: clip {r['clip'] or '-'} {r['duration_s']} s, {r.get('joints', 0)} joints, "
              f"{sum(m['points'] for m in r['meshes'])} points, {r['bytes'] / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
