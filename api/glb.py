"""Read and replace the base color texture of a template GLB.

Nothing here touches the skeleton, the skin weights, the animations or the
node hierarchy. Stage 1 authored those and the rig is what makes the pet move;
stage 3 only repaints it. Concretely: `nodes`, `skins`, `animations`, `meshes`
and every accessor are copied through untouched, and the one field written is
the bufferView backing the material's baseColorTexture image.

Which image that is has to be looked up rather than assumed. In
skeleton_shiba_small.glb the base color is image 3 while image 0 is the
specular map, so a writer that reached for images[0] would silently repaint the
wrong channel.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image
from pygltflib import GLTF2, BufferView, Image as GLTFImage

# A repainted coat is a matte dielectric. The template ships values tuned
# for its own look, and a stray baseColorFactor or metallic term tints or
# mirrors the new texture instead of showing it.
MATTE_ROUGHNESS = 0.85


class TextureSlotMissing(RuntimeError):
    """The GLB has no material with a base color texture to replace."""


def body_material_index(gltf: GLTF2, body_material: str | None = None) -> int:
    """The material that wears the coat, and the only one a repaint touches.

    A template with one material has no choice to make. One with several -
    body, eyes, whiskers, nose - names the body material, or if it does not,
    the body is the textured material covering the most vertices: eyes and
    whiskers are small meshes with their own maps and alpha, and those maps
    are never rewritten.
    """
    candidates = [
        index for index, material in enumerate(gltf.materials or [])
        if material.pbrMetallicRoughness is not None
        and material.pbrMetallicRoughness.baseColorTexture is not None
        and gltf.textures[material.pbrMetallicRoughness.baseColorTexture.index].source is not None
    ]
    if not candidates:
        raise TextureSlotMissing(
            "no material in this GLB declares pbrMetallicRoughness.baseColorTexture"
        )
    if body_material is not None:
        for index in candidates:
            if (gltf.materials[index].name or "") == body_material:
                return index
        raise TextureSlotMissing(f"no textured material named {body_material!r}")
    if len(candidates) == 1:
        return candidates[0]
    named = [index for index in candidates if "body" in (gltf.materials[index].name or "").lower()]
    if len(named) == 1:
        return named[0]
    # the material covering the most vertices
    weight = {index: 0 for index in candidates}
    for mesh in gltf.meshes or []:
        for primitive in mesh.primitives:
            if primitive.material in weight and primitive.attributes.POSITION is not None:
                weight[primitive.material] += gltf.accessors[primitive.attributes.POSITION].count
    return max(candidates, key=lambda index: weight[index])


def _base_color_image_index(gltf: GLTF2, body_material: str | None = None) -> int:
    material = gltf.materials[body_material_index(gltf, body_material)]
    return gltf.textures[material.pbrMetallicRoughness.baseColorTexture.index].source


def _view_bytes(gltf: GLTF2, blob: bytes, view_index: int) -> bytes:
    view = gltf.bufferViews[view_index]
    start = view.byteOffset or 0
    return blob[start:start + view.byteLength]


def animation_names(glb_path: str | Path) -> list[str]:
    """The clips the template actually carries, for the tap-reaction fallback
    to check against instead of assuming."""
    gltf = GLTF2().load(str(glb_path))
    return [clip.name for clip in (gltf.animations or []) if clip.name]


def extract_base_texture(glb_path: str | Path) -> Image.Image:
    """Return the GLB's base color texture as a PIL image."""
    gltf = GLTF2().load(str(glb_path))
    blob = gltf.binary_blob()
    image = gltf.images[_base_color_image_index(gltf)]
    if image.bufferView is None:
        raise TextureSlotMissing(
            f"base color image '{image.name}' is a URI reference, not embedded"
        )
    return Image.open(io.BytesIO(_view_bytes(gltf, blob, image.bufferView))).convert("RGB")


def _make_matte(gltf: GLTF2, only: int | None = None) -> None:
    """Force the material to show the texture as painted.

    baseColorFactor multiplies the texture, so anything but white tints the
    whole coat; a non-zero metallicFactor turns fur into a mirror that takes
    its colour from the environment rather than from the albedo. Both are set
    on the template for its own original look and neither survives a repaint.
    """
    for index, material in enumerate(gltf.materials or []):
        pbr = material.pbrMetallicRoughness
        if pbr is None or pbr.baseColorTexture is None or (only is not None and index != only):
            continue
        pbr.baseColorFactor = [1.0, 1.0, 1.0, 1.0]
        pbr.metallicFactor = 0.0
        pbr.roughnessFactor = MATTE_ROUGHNESS
        # the factors only scale a metallic-roughness texture where one is
        # attached, and the template's made the tail a gloss-black sausage
        # with a highlight running its length; without the texture the
        # factors are the whole story and the coat is matte everywhere
        pbr.metallicRoughnessTexture = None


def _drop_vertex_colors(gltf: GLTF2, only: int | None = None) -> None:
    """Remove COLOR_0 if the mesh carries it.

    A vertex colour multiplies the base colour per vertex, so a template that
    tinted its mesh that way would keep tinting the coat we just painted, and
    the cause would be invisible in the texture. Only the attribute reference
    is dropped: the accessor and its bytes are left alone, so nothing else
    shifts.
    """
    for mesh in gltf.meshes or []:
        for primitive in mesh.primitives:
            if only is not None and primitive.material != only:
                continue
            if getattr(primitive.attributes, "COLOR_0", None) is not None:
                primitive.attributes.COLOR_0 = None


def _normal_image_index(gltf: GLTF2, body_material: str | None = None) -> int | None:
    material = gltf.materials[body_material_index(gltf, body_material)]
    if material.normalTexture is None:
        return None
    return gltf.textures[material.normalTexture.index].source


def extract_normal_texture(glb_path: str | Path) -> Image.Image | None:
    """The body material's normal map, or None when it has none."""
    gltf = GLTF2().load(str(glb_path))
    index = _normal_image_index(gltf)
    if index is None or gltf.images[index].bufferView is None:
        return None
    blob = gltf.binary_blob()
    return Image.open(io.BytesIO(_view_bytes(gltf, blob, gltf.images[index].bufferView))).convert("RGB")


def replace_base_texture(
    glb_path: str | Path, texture: Image.Image, out_path: str | Path, fmt: str = "PNG",
    body_material: str | None = None, normal: Image.Image | None = None,
) -> Path:
    """Write a copy of the GLB whose body base color texture is `texture`.

    Only the body material is touched (see body_material_index): the eyes,
    whiskers and nose keep their own textures, alpha and material settings.

    The binary chunk is repacked rather than appended to, so the new file does
    not carry the replaced image as dead weight. Repacking is safe because an
    accessor addresses its bufferView by index and its own byteOffset is
    relative to that view, so moving views inside the buffer leaves every
    accessor - and therefore every animation and skin - valid.
    """
    gltf = GLTF2().load(str(glb_path))
    blob = gltf.binary_blob()
    body = body_material_index(gltf, body_material)
    target = _base_color_image_index(gltf, body_material)

    encoded = io.BytesIO()
    texture.save(encoded, format=fmt)
    replacements = {gltf.images[target].bufferView: encoded.getvalue()}

    # the fur normal map goes into the same slot the template's own normal
    # map occupies, so the material needs no new texture or sampler; a
    # template without one keeps having none
    normal_index = _normal_image_index(gltf, body_material) if normal is not None else None
    if normal_index is not None and gltf.images[normal_index].bufferView is not None:
        encoded = io.BytesIO()
        normal.save(encoded, format="PNG")
        replacements[gltf.images[normal_index].bufferView] = encoded.getvalue()
        gltf.images[normal_index].mimeType = "image/png"

    payloads = [
        replacements.get(index, _view_bytes(gltf, blob, index))
        for index in range(len(gltf.bufferViews))
    ]

    packed = bytearray()
    for index, (view, payload) in enumerate(zip(gltf.bufferViews, payloads)):
        # every view starts 4-byte aligned; accessors inherit the alignment
        # they need from their view's offset
        if len(packed) % 4:
            packed.extend(b"\x00" * (4 - len(packed) % 4))
        gltf.bufferViews[index] = BufferView(
            buffer=0,
            byteOffset=len(packed),
            byteLength=len(payload),
            byteStride=view.byteStride,
            target=view.target,
        )
        packed.extend(payload)

    _make_matte(gltf, only=body)
    _drop_vertex_colors(gltf, only=body)

    gltf.images[target] = GLTFImage(
        name=gltf.images[target].name,
        bufferView=gltf.images[target].bufferView,
        mimeType=f"image/{fmt.lower()}",
    )
    gltf.buffers[0].byteLength = len(packed)
    gltf.buffers[0].uri = None
    gltf.set_binary_blob(bytes(packed))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gltf.save(str(out_path))
    return out_path
