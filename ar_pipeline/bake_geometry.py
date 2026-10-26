"""Bake the template's world-space position and normal into UV space.

WHY
---
Projecting a view onto the mesh means asking, for every texel, "where is this
point in the world, which way does it face, and can camera X see it?" The first
two are properties of the template alone, so they are baked once here and every
projection afterwards is plain array arithmetic with no Blender in the loop.

The rasteriser is the one build_template.py already uses for region masks -
per-vertex values splatted across each UV triangle - fed with vertex positions
and normals instead of vertex-group weights. Same code path, same UV layout,
same padding behaviour, so the geometry lines up with the masks texel for texel.

STORAGE
-------
Position, normal, validity and the canonical depths are written as float32
.npy arrays. An earlier version encoded them into 16-bit PNGs, but Pillow reads
16-bit RGB PNGs down to 8 bits, which put position on a 4.5 mm grid and depth
on a 34 mm one - useless for a depth test with a 7 mm tolerance. Arrays are
exact and need no decoding. 8-bit PNG previews sit beside them for looking at,
and for nothing else. cameras.json is read here and never written.

Usage:
    blender --background --python ar_pipeline/bake_geometry.py -- \
        output/skeleton_shiba_small.glb output/canonical output/geometry
"""

import json
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_template import (  # noqa: E402
    bake_weights_to_uv,
    detect_texture_size,
    find_main_mesh,
    import_glb,
    mesh_uv_triangles,
    reset_scene,
)


def rest_pose() -> None:
    for obj in bpy.data.objects:
        if obj.type != "ARMATURE":
            continue
        if obj.animation_data is not None:
            obj.animation_data.action = None
        obj.data.pose_position = "REST"


def evaluated_world_vertices(mesh_obj):
    """Vertex positions and normals in world space, after the armature has
    been applied in rest pose - the same shape the canonical cameras saw."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = mesh_obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    matrix = mesh_obj.matrix_world
    normal_matrix = matrix.to_3x3().inverted().transposed()
    positions = np.array([tuple(matrix @ v.co) for v in mesh.vertices], dtype=np.float64)
    normals = np.array(
        [tuple((normal_matrix @ v.normal).normalized()) for v in mesh.vertices], dtype=np.float64
    )
    evaluated.to_mesh_clear()
    return positions, normals


def save_preview(path: str, array: np.ndarray) -> None:
    """8-bit PNG of an array in 0..1, for eyes only."""
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    h, w, _ = array.shape
    image = bpy.data.images.new(os.path.basename(path), width=w, height=h, alpha=False)
    image.colorspace_settings.name = "Non-Color"
    rgba = np.empty((h, w, 4), dtype=np.float32)
    rgba[..., :3] = np.clip(array, 0.0, 1.0)
    rgba[..., 3] = 1.0
    image.pixels.foreach_set(np.ascontiguousarray(rgba[::-1]).ravel())
    image.filepath_raw = path
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)


def load_exr_channel(path: str) -> np.ndarray:
    image = bpy.data.images.load(path)
    w, h = image.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    image.pixels.foreach_get(buf)
    bpy.data.images.remove(image)
    return buf.reshape(h, w, 4)[::-1, :, 0].copy()   # top row first, R channel


def main() -> None:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    if len(argv) != 3:
        print("[geometry] usage: ... -- template.glb canonical_dir out_dir")
        sys.exit(1)
    glb_path, canonical_dir, out_dir = (os.path.abspath(p) for p in argv)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(canonical_dir, "cameras.json"), encoding="utf-8") as handle:
        cameras = json.load(handle)

    reset_scene()
    import_glb(glb_path)
    rest_pose()
    bpy.context.view_layer.update()

    mesh_obj = find_main_mesh()
    if mesh_obj is None:
        print("[geometry] ERROR: no UV-mapped mesh found")
        sys.exit(1)
    size = detect_texture_size(mesh_obj, 0)
    width, height = size
    print(f"[geometry] mesh '{mesh_obj.name}', baking at {width}x{height}")

    positions, normals = evaluated_world_vertices(mesh_obj)
    tri_loops, tri_verts, uvs = mesh_uv_triangles(mesh_obj.data)

    lo = positions.min(axis=0)
    hi = positions.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)

    # the rasteriser skips a triangle whose values are all zero, so bake the
    # positions shifted to be strictly positive and shift back afterwards
    shifted = positions - lo + 1.0
    channels = [bake_weights_to_uv(tri_loops, tri_verts, uvs, shifted[:, axis], size) for axis in range(3)]
    valid = bake_weights_to_uv(tri_loops, tri_verts, uvs, np.ones(len(positions)), size)
    valid = (valid > 0.5).astype(np.float32)
    position_map = (np.stack(channels, axis=-1) - 1.0 + lo).astype(np.float32) * valid[..., None]

    shifted_n = normals + 2.0
    channels = [bake_weights_to_uv(tri_loops, tri_verts, uvs, shifted_n[:, axis], size) for axis in range(3)]
    normal_map = (np.stack(channels, axis=-1) - 2.0).astype(np.float32)
    length = np.maximum(np.linalg.norm(normal_map, axis=-1, keepdims=True), 1e-6)
    normal_map = normal_map / length * valid[..., None]
    print(f"[geometry] UV coverage: {valid.mean():.1%} of texels lie on a triangle")

    np.save(os.path.join(out_dir, "position.npy"), position_map)
    np.save(os.path.join(out_dir, "normal.npy"), normal_map)
    np.save(os.path.join(out_dir, "valid.npy"), valid)
    save_preview(os.path.join(out_dir, "position_preview.png"), (position_map - lo) / span)
    save_preview(os.path.join(out_dir, "normal_preview.png"), (normal_map + 1.0) / 2.0)
    save_preview(os.path.join(out_dir, "valid_preview.png"), valid)

    # canonical depth in metres, straight from the EXR; cameras.json untouched
    near, far = cameras["clip"]
    for name in cameras["views"]:
        depth = load_exr_channel(os.path.join(canonical_dir, f"{name}_depth.exr")).astype(np.float32)
        np.save(os.path.join(out_dir, f"{name}_depth.npy"), depth)
        print(f"[geometry] {name} depth ({depth[depth > near].min():.3f}..{depth.max():.3f} m)")

    manifest = {
        "template": os.path.basename(glb_path),
        "size": [width, height],
        "position": {"file": "position.npy", "min": lo.tolist(), "max": hi.tolist()},
        "normal": {"file": "normal.npy"},
        "valid": {"file": "valid.npy"},
        "depth": {"units": "metres", "near": near, "far": far,
                  "files": {name: f"{name}_depth.npy" for name in cameras["views"]}},
        "cameras": "../canonical/cameras.json",
    }
    with open(os.path.join(out_dir, "geometry.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[geometry] wrote {os.path.join(out_dir, 'geometry.json')}")


if __name__ == "__main__":
    main()
