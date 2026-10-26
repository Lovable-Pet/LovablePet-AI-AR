"""
Mask/UV alignment check for the templates built by build_template.py.

Paints one region mask red on top of the model's base texture and writes a
throwaway GLB, so you can load it in test.html (or Blender) and confirm that
exactly the intended body part turns red. Nothing here feeds the real
pipeline -- the output lives in output/_preview/ and can be deleted.

Usage (headless):
    blender --background --python preview_mask.py -- output/skeleton_shiba_small.glb tail

Pass the region "gap" instead of a body part to tint everything that no
region claims -- those spots keep the base texture when the regions get
recolored, so they read as stripes between body parts.

Optional flags (after --):
    --mask path/to/mask.png   use this mask instead of the conventional
                              templates/{species}_{size}/masks/{region}.png
    --strength 0.65           how strongly the red tint is blended in (0-1)
    --output path.glb         override the output GLB path
"""

import argparse
import os
import re
import sys

import bpy
import numpy as np

OUTPUT_NAME_RE = re.compile(
    r"^skeleton_(?P<species>.+)_(?P<size>small|medium|large)\.glb$", re.IGNORECASE
)
# REGIONS lives in api/regions.py, which carries the boundary definitions and
# pulls in no Blender modules, so every stage can share the one list.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.regions import REGIONS  # noqa: E402
GAP_REGION = "gap"
TINT_RGB = (1.0, 0.0, 0.0)
MASK_THRESHOLD = 0.5


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("glb", help="GLB built by build_template.py")
    parser.add_argument("region", help="region name (head, tail, ...) or 'gap'")
    parser.add_argument("--mask", default=None, help="explicit mask PNG path")
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--output", default=None, help="output GLB path")
    return parser.parse_args(argv)


def templates_dir_for(glb_path):
    name = os.path.basename(glb_path)
    match = OUTPUT_NAME_RE.match(name)
    if not match:
        print(f"[preview_mask] ERROR: '{name}' does not follow the "
              "skeleton_{species}_{size}.glb convention, pass --mask explicitly.")
        sys.exit(1)
    folder = f"{match.group('species').lower()}_{match.group('size').lower()}"
    return os.path.join(os.path.dirname(os.path.abspath(glb_path)), "templates", folder)


def load_gray(path, non_color=True):
    image = bpy.data.images.load(path)
    try:
        return read_image(image, non_color=non_color)
    finally:
        bpy.data.images.remove(image)


def compute_gap_mask(templates_dir):
    """Every UV-island pixel that no region claims. Those spots keep the
    original texture when the regions get recolored, so they show up as
    stripes between body parts."""
    uv_layout, size = load_gray(os.path.join(templates_dir, "uv_layout.png"))
    island = uv_layout[..., 3] > 0.0

    claims = np.zeros(island.shape, dtype=np.int16)
    for region in REGIONS:
        path = os.path.join(templates_dir, "masks", f"{region}.png")
        if not os.path.isfile(path):
            continue
        mask, mask_size = load_gray(path)
        if mask_size != size:
            print(f"[preview_mask] WARNING: {region}.png is {mask_size}, skipping")
            continue
        claims += (mask[..., 0] > MASK_THRESHOLD).astype(np.int16)

    gap = (island & (claims == 0)).astype(np.float32)
    return gap[..., None].repeat(4, axis=2), size


def find_base_color_image(mesh_obj):
    """The image feeding Base Color, falling back to the largest texture."""
    fallback = None
    for slot in mesh_obj.material_slots:
        mat = slot.material
        if mat is None or mat.node_tree is None:
            continue
        for node in mat.node_tree.nodes:
            if node.type != "BSDF_PRINCIPLED":
                continue
            link = next((l for l in mat.node_tree.links
                         if l.to_node == node and l.to_socket.name == "Base Color"), None)
            if link and link.from_node.type == "TEX_IMAGE" and link.from_node.image:
                return link.from_node
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image and node.image.size[0]:
                if fallback is None or node.image.size[0] > fallback.image.size[0]:
                    fallback = node
    return fallback


def read_image(image, non_color):
    if non_color:
        image.colorspace_settings.name = "Non-Color"
    w, h = image.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    image.pixels.foreach_get(buf)
    return buf.reshape(h, w, 4), (w, h)


def main():
    args = parse_args()
    templates_dir = None if args.mask else templates_dir_for(args.glb)
    mask_file = args.mask
    if mask_file is None and args.region != GAP_REGION:
        mask_file = os.path.join(templates_dir, "masks", f"{args.region}.png")
    if mask_file is not None and not os.path.isfile(mask_file):
        print(f"[preview_mask] ERROR: mask not found: {mask_file}")
        sys.exit(1)

    out_glb = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.glb)), "_preview",
        f"mask_check_{args.region}.glb")
    os.makedirs(os.path.dirname(out_glb), exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=args.glb)

    meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.data.uv_layers]
    if not meshes:
        print("[preview_mask] ERROR: no UV-mapped mesh in the GLB.")
        sys.exit(1)
    mesh_obj = max(meshes, key=lambda o: len(o.data.polygons))

    node = find_base_color_image(mesh_obj)
    if node is None:
        print(f"[preview_mask] ERROR: no base color texture on '{mesh_obj.name}'.")
        sys.exit(1)

    base, base_size = read_image(node.image, non_color=False)

    if mask_file is None:
        mask, mask_size = compute_gap_mask(templates_dir)
        label = "uncovered area"
    else:
        mask, mask_size = load_gray(mask_file)
        label = os.path.basename(mask_file)

    if mask_size != base_size:
        print(f"[preview_mask] ERROR: mask is {mask_size[0]}x{mask_size[1]} but the "
              f"base texture is {base_size[0]}x{base_size[1]}.")
        sys.exit(1)

    coverage = float((mask[..., 0] > MASK_THRESHOLD).mean())
    if coverage == 0.0:
        print(f"[preview_mask] WARNING: '{args.region}' mask is empty, the "
              "preview will look exactly like the untouched model.")
    print(f"[preview_mask] '{label}' covers {coverage:.1%} of the UV space")

    alpha = np.clip(mask[..., 0:1], 0.0, 1.0) * args.strength
    tinted = base.copy()
    tinted[..., :3] = base[..., :3] * (1.0 - alpha) + np.array(TINT_RGB, dtype=np.float32) * alpha

    tint_png = os.path.splitext(out_glb)[0] + ".png"
    preview_image = bpy.data.images.new("mask_preview", width=base_size[0],
                                        height=base_size[1], alpha=True)
    preview_image.pixels.foreach_set(np.ascontiguousarray(tinted).ravel())
    preview_image.filepath_raw = tint_png
    preview_image.file_format = "PNG"
    preview_image.save()
    bpy.data.images.remove(preview_image)

    # reload from disk so the exporter embeds the saved file, not a generated buffer
    node.image = bpy.data.images.load(tint_png)

    bpy.ops.export_scene.gltf(filepath=out_glb, export_format="GLB",
                              export_animations=False)
    print(f"[preview_mask] wrote {out_glb}")
    print(f"[preview_mask] open test.html and pick '{args.region}' in the mask "
          "dropdown, or drag this GLB into Blender.")


if __name__ == "__main__":
    main()
