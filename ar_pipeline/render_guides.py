"""Render the template's structure guides: the mesh from the four canonical
cameras, flat grey on pure white, plus the cameras that made them.

WHY
---
The generated views used to be drawn from a silhouette sheet, and the image
model drew a rounder, shorter head inside the outline than the mesh has, so
the face markings landed off the mesh when projected. A guide that shows the
form itself - the muzzle's length, where the eyes sit, how the ears stand -
conditions the generation on the template's proportions, and an image drawn
to the guide projects back with the same cameras exactly.

THE CAMERAS ARE THE CONTRACT
----------------------------
The guide camera and the projection camera must be one and the same, so the
cameras are not defined here: this script imports canonical_views and uses
its camera placement, ortho scale and resolution unchanged, and writes the
same manifest schema as guides/{template}_cameras.json. The projection reads
that file. Nothing in either step may adjust a camera by hand.

Usage:
    blender --background --python ar_pipeline/render_guides.py -- \
        output/skeleton_shiba_small.glb output/guides
Outputs:
    output/guides/{template}_{front|left|right|back}.png   (RGB, white ground)
    output/guides/{template}_{view}_mask.png              (silhouette)
    output/guides/{template}_cameras.json
"""

import json
import os
import sys

import bpy
import mathutils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import canonical_views as cv  # noqa: E402

# The guide is the form, not the coat. A lit grey would print shadows into
# the reference; a flat grey is only a silhouette, which is what the image
# model had before and drew a rounder head into. So the grey is shaded by
# how squarely each point faces the camera - a matcap, in effect: no light,
# no shadow, but the stop, the brow ridge and the cheek read as form.
GUIDE_GREY = 0.62
GUIDE_SHADE_FLOOR = 0.55     # share of the grey at a grazing angle
GUIDE_SHADE_RANGE = 0.45     # added when facing the camera squarely


def guide_material():
    material = bpy.data.materials.new("guide_grey")
    material.use_nodes = True
    tree = material.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    layer = tree.nodes.new("ShaderNodeLayerWeight")      # Facing: 0 head-on, 1 grazing
    invert = tree.nodes.new("ShaderNodeMath")
    invert.operation = "SUBTRACT"
    invert.inputs[0].default_value = 1.0
    scale = tree.nodes.new("ShaderNodeMath")
    scale.operation = "MULTIPLY_ADD"
    scale.inputs[1].default_value = GUIDE_SHADE_RANGE
    scale.inputs[2].default_value = GUIDE_SHADE_FLOOR
    grey = tree.nodes.new("ShaderNodeMath")
    grey.operation = "MULTIPLY"
    grey.inputs[1].default_value = GUIDE_GREY
    emission = tree.nodes.new("ShaderNodeEmission")
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(layer.outputs["Facing"], invert.inputs[1])
    tree.links.new(invert.outputs["Value"], scale.inputs[0])
    tree.links.new(scale.outputs["Value"], grey.inputs[0])
    tree.links.new(grey.outputs["Value"], emission.inputs["Color"])
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def main() -> None:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    if len(argv) != 2:
        print("[guides] usage: ... -- template.glb out_dir")
        sys.exit(1)
    glb_path, out_dir = argv
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(glb_path))[0]

    cv.reset()
    bpy.ops.import_scene.gltf(filepath=glb_path)
    cv.rest_pose()
    cv.isolate_main_mesh()
    bpy.context.view_layer.update()

    centre, span, lo, hi = cv.model_bounds()
    ortho_scale = span * cv.VIEW_MARGIN
    distance = span * 3.0
    cv.flat_lighting()

    scene = bpy.context.scene
    engines = scene.render.bl_rna.properties["engine"].enum_items.keys()
    scene.render.engine = next(
        (name for name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE") if name in engines),
        "BLENDER_WORKBENCH",
    )
    scene.render.resolution_x = cv.VIEW_RESOLUTION
    scene.render.resolution_y = cv.VIEW_RESOLUTION
    scene.render.resolution_percentage = 100
    # the guide is handed to an image model as a picture: opaque, white ground
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_layers[0].material_override = guide_material()

    camera_data = bpy.data.cameras.new("guide")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = ortho_scale
    camera_data.clip_start = 0.01
    camera_data.clip_end = distance * 2.5
    camera = bpy.data.objects.new("guide", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera

    manifest = {
        "template": os.path.basename(glb_path),
        "pose": "rest",
        "projection": "orthographic",
        "resolution": [cv.VIEW_RESOLUTION, cv.VIEW_RESOLUTION],
        "ortho_scale": ortho_scale,
        "clip": [camera_data.clip_start, camera_data.clip_end],
        "model_centre": centre,
        "model_bounds": {"min": lo, "max": hi},
        "up": [0, 0, 1],
        "forward": [0, -1, 0],
        "shading": {"material": "facing-shaded grey emission (no light, no shadow)",
                    "grey": GUIDE_GREY, "floor": GUIDE_SHADE_FLOOR, "range": GUIDE_SHADE_RANGE,
                    "background": "white"},
        "views": {},
    }

    for name, offset, look in cv.CANONICAL_VIEWS:
        location = [centre[i] + offset[i] * distance for i in range(3)]
        camera.location = location
        camera.rotation_euler = mathutils.Vector(look).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()

        guide_path = os.path.join(out_dir, f"{stem}_{name}.png")
        scene.render.filepath = guide_path
        bpy.ops.render.render(write_still=True)

        # the silhouette, for cutting the generated view out of its ground
        scene.render.film_transparent = True
        scene.render.image_settings.color_mode = "RGBA"
        alpha_path = os.path.join(out_dir, f"_{stem}_{name}_rgba.png")
        scene.render.filepath = alpha_path
        bpy.ops.render.render(write_still=True)
        mask_path = os.path.join(out_dir, f"{stem}_{name}_mask.png")
        covered = cv.write_mask(alpha_path, mask_path)
        os.remove(alpha_path)
        scene.render.film_transparent = False
        scene.render.image_settings.color_mode = "RGB"

        manifest["views"][name] = {
            "location": location,
            "look": list(look),
            "rotation_euler": [float(v) for v in camera.rotation_euler],
            "matrix_world": [list(row) for row in camera.matrix_world],
            "guide": os.path.basename(guide_path),
            "mask": os.path.basename(mask_path),
        }
        share = covered / (cv.VIEW_RESOLUTION * cv.VIEW_RESOLUTION)
        print(f"[guides] {name}: {os.path.basename(guide_path)} ({share:.1%} of frame)")

    cameras_path = os.path.join(out_dir, f"{stem}_cameras.json")
    with open(cameras_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[guides] wrote {cameras_path}")


if __name__ == "__main__":
    main()
