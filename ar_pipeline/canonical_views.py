"""Render the template from four fixed orthographic cameras, in rest pose.

WHAT THIS IS FOR
----------------
Later stages hand an image model the shelter photo plus a 2x2 sheet of
silhouettes and ask for the same animal drawn into those four cells. What comes
back is only useful if it can be put back onto the mesh, and that needs the
cameras that made the silhouettes to be known exactly. So every view here
writes colour, metric depth and a silhouette, and cameras.json records the
camera that produced them.

cameras.json is the fixed point of the whole projection path. Nothing after
this step may adjust it; if the views have to change, this step runs again and
everything downstream is rebuilt against the new file.

CONVENTIONS
-----------
The template faces -Y with +Z up. Its left flank is therefore +X (right-hand
rule), so the "left" camera sits at +X looking along -X. All four cameras sit
at zero elevation, orthographic, sharing one ortho_scale set from the model's
longest side, and all look at the model's bounding-box centre.

Depth is written as 32-bit OpenEXR straight from the Z pass, not a normalised
PNG, because the projection step compares distances across views and a
per-view 0..1 stretch would throw that away.

Usage:
    blender --background --python ar_pipeline/canonical_views.py -- \
        output/skeleton_shiba_small.glb output/canonical
"""

import json
import os
import sys

import bpy
import mathutils

VIEW_RESOLUTION = 1024

# Ortho scale is the model's longest side times this, so every view holds the
# whole animal with a little air and all four share one scale.
VIEW_MARGIN = 1.15

CANONICAL_VIEWS = (
    # name,   camera offset (unit),  look direction
    ("front", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
    ("left",  (1.0, 0.0, 0.0),  (-1.0, 0.0, 0.0)),
    ("right", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    ("back",  (0.0, 1.0, 0.0),  (0.0, -1.0, 0.0)),
)


def reset() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def rest_pose() -> None:
    """Freeze every armature in its bind pose and detach any clip.

    The importer leaves one action bound, and a canonical view must not depend
    on which clip that was: the projection step reconstructs these cameras
    against the same rest pose, so the two have to agree exactly.
    """
    for obj in bpy.data.objects:
        if obj.type != "ARMATURE":
            continue
        if obj.animation_data is not None:
            obj.animation_data.action = None
        obj.data.pose_position = "REST"


def main_mesh():
    """The animal is the mesh with the most vertices. The template also carries
    a unit Icosphere helper, and letting it into the bounds framed the dog at
    4% of the picture."""
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    return max(meshes, key=lambda obj: len(obj.data.vertices))


def isolate_main_mesh() -> None:
    keep = main_mesh()
    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj is not keep:
            obj.hide_render = True


def model_bounds():
    obj = main_mesh()
    corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    lo = [min(c[i] for c in corners) for i in range(3)]
    hi = [max(c[i] for c in corners) for i in range(3)]
    centre = [(lo[i] + hi[i]) / 2 for i in range(3)]
    span = max(hi[i] - lo[i] for i in range(3))
    return centre, span, lo, hi


def flat_lighting() -> None:
    """Even, shadowless light. The colour view is a reference for framing and
    coat placement, and a shadow across it would read as a marking."""
    world = bpy.data.worlds.new("flat")
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    background.inputs[1].default_value = 1.0
    bpy.context.scene.world = world


def depth_material():
    """A material that emits the camera-space depth of every fragment.

    Written as an emission of the Camera Data node's View Z Depth, rendered
    through a material override to a float EXR with colour management set to
    Standard, so the stored value is the distance itself. This avoids the
    compositor, whose API changed in Blender 5 and which is more machinery than
    a single pass warrants.
    """
    material = bpy.data.materials.new("depth_override")
    material.use_nodes = True
    tree = material.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    camera = tree.nodes.new("ShaderNodeCameraData")
    emission = tree.nodes.new("ShaderNodeEmission")
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(camera.outputs["View Z Depth"], emission.inputs["Color"])
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def render_depth(depth_path: str, override) -> None:
    scene = bpy.context.scene
    layer = scene.view_layers[0]
    settings = scene.render.image_settings
    keep = (settings.file_format, settings.color_mode, settings.color_depth,
            scene.view_settings.view_transform, scene.render.filepath)
    layer.material_override = override
    settings.file_format = "OPEN_EXR"
    settings.color_mode = "RGB"
    settings.color_depth = "32"
    scene.view_settings.view_transform = "Standard"
    scene.render.filepath = depth_path
    bpy.ops.render.render(write_still=True)
    layer.material_override = None
    (settings.file_format, settings.color_mode, settings.color_depth,
     scene.view_settings.view_transform, scene.render.filepath) = keep


def write_mask(colour_path: str, mask_path: str) -> int:
    """The silhouette is the alpha of the transparent-film render. Returns the
    number of covered pixels so the log can show each view actually held the
    animal."""
    image = bpy.data.images.load(colour_path)
    width, height = image.size
    pixels = list(image.pixels)
    alpha = pixels[3::4]
    flat = [0.0] * (width * height * 4)
    covered = 0
    for i, a in enumerate(alpha):
        v = 1.0 if a > 0.5 else 0.0
        covered += int(v)
        flat[i * 4:i * 4 + 4] = (v, v, v, 1.0)
    mask = bpy.data.images.new("mask", width=width, height=height, alpha=False)
    mask.pixels.foreach_set(flat)
    mask.filepath_raw = mask_path
    mask.file_format = "PNG"
    mask.save()
    bpy.data.images.remove(image)
    bpy.data.images.remove(mask)
    return covered


def main() -> None:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    if len(argv) != 2:
        print("[canonical] usage: ... -- template.glb out_dir")
        sys.exit(1)
    glb_path, out_dir = argv
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    reset()
    bpy.ops.import_scene.gltf(filepath=glb_path)
    rest_pose()
    isolate_main_mesh()
    bpy.context.view_layer.update()

    centre, span, lo, hi = model_bounds()
    print(f"[canonical] mesh '{main_mesh().name}' bounds {[round(v, 3) for v in lo]} .. "
          f"{[round(v, 3) for v in hi]}, span {span:.3f}")
    ortho_scale = span * VIEW_MARGIN
    # orthographic, so the distance only has to clear the mesh and set the clip
    distance = span * 3.0

    flat_lighting()

    scene = bpy.context.scene
    engines = scene.render.bl_rna.properties["engine"].enum_items.keys()
    scene.render.engine = next(
        (name for name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE") if name in engines),
        "BLENDER_WORKBENCH",
    )
    scene.render.resolution_x = VIEW_RESOLUTION
    scene.render.resolution_y = VIEW_RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    override = depth_material()

    camera_data = bpy.data.cameras.new("canonical")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = ortho_scale
    camera_data.clip_start = 0.01
    camera_data.clip_end = distance * 2.5
    camera = bpy.data.objects.new("canonical", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera

    manifest = {
        "template": os.path.basename(glb_path),
        "pose": "rest",
        "projection": "orthographic",
        "resolution": [VIEW_RESOLUTION, VIEW_RESOLUTION],
        "ortho_scale": ortho_scale,
        "clip": [camera_data.clip_start, camera_data.clip_end],
        "model_centre": centre,
        "model_bounds": {"min": lo, "max": hi},
        "up": [0, 0, 1],
        "forward": [0, -1, 0],
        "views": {},
    }

    for name, offset, look in CANONICAL_VIEWS:
        location = [centre[i] + offset[i] * distance for i in range(3)]
        camera.location = location
        camera.rotation_euler = mathutils.Vector(look).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()

        colour_path = os.path.join(out_dir, f"{name}_color.png")
        scene.render.filepath = colour_path
        bpy.ops.render.render(write_still=True)

        render_depth(os.path.join(out_dir, f"{name}_depth.exr"), override)

        mask_path = os.path.join(out_dir, f"{name}_mask.png")
        covered = write_mask(colour_path, mask_path)

        manifest["views"][name] = {
            "location": location,
            "look": list(look),
            "rotation_euler": [float(v) for v in camera.rotation_euler],
            "matrix_world": [list(row) for row in camera.matrix_world],
            "color": f"{name}_color.png",
            "depth": f"{name}_depth.exr",
            "mask": f"{name}_mask.png",
        }
        share = covered / (VIEW_RESOLUTION * VIEW_RESOLUTION)
        print(f"[canonical] {name}: colour, depth, mask ({share:.1%} of frame covered)")

    with open(os.path.join(out_dir, "cameras.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[canonical] wrote {os.path.join(out_dir, 'cameras.json')}")


if __name__ == "__main__":
    main()
