"""Render a built pet GLB to a PNG, so a coat can be judged against the photo.

Looking at a UV texture tells you which mask got painted; it does not tell you
whether the animal looks right. This puts the model under plain lighting at a
three-quarter front view - roughly how a shelter photo is taken - and writes an
image you can hold next to the original.

The pose matters as much as the coat. Imported straight from the GLB the
armature sits in whatever clip the exporter left active - play_dead on these
templates - which renders a dog on its back and tells you nothing about the
markings, so a clip and a frame within it can be named instead:
    ... -- pet.glb out.png --clip sitting
    ... -- pet.glb out.png --clip standing --frame 140

Usage:
    blender --background --python ar_pipeline/render_preview.py -- \
        output/live_render/D2023072630.glb output/live_render/preview.png
"""

import math
import os
import sys

import bpy


def reset() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def head_focus():
    """Centre and span of the head, from the rig's head/nose/ear joints, for
    the face close-up. None when the rig has no such bones."""
    import mathutils

    points = []
    for obj in bpy.data.objects:
        if obj.type != "ARMATURE":
            continue
        for bone in obj.pose.bones:
            name = bone.name.lower()
            if any(key in name for key in ("head", "nose", "ear_base", "ear_tip", "eye", "jaw")):
                points.append(obj.matrix_world @ bone.head)
    if len(points) < 3:
        return None
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    zs = [p.z for p in points]
    centre = mathutils.Vector(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2))
    span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    return centre, span * 1.6


def frame_subject(camera_object, target, distance_scale: float = 2.3,
                  focus=None, angle_deg: float = 35.0, height_share: float = 0.35) -> None:
    """Point the camera at the mesh's centre from the front-left and back off
    far enough to hold the whole animal. `focus` = (centre, span) overrides
    the mesh bounds, for a close-up on part of it."""
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    corners = [obj.matrix_world @ corner
               for obj in meshes for corner in
               [__import__("mathutils").Vector(c) for c in obj.bound_box]]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    centre = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2)
    span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    if focus is not None:
        centre, span = (focus[0].x, focus[0].y, focus[0].z), focus[1]

    radius = span * distance_scale
    # -Y is forward on these templates, so the camera sits forward and to one
    # side, a little above the shoulder
    angle = math.radians(angle_deg)
    camera_object.location = (
        centre[0] + radius * math.sin(angle),
        centre[1] - radius * math.cos(angle),
        centre[2] + span * height_share,
    )
    direction = (
        centre[0] - camera_object.location[0],
        centre[1] - camera_object.location[1],
        centre[2] - camera_object.location[2],
    )
    import mathutils

    camera_object.rotation_euler = mathutils.Vector(direction).to_track_quat("-Z", "Y").to_euler()
    target.location = centre


def build_lighting(span: float, key_angle: float = 35.0) -> None:
    """A key, a fill and a rim. Flat ambient would hide exactly what we are
    checking - whether the coat reads as fur - so the key is angled.

    key_angle swings the key light around the model. Rendering the same pet at
    three angles is the delighting check: a shadow baked into the albedo stays
    put while the light moves, a real one follows it."""
    radians = math.radians(key_angle)
    key_location = (span * 2.8 * math.sin(radians), -span * 2.8 * math.cos(radians), span * 2.2)
    for name, location, energy, size in (
        ("key", key_location, 900, span),
        ("fill", (-span * 2.2, -span * 1.4, span * 0.8), 250, span * 1.5),
        ("rim", (0, span * 2.4, span * 1.6), 350, span),
    ):
        light_data = bpy.data.lights.new(name=name, type="AREA")
        light_data.energy = energy
        light_data.size = size
        light = bpy.data.objects.new(name, light_data)
        light.location = location
        bpy.context.collection.objects.link(light)
        direction = (-location[0], -location[1], -location[2])
        import mathutils

        light.rotation_euler = mathutils.Vector(direction).to_track_quat("-Z", "Y").to_euler()

    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.55, 0.55, 0.57, 1.0)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.6
    bpy.context.scene.world = world


def apply_clip(clip_name, frame):
    """Put the armature into `clip_name` and return the frame to render.

    The glTF importer leaves one action bound arbitrarily, so the clip is
    always chosen explicitly. An unknown name is reported with the real ones
    rather than quietly rendering the wrong pose.
    """
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if not armatures:
        print("[render_preview] no armature; rendering the mesh as imported")
        return 1

    available = {action.name.lower(): action for action in bpy.data.actions}
    action = available.get((clip_name or "").lower())
    if action is None:
        # a template built from an unrigged model carries only the clips
        # build_template.py adds (idle, tail_wag): render the rest pose
        print(f"[render_preview] no clip named {clip_name!r}; have "
              f"{sorted(a.name for a in bpy.data.actions)}; rendering the rest pose")
        return 1

    start, end = (int(value) for value in action.frame_range)
    # halfway in, where the pose has settled: these clips open on the neutral
    # bind pose and ease into the motion
    chosen = frame if frame is not None else (start + end) // 2
    chosen = max(start, min(end, chosen))

    for armature in armatures:
        if armature.animation_data is None:
            armature.animation_data_create()
        armature.animation_data.action = action
        slots = getattr(armature.animation_data, "action_suitable_slots", None)
        if slots:
            armature.animation_data.action_slot = slots[0]

    print(f"[render_preview] clip '{action.name}' frame {chosen} of {start}-{end}")
    return chosen


def add_ground(span: float) -> None:
    """A floor under the animal.

    Without one the pet hangs in empty space and reads as a cut-out, which
    makes it harder to judge against a photo taken on a shelter floor - the
    contact shadow is most of what tells you the pose is sitting rather than
    hovering.
    """
    lowest = min(
        (obj.matrix_world @ __import__("mathutils").Vector(corner)).z
        for obj in bpy.data.objects if obj.type == "MESH"
        for corner in obj.bound_box
    )
    bpy.ops.mesh.primitive_plane_add(size=span * 12, location=(0, 0, lowest))
    plane = bpy.context.active_object
    material = bpy.data.materials.new("ground")
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.58, 0.58, 0.59, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.95
    plane.data.materials.append(material)


def main() -> None:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    if "--views" in argv:
        # canonical orthographic views of the template itself, for the
        # projection path; the implementation lives beside this file
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import canonical_views
        index = argv.index("--views")
        sys.argv = [sys.argv[0], "--", argv[0], argv[index + 1]]
        canonical_views.main()
        return
    clip_name, frame = None, None
    if "--clip" in argv:
        index = argv.index("--clip")
        clip_name = argv[index + 1]
        del argv[index:index + 2]
    if "--frame" in argv:
        index = argv.index("--frame")
        frame = int(argv[index + 1])
        del argv[index:index + 2]
    zoom = 2.3
    if "--zoom" in argv:
        index = argv.index("--zoom")
        zoom = float(argv[index + 1])
        del argv[index:index + 2]
    height_share = None
    if "--height" in argv:
        # camera height above the model's centre, as a share of its span (0.35 default)
        index = argv.index("--height")
        height_share = float(argv[index + 1])
        del argv[index:index + 2]
    view_angle = 35.0
    angle_given = "--angle" in argv
    if "--angle" in argv:
        index = argv.index("--angle")
        view_angle = float(argv[index + 1])
        del argv[index:index + 2]
    face = False
    if "--face" in argv:
        # a close-up of the head, framed on the rig's head joints, from
        # nearer the front so both eyes and the muzzle are in view
        face = True
        argv.remove("--face")
        zoom = min(zoom, 1.6)
        # 20 degrees unless an angle was asked for (a straight-on face shows
        # whether a marking sits on the midline)
        view_angle = view_angle if angle_given else 20.0
    grey = False
    if "--grey" in argv:
        # every material one plain grey: the outline alone, for comparing
        # templates by shape (api/model_sheet.py)
        grey = True
        argv.remove("--grey")
    # no floor and a transparent background, to cut the animal out (the grey
    # sheet; tools/template_compare.py's AR-like view, where the floor is the
    # viewer's own)
    clear = grey
    if "--clear" in argv:
        clear = True
        argv.remove("--clear")
    key_angle = 35.0
    if "--light" in argv:
        index = argv.index("--light")
        key_angle = float(argv[index + 1])
        del argv[index:index + 2]
    if len(argv) != 2:
        print("[render_preview] usage: ... -- in.glb out.png [--clip NAME] [--frame N] "
              "[--zoom F] [--angle DEG] [--face] [--light DEG] [--grey] [--clear]")
        sys.exit(1)
    glb_path, out_png = argv

    reset()
    bpy.ops.import_scene.gltf(filepath=glb_path)
    pose_frame = apply_clip(clip_name, frame) if clip_name else 1

    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if not meshes:
        print("[render_preview] ERROR: the GLB contains no mesh")
        sys.exit(1)
    if grey:
        plain = bpy.data.materials.new("grey")
        plain.use_nodes = True
        bsdf = plain.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.62, 0.62, 0.62, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.8
        for obj in meshes:
            obj.data.materials.clear()
            obj.data.materials.append(plain)
    span = max(max(obj.dimensions) for obj in meshes)

    target = bpy.data.objects.new("focus", None)
    bpy.context.collection.objects.link(target)
    camera_data = bpy.data.cameras.new("camera")
    camera_data.lens = 70
    camera = bpy.data.objects.new("camera", camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    bpy.context.scene.frame_set(pose_frame)
    bpy.context.view_layer.update()
    focus = head_focus() if face else None
    if face and focus is None:
        print("[render_preview] no head joints found; framing the whole animal")
    frame_subject(camera, target, zoom, focus, view_angle,
                  height_share if height_share is not None else (0.15 if face else 0.35))
    build_lighting(span, key_angle)
    if not clear:
        add_ground(span)

    scene = bpy.context.scene
    # the EEVEE identifier was renamed between Blender versions, so pick
    # whichever one this build actually offers
    engines = scene.render.bl_rna.properties["engine"].enum_items.keys()
    scene.render.engine = next(
        (name for name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE") if name in engines),
        "BLENDER_WORKBENCH",
    )
    # the check has to show the texture's own colour: Blender's default AgX
    # view transform desaturates (the yellow jindo's fawn back, LAB chroma
    # 25 in the texture, rendered at 12 and the coat read as cream, fixture
    # 2222). Standard is what a GLB viewer shows.
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    # the lights were set for AgX's soft shoulder; without it the lit fur
    # rendered 16 L above its texture and the floor clipped to white
    scene.view_settings.exposure = -0.7
    scene.render.resolution_x = 700
    scene.render.resolution_y = 900
    scene.render.film_transparent = clear
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if clear else "RGB"
    scene.render.filepath = os.path.abspath(out_png)
    scene.frame_set(pose_frame)
    bpy.ops.render.render(write_still=True)
    print(f"[render_preview] wrote {out_png}")


if __name__ == "__main__":
    main()
