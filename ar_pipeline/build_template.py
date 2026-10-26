"""
Lovable Pet - AR pipeline step 1: base template builder.

Takes an already-rigged animal GLB (e.g. from Sketchfab) and produces the
per-species/per-size base template used by the later texture-composition
stages:

  1. two animation clips added on top of the EXISTING armature
       - idle      : subtle breathing motion (prefers a "breathe" bone)
       - tail_wag  : left-right rotation of the tail bone chain
  2. the mesh UV layout as a PNG, at the base texture's resolution
  3. nine region masks (head, ears, muzzle, chest, back, belly, legs,
     paws, tail), either baked from existing region vertex groups or
     emitted as blank paint-me templates with the UV layout underneath

No new mesh or bones are created. Each clip is kept as its own NLA
track/strip so the glTF exporter writes them out as separate,
independently-selectable animations (matches by name in
<model-viewer animation-name="idle|tail_wag">).

Usage (headless):
    blender --background --python build_template.py -- input.glb output/skeleton_shiba_small.glb

The input can also be a .blend prepared by prepare_regions.py, which is how
region vertex groups tagged in the Blender GUI get here -- glTF has no way to
carry them. Tagged regions are baked into real masks; untagged ones fall back
to blank paint-me templates.

The output filename is a fixed convention, skeleton_{species}_{size}.glb
with size in small|medium|large, because the size grade has to line up with
the weight-based grading in stage 5.2.

Outputs:
    output/skeleton_{species}_{size}.glb
    output/templates/{species}_{size}/uv_layout.png
    output/templates/{species}_{size}/masks/{region}.png   x9

Optional flags (after --):
    --fps 24            frame rate for the new clips (default 24)
    --idle-frames 96    length of the idle loop in frames (default 96, ~4s @24fps)
    --wag-frames 32     length of the tail_wag loop in frames (default 32, ~1.3s @24fps)
    --texture-size 2048 override the auto-detected base texture resolution
"""

import argparse
import os
import re
import sys

import bpy
import numpy as np

# REGIONS lives in api/regions.py, which carries the boundary definitions and
# pulls in no Blender modules, so every stage can share the one list.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.regions import MERGE, REGIONS, SOURCE_REGIONS, SOURCES  # noqa: E402

OUTPUT_NAME_RE = re.compile(
    r"^skeleton_(?P<species>.+)_(?P<size>small|medium|large)\.glb$", re.IGNORECASE
)

TAIL_NAME_HINTS = ["tail", "꼬리"]
BREATHE_NAME_HINTS = ["breathe", "breath"]
SPINE_NAME_HINTS = ["spine", "chest", "torso", "spine1", "spine01", "body"]

# Masks are consumed with a 0.5 threshold, so the UV guide baked into blank
# templates sits well below it and never reads as "inside the region".
GUIDE_LINE_VALUE = 0.18
MASK_THRESHOLD = 0.5
DEFAULT_TEXTURE_SIZE = 1024
SEAM_DILATION_PX = 2

# How wide a region's edge fades, in pixels of the 2048 map. A hard edge is
# right where the coat really does change abruptly - a Doberman's tan points
# have a visible line - but on a Shiba the inner/outer leg and the urajiro
# boundary are a gradient, and a hard split reads as a sticker laid on the fur.
# Only the boundaries that are soft on the animal are softened here; the same
# argument applies to throat and cheeks, so raise those if the urajiro on the
# face looks cut out.
DEFAULT_FEATHER_PX = 2
REGION_FEATHER_PX = {
    # the urajiro boundary - belly, legs, throat, cheeks, chest - is a
    # gradient on the animal, not a line. Rendered hard it reads as a decal,
    # and the tan that a real black-and-tan shows where black meets cream never
    # appears, because nothing in the pipeline paints a transition: the blend
    # through the soft edge is what produces it. Keyed by SOURCE region: the
    # feather is applied before the sources are unioned into a region.
    "underside": 10,
    "legs_lower": 10,
    "legs_upper": 10,
    "throat": 8,
    "cheeks": 8,
    "chest": 8,
    "ears_inner": 6,
}


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="path to the input rigged GLB")
    parser.add_argument("output", help="output GLB path, named skeleton_{species}_{size}.glb")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--idle-frames", type=int, default=96)
    parser.add_argument("--wag-frames", type=int, default=32)
    parser.add_argument("--texture-size", type=int, default=0,
                        help="override the auto-detected base texture resolution")
    return parser.parse_args(argv)


def parse_output_name(output_path):
    name = os.path.basename(output_path)
    match = OUTPUT_NAME_RE.match(name)
    if not match:
        print(f"[build_template] ERROR: output filename '{name}' does not follow the "
              "required convention skeleton_{species}_{size}.glb with size in "
              "small|medium|large. Example: output/skeleton_shiba_small.glb")
        sys.exit(1)
    return match.group("species").lower(), match.group("size").lower()


# --------------------------------------------------------------------------
# scene / rig helpers
# --------------------------------------------------------------------------

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_glb(path):
    bpy.ops.import_scene.gltf(filepath=path)


def load_input(path):
    """A .blend input carries the region vertex groups tagged in the GUI;
    glTF cannot store them, so that round trip has to go through a .blend."""
    if path.lower().endswith(".blend"):
        bpy.ops.wm.open_mainfile(filepath=os.path.abspath(path))
        print(f"[build_template] opened .blend input: {path}")
    else:
        reset_scene()
        import_glb(path)


def find_armature():
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            return obj
    return None


def find_main_mesh():
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.data.uv_layers]
    if not meshes:
        return None
    meshes.sort(key=lambda o: len(o.data.polygons), reverse=True)
    if len(meshes) > 1:
        others = ", ".join(o.name for o in meshes[1:])
        print(f"[build_template] NOTE: several UV-mapped meshes found, using the largest "
              f"'{meshes[0].name}' for UV layout and masks (ignoring: {others}).")
    return meshes[0]


def bone_depth(pose_bone):
    depth = 0
    b = pose_bone.parent
    while b is not None:
        depth += 1
        b = b.parent
    return depth


def find_matching_bones(armature, hints):
    matches = [
        pb for pb in armature.pose.bones
        if any(hint.lower() in pb.name.lower() for hint in hints)
    ]
    matches.sort(key=bone_depth)
    return matches


# --------------------------------------------------------------------------
# animation clips
# --------------------------------------------------------------------------

def new_action(armature, name):
    if armature.animation_data is None:
        armature.animation_data_create()
    # a source model shipping its own "idle" would otherwise push ours to
    # "idle.001" and the clip name model-viewer selects by would be wrong
    clash = bpy.data.actions.get(name)
    if clash is not None:
        print(f"[build_template] replacing the input's existing '{name}' action")
        bpy.data.actions.remove(clash)
    action = bpy.data.actions.new(name=name)
    action.use_fake_user = True
    armature.animation_data.action = action
    return action


def push_action_to_nla(armature, action, frame_start=1):
    track = armature.animation_data.nla_tracks.new()
    track.name = action.name
    track.strips.new(action.name, frame_start, action)
    armature.animation_data.action = None


def iter_fcurves(action):
    """Blender >= 4.4 moved fcurves under action.layers/strips/channelbags
    (the "slotted actions" data model); older Blender kept them directly
    on the action. Support both."""
    if hasattr(action, "fcurves"):
        yield from action.fcurves
        return
    for layer in action.layers:
        for strip in layer.strips:
            for channelbag in getattr(strip, "channelbags", []):
                yield from channelbag.fcurves


def set_smooth_interpolation(action):
    for fcurve in iter_fcurves(action):
        for kp in fcurve.keyframe_points:
            kp.interpolation = "SINE"


def key_scale(pose_bone, frame, scale):
    pose_bone.scale = scale
    pose_bone.keyframe_insert(data_path="scale", frame=frame, group=pose_bone.name)


def key_rotation(pose_bone, frame, euler_xyz):
    pose_bone.rotation_mode = "XYZ"
    pose_bone.rotation_euler = euler_xyz
    pose_bone.keyframe_insert(data_path="rotation_euler", frame=frame, group=pose_bone.name)


def build_idle_action(armature, num_frames):
    # a dedicated "breathe" bone (common in production rigs) is the best
    # target for idle breathing; fall back to spine/chest, then root.
    breathe_bones = find_matching_bones(armature, BREATHE_NAME_HINTS)
    spine_bones = breathe_bones or find_matching_bones(armature, SPINE_NAME_HINTS)
    if not spine_bones:
        spine_bones = [armature.pose.bones[0]] if armature.pose.bones else []
    if not spine_bones:
        print("[build_template] WARNING: no bones found at all, skipping idle action.")
        return None

    chest = spine_bones[0]
    action = new_action(armature, "idle")

    base_scale = tuple(chest.scale)
    breathe_scale = (base_scale[0], base_scale[1] * 1.03, base_scale[2])

    half = max(1, num_frames // 2)
    key_scale(chest, 1, base_scale)
    key_scale(chest, half, breathe_scale)
    key_scale(chest, num_frames, base_scale)

    action.use_frame_range = True
    action.frame_start = 1
    action.frame_end = num_frames
    set_smooth_interpolation(action)

    print(f"[build_template] idle action created on bone '{chest.name}' "
          f"({num_frames} frames, breathing scale animation).")
    return action


def build_tail_wag_action(armature, num_frames):
    tail_bones = find_matching_bones(armature, TAIL_NAME_HINTS)
    if not tail_bones:
        print("[build_template] WARNING: no tail bone found (looked for names "
              f"containing {TAIL_NAME_HINTS}). Skipping tail_wag action.")
        return None

    action = new_action(armature, "tail_wag")

    quarter = max(1, num_frames // 4)
    half = max(1, num_frames // 2)
    three_quarter = max(1, (3 * num_frames) // 4)

    for i, bone in enumerate(tail_bones):
        # amplitude shrinks further down the tail chain for a natural whip motion
        amplitude_deg = 20.0 / (1 + i * 0.6)
        amplitude = amplitude_deg * (3.14159265 / 180.0)
        base_rot = tuple(bone.rotation_euler) if bone.rotation_mode != "QUATERNION" else (0.0, 0.0, 0.0)

        key_rotation(bone, 1, base_rot)
        key_rotation(bone, quarter, (base_rot[0], base_rot[1], base_rot[2] + amplitude))
        key_rotation(bone, half, base_rot)
        key_rotation(bone, three_quarter, (base_rot[0], base_rot[1], base_rot[2] - amplitude))
        key_rotation(bone, num_frames, base_rot)

    action.use_frame_range = True
    action.frame_start = 1
    action.frame_end = num_frames
    set_smooth_interpolation(action)

    names = ", ".join(b.name for b in tail_bones)
    print(f"[build_template] tail_wag action created on bone(s): {names} "
          f"({num_frames} frames).")
    return action


# --------------------------------------------------------------------------
# glTF export + animation verification
# --------------------------------------------------------------------------

def export_glb(filepath, expected_clip_names):
    export_op = bpy.ops.export_scene.gltf
    valid_props = set(export_op.get_rna_type().properties.keys())

    desired_kwargs = {
        "filepath": filepath,
        "export_format": "GLB",
        "export_animations": True,
        "export_animation_mode": "NLA_TRACKS",
        "export_force_sampling": True,
        "export_apply": False,
    }
    kwargs = {k: v for k, v in desired_kwargs.items() if k in valid_props}
    dropped = set(desired_kwargs) - set(kwargs)
    if dropped:
        print(f"[build_template] NOTE: this Blender's glTF exporter doesn't "
              f"recognize {sorted(dropped)}, exporting without them.")

    export_op(**kwargs)
    print(f"[build_template] Exported: {filepath}")

    verify_animations_in_export(filepath, expected_clip_names)


def verify_animations_in_export(filepath, expected_clip_names):
    """Best-effort check right after export. Blender's own Python usually
    doesn't have pygltflib installed, so this is skipped gracefully if it's
    missing -- use verify_export.py with a regular Python + pygltflib for
    the authoritative check."""
    try:
        import pygltflib
    except ImportError:
        print("[build_template] pygltflib not available in Blender's Python, "
              "skipping in-process animation verification. Run verify_export.py "
              "separately to confirm the exported animations.")
        return

    gltf = pygltflib.GLTF2().load(filepath)
    found_names = {anim.name for anim in gltf.animations}
    missing = [n for n in expected_clip_names if n not in found_names]

    if missing:
        print(f"[build_template] VERIFICATION FAILED: missing animations {missing} "
              f"in exported file. Found: {sorted(found_names)}")
        sys.exit(1)

    print(f"[build_template] VERIFICATION OK: found animations {sorted(found_names)}")


# --------------------------------------------------------------------------
# UV layout
# --------------------------------------------------------------------------

def detect_texture_size(mesh_obj, override):
    if override:
        print(f"[build_template] texture size overridden: {override}x{override}")
        return (override, override)

    best = None
    for slot in mesh_obj.material_slots:
        mat = slot.material
        if mat is None or mat.node_tree is None:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image and node.image.size[0]:
                w, h = node.image.size
                if best is None or w * h > best[1] * best[2]:
                    best = (node.image.name, w, h)

    if best is None:
        print(f"[build_template] WARNING: no image texture found on '{mesh_obj.name}', "
              f"falling back to {DEFAULT_TEXTURE_SIZE}x{DEFAULT_TEXTURE_SIZE}.")
        return (DEFAULT_TEXTURE_SIZE, DEFAULT_TEXTURE_SIZE)

    print(f"[build_template] base texture '{best[0]}' is {best[1]}x{best[2]}, "
          "using it for UV layout and masks.")
    return (best[1], best[2])


def ensure_gpu_ready():
    """The PNG UV layout exporter draws into a GPU offscreen buffer, which
    stays uninitialized under --background until gpu.init() (Blender 5.0+)."""
    import gpu
    if not hasattr(gpu, "init"):
        return
    try:
        gpu.init()
    except SystemError as exc:
        print(f"[build_template] NOTE: headless GPU init failed ({exc}).")


def export_uv_layout(mesh_obj, filepath, size):
    try:
        bpy.ops.preferences.addon_enable(module="io_mesh_uv_layout")
    except Exception:
        pass  # already enabled, or bundled differently in this build

    ensure_gpu_ready()

    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj

    try:
        bpy.ops.uv.export_layout(filepath=filepath, export_all=True, mode="PNG",
                                 size=size, opacity=0.35)
    except Exception as exc:
        print(f"[build_template] WARNING: UV layout export failed ({exc}). "
              "Blank mask templates will have no UV guide underneath.")
        return False

    print(f"[build_template] UV layout exported: {filepath} ({size[0]}x{size[1]})")
    return True


def load_uv_guide(filepath, size):
    """Read the exported UV layout back as a faint grayscale guide layer."""
    if not os.path.isfile(filepath):
        return np.zeros((size[1], size[0]), dtype=np.float32)

    img = bpy.data.images.load(filepath)
    try:
        img.colorspace_settings.name = "Non-Color"
        w, h = img.size
        if (w, h) != tuple(size):
            print(f"[build_template] WARNING: UV layout is {w}x{h} but masks are "
                  f"{size[0]}x{size[1]}, skipping the guide layer.")
            return np.zeros((size[1], size[0]), dtype=np.float32)
        buf = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(buf)
    finally:
        bpy.data.images.remove(img)

    # exported layout is white lines over transparency; alpha is the coverage
    alpha = buf.reshape(h, w, 4)[::-1, :, 3]
    return alpha * GUIDE_LINE_VALUE


# --------------------------------------------------------------------------
# region masks
# --------------------------------------------------------------------------

def region_vertex_groups(mesh_obj):
    """Match region vertex groups by exact name only: a rigged model has one
    vertex group per bone ('head_jnt', 'tail_1_jnt', ...) and substring
    matching would silently mistake those for region groups.

    Looks for the sixteen SOURCE_REGIONS (what an existing template tags);
    a group tagged at a ten-region name is picked up under that name too,
    so a new template may tag either set."""
    by_name = {vg.name.lower(): vg for vg in mesh_obj.vertex_groups}
    found = {}
    for region in SOURCE_REGIONS:
        for candidate in (region, f"region_{region}", f"{region}_region", f"mask_{region}"):
            vg = by_name.get(candidate)
            if vg is not None:
                found[region] = vg
                break
    return found


def mesh_uv_triangles(mesh):
    if hasattr(mesh, "calc_loop_triangles"):
        mesh.calc_loop_triangles()

    tri_count = len(mesh.loop_triangles)
    tri_loops = np.empty(tri_count * 3, dtype=np.int32)
    tri_verts = np.empty(tri_count * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("loops", tri_loops)
    mesh.loop_triangles.foreach_get("vertices", tri_verts)

    uv_data = np.empty(len(mesh.loops) * 2, dtype=np.float32)
    mesh.uv_layers.active.data.foreach_get("uv", uv_data)

    return tri_loops.reshape(-1, 3), tri_verts.reshape(-1, 3), uv_data.reshape(-1, 2)


def vertex_weights(mesh, group_indices):
    """One pass over the vertices collecting the weights of every region
    group at once."""
    wanted = set(group_indices)
    weights = {idx: np.zeros(len(mesh.vertices), dtype=np.float32) for idx in wanted}
    for vert in mesh.vertices:
        for g in vert.groups:
            if g.group in wanted:
                weights[g.group][vert.index] = g.weight
    return weights


def rasterize_triangle(img, tri_uv, tri_val):
    h, w = img.shape
    xs = tri_uv[:, 0] * (w - 1)
    ys = (1.0 - tri_uv[:, 1]) * (h - 1)  # UV origin is bottom-left, image is top-down

    x0 = max(int(np.floor(xs.min())), 0)
    x1 = min(int(np.ceil(xs.max())), w - 1)
    y0 = max(int(np.floor(ys.min())), 0)
    y1 = min(int(np.ceil(ys.max())), h - 1)
    if x1 < x0 or y1 < y0:
        return

    denom = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
    if abs(denom) < 1e-9:
        return

    px = np.arange(x0, x1 + 1, dtype=np.float32)[None, :]
    py = np.arange(y0, y1 + 1, dtype=np.float32)[:, None]

    l0 = ((ys[1] - ys[2]) * (px - xs[2]) + (xs[2] - xs[1]) * (py - ys[2])) / denom
    l1 = ((ys[2] - ys[0]) * (px - xs[2]) + (xs[0] - xs[2]) * (py - ys[2])) / denom
    l2 = 1.0 - l0 - l1

    inside = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4)
    vals = np.where(inside, l0 * tri_val[0] + l1 * tri_val[1] + l2 * tri_val[2], 0.0)

    block = img[y0:y1 + 1, x0:x1 + 1]
    np.maximum(block, vals.astype(np.float32), out=block)


def bake_weights_to_uv(tri_loops, tri_verts, uvs, weights, size):
    img = np.zeros((size[1], size[0]), dtype=np.float32)
    for loops, verts in zip(tri_loops, tri_verts):
        tri_val = weights[verts]
        if tri_val.max() <= 0.0:
            continue
        rasterize_triangle(img, uvs[loops], tri_val)
    return img


def dilate(img, pixels):
    """Bleed the mask a few pixels past the UV islands so texture filtering
    doesn't sample background across a seam."""
    out = img
    for _ in range(pixels):
        out = np.maximum.reduce([
            out,
            np.roll(out, 1, axis=0), np.roll(out, -1, axis=0),
            np.roll(out, 1, axis=1), np.roll(out, -1, axis=1),
        ])
    return out


def _box_blur(img, radius, axis):
    """One separable box pass, via a cumulative sum so the cost does not grow
    with the radius."""
    pad = [(0, 0), (0, 0)]
    pad[axis] = (radius, radius)
    padded = np.pad(img, pad, mode="edge")

    shape = list(padded.shape)
    shape[axis] += 1
    cumulative = np.zeros(shape, dtype=np.float32)
    index = [slice(None), slice(None)]
    index[axis] = slice(1, None)
    cumulative[tuple(index)] = np.cumsum(padded, axis=axis)

    width = 2 * radius + 1
    hi, lo = [slice(None), slice(None)], [slice(None), slice(None)]
    hi[axis] = slice(width, None)
    lo[axis] = slice(0, -width)
    return (cumulative[tuple(hi)] - cumulative[tuple(lo)]) / width


def feather(img, pixels):
    """Fade the mask edge over roughly `pixels`, so the composited color eases
    into the one beside it instead of butting against it.

    Two box passes rather than one: a single pass falls off in a straight line
    and still shows where it starts, while two give the S-curve the eye reads
    as fur rather than as a cut.

    The fade spreads outward as well as in, so it wants to stay inside the UV
    padding; at 16px padding on a 2048 map, anything past about 12 starts to
    reach the next island.
    """
    if pixels < 1:
        return img
    radius = max(1, pixels // 2)
    out = img
    for _ in range(2):
        out = _box_blur(out, radius, 0)
        out = _box_blur(out, radius, 1)
    return np.clip(out, 0.0, 1.0)


def save_gray_png(array, filepath):
    h, w = array.shape
    img = bpy.data.images.new(os.path.basename(filepath), width=w, height=h, alpha=False)
    try:
        img.colorspace_settings.name = "Non-Color"
        rgba = np.empty((h, w, 4), dtype=np.float32)
        rgba[..., 0] = array
        rgba[..., 1] = array
        rgba[..., 2] = array
        rgba[..., 3] = 1.0
        img.pixels.foreach_set(np.ascontiguousarray(rgba[::-1]).ravel())
        img.filepath_raw = filepath
        img.file_format = "PNG"
        img.save()
    finally:
        bpy.data.images.remove(img)


def write_region_masks(mesh_obj, masks_dir, guide, size):
    """Bake every tagged source group to masks_source/<name>.png, then write
    masks/<region>.png as the union (texel maximum) of that region's sources.
    A region none of whose sources is tagged gets a blank paint-me template."""
    groups = region_vertex_groups(mesh_obj)
    templates_written = []
    source_dir = os.path.join(os.path.dirname(os.path.normpath(masks_dir)), "masks_source")
    os.makedirs(source_dir, exist_ok=True)

    if not groups:
        print("[build_template] no region vertex groups found "
              f"(looked for exact names {SOURCE_REGIONS}). Writing {len(REGIONS)} blank "
              "paint-me templates instead.")
        for region in REGIONS:
            save_gray_png(guide, os.path.join(masks_dir, f"{region}.png"))
            templates_written.append(region)
        return templates_written

    print(f"[build_template] region vertex groups found: {sorted(groups)}")
    mesh = mesh_obj.data
    tri_loops, tri_verts, uvs = mesh_uv_triangles(mesh)
    weights = vertex_weights(mesh, [vg.index for vg in groups.values()])

    baked_sources = {}
    for source in SOURCE_REGIONS:
        vg = groups.get(source)
        if vg is None:
            print(f"  - {source}: no vertex group")
            continue
        mask = bake_weights_to_uv(tri_loops, tri_verts, uvs, weights[vg.index], size)
        coverage = float((mask > MASK_THRESHOLD).mean())
        if coverage == 0.0:
            print(f"  - {source}: vertex group '{vg.name}' is empty")
            continue
        softness = REGION_FEATHER_PX.get(source, DEFAULT_FEATHER_PX)
        baked = feather(dilate(mask, SEAM_DILATION_PX), softness)
        baked_sources[source] = baked
        save_gray_png(baked, os.path.join(source_dir, f"{source}.png"))
        print(f"  - {source}: baked from vertex group '{vg.name}' "
              f"(covers {coverage:.1%} of the UV space, feather {softness}px)")

    for region in REGIONS:
        path = os.path.join(masks_dir, f"{region}.png")
        parts = [baked_sources[s] for s in SOURCES[region] if s in baked_sources]
        if not parts:
            # keep the paintable template rather than writing a guide-less black image
            save_gray_png(guide, path)
            templates_written.append(region)
            print(f"  - {region}: none of {SOURCES[region]} tagged, blank template written")
            continue
        union = parts[0]
        for part in parts[1:]:
            union = np.maximum(union, part)
        save_gray_png(union, path)
        print(f"  - {region}: union of {[s for s in SOURCES[region] if s in baked_sources]}")

    # The features mask is not a region and is not in REGIONS: it marks the
    # eyes, nose and mouth, which every texture build leaves as the template
    # painted them. Underscored so nothing that globs region masks picks it up.
    features = mesh_obj.vertex_groups.get("features")
    if features is not None:
        weights_f = vertex_weights(mesh, [features.index])
        mask = bake_weights_to_uv(tri_loops, tri_verts, uvs, weights_f[features.index], size)
        coverage = float((mask > MASK_THRESHOLD).mean())
        save_gray_png(feather(dilate(mask, SEAM_DILATION_PX), 3), os.path.join(masks_dir, "_features.png"))
        print(f"  - _features: eyes/nose/mouth, kept from the template (covers {coverage:.1%})")

    return templates_written


# --------------------------------------------------------------------------
# mask verification
# --------------------------------------------------------------------------

def verify_masks(masks_dir, size):
    missing, wrong_size, blank = [], [], []

    for region in REGIONS:
        path = os.path.join(masks_dir, f"{region}.png")
        if not os.path.isfile(path):
            missing.append(region)
            continue

        img = bpy.data.images.load(path)
        try:
            img.colorspace_settings.name = "Non-Color"
            w, h = img.size
            if (w, h) != tuple(size):
                wrong_size.append(f"{region} ({w}x{h}, expected {size[0]}x{size[1]})")
                continue
            buf = np.empty(w * h * 4, dtype=np.float32)
            img.pixels.foreach_get(buf)
            if buf.reshape(-1, 4)[:, 0].max() <= MASK_THRESHOLD:
                blank.append(region)
        finally:
            bpy.data.images.remove(img)

    print(f"[build_template] mask check in {masks_dir}")
    print(f"  - present : {len(REGIONS) - len(missing)}/{len(REGIONS)}")
    if missing:
        print(f"  - MISSING : {missing}")
    if wrong_size:
        print(f"  - WRONG SIZE : {wrong_size}")
    if blank:
        print(f"  - TODO, still unpainted (all black): {blank}")
        print("    open each one in an image editor, paint the region white "
              "over the UV guide, save as PNG at the same resolution.")

    ok = not missing and not wrong_size
    print(f"[build_template] MASK VERIFICATION {'OK' if ok else 'FAILED'}")
    return ok


# --------------------------------------------------------------------------

def main():
    args = parse_args()
    species, size_grade = parse_output_name(args.output)
    print(f"[build_template] species='{species}', size grade='{size_grade}'")

    load_input(args.input)

    armature = find_armature()
    if armature is None:
        print("[build_template] ERROR: no armature found in the imported GLB. "
              "This tool only adds animation to an already-rigged model.")
        sys.exit(1)

    bpy.context.scene.render.fps = args.fps

    created_clips = []
    for action in (build_idle_action(armature, args.idle_frames),
                   build_tail_wag_action(armature, args.wag_frames)):
        if action is not None:
            push_action_to_nla(armature, action, frame_start=1)
            created_clips.append(action.name)

    if not created_clips:
        print("[build_template] ERROR: no animation clips could be created "
              "(no usable bones found). Aborting export.")
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    export_glb(args.output, expected_clip_names=created_clips)

    mesh_obj = find_main_mesh()
    if mesh_obj is None:
        print("[build_template] ERROR: no UV-mapped mesh found, cannot build the "
              "UV layout or region masks.")
        sys.exit(1)

    template_dir = os.path.join(out_dir, "templates", f"{species}_{size_grade}")
    masks_dir = os.path.join(template_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    size = detect_texture_size(mesh_obj, args.texture_size)
    uv_path = os.path.join(template_dir, "uv_layout.png")
    export_uv_layout(mesh_obj, uv_path, size)

    guide = load_uv_guide(uv_path, size)
    write_region_masks(mesh_obj, masks_dir, guide, size)

    if not verify_masks(masks_dir, size):
        sys.exit(1)


if __name__ == "__main__":
    main()
