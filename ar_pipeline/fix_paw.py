"""Blender: rebuild one paw of a template after its mirrored twin.

WHY
---
The Meshy maltese (2026-10-12) has its right front paw as a handful of
large flat facets with a flat polygonal sole; the other three paws are
rounded. With a plain coat colour the flat paw reads as broken. Pulling
its few vertices onto the good paw's shape (mirror_paw.py, numpy) moved
them 2 mm and changed nothing: there are no vertices to bend.

HOW
---
In Blender: the bad paw's faces are subdivided (two cuts, so the sole and
the toes get vertices), the good paw's faces are duplicated, mirrored
across the midline and slid onto the bad paw, and a Shrinkwrap modifier
(nearest surface point, weighted by a vertex group that fades out up the
pastern) pulls the subdivided paw onto that surface. Custom split normals
are dropped so the moved faces shade from their new geometry. The GLB is
exported the way build_template.py exports it (NLA tracks, no modifier
apply), so the rig, the clips and the texture come through.

Usage:
    blender --background --python-exit-code 1 --python ar_pipeline/fix_paw.py -- \\
        output/skeleton_maltese_small.glb output/skeleton_maltese_small.glb --good LF --bad RF
"""

import argparse
import sys

import bpy
from mathutils import Vector

PAW_BONES = {  # bones whose weight marks a paw
    "LF": ("LF_toes_jnt", "L_hand_jnt"), "RF": ("RF_toes_jnt", "R_hand_jnt"),
    "LB": ("LB_toes_jnt", "L_foot_jnt"), "RB": ("RB_toes_jnt", "R_foot_jnt"),
}
PAW_WEIGHT_MIN = 0.3
PAW_HEIGHT = 0.16      # m above the ground: the paw and the pastern
FADE_FROM = 0.10       # full pull below this height, fading to zero at PAW_HEIGHT
CUTS = 2
# A relaxing pass after the pull (Smooth modifier, 0.5 x 6) tore the paw
# open: the Meshy fur flakes are disconnected shells and each shell relaxed
# on its own. Left at 0 rounds; the nearest-point pull alone keeps them together.
SMOOTH_FACTOR, SMOOTH_ROUNDS = 0.5, 0


def paw_vertices(obj, bones, ground):
    names = {vg.index: vg.name for vg in obj.vertex_groups}
    out = set()
    for v in obj.data.vertices:
        w = max((g.weight for g in v.groups if names.get(g.group) in bones), default=0.0)
        if w >= PAW_WEIGHT_MIN and (obj.matrix_world @ v.co).z < ground + PAW_HEIGHT:
            out.add(v.index)
    return out


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--good", default="LF", choices=sorted(PAW_BONES))
    ap.add_argument("--bad", default="RF", choices=sorted(PAW_BONES))
    args = ap.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=args.src)
    obj = max((o for o in bpy.data.objects if o.type == "MESH"), key=lambda o: len(o.data.vertices))
    bpy.context.view_layer.objects.active = obj
    ground = min((obj.matrix_world @ v.co).z for v in obj.data.vertices)
    good = paw_vertices(obj, PAW_BONES[args.good], ground)
    bad = paw_vertices(obj, PAW_BONES[args.bad], ground)
    print(f"[fix_paw] {args.good} {len(good)} vertices, {args.bad} {len(bad)} vertices, ground z {ground:.3f}")

    # the target: the good paw's faces, mirrored across x and slid onto the bad paw
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.ops.object.duplicate()
    target = bpy.context.view_layer.objects.active
    target.name = "paw_target"
    target.modifiers.clear()
    target.parent = None
    target.matrix_world = obj.matrix_world.copy()
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    for p in target.data.polygons:
        p.select = not all(i in good for i in p.vertices)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.delete(type="FACE")
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    good_centre = sum((v.co for v in target.data.vertices), Vector()) / len(target.data.vertices)
    for v in target.data.vertices:
        v.co.x = -v.co.x
    bad_centre = sum(((obj.matrix_world @ obj.data.vertices[i].co) for i in bad), Vector()) / len(bad)
    shift = bad_centre - Vector((-good_centre.x, good_centre.y, good_centre.z))
    shift.z = 0.0
    for v in target.data.vertices:
        v.co += shift
    target.data.update()
    print(f"[fix_paw] target {len(target.data.polygons)} faces, slid {shift.length * 1000:.0f} mm")

    # subdivide the bad paw's faces so there is something to bend
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    n_faces = 0
    for p in obj.data.polygons:
        p.select = all(i in bad for i in p.vertices)
        n_faces += p.select
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.subdivide(number_cuts=CUTS, smoothness=0.0)
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    print(f"[fix_paw] subdivided {n_faces} faces -> mesh {len(obj.data.vertices)} vertices")

    # the pull weight: 1 on the paw, fading to 0 up the pastern
    vg = obj.vertex_groups.new(name="paw_fix")
    sx = -1 if args.bad.startswith("R") else 1
    sz = 1 if args.bad.endswith("F") else -1
    names = {g.index: g.name for g in obj.vertex_groups}
    count = 0
    for v in obj.data.vertices:
        w = max((g.weight for g in v.groups if names.get(g.group) in PAW_BONES[args.bad]), default=0.0)
        co = obj.matrix_world @ v.co
        if w < PAW_WEIGHT_MIN * 0.5 or (co.x * sx) < 0 or (co.y * -sz) < 0:
            continue
        h = co.z - ground
        if h >= PAW_HEIGHT:
            continue
        pull = 1.0 if h <= FADE_FROM else (PAW_HEIGHT - h) / (PAW_HEIGHT - FADE_FROM)
        vg.add([v.index], pull, "REPLACE")
        count += 1
    print(f"[fix_paw] pull weights on {count} vertices")

    mod = obj.modifiers.new("paw_fix", "SHRINKWRAP")
    mod.target = target
    mod.wrap_method = "NEAREST_SURFACEPOINT"
    mod.vertex_group = "paw_fix"
    # the good paw's surface is itself fur flakes; a nearest-point pull onto
    # it zigzags, so the pulled paw is relaxed a little afterwards
    smooth = obj.modifiers.new("paw_smooth", "SMOOTH")
    smooth.factor = SMOOTH_FACTOR
    smooth.iterations = SMOOTH_ROUNDS
    smooth.vertex_group = "paw_fix"
    # apply by hand: evaluate with only the shrinkwrap on (the armature modifier
    # off, so the idle pose does not leak into the rest mesh) and copy the
    # coordinates back - modifier_apply trips over bones that own no vertex group
    keep = {mod.name, smooth.name}
    others = [(m, m.show_viewport) for m in obj.modifiers if m.name not in keep]  # RNA wrappers: compare by name
    for m, _ in others:
        m.show_viewport = False
    bpy.context.view_layer.update()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
    assert len(evaluated.vertices) == len(obj.data.vertices)
    moved = 0.0
    for v, e in zip(obj.data.vertices, evaluated.vertices):
        moved = max(moved, (v.co - e.co).length)
        v.co = e.co.copy()
    obj.modifiers.remove(smooth)
    obj.modifiers.remove(mod)
    for m, shown in others:
        m.show_viewport = shown
    obj.data.update()
    # drop the imported custom split normals: the moved and the new faces
    # must shade from their geometry (Blender 4.1+ keeps them as a corner
    # attribute; the operator is the fallback)
    attr = obj.data.attributes.get("custom_normal")
    if attr is not None:
        obj.data.attributes.remove(attr)
    if getattr(obj.data, "has_custom_normals", False):
        bpy.context.view_layer.objects.active = obj
        bpy.ops.mesh.customdata_custom_splitnormals_clear()
    print(f"[fix_paw] custom normals left: {getattr(obj.data, 'has_custom_normals', None)}")
    obj.vertex_groups.remove(vg)
    print(f"[fix_paw] shrinkwrap applied, max move {moved * 1000:.0f} mm")
    bpy.data.objects.remove(target, do_unlink=True)
    for p in obj.data.polygons:
        p.use_smooth = True

    valid = set(bpy.ops.export_scene.gltf.get_rna_type().properties.keys())
    kwargs = {"filepath": args.dst, "export_format": "GLB", "export_animations": True,
              "export_animation_mode": "NLA_TRACKS", "export_force_sampling": True, "export_apply": False}
    bpy.ops.export_scene.gltf(**{k: v for k, v in kwargs.items() if k in valid})
    print(f"[fix_paw] wrote {args.dst}")


main()
