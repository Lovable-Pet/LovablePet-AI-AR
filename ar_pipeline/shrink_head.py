"""Blender: shrink a template's head about the neck base, bones included.

WHY
---
The Meshy jindo puppy (2026-10-12) has a head almost twice as wide, relative
to the withers, as the adult jindo mesh. Kim et al. 1989 (249 jindos) put a
2-month puppy's head width at 0.36 of the withers against 0.29 for an adult:
only a quarter bigger, not twice. Scaled to a real puppy's height the model
reads as a big-headed toy (user: "너무 대두로 나옴").

HOW
---
Every vertex is scaled towards a pivot on the midline at the neck base,
by a separate factor per axis (width, length, height: the muzzle must keep
its length while the cranium and the neck ruff lose width), weighted by
its skin weight to the head: the head bone and everything under it count
in full, the neck middle at NECK_MID, the neck base not at all, so the neck
narrows into the smaller head without a step and the front legs (which
also stand in front of the neck base) are untouched. Bone heads and tails
move by the same weights. Doing it in Blender's
edit mode keeps the skin weights, recomputes the inverse bind matrices and
re-samples the (rest-relative) clips on export, so the rig still works. The
GLB is exported the way build_template.py exports it.

Usage:
    blender --background --python-exit-code 1 --python ar_pipeline/shrink_head.py -- \
        output/skeleton_jindo_puppy_small.glb <out.glb> --factor 0.7,0.95,0.75
"""

import argparse
import sys

import bpy
from mathutils import Vector

NECK_MID = 0.6   # share of the shrink the neck middle gets (neck base 0, head chain 1)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--factor", default="0.7,0.95,0.75",
                    help="scale as width,length,height (dog frame), or one number for all three")
    ap.add_argument("--pivot-bone", default="neck_base_jnt")
    ap.add_argument("--blend-bone", default="neck_mid_jnt")
    ap.add_argument("--head-bone", default="head_jnt")
    args = ap.parse_args(argv)
    parts = [float(x) for x in args.factor.split(",")]
    fx, fy, fz = parts if len(parts) == 3 else parts * 3
    # Blender frame after import: x width, y along the body, z up
    factor = Vector((fx, fy, fz))

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=args.src)
    obj = max((o for o in bpy.data.objects if o.type == "MESH"), key=lambda o: len(o.data.vertices))
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    # bone weights: 0 at the neck base, NECK_MID at the neck middle, 1 for
    # the head bone and every bone under it
    bones = arm.data.bones
    head_chain = {args.head_bone} | {b.name for b in bones[args.head_bone].children_recursive}
    bone_w = {b.name: 1.0 if b.name in head_chain else (NECK_MID if b.name == args.blend_bone else 0.0) for b in bones}
    pivot = arm.matrix_world @ bones[args.pivot_bone].head_local
    pivot.x = 0.0
    print(f"[shrink_head] pivot {tuple(round(c, 3) for c in pivot)}, head chain {sorted(head_chain)}")

    def moved(p: Vector, w: float) -> Vector:
        s = Vector((1.0, 1.0, 1.0)) + (factor - Vector((1.0, 1.0, 1.0))) * w
        return pivot + Vector((d * k for d, k in zip(p - pivot, s)))

    # vertices, by their skin weight to the moved bones
    names = {vg.index: vg.name for vg in obj.vertex_groups}
    inv = obj.matrix_world.inverted()
    n = 0
    for v in obj.data.vertices:
        w = min(1.0, sum(bone_w.get(names.get(g.group, ""), 0.0) * g.weight for g in v.groups))
        if w <= 1e-4:
            continue
        v.co = inv @ moved(obj.matrix_world @ v.co, w)
        n += 1
    obj.data.update()
    # drop custom split normals so the squeezed faces shade from their geometry
    attr = obj.data.attributes.get("custom_normal")
    if attr is not None:
        obj.data.attributes.remove(attr)
    print(f"[shrink_head] moved {n} of {len(obj.data.vertices)} vertices")

    # bones: heads and tails in edit mode. A connected child's head is its
    # parent's tail, so the tail takes the child's weight.
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    ainv = arm.matrix_world.inverted()
    nb_moved = []
    for eb in arm.data.edit_bones:
        wh = bone_w[eb.name]
        connected = [c for c in eb.children if c.use_connect]
        wt = bone_w[connected[0].name] if connected else wh
        if wh <= 0 and wt <= 0:
            continue
        eb.head = ainv @ moved(arm.matrix_world @ eb.head, wh)
        eb.tail = ainv @ moved(arm.matrix_world @ eb.tail, wt)
        nb_moved.append(eb.name)
    bpy.ops.object.mode_set(mode="OBJECT")
    print(f"[shrink_head] moved bones: {', '.join(nb_moved)}")
    # pose-bone locations are rest-relative; the clips were force-sampled from
    # a rig whose rest matched, so their location keys are (near) zero and
    # the new rest carries the head. Check that assumption.
    worst = 0.0
    for act in bpy.data.actions:
        curves = list(getattr(act, "fcurves", []))
        for layer in getattr(act, "layers", []):          # Blender 4.4+ layered actions
            for strip in layer.strips:
                for bag in strip.channelbags:
                    curves += list(bag.fcurves)
        for fc in curves:
            if fc.data_path.endswith(".location") and any(f'"{b}"' in fc.data_path for b in nb_moved):
                worst = max(worst, max(abs(k.co[1]) for k in fc.keyframe_points))
    print(f"[shrink_head] largest location key on a moved bone: {worst * 1000:.1f} mm")
    for p in obj.data.polygons:
        p.use_smooth = True

    valid = set(bpy.ops.export_scene.gltf.get_rna_type().properties.keys())
    kwargs = {"filepath": args.dst, "export_format": "GLB", "export_animations": True,
              "export_animation_mode": "NLA_TRACKS", "export_force_sampling": True, "export_apply": False}
    bpy.ops.export_scene.gltf(**{k: v for k, v in kwargs.items() if k in valid})
    print(f"[shrink_head] wrote {args.dst}")


main()
