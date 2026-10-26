"""Stage 0 for an UNRIGGED model: normalise, re-unwrap, re-bake, auto-rig.

The template stages after this one (build_template.py, seed_regions.py,
refine_masks.py, build_landmarks.py) all read the rig: bone weights seed the
region masks, bone chains carve the body, joints place the landmarks. A
generated model (Meshy, Tripo) has no rig, so this script builds one with
the SAME bone names as the shiba source rig - pelvis_jnt, spine_1_jnt,
L_shoulder_jnt, LF_toes_jnt, tail_1_jnt..tail_7_jnt, L_ear_base_jnt,
L_eye_jnt, nose_jnt, breathe_jnt - so every later stage runs unchanged.

What it does, in order:

  1. NORMALISE  import, apply transforms, turn the head to -Y (the template
                convention: -Y forward, +Z up), paws on z = 0, centred on x,
                scaled to --height metres (shiba template: 0.83).
  2. UNWRAP     a Meshy atlas is ~2000 confetti islands; every UV-space
                step (feathering, the image model's fur pass, seam
                post-processing) needs coherent islands. Smart UV Project
                made it worse (4000 islands: the generated surface is bumpy
                and the angle limit cuts almost every face), so the seams
                come from the RIG instead: the boundary between two bone
                regions (head, ear, neck, back, each leg, tail) plus the
                midline plane, and for each leg the plane through its own
                centre - one or two flattenable patches per body part, a
                few dozen islands in all. Then an angle-based unwrap. The
                old map is kept as UVMap_old until the bake is done.
  3. BAKE       the model's own base colour and normal map are baked from
                the old UV onto the new one (Cycles EMIT bake), and the
                material rebuilt on the baked images.
  4. JOINTS     joint positions estimated from the vertices alone: paw
                clusters, leg columns sampled at fractions of the height,
                a spine through slice centroids, the nose as the foremost
                point, ears as what rises above the crown, the tail by
                edge-graph distance from its root.
  5. RIG        an armature in the shiba hierarchy, automatic (heat)
                weights, envelope fallback.
  6. EXPORT     GLB with skin (no clips: build_template.py adds idle and
                tail_wag), a .blend for hand fixes, joints.json, and a
                debug render with a sphere on every joint.

Usage:
    blender --background --python ar_pipeline/auto_rig.py -- \
        work/new_dog/model.glb work/new_dog/jindo_rigged.glb \
        [--height 0.83] [--no-unwrap] [--debug-dir work/new_dog]

The joint estimate is a draft: check the debug render, and fix any joint
in --joints-override JSON ({"L_eye_jnt": [x, y, z], ...}) or in the .blend.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import deque

import bpy
import numpy as np
from mathutils import Vector

TEXTURE_SIZE = 2048
UNWRAP_ANGLE_DEG = 70.0
UNWRAP_MARGIN = 0.006
BAKE_MARGIN_PX = 12
# a folded patch this many faces or larger is re-unwrapped by projection (a paw is ~1,000 faces)
MIN_FOLD_FACES = 200


# --------------------------------------------------------------------------- helpers

def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--height", type=float, default=0.83, help="target height in metres (shiba template: 0.83)")
    p.add_argument("--no-unwrap", action="store_true", help="keep the model's UV and textures")
    p.add_argument("--debug-dir", default=None)
    p.add_argument("--joints-override", default=None, help="JSON of joint name -> [x, y, z] to replace estimates")
    return p.parse_args(argv)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def main_mesh():
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    meshes.sort(key=lambda o: len(o.data.polygons), reverse=True)
    if not meshes:
        raise SystemExit("[auto_rig] no mesh in the input")
    if len(meshes) > 1:
        print(f"[auto_rig] NOTE: {len(meshes)} meshes; joining into '{meshes[0].name}'")
        bpy.ops.object.select_all(action="DESELECT")
        for m in meshes:
            m.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()
    return meshes[0]


def vertices_world(obj) -> np.ndarray:
    mw = np.array(obj.matrix_world)
    co = np.array([v.co[:] for v in obj.data.vertices], dtype=np.float64)
    return (co @ mw[:3, :3].T) + mw[:3, 3]


def apply_transforms(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


# --------------------------------------------------------------------------- 1. normalise

def normalise(obj, height: float) -> np.ndarray:
    # unparent from any import empties so the transforms are the mesh's own
    if obj.parent is not None:
        mw = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = mw
    for o in list(bpy.data.objects):
        if o.type == "EMPTY":
            bpy.data.objects.remove(o)
    apply_transforms(obj)
    # a generated mesh splits its vertices along every island of its own
    # atlas (33k vertices on 21k positions): welded, or the unwrap treats
    # each of those 2000 pieces as an island whatever the seams say. UVs
    # live on face corners, so the old map survives for the bake.
    before = len(obj.data.vertices)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=1e-5)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    print(f"[auto_rig] welded {before} -> {len(obj.data.vertices)} vertices")
    v = vertices_world(obj)
    # head end: the highest points (ear tips) sit at the head end
    top = v[v[:, 2] > v[:, 2].min() + 0.97 * (v[:, 2].max() - v[:, 2].min())]
    if top[:, 1].mean() > v[:, 1].mean():
        obj.rotation_euler = (0.0, 0.0, math.pi)
        apply_transforms(obj)
        v = vertices_world(obj)
        print("[auto_rig] turned the model 180 deg so the head faces -Y")
    scale = height / (v[:, 2].max() - v[:, 2].min())
    obj.scale = (scale, scale, scale)
    apply_transforms(obj)
    v = vertices_world(obj)
    centre = np.array([(v[:, 0].max() + v[:, 0].min()) / 2, (v[:, 1].max() + v[:, 1].min()) / 2, v[:, 2].min()])
    obj.location = tuple(-centre)
    apply_transforms(obj)
    v = vertices_world(obj)
    print(f"[auto_rig] normalised: scale x{scale:.3f}, extents {np.round(v.max(0) - v.min(0), 3)}, "
          f"y {v[:, 1].min():.3f}..{v[:, 1].max():.3f}")
    return v


# --------------------------------------------------------------------------- 2-3. unwrap + bake

def find_material_images(obj):
    """(base colour image, normal image) of the imported material, either may be None."""
    base = normal = None
    # by name first: a glTF import may route the base colour through mix or
    # separate nodes, and Meshy names its images texture_0 / normal
    for image in bpy.data.images:
        lname = image.name.lower()
        if "normal" in lname and normal is None:
            normal = image
        elif "metallic" not in lname and "roughness" not in lname and "render result" not in lname                 and "viewer" not in lname and base is None:
            base = image
    for slot in obj.material_slots:
        mat = slot.material
        if mat is None or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type != "TEX_IMAGE" or node.image is None:
                continue
            for link in mat.node_tree.links:
                if link.from_node is node:
                    to = link.to_node
                    if to.type == "BSDF_PRINCIPLED" and link.to_socket.name == "Base Color":
                        base = node.image
                    elif to.type == "NORMAL_MAP":
                        normal = node.image
    return base, normal


# Bone-name fragment -> seam region, the coarse body parts an unwrap cuts
# between. Order matters: the first fragment found wins.
SEAM_PARTS = [
    ("ear", "ear"), ("eye", "head"), ("nose", "head"), ("jaw", "head"), ("head", "head"),
    ("neck", "neck"), ("tail", "tail"),
    ("clavicle", "body"), ("breathe", "body"), ("central", "body"), ("spine", "body"), ("pelvis", "body"),
    ("chest", "body"), ("COG", "body"), ("root", "body"),
    ("LF_", "leg_LF"), ("RF_", "leg_RF"), ("LB_", "leg_LB"), ("RB_", "leg_RB"),
    ("L_shoulder", "leg_LF"), ("L_elbow", "leg_LF"), ("L_wrist", "leg_LF"), ("L_hand", "leg_LF"),
    ("R_shoulder", "leg_RF"), ("R_elbow", "leg_RF"), ("R_wrist", "leg_RF"), ("R_hand", "leg_RF"),
    ("L_hip", "leg_LB"), ("L_knee", "leg_LB"), ("L_ankle", "leg_LB"), ("L_foot", "leg_LB"),
    ("R_hip", "leg_RB"), ("R_knee", "leg_RB"), ("R_ankle", "leg_RB"), ("R_foot", "leg_RB"),
]
SMOOTH_PASSES = 3


def seam_part(bone: str) -> str:
    for fragment, part in SEAM_PARTS:
        if fragment.lower() in bone.lower():
            return part
    return "body"


def seams_from_rig(obj, v: np.ndarray, dominant: list[str], joints: dict[str, list[float]]) -> int:
    """Mark seams: between body parts, along the midline, and through each
    leg's own centre plane. Returns the number of seam edges."""
    parts = np.array([seam_part(b) for b in dominant])
    names = sorted(set(parts.tolist()))
    ids = np.array([names.index(x) for x in parts])
    edges = np.array([e.vertices[:] for e in obj.data.edges], dtype=np.int64)
    # a few passes of neighbour majority so the part boundary is a line, not
    # a fringe of single-vertex islands
    for _ in range(SMOOTH_PASSES):
        votes = np.zeros((len(v), len(names)), dtype=np.int32)
        np.add.at(votes, (edges[:, 0], ids[edges[:, 1]]), 1)
        np.add.at(votes, (edges[:, 1], ids[edges[:, 0]]), 1)
        has = votes.sum(axis=1) > 0
        ids = np.where(has, np.argmax(votes, axis=1), ids)
    # the cut plane: the midline for the body, the leg's own column for a leg.
    # The trunk and neck are also cut at the spine's height into a back and
    # a belly shell: as two half-tubes they unwrapped stretched two to one
    # along the flank and the fur strokes smeared on the mesh.
    cut = np.sign(v[:, 0] + 1e-9)
    for part in ("body", "neck"):
        if part in names:
            sel = ids == names.index(part)
            # the horizontal cut sits LOW on the flank, inside the belly's
            # colour: at mid-height it ran along the belly/flank colour
            # boundary, and the refinement cannot blend across a UV seam,
            # so the boundary came out as a hard line with a mark at the seam
            z_cut = float(np.percentile(v[sel, 2], 28))
            cut[sel] = 2 * np.sign(v[sel, 0] + 1e-9) + np.sign(v[sel, 2] - z_cut + 1e-9)
    for part, upper in (("leg_LF", "L_shoulder_jnt"), ("leg_RF", "R_shoulder_jnt"),
                        ("leg_LB", "L_hip_jnt"), ("leg_RB", "R_hip_jnt")):
        if part in names and upper in joints:
            sel = ids == names.index(part)
            cut[sel] = np.sign(v[sel, 0] - joints[upper][0] + 1e-9)
    # the cut key gets the same neighbour-majority passes as the part id:
    # on a bumpy mesh single triangles straddle the cut plane and came out
    # as hundreds of one-face islands, which the image model later painted
    # as grey shards all over the coat
    key = ids * 16 + (cut + 4).astype(int)
    keys = np.unique(key)
    kidx = np.searchsorted(keys, key)
    for _ in range(SMOOTH_PASSES + 2):
        votes = np.zeros((len(v), len(keys)), dtype=np.int32)
        np.add.at(votes, (edges[:, 0], kidx[edges[:, 1]]), 1)
        np.add.at(votes, (edges[:, 1], kidx[edges[:, 0]]), 1)
        has = votes.sum(axis=1) > 0
        kidx = np.where(has, np.argmax(votes, axis=1), kidx)
    key = keys[kidx]
    ids = key // 16
    cut = (key % 16) - 4
    a, b = edges[:, 0], edges[:, 1]
    seam = (ids[a] != ids[b]) | (cut[a] != cut[b])
    for e, flag in zip(obj.data.edges, seam):
        e.use_seam = bool(flag)
    counts = {names[k]: int((ids == k).sum()) for k in range(len(names))}
    print(f"[auto_rig] seams: {int(seam.sum())} edges; parts {counts}")
    return int(seam.sum())


def reunwrap(obj, v: np.ndarray, dominant: list[str], joints: dict[str, list[float]]):
    mesh = obj.data
    old = mesh.uv_layers.active
    old.name = "UVMap_old"
    new = mesh.uv_layers.new(name="UVMap")
    mesh.uv_layers.active = new
    new.active_render = True
    seams_from_rig(obj, v, dominant, joints)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    # angle-based. Minimum-stretch (SLIM) failed to solve one body half on
    # the Meshy mesh and collapsed it into a disc; the stretch angle-based
    # leaves is kept down by cutting the trunk into quarters (seams_from_rig)
    bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=UNWRAP_MARGIN)
    # an island the solver cannot flatten (both forepaws on the Meshy jindo:
    # toes with self-intersections) comes back folded to a point, UV area
    # zero, and the whole paw then samples one texel. Those faces are
    # re-unwrapped by projection, which never fails, and packed with the rest.
    import bmesh
    bm = bmesh.from_edit_mesh(mesh)
    uv_layer = bm.loops.layers.uv.active
    bm.faces.ensure_lookup_table()
    # islands = faces joined across non-seam edges, manifold or not (the
    # forepaws' toes are non-manifold; a walk that skipped those edges fell
    # them apart into singletons and the collapsed islands went unnoticed)
    parent = list(range(len(bm.faces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for edge in bm.edges:
        if edge.seam:
            continue
        faces = [f.index for f in edge.link_faces]
        for other in faces[1:]:
            ra, rb = find(faces[0]), find(other)
            if ra != rb:
                parent[ra] = rb
    uv_areas = np.zeros(len(bm.faces)); areas_3d = np.zeros(len(bm.faces)); roots = np.zeros(len(bm.faces), dtype=np.int64)
    for face in bm.faces:
        pts = [loop[uv_layer].uv for loop in face.loops]
        a = 0.0
        for i in range(1, len(pts) - 1):
            a += 0.5 * abs((pts[i].x - pts[0].x) * (pts[i + 1].y - pts[0].y) - (pts[i + 1].x - pts[0].x) * (pts[i].y - pts[0].y))
        uv_areas[face.index] = a
        areas_3d[face.index] = face.calc_area()
        roots[face.index] = find(face.index)
    density = uv_areas.sum() / max(areas_3d.sum(), 1e-9)
    island_uv = {}; island_3d = {}
    for r, a_uv, a_3d in zip(roots, uv_areas, areas_3d):
        island_uv[r] = island_uv.get(r, 0.0) + a_uv
        island_3d[r] = island_3d.get(r, 0.0) + a_3d
    # the solver folds PART of an island (the paw at the end of a leg
    # island, whose toes are non-manifold) to a point: those faces have no
    # UV area while the island as a whole looks fine. Flag faces under
    # 0.2% of their expected area, cluster them across shared edges, and
    # re-unwrap only clusters of at least MIN_FOLD_FACES by projection at
    # the widest angle - single squeezed faces in the chest ruff and the
    # tail are left where the solver put them (re-projecting each one
    # scattered 8,000 confetti islands over the coat)
    expected = areas_3d * density
    flagged = (expected > 1e-9) & (uv_areas < 0.002 * expected)
    fparent = list(range(len(bm.faces)))

    def ffind(i):
        while fparent[i] != i:
            fparent[i] = fparent[fparent[i]]
            i = fparent[i]
        return i

    for edge in bm.edges:
        faces = [f.index for f in edge.link_faces if flagged[f.index]]
        for other in faces[1:]:
            ra, rb = ffind(faces[0]), ffind(other)
            if ra != rb:
                fparent[ra] = rb
    clusters: dict[int, list[int]] = {}
    for i in np.nonzero(flagged)[0]:
        clusters.setdefault(ffind(int(i)), []).append(int(i))
    folded = [c for c in clusters.values() if len(c) >= MIN_FOLD_FACES]
    if folded:
        chosen = {i for c in folded for i in c}
        for face in bm.faces:
            face.select = face.index in chosen
        bmesh.update_edit_mesh(mesh)
        bpy.ops.uv.smart_project(angle_limit=math.radians(89), island_margin=UNWRAP_MARGIN, correct_aspect=True,
                                 scale_to_bounds=False)
        print(f"[auto_rig] {len(folded)} folded patch(es) of {[len(c) for c in folded]} faces "
              f"({int(flagged.sum())} faces flagged in all): re-unwrapped by projection")
        bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.select_all(action="SELECT")
    # one texel density for every island: the unwrap leaves each island at
    # its own scale (one foreleg came out 28% larger than the other), and
    # with it the fur strokes, the feather widths and the eye/nose blobs
    # differ from island to island
    try:
        bpy.ops.uv.average_islands_scale()
    except RuntimeError as exc:
        print(f"[auto_rig] average_islands_scale skipped ({exc})")
    try:
        bpy.ops.uv.pack_islands(rotate=True, margin=UNWRAP_MARGIN)
    except RuntimeError as exc:
        print(f"[auto_rig] pack_islands skipped ({exc})")
    bpy.ops.object.mode_set(mode="OBJECT")
    # count islands: faces connected through shared UV coordinates
    uv = mesh.uv_layers["UVMap"].data
    key_of_loop = {}
    parent = list(range(len(mesh.polygons)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for poly in mesh.polygons:
        for li in poly.loop_indices:
            vi = mesh.loops[li].vertex_index
            key = (vi, round(uv[li].uv[0], 5), round(uv[li].uv[1], 5))
            if key in key_of_loop:
                ra, rb = find(key_of_loop[key]), find(poly.index)
                if ra != rb:
                    parent[ra] = rb
            else:
                key_of_loop[key] = poly.index
    roots = {}
    for i in range(len(mesh.polygons)):
        roots[find(i)] = roots.get(find(i), 0) + 1
    sizes = sorted(roots.values(), reverse=True)
    print(f"[auto_rig] re-unwrapped: {len(sizes)} islands, largest {sizes[:6]} faces, "
          f"{sum(1 for x in sizes if x < 20)} under 20 faces")


def bake_image(obj, source_image, name: str, non_color: bool) -> bpy.types.Image:
    """Bake `source_image` (sampled through UVMap_old) into a new image on UVMap."""
    mesh = obj.data
    target = bpy.data.images.new(name, TEXTURE_SIZE, TEXTURE_SIZE, alpha=False)
    target.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    if non_color:
        source_image.colorspace_settings.name = "Non-Color"
    mat = bpy.data.materials.new(f"_bake_{name}")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    emit = nodes.new("ShaderNodeEmission")
    src = nodes.new("ShaderNodeTexImage")
    src.image = source_image
    src.interpolation = "Linear"
    uvn = nodes.new("ShaderNodeUVMap")
    uvn.uv_map = "UVMap_old"
    dst = nodes.new("ShaderNodeTexImage")
    dst.image = target
    links.new(uvn.outputs["UV"], src.inputs["Vector"])
    links.new(src.outputs["Color"], emit.inputs["Color"])
    links.new(emit.outputs["Emission"], out.inputs["Surface"])
    nodes.active = dst
    dst.select = True
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    mesh.uv_layers["UVMap"].active = True
    mesh.uv_layers["UVMap"].active_render = True

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 4
    scene.render.bake.use_clear = True
    scene.render.bake.margin = BAKE_MARGIN_PX
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.bake(type="EMIT", uv_layer="UVMap")
    print(f"[auto_rig] baked {name} ({TEXTURE_SIZE}px)")
    return target


def rebuild_material(obj, base_image, normal_image):
    mat = bpy.data.materials.new("coat")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.85
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = base_image
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    if normal_image is not None:
        ntex = nodes.new("ShaderNodeTexImage")
        ntex.image = normal_image
        nmap = nodes.new("ShaderNodeNormalMap")
        nmap.uv_map = "UVMap"
        links.new(ntex.outputs["Color"], nmap.inputs["Color"])
        links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    for image in (base_image, normal_image):
        if image is not None:
            image.pack()


# --------------------------------------------------------------------------- 4. joints

def slice_centroid(v: np.ndarray, sel: np.ndarray) -> np.ndarray | None:
    return v[sel].mean(axis=0) if sel.sum() >= 3 else None


def local_thickness(obj, v: np.ndarray, max_depth: float) -> np.ndarray:
    """How thick the body is under every vertex: a ray cast inward along the
    vertex normal, the distance to the far wall. A tail or an ear is thin,
    the torso is not - the one measure that tells them apart without a rig."""
    from mathutils.bvhtree import BVHTree

    depsgraph = bpy.context.evaluated_depsgraph_get()
    tree = BVHTree.FromObject(obj, depsgraph)
    mw = obj.matrix_world
    inv = mw.inverted()
    out = np.full(len(v), max_depth, dtype=np.float64)
    for i, vert in enumerate(obj.data.vertices):
        n = (mw.to_3x3() @ vert.normal).normalized()
        # the normals of a generated mesh may point either way: cast both
        # ways and take the nearer far wall
        best = max_depth
        for sign in (-1.0, 1.0):
            direction = (inv.to_3x3() @ (n * sign)).normalized()
            origin = inv @ (Vector(v[i]) + n * sign * 0.002)
            hit, _, _, dist = tree.ray_cast(origin, direction, max_depth)
            if hit is not None and dist is not None and 0.004 < dist < best:
                best = float(dist)
        out[i] = best
    thin = float((out < 0.1 * max_depth).mean())
    print(f"[auto_rig] thickness: median {np.median(out):.3f} m, thin share {thin:.2f}")
    return out


def vertex_darkness(obj, image) -> np.ndarray | None:
    """Per vertex, the darkest luma the model's own base colour texture puts
    on it (sampled through the face-corner UVs). The eyes and the nose
    leather are the dark spots on a head; no rig needed to find them."""
    if image is None:
        return None
    w, h = image.size
    px = np.array(image.pixels[:], dtype=np.float32).reshape(h, w, -1)[..., :3]
    luma = px @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    mesh = obj.data
    uv = mesh.uv_layers.active.data
    out = np.ones(len(mesh.vertices), dtype=np.float32)
    for poly in mesh.polygons:
        for li in poly.loop_indices:
            u, vv = uv[li].uv
            x = min(w - 1, max(0, int(u * w)))
            y = min(h - 1, max(0, int(vv * h)))
            vi = mesh.loops[li].vertex_index
            if luma[y, x] < out[vi]:
                out[vi] = luma[y, x]
    return out


DARK_LUMA = 0.22


def estimate_joints(v: np.ndarray, edges: np.ndarray, thick: np.ndarray,
                    darkness: np.ndarray | None = None) -> dict[str, list[float]]:
    H = float(v[:, 2].max() - v[:, 2].min())
    J: dict[str, np.ndarray] = {}

    # --- paws: the lowest vertices, split into four by side and by front/back
    low = v[v[:, 2] < 0.05 * H]
    # four clusters on the ground plane: greedy merge within a paw's width.
    # A median split on y put the hind toes on the ground in front of the
    # hind paws when the model's fore and hind paws were not the same
    # distance apart as its median suggested.
    clusters: list[list[np.ndarray]] = []
    sums: list[np.ndarray] = []
    for pt in low:
        for c, total in zip(clusters, sums):
            if np.linalg.norm(total[:2] / len(c) - pt[:2]) < 0.07 * H:
                c.append(pt)
                total += pt
                break
        else:
            clusters.append([pt])
            sums.append(pt.copy())
    clusters.sort(key=len, reverse=True)
    if len(clusters) < 4:
        raise SystemExit(f"[auto_rig] found {len(clusters)} paw clusters among the lowest vertices, need 4")
    centres = [np.mean(c, axis=0) for c in clusters[:4]]
    front = sorted(centres, key=lambda c: c[1])[:2]
    back = sorted(centres, key=lambda c: c[1])[2:]
    paws = {}
    for tag, group, xs in (("LF", front, 1), ("RF", front, -1), ("LB", back, 1), ("RB", back, -1)):
        paws[tag] = max(group, key=lambda c: c[0] * xs).copy()
        paws[tag][2] = 0.0
    print(f"[auto_rig] paws: {[(t, np.round(p, 3).tolist()) for t, p in paws.items()]}")
    # L = +x (the animal's own left when it faces -Y), as in the shiba rig
    side_of = {"LF": "L", "RF": "R", "LB": "L", "RB": "R"}

    # --- legs: climb the leg column, re-centring at every band
    def column(start_xy, z, radius, prev=None):
        band = (np.abs(v[:, 2] - z) < 0.02 * H)
        ref = start_xy if prev is None else prev[:2]
        near = band & (np.linalg.norm(v[:, :2] - ref, axis=1) < radius)
        c = slice_centroid(v, near)
        return c if c is not None else np.array([ref[0], ref[1], z])

    for tag, paw in paws.items():
        side = side_of[tag]
        front = tag[1] == "F"
        toes = paw.copy()
        toes[2] = 0.02 * H
        J[f"{tag}_toes_jnt"] = toes
        hand = column(paw[:2], 0.07 * H, 0.09 * H)
        J[f"{side}_{'hand' if front else 'foot'}_jnt"] = hand
        wrist = column(paw[:2], 0.22 * H, 0.09 * H, hand)
        J[f"{side}_{'wrist' if front else 'ankle'}_jnt"] = wrist
        elbow = column(paw[:2], 0.45 * H, 0.10 * H, wrist)
        J[f"{side}_{'elbow' if front else 'knee'}_jnt"] = elbow
        shoulder = column(paw[:2], 0.64 * H, 0.12 * H, elbow)
        # the upper joint sits where the leg meets the body: the band there is
        # the torso, whose centroid is the midline, so keep the paw's side
        shoulder[0] = np.sign(paw[0]) * max(abs(shoulder[0]), 0.8 * abs(paw[0]))
        J[f"{side}_{'shoulder' if front else 'hip'}_jnt"] = shoulder
        if front:
            clav = shoulder.copy()
            clav[0] *= 0.4
            clav[2] += 0.08 * H
            J[f"{side}_clavicle_jnt"] = clav

    y_front = float(np.mean([J["L_shoulder_jnt"][1], J["R_shoulder_jnt"][1]]))
    y_hind = float(np.mean([J["L_hip_jnt"][1], J["R_hip_jnt"][1]]))

    # --- torso: belly and back heights between the legs
    # between the legs (not over them) and near the midline, so the belly
    # is the belly and not a paw
    torso = (v[:, 1] > y_front + 0.1 * H) & (v[:, 1] < y_hind - 0.1 * H) & (np.abs(v[:, 0]) < 0.05 * H)
    z_belly = float(np.percentile(v[torso][:, 2], 3))
    z_back = float(np.percentile(v[torso][:, 2], 97))
    print(f"[auto_rig] torso: y {y_front:.3f}..{y_hind:.3f}, belly z {z_belly:.3f}, back z {z_back:.3f}, H {H:.3f}")

    def spine_point(y, lift=0.55):
        sel = (np.abs(v[:, 1] - y) < 0.03 * H) & (v[:, 2] > z_belly - 0.02) & (v[:, 2] < z_back + 0.02)
        c = slice_centroid(v, sel)
        if c is None:
            return np.array([0.0, y, (z_belly + z_back) / 2])
        return np.array([0.0, y, z_belly + lift * (z_back - z_belly)])

    J["pelvis_jnt"] = spine_point(y_hind)
    J["spine_1_jnt"] = spine_point(y_hind + (y_front - y_hind) * 0.35)
    J["spine_2_jnt"] = spine_point(y_hind + (y_front - y_hind) * 0.7)
    J["chest_jnt"] = spine_point(y_front)
    J["COG_jnt"] = J["spine_1_jnt"].copy()
    J["root"] = np.array([0.0, J["spine_1_jnt"][1], 0.0])
    J["breathe_jnt"] = np.array([0.0, y_hind + (y_front - y_hind) * 0.6, z_belly + 0.35 * (z_back - z_belly)])
    J["central_corrective"] = J["breathe_jnt"].copy()

    # --- head: the nose is the foremost point at head height
    headish = v[(v[:, 2] > 0.5 * H)]
    nose = headish[np.argmin(headish[:, 1])]
    J["nose_jnt"] = nose + np.array([0.0, 0.015, 0.0])
    head_sel = (v[:, 1] < nose[1] + 0.30 * H) & (v[:, 2] > 0.55 * H)
    head_c = v[head_sel].mean(axis=0)
    head_c[0] = 0.0
    # crown line at the centre: what rises above it on either side is ear
    centre_head = head_sel & (np.abs(v[:, 0]) < 0.04 * H)
    z_crown = float(np.percentile(v[centre_head][:, 2], 92))
    ears = {}
    for side, xs in (("L", 1), ("R", -1)):
        sel = head_sel & (np.sign(v[:, 0]) == xs) & (v[:, 2] > z_crown + 0.01 * H)
        if sel.sum() < 10:
            print(f"[auto_rig] WARNING: no {side} ear found above the crown; placing a guess")
            base = np.array([xs * 0.06 * H, head_c[1] + 0.02 * H, z_crown])
            tip = base + np.array([xs * 0.01 * H, 0.0, 0.12 * H])
        else:
            ear = v[sel]
            bottom = ear[ear[:, 2] < ear[:, 2].min() + 0.03 * H]
            top = ear[ear[:, 2] > np.percentile(ear[:, 2], 90)]
            base, tip = bottom.mean(axis=0), top.mean(axis=0)
        ears[side] = (base, tip)
        J[f"{side}_ear_base_jnt"] = base
        J[f"{side}_ear_tip_jnt"] = tip
    y_ear = float(np.mean([ears["L"][0][1], ears["R"][0][1]]))
    head_len = y_ear - nose[1]
    J["head_jnt"] = np.array([0.0, y_ear - 0.15 * head_len, z_crown - 0.12 * head_len])
    # the jaw hangs UNDER the muzzle, halfway from the nose to the ear base;
    # placed at eye level it took the muzzle top from the nose and head
    # bones, and seed_regions then read the muzzle as throat
    J["jaw_jnt"] = np.array([0.0, nose[1] + 0.45 * head_len, nose[2] - 0.06 * H])
    # eyes: the two dark spots on the head, one each side, that are not the
    # nose (the nose is the most forward dark spot); else a guess from the
    # head's proportions, snapped to the surface
    dark_eyes = {}
    if darkness is not None:
        dark = head_sel & (darkness < DARK_LUMA)
        uniq_d, inv_d = np.unique(np.round(v, 5), axis=0, return_inverse=True)
        inv_d = np.asarray(inv_d).reshape(-1)
        dark_r = np.zeros(len(uniq_d), dtype=bool)
        dark_r[inv_d[dark]] = True
        er_d = inv_d[edges]
        keep = dark_r[er_d[:, 0]] & dark_r[er_d[:, 1]]
        parent = np.arange(len(uniq_d))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for a, b in er_d[keep]:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        roots = np.array([find(i) if dark_r[i] else -1 for i in range(len(uniq_d))])
        blobs = []
        for r in np.unique(roots[roots >= 0]):
            sel = roots == r
            if sel.sum() >= 8:
                blobs.append((int(sel.sum()), uniq_d[sel].mean(axis=0)))
        blobs.sort(key=lambda b: -b[0])
        if blobs:
            nose_blob = min(blobs, key=lambda b: b[1][1])
            J["nose_jnt"] = nose_blob[1] + np.array([0.0, 0.01, 0.0])
            # the eyes: a left/right pair at the same height and depth, above
            # the nose, and the most FORWARD such pair - the ear cups are a
            # dark pair too, and larger, but they sit further back
            pairs = []
            for a in blobs:
                for b in blobs:
                    if a is nose_blob or b is nose_blob or a[1][0] <= 0 or b[1][0] >= 0:
                        continue
                    if abs(a[1][1] - b[1][1]) > 0.03 * H or abs(a[1][2] - b[1][2]) > 0.03 * H:
                        continue
                    if abs(abs(a[1][0]) - abs(b[1][0])) > 0.03 * H:
                        continue
                    ymid = (a[1][1] + b[1][1]) / 2
                    if ymid - nose_blob[1][1] < 0.03 * H or (a[1][2] + b[1][2]) / 2 < nose_blob[1][2]:
                        continue
                    pairs.append((ymid, a, b))
            if pairs:
                _, left, right = min(pairs, key=lambda t: t[0])
                dark_eyes["L"], dark_eyes["R"] = left[1], right[1]
        print(f"[auto_rig] dark spots on the head: {[(n, np.round(c, 3).tolist()) for n, c in blobs[:6]]}; "
              f"eyes from texture: {sorted(dark_eyes)}")
    for side, xs in (("L", 1), ("R", -1)):
        if side in dark_eyes:
            J[f"{side}_eye_jnt"] = dark_eyes[side]
            continue
        guess = np.array([0.0, nose[1] + 0.62 * head_len, z_crown - 0.22 * head_len])
        cand = head_sel & (np.sign(v[:, 0]) == xs) & (np.abs(v[:, 1] - guess[1]) < 0.06 * H) \
            & (np.abs(v[:, 2] - guess[2]) < 0.06 * H)
        if cand.sum() >= 3:
            pts = v[cand]
            # the eye sits where the face is widest at that height: outboard verts
            outer = pts[np.abs(pts[:, 0]) > np.percentile(np.abs(pts[:, 0]), 70)]
            guess = outer.mean(axis=0)
        else:
            guess[0] = xs * 0.35 * head_len
        J[f"{side}_eye_jnt"] = guess

    # --- neck
    J["neck_base_jnt"] = np.array([0.0, y_front - 0.06 * H, z_back + 0.02 * H])
    J["neck_mid_jnt"] = (J["neck_base_jnt"] + J["head_jnt"]) / 2

    # --- tail. Thickness cannot separate a curled tail from the rump it
    #     lies against (tried: the thin set was confetti), so the tail is
    #     read as the lobe of the side profile behind and above the hips: a
    #     circle fitted to that lobe's rear and top extremes, and seven
    #     joints along its rim from the root on the rump round to the tip.
    #     A hanging tail shows as a lobe below the rump instead and gets the
    #     same treatment mirrored downward.
    # the back's height profile along y near the midline, from mid-back to
    # the rear end: a curl lying over the rump lifts the profile above the
    # croup, and where that lift starts is the front of the curl
    y_end = float(v[:, 1].max())
    bins = np.arange(y_hind - 0.2 * H, y_end + 0.02 * H, 0.02 * H)
    mid = np.abs(v[:, 0]) < 0.06 * H
    profile = []
    for y0 in bins:
        sel = mid & (v[:, 1] >= y0) & (v[:, 1] < y0 + 0.02 * H)
        profile.append(float(np.percentile(v[sel][:, 2], 97)) if sel.sum() >= 5 else np.nan)
    profile = np.array(profile)
    croup_i = int(np.argmin(np.abs(bins - (y_hind - 0.12 * H))))
    z_croup = float(np.nanmin(profile[max(0, croup_i - 2):croup_i + 3]))
    lifted = np.nonzero(np.nan_to_num(profile, nan=-1) > z_croup + 0.02 * H)[0]
    lifted = lifted[bins[lifted] > y_hind - 0.12 * H]
    rear = (v[:, 1] > y_hind + 0.02 * H) & (v[:, 2] > 0.45 * H)
    lobe = v[rear]
    z_rump = z_croup
    print(f"[auto_rig] tail: back profile z {np.round(profile, 3).tolist()}, croup {z_croup:.3f}, "
          f"lifted bins {lifted.size}")
    if lobe.shape[0] < 50 or lifted.size < 2:
        # nothing lifts the back line: a hanging tail, the lobe below the rump
        rear = (v[:, 1] > y_hind + 0.06 * H) & (v[:, 2] < 0.6 * H) & (v[:, 2] > 0.25 * H)
        lobe = v[rear]
        x_c = float(np.median(lobe[:, 0])) if lobe.shape[0] else 0.0
        base = np.array([x_c, y_hind + 0.06 * H, z_rump - 0.02 * H])
        tip = np.array([x_c, lobe[:, 1].max() if lobe.shape[0] else base[1] + 0.2,
                        lobe[:, 2].min() if lobe.shape[0] else base[2] - 0.3])
        for k in range(7):
            J[f"tail_{k + 1}_jnt"] = base + (tip - base) * k / 6
        print("[auto_rig] tail: hanging, straight chain")
    else:
        y_front_curl = float(bins[lifted[0]])
        curl = v[(v[:, 1] > y_front_curl) & (v[:, 2] > z_croup - 0.02 * H)]
        x_c = float(np.median(curl[:, 0]))
        y_max, z_max = float(curl[:, 1].max()), float(curl[:, 2].max())
        z_low = z_croup - 0.01 * H
        y_at_top = float(curl[np.argmax(curl[:, 2]), 1])
        r_guess = 0.5 * ((y_max - y_front_curl) + (z_max - z_low)) / 2
        if y_max - y_at_top < 0.6 * r_guess:
            # the highest point is also (nearly) the rearmost: a raised
            # straight tail, not a curl. Joints along the lobe's own axis
            # from the root on the rump to the tip, centroids of slices.
            root = np.array([x_c, y_hind + 0.05 * H, z_croup - 0.01 * H])
            tip = curl[np.argmax((curl[:, 1] - root[1]) + (curl[:, 2] - root[2]))]
            axis = tip - root
            t = ((curl - root) @ axis) / max(float(axis @ axis), 1e-9)
            J["tail_1_jnt"] = root
            for k in range(1, 7):
                lo, hi = (k - 0.5) / 6, (k + 0.5) / 6
                sel = (t >= lo) & (t < hi)
                J[f"tail_{k + 1}_jnt"] = curl[sel].mean(axis=0) if sel.sum() >= 3 else root + axis * (k / 6)
            print(f"[auto_rig] tail: straight, {curl.shape[0]} vertices, root {np.round(root, 3)}, tip {np.round(tip, 3)}")
            return {k: [round(float(x), 4) for x in p] for k, p in J.items()}
        centre = np.array([x_c, (y_front_curl + y_max) / 2, (z_low + z_max) / 2])
        r = 0.5 * ((y_max - y_front_curl) + (z_max - z_low)) / 2
        rim = 0.72 * r  # the joints run inside the tail, not on its skin
        angles = np.radians(np.linspace(-65, 195, 7))  # 0 = rear (+y), 90 = up (+z)
        for k, a in enumerate(angles):
            J[f"tail_{k + 1}_jnt"] = centre + np.array([0.0, rim * math.cos(a), rim * math.sin(a)])
        J["tail_1_jnt"][2] = min(J["tail_1_jnt"][2], z_rump)
        print(f"[auto_rig] tail: curled lobe of {lobe.shape[0]} vertices, centre {np.round(centre, 3)}, r {r:.3f}")

    return {k: [round(float(x), 4) for x in p] for k, p in J.items()}


# --------------------------------------------------------------------------- 5. rig

# parent -> children, the shiba hierarchy (features the estimate cannot place are left out)
HIERARCHY = {
    "root": ["COG_jnt"],
    "COG_jnt": ["pelvis_jnt", "spine_1_jnt"],
    "pelvis_jnt": ["tail_1_jnt", "L_hip_jnt", "R_hip_jnt", "central_corrective"],
    "tail_1_jnt": ["tail_2_jnt"], "tail_2_jnt": ["tail_3_jnt"], "tail_3_jnt": ["tail_4_jnt"],
    "tail_4_jnt": ["tail_5_jnt"], "tail_5_jnt": ["tail_6_jnt"], "tail_6_jnt": ["tail_7_jnt"],
    "L_hip_jnt": ["L_knee_jnt"], "L_knee_jnt": ["L_ankle_jnt"], "L_ankle_jnt": ["L_foot_jnt"], "L_foot_jnt": ["LB_toes_jnt"],
    "R_hip_jnt": ["R_knee_jnt"], "R_knee_jnt": ["R_ankle_jnt"], "R_ankle_jnt": ["R_foot_jnt"], "R_foot_jnt": ["RB_toes_jnt"],
    "spine_1_jnt": ["spine_2_jnt"], "spine_2_jnt": ["chest_jnt", "breathe_jnt"],
    "chest_jnt": ["neck_base_jnt", "L_clavicle_jnt", "R_clavicle_jnt"],
    "neck_base_jnt": ["neck_mid_jnt"], "neck_mid_jnt": ["head_jnt"],
    "head_jnt": ["jaw_jnt", "nose_jnt", "L_ear_base_jnt", "R_ear_base_jnt", "L_eye_jnt", "R_eye_jnt"],
    "L_ear_base_jnt": ["L_ear_tip_jnt"], "R_ear_base_jnt": ["R_ear_tip_jnt"],
    "L_clavicle_jnt": ["L_shoulder_jnt"], "L_shoulder_jnt": ["L_elbow_jnt"], "L_elbow_jnt": ["L_wrist_jnt"],
    "L_wrist_jnt": ["L_hand_jnt"], "L_hand_jnt": ["LF_toes_jnt"],
    "R_clavicle_jnt": ["R_shoulder_jnt"], "R_shoulder_jnt": ["R_elbow_jnt"], "R_elbow_jnt": ["R_wrist_jnt"],
    "R_wrist_jnt": ["R_hand_jnt"], "R_hand_jnt": ["RF_toes_jnt"],
}
# leaf bones point this way (metres) since they have no child to aim at
LEAF_TAILS = {
    "LB_toes_jnt": (0, -0.03, 0), "RB_toes_jnt": (0, -0.03, 0), "LF_toes_jnt": (0, -0.03, 0), "RF_toes_jnt": (0, -0.03, 0),
    "tail_7_jnt": None, "jaw_jnt": (0, -0.04, -0.01), "nose_jnt": (0, -0.02, 0),
    "L_ear_tip_jnt": (0, 0, 0.02), "R_ear_tip_jnt": (0, 0, 0.02),
    "L_eye_jnt": (0, -0.015, 0), "R_eye_jnt": (0, -0.015, 0),
    "breathe_jnt": (0, 0, 0.04), "central_corrective": (0, 0, -0.03),
}


def build_armature(joints: dict[str, list[float]]):
    arm_data = bpy.data.armatures.new("skeletal")
    arm = bpy.data.objects.new("skeletal", arm_data)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    bones = {}

    def add(name, parent=None):
        if name not in joints:
            return
        head = Vector(joints[name])
        children = [c for c in HIERARCHY.get(name, []) if c in joints]
        if children:
            tail = Vector(np.mean([joints[c] for c in children], axis=0).tolist())
            if (tail - head).length < 0.01:
                tail = head + Vector((0, 0.02, 0))
        else:
            offset = LEAF_TAILS.get(name)
            if offset is None:
                # continue the parent's direction
                parent_head = Vector(joints[parent]) if parent else head - Vector((0, 0.02, 0))
                direction = (head - parent_head)
                tail = head + (direction.normalized() * 0.03 if direction.length > 1e-6 else Vector((0, 0.02, 0)))
            else:
                tail = head + Vector(offset)
        b = arm_data.edit_bones.new(name)
        b.head, b.tail = head, tail
        if parent is not None and parent in bones:
            b.parent = bones[parent]
        bones[name] = b
        for c in children:
            add(c, name)

    add("root")
    bpy.ops.object.mode_set(mode="OBJECT")
    print(f"[auto_rig] armature: {len(bones)} bones")
    return arm


BONE_RADIUS = {  # metres: how far a bone's influence reaches before the Gaussian falls off
    "default": 0.05, "spine": 0.14, "pelvis": 0.14, "COG": 0.14, "chest": 0.13, "neck": 0.09, "head": 0.10,
    # eye and nose radii set the _features (protected, restored from the
    # template's own pixels) blobs: seed_regions takes weight >= 0.15, i.e.
    # 1.4 radii. 0.018 / 0.025 restored a ring of the white template fur
    # around the eyes and nose of a black dog.
    "jaw": 0.05, "nose": 0.016, "eye": 0.010, "ear": 0.03, "tail": 0.035, "breathe": 0.10, "central_corrective": 0.08,
    "root": 0.0, "clavicle": 0.06, "shoulder": 0.07, "hip": 0.08, "elbow": 0.05, "knee": 0.06, "wrist": 0.04,
    "ankle": 0.04, "hand": 0.035, "foot": 0.035, "toes": 0.03,
}


def bone_radius(name: str) -> float:
    lowered = name.lower()
    for key, r in BONE_RADIUS.items():
        if key != "default" and key.lower() in lowered:
            return r
    return BONE_RADIUS["default"]


def skin(obj, arm, v: np.ndarray):
    """Vertex groups from the distance of every vertex to every bone segment,
    a Gaussian of that distance over the bone's radius, the three strongest
    kept and normalised. Blender's heat solver is the better skin when it
    converges, but on a generated mesh it returned no weights at all and
    said so only in the console; this always produces a usable skin."""
    heads, tails, names, radii = [], [], [], []
    for b in arm.data.bones:
        r = bone_radius(b.name)
        if r <= 0:
            continue
        names.append(b.name)
        heads.append(np.array(b.head_local, dtype=np.float64))
        tails.append(np.array(b.tail_local, dtype=np.float64))
        radii.append(r)
    heads, tails, radii = np.stack(heads), np.stack(tails), np.array(radii)
    ab = tails - heads
    ab_len2 = np.maximum((ab * ab).sum(axis=1), 1e-9)
    weights = np.zeros((len(v), len(names)), dtype=np.float64)
    for k in range(len(names)):
        t = np.clip(((v - heads[k]) @ ab[k]) / ab_len2[k], 0.0, 1.0)
        closest = heads[k] + t[:, None] * ab[k]
        d = np.linalg.norm(v - closest, axis=1)
        weights[:, k] = np.exp(-(d / radii[k]) ** 2)
    # keep the three strongest bones per vertex
    order = np.argsort(-weights, axis=1)[:, :3]
    kept = np.zeros_like(weights)
    rows = np.arange(len(v))[:, None]
    kept[rows, order] = weights[rows, order]
    kept[kept < 1e-4] = 0.0
    total = kept.sum(axis=1, keepdims=True)
    total[total == 0] = 1.0
    kept /= total

    for vg in list(obj.vertex_groups):
        obj.vertex_groups.remove(vg)
    for k, name in enumerate(names):
        vg = obj.vertex_groups.new(name=name)
        idx = np.nonzero(kept[:, k] > 0)[0]
        for i in idx:
            vg.add([int(i)], float(kept[i, k]), "REPLACE")
    obj.parent = arm
    obj.matrix_parent_inverse = arm.matrix_world.inverted()
    mod = obj.modifiers.new("Armature", "ARMATURE")
    mod.object = arm
    dominant = np.bincount(order[:, 0], minlength=len(names))
    quiet = [names[k] for k in range(len(names)) if dominant[k] == 0]
    print(f"[auto_rig] skinned {len(v)} vertices over {len(names)} bones; bones dominating no vertex: {quiet or 'none'}")
    return [names[k] for k in order[:, 0]]


# --------------------------------------------------------------------------- 6. export + debug

def export(obj, arm, path):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    kwargs = {"filepath": path, "export_format": "GLB", "export_animations": False, "export_skins": True,
              "export_apply": False, "export_image_format": "AUTO", "use_selection": True}
    valid = set(bpy.ops.export_scene.gltf.get_rna_type().properties.keys())
    bpy.ops.export_scene.gltf(**{k: v for k, v in kwargs.items() if k in valid})
    print(f"[auto_rig] exported {path}")


def debug_render(obj, joints, path):
    """Workbench render from the side and the front-left with a red sphere on every joint."""
    import bmesh  # noqa: F401 - ensures bmesh is importable in this Blender
    red = bpy.data.materials.new("_joint")
    red.diffuse_color = (1.0, 0.05, 0.05, 1.0)
    spheres = []
    for name, p in joints.items():
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.012, location=p, segments=8, ring_count=6)
        s = bpy.context.active_object
        s.name = f"_j_{name}"
        s.data.materials.append(red)
        spheres.append(s)
    v = vertices_world(obj)
    centre = (v.max(0) + v.min(0)) / 2
    span = float(np.max(v.max(0) - v.min(0)))
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    # see-through body so the joints inside show
    scene.display.shading.show_xray = True
    scene.display.shading.xray_alpha = 0.25
    scene.render.resolution_x, scene.render.resolution_y = 1400, 900
    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    height = float(v[:, 2].max() - v[:, 2].min())
    cam_data.ortho_scale = max(span, height / 0.64) * 1.15
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    for tag, offset in (("side", (span * 3, 0, 0)), ("front", (span * 2, -span * 2.5, span * 0.6))):
        cam.location = Vector(centre) + Vector(offset)
        direction = Vector(centre) - cam.location
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = path.replace(".png", f"_{tag}.png")
        bpy.ops.render.render(write_still=True)
        print(f"[auto_rig] debug render {scene.render.filepath}")
    for s in spheres:
        bpy.data.objects.remove(s)


# --------------------------------------------------------------------------- main

def main():
    args = parse_args()
    reset_scene()
    bpy.ops.import_scene.gltf(filepath=os.path.abspath(args.input))
    obj = main_mesh()
    base_img, normal_img = find_material_images(obj)
    print(f"[auto_rig] input images: base {getattr(base_img, 'name', None)}, normal {getattr(normal_img, 'name', None)}")
    v = normalise(obj, args.height)

    edges = np.array([e.vertices[:] for e in obj.data.edges], dtype=np.int64)
    thick = local_thickness(obj, v, max_depth=0.6 * args.height)
    darkness = vertex_darkness(obj, base_img)
    joints = estimate_joints(v, edges, thick, darkness)
    if args.joints_override:
        override = json.loads(open(args.joints_override, encoding="utf-8").read())
        joints.update({k: [float(x) for x in p] for k, p in override.items()})
        print(f"[auto_rig] overrode {len(override)} joint(s)")
    debug_dir = os.path.abspath(args.debug_dir or os.path.dirname(os.path.abspath(args.output)))
    os.makedirs(debug_dir, exist_ok=True)
    with open(os.path.join(debug_dir, "joints.json"), "w", encoding="utf-8") as f:
        json.dump(joints, f, indent=1)

    arm = build_armature(joints)
    dominant = skin(obj, arm, v)

    if not args.no_unwrap:
        bpy.context.view_layer.objects.active = obj
        reunwrap(obj, v, dominant, joints)
        baked_base = bake_image(obj, base_img, "base_color", non_color=False) if base_img else None
        # the model's normal map is NOT carried over: a tangent-space normal
        # map is relative to the UV layout it was baked for, and re-sampled
        # onto the new layout it lit the fur in wrong directions (the viewer
        # showed it as a mottled, greasy coat). The mask build paints its own
        # hair grain into the colour map; flat shading is the honest choice.
        if baked_base is None:
            raise SystemExit("[auto_rig] the input has no base colour texture to bake")
        rebuild_material(obj, baked_base, None)
        obj.data.uv_layers.remove(obj.data.uv_layers["UVMap_old"])

    debug_render(obj, joints, os.path.abspath(os.path.join(debug_dir, "_joints.png")))
    export(obj, arm, os.path.abspath(args.output))
    blend = os.path.splitext(os.path.abspath(args.output))[0] + ".blend"
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    print(f"[auto_rig] saved {blend}")


if __name__ == "__main__":
    main()
