"""Seed the sixteen SOURCE region vertex groups from the rig, so the hand work starts
from a rough draft instead of an empty mesh.

The rig already encodes most of the anatomy the regions are defined by: there
are brow joints, ear base and tip joints, per-toe joints, a jaw, a neck chain
and a tail chain. Assigning each vertex to the region of its dominant bone gets
the coarse split for free.

What the bones cannot express is the top/bottom and inner/outer splits, because
one bone drives both sides of the body part it moves: back against underside,
ears_outer against ears_inner, collar against throat, legs_upper against the
inner-leg part of underside. Those are decided by the vertex normal instead,
which follows the surface rather than a flat cut through it.

The result is a draft. Face regions especially - eyebrows, muzzle_bridge,
cheeks, skull - sit within millimetres of each other on a small dog and want
checking by eye. The coverage report at the end says which regions came out
suspiciously small or empty.

Usage:
    blender --background work/shiba_small.blend \
        --python ar_pipeline/seed_regions.py -- work/shiba_small_seeded.blend

Then open the result in Blender, fix what looks wrong, and bake:
    blender --background --python ar_pipeline/build_template.py -- \
        work/shiba_small_seeded.blend output/skeleton_shiba_small.glb
"""

import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_template import SOURCE_REGIONS as REGIONS, find_main_mesh

# Bone-name fragment -> region, tried in order so the specific wins over the
# general ("L_ear_tip" before "ear", "wrist" before "L_").
# The nine groups the old region set left behind. Only these are cleared:
# everything else without a _jnt suffix - root, the corrective groups - is part
# of the skinning and removing it would change how the mesh deforms.
LEGACY_REGIONS = (
    "head", "ears", "muzzle", "chest", "back", "belly", "legs", "paws", "tail",
)

# Not a coat region. Eyes, nose, lips, tongue and the inner mouth are not fur,
# have no colour in the fur palette, and must never be painted over - by a
# mask fill, by a projected view that forgot to draw them, or by delighting.
# They are baked to their own mask and the template's own pixels are put back
# there at the very end of every texture build. The rig names these bones.
FEATURES_GROUP = "features"
# lip and mouth_corner are left out on purpose: those bones move the fur
# around the mouth, and including them kept the template's red muzzle fur on a
# cream-muzzled dog. The mouth line itself survives through the brightness
# map, which keeps hard edges.
FEATURE_FRAGMENTS = ("eye", "lid", "squint", "nose", "nostril", "tongue")
FEATURE_MIN_WEIGHT = 0.15

# Small joints that place a feature precisely but almost never outweigh the big
# joint they sit on: head_jnt drives the whole skull, so a vertex on the brow
# ridge is still mostly head_jnt. Where one of these carries real weight it
# names the region even though it does not dominate.
# fragment -> the weight at which it takes over. The brow joints barely move
# the surface, so they are blended thinly over the ridge and never reach a
# quarter; the nose joints do carry the nose and only deserve it when they lead.
PRIORITY_FRAGMENTS = {"brow": 0.05, "nostril": 0.25, "nose": 0.25}

BONE_RULES = [
    # --- head ---
    ("brow", "eyebrows"),
    ("nostril", "muzzle_bridge"),
    ("nose", "muzzle_bridge"),
    ("whisker", "cheeks"),
    ("upperlip", "cheeks"),
    ("upper_lip", "cheeks"),
    ("lower_lip", "cheeks"),
    ("lowerlip", "cheeks"),
    ("mouth_corner", "cheeks"),
    ("squint", "cheeks"),
    ("lid", "cheeks"),
    ("eye", "cheeks"),
    ("tongue", None),          # inside the mouth, never visible on the texture
    ("jaw", "throat"),         # normal test lifts the sides back into cheeks
    ("ear", "ears_outer"),     # normal test carves out ears_inner
    ("head", "skull"),         # normal test carves out cheeks
    # --- neck ---
    ("neck", "collar"),        # normal test carves out throat
    # --- tail: tip joints first ---
    ("tail_6", "tail_tip"),
    ("tail_7", "tail_tip"),
    ("tail", "tail_top"),      # normal test carves out underside
    # --- legs, distal first ---
    ("toes", "paws"),
    ("index", "paws"),
    ("mid", "paws"),
    ("ring", "paws"),
    ("pinkie", "paws"),
    ("hand", "paws"),
    ("foot", "paws"),
    ("wrist", "legs_lower"),
    ("ankle", "legs_lower"),
    ("elbow", "legs_lower"),
    ("knee", "legs_lower"),
    ("shoulder", "legs_upper"),
    ("clavicle", "chest"),
    ("hip", "legs_upper"),
    ("leg_corrective", "legs_upper"),
    # --- torso ---
    ("chest", "chest"),
    ("spine", "back"),         # normal test carves out underside
    ("pelvis", "back"),
    ("breathe", "underside"),
    ("COG", "back"),
    ("central_corrective", "underside"),
    ("root", None),
]

# Regions whose bone covers both sides of the part, resolved by normal:
#   region -> (test, region when the test passes)
# nz is the vertex normal's z, outward is the normal's x times the vertex's
# side (positive when the normal points away from the centreline).
NORMAL_SPLITS = {
    "back": (lambda nz, outward, v: nz < -0.25, "underside"),
    # The chest and clavicle bones move the whole shoulder girdle, top
    # included, and the mask painted the withers white on a black dog. The
    # chest is the front of the chest (see REGION_DESCRIPTIONS); what faces
    # up there is the shoulder, and the shoulder is back.
    "chest": (lambda nz, outward, v: nz > 0.25, "back"),
    "tail_top": (lambda nz, outward, v: nz < -0.35, "underside"),
    "collar": (lambda nz, outward, v: nz < -0.2, "throat"),
    # Inner or rear surface -> underside, on the upper and the lower leg
    # alike, so the cream runs unbroken from the groin to the paw. -Y is
    # forward, so a normal with positive y faces the back of the leg.
    # The rear surface of the upper leg is the underside, but not where it
    # curves over the top of the rump under the tail: that faces up and is
    # back, and painted cream it showed as a pale ring round the tail base.
    "legs_upper": (lambda nz, outward, v: outward < -0.3 or (v.normal.y > 0.35 and nz < 0.2),
                   "underside"),
    "legs_lower": (lambda nz, outward, v: outward < -0.2 or v.normal.y > 0.3,
                   "underside"),
    # the jaw drives the throat and the lower cheek; only what faces downward
    # is really the throat
    "throat": (lambda nz, outward, v: nz > -0.15, "cheeks"),
    # head_jnt runs from the crown to the cheekbone; the skull is what faces up
    "skull": (lambda nz, outward, v: nz < 0.25, "cheeks"),
}


# The muzzle is forward of the stop and no wider than the nose; -Y is forward
# and on the shiba the nose sits at y=-0.61, the stop around y=-0.45. These
# are the fallbacks: with nose and eye joints in the rig, muzzle_bounds()
# places the stop a little behind the eyes and takes the width from the eye
# spacing, so another template (auto_rig.py builds the same joints) fits.
MUZZLE_FRONT_Y = -0.45
MUZZLE_HALF_WIDTH = 0.045


def muzzle_bounds():
    armature = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    if armature is None:
        return MUZZLE_FRONT_Y, MUZZLE_HALF_WIDTH
    bones = armature.data.bones
    if not all(name in bones for name in ("nose_jnt", "L_eye_jnt", "R_eye_jnt")):
        return MUZZLE_FRONT_Y, MUZZLE_HALF_WIDTH
    mw = armature.matrix_world
    nose = mw @ bones["nose_jnt"].head_local
    left, right = mw @ bones["L_eye_jnt"].head_local, mw @ bones["R_eye_jnt"].head_local
    eye_y = (left.y + right.y) / 2
    # the strip runs from the nose up to just behind the eye centres; at
    # 0.6 it ran on over the forehead (a blue cap between the ears on the
    # jindo) and the face step keeps muzzle_bridge exactly as tagged
    front = eye_y + 0.3 * (eye_y - nose.y)
    half = 0.35 * (left - right).length
    print(f"[seed_regions] muzzle from the rig: forward of y={front:.3f}, half width {half:.3f}")
    return front, half


def ear_hangs(side: str) -> bool:
    """True when the ear tip bone sits below the ear base bone (a drop ear)."""
    armature = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    if armature is None:
        return False
    bones = armature.data.bones
    base, tip = f"{side}_ear_base_jnt", f"{side}_ear_tip_jnt"
    if base not in bones or tip not in bones:
        return False
    mw = armature.matrix_world
    hangs = (mw @ bones[tip].head_local).z < (mw @ bones[base].head_local).z
    print(f"[seed_regions] {side} ear {'hangs' if hangs else 'stands'}")
    return hangs


def ear_shell_split(mesh, index_to_name):
    """Decide ears_inner against ears_outer from the ear's own geometry.

    An ear is a thin shell, so its two faces carry nearly opposite normals and
    the direction they differ along is the shell's thickness axis. That axis is
    whatever the first singular vector of the normals says it is, which matters
    because it is not the lateral one: on an erect ear it comes out close to
    forward-and-down, and a test on normal.x - the obvious guess - lands near
    chance. Each ear is fitted separately so a model with one ear cocked still
    splits correctly.

    Which face is the concave one is then read off the positions. On an
    erect ear the inner face sits forward of the outer (-Y is forward). On
    a hanging ear (the tip bone below the base bone: maltese, retriever,
    poodle, 2026-10-12) the flap lies against the cheek and its inner face
    is the one nearer the midline; the forward rule put three quarters of
    the maltese ear in ears_inner.
    """
    sides = {1: [], -1: []}
    for vertex in mesh.vertices:
        bone = dominant_bone(vertex, index_to_name)
        if bone is not None and region_of_bone(bone) == "ears_outer":
            sides[1 if vertex.co.x >= 0 else -1].append(vertex)

    hanging = {1: ear_hangs("L"), -1: ear_hangs("R")}
    decided = {}
    for side, vertices in sides.items():
        if len(vertices) < 8:
            continue
        normals = np.array([tuple(v.normal) for v in vertices])
        axis = np.linalg.svd(normals, full_matrices=False)[2][0]
        projection = normals @ axis
        positions = np.array([tuple(v.co) for v in vertices])

        forward, backward = projection > 0, projection < 0
        if not forward.any() or not backward.any():
            continue
        if hanging[side]:
            inner_is_positive = (np.abs(positions[forward][:, 0]).mean()
                                 < np.abs(positions[backward][:, 0]).mean())
        else:
            inner_is_positive = positions[forward][:, 1].mean() < positions[backward][:, 1].mean()

        for vertex, value in zip(vertices, projection):
            is_inner = (value > 0) == inner_is_positive
            decided[vertex.index] = "ears_inner" if is_inner else "ears_outer"
    return decided


def region_of_bone(name):
    lowered = name.lower()
    for fragment, region in BONE_RULES:
        if fragment.lower() in lowered:
            return region
    return None


def dominant_bone(vertex, index_to_name):
    best_name, best_weight = None, 0.0
    priority_name, priority_weight = None, 0.0

    for group in vertex.groups:
        name = index_to_name.get(group.group)
        if name is None or "_jnt" not in name:
            continue
        if group.weight > best_weight:
            best_name, best_weight = name, group.weight
        lowered = name.lower()
        for fragment, minimum in PRIORITY_FRAGMENTS.items():
            if fragment in lowered and group.weight >= minimum:
                if group.weight > priority_weight:
                    priority_name, priority_weight = name, group.weight
                break

    return priority_name or best_name


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    if len(argv) != 1:
        print("[seed_regions] usage: ... --python seed_regions.py -- output.blend")
        sys.exit(1)
    out_blend = argv[0]

    mesh_obj = find_main_mesh()
    if mesh_obj is None:
        print("[seed_regions] ERROR: no UV-mapped mesh found.")
        sys.exit(1)
    print(f"[seed_regions] mesh '{mesh_obj.name}', {len(mesh_obj.data.vertices)} vertices")

    # Every removal renumbers the groups after it, so all removals happen
    # first and the index map is built afterwards. Creating groups is safe in
    # between only because new ones are appended past the existing indices.
    for vg in list(mesh_obj.vertex_groups):
        if vg.name in LEGACY_REGIONS or vg.name in REGIONS or vg.name == FEATURES_GROUP:
            print(f"  clearing region group '{vg.name}'")
            mesh_obj.vertex_groups.remove(vg)

    index_to_name = {vg.index: vg.name for vg in mesh_obj.vertex_groups}
    print(f"[seed_regions] {len(index_to_name)} rig groups kept")

    groups = {region: mesh_obj.vertex_groups.new(name=region) for region in REGIONS}

    ear_regions = ear_shell_split(mesh_obj.data, index_to_name)
    print(f"[seed_regions] ear shell split: {len(ear_regions)} vertices, "
          f"{sum(1 for r in ear_regions.values() if r == 'ears_inner')} inner")

    assignment = {region: [] for region in REGIONS}
    unassigned = 0
    muzzle_front_y, muzzle_half_width = muzzle_bounds()

    for vertex in mesh_obj.data.vertices:
        bone = dominant_bone(vertex, index_to_name)
        if bone is None:
            unassigned += 1
            continue
        region = region_of_bone(bone)
        if region is None:
            unassigned += 1
            continue

        if vertex.index in ear_regions:
            assignment[ear_regions[vertex.index]].append(vertex.index)
            continue

        # the jaw bone reaches the muzzle top on a rig whose jaw joint sits
        # high (auto_rig's first draft), so throat is tested too: what faces
        # up on the muzzle's midline is the bridge whichever bone drives it
        if region in ("cheeks", "skull", "throat"):
            co, normal = vertex.co, vertex.normal
            if co.y < muzzle_front_y and normal.z > 0.2 and abs(co.x) < muzzle_half_width:
                region = "muzzle_bridge"

        split = NORMAL_SPLITS.get(region)
        if split is not None:
            test, other = split
            normal = vertex.normal
            side = 1.0 if vertex.co.x >= 0 else -1.0
            outward = normal.x * side
            if test(normal.z, outward, vertex):
                region = other

        assignment[region].append(vertex.index)

    for region, indices in assignment.items():
        if indices:
            groups[region].add(indices, 1.0, "REPLACE")

    # the features mask: any vertex a face-feature bone moves appreciably
    feature_indices = []
    for vertex in mesh_obj.data.vertices:
        for group in vertex.groups:
            name = index_to_name.get(group.group, "").lower()
            if "_jnt" in name and any(f in name for f in FEATURE_FRAGMENTS) \
                    and group.weight >= FEATURE_MIN_WEIGHT:
                feature_indices.append(vertex.index)
                break
    features = mesh_obj.vertex_groups.new(name=FEATURES_GROUP)
    if feature_indices:
        features.add(feature_indices, 1.0, "REPLACE")
    print(f"[seed_regions] features (eyes/nose/mouth): {len(feature_indices)} verts "
          f"({len(feature_indices) / len(mesh_obj.data.vertices):.1%})")

    total = len(mesh_obj.data.vertices)
    print(f"\n[seed_regions] coverage ({total} vertices, {unassigned} left unassigned):")
    thin = []
    for region in REGIONS:
        count = len(assignment[region])
        flag = ""
        if count == 0:
            flag = "  <- EMPTY, tag this one by hand"
            thin.append(region)
        elif count < total * 0.005:
            flag = "  <- very small, check it"
            thin.append(region)
        print(f"  {region:16s} {count:6d} verts ({count / total:5.1%}){flag}")

    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out_blend))
    print(f"\n[seed_regions] saved {out_blend}")
    if thin:
        print(f"[seed_regions] needs attention: {', '.join(thin)}")
    print("[seed_regions] open it in Blender, fix the face regions by eye, then run "
          "build_template.py")


if __name__ == "__main__":
    main()
