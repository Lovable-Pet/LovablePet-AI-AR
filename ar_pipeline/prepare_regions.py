"""
Prepare a .blend for region tagging.

Imports the rigged GLB and adds one empty vertex group per region, so the
selection work in Blender is just: pick faces -> click the region's group ->
Assign. Creating the groups here (instead of typing nine names by hand) keeps
them spelled exactly the way build_template.py looks for them; a typo would
silently produce a blank mask instead of a baked one.

Usage (headless):
    blender --background --python prepare_regions.py -- input.glb work/shiba_small.blend

Then, in the Blender GUI:
    1. File > Open the .blend
    2. select the body mesh, Tab into Edit Mode, press 3 for face select
    3. Alt+Z for x-ray, then B (box) or C (circle) to select a body part
    4. Properties > Object Data (green triangle) > Vertex Groups: click the
       region, Weight 1.0, press Assign
    5. repeat for the other regions, Ctrl+S

Finally, feed the .blend back to build_template.py, which bakes every
tagged region into masks/{region}.png:
    blender --background --python build_template.py -- work/shiba_small.blend output/skeleton_shiba_small.glb
"""

import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_template import SOURCE_REGIONS as REGIONS, find_main_mesh, import_glb, reset_scene


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    if len(argv) != 2:
        print("[prepare_regions] usage: ... -- input.glb output.blend")
        sys.exit(1)
    input_glb, out_blend = argv

    reset_scene()
    import_glb(input_glb)

    mesh_obj = find_main_mesh()
    if mesh_obj is None:
        print("[prepare_regions] ERROR: no UV-mapped mesh found in the GLB.")
        sys.exit(1)

    existing = {vg.name.lower() for vg in mesh_obj.vertex_groups}
    created = []
    for region in REGIONS:
        if region in existing:
            continue
        mesh_obj.vertex_groups.new(name=region)
        created.append(region)

    print(f"[prepare_regions] mesh '{mesh_obj.name}': added {len(created)} region "
          f"groups {created}")

    out_blend = os.path.abspath(out_blend)
    os.makedirs(os.path.dirname(out_blend), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    print(f"[prepare_regions] wrote {out_blend}")
    print("[prepare_regions] open it, tag the regions, save, then run "
          "build_template.py with the .blend as input.")


if __name__ == "__main__":
    main()
