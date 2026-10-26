"""Quick diagnostic: list what object types a GLB actually contains.
Usage: blender --background --python ar_pipeline/inspect_glb.py -- dog.glb
"""
import sys
import bpy

argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=argv[0])

print("\n=== OBJECTS IN FILE ===")
for obj in bpy.data.objects:
    print(f"- {obj.name}  (type={obj.type})")

print("\n=== ARMATURES ===")
if not bpy.data.armatures:
    print("(none found)")
for arm in bpy.data.armatures:
    print(f"- armature '{arm.name}', bones: {[b.name for b in arm.bones]}")
