"""
Standalone check for build_template.py's output.

Runs without Blender, so it works in CI or a plain Python environment:
  1. opens the GLB with pygltflib and confirms the expected animation
     clips are present (idle, tail_wag by default)
  2. confirms uv_layout.png and all nine region masks exist next to it,
     at the same resolution as the GLB's base texture

Usage:
    pip install pygltflib
    python verify_export.py output/skeleton_shiba_small.glb
    python verify_export.py output/skeleton_shiba_small.glb --expect idle tail_wag

The templates folder is derived from the GLB name by convention:
    output/skeleton_{species}_{size}.glb
    -> output/templates/{species}_{size}/{uv_layout.png, masks/*.png}

Exit code 0 = everything required is present, 1 = something is missing.
Unpainted (all black) masks are reported as TODO, not as a failure.
Install Pillow to enable that paint-state check.
"""

import argparse
import os
import re
import struct
import sys

from pygltflib import GLTF2

try:
    import numpy as np
    from PIL import Image
except ImportError:
    np = None
    Image = None

# REGIONS lives in api/regions.py, which carries the boundary definitions and
# pulls in no Blender modules, so every stage can share the one list.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.regions import REGIONS  # noqa: E402
OUTPUT_NAME_RE = re.compile(
    r"^skeleton_(?P<species>.+)_(?P<size>small|medium|large)\.glb$", re.IGNORECASE
)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_size_from_bytes(data):
    if len(data) < 24 or data[:8] != PNG_SIGNATURE or data[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", data[16:24])


def png_size(path):
    with open(path, "rb") as fh:
        return png_size_from_bytes(fh.read(24))


def templates_dir_for(glb_path):
    name = os.path.basename(glb_path)
    match = OUTPUT_NAME_RE.match(name)
    if not match:
        return None
    species = match.group("species").lower()
    size_grade = match.group("size").lower()
    return os.path.join(os.path.dirname(os.path.abspath(glb_path)), "templates",
                        f"{species}_{size_grade}")


def base_texture_size(gltf):
    """Largest embedded PNG in the GLB, used as the reference resolution."""
    blob = gltf.binary_blob()
    if not blob:
        return None
    best = None
    for image in gltf.images or []:
        if image.bufferView is None:
            continue
        view = gltf.bufferViews[image.bufferView]
        offset = view.byteOffset or 0
        size = png_size_from_bytes(blob[offset:offset + 24])
        if size and (best is None or size[0] * size[1] > best[0] * best[1]):
            best = size
    return best


def check_structure(gltf, expected):
    ok = True

    found = [anim.name for anim in gltf.animations]
    missing = [name for name in expected if name not in set(found)]
    print(f"Animations found: {found}")
    if missing:
        print(f"  FAIL: missing expected animation clip(s): {missing}")
        ok = False
    else:
        print(f"  OK: all expected clips present: {expected}")

    skins = gltf.skins or []
    if not skins:
        print("  FAIL: no skins in the GLB -- the rig was lost on export, so the "
              "animations cannot deform the mesh")
        ok = False
    else:
        joints = ", ".join(str(len(skin.joints or [])) for skin in skins)
        print(f"  OK: {len(skins)} skin(s), joint counts: {joints}")

    return ok


def is_blank(path):
    """True if the mask has no painted (bright) pixels. None if unknown."""
    if Image is None:
        return None
    with Image.open(path) as img:
        return img.convert("L").getextrema()[1] <= 128


def check_coverage(templates_dir):
    """How much of the mesh's UV area no region claims (a gap shows up as a
    stripe of untouched base texture) and how much two or more claim."""
    if np is None or Image is None:
        return
    uv_path = os.path.join(templates_dir, "uv_layout.png")
    if not os.path.isfile(uv_path):
        return

    with Image.open(uv_path) as img:
        island = np.array(img.convert("RGBA"))[..., 3] > 0
    island_px = int(island.sum())
    if island_px == 0:
        return

    claims = np.zeros(island.shape, dtype=np.uint8)
    for region in REGIONS:
        path = os.path.join(templates_dir, "masks", f"{region}.png")
        if not os.path.isfile(path):
            continue
        with Image.open(path) as img:
            claims += (np.array(img.convert("L")) > 128).astype(np.uint8)

    uncovered = int((island & (claims == 0)).sum()) / island_px
    overlapped = int((island & (claims >= 2)).sum()) / island_px
    print(f"  coverage: {1 - uncovered:.1%} of the UV islands claimed, "
          f"{overlapped:.1%} claimed by 2+ regions")
    if uncovered > 0.02:
        print(f"  WARN: {uncovered:.1%} of the mesh belongs to no region -- those "
              "spots keep the original texture when regions are recolored")


def check_templates(templates_dir, expected_size):
    print(f"Templates dir: {templates_dir}")
    if not os.path.isdir(templates_dir):
        print("  FAIL: templates folder does not exist")
        return False

    ok = True
    uv_path = os.path.join(templates_dir, "uv_layout.png")
    if os.path.isfile(uv_path):
        print(f"  OK: uv_layout.png {png_size(uv_path)}")
    else:
        print("  FAIL: uv_layout.png is missing")
        ok = False

    masks_dir = os.path.join(templates_dir, "masks")
    missing, wrong_size, blank = [], [], []
    for region in REGIONS:
        path = os.path.join(masks_dir, f"{region}.png")
        if not os.path.isfile(path):
            missing.append(region)
            continue
        size = png_size(path)
        if expected_size and size != expected_size:
            wrong_size.append(f"{region} {size}")
            continue
        if is_blank(path):
            blank.append(region)

    print(f"  masks present: {len(REGIONS) - len(missing)}/{len(REGIONS)} in {masks_dir}")
    if missing:
        print(f"  FAIL: missing masks: {missing}")
        ok = False
    if wrong_size:
        print(f"  FAIL: wrong resolution (expected {expected_size}): {wrong_size}")
        ok = False
    if blank:
        print(f"  TODO: still unpainted (all black): {blank}")
    if Image is None:
        print("  NOTE: install Pillow and numpy for the paint-state and coverage checks")
    elif not missing:
        check_coverage(templates_dir)

    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("glb_path")
    parser.add_argument("--expect", nargs="+", default=["idle", "tail_wag"],
                        help="animation clip names that must be present")
    parser.add_argument("--templates-dir", default=None,
                        help="override the conventional templates folder location")
    parser.add_argument("--texture-size", type=int, default=0,
                        help="override the expected mask resolution")
    args = parser.parse_args()

    try:
        gltf = GLTF2().load(args.glb_path)
    except Exception as exc:
        print(f"FAIL: could not open '{args.glb_path}': {exc}")
        sys.exit(1)

    expected_size = None
    if args.texture_size:
        expected_size = (args.texture_size, args.texture_size)
    else:
        expected_size = base_texture_size(gltf)
    print(f"Base texture resolution: {expected_size or 'unknown'}")

    templates_dir = args.templates_dir or templates_dir_for(args.glb_path)
    if templates_dir is None:
        print(f"FAIL: '{os.path.basename(args.glb_path)}' does not follow the "
              "skeleton_{species}_{size}.glb convention, cannot locate templates")
        sys.exit(1)

    structure_ok = check_structure(gltf, args.expect)
    templates_ok = check_templates(templates_dir, expected_size)

    if structure_ok and templates_ok:
        print("PASS")
        sys.exit(0)
    print("FAIL")
    sys.exit(1)


if __name__ == "__main__":
    main()
