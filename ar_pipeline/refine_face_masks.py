"""Redefine the three face masks - eyebrows, cheeks, skull - from the eyes.

WHY
---
The hand-tagged face masks were drawn around the template's own red-shiba
coat, not around a face: "eyebrows" is one wide band across the whole
forehead between the eyes, and "cheeks" runs from the ear base to the jaw and
takes the temples above the eyes with it. On a tan-point coat those two
regions are tan, so the tan climbed over the forehead and around the eyes
where a real black-and-tan is black. The refinement blends the edges but
cannot move them.

WHAT THIS DOES
--------------
Every texel the four face masks cover (skull, eyebrows, cheeks,
muzzle_bridge; the eyes, nose and mouth excluded) is reassigned from where
it sits relative to the two eye sockets, in world space:

  eyebrows   two ellipses just above each eye, the tan-point pip
             (the same ellipse projection.py draws: PIP_* in eye radii)
  cheeks     below the eyes and in front of the ear base - the side of
             the face and of the muzzle down to the jaw
  skull      the rest: forehead, temples, around the eyes, back of the head
  muzzle_bridge is kept as tagged.

The old face masks are moved to masks_face_hand/ so the change can be
undone; masks/ is rebuilt as the ten-region union.

Usage:
    python ar_pipeline/refine_face_masks.py [--template shiba_small] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from api.regions import REGIONS, SOURCES  # noqa: E402
from build_landmarks import EAR_BASE_JOINTS, load_mask  # noqa: E402

FACE = ("skull", "eyebrows", "cheeks", "muzzle_bridge")
# the pip above each eye, in eye-socket radii (projection.py draws the same
# ellipse when it forces the brow colour)
PIP_HALF_WIDTH = 1.0
PIP_HALF_HEIGHT = 0.7
# a real pip sits clear of the eyelid with black between: 1.3 (projection's
# value) put its lower edge on the lid, so it rides higher here
PIP_RISE = 1.9
# the cheek starts this many eye radii below the eye centre. On a shiba the
# cheek's light puff begins right beside the eye and spreads back toward the
# ear, so only the lower lid's rim stays with the skull. 0.8 left the whole
# side of the face black and the cheeks a small patch on the jaw.
CHEEK_BELOW_EYE = 0.2
# ... and beside the eye it rises to this many eye radii ABOVE the eye
# centre: the cheek puff of a shiba stands level with the eye and spreads
# back to the ear, outboard of the eye's outer corner (CHEEK_OUTBOARD radii)
CHEEK_BESIDE_EYE = 0.5
CHEEK_OUTBOARD = 0.8
# ... and reaches back to the ear base, no further
CHEEK_BEHIND_EAR = 0.02
# the top of the muzzle between the eyes and the nose stays with the skull
# (the tan-point's black nose bar is wider than the thin muzzle_bridge tag):
# the muzzle texels that face UP, within this half-width of the midline in
# eye distances. The sides of the muzzle face sideways and are cheek.
MUZZLE_TOP_HALF_WIDTH = 0.25
MUZZLE_TOP_FACING_UP = 0.3
# the eye's radius as a share of the eye distance. The socket feature island
# is the whole eyelid rim, half again the eye's size, so it is not used.
EYE_RADIUS = 0.2
FEATHER_PX = 2


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8), mode="L").save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default="shiba_small")
    parser.add_argument("--glb", default="output/skeleton_shiba_small.glb")
    parser.add_argument("--geometry", default="output/geometry")
    parser.add_argument("--dry-run", action="store_true", help="write to masks_refined_face/ only")
    parser.add_argument("--normal-blur", type=float, default=0.0, help="see refine_masks.py")
    args = parser.parse_args()

    template_dir = Path("output/templates") / args.template
    masks_dir = template_dir / "masks"
    source_dir = template_dir / "masks_source"
    manifest = json.loads((Path(args.geometry) / "geometry.json").read_text(encoding="utf-8"))
    size = tuple(manifest["size"])
    position = np.load(Path(args.geometry) / manifest["position"]["file"]).astype(np.float32)
    normal = np.load(Path(args.geometry) / manifest["normal"]["file"]).astype(np.float32)
    valid = np.load(Path(args.geometry) / manifest["valid"]["file"]) > 0.5
    if args.normal_blur > 0:
        from refine_masks import smooth_normals
        normal = smooth_normals(normal, valid, args.normal_blur)

    from api.mesh import TemplateMesh
    joints = TemplateMesh.load(args.glb, size).joints

    source = {name: load_mask(source_dir, name, size) for name in FACE}
    features = load_mask(masks_dir, "_features", size)
    face = np.zeros(valid.shape, dtype=bool)
    for name in FACE:
        face |= source[name] > 0.5
    face &= valid & ~(features > 0.5)
    bridge = (source["muzzle_bridge"] > 0.5) & face
    free = face & ~bridge

    # the eyes: landmarks.json (build_landmarks.py verified them against the
    # feature islands)
    landmarks = json.loads((template_dir / "landmarks.json").read_text(encoding="utf-8"))
    eye_points = [np.array(landmarks["points"][k]["world"], dtype=np.float32) for k in ("left_eye", "right_eye")]
    eye_dist = float(landmarks.get("eye_distance_world") or np.linalg.norm(eye_points[0] - eye_points[1]))
    nose = np.array(landmarks["points"]["nose"]["world"], dtype=np.float32)
    eyes = [(centre, EYE_RADIUS * eye_dist) for centre in eye_points]
    ear_y = float(np.mean([joints[name][1] for name in EAR_BASE_JOINTS]))

    x, y, z = position[..., 0], position[..., 1], position[..., 2]
    brows = np.zeros(valid.shape, dtype=bool)
    for centre, r in eyes:
        dx = (x - centre[0]) / (PIP_HALF_WIDTH * r)
        dz = (z - (centre[2] + PIP_RISE * r)) / (PIP_HALF_HEIGHT * r)
        near = np.abs(y - centre[1]) < 3.0 * r          # the front of the head only
        brows |= (dx * dx + dz * dz <= 1.0) & near
    brows &= free

    eye_z = float(np.mean([c[2] for c, _ in eyes]))
    eye_y = float(np.mean([c[1] for c, _ in eyes]))
    eye_r = float(np.mean([r for _, r in eyes]))
    muzzle_top = ((np.abs(x) < MUZZLE_TOP_HALF_WIDTH * eye_dist) & (y < eye_y - 0.01)
                  & (z > nose[2] - 0.01) & (normal[..., 2] > MUZZLE_TOP_FACING_UP))
    eye_x = float(np.mean([abs(c[0]) for c, _ in eyes]))
    below = z < eye_z - CHEEK_BELOW_EYE * eye_r
    beside = (np.abs(x) > eye_x + CHEEK_OUTBOARD * eye_r) & (z < eye_z + CHEEK_BESIDE_EYE * eye_r)
    cheeks = (free & ~brows & (below | beside) & (y < ear_y + CHEEK_BEHIND_EAR) & ~muzzle_top)
    skull = free & ~brows & ~cheeks

    print(f"[face] eyes at z {eye_z:.3f}, eye radius {eye_r:.3f}, ear base y {ear_y:.3f}")
    print(f"[face] texels: brows {int(brows.sum())}, cheeks {int(cheeks.sum())}, skull {int(skull.sum())}, "
          f"bridge {int(bridge.sum())} (was: brows {int((source['eyebrows'] > 0.5).sum())}, "
          f"cheeks {int((source['cheeks'] > 0.5).sum())}, skull {int((source['skull'] > 0.5).sum())})")

    # feather and pad past the islands from the nearest island texel, as
    # build_template and refine_masks do
    indices = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    new = {}
    for name, hard in (("eyebrows", brows), ("cheeks", cheeks), ("skull", skull)):
        # outside the face the old mask (ears_outer merged into skull etc.)
        # is kept: only face texels were reassigned
        keep = np.where(face, 0.0, source[name])
        mask = np.maximum(keep, hard.astype(np.float32))
        mask = mask[indices[0], indices[1]]
        mask = ndimage.gaussian_filter(mask, FEATHER_PX / 2.0)
        new[name] = np.clip(mask, 0, 1)

    if args.dry_run:
        out = template_dir / "masks_refined_face"
        out.mkdir(parents=True, exist_ok=True)
        for name, mask in new.items():
            save_mask(out / f"{name}.png", mask)
        print(f"[face] dry run: wrote {sorted(new)} to {out}")
        return

    backup = template_dir / "masks_face_hand"
    if not backup.exists():
        backup.mkdir(parents=True)
        for name in new:
            shutil.copy2(source_dir / f"{name}.png", backup / f"{name}.png")
        print(f"[face] hand-tagged face masks backed up to {backup}")
    for name, mask in new.items():
        save_mask(source_dir / f"{name}.png", mask)

    # the ten-region union
    for region in REGIONS:
        parts = [np.asarray(Image.open(source_dir / f"{s}.png").convert("L"), dtype=np.uint8) for s in SOURCES[region]]
        Image.fromarray(np.max(np.stack(parts), axis=0), mode="L").save(masks_dir / f"{region}.png")
    print(f"[face] wrote {len(new)} face source masks and rebuilt {len(REGIONS)} region masks in {masks_dir}")


if __name__ == "__main__":
    main()
