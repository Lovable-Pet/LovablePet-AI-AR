"""Reshape a template's face masks around the eyes and down the nose.

WHY
---
On the generated jindo the cheeks stopped in a straight line at eye level
and the band round each eye was skull, so a black-and-tan's cream
"spectacles" came out black (user on eval demo e08); the muzzle_bridge ran
+-2.4 cm wide from the nose to 5 cm above the eyes, so a dark muzzle top
blacked the sides of the muzzle too; and the brow spots were hard-edged
ovals that read as stickers.

HOW
---
All in 3D from the position map, in units of the eye distance E so the rule
carries to another template:
  - skull texels round each eye - beside and below it, and up to the brow
    line - are written as _spectacle.png, painted the cheeks' colour by the
    texture build when the cheeks are light and the dog is not piebald;
  - muzzle_bridge wider than MUZZLE_HALF_E either side of the midline goes
    to the cheeks below eye level and to the skull above; the strip above
    eye level goes to the skull as well (a blaze up the forehead is its own
    marking);
  - the four face masks are then smoothed on the surface (FACE_SIGMA_M) so
    their edges, the brows' included, are soft, and every mask is
    renormalised;
  - the cheek texels round and just under each eye are written as
    _eye_ring.png, painted the skull's colour by the texture build for a dog
    the head close-up saw dark round the eyes (EYE_RING_BELOW).

Usage:
    python -m ar_pipeline.open_face_masks --slug jindo --size small [--eye-ring-only]
Back up output/templates/<slug>_<size>/masks first; this rewrites it.
--eye-ring-only writes _eye_ring.png from the masks as they are and nothing else.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ar_pipeline.smooth_masks import smooth_channels  # noqa: E402

# the spectacle round each eye, in eye distances E
SPECTACLE_X = (0.2, 1.0)        # lateral, from the midline
SPECTACLE_BACK = 0.6            # behind the eye (toward the ears)
SPECTACLE_FRONT = 0.8           # in front of the eye (toward the nose)
SPECTACLE_BELOW = 0.45
SPECTACLE_ABOVE = 0.3           # up to the brow line
MUZZLE_HALF_E = 0.22            # the dark strip down the nose: +-1.4 cm on the jindo
ABOVE_EYES_E = 0.08
FACE_REGIONS = ("skull", "cheeks", "muzzle_bridge", "eyebrows")
FACE_SIGMA_M = 0.005
# A dog dark round the eyes (face_colours eye_ring "dark": a black shiba) has
# the head colour above, beside and just under each eye; its light cheeks
# start below the eye. The templates' cheeks reach eye level (59% of the
# jindo's texels up to 0.15 E above the eye centre are cheek), so the face
# within EYE_RING_REACH eye radii of an eye (across the face) and above
# EYE_RING_BELOW eye radii under its centre goes to _eye_ring.png, faded over
# EYE_RING_SOFT eye radii - all but the brow spots and the muzzle strip,
# which keep their own colours. Not the cheek mask times that: the cheeks
# are soft (0.8 at eye level), and the ring painted at 0.8 left the cream
# showing through grey.
EYE_RING_REACH = 3.0
EYE_RING_BELOW = 1.3
EYE_RING_SOFT = 0.4


def eye_ring(position: np.ndarray, valid: np.ndarray, masks: dict[str, np.ndarray], eyes: list[np.ndarray],
             radius: float) -> np.ndarray:
    """The face round and just under the eyes but the brows and the muzzle
    strip (EYE_RING_*), soft, the padding filled from the nearest island
    texel."""
    from scipy import ndimage
    ring = np.zeros(valid.shape, dtype=np.float32)
    soft = EYE_RING_SOFT * radius
    for eye in eyes:
        across = np.linalg.norm(position[..., :2] - eye[:2], axis=-1)
        near = np.clip((EYE_RING_REACH * radius - across) / soft + 0.5, 0.0, 1.0)
        high = np.clip((position[..., 2] - (eye[2] - EYE_RING_BELOW * radius)) / soft + 0.5, 0.0, 1.0)
        ring = np.maximum(ring, (near * high).astype(np.float32))
    own = np.clip(1.0 - masks["eyebrows"] - masks["muzzle_bridge"], 0.0, 1.0)
    ring = np.where(valid, ring * own, 0.0)
    reach = (EYE_RING_REACH + 2.0) * radius
    head = valid & (np.min([np.linalg.norm(position - eye, axis=-1) for eye in eyes], axis=0) < reach)
    ring = np.where(head, smooth_channels(ring[..., None], position, head, FACE_SIGMA_M, stride=2)[..., 0], ring)
    nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return np.clip(ring, 0.0, 1.0)[nearest[0], nearest[1]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", required=True)
    parser.add_argument("--size", default="small")
    parser.add_argument("--eye-ring-only", action="store_true", help="write _eye_ring.png only")
    args = parser.parse_args()
    template = ROOT / "output" / "templates" / f"{args.slug}_{args.size}"
    masks_dir = template / "masks"
    position = np.load(template / "geometry" / "position.npy")
    valid = np.load(template / "geometry" / "valid.npy") > 0.5
    points = json.loads((template / "landmarks.json").read_text(encoding="utf-8"))["points"]
    left, right = np.array(points["left_eye"]["world"]), np.array(points["right_eye"]["world"])
    eye = (left + right) / 2.0
    e = float(np.linalg.norm(left - right))
    # the muzzle's width goes with the size of the face, not with how far
    # apart the eyes sit: the eye distance the measured eye would have on
    # the jindo (refine_face_masks.eye_radius)
    from ar_pipeline.refine_face_masks import EYE_RADIUS, eye_radius
    face_e = eye_radius(template, position, valid, [left, right], e) / EYE_RADIUS

    names = sorted(p.stem for p in masks_dir.glob("*.png") if not p.stem.startswith("_"))
    masks = {n: np.asarray(Image.open(masks_dir / f"{n}.png").convert("L"), dtype=np.float32) / 255.0 for n in names}
    radius = face_e * EYE_RADIUS
    if args.eye_ring_only:
        ring = eye_ring(position, valid, masks, [left, right], radius)
        Image.fromarray((ring * 255 + 0.5).astype(np.uint8), mode="L").save(masks_dir / "_eye_ring.png")
        print(f"[open_face_masks] eye radius {radius * 100:.2f} cm; eye ring texels {int((ring > 0.5).sum())}")
        return
    x, y, z = position[..., 0], position[..., 1], position[..., 2]
    ax = np.abs(x)
    # blender space: -y is forward (the nose), z up
    near_eyes = (valid & (ax > SPECTACLE_X[0] * e) & (ax < SPECTACLE_X[1] * e)
                 & (y > eye[1] - SPECTACLE_FRONT * e) & (y < eye[1] + SPECTACLE_BACK * e)
                 & (z > eye[2] - SPECTACLE_BELOW * e) & (z < eye[2] + SPECTACLE_ABOVE * e))
    # the spectacle stays skull in the region masks and is written as its own
    # mask (_spectacle.png): texture.compose_body paints it the cheeks'
    # colour only for a light-cheeked dog that is not piebald - a piebald's
    # dark cap covers its eyes (eval e13, e16)
    spectacle = near_eyes * masks["skull"]
    moved_spectacle = int((spectacle > 0.5).sum())

    bridge = masks["muzzle_bridge"]
    above = z > eye[2] + ABOVE_EYES_E * e
    wide = valid & (ax > MUZZLE_HALF_E * face_e)
    to_cheek = bridge * (wide & ~above)
    to_skull = bridge * (valid & above)
    moved_bridge = (int((to_cheek > 0.5).sum()), int((to_skull > 0.5).sum()))
    masks["cheeks"] = masks["cheeks"] + to_cheek
    masks["skull"] = masks["skull"] + to_skull
    masks["muzzle_bridge"] = bridge - to_cheek - to_skull

    face = [n for n in FACE_REGIONS if n in masks]
    head = valid & (z > eye[2] - 3.0 * e) & (y < eye[1] + 3.0 * e)
    smoothed = smooth_channels(np.stack([masks[n] for n in face] + [spectacle], -1), position, head,
                               FACE_SIGMA_M, stride=2)
    for k, n in enumerate(face):
        masks[n] = np.where(head, smoothed[..., k], masks[n])
    spectacle = np.where(head, smoothed[..., len(face)], spectacle)
    total = sum(masks.values())
    # texels without geometry (the padding round each island, which texture
    # filtering reads across a seam) take the nearest island texel's value:
    # left as they were they kept the old muzzle strip as a zigzag line down
    # the forehead seam
    from scipy import ndimage
    nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    for n in names:
        masks[n] = np.where(valid & (total > 1e-4), masks[n] / np.maximum(total, 1e-4), masks[n])
        masks[n] = masks[n][nearest[0], nearest[1]]
        Image.fromarray(np.clip(masks[n] * 255 + 0.5, 0, 255).astype(np.uint8), mode="L").save(masks_dir / f"{n}.png")
    spectacle = np.clip(spectacle, 0.0, 1.0)[nearest[0], nearest[1]]
    Image.fromarray((spectacle * 255 + 0.5).astype(np.uint8), mode="L").save(masks_dir / "_spectacle.png")
    ring = eye_ring(position, valid, masks, [left, right], radius)
    Image.fromarray((ring * 255 + 0.5).astype(np.uint8), mode="L").save(masks_dir / "_eye_ring.png")
    print(f"[open_face_masks] E {e * 100:.1f} cm; spectacle texels to cheeks {moved_spectacle}; "
          f"muzzle_bridge to cheeks {moved_bridge[0]}, to skull {moved_bridge[1]}")


if __name__ == "__main__":
    main()
