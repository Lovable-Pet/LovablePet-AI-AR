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
    renormalised.

Usage:
    python -m ar_pipeline.open_face_masks --slug jindo --size small
Back up output/templates/<slug>_<size>/masks first; this rewrites it.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", required=True)
    parser.add_argument("--size", default="small")
    args = parser.parse_args()
    template = ROOT / "output" / "templates" / f"{args.slug}_{args.size}"
    masks_dir = template / "masks"
    position = np.load(template / "geometry" / "position.npy")
    valid = np.load(template / "geometry" / "valid.npy") > 0.5
    points = json.loads((template / "landmarks.json").read_text(encoding="utf-8"))["points"]
    left, right = np.array(points["left_eye"]["world"]), np.array(points["right_eye"]["world"])
    eye = (left + right) / 2.0
    e = float(np.linalg.norm(left - right))

    names = sorted(p.stem for p in masks_dir.glob("*.png") if not p.stem.startswith("_"))
    masks = {n: np.asarray(Image.open(masks_dir / f"{n}.png").convert("L"), dtype=np.float32) / 255.0 for n in names}
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
    wide = valid & (ax > MUZZLE_HALF_E * e)
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
    print(f"[open_face_masks] E {e * 100:.1f} cm; spectacle texels to cheeks {moved_spectacle}; "
          f"muzzle_bridge to cheeks {moved_bridge[0]}, to skull {moved_bridge[1]}")


if __name__ == "__main__":
    main()
