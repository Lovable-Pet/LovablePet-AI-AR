"""Paint an iris, pupil, limbus and catch-light into a template's flat eyes.

The toy poodle's generated model came with each eye one flat grey disc
(L* 20, chroma 4) - no iris, no pupil, no highlight - and every poodle came
out with dead grey eyes (user, 11-02: "눈이 너무 죽었는데"). The other
templates' eyes carry a painted iris. Here the disc is found by the eye
landmark (the dark texels within 1.5 cm) and painted in rings on the
SURFACE: each texel's 3D distance from the disc's centre over its radius,
the centre taken from the disc's rim (the atlas packs a disc denser on one
side, and the texels' own mean sat off-centre). A dark brown iris: the
reading said dark_brown for nine of ten shelter poodles, and the small eye
island is not one eyes.recolour_iris recolours.

    python -m ar_pipeline.paint_eyes --slug toypoodle            # preview only
    python -m ar_pipeline.paint_eyes --slug toypoodle --install  # into the GLB and base_color.png

--install backs the GLB and base_color.png up to work/backup_eyes_<slug>/
first (once). Run it on the template's own flat eyes once: on painted eyes
the catch-light leaves a hole in the dark disc and the rings shift (toy
poodle: 135 texels). Done for toypoodle on 2026-11-02 (its originals in
work/backup_eyes_toypoodle/). The template assets go into the next release
(assets-v2).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from api.fur import to_linear, to_srgb
from api.glb import extract_base_texture, replace_base_texture
from api.landmarks import Landmarks
from api.projection import lab_to_linear, linear_to_lab

# ring radii as shares of the disc's, and the ramp between rings
PUPIL, IRIS = 0.40, 0.90
SOFT = 0.06
PUPIL_LAB = np.array([5.0, 0.0, 0.0])
IRIS_LAB = np.array([25.0, 11.0, 17.0])
LIMBUS_LAB = np.array([11.0, 3.0, 4.0])
HIGHLIGHT_LAB = np.array([97.0, 0.0, -1.0])
# the disc: texels darker than this within this distance of the eye landmark
DISC_MAX_L = 40.0
DISC_REACH_M = 0.015


def paint(slug: str) -> tuple[Image.Image, Image.Image, list[str]]:
    """(painted texture, before/after preview, report lines)."""
    tpl = Path(f"output/templates/{slug}_small")
    texture = extract_base_texture(Path(f"output/skeleton_{slug}_small.glb")).convert("RGB")
    lm = Landmarks.load(tpl, tpl / "geometry")
    pos, valid, normal = lm.position, lm.valid, lm.normal
    lab = linear_to_lab(to_linear(np.asarray(texture).astype(np.float32)))
    out_lab = lab.copy()
    rng = np.random.default_rng(7)
    boxes, report = [], []
    for side in ("left_eye", "right_eye"):
        near = valid & (np.linalg.norm(pos - np.array(lm.world(side)), axis=-1) < DISC_REACH_M)
        dark = near & (lab[..., 0] < DISC_MAX_L)
        labels, count = ndimage.label(dark)
        if not count:
            report.append(f"{side}: no dark disc")
            continue
        disc = labels == (np.argmax(ndimage.sum(dark, labels, range(1, count + 1))) + 1)
        rim = disc & ~ndimage.binary_erosion(disc)
        centre = pos[rim].mean(0)
        radius = float(np.median(np.linalg.norm(pos[rim] - centre, axis=-1)))
        rho = np.linalg.norm(pos - centre, axis=-1) / radius
        gaze = normal[disc].mean(0)
        gaze /= np.linalg.norm(gaze)
        up = np.array([0.0, 0.0, 1.0]) - gaze[2] * gaze
        up /= np.linalg.norm(up)
        across = np.cross(gaze, up)
        # the iris's fibres: noise by the angle round the gaze
        offset = pos - centre
        angle = np.arctan2(offset @ across, offset @ up)
        fibres = np.interp(angle, np.linspace(-np.pi, np.pi, 49), rng.normal(0, 1, 49))
        w_iris = np.clip((rho - PUPIL) / SOFT + 0.5, 0.0, 1.0)[..., None]
        w_limbus = np.clip((rho - IRIS) / SOFT + 0.5, 0.0, 1.0)[..., None]
        iris = IRIS_LAB + np.stack([3.5 * fibres + 4.0 * (rho - 0.6), 1.5 * fibres, 2.0 * fibres], -1)
        colour = PUPIL_LAB * (1 - w_iris) + iris * w_iris
        colour = colour * (1 - w_limbus) + LIMBUS_LAB * w_limbus
        # the catch-light: up and a little across from the centre
        spot = centre + radius * (0.42 * up + 0.22 * across)
        light = np.clip(1.0 - np.linalg.norm(pos - spot, axis=-1) / (0.16 * radius), 0.0, 1.0) ** 0.7
        colour = colour * (1 - light)[..., None] + HIGHLIGHT_LAB * light[..., None]
        out_lab[disc] = colour[disc]
        ys, xs = np.nonzero(disc)
        boxes.append((slice(max(ys.min() - 6, 0), ys.max() + 7), slice(max(xs.min() - 6, 0), xs.max() + 7)))
        report.append(f"{side}: {int(disc.sum())} texels, radius {radius * 1000:.1f} mm")
    rgb = to_srgb(lab_to_linear(out_lab.astype(np.float32)))
    painted = Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8))
    cells = []
    for box in boxes:
        for source in (np.asarray(texture), np.asarray(painted)):
            cell = Image.fromarray(source[box])
            cells.append(cell.resize((cell.width * 10, cell.height * 10), Image.NEAREST))
    preview = Image.new("RGB", (sum(c.width + 20 for c in cells) or 1, max((c.height for c in cells), default=1)), "white")
    x = 0
    for cell in cells:
        preview.paste(cell, (x, 0))
        x += cell.width + 20
    return painted, preview, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slug", required=True)
    parser.add_argument("--out", default=None, help="where the preview goes (default work/eyes_<slug>/)")
    parser.add_argument("--install", action="store_true", help="write into the template GLB and base_color.png")
    args = parser.parse_args()
    out = Path(args.out or f"work/eyes_{args.slug}")
    out.mkdir(parents=True, exist_ok=True)
    painted, preview, report = paint(args.slug)
    painted.save(out / "base_color.png")
    preview.save(out / "eyes_preview.png")
    print("\n".join(report), f"\npreview {out / 'eyes_preview.png'}")
    if args.install:
        glb = Path(f"output/skeleton_{args.slug}_small.glb")
        png = Path(f"output/templates/{args.slug}_small/base_color.png")
        backup = Path(f"work/backup_eyes_{args.slug}")
        if not backup.exists():
            backup.mkdir(parents=True)
            shutil.copy(glb, backup / glb.name)
            shutil.copy(png, backup / png.name)
        replace_base_texture(glb, painted, out / glb.name)
        shutil.copy(out / glb.name, glb)
        painted.save(png)
        print(f"installed into {glb} and {png} (backup {backup})")


if __name__ == "__main__":
    main()
