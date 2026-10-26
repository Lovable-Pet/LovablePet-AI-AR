"""Render the pattern anchors: one picture of the template per coat pattern.

The pattern classifier (api/pattern_classify.py) shows the model the shelter
photo next to these five renders and asks which layout the photo matches.
Each anchor is the template wearing that pattern's DEFAULT layout from
patterns.py, in a representative pair of colours, rendered by
render_preview.py in a standing pose so the legs, chest and belly are all
in view. They are a picture of a layout, not of a breed: the colours are
chosen to make the layout legible, nothing more.

Re-run after any change to the compositor or to PATTERN_LAYOUT, so the
anchors show what the pipeline would actually build for that answer.

Usage:
    python ar_pipeline/render_anchors.py [--template shiba_small] [--only tan_point]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.assets import BLENDER, template_paths  # noqa: E402
from api.glb import extract_base_texture, replace_base_texture  # noqa: E402
from api.patterns import instantiate_defaults  # noqa: E402
from api.texture import synthesize_texture  # noqa: E402

# (base, second) that make each layout legible
ANCHOR_COLOURS: dict[str, tuple[str, str | None]] = {
    "tan_point": ("black", "tan"),
    "urajiro": ("red", "cream"),
    "irish": ("black", "white"),
    "saddle": ("tan", "black"),
    "solid": ("fawn", None),
}

CLIP = "standing"
# closer than the default preview: the anchor is about the layout, not the scene
ZOOM = 1.7


def render_anchor(pattern: str, template: str, anchors_dir: Path, work_dir: Path) -> Path:
    base, second = ANCHOR_COLOURS[pattern]
    appearance = {
        "pet_id": f"anchor_{pattern}",
        "base_template": f"skeleton_{template}.glb",
        "base_color": base,
        "pattern": pattern,
        "region_colors": instantiate_defaults(pattern, base, second),
        "markings": [],
    }
    template_glb, masks_dir = template_paths(appearance["base_template"])
    work_dir.mkdir(parents=True, exist_ok=True)
    base_path = work_dir / f"{pattern}_base.png"
    extract_base_texture(template_glb).save(base_path)

    coat = synthesize_texture(base_path, masks_dir, appearance)
    coat.save(work_dir / f"{pattern}_texture.png")
    glb_path = replace_base_texture(template_glb, coat, work_dir / f"{pattern}.glb")

    anchors_dir.mkdir(parents=True, exist_ok=True)
    out = anchors_dir / f"{pattern}.png"
    subprocess.run([
        BLENDER, "--background", "--python", "ar_pipeline/render_preview.py", "--",
        str(glb_path), str(out), "--clip", CLIP, "--zoom", str(ZOOM), "--angle", "40",
    ], check=True, capture_output=True)
    print(f"[anchors] {pattern}: {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default="shiba_small")
    parser.add_argument("--only", default=None)
    args = parser.parse_args()

    anchors_dir = Path("output/templates") / args.template / "anchors"
    work_dir = Path("output/_anchors_work")
    patterns = [args.only] if args.only else list(ANCHOR_COLOURS)
    for pattern in patterns:
        render_anchor(pattern, args.template, anchors_dir, work_dir)


if __name__ == "__main__":
    main()
