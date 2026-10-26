"""Our templates side by side, plain grey, lettered A-F: the picture the
body-shape call (body_shape.py) compares a dog's photograph with to say
which of our models it looks like, or that none would pass for it.

The question is the user's (2026-10-18): built from that model and painted
in its coat, would this read as the dog, without looking forced? The grey
keeps the comparison on the outline - ears, coat, legs, head, muzzle - and
off the colours, which are painted later. Each vote of the call sees its
own letter order (ORDERS), so a habit of picking the first or the middle
picture cannot pass for an answer.

    python -m api.model_sheet          # render the greys (Blender) and the sheets
    python -m api.model_sheet --force  # again, after a template changed

Files: output/model_sheet/<slug>.png (one grey render each) and
sheet_<n>.png, beside the templates and, like them, not in git.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TEMPLATES = ["jindo", "jindo_puppy", "maltese", "toypoodle", "chihuahua", "retriever"]
# what a model can change beyond its grey outline, told to the call
CAN_CHANGE = {"jindo": "its ears can also be tipped or dropped, its tail lowered, its build slimmer or "
                       "stockier and its muzzle shorter or longer"}
LETTERS = "ABCDEF"
# one letter order per vote; the first is the list order
ORDERS = [[0, 1, 2, 3, 4, 5], [3, 5, 0, 4, 1, 2], [4, 2, 5, 1, 3, 0]]
SHEET_DIR = Path("output") / "model_sheet"
CELL = 340
DETAIL = "low"   # one 512 px tile, 85 tokens: the outlines read at that size


def render_greys(blender: str, force: bool = False) -> None:
    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    for slug in TEMPLATES:
        out = SHEET_DIR / f"{slug}.png"
        if out.exists() and not force:
            continue
        glb = Path("output") / f"skeleton_{slug}_small.glb"
        # three-quarter front, standing: the ears, face and legs in one view
        subprocess.run([blender, "--background", "--python", "ar_pipeline/render_preview.py", "--", str(glb), str(out),
                        "--clip", "standing", "--zoom", "2.0", "--angle", "50", "--height", "0.1", "--grey"],
                       check=True, capture_output=True)
        print(f"[model_sheet] {out}")


def _cell(slug: str) -> Image.Image:
    image = Image.open(SHEET_DIR / f"{slug}.png").convert("RGBA")
    box = image.getchannel("A").point(lambda a: 255 if a > 8 else 0).getbbox()
    image = image.crop(box)
    scale = (CELL - 40) / max(image.size)
    image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)
    cell = Image.new("RGB", (CELL, CELL), "white")
    cell.paste(image, ((CELL - image.width) // 2, CELL - image.height - 6), image)
    return cell


def sheet(n: int) -> tuple[Path, dict[str, str]]:
    """Sheet n (letter order ORDERS[n % len]) and its letter -> template map."""
    order = ORDERS[n % len(ORDERS)]
    letters = {LETTERS[i]: TEMPLATES[t] for i, t in enumerate(order)}
    path = SHEET_DIR / f"sheet_{n % len(ORDERS)}.png"
    if not path.exists():
        canvas = Image.new("RGB", (CELL * 3, CELL * 2), "white")
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("arialbd.ttf", 54)
        except OSError:
            font = ImageFont.load_default(size=54)
        for i, letter in enumerate(LETTERS):
            x, y = (i % 3) * CELL, (i // 3) * CELL
            canvas.paste(_cell(letters[letter]), (x, y))
            draw.text((x + 12, y + 6), letter, fill=(200, 30, 30), font=font)
            draw.rectangle([x, y, x + CELL - 1, y + CELL - 1], outline=(190, 190, 190), width=2)
        canvas.save(path)
    return path, letters


def describe(letters: dict[str, str]) -> str:
    """The text that goes with a sheet: what the letters show and what each
    model can still change."""
    notes = [f"{letter}: {CAN_CHANGE[slug]}." for letter, slug in letters.items() if slug in CAN_CHANGE]
    return ("The second image shows our six 3D dog models, A to F, in plain grey. The coat colours and markings "
            "are painted on afterwards, and the model is resized to the dog's weight, so judge only the shape: "
            "ears, coat length and texture, legs, head, muzzle, tail, body proportions. "
            + " ".join(notes))


def main() -> None:
    from .assets import BLENDER

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="render the greys and the sheets again")
    args = ap.parse_args()
    if args.force:
        for old in SHEET_DIR.glob("sheet_*.png"):
            old.unlink()
    render_greys(BLENDER, force=args.force)
    for n in range(len(ORDERS)):
        path, letters = sheet(n)
        print(f"[model_sheet] {path}: {letters}")


if __name__ == "__main__":
    main()


__all__ = ["DETAIL", "ORDERS", "TEMPLATES", "describe", "sheet"]
