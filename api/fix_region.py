"""Repaint one or more coat regions of a finished atlas, and nothing else.

    python -m api.fix_region --texture output/runs/combined/final/D2023072630.png \
        --appearance output/runs/live_paint4/build_round0/appearance.json \
        --regions legs_lower --out output/runs/fix_legs [--photo tests/fixtures/image.png]

The image model repainted the whole atlas four times and never put the tan
on the lower legs; asked for the whole coat it paints "roughly black-and-
tan" and the small areas fall where they may. Asked for ONE area, with the
rest of the atlas locked, it has one job. That is what the edit endpoint's
mask does: the atlas goes in whole, the mask marks the regions to change
(transparent = may change), and the model paints fur of the target colour
there, matched to the fur around it. Every other texel is then restored
from the input exactly, so a fix never disturbs what was already right.

One call per fix; the fix is cached like a repaint.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from .fur import FUR_PALETTE, to_linear, to_srgb
from .regions import REGIONS
from .uv_paint import COLOUR_WORDS, CACHE_DIR, ScriptedImageTransport, _png, finish_painted

EDIT_SIZE = "1024x1024"
# The editable area is grown this far past the region so the model can
# blend the new fur into the old along the edge; the paste-back feathers
# over the same width.
GROW_PX = 6


def fix_prompt(regions: list[str], colours: dict[str, str], appearance: dict[str, Any]) -> str:
    wants = "; ".join(
        f"{r.replace('_', ' ')}: {COLOUR_WORDS.get(colours[r], colours[r])} fur" for r in regions if r in colours
    )
    base = COLOUR_WORDS.get(appearance.get("base_color", ""), appearance.get("base_color", ""))
    return (
        "The first image is the UV texture atlas of a 3D dog model (a "
        f"{base} dog with markings). Only the transparent area of the mask may change. "
        f"In that area paint the dog's fur in this colour: {wants}. Match the fur texture, hair "
        "direction and lighting of the surrounding atlas so the new fur continues into the old "
        "without a visible edge; fur boundaries between colours are clear but soft at the scale "
        "of hairs. The second image is a photograph of the dog for the exact shade. Do not change "
        "anything outside the mask, do not move or redraw any piece, no background, no shadows."
    )


def region_mask(regions: list[str], masks_dir: Path, valid: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from .texture import MaskLibrary

    masks = MaskLibrary(masks_dir, size)
    union = np.zeros((size[1], size[0]), dtype=bool)
    for region in regions:
        mask = masks.get(region)
        if mask is not None:
            union |= mask > 0.5
    protect = masks.extra("protect")
    editable = ndimage.binary_dilation(union, iterations=GROW_PX) & valid
    if protect is not None:
        editable &= ~(protect > 0.5)
    return editable


def mask_png(editable: np.ndarray, size: tuple[int, int]) -> bytes:
    """RGBA mask for the edit endpoint: alpha 0 where the model may paint."""
    alpha = np.where(editable, 0, 255).astype(np.uint8)
    rgba = np.zeros(editable.shape + (4,), dtype=np.uint8)
    rgba[..., 3] = alpha
    image = Image.fromarray(rgba, mode="RGBA").resize(size, Image.NEAREST)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


async def fix_regions(transport, texture: Image.Image, photo_path: str | Path, appearance: dict[str, Any],
                      regions: list[str], masks_dir: str | Path, valid: np.ndarray,
                      cache_dir: str | Path = CACHE_DIR, use_cache: bool = True) -> tuple[Image.Image, dict[str, Any]]:
    """Returns the atlas with the regions repainted (unfinished) and a report."""
    masks_dir = Path(masks_dir)
    size = texture.size
    regions = [r for r in regions if r in REGIONS]
    colours = {e["region"]: e["color"] for e in appearance.get("region_colors") or [] if e.get("color") in FUR_PALETTE}
    prompt = fix_prompt(regions, colours, appearance)
    editable = region_mask(regions, masks_dir, valid, size)

    small = texture.convert("RGB").resize((1024, 1024), Image.LANCZOS)
    photo = Image.open(photo_path).convert("RGB")
    if max(photo.size) > 1024:
        scale = 1024 / max(photo.size)
        photo = photo.resize((int(photo.size[0] * scale), int(photo.size[1] * scale)), Image.LANCZOS)
    atlas_bytes, photo_bytes, mask_bytes = _png(small), _png(photo), mask_png(editable, (1024, 1024))

    pet_id = str(appearance.get("pet_id") or "pet")
    key = hashlib.sha256(prompt.encode("utf-8") + hashlib.sha256(atlas_bytes).digest()
                         + hashlib.sha256(mask_bytes).digest()).hexdigest()[:16]
    cached = Path(cache_dir) / f"{pet_id}-fix-{key}.png"
    report: dict[str, Any] = {"regions": regions, "prompt": prompt, "cache": str(cached),
                              "editable_share": round(float(editable.mean()), 4)}
    use_cache = use_cache and not isinstance(transport, ScriptedImageTransport)
    if use_cache and cached.exists():
        painted = Image.open(cached).convert("RGB")
        report["source"] = "cache"
    else:
        data = await transport.edit(prompt=prompt, references=[atlas_bytes, photo_bytes], size=EDIT_SIZE,
                                    mask=mask_bytes)
        painted = Image.open(io.BytesIO(data)).convert("RGB")
        report["source"] = "model"
        if use_cache:
            cached.parent.mkdir(parents=True, exist_ok=True)
            painted.save(cached)

    # paste back: only the editable area takes the model's pixels, feathered
    new = to_linear(np.asarray(painted.resize(size, Image.LANCZOS), dtype=np.float32))
    old = to_linear(np.asarray(texture.convert("RGB"), dtype=np.float32))
    weight = ndimage.gaussian_filter(editable.astype(np.float32), GROW_PX / 2.0)
    weight = np.minimum(weight, editable.astype(np.float32) * 1.0 + ndimage.gaussian_filter(editable.astype(np.float32), 1.0))
    out = old * (1 - weight[..., None]) + new * weight[..., None]
    # how much the model actually changed inside the mask (a no-op fix is worth knowing)
    changed = float(np.abs(new - old)[editable].mean())
    report["mean_change_in_mask"] = round(changed, 4)
    return Image.fromarray((to_srgb(np.clip(out, 0, 1)) * 255).astype(np.uint8), mode="RGB"), report


def main() -> None:
    from .glb import extract_base_texture, replace_base_texture
    from .pipeline import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--texture", required=True, help="the finished atlas to fix (PNG)")
    parser.add_argument("--appearance", required=True, help="appearance.json with the target region colours")
    parser.add_argument("--regions", required=True, help="comma-separated region names")
    parser.add_argument("--out", required=True)
    parser.add_argument("--photo", default="tests/fixtures/image.png")
    parser.add_argument("--template", default="output/skeleton_shiba_small.glb")
    parser.add_argument("--masks", default="output/templates/shiba_small/masks")
    parser.add_argument("--scripted", action="store_true", help="offline: the stand-in returns the input")
    args = parser.parse_args()

    texture = Image.open(args.texture).convert("RGB")
    appearance = json.loads(Path(args.appearance).read_text(encoding="utf-8"))
    valid = np.load("output/geometry/valid.npy") > 0.5
    if args.scripted:
        transport = ScriptedImageTransport()
    else:
        from .imagegen import OpenAIImageTransport
        transport = OpenAIImageTransport()
    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    fixed, report = asyncio.run(fix_regions(transport, texture, args.photo, appearance, regions, args.masks, valid))

    out = Path(args.out)
    (out / "final").mkdir(parents=True, exist_ok=True)
    fixed.save(out / "fixed_raw.png")
    base = extract_base_texture(args.template)
    coat, finish = finish_painted(fixed, base, args.masks, args.template, valid)
    pet_id = str(appearance.get("pet_id"))
    coat.save(out / "final" / f"{pet_id}.png")
    glb = replace_base_texture(args.template, coat, out / "final" / f"{pet_id}.glb")
    report["finish"] = finish
    (out / "fix_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[fix] {regions}: source={report['source']}, change in mask={report['mean_change_in_mask']}, -> {glb}")


if __name__ == "__main__":
    main()
