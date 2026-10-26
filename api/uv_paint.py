"""Paint mode - the image model repaints the template's own UV texture.

WHY
---
The mask build colours ten fixed regions, and the template's masks do
not follow the anatomy a coat follows: the "chest" is the shoulder fronts,
the "underside" runs down the inside of every leg. Every rule added to
compensate (chest only where facing forward, belly only so far up the
flank, legs excluded from the belly...) fixed one animal and left another
with a white spot on its rump. The person looking at the model sees the
masks.

The template ships a hand-painted UV texture of a shiba: real fur, real
markings, natural boundaries, painted by someone who knew where a dog's
colours go. This mode asks an image-editing model to REPAINT that texture
in the target coat - "this red shiba's texture, made black-and-tan with
these features" - keeping every shape and position exactly. The edit model
keeps the layout because it is told to and because the input is the layout;
what it changes is the colour of the fur, which is what we want changed.

WHAT STAYS OURS
---------------
The geometry, rig, UVs and masks are untouched. The eyes, nose, pads and
mouth come back from the template through protect.png. The UV seams go
through the seam post-process. The result is checked against the UV
layout: an output whose painted islands do not sit where the template's
are (the model drew a dog instead of repainting the atlas) is rejected and
the mask build stands in. The critic runs in the feature vocabulary as in
simple mode and, if it finds a difference, the repaint is asked for once
more with the correction in the prompt.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from .fur import to_linear, to_srgb
from .imagegen import ImageTransport
from .simple import FEATURES

EDIT_SIZE = "1024x1024"
# The repainted atlas must keep the template's island outlines at least this
# well (the "did it draw a dog instead" check) ...
LAYOUT_IOU_FLOOR = 0.7
# ... and its eyeballs, nose and mouth - the parts it was told not to touch -
# must still sit where the template's are: structural similarity of the
# luminance inside those islands, the alignment check that catches a shifted
# or rescaled atlas the outline check lets through. Measured: the template
# 1.0, real repaints 0.61-0.91, the atlas shifted 20 px 0.38, rescaled 5%
# 0.10, a drawing of a dog 0.00.
FEATURE_SSIM_FLOOR = 0.5
CACHE_DIR = Path("output/cache/uv_paint")

COLOUR_WORDS = {
    "black": "black", "white": "white", "cream": "cream", "fawn": "pale fawn", "red": "red-orange",
    "tan": "tan", "chocolate": "dark chocolate brown", "gray": "grey",
    "pink": "pale pink bare skin",
}


def describe_coat(appearance: dict[str, Any]) -> str:
    """The target coat in the words a person would use, from the pattern,
    the two colours and the features."""
    base = COLOUR_WORDS.get(appearance.get("base_color", ""), appearance.get("base_color", ""))
    second = appearance.get("second_color")
    second_word = COLOUR_WORDS.get(second, second) if second else None
    pattern = appearance.get("pattern", "solid")
    lines = [f"Base coat colour: {base}."]
    layouts = {
        "tan_point": f"A {base} dog with {second_word or 'tan'} points. Where each colour goes: "
                     f"{base} on the back, sides, top of the head, outer ears, muzzle top, tail and the "
                     f"outside of the upper legs; {second_word or 'tan'} as a spot above each eye, on the "
                     f"cheeks and the sides of the muzzle, on the front of the lower legs, and as a patch on "
                     f"the front of each shoulder. All three colours must be present and each clearly its "
                     f"own: the {second_word or 'tan'} is a warm mid brown, neither faint nor covering the "
                     f"white areas listed in the features below.",
        "urajiro": f"A {base} dog with {second_word or 'cream'} on the cheeks, throat, chest, belly, "
                   f"inner legs and underside of the tail (urajiro).",
        "irish": f"A {base} dog with a {second_word or 'white'} collar, chest, belly, paws and tail tip.",
        "saddle": f"A {base} dog with a {second_word or 'black'} saddle over the back and tail top and a dark muzzle.",
        "solid": f"A solid {base} dog, one colour all over.",
        "brindle": f"A {base} dog with {second_word or 'black'} brindle stripes across the body.",
        "merle": f"A {base} dog with {second_word or 'dark'} merle mottling.",
        "other": f"A {base} dog" + (f" with {second_word} patches." if second_word else "."),
    }
    lines.append(layouts.get(pattern, layouts["other"]))
    features = [FEATURES[name][0] for name in appearance.get("features") or [] if name in FEATURES]
    if features:
        lines.append("Features this dog has, each painted plainly: " + "; ".join(features) + ".")
    return " ".join(lines)


def build_prompt(appearance: dict[str, Any], corrections: list[str] | None = None,
                 notes: list[str] | None = None) -> str:
    coat = describe_coat(appearance)
    fixes = ""
    if notes:
        fixes += " About the photograph: " + " ".join(notes)
    if corrections:
        fixes += " Corrections from a review of the previous attempt: " + " ".join(corrections)
    return (
        "The first image is the UV texture atlas of a 3D dog model, UNCOLOURED: neutral grey fur laid out "
        "flat in pieces (head, body, legs, tail, ears, paws), carrying only the hair direction and stroke "
        "texture, with the eyes, nose, mouth and paw pads already finished. The second image is a "
        "photograph of the actual dog. The third image is a COLOUR GUIDE: the same atlas, same pieces in "
        "the same places, with each area filled flat in the colour the fur there must become.\n\n"
        "Colour the fur of the first image so that every area takes the colour the guide shows for that "
        "area - the guide decides where each colour goes; the photograph decides the exact shade and the "
        "fur's look. The grey carries no colour information: never keep it grey, and never let one area "
        "be darker or lighter than the guide says because of the grey's shading. Keep the first image's "
        "fur texture, hair direction and detail; do not copy the guide's flatness or its soft edges.\n\n"
        f"Target coat, for reference: {coat}\n\n"
        "Rules: keep every piece exactly where it is, at exactly the same size and outline - the output "
        "must line up with the input pixel for pixel, or it will not fit the model. Do not draw a dog, "
        "do not add a background, do not move, rotate or redraw any piece. Keep the eyes, nose, mouth "
        "and paw pads as they are. Change only the colour of the fur. Fur boundaries between colours "
        "are clear but soft at the scale of individual hairs, as on a real coat. Style: clean, "
        "slightly illustrated fur - even lighting, no shadows, no highlights, gentle natural colour, "
        "not oversaturated."
        + fixes
    )


def guide_texture(appearance: dict[str, Any], masks_dir: str | Path, valid: np.ndarray,
                  size: tuple[int, int], palette=None, base_texture: Image.Image | None = None,
                  targets: dict[str, Any] | None = None) -> Image.Image:
    """The target coat as a flat colour map on the atlas: the template's own
    texture with every coat region filled flat in its colour's mid stop.
    Eyes, tongue, pads and the atlas background stay the template's, so the
    guide reads as "this atlas, these colours". Three prompts in a row put
    the same words in different places on the dog; a picture of where each
    colour goes is a reference an image model actually follows.

    `targets` (region -> LAB), when given, overrides the palette word for
    that region: the guide then shows the colour the combination step will
    score against, so the model is asked for exactly what it is judged on."""
    from .fur import FUR_PALETTE, hex_to_linear
    from .projection import lab_to_linear
    from .texture import MaskLibrary

    masks = MaskLibrary(masks_dir, size)
    if base_texture is not None:
        fill = to_linear(np.asarray(base_texture.convert("RGB").resize(size), dtype=np.float32))
    else:
        fill = np.full((size[1], size[0], 3), 0.8, dtype=np.float32)
    # the eyes, nose, mouth and pads are not fur: they keep the template's
    # pixels in the guide, so the model is never told the eyeballs are tan
    protect = masks.extra("protect")
    keep = (1.0 - protect) if protect is not None else np.ones(valid.shape, dtype=np.float32)
    for entry in appearance.get("region_colors") or []:
        mask = masks.get(entry.get("region"))
        colour = entry.get("color")
        if mask is None or colour not in FUR_PALETTE:
            continue
        if targets and entry.get("region") in targets:
            lab = np.asarray(targets[entry["region"]], dtype=np.float32).reshape(1, 1, 3)
            rgb = np.clip(lab_to_linear(lab), 0.0, 1.0)[0, 0]
        else:
            rgb = hex_to_linear((palette or {}).get(colour, FUR_PALETTE[colour])[1])
        weight = (np.clip(ndimage.gaussian_filter(mask, 2.0), 0, 1) * valid * keep)[..., None]
        fill = fill * (1 - weight) + rgb * weight
    return Image.fromarray((to_srgb(np.clip(fill, 0, 1)) * 255).astype(np.uint8), mode="RGB")


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def layout_iou(painted: Image.Image, valid: np.ndarray, base: Image.Image) -> float:
    """How well the painted atlas keeps the template's layout: the share of
    the template's island-edge pixels that are also edges in the painting.
    Colours may change freely; the outlines of the pieces may not. Judged
    at 512 px so a few pixels of drift do not count and a redrawn atlas
    does. 1.0 for the template itself."""
    size = (512, 512)
    small_valid = np.asarray(Image.fromarray((valid * 255).astype(np.uint8)).resize(size, Image.BILINEAR)) > 127
    boundary = small_valid ^ ndimage.binary_erosion(small_valid, iterations=1)
    band = ndimage.binary_dilation(boundary, iterations=3)

    def edges(image: Image.Image) -> np.ndarray:
        gray = np.asarray(image.convert("L").resize(size, Image.LANCZOS), dtype=np.float32) / 255.0
        magnitude = np.hypot(ndimage.sobel(gray, axis=0), ndimage.sobel(gray, axis=1))
        threshold = np.percentile(magnitude[band], 40)
        return magnitude > max(threshold, 0.05)

    base_edges = edges(base) & band
    paint_edges = ndimage.binary_dilation(edges(painted), iterations=1) & band
    if base_edges.sum() == 0:
        return 0.0
    return float((base_edges & paint_edges).sum() / base_edges.sum())


def feature_alignment(painted: Image.Image, base: Image.Image, masks_dir: str | Path) -> float | None:
    """SSIM between the repaint and the template inside the whole protected
    feature islands (eyeballs, nose leather, mouth), at the atlas size.
    Colour is free there too, so luminance is normalised per island."""
    from skimage.metrics import structural_similarity

    from .eyes import feature_islands, WHOLE_SHARE

    size = base.size
    protect_path = Path(masks_dir).parent / "protect.png"
    if not protect_path.exists():
        return None
    protect = np.asarray(Image.open(protect_path).convert("L").resize(size)) > 127
    _, labels, count = feature_islands(masks_dir, size)
    a = np.asarray(painted.convert("L").resize(size, Image.LANCZOS), dtype=np.float32) / 255.0
    b = np.asarray(base.convert("L"), dtype=np.float32) / 255.0
    scores, weights = [], []
    for index in range(1, count + 1):
        island = labels == index
        if island.sum() < 500 or protect[island].mean() < WHOLE_SHARE:
            continue
        rows, cols = np.nonzero(island)
        y0, y1, x0, x1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
        wa, wb = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
        wa = (wa - wa.mean()) / (wa.std() + 1e-3)
        wb = (wb - wb.mean()) / (wb.std() + 1e-3)
        rng = float(max(wa.max() - wa.min(), wb.max() - wb.min(), 1e-3))
        score, full = structural_similarity(wa, wb, data_range=rng, full=True)
        scores.append(float(full[island[y0:y1, x0:x1]].mean()))
        weights.append(int(island.sum()))
    if not scores:
        return None
    return float(np.average(scores, weights=weights))


@dataclass
class PaintResult:
    image: Image.Image | None
    accepted: bool
    report: dict[str, Any] = field(default_factory=dict)


async def paint_uv(
    transport: ImageTransport,
    base_texture: Image.Image,
    photo_path: str | Path,
    appearance: dict[str, Any],
    valid: np.ndarray,
    corrections: list[str] | None = None,
    cache_dir: str | Path = CACHE_DIR,
    use_cache: bool = True,
    guide: Image.Image | None = None,
    masks_dir: str | Path | None = None,
    notes: list[str] | None = None,
) -> PaintResult:
    """One repaint of the atlas. Returns the texture at the template's size
    when the layout check passes, else None."""
    prompt = build_prompt(appearance, corrections, notes)
    pet_id = str(appearance.get("pet_id") or "pet")
    guide_bytes = _png(guide.resize((1024, 1024), Image.LANCZOS)) if guide is not None else b""
    key = hashlib.sha256((prompt + pet_id).encode("utf-8") + hashlib.sha256(guide_bytes).digest()).hexdigest()[:16]
    cache_dir = Path(cache_dir)
    cached = cache_dir / f"{pet_id}-{key}.png"
    report: dict[str, Any] = {"prompt": prompt, "cache": str(cached)}

    if use_cache and cached.exists():
        painted = Image.open(cached).convert("RGB")
        report["source"] = "cache"
    else:
        small = base_texture.convert("RGB").resize((1024, 1024), Image.LANCZOS)
        photo = Image.open(photo_path).convert("RGB")
        if max(photo.size) > 1024:
            scale = 1024 / max(photo.size)
            photo = photo.resize((int(photo.size[0] * scale), int(photo.size[1] * scale)), Image.LANCZOS)
        references = [_png(small), _png(photo)] + ([guide_bytes] if guide_bytes else [])
        data = await transport.edit(prompt=prompt, references=references, size=EDIT_SIZE)
        painted = Image.open(io.BytesIO(data)).convert("RGB")
        report["source"] = "model"
        if use_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            painted.save(cached)

    iou = layout_iou(painted, valid, base_texture)
    report["layout_iou"] = round(iou, 3)
    if iou < LAYOUT_IOU_FLOOR:
        report["rejected"] = f"layout IoU {iou:.2f} < {LAYOUT_IOU_FLOOR}: the model did not keep the atlas"
        return PaintResult(None, False, report)
    alignment = feature_alignment(painted, base_texture, masks_dir) if masks_dir is not None else None
    report["feature_ssim"] = None if alignment is None else round(alignment, 3)
    if alignment is not None and alignment < FEATURE_SSIM_FLOOR:
        report["rejected"] = f"feature SSIM {alignment:.2f} < {FEATURE_SSIM_FLOOR}: eyes/nose moved"
        return PaintResult(None, False, report)
    full = painted.resize(base_texture.size, Image.LANCZOS)
    return PaintResult(full, True, report)


# The band along an island's edge where the image model tends to draw its own
# outline, and the band just inside it that says what the coat there really is.
EDGE_BAND_PX = 3
EDGE_REFERENCE_PX = (4, 10)
# The rim correction never moves luminance further than this.
EDGE_MAX_L = 10.0


def flatten_island_rims(composed: np.ndarray, valid: np.ndarray, fur: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove the systematic light or dark rim the image model draws around
    each UV island.

    Asked to repaint an atlas, the model treats every island as a shape and
    shades its border. On the model that border is a UV seam, so the shading
    renders as a line drawn across the coat. The correction is the MEDIAN
    offset of the outer three texels against the texels just inside them,
    taken over every island at once: a real coat does not change tone within
    three texels, so that offset is the artefact and nothing else. It is
    applied with a weight that falls to zero four texels in, and capped, so
    a marking that genuinely reaches the edge survives."""
    from .projection import lab_to_linear, linear_to_lab

    distance = ndimage.distance_transform_edt(valid)
    lab = linear_to_lab(np.clip(composed, 1e-4, 1.0))
    edge = fur & (distance <= EDGE_BAND_PX)
    inner = fur & (distance > EDGE_REFERENCE_PX[0]) & (distance <= EDGE_REFERENCE_PX[1])
    report: dict[str, Any] = {"edge_texels": int(edge.sum())}
    if edge.sum() < 1000 or inner.sum() < 1000:
        report["skipped"] = "too few texels"
        return composed, report
    reference = float(np.median(lab[inner][:, 0]))
    # one offset per texel of distance: the rim is a profile, not a step, and
    # a single number leaves the outermost texel still half wrong
    offsets = np.zeros(EDGE_BAND_PX + 2, dtype=np.float32)
    filled = np.zeros(EDGE_BAND_PX + 2, dtype=bool)
    for d in range(EDGE_BAND_PX + 2):
        band = fur & (distance >= d) & (distance < d + 1)
        if band.sum() > 200:
            offsets[d] = float(np.clip(reference - np.median(lab[band][:, 0]), -EDGE_MAX_L, EDGE_MAX_L))
            filled[d] = True
    # the outermost band can be empty (a texel touching the padding is already
    # at distance 1); leaving it at zero would drag its neighbour back down
    for d in range(len(offsets)):
        if not filled[d]:
            nearer = [k for k in range(len(offsets)) if filled[k]]
            if nearer:
                offsets[d] = offsets[min(nearer, key=lambda k: abs(k - d))]
    report["rim_profile_l"] = [round(float(-v), 2) for v in offsets]
    if float(np.abs(offsets).max()) < 1.0:
        report["applied"] = False
        return composed, report
    index = np.clip(distance, 0, len(offsets) - 1).astype(np.int32)
    correction = np.where(fur, offsets[index], 0.0).astype(np.float32)
    # a texel beyond the band gets nothing, and the step between bands is softened
    correction = np.where(distance > EDGE_BAND_PX + 1, 0.0, correction)
    lab[..., 0] = lab[..., 0] + ndimage.gaussian_filter(correction, 0.7)
    report["applied"] = True
    report["correction_l"] = [round(float(v), 2) for v in offsets]
    return np.clip(lab_to_linear(lab), 0.0, 1.0), report


def finish_painted(painted: Image.Image, base_texture: Image.Image, masks_dir: str | Path,
                   mesh_glb: str | Path | None, valid: np.ndarray,
                   eye_colour: str | None = None, flatten_rims: bool = True) -> tuple[Image.Image, dict[str, Any]]:
    """Protect the template's non-fur parts (eyeballs, nose, pads, mouth -
    not the fur around the eyes, which the repaint drew), recolour the iris
    to the dog's eye colour, and smooth the seams. `flatten_rims` False
    skips the island-rim correction: for a texture whose rims are already
    the mask build's (the lite path) that correction only measures the
    model's outline just inside the rim and pulls the rim toward it."""
    from .eyes import protect_for_paint, recolour_iris
    from .texture import MaskLibrary, _blur

    report: dict[str, Any] = {}
    composed = to_linear(np.asarray(painted, dtype=np.float32))
    template = to_linear(np.asarray(base_texture.convert("RGB"), dtype=np.float32))
    # outside the islands (the padding texture filtering reads across every
    # seam) every texel takes the nearest island texel's painted colour. The
    # template's own padding is the old coat's colour and bled in as an
    # orange rim along every seam.
    inside = ndimage.binary_erosion(valid, iterations=1)
    indices = ndimage.distance_transform_edt(~inside, return_distances=False, return_indices=True)
    composed = composed[indices[0], indices[1]]

    masks = MaskLibrary(masks_dir, base_texture.size)
    protect = masks.extra("protect")
    fur = valid & (protect < 0.5 if protect is not None else True)
    if flatten_rims:
        composed, report["rims"] = flatten_island_rims(composed, valid, fur)
    else:
        report["rims"] = {"skipped": "rims are the build's"}
    if protect is not None:
        keep = protect_for_paint(masks_dir, protect > 0.5).astype(np.float32)
        weight = np.maximum(_blur(keep, 1.5), keep)[..., None]
        composed = composed * (1 - weight) + template * weight
        report["protect_texels"] = int((keep > 0.5).sum())
        if eye_colour:
            composed, report["eyes"] = recolour_iris(composed, masks_dir, keep > 0.5, eye_colour)

    if mesh_glb is not None and Path(mesh_glb).exists():
        from .mesh import TemplateMesh
        from .seams import smooth_seams
        mesh = TemplateMesh.load(mesh_glb, base_texture.size)
        composed, report["seams"] = smooth_seams(np.clip(composed, 0, 1), mesh)

    return Image.fromarray((to_srgb(np.clip(composed, 0, 1)) * 255).astype(np.uint8), mode="RGB"), report


class ScriptedImageTransport:
    """Offline stand-in: returns the first reference unchanged, so the
    plumbing (cache, layout check, protect, seams, critic) runs without a
    key. The coat therefore stays the template's red shiba."""

    async def edit(self, *, prompt: str, references: list[bytes], size: str,
                   mask: bytes | None = None) -> bytes:
        return references[0]


__all__ = ["PaintResult", "ScriptedImageTransport", "build_prompt", "describe_coat", "finish_painted",
           "guide_texture", "layout_iou", "paint_uv"]
