"""Combine several repaints of the atlas into one, region by region.

Four repaints of the same dog came back from the image model, each right
somewhere and wrong somewhere else: one had the white chest and belly, one
the tan legs, one the face. The model does not take a seed, so a fifth call
would be a fifth roll. Instead the repaints already paid for are combined:
for every coat region the candidate whose fur colour there is closest to
the target colour is taken, and the regions are blended along the masks
with a fur-level edge. Every texel is still a texel the image model
painted - real fur, natural boundaries inside a region - and no call is
made.

HOW STRONG A COLOUR IS
----------------------
A colour name says which ramp; it does not say how much. The region
reading (agents.py) also says whether the colour is faint, medium or
strong, and that share mixes the colour's candidate over the candidate
whose fur there is most neutral (the least chroma): a faint tan on a pale
leg is a third of the tan atlas over the pale one. When the photograph's own
measurement of the region is available (photo_colours.py) the mix is
solved to land on it instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from .fur import FUR_PALETTE, hex_to_linear, to_linear, to_srgb
from .regions import REGIONS

# A candidate's luminance gradient along the mask ring may be this many
# times its interior gradient before it counts as a mask-shaped fill; past
# that, each unit of the ratio costs this much LAB distance.
MASK_EDGE_FREE = 2.0
MASK_EDGE_WEIGHT = 10.0

# How much of the region's colour a reading's strength stands for, as the
# share of the colour candidate over the base-coat candidate.
STRENGTH_SHARE = {"faint": 0.35, "medium": 0.65, "strong": 1.0}
# When every candidate is further than this from a region's target, none
# of them painted that colour there; the region keeps the base candidate
# rather than taking the least-wrong colour (an orange shoulder for a white
# chest).
NO_CANDIDATE_LAB = 40.0
# Candidates within this much chroma of the least-chromatic one count as neutral fur.
NEUTRAL_CHROMA_SLACK = 6.0
# The edge where one candidate hands over to another is blended this wide
# (texels); a fur-level 3 px showed the seam when the two repaints' tones
# differed.
HANDOVER_PX = 8.0
# Before combining, every candidate is tone-matched to the base candidate
# over the regions both painted the same colour: the median L, a, b there
# are made equal. Two repaints of one prompt come back at slightly
# different exposures, and without this a region taken from the other
# repaint sits a shade off.
TONE_MATCH = True


def _lab(rgb_linear: np.ndarray) -> np.ndarray:
    from .projection import linear_to_lab
    return linear_to_lab(np.clip(rgb_linear, 1e-4, 1.0))


def _palette_lab(colour: str, palette=None) -> np.ndarray:
    return _lab(hex_to_linear((palette or {}).get(colour, FUR_PALETTE[colour])[1])[None, None, :])[0, 0]


# How far the photograph's hue may pull a region away from its palette word,
# in LAB a/b distance. A measurement is a small window the vision model
# pointed at, so it can land on grass or on a shadow's edge; beyond this it
# is not a shade of the named colour any more and is reeled back in.
#
# The allowance is PROPORTIONAL to the word's own chroma, because a word
# with little chroma has little to argue about. "tan" (chroma 43) covers
# everything from a pale wash to a deep red-gold, and that spread is the
# whole reason to measure. "white" (chroma 4.4) does not: a flat 22 let a
# white dog's back be pulled to b -9.5 by a pavement-and-sky grey card, and
# the guide then asked for a cold blue-grey coat. The model painted one.
MEASURED_AB_FLOOR = 3.0
MEASURED_AB_SHARE = 0.5


def _blend_measured(colour: str, lab: np.ndarray, palette=None) -> np.ndarray:
    """The target for a region the photograph was measured at: the palette
    word's LUMINANCE with the measurement's hue and chroma.

    Brightness is surface colour times illumination and a texture must carry
    only the first, so the photograph's L is dropped - copying it would bake
    the room's shadow into the albedo, where it would be lit again. Hue and
    relative chroma survive shading, so those are the photograph's to give,
    and they are what makes one dog's tan different from another's."""
    base = _palette_lab(colour, palette)
    ab = np.asarray(lab, dtype=np.float32)[1:3]
    delta = ab - base[1:3]
    distance = float(np.linalg.norm(delta))
    limit = MEASURED_AB_FLOOR + MEASURED_AB_SHARE * float(np.hypot(base[1], base[2]))
    if distance > limit:
        ab = base[1:3] + delta * (limit / distance)
    return np.array([base[0], ab[0], ab[1]], dtype=np.float32)


def region_targets(appearance: dict[str, Any], palette=None,
                   measured: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    """The LAB colour every region is matched against: the named colour's mid
    stop, with the photograph's own hue and chroma where a measurement
    exists (photo_colours.py). See _blend_measured for why only hue and
    chroma come from the photograph."""
    targets: dict[str, np.ndarray] = {}
    for entry in appearance.get("region_colors") or []:
        region, colour = entry.get("region"), entry.get("color")
        if colour not in FUR_PALETTE:
            continue
        if measured and region in measured:
            targets[region] = _blend_measured(colour, measured[region]["lab"], palette)
        else:
            targets[region] = _palette_lab(colour, palette)
    return targets


# A region whose LUMINANCE sits this far from its target is wrong whatever
# its hue scores. On a pale coat a twenty-point L gap reads as grey dirt
# while the whole LAB distance stays under the miss threshold, which is how
# a white dog's chest came back at L73 - the neutral atlas's own grey, left
# uncoloured - without anything flagging it.
MISS_L = 12.0


def luma_gaps(candidate: np.ndarray, masks, appearance: dict[str, Any], palette=None,
              measured: dict[str, Any] | None = None) -> dict[str, float]:
    """Per region, how far the candidate's median luminance is from the
    target's. Unlike the hue, this is worth checking on a region the
    photograph never answered: the pattern's base colour fixes how light
    the coat is, so a region left at the atlas's grey is a defect even when
    nobody reported a colour for it."""
    gaps: dict[str, float] = {}
    for region, target in region_targets(appearance, palette, measured).items():
        mask = masks.get(region)
        if mask is None:
            continue
        inside = mask > 0.5
        if inside.sum() < 50:
            continue
        gaps[region] = float(abs(np.median(_lab(candidate[inside])[:, 0]) - target[0]))
    return gaps


def region_scores(candidates: list[np.ndarray], masks, appearance: dict[str, Any],
                  palette=None, measured: dict[str, Any] | None = None) -> dict[str, list[float]]:
    """Per region, each candidate's LAB distance to the target colour, plus
    a penalty for a colour edge that runs along the mask edge (a masked
    edit paints to the mask; natural paintings put their boundaries
    elsewhere)."""
    scores: dict[str, list[float]] = {}
    targets = region_targets(appearance, palette, measured)
    for region, target in targets.items():
        mask = masks.get(region)
        if mask is None:
            continue
        inside = mask > 0.5
        if inside.sum() < 50:
            continue
        ring = ndimage.binary_dilation(inside, iterations=3) & ~ndimage.binary_erosion(inside, iterations=3)
        core = ndimage.binary_erosion(inside, iterations=4)
        scores[region] = []
        for candidate in candidates:
            lab = _lab(candidate[inside])
            # the median, so a marking the model drew inside the region does
            # not drag the region's colour
            distance = float(np.linalg.norm(np.median(lab, axis=0) - target))
            luma = candidate @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
            grad = np.hypot(ndimage.sobel(luma, axis=0), ndimage.sobel(luma, axis=1))
            edge_ratio = float(grad[ring].mean() / (grad[core].mean() + 1e-4)) if core.any() and ring.any() else 1.0
            penalty = MASK_EDGE_WEIGHT * max(0.0, edge_ratio - MASK_EDGE_FREE)
            scores[region].append(round(distance + penalty, 2))
    return scores


def _mix_ratio(c1: np.ndarray, c2: np.ndarray, target: np.ndarray) -> float:
    """The share of c2 in a c1/c2 mix whose LAB lands nearest the target."""
    d = c2 - c1
    denom = float(d @ d)
    if denom < 1e-6:
        return 0.0
    return float(np.clip((target - c1) @ d / denom, 0.0, 1.0))


def tone_match(candidates: list[np.ndarray], base_index: int, masks, scores: dict[str, list[float]],
               agree_lab: float = 12.0) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Shift each candidate's LAB so that, over the regions where it and the
    base candidate scored alike (painted the same colour), their medians
    coincide. Returns the adjusted candidates and the shifts applied."""
    from .projection import lab_to_linear

    base = candidates[base_index]
    out = list(candidates)
    shifts: dict[str, Any] = {}
    for index, candidate in enumerate(candidates):
        if index == base_index:
            continue
        deltas = []
        weights = []
        for region, s in scores.items():
            if abs(s[index] - s[base_index]) > agree_lab:
                continue
            mask = masks.get(region)
            inside = mask > 0.5
            if inside.sum() < 200:
                continue
            deltas.append(np.median(_lab(base[inside]), axis=0) - np.median(_lab(candidate[inside]), axis=0))
            weights.append(int(inside.sum()))
        if not deltas:
            continue
        delta = np.average(np.stack(deltas), axis=0, weights=weights)
        delta = np.clip(delta, [-12.0, -8.0, -8.0], [12.0, 8.0, 8.0])
        lab = _lab(candidate) + delta.astype(np.float32)
        out[index] = np.clip(lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
        shifts[str(index)] = [round(float(v), 2) for v in delta]
    return out, shifts


def combine(candidates: list[np.ndarray], masks, appearance: dict[str, Any], valid: np.ndarray,
            palette=None, edge_px: float = HANDOVER_PX, measured: dict[str, Any] | None = None,
            mix_threshold: float = 8.0) -> tuple[np.ndarray, dict[str, Any]]:
    """`candidates` are linear (h, w, 3) atlases of the same layout.

    Per region the best candidate is taken. A region the reading called
    faint or medium takes its colour candidate over the base-coat candidate
    in the strength's share. A region with a measured target that the best
    candidate still misses by more than `mix_threshold` takes the best two
    mixed in the proportion that lands on the measurement.
    """
    scores = region_scores(candidates, masks, appearance, palette, measured)
    targets = region_targets(appearance, palette, measured)
    choice = {region: int(np.argmin(s)) for region, s in scores.items()}
    unpainted = [region for region, s in scores.items() if min(s) > NO_CANDIDATE_LAB]
    # a region the photograph did not answer (still at the layout default)
    # is not a reason to override the base candidate: the whole-atlas
    # repaint already painted it the way a typical dog of this coat has it,
    # which is all the default knows too
    inferred = {e.get("region") for e in appearance.get("region_colors") or []
                if str(e.get("source", "")).startswith("pattern_default")}
    unpainted = sorted(set(unpainted) | inferred)
    counts = np.bincount(list(choice.values()), minlength=len(candidates)) if choice else [1]
    base_index = int(np.argmax(counts))
    shifts: dict[str, Any] = {}
    if TONE_MATCH and len(candidates) > 1:
        candidates, shifts = tone_match(candidates, base_index, masks, scores)
    out = candidates[base_index].copy()
    mixes: dict[str, Any] = {}
    entries = {e.get("region"): e for e in appearance.get("region_colors") or []}

    for region in REGIONS:
        index = choice.get(region)
        if index is None or region in unpainted:
            continue
        mask = masks.get(region)
        inside = mask > 0.5
        weight = ndimage.gaussian_filter(np.clip(mask, 0, 1), edge_px)[..., None]
        layer = candidates[index]
        entry = entries.get(region, {})
        strength = str(entry.get("strength") or "strong").lower()
        share = STRENGTH_SHARE.get(strength, 1.0)

        if share < 1.0 and len(candidates) > 1:
            # a faint colour is mostly the neutral fur underneath with a
            # little of the colour on it. The candidate whose fur here has
            # the least chroma is that neutral fur (white, grey or black,
            # whichever the model painted); the colour candidate goes over
            # it in the strength's share
            medians = [np.median(_lab(c[inside]), axis=0) for c in candidates]
            chroma = [float(np.hypot(m[1], m[2])) for m in medians]
            neutral = [i for i, ch in enumerate(chroma) if ch <= min(chroma) + NEUTRAL_CHROMA_SLACK]
            # among the neutral candidates, the lightness most repaints gave
            # the region: black and white are both neutral, and which one
            # the region "is" under a faint tint is what the rolls agreed on
            median_l = float(np.median([m[0] for m in medians]))
            under = min(neutral, key=lambda i: abs(medians[i][0] - median_l))
            if under == index:
                under = int(np.argsort(scores[region])[1])
            layer = candidates[under] * (1 - share) + candidates[index] * share
            mixes[region] = {"candidates": [under, index], "share_of_colour": share, "strength": strength}
        elif measured and region in measured and len(candidates) > 1:
            order = np.argsort(scores[region])
            first, second = int(order[0]), int(order[1])
            c1 = np.median(_lab(candidates[first][inside]), axis=0)
            c2 = np.median(_lab(candidates[second][inside]), axis=0)
            if float(np.linalg.norm(c1 - targets[region])) > mix_threshold:
                ratio = _mix_ratio(c1, c2, targets[region])
                if ratio > 0.02:
                    layer = candidates[first] * (1 - ratio) + candidates[second] * ratio
                    mixes[region] = {"candidates": [first, second], "share_of_second": round(ratio, 3)}

        if index == base_index and region not in mixes:
            continue
        out = out * (1 - weight) + layer * weight

    report = {"base_candidate": base_index, "choice": choice, "mixes": mixes, "no_candidate": unpainted,
              "tone_shifts": shifts,
              "measured_regions": sorted(measured) if measured else [],
              "scores": {r: [round(v, 1) for v in s] for r, s in scores.items()}}
    return out, report


def combine_runs(run_dirs: list[str | Path], appearance: dict[str, Any], masks_dir: str | Path,
                 base_texture: Image.Image, valid: np.ndarray, mesh_glb: str | Path,
                 palette=None, measured: dict[str, Any] | None = None,
                 symmetric: bool = True) -> tuple[Image.Image, dict[str, Any]]:
    """The repaints of several pipeline runs (their build_round0 raw atlases),
    combined, mirrored left/right (symmetry.py) and finished like a single
    repaint."""
    from .texture import MaskLibrary
    from .uv_paint import finish_painted

    pet_id = str(appearance.get("pet_id"))
    candidates, names = [], []
    for run in run_dirs:
        raw = Path(run) / "build_round0" / f"{pet_id}_painted_raw.png"
        if raw.exists():
            candidates.append(to_linear(np.asarray(Image.open(raw).convert("RGB").resize(base_texture.size), dtype=np.float32)))
            names.append(str(run))
    if not candidates:
        raise FileNotFoundError("no repaints found under the given runs")
    masks = MaskLibrary(masks_dir, base_texture.size)
    combined, report = combine(candidates, masks, appearance, valid, palette, measured=measured)
    report["candidates"] = names
    if symmetric:
        from .symmetry import symmetrize
        position = np.load(Path("output/geometry/position.npy"))
        combined, report["symmetry"] = symmetrize(combined, position, valid, masks)
    image = Image.fromarray((to_srgb(np.clip(combined, 0, 1)) * 255).astype(np.uint8), mode="RGB")
    finished, finish = finish_painted(image, base_texture, masks_dir, mesh_glb, valid,
                                      eye_colour=appearance.get("eye_color"))
    report["finish"] = finish
    return finished, report


__all__ = ["STRENGTH_SHARE", "combine", "combine_runs", "region_scores", "region_targets"]


def main() -> None:
    """Read the photograph (every region, with strength) and combine the
    repaints already on disk - no image edits, only the region reading.

        python -m api.combine_paints --record tests/record_black.json --photo tests/fixtures/image.png \
            --runs output/runs/live_paint,output/runs/live_paint2,output/runs/live_paint3,output/runs/live_paint4,output/runs/live_parts \
            --out output/runs/combined_live [--reuse appearance.json] [--scripted black_shiba]
    """
    import argparse
    import asyncio
    import json
    import subprocess
    import sys

    from .assets import BLENDER, template_paths
    from .glb import extract_base_texture, replace_base_texture
    from .metadata import process_metadata
    from .pipeline import default_transport, load_dotenv
    from .schemas import PetMetadata

    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", required=True)
    parser.add_argument("--photo", required=True)
    parser.add_argument("--runs", required=True, help="comma-separated run folders holding build_round0/*_painted_raw.png")
    parser.add_argument("--out", required=True)
    parser.add_argument("--reuse", default=None, help="skip the reading; use this appearance json")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--scripted", default=None)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    record = json.loads(Path(args.record).read_text(encoding="utf-8"))
    meta = PetMetadata.model_validate(record)
    resolved = process_metadata(meta.pet_id, meta.species, meta.weight_kg, meta.coat_color)
    template_glb, masks_dir = template_paths(resolved.base_template)
    out = Path(args.out)
    (out / "final").mkdir(parents=True, exist_ok=True)

    if args.reuse:
        from .regions import canonicalize_appearance
        appearance = canonicalize_appearance(json.loads(Path(args.reuse).read_text(encoding="utf-8")))
        reading_log = {"reused_from": args.reuse}
    else:
        from .extract import extract_appearance
        if args.scripted:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
            from scripts import CASES
            transport = CASES[args.scripted]()
        else:
            transport = default_transport(args.provider)
        extracted = asyncio.run(extract_appearance(transport, [args.photo], resolved.pet_id, resolved.base_color,
                                                   masks_dir.parent))
        appearance = {**extracted.appearance, "base_template": resolved.base_template,
                      "eye_color": extracted.decision.eye_color}
        reading_log = extracted.log
    appearance["pet_id"] = resolved.pet_id
    (out / "appearance.json").write_text(json.dumps(appearance, ensure_ascii=False, indent=1), encoding="utf-8")
    strengths = {e["region"]: (e["color"], e.get("strength", "strong")) for e in appearance.get("region_colors") or []}
    print(f"[combine] regions: {strengths}")

    base = extract_base_texture(template_glb)
    valid = np.load("output/geometry/valid.npy") > 0.5
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    coat, report = combine_runs(runs, appearance, masks_dir, base, valid, template_glb)
    coat.save(out / "final" / f"{resolved.pet_id}.png")
    glb = replace_base_texture(template_glb, coat, out / "final" / f"{resolved.pet_id}.glb")
    (out / "report.json").write_text(json.dumps({"reading": reading_log, "combine": report}, ensure_ascii=False,
                                                indent=1, default=str), encoding="utf-8")
    print(f"[combine] choice {report['choice']}\n[combine] mixes {report['mixes']}")
    if not args.no_render and Path(BLENDER).exists():
        for tag, extra in (("body", ["--clip", "sitting", "--zoom", "2.2"]),
                           ("side", ["--clip", "standing", "--zoom", "2.0", "--angle", "70", "--height", "0.05"]),
                           ("face", ["--clip", "sitting", "--face"])):
            subprocess.run([BLENDER, "--background", "--python", "ar_pipeline/render_preview.py", "--",
                            str(glb), str(out / f"render_{tag}.png"), *extra], check=True, capture_output=True)
    print(f"[combine] -> {glb}")


if __name__ == "__main__":
    main()
