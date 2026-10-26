"""The pipeline: photograph and record in, a wearable GLB out.

The DEFAULT is now the lite path (api/lite.py, --mode lite): mask build first,
one refining image edit, five vision calls. Everything below describes the
earlier whole-atlas repaint path, kept as --mode paint.

    python -m api.run --record tests/record_black.json --photo tests/fixtures/image.png --out output/runs/<name>
        [--mode lite|paint]           lite (default) or the repaint path described here
        [--no-refine]                 lite: the mask build alone, no image model
        [--reuse <appearance.json>]   skip the reading, reuse a previous one (no vision calls)
        [--no-edit]                   never call the image model; combine cached repaints only
        [--rounds N]                  whole-atlas repaints at most (default 2)
        [--no-render]                 skip the Blender renders

THE STEPS
---------
0. prepare       the photograph gated and cut (prepare_photo.py): two small
                   vision samples answer a factual form (is it one dog, does a
                   person or text cover it, harness); a rejection needs both
                   to agree and stops the run with the reason in Korean.
                   Then rembg lifts the dog, drops stray pieces, fills the
                   rest with grey and crops - that file is what every later
                   stage sees. Harness or back view become notes in the
                   reading and repaint prompts. --accept-photo runs on despite
                   a rejection.
1. record        size class, template, base colour - no model call (metadata.py)
2. reading       what the photograph says, voted three ways (extract.py):
                   pattern against the template's five anchor renders, with the
                   eye colour; then the six face regions off a head crop and the
                   ten body regions off the whole photo, each region a colour, a
                   coverage and a STRENGTH (faint/medium/strong). Silence keeps
                   the pattern's layout colour and the region is logged as
                   inferred. A failed sample is retried so the vote is never short.
2b. measure      the photograph's own colour for each region: two readings
                   point at a grid cell of fur per region, the cell is
                   white-balanced against the frame's border and read in LAB
                   (photo_colours.py). Only the HUE and CHROMA are kept; the
                   luminance stays the palette word's, because a photograph's
                   brightness carries the room's light and would bake a
                   shadow into the texture. This is what makes one dog's tan
                   different from another's instead of every tan being the
                   same table orange. --no-measure skips it.
3. repaint       the template's UV atlas, first made NEUTRAL - every fur
                   texel a grey keeping only the hair-scale grain, so none of
                   the template's own colour layout (a red shiba's) can leak
                   (neutral_atlas.py) - coloured whole by the image model in
                   the target coat, with the photograph and a flat colour
                   guide as references (uv_paint.py). The result is
                   scored region by region; regions that missed go into a second
                   repaint's prompt as corrections. At most --rounds calls.
4. combine       every repaint of this animal - this run's and the cached ones
                   from earlier runs - combined region by region
                   (combine_paints.py): the candidate nearest the region's colour
                   wins; a faint or medium colour is mixed over the neutral fur
                   in the strength's share; a region no candidate painted, or
                   the photograph did not answer, keeps the base repaint.
5. symmetry      left and right of the body agree (symmetry.py)
6. finish        the template's eyeballs, nose, pads and mouth put back, the
                   iris recoloured to the eye colour read from the photo
                   (eyes.py), UV seams smoothed (seams.py)
7. critic        the finished model rendered (whole animal sitting, standing
                   from the side, the face) and compared with the photograph
                   by the vision model, two samples, only agreed differences
                   (critic.py). A region the critic disagrees with is
                   corrected in the reading and the combination redone
                   without a call; a colour no repaint holds earns one more
                   repaint with the correction in its prompt. This is the one
                   step that judges the result against the photograph rather
                   than against the reading.
8. GLB + log     base colour texture swapped into a copy of the template
                   (rig, animations, UVs untouched), renders, pipeline_log.json

Costs per animal: two small vision calls for the photo gate, about ten for the reading, one or
two image edits. Nothing else. The image model's output is cached by prompt
so a re-run with the same reading makes no calls at all.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .assets import BLENDER, BuiltAsset, template_paths
from .combine_paints import MISS_L, combine, luma_gaps, region_targets
from .fur import to_linear, to_srgb
from .glb import animation_names, extract_base_texture, replace_base_texture
from .metadata import process_metadata
from .behavior import resolve_behavior
from .pipeline import PipelineResult, _json, default_transport, load_dotenv
from .schemas import PetMetadata
from .symmetry import symmetrize
from .texture import MaskLibrary
from .neutral_atlas import neutral_atlas_for
from .uv_paint import CACHE_DIR, ScriptedImageTransport, finish_painted, guide_texture, paint_uv

MISS_LAB = 25.0


def _now() -> float:
    return time.perf_counter()


def cached_repaints(pet_id: str, size: tuple[int, int]) -> tuple[list[np.ndarray], list[str]]:
    """Every whole-atlas repaint of this animal already in the cache."""
    atlases, names = [], []
    for path in sorted(glob.glob(str(CACHE_DIR / f"{pet_id}-*.png"))):
        if "-fix-" in Path(path).name:
            continue   # a masked region edit, not a whole atlas
        atlases.append(to_linear(np.asarray(Image.open(path).convert("RGB").resize(size), dtype=np.float32)))
        names.append(Path(path).name)
    return atlases, names


async def run(
    record: dict[str, Any],
    photos: list[str | Path],
    out_dir: str | Path,
    transport=None,
    image_transport=None,
    reuse_appearance: str | Path | None = None,
    rounds: int = 2,
    edit: bool = True,
    render: bool = True,
    samples: int = 3,
    blender: str = BLENDER,
    critic: bool = True,
    prepare: bool = True,
    strict_photo: bool = True,
    fresh: bool = False,
    measure: bool = True,
) -> PipelineResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    photos = [Path(p) for p in photos]
    log: dict[str, Any] = {"mode": "run", "steps": {}, "timings_s": {}}
    started = _now()

    # 0. prepare: gate the photograph (skipped with a reused reading, which
    #    already passed it) and cut the animal out for every later stage
    photo_notes: list[str] = []
    if prepare:
        from .prepare_photo import prepare_photo

        t = _now()
        gate_transport = None if reuse_appearance else (transport or default_transport())
        prepared = await prepare_photo(gate_transport, photos[0], out_dir, species="dog", strict=strict_photo)
        photos = [prepared.path] + photos[1:]
        photo_notes = prepared.notes
        log["steps"]["prepare"] = prepared.report()
        log["timings_s"]["prepare"] = round(_now() - t, 2)
        print(f"[run] photo prepared: cut={prepared.cut}, coverage {prepared.coverage:.2f}, "
              f"dropped {prepared.pieces_dropped} piece(s), notes {prepared.notes or 'none'}"
              + (f", REJECTED but running on: {prepared.rejected}" if prepared.rejected else ""))

    # 1. record
    meta = PetMetadata.model_validate(record)
    resolved = process_metadata(meta.pet_id, meta.species, meta.weight_kg, meta.coat_color)
    template_glb, masks_dir = template_paths(resolved.base_template)
    template_dir = masks_dir.parent
    log["steps"]["record"] = {"pet_id": resolved.pet_id, "base_color": resolved.base_color,
                              "size_class": resolved.size_class, "base_template": resolved.base_template}

    # 2. reading
    t = _now()
    if reuse_appearance:
        from .regions import canonicalize_appearance
        # an older reading may still use the sixteen-region names
        appearance = canonicalize_appearance(json.loads(Path(reuse_appearance).read_text(encoding="utf-8")))
        log["steps"]["reading"] = {"reused_from": str(reuse_appearance)}
    else:
        from .extract import extract_appearance
        transport = transport or default_transport()
        extracted = await extract_appearance(transport, photos, resolved.pet_id, resolved.base_color,
                                             template_dir, samples=samples, notes=photo_notes)
        appearance = {**extracted.appearance, "eye_color": extracted.decision.eye_color}
        if extracted.face is not None:
            extracted.face.image.save(out_dir / "face_crop.png")
        log["steps"]["reading"] = extracted.log
    appearance.update({"pet_id": resolved.pet_id, "base_template": resolved.base_template,
                       "size_class": resolved.size_class})
    _json(out_dir / "appearance.json", appearance)
    log["timings_s"]["reading"] = round(_now() - t, 2)
    observed = [e["region"] for e in appearance["region_colors"] if not str(e.get("source", "")).startswith("pattern")]
    print(f"[run] reading: {appearance.get('pattern')} {appearance.get('base_color')}/{appearance.get('second_color')}, "
          f"eyes {appearance.get('eye_color')}, observed {observed}")

    # 2b. the photograph's own colour per region, carried in the appearance so
    #     that a --reuse run scores against the same measurement
    measured: dict[str, Any] = appearance.get("measured_colors") or {}
    if measure and not measured:
        from .photo_colours import measure_region_colours

        t = _now()
        try:
            # the ORIGINAL frame, not the cut-out: the white balance reads the
            # border as a grey card, and our cut-out replaced it with a flat grey
            measured = await measure_region_colours(prepared.original if prepare else photos[0],
                                                    transport or default_transport())
        except Exception as exc:  # noqa: BLE001 - the palette word is the fallback
            print(f"[run] colour measurement failed ({type(exc).__name__}: {exc}); using the palette")
            measured = {}
        appearance["measured_colors"] = measured
        _json(out_dir / "appearance.json", appearance)
        log["steps"]["measure"] = {"regions": {r: v["lab"] for r, v in measured.items()}}
        log["timings_s"]["measure"] = round(_now() - t, 2)
        print(f"[run] measured {len(measured)} region(s) from the photo: {sorted(measured)}")

    base = extract_base_texture(template_glb)
    valid = np.load(Path("output/geometry/valid.npy")) > 0.5
    # the template's shape and grain without its colour: what the image model colours
    paint_base = neutral_atlas_for(base, valid, masks_dir)
    paint_base.save(out_dir / "neutral_atlas.png")
    masks = MaskLibrary(masks_dir, base.size)

    # 3. repaint
    t = _now()
    paint_log: list[dict[str, Any]] = []
    new_atlases: list[np.ndarray] = []
    if edit:
        if image_transport is None:
            import os
            if os.environ.get("OPENAI_API_KEY") or os.environ.get("LUNA_API_KEY"):
                from .imagegen import OpenAIImageTransport
                image_transport = OpenAIImageTransport()
        if image_transport is None:
            paint_log.append({"skipped": "no OPENAI_API_KEY for the image model"})
        else:
            use_cache = not isinstance(image_transport, ScriptedImageTransport)
            guide = guide_texture(appearance, masks_dir, valid, base.size, base_texture=paint_base,
                                  targets=region_targets(appearance, measured=measured))
            guide.save(out_dir / "guide.png")
            corrections: list[str] = []
            from .combine_paints import region_scores
            from .uv_paint import COLOUR_WORDS
            for roll in range(rounds):
                result = await paint_uv(image_transport, paint_base, photos[0], appearance, valid, corrections,
                                        use_cache=use_cache, guide=guide, masks_dir=masks_dir, notes=photo_notes)
                entry = {"roll": roll + 1, **{k: v for k, v in result.report.items() if k != "prompt"},
                         "corrections": list(corrections)}
                if not result.accepted:
                    paint_log.append(entry)
                    break
                atlas = to_linear(np.asarray(result.image, dtype=np.float32))
                new_atlases.append(atlas)
                result.image.save(out_dir / f"repaint_roll{roll + 1}.png")
                scores = region_scores([atlas], masks, appearance, measured=measured)
                # a hue only counts against a region the photograph answered; a
                # luminance counts everywhere, because the pattern fixes it
                gaps = luma_gaps(atlas, masks, appearance, measured=measured)
                missed = sorted({r for r, s in scores.items() if s[0] > MISS_LAB and r in observed}
                                | {r for r, g in gaps.items() if g > MISS_L})
                entry["luma_gaps"] = {r: round(g, 1) for r, g in sorted(gaps.items()) if g > MISS_L / 2}
                entry["missed"] = missed
                paint_log.append(entry)
                if not missed:
                    break
                colours = {e["region"]: e["color"] for e in appearance["region_colors"]}
                corrections = [f"The {r.replace('_', ' ')} must be {COLOUR_WORDS.get(colours[r], colours[r])} fur."
                               for r in missed]
    else:
        paint_log.append({"skipped": "--no-edit"})
    log["steps"]["repaint"] = paint_log
    log["timings_s"]["repaint"] = round(_now() - t, 2)

    # 4. combine (this run's repaints + every cached one of this animal), 5. symmetry, 6. finish
    cached, cached_names = ([], []) if fresh else cached_repaints(resolved.pet_id, base.size)
    candidates = new_atlases + [c for c in cached]
    names = [f"roll{i + 1}" for i in range(len(new_atlases))] + cached_names
    if not candidates:
        raise RuntimeError("no repaint to build from: the image model was not called and nothing is cached")
    position = np.load(Path("output/geometry/position.npy"))
    final_dir = out_dir / "final"
    final_dir.mkdir(exist_ok=True)

    def build(current: dict[str, Any], tag: str) -> tuple[Path, dict[str, Any]]:
        combined, report = combine(candidates, masks, current, valid, measured=measured)
        report["candidates"] = names
        combined, report["symmetry"] = symmetrize(combined, position, valid, masks)
        image = Image.fromarray((to_srgb(np.clip(combined, 0, 1)) * 255).astype(np.uint8), mode="RGB")
        coat, report["finish"] = finish_painted(image, base, masks_dir, template_glb, valid,
                                                eye_colour=current.get("eye_color"))
        coat.save(final_dir / f"{resolved.pet_id}.png")
        return replace_base_texture(template_glb, coat, final_dir / f"{resolved.pet_id}.glb"), report

    t = _now()
    glb, combine_report = build(appearance, "first")
    log["steps"]["combine"] = combine_report
    log["timings_s"]["combine_finish"] = round(_now() - t, 2)
    print(f"[run] combined {len(candidates)} repaint(s): choice {combine_report['choice']}, mixes {combine_report['mixes']}")

    # 7. critic: the render against the photograph
    critic_log: dict[str, Any] = {"ran": False}
    if critic and render and Path(blender).exists():
        from .critic import render_views, sampled_critique

        t = _now()
        body, face = render_views(glb, out_dir / "renders", blender, "sitting")
        patches, readings, notes = await sampled_critique(transport or default_transport(), photos[0], body, face,
                                                          appearance)
        critic_log = {"ran": True, "patches": patches, "readings": readings, "note": notes,
                      "renders": [str(body), str(body.parent / "render_side.png"), str(face)]}
        if patches:
            from .critic import apply_patches
            corrected = apply_patches(appearance, patches, 1)
            # regions the critic corrected are observations now, and paint at full strength
            fixed = {p["region"] for p in patches}
            for e in corrected["region_colors"]:
                if e["region"] in fixed:
                    e["source"] = "critic"
                    e["strength"] = "strong"
            corrected["inferred_regions"] = [r for r in corrected.get("inferred_regions", []) if r not in fixed]
            _json(out_dir / "appearance_after_critic.json", corrected)
            glb, combine_report2 = build(corrected, "after_critic")
            critic_log["recombined"] = {"no_candidate": combine_report2.get("no_candidate"),
                                        "choice": combine_report2["choice"]}
            # Did the correction actually reach the texture? Ask the output, not
            # the bookkeeping: with a single candidate the combination has
            # nothing to swap in, so a patch everyone agreed on silently did
            # nothing. Anything still off earns the one extra edit.
            after = to_linear(np.asarray(Image.open(final_dir / f"{resolved.pet_id}.png").convert("RGB"),
                                         dtype=np.float32))
            from .combine_paints import region_scores as _scores
            still = _scores([after], masks, corrected, measured=measured)
            still_l = luma_gaps(after, masks, corrected, measured=measured)
            needs_paint = [p["region"] for p in patches
                           if still.get(p["region"], [0.0])[0] > MISS_LAB or still_l.get(p["region"], 0.0) > MISS_L]
            critic_log["not_honoured"] = needs_paint
            if needs_paint and edit and image_transport is not None:
                from .uv_paint import COLOUR_WORDS
                colours = {e["region"]: e["color"] for e in corrected["region_colors"]}
                corrections = [f"The {r.replace('_', ' ')} must be {COLOUR_WORDS.get(colours[r], colours[r])} fur."
                               for r in needs_paint]
                result = await paint_uv(image_transport, paint_base, photos[0], corrected, valid, corrections,
                                        use_cache=not isinstance(image_transport, ScriptedImageTransport),
                                        guide=guide_texture(corrected, masks_dir, valid, base.size, base_texture=paint_base,
                                                            targets=region_targets(corrected, measured=measured)),
                                        masks_dir=masks_dir, notes=photo_notes)
                critic_log["extra_repaint"] = {k: v for k, v in result.report.items() if k != "prompt"}
                if result.accepted:
                    candidates.append(to_linear(np.asarray(result.image, dtype=np.float32)))
                    names.append("critic_roll")
                    glb, combine_report3 = build(corrected, "after_critic_repaint")
                    critic_log["recombined_after_repaint"] = {"choice": combine_report3["choice"]}
            appearance = corrected
            body, face = render_views(glb, out_dir / "renders_after_critic", blender, "sitting")
            critic_log["renders_after"] = [str(body), str(face)]
        log["timings_s"]["critic"] = round(_now() - t, 2)
        print(f"[run] critic: {patches or 'no agreed differences'}")
    elif render and Path(blender).exists():
        from .critic import render_views
        body, face = render_views(glb, out_dir / "renders", blender, "sitting")
        critic_log["renders"] = [str(body), str(face)]
    log["steps"]["critic"] = critic_log

    behavior = resolve_behavior(animation_names(glb))
    log["final"] = {"glb": str(glb), "texture": str(final_dir / f"{resolved.pet_id}.png"),
                    "served_path": resolved.asset_path, "behavior": behavior,
                    "inferred_regions": appearance.get("inferred_regions", []), "eye_color": appearance.get("eye_color")}
    log["timings_s"]["total"] = round(_now() - started, 2)
    _json(out_dir / "pipeline_log.json", log)
    print(f"[run] done in {log['timings_s']['total']}s -> {glb}")
    asset = BuiltAsset(pet_id=resolved.pet_id, glb_path=glb, texture_path=final_dir / f"{resolved.pet_id}.png",
                       served_path=resolved.asset_path, behavior=behavior, report={"path": "run", "combine": combine_report})
    return PipelineResult(resolved.pet_id, appearance, asset, log, out_dir)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", required=True)
    parser.add_argument("--photo", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reuse", default=None, help="a previous run's appearance.json: no reading calls")
    parser.add_argument("--no-edit", action="store_true", help="no image-model calls; cached repaints only")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-critic", action="store_true", help="skip the render-vs-photo critic")
    parser.add_argument("--no-prepare", action="store_true", help="skip the photo gate and cut-out")
    parser.add_argument("--fresh", action="store_true", help="ignore this animal's earlier repaints in the cache")
    parser.add_argument("--no-measure", action="store_true",
                        help="do not measure region colours from the photo; use the palette words alone")
    parser.add_argument("--accept-photo", action="store_true", help="run on even when the gate rejects the photo")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--scripted", default=None, help="offline: canned reading from tests/scripts.py")
    parser.add_argument("--template", default=None,
                        help="template slug (e.g. jindo -> output/skeleton_jindo_<size>.glb); default: the one the "
                             "photo looks like (api/template_pick.py)")
    parser.add_argument("--mode", choices=("lite", "paint"), default="lite",
                        help="lite (default): mask build + one refining edit, five vision calls; "
                             "paint: the whole-atlas repaint with combine and critic")
    parser.add_argument("--no-refine", action="store_true", help="lite: skip the image-model refinement")
    parser.add_argument("--read-shape", action="store_true",
                        help="lite with --reuse: read the body shape fresh (one call x3), keep the colour reading")
    parser.add_argument("--fit-stop", action="store_true",
                        help="lite: stop before the measurement and the image edit when the template does not "
                             "suit the dog (fit.py verdict 부적합); the verdict is in <out>/fit.json either way")
    parser.add_argument("--no-real-size", action="store_true",
                        help="lite: keep the template's own size instead of scaling to the record's weight")
    args = parser.parse_args()

    transport = None
    image_transport = None
    if args.scripted:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
        from scripts import CASES
        transport = CASES[args.scripted]()
        image_transport = ScriptedImageTransport()
    elif args.provider:
        transport = default_transport(args.provider)
    record = json.loads(Path(args.record).read_text(encoding="utf-8"))
    if args.template:
        import os
        os.environ["PET_TEMPLATE"] = args.template
    from .fit import FitRejected
    from .prepare_photo import PhotoRejected
    try:
        if args.mode == "lite":
            from .lite import run_lite
            asyncio.run(run_lite(record, args.photo, args.out, transport=transport, image_transport=image_transport,
                                 refine=not (args.no_refine or args.no_edit), render=not args.no_render,
                                 prepare=not args.no_prepare, strict_photo=not args.accept_photo,
                                 reuse_appearance=args.reuse, measure=not args.no_measure,
                                 read_shape=args.read_shape, fit_stop=args.fit_stop,
                                 real_size=not args.no_real_size))
            return
        asyncio.run(run(record, args.photo, args.out, transport=transport, image_transport=image_transport,
                        reuse_appearance=args.reuse, rounds=args.rounds, edit=not args.no_edit,
                        render=not args.no_render, critic=not args.no_critic,
                        prepare=not args.no_prepare, strict_photo=not args.accept_photo, fresh=args.fresh,
                        measure=not args.no_measure))
    except PhotoRejected as exc:
        _json(Path(args.out) / "photo_rejected.json", {"reasons": exc.reasons, "message": str(exc),
                                                       "gate": exc.readings})
        raise SystemExit(f"[run] photo rejected: {exc}  (--accept-photo to run anyway)")
    except FitRejected as exc:
        raise SystemExit(f"[run] template does not suit this dog: {exc}  (drop --fit-stop to build anyway)")


if __name__ == "__main__":
    main()
