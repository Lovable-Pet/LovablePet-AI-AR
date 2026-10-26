"""The whole pipeline, record and photographs to a wearable GLB, in one call.

DEFAULT PATH
------------
  1. metadata          the record settles size class, template, base colour
  2. appearance        pattern classifier + face/body region agents (extract.py)
  2b. views prior      four views drawn by the image model, READ for the
                       regions the photo left at the default and reduced to
                       colour labels for the marked regions' shapes
                       (views_prior.py); skipped without a key or guides
  3. palette           the coat's colours calibrated against the photographs
  4. face layer        the photo's face warped onto the face island, when the
                       head is frontal (face_photo.py); gated by the ring
                       check inside the compositor
  5. build             the coat composed (texture.py) into the template (glb.py)
  6. critic            render, compare with the photo, patch, rebuild; at
                       most two rounds (critic.py)

Every step writes what it produced under out_dir, and pipeline_log.json
records the decisions: which pattern and why, which regions were read and
which assumed, whether the face layer was accepted, every critic round and
whether the loop oscillated.

EXPERIMENTAL
------------
--experimental switches the coat build to the generated-view projection
(imagegen/luna -> consistency -> projection). It needs an image-generation
key and the canonical renders under output/canonical; when those are
missing it says so in the log and the default build stands.

Usage:
    python -m api.pipeline --record tests/record_black.json --photo tests/fixtures/image.png \
        --out output/runs/D2023072630 [--experimental] [--no-render]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import time
from dataclasses import dataclass, field

from PIL import Image
from pathlib import Path
from typing import Any

from .agents import Transport
from .assets import BLENDER, BuiltAsset, build_pet_asset, template_paths
from .critic import CriticOutcome, critic_loop
from .extract import extract_appearance
from .face_photo import build_face_layer, read_face_landmarks
from .metadata import MetadataResult, process_metadata
from .palette import palette_for
from .behavior import resolve_behavior
from .regions import canonicalize_appearance
from .schemas import PetMetadata
from .texture import MaskLibrary


@dataclass
class PipelineResult:
    pet_id: str
    appearance: dict[str, Any]
    asset: BuiltAsset
    log: dict[str, Any] = field(default_factory=dict)
    out_dir: Path = Path(".")


def _now() -> float:
    return time.perf_counter()


def default_transport(provider: str | None = None) -> Transport:
    """The vision transport: the provider asked for, else whichever key is
    present (OpenAI when only OPENAI_API_KEY is set), else Anthropic."""
    if provider is None and not os.environ.get("APPEARANCE_PROVIDER"):
        if os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
            provider = "openai"
    provider = (provider or os.environ.get("APPEARANCE_PROVIDER", "anthropic")).lower()
    if provider == "openai":
        from .openai_transport import OpenAITransport
        return OpenAITransport()
    from .agents import AnthropicTransport
    return AnthropicTransport()


def _json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


async def run_pipeline(
    record: dict[str, Any],
    photos: list[str | Path],
    out_dir: str | Path,
    transport: Transport,
    experimental: bool = False,
    render: bool = True,
    critic_rounds: int = 2,
    samples: int = 3,
    blender: str = BLENDER,
    render_clip: str = "sitting",
    use_views: bool = True,
    mode: str = "paint",
    image_transport=None,
    reuse_appearance: str | Path | None = None,
) -> PipelineResult:
    if mode == "parts":
        return await run_parts(record, photos, out_dir, transport, render=render, samples=samples,
                               blender=blender, render_clip=render_clip, critic_rounds=critic_rounds,
                               image_transport=image_transport, reuse_appearance=reuse_appearance)
    if mode in ("simple", "paint"):
        return await run_simple(record, photos, out_dir, transport, render=render,
                                samples=samples, blender=blender, render_clip=render_clip,
                                critic_rounds=critic_rounds, paint=(mode == "paint"),
                                image_transport=image_transport, reuse_appearance=reuse_appearance)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    photos = [Path(p) for p in photos]
    log: dict[str, Any] = {"experimental": experimental, "steps": {}, "timings_s": {}}
    started = _now()

    # 1. metadata ------------------------------------------------------------
    meta = PetMetadata.model_validate(record)
    resolved: MetadataResult = process_metadata(meta.pet_id, meta.species, meta.weight_kg, meta.coat_color)
    template_glb, masks_dir = template_paths(resolved.base_template)
    template_dir = masks_dir.parent
    log["steps"]["metadata"] = {
        "pet_id": resolved.pet_id, "base_color": resolved.base_color,
        "size_class": resolved.size_class, "base_template": resolved.base_template,
        "photos": [str(p) for p in photos],
    }
    log["timings_s"]["metadata"] = round(_now() - started, 2)

    # 2. appearance ----------------------------------------------------------
    t = _now()
    extracted = await extract_appearance(
        transport, photos, resolved.pet_id, resolved.base_color, template_dir, samples=samples
    )
    appearance = {**extracted.appearance, "base_template": resolved.base_template,
                  "size_class": resolved.size_class}
    _json(out_dir / "appearance_stage2.json", appearance)
    if extracted.face is not None:
        extracted.face.image.save(out_dir / "face_crop.png")
    log["steps"]["appearance"] = extracted.log
    log["timings_s"]["appearance"] = round(_now() - t, 2)
    print(f"[pipeline] appearance: {extracted.log['pattern_summary']}; "
          f"inferred {appearance['inferred_regions']}")

    # 2b. the views prior --------------------------------------------------
    t = _now()
    label_layer = None
    if use_views:
        from .views_prior import fill_from_views, views_prior

        prior = await views_prior(
            transport, photos[0], appearance, Path(resolved.base_template).stem,
            out_dir / "views", Path("output/geometry"), samples=samples,
        )
        appearance, filled = fill_from_views(appearance, prior)
        if prior.label_map is not None:
            label_layer = (prior.label_map, prior.label_weight, prior.colours)
        log["steps"]["views_prior"] = {**prior.report, "filled_regions": filled}
        _json(out_dir / "appearance_after_views.json", appearance)
        print(f"[pipeline] views prior: {prior.report.get('skipped') or f'filled {filled}'}"
              + (f", shapes {prior.report['shapes']['labelled_share_of_valid']:.0%} of the surface"
                 if prior.label_map is not None else ""))
    else:
        log["steps"]["views_prior"] = {"skipped": "disabled"}
    log["timings_s"]["views_prior"] = round(_now() - t, 2)

    # 3. palette -------------------------------------------------------------
    t = _now()
    palette, stats, story = palette_for(appearance, photos)
    log["steps"]["palette"] = {"story": story, "usable": stats.usable, "palette": palette}
    log["timings_s"]["palette"] = round(_now() - t, 2)

    # 4. face layer ----------------------------------------------------------
    t = _now()
    face_layer = None
    face_ratio = None
    face_log: dict[str, Any] = {"attempted": False}
    crop = extracted.face
    if crop is not None and crop.visible and crop.frontal:
        face_log["attempted"] = True
        try:
            marks = await read_face_landmarks(crop.image, transport)
            face_log["landmarks_crop_px"] = marks
            masks = MaskLibrary(masks_dir, (2048, 2048))
            layer = build_face_layer(photos[0], crop.box, marks, template_dir, masks, palette,
                                     trust_all=crop.frontal)
            if layer is None:
                face_log["result"] = "not enough visible landmarks; no face layer"
            else:
                face_layer = (layer.rgb, layer.weight)
                face_ratio = layer.report.get("triangle_ratio")
                face_log["result"] = "built"
                face_log["layer"] = layer.report
        except Exception as exc:  # noqa: BLE001 - the face layer is optional
            face_log["result"] = f"failed ({type(exc).__name__}: {exc}); no face layer"
    else:
        face_log["result"] = ("no face crop" if crop is None else
                              "head not frontal" if crop.visible else "no head found")
    log["steps"]["face_layer"] = face_log
    log["timings_s"]["face_layer"] = round(_now() - t, 2)
    print(f"[pipeline] face layer: {face_log['result']}")

    # 5 + 6. build and critic --------------------------------------------------
    t = _now()
    builds: list[dict[str, Any]] = []
    views_dir = None
    if experimental:
        views_dir, experimental_log = await _experimental_views(
            transport, photos[0], appearance, out_dir / "views"
        )
        log["steps"]["experimental"] = experimental_log

    def build(current: dict[str, Any], round_index: int) -> Path:
        asset_dir = out_dir / f"build_round{round_index}"
        built = build_pet_asset(
            current, asset_dir=asset_dir, photos=photos, palette=palette,
            face_layer=face_layer, debug_dir=asset_dir / "debug" if round_index == 0 else None,
            experimental=experimental, views_dir=views_dir, face_triangle_ratio=face_ratio,
            label_layer=label_layer,
        )
        builds.append({"round": round_index, "glb": str(built.glb_path), "report": built.report})
        _json(asset_dir / "appearance.json", current)
        return built.glb_path

    critic: CriticOutcome | None = None
    can_render = render and Path(blender).exists()
    if can_render and critic_rounds > 0:
        critic = await critic_loop(
            transport, photos[0], appearance, build, out_dir, blender, max_rounds=critic_rounds,
            clip=render_clip,
        )
        appearance = critic.appearance
        log["steps"]["critic"] = critic.log()
    else:
        build(appearance, 0)
        log["steps"]["critic"] = {
            "rounds": 0, "final_round": 0, "oscillated": False, "history": [],
            "skipped": "rendering disabled" if not render else f"Blender not found at {blender}",
        }
    log["steps"]["builds"] = builds
    log["timings_s"]["build_and_critic"] = round(_now() - t, 2)

    # the final asset: the round the critic kept, copied to out_dir
    final_round = critic.final_round if critic is not None else 0
    final_build = next(b for b in builds if b["round"] == final_round)
    final_dir = out_dir / "final"
    final_dir.mkdir(exist_ok=True)
    for name in (f"{resolved.pet_id}.glb", f"{resolved.pet_id}.png", f"{resolved.pet_id}_normal.png"):
        source = Path(final_build["glb"]).parent / name
        if source.exists():
            shutil.copy2(source, final_dir / name)
    from .glb import animation_names
    behavior = resolve_behavior(animation_names(final_dir / f"{resolved.pet_id}.glb"))
    asset = BuiltAsset(
        pet_id=resolved.pet_id,
        glb_path=final_dir / f"{resolved.pet_id}.glb",
        texture_path=final_dir / f"{resolved.pet_id}.png",
        served_path=resolved.asset_path,
        behavior=behavior,
        normal_path=(final_dir / f"{resolved.pet_id}_normal.png"
                     if (final_dir / f"{resolved.pet_id}_normal.png").exists() else None),
        report=final_build["report"],
    )
    _json(out_dir / "appearance_final.json", appearance)

    log["final"] = {
        "round": final_round, "glb": str(asset.glb_path), "texture": str(asset.texture_path),
        "served_path": asset.served_path, "behavior": behavior,
        "inferred_regions": appearance.get("inferred_regions", []),
    }
    log["timings_s"]["total"] = round(_now() - started, 2)
    _json(out_dir / "pipeline_log.json", log)
    print(f"[pipeline] done in {log['timings_s']['total']}s -> {asset.glb_path}")
    return PipelineResult(resolved.pet_id, appearance, asset, log, out_dir)


async def run_simple(
    record: dict[str, Any],
    photos: list[str | Path],
    out_dir: str | Path,
    transport: Transport,
    render: bool = True,
    samples: int = 3,
    blender: str = BLENDER,
    render_clip: str = "sitting",
    critic_rounds: int = 1,
    paint: bool = False,
    image_transport=None,
    reuse_appearance: str | Path | None = None,
) -> PipelineResult:
    """SIMPLE MODE: pattern + two colours + a few named features on the
    pattern's natural layout (simple.py). With `paint` (PAINT MODE, the
    default) the same reading drives an image-editing model that repaints
    the template's own UV texture (uv_paint.py); the mask build is the
    fallback when the repaint fails its layout check."""
    from .critic import render_views
    from .extract import photo_blocks
    from .face_crop import crop_face
    from .pattern_classify import classify_pattern
    from .pattern_classify import describe as describe_decision
    from .simple import apply_features, read_features, simple_critique

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    photos = [Path(p) for p in photos]
    log: dict[str, Any] = {"mode": "simple", "steps": {}, "timings_s": {}}
    started = _now()

    meta = PetMetadata.model_validate(record)
    resolved = process_metadata(meta.pet_id, meta.species, meta.weight_kg, meta.coat_color)
    template_glb, masks_dir = template_paths(resolved.base_template)
    template_dir = masks_dir.parent
    log["steps"]["metadata"] = {"pet_id": resolved.pet_id, "base_color": resolved.base_color,
                                "size_class": resolved.size_class, "base_template": resolved.base_template,
                                "photos": [str(p) for p in photos]}

    # 1. pattern, colours, features - or a previous run's reading, reused so
    #    a prompt or finishing change costs one image edit and nothing else
    t = _now()
    blocks = photo_blocks(photos)
    if reuse_appearance:
        previous = canonicalize_appearance(json.loads(Path(reuse_appearance).read_text(encoding="utf-8")))
        features = list(previous.get("features") or [])
        votes = {f: 1.0 for f in features}
        feature_note = previous.get("description", "")
        decision = _ReusedDecision(previous, str(reuse_appearance))
        summary = f"reused {reuse_appearance}: {decision.pattern} ({decision.base_color}/{decision.second_color})"
    else:
        decision = await classify_pattern(transport, blocks, template_dir / "anchors", resolved.base_color, samples=samples)
        features, votes, feature_note = await read_features(transport, blocks, decision.base_color, samples)
        summary = describe_decision(decision)
    region_colors = apply_features(decision.pattern, decision.base_color, decision.second_color, features)
    appearance = {
        "pet_id": resolved.pet_id, "base_template": resolved.base_template, "size_class": resolved.size_class,
        "base_color": decision.base_color, "second_color": decision.second_color,
        "pattern": decision.pattern, "overlay": decision.overlay,
        "eye_color": getattr(decision, "eye_color", None) or (previous.get("eye_color") if reuse_appearance else None),
        "region_colors": region_colors, "markings": [], "features": features,
        "inferred_regions": [e["region"] for e in region_colors if str(e.get("source", "")).startswith("pattern")],
        "description": feature_note, "confidence": min([1.0, *[votes.get(f, 1.0) for f in features]]),
    }
    _json(out_dir / "appearance_stage2.json", appearance)
    log["steps"]["appearance"] = {"pattern": decision.as_dict(), "pattern_summary": summary,
                                  "features": features, "feature_votes": votes, "note": feature_note}
    log["timings_s"]["appearance"] = round(_now() - t, 2)
    print(f"[pipeline] simple: {summary}; features {features} (votes {votes})")

    # 2. palette
    t = _now()
    palette, stats, story = palette_for(appearance, photos)
    log["steps"]["palette"] = {"story": story, "usable": stats.usable, "palette": palette}
    log["timings_s"]["palette"] = round(_now() - t, 2)

    # 3. the face from the photograph, when frontal and when it lands
    t = _now()
    face_layer, face_ratio = None, None
    face_log: dict[str, Any] = {"attempted": False}
    try:
        if paint:
            # the image model is given the photograph itself; the warped face
            # layer is the mask build's device, and its calls are saved
            raise StopIteration("skipped (paint mode: the image model sees the photograph)")
        crop = await crop_face(photos[0], transport)
        crop.image.save(out_dir / "face_crop.png")
        face_log["crop"] = crop.report()
        if crop.visible and crop.frontal:
            face_log["attempted"] = True
            marks = await read_face_landmarks(crop.image, transport)
            face_log["landmarks_crop_px"] = marks
            masks = MaskLibrary(masks_dir, (2048, 2048))
            layer = build_face_layer(photos[0], crop.box, marks, template_dir, masks, palette, trust_all=True)
            if layer is None:
                face_log["result"] = "not enough landmarks; no face layer"
            else:
                face_layer, face_ratio = (layer.rgb, layer.weight), layer.report.get("triangle_ratio")
                face_log["result"], face_log["layer"] = "built", layer.report
        else:
            face_log["result"] = "head not frontal" if crop.visible else "no head found"
    except StopIteration as why:
        face_log["result"] = str(why)
    except Exception as exc:  # noqa: BLE001
        face_log["result"] = f"failed ({type(exc).__name__}: {exc}); no face layer"
    log["steps"]["face_layer"] = face_log
    log["timings_s"]["face_layer"] = round(_now() - t, 2)
    print(f"[pipeline] face layer: {face_log['result']}")

    # 4. build, one critic round in the feature vocabulary, keep the better
    t = _now()
    builds: list[dict[str, Any]] = []

    paint_state: dict[str, Any] = {"corrections": [], "log": []}

    def build(current: dict[str, Any], round_index: int) -> Path:
        asset_dir = out_dir / f"build_round{round_index}"
        if paint:
            painted = _paint_build(current, asset_dir, photos, resolved, image_transport, paint_state)
            if painted is not None:
                builds.append({"round": round_index, "glb": str(painted.glb_path), "report": painted.report})
                _json(asset_dir / "appearance.json", current)
                return painted.glb_path
        built = build_pet_asset(current, asset_dir=asset_dir, photos=photos, palette=palette,
                                face_layer=face_layer, face_triangle_ratio=face_ratio, feather_all=False,
                                debug_dir=asset_dir / "debug" if round_index == 0 else None)
        builds.append({"round": round_index, "glb": str(built.glb_path), "report": built.report})
        _json(asset_dir / "appearance.json", current)
        return built.glb_path

    glb = build(appearance, 0)
    log["mode"] = "paint" if paint else "simple"
    critic_log: dict[str, Any] = {"rounds": 0, "final_round": 0, "history": []}
    final_round = 0
    if render and Path(blender).exists() and critic_rounds > 0:
        body, face = render_views(glb, out_dir / "critic_round1", blender, render_clip)
        add, remove, readings, notes = await simple_critique(transport, blocks, body, face, features)
        record1 = {"round": 1, "add": add, "remove": remove, "readings": readings, "note": notes,
                   "renders": [str(body), str(face)]}
        score0 = sum(len(r.get("add") or []) + len(r.get("remove") or []) for r in readings) / max(len(readings), 1)
        scores = {"0": round(score0, 2)}
        if add or remove:
            features2 = [f for f in features if f not in remove] + [f for f in add if f not in features]
            features2 = features2[:MAX_FEATURES_SIMPLE]
            from .simple import FEATURES as _FEATURES
            paint_state["corrections"] = (
                [f"The dog does have {_FEATURES[f][0]}; paint it." for f in add if f in _FEATURES]
                + [f"The dog does NOT have {_FEATURES[f][0]}; remove it." for f in remove if f in _FEATURES])
            appearance2 = {**appearance, "features": features2,
                           "region_colors": apply_features(decision.pattern, decision.base_color,
                                                           decision.second_color, features2)}
            glb2 = build(appearance2, 1)
            body2, face2 = render_views(glb2, out_dir / "critic_round2", blender, render_clip)
            add2, remove2, readings2, notes2 = await simple_critique(transport, blocks, body2, face2, features2)
            score1 = sum(len(r.get("add") or []) + len(r.get("remove") or []) for r in readings2) / max(len(readings2), 1)
            scores["1"] = round(score1, 2)
            record1["outcome"] = f"applied add={add} remove={remove} and rebuilt"
            critic_log["history"] = [record1, {"round": 2, "add": add2, "remove": remove2, "readings": readings2,
                                               "note": notes2, "renders": [str(body2), str(face2)],
                                               "outcome": "scored the rebuilt model"}]
            if score1 <= score0:
                appearance, features, final_round = appearance2, features2, 1
        else:
            record1["outcome"] = "no agreed differences"
            critic_log["history"] = [record1]
        critic_log.update({"rounds": len(critic_log["history"]), "final_round": final_round,
                           "mean_differences_per_build": scores})
    else:
        critic_log["skipped"] = "rendering disabled" if not render else f"Blender not found at {blender}"
    log["steps"]["critic"] = critic_log
    log["steps"]["builds"] = builds
    if paint:
        log["steps"]["paint"] = paint_state["log"]
    log["timings_s"]["build_and_critic"] = round(_now() - t, 2)

    # 5. final copy
    final_build = next(b for b in builds if b["round"] == final_round)
    final_dir = out_dir / "final"
    final_dir.mkdir(exist_ok=True)
    for name in (f"{resolved.pet_id}.glb", f"{resolved.pet_id}.png", f"{resolved.pet_id}_normal.png"):
        source = Path(final_build["glb"]).parent / name
        if source.exists():
            shutil.copy2(source, final_dir / name)
    from .glb import animation_names
    behavior = resolve_behavior(animation_names(final_dir / f"{resolved.pet_id}.glb"))
    asset = BuiltAsset(pet_id=resolved.pet_id, glb_path=final_dir / f"{resolved.pet_id}.glb",
                       texture_path=final_dir / f"{resolved.pet_id}.png", served_path=resolved.asset_path,
                       behavior=behavior, report=final_build["report"])
    _json(out_dir / "appearance_final.json", appearance)
    log["final"] = {"round": final_round, "glb": str(asset.glb_path), "features": features,
                    "served_path": asset.served_path, "behavior": behavior}
    log["timings_s"]["total"] = round(_now() - started, 2)
    _json(out_dir / "pipeline_log.json", log)
    print(f"[pipeline] done in {log['timings_s']['total']}s -> {asset.glb_path}")
    return PipelineResult(resolved.pet_id, appearance, asset, log, out_dir)


# the first read names at most four features; the critic may add up to two more
MAX_FEATURES_SIMPLE = 6

# Paint mode: how many whole-atlas repaints at most, and the LAB distance
# past which a region counts as missed and earns the second roll.
PAINT_ROLLS = 2
MISS_LAB = 25.0


async def run_parts(
    record: dict[str, Any],
    photos: list[str | Path],
    out_dir: str | Path,
    transport: Transport,
    render: bool = True,
    samples: int = 3,
    blender: str = BLENDER,
    render_clip: str = "sitting",
    critic_rounds: int = 0,
    image_transport=None,
    reuse_appearance: str | Path | None = None,
) -> PipelineResult:
    """PARTS MODE (the default): every region read from the photograph
    (extract.py), every colour group painted by a masked edit
    (paint_parts.py). See paint_parts.py."""
    import numpy as np

    from .glb import animation_names, extract_base_texture, replace_base_texture
    from .paint_parts import colour_groups, paint_by_parts
    from .uv_paint import ScriptedImageTransport, finish_painted

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    photos = [Path(p) for p in photos]
    log: dict[str, Any] = {"mode": "parts", "steps": {}, "timings_s": {}}
    started = _now()

    meta = PetMetadata.model_validate(record)
    resolved = process_metadata(meta.pet_id, meta.species, meta.weight_kg, meta.coat_color)
    template_glb, masks_dir = template_paths(resolved.base_template)
    template_dir = masks_dir.parent
    log["steps"]["metadata"] = {"pet_id": resolved.pet_id, "base_color": resolved.base_color,
                                "size_class": resolved.size_class, "base_template": resolved.base_template,
                                "photos": [str(p) for p in photos]}

    # 1. every region, from the photograph (or a previous run's reading)
    t = _now()
    if reuse_appearance:
        appearance = canonicalize_appearance(json.loads(Path(reuse_appearance).read_text(encoding="utf-8")))
        appearance.update({"pet_id": resolved.pet_id, "base_template": resolved.base_template})
        log["steps"]["appearance"] = {"reused_from": str(reuse_appearance),
                                      "pattern_summary": f"reused: {appearance.get('pattern')}"}
    else:
        extracted = await extract_appearance(transport, photos, resolved.pet_id, resolved.base_color,
                                             template_dir, samples=samples)
        appearance = {**extracted.appearance, "base_template": resolved.base_template,
                      "size_class": resolved.size_class}
        if extracted.face is not None:
            extracted.face.image.save(out_dir / "face_crop.png")
        log["steps"]["appearance"] = extracted.log
    _json(out_dir / "appearance_stage2.json", appearance)
    log["timings_s"]["appearance"] = round(_now() - t, 2)
    groups = colour_groups(appearance)
    print(f"[pipeline] parts: {log['steps']['appearance']['pattern_summary']}; "
          f"inferred {appearance.get('inferred_regions')}; groups {[(c, len(r)) for c, r in groups]}")

    # 2. paint, one masked edit per colour group
    t = _now()
    if image_transport is None:
        if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("LUNA_API_KEY")):
            raise RuntimeError("parts mode needs OPENAI_API_KEY for the image model (or an image_transport)")
        from .imagegen import OpenAIImageTransport
        image_transport = OpenAIImageTransport()
    base = extract_base_texture(template_glb)
    valid = np.load(Path("output/geometry/valid.npy")) > 0.5
    atlas, paint_report = await paint_by_parts(image_transport, base, photos[0], appearance, masks_dir, valid,
                                               work_dir=out_dir / "paint_steps",
                                               use_cache=not isinstance(image_transport, ScriptedImageTransport))
    build_dir = out_dir / "build_round0"
    build_dir.mkdir(exist_ok=True)
    atlas.save(build_dir / f"{resolved.pet_id}_painted_raw.png")
    coat, finish = finish_painted(atlas, base, masks_dir, template_glb, valid)
    coat.save(build_dir / f"{resolved.pet_id}.png")
    glb = replace_base_texture(template_glb, coat, build_dir / f"{resolved.pet_id}.glb")
    _json(build_dir / "appearance.json", appearance)
    log["steps"]["paint"] = {**paint_report, "finish": finish}
    log["timings_s"]["paint"] = round(_now() - t, 2)
    print(f"[pipeline] painted {len(paint_report['groups'])} colour group(s): "
          f"{[(g['colour'], g['source'], g['mean_change_in_mask']) for g in paint_report['groups']]}")

    # 3. renders for the log (the critic in this mode is the person looking)
    if render and Path(blender).exists():
        from .critic import render_views
        body, face = render_views(glb, out_dir / "renders", blender, render_clip)
        log["steps"]["renders"] = [str(body), str(face)]

    final_dir = out_dir / "final"
    final_dir.mkdir(exist_ok=True)
    for name in (f"{resolved.pet_id}.glb", f"{resolved.pet_id}.png"):
        shutil.copy2(build_dir / name, final_dir / name)
    behavior = resolve_behavior(animation_names(final_dir / f"{resolved.pet_id}.glb"))
    asset = BuiltAsset(pet_id=resolved.pet_id, glb_path=final_dir / f"{resolved.pet_id}.glb",
                       texture_path=final_dir / f"{resolved.pet_id}.png", served_path=resolved.asset_path,
                       behavior=behavior, report={"path": "paint_parts", "paint": paint_report})
    _json(out_dir / "appearance_final.json", appearance)
    log["final"] = {"glb": str(asset.glb_path), "served_path": asset.served_path, "behavior": behavior,
                    "inferred_regions": appearance.get("inferred_regions", [])}
    log["timings_s"]["total"] = round(_now() - started, 2)
    _json(out_dir / "pipeline_log.json", log)
    print(f"[pipeline] done in {log['timings_s']['total']}s -> {asset.glb_path}")
    return PipelineResult(resolved.pet_id, appearance, asset, log, out_dir)


class _ReusedDecision:
    """The shape classify_pattern returns, read back from a previous run."""

    def __init__(self, previous: dict[str, Any], source: str) -> None:
        self.pattern = previous["pattern"]
        self.base_color = previous["base_color"]
        self.second_color = previous.get("second_color")
        self.overlay = previous.get("overlay")
        self.source = source

    def as_dict(self) -> dict[str, Any]:
        return {"pattern": self.pattern, "base_color": self.base_color,
                "second_color": self.second_color, "reused_from": self.source}


def _paint_build(current, asset_dir: Path, photos, resolved, image_transport, state) -> BuiltAsset | None:
    """The repainted-atlas build. None when the repaint is unavailable or
    fails its layout check, so the caller falls back to the mask build."""
    import numpy as np

    from .glb import animation_names, extract_base_texture, replace_base_texture
    from .uv_paint import finish_painted, guide_texture, paint_uv

    entry: dict[str, Any] = {"round": len(state["log"])}
    if image_transport is None:
        if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("LUNA_API_KEY")):
            entry["skipped"] = "no OPENAI_API_KEY for the image model; mask build"
            state["log"].append(entry)
            return None
        from .imagegen import OpenAIImageTransport
        image_transport = OpenAIImageTransport()

    template_glb, masks_dir = template_paths(resolved.base_template)
    asset_dir.mkdir(parents=True, exist_ok=True)
    base = extract_base_texture(template_glb)
    valid = np.load(Path("output/geometry/valid.npy")) > 0.5
    guide = guide_texture(current, masks_dir, valid, base.size, base_texture=base)
    guide.save(asset_dir / f"{resolved.pet_id}_guide.png")
    try:
        # the build runs inside the pipeline's event loop, so the repaint
        # coroutine gets a loop of its own on a worker thread
        from concurrent.futures import ThreadPoolExecutor
        # a stand-in transport must never write the shared cache: an offline
        # run once left the template itself under a live prompt's key
        from .uv_paint import ScriptedImageTransport
        use_cache = not isinstance(image_transport, ScriptedImageTransport)
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(asyncio.run, paint_uv(image_transport, base, photos[0], current, valid,
                                                        list(state["corrections"]), use_cache=use_cache,
                                                        guide=guide)).result()
    except Exception as exc:  # noqa: BLE001
        entry["error"] = f"{type(exc).__name__}: {exc}"
        state["log"].append(entry)
        return None
    entry.update(result.report)
    if not result.accepted:
        state["log"].append(entry)
        return None

    # score every region of the repaint against its target; the regions that
    # missed go into a second roll's prompt as corrections, and the rolls are
    # combined region by region (combine_paints.py) - the second roll only
    # ever replaces regions where it scores better
    from .combine_paints import combine, region_scores
    from .fur import to_linear, to_srgb
    from .texture import MaskLibrary
    from .uv_paint import COLOUR_WORDS

    masks = MaskLibrary(masks_dir, base.size)
    rolls = [to_linear(np.asarray(result.image, dtype=np.float32))]
    scores = region_scores(rolls, masks, current)
    missed = [r for r, sc in scores.items() if sc[0] > MISS_LAB]
    entry["roll1_scores"] = {r: round(sc[0], 1) for r, sc in scores.items()}
    entry["roll1_missed"] = missed
    if missed and len(rolls) < PAINT_ROLLS:
        colours = {e["region"]: e["color"] for e in current.get("region_colors") or []}
        corrections = [f"The {r.replace('_', ' ')} must be {COLOUR_WORDS.get(colours[r], colours[r])} fur."
                       for r in missed if r in colours]
        with ThreadPoolExecutor(max_workers=1) as pool:
            second = pool.submit(asyncio.run, paint_uv(image_transport, base, photos[0], current, valid,
                                                        list(state["corrections"]) + corrections,
                                                        use_cache=use_cache, guide=guide)).result()
        entry["roll2"] = {k: v for k, v in second.report.items() if k != "prompt"}
        entry["roll2_corrections"] = corrections
        if second.accepted:
            rolls.append(to_linear(np.asarray(second.image, dtype=np.float32)))
            second.image.save(asset_dir / f"{resolved.pet_id}_painted_raw_roll2.png")
    if len(rolls) > 1:
        combined, combo = combine(rolls, masks, current, valid)
        entry["combine"] = combo
        painted = Image.fromarray((to_srgb(np.clip(combined, 0, 1)) * 255).astype(np.uint8), mode="RGB")
    else:
        painted = result.image
    painted.save(asset_dir / f"{resolved.pet_id}_painted_raw.png")
    coat, finish = finish_painted(painted, base, masks_dir, template_glb, valid,
                                  eye_colour=current.get("eye_color"))
    entry["finish"] = finish
    coat.save(asset_dir / f"{resolved.pet_id}.png")
    glb_path = replace_base_texture(template_glb, coat, asset_dir / f"{resolved.pet_id}.glb")
    state["log"].append(entry)
    return BuiltAsset(pet_id=resolved.pet_id, glb_path=glb_path, texture_path=asset_dir / f"{resolved.pet_id}.png",
                      served_path=resolved.asset_path,
                      behavior=resolve_behavior(animation_names(glb_path)),
                      report={"path": "uv_paint", "paint": entry})


async def _experimental_views(transport, photo: Path, appearance: dict[str, Any], views_dir: Path):
    """EXPERIMENTAL: the generated-view path's inputs - four views drawn by
    the image model against the template's structure guides, then scored
    for consistency with the appearance. Best effort: every missing piece
    is reported in the log rather than raised, and the default build stands."""
    from .assets import CANONICAL_DIR

    guides_dir = Path("output/guides")
    template_stem = Path(appearance["base_template"]).stem
    report: dict[str, Any] = {"views_dir": str(views_dir), "guides_dir": str(guides_dir)}
    if not (CANONICAL_DIR / "cameras.json").exists():
        report["skipped"] = f"no canonical renders under {CANONICAL_DIR} (run render_preview.py --views)"
        return None, report
    if not (guides_dir / f"{template_stem}_cameras.json").exists():
        report["skipped"] = f"no structure guides under {guides_dir} (run ar_pipeline/render_guides.py)"
        return None, report
    if not (os.environ.get("LUNA_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        report["skipped"] = "no LUNA_API_KEY / OPENAI_API_KEY for the image model"
        return None, report
    try:
        from .imagegen import appearance_to_text
        from .luna import OpenAILunaTransport, generate_views

        views = await generate_views(photo, guides_dir, template_stem, OpenAILunaTransport(), views_dir,
                                     appearance_text=appearance_to_text(appearance))
        report["views"] = {k: str(v) for k, v in views.views.items()}
    except Exception as exc:  # noqa: BLE001
        report["skipped"] = f"view generation failed ({type(exc).__name__}: {exc})"
        return None, report
    try:
        from .consistency import read_back, compare_appearance

        observed = await read_back(views.views, appearance.get("base_color", ""), transport)
        score, mismatches, pattern_match, jaccard = compare_appearance(appearance, observed)
        (views_dir / "consistency.json").write_text(json.dumps({
            "score": round(score, 4), "mismatches": mismatches, "pattern_match": pattern_match,
            "jaccard": round(jaccard, 4)}, ensure_ascii=False, indent=2), encoding="utf-8")
        report["consistency"] = {"score": round(score, 4), "mismatches": mismatches}
    except Exception as exc:  # noqa: BLE001
        report["consistency_error"] = f"{type(exc).__name__}: {exc}"
    return views_dir, report


def load_dotenv(path: str | Path = ".env") -> list[str]:
    """KEY=value lines from a .env in the working directory into the
    environment (existing variables win). Returns the names loaded."""
    path = Path(path)
    if not path.exists():
        return []
    loaded = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def main() -> None:
    loaded = load_dotenv()
    if loaded:
        print(f"[pipeline] .env: loaded {loaded}")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", required=True, help="shelter record JSON (Korean keys)")
    parser.add_argument("--photo", action="append", required=True, help="photo path; repeat for several")
    parser.add_argument("--out", required=True)
    parser.add_argument("--experimental", action="store_true", help="generated-view projection instead of the mask build")
    parser.add_argument("--no-render", action="store_true", help="skip the critic loop's renders")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--provider", default=None, help="anthropic | openai")
    parser.add_argument("--scripted", default=None, help="run offline with a canned script from tests/scripts.py")
    parser.add_argument("--no-views", action="store_true", help="skip the generated-views prior (detailed mode)")
    parser.add_argument("--reuse", default=None, metavar="APPEARANCE_JSON",
                        help="reuse a previous run's reading (its appearance_final.json): no classifier and "
                             "no feature calls - only the image edit, plus the critic unless --rounds 0")
    parser.add_argument("--mode", choices=["paint", "parts", "simple", "detailed"], default="paint",
                        help="paint (default): whole-atlas repaint with a colour guide, a second roll with "
                             "corrections when regions miss, the rolls combined per region; parts: every "
                             "region read, one masked edit per colour group (paints to the mask edge); "
                             "simple: the features reading on the mask build; detailed: ten-region mask build")
    args = parser.parse_args()

    if args.scripted:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
        from scripts import CASES
        transport = CASES[args.scripted]()
    else:
        transport = default_transport(args.provider)

    record = json.loads(Path(args.record).read_text(encoding="utf-8"))
    asyncio.run(run_pipeline(
        record, args.photo, args.out, transport, experimental=args.experimental,
        render=not args.no_render, critic_rounds=args.rounds,
        use_views=not args.no_views, mode=args.mode, reuse_appearance=args.reuse,
    ))


if __name__ == "__main__":
    main()
