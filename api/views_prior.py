"""Stage 2-6 - what a "typical dog like this" looks like, from generated views.

WHY
---
A shelter photo shows one side of a sitting animal. The regions it does not
show fall back to the coat pattern's defaults, which know five layouts and
nothing about a mixed-breed dog. An image model shown the photo can draw
the animal from the other sides, and that drawing carries a prior the
table does not have: where the white on this dog's chest would stop, what
the underside of its tail looks like, how far the dark back reaches down
the flank.

The drawing is NOT put on the mesh. Projecting its pixels drags along its
outlines, its flat cartoon shading and every place its pose differs from
the mesh (the experimental path shows what that looks like). The views are
used in two narrower ways:

  B. READ. The four views go through the same region-colour agent as the
     photograph, voted the same way, and what they say fills the regions
     the photograph left at the default. Priority stays
         face call > body call > views > pattern default.

  C. SHAPES. Each view is reduced to a label map - every pixel snapped to
     the nearest of the coat's few colour names - and the labels, not the
     pixels, are projected onto the mesh with the same cameras the
     projection path uses. Thin dark strokes (drawn outlines) are dropped
     before projecting. The result is, per texel, "which of this coat's
     colours the drawing put here", which the compositor uses only inside
     regions that carry a marking or a partial coverage - the places where
     a region's single colour is known to be too coarse - and paints with
     its own fur ramps. The picture never reaches the texture; its
     silhouettes of the markings do.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from .agents import REGION_COLOR_AGENT, AgentResult, Transport, run_agent
from .extract import photo_blocks
from .fur import FUR_COLORS, FUR_PALETTE, hex_to_linear
from .regions import REGIONS

GUIDES_DIR = Path("output/guides")
VIEW_NAMES = ("front", "left", "right", "back")
SOURCE = "views_read"

# A pixel darker than this (linear luma) that lies within OUTLINE_PX of a
# lighter pixel is a drawn stroke, not fur, and casts no label.
OUTLINE_LUMA = 0.04
OUTLINE_PX = 4
# Labels are trusted where at least this much projection weight landed...
MIN_LABEL_WEIGHT = 0.15
# ...after the votes are blurred at this scale (texels), and only where the
# winning colour holds this share of the neighbourhood's votes.
LABEL_SMOOTH_SIGMA = 5.0
LABEL_MIN_SHARE = 0.6


@dataclass
class ViewsPrior:
    views_dir: Path
    regions: AgentResult | None
    label_map: np.ndarray | None        # (h, w) int, index into `colours`, -1 where none
    label_weight: np.ndarray | None     # (h, w) 0..1
    colours: list[str]
    report: dict[str, Any] = field(default_factory=dict)


def available(template_stem: str) -> tuple[bool, str]:
    if not (GUIDES_DIR / f"{template_stem}_cameras.json").exists():
        return False, f"no structure guides under {GUIDES_DIR} (run ar_pipeline/render_guides.py)"
    if not (os.environ.get("LUNA_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        return False, "no LUNA_API_KEY / OPENAI_API_KEY for the image model"
    return True, ""


def _palette_lab(colours: list[str], palette: dict[str, tuple[str, str, str]] | None) -> np.ndarray:
    from .projection import linear_to_lab

    mids = []
    for name in colours:
        stops = (palette or {}).get(name) or FUR_PALETTE[name]
        mids.append(linear_to_lab(hex_to_linear(stops[1])[None, None, :])[0, 0])
    return np.stack(mids).astype(np.float32)


def label_view(rgb_linear: np.ndarray, alpha: np.ndarray, mids_lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest coat colour per pixel, and a weight that is 0 on outlines and
    outside the animal."""
    from .projection import linear_to_lab

    lab = linear_to_lab(np.clip(rgb_linear, 1e-4, 1.0))
    weights = np.array([0.6, 1.0, 1.0], dtype=np.float32)
    distance = (((lab[..., None, :] - mids_lab[None, None, :, :]) * weights) ** 2).sum(-1)
    label = distance.argmin(-1)

    luma = rgb_linear @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    dark = luma < OUTLINE_LUMA
    near_light = ndimage.binary_dilation(~dark & (alpha > 0.5), iterations=OUTLINE_PX)
    outline = dark & near_light & ~ndimage.binary_erosion(dark, iterations=OUTLINE_PX)
    weight = (alpha > 0.5) & ~outline
    return label, weight.astype(np.float32)


def project_labels(views_dir: Path, cameras_path: Path, geometry_dir: Path,
                   colours: list[str], palette=None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Vote every view's labels onto the mesh. Returns label map, weight, report."""
    from .projection import Geometry, load_views, project_view

    geometry = Geometry.load(geometry_dir)
    views = load_views(cameras_path, geometry_dir, views_dir)
    mids = _palette_lab(colours, palette)
    h, w = geometry.valid.shape
    votes = np.zeros((h, w, len(colours)), dtype=np.float32)
    per_view: dict[str, float] = {}
    for view in views:
        label, weight = label_view(view.rgb, view.alpha, mids)
        # project the labels as one-hot planes, three at a time, through
        # the ordinary colour projection (bilinear on one-hot = soft votes)
        for start in range(0, len(colours), 3):
            planes = np.zeros(label.shape + (3,), dtype=np.float32)
            for k in range(3):
                index = start + k
                if index < len(colours):
                    planes[..., k] = (label == index) * weight
            from dataclasses import replace
            colour, texel_weight = project_view(geometry, replace(view, rgb=planes))
            for k in range(3):
                index = start + k
                if index < len(colours):
                    votes[..., index] += colour[..., k] * texel_weight
        per_view[view.name] = round(float((weight > 0).mean()), 3)
    # smooth the votes before deciding: a drawing's edge lands a few texels
    # off where the mesh's edge is, and at grazing angles the votes alternate
    # texel by texel, which read as stripes on the cheek. A blur at the scale
    # of a marking's edge and a clear majority make each texel's label the
    # label of its neighbourhood, not of one projected pixel.
    smoothed = np.stack([ndimage.gaussian_filter(votes[..., i], LABEL_SMOOTH_SIGMA)
                         for i in range(votes.shape[-1])], -1)
    total = smoothed.sum(-1)
    share = smoothed.max(-1) / np.maximum(total, 1e-6)
    label_map = np.where((total > MIN_LABEL_WEIGHT) & (share >= LABEL_MIN_SHARE), smoothed.argmax(-1), -1)
    label_weight = np.clip(total / max(float(np.percentile(total[total > 0], 90)), 1e-6), 0, 1) if (total > 0).any() else total
    label_weight = (label_weight * np.clip((share - LABEL_MIN_SHARE) / (1 - LABEL_MIN_SHARE), 0, 1)).astype(np.float32)
    label_weight = np.where(label_map >= 0, label_weight, 0.0).astype(np.float32)
    report = {"labelled_share_of_valid": round(float((label_map >= 0).sum() / max(geometry.valid.sum(), 1)), 3),
              "view_coverage": per_view, "colours": colours}
    return label_map, label_weight, report


async def views_prior(
    transport: Transport,
    photo: str | Path,
    appearance: dict[str, Any],
    template_stem: str,
    out_dir: str | Path,
    geometry_dir: str | Path,
    palette=None,
    samples: int = 3,
    generate: bool = True,
) -> ViewsPrior:
    """Generate the views (or reuse the ones in out_dir), read them, and
    project their labels. Every failure is reported, never raised."""
    out_dir = Path(out_dir)
    report: dict[str, Any] = {"views_dir": str(out_dir)}
    ok, why = available(template_stem)
    have_views = all((out_dir / f"{n}.png").exists() for n in VIEW_NAMES)
    if not have_views:
        if not ok:
            report["skipped"] = why
            return ViewsPrior(out_dir, None, None, None, [], report)
        if not generate:
            report["skipped"] = "no views on disk and generation disabled"
            return ViewsPrior(out_dir, None, None, None, [], report)
        try:
            from .imagegen import appearance_to_text
            from .luna import OpenAILunaTransport, generate_views

            result = await generate_views(photo, GUIDES_DIR, template_stem, OpenAILunaTransport(), out_dir,
                                          appearance_text=appearance_to_text(appearance))
            report["generated"] = {k: str(v) for k, v in result.views.items()}
        except Exception as exc:  # noqa: BLE001
            report["skipped"] = f"view generation failed ({type(exc).__name__}: {exc})"
            return ViewsPrior(out_dir, None, None, None, [], report)
    else:
        report["generated"] = "reused views on disk"

    # B. read the views like a photograph
    regions = None
    try:
        estimate = {e["region"]: e["color"] for e in appearance.get("region_colors") or []}
        blocks = photo_blocks([out_dir / f"{n}.png" for n in VIEW_NAMES])
        regions = await run_agent(REGION_COLOR_AGENT, transport, blocks, appearance.get("base_color", ""),
                                  tuple(REGIONS), estimate, samples)
        report["read"] = {
            "reported": [{k: e.get(k) for k in ("region", "color", "coverage", "gradient_to", "gradient_anchor")}
                         for e in regions.payload.get("region_colors") or []],
            "disagreed": regions.disagreed, "confidence": regions.confidence,
        }
    except Exception as exc:  # noqa: BLE001
        report["read_error"] = f"{type(exc).__name__}: {exc}"

    # C. the shapes
    label_map = label_weight = None
    colours = sorted({appearance.get("base_color", "")} | {e["color"] for e in appearance.get("region_colors") or []}
                     | {m["color"] for m in appearance.get("markings") or []}
                     | ({e["color"] for e in regions.payload.get("region_colors") or []} if regions else set()))
    colours = [c for c in colours if c in FUR_COLORS]
    try:
        cameras = GUIDES_DIR / f"{template_stem}_cameras.json"
        label_map, label_weight, shape_report = project_labels(out_dir, cameras, Path(geometry_dir), colours, palette)
        report["shapes"] = shape_report
        preview = np.zeros(label_map.shape + (3,), dtype=np.uint8)
        for index, name in enumerate(colours):
            rgb = (hex_to_linear(FUR_PALETTE[name][1]) ** (1 / 2.2) * 255).astype(np.uint8)
            preview[label_map == index] = rgb
        Image.fromarray(preview).save(out_dir / "labels_preview.png")
    except Exception as exc:  # noqa: BLE001
        report["shapes_error"] = f"{type(exc).__name__}: {exc}"

    return ViewsPrior(out_dir, regions, label_map, label_weight, colours, report)


def fill_from_views(appearance: dict[str, Any], prior: ViewsPrior) -> tuple[dict[str, Any], list[str]]:
    """Regions still at the pattern default take the views' reading."""
    if prior.regions is None:
        return appearance, []
    read = {e["region"]: e for e in prior.regions.payload.get("region_colors") or []}
    filled: list[str] = []
    entries = []
    for entry in appearance.get("region_colors") or []:
        region = entry["region"]
        if str(entry.get("source", "")).startswith("pattern_default") and region in read:
            entries.append({**read[region], "source": SOURCE, "weight": 1.0})
            filled.append(region)
        else:
            entries.append(entry)
    # a region with no entry at all (pattern "other") is added outright
    present = {e["region"] for e in entries}
    for region, entry in read.items():
        if region not in present:
            entries.append({**entry, "source": SOURCE, "weight": 1.0})
            filled.append(region)
    entries = [e for r in REGIONS for e in entries if e["region"] == r]
    inferred = [r for r in appearance.get("inferred_regions") or [] if r not in filled]
    return {**appearance, "region_colors": entries, "inferred_regions": inferred}, filled


__all__ = ["ViewsPrior", "available", "fill_from_views", "label_view", "project_labels", "views_prior"]
