"""Deterministic merge of the agent results. No model call happens here.

PRIORITY
--------
Three sources can speak for a region, and they are ranked:

    face call  >  body call  >  pattern default

The face call read a crop of the head at full resolution, so on the six
face regions it is the better witness than the body call, which saw the
head as a corner of the frame. Both outrank the default, which is not an
observation at all - it is what the coat pattern says a region usually is,
kept only where no photograph could answer. A region no call reported keeps
its default and is listed in `inferred_regions`, so everyone downstream
knows which parts of this animal were read and which were assumed.

The signature takes primitives rather than the shelter record, so display-only
fields have no path into the merged appearance at all.
"""

from __future__ import annotations

from typing import Any

from .agents import AgentResult
from .regions import REGIONS

SOURCE_FACE = "face_call"
SOURCE_BODY = "body_call"


def supersede_markings(
    region_colors: list[dict[str, Any]], markings: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop markings that only repeat an observed region colour.

    Both agents see the same photo and the regions are fine enough that one
    feature can honestly read as either - tan eyebrows are a region color to
    one agent and a patch to the other. When the marking's colour is the
    colour the region was observed to be, it says nothing the region entry
    does not, and painting it again would only cut a second edge into the
    same fur. A marking in a DIFFERENT colour on an observed region - a
    chocolate patch on a white skull - is exactly what the markings agent is
    for, and it stays. A default colour never claims a region.
    """
    from .patterns import color_family

    observed = {
        entry.get("region"): str(entry.get("color", "")).lower() for entry in region_colors
        if not str(entry.get("source", "")).startswith("pattern_default")
    }

    def redundant(marking: dict[str, Any]) -> bool:
        region = marking.get("region")
        if region not in observed:
            return False
        colour = str(marking.get("color", "")).lower()
        return colour == observed[region] or color_family(colour) == color_family(observed[region])

    kept = [m for m in markings if not redundant(m)]
    superseded = [m for m in markings if redundant(m)]
    return kept, superseded


def _observations(result: AgentResult | None, source: str) -> dict[str, dict[str, Any]]:
    if result is None:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for item in result.payload.get("region_colors") or []:
        region = item.get("region")
        if region in REGIONS and region in result.scope and region not in out:
            out[region] = {**item, "source": source}
    return out


def merge_scoped(
    pet_id: str,
    base_color: str,
    defaults: list[dict[str, Any]],
    body: AgentResult | None,
    face: AgentResult | None,
    body_markings: AgentResult | None = None,
    face_markings: AgentResult | None = None,
    pattern: str = "other",
    overlay: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the merged appearance and a provenance report.

    `defaults` is the ten-region estimate (possibly empty). Every region
    ends up with at most one colour entry; the entry carries `source` so the
    compositor and the log can tell an observation from an assumption.
    """
    by_region: dict[str, dict[str, Any]] = {}
    for entry in defaults:
        region = entry.get("region")
        if region in REGIONS:
            by_region[region] = dict(entry)

    provenance: dict[str, str] = {r: "default" if r in by_region else "none" for r in REGIONS}
    for source, result in ((SOURCE_BODY, body), (SOURCE_FACE, face)):
        for region, entry in _observations(result, source).items():
            by_region[region] = entry
            provenance[region] = source

    region_colors = [by_region[r] for r in REGIONS if r in by_region]

    markings: list[dict[str, Any]] = []
    for result in (face_markings, body_markings):
        if result is None:
            continue
        for item in result.payload.get("markings") or []:
            if item.get("region") in REGIONS and item.get("region") in result.scope:
                markings.append(dict(item))
    markings, superseded = supersede_markings(region_colors, markings)

    notes = [str(e.get("note", "")).strip() for e in region_colors if e.get("source") != "default"
             and not str(e.get("source", "")).startswith("pattern_default")]
    for result in (face_markings, body_markings):
        if result is not None:
            notes.append(str(result.payload.get("note", "")).strip())

    inferred = [r for r in REGIONS if provenance[r] == "default"]
    unknown = [r for r in REGIONS if provenance[r] == "none"]
    disagreed = sorted(set(
        region for result in (body, face, body_markings, face_markings) if result is not None
        for region in result.disagreed
    ))
    confidences = [r.confidence for r in (body, face, body_markings, face_markings) if r is not None]

    merged = {
        "pet_id": pet_id,
        "base_color": base_color,
        "pattern": pattern,
        "overlay": overlay,
        "color_count": len({base_color, *(e.get("color") for e in region_colors)}),
        "region_colors": region_colors,
        "markings": markings,
        "description": " ".join(note for note in notes if note),
        # agreement between independent readings, never the model's own number
        "confidence": min(confidences) if confidences else 0.0,
        "inferred_regions": inferred,
        "unobserved_regions": unknown,
    }
    report = {
        "provenance": provenance,
        "inferred_regions": inferred,
        "unobserved_regions": unknown,
        "disagreed_regions": disagreed,
        "superseded_markings": superseded,
        "observed_face": sorted(r for r, s in provenance.items() if s == SOURCE_FACE),
        "observed_body": sorted(r for r, s in provenance.items() if s == SOURCE_BODY),
    }
    return merged, report


def merge_appearance(
    pet_id: str, base_color: str, region_result: AgentResult, feature_result: AgentResult
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The unscoped merge the consistency check (experimental path) still
    uses: one region reading and one markings reading over the whole animal."""
    merged, report = merge_scoped(
        pet_id, base_color, [], region_result, None, feature_result, None,
        pattern=region_result.payload.get("pattern", "other"),
    )
    return merged, report["superseded_markings"]
