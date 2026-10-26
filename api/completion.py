"""A third agent, after the parallel two, for the regions no photo showed.

WHY IT EXISTS
-------------
Stage 2 cannot tell the difference between "the belly is black" and "the belly
was never in frame". Both arrive as silence. The pattern table in patterns.py
works around that by inferring from the regions that did appear, which is sound
but indirect.

Asking a model which regions are actually visible in the photographs is not an
inference at all - it is an observation, and it is the one piece of information
the first two agents were never asked for. That is what this agent adds. Only
once a region is established as unseen does it say what the visible coat
implies about it.

WHY IT RUNS AFTER, NOT ALONGSIDE
--------------------------------
A third parallel agent would judge the animal from scratch without knowing what
the other two concluded, and every overlap between region colors and markings
that stage 2 just finished resolving would come back as a three-way conflict.
Running afterwards, it is handed what is already settled and may only fill
what is not.

WHAT IT IS NOT ALLOWED TO DO
----------------------------
A model asked to complete a coat will cheerfully fill every region with
plausible colors, and this describes a real animal someone may travel to meet.
So the schema is the smaller half of the guard and the code is the rest: a
completion survives only if its region is in the enum, was named as not
visible, and carries no observed color already. Survivors are marked with their
source and painted over a smaller area than an observed region, never more
faintly - half-opaque white over black fur is grey fur, which no animal has.

DETERMINISM
-----------
A pet's texture is seeded from its 관리번호 so it never changes between runs. A
model call would break that, so the result is cached against the pet and the
observations it was based on; a re-render reuses it instead of asking again.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .agents import AgentRefusal, Transport

# This agent is off the default path (the pattern defaults now stand in for
# the regions no photo showed). It still gates on the model's own confidence,
# which the default path never does; kept only so the legacy caller runs.
CONFIDENCE_FLOOR = 0.7
from .patterns import INFERRED_WEIGHT, complete_appearance, describe, match_pattern
from .regions import REGION_DESCRIPTIONS, REGIONS

CACHE_DIR = Path("output/cache/completions")

_REGION_LIST = "\n".join(f"- {region}: {REGION_DESCRIPTIONS[region]}" for region in REGIONS)

COMPLETION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "not_visible": {
            "type": "array",
            "description": "Regions no photograph shows well enough to judge.",
            "items": {"type": "string", "enum": list(REGIONS)},
        },
        "completions": {
            "type": "array",
            "description": "Only regions listed in not_visible.",
            "items": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "enum": list(REGIONS)},
                    "color": {"type": "string", "description": "Color name in English."},
                    "coverage": {"type": "string", "enum": ["full", "partial"]},
                    "reason": {
                        "type": "string",
                        "description": "Which visible regions imply this, in one phrase.",
                    },
                },
                "required": ["region", "color", "coverage", "reason"],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["not_visible", "completions", "confidence"],
    "additionalProperties": False,
}

COMPLETION_SYSTEM = f"""You decide which parts of a pet the photographs do not show, and what the parts that ARE shown imply about them.

The user message gives you the base coat color and the region colors already
established from these same photographs. Those are settled. Never restate them,
never contradict them, and never return one of those regions.

Work in two steps.

First, list in "not_visible" every region that no photograph shows well enough
to judge. A dog sitting facing the camera hides its belly, the inside of its
legs and the underside of its tail; a head-on portrait hides the back. Be
strict: a region you can see and that simply matches the base coat color is
visible, not hidden, and does not belong in this list.

Second, for regions in that list only, say in "completions" what the visible
coat implies. Coat patterns are not arbitrary - urajiro puts cream along the
whole underside, tan points appear above the eyes and low on the legs together,
Irish spotting runs collar to chest to feet - so visible regions constrain
hidden ones. State in "reason" which visible regions drive each conclusion.

Complete a region only when the visible coat genuinely implies it. If the
pattern says nothing about a hidden region, leave it out: it keeps the base
coat color, which is the honest answer. Returning nothing at all is a normal
result.

The region names and their boundaries:

{_REGION_LIST}

Set confidence to how strongly the visible regions constrain the hidden ones:
above 0.8 when they form a clear, well-known pattern, below 0.7 when the coat
is ambiguous or too little of the animal is in frame to tell."""


def inferred_regions(appearance: dict[str, Any]) -> list[str]:
    """The regions whose colour is an assumption rather than an observation:
    the ones still carrying a pattern default (or a legacy completion) after
    the merge. Recorded on the appearance so the compositor, the critic and
    the log can treat them as such."""
    return [
        entry["region"] for entry in appearance.get("region_colors") or []
        if entry.get("region") in REGIONS and (
            str(entry.get("source", "")).startswith("pattern_default")
            or str(entry.get("source", "")).startswith("pattern:")
            or entry.get("source") == "llm_completion"
        )
    ]


def _fingerprint(appearance: dict[str, Any]) -> str:
    """Identifies the observations a completion was based on, so a cached one is
    only reused for the same inputs."""
    payload = json.dumps(
        {
            "base_color": appearance.get("base_color"),
            "region_colors": sorted(
                (entry.get("region"), entry.get("color"), entry.get("coverage"))
                for entry in appearance.get("region_colors") or []
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _cache_path(appearance: dict[str, Any], cache_dir: Path) -> Path:
    pet_id = str(appearance.get("pet_id") or appearance.get("petId") or "unknown")
    return cache_dir / f"{pet_id}-{_fingerprint(appearance)}.json"


def _user_text(appearance: dict[str, Any]) -> str:
    observed = appearance.get("region_colors") or []
    if observed:
        lines = "\n".join(
            f"- {entry.get('region')}: {entry.get('color')} ({entry.get('coverage')})"
            for entry in observed
        )
    else:
        lines = "- (none; the whole animal matches the base coat color)"
    return (
        f"The shelter record gives the base coat color as "
        f"'{appearance.get('base_color')}'.\n\n"
        f"Already established from these photographs:\n{lines}\n\n"
        "Using every photo provided, list the regions these photographs do not "
        "show, and complete only those."
    )


def _accept(payload: dict[str, Any], observed: set[str]) -> list[dict[str, Any]]:
    """Keep only completions the guard allows, in order, one per region."""
    hidden = {region for region in payload.get("not_visible") or [] if region in REGIONS}
    accepted: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in payload.get("completions") or []:
        region = item.get("region")
        if region not in REGIONS or region in observed or region not in hidden:
            continue
        if region in seen:
            continue
        seen.add(region)
        accepted.append(
            {
                "region": region,
                "color": str(item.get("color", "")),
                "coverage": item.get("coverage", "partial"),
                "note": str(item.get("reason", "")),
                "source": "llm_completion",
                "weight": INFERRED_WEIGHT,
            }
        )
    return accepted


async def complete_with_agent(
    appearance: dict[str, Any],
    transport: Transport,
    image_blocks: list[dict[str, Any]],
    cache_dir: str | Path = CACHE_DIR,
    use_cache: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Fill the regions the photos do not show. Returns the appearance, the
    entries added, and a one-line account of what happened.

    Falls back to the deterministic pattern table whenever the model cannot be
    reached, declines, or is not confident enough to be worth showing.
    """
    cache_dir = Path(cache_dir)
    cached = _cache_path(appearance, cache_dir)
    observed = {entry.get("region") for entry in appearance.get("region_colors") or []}

    if use_cache and cached.exists():
        added = json.loads(cached.read_text(encoding="utf-8"))
        if not added:
            return appearance, [], "cached: nothing to complete"
        return (
            {**appearance, "region_colors": list(appearance["region_colors"]) + added},
            added,
            f"cached: {[entry['region'] for entry in added]}",
        )

    try:
        payload = await transport.complete(
            system=COMPLETION_SYSTEM,
            blocks=[*image_blocks, {"type": "text", "text": _user_text(appearance)}],
            schema=COMPLETION_SCHEMA,
        )
    except (AgentRefusal, Exception) as exc:  # noqa: BLE001 - any failure falls back
        filled, added = complete_appearance(appearance)
        return filled, added, f"agent unavailable ({type(exc).__name__}); used the pattern table"

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    if confidence < CONFIDENCE_FLOOR:
        filled, added = complete_appearance(appearance)
        return (
            filled,
            added,
            f"agent confidence {confidence:.2f} below {CONFIDENCE_FLOOR}; "
            f"used the pattern table ({describe(match_pattern(appearance.get('region_colors') or []))})",
        )

    added = _accept(payload, observed)
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(added, ensure_ascii=False, indent=2), encoding="utf-8")

    hidden = [region for region in payload.get("not_visible") or [] if region in REGIONS]
    if not added:
        return appearance, [], f"agent (confidence {confidence:.2f}): nothing to complete, hidden={hidden}"

    return (
        {**appearance, "region_colors": list(appearance["region_colors"]) + added},
        added,
        f"agent (confidence {confidence:.2f}): hidden={hidden}, "
        f"completed={[entry['region'] for entry in added]}",
    )
