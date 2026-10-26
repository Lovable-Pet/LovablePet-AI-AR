"""A full census of all regions, voted across several samples.

WHY A CENSUS AND NOT "ONLY THE DIFFERENCES"
-------------------------------------------
The stage-2 region agent was asked to list only regions whose colour differs
from the recorded base coat, to keep answers short. On the test dog it named
five of sixteen. The other eleven fell to the base colour by default, and the
result looked nothing like the same appearance with all sixteen filled in by
hand - which rendered well. The gap was not the compositor; it was that the
compositor was given a third of the information.

Sixteen regions times an eight-word colour enum is a small answer. Asking for
every region, every time, costs nothing and removes the ambiguity that has
haunted the pipeline since the start: with a census, "the belly is black" and
"I could not see the belly" are two different answers, because visibility is
its own field.

WHY SEVERAL SAMPLES
-------------------
The same photo, sent twice, has come back with contradictory readings that both
carried 0.95 confidence. A model's self-reported confidence does not track its
accuracy here, so it is not used as a gate. Instead N samples are drawn in
parallel and each region takes the colour most of them agree on; the share that
agreed is the region's confidence, which is a measurement rather than a claim.
A region that most samples say is not visible is left unfilled for the
completion step, whatever colour the minority guessed.

The output has the same shape as the old region agent's, so nothing downstream
changes.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .agents import AgentRefusal, Transport
from .fur import FUR_COLORS
from .regions import REGION_DESCRIPTIONS, REGIONS

# How many independent readings to take. Three is the smallest count with a
# majority; five is noticeably steadier on a single low-quality photo.
DEFAULT_SAMPLES = 3

# A region needs this share of samples to call it visible before its colour is
# used at all. Below it, the region is treated as unseen.
VISIBLE_SHARE = 0.5

_REGION_LIST = "\n".join(f"- {region}: {REGION_DESCRIPTIONS[region]}" for region in REGIONS)

CENSUS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "regions": {
            "type": "array",
            "description": "Exactly one entry for every one of the regions listed.",
            "items": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "enum": list(REGIONS)},
                    "color": {
                        "type": "string",
                        "enum": list(FUR_COLORS),
                        "description": "Dominant coat colour of this region.",
                    },
                    "coverage": {
                        "type": "string",
                        "enum": ["full", "partial"],
                        "description": "full if essentially the whole region is that colour.",
                    },
                    "visible": {
                        "type": "boolean",
                        "description": "false if no photograph shows this region well enough to judge.",
                    },
                },
                "required": ["region", "color", "coverage", "visible"],
                "additionalProperties": False,
            },
        },
        "pattern": {
            "type": "string",
            "enum": ["solid", "bicolor", "tricolor", "spotted", "brindle", "merle", "other"],
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["regions", "pattern", "confidence"],
    "additionalProperties": False,
}

CENSUS_SYSTEM = f"""You read the coat of a pet from photographs, one body region at a time, and report every region.

For EACH of the regions below give the dominant coat colour, chosen
from this fixed palette and nothing else: {", ".join(FUR_COLORS)}.
Pick the nearest palette word - a warm beige is cream or fawn, a dark
reddish-brown is chocolate, a mid brown is tan. Report the colour of the fur,
not of the nose, eyes, pads or tongue.

Mark a region visible=false when no photograph shows it well enough to judge:
a dog sitting toward the camera hides its belly and the inside of its legs; a
head-on portrait hides the back. Still give your best-guess colour for a hidden
region, but be honest about visibility - that flag decides whether the colour
is used as an observation or replaced by inference.

Set coverage to "full" when essentially the whole region is that colour and
"partial" when a substantial part of it is a different colour.

The base coat colour from the shelter record is given in the user message as a
hint about the animal's overall colour; do not simply copy it into every
region, and do report a region as the base colour when that is what you see.

Regions and their boundaries:

{_REGION_LIST}

Return exactly one entry per region listed, in any order."""


@dataclass
class CensusResult:
    region_colors: list[dict[str, Any]]
    hidden: list[str]
    pattern: str
    samples: int
    agreement: dict[str, float] = field(default_factory=dict)
    raw: list[dict[str, Any]] = field(default_factory=list)


def _user_text(base_color: str) -> str:
    return (
        f"The shelter record gives the base coat colour as '{base_color}'. "
        "Using every photo provided, report every region listed."
    )


def _normalise(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """One entry per region, dropping anything outside the enums."""
    entries: dict[str, dict[str, Any]] = {}
    for item in payload.get("regions") or []:
        region = item.get("region")
        color = str(item.get("color", "")).strip().lower()
        if region not in REGIONS or color not in FUR_COLORS or region in entries:
            continue
        entries[region] = {
            "color": color,
            "coverage": item.get("coverage", "full"),
            "visible": bool(item.get("visible", True)),
        }
    return entries


def vote(samples: list[dict[str, dict[str, Any]]]) -> CensusResult:
    """Combine several census readings into one.

    Per region, the samples that call it visible vote on the colour; the winning
    colour's share of those votes is the region's confidence. A region most
    samples call hidden is left out and named in `hidden`, so the completion
    step knows it is genuinely unseen rather than merely unmentioned.
    """
    count = len(samples)
    region_colors: list[dict[str, Any]] = []
    hidden: list[str] = []
    agreement: dict[str, float] = {}

    for region in REGIONS:
        readings = [sample[region] for sample in samples if region in sample]
        if not readings:
            hidden.append(region)
            continue

        visible = [reading for reading in readings if reading["visible"]]
        if len(visible) / count < VISIBLE_SHARE:
            hidden.append(region)
            continue

        colour, votes = Counter(reading["color"] for reading in visible).most_common(1)[0]
        coverage = Counter(
            reading["coverage"] for reading in visible if reading["color"] == colour
        ).most_common(1)[0][0]
        share = votes / len(visible)
        agreement[region] = share
        region_colors.append({
            "region": region,
            "color": colour,
            "coverage": coverage,
            "confidence": round(share, 3),
            "note": f"{votes}/{len(visible)} samples agree" if count > 1 else "single sample",
        })

    patterns = Counter(sample.get("__pattern", "other") for sample in samples)
    return CensusResult(
        region_colors=region_colors,
        hidden=hidden,
        pattern=patterns.most_common(1)[0][0] if patterns else "other",
        samples=count,
        agreement=agreement,
    )


async def run_census(
    transport: Transport,
    image_blocks: list[dict[str, Any]],
    base_color: str,
    samples: int = DEFAULT_SAMPLES,
) -> CensusResult:
    """Take `samples` independent readings in parallel and vote them."""
    blocks = [*image_blocks, {"type": "text", "text": _user_text(base_color)}]

    async def one() -> dict[str, Any]:
        return await transport.complete(system=CENSUS_SYSTEM, blocks=blocks, schema=CENSUS_SCHEMA)

    payloads = await asyncio.gather(*(one() for _ in range(samples)), return_exceptions=True)

    readings: list[dict[str, dict[str, Any]]] = []
    raw: list[dict[str, Any]] = []
    for payload in payloads:
        if isinstance(payload, Exception):
            if isinstance(payload, AgentRefusal):
                raise payload
            continue
        raw.append(payload)
        normalised = _normalise(payload)
        normalised["__pattern"] = payload.get("pattern", "other")  # type: ignore[assignment]
        readings.append(normalised)

    if not readings:
        raise RuntimeError("every census sample failed")

    result = vote(readings)
    result.raw = raw
    return result


def describe(result: CensusResult) -> str:
    visible = len(result.region_colors)
    weak = [r for r, share in result.agreement.items() if share < 1.0]
    return (
        f"{result.samples} samples: {visible} regions visible, "
        f"{len(result.hidden)} hidden {result.hidden}, "
        f"pattern={result.pattern}, "
        f"{'unanimous' if not weak else f'disagreement on {weak}'}"
    )
