"""The two vision agents and the sampling policy around them.

Both agents see the same photos and are told the base coat color from the
shelter record; neither is asked to decide it. Their scopes are disjoint and
neither system prompt mentions the other's, so they don't negotiate a split
between themselves.

SCOPE AND ESTIMATE
------------------
An agent is run on a subset of the regions (`regions_in_scope`) and handed
the current estimate for them (`current_estimate`, from the pattern
defaults). It is asked to report only the regions whose colour differs from
the estimate: a short answer about what the photograph contradicts, rather
than a long one that re-describes what the layout already said. The face
call sees a face crop and the five face regions; the body call sees the whole
photo and the other ten. With no estimate at all (pattern "other") the agent
is asked for every region in its scope.

CONFIDENCE
----------
The model's own confidence number is not in the schema any more. Two
readings of one photo once came back contradictory at 0.95 each, so nothing
is gated on it. Every agent is sampled three times in parallel and each
region is voted: a reading that names the region votes for that colour, a
reading that does not name it votes for the current estimate (silence means
"the estimate is right"), and a reading that names the estimate's own
colour is the same vote. A colour with a majority wins; failing that, a
colour FAMILY (light / point / dark) with a majority wins and the family's
most-named colour is used, so "tan" against "cream" on a pale throat is a
disagreement about the word, not the coat. A region with no majority is
dropped and listed, so downstream treats it as unobserved. The result's
confidence is the mean winning share over the regions it reports.

Output is constrained with output_config.format (json_schema). The schemas are
written out by hand rather than generated from Pydantic models so they stay
flat - an array of plain objects, no $defs and no $ref.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Protocol

import anthropic

from .fur import FUR_COLORS
from .regions import FACE_REGIONS, REGION_DESCRIPTIONS, REGIONS

MODEL = "claude-opus-5"
EFFORT = "medium"
MAX_TOKENS = 16000

# Server-side refusal fallbacks: on a policy decline the API re-runs the same
# request on a fallback model within the same call. Set to False if the beta
# is not enabled for the account.
USE_REFUSAL_FALLBACK = True
REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"

MAX_ATTEMPTS = 3

# Every agent is read this many times, in parallel, and a region is kept when
# a majority of the readings agree on it. Two readings had to be unanimous
# and, live, gpt-4o's two readings of one photo agreed on nothing - every
# region fell back to the layout default. Three with a majority is the
# smallest count that survives one stray reading.
SAMPLES = 3
# How many times a failed sample is re-issued before the vote goes ahead short.
SAMPLE_RETRIES = 2

# The vocabulary the texture stage can place things against (landmarks.json).
GRADIENT_ANCHORS = ("eye", "nose", "belly_center", "paw", "ear_base", "none")
MARKING_ANCHORS = ("left_eye", "right_eye", "nose", "ear_base", "chest_center", "none")
MARKING_OFFSETS = ("above", "below", "left", "right", "on")
MARKING_DISTANCES = ("near", "far")

_REGION_PROPERTY = {
    "type": "string",
    "enum": list(REGIONS),
    "description": "Body region. Must be exactly one of the listed values.",
}

# The texture stage turns a colour name into a three-stop fur ramp, so a name
# outside the palette has nothing to map to. Constraining it here means the
# model never offers one.
_COLOR_PROPERTY = {
    "type": "string",
    "enum": list(FUR_COLORS),
    "description": "Coat colour. Must be exactly one of the listed values.",
}

REGION_COLOR_SCHEMA = {
    "type": "object",
    "properties": {
        "region_colors": {
            "type": "array",
            "description": "Only regions whose colour differs from the current estimate.",
            "items": {
                "type": "object",
                "properties": {
                    "region": _REGION_PROPERTY,
                    "color": _COLOR_PROPERTY,
                    "coverage": {"type": "string", "enum": ["full", "partial"]},
                    "strength": {
                        "type": "string",
                        "enum": ["faint", "medium", "strong"],
                        "description": (
                            "How strongly the colour shows against the fur next to it: faint is a "
                            "tint you notice only on a second look, strong is unmistakable."
                        ),
                    },
                    "gradient_to": {
                        "type": "string",
                        "enum": [*FUR_COLORS, "none"],
                        "description": (
                            "If the region fades from `color` into another colour, that "
                            "colour; none when the region is one colour."
                        ),
                    },
                    "gradient_anchor": {
                        "type": "string",
                        "enum": list(GRADIENT_ANCHORS),
                        "description": (
                            "Where `color` is strongest, when there is a gradient: the fade "
                            "runs away from this landmark. none when there is no gradient."
                        ),
                    },
                    "note": {"type": "string", "description": "One short phrase."},
                },
                "required": ["region", "color", "coverage", "strength", "gradient_to", "gradient_anchor", "note"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["region_colors"],
    "additionalProperties": False,
}

MARKINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "markings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["spot", "patch", "stripe", "mask", "socks", "blaze", "other"],
                    },
                    "region": _REGION_PROPERTY,
                    "color": _COLOR_PROPERTY,
                    "size": {"type": "string", "enum": ["small", "medium", "large"]},
                    "anchor": {
                        "type": "string",
                        "enum": list(MARKING_ANCHORS),
                        "description": "The landmark the marking sits by; none if no landmark is near.",
                    },
                    "offset": {
                        "type": "string",
                        "enum": list(MARKING_OFFSETS),
                        "description": "Where the marking is relative to the anchor, as seen in the photo.",
                    },
                    "distance": {
                        "type": "string",
                        "enum": list(MARKING_DISTANCES),
                        "description": "near: touching or within one eye-width; far: further than that.",
                    },
                },
                "required": ["type", "region", "color", "size", "anchor", "offset", "distance"],
                "additionalProperties": False,
            },
        },
        "note": {"type": "string", "description": "One short sentence, or empty if none."},
    },
    "required": ["markings", "note"],
    "additionalProperties": False,
}


def region_rule(regions: tuple[str, ...] | list[str]) -> str:
    listing = "\n".join(f"- {region}: {REGION_DESCRIPTIONS[region]}" for region in regions)
    return (
        'The "region" field must be exactly one of the regions below. Their edges '
        "are anatomical and several of them meet, so decide by the definition "
        "rather than by what the name sounds like:\n\n"
        f"{listing}\n\n"
        "Never name a region outside this list. If something you see does not fall "
        "inside one of these, leave it out entirely."
    )


REGION_COLOR_SYSTEM = """You read the large-area coat coloring of a pet from photographs, region by region.

The base coat color is given to you in the user message. It is established fact
from the shelter record: never restate it, re-judge it, or second-guess it.

The user message also gives the CURRENT ESTIMATE of each region's colour, from
the coat pattern this dog was matched to. Report a region only when the
photograph clearly shows a different colour from its estimate. A region whose
estimate is right must not appear in your answer at all - that keeps the
response short and is the whole point of being given the estimate. If no
estimate is given, report every region in scope that you can see.

{region_rule}

List each region at most once. If a region carries more than one color,
report the one covering the largest area of it, and if the region visibly
FADES from that color into another - a tan cheek paling to cream toward the
throat, black on the back thinning to grey toward the belly - name the other
color in gradient_to and, in gradient_anchor, the landmark the main color is
strongest at. Otherwise both are "none".

Use coverage "full" when essentially the entire region is that color, and
"partial" when only a portion of it is. Set strength to how strongly the
color shows: "faint" for a light wash or tint (a warm cast on pale legs, a
hint of cream on a cheek), "medium" for a clear but soft colour, "strong"
for a saturated, unmistakable one. Judge the fur's own colour, not the
lighting.

Report broad areas of color only. Small localized features are out of your
scope and must not be listed as region colors. A region you cannot see in
any photograph must not be reported.

Use "pink" only for BARE SKIN showing through thin hair - the inside of an
ear, a bald belly - never for fur that merely looks warm. A pale dog's ear
lining is pink, not red."""

MARKINGS_SYSTEM = """You catalogue the markings on a pet that its region coloring cannot express.

The base coat color is given to you in the user message as established fact
from the shelter record: never restate it, re-judge it, or report it as a
marking.

{region_rule}

Those regions are small and anatomical, and each one's color is recorded
separately. A feature that simply fills its region is therefore that region's
color and not a marking: tan eyebrows, a cream throat, white paws, a dark
muzzle bridge all belong to the region, however striking they look. Leave them
out.

Report only what the region colors cannot say:
- a feature that divides a region instead of filling it, such as a blaze
  running down the middle of the skull;
- a feature that ignores the region edges, such as spots scattered across
  several of them;
- a feature on one side of the animal only, where the other side differs.

Place each marking against a landmark when one is near: the eye it sits by,
the nose, an ear base or the centre of the chest, with offset saying which
side of that landmark it is on as seen in the photograph and distance saying
whether it touches the landmark (near, within one eye-width) or sits further
off (far). Use anchor "none" when no landmark is nearby.

Judge size relative to the region the marking sits on. Anything covering most
of its region is that region's color and should not be here at all, so "large"
ought to be rare.

Set note to one short sentence summarizing the markings, or an empty string if
there are none. Finding none is a normal and common answer."""


class AgentRefusal(RuntimeError):
    """The model declined the request and no fallback produced an answer."""


@dataclass(frozen=True)
class AgentSpec:
    name: str
    system: str          # with a {region_rule} placeholder
    schema: dict[str, Any]
    list_key: str
    user_text: str

    def system_for(self, regions: tuple[str, ...] | list[str]) -> str:
        return self.system.format(region_rule=region_rule(regions))


REGION_COLOR_AGENT = AgentSpec(
    name="region_colors",
    system=REGION_COLOR_SYSTEM,
    schema=REGION_COLOR_SCHEMA,
    list_key="region_colors",
    user_text=(
        "The shelter record gives the base coat color as '{base_color}'.\n\n"
        "{estimate}\n\n"
        "Using every photo provided, list the regions in scope whose coloring "
        "clearly differs from the estimate."
    ),
)

MARKINGS_AGENT = AgentSpec(
    name="markings",
    system=MARKINGS_SYSTEM,
    schema=MARKINGS_SCHEMA,
    list_key="markings",
    user_text=(
        "The shelter record gives the base coat color as '{base_color}'.\n\n"
        "{estimate}\n\n"
        "Using every photo provided, catalogue the localized markings on the "
        "regions in scope that stand out from their coloring."
    ),
)


class Transport(Protocol):
    """Seam between the sampling policy and the API, so the policy can be
    exercised without network access."""

    async def complete(
        self, *, system: str, blocks: list[dict[str, Any]], schema: dict[str, Any]
    ) -> dict[str, Any]: ...


class AnthropicTransport:
    def __init__(
        self,
        client: anthropic.AsyncAnthropic | None = None,
        model: str = MODEL,
        effort: str = EFFORT,
    ) -> None:
        self._client = client or anthropic.AsyncAnthropic()
        self._model = model
        self._effort = effort

    async def complete(
        self, *, system: str, blocks: list[dict[str, Any]], schema: dict[str, Any]
    ) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if USE_REFUSAL_FALLBACK:
            extra = {"betas": [REFUSAL_FALLBACK_BETA], "fallbacks": "default"}

        response = await self._client.beta.messages.create(
            model=self._model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": blocks}],
            output_config={
                "effort": self._effort,
                "format": {"type": "json_schema", "schema": schema},
            },
            **extra,
        )

        if response.stop_reason == "refusal":
            detail = getattr(response.stop_details, "category", None)
            raise AgentRefusal(f"model declined the request (category={detail})")

        # thinking blocks can precede the answer, so pick the text block out
        text = next(block.text for block in response.content if block.type == "text")
        return json.loads(text)


@dataclass
class AgentResult:
    name: str
    payload: dict[str, Any]
    confidence: float
    calls: int = 0
    agreed: bool = False
    dropped: list[dict[str, Any]] = field(default_factory=list)
    # regions the readings did not agree on - left out, treated as unobserved
    disagreed: list[str] = field(default_factory=list)
    scope: tuple[str, ...] = ()
    # per (region, colour) pair: the share of readings that named it
    agreement: dict[str, float] = field(default_factory=dict)
    # calls that failed and were retried
    errors: list[str] = field(default_factory=list)


def estimate_text(regions: tuple[str, ...] | list[str],
                  current_estimate: dict[str, str] | None) -> str:
    """The scope and the estimate, as the user message states them."""
    scope = ", ".join(regions)
    if not current_estimate:
        return (
            f"Regions in scope: {scope}.\n"
            "There is no current estimate for this dog (its coat matched no known "
            "layout), so report every region in scope you can see."
        )
    lines = "\n".join(
        f"- {region}: {current_estimate[region]}" for region in regions if region in current_estimate
    )
    missing = [region for region in regions if region not in current_estimate]
    text = f"Regions in scope: {scope}.\n\nCurrent estimate:\n{lines}"
    if missing:
        text += f"\n\nNo estimate for: {', '.join(missing)} - report those if visible."
    if set(regions) <= set(FACE_REGIONS):
        text += "\n\n" + FACE_HINT
    return text


# The face's most common mistake: the fur beside the eyes is one colour and
# the lower cheek another (a dark mask fading into cream), and the region
# agent answers with whichever it looked at first. Asked for explicitly, it
# comes back as a gradient the compositor can draw.
FACE_HINT = (
    "On the face, look at the cheeks in two places: right beside the eye, and "
    "down by the jaw. If those two are different colours - a dark mask around "
    "the eyes running into pale fur below, or pale brows over a dark cheek - "
    "report cheeks with the colour by the jaw, gradient_to the colour beside "
    "the eye, and gradient_anchor eye. Report eyebrows as the colour of the "
    "fur directly above each eye, whatever the skull behind it is. Report "
    "muzzle_bridge as the colour of the top of the muzzle only."
)


def _valid_items(
    payload: dict[str, Any], list_key: str, scope: tuple[str, ...]
) -> tuple[dict[str, Any], list[dict]]:
    """Drop entries the texture stage cannot use, and say which.

    Three kinds: a region outside the enum, a region outside this call's
    scope, and a second entry for a region already claimed. The schema
    constrains none of them, so all are anomalies and all are worth one
    re-ask. Only region_colors is deduplicated: several markings on one
    region is normal.
    """
    items = payload.get(list_key) or []
    dedupe = list_key == "region_colors"

    kept, dropped, seen = [], [], set()
    for item in items:
        region = item.get("region")
        if region not in REGIONS or region not in scope:
            dropped.append(item)
        elif dedupe and region in seen:
            dropped.append(item)
        else:
            seen.add(region)
            kept.append(item)

    if dropped:
        payload = {**payload, list_key: kept}
    return payload, dropped


def _signature(payload: dict[str, Any], list_key: str) -> frozenset[tuple[str, str]]:
    """Two readings agree when they name the same (region, color) pairs.
    Differences in coverage, size or placement are not disagreement."""
    return frozenset(
        (item["region"], str(item.get("color", "")).strip().lower())
        for item in payload.get(list_key) or []
    )


async def run_agent(
    spec: AgentSpec,
    transport: Transport,
    image_blocks: list[dict[str, Any]],
    base_color: str,
    regions_in_scope: tuple[str, ...] | list[str] | None = None,
    current_estimate: dict[str, str] | None = None,
    samples: int = SAMPLES,
) -> AgentResult:
    """Sample the agent `samples` times over `regions_in_scope` and keep what
    the readings agree on. `current_estimate` maps region -> colour and may
    cover only some of the scope; None or {} means no estimate."""
    scope = tuple(regions_in_scope) if regions_in_scope else tuple(REGIONS)
    scoped_estimate = {r: c for r, c in (current_estimate or {}).items() if r in scope}
    system = spec.system_for(scope)
    user = spec.user_text.format(
        base_color=base_color, estimate=estimate_text(scope, scoped_estimate)
    )
    blocks = [*image_blocks, {"type": "text", "text": user}]

    async def one_call() -> tuple[dict[str, Any], list[dict]]:
        payload = await transport.complete(system=system, blocks=blocks, schema=spec.schema)
        return _valid_items(payload, spec.list_key, scope)

    async def sample() -> tuple[dict[str, Any], list[dict], int]:
        payload, dropped = await one_call()
        calls = 1
        if dropped:
            # one re-ask when the model went outside the enum, the scope, or repeated a region
            payload, dropped_again = await one_call()
            calls += 1
            dropped = dropped + dropped_again
        return payload, dropped, calls

    # a sample that fails (a rate limit, a network error) is retried, so
    # the vote always has its full count: two readings that split 1-1
    # discard every region, and one failed call must not cause that
    readings: list[tuple[dict[str, Any], list[dict], int]] = []
    errors: list[str] = []
    pending = samples
    for _attempt in range(SAMPLE_RETRIES + 1):
        if pending <= 0:
            break
        results = await asyncio.gather(*(sample() for _ in range(pending)), return_exceptions=True)
        pending = 0
        for result in results:
            if isinstance(result, AgentRefusal):
                raise result
            if isinstance(result, Exception):
                errors.append(f"{type(result).__name__}: {result}")
                pending += 1
            else:
                readings.append(result)
    if not readings:
        raise RuntimeError(f"every sample of {spec.name} failed: {errors[-1] if errors else 'unknown'}")
    payloads = [reading[0] for reading in readings]
    dropped_all = [item for reading in readings for item in reading[1]]
    calls = sum(reading[2] for reading in readings) + len(errors)
    resolved = _resolve(spec, payloads, calls, dropped_all, scope, scoped_estimate)
    resolved.errors = errors
    return resolved


def _resolve(
    spec: AgentSpec,
    attempts: list[dict[str, Any]],
    calls: int,
    dropped: list[dict[str, Any]],
    scope: tuple[str, ...] = tuple(REGIONS),
    estimate: dict[str, str] | None = None,
) -> AgentResult:
    """Vote the readings region by region (see CONFIDENCE in the module
    docstring). Markings are voted as (region, colour-family) pairs and need
    a majority of the readings to name them."""
    from collections import Counter

    from .patterns import color_family

    n = max(len(attempts), 1)
    majority = n // 2 + 1
    estimate = estimate or {}
    kept: list[dict[str, Any]] = []
    disagreed: list[str] = []
    agreement: dict[str, float] = {}
    shares: list[float] = []

    if spec.list_key == "region_colors":
        per_reading = []
        for attempt in attempts:
            named = {}
            for item in attempt.get("region_colors") or []:
                colour = str(item.get("color", "")).strip().lower()
                if item.get("region") in scope and colour in FUR_COLORS and item["region"] not in named:
                    named[item["region"]] = (colour, item)
            per_reading.append(named)

        regions_named = {region for named in per_reading for region in named}
        for region in scope:
            if region not in regions_named:
                continue
            votes: list[str] = []
            for named in per_reading:
                if region in named:
                    votes.append(named[region][0])
                elif region in estimate:
                    votes.append(estimate[region])   # silence confirms the estimate
            counts = Counter(votes)
            for colour, count in counts.items():
                agreement[f"{region}:{colour}"] = round(count / n, 3)
            winner, top = counts.most_common(1)[0]
            share = top / n
            if top < majority:
                families = Counter(color_family(v) for v in votes)
                family, family_top = families.most_common(1)[0]
                if family_top >= majority and family != "unknown":
                    in_family = Counter(v for v in votes if color_family(v) == family)
                    winner, share = in_family.most_common(1)[0][0], family_top / n
                else:
                    disagreed.append(region)
                    continue
            # the item from the first reading that named this colour (or family)
            item = next((named[region][1] for named in per_reading
                         if region in named and named[region][0] == winner), None)
            if item is None:
                item = next(named[region][1] for named in per_reading if region in named)
                item = {**item, "color": winner}
            # a reading that named the estimate's own colour confirmed it: that
            # is an observation of the region, not silence, and the merge
            # records it as one (the colour does not change)
            confirmed = winner == estimate.get(region)
            kept.append({**item, "confidence": round(share, 3), "confirmed": confirmed})
            shares.append(share)
    else:
        keys_per_reading = []
        for attempt in attempts:
            keys = {}
            for item in attempt.get(spec.list_key) or []:
                colour = str(item.get("color", "")).strip().lower()
                if item.get("region") in scope:
                    keys.setdefault((item["region"], color_family(colour)), item)
            keys_per_reading.append(keys)
        counts = Counter(key for keys in keys_per_reading for key in keys)
        for key, count in counts.items():
            agreement[f"{key[0]}:{key[1]}"] = round(count / n, 3)
            if count >= majority:
                item = next(keys[key] for keys in keys_per_reading if key in keys)
                kept.append({**item, "confidence": round(count / n, 3)})
                shares.append(count / n)
            else:
                disagreed.append(key[0])
        disagreed = sorted(set(disagreed))

    lead = attempts[0] if attempts else {}
    payload = {**lead, spec.list_key: kept}
    return AgentResult(
        name=spec.name,
        payload=payload,
        confidence=round(sum(shares) / len(shares), 3) if shares else (1.0 if not disagreed else 0.0),
        calls=calls,
        agreed=not disagreed,
        dropped=dropped,
        disagreed=sorted(set(disagreed)),
        scope=scope,
        agreement=agreement,
    )
