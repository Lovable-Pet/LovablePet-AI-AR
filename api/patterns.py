"""Complete the regions a photograph could not answer, using coat patterns.

A shelter photo is usually one frame of a sitting dog, so whole areas of the
animal - the belly, the inside of the legs - are simply not in it, and stage 2
correctly says nothing about them. Leaving them at the base color renders a
black-and-tan shiba with a black belly, which is not what the animal looks
like.

Dog coats are not arbitrary. A handful of patterns account for almost all of
them, and each covers a fixed set of body parts: urajiro puts cream on the
ventral surface, tan points put tan above the eyes and on the lower legs, Irish
spotting puts white on the collar, chest, feet and tail tip. So the regions that
ARE visible identify the pattern, and the pattern says what the hidden ones look
like.

WHAT THIS IS ALLOWED TO DO
--------------------------
This is inference about a real animal that someone may travel to meet, so it is
deliberately narrow:

  - it only fills regions stage 2 said nothing about; an observed region is
    never overwritten, however badly it fits the pattern;
  - it only fires when the observed regions actually support the pattern, by a
    margin - a coat that contradicts urajiro as often as it supports it gets
    nothing filled;
  - it fills with the color the supporting regions actually showed, not with a
    color from the table, so the completion follows this animal;
  - every filled entry is marked with its source and carries a reduced weight,
    so it renders softer than something the photograph showed.

The honest framing is "this dog's visible markings are urajiro, and urajiro
includes the belly", not "dogs like this usually have a cream belly".
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .regions import REGIONS

# Colour families, because a pattern constrains which KIND of colour covers a
# region, not its exact name: urajiro is cream or white or ivory depending on
# the dog, and calling all of those "light" is what lets one table match them.
COLOR_FAMILY: dict[str, str] = {
    "white": "light", "cream": "light", "pink": "light",
    "fawn": "point", "red": "point", "tan": "point",
    "black": "dark", "chocolate": "dark", "gray": "dark",
}


def color_family(name: str) -> str:
    key = (name or "").strip().lower()
    if key in COLOR_FAMILY:
        return COLOR_FAMILY[key]
    for qualifier in ("dark ", "light ", "pale "):
        if key.startswith(qualifier):
            return COLOR_FAMILY.get(key[len(qualifier):], "unknown")
    return "unknown"


@dataclass(frozen=True)
class CoatPattern:
    name: str
    korean: str
    family: str
    regions: tuple[str, ...]
    example: str


# Each pattern lists the regions it covers with its family's colour. The base
# coat covers everything else, which is why no pattern mentions back or skull.
COAT_PATTERNS: tuple[CoatPattern, ...] = (
    CoatPattern(
        name="urajiro",
        korean="우라지로 - 배쪽 크림",
        family="light",
        regions=("cheeks", "chest", "underside", "legs_lower", "paws", "ears_inner"),
        example="시바, 아키타",
    ),
    CoatPattern(
        name="tan_point",
        korean="탄포인트 - 눈썹·다리 갈색",
        family="point",
        regions=("eyebrows", "cheeks", "chest", "legs_lower", "paws"),
        example="도베르만, 로트와일러",
    ),
    CoatPattern(
        name="irish",
        korean="아이리시 스포팅 - 가슴·배·발 흰색",
        family="light",
        regions=("chest", "underside", "legs_lower", "paws"),
        example="보더콜리, 콜리",
    ),
)

# A pattern needs this many regions agreeing before it may fill anything. Two
# is a coincidence - almost every light-chested dog has a white chest and white
# paws - and three is a pattern.
MIN_SUPPORT = 3

# And it must agree at least this often among the regions it covers, so a coat
# that half fits gets nothing.
MIN_AGREEMENT = 0.7

# Filled regions render at this share of an observed one. They are a sound
# inference, not an observation, and should not look equally certain.
INFERRED_WEIGHT = 0.6

# Nothing observed says how much of a hidden region the colour covers, so the
# conservative reading is "some of it".
INFERRED_COVERAGE = "partial"


@dataclass(frozen=True)
class PatternMatch:
    pattern: CoatPattern
    support: int
    contradiction: int
    color: str

    @property
    def agreement(self) -> float:
        total = self.support + self.contradiction
        return self.support / total if total else 0.0


def match_pattern(region_colors: list[dict[str, Any]]) -> PatternMatch | None:
    """Find the pattern the observed regions support, if any.

    A region inside the pattern whose colour is in the pattern's family counts
    for it; one in the pattern whose colour is in a different family counts
    against. Regions outside the pattern are ignored rather than held against
    it, because a dog wears more than one pattern at once - a black-and-tan
    shiba has urajiro AND tan points, and each has to be able to match on its
    own regions without the other's counting as a failure.
    """
    best: PatternMatch | None = None

    for pattern in COAT_PATTERNS:
        supporting = [
            entry for entry in region_colors
            if entry.get("region") in pattern.regions
            and color_family(entry.get("color", "")) == pattern.family
        ]
        contradiction = sum(
            1 for entry in region_colors
            if entry.get("region") in pattern.regions
            and color_family(entry.get("color", "")) not in (pattern.family, "unknown")
        )
        if len(supporting) < MIN_SUPPORT:
            continue

        # fill with the colour this animal actually showed, not the table's
        colour = Counter(
            str(entry.get("color", "")).strip().lower() for entry in supporting
        ).most_common(1)[0][0]

        candidate = PatternMatch(pattern, len(supporting), contradiction, colour)
        if candidate.agreement < MIN_AGREEMENT:
            continue
        if best is None or (candidate.support, candidate.agreement) > (best.support, best.agreement):
            best = candidate

    return best


def complete_appearance(appearance: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return the appearance with unobserved pattern regions filled in, and the
    list of what was added."""
    region_colors = list(appearance.get("region_colors") or [])
    match = match_pattern(region_colors)
    if match is None:
        return appearance, []

    observed = {entry.get("region") for entry in region_colors}
    added = [
        {
            "region": region,
            "color": match.color,
            "coverage": INFERRED_COVERAGE,
            "note": f"not visible in the photos; {match.pattern.name} covers this region",
            "source": f"pattern:{match.pattern.name}",
            "weight": INFERRED_WEIGHT,
        }
        for region in match.pattern.regions
        if region not in observed and region in REGIONS
    ]
    if not added:
        return appearance, []

    return {**appearance, "region_colors": region_colors + added}, added


def describe(match: PatternMatch | None) -> str:
    if match is None:
        return "no coat pattern matched; nothing filled in"
    return (
        f"{match.pattern.name} ({match.pattern.korean}, {match.pattern.example}): "
        f"{match.support} regions agree, {match.contradiction} disagree "
        f"({match.agreement:.0%}), filling with '{match.color}'"
    )


# --------------------------------------------------------------------------
# defaults: a whole coat from (pattern, base, second)
# --------------------------------------------------------------------------

# The classifier's vocabulary (see pattern_classify.py). The first five have a
# layout the anchors are rendered from; brindle and merle are a solid coat
# with a texture overlay; other has no layout at all, so every region has to
# come from the photograph. sable is the same kind: a warm coat with dark
# tipping shaded over the top of the dog (texture.apply_overlay), never a
# hard-edged saddle - the sable jindo mix read as saddle and came out with a
# black blanket and a black nose bar (fixture_sable).
# piebald: a white dog with a coloured cap over the skull and ears (eval
# e13, e15, e16 all read "other" and came out plain white); patches on the
# body are the body_patches feature.
PATTERN_NAMES = ("tan_point", "urajiro", "irish", "saddle", "solid", "brindle", "merle", "sable", "piebald",
                 "other")

# Regions the second colour covers, per pattern. Everything else is the base.
# saddle is written from the point colour's side - the base is the tan and
# the second colour is the dark saddle - because that is how a person
# describes the dog and how the classifier is asked to answer.
# With the ten regions an Irish coat's white collar and tail tip are not
# representable (they are part of back); the chest-to-feet run is.
PATTERN_LAYOUT: dict[str, tuple[str, ...]] = {
    "tan_point": ("eyebrows", "cheeks", "chest", "underside", "legs_lower", "paws", "ears_inner"),
    "urajiro": ("cheeks", "chest", "underside", "legs_lower", "paws", "ears_inner"),
    "irish": ("chest", "underside", "legs_lower", "paws"),
    "saddle": ("back", "muzzle_bridge"),
    "solid": (),
    "brindle": (),
    "merle": (),
    "sable": (),
    "piebald": ("skull", "eyebrows", "ears_inner"),
}

# brindle and merle keep a solid layout and carry the texture as a flag the
# compositor reads; other has no defaults at all.
PATTERN_OVERLAY: dict[str, str | None] = {name: None for name in PATTERN_NAMES}
PATTERN_OVERLAY["brindle"] = "brindle"
PATTERN_OVERLAY["merle"] = "merle"
PATTERN_OVERLAY["sable"] = "sable"

DEFAULT_SOURCE = "pattern_default"


def instantiate_defaults(pattern: str, base: str, second: str | None) -> list[dict[str, Any]]:
    """All ten regions for a pattern: the starting estimate the vision
    agents are asked to correct, region by region.

    Returns [] for "other" (and for an unknown name), which tells the agents
    there is no estimate and every region must be read from the photograph.
    A pattern with no second colour given falls back to a solid coat.
    """
    key = (pattern or "").strip().lower()
    if key not in PATTERN_LAYOUT:
        return []
    second_regions = PATTERN_LAYOUT[key] if second and second != "none" else ()
    return [
        {
            "region": region,
            "color": second if region in second_regions else base,
            "coverage": "full",
            "note": f"{key} default",
            "source": f"{DEFAULT_SOURCE}:{key}",
            "weight": 1.0,
        }
        for region in REGIONS
    ]


def overlay_for(pattern: str) -> str | None:
    return PATTERN_OVERLAY.get((pattern or "").strip().lower())

