"""Simple mode - a few features, laid on a natural coat.

WHY
---
Reading a photograph in ten independent regions, each with its own
colour, gradient and shape, produces ten independent answers, and every
disagreement between them - between two samples, between the face call and
the body call, between the reader and the critic - lands on the model as a
patch of a slightly different colour. The dog came out mottled, and no
single answer was wrong.

This mode asks for what a person would say about the dog: its coat pattern,
its two colours, and the handful of things you would mention to pick it out
of a line-up - "white chest, white paws, tan over the eyes". Those go onto
the pattern's default layout, which was drawn to look like a dog, and the
layout does the rest. Nothing is read per region; nothing is painted per
texel from a picture; every boundary is feathered so no mask shows.

WHAT IS READ
------------
  pattern + base + second     pattern_classify.py, three samples voted
  features (up to 4)          one call, three samples, majority per feature;
                              each feature is a named recolouring of a few
                              regions (FEATURES below)

The critic, when it runs, speaks the same vocabulary: which features the
photograph shows that the render lacks, and which the render shows that
the photograph does not. One round, two samples, only agreed changes.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .agents import AgentRefusal, Transport
from .fur import FUR_COLORS
from .patterns import color_family, instantiate_defaults
from .regions import REGIONS

SAMPLES = 3
MAX_FEATURES = 4

# name -> (description for the model, {region: colour-role})
# colour roles: "white" / "cream" / "tan" / "black" literal palette words,
# "second" the pattern's second colour, "dark" the darker of base/second.
FEATURES: dict[str, tuple[str, dict[str, str]]] = {
    # the chest region runs from the chin down the throat to the breastbone
    # and out to the shoulder fronts; the belly rule carries a light chest
    # on between the forelegs
    "white_chest": ("a white chest and throat", {"chest": "white"}),
    "white_belly": ("a white belly and underside", {"underside": "white"}),
    # the two sock heights are told apart at the wrist: white that stops at
    # the toes or the pastern is white_paws, white that reaches the wrist or
    # climbs above it is white_socks. A dog wears one or the other, not both.
    "white_paws": ("white on the paws only: the toes and the pastern, stopping below the wrist and hock",
                   {"paws": "white"}),
    "white_socks": ("white socks that reach the wrist or hock, or climb above them toward the elbow or knee "
                    "(white paws are part of this, do not list white_paws as well)",
                    {"paws": "white", "legs_lower": "white"}),
    "white_muzzle": ("a white muzzle or blaze up the face", {"muzzle_bridge": "white", "cheeks": "white"}),
    "white_bridge": ("a white blaze along the top of the muzzle", {"muzzle_bridge": "white"}),
    "cream_bridge": ("a cream blaze along the top of the muzzle", {"muzzle_bridge": "cream"}),
    "cream_cheeks": ("cream cheeks and jaw", {"cheeks": "cream"}),
    "white_cheeks": ("white cheeks and jaw", {"cheeks": "white"}),
    # the inside of the ears, from the face call (face_colours.py); "point"
    # so a red-pointed dog gets red ears, not tan
    "pink_ears": ("pink bare skin inside the ears", {"ears_inner": "pink"}),
    # "the same as the head": the base colour, not black. The face call
    # answers "dark" for the shadowed cup of a white or yellow dog's pricked
    # ear (eval e01-e04, e13: 3-5 of 5), and black cups on a white dog were
    # the most visible error in the eval set
    "dark_ears": ("dark fur inside the ears, the same as the head", {"ears_inner": "base"}),
    "tan_ears": ("tan or red fur inside the ears", {"ears_inner": "point"}),
    "cream_ears": ("cream fur inside the ears", {"ears_inner": "cream"}),
    "white_ears": ("white fur inside the ears", {"ears_inner": "white"}),
    "cream_underside": ("cream on the throat, chest and belly (urajiro)",
                        {"chest": "cream", "underside": "cream"}),
    "tan_brows": ("tan spots above the eyes", {"eyebrows": "point"}),
    "cream_brows": ("cream spots above the eyes", {"eyebrows": "cream"}),
    "white_brows": ("white spots above the eyes", {"eyebrows": "white"}),
    "no_brows": ("no spots above the eyes: the forehead colour runs over them", {"eyebrows": "base"}),
    "tan_cheeks": ("tan or red on the cheeks", {"cheeks": "tan"}),
    "tan_legs": ("tan or red on the lower legs (the paws keep their own colour)", {"legs_lower": "tan"}),
    "dark_muzzle": ("a dark muzzle or face mask", {"muzzle_bridge": "dark"}),
    "tan_bridge": ("tan or red fur along the top of the muzzle", {"muzzle_bridge": "warm"}),
    "dark_back": ("a dark saddle or blanket over the back and tail", {"back": "dark"}),
    # placed as a marking by lite.py (random large patches on the trunk)
    "body_patches": ("large patches of the second colour on a white body (piebald spots)", {}),
}

FEATURES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "features": {
            "type": "array",
            "maxItems": MAX_FEATURES,
            "description": f"Up to {MAX_FEATURES} features clearly visible on this dog; fewer is normal.",
            "items": {"type": "string", "enum": list(FEATURES)},
        },
        "note": {"type": "string", "description": "One short phrase, or empty."},
    },
    "required": ["features", "note"],
    "additionalProperties": False,
}

_FEATURE_LIST = "\n".join(f"- {name}: {text}" for name, (text, _) in FEATURES.items())

FEATURES_SYSTEM = f"""You name the few things that stand out about a dog's coat in a photograph.

Choose from this fixed list only, and only what is CLEARLY visible - the
features a person would mention to pick this dog out of a line-up. Up to
{MAX_FEATURES}; one or two is a normal answer, and an empty list is fine for a
plain coat. Do not list a feature because dogs of this kind usually have it.

{_FEATURE_LIST}

The base coat colour is given in the user message as established fact."""


@dataclass
class SimpleAppearance:
    pattern: str
    base_color: str
    second_color: str | None
    features: list[str]
    region_colors: list[dict[str, Any]]
    votes: dict[str, float] = field(default_factory=dict)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"pattern": self.pattern, "base_color": self.base_color, "second_color": self.second_color,
                "features": self.features, "feature_votes": self.votes, "note": self.note}


def _role_colour(role: str, base: str, second: str | None) -> str:
    if role == "second":
        return second or base
    if role == "base":
        return base
    if role == "point":
        # the warm colour of the coat's points: the second colour when it is
        # one, else tan
        return second if second and color_family(second) == "point" else "tan"
    if role == "warm":
        # the coat's warm colour: the base when it is one, else the points
        for colour in (base, second):
            if colour and color_family(colour) == "point":
                return colour
        return "tan"
    if role == "dark":
        candidates = [c for c in (base, second) if c and color_family(c) == "dark"]
        return candidates[0] if candidates else "black"
    return role


def apply_features(pattern: str, base: str, second: str | None, features: list[str]) -> list[dict[str, Any]]:
    """The pattern's default layout with the features laid over it, all
    ten regions, full coverage, no gradients."""
    layout = {e["region"]: e for e in instantiate_defaults(pattern, base, second)}
    if not layout:
        layout = {r: {"region": r, "color": base, "coverage": "full", "note": "solid fallback",
                      "source": "pattern_default:solid", "weight": 1.0} for r in REGIONS}
    for name in features:
        text, overrides = FEATURES[name]
        for region, role in overrides.items():
            colour = _role_colour(role, base, second)
            if colour not in FUR_COLORS:
                continue
            layout[region] = {**layout[region], "color": colour, "note": text, "source": f"feature:{name}"}

    # a white or cream chest continues down the belly on nearly every dog
    # that has one; the belly is the one region a sitting photo never shows,
    # so where nothing else spoke for it, it follows the chest
    chest = layout["chest"]
    if (color_family(chest["color"]) == "light"
            and str(layout["underside"].get("source", "")).startswith("pattern_default")):
        layout["underside"] = {**layout["underside"], "color": chest["color"],
                               "note": "follows the throat and chest", "source": "inferred:chest"}
    return [layout[r] for r in REGIONS]


# Where a coat's colours blend rather than meet, as defaults that need no
# reading. Each rule: (region, condition on the region's colour and the coat)
# -> the entry's colour is strongest at the anchor and fades into gradient_to
# with distance (texture._gradient_weight). Only what is true of nearly every
# dog that has the marking at all:
#   socks      white legs fade upward into the leg's colour; the sock's top
#              edge is never a line
#   cheeks     a cream cheek is warmer (the points colour) next to the eye
#   ear cup    a tan or red cup is the points colour at the base and cream
#              toward the tip (the hair inside an ear is sparse and pale)
# NOT the belly: a fade from the belly centre also darkens the underside
# between the forelegs, cutting the white bib off from the white belly
# (tried on the black-and-tan: a black gap appeared behind the elbow). The
# belly's flank edge is left to the refinement.
# where a white-fronted dog's light leg meets its body: the top share of the
# leg (percentiles of the distance from the paw) fades this far toward the
# body colour
LEG_TOP_REACH = (82.0, 100.0)
LEG_TOP_CAP = 0.75


def default_gradients(appearance: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """The appearance with default gradients on the regions that blend, and
    the list of what was added. Entries that already carry a gradient (an
    older reading) are left alone."""
    base = str(appearance.get("base_color", "")).lower()
    second = appearance.get("second_color")
    second = str(second).lower() if second else None
    points = second if second and color_family(second) == "point" else None
    entries = {e["region"]: dict(e) for e in appearance.get("region_colors") or []}
    added: list[str] = []

    def fade(region: str, colour: str, to: str, anchor: str) -> None:
        entry = entries.get(region)
        if entry is None or str(entry.get("gradient_to") or "none") != "none":
            return
        if to not in FUR_COLORS or colour not in FUR_COLORS or to == colour:
            return
        entries[region] = {**entry, "color": colour, "gradient_to": to, "gradient_anchor": anchor,
                           "note": f"{entry.get('note', '')} / default gradient {colour}->{to} from {anchor}".strip(" /")}
        added.append(f"{region}:{colour}->{to}@{anchor}")

    legs = entries.get("legs_lower", {})
    # ... except on a white-fronted dog: its white chest runs on into white
    # legs, and the fade drew a tan wedge at each elbow (eval demo e08)
    front_light = color_family(str(entries.get("chest", {}).get("color", ""))) == "light"
    if color_family(legs.get("color", "")) == "light" and front_light:
        # the white-fronted dog's legs: the chest's own light colour (pure
        # white socks under a cream chest read as bright stockings, eval
        # demo e08), meeting the body over the top of the leg only, so the
        # elbow is not a straight cut
        chest_colour = str(entries["chest"]["color"])
        for region in ("legs_lower", "paws"):
            if region in entries and color_family(str(entries[region].get("color", ""))) == "light":
                entries[region] = {**entries[region], "color": chest_colour}
        legs = entries["legs_lower"]
        if base in FUR_COLORS and color_family(base) != "light":
            entries["legs_lower"] = {**legs, "gradient_to": base, "gradient_anchor": "paw",
                                     "gradient_reach": list(LEG_TOP_REACH), "gradient_cap": LEG_TOP_CAP,
                                     "note": f"{legs.get('note', '')} / meets the {base} body at the top".strip(" /")}
            added.append(f"legs_lower:{chest_colour}->{base}@paw top")
    if color_family(legs.get("color", "")) == "light" and not front_light:
        # the leg's own colour above the sock: the points on a pointed coat, else the base
        fade("legs_lower", legs["color"], points or base, "paw")
    # the inside of the legs: on a dog whose chest and paws are light the
    # inner and front faces of the legs are that light colour too (the
    # shiba's urajiro; the black-and-tan's tan sits on the outer face of
    # the leg only). The reading cannot see the inside of a leg, and the
    # user read a tan inner leg on a dog whose photo shows white there.
    chest = entries.get("chest", {})
    paws = entries.get("paws", {})
    if (color_family(legs.get("color", "")) != "light" and color_family(chest.get("color", "")) == "light"
            and color_family(paws.get("color", "")) == "light" and str(legs.get("inner_color") or "") == ""):
        entries["legs_lower"] = {**entries["legs_lower"], "inner_color": chest["color"],
                                 "note": f"{legs.get('note', '')} / inside {chest['color']} like the chest".strip(" /")}
        added.append(f"legs_lower inner:{chest['color']}")
    cheeks = entries.get("cheeks", {})
    # not when the face call saw the cheeks: its "cream" is the colour up to
    # the eyes, and the tan fade put tan round the eyes of a black-and-tan
    # whose photo shows cream there (eval demo e08)
    seen_cheeks = str(cheeks.get("source", "")).startswith("feature:")
    if color_family(cheeks.get("color", "")) == "light" and points and not seen_cheeks:
        fade("cheeks", points, cheeks["color"], "eye")
    # the ear cup: the hair inside an ear is sparse and pale, warm at the
    # base and near-white toward the tip. Measured on the black-and-tan
    # shiba's photo the cup is LAB (57..78, 3, 12..19) - between cream and
    # fawn - while the vision read calls it "tan" (three of three) and the
    # tan ramp (56, 15, 40) painted it a saturated orange the user read as
    # wrong. So a points-coloured cup is painted as a fade: the points
    # colour at the ear base, cream toward the tip.
    # a dark muzzle on a light or warm dog is a mask: darkest at the nose,
    # gone by the eyes. The jindo's muzzle_bridge reaches up between the
    # eyes, and a solid black one drew a stripe onto the forehead (eval v3
    # e12, e20)
    # The same for a light muzzle top on a darker head: a cream bridge ran up
    # to the forehead as a white stripe (eval demo e09). A blaze that really
    # goes up between the eyes is its own marking (texture._blaze_strip).
    muzzle = entries.get("muzzle_bridge", {})
    skull = str(entries.get("skull", {}).get("color", base))
    if (muzzle.get("color") and muzzle["color"] != skull and skull in FUR_COLORS
            and color_family(muzzle["color"]) != color_family(skull)):
        fade("muzzle_bridge", muzzle["color"], skull, "nose")
    ears = entries.get("ears_inner", {})
    if color_family(ears.get("color", "")) == "point":
        fade("ears_inner", ears["color"], "cream", "ear_base")
    return {**appearance, "region_colors": [entries[r] for r in REGIONS if r in entries]}, added


async def read_features(transport: Transport, photo_blocks: list[dict[str, Any]], base_color: str,
                        samples: int = SAMPLES) -> tuple[list[str], dict[str, float], str]:
    """The features a majority of the readings named."""
    blocks = [*photo_blocks, {"type": "text", "text": (
        f"The shelter record gives the base coat colour as '{base_color}'. "
        "Which of the listed features does this dog clearly show?")}]

    async def one() -> dict[str, Any]:
        return await transport.complete(system=FEATURES_SYSTEM, blocks=blocks, schema=FEATURES_SCHEMA)

    payloads = await asyncio.gather(*(one() for _ in range(samples)), return_exceptions=True)
    readings: list[list[str]] = []
    notes: list[str] = []
    for payload in payloads:
        if isinstance(payload, AgentRefusal):
            raise payload
        if isinstance(payload, Exception):
            continue
        readings.append([f for f in payload.get("features") or [] if f in FEATURES][:MAX_FEATURES])
        if payload.get("note"):
            notes.append(str(payload["note"]))
    n = max(len(readings), 1)
    counts = Counter(f for reading in readings for f in set(reading))
    votes = {name: round(count / n, 3) for name, count in counts.most_common()}
    agreed = [name for name, count in counts.most_common() if count > n / 2][:MAX_FEATURES]
    return agreed, votes, " / ".join(notes)


# --------------------------------------------------------------------------
# the critic, in the same vocabulary
# --------------------------------------------------------------------------

SIMPLE_CRITIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "add": {
            "type": "array",
            "description": "Features the photograph clearly shows that the render lacks.",
            "items": {"type": "string", "enum": list(FEATURES)},
        },
        "remove": {
            "type": "array",
            "description": "Features the render shows that the photograph clearly does not have.",
            "items": {"type": "string", "enum": list(FEATURES)},
        },
        "note": {"type": "string"},
    },
    "required": ["add", "remove", "note"],
    "additionalProperties": False,
}

SIMPLE_CRITIC_SYSTEM = f"""You compare a 3D render of a pet model with the photograph it was built from.

You are shown the photograph, a full-body render and a face close-up of the
model, and the list of coat features the model was built with. The render is
a simplified model under studio light; IGNORE style, fur texture, shading,
lighting, pose, background, eyes and nose.

Answer only in terms of this feature list:

{_FEATURE_LIST}

List in "add" the features the photograph clearly shows and the render
lacks, and in "remove" the features the render shows that the photograph
clearly does not have. Empty lists mean the render matches, and that is a
normal answer. Do not add a feature because dogs like this usually have it."""


async def simple_critique(transport: Transport, photo_blocks: list[dict[str, Any]], body_render, face_render,
                          features: list[str], samples: int = 2) -> tuple[list[str], list[str], list[dict], str]:
    """Agreed additions and removals (a majority of the samples), plus every
    reading and the notes."""
    from .critic import _image_block, side_render_for
    from pathlib import Path
    from PIL import Image

    side = side_render_for(Path(body_render))
    blocks = [
        {"type": "text", "text": "The photograph:"}, *photo_blocks,
        {"type": "text", "text": "Full-body render of the model:"}, _image_block(Image.open(body_render)),
        *([{"type": "text", "text": "The model standing, seen from the side and low:"},
           _image_block(Image.open(side))] if side else []),
        {"type": "text", "text": "Face close-up render of the model:"}, _image_block(Image.open(face_render)),
        {"type": "text", "text": "The model was built with these features: "
                                 + (", ".join(features) or "(none)") + "."},
    ]

    async def one() -> dict[str, Any]:
        return await transport.complete(system=SIMPLE_CRITIC_SYSTEM, blocks=blocks, schema=SIMPLE_CRITIC_SCHEMA)

    payloads = await asyncio.gather(*(one() for _ in range(samples)), return_exceptions=True)
    readings = [p for p in payloads if isinstance(p, dict)]
    n = max(len(readings), 1)
    adds = Counter(f for r in readings for f in set(r.get("add") or []) if f in FEATURES and f not in features)
    removes = Counter(f for r in readings for f in set(r.get("remove") or []) if f in features)
    add = [f for f, c in adds.most_common() if c > n / 2]
    remove = [f for f, c in removes.most_common() if c > n / 2]
    notes = " / ".join(str(r.get("note", "")) for r in readings if r.get("note"))
    return add, remove, [{"add": r.get("add"), "remove": r.get("remove")} for r in readings], notes


__all__ = ["FEATURES", "FEATURES_SCHEMA", "SIMPLE_CRITIC_SCHEMA", "SimpleAppearance", "apply_features",
           "read_features", "simple_critique"]
