"""The chest's and the belly's colours, asked from the photograph alone.

WHY
---
The pattern's layout says where the second colour goes, and for a
tan-point that is the whole chest, throat and belly - a rottweiler's
layout. A doberman-type dog is black there with two small tan spots on the
breastbone, and came out with a tan front half (fixture_dobie). The layout
call cannot be asked "how much": a colour question asked beside a layout
question comes back as whatever the layout implies (face_colours.py). So,
like the face, the chest and belly are asked in their own call with
absolute colour words and no pattern names.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

from .agents import AgentRefusal, Transport

SAMPLES = 3

CHEST_ANSWERS = ("dark", "dark_tan_spots", "tan", "cream", "white", "unknown")
BELLY_ANSWERS = ("dark", "tan", "cream", "white", "unknown")
# The feet. The layout call's white_legs said "paws" on a tan-footed dobie
# (fixture_dobie4): asked beside the layout, a colour comes back as what the
# layout implies. A seen, non-white answer here drops the white leg features.
PAW_ANSWERS = ("dark", "tan", "cream", "white", "unknown")
# how much lighter a cream chest is than the back: the cream ramp is taken
# this share of the way from the base to the table's cream (palette.
# relative_cream). The pale front of a tan dog ranges from barely lighter to
# near-white; one fixed share painted every one of them alike.
LIGHTNESS_ANSWERS = ("slightly", "clearly", "near_white", "not_cream")
# Whether the top of the back is tipped grey-black: the sable overlay is
# drawn only when this agrees with the layout call. The layout call picked
# sable 3/3 for a red dog with a black face mask whose back is a deep red
# (eval v3 e11) through every wording of its own sable rule; a question on
# the back alone, compared with the flank, is what the calls read well.
TIPPING_ANSWERS = ("none", "slight", "clear", "unknown")
CREAM_SHARE = {"slightly": 0.35, "clearly": 0.6, "near_white": 0.9}
WHITE_LEG_FEATURES = ("white_paws", "white_socks")

BODY_COLOURS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "chest": {
            "type": "string",
            "enum": list(CHEST_ANSWERS),
            "description": "The colour of MOST of the front of the chest, from the throat under the chin down "
                           "to between the front legs: dark (black, chocolate or grey), dark_tan_spots (black "
                           "or chocolate, broken only by two small tan spots on the breastbone), tan (any warm "
                           "brown, red or fawn - also when a small white spot sits on it), cream, white, "
                           "unknown (not visible).",
        },
        "belly": {
            "type": "string",
            "enum": list(BELLY_ANSWERS),
            "description": "The colour of the belly and the underside between the legs: dark, tan, cream, "
                           "white, unknown (not visible in this photograph - a sitting or standing dog seen "
                           "from the front usually hides it).",
        },
        "paws": {
            "type": "string",
            "enum": list(PAW_ANSWERS),
            "description": "The colour of the feet - the toes and the fur just above them: dark, tan, cream, "
                           "white, unknown (the feet are hidden or out of the picture).",
        },
        "back_tipping": {
            "type": "string",
            "enum": list(TIPPING_ANSWERS),
            "description": "The top of the back and neck compared with the lower flank: clear (plainly "
                           "greyer or blacker - dark-tipped hair), slight (a little greyer or blacker), none "
                           "(the same colour, or only a deeper shade of it - a richer red or brown is none), "
                           "unknown (the back is not visible).",
        },
        "chest_lightness": {
            "type": "string",
            "enum": list(LIGHTNESS_ANSWERS),
            "description": "Only when the chest is cream: how much lighter it is than the back - slightly "
                           "(a paler shade of the same colour), clearly (plainly pale, still warm), near_white "
                           "(almost white). not_cream for any other chest answer.",
        },
        "note": {"type": "string", "description": "One short phrase on what you saw, or empty."},
    },
    "required": ["chest", "belly", "paws", "back_tipping", "chest_lightness", "note"],
    "additionalProperties": False,
}

BODY_COLOURS_SYSTEM = """You look at a photograph of a dog and name three fur colours.

Answer only from the fur you can see in this picture. Do not reason from
the breed or from what dogs with this kind of coat usually have.

chest: the front of the chest, from the throat under the chin down to
between the front legs. Name the colour of MOST of that area. A black or
chocolate chest with two small tan spots on the breastbone is
dark_tan_spots. A small white spot on a tan chest is still tan. Shading
and shine on black fur are still dark.

back_tipping: look at the top of the back and neck and compare it with the
lower flank. clear when the top is plainly greyer or blacker, the hair
dark-tipped; slight when it is a little greyer or blacker; none when it is
the same colour or only a deeper, richer shade of it (a deep red or brown
back over a lighter red flank is none). A black face mask does not count.

chest_lightness: when the chest is cream, how much lighter than the back
it is - slightly, clearly, or near_white. Otherwise not_cream.

Compare each part with the dog's back and sides. A chest, belly or feet
clearly LIGHTER than the back are cream or white, not tan, even when warm
light or a coloured background tints them: the pale front of a tan dog is
cream, its white socks are white. Tan means the same warm colour as the
back or close to it.

belly: the belly and the underside between the legs. Answer unknown unless
you can actually see it.

paws: the feet - the toes and the fur just above them. White when the fur
there is white or near-white, lighter than the legs above; tan only when
the feet are the same warm colour as the legs.

dark is black, chocolate or grey; tan is any clearly warm brown, red or
fawn; cream is off-white with a warm tint; white is a plain white."""


async def read_body_colours(transport: Transport, photo_blocks: list[dict[str, Any]],
                            samples: int = SAMPLES) -> tuple[str, str, str, dict[str, Any]]:
    """(chest, belly, paws, votes): the answer more than half of ALL the
    samples gave; unknown otherwise, which keeps the pattern's usual colour.
    A part most samples could not see is not painted from the one that
    guessed (eval v2 e14: belly tan from 1 of 3, the other two unknown)."""
    blocks = [*photo_blocks, {"type": "text", "text": "The front of the chest, the belly and the feet: "
                                                        "what colour is each?"}]

    async def one() -> dict[str, Any]:
        return await transport.complete(system=BODY_COLOURS_SYSTEM, blocks=blocks, schema=BODY_COLOURS_SCHEMA)

    payloads = await asyncio.gather(*(one() for _ in range(samples)), return_exceptions=True)
    readings: list[dict[str, Any]] = []
    for payload in payloads:
        if isinstance(payload, AgentRefusal):
            raise payload
        if isinstance(payload, Exception):
            continue
        readings.append(payload)

    def vote(key: str, allowed: tuple[str, ...]) -> tuple[str, dict[str, int]]:
        said = Counter(r.get(key) for r in readings if r.get(key) in allowed and r.get(key) != "unknown")
        top = said.most_common(1)
        return (top[0][0] if top and top[0][1] * 2 > len(readings) else "unknown"), dict(said)

    chest, chest_votes = vote("chest", CHEST_ANSWERS)
    belly, belly_votes = vote("belly", BELLY_ANSWERS)
    paws, paw_votes = vote("paws", PAW_ANSWERS)
    # the lightness among the samples that called the chest cream
    said = Counter(r.get("chest_lightness") for r in readings
                   if r.get("chest") == "cream" and r.get("chest_lightness") in CREAM_SHARE)
    lightness = said.most_common(1)[0][0] if said else None
    tipping, tipping_votes = vote("back_tipping", TIPPING_ANSWERS)
    return chest, belly, paws, {"chest": chest_votes, "belly": belly_votes, "paws": paw_votes,
                                "back_tipping": tipping, "back_tipping_votes": tipping_votes,
                                "chest_lightness": dict(said), "cream_share": CREAM_SHARE.get(lightness),
                                "samples": len(readings),
                                "notes": [str(r.get("note", "")) for r in readings if r.get("note")]}


# answer -> colour role (simple._role_colour); "dark_tan_spots" paints the
# chest dark and places a chest_spots marking in the points colour. The
# answer was named "spots" and a tan dog with a white chest spot got it 3/3
# ("a white spot on the chest") - a black front on a tan dog (eval v2 e05).
CHEST_ROLES = {"dark": "dark", "dark_tan_spots": "dark", "tan": "point", "cream": "cream", "white": "white"}
BELLY_ROLES = {"dark": "dark", "tan": "point", "cream": "cream", "white": "white"}
# the feet only when lighter than the coat: the answer used to only drop the
# white-leg features, so cream feet stayed the leg's tan (eval v7 e18)
PAW_ROLES = {"cream": "cream", "white": "white"}


def filter_leg_features(features: list[str], paws: str) -> list[str]:
    """The features without the white leg ones when the feet were seen and
    are not white; with white_paws added when they are white and the layout
    call named no white on the legs."""
    if paws in ("dark", "tan", "cream"):
        return [f for f in features if f not in WHITE_LEG_FEATURES]
    if paws == "white" and not any(f in features for f in WHITE_LEG_FEATURES):
        return [*features, "white_paws"]
    return list(features)


def apply_body_colours(region_colors: list[dict[str, Any]], markings: list[dict[str, Any]],
                       chest: str, belly: str, base: str, second: str | None,
                       paws: str = "unknown") -> list[str]:
    """Recolour the chest and underside entries in place from the answers;
    returns what changed. An unseen belly follows a dark chest (the layout's
    tan belly under a black chest is the rottweiler's, not what was seen)."""
    from .simple import _role_colour

    from .patterns import color_family

    entries = {e["region"]: e for e in region_colors}
    changed: list[str] = []
    # a black front with tan spots is a black dog's; on a light or tan coat
    # the answer misread a white spot
    if chest == "dark_tan_spots" and color_family(base) != "dark":
        chest = "unknown"
        changed.append("chest dark_tan_spots ignored: base not dark")

    def put(region: str, role: str, why: str) -> None:
        colour = _role_colour(role, base, second)
        entry = entries.get(region)
        if entry is None or entry.get("color") == colour:
            return
        entry.update({"color": colour, "note": why, "source": f"body:{why}"})
        changed.append(f"{region}:{colour}")

    # on a white or cream dog "lighter than the back" has nowhere to go: the
    # calls answered cream for the white chests of e01, e02, e13, e16 (eval
    # v3) and the bib came out beige. The layout's own light front stays.
    if color_family(base) == "light":
        chest = "unknown" if chest in ("cream", "white") else chest
        belly = "unknown" if belly in ("cream", "white") else belly
        paws = "unknown" if paws in ("cream", "white") else paws
    if chest in CHEST_ROLES:
        put("chest", CHEST_ROLES[chest], f"chest {chest}")
        if chest == "dark_tan_spots":
            colour = _role_colour("point", base, second)
            markings.append({"type": "chest_spots", "region": "chest", "color": colour, "size": "small",
                             "anchor": "none", "note": f"two small {colour} spots on the breastbone"})
            changed.append(f"chest_spots:{colour}")
    if belly in BELLY_ROLES:
        put("underside", BELLY_ROLES[belly], f"belly {belly}")
    elif chest in ("dark", "dark_tan_spots"):
        put("underside", "dark", "belly follows the dark chest")
    elif chest in CHEST_ROLES:
        # an unseen belly is the chest's colour: a cream bib over a tan belly
        # stopped at the ribs (eval v7 e18), and a tan chest over the
        # white_chest feature's white belly left a white belly (v3 e05)
        put("underside", CHEST_ROLES[chest], f"belly follows the {chest} chest")
    if paws in PAW_ROLES:
        put("paws", PAW_ROLES[paws], f"paws {paws}")
    return changed


__all__ = ["read_body_colours", "apply_body_colours", "filter_leg_features", "PAW_ANSWERS", "BODY_COLOURS_SCHEMA", "CHEST_ANSWERS", "BELLY_ANSWERS"]
