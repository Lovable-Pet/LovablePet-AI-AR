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
# Where the coloured patches of a white-bodied coat sit and how much of the
# body they cover. The layout call says only that there are patches, and one
# fixed placement fit neither a white puppy with one small flank patch
# (shelter maltese 6, drawn as a saddle) nor a parti with a saddle and a rump
# patch (maltese 7, drawn without the rump). On five patched dogs (11-02:
# maltese 6 and 7, eval e13 and e15, puppy 13) the amount split small /
# large as the photos do. Where: asked as a saddle yes/no beside a flank
# yes/no, saddle came back yes in 15 of 15 samples, the flank puppy's too
# ("small patch on back"); asked instead whether the top line of the back
# right above the largest patch is white, the flank puppy came back side 3/3
# and the saddled parti over_spine 3/3. The rump is its own yes/no - a dog
# can have a saddle and a rump patch.
PATCH_PLACE_ANSWERS = ("yes", "no", "unknown")
PATCH_LARGEST_ANSWERS = ("over_spine", "side", "none", "unknown")
PATCH_AMOUNT_ANSWERS = ("none", "small", "large", "most", "unknown")
# The top line of the neck behind a coloured head: the head's colour
# running on down to the shoulders, or white. Asked as "the back and upper
# sides of the neck", 2 of 3 white-necked dogs with coloured hanging ears
# came back head (shelter maltese 6, eval e15). Patches stay off the
# neck (texture.PATCH_OFF_SOURCES: four of five patched dogs had a white
# neck), and a parti whose brown head ran down its neck to a white band at
# the shoulders came out with a white neck under a brown head (shelter
# maltese 7, user 11-07: "갈색털이 뒷목 거의 덮었는데").
NAPE_ANSWERS = ("head", "light", "unknown")
# The trunk: white or coloured. Asked only of a model whose profile lists it
# (reading_model body_extra): gpt-6-luna reads a white dog with a black head
# seen from the front as "irish black/white" - its literal wording says a dark
# head with a white blaze is irish when the body is not in the photograph -
# and the irish layout then paints the back black (new shelter photos 18 and
# 17 of 2026-11-13 and -14: "mostly white body with a dark head and ears" in
# the samples' own words). The other answers of this call do not say it:
# patch_amount's "none" is a white body and a dog coloured all over alike,
# and the nape came back unknown from a front view. The whole animal's
# lightness does not say it either (a white-collared black collie measured
# lighter than the white terrier in shade).
TRUNK_ANSWERS = ("white", "coloured", "unknown")
EXTRA_QUESTIONS: dict[str, tuple[dict[str, Any], str]] = {
    "trunk": ({
        "type": "string",
        "enum": list(TRUNK_ANSWERS),
        "description": "The trunk - the back, the sides and the rump, behind the neck; not the head, the front "
                       "of the chest, the legs or the tail: white (white, cream or near-white all over, or "
                       "white with a few coloured patches), coloured (black, brown, tan, grey or another "
                       "colour covers most of it - also a dark dog with a white chest, collar or feet), "
                       "unknown (no part of the trunk is in the photograph).",
    }, """trunk: the back, the sides and the rump - the body behind the neck, leaving
out the head, the front of the chest, the legs and the tail. white when the
fur there is white, cream or near-white, all over or with a few coloured
patches on it; coloured when black, brown, tan, grey or another colour
covers most of it. A dark dog with a white chest, collar or feet is
coloured; a white dog whose head or ears alone are dark is white. From the
front, go by the shoulders and the sides you can see behind the chest."""),
}

BODY_COLOURS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "chest": {
            "type": "string",
            "enum": list(CHEST_ANSWERS),
            "description": "The colour of MOST of the front of the chest, from the throat under the chin down "
                           "to between the front legs: dark (black, chocolate or grey), dark_tan_spots (black "
                           "or chocolate, broken only by two small tan spots on the breastbone), tan (any warm "
                           "brown, red or fawn - also when a small white spot sits on it), cream, white "
                           "(only when white covers most of the front - a small white spot or a few white "
                           "hairs on a dark or tan chest is dark or tan), unknown (not visible).",
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
        "patch_amount": {
            "type": "string",
            "enum": list(PATCH_AMOUNT_ANSWERS),
            "description": "Patches of colour on a white or near-white body, from behind the neck to the root "
                           "of the tail (not the head, the neck or the tail): none (no such patches - a white "
                           "body, colour only on the head, ears or tail, or a dog coloured all over), small (one "
                           "or two patches, each smaller than the dog's head), large (patches larger than the "
                           "head, the body still mostly white), most (half the body or more is coloured), "
                           "unknown (the body is not visible).",
        },
        "patch_largest": {
            "type": "string",
            "enum": list(PATCH_LARGEST_ANSWERS),
            "description": "The largest patch on the body: over_spine (it covers the middle line along the top "
                           "of the back - the top line of the back right above it is coloured too), side (it "
                           "sits on the side of the body with white fur between it and the spine - the top "
                           "line of the back right above it is white), none (no patches on the body), unknown "
                           "(the top of the back is not visible).",
        },
        "patch_rump": {
            "type": "string",
            "enum": list(PATCH_PLACE_ANSWERS),
            "description": "Is a patch over the hips, the top of the hindquarters or the outside of the "
                           "thighs, in front of the tail? yes, no (also when there are no patches), unknown "
                           "(the hindquarters are not visible).",
        },
        "nape": {
            "type": "string",
            "enum": list(NAPE_ANSWERS),
            "description": "The top line of the neck, from the back of the skull to the shoulders - not the "
                           "ears (long ears hanging over the neck do not count): head (the head's colour carries "
                           "on along the top of the neck to the shoulders or most of the way - also for a dog "
                           "coloured all over), light (white or near-white there - the colour stops at the skull "
                           "and the ears), unknown (not visible).",
        },
        "note": {"type": "string", "description": "One short phrase on what you saw, or empty."},
    },
    "required": ["chest", "belly", "paws", "back_tipping", "chest_lightness", "patch_amount", "patch_largest",
                 "patch_rump", "nape", "note"],
    "additionalProperties": False,
}

BODY_COLOURS_SYSTEM = """You look at a photograph of a dog and name the colours of its body fur.

Answer only from the fur you can see in this picture. Do not reason from
the breed or from what dogs with this kind of coat usually have.

chest: the front of the chest, from the throat under the chin down to
between the front legs. Name the colour of MOST of that area. A black or
chocolate chest with two small tan spots on the breastbone is
dark_tan_spots. A small white spot or a few white hairs on a tan chest is
still tan, on a black chest still dark - white only when white covers most
of the front. Shading and shine on black fur are still dark.

nape: the top line of the neck, between the back of the skull and the
shoulders. Look there, not at the ears: coloured ears hanging down over a
white neck do not make it coloured. head when the head's colour carries on
along it to the shoulders or most of the way; light when the neck behind
the skull is white or near-white.

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
fawn; cream is off-white with a warm tint; white is a plain white.

patches: only patches of colour on a white or near-white body, between the
neck and the root of the tail - not the head, the ears or the tail. Say how
much of the body they cover, and whether a patch sits over the hips and
thighs (rump). For the largest patch, look at the top line of the back
right above it: coloured there, it covers the spine (over_spine); white
there, with white fur between the patch and the spine, it is on the side
(side). A coloured dog with a white chest or white feet has no patches
(none, no)."""


def _asked(extra: tuple[str, ...]) -> tuple[str, dict[str, Any]]:
    """The call's system text and schema with the `extra` questions added
    (EXTRA_QUESTIONS); with none, the two as they stand."""
    if not extra:
        return BODY_COLOURS_SYSTEM, BODY_COLOURS_SCHEMA
    properties = dict(BODY_COLOURS_SCHEMA["properties"])
    note = properties.pop("note")
    system = BODY_COLOURS_SYSTEM
    for name in extra:
        properties[name], text = EXTRA_QUESTIONS[name]
        system += "\n\n" + text
    properties["note"] = note
    return system, {**BODY_COLOURS_SCHEMA, "properties": properties, "required": list(properties)}


async def read_body_colours(transport: Transport, photo_blocks: list[dict[str, Any]],
                            samples: int = SAMPLES, requests: int = 1,
                            margins: tuple[float, ...] = (),
                            extra: tuple[str, ...] = ()) -> tuple[str, str, str, dict[str, Any]]:
    """(chest, belly, paws, votes): the answer more than half of ALL the
    samples gave; unknown otherwise, which keeps the pattern's usual colour.
    A part most samples could not see is not painted from the one that
    guessed (eval v2 e14: belly tan from 1 of 3, the other two unknown).

    `requests` over one: the same question in that many requests at once,
    each on its own view of the photograph when `margins` is given
    (sampling.sample_views). The answers above come from the first, as
    ever; the others only add votes["asks"], the three colours voted over
    every request's samples (lite's body_by_asks reads it).

    `extra`: questions asked of this model only (EXTRA_QUESTIONS); their
    answers come back in votes[name] with votes[name + "_votes"], voted
    over every request's samples."""
    from .sampling import sample_views

    system, schema = _asked(tuple(extra))

    blocks = [*photo_blocks, {"type": "text", "text": "The front of the chest, the belly and the feet: "
                                                        "what colour is each? And the patches on the body, "
                                                        "if any: where, and how much?"}]

    batches = await sample_views(transport, system=system, blocks=blocks, schema=schema,
                                 n=samples, requests=requests, margins=margins)
    readings: list[dict[str, Any]] = []
    for payload in batches[0]:
        if isinstance(payload, AgentRefusal):
            raise payload
        if isinstance(payload, Exception):
            continue
        readings.append(payload)

    def vote(key: str, allowed: tuple[str, ...], over: list[dict[str, Any]] | None = None) -> tuple[str, dict[str, int]]:
        over = readings if over is None else over
        said = Counter(r.get(key) for r in over if r.get(key) in allowed and r.get(key) != "unknown")
        top = said.most_common(1)
        return (top[0][0] if top and top[0][1] * 2 > len(over) else "unknown"), dict(said)

    def lightness_of(over: list[dict[str, Any]]) -> tuple[str | None, dict[str, int]]:
        # the lightness among the samples that called the chest cream
        said = Counter(r.get("chest_lightness") for r in over
                       if r.get("chest") == "cream" and r.get("chest_lightness") in CREAM_SHARE)
        return (said.most_common(1)[0][0] if said else None), dict(said)

    parts = (("chest", CHEST_ANSWERS), ("belly", BELLY_ANSWERS), ("paws", PAW_ANSWERS))
    (chest, chest_votes), (belly, belly_votes), (paws, paw_votes) = (vote(key, allowed) for key, allowed in parts)
    lightness, lightness_votes = lightness_of(readings)
    tipping, tipping_votes = vote("back_tipping", TIPPING_ANSWERS)
    nape, nape_votes = vote("nape", NAPE_ANSWERS)
    votes = {"chest": chest_votes, "belly": belly_votes, "paws": paw_votes,
             "back_tipping": tipping, "back_tipping_votes": tipping_votes,
             "nape": nape, "nape_votes": nape_votes,
             "chest_lightness": lightness_votes, "cream_share": CREAM_SHARE.get(lightness),
             **vote_patches(readings),
             "samples": len(readings),
             "notes": [str(r.get("note", "")) for r in readings if r.get("note")]}
    each = [[p for p in batch if isinstance(p, dict)] for batch in batches]
    every = [p for batch in each for p in batch]
    for name in extra:
        votes[name], votes[f"{name}_votes"] = vote(name, tuple(EXTRA_QUESTIONS[name][0]["enum"]), every)
        votes[f"{name}_samples"] = len(every)
    if len(batches) > 1:
        asks: dict[str, Any] = {"requests": len(batches), "samples": len(every)}
        for key, allowed in parts:
            asks[f"{key}_answer"], asks[key] = vote(key, allowed, every)
        asks["cream_share"] = CREAM_SHARE.get(lightness_of(every)[0])
        # what each request alone would have answered (chest/belly/paws), for the log
        asks["each"] = ["/".join(vote(key, allowed, batch)[0] for key, allowed in parts) for batch in each]
        votes["asks"] = asks
    return chest, belly, paws, votes


def light_by_family(answer: str, said: dict[str, int] | None, samples: int | None, base_color: str) -> str:
    """A part no single colour won (`answer` unknown) that most samples
    called light is light: the light colour more of them named, white on a
    tie for a white dog and cream for any other. A black-and-tan shiba's
    white bib, read dark / cream / white, kept the layout's tan chest
    (shelter adult jindo 6); on file the split light majorities were all
    light in their photos (a cream chihuahua's feet, a white puppy's chest,
    a white dog's feet and belly)."""
    if answer != "unknown" or not said or not samples:
        return answer
    cream, white = int(said.get("cream", 0)), int(said.get("white", 0))
    if (cream + white) * 2 <= samples:
        return answer
    if cream == white:
        return "white" if str(base_color).lower() == "white" else "cream"
    return "white" if white > cream else "cream"


def plain_front(chest: str, belly: str, base: str | None, second: str | None,
                overlay: str | None = None) -> tuple[str, str, list[str]]:
    """(chest, belly, what was dropped): a "dark" chest or belly on a coat
    with no dark colour in it is not painted. The body call says dark of a
    front in shadow, and a black bib or belly on a tan dog is a dog nobody
    photographed: a fawn mastiff sitting, belly dark 5 of 9, got a black
    belly (new shelter photo 1, 2026-11-14), a red dog bending forward,
    chest dark 9 of 9, a black chest and belly (photo 20 of 11-13), a
    grey-brown dog read tan, chest dark 3 of 3, the same (photo 13). Those
    are the three plain warm coats of 90 on file with a dark vote for the
    front, and none has a dark front. lite.dark_chest_seen_light undid it
    only where the chest was measured and measured light. A coat with a dark
    second colour, or brindled, keeps the answer: its front can be dark."""
    from .patterns import color_family

    dark = any(color_family(str(c or "")) == "dark" for c in (base, second))
    if dark or overlay == "brindle":
        return chest, belly, []
    dropped = []
    if chest in ("dark", "dark_tan_spots"):
        dropped.append(f"chest {chest}")
        chest = "unknown"
    if belly == "dark":
        dropped.append("belly dark")
        belly = "unknown"
    return chest, belly, dropped


def front_not_deeper(chest: str, belly: str, pattern: str | None, base: str | None) -> tuple[str, str, list[str]]:
    """(chest, belly, what was dropped): a "tan" chest or belly on a solid
    white or cream coat is not painted. A dog's front is its coat's colour
    or paler; a tan bib and belly on a cream poodle is a dog nobody has -
    the body call said tan 3 of 3 for the front of a matted, dirty one read
    solid cream (hard photo 73, 2026-11-14), the one of 214 dogs on file
    with a front deeper than its plain coat."""
    from .patterns import color_family

    if pattern != "solid" or color_family(str(base or "")) != "light":
        return chest, belly, []
    dropped = []
    if chest == "tan":
        dropped.append("chest tan")
        chest = "unknown"
    if belly == "tan":
        dropped.append("belly tan")
        belly = "unknown"
    return chest, belly, dropped


def vote_patches(readings: list[dict[str, Any]]) -> dict[str, Any]:
    """patch_amount: the amount more than half of all the samples gave, else
    unknown; patch_places: "side" when more than half put the largest patch
    on the side, "rump" when more than half said yes to the rump."""
    def majority(key: str, allowed: tuple[str, ...]) -> tuple[str, dict[str, int]]:
        said = Counter(r.get(key) for r in readings if r.get(key) in allowed)
        top = said.most_common(1)
        return (top[0][0] if top and top[0][1] * 2 > len(readings) else "unknown"), dict(said)

    amount, amounts = majority("patch_amount", PATCH_AMOUNT_ANSWERS)
    largest, largests = majority("patch_largest", PATCH_LARGEST_ANSWERS)
    rump, rumps = majority("patch_rump", PATCH_PLACE_ANSWERS)
    places = (["side"] if largest == "side" else []) + (["rump"] if rump == "yes" else [])
    return {"patch_amount": amount, "patch_places": places,
            "patch_votes": {"amount": amounts, "largest": largests, "rump": rumps}}


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
    # tan or dark feet under the layout's light legs on a warm coat: the legs
    # are the coat's colour - an urajiro's cream legs where the photo shows
    # ginger forelegs and the feet read tan 2 of 2 (shelter chihuahua 9, user
    # 11-07). The layout's light paws stay, the toes under the coloured leg
    # (simple.default_gradients)
    legs = entries.get("legs_lower")
    if (paws in ("tan", "dark") and legs is not None and str(legs.get("source", "")).startswith("pattern_default")
            and color_family(str(legs.get("color", ""))) == "light" and color_family(base) == "point"):
        put("legs_lower", "base", f"legs follow the {paws} feet")
    return changed


__all__ = ["read_body_colours", "apply_body_colours", "filter_leg_features", "vote_patches", "plain_front",
           "PAW_ANSWERS", "BODY_COLOURS_SCHEMA", "CHEST_ANSWERS", "BELLY_ANSWERS", "TRUNK_ANSWERS", "EXTRA_QUESTIONS"]
