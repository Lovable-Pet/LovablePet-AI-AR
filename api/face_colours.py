"""The face's four colour questions, asked from the head close-up alone.

WHY A SEPARATE CALL
-------------------
Asked inside the layout call - beside five anchor renders and the question
"which layout is this" - the cheek and brow colours came back as what the
layout implies, three samples out of three, on a dog whose cheeks are white:
the samples' own reasons said "tan points on cheeks and brows". Absolute
colour words did not help and neither did adding the close-up to that call.
Here the model sees the head close-up and nothing else: no anchors, no
pattern names, no record hint. It can only answer with what it sees.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

from .agents import AgentRefusal, Transport

# Five, not three: with three the cheeks of a white-cheeked dog came back
# cream 2 / white 1 on one run and tan 2 / cream 1 on the next - a two-vote
# majority flips on one sample. The crop is small and the call cheap.
SAMPLES = 5

CHEEK_ANSWERS = ("white", "cream", "tan", "dark", "unknown")
BROW_ANSWERS = ("none", "white", "cream", "tan", "unknown")
MUZZLE_ANSWERS = ("dark", "white", "cream", "tan", "unknown")
# The inside of the ears was never read from the photograph before: the lite
# path painted it the pattern default (tan on every tan-point), and the one
# feature that could say otherwise (pink_ears, in the layout call) never
# fired. The head close-up is the only picture that shows the ear cup.
EAR_ANSWERS = ("dark", "tan", "cream", "white", "pink", "unknown")
# The strip between the eyes and up the forehead. The muzzle question stops
# at the eyes; a corgi's white blaze runs on up the forehead and the skull
# is one region, so without this answer the forehead stays the head colour.
BLAZE_ANSWERS = ("none", "white", "cream", "unknown")
# The crown - the top of the head between the ears. The layout call read a
# white jindo's faintly cream ears as a coloured cap ("piebald", eval e01,
# e02); the crown's own colour from the close-up settles it (lite.py). gray
# apart from dark: a white puppy's smoky grey cap was painted black (shelter
# puppy 14, user: "검은색은 좀 더 연한 느낌").
CROWN_ANSWERS = ("white", "cream", "tan", "dark", "gray", "unknown")
# The fur directly above each eye, between the eyelid and the brow spot.
# Two black-and-tans with the same answers everywhere else differ here: one
# has cream "spectacles" round the eyes (eval e08), the other is black above
# and beside the eyes and light only below them (a black shiba, jindo29) -
# the build painted both with spectacles and the refine model, following the
# photo, repainted the second dog's (texture.compose_body). Asked as "round
# the eye" the black shiba came back light 3 of 5: the samples counted the
# tan under its eyes ("light eye rings"). Above the eye is where they differ.
EYE_RING_ANSWERS = ("light", "dark", "unknown")
# The cheeks against the brow spots, asked as a comparison: named alone, a
# black-and-tan puppy's tan cheeks (photo L 55, chroma 14) and e08's cream
# cheeks fell on either side of every wording - one wording read both tan,
# the next both cream (11-01). "same" with tan brows makes the cheeks tan
# (lite.py).
CHEEK_BROW_ANSWERS = ("same", "paler", "none", "unknown")
# ... and dark wins on this share of the samples, not on a majority. Seven
# black-headed dogs, five samples each (10-31): the one with cream above the
# eyes (e08) drew no dark vote in ten; the black shiba, black above the eyes
# under tan brow spots set close to them, drew 2 of 5 twice; e16 3, the
# others 5. A majority missed the shiba; two votes of five separates all.
EYE_RING_DARK_SHARE = 0.4

FACE_COLOURS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "cheeks": {
            "type": "string",
            "enum": list(CHEEK_ANSWERS),
            "description": "The colour of the fur on the cheeks and the sides of the muzzle: white, cream "
                           "(off-white with a warm tint), tan (any warm brown, red or fawn), dark (black, "
                           "chocolate or grey - the same as the top of the head), unknown (not visible).",
        },
        "brows": {
            "type": "string",
            "enum": list(BROW_ANSWERS),
            "description": "The colour of the small spots of fur just above the eyes, if there are any: "
                           "none (no spots; the forehead fur runs over the eyes unbroken), white, cream, tan, "
                           "unknown (not visible).",
        },
        "muzzle": {
            "type": "string",
            "enum": list(MUZZLE_ANSWERS),
            "description": "The colour of the fur on the TOP of the muzzle, the strip from between the eyes "
                           "down to the nose (not the sides of the muzzle): dark (black, chocolate or grey), "
                           "white, cream, tan, unknown (not visible).",
        },
        "ears": {
            "type": "string",
            "enum": list(EAR_ANSWERS),
            "description": "The colour INSIDE the ears - the cupped inner surface, not the back of the ear: "
                           "dark (black, chocolate or grey), tan (warm brown, red or fawn fur), cream, white, "
                           "pink (bare skin showing through thin hair), unknown (the ear cup is not visible).",
        },
        "blaze": {
            "type": "string",
            "enum": list(BLAZE_ANSWERS),
            "description": "A light stripe that continues from the top of the muzzle UP BETWEEN the eyes onto "
                           "the forehead: none (the fur between and above the eyes is the head colour), white, "
                           "cream, unknown (the forehead is not visible). A light muzzle top that stops below "
                           "the eyes is none.",
        },
        "crown": {
            "type": "string",
            "enum": list(CROWN_ANSWERS),
            "description": "The colour of the fur on the top of the head between the ears (not the ears): "
                           "white, cream, tan (any warm brown, red or fawn), dark (black or chocolate), gray "
                           "(grey or blue-grey, clearly lighter than black), unknown.",
        },
        "eye_ring": {
            "type": "string",
            "enum": list(EYE_RING_ANSWERS),
            "description": "The fur directly ABOVE each eye, between the upper eyelid and the brow spot, and "
                           "at the eye's outer corner (not below the eye): light (lighter than the top of the "
                           "head - light fur runs up over the eye), dark (the head colour - the eye's upper side "
                           "sits in dark fur), unknown (not visible).",
        },
        "cheeks_vs_brows": {
            "type": "string",
            "enum": list(CHEEK_BROW_ANSWERS),
            "description": "The cheeks compared with the brow spots above the eyes: same (the cheeks are as "
                           "deep and as warm as the brow spots - the same points colour), paler (the cheeks are "
                           "plainly paler than the brow spots - beige, cream or white), none (there are no brow "
                           "spots), unknown (not visible).",
        },
        "note": {"type": "string", "description": "One short phrase on what you saw, or empty."},
    },
    "required": ["cheeks", "brows", "muzzle", "ears", "blaze", "crown", "eye_ring", "cheeks_vs_brows", "note"],
    "additionalProperties": False,
}

FACE_COLOURS_SYSTEM = """You look at a close-up photograph of a dog's head and name seven fur colours
and make one comparison.

Answer only from the fur you can see in this picture. Do not reason from
the breed or from what dogs with this kind of coat usually have.

cheeks: the fur on the cheeks and the sides of the muzzle, below and beside
the eyes. Name the colour of MOST of that area. A tan or dark patch right
under or beside the eye is a marking and does not count: a cheek that is
white or cream over most of its area with a tan patch under the eye is
white or cream, not tan. Compare the cheeks with the top of the head: if
they are clearly LIGHTER than the crown, they are cream or white, not tan,
even when warm light tints them. Tan means the same warm colour as the
crown or close to it.

brows: the small spots of fur directly above each eye, if the forehead
colour is broken by them. Name the spot's own colour. If the forehead fur
runs over the eyes with no spot, answer none.

muzzle: the fur on the top of the muzzle only - the strip running from
between the eyes down to the nose. The sides of the muzzle are the cheeks
and do not count here. A white blaze that runs up this strip is white. The
black nose leather, the lips and the open mouth are not fur and do not
count: a tan muzzle top ending in a black nose is tan. Answer dark only
when the FUR of the strip itself is black or dark grey, as on a masked dog.

ears: the fur or skin INSIDE the ears - the cupped surface that faces the
camera when the ears stand up or the side when they hang. The back of the
ear and its dark rim are the head colour and do not count. The hair inside
an ear is sparse and pale: whitish or off-white hair is white or cream, and
tan means clearly orange or brown hair, not a pale warm tint. Bare skin
showing through thin hair is pink. If neither ear cup is visible, answer
unknown.

blaze: whether a light stripe runs on from the top of the muzzle up
BETWEEN the eyes and onto the forehead. Answer none when the fur between
the eyes and above them is the head colour, even if the muzzle top below
the eyes is white. Answer white or cream only for a stripe that clearly
reaches above the eyes.

crown: the fur on the top of the head between the ears, not the ears
themselves. A white head with faintly cream ear tips has a white crown.
gray is a dark fur plainly lighter than black - a smoky or blue-grey cap;
a black crown with a sheen in the light is still dark.

eye_ring: the fur directly ABOVE each eye - the strip between the upper
eyelid and the brow spot - and at the eye's outer corner. The fur BELOW the
eye is the cheek and does not count here, even when it is light; neither do
the brow spot itself or the thin black rim of the eyelid. light: that strip
is clearly lighter than the top of the head - light fur runs up over the
eye to the brow. dark: the eye's upper side sits in the head colour. On a
dog whose head is one colour all over, answer dark.

cheeks_vs_brows: put the cheeks beside the brow spots, ignoring the
crown. same: the cheeks carry the brow spots' colour, as deep and as warm,
even if both are much lighter than a black crown. paler: the cheeks are
plainly paler than the brow spots - beige, cream or white over most of the
cheek. none when there are no brow spots.

white is a plain white; cream is off-white with a warm tint; tan is any
clearly warm brown, red or fawn; dark is black, chocolate or grey."""


async def read_face_colours(transport: Transport, crop_block: dict[str, Any],
                            samples: int = SAMPLES) -> tuple[str, str, str, str, str, dict[str, Any]]:
    # the crown and the eye ring are returned in votes["crown_answer"] and
    # votes["eye_ring_answer"] so the five-tuple callers keep working
    """(cheeks, brows, muzzle, ears, blaze, votes): each the answer more than
    half of ALL the samples gave; unknown otherwise, and an unknown region
    keeps the pattern's usual colour. Taking the most-named answer among the
    samples that saw the region let one sample of five paint tan brow spots
    the other four did not see (eval v2 e20)."""
    blocks = [crop_block, {"type": "text", "text": "The cheeks, the brow spots, the top of the muzzle, the "
                                                   "inside of the ears, the fur round the eyes, and whether a "
                                                   "light blaze runs up between the eyes: what colour is each?"}]

    async def one() -> dict[str, Any]:
        return await transport.complete(system=FACE_COLOURS_SYSTEM, blocks=blocks, schema=FACE_COLOURS_SCHEMA)

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

    cheeks, cheek_votes = vote("cheeks", CHEEK_ANSWERS)
    brows, brow_votes = vote("brows", BROW_ANSWERS)
    muzzle, muzzle_votes = vote("muzzle", MUZZLE_ANSWERS)
    ears, ear_votes = vote("ears", EAR_ANSWERS)
    blaze, blaze_votes = vote("blaze", BLAZE_ANSWERS)
    crown, crown_votes = vote("crown", CROWN_ANSWERS)
    eye_ring, eye_ring_votes = vote("eye_ring", EYE_RING_ANSWERS)
    cheek_brow, cheek_brow_votes = vote("cheeks_vs_brows", CHEEK_BROW_ANSWERS)
    if readings and eye_ring_votes.get("dark", 0) >= EYE_RING_DARK_SHARE * len(readings):
        eye_ring = "dark"
    votes = {"crown_answer": crown, "crown": crown_votes, "eye_ring_answer": eye_ring, "eye_ring": eye_ring_votes,
             "cheeks_vs_brows_answer": cheek_brow, "cheeks_vs_brows": cheek_brow_votes, "cheeks": cheek_votes, "brows": brow_votes, "muzzle": muzzle_votes, "ears": ear_votes,
             "blaze": blaze_votes, "samples": len(readings),
             "notes": [str(r.get("note", "")) for r in readings if r.get("note")]}
    return cheeks, brows, muzzle, ears, blaze, votes


__all__ = ["read_face_colours", "FACE_COLOURS_SCHEMA", "CHEEK_ANSWERS", "BROW_ANSWERS", "MUZZLE_ANSWERS",
           "EAR_ANSWERS", "BLAZE_ANSWERS", "EYE_RING_ANSWERS", "CHEEK_BROW_ANSWERS"]
