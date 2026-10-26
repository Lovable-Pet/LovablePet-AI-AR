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
# e02); the crown's own colour from the close-up settles it (lite.py).
CROWN_ANSWERS = ("white", "cream", "tan", "dark", "unknown")

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
                           "white, cream, tan (any warm brown, red or fawn), dark (black, chocolate or grey), "
                           "unknown.",
        },
        "note": {"type": "string", "description": "One short phrase on what you saw, or empty."},
    },
    "required": ["cheeks", "brows", "muzzle", "ears", "blaze", "crown", "note"],
    "additionalProperties": False,
}

FACE_COLOURS_SYSTEM = """You look at a close-up photograph of a dog's head and name six fur colours.

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

white is a plain white; cream is off-white with a warm tint; tan is any
clearly warm brown, red or fawn; dark is black, chocolate or grey."""


async def read_face_colours(transport: Transport, crop_block: dict[str, Any],
                            samples: int = SAMPLES) -> tuple[str, str, str, str, str, dict[str, Any]]:
    # the crown is returned in votes["crown_answer"] so the five-tuple
    # callers keep working
    """(cheeks, brows, muzzle, ears, blaze, votes): each the answer more than
    half of ALL the samples gave; unknown otherwise, and an unknown region
    keeps the pattern's usual colour. Taking the most-named answer among the
    samples that saw the region let one sample of five paint tan brow spots
    the other four did not see (eval v2 e20)."""
    blocks = [crop_block, {"type": "text", "text": "The cheeks, the brow spots, the top of the muzzle, the "
                                                   "inside of the ears, and whether a light blaze runs up "
                                                   "between the eyes: what colour is each?"}]

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
    votes = {"crown_answer": crown, "crown": crown_votes, "cheeks": cheek_votes, "brows": brow_votes, "muzzle": muzzle_votes, "ears": ear_votes,
             "blaze": blaze_votes, "samples": len(readings),
             "notes": [str(r.get("note", "")) for r in readings if r.get("note")]}
    return cheeks, brows, muzzle, ears, blaze, votes


__all__ = ["read_face_colours", "FACE_COLOURS_SCHEMA", "CHEEK_ANSWERS", "BROW_ANSWERS", "MUZZLE_ANSWERS",
           "EAR_ANSWERS", "BLAZE_ANSWERS"]
