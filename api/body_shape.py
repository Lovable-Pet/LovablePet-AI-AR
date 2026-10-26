"""The dog's outline, read from the photograph: ears, tail, build, muzzle,
and three proportions the template cannot change (coat length, leg length,
head size) that fit.py uses to judge whether the template suits the dog.

The answers become morph-target weights on the template (shape_keys.py,
shape.py). Asked in their own call, like the face and body colours: a shape
question beside a layout question would be answered from the layout too.
Every question has "unknown", which leaves the template's own shape.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

from .agents import AgentRefusal, Transport

SAMPLES = 3
RETRIES = 4
RETRY_WAIT_S = 2.0

EAR_ANSWERS = ("erect", "tipped", "drop", "unknown")
TAIL_BUILD_ANSWERS = ("bushy", "thin", "unknown")
TAIL_CARRIAGE_ANSWERS = ("up", "down", "unknown")
BUILD_ANSWERS = ("slim", "medium", "stocky", "unknown")
MUZZLE_ANSWERS = ("short", "medium", "long", "unknown")
COAT_LENGTH_ANSWERS = ("short", "medium", "long", "unknown")
LEG_LENGTH_ANSWERS = ("short", "normal", "long", "unknown")
HEAD_SIZE_ANSWERS = ("normal", "large", "unknown")

BODY_SHAPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ears": {"type": "string", "enum": list(EAR_ANSWERS),
                 "description": "erect (standing up to the tip), tipped (standing at the base, the top third "
                                "folding forward), drop (hanging down beside the head), unknown."},
        "tail_build": {"type": "string", "enum": list(TAIL_BUILD_ANSWERS),
                       "description": "bushy (thick, plumed fur), thin (short-haired, whip-like), unknown."},
        "tail_carriage": {"type": "string", "enum": list(TAIL_CARRIAGE_ANSWERS),
                          "description": "up (held level, raised or curled over the back), down (hanging "
                                         "below the level of the back), unknown (not visible)."},
        "build": {"type": "string", "enum": list(BUILD_ANSWERS),
                  "description": "slim (narrow chest, tucked waist, visible leanness), medium, stocky "
                                 "(broad, deep, heavy-set), unknown."},
        "muzzle": {"type": "string", "enum": list(MUZZLE_ANSWERS),
                   "description": "short (flat face, muzzle much shorter than the skull), medium, long "
                                  "(narrow muzzle about as long as the skull or longer), unknown."},
        "coat_length": {"type": "string", "enum": list(COAT_LENGTH_ANSWERS),
                        "description": "short (smooth, close to the skin), medium (a dense double coat, "
                                       "fluffy but the body outline clear), long (hair hangs or flows and hides "
                                       "the outline, as on a maltese, shih tzu or yorkie), unknown."},
        "leg_length": {"type": "string", "enum": list(LEG_LENGTH_ANSWERS),
                       "description": "compared with the depth of the chest: short (legs much shorter than the "
                                      "chest is deep, as on a corgi or dachshund), normal, long (lanky, as on a "
                                      "greyhound), unknown."},
        "head_size": {"type": "string", "enum": list(HEAD_SIZE_ANSWERS),
                      "description": "normal, or large (the head big for the body, as on a toy breed or a young "
                                     "puppy), unknown."},
        "note": {"type": "string", "description": "One short phrase, or empty."},
    },
    "required": ["ears", "tail_build", "tail_carriage", "build", "muzzle", "coat_length", "leg_length",
                 "head_size", "note"],
    "additionalProperties": False,
}

BODY_SHAPE_SYSTEM = """You look at a photograph of a dog and describe its outline in eight words.

Answer only from what you can see. Do not reason from the breed. Ignore
the coat's colours entirely. Answer unknown for a part you cannot see.

ears: erect when they stand to the tip; tipped when they stand at the base
and the top folds forward; drop when they hang down beside the head.
tail_build: bushy for a thick plumed tail, thin for a short-haired one.
tail_carriage: how the tail is held in this picture - up (level, raised or
curled) or down (hanging below the back).
build: slim, medium or stocky, for the trunk.
muzzle: short, medium or long, compared with the length of the skull.
coat_length: short, medium (dense double coat, outline still clear) or long
(hair hangs and hides the outline).
leg_length: short, normal or long, compared with how deep the chest is.
head_size: normal, or large for the body (toy breed or young puppy)."""

# with the model sheet (model_sheet.py): which of our models this dog looks
# like, and where it differs from it. Asked in the same call to cost nothing
# extra; the eight answers above stay about the photographed dog. A first
# version asked for close / rough / poor outright and the call never said
# poor - a pug and a pomeranian were "close" to the chihuahua (30 dogs,
# 2026-10-18) - so it now names the differences and the count decides.
MODEL_MATCH_ANSWERS = ("close", "rough", "poor")
DIFFERENCES = ("ears", "coat length", "coat texture", "muzzle", "head", "legs", "body", "tail")
POOR_FROM = 2   # this many clear differences: the model would look forced
MODELS_SCHEMA: dict[str, Any] = {
    **BODY_SHAPE_SCHEMA,
    "properties": {
        **BODY_SHAPE_SCHEMA["properties"],
        "closest_model": {"type": "string", "enum": ["A", "B", "C", "D", "E", "F"],
                          "description": "The letter of the grey model that, painted in this dog's coat, would "
                                         "most look like this dog."},
        "differences": {"type": "array", "items": {"type": "string", "enum": list(DIFFERENCES)},
                        "description": "Every part where this dog clearly differs from that model, beyond what the "
                                       "model can change. Empty when none."},
    },
    "required": [*BODY_SHAPE_SCHEMA["required"], "closest_model", "differences"],
}
MODELS_SYSTEM = BODY_SHAPE_SYSTEM + """

A second image shows our grey 3D dog models, lettered. The eight answers
above describe the dog in the photograph, never the models. Then:
closest_model: the model that, painted in this dog's coat and resized to its
weight, would most look like this dog. Compare the shape only - ears, coat
length and texture, legs, head, muzzle, tail, proportions - not the size of
the picture and not colour.
differences: be strict, as a person who knows this dog would be. List each
part where the dog clearly differs from the chosen model: ears (shape, size,
set), coat length, coat texture (smooth, fluffy, curly, wiry), muzzle (length,
width, a flat face), head (size and shape for the body), legs (length,
thickness), body (depth, length, a barrel or a slim waist), tail. Leave out
what the model can change and leave out colour, markings and size."""


def match_from(differences: list[str]) -> str:
    """close with no clear difference, rough with one, poor from POOR_FROM."""
    n = len(set(differences))
    return "close" if n == 0 else "poor" if n >= POOR_FROM else "rough"

# answers -> morph-target weights (shape_keys.py target names)
EAR_TARGETS = {"tipped": {"ear_fold": 1.0}, "drop": {"ear_drop": 1.0}}
BUILD_TARGETS = {"slim": {"body_slim": 1.0}, "stocky": {"body_slim": -0.8}}
MUZZLE_TARGETS = {"short": {"muzzle_long": -1.0}, "long": {"muzzle_long": 0.6}}


def tail_target(build: str, carriage: str) -> dict[str, float]:
    # a thinned tail lost its volume and read as wrong on the model (user,
    # eval demo e08): thin is answered but no longer drawn
    thin = False
    down = carriage == "down"
    if thin and down:
        return {"tail_down_thin": 1.0}
    if down:
        return {"tail_down": 1.0}
    if thin:
        return {"tail_thin": 1.0}
    return {}


def shape_weights(answers: dict[str, str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    weights.update(EAR_TARGETS.get(answers.get("ears", ""), {}))
    weights.update(tail_target(answers.get("tail_build", ""), answers.get("tail_carriage", "")))
    weights.update(BUILD_TARGETS.get(answers.get("build", ""), {}))
    weights.update(MUZZLE_TARGETS.get(answers.get("muzzle", ""), {}))
    return weights


def _sheet_blocks(n: int) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """The model sheet for vote n (its own letter order) and its letters."""
    import base64

    from .model_sheet import DETAIL, describe, sheet

    path, letters = sheet(n)
    image = {"type": "image", "detail": DETAIL,
             "source": {"type": "base64", "media_type": "image/png",
                        "data": base64.standard_b64encode(path.read_bytes()).decode("ascii")}}
    return [image, {"type": "text", "text": describe(letters)}], letters


async def read_body_shape(transport: Transport, photo_blocks: list[dict[str, Any]],
                          samples: int = SAMPLES, models: bool = False) -> tuple[dict[str, str], dict[str, Any]]:
    """(answers, votes): each question's most-named seen answer, unknown when
    no sample saw it. With `models`, each vote also sees the model sheet in
    its own letter order and names the closest model (answers
    "closest_model", a template slug) and how well it matches
    ("model_match")."""
    ask = {"type": "text", "text": "Describe this dog's ears, tail, build, muzzle, coat length, leg length and "
                                   "head size." + (" Then say which grey model it looks like." if models else "")}

    async def one(n: int) -> tuple[dict[str, Any], dict[str, str]]:
        if not models:
            reading = await transport.complete(system=BODY_SHAPE_SYSTEM, blocks=[*photo_blocks, ask],
                                               schema=BODY_SHAPE_SCHEMA)
            return reading, {}
        sheet_blocks, letters = _sheet_blocks(n)
        reading = await transport.complete(system=MODELS_SYSTEM, blocks=[*photo_blocks, *sheet_blocks, ask],
                                           schema=MODELS_SCHEMA)
        return reading, letters

    readings: list[dict[str, Any]] = []
    errors: list[str] = []

    def keep(reading: dict[str, Any], letters: dict[str, str]) -> None:
        # the letter means a different model on each vote's sheet
        if models:
            diffs = sorted({d for d in reading.get("differences") or [] if d in DIFFERENCES})
            reading = {**reading, "closest_model": letters.get(str(reading.get("closest_model")), "unknown"),
                       "differences": diffs, "model_match": match_from(diffs)}
        readings.append(reading)

    payloads = await asyncio.gather(*(one(n) for n in range(samples)), return_exceptions=True)
    for payload in payloads:
        if isinstance(payload, AgentRefusal):
            raise payload
        if isinstance(payload, Exception):
            errors.append(f"{type(payload).__name__}: {str(payload)[:160]}")
            continue
        keep(*payload)
    # a failed sample (rate limit, dropped connection - 16 of 60 in one
    # eval pass) is retried one at a time; three missing samples made the
    # outline "unknown" and the fit verdict "could not see"
    for attempt in range(RETRIES):
        if len(readings) >= samples:
            break
        await asyncio.sleep(RETRY_WAIT_S * (attempt + 1))
        try:
            keep(*await one(len(readings) + attempt))
        except AgentRefusal:
            raise
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {str(exc)[:160]}")

    answers: dict[str, str] = {}
    votes: dict[str, Any] = {"samples": len(readings)}
    for key, allowed in (("ears", EAR_ANSWERS), ("tail_build", TAIL_BUILD_ANSWERS),
                         ("tail_carriage", TAIL_CARRIAGE_ANSWERS), ("build", BUILD_ANSWERS),
                         ("muzzle", MUZZLE_ANSWERS), ("coat_length", COAT_LENGTH_ANSWERS),
                         ("leg_length", LEG_LENGTH_ANSWERS), ("head_size", HEAD_SIZE_ANSWERS)):
        said = Counter(r.get(key) for r in readings if r.get(key) in allowed and r.get(key) != "unknown")
        answers[key] = said.most_common(1)[0][0] if said else "unknown"
        votes[key] = dict(said)
    if models:
        from .model_sheet import TEMPLATES
        said = Counter(r.get("closest_model") for r in readings if r.get("closest_model") in TEMPLATES)
        answers["closest_model"] = said.most_common(1)[0][0] if said else "unknown"
        votes["closest_model"] = dict(said)
        # the differences most of the votes for that model named
        backing = [r for r in readings if r.get("closest_model") == answers["closest_model"]]
        named = Counter(d for r in backing for d in r["differences"])
        answers["differences"] = sorted(d for d, n in named.items() if n * 2 > len(backing))
        answers["model_match"] = match_from(answers["differences"]) if backing else "unknown"
        votes["model_match"] = dict(Counter(r["model_match"] for r in readings))
        # each vote's own model and differences, so a split pick shows its doubt
        votes["by_model"] = [f"{r.get('closest_model')}:{'+'.join(r['differences']) or '-'}" for r in readings]
    votes["notes"] = [str(r.get("note", "")) for r in readings if r.get("note")]
    if errors:
        votes["errors"] = errors
    return answers, votes


__all__ = ["BODY_SHAPE_SCHEMA", "DIFFERENCES", "MODELS_SCHEMA", "POOR_FROM", "match_from", "read_body_shape",
           "shape_weights"]
