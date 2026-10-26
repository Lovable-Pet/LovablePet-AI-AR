"""Stage 2-1 - which coat pattern the photograph shows, judged against anchors.

WHY A CLASSIFIER BEFORE THE REGION AGENTS
-----------------------------------------
Asked for every region cold, a vision model answers the visible ones from
the photo and the hidden ones from habit, and the two are indistinguishable
in the output. Asked first "which of these five layouts is this dog", it
answers a question a photograph can actually settle, and the layout then
supplies a complete ten-region estimate (patterns.instantiate_defaults)
that the region agents only have to correct where the photo disagrees.

The anchors make the question concrete: five renders of the template itself
wearing each layout, so "tan_point" is not a word the model has to define
but a picture it can hold next to the photo.

THE VOTE
--------
Three samples are drawn in parallel (SAMPLES). The model's own confidence is never read;
agreement between independent readings is the confidence. When the two
disagree, three rules settle the cases that are not really disagreements:

  - saddle against tan_point: saddle. A saddle IS a tan-point coat whose tan
    has spread up the legs and head; one reading naming the general case
    and the other the specific one agree on the layout.
  - urajiro against tan_point: whichever the second colour says. The two
    layouts differ mainly in what the light regions are - cream/white
    (urajiro) or tan/red (tan points) - so the colour family of the second
    colour is the deciding observation.
  - anything else: a plain majority wins; with none, sample again, up to
    three rounds. Still split after three, the answer is "other": no layout
    is assumed and every region has to come from the photograph.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .agents import AgentRefusal, Transport
from .fur import FUR_COLORS
from .patterns import (
    PATTERN_NAMES,
    color_family,
    instantiate_defaults,
    overlay_for,
)

ANCHOR_PATTERNS = ("tan_point", "urajiro", "irish", "saddle", "solid")
SAMPLES = 3
MAX_ROUNDS = 3
ANCHOR_WIDTH = 512

CLASSIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "enum": list(PATTERN_NAMES),
            "description": "The anchor layout the photographed coat matches, or brindle/merle/other.",
        },
        "base_color": {
            "type": "string",
            "enum": list(FUR_COLORS),
            "description": "The colour covering most of the animal.",
        },
        "second_color": {
            "type": "string",
            "enum": [*FUR_COLORS, "none"],
            "description": "The colour of the pattern's marked regions; none for a solid coat.",
        },
        "eye_color": {
            "type": "string",
            "enum": ["dark_brown", "brown", "amber", "hazel", "blue", "gray", "unknown"],
            "description": "The iris colour, if an eye is visible; unknown otherwise.",
        },
        "white_legs": {
            "type": "string",
            "enum": ["none", "paws", "socks", "unknown"],
            "description": "How far white reaches up the legs: none (no white on the legs), paws (white on "
                           "the toes and pastern only, stopping BELOW the wrist and hock), socks (white "
                           "reaching the wrist or hock or climbing above them), unknown (legs not visible).",
        },
        "features": {
            "type": "array",
            "maxItems": 4,
            "description": "Up to 4 coat features CLEARLY visible in the photograph that the chosen "
                           "layout does not already imply; an empty list is a normal answer.",
            "items": {"type": "string", "enum": []},   # filled below from simple.FEATURES
        },
        "reason": {"type": "string", "description": "One short phrase naming what decided it."},
    },
    "required": ["pattern", "base_color", "second_color", "eye_color", "white_legs", "features", "reason"],
    "additionalProperties": False,
}


# The sock height is asked as its own single-choice question (white_legs)
# rather than as two features: offered "white_paws" and "white_socks" side
# by side, the samples split between them or dropped both, and the legs came
# out tan on a dog with white legs. One ordinal answer is a question a
# photograph settles.
LEG_FEATURES = {"none": (), "paws": ("white_paws",), "socks": ("white_socks",), "unknown": ()}
# The cheek and brow colours are asked in their own call over the head
# close-up (face_colours.py): asked here, beside the anchors, they came back
# as whatever the layout implies. These map that call's answers onto
# features; "tan" and "dark" add nothing the layout does not already say.
CHEEK_FEATURES = {"white": ("white_cheeks",), "cream": ("cream_cheeks",), "tan": (), "dark": (), "unknown": ()}
BROW_FEATURES = {"none": ("no_brows",), "white": ("white_brows",), "cream": ("cream_brows",), "tan": ("tan_brows",),
                 "unknown": ()}
# "dark" and "tan" are what the layouts already put there (a dark nose bar on
# a tan-point, the base colour elsewhere); only a light blaze is news
MUZZLE_FEATURES = {"dark": ("dark_muzzle",), "white": ("white_bridge",), "cream": ("cream_bridge",),
                   # the saddle layout puts a dark bar there; a tan answer removes it
                   "tan": ("tan_bridge",),
                   "unknown": ()}
# the inside of the ears: every answer is news, because the layouts only ever
# put the points colour (tan-point, urajiro) or the base colour there
EAR_FEATURES = {"dark": ("dark_ears",), "tan": ("tan_ears",), "cream": ("cream_ears",), "white": ("white_ears",),
                "pink": ("pink_ears",), "unknown": ()}


def _feature_vocabulary() -> dict[str, str]:
    """simple.FEATURES without importing simple at module load (it imports
    this module's neighbours); the enum is filled in on first use. The two
    sock features are left out: white_legs asks that."""
    from .simple import FEATURES
    asked_elsewhere = {f for names in (*LEG_FEATURES.values(), *CHEEK_FEATURES.values(), *BROW_FEATURES.values(),
                                       *MUZZLE_FEATURES.values(), *EAR_FEATURES.values()) for f in names}
    # the face call answers the cheeks and the muzzle top, whatever they are;
    # gpt-4.1-mini listed white_muzzle here for a dog with a black nose bar
    # and white muzzle SIDES, and painted a white stripe down the bridge
    asked_elsewhere.update({"tan_cheeks", "white_muzzle"})
    return {name: text for name, (text, _) in FEATURES.items() if name not in asked_elsewhere}


def _classify_schema() -> dict[str, Any]:
    vocabulary = _feature_vocabulary()
    schema = json.loads(json.dumps(CLASSIFY_SCHEMA))
    schema["properties"]["features"]["items"]["enum"] = list(vocabulary)
    return schema


def _classify_system() -> str:
    vocabulary = _feature_vocabulary()
    listing = "\n".join(f"- {name}: {text}" for name, text in vocabulary.items())
    return CLASSIFY_SYSTEM + f"""

Then say how far white reaches up the legs (white_legs). Look at the
wrist (front legs) and the hock (hind legs): white that stops below them is
"paws"; white that reaches them or climbs higher is "socks"; no white on
the legs is "none"; legs not visible is "unknown".

Finally, name the few coat features that stand out in the photograph and
that the layout you chose does NOT already imply - the things a person
would mention to pick this dog out of a line-up. Choose from this fixed
list only, up to 4, and only what is CLEARLY visible; one or two is a
normal answer and an empty list is fine. Do not list a feature because dogs
of this kind usually have it.

{listing}"""

CLASSIFY_SYSTEM = f"""You decide which coat layout a photographed dog has, by comparing the photograph with five reference renders.

The five renders show the SAME 3D dog model wearing five layouts. Their
colours are only there to make the layout visible - ignore breed, pose, fur
style and lighting, and compare only WHERE the second colour sits:

- tan_point: a dark coat with points of a warmer colour above the eyes, on
  the cheeks, throat, chest, lower legs and paws.
- urajiro: a coat whose cheeks, throat, chest, belly and inner legs are
  cream or white, with the back and head in the base colour.
- irish: a white collar, chest, belly, paws and tail tip on a coloured coat.
- saddle: a light body with a dark blanket over the back and tail top, and
  usually a dark muzzle.
- solid: one colour all over.

Answer brindle for a striped coat and merle for a mottled/marbled one,
whatever their layout. Answer sable for a warm coat (tan, red, fawn) whose
hairs are tipped darker, darkest along the back, the neck, the top of the
head and the tail, shading gradually into the warm base with NO clear edge;
a saddle is a solid dark blanket with a clean edge. For sable, base_color
is the warm colour and second_color the colour of the dark tips. Sable
needs a visibly GREY-BLACK overlay over the back and neck. A red, tan or
fawn coat that is one colour, only darker in shadow or on the ear tips, or
has a dark muzzle, is solid (with dark_muzzle), not sable. Compare the top
of the back with the lower flank in the same light: sable only when the
back is plainly greyer or blacker than the flank; a deep red or fawn back
that is merely a richer shade of the flank's colour is solid. A black mask
on the face never makes a dog sable on its own.

Answer piebald for a WHITE dog whose head and ears are coloured (a cap
over the skull and ears, the muzzle often white), with or without coloured
patches on the body; base_color is white and second_color the cap's
colour. Name the body patches with the body_patches feature. The cap
must be a strong colour (black, brown, red, orange) covering most of the
skull; a white dog with faintly cream ears or shading is solid. Answer other when none of these describes the coat -
random patches, piebald spotting, or a coat you cannot see enough of.

Choose base_color as the colour covering most of the animal, and
second_color as the colour of the pattern's marked regions (none for solid,
and the stripe colour for brindle), each from this fixed palette:
{", ".join(FUR_COLORS)}. Pick the nearest palette word; "pink" means bare
skin showing through thin hair, never fur. The shelter record's base colour
is given as a hint; report what you see.

Also give eye_color: the colour of the iris as seen in the photograph -
dark_brown (nearly black), brown, amber (orange-gold), hazel, blue or gray -
or unknown when no eye is clearly visible."""


def _image_block(image: Image.Image) -> dict[str, Any]:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.standard_b64encode(buffer.getvalue()).decode("ascii"),
        },
    }


def anchor_blocks(anchors_dir: str | Path) -> list[dict[str, Any]]:
    """The five anchors, each preceded by its name."""
    anchors_dir = Path(anchors_dir)
    blocks: list[dict[str, Any]] = []
    for pattern in ANCHOR_PATTERNS:
        path = anchors_dir / f"{pattern}.png"
        if not path.exists():
            raise FileNotFoundError(f"anchor missing: {path} (run ar_pipeline/render_anchors.py)")
        image = Image.open(path).convert("RGB")
        if image.size[0] > ANCHOR_WIDTH:
            height = int(image.size[1] * ANCHOR_WIDTH / image.size[0])
            image = image.resize((ANCHOR_WIDTH, height), Image.LANCZOS)
        blocks.append({"type": "text", "text": f"Reference render, layout = {pattern}:"})
        # low detail: one 512px tile. The anchor shows where the second colour
        # sits on the body, which survives that; at "auto" the five of them
        # were most of the call's image tokens, three samples per animal.
        blocks.append({**_image_block(image), "detail": "low"})
    return blocks


@dataclass
class PatternDecision:
    pattern: str
    base_color: str
    second_color: str | None
    rounds: int
    samples: list[dict[str, Any]] = field(default_factory=list)
    resolution: str = ""
    overlay: str | None = None
    eye_color: str | None = None
    features: list[str] = field(default_factory=list)
    feature_votes: dict[str, float] = field(default_factory=dict)
    white_legs: str = "unknown"
    cheeks: str = "unknown"
    brows: str = "unknown"

    @property
    def region_colors(self) -> list[dict[str, Any]]:
        return instantiate_defaults(self.pattern, self.base_color, self.second_color)

    def as_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "base_color": self.base_color,
            "second_color": self.second_color,
            "overlay": self.overlay,
            "eye_color": self.eye_color,
            "rounds": self.rounds,
            "resolution": self.resolution,
            "features": self.features,
            "feature_votes": self.feature_votes,
            "white_legs": self.white_legs,
            "cheeks": self.cheeks,
            "brows": self.brows,
            "samples": self.samples,
        }


# What a dog's eyes are when the readings do not agree. Not a guess about
# this animal: the commonest iris in dogs, used so an unresolved vote cannot
# put a pale eye on a model.
DEFAULT_EYE_COLOR = "dark_brown"


def _majority(values: list[str], tie_break: str | None) -> str | None:
    counted = Counter(v for v in values if v)
    if not counted:
        return tie_break
    top = counted.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return tie_break if tie_break in counted else top[0][0]
    return top[0][0]


def resolve_patterns(named: list[str], seconds: list[str]) -> tuple[str | None, str]:
    """Apply the disagreement rules to one round's readings. Returns the
    pattern (None when unresolved) and a note saying which rule fired."""
    distinct = set(named)
    if len(distinct) == 1:
        return named[0], "agreed"
    # the two named splits are settled by their rule before any head count:
    # the second colour is a better witness than one more vote
    if distinct == {"saddle", "tan_point"}:
        return "saddle", "saddle vs tan_point: saddle (a tan-point coat with the tan spread)"
    if distinct == {"urajiro", "tan_point"}:
        families = {color_family(second) for second in seconds if second and second != "none"}
        if families == {"light"}:
            return "urajiro", "urajiro vs tan_point: second colour is light, so urajiro"
        if families == {"point"}:
            return "tan_point", "urajiro vs tan_point: second colour is a point colour, so tan_point"
    counts = Counter(named)
    top, votes = counts.most_common(1)[0]
    if votes > len(named) / 2:
        return top, f"majority {votes}/{len(named)}"
    if distinct == {"urajiro", "tan_point"}:
        return None, "urajiro vs tan_point: second colours disagree too"
    return None, f"split {sorted(distinct)}"


async def classify_pattern(
    transport: Transport,
    photo_blocks: list[dict[str, Any]],
    anchors_dir: str | Path,
    record_base_color: str,
    samples: int = SAMPLES,
    max_rounds: int = MAX_ROUNDS,
) -> PatternDecision:
    """Vote the pattern across `samples` readings, re-sampling on a split."""
    blocks = [
        *anchor_blocks(anchors_dir),
        {"type": "text", "text": "The photograph(s) to classify:"},
        *photo_blocks,
        {"type": "text", "text": (
            f"The shelter record gives the base coat colour as '{record_base_color}'. "
            "Which layout does this dog wear, and in which two colours?"
        )},
    ]

    system, schema = _classify_system(), _classify_schema()

    async def one() -> dict[str, Any]:
        return await transport.complete(system=system, blocks=blocks, schema=schema)

    all_samples: list[dict[str, Any]] = []
    resolution = ""
    pattern: str | None = None
    rounds = 0
    while rounds < max_rounds and pattern is None:
        rounds += 1
        readings = []
        errors: list[Exception] = []
        pending = samples
        for _attempt in range(3):
            if pending <= 0:
                break
            payloads = await asyncio.gather(*(one() for _ in range(pending)), return_exceptions=True)
            pending = 0
            for payload in payloads:
                if isinstance(payload, AgentRefusal):
                    raise payload
                if isinstance(payload, Exception):
                    errors.append(payload)
                    pending += 1     # retried
                    continue
                if payload.get("pattern") not in PATTERN_NAMES:
                    continue
                readings.append(payload)
        if not readings:
            first = errors[0] if errors else None
            raise RuntimeError(f"every pattern sample failed: {type(first).__name__}: {first}") from first
        all_samples.extend({**reading, "round": rounds} for reading in readings)
        pattern, resolution = resolve_patterns(
            [reading["pattern"] for reading in readings],
            [reading.get("second_color", "none") for reading in readings],
        )

    if pattern is None:
        pattern = "other"
        resolution = f"{resolution}; unresolved after {rounds} rounds, so other"

    agreeing = [s for s in all_samples if s["pattern"] == pattern] or all_samples
    base = _majority([s.get("base_color", "") for s in agreeing], record_base_color) or record_base_color
    if base not in FUR_COLORS:
        base = record_base_color
    second = _majority([s.get("second_color", "") for s in agreeing], None)
    if second in (None, "none") or second not in FUR_COLORS or pattern == "solid":
        second = None

    # Most dogs have dark brown eyes, so a split or silent vote takes that
    # rather than whichever sample answered first: a wrong pale iris is far
    # more visible on the model than a dark one.
    said = [s.get("eye_color", "") for s in all_samples if s.get("eye_color") not in (None, "", "unknown")]
    counted = Counter(said)
    top = counted.most_common()
    if not top or (len(top) > 1 and top[0][1] == top[1][1]):
        eyes = DEFAULT_EYE_COLOR
    else:
        eyes = top[0][0]

    # features: a majority of the agreeing samples, most-named first, at
    # most four. A sample that answered no features (or an older script
    # without the field) simply votes for none.
    vocabulary = _feature_vocabulary()
    named = Counter(f for s in agreeing for f in set(s.get("features") or []) if f in vocabulary)
    n = max(len(agreeing), 1)
    feature_votes = {name: round(count / n, 3) for name, count in named.most_common()}
    features = [name for name, count in named.most_common() if count > n / 2][:4]

    # the sock height: the most-named answer among those that saw the legs;
    # a tie between two heights takes the lower one
    legs_said = Counter(s.get("white_legs") for s in agreeing if s.get("white_legs") in ("none", "paws", "socks"))
    order = ("none", "paws", "socks")
    if legs_said:
        best = max(legs_said.values())
        white_legs = min((k for k, v in legs_said.items() if v == best), key=order.index)
    else:
        white_legs = "unknown"
    for name in LEG_FEATURES[white_legs]:
        if name not in features:
            features.append(name)

    return PatternDecision(
        pattern=pattern,
        base_color=base,
        second_color=second,
        rounds=rounds,
        samples=all_samples,
        resolution=resolution,
        overlay=overlay_for(pattern),
        eye_color=eyes,
        features=features,
        feature_votes=feature_votes,
        white_legs=white_legs,
    )


def describe(decision: PatternDecision) -> str:
    return (
        f"{decision.pattern} (base {decision.base_color}, second {decision.second_color or '-'}), "
        f"{len(decision.samples)} samples in {decision.rounds} round(s): {decision.resolution}"
        + (f"; overlay {decision.overlay}" if decision.overlay else "")
    )


__all__ = ["ANCHOR_PATTERNS", "PatternDecision", "anchor_blocks", "classify_pattern",
           "describe", "resolve_patterns"]
