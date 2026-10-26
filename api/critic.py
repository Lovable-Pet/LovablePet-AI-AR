"""Stage 3-3 - a critic looks at the rendered pet beside the photograph.

WHY
---
Everything before this judged the photograph. Nothing judged the result.
A region colour can be right and still render wrong - a mask larger than
the marking it stands for, a default that the photo never contradicted
because the region was out of frame and the default was wrong anyway.
Rendering the built asset and asking "which regions differ from the photo"
is the one check that sees what the person will see.

WHAT IT IS ALLOWED TO SAY
-------------------------
Only colours. The critic is told to ignore style, fur texture and lighting -
the render is a simplified model under studio light and the photo is a
shelter floor under fluorescents - and to compare the colour and the
placement of the coat's regions, naming only the ones that differ:

    {region, current, expected, kind}     kind = color | missing | extra

`color`: the region is the wrong colour, `expected` replaces it. `missing`:
the photo shows a second colour on this region that the render lacks,
`expected` is painted on it. `extra`: the render shows a colour the photo
does not have there; the region goes back to the base coat. Every answer is
a palette word, so a patch is always something the compositor can paint.

THE LOOP, AND WHEN IT STOPS
---------------------------
At most two rounds. After each, the patches are applied to region_colors
and the coat is composed and rendered again. Two patches on one region that
reverse each other - round one says tan, round two says black again - mean
the critic cannot decide and the loop is oscillating; the round-one result
is kept and the log says so. No patches ends the loop early. Every round's
patches, and whether the loop oscillated, go to the pipeline log.
"""

from __future__ import annotations

import base64
import io
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .agents import Transport
from .fur import FUR_COLORS
from .regions import REGION_DESCRIPTIONS, REGIONS

MAX_ROUNDS = 2
RENDER_SIDE = 900
# The critic is sampled like every other agent: a patch is applied only when
# a majority of the readings name the same region with the same expected
# colour family. One reading once repainted a whole leg.
SAMPLES = 2
# A round changes at most this many regions. Fewer, surer changes; the next
# round sees them and can go on.
MAX_PATCHES_PER_ROUND = 3

CRITIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "patches": {
            "type": "array",
            "description": "Only the regions whose colour or marking placement differs between the render and the photo.",
            "items": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "enum": list(REGIONS)},
                    "current": {
                        "type": "string",
                        "enum": [*FUR_COLORS, "none"],
                        "description": "The colour the render shows there, from the list given.",
                    },
                    "expected": {
                        "type": "string",
                        "enum": list(FUR_COLORS),
                        "description": "The colour the photo shows there.",
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["color", "missing", "extra"],
                        "description": "color: wrong colour; missing: the photo has this colour here and the render does not; extra: the render has a colour the photo does not.",
                    },
                },
                "required": ["region", "current", "expected", "kind"],
                "additionalProperties": False,
            },
        },
        "note": {"type": "string", "description": "One short sentence, or empty."},
    },
    "required": ["patches", "note"],
    "additionalProperties": False,
}

_REGION_LIST = "\n".join(f"- {region}: {REGION_DESCRIPTIONS[region]}" for region in REGIONS)

CRITIC_SYSTEM = f"""You compare a 3D render of a pet model with the photograph it was built from, and report only where the coat's colours differ.

You are shown the photograph, a full-body render of the model and a close-up
render of its face, and the list of region colours the model was painted
with. The render is a simplified model under studio light; the photograph is
a real animal in a shelter. IGNORE style, fur texture, shading, lighting,
pose, the background, the eyes and the nose. Compare ONLY, region by region,
which colour the coat is and where the second colour sits.

Report a region only when it is clearly different. For each one give the
colour the render currently shows (current), the colour the photograph shows
(expected) and the kind of difference:
- color: the region is painted the wrong colour;
- missing: the photograph has a colour on this region that the render lacks;
- extra: the render has a colour on this region that the photograph does not.

All colours are palette words: {", ".join(FUR_COLORS)}. Pick the nearest.
A region you cannot see in the photograph must not be reported. Returning an
empty list is a normal answer and means the render matches.

The regions and their boundaries:

{_REGION_LIST}"""


def _image_block(image: Image.Image, side: int = RENDER_SIDE) -> dict[str, Any]:
    if max(image.size) > side:
        scale = side / max(image.size)
        image = image.resize((max(1, int(image.size[0] * scale)), max(1, int(image.size[1] * scale))), Image.LANCZOS)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                        "data": base64.standard_b64encode(buffer.getvalue()).decode("ascii")}}


def render_views(glb_path: str | Path, out_dir: str | Path, blender: str, clip: str = "sitting") -> tuple[Path, Path]:
    """The renders the critic sees: the whole animal in the given clip from
    the front-left, the whole animal STANDING from the side and low (the
    belly, the flank and all four legs in view), and the face. Returns
    (body, face); the standing view sits beside them as render_side.png."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    body = out_dir / "render_body.png"
    side = out_dir / "render_side.png"
    face = out_dir / "render_face.png"
    for target, extra in ((body, ["--clip", clip, "--zoom", "2.2"]),
                          (side, ["--clip", "standing", "--zoom", "2.0", "--angle", "70", "--height", "0.05"]),
                          (face, ["--clip", clip, "--face"])):
        subprocess.run([blender, "--background", "--python", "ar_pipeline/render_preview.py", "--",
                        str(glb_path), str(target), *extra], check=True, capture_output=True)
    return body, face


def side_render_for(body: Path) -> Path | None:
    side = Path(body).parent / "render_side.png"
    return side if side.exists() else None


def _colours_text(appearance: dict[str, Any]) -> str:
    lines = []
    for entry in appearance.get("region_colors") or []:
        extra = ""
        if entry.get("gradient_to") not in (None, "none"):
            extra = f", fading to {entry['gradient_to']}"
        lines.append(f"- {entry['region']}: {entry['color']}{extra}")
    for marking in appearance.get("markings") or []:
        lines.append(f"- marking on {marking['region']}: {marking['color']} {marking.get('type')}")
    return "\n".join(lines) or "- (all base colour)"


async def critique(
    transport: Transport, photo_path: str | Path, body_render: Path, face_render: Path,
    appearance: dict[str, Any],
) -> dict[str, Any]:
    blocks = [
        {"type": "text", "text": "The photograph:"},
        _image_block(Image.open(photo_path)),
        {"type": "text", "text": "Full-body render of the model:"},
        _image_block(Image.open(body_render)),
        *([{"type": "text", "text": "The model standing, seen from the side and low:"},
           _image_block(Image.open(side_render_for(Path(body_render))))]
          if side_render_for(Path(body_render)) else []),
        {"type": "text", "text": "Face close-up render of the model:"},
        _image_block(Image.open(face_render)),
        {"type": "text", "text": (
            f"The model's base coat colour is '{appearance.get('base_color')}' and its regions "
            f"are painted:\n{_colours_text(appearance)}\n\n"
            "List only the regions whose colour or second-colour placement differs from the photograph."
        )},
    ]
    return await transport.complete(system=CRITIC_SYSTEM, blocks=blocks, schema=CRITIC_SCHEMA)


def _valid_patches(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out, seen = [], set()
    for item in payload.get("patches") or []:
        region = item.get("region")
        expected = str(item.get("expected", "")).lower()
        if region not in REGIONS or region in seen or expected not in FUR_COLORS:
            continue
        seen.add(region)
        out.append({"region": region, "current": str(item.get("current", "none")).lower(),
                    "expected": expected, "kind": item.get("kind", "color")})
    return out


def apply_patches(appearance: dict[str, Any], patches: list[dict[str, Any]], round_index: int) -> dict[str, Any]:
    """A new appearance with the patches painted into region_colors.

    Patches are deliberately light-handed. `color` swaps the colour and
    keeps the entry's coverage and gradient - the critic said the colour is
    wrong, not that the region's shape is - and where the old colour was
    the gradient's far end, that end follows. `missing` adds the expected
    colour as a partial fade from the region's existing colour rather than
    repainting the region. `extra` returns the region to the base coat.
    """
    base = appearance.get("base_color", "fawn")
    entries = {e["region"]: dict(e) for e in appearance.get("region_colors") or []}
    for patch in patches:
        region, expected, kind = patch["region"], patch["expected"], patch["kind"]
        entry = entries.get(region, {"region": region, "coverage": "full", "note": "",
                                     "gradient_to": "none", "gradient_anchor": "none"})
        old = entry.get("color")
        gradient_to = entry.get("gradient_to", "none") or "none"
        anchor = entry.get("gradient_anchor", "none") or "none"
        if kind == "extra":
            colour, coverage = base, entry.get("coverage", "full")
            if gradient_to == patch.get("current"):
                gradient_to, anchor = "none", "none"
        elif kind == "missing":
            colour, coverage = old or base, "partial"
            if gradient_to == "none":
                gradient_to = expected
        else:
            colour, coverage = expected, entry.get("coverage", "full")
            if gradient_to == expected:
                gradient_to, anchor = "none", "none"
        entry = {**entry, "color": colour, "coverage": coverage,
                 "source": f"critic:round{round_index}",
                 "note": f"critic {kind}: {patch['current']} -> {expected}",
                 "gradient_to": gradient_to, "gradient_anchor": anchor if gradient_to != "none" else "none"}
        entries[region] = entry
    region_colors = [entries[r] for r in REGIONS if r in entries]
    inferred = [r for r in appearance.get("inferred_regions") or [] if r not in {p["region"] for p in patches}]
    return {**appearance, "region_colors": region_colors, "inferred_regions": inferred}


def reverses(earlier: list[dict[str, Any]], later: list[dict[str, Any]]) -> list[str]:
    """Regions whose second patch undoes the first: the oscillation test."""
    first = {p["region"]: p for p in earlier}
    out = []
    for patch in later:
        previous = first.get(patch["region"])
        if previous is not None and patch["expected"] == previous["current"] and previous["current"] != "none":
            out.append(patch["region"])
    return out


@dataclass
class CriticOutcome:
    appearance: dict[str, Any]
    rounds: list[dict[str, Any]] = field(default_factory=list)
    oscillated: bool = False
    final_round: int = 0
    scores: dict[str, float] = field(default_factory=dict)

    def log(self) -> dict[str, Any]:
        return {
            "rounds": len(self.rounds),
            "final_round": self.final_round,
            "oscillated": self.oscillated,
            "mean_differences_per_build": self.scores,
            "history": self.rounds,
        }


# Two regions of one body part that a render does not separate reliably
# vote as one: they agree if they agree on the COLOUR, and the patch then
# covers the regions the readings named. Empty since the merge to ten
# regions (the ear halves and the tail halves it used to pair are now one
# region each); kept so a future split has somewhere to go.
VOTE_GROUPS: dict[str, str] = {}


async def sampled_critique(
    transport: Transport, photo_path: str | Path, body: Path, face: Path,
    appearance: dict[str, Any], samples: int = SAMPLES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """`samples` readings; a patch survives when a majority name the region
    with the same expected colour family. Returns (agreed patches, every
    reading's patches, the notes)."""
    import asyncio
    from collections import Counter

    from .patterns import color_family

    payloads = await asyncio.gather(*(critique(transport, photo_path, body, face, appearance)
                                      for _ in range(samples)))
    readings = [_valid_patches(p) for p in payloads]
    majority = samples // 2 + 1
    counts: Counter = Counter()
    first: dict[tuple[str, str], dict[str, Any]] = {}
    named: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for patches in readings:
        seen = set()
        for patch in patches:
            key = (VOTE_GROUPS.get(patch["region"], patch["region"]), color_family(patch["expected"]))
            named.setdefault(key, {}).setdefault(patch["region"], patch)
            if key in seen:
                continue
            seen.add(key)
            counts[key] += 1
            first.setdefault(key, patch)
    agreed = [p for key, count in counts.most_common() if count >= majority
              for p in named[key].values()]
    agreed = agreed[:MAX_PATCHES_PER_ROUND]
    notes = " / ".join(str(p.get("note", "")).strip() for p in payloads if p.get("note"))
    return agreed, readings, notes


async def critic_loop(
    transport: Transport,
    photo_path: str | Path,
    appearance: dict[str, Any],
    build: Callable[[dict[str, Any], int], Path],
    out_dir: str | Path,
    blender: str,
    max_rounds: int = MAX_ROUNDS,
    clip: str = "sitting",
) -> CriticOutcome:
    """`build(appearance, round_index)` composes and writes the GLB for an
    appearance and returns its path; round 0 is the build the loop starts
    from. Returns the appearance to keep and the log of what happened.

    Every built round is critiqued, including the last one, and the round
    kept is the one the critic found the fewest differences on (the latest
    on a tie). The last patch is therefore never taken on faith."""
    out_dir = Path(out_dir)
    builds: dict[int, dict[str, Any]] = {0: appearance}
    glb = build(appearance, 0)
    history: list[dict[str, Any]] = []
    scores: dict[int, int] = {}
    oscillated = False
    current = appearance
    round_index = 0

    while True:
        round_index += 1
        body, face = render_views(glb, out_dir / f"critic_round{round_index}", blender, clip)
        patches, readings, notes = await sampled_critique(transport, photo_path, body, face, current)
        built_round = round_index - 1
        scores[built_round] = sum(len(r) for r in readings) / max(len(readings), 1)
        record = {"round": round_index, "critiqued_build": built_round, "patches": patches,
                  "readings": readings, "mean_differences": round(scores[built_round], 2),
                  "note": notes, "renders": [str(body), str(face)]}

        if not patches:
            record["outcome"] = "no agreed differences; loop ends"
            history.append(record)
            break
        if history and history[-1].get("patches"):
            flipped = reverses(history[-1]["patches"], patches)
            if flipped:
                oscillated = True
                record["outcome"] = f"reverses round {round_index - 1} on {flipped}; oscillation, loop ends"
                record["reversed_regions"] = flipped
                history.append(record)
                break
        if round_index > max_rounds:
            record["outcome"] = "round limit; differences noted, not applied"
            history.append(record)
            break

        current = apply_patches(current, patches, round_index)
        glb = build(current, round_index)
        builds[round_index] = current
        record["outcome"] = f"applied {len(patches)} patch(es) and rebuilt"
        history.append(record)

    best = min(scores, key=lambda r: (scores[r], -r))
    return CriticOutcome(appearance=builds[best], rounds=history, oscillated=oscillated, final_round=best,
                         scores={str(k): round(v, 2) for k, v in scores.items()})


__all__ = ["CRITIC_SCHEMA", "CriticOutcome", "apply_patches", "critic_loop", "critique",
           "render_views", "reverses", "sampled_critique"]
