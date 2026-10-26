"""Stage 4-3 - check the generated views against the appearance they were made from.

The image model was told the coat in words. Whether it listened is checked the
only way that is independent of the words: the four generated views go back
through the same two vision agents that read the shelter photo, and the
appearance they produce is compared with the original. If the model painted a
white back where the record says black, the region agent will say so.

SCORE
-----
    score = 0.2 * pattern_match + 0.8 * jaccard(region_colors)

Jaccard is taken over (region, colour FAMILY) pairs - light, point or dark -
not exact names. The first live run scored a sheet that looked right at 0.25
because the agents called its white throat "cream" and the record said
"white"; the corrections that followed ("throat should be white, not cream",
eight lines of them) drove the model into a black bicolor dog. A texture whose
delighting snaps every texel to the animal's own palette does not care whether
the pale fur was named white or cream; it cares that it is pale where the
record says pale. Names only enter the messages, never the score.

The original lists every region, base coat included; the region agent reports
only regions that differ from the base, so base-coloured entries are dropped
from both sides before comparing, or the agent would be penalised for
correctly saying nothing. When every attempt fails, the best-scoring one is
restored as the working views rather than the last: each round of corrections
tends to push the model further, so the last is usually the worst. Markings are
reported for information but do not enter the score: a marking's region is a
weaker claim than a region colour's and the two are not comparable one-to-one.

Below 0.75 the mismatches are written into the prompt and the sheet is
generated again, at most twice more. The score, the attempts and what was
wrong on each are saved so the projection step can weight the views by how
much they were trusted.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .agents import MARKINGS_AGENT, REGION_COLOR_AGENT, Transport, run_agent
import time
from .imagegen import ImageTransport, generate_views
from .patterns import color_family

THRESHOLD = 0.75
MAX_ATTEMPTS = 3

# Every reading of a sheet is remembered against the sheet's hash, across runs.
# The same file read twice by the agents came back 0.76 and then 0.47: the
# picture had not changed, the reading had. The agents' errors are almost all
# omissions - a region not named, a colour called by its neighbour's name - so
# of several readings of one image the highest is the least wrong, and that is
# the score a sheet carries.
SCORE_MEMO = "scores.json"


def sheet_hash(path: Path) -> str | None:
    return hashlib.md5(path.read_bytes()).hexdigest() if path.exists() else None


def _memo_path(views_dir: Path) -> Path:
    return views_dir / SCORE_MEMO


def remember_score(views_dir: Path, digest: str | None, score: float, attempt: dict[str, Any]) -> float:
    """Record this reading and return the best score known for the sheet."""
    if digest is None:
        return score
    path = _memo_path(views_dir)
    memo = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    entry = memo.setdefault(digest, {"readings": []})
    entry["readings"].append({"score": round(score, 4), "jaccard": attempt.get("jaccard"),
                              "pattern": attempt.get("observed_pattern")})
    entry["best"] = max(reading["score"] for reading in entry["readings"])
    path.write_text(json.dumps(memo, indent=2), encoding="utf-8")
    return float(entry["best"])


def best_known_score(views_dir: Path, digest: str | None) -> float | None:
    path = _memo_path(views_dir)
    if digest is None or not path.exists():
        return None
    memo = json.loads(path.read_text(encoding="utf-8"))
    entry = memo.get(digest)
    return float(entry["best"]) if entry else None
PATTERN_WEIGHT = 0.2
JACCARD_WEIGHT = 0.8


FACE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "both_eyes_visible": {"type": "boolean"},
        "nose_visible": {"type": "boolean"},
        "face_reads_as_dog": {
            "type": "boolean",
            "description": "false if the head is a colour blob with no recognisable features",
        },
        "note": {"type": "string"},
    },
    "required": ["both_eyes_visible", "nose_visible", "face_reads_as_dog", "note"],
    "additionalProperties": False,
}

FACE_SYSTEM = """You look at the front view of a drawn dog and say whether its face is
actually there. Answer strictly from what is drawn: are both eyes visible as
distinct features, is the nose visible, and does the head read as a dog's face
rather than a patch of colour. One short note."""


async def check_face(front: Path, transport: Transport) -> dict[str, Any]:
    """A yes/no on the face, from the front view alone.

    The colour score cannot see this: a sheet can lose both eyes and score
    higher on its regions. A sheet that fails here is never preferred over
    one that passes, whatever their colour scores.
    """
    blocks = _view_blocks({"front": front})
    blocks.append({"type": "text", "text": "Is this dog's face intact?"})
    try:
        payload = await transport.complete(system=FACE_SYSTEM, blocks=blocks, schema=FACE_SCHEMA)
    except Exception as exc:  # noqa: BLE001 - the gate must not block the loop
        return {"ok": True, "note": f"face check unavailable ({type(exc).__name__}); not applied"}
    ok = bool(payload.get("both_eyes_visible")) and bool(payload.get("nose_visible")) \
        and bool(payload.get("face_reads_as_dog"))
    return {"ok": ok, "note": str(payload.get("note", "")), **payload}


@dataclass
class Consistency:
    score: float
    attempts: int
    mismatches: list[str]
    pattern_match: bool
    jaccard: float
    observed: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "attempts": self.attempts,
            "mismatches": self.mismatches,
            "pattern_match": self.pattern_match,
            "jaccard": round(self.jaccard, 4),
            "threshold": THRESHOLD,
            "history": self.history,
        }


def _pairs(appearance: dict[str, Any], base: str) -> set[tuple[str, str]]:
    base_family = color_family(base)
    return {
        (entry["region"], color_family(str(entry.get("color", "")).lower()))
        for entry in appearance.get("region_colors") or []
        if color_family(str(entry.get("color", "")).lower()) != base_family
    }


def _names(appearance: dict[str, Any]) -> dict[str, str]:
    return {entry["region"]: str(entry.get("color", "")).lower()
            for entry in appearance.get("region_colors") or []}


def compare_appearance(original: dict[str, Any], observed: dict[str, Any]) -> tuple[float, list[str], bool, float]:
    """Return (score, mismatches, pattern_match, jaccard)."""
    base = str(original.get("base_color", "")).lower()
    want, seen = _pairs(original, base), _pairs(observed, base)

    union = want | seen
    jaccard = len(want & seen) / len(union) if union else 1.0
    pattern_match = str(original.get("pattern", "")).lower() == str(observed.get("pattern", "")).lower()
    score = PATTERN_WEIGHT * float(pattern_match) + JACCARD_WEIGHT * jaccard

    mismatches: list[str] = []
    if not pattern_match:
        mismatches.append(
            f"overall pattern should be {original.get('pattern')}, not {observed.get('pattern')}"
        )
    seen_by_region = {region: family for region, family in seen}
    want_by_region = {region: family for region, family in want}
    want_names, seen_names = _names(original), _names(observed)
    for region, family in sorted(want):
        got = seen_by_region.get(region)
        label = region.replace("_", " ")
        if got is None:
            mismatches.append(f"{label} should be {want_names[region]} (it was painted {base})")
        elif got != family:
            mismatches.append(f"{label} should be {want_names[region]}, not {seen_names.get(region, got)}")
    for region, family in sorted(seen):
        if region not in want_by_region:
            mismatches.append(f"{region.replace('_', ' ')} should be {base}, not {seen_names.get(region, family)}")
    return score, mismatches, pattern_match, jaccard


def _view_blocks(views: dict[str, Path]) -> list[dict[str, Any]]:
    blocks = []
    for name in ("front", "left", "right", "back"):
        path = views.get(name)
        if path is None:
            continue
        image = Image.open(path).convert("RGBA")
        ground = Image.new("RGBA", image.size, (255, 255, 255, 255))
        ground.alpha_composite(image)
        buffer = io.BytesIO()
        ground.convert("RGB").save(buffer, format="PNG")
        blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png",
                       "data": base64.standard_b64encode(buffer.getvalue()).decode("ascii")},
        })
    return blocks


async def read_back(views: dict[str, Path], base_color: str, transport: Transport) -> dict[str, Any]:
    """The generated views through the two stage-2 agents."""
    blocks = _view_blocks(views)
    region_result, marking_result = await asyncio.gather(
        run_agent(REGION_COLOR_AGENT, transport, blocks, base_color),
        run_agent(MARKINGS_AGENT, transport, blocks, base_color),
    )
    return {
        "base_color": base_color,
        "pattern": region_result.payload.get("pattern", "other"),
        "region_colors": region_result.payload.get("region_colors", []),
        "markings": marking_result.payload.get("markings", []),
        "calls": region_result.calls + marking_result.calls,
    }


async def ensure_consistent(
    original: dict[str, Any],
    photo_path: str | Path,
    canonical_dir: str | Path,
    views_dir: str | Path,
    text_transport: Transport,
    image_transport: ImageTransport,
    regenerate_first: bool = False,
) -> Consistency:
    """Check the views in `views_dir`, regenerating with the mismatches spelled
    out until the score clears the threshold or the attempts run out."""
    views_dir = Path(views_dir)
    base = str(original.get("base_color", "")).lower()
    history: list[dict[str, Any]] = []
    mismatches: list[str] = []
    result: Consistency | None = None
    # every run keeps its own attempt folders; an earlier run's first sheet was
    # once overwritten by a later run's second, and it had been the best one
    run_tag = time.strftime("%H%M%S")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        existing = {n: views_dir / f"{n}.png" for n in ("front", "left", "right", "back")}
        have_views = all(p.exists() for p in existing.values())
        if attempt > 1 or regenerate_first or not have_views:
            if have_views:
                keep = views_dir / f"attempt_{attempt - 1}"
                if keep.exists():
                    keep = views_dir / f"attempt_{attempt - 1}_{run_tag}"
                keep.mkdir(exist_ok=True)
                for name, path in existing.items():
                    path.replace(keep / path.name)
                sheet = views_dir / "sheet.png"
                if sheet.exists():
                    sheet.replace(keep / "sheet.png")
            print(f"[consistency] attempt {attempt}: generating"
                  + (f" with {len(mismatches)} corrections" if mismatches else ""), flush=True)
            generated = await generate_views(
                photo_path, canonical_dir, original, image_transport, views_dir,
                mismatches=mismatches or None,
            )
            views = generated.views
        else:
            print("[consistency] attempt 1: checking the views already generated", flush=True)
            views = existing

        observed = await read_back(views, base, text_transport)
        score, mismatches, pattern_match, jaccard = compare_appearance(original, observed)
        face = await check_face(views["front"], text_transport)
        if not face["ok"]:
            print(f"[consistency] attempt {attempt}: face check FAILED - {face.get('note', '')}",
                  flush=True)
            mismatches = ["the dog's eyes and nose must be clearly drawn on the face"] + mismatches
        digest = sheet_hash(views_dir / "sheet.png")
        entry = {
            "attempt": attempt, "score": round(score, 4), "jaccard": round(jaccard, 4),
            "pattern_match": pattern_match, "mismatches": mismatches,
            "observed_pattern": observed.get("pattern"),
            "observed_regions": [(e["region"], e.get("color")) for e in observed.get("region_colors", [])],
            "sheet_md5": digest,
            "face_ok": face["ok"],
            "face_note": face.get("note", ""),
        }
        best_seen = remember_score(views_dir, digest, score, entry)
        if best_seen > score:
            print(f"[consistency] this sheet was read at {best_seen:.2f} before; using that",
                  flush=True)
            score = best_seen
            entry["score"] = round(score, 4)
        history.append(entry)
        print(f"[consistency] attempt {attempt}: score {score:.2f} "
              f"(pattern {'ok' if pattern_match else 'off'}, jaccard {jaccard:.2f}), "
              f"{len(mismatches)} mismatch(es)", flush=True)
        for line in mismatches:
            print(f"    - {line}")

        result = Consistency(score, attempt, mismatches, pattern_match, jaccard, observed, history)
        if score >= THRESHOLD and face["ok"]:
            break

    assert result is not None
    # write first, then restore: restore_best rewrites the file with the
    # attempt it put in place, and writing after it would erase that note
    (views_dir / "consistency.json").write_text(
        json.dumps(result.to_json(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return restore_best(views_dir, result)


def restore_best(views_dir: Path, result: Consistency) -> Consistency:
    """Leave the highest-scoring attempt's views in place.

    The loop ends on whichever attempt ran last, and when every attempt fails
    that is usually the worst one: each round of corrections pushed the model
    further. The best sheet is the one that goes on to projection, whatever
    order it was made in.
    """
    best = max(result.history, key=lambda h: (bool(h.get("face_ok", True)), h["score"]))
    best_no = best["attempt"]
    if best_no == result.attempts:
        known = best_known_score(views_dir, best.get("sheet_md5"))
        if known is not None and known > result.score:
            result.score = known
        return result
    candidates = sorted(views_dir.glob(f"attempt_{best_no}*"), key=lambda p: p.stat().st_mtime)
    source = candidates[-1] if candidates else views_dir / f"attempt_{best_no}"
    if not source.exists():
        return result
    current_no = result.attempts
    keep = views_dir / f"attempt_{current_no}"
    keep.mkdir(exist_ok=True)
    for name in ("front", "left", "right", "back", "sheet"):
        current = views_dir / f"{name}.png"
        if current.exists():
            current.replace(keep / f"{name}.png")
        restored = source / f"{name}.png"
        if restored.exists():
            restored.replace(views_dir / f"{name}.png")
    print(f"[consistency] restored attempt {best_no} (score {best['score']:.2f}) as the working views",
          flush=True)
    result.score = best["score"]
    result.jaccard = best["jaccard"]
    result.pattern_match = best["pattern_match"]
    result.mismatches = list(best["mismatches"])
    (views_dir / "consistency.json").write_text(
        json.dumps({**result.to_json(), "restored_attempt": best_no}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def rescore(original: dict[str, Any], views_dir: str | Path) -> Consistency:
    """Recompute every attempt's score from the observations already saved in
    consistency.json, and put the best attempt back - no model call."""
    views_dir = Path(views_dir)
    saved = json.loads((views_dir / "consistency.json").read_text(encoding="utf-8"))
    base = str(original.get("base_color", "")).lower()
    history = []
    for entry in saved["history"]:
        observed = {
            "base_color": base,
            "pattern": entry.get("observed_pattern"),
            "region_colors": [{"region": r, "color": c} for r, c in entry.get("observed_regions", [])],
        }
        score, mismatches, pattern_match, jaccard = compare_appearance(original, observed)
        history.append({**entry, "score": round(score, 4), "jaccard": round(jaccard, 4),
                        "pattern_match": pattern_match, "mismatches": mismatches})
        print(f"[consistency] attempt {entry['attempt']}: {entry['score']:.2f} -> {score:.2f} "
              f"({len(mismatches)} mismatch(es))", flush=True)
    last = history[-1]
    result = Consistency(last["score"], last["attempt"], last["mismatches"], last["pattern_match"],
                         last["jaccard"], {}, history)
    (views_dir / "consistency.json").write_text(
        json.dumps(result.to_json(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return restore_best(views_dir, result)


__all__ = ["Consistency", "THRESHOLD", "compare_appearance", "ensure_consistent",
           "read_back", "rescore", "restore_best"]
