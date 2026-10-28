"""Stage 0: the photograph made fit for the reading.

Shelter photographs are not studio shots. A hand holds the dog, another dog
sits behind it, the intake date is printed across a corner, a harness covers
the chest. Downstream every stage - the region agents, the image model's
reference, the critic - takes the photograph at face value, so what is not
the dog has to leave before any of them see it.

A known shelter logo stamped over the photograph is painted out first
(watermark.py), so that the gate does not refuse it as text on the animal.
Then two parts, in this order:

1. GATE (one small vision call, two samples). The vision model answers a
   short factual form about the whole photograph: what the subject is, how
   many animals, whether a person or an object covers part of the animal,
   whether text or a watermark lies over the animal, whether it wears
   clothing (a rejection: a garment hides the torso coat) or only a
   harness, which side is shown. Nothing here is a
   self-reported confidence; the rejection rules are fixed below, and a
   rejection needs both samples to agree on it (a photograph is never
   refused on one reading). Conditions that do not stop the run - a
   harness, a back view - become NOTES the reading and repaint prompts
   carry ("ignore the harness; paint fur only").

2. CUT (local, no call). rembg lifts the foreground. The gate also named
   the grid cell the animal's body sits in, so of rembg's pieces the one
   under that cell is the animal; the others (a person standing beside it,
   a second dog, a stray blob) go, except small fragments touching the
   animal's box, which may be its own detached tail or ear. Without a gate
   the largest piece is taken. The rest of the frame is filled with a
   neutral grey and the result is cropped to the animal with a margin. That file replaces the photograph for every later stage. What
   rembg cannot separate - an arm across the dog's back, text printed over
   the fur - is exactly what the gate refuses.

Cage bars or a wire fence between the camera and the dog are not refused:
the coat, the outline and the looks show between the bars as they do to a
person. What they break is the cut - the bars slice rembg's mask into
strips and the anchor keeps one (synthetic bars on e14, 8 px every 45: one
strip, 1.3 % of the frame, head and legs gone). So when the pieces chain
across bar-wide gaps they are taken together, the gaps are kept as photo
(the bars stay visible, not painted grey like fur), and a note tells the
reading and repaint prompts to look through them. The colour measurement
samples pixels, so the lite run skips it then. Only a mesh so dense that
little coat shows is refused, as object_covers.

Measured on tests/fixtures/image.png with a date string, a red caption and a
skin-coloured block composited in: the caption and the block outside the
dog vanish completely; the part of the caption that crossed the dog stays
(9 % of its band), which is why "text over the animal" is a gate rule and
not a cut rule. On tests/fixtures/images.jpg (white shiba, a person's leg at
the frame's edge) rembg keeps the leg as its own piece, larger than a tenth
of the dog - a size rule cannot tell them apart, the anchor cell can.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from .agents import Transport

# The whole photograph is shown to the gate at most this wide.
GATE_SIDE = 768
# Margin around the animal's box, as a share of the box.
CROP_MARGIN = 0.15
# Fill outside the cut-out.
BACKDROP = (128, 128, 128)
# Below this share of the frame rembg found nothing worth trusting; the
# photograph is used whole.
MIN_COVERAGE = 0.01
GATE_SAMPLES = 2

GATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "subject": {"type": "string", "enum": ["dog", "cat", "other_animal", "none"]},
        "animals": {"type": "integer", "minimum": 0},
        "person_covers_animal": {"type": "boolean"},
        "object_covers_animal": {"type": "boolean"},
        "bars_in_front": {"type": "boolean"},
        "text_on_animal": {"type": "boolean"},
        "clothing": {"type": "boolean"},
        "harness": {"type": "boolean"},
        "view": {"type": "string", "enum": ["front", "side", "back", "top", "mixed"]},
        "animal_cell": {
            "type": "object",
            "description": "The grid cell at the centre of the main animal's body (torso, not the head).",
            "properties": {"col": {"type": "string"}, "row": {"type": "integer", "minimum": 1}},
            "required": ["col", "row"],
        },
        "note": {"type": "string"},
    },
    "required": ["subject", "animals", "person_covers_animal", "object_covers_animal", "bars_in_front",
                 "text_on_animal", "clothing", "harness", "view", "animal_cell"],
}

GATE_SYSTEM = """You check whether a shelter photograph can be used to read one animal's coat colours. Answer the form about the WHOLE photograph, factually.

- subject: the main animal in the photograph.
- animals: how many animals are visible, in total (a reflection or a picture on the wall does not count).
- person_covers_animal: true when a person's hand, arm, leg or body hides part of the main animal's coat (a person standing beside it, not touching, is false).
- object_covers_animal: true when a blanket, bowl, leash loop, a mesh so dense that little coat shows through, or another object hides a large part of the coat (a thin leash line is false; cage bars or a fence with the coat visible between them are false - answer them in bars_in_front).
- bars_in_front: true when cage bars, a wire fence or a mesh stands between the camera and the animal, crossing its body (a fence behind the animal is false).
- text_on_animal: true when printed text, a date stamp, a logo or a watermark lies over the animal's body (text over the background only is false).
- clothing: true when the animal wears clothing - a shirt, sweater, coat, dress or any garment with sleeves or a body panel covering the torso (a harness, bandana or collar is not clothing).
- harness: true when the animal wears a harness, a bandana or a wide collar, without clothing.
- view: which side of the animal the photograph mostly shows.
- animal_cell: the photograph has a lettered grid drawn over it (columns A.. left to right, rows 1.. top to bottom, each cell labelled in its corner). Name the cell at the centre of the main animal's BODY (its torso, not its head or legs).
- note: one short sentence, only when something else would mislead a colour reading (heavy shadow, colour cast, motion blur).

Do not judge quality or give a confidence. Answer with the form only."""

# Conditions that stop the run, with what to tell the shelter.
REJECT_RULES: dict[str, str] = {
    "not_a_dog": "사진의 주 피사체가 개가 아닙니다.",
    "several_animals": "동물이 둘 이상 보입니다. 이 개만 나온 사진이 필요합니다.",
    "person_covers": "사람의 손·팔·몸이 개의 털을 가리고 있습니다. 가리지 않은 사진이 필요합니다.",
    "object_covers": "물체가 개의 털을 크게 가리고 있습니다.",
    "text_on_animal": "글자·도장·워터마크가 개의 몸 위에 있습니다. 글자 없는 원본이 필요합니다.",
    "clothing": "개가 옷을 입고 있습니다. 옷을 벗은 사진이 필요합니다.",
}


class PhotoRejected(RuntimeError):
    def __init__(self, reasons: list[str], readings: list[dict[str, Any]]) -> None:
        self.reasons = reasons
        self.readings = readings
        super().__init__("; ".join(REJECT_RULES[r] for r in reasons))


@dataclass
class PreparedPhoto:
    path: Path                       # the file every later stage reads
    original: Path
    cut: bool                        # rembg cut applied (else the photo whole)
    box: tuple[int, int, int, int]   # crop box in original pixels
    coverage: float                  # share of the original frame that is animal
    pieces_dropped: int
    rejected: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    readings: list[dict[str, Any]] = field(default_factory=list)
    quality: dict[str, Any] | None = None   # photo_quality.measure_quality on the cut-out
    bars: bool = False               # seen through cage bars or a fence (pieces bridged)
    subject_box: tuple[int, int, int, int] | None = None   # the animal in `path`'s pixels (None: not cut)
    watermarks: list[str] = field(default_factory=list)   # shelter logos painted out (watermark.py)

    def report(self) -> dict[str, Any]:
        return {
            "path": str(self.path), "original": str(self.original), "cut": self.cut,
            "box": list(self.box), "coverage": round(self.coverage, 3), "pieces_dropped": self.pieces_dropped,
            "bars": self.bars, "subject_box": list(self.subject_box) if self.subject_box else None,
            "watermarks": self.watermarks,
            "rejected": self.rejected, "notes": self.notes, "gate": self.readings, "quality": self.quality,
        }


def _image_block(image: Image.Image) -> dict[str, Any]:
    import base64

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                        "data": base64.standard_b64encode(buffer.getvalue()).decode("ascii")}}


def _shrink(image: Image.Image, side: int) -> Image.Image:
    if max(image.size) <= side:
        return image
    scale = side / max(image.size)
    return image.resize((max(1, int(image.size[0] * scale)), max(1, int(image.size[1] * scale))), Image.LANCZOS)


# --------------------------------------------------------------------------- gate

def reject_reasons(reading: dict[str, Any], species: str = "dog") -> list[str]:
    reasons = []
    if reading.get("subject") != species:
        reasons.append("not_a_dog")
    if int(reading.get("animals") or 0) > 1:
        reasons.append("several_animals")
    if reading.get("person_covers_animal"):
        reasons.append("person_covers")
    if reading.get("object_covers_animal"):
        reasons.append("object_covers")
    if reading.get("text_on_animal"):
        reasons.append("text_on_animal")
    if reading.get("clothing"):
        reasons.append("clothing")
    return reasons


BARS_NOTE = ("The animal is photographed through cage bars or a fence: ignore the bars entirely and treat "
             "the coat behind each bar as continuing the coat on either side of it.")


def gate_notes(readings: list[dict[str, Any]]) -> list[str]:
    """Prompt lines for conditions EVERY sample reported that do not stop the
    run. A note goes into the reading and repaint prompts, so one sample
    mistaking a collar for a harness must not put "ignore the harness" in
    front of the image model."""
    notes = []
    if readings and all(r.get("harness") for r in readings):
        notes.append("The animal in the photograph wears a harness, bandana or collar: ignore it entirely and "
                     "treat the fur beneath as continuing the neighbouring coat.")
    if readings and all(r.get("bars_in_front") for r in readings):
        notes.append(BARS_NOTE)
    views = [r.get("view") for r in readings]
    if views and all(v == "back" for v in views):
        notes.append("The photograph shows the animal from behind; the face and chest are not visible.")
    if views and all(v == "top" for v in views):
        notes.append("The photograph looks down on the animal; the underside and lower legs are not visible.")
    if readings and all(str(r.get("note", "")).strip() for r in readings):
        notes.append(str(readings[0]["note"]).strip())
    return notes


def gate_anchors(readings: list[dict[str, Any]], size: tuple[int, int]) -> list[tuple[float, float]]:
    """The animal-cell centres of every reading, in the original's pixels."""
    from .face_photo import cell_to_pixel

    anchors = []
    for r in readings:
        cell = r.get("animal_cell") or {}
        try:
            anchors.append(cell_to_pixel(str(cell.get("col", "")), int(cell.get("row", 1)), "center", size))
        except (TypeError, ValueError):
            continue
    return anchors


async def gate_photo(transport: Transport, image: Image.Image, species: str = "dog",
                     samples: int = GATE_SAMPLES) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Returns (agreed rejection reasons, notes, the raw readings). A reason
    counts only when every sample gave it."""
    from .face_photo import draw_grid

    shown = draw_grid(_shrink(image, GATE_SIDE))
    blocks = [_image_block(shown), {"type": "text", "text": "Fill in the form for this photograph."}]
    readings: list[dict[str, Any]] = []
    for _ in range(samples):
        try:
            readings.append(await transport.complete(system=GATE_SYSTEM, blocks=blocks, schema=GATE_SCHEMA))
        except Exception as exc:  # noqa: BLE001 - one failed sample must not refuse a photo
            print(f"[prepare] gate sample failed ({type(exc).__name__}: {exc})")
    if not readings:
        return [], [], []
    per_sample = [set(reject_reasons(r, species)) for r in readings]
    agreed = set.intersection(*per_sample) if per_sample else set()
    reasons = [r for r in REJECT_RULES if r in agreed]
    return reasons, gate_notes(readings), readings


# --------------------------------------------------------------------------- cut

# A fragment next to the chosen piece survives when it is at most this share
# of it and touches the piece's box grown by FRAGMENT_REACH.
FRAGMENT_SHARE = 0.25
FRAGMENT_REACH = 0.05


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(mask)
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def choose_pieces(labels: np.ndarray, sizes: np.ndarray, anchors: list[tuple[float, float]]) -> tuple[list[int], int]:
    """Which labelled pieces are the animal. With anchors, the piece under
    (or nearest) the most anchors; without, the largest. Small fragments
    touching its grown box come along. Returns (labels kept, chosen label)."""
    from scipy import ndimage

    count = len(sizes)
    if anchors:
        votes: dict[int, int] = {}
        for x, y in anchors:
            xi, yi = int(np.clip(round(x), 0, labels.shape[1] - 1)), int(np.clip(round(y), 0, labels.shape[0] - 1))
            hit = int(labels[yi, xi])
            if hit == 0:
                # nearest piece to the anchor
                distance = ndimage.distance_transform_edt(labels == 0, return_indices=True)
                indices = distance[1]
                hit = int(labels[indices[0][yi, xi], indices[1][yi, xi]])
            votes[hit] = votes.get(hit, 0) + 1
        chosen = max(votes, key=lambda k: (votes[k], sizes[k - 1]))
    else:
        chosen = int(np.argmax(sizes)) + 1
    x0, y0, x1, y1 = _bbox(labels == chosen)
    rx, ry = (x1 - x0) * FRAGMENT_REACH, (y1 - y0) * FRAGMENT_REACH
    grown = (x0 - rx, y0 - ry, x1 + rx, y1 + ry)
    kept = [chosen]
    for index in range(1, count + 1):
        if index == chosen or sizes[index - 1] > FRAGMENT_SHARE * sizes[chosen - 1]:
            continue
        fx0, fy0, fx1, fy1 = _bbox(labels == index)
        if fx0 < grown[2] and fx1 > grown[0] and fy0 < grown[3] and fy1 > grown[1]:
            kept.append(index)
    return kept, chosen


# Bars slice the mask into strips with bar-wide gaps between them. Pieces
# chained to the chosen one across gaps up to this share of the frame's long
# side (e14 at 650 px: 19 px) are one animal behind bars when at least
# BAR_STRIPS of them are large (each above FRAGMENT_SHARE of the chosen):
# a person's leg beside the dog is one large piece, not a chain.
BAR_GAP = 0.03
BAR_STRIPS = 3


def bridge_bars(solid: np.ndarray, labels: np.ndarray, sizes: np.ndarray, chosen: int,
                hint: bool = False) -> np.ndarray | None:
    """The animal's mask with the bars' gaps filled in, or None when the
    pieces do not look cut by bars. With `hint` (the gate saw bars) two
    chained pieces are enough."""
    from scipy import ndimage

    k = max(3, int(round(BAR_GAP * max(solid.shape)))) | 1   # odd: max then min is a closing
    grown = ndimage.maximum_filter(solid.astype(np.uint8), size=k)
    closed = ndimage.minimum_filter(grown, size=k).astype(bool) | solid
    groups, _ = ndimage.label(closed)
    region = groups == int(groups[labels == chosen][0])
    members = [int(m) for m in np.unique(labels[region]) if m > 0]
    large = [m for m in members if sizes[m - 1] > FRAGMENT_SHARE * sizes[chosen - 1]]
    if len(large) >= BAR_STRIPS or (hint and len(members) > 1):
        return region
    return None


def cut_out(image: Image.Image, anchors: list[tuple[float, float]] | None = None,
            ) -> tuple[Image.Image | None, tuple[int, int, int, int], float, int]:
    """The animal on a grey backdrop, cropped with a margin. Returns
    (image or None when rembg found nothing, box, coverage, pieces dropped)."""
    cut, box, coverage, dropped, _, _ = cut_out_with_mask(image, anchors)
    return cut, box, coverage, dropped


def cut_out_with_mask(image: Image.Image, anchors: list[tuple[float, float]] | None = None,
                      bars_hint: bool = False,
                      ) -> tuple[Image.Image | None, tuple[int, int, int, int], float, int, np.ndarray | None, bool]:
    """cut_out plus the animal's mask in the cut-out's own pixels (None
    when nothing was cut) for the quality measurement, and whether the
    pieces were bridged across bars (bridge_bars; `bars_hint`: the gate
    saw bars)."""
    from scipy import ndimage

    from .imagegen import _rembg_alpha

    rgb = image.convert("RGB")
    width, height = rgb.size
    alpha = _rembg_alpha(rgb)
    if alpha is None:
        return None, (0, 0, width, height), 0.0, 0, None, False
    solid = alpha > 0.5
    if solid.mean() < MIN_COVERAGE:
        return None, (0, 0, width, height), float(solid.mean()), 0, None, False
    labels, count = ndimage.label(solid)
    sizes = np.asarray(ndimage.sum(solid, labels, index=range(1, count + 1)), dtype=np.float64)
    kept, chosen = choose_pieces(labels, sizes, anchors or [])
    keep = np.isin(labels, kept)
    # behind bars: every strip, and the bars between them kept as photo
    bridged = bridge_bars(solid, labels, sizes, chosen, bars_hint) if count > 1 else None
    if bridged is not None:
        keep = keep | bridged
        kept = [int(m) for m in np.unique(labels[keep]) if m > 0]
    dropped = count - len(kept)
    # soft edge from rembg inside the kept pieces only
    inside = np.where(keep, alpha, 0.0)
    if bridged is not None:
        inside = np.where(keep & ~solid, 1.0, inside)
    soft = inside.astype(np.float32)
    soft = np.asarray(Image.fromarray((soft * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3)),
                      dtype=np.float32) / 255.0
    soft = np.maximum(soft, inside)
    pixels = np.asarray(rgb, dtype=np.float32)
    backdrop = np.asarray(BACKDROP, dtype=np.float32)
    out = pixels * soft[..., None] + backdrop * (1.0 - soft[..., None])
    rows, cols = np.nonzero(keep)
    x0, y0, x1, y1 = int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1
    mx, my = (x1 - x0) * CROP_MARGIN, (y1 - y0) * CROP_MARGIN
    box = (int(max(0, round(x0 - mx))), int(max(0, round(y0 - my))),
           int(min(width, round(x1 + mx))), int(min(height, round(y1 + my))))
    cut = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB").crop(box)
    return cut, box, float(keep.mean()), dropped, keep[box[1]:box[3], box[0]:box[2]], bridged is not None


# --------------------------------------------------------------------------- both

async def prepare_photo(
    transport: Transport | None,
    photo_path: str | Path,
    out_dir: str | Path,
    species: str = "dog",
    gate: bool = True,
    strict: bool = True,
) -> PreparedPhoto:
    """Gate (when a transport is given) then cut. With strict, an agreed
    rejection raises PhotoRejected; otherwise the reasons are only logged."""
    photo_path = Path(photo_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(photo_path).convert("RGB")
    from .watermark import WATERMARK_NOTE, remove_watermarks
    image, watermarks = remove_watermarks(image)

    reasons: list[str] = []
    notes: list[str] = []
    readings: list[dict[str, Any]] = []
    if gate and transport is not None:
        reasons, notes, readings = await gate_photo(transport, image, species)
        if reasons and strict:
            raise PhotoRejected(reasons, readings)
    if watermarks:
        notes = notes + [WATERMARK_NOTE]

    seen_bars = bool(readings) and all(r.get("bars_in_front") for r in readings)
    cut, box, coverage, dropped, mask, bars = cut_out_with_mask(image, gate_anchors(readings, image.size),
                                                                bars_hint=seen_bars)
    # the strips say bars even when a sample did not: the note goes in then too
    if bars and BARS_NOTE not in notes:
        notes = notes + [BARS_NOTE]
    target = out_dir / "prepared.png"
    if cut is None:
        image.save(target)
    else:
        cut.save(target)
    # how far the photograph can be trusted (size, sharpness, exposure of
    # the animal's own pixels): no call, read by fit.py and the lite run
    from .photo_quality import measure_quality
    quality = measure_quality(cut if cut is not None else image, mask)
    # the face crop's subject box, so it does not run the model again on the cut-out
    subject = None
    if cut is not None and mask is not None and mask.any():
        rows, cols = np.nonzero(mask)
        subject = (int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1)
    return PreparedPhoto(path=target, original=photo_path, cut=cut is not None, box=box, coverage=coverage,
                         pieces_dropped=dropped, rejected=reasons, notes=notes, readings=readings, quality=quality,
                         bars=bars, subject_box=subject, watermarks=watermarks)


__all__ = ["BARS_NOTE", "GATE_SCHEMA", "PhotoRejected", "PreparedPhoto", "bridge_bars", "choose_pieces", "cut_out",
           "cut_out_with_mask", "gate_anchors", "gate_photo", "prepare_photo", "reject_reasons"]
