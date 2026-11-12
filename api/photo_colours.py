"""Measure each coat region's colour in the photograph itself.

A colour NAME from the vision model ("tan") says which of eight ramps a
region gets. It does not say how strong: the tan on this dog's lower legs
is a faint wash, on a Doberman's it is a saturated brown, and both are
"tan". The number has to come from the pixels.

Where a region is in the photograph is the hard part. The vision model is
poor at pixel coordinates but good at reading a labelled grid (see
face_photo.py), so the photo is sent with a lettered grid and the model
names, for each region it can see, the cell where that region's fur is
best seen - fur only, not a nose, not a shadow. The cell's pixels, white-
balanced by the photo's border as palette.py does, give the region's
measured colour as a LAB median. Regions the model cannot see get no
measurement and keep their named colour.

The grid is coarse against a dog that fills part of the frame, and a named
cell often lies off the dog. On public shelter photos (2026-10-28), 5 of a
dog's 7 regions landed on a truck and the sky, so all measured blue and
were dropped, and paws landed on a black mat (L 5). So when the photo
was cut out, only the animal's own pixels in a cell are read (the cut-out's
mask), and a cell with too few of them counts as a miss, not a reading.

The measurement is what the combination step (combine_paints.py) matches
against: the candidate closest to it, or a mix of two candidates in the
proportion that lands on it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .agents import Transport
from .face_photo import GRID_COLS, GRID_ROWS, _COLS, _image_block, draw_grid
from .fur import to_linear
from .regions import REGION_DESCRIPTIONS, REGIONS
from .sampling import sample

QUERY_SIDE = 1024
SAMPLES = 2
# The cell's pixels are read in a window this share of a cell wide around
# the cell centre, and the darkest/lightest tails are dropped before the
# median so an edge of shadow or a nose in the cell does not count.
WINDOW_SHARE = 0.8
TRIM = 0.15
# With the cut-out's mask, a cell whose window is less than this share animal
# is a miss: the model pointed off the dog. The mask is first eroded by
# ANIMAL_ERODE_PX (at the measured size) so the cut's soft edge, half fur and
# half background, is not read as fur. Re-measuring the recorded cells of the
# 20 eval dogs and 2 shelter dogs (340 cells, 2026-10-28): 126 cells were
# under half animal and 48 had none. With 0.5, 125 of 170 region readings
# survive; with 0.25 and the erosion, 147. The extra ones are edge cells
# whose animal part is real fur (e07's tan legs: L 62 from the dog, 65 from
# the grass around it before).
MIN_ANIMAL_SHARE = 0.25
ANIMAL_ERODE_PX = 2

REGION_CELL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "cells": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "enum": list(REGIONS)},
                    "col": {"type": "string", "enum": list(_COLS)},
                    "row": {"type": "integer", "minimum": 1, "maximum": GRID_ROWS},
                },
                "required": ["region", "col", "row"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["cells"],
    "additionalProperties": False,
}

_REGION_LIST = "\n".join(f"- {r}: {REGION_DESCRIPTIONS[r]}" for r in REGIONS)

REGION_CELL_SYSTEM = f"""You point at body regions of a dog in a photograph, using the grid drawn over it.

The image has a labelled grid: columns {_COLS[0]} to {_COLS[-1]} left to right, rows 1 to {GRID_ROWS}
top to bottom, the label of each cell in its corner. For every region listed below that is
clearly visible, name ONE cell that is filled with that region's FUR - not its edge, not the
eyes, nose, mouth, pads, floor or shadow. Skip a region you cannot see well; skipping is normal.

{_REGION_LIST}"""


def _measure(image: Image.Image, col: str, row: int, illuminant: np.ndarray,
             animal: np.ndarray | None = None) -> np.ndarray | None:
    from .projection import linear_to_lab

    width, height = image.size
    cw, ch = width / GRID_COLS, height / GRID_ROWS
    c = _COLS.index(col.upper()) if col and col.upper() in _COLS else None
    if c is None:
        return None
    r = int(np.clip(int(row) - 1, 0, GRID_ROWS - 1))
    x0, x1 = int((c + 0.5 - WINDOW_SHARE / 2) * cw), int((c + 0.5 + WINDOW_SHARE / 2) * cw)
    y0, y1 = int((r + 0.5 - WINDOW_SHARE / 2) * ch), int((r + 0.5 + WINDOW_SHARE / 2) * ch)
    window = np.asarray(image, dtype=np.float32)[y0:y1, x0:x1]
    if animal is not None:
        inside = animal[y0:y1, x0:x1]
        if inside.size == 0 or float(inside.mean()) < MIN_ANIMAL_SHARE:
            return None
        rgb = window[inside] / 255.0
    else:
        rgb = window.reshape(-1, 3) / 255.0
    if rgb.shape[0] < 16:
        return None
    rgb = np.clip(rgb / np.maximum(illuminant, 1e-3), 0.0, 1.0)
    lab = linear_to_lab(to_linear(rgb))
    order = np.argsort(lab[:, 0])
    keep = order[int(len(order) * TRIM): max(int(len(order) * (1 - TRIM)), int(len(order) * TRIM) + 1)]
    return np.median(lab[keep], axis=0)


# The border stands in for a grey card only when it is near neutral. A
# studio backdrop (e18: saturated red) or dirt floor divided out turned the
# whole dog green-cyan (measured a -16..-34) and cream rendered pale blue.
# A border whose spread across channels exceeds BORDER_MAX_SPREAD of its mean
# is not a grey card; any correction is clamped to +-BORDER_MAX_CORRECTION.
BORDER_MAX_SPREAD = 0.25
BORDER_MAX_CORRECTION = 0.10


def border_is_neutral(background: np.ndarray) -> bool:
    mean = max(float(background.mean()), 1e-4)
    return float(background.max() - background.min()) / mean <= BORDER_MAX_SPREAD


def border_illuminant(background: np.ndarray) -> np.ndarray:
    mean = max(float(background.mean()), 1e-4)
    if not border_is_neutral(background):
        return np.ones(3, dtype=np.float32)
    return np.clip(background / mean, 1.0 - BORDER_MAX_CORRECTION, 1.0 + BORDER_MAX_CORRECTION).astype(np.float32)


# A saturated studio backdrop (e18: coral red) is not only no grey card: its
# light is reflected in the fur, and the vision calls then read the white
# chest, legs and cheeks as tan (5/5 cheeks, 3/3 feet, eval v2). The photo
# they see is corrected along the backdrop's colour only, by as much as
# takes the dog's brightest fur (top BRIGHT_SHARE, assumed near-white)
# toward neutral, capped at CAST_MAX_ALPHA so a tan dog with no white is
# only nudged. Dirt and grass get no cast correction: their light is weak
# and the correction pushed the yellow jindo on dark grass (fixture 2222)
# toward magenta. Saturation is judged in LAB chroma, not the channel
# spread: dark grass spreads 0.77 against the red wall's 0.82 only because
# its mean is low, while its chroma is 34 against the eval set's grass and
# dirt at 18-34 and the wall's 56. (Either kind still drops the measured
# hue in palette.ramps_from_measurement; see the note there.)
BACKDROP_MIN_CHROMA = 45.0
BRIGHT_SHARE = 0.03
CAST_MAX_ALPHA = 0.15


def border_kind(background: np.ndarray) -> str:
    """'neutral' (a grey card: white-balance by it), 'natural' (dirt, grass,
    a wall: no balance) or 'backdrop' (a saturated studio colour cast into
    the fur: the photo the calls see is corrected for it)."""
    from .projection import linear_to_lab

    if border_is_neutral(background):
        return "neutral"
    lab = linear_to_lab(to_linear(np.asarray(background, dtype=np.float32)[None, None, :]))[0, 0]
    return "backdrop" if float(np.hypot(lab[1], lab[2])) >= BACKDROP_MIN_CHROMA else "natural"


def border_kind_of(photo_path) -> str:
    """border_kind of a photograph on disk (the original, not the cutout)."""
    from .palette import _background

    shown = Image.open(photo_path).convert("RGB")
    shown.thumbnail((QUERY_SIDE, QUERY_SIDE))
    return border_kind(_background(np.asarray(shown, dtype=np.float32) / 255.0))


def backdrop_cast(original: Image.Image, subject: Image.Image) -> tuple[np.ndarray, float] | None:
    """(illuminant to divide by, alpha) when `original`'s border is a
    saturated colour; None otherwise. `subject` is the cutout the calls see
    (its grey background is ignored)."""
    from .palette import _background

    shown = original.convert("RGB")
    shown.thumbnail((QUERY_SIDE, QUERY_SIDE))
    background = _background(np.asarray(shown, dtype=np.float32) / 255.0)
    mean = max(float(background.mean()), 1e-4)
    if border_kind(background) != "backdrop":
        return None
    direction = background / mean - 1.0
    rgb = np.asarray(subject.convert("RGB"), dtype=np.float32) / 255.0
    flat = rgb.reshape(-1, 3)
    fur = flat[np.abs(flat - 128 / 255.0).max(axis=1) > 0.02]     # not the cutout's grey
    if len(fur) < 1000:
        return None
    luma = fur.mean(axis=1)
    bright = fur[luma >= np.quantile(luma, 1.0 - BRIGHT_SHARE)]
    tint = bright.mean(axis=0) / max(float(bright.mean()), 1e-4) - 1.0
    alpha = float(np.clip(np.dot(tint, direction) / max(float(np.dot(direction, direction)), 1e-6),
                          0.0, CAST_MAX_ALPHA))
    return (1.0 + alpha * direction).astype(np.float32), alpha


def remove_backdrop_cast(original_path, subject_path, out_path) -> dict[str, Any] | None:
    """Write `subject_path` corrected for a saturated backdrop's cast to
    `out_path`; returns a report, or None when there is no such backdrop."""
    original, subject = Image.open(original_path), Image.open(subject_path).convert("RGB")
    cast = backdrop_cast(original, subject)
    if cast is None:
        return None
    illuminant, alpha = cast
    rgb = np.asarray(subject, dtype=np.float32) / 255.0
    corrected = np.clip(rgb / illuminant, 0.0, 1.0)
    # the cutout's grey stays grey
    backdrop = np.abs(rgb - 128 / 255.0).max(axis=-1) <= 0.02
    corrected[backdrop] = rgb[backdrop]
    Image.fromarray((corrected * 255 + 0.5).astype(np.uint8)).save(out_path)
    return {"alpha": round(alpha, 3), "illuminant": [round(float(x), 3) for x in illuminant]}


def _shown(image: Image.Image) -> Image.Image:
    scale = QUERY_SIDE / max(image.size)
    if scale >= 1.0:
        return image
    return image.resize((int(image.size[0] * scale), int(image.size[1] * scale)), Image.LANCZOS)


async def measure_region_colours(photo_path, transport: Transport, samples: int = SAMPLES,
                                 animal_mask: np.ndarray | None = None) -> dict[str, Any]:
    """region -> {"lab": [L, a, b], "cell": "D6", "samples": n} for the
    regions a majority of the readings could point at. `animal_mask`: the
    cut-out's animal in the photo's own pixels (PreparedPhoto.animal_mask)."""
    image = Image.open(photo_path).convert("RGB")
    shown = _shown(image)

    blocks = [_image_block(draw_grid(shown)), {"type": "text", "text": (
        f"The grid is {GRID_COLS} columns ({_COLS[0]}-{_COLS[-1]}) by {GRID_ROWS} rows. "
        "Give one fur cell per visible region.")}]
    payloads = await sample(transport, system=REGION_CELL_SYSTEM, blocks=blocks, schema=REGION_CELL_SCHEMA, n=samples)
    return measure_cells(image, [p for p in payloads if isinstance(p, dict)], animal_mask)


def measure_cells(image: Image.Image, readings: list[dict[str, Any]],
                  animal_mask: np.ndarray | None = None) -> dict[str, Any]:
    """The measurement for cell readings already taken (the region-cell
    calls' answers), so recorded cells can be measured again offline."""
    from .palette import _background

    shown = _shown(image)
    rgb = np.asarray(shown, dtype=np.float32) / 255.0
    background = _background(rgb)
    illuminant = border_illuminant(background)
    # a saturated backdrop's colour is also reflected in the fur: the hue
    # measured is the scene's (palette.ramps_from_measurement then keeps
    # the table's hue and takes the lightness only)
    kind = border_kind(background)
    animal = None
    if animal_mask is not None and animal_mask.shape == (image.size[1], image.size[0]):
        from scipy import ndimage

        animal = np.asarray(Image.fromarray(animal_mask.astype(np.uint8) * 255).resize(shown.size, Image.NEAREST)) > 127
        animal = ndimage.binary_erosion(animal, iterations=ANIMAL_ERODE_PX)

    per_region: dict[str, list[tuple[np.ndarray, str]]] = {}
    missed: dict[str, int] = {}
    for payload in readings:
        seen = set()
        for item in payload.get("cells") or []:
            region = item.get("region")
            if region not in REGIONS or region in seen:
                continue
            seen.add(region)
            lab = _measure(shown, str(item.get("col", "")), int(item.get("row", 1)), illuminant, animal)
            if lab is not None:
                per_region.setdefault(region, []).append((lab, f"{item.get('col')}{item.get('row')}"))
            elif animal is not None:
                missed[region] = missed.get(region, 0) + 1
    out: dict[str, Any] = {}
    majority = len(readings) // 2 + 1 if readings else 1
    for region, values in per_region.items():
        # a reading that pointed off the animal is a miss, not a disagreement
        if len(values) < majority - missed.get(region, 0):
            continue
        labs = np.stack([v[0] for v in values])
        out[region] = {"lab": [round(float(x), 1) for x in np.median(labs, axis=0)],
                       "cells": [v[1] for v in values], "samples": len(values),
                       "neutral_border": kind == "neutral", "border": kind}
        if missed.get(region):
            out[region]["missed"] = missed[region]
    return out


# The coat's chroma from every warm fur pixel of the cut-out, white-balanced
# by the border as the cells are: fur at least COAT_MIN_CHROMA, of hue
# COAT_HUE, neither black nor white, its chroma at the COAT_PERCENTILE-th
# percentile - above the shade's greyed hair and a cream bib's or white
# face's pale edge, below the glare. A few cells read a golden puppy's coat
# khaki (11-01) and a red papillon's brown dull (shelter maltese 7, chroma
# 12 on the model, 21 in its photo); the whole coat's pixels, rendered and
# compared on 22 dogs, put each coat's chroma where its photo has it
# (palette.chroma_from_coat).
COAT_MIN_CHROMA = 10.0
COAT_HUE = (30.0, 100.0)
COAT_L = (8.0, 95.0)
COAT_PERCENTILE = 65


def _photo_parts(image: Image.Image, animal_mask: np.ndarray | None,
                 balance: bool = True) -> tuple[np.ndarray, str, np.ndarray] | None:
    """(the shown photo's RGB, white-balanced by the border unless not
    `balance`; the border's kind; the animal mask at that size), or None
    without a mask of the photo's size."""
    from .palette import _background

    if animal_mask is None or animal_mask.shape != (image.size[1], image.size[0]):
        return None
    shown = _shown(image)
    rgb = np.asarray(shown, dtype=np.float32) / 255.0
    background = _background(rgb)
    illuminant = border_illuminant(background) if balance else np.ones(3, dtype=np.float32)
    animal = np.asarray(Image.fromarray(animal_mask.astype(np.uint8) * 255).resize(shown.size, Image.NEAREST)) > 127
    return np.clip(rgb / np.maximum(illuminant, 1e-3), 0.0, 1.0), border_kind(background), animal


def _animal_lab(image: Image.Image, animal_mask: np.ndarray | None,
                balance: bool = True) -> tuple[np.ndarray, str] | None:
    """(LAB of every animal pixel, white-balanced by the border unless not
    `balance`; the border's kind), or None without an animal mask or with
    too little animal."""
    from scipy import ndimage

    from .projection import linear_to_lab

    parts = _photo_parts(image, animal_mask, balance)
    if parts is None:
        return None
    rgb, kind, animal = parts
    animal = ndimage.binary_erosion(animal, iterations=ANIMAL_ERODE_PX + 1)
    if animal.sum() < 500:
        return None
    return linear_to_lab(to_linear(rgb[animal])), kind


def coat_chroma(image: Image.Image, animal_mask: np.ndarray | None) -> dict[str, Any] | None:
    """{chroma, pixels, share, border} of the photo's warm fur, or None
    without an animal mask or warm fur."""
    read = _animal_lab(image, animal_mask)
    if read is None:
        return None
    lab, kind = read
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    hue = np.degrees(np.arctan2(lab[:, 2], lab[:, 1]))
    warm = ((chroma >= COAT_MIN_CHROMA) & (hue >= COAT_HUE[0]) & (hue <= COAT_HUE[1])
            & (lab[:, 0] > COAT_L[0]) & (lab[:, 0] < COAT_L[1]))
    if not warm.any():
        return None
    return {"chroma": round(float(np.percentile(chroma[warm], COAT_PERCENTILE)), 1), "pixels": int(warm.sum()),
            "share": round(float(warm.mean()), 3), "border": kind}


# A light coat's own colour from the whole animal, for a dog read cream
# (palette.cream_from_coat). A cell is too small for it: on a tan dog's
# cream bib it lands on the tan beside it or the shadow under the chin, and
# that is why cream keeps the table (palette.MEASURED_NAMES). The whole
# animal has the same shadows, so only its lit half counts - LIGHT_COAT_BAND
# percentiles of lightness, above the shade and the dark eyes and nose,
# below the glare - and only fur that is grey or warm (chroma under
# LIGHT_COAT_GREY_CHROMA, or hue within LIGHT_COAT_HUE): a collar, a blue
# shirt or a pink tongue is neither. Pale yellow runs to hue 100-110 after
# the white balance (shelter maltese 3), past COAT_HUE's 100.
# The lightness is all the lit fur's. The colour is the warm part's alone
# when it is at least LIGHT_COAT_WARM_SHARE of it: a cream chihuahua's white
# chest and socks are a third of its lit fur and halved the chroma.
# The lightness is the camera's as much as the coat's: auto exposure puts
# what fills the frame near mid grey, and the seven white dogs on the models
# page read L 55-90 (a white toy poodle 55). So the scene's own light end -
# the LIGHT_COAT_SCENE_PERCENTILE of everything around the animal - comes
# with it, and palette.cream_from_coat goes by the fur's distance below it.
LIGHT_COAT_BAND = (50, 97)
LIGHT_COAT_GREY_CHROMA = 8.0
LIGHT_COAT_HUE = (30.0, 115.0)
LIGHT_COAT_WARM_SHARE = 0.3
LIGHT_COAT_SCENE_PERCENTILE = 95
LIGHT_COAT_MIN_SCENE = 2000


def light_coat(image: Image.Image, animal_mask: np.ndarray | None) -> dict[str, Any] | None:
    """{lab, warm_share, pixels, border, scene_l} of the photo's lit fur,
    or None without an animal mask; scene_l is None when the animal leaves
    too little of the frame."""
    from scipy import ndimage

    from .projection import linear_to_lab

    parts = _photo_parts(image, animal_mask)
    if parts is None:
        return None
    rgb, kind, animal = parts
    inside = ndimage.binary_erosion(animal, iterations=ANIMAL_ERODE_PX + 1)
    around = ~ndimage.binary_dilation(animal, iterations=ANIMAL_ERODE_PX + 1)
    if inside.sum() < 500:
        return None
    lab = linear_to_lab(to_linear(rgb[inside]))
    scene_l = None
    if around.sum() >= LIGHT_COAT_MIN_SCENE:
        scene = linear_to_lab(to_linear(rgb[around]))
        scene_l = round(float(np.percentile(scene[:, 0], LIGHT_COAT_SCENE_PERCENTILE)), 1)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    hue = np.degrees(np.arctan2(lab[:, 2], lab[:, 1]))
    warm = (hue >= LIGHT_COAT_HUE[0]) & (hue <= LIGHT_COAT_HUE[1]) & (chroma < 60)
    fur = lab[(chroma < LIGHT_COAT_GREY_CHROMA) | warm]
    if len(fur) < 500:
        return None
    low, high = np.percentile(fur[:, 0], LIGHT_COAT_BAND)
    lit = fur[(fur[:, 0] >= low) & (fur[:, 0] <= high)]
    lit_warm = lit[np.hypot(lit[:, 1], lit[:, 2]) >= LIGHT_COAT_GREY_CHROMA]
    share = len(lit_warm) / max(len(lit), 1)
    colour = np.median(lit_warm if share >= LIGHT_COAT_WARM_SHARE else lit, axis=0)
    out = [float(np.median(lit[:, 0])), float(colour[1]), float(colour[2])]
    return {"lab": [round(x, 1) for x in out], "warm_share": round(share, 2), "pixels": int(len(lit)), "border": kind,
            "scene_l": scene_l}


# Sunlight on the fur. A photo cannot tell a yellow dog in white light from
# a white dog in yellow light, but a coat's own colour stays in the shade
# while low evening sun colours only what it reaches: shelter jindo puppy 4,
# recorded 흰색, read chroma 11 in its mid-lit fur and 32 in the sun (the
# person's arms orange, the floor in shade blue), while a cream chihuahua's
# reads 15 and 13, a cream maltese's 11 and 10. So the lit fur's
# colourfulness (chroma over lightness, which shade alone keeps) is set
# against the mid-lit fur's, on the photo as taken - no white balance, which
# would scale both. SUN_RATIO and over, with the lit fur at SUN_MIN_CHROMA
# or more (a white dog's chroma 1 -> 4 is noise, not sun), and the photo's
# hue and chroma are the light's.
# Only for a light coat, its mid-lit fur at SUN_MIN_MID_L or more: the two
# bands must be the same fur. On the shelter sample's 398 cut-outs the test
# without it flagged six, five of them two things side by side - a
# black-and-tan's black (mid L 34-43) against its tan, yellow leather gloves
# and a bare arm beside black dogs (L 31-34), a red coat against its white
# chest (L 47). With it, puppy 4 alone (L 66); of the 244 light coats, the
# sixteen nearest the line are indoor or overcast. One sunlit photo in 400
# is all there is to tune on, so the line is the safe side of it.
# A grey card does not save such a photo: the floor in the shade is blue
# while the dog is in the sun, and balancing by it turns the dog yellower.
# A grey coat's own grey: the animal's pixels of low chroma that are neither
# black nor white fur. The table's grey is one mid blue-grey for a charcoal
# husky and a pale dirty poodle alike.
GRAY_COAT_MAX_CHROMA = 8.0
GRAY_COAT_L = (15.0, 75.0)
GRAY_COAT_MIN_SHARE = 0.1
GRAY_COAT_MIN_PIXELS = 500


def gray_coat(image: Image.Image, animal_mask: np.ndarray | None) -> dict[str, Any] | None:
    """The grey fur of the animal (GRAY_COAT_*): its median LAB, the quartiles
    of its lightness, its share of the animal and the border's kind; None
    when there is too little of it to go by."""
    got = _animal_lab(image, animal_mask)
    if got is None:
        return None
    lab, kind = got
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    low, high = GRAY_COAT_L
    grey = (chroma < GRAY_COAT_MAX_CHROMA) & (lab[:, 0] > low) & (lab[:, 0] < high)
    if grey.sum() < GRAY_COAT_MIN_PIXELS or grey.mean() < GRAY_COAT_MIN_SHARE:
        return None
    fur = lab[grey]
    return {"lab": [round(float(v), 1) for v in np.median(fur, axis=0)],
            "l_quartiles": [round(float(v), 1) for v in np.percentile(fur[:, 0], [25, 75])],
            "share": round(float(grey.mean()), 3), "border": kind}


SUN_BANDS = ((40, 65), (65, 97))
SUN_MIN_CHROMA = 10.0
SUN_RATIO = 1.5
SUN_MIN_MID_L = 55.0


def sun_cast_of(lab: np.ndarray) -> dict[str, Any]:
    """{ratio, mid_l, mid_chroma, lit_chroma, sunlit} of an animal's LAB
    pixels (unbalanced)."""
    lightness = lab[:, 0]
    medians = []
    for low, high in SUN_BANDS:
        a, b = np.percentile(lightness, (low, high))
        medians.append(np.median(lab[(lightness >= a) & (lightness <= b)], axis=0))
    (mid_l, *mid_ab), (lit_l, *lit_ab) = medians
    mid_c, lit_c = float(np.hypot(*mid_ab)), float(np.hypot(*lit_ab))
    ratio = (lit_c / max(float(lit_l), 1.0)) / max(mid_c / max(float(mid_l), 1.0), 0.02)
    return {"ratio": round(ratio, 2), "mid_l": round(float(mid_l), 1), "mid_chroma": round(mid_c, 1),
            "lit_chroma": round(lit_c, 1),
            "sunlit": bool(lit_c >= SUN_MIN_CHROMA and ratio >= SUN_RATIO and float(mid_l) >= SUN_MIN_MID_L)}


def sun_cast(image: Image.Image, animal_mask: np.ndarray | None) -> dict[str, Any] | None:
    """sun_cast_of the photo's animal, or None without an animal mask."""
    read = _animal_lab(image, animal_mask, balance=False)
    return sun_cast_of(read[0]) if read is not None else None


__all__ = ["REGION_CELL_SCHEMA", "border_kind", "border_kind_of", "coat_chroma", "light_coat", "measure_cells",
           "measure_region_colours", "sun_cast", "sun_cast_of"]
