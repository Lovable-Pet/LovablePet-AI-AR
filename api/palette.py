"""Calibrate the coat palette against the actual photographs.

WHY THE TABLE ALONE IS NOT ENOUGH
---------------------------------
COAT_COLORS turns a name into a fixed RGB, so "tan" is the same orange for
every animal. Measured against a real shelter photo that is far too strong:
this dog's subject pixels sit at a relative chroma around 0.12, while the
table's tan is 0.68 - five times more saturated than anything on the animal.
Rendered, the coat looks like it was coloured in rather than photographed.

WHAT IS MEASURED
----------------
Relative chroma, (max-min)/mean per pixel, rather than absolute saturation.
Shading scales all three channels together, so it moves absolute chroma but
leaves the ratio alone; a dog half in shadow still reports the same relative
chroma as one in full light. That makes it the one statistic that can be read
off an uncontrolled photo without first solving for the lighting.

Luminance is deliberately NOT taken from the photo. A photograph's brightness
is surface colour times illumination, and a texture must carry only the first:
copying the photo's dark side into the albedo would bake that shadow onto the
model, where it would then be lit again.

WHITE BALANCE
-------------
The border of a shelter photo is floor or wall - close enough to neutral to
serve as a grey card. Its median gives the illuminant, and dividing it out
removes the room's colour cast before any statistic is taken, so a warm
fluorescent does not read as a warmer coat.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .fur import FUR_PALETTE

# Photos are analysed at this width; the statistics are distributional and a
# larger image only costs time.
ANALYSIS_WIDTH = 200

# How wide a border to read the background from, as a fraction of the image.
BORDER_FRACTION = 0.04

# A pixel this far from the background colour is taken to be the animal.
SUBJECT_DISTANCE = 0.13

# Below this share of the frame the subject estimate is not trustworthy - the
# animal may be small, off-centre, or the same colour as the floor.
MIN_SUBJECT_SHARE = 0.10

# The reference for "how saturated is this coat's colour" is the most saturated
# few percent of the animal's pixels, not a percentile of all of them. On a
# black-and-tan the tan pips are a sliver of the frame: p90 over the whole dog
# landed on black-and-white fur and reported 0.29, which scaled the tan ramp to
# ×0.40 and turned it beige. The top 5% ARE the tan, and they read 0.72.
TOP_SATURATED_SHARE = 0.05

# Never pull saturation below this share of the palette's own value. The palette
# was authored for indoor fur; a photo can only justify a moderate correction.
MIN_SCALE = 0.6

# How far to pull the table toward the photograph. Not the whole way: a shelter
# photo is not a colour-managed capture, and compression and the camera's own
# saturation both leave their mark on the measurement.
CALIBRATION_STRENGTH = 0.85

# HUE FROM THE PHOTOGRAPH
# -----------------------
# Scaling saturation keeps the table's hue, and the table's hue is a guess:
# its "white" is a warm ivory and its "tan" an orange, while this dog's white
# is neutral and its tan a muted brown. So each named colour's a/b is read off
# the photograph too. The photo is not clustered - k-means dragged "tan" onto
# the grey where black fur shades into white - but split by the one thing that
# separates the coat's colours in an uncontrolled photo: chroma for the hued
# colour, lightness for the two neutrals.
#
# Photos are read at this width for the colour measurement; the medians below
# need more pixels than the chroma statistic does.
PHOTO_COLOUR_WIDTH = 400

# Names that are a hue rather than a shade of grey. The photo's most saturated
# pixels are that hue - when the coat names exactly one of them. Two hued
# names cannot be told apart this way and keep the table's hue.
HUED_NAMES = frozenset({"tan", "red", "fawn", "chocolate", "cream"})

# Pixels below this percentile of chroma are read as neutral fur.
NEUTRAL_CHROMA_PERCENTILE = 60

# Lightness bounds for the two neutral names.
WHITE_MIN_L = 60.0
BLACK_MAX_L = 25.0

# Fewer pixels than this and a colour is not measured.
MIN_COLOUR_PIXELS = 200


@dataclass(frozen=True)
class PhotoStats:
    illuminant: tuple[float, float, float]
    relative_chroma: float
    subject_share: float
    pixels: int

    @property
    def usable(self) -> bool:
        return self.pixels > 500 and self.subject_share >= MIN_SUBJECT_SHARE


def _load(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    width = ANALYSIS_WIDTH
    height = max(1, int(width * image.size[1] / image.size[0]))
    return np.asarray(image.resize((width, height), Image.LANCZOS), dtype=np.float32) / 255.0


def _background(image: np.ndarray) -> np.ndarray:
    """The frame's border, as a stand-in for floor or wall."""
    height, width, _ = image.shape
    margin = max(2, int(min(height, width) * BORDER_FRACTION))
    border = np.concatenate([
        image[:margin].reshape(-1, 3), image[-margin:].reshape(-1, 3),
        image[:, :margin].reshape(-1, 3), image[:, -margin:].reshape(-1, 3),
    ])
    return np.median(border, axis=0)


def measure(photo_paths: list[str | Path]) -> PhotoStats:
    """Read the illuminant and the coat's relative chroma off the photographs."""
    subjects: list[np.ndarray] = []
    illuminants: list[np.ndarray] = []
    total = 0

    for path in photo_paths:
        image = _load(path)
        background = _background(image)
        illuminants.append(background)

        flat = image.reshape(-1, 3)
        distance = np.linalg.norm(flat - background, axis=1)
        subject = flat[distance > SUBJECT_DISTANCE]
        total += flat.shape[0]
        if subject.shape[0]:
            subjects.append(subject)

    if not subjects:
        return PhotoStats((1.0, 1.0, 1.0), 0.0, 0.0, 0)

    illuminant = np.median(np.stack(illuminants), axis=0)
    # normalise so white balancing changes colour, not exposure
    illuminant = illuminant / max(float(illuminant.mean()), 1e-4)

    pixels = np.concatenate(subjects)
    balanced = np.clip(pixels / np.maximum(illuminant, 1e-3), 0.0, 1.0)

    mean = balanced.mean(axis=1)
    bright = mean > 0.04          # near-black pixels carry no usable hue
    if bright.sum() < 100:
        bright = np.ones_like(mean, dtype=bool)
    usable = balanced[bright]
    spread = usable.max(axis=1) - usable.min(axis=1)
    relative = spread / np.maximum(usable.mean(axis=1), 1e-3)

    top = max(int(relative.size * TOP_SATURATED_SHARE), 50)
    strongest = np.sort(relative)[-top:]

    return PhotoStats(
        illuminant=tuple(float(value) for value in illuminant),
        relative_chroma=float(np.median(strongest)),
        subject_share=pixels.shape[0] / max(total, 1),
        pixels=int(pixels.shape[0]),
    )


def _subject_mask(image: Image.Image, rgb: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Where the animal is: rembg when installed, else distance from the border."""
    try:
        from .imagegen import _rembg_alpha
        alpha = _rembg_alpha(image)
    except Exception:  # noqa: BLE001 - rembg is optional
        alpha = None
    if alpha is not None:
        from PIL import ImageFilter
        mask = Image.fromarray(((alpha > 0.5) * 255).astype(np.uint8), mode="L")
        # a few pixels in from the cut: the edge mixes fur with floor
        return np.asarray(mask.filter(ImageFilter.MinFilter(9))) > 127
    return np.linalg.norm(rgb - background, axis=-1) > SUBJECT_DISTANCE


def photo_colours(names: list[str], photo_paths: list[str | Path]) -> dict[str, tuple[float, float]]:
    """The a/b chroma of each named fur colour as it appears in the photos,
    white-balanced. Only names the photo can settle are returned."""
    from .fur import to_linear
    from .projection import linear_to_lab

    labs: list[np.ndarray] = []
    for path in photo_paths:
        image = Image.open(path).convert("RGB")
        width = PHOTO_COLOUR_WIDTH
        height = max(1, int(width * image.size[1] / image.size[0]))
        image = image.resize((width, height), Image.LANCZOS)
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        background = _background(rgb)
        illuminant = background / max(float(background.mean()), 1e-4)
        balanced = np.clip(rgb / np.maximum(illuminant, 1e-3), 0.0, 1.0)
        subject = _subject_mask(image, rgb, background)
        if subject.sum() < MIN_COLOUR_PIXELS:
            continue
        labs.append(linear_to_lab(to_linear(balanced))[subject])
    if not labs:
        return {}

    lab = np.concatenate(labs)
    lightness, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    chroma = np.hypot(a, b)
    wanted = {(name or "").strip().lower() for name in names}
    out: dict[str, tuple[float, float]] = {}

    def take(name: str, select: np.ndarray) -> None:
        if select.sum() >= MIN_COLOUR_PIXELS:
            out[name] = (float(np.median(a[select])), float(np.median(b[select])))

    hued = sorted(wanted & HUED_NAMES)
    if len(hued) == 1:
        take(hued[0], chroma >= np.percentile(chroma, 100 * (1 - TOP_SATURATED_SHARE)))
    neutral = chroma < np.percentile(chroma, NEUTRAL_CHROMA_PERCENTILE)
    if "white" in wanted:
        take("white", neutral & (lightness > WHITE_MIN_L))
    if "black" in wanted:
        take("black", lightness < BLACK_MAX_L)
    return out


def apply_photo_colours(
    palette: dict[str, tuple[str, str, str]],
    colours: dict[str, tuple[float, float]],
    strength: float = CALIBRATION_STRENGTH,
) -> dict[str, tuple[str, str, str]]:
    """Turn each measured ramp toward the photograph's hue. Lightness is kept,
    and each stop keeps its share of the mid stop's chroma, so a ramp stays a
    ramp - only its direction in a/b changes."""
    from .fur import hex_to_linear, to_srgb
    from .projection import lab_to_linear, linear_to_lab

    out = dict(palette)
    for name, (photo_a, photo_b) in colours.items():
        stops = palette.get(name)
        if stops is None:
            continue
        labs = [linear_to_lab(hex_to_linear(stop)[None, None, :])[0, 0] for stop in stops]
        mid_chroma = max(float(np.hypot(labs[1][1], labs[1][2])), 1e-3)
        photo = np.array([photo_a, photo_b], dtype=np.float32)
        new_stops = []
        for lab in labs:
            ratio = float(np.hypot(lab[1], lab[2])) / mid_chroma
            target = photo * ratio
            ab = lab[1:] + strength * (target - lab[1:])
            rgb = to_srgb(lab_to_linear(np.array([lab[0], ab[0], ab[1]], dtype=np.float32)[None, None, :])[0, 0])
            new_stops.append(_rgb_to_hex(tuple(int(round(float(c) * 255)) for c in rgb)))
        out[name] = tuple(new_stops)
    return out


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    text = value.lstrip("#")
    return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def relative_chroma(rgb: tuple[int, int, int]) -> float:
    values = np.asarray(rgb, dtype=np.float32) / 255.0
    return float((values.max() - values.min()) / max(values.mean(), 1e-3))


def chroma_scale(names: list[str], stats: PhotoStats,
                 strength: float = CALIBRATION_STRENGTH) -> float:
    """One factor for the whole palette.

    Every colour is scaled by the same amount rather than each being pushed to
    the measured chroma individually. The photograph gives one reading of how
    saturated this coat is; it does not say how black should differ from tan,
    and FUR_PALETTE already does. A shared factor takes the overall level from
    the photo and leaves the relationships between the named colours intact.
    """
    if not stats.usable:
        return 1.0
    chromas = [
        relative_chroma(_hex_to_rgb(stop))
        for name in names if name
        for stop in FUR_PALETTE.get((name or "").strip().lower(), ())
    ]
    strongest = max(chromas) if chromas else 0.0
    if strongest <= 1e-3:
        return 1.0
    scale = 1.0 + (stats.relative_chroma / strongest - 1.0) * strength
    return max(MIN_SCALE, min(1.0, scale))


def calibrate_color(rgb: tuple[int, int, int], scale: float) -> tuple[int, int, int]:
    """Scale one colour's saturation, keeping its hue and its brightness.

    Only the length of the colour's offset from grey changes, so tan stays tan
    and stops being highlighter-orange. Brightness is untouched for the reason
    in the module docstring.
    """
    values = np.asarray(rgb, dtype=np.float32) / 255.0
    luminance = float(values.mean())
    adjusted = np.clip(luminance + (values - luminance) * max(scale, 0.0), 0.0, 1.0)
    return tuple(int(round(channel * 255)) for channel in adjusted)


def build_palette(
    names: list[str], stats: PhotoStats, strength: float = CALIBRATION_STRENGTH
) -> dict[str, tuple[str, str, str]]:
    """Calibrated fur ramps for every colour name the appearance mentions.

    All three stops of a ramp are scaled by the same factor, so the dark/mid/
    light relationship that makes the coat read as fur is preserved while its
    overall saturation comes down to what the photograph shows.
    """
    scale = chroma_scale(names, stats, strength)
    palette: dict[str, tuple[str, str, str]] = {}
    for name in names:
        key = (name or "").strip().lower()
        if not key or key in palette or key not in FUR_PALETTE:
            continue
        palette[key] = tuple(
            _rgb_to_hex(calibrate_color(_hex_to_rgb(stop), scale))
            for stop in FUR_PALETTE[key]
        )
    return palette


def palette_for(appearance: dict, photo_paths: list[str | Path]) -> tuple[dict, PhotoStats, str]:
    """Calibrate every colour this appearance names. Returns the palette, the
    measurements, and a line describing what happened."""
    names = [appearance.get("base_color", "")]
    names += [entry.get("color", "") for entry in appearance.get("region_colors") or []]
    names += [marking.get("color", "") for marking in appearance.get("markings") or []]

    stats = measure(list(photo_paths))
    if not stats.usable:
        return (
            {},
            stats,
            f"photos unusable for calibration (subject {stats.subject_share:.0%} "
            f"of frame, {stats.pixels} px); keeping the table colours",
        )

    scale = chroma_scale(names, stats)
    palette = build_palette(names, stats, strength=CALIBRATION_STRENGTH)
    colours = photo_colours(names, list(photo_paths))
    palette = apply_photo_colours(palette, colours)
    hues = ", ".join(f"{name} a{a:+.0f} b{b:+.0f}" for name, (a, b) in sorted(colours.items()))
    shifts = ", ".join(
        f"{name} {relative_chroma(_hex_to_rgb(FUR_PALETTE[name][1])):.2f}"
        f"->{relative_chroma(_hex_to_rgb(ramp[1])):.2f}"
        for name, ramp in palette.items()
        if abs(relative_chroma(_hex_to_rgb(FUR_PALETTE[name][1]))
               - relative_chroma(_hex_to_rgb(ramp[1]))) > 0.02
    )
    return (
        palette,
        stats,
        f"photo chroma (top {TOP_SATURATED_SHARE:.0%})={stats.relative_chroma:.2f}, "
        f"saturation x{scale:.2f}, illuminant "
        f"{tuple(round(value, 2) for value in stats.illuminant)}; {shifts or 'no change'}; "
        f"hue from photo: {hues or 'none measurable'}",
    )


# RAMPS FROM THE REGION MEASUREMENT (lite path)
# ----------------------------------------------
# photo_colours.measure_region_colours reads each region's fur colour off a
# grid cell the vision model pointed at: a trimmed median, white-balanced
# by the photo's border. The lite build paints a region with a named ramp,
# and the ramp's mid stop is where the region's mean lands; a corgi read as
# "fawn" measured L 52 against fawn's mid of L 68 and rendered 20 L too
# pale. So every hued ramp the appearance uses is moved toward the median
# of the measurements of the regions painted with it. Not all the way, and
# not without a cap: the module docstring's reason still holds - part of a
# photo's lightness is the room's light - and one cell in shadow must not
# turn a fawn coat chocolate. Neutral names (white, black, gray) keep the
# table: a white bib in shadow is still white.
MEASURED_STRENGTH = 0.7
# (18 until 11-01: a yellow puppy's cells, in the shade of its own coat, read
# L 52 against its photo's coat at 63 and the coat went muddy)
MEASURED_MAX_SHIFT_L = 10.0
# ... and lighter by at most this much: the cells land on sunlit fur, glare
# and pale markings more often than the coat's own tone, and "paler than the
# photo" was the user's call on four dogs (shelter puppies 4, 6, 7; fixture
# 2222). Puppy 6's fawn went +10 L from a sunlit forehead (L 83) and a cell on
# its white muzzle stripe (L 92) though its coat in the photo is a deep gold.
MEASURED_MAX_LIGHTEN_L = 5.0
MEASURED_MAX_SHIFT_AB = 10.0
# cream is a light name: the cell pointed at for a cream chest or cheek on a
# tan dog lands on the tan beside it or in the shadow under the chin (eval
# v7 e05, e09: measured L 48-55 against cream's 88, the capped -18 L turned
# the bib grey-beige). It keeps the table.
MEASURED_NAMES = HUED_NAMES - {"cream"}
# A cell the model pointed at for a "tan" region can land on black fur or
# a white sock next to it (the scripted black shiba's legs_lower cell read
# LAB (0.1, 0, 0) and dragged tan to the cap; the red shiba's skull cell read
# (75, 9, 5) - the cream face, chroma 10 - and desaturated red to a peach).
# A reading counts for a hued name only when it is itself clearly a hue -
# chroma at least this, which cream (mid chroma 11) never reaches, so cream
# keeps the table - and neither near-black nor near-white.
MEASURED_MIN_CHROMA = 15.0
MEASURED_L_RANGE = (15.0, 92.0)
# Fur is red-orange to yellow: the table's hued stops sit at LAB hue 45..90
# degrees and the eval set's good readings at 39..92. Outside that the cell
# measured the scene - e14's side-on head read grass (hue 118) and turned
# the cream cheeks green; e18's red backdrop cast the tan coat hue 34 and
# the capped shift painted the whole dog salmon (eval v2).
MEASURED_HUE_RANGE = (40.0, 100.0)


def _plausible_hue(lab: np.ndarray) -> bool:
    chroma = float(np.hypot(lab[1], lab[2]))
    hue = float(np.degrees(np.arctan2(lab[2], lab[1])))
    return (chroma >= MEASURED_MIN_CHROMA and MEASURED_L_RANGE[0] <= float(lab[0]) <= MEASURED_L_RANGE[1]
            and MEASURED_HUE_RANGE[0] <= hue <= MEASURED_HUE_RANGE[1])


# "cream" on a warm dog is its pale front, which the calls now name by being
# lighter than the back - on a tan dog that is a pale tan, not the table's
# near-white cream (L 88): the tan mix's cream bib rendered bright white
# (user on eval v7 e05). On a tan, red, fawn or chocolate coat the cream
# ramp is taken this share of the way from the base's ramp to cream, in LAB.
CREAM_ON_WARM = 0.6
WARM_BASES = ("tan", "red", "fawn", "chocolate")


# "cream" cheeks under a dark crown run from beige to near white: measured in
# the head close-ups, a black-and-tan puppy's L 55, e08's 67, a black shiba's
# 83, while the cream ramp sits at L 88 - the puppy's beige cheeks came out a
# white husky mask (user, 11-01). On such a dog the cream ramp takes the
# colour of the light fur in the lower half of its own face crop (no call;
# face_light_lab), FACE_CREAM_STRENGTH of the way: at most
# FACE_CREAM_MAX_SHIFT_L darker and never lighter than the table, its a*/b*
# at most FACE_CREAM_MAX_SHIFT_AB warmer - darkened alone, at cream's chroma,
# the puppy's cheeks came out olive grey.
FACE_CREAM_STRENGTH = 0.85
FACE_CREAM_MAX_SHIFT_L = 30.0
FACE_CREAM_MAX_SHIFT_AB = 8.0
FACE_CREAM_MIN_PIXELS = 400
# the face between the eyes and the mouth in the crop, as shares of its
# height and width: down to 0.85 and out to the edges it took in the white
# chest under a black puppy's chin, and a black face read white (shelter
# puppy 11, user: "얼굴 검은색인 것 같고")
FACE_LOWER = ((0.40, 0.72), (0.18, 0.82))


def face_light(crop: Image.Image) -> tuple[float | None, list[float] | None]:
    """(share, lab) of the light fur in the lower face of a head crop - L*
    over 40, not plainly blue (a glove, a backdrop; a cool white coat is
    kept) - among the dog's pixels there (the cut-out's flat grey fill left
    out): its share, and its median CIE LAB (None with too few such
    pixels)."""
    from .fur import to_linear
    from .projection import linear_to_lab

    rgb = np.asarray(crop.convert("RGB"), dtype=np.float32)
    h, w = rgb.shape[:2]
    (y0, y1), (x0, x1) = FACE_LOWER
    rgb = rgb[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    lab = linear_to_lab(to_linear(rgb))
    fill = (np.abs(rgb - rgb.mean(axis=-1, keepdims=True)).max(axis=-1) < 1.0)
    # cool pixels count only as a white coat: bright and near grey. A black
    # coat's sheen under daylight is blue-grey at L* 45-60 (shelter puppy 10),
    # a glove or backdrop plainly blue
    cool = lab[..., 2] < -2.0
    white = (lab[..., 0] > 65.0) & (np.hypot(lab[..., 1], lab[..., 2]) <= 8.0)
    light = (lab[..., 0] > 40.0) & ~fill & (~cool | white)
    dog = int((~fill).sum())
    share = round(float(light.sum()) / dog, 3) if dog else None
    if int(light.sum()) < FACE_CREAM_MIN_PIXELS:
        return share, None
    return share, [round(float(v), 1) for v in np.median(lab[light], axis=0)]


# The light cheeks of a dark-headed dog checked against its own face crop
# (face_light): the face call said "cream" for a black-and-tan puppy's tan
# cheeks (photo L 56, a* 3, b* 17) and for one whose lower face is dark grey
# with a tan streak, and both came out with a pale mask (user, 11-01: "흰색이
# 애초에 아닌데 얼굴이"). Measured on nine dark-headed dogs (FACE_LOWER):
# light fur under FACE_LIGHT_MIN_SHARE of the face is a dark face (0.015-0.17;
# tan faces 0.20, cream and white 0.22-0.53 - a narrow margin, ten dogs), and
# light fur nearer the tan ramp than cream or white is tan.
FACE_LIGHT_MIN_SHARE = 0.18


# The crown in the upper face of a head crop (shares of its height and
# width), and the dark fur there: a smoky grey cap (shelter puppy 14,
# median L* 27) against black heads (8-20). A crown with dark fur on at
# least FACE_DARK_MIN_SHARE of the dog's pixels there and a median L* of
# FACE_GRAY_MIN_L or more is grey, not black.
FACE_UPPER = ((0.05, 0.40), (0.15, 0.85))
FACE_DARK_MIN_SHARE = 0.3
FACE_GRAY_MIN_L = 24.0
# ... read only on a cut-out crop (this share of it the cut's flat grey
# fill): an uncut crop's background shadows counted as dark fur, and a
# black puppy photographed in front of white clothes measured L* 35 and
# came out grey (shelter puppy 10)
FACE_CUT_MIN_SHARE = 0.05


def face_dark(crop: Image.Image) -> tuple[float | None, float | None, float, float | None, float | None]:
    """(share, median L*, cut, a*, b*) of the dark fur (L* under 55) in the
    upper face of a head crop, among the dog's pixels there; `cut` is the
    share of the whole crop that is the cut-out's flat grey fill, and a*/b*
    the dark fur's median taken relative to the crop's own white fur (L*
    over 70, near grey) when it has some, so a cast in the photo cancels."""
    from .fur import to_linear
    from .projection import linear_to_lab

    whole = np.asarray(crop.convert("RGB"), dtype=np.float32)
    cut = round(float(((np.abs(whole - whole.mean(axis=-1, keepdims=True)).max(axis=-1) < 1.0)
                       & (np.abs(whole.mean(axis=-1) - 128.0) < 3.0)).mean()), 3)
    h, w = whole.shape[:2]
    (y0, y1), (x0, x1) = FACE_UPPER
    rgb = whole[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    lab = linear_to_lab(to_linear(rgb))
    fill = (np.abs(rgb - rgb.mean(axis=-1, keepdims=True)).max(axis=-1) < 1.0)
    dark = (lab[..., 0] < 55.0) & ~fill
    dog = int((~fill).sum())
    if not dog or not dark.any():
        return None, None, cut, None, None
    tint = np.median(lab[dark][:, 1:], axis=0)
    white = (lab[..., 0] > 70.0) & ~fill & (np.hypot(lab[..., 1], lab[..., 2]) < 12.0)
    if int(white.sum()) >= FACE_CREAM_MIN_PIXELS:
        tint = tint - np.median(lab[white][:, 1:], axis=0)
    return (round(float(dark.sum()) / dog, 3), round(float(np.median(lab[..., 0][dark])), 1), cut,
            round(float(tint[0]), 1), round(float(tint[1]), 1))


# A crown read grey (lite gray_for_black) is painted the grey of its own
# dark fur: the gray ramp (mid L* 53, a touch blue) made a white puppy's
# smoky grey-brown eye patches (photo L* 27, a* +3.6 against its white fur)
# a light blue-grey cap (shelter puppy 14, user: "저런 회색이 아닐 것 - 따뜻한
# 회갈색"). The ramp moves so its mid stop is the measured fur
# GRAY_FACE_LIFT lighter (the median takes in the shadows, as a black dog's
# fur measures L* 12-16 against the black mid 7), L* within GRAY_FACE_L, its
# a*/b* within GRAY_FACE_MAX_AB of neutral.
GRAY_FACE_LIFT = 8.0
GRAY_FACE_L = (30.0, 50.0)
GRAY_FACE_MAX_AB = 6.0
# The tint is kept along the warm axis only (a fawn's hue, GRAY_WARM_HUE
# degrees), GRAY_WARM_GAIN times: taken as measured, puppy 14's (+3.6, -0.2)
# came out mauve; a cool tint is a cast, not a grey-brown coat.
GRAY_WARM_HUE = 65.0
GRAY_WARM_GAIN = 1.5


def gray_target(dark_l: float | None, tint_a: float | None, tint_b: float | None,
                lift: float = GRAY_FACE_LIFT) -> list[float] | None:
    """The gray ramp's mid LAB for a measured dark fur (GRAY_FACE_*,
    GRAY_WARM_*). `lift` 0 for a dark brindle: its lighter hairs are drawn
    over the base and lift it already."""
    if dark_l is None:
        return None
    low, high = GRAY_FACE_L
    axis = np.array([np.cos(np.radians(GRAY_WARM_HUE)), np.sin(np.radians(GRAY_WARM_HUE))])
    warmth = max(0.0, float(np.dot([tint_a or 0.0, tint_b or 0.0], axis))) * GRAY_WARM_GAIN
    a, b = np.clip(warmth * axis, -GRAY_FACE_MAX_AB, GRAY_FACE_MAX_AB)
    return [round(float(np.clip(dark_l + lift, low, high)), 1), round(float(a), 1), round(float(b), 1)]


def gray_from_face(appearance: dict, palette: dict[str, tuple[str, str, str]]) -> dict[str, tuple[str, str, str]]:
    """The palette with the gray ramp moved to appearance["gray_lab"] (every
    stop by the same LAB delta); unchanged without it."""
    from .fur import hex_to_linear, to_srgb
    from .projection import lab_to_linear, linear_to_lab

    target = appearance.get("gray_lab")
    if not target:
        return palette
    stops = palette.get("gray") or FUR_PALETTE["gray"]
    labs = [linear_to_lab(hex_to_linear(stop)[None, None, :])[0, 0] for stop in stops]
    delta = np.asarray(target, dtype=np.float32) - labs[1]
    moved = []
    for lab in labs:
        lab = np.clip(lab + delta, [0.0, -128.0, -128.0], [100.0, 127.0, 127.0]).astype(np.float32)
        rgb = to_srgb(lab_to_linear(lab[None, None, :])[0, 0])
        moved.append(_rgb_to_hex(tuple(int(round(float(np.clip(c, 0, 1)) * 255)) for c in rgb)))
    return {**palette, "gray": tuple(moved)}


def cheeks_from_photo(cheeks: str, share: float | None, lab: list[float] | None) -> str:
    """The cheek colour the face crop supports: "dark", "tan", or `cheeks`."""
    from .fur import hex_to_linear
    from .projection import linear_to_lab

    if share is not None and share < FACE_LIGHT_MIN_SHARE:
        return "dark"
    if lab is None:
        return cheeks
    def distance(name: str) -> float:
        mid = linear_to_lab(hex_to_linear(FUR_PALETTE[name][1])[None, None, :])[0, 0]
        return float(np.linalg.norm(np.asarray(lab, dtype=np.float32) - mid))
    return "tan" if distance("tan") < min(distance("cream"), distance("white")) else cheeks


def cream_from_face(appearance: dict, palette: dict[str, tuple[str, str, str]]) -> dict[str, tuple[str, str, str]]:
    """The palette with the cream ramp moved to the face's own light fur
    (appearance["face_light_lab"], set for a dark-crowned dog with cream
    cheeks); unchanged without it."""
    from .fur import hex_to_linear, to_srgb
    from .projection import lab_to_linear, linear_to_lab

    target = appearance.get("face_light_lab")
    if not target:
        return palette
    stops = palette.get("cream") or FUR_PALETTE["cream"]
    labs = [linear_to_lab(hex_to_linear(stop)[None, None, :])[0, 0] for stop in stops]
    delta = (np.asarray(target, dtype=np.float32) - labs[1]) * FACE_CREAM_STRENGTH
    delta[0] = float(np.clip(delta[0], -FACE_CREAM_MAX_SHIFT_L, 0.0))
    # warmer only: a cooler reading is the grey of a muzzle or a white glove,
    # and cream turned grey read as dirt
    delta[1:] = np.clip(delta[1:], 0.0, FACE_CREAM_MAX_SHIFT_AB)
    if delta[0] == 0.0:
        return palette
    moved = []
    for lab in labs:
        lab = (lab + delta).astype(np.float32)
        rgb = to_srgb(lab_to_linear(lab[None, None, :])[0, 0])
        moved.append(_rgb_to_hex(tuple(int(round(float(c) * 255)) for c in rgb)))
    return {**palette, "cream": tuple(moved)}


def relative_cream(appearance: dict, palette: dict[str, tuple[str, str, str]]) -> dict[str, tuple[str, str, str]]:
    """The palette with a cream between the base and the table's cream, when
    the base is warm and some region is cream; unchanged otherwise."""
    from .fur import hex_to_linear, to_srgb
    from .projection import lab_to_linear, linear_to_lab

    base = str(appearance.get("base_color") or "").lower()
    uses_cream = any(str(e.get("color", "")).lower() == "cream" or str(e.get("gradient_to", "")).lower() == "cream"
                     for e in appearance.get("region_colors") or [])
    if base not in WARM_BASES or not uses_cream:
        return palette
    base_stops = palette.get(base) or FUR_PALETTE.get(base)
    cream_stops = FUR_PALETTE["cream"]
    # the body call's "how much lighter" (body_colours.CREAM_SHARE), else the default
    share = float(appearance.get("cream_share") or CREAM_ON_WARM)
    mixed = []
    for b, c in zip(base_stops, cream_stops):
        lb = linear_to_lab(hex_to_linear(b)[None, None, :])[0, 0]
        lc = linear_to_lab(hex_to_linear(c)[None, None, :])[0, 0]
        lab = (lb + share * (lc - lb)).astype(np.float32)
        rgb = to_srgb(lab_to_linear(lab[None, None, :])[0, 0])
        mixed.append(_rgb_to_hex(tuple(int(round(float(x) * 255)) for x in rgb)))
    return {**palette, "cream": tuple(mixed)}


# A warm ramp under a neutral border keeps at least this share of the table's
# chroma, and its hue turns at most this many degrees from the table's.
# Neither, now (11-01): the measured cells of yellow shelter puppies read hue
# 78-79 and chroma 20 where the photos' coats measure hue 67-69 and chroma
# 20-30 (pixels of the whole coat), and the turn plus the greying painted
# the golden puppies khaki beige; the table's hue and chroma are kept and the
# lightness alone follows the measurement.
WARM_MIN_CHROMA_SHARE = 1.0
WARM_MAX_HUE_TURN = 0.0


def _warm_chroma_shift(mid: np.ndarray, target: np.ndarray, strength: float, max_shift_ab: float) -> np.ndarray:
    """The a*/b* shift of a warm ramp (tan, red, fawn, chocolate) under a
    neutral border: the usual shift, then held to the table's chroma and hue
    within WARM_MIN_CHROMA_SHARE and WARM_MAX_HUE_TURN. A warm coat's
    measured colour is greyed and turned by the room's light: on e20 (small
    tan dog, beige-and-white room) the leg read a* -1, chroma 15 against the
    table fawn's 28, and the free shift turned the whole fawn ramp
    olive-grey (chroma 19, hue 86 against 76). The lightness still follows
    the measurement, and a coat measured more saturated is still followed."""
    free = np.clip((target[1:] - mid[1:]) * strength, -max_shift_ab, max_shift_ab)
    table_chroma = float(np.hypot(mid[1], mid[2]))
    if table_chroma < 1e-3:
        return free.astype(np.float32)
    ab = mid[1:] + free
    chroma = max(float(np.hypot(ab[0], ab[1])), WARM_MIN_CHROMA_SHARE * table_chroma)
    table_hue = float(np.arctan2(mid[2], mid[1]))
    turn = (float(np.arctan2(ab[1], ab[0])) - table_hue + np.pi) % (2 * np.pi) - np.pi
    hue = table_hue + float(np.clip(turn, -np.radians(WARM_MAX_HUE_TURN), np.radians(WARM_MAX_HUE_TURN)))
    return (np.array([chroma * np.cos(hue), chroma * np.sin(hue)]) - mid[1:]).astype(np.float32)


def ramps_from_measurement(
    appearance: dict, measured: dict, strength: float = MEASURED_STRENGTH,
    max_shift_l: float = MEASURED_MAX_SHIFT_L, max_shift_ab: float = MEASURED_MAX_SHIFT_AB,
) -> tuple[dict[str, tuple[str, str, str]], dict]:
    """(palette, report): the hued ramps the appearance names, each shifted
    in LAB toward the measured colour of the regions that carry it. Only
    fully-covered regions count, and only names with a measurement."""
    from .fur import hex_to_linear, to_srgb
    from .projection import lab_to_linear, linear_to_lab

    by_name: dict[str, list[np.ndarray]] = {}
    skipped: list[str] = []
    for entry in appearance.get("region_colors") or []:
        name = str(entry.get("color", "")).strip().lower()
        region = entry.get("region")
        if name not in MEASURED_NAMES or str(entry.get("coverage", "full")) != "full":
            continue
        reading = measured.get(region) if isinstance(measured, dict) else None
        if reading and reading.get("lab"):
            lab = np.asarray(reading["lab"], dtype=np.float32)
            if _plausible_hue(lab):
                by_name.setdefault(name, []).append(lab)
            else:
                skipped.append(f"{region}:{name}@L{lab[0]:.0f}")

    # a photo whose border was no grey card was not white-balanced, and the
    # surroundings tint the fur too: e18's red studio wall made the tan coat
    # salmon even inside the hue range, and letting grass and dirt borders
    # ("natural") keep the measured hue was tried on 2026-09-30 and dropped -
    # e07's tan went olive (a* -10, grass reflected in the fur) and e05's
    # dull brown (shaded dirt), while the yellow jindo on grass it was meant
    # for (fixture 2222) had measured the table's fawn anyway. Lightness only
    # unless the border is neutral.
    kinds = {str(r.get("border", "neutral" if r.get("neutral_border", True) else "natural"))
             for r in (measured or {}).values() if isinstance(r, dict)}
    tinted = bool(kinds - {"neutral"})
    palette: dict[str, tuple[str, str, str]] = {}
    report: dict = {"skipped": skipped} if skipped else {}
    if tinted:
        report["hue_kept"] = ("saturated backdrop" if "backdrop" in kinds else "border not neutral") + ": lightness only"
    for name, labs in by_name.items():
        stops = FUR_PALETTE.get(name)
        if not stops:
            continue
        target = np.median(np.stack(labs), axis=0)
        stop_labs = [linear_to_lab(hex_to_linear(stop)[None, None, :])[0, 0] for stop in stops]
        delta = (target - stop_labs[1]) * strength
        delta[0] = float(np.clip(delta[0], -max_shift_l, min(max_shift_l, MEASURED_MAX_LIGHTEN_L)))
        if tinted:
            delta[1:] = 0.0
        elif name in WARM_BASES:
            delta[1:] = _warm_chroma_shift(stop_labs[1], target, strength, max_shift_ab)
        else:
            delta[1:] = np.clip(delta[1:], -max_shift_ab, max_shift_ab)
        moved = []
        for lab in stop_labs:
            rgb = to_srgb(lab_to_linear((lab + delta).astype(np.float32)[None, None, :])[0, 0])
            moved.append(_rgb_to_hex(tuple(int(round(float(c) * 255)) for c in rgb)))
        palette[name] = tuple(moved)
        report[name] = {"measured": [round(float(x), 1) for x in target],
                        "table_mid": [round(float(x), 1) for x in stop_labs[1]],
                        "shift": [round(float(x), 1) for x in delta], "regions": len(labs),
                        "ramp": palette[name]}
    return palette, report


__all__ = [
    "PhotoStats", "apply_photo_colours", "build_palette", "calibrate_color",
    "chroma_scale", "measure", "palette_for", "photo_colours", "ramps_from_measurement", "relative_chroma",
]
