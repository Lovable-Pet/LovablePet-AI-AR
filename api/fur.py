"""Fur colours as three-stop ramps, and the gradient map that applies them.

WHY A RAMP AND NOT A COLOUR
---------------------------
Fur is not one colour. A single hair runs dark at the root and pale at the tip,
and a coat reads as fur precisely because of that spread. Painting a region one
RGB value and modulating it by brightness gives a plastic surface: the hue is
constant and only the exposure moves. Mapping the template's brightness through
a dark/mid/light ramp instead lets the shadowed undercoat and the lit guard
hairs be genuinely different colours, which is what the eye uses to tell fur
from paint.

WHY NEITHER END IS PURE
-----------------------
Nothing on an animal is #000000 or #FFFFFF. Black fur in shade still returns a
little light and still has a hue; white fur is warm and takes on its
surroundings. Pure values also destroy information - a pure black ramp makes
every shadow identical, so the fur detail underneath disappears - so the ramps
here bottom out around #0D0B0A and top out around #FAF8F4.

LINEAR SPACE
------------
Every operation runs on linear values. sRGB is perceptually spaced, so blending
two colours in it lands somewhere neither belongs: mid-grey between black and
white comes out far too light, and a gradient between two fur tones develops a
dead band in the middle. Decode on load, encode on save, and everything in
between is linear.
"""

from __future__ import annotations

import numpy as np

GAMMA = 2.2

# The only colour names the pipeline accepts. Free-text colours from a model
# cannot be turned into a ramp, and "dark brown" against "brown" against
# "chocolate" is a distinction no texture can honour anyway.
FUR_COLORS = (
    "black",
    "white",
    "cream",
    "fawn",
    "red",
    "tan",
    "chocolate",
    "gray",
    # bare skin, not fur: the inside of a thin ear, a pink nose, a sparsely
    # furred belly. Without it a white dog's pink ear is answered "red" (the
    # nearest fur word) and gets painted orange.
    "pink",
)

# (dark, mid, light) per colour: the undercoat in shadow, the body of the coat,
# and the lit tips. Warm and desaturated throughout, because a coat photographed
# indoors never reaches the saturation a colour picker suggests.
FUR_PALETTE: dict[str, tuple[str, str, str]] = {
    # never #000000: black fur keeps a warm cast and needs range for the
    # detail underneath to survive
    # the lit tips stay well below mid-grey: a black coat under a bright
    # key light was rendering charcoal, not black
    "black": ("#0A0908", "#1F1B18", "#3E3732"),
    # never #FFFFFF: white fur is an off-white that picks up its surroundings
    "white": ("#D9D3C9", "#EEE9E1", "#FAF8F4"),
    # off-white with a warm tint, a step below white - not beige. The earlier
    # ramp (#8A7E68 / #D8CBAE / #F2EADA: mid L 82, shadow L 54) painted the
    # cream cheeks of a black-and-tan shiba as beige-tan against a photo
    # whose cheek measures LAB (90, 0, 9); the vision read said "cream"
    # meaning off-white. Mid L 88 b 11, shadow L 72.
    "cream": ("#B8AC94", "#E7DCC5", "#F6F0E4"),
    "fawn": ("#7A6247", "#C4A177", "#E6D2B4"),
    "red": ("#6B3A22", "#B5672F", "#DCA268"),
    "tan": ("#6E4526", "#B07A42", "#D6A876"),
    "chocolate": ("#2A1A12", "#4E3325", "#8A6448"),
    "gray": ("#3A3A3C", "#7E7E82", "#BEBEC2"),
    # skin showing through fine hair: desaturated and light, never a saturated
    # pink, because it is always seen through some hair
    "pink": ("#A8837C", "#D8B3AA", "#EFD8D1"),
}

DEFAULT_COLOR = "fawn"

# Where the brightness map's own distribution is pinned to the ramp. Using
# percentiles rather than absolute levels means the ramp lands correctly however
# the brightness map was normalised, and a flat template still gets a full
# gradient instead of collapsing onto one stop.
RAMP_LOW, RAMP_MID, RAMP_HIGH = 5.0, 50.0, 95.0


def hex_to_linear(value: str) -> np.ndarray:
    """A hex colour as linear RGB in 0..1."""
    text = value.lstrip("#")
    srgb = np.array([int(text[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0
    return srgb ** GAMMA


def to_linear(srgb: np.ndarray) -> np.ndarray:
    """Decode 0..255 sRGB (or 0..1) into linear."""
    values = srgb.astype(np.float32)
    if values.max() > 1.0:
        values = values / 255.0
    return np.clip(values, 0.0, 1.0) ** GAMMA


def to_srgb(linear: np.ndarray) -> np.ndarray:
    """Encode linear back to 0..1 sRGB."""
    return np.clip(linear, 0.0, 1.0) ** (1.0 / GAMMA)


def ramp_for(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The (dark, mid, light) ramp for a colour name, in linear RGB."""
    key = (name or "").strip().lower()
    stops = FUR_PALETTE.get(key) or FUR_PALETTE[DEFAULT_COLOR]
    return tuple(hex_to_linear(stop) for stop in stops)


def gradient_map(
    gray: np.ndarray,
    dark: np.ndarray,
    mid: np.ndarray,
    light: np.ndarray,
    region: np.ndarray | None = None,
) -> np.ndarray:
    """Map a brightness map through a three-stop ramp, in linear space.

    The percentiles that pin the ramp are taken over the WHOLE map, never
    inside one region. Re-fitting them per region looks reasonable - every
    region then gets the full ramp - but it rescales whatever variation that
    region happens to have up to the ramp's full width, so a back that varies
    by a few percent in the template comes out with visible light and dark
    patches of black. Measured, that one choice raised the low-frequency spread
    of the back from 0.017 to 0.090: the blotching was the ramp amplifying
    noise, not the noise itself.

    Fitting globally means a uniformly dark region receives a narrow slice of
    the ramp and stays uniform, which is both what the eye expects and what the
    brightness map actually says.

    `region` is accepted and ignored so callers can pass the mask they are
    working on without the signature implying it changes the fit.
    """
    low, middle, high = np.percentile(gray, [RAMP_LOW, RAMP_MID, RAMP_HIGH])
    if high - low < 1e-4:
        low, middle, high = 0.0, 0.5, 1.0
    middle = min(max(middle, low + 1e-4), high - 1e-4)

    lower = np.clip((gray - low) / (middle - low), 0.0, 1.0)
    upper = np.clip((gray - middle) / (high - middle), 0.0, 1.0)

    blended = dark + (mid - dark) * lower[..., None]
    return blended + (light - mid) * upper[..., None]


def normalize_brightness(gray: np.ndarray, target: float = 210.0 / 255.0) -> np.ndarray:
    """Pin a brightness map's mean.

    A multiply blend can only ever darken, which is why it was abandoned; this
    map is the input to a gradient instead, and pinning its mean keeps the
    gradient's input distribution stable across templates.
    """
    mean = float(gray.mean())
    if mean <= 1e-4:
        return gray
    return np.clip(gray * (target / mean), 0.0, 1.0)




# --------------------------------------------------------------------------
# fur detail: anisotropic noise along the hair, and the normal map it implies
# --------------------------------------------------------------------------

# Brightness jitter of the hair overlay, as a share: +-5%.
FUR_JITTER = 0.03
# How far along the hair direction one noise sample is smeared, in texels.
FUR_STREAK_PX = 10
# Size of the noise grain before smearing, in texels.
FUR_GRAIN_PX = 1.5
# How strongly the streaks tilt the normal.
FUR_NORMAL_STRENGTH = 0.2


def read_direction(path) -> tuple[np.ndarray, np.ndarray]:
    """fur_direction.png -> (direction (h, w, 2) unit vectors in image space,
    defined (h, w) bool). See ar_pipeline/build_landmarks.py for the encoding."""
    from PIL import Image

    data = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    direction = np.stack([data[..., 0] / 255.0 * 2 - 1, data[..., 1] / 255.0 * 2 - 1], axis=-1)
    defined = data[..., 2] > 127
    length = np.linalg.norm(direction, axis=-1, keepdims=True)
    direction = np.where(length > 1e-6, direction / np.maximum(length, 1e-6), 0.0)
    return direction.astype(np.float32), defined


def anisotropic_noise(shape: tuple[int, int], direction: np.ndarray, seed: int,
                      streak: int = FUR_STREAK_PX, grain: float = FUR_GRAIN_PX) -> np.ndarray:
    """Zero-mean, unit-ish noise smeared along `direction`: a line-integral
    convolution of blurred white noise, which is what individual hairs look
    like at texture scale. Where the direction is undefined the noise is
    isotropic, so the seam between the two never shows as a hard line."""
    from scipy.ndimage import gaussian_filter, map_coordinates

    rng = np.random.default_rng(seed)
    height, width = shape
    base = gaussian_filter(rng.standard_normal(shape).astype(np.float32), grain)
    base /= max(float(base.std()), 1e-6)

    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    total = np.zeros(shape, dtype=np.float32)
    steps = np.arange(-streak, streak + 1, dtype=np.float32)
    for step in steps:
        coords = [ys + direction[..., 1] * step, xs + direction[..., 0] * step]
        total += map_coordinates(base, coords, order=1, mode="reflect")
    total /= len(steps)
    total /= max(float(total.std()), 1e-6)
    return total.astype(np.float32)


def fur_multiplier(noise: np.ndarray, jitter: float = FUR_JITTER) -> np.ndarray:
    """The brightness factor the hair overlay applies: 1 +- jitter, with the
    noise's tails clipped so no single hair blows out."""
    return (1.0 + jitter * np.clip(noise, -2.0, 2.0) / 2.0).astype(np.float32)


def fur_normal_map(noise: np.ndarray, strength: float = FUR_NORMAL_STRENGTH,
                   base_normal: np.ndarray | None = None) -> np.ndarray:
    """A tangent-space normal map (h, w, 3) in 0..1 from the hair height field,
    blended over the template's own normal map when one is given.

    glTF normal maps are OpenGL-style: +x right, +y UP in texture space, +z
    out. Image rows run downward, so the y slope is negated. The blend is
    the whitened UDN sum - the two maps' x/y offsets add and z is renormalised
    - which keeps the template's large-scale relief under the fur streaks."""
    dx = np.gradient(noise, axis=1)
    dy = np.gradient(noise, axis=0)
    nx = -dx * strength
    ny = dy * strength
    if base_normal is not None:
        base = base_normal.astype(np.float32) * 2.0 - 1.0
        nx = nx + base[..., 0]
        ny = ny + base[..., 1]
        nz = np.maximum(base[..., 2], 0.2)
    else:
        nz = np.ones_like(nx)
    normal = np.stack([nx, ny, nz], axis=-1)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6)
    return ((normal + 1.0) / 2.0).astype(np.float32)


__all__ = [
    "DEFAULT_COLOR", "FUR_COLORS", "FUR_PALETTE", "GAMMA", "anisotropic_noise",
    "fur_multiplier", "fur_normal_map", "gradient_map", "hex_to_linear",
    "normalize_brightness", "ramp_for", "read_direction", "to_linear", "to_srgb",
]
