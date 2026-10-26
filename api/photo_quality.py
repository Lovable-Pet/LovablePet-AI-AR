"""Stage 0b: how much the photograph can be trusted, measured locally.

Nothing upstream judges quality: the gate is told not to, the colour calls
take the picture at face value, and a dark, blurred or thumbnail-sized photo
goes through like a studio shot and comes out as a wobbly reading. Three
measurements on the cut-out animal, no model call:

  size      the animal's box, short side in pixels of the original. The
            colour calls see the cut-out at most ~768 wide; below ~150 px
            the fur has no texture left and the face is a few blocks.
  sharpness variance of the Laplacian of luminance inside the animal, with
            the animal first scaled to a fixed width so the number does not
            grow with resolution. Fur is high-frequency: a focused coat
            scores in the hundreds, motion blur or a heavy upscale in the
            tens. The raw variance also falls when a photo is merely dim
            (every gradient is smaller), so a second, contrast-free form
            (variance / luminance variance) is taken too and the photo gets
            the BETTER of the two verdicts - a dim-but-focused photo is not
            called blurred, and a soft web JPEG with strong contrast is not
            either. Both have to say blurred.
  exposure  share of animal pixels crushed to black or blown to white. A
            black dog in a dim room is dark by nature (e06 reads 9 %
            crushed, the black shiba fixture 19 %), so only the CLIPPED
            share counts, and the fair line sits above those: fur at L 0
            or L 255 carries no colour at all. Plain underexposure without
            clipping is not caught here - a dim white dog and a black dog
            look alike to a local statistic.

Each measurement maps to good / fair / poor with the thresholds below, the
worst one is the photo's level, and fit.py turns a fair or poor level into
a photo check with a Korean reason for the shelter. Thresholds come from the
eval set (tests/evalset, 20 shelter photos: every original reads good) and
the same photos degraded on purpose - scaled to 1/3 and 1/6, gaussian blur
2 and 4 px, brightness x0.3 and x2.5 (scratch calibration, 2026-10-06):
1/6 scale and 4 px blur are poor on every photo under 1500 px, 1/3 scale is
fair-to-poor, x2.5 blows out 15 of 20, x0.3 crushes only the dark dogs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

# Width the animal is scaled to before the sharpness measurement.
SHARP_WIDTH = 400
# Short side of the animal's box, original pixels.
SIZE_FAIR, SIZE_POOR = 220, 120
# Laplacian variance at SHARP_WIDTH, and the contrast-free ratio.
SHARP_FAIR, SHARP_POOR = 60.0, 25.0
RATIO_FAIR, RATIO_POOR = 0.035, 0.015
# Share of animal pixels at the luminance limits.
CLIP_FAIR, CLIP_POOR = 0.20, 0.40
# Luminance (0..255) counted as crushed / blown.
BLACK_L, WHITE_L = 8, 247

LEVELS = ("good", "fair", "poor")


def _level(value: float, fair: float, poor: float, higher_is_better: bool) -> str:
    if higher_is_better:
        return "good" if value >= fair else "fair" if value >= poor else "poor"
    return "good" if value <= fair else "fair" if value <= poor else "poor"


def _laplacian_variance(gray: np.ndarray, mask: np.ndarray) -> float:
    from scipy import ndimage

    lap = ndimage.laplace(gray.astype(np.float64))
    inside = ndimage.binary_erosion(mask, iterations=3)
    if inside.sum() < 100:
        inside = mask
    if inside.sum() < 10:
        return 0.0
    return float(lap[inside].var())


def measure_quality(image: Image.Image, mask: np.ndarray | None = None,
                    box: tuple[int, int, int, int] | None = None) -> dict[str, Any]:
    """`image` is the frame the measurements are taken on (the cut-out, or
    the original), `mask` where the animal is in it (None: the whole
    frame), `box` the animal's box in ORIGINAL pixels for the size measure
    (None: the mask's box in `image`)."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = rgb.shape[:2]
    if mask is None:
        mask = np.ones((height, width), dtype=bool)
    mask = mask.astype(bool)
    if mask.sum() < 10:
        mask = np.ones((height, width), dtype=bool)
    rows, cols = np.nonzero(mask)
    mx0, my0, mx1, my1 = int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1
    if box is None:
        box = (mx0, my0, mx1, my1)
    short_side = int(min(box[2] - box[0], box[3] - box[1]))

    # sharpness at a fixed width of the animal's box
    crop = Image.fromarray(rgb[my0:my1, mx0:mx1].astype(np.uint8), mode="RGB").convert("L")
    crop_mask = Image.fromarray((mask[my0:my1, mx0:mx1] * 255).astype(np.uint8), mode="L")
    scale = SHARP_WIDTH / max(crop.size[0], 1)
    size = (max(1, int(round(crop.size[0] * scale))), max(1, int(round(crop.size[1] * scale))))
    gray = np.asarray(crop.resize(size, Image.LANCZOS), dtype=np.float32)
    small_mask = np.asarray(crop_mask.resize(size, Image.NEAREST)) > 127
    sharpness = _laplacian_variance(gray, small_mask)
    contrast = float(gray[small_mask].std()) if small_mask.sum() >= 10 else 0.0
    sharpness_ratio = sharpness / max(contrast * contrast, 1e-3)

    # exposure on the animal's own pixels at full resolution
    luma = rgb @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    animal = luma[mask]
    clipped = float(((animal <= BLACK_L) | (animal >= WHITE_L)).mean())
    dark_share = float((animal <= BLACK_L).mean())
    median_l = float(np.median(animal))

    raw = _level(sharpness, SHARP_FAIR, SHARP_POOR, True)
    ratio = _level(sharpness_ratio, RATIO_FAIR, RATIO_POOR, True)
    levels = {
        "size": _level(short_side, SIZE_FAIR, SIZE_POOR, True),
        "sharpness": min(raw, ratio, key=LEVELS.index),
        "exposure": _level(clipped, CLIP_FAIR, CLIP_POOR, False),
    }
    level = max(levels.values(), key=LEVELS.index)
    return {
        "level": level,
        "levels": levels,
        "short_side_px": short_side,
        "sharpness": round(sharpness, 1),
        "sharpness_ratio": round(sharpness_ratio, 4),
        "contrast": round(contrast, 1),
        "clipped_share": round(clipped, 3),
        "dark_share": round(dark_share, 3),
        "median_luma": round(median_l, 1),
    }


def quality_of_file(path, use_rembg: bool = True) -> dict[str, Any]:
    """Measure a photograph on disk (EXIF rotation applied), cutting the
    animal out with rembg when available so the measurement is of the fur,
    not the floor."""
    from PIL import ImageOps

    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    mask = None
    if use_rembg:
        from .imagegen import _rembg_alpha

        alpha = _rembg_alpha(image)
        if alpha is not None and (alpha > 0.5).mean() >= 0.01:
            mask = alpha > 0.5
    return measure_quality(image, mask)


# Korean reasons for the shelter, by measurement and level.
REASONS = {
    ("size", "fair"): "사진 속 개가 작게 찍힘(짧은 변 {short_side_px}px): 털 결과 얼굴 세부가 흐릿하게 나올 수 있음",
    ("size", "poor"): "사진 속 개가 너무 작음(짧은 변 {short_side_px}px): 색과 얼굴을 읽기 어려움, 개가 크게 나온 사진 필요",
    ("sharpness", "fair"): "사진이 약간 흐림: 털 결과 얼굴 세부가 뭉개질 수 있음",
    ("sharpness", "poor"): "사진이 많이 흐림(초점이나 흔들림): 색은 읽히지만 얼굴과 무늬를 믿기 어려움, 선명한 사진 필요",
    ("exposure", "fair"): "사진이 너무 어둡거나 밝음(털의 {clipped_pct}%가 완전 검정/흰색): 색이 실제와 다르게 나올 수 있음",
    ("exposure", "poor"): "사진 노출이 나쁨(털의 {clipped_pct}%가 완전 검정/흰색): 색을 믿을 수 없음, 밝은 곳에서 찍은 사진 필요",
}


def describe_quality(quality: dict[str, Any] | None) -> list[tuple[str, str, str]]:
    """(measurement, level, Korean reason) for every measurement that is
    not good, worst first."""
    if not quality:
        return []
    out = []
    for key, level in sorted(quality.get("levels", {}).items(), key=lambda kv: -LEVELS.index(kv[1])):
        if level == "good":
            continue
        text = REASONS[(key, level)].format(short_side_px=quality.get("short_side_px", 0),
                                            clipped_pct=round(100 * quality.get("clipped_share", 0.0)))
        out.append((key, level, text))
    return out


def quality_view(quality: dict[str, Any] | None) -> dict[str, Any] | None:
    """The API's view: the level, each measurement's level and the numbers
    behind them."""
    if not quality:
        return None
    return {"level": quality["level"], "levels": dict(quality["levels"]),
            "short_side_px": quality["short_side_px"], "sharpness": quality["sharpness"],
            "clipped_share": quality["clipped_share"]}


__all__ = ["LEVELS", "describe_quality", "measure_quality", "quality_of_file", "quality_view"]
