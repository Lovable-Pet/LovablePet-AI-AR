"""Parts mode - every region read from the photo, every colour painted by mask.

THE TWO HALVES
--------------
Reading. No list of "up to four features". The pattern classifier gives a
ten-region estimate, and the region agents (a face call over the head
crop, a body call over the whole photo, three samples each, majority vote)
correct it region by region. A region no reading contradicts keeps the
layout's colour; the log says which were read and which were assumed.

Painting. The atlas is not repainted in one go - asked for a whole coat the
image model paints "roughly this kind of dog" and the small regions land
where they may. Instead the regions are grouped by target colour and each
group is one masked edit: the atlas goes in whole, the mask opens only that
group's regions, and the model paints fur of that one colour there,
matched to the fur around it. Everything outside the mask is restored from
the input exactly, so each group leaves the others untouched. The base
colour's group goes first (it is the largest), then the rest by size. Two
to four calls per dog, each with one job.

The template's own atlas is the starting point, so the hair, the eyes and
the layout are the artist's; only the fur colour changes, region by region.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .fix_region import fix_regions
from .fur import FUR_PALETTE
from .regions import REGIONS


def colour_groups(appearance: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """(colour, regions) in painting order: the base colour first, then the
    others by how many regions they cover."""
    groups: dict[str, list[str]] = defaultdict(list)
    for entry in appearance.get("region_colors") or []:
        region, colour = entry.get("region"), entry.get("color")
        if region in REGIONS and colour in FUR_PALETTE:
            groups[colour].append(region)
    base = appearance.get("base_color")
    ordered = sorted(groups.items(), key=lambda item: (item[0] != base, -len(item[1])))
    return [(colour, [r for r in REGIONS if r in regions]) for colour, regions in ordered]


async def paint_by_parts(
    transport,
    base_texture: Image.Image,
    photo_path: str | Path,
    appearance: dict[str, Any],
    masks_dir: str | Path,
    valid: np.ndarray,
    work_dir: str | Path | None = None,
    use_cache: bool = True,
) -> tuple[Image.Image, dict[str, Any]]:
    """One masked edit per colour group, each starting from the previous
    result. Returns the unfinished atlas and a report per group."""
    atlas = base_texture.convert("RGB")
    report: dict[str, Any] = {"groups": []}
    for index, (colour, regions) in enumerate(colour_groups(appearance)):
        atlas, entry = await fix_regions(transport, atlas, photo_path, appearance, regions, masks_dir, valid,
                                         use_cache=use_cache)
        entry = {k: v for k, v in entry.items() if k != "prompt"}
        entry.update({"step": index, "colour": colour})
        report["groups"].append(entry)
        if work_dir is not None:
            Path(work_dir).mkdir(parents=True, exist_ok=True)
            atlas.save(Path(work_dir) / f"step{index}_{colour}.png")
    return atlas, report


__all__ = ["colour_groups", "paint_by_parts"]
