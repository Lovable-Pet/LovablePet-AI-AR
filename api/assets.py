"""Stage 3 end to end: an appearance result in, a wearable GLB out.

The pet's file is keyed by 관리번호, matching metadata.asset_path(). That
function returns the URL the client fetches (/assets/pets/{petId}.glb); this
one writes the file, so it needs a real directory to write into. The two are
deliberately separate - where the bytes live on disk is a deployment question
and the served path must not change with it.

DEFAULT AND EXPERIMENTAL
------------------------
The default path composes the coat from the region masks (texture.py), with
the photograph's face layered in when face_photo.py produced one. The
generated-view projection - image model draws four views, they are checked
and projected onto the mesh (imagegen/luna, consistency, projection) - is
kept behind `experimental=True`: it is not run, and its inputs are not
looked for, unless asked.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .glb import animation_names, extract_base_texture, extract_normal_texture, replace_base_texture
from .merge import supersede_markings
from .palette import palette_for
from .patterns import complete_appearance, describe, match_pattern
from .behavior import resolve_behavior
from .texture import TextureResult, compose_texture

# Where the experimental projection path's inputs live.
CANONICAL_DIR = Path("output/canonical")
GEOMETRY_DIR = Path("output/geometry")


def geometry_dir_for(masks_dir: str | Path) -> Path:
    """The baked geometry (position/normal/valid maps) for a template: its
    own output/templates/<slug>/geometry when it has one, else the shared
    output/geometry the shiba template was baked to."""
    own = Path(masks_dir).parent / "geometry"
    return own if (own / "geometry.json").exists() else GEOMETRY_DIR
VIEW_NAMES = ("front", "left", "right", "back")

# Where the built files land locally. metadata.ASSET_ROOT is the URL prefix.
ASSET_DIR = Path("output/assets/pets")

# The renderer every preview and anchor goes through. Overridable so a
# machine with Blender elsewhere does not have to edit code.
BLENDER = os.environ.get(
    "BLENDER_EXE", "C:/Program Files/Blender Foundation/Blender 5.2/blender.exe"
)


@dataclass(frozen=True)
class BuiltAsset:
    pet_id: str
    glb_path: Path
    texture_path: Path
    served_path: str
    behavior: dict[str, Any]
    normal_path: Path | None = None
    report: dict[str, Any] = field(default_factory=dict)


def template_paths(base_template: str, templates_root: str | Path = "output/templates") -> tuple[Path, Path]:
    """Locate the template GLB and its mask folder from the stage-2 filename.

    base_template is "skeleton_{species}_{size}.glb" and the masks live in
    templates/{species}_{size}/, so the one name settles both.
    """
    stem = Path(base_template).stem
    if not stem.startswith("skeleton_"):
        raise ValueError(f"expected skeleton_{{species}}_{{size}}.glb, got {base_template!r}")
    slug = stem[len("skeleton_"):]
    glb = Path("output") / base_template
    if not glb.exists():
        # one body per template, sized by weight at the end (size.py): a
        # size class without its own build uses the one that exists
        name = slug.rsplit("_", 1)[0]
        for size in ("small", "medium", "large"):
            other = Path("output") / f"skeleton_{name}_{size}.glb"
            if other.exists():
                glb, slug = other, f"{name}_{size}"
                break
    masks = Path(templates_root) / slug / "masks"
    return glb, masks


def _legacy_complete(appearance: dict[str, Any]) -> dict[str, Any]:
    """The old pattern-table completion, for an appearance that carries no
    full-region estimate (stage-2 output from before the classifier)."""
    match = match_pattern(appearance.get("region_colors") or [])
    appearance, filled = complete_appearance(appearance)
    print(f"[pattern] {describe(match)}")
    if filled:
        kept, superseded = supersede_markings(
            appearance["region_colors"], appearance.get("markings") or []
        )
        if superseded:
            print(f"[pattern] dropped {[m.get('region') for m in superseded]} "
                  f"marking(s) now covered by a filled region colour")
        appearance = {**appearance, "markings": kept,
                      "inferred_regions": [entry["region"] for entry in filled]}
    return appearance


def build_pet_asset(
    appearance: dict[str, Any],
    asset_dir: str | Path = ASSET_DIR,
    debug_dir: str | Path | None = None,
    complete_pattern: bool = True,
    photos: list[str | Path] | None = None,
    face_layer: tuple[np.ndarray, np.ndarray] | None = None,
    geometry_dir: str | Path = GEOMETRY_DIR,
    experimental: bool = False,
    views_dir: str | Path | None = None,
    landmarks: str | Path | dict | None = None,
    palette: dict[str, tuple[str, str, str]] | None = None,
    face_triangle_ratio: float | None = None,
    label_layer: tuple[np.ndarray, np.ndarray, list[str]] | None = None,
    feather_all: bool = False,
) -> BuiltAsset:
    """Compose the coat, write it into a copy of the template, and report the
    behavior settings that go with it.

    The base texture is read out of the template rather than taken as a path,
    so the texture the coat is painted over is always the one that GLB actually
    ships with. `face_layer` is (linear rgb, weight) from face_photo.py.
    `palette` given skips the photo calibration (the critic loop reuses one).
    """
    pet_id = str(appearance.get("pet_id") or appearance.get("petId") or "")
    if not pet_id:
        raise ValueError("appearance has no pet_id / petId to key the asset by")

    has_estimate = any(
        str(entry.get("source", "")).startswith("pattern") for entry in appearance.get("region_colors") or []
    ) or len(appearance.get("region_colors") or []) >= len(__import__("api.regions", fromlist=["REGIONS"]).REGIONS)
    if complete_pattern and not has_estimate:
        appearance = _legacy_complete(appearance)

    template_glb, masks_dir = template_paths(appearance["base_template"])
    template_dir = masks_dir.parent

    asset_dir = Path(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    texture_path = asset_dir / f"{pet_id}.png"
    normal_path = asset_dir / f"{pet_id}_normal.png"

    base_texture = extract_base_texture(template_glb)
    base_texture_path = asset_dir / f"{pet_id}_template_base.png"
    base_texture.save(base_texture_path)
    base_normal = extract_normal_texture(template_glb)

    palette_story = None
    if palette is None:
        if photos:
            palette, stats, palette_story = palette_for(appearance, list(photos))
            print(f"[palette] {palette_story}")
        else:
            print("[palette] no photos given; using the name table uncalibrated")
            palette = None

    report: dict[str, Any] = {"path": "masks", "palette": palette_story}
    normal_image = None

    projection_ready = experimental and (
        views_dir is not None
        and all((Path(views_dir) / f"{name}.png").exists() for name in VIEW_NAMES)
        and (CANONICAL_DIR / "cameras.json").exists()
        and (Path(geometry_dir) / "geometry.json").exists()
    )
    if projection_ready:
        # EXPERIMENTAL: the generated views placed on the mesh, delit, and the
        # mask build filling whatever no view reached
        from .projection import compose_projected

        consistency = Path(views_dir) / "consistency.json"
        score = None
        if consistency.exists():
            import json as _json
            score = _json.loads(consistency.read_text(encoding="utf-8")).get("score")

        face_photo = None
        if photos:
            from .face_photo import FacePhoto

            source = landmarks if landmarks is not None else Path(views_dir) / "landmarks.json"
            if isinstance(source, dict):
                face_photo = FacePhoto(Path(photos[0]), {
                    k: (float(v["x"]), float(v["y"])) for k, v in source.items()
                    if isinstance(v, dict) and "x" in v})
            elif Path(source).exists():
                face_photo = FacePhoto.load(photos[0], source)
        coat, projection_report = compose_projected(
            base_texture_path, masks_dir, appearance, CANONICAL_DIR, geometry_dir, views_dir,
            palette=palette, consistency_score=score, debug_dir=debug_dir, face_photo=face_photo,
        )
        report.update({"path": "projection (experimental)", "projection": projection_report})
        print(f"[assets] experimental projection path: {projection_report['coverage_of_valid']:.0%} "
              f"of the surface from the views")
    else:
        if experimental and views_dir is not None:
            print("[assets] experimental projection inputs missing; using the mask build")
        result: TextureResult = compose_texture(
            base_texture_path, masks_dir, appearance, debug_dir=debug_dir, palette=palette,
            template_dir=template_dir, geometry_dir=geometry_dir, mesh_glb=template_glb,
            face_layer=face_layer, base_normal=base_normal,
            face_triangle_ratio=face_triangle_ratio, label_layer=label_layer,
            feather_all=feather_all,
        )
        coat = result.image
        normal_image = result.normal
        report["texture"] = result.report
    coat.save(texture_path)
    if normal_image is not None:
        normal_image.save(normal_path)

    glb_path = replace_base_texture(template_glb, coat, asset_dir / f"{pet_id}.glb", normal=normal_image)
    base_texture_path.unlink(missing_ok=True)

    behavior = resolve_behavior(animation_names(glb_path))

    return BuiltAsset(
        pet_id=pet_id,
        glb_path=glb_path,
        texture_path=texture_path,
        served_path=f"/assets/pets/{pet_id}.glb",
        behavior=behavior,
        normal_path=normal_path if normal_image is not None else None,
        report=report,
    )
