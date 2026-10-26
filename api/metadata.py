"""Deterministic handling of the shelter record.

This runs to completion before any model call. Anything the record already
states is settled here, so the vision agents are only ever asked what a
photograph can answer.

Display-only fields (이름, 추정나이) are deliberately absent from everything
below: they reach the response DTO directly and never enter a decision.
"""

from dataclasses import dataclass

from .fur import DEFAULT_COLOR, FUR_COLORS


class UnknownSpecies(ValueError):
    """No stage-1 base template exists for this species."""


# (small/medium cut, medium/large cut) in kg. Add a species key to override;
# body-mass ranges differ enough between species that one table cannot serve
# all of them.
SIZE_THRESHOLDS_KG = {
    "default": (7.0, 25.0),
}

# Species -> the slug used in stage 1's skeleton_{slug}_{size}.glb filenames.
# The shiba template (the first one, 2026-09) was retired on 2026-10-01 and
# moved to archive/shiba/; the jindo built from the Meshy model replaced it.
SPECIES_TEMPLATE_SLUG = {
    "개": "jindo",
}

# The record's colour has to land inside FUR_COLORS, because that is what the
# texture stage can build a ramp from. Several shelter words collapse onto one
# entry - 아이보리 and 베이지 are both cream to a texture - and 삼색 and 얼룩 name
# a pattern rather than a colour, so they map to the base coat those patterns
# sit on and the pattern itself is the vision agents' job.
COLOR_NAMES_EN = {
    "검정": "black", "검정색": "black", "블랙": "black", "흑색": "black",
    "흰색": "white", "하양": "white", "백색": "white", "흰": "white",
    "아이보리": "cream", "크림": "cream", "베이지": "cream", "미색": "cream",
    "갈색": "tan", "밤색": "tan", "황갈색": "tan",
    "노랑": "fawn", "황색": "fawn", "금색": "fawn", "베이지색": "cream",
    "적갈색": "red", "붉은색": "red", "주황": "red",
    "초콜릿": "chocolate", "진갈색": "chocolate",
    "회색": "gray", "쥐색": "gray", "은색": "gray",
    "삼색": "tan", "얼룩": "tan",
}

ASSET_ROOT = "/assets/pets"


@dataclass(frozen=True)
class MetadataResult:
    pet_id: str
    base_color: str
    size_class: str
    base_template: str
    asset_path: str


def get_size_class(species: str, weight_kg: float) -> str:
    small_max, medium_max = SIZE_THRESHOLDS_KG.get(species, SIZE_THRESHOLDS_KG["default"])
    if weight_kg < small_max:
        return "small"
    if weight_kg < medium_max:
        return "medium"
    return "large"


def resolve_base_template(species: str, size_class: str) -> str:
    """skeleton_{slug}_{size}.glb. The slug is the species' default template
    unless PET_TEMPLATE names another one (python -m api.run --template
    jindo sets it): every template lives under its own slug in
    output/templates/."""
    import os

    slug = os.environ.get("PET_TEMPLATE") or SPECIES_TEMPLATE_SLUG.get(species)
    if slug is None:
        raise UnknownSpecies(
            f"no base template for species '{species}'; "
            f"known: {sorted(SPECIES_TEMPLATE_SLUG)}"
        )
    return f"skeleton_{slug}_{size_class}.glb"


def normalize_color(coat_color: str) -> str:
    """Map the record's coat colour onto the fur palette.

    An unrecognised word falls back to the palette's default rather than
    passing through: a colour the texture stage cannot build a ramp from would
    fail later, further from the record that caused it.
    """
    words = _coat_words(coat_color)
    known = [w for w in (_single_color(w) for w in words) if w]
    if known:
        # A compound entry ("갈색/흰색", "흰색, 검정") names the markings as
        # well as the coat; the base is the first word that is not white -
        # white is nearly always the second colour of a pattern, and the
        # pattern reader anchors on the base colour we hand it.
        coloured = [w for w in known if w != "white"]
        return coloured[0] if coloured else known[0]
    print(f"[metadata] coat colour {coat_color!r} is not in the fur palette; "
          f"using {DEFAULT_COLOR}")
    return DEFAULT_COLOR


def _coat_words(coat_color: str) -> list[str]:
    """The colour words of a record entry, split on the separators shelters
    use: "갈색/흰색", "흰색, 검정", "갈색·흰색", "갈색 흰색", "갈+흰"."""
    import re

    return [w for w in re.split(r"[\s/,·&+]+", coat_color.strip()) if w]


def _single_color(word: str) -> str | None:
    if word in COLOR_NAMES_EN:
        return COLOR_NAMES_EN[word]
    lowered = word.lower()
    if lowered in FUR_COLORS:
        return lowered
    # "갈" for "갈색", "흰" for "흰색": the shelter shorthand
    for name, colour in COLOR_NAMES_EN.items():
        if name.endswith("색") and word == name[:-1]:
            return colour
    return None


def asset_path(pet_id: str) -> str:
    """Files are keyed by 관리번호, never by name: names collide and pets get
    renamed after adoption."""
    return f"{ASSET_ROOT}/{pet_id}.glb"


def process_metadata(pet_id: str, species: str, weight_kg: float, coat_color: str) -> MetadataResult:
    size_class = get_size_class(species, weight_kg)
    return MetadataResult(
        pet_id=pet_id,
        base_color=normalize_color(coat_color),
        size_class=size_class,
        base_template=resolve_base_template(species, size_class),
        asset_path=asset_path(pet_id),
    )
