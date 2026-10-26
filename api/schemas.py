"""Request and response models for the appearance-extraction endpoint."""

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from .fur import FUR_COLORS
from .patterns import PATTERN_NAMES
from .regions import REGIONS

Region = Literal[
    "skull", "eyebrows", "muzzle_bridge", "cheeks", "ears_inner",
    "back", "chest", "underside",
    "legs_lower", "paws",
]

# Literal cannot be built from REGIONS at runtime, so it is spelled out above
# and checked here instead: a region added to one and not the other would
# otherwise surface as a validation error on a live upload.
assert set(get_args(Region)) == set(REGIONS), (
    f"Region literal and REGIONS disagree: {set(get_args(Region)) ^ set(REGIONS)}"
)
Pattern = Literal["tan_point", "urajiro", "irish", "saddle", "solid", "brindle", "merle", "sable", "piebald",
                  "other"]
assert set(get_args(Pattern)) == set(PATTERN_NAMES), (
    f"Pattern literal and PATTERN_NAMES disagree: {set(get_args(Pattern)) ^ set(PATTERN_NAMES)}"
)
Coverage = Literal["full", "partial"]
MarkingType = Literal["spot", "patch", "stripe", "mask", "socks", "blaze", "other"]
Size = Literal["small", "medium", "large"]
GradientAnchor = Literal["eye", "nose", "belly_center", "paw", "ear_base", "none"]
MarkingAnchor = Literal["left_eye", "right_eye", "nose", "ear_base", "chest_center", "none"]
MarkingOffset = Literal["above", "below", "left", "right", "on"]
MarkingDistance = Literal["near", "far"]


class PetMetadata(BaseModel):
    """The shelter record, as posted alongside the photos.

    이름 and 추정나이 are parsed so they can be echoed back for display. No
    other module reads them: age-based temperament inference is an unvalidated
    generalization and is out of scope for this pipeline.
    """

    model_config = ConfigDict(populate_by_name=True)

    pet_id: str = Field(alias="관리번호")
    species: str = Field(alias="종")
    weight_kg: float = Field(alias="몸무게", gt=0)
    coat_color: str = Field(alias="털색")
    name: str = Field(alias="이름", default="")
    sex: str = Field(alias="성별", default="")
    neutered: bool = Field(alias="중성화", default=False)
    age_text: str = Field(alias="추정나이", default="")


# Free-text colours cannot be turned into a fur ramp, and "dark brown" against
# "brown" against "chocolate" is a distinction no texture can honour. The
# palette in api/fur.py is the contract.
FurColor = Literal[
    "black", "white", "cream", "fawn", "red", "tan", "chocolate", "gray", "pink",
]

assert set(get_args(FurColor)) == set(FUR_COLORS), (
    f"FurColor and FUR_PALETTE disagree: {set(get_args(FurColor)) ^ set(FUR_COLORS)}"
)


class RegionColor(BaseModel):
    region: Region
    color: FurColor
    coverage: Coverage
    note: str = ""
    # a fade from `color` (strongest at gradient_anchor) into gradient_to
    gradient_to: Literal[
        "black", "white", "cream", "fawn", "red", "tan", "chocolate", "gray", "pink", "none"
    ] = "none"
    gradient_anchor: GradientAnchor = "none"
    # face_call / body_call / pattern_default:<pattern>
    source: str = ""
    # share of independent readings that agreed; never the model's own number
    confidence: float = 1.0


class Marking(BaseModel):
    type: MarkingType
    region: Region
    color: FurColor
    size: Size
    anchor: MarkingAnchor = "none"
    offset: MarkingOffset = "on"
    distance: MarkingDistance = "near"


class AppearanceResponse(BaseModel):
    pet_id: str = Field(serialization_alias="petId")
    base_color: FurColor
    second_color: FurColor | None = None
    color_count: int
    pattern: Pattern
    # brindle / merle texture over a solid layout, else null
    overlay: str | None = None
    region_colors: list[RegionColor]
    markings: list[Marking]
    description: str
    confidence: float

    # regions no photograph answered, kept at the pattern's default
    inferred_regions: list[str] = []

    # resolved from the record alone, before any model call
    size_class: Size
    base_template: str
    asset_path: str

    # display only - never read by any logic
    display_name: str = ""
    display_age: str = ""

    model_config = ConfigDict(populate_by_name=True)
