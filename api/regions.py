"""The ten coat regions, shared with stage 1's mask templates.

Kept as a standalone module (rather than imported from ar_pipeline) because
those scripts import bpy at module level and only run inside Blender. REGIONS
here and the masks under output/templates/<name>/masks/ must stay identical: a
region name here with no matching masks/{region}.png is a texture that can
never be composited.

WHY TEN
-------
There used to be sixteen. The split was fine-grained enough to tell an ear's
inside from its outside and a tail's tip from its top, but the photographs
never answered at that resolution: over every live run, six to fifteen of the
sixteen came from the pattern's default layout, the two critic samples agreed
on almost no region-level difference, and the per-region combine had
candidates for one region in sixteen. Each extra region was a place for the
reading to miss, and a miss buys another image-model round.

The ten that remain are the regions that named coat patterns actually
distinguish - tan points, urajiro, Irish spotting, saddle, the dark mask, and
the tricolour's white socks - plus the base coat. Two regions that receive the
same colour under every one of those patterns are one region here:

    skull          = skull + ears_outer          (base coat on the head)
    back           = back + collar + legs_upper + tail_top + tail_tip
    chest          = chest + throat              (the white or cream bib)

ears_inner stays on its own although the patterns treat it like cheeks: on
the white dogs it was the one region the photograph answered at all (pink or
red inside the ear), and folded into cheeks that answer either vanishes into
the cheek's dominant white or tints the whole cheek pink.

What that gives up: an Irish white collar and white tail tip, and a dark mask
that reaches the ears. None of the three was ever read from a photograph.

The Blender-side vertex groups are still tagged at the old sixteen
(SOURCE_REGIONS): stage 1 bakes those and unions them into REGIONS through
MERGE, so an existing template needs no re-tagging and a new one may tag
either set.
"""

REGIONS = (
    # --- head ---
    "skull",
    "eyebrows",
    "muzzle_bridge",
    "cheeks",
    "ears_inner",
    # --- body ---
    "back",
    "chest",
    "underside",
    # --- legs ---
    "legs_lower",
    "paws",
)

# Sent to the vision agents verbatim. The names alone are not decidable -
# eyebrows against skull, muzzle_bridge against cheeks - so the definition
# travels with the name.
REGION_DESCRIPTIONS = {
    # 바탕색. 블레이즈는 마킹으로. 예: 시바 정수리·귀 바깥 붉은색
    "skull": (
        "The top of the head from the stop back to the occiput, between the "
        "ears, together with the outer surface of both ears."
    ),
    # 탄포인트 pips 자리. 예: 도베르만·로트와일러 눈 위 갈색 점
    "eyebrows": (
        "Two ovals directly above the eyes, each about as wide as the eye. "
        "The eyes themselves are not included."
    ),
    # 마스크 진한 부분, 탄포인트 nose bar. 예: 저먼 셰퍼드 검은 콧등
    "muzzle_bridge": (
        "The top of the muzzle only, from the stop down to the nose, no wider "
        "than the nose. The sides of the muzzle are not part of it."
    ),
    # 탄포인트·우라지로 뺨. 예: 시바 뺨 크림색
    "cheeks": (
        "The side of the face from under the eye down to the jawline and "
        "forward to the corner of the mouth, including the sides of the muzzle."
    ),
    # 우라지로·탄포인트, 흰 개의 분홍 맨살. 예: 시바 귀 안쪽 크림색
    "ears_inner": "The inner, cupped surface of both ears.",
    # 바탕색·새들·세이블. 예: 저먼 셰퍼드 검은 새들
    "back": (
        "The neck behind the head (sides and back, not the front), the back "
        "from the shoulder to the base of the tail including the upper flank, "
        "the outer surface of the upper legs down to the elbow and knee, and "
        "the whole tail except its underside. Its lower edge on the trunk is a "
        "horizontal line just below the widest point of the ribcage."
    ),
    # 탄포인트 삼각형, 아이리시·우라지로 흰 목·가슴. 예: 시바 턱밑부터 가슴까지 크림색
    "chest": (
        "Under the jaw, down the front of the neck and over the front of the "
        "chest: centred on the breastbone between the forelegs, reaching out to "
        "the front of both shoulders, from the chin down to elbow height."
    ),
    # 탄포인트·우라지로 배쪽 전체. 예: 시바 배·다리 안쪽 크림색
    # 다리 안쪽·뒷면은 손목/비절에서 끊지 않고 발까지 내려간다. 시바의 크림색이
    # 팔꿈치 아래에서 뒷면을 따라 이어지는데, legs_lower가 안팎을 묶고 있으면
    # 그 배색을 표현할 방법이 없기 때문.
    "underside": (
        "Everything below the lower edge of the back: the belly, the underside "
        "of the tail, and the inner and rear surfaces of all four legs running "
        "unbroken from the groin down to the paws."
    ),
    # 탄포인트 다리 하단, 긴 양말. 예: 도베르만 다리 아래 갈색(앞면)
    # 안쪽·뒷면은 underside가 가져간다.
    "legs_lower": (
        "The front and outer surfaces of the forelegs from elbow to wrist and "
        "of the hind legs from knee to hock. The inner and rear surfaces "
        "belong to underside."
    ),
    # 짧은 양말, 발끝 흰색. 예: 보더콜리 흰 발
    "paws": "Below the wrist and hock, including the toes.",
}

assert set(REGION_DESCRIPTIONS) == set(REGIONS), (
    "every region needs a description sent to the agents: "
    f"{set(REGION_DESCRIPTIONS) ^ set(REGIONS)}"
)


# The split the two stage-2 calls work on. The face call reads a crop of the
# head and answers for the five regions that live on it; the body call reads
# the whole photograph and answers for the other five. The chest is a body
# region: it runs from the chin down the front of the neck to the breastbone
# and a head crop cuts it off.
FACE_REGIONS = ("skull", "eyebrows", "muzzle_bridge", "cheeks", "ears_inner")
BODY_REGIONS = tuple(region for region in REGIONS if region not in FACE_REGIONS)
assert len(FACE_REGIONS) == 5 and len(BODY_REGIONS) == 5


# --------------------------------------------------------------------------
# the sixteen source regions
# --------------------------------------------------------------------------

# The vertex groups stage 1 tags in Blender and bakes, one mask each, into
# masks_source/. A region here is either a REGIONS name or maps to one
# through MERGE; masks/{region}.png is the union of its sources.
SOURCE_REGIONS = (
    "eyebrows", "muzzle_bridge", "cheeks", "throat", "skull", "ears_outer", "ears_inner",
    "collar", "back", "chest", "underside",
    "legs_upper", "legs_lower", "paws",
    "tail_top", "tail_tip",
)

# source region -> region. Identity for the names that survived.
MERGE: dict[str, str] = {
    "skull": "skull",
    "ears_outer": "skull",
    "eyebrows": "eyebrows",
    "muzzle_bridge": "muzzle_bridge",
    "cheeks": "cheeks",
    "ears_inner": "ears_inner",
    "back": "back",
    "collar": "back",
    "legs_upper": "back",
    "tail_top": "back",
    "tail_tip": "back",
    "chest": "chest",
    "throat": "chest",
    "underside": "underside",
    "legs_lower": "legs_lower",
    "paws": "paws",
}

assert set(MERGE) == set(SOURCE_REGIONS) and set(MERGE.values()) == set(REGIONS), (
    "MERGE must map every source region onto a region and reach every region"
)

# region -> its source regions: the namesake first (the larger, better-read
# part), then the rest in SOURCE_REGIONS order
SOURCES: dict[str, tuple[str, ...]] = {
    region: (region,) + tuple(s for s in SOURCE_REGIONS if MERGE[s] == region and s != region)
    for region in REGIONS
}


def canonical_region(name: str) -> str | None:
    """A region name from the current or the old sixteen-region vocabulary,
    folded onto REGIONS; None for anything else."""
    key = (name or "").strip().lower()
    if key in MERGE:
        return MERGE[key]
    return None


def canonicalize_entries(entries: list[dict] | None) -> list[dict]:
    """Region entries (region_colors, markings, critic patches) written under
    the old sixteen names, folded onto the ten. When several old entries
    land on one region, one that was read from the photograph beats a
    pattern default, and among equals the first in the region's source
    order wins - the larger part (skull over ears_outer, back over collar,
    chest over throat), which is what a fresh
    reading would have answered for the merged region. Entries naming no
    known region are dropped. Markings keep duplicates: two patches on one
    region are still two patches."""
    out: list[dict] = []
    seen: dict[str, int] = {}
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        region = canonical_region(str(entry.get("region", "")))
        if region is None:
            continue
        folded = {**entry, "region": region}
        if "type" in entry:            # a marking
            out.append(folded)
            continue
        if region in seen:
            if _rank(entry, region) < _rank(out[seen[region]], region):
                out[seen[region]] = folded
            continue
        seen[region] = len(out)
        out.append(folded)
    return out


def _rank(entry: dict, region: str) -> tuple[int, int]:
    """Lower wins: read entries before defaults, then source order."""
    source = str(entry.get("source", ""))
    defaulted = 1 if source.startswith(("pattern", "inferred")) else 0
    key = str(entry.get("region", "")).strip().lower()
    order = SOURCES[region]
    return defaulted, order.index(key) if key in order else 0


def canonicalize_appearance(appearance: dict) -> dict:
    """An appearance.json from before the merge (or after; this is
    idempotent) with every region list folded onto REGIONS."""
    if not isinstance(appearance, dict):
        return appearance
    out = dict(appearance)
    for key in ("region_colors", "markings"):
        if key in out:
            out[key] = canonicalize_entries(out.get(key))
    if "inferred_regions" in out:
        folded = []
        for name in out.get("inferred_regions") or []:
            region = canonical_region(str(name))
            if region is not None and region not in folded:
                folded.append(region)
        # a region is inferred only if none of its sources was read
        read = {e["region"] for e in out.get("region_colors") or []
                if not str(e.get("source", "")).startswith(("pattern", "inferred"))}
        out["inferred_regions"] = [r for r in folded if r not in read]
    return out
