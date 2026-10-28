"""Does the template suit this dog? A verdict before (or beside) the build.

The template is the model the photograph looks most like
(template_pick.combine: the vision call's pick against the grey model
sheet, else the DINOv2 looks), and the question is the user's (2026-10-18):
built from that model and painted in the dog's coat, does it read as the
dog without looking forced? Without a pick (an old caller) the template is
the adult jindo.

Refused: looks no template covers (the DINOv2 "none": pom-type spitz, flat
faces) and a model the vision call found different in POOR_FROM or more
parts (ears, coat, muzzle, head, legs, body, tail; body_shape.py). On 30
dogs judged by the user the two together refused 5 of the 8 the user called
forced and 2 of the 22 the user would build (tools/template_judge.py). One
difference is a caveat. Also counted: short legs (the size rule fails on
them), a weight outside what dogs of that look weigh, a coat layout outside
the pattern set, the photograph. Age is not asked: the looks tell a puppy.
Each check
costs points (fair 20, poor 50); a check that alone makes the result look
wrong is "poor" and decides the verdict, and three caveats together are poor
too. A poor answer (long coat, short legs) counts as poor only when every
shape sample gave it; a split vote is a caveat - the English setter's
feathering read long 2/3 in one pass and 1/3 in the next.

    good  적합    no caveat: build it
    fair  주의    build it; it will read as the dog, with the named caveats
    poor  부적합  the result will not look like this dog
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import Any

PENALTY = {"fair": 20, "poor": 50}
LABEL = {"good": "적합", "fair": "주의", "poor": "부적합"}
FAIR_MIN = 50

TEMPLATE_KO = {"jindo": "진도", "jindo_puppy": "진도 강아지", "maltese": "말티즈", "toypoodle": "토이푸들",
               "chihuahua": "치와와", "retriever": "리트리버", "none": "없음"}
LONG_COAT = {"maltese", "toypoodle"}                             # hair that hides the outline is theirs
BIG_HEAD = {"jindo_puppy", "chihuahua", "maltese", "toypoodle"}  # a head large for the body is theirs
DIFF_KO = {"ears": "귀", "coat length": "털 길이", "coat texture": "털 결", "muzzle": "주둥이", "head": "머리",
           "legs": "다리", "body": "몸통", "tail": "꼬리"}


def parse_age(text: str, today: _dt.date | None = None) -> dict[str, Any]:
    """Shelter age text -> {"days_max": upper bound on age in days or None,
    "years": whole years or None}. Forms seen: "2026(60일미만)(년생)",
    "2022(년생)", "3살", "5개월"."""
    today = today or _dt.date.today()
    text = str(text or "")
    out: dict[str, Any] = {"text": text, "days_max": None, "years": None}
    m = re.search(r"(\d+)\s*일\s*미만", text)
    if m:
        out["days_max"] = int(m.group(1))
    m = re.search(r"(\d+)\s*개월", text)
    if m:
        out["days_max"] = int(m.group(1)) * 31
        out["years"] = int(m.group(1)) // 12
    m = re.search(r"(\d+)\s*(살|세)", text)
    if m:
        out["years"] = int(m.group(1))
    m = re.search(r"(19|20)\d{2}", text)
    if m and out["years"] is None:
        out["years"] = max(0, today.year - int(m.group(0)))
    return out


def assess_fit(species: str, weight_kg: float, age_text: str = "",
               shape: dict[str, str] | None = None, pattern: str | None = None,
               today: _dt.date | None = None, votes: dict[str, Any] | None = None,
               photo_quality: dict[str, Any] | None = None, pick: dict[str, Any] | None = None,
               photo_bars: bool = False, from_above: bool = False) -> dict[str, Any]:
    """The verdict: level (good/fair/poor), its Korean label, a 0-100 score
    and every check that cost points, with its reason in Korean. `pick` is
    template_pick.combine's answer; without it the template is the adult
    jindo. `votes` is body_shape's per-question tally; without it the
    answers count as unanimous. `photo_quality` is
    photo_quality.measure_quality's result: a small, blurred or clipped
    photograph costs points like a body check does, because the coat read
    from it is as unreliable as a coat the template cannot show.
    `photo_bars`: the dog was photographed through cage bars or a fence
    (prepare_photo.bridge_bars) - the coat behind the bars is guessed and
    the colour measurement skipped, a caveat. `from_above`: every gate
    sample saw the dog from above (prepare_photo.seen_from_above). The legs
    are foreshortened then, so the leg answers and a "legs" difference are
    not counted. On public shelter photos (2026-10-28) the two puppies seen
    from above were read short-legged 1/3 and 3/3. The age is only carried
    into the inputs."""
    shape = shape or {}
    if from_above and pick and "legs" in (pick.get("differences") or []):
        from .body_shape import match_from
        rest = [d for d in pick["differences"] if d != "legs"]
        pick = {**pick, "differences": rest, "match": match_from(rest) if pick.get("match") else pick.get("match")}
    built = (pick or {}).get("template") or "jindo"
    name = TEMPLATE_KO.get(built, built)
    # the vision call compared the dog with the model itself: its differences
    # stand in for the coat, head and muzzle checks against a fixed template
    by_vision = (pick or {}).get("by") == "vision"

    def tally(key: str, value: str) -> tuple[int, int]:
        """(samples that said value, samples) for one shape question."""
        if votes and isinstance(votes.get(key), dict) and votes.get("samples"):
            return int(votes[key].get(value, 0)), int(votes["samples"])
        return (1 if shape.get(key) == value else 0), 1
    age = parse_age(age_text, today)
    checks: list[dict[str, str]] = []
    notes: list[str] = []

    def hit(check: str, level: str, why: str) -> None:
        checks.append({"check": check, "level": level, "label": LABEL[level], "why": why})

    if species != "개":
        hit("species", "poor", f"종이 '{species}': 템플릿은 개뿐")

    if pick:
        diffs = "·".join(DIFF_KO.get(d, d) for d in pick.get("differences") or [])
        if pick.get("no_template"):
            hit("template", "poor", f"맞는 템플릿 없음: 포메형·단두형 생김새, 가장 가까운 {name} 몸으로는 이 개처럼 안 보임")
        elif pick.get("match") == "poor":
            hit("template", "poor", f"{name} 모델과 다른 곳이 많음({diffs}): 다른 종류의 개처럼 보임")
        elif pick.get("match") == "rough":
            hit("template", "fair", f"{name} 모델과 다른 곳: {diffs}")
        notes.append(f"템플릿: {name}" + (" (사진과 모델 비교)" if by_vision else " (생김새 분류)"))
        if pick.get("heavy_puppy"):
            notes.append(f"생김새는 진도 강아지지만 {weight_kg:g}kg이라 성견 템플릿으로 만듦")

    # the weights of the shelter dogs of this look (template_pick's training set)
    from .template_pick import weight_range
    try:
        span = weight_range(built)
    except FileNotFoundError:
        span = None
    if span and not span[0] <= weight_kg <= span[1]:
        hit("weight", "fair", f"{weight_kg:g}kg: {name} 생김새 보호소 개 대부분({span[0]:.1f}~{span[1]:.1f}kg) 밖이라 "
                              f"크기는 맞춰도 체형이 다를 수 있음")

    if built not in LONG_COAT and not by_vision:
        yes, n = tally("coat_length", "long")
        if yes and yes == n:
            hit("coat_length", "poor", f"장모: 털이 늘어져 몸 윤곽을 덮는 개는 {name} 몸으로 표현이 안 됨")
        elif yes:
            hit("coat_length", "fair", f"털이 긴 편({yes}/{n}표 장모): 늘어진 털은 표현이 안 됨")

    legs = shape.get("leg_length", "unknown")
    yes, n = tally("leg_length", "short")
    if from_above:
        notes.append("위에서 찍은 사진이라 다리 길이는 판단하지 않음")
    elif yes and yes == n:
        hit("leg_length", "poor", f"단족(코기·닥스훈트형): 다리 길이가 {name} 템플릿과 크게 달라 비율이 어색함")
    elif yes:
        hit("leg_length", "fair", f"다리가 짧은 편({yes}/{n}표 단족): {name} 다리보다 짧아 보일 수 있음")
    elif legs == "long":
        hit("leg_length", "fair", f"다리가 긴 체형: {name} 다리로는 조금 짧아 보임")

    if shape.get("head_size") == "large" and built not in BIG_HEAD and not by_vision:
        hit("head_size", "fair", f"몸에 비해 머리가 큼(초소형견·어린 강아지형): {name} 머리 비율과 다름")

    if shape.get("muzzle") == "short" and not by_vision:
        hit("muzzle", "fair", "주둥이가 짧은 얼굴: 주둥이 모프로 줄여도 납작한 얼굴까지는 안 됨")

    if pattern == "other":
        hit("pattern", "fair", "템플릿 패턴에 없는 털 배치: 가장 가까운 기본 배치로 칠함")
    elif pattern in ("brindle", "merle"):
        hit("pattern", "fair", f"{pattern} 무늬: 덧칠 무늬로 근사만 함")

    unknown = [k for k in ("coat_length", "leg_length", "head_size") if shape.get(k, "unknown") == "unknown"]
    # no shape answers at all: the outline was not asked (a template without
    # morph targets), not unseen
    if shape and len(unknown) >= 2:
        hit("photo", "fair", "사진에서 털 길이·다리·머리 비율을 확인하지 못함")

    # the photograph itself (photo_quality.py): too small, blurred, clipped
    from .photo_quality import describe_quality
    for measure, level, why in describe_quality(photo_quality):
        hit(f"photo_{measure}", level, why)
    if photo_bars:
        hit("photo_bars", "fair", "철장·울타리 너머로 찍힌 사진: 막대에 가린 털은 주변 털로 추정, 부위 색 측정 생략")

    score = max(0, 100 - sum(PENALTY[c["level"]] for c in checks))
    if any(c["level"] == "poor" for c in checks) or score < FAIR_MIN:
        level = "poor"
    elif checks:
        level = "fair"
    else:
        level = "good"
    return {"level": level, "label": LABEL[level], "score": score, "checks": checks, "notes": notes,
            "template": built, "pick": pick,
            "inputs": {"species": species, "weight_kg": weight_kg, "age": age, "pattern": pattern,
                       "shape": {k: shape.get(k, "unknown") for k in
                                 ("coat_length", "leg_length", "head_size", "muzzle", "build", "ears")},
                       "photo_quality": photo_quality}}


def describe(fit: dict[str, Any]) -> str:
    head = f"{fit['label']} ({fit['score']}점)"
    if not fit["checks"]:
        return head
    return head + ": " + "; ".join(c["why"] for c in fit["checks"])


class FitRejected(RuntimeError):
    """Raised by the lite run when --fit-stop is on and the verdict is poor."""

    def __init__(self, fit: dict[str, Any]):
        super().__init__(describe(fit))
        self.fit = fit


__all__ = ["FitRejected", "LABEL", "TEMPLATE_KO", "assess_fit", "describe", "parse_age"]
