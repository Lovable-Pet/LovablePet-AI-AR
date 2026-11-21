"""Which template a dog looks like, from its photograph alone. Local, no
API call.

The photograph becomes one vector (DINOv2 ViT-L/14, image features only,
no text) and a linear layer over it scores the templates: jindo,
jindo_puppy, maltese, toypoodle, chihuahua, retriever, and "none" for the
looks no template covers (pom-type spitz, flat-faced breeds). Age is not
asked: a puppy is told by its looks (2026-10-18, user - the template and
the coat need the looks, the size needs the weight). One exception since
2026-11-15, in combine: a record under 60 days is not built as an adult
retriever, nor as an adult jindo one sample doubted.

Which model to BUILD is the vision call's (body_shape.py with the model
sheet; combine below): judged against a person on 30 dogs it named the
person's model 20 times in 22, this layer 13 (tools/template_judge.py). This
layer stays for what it does better - its "none", pom-type and flat faces,
caught 4 of the 8 dogs the person called forced - and as the pick when the
call fails. Its confidence told forced from fine not at all (0.63-1.00 for
the forced, 0.75-0.98 for the clear matches).

The layer was fitted on 398 shelter photographs labelled by hand plus
1,307 shelter photographs whose record names a breed
(tools/shelter_stats/train_template_pick.py, api/data/template_pick.npz).
Held out by shelter and day, it picked the hand-labelled template for 79%
of the 398 (75% without the breed photos), 87.5% of the 70% it was most
sure of. The drop-ear mixes are what is left (48%: read as jindo or jindo
puppy), and the hand labels share that doubt.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

BACKBONE = "vit_large_patch14_dinov2.lvd142m"
TEMPLATES = ["jindo", "jindo_puppy", "maltese", "toypoodle", "chihuahua", "retriever"]
WEIGHTS = Path(__file__).with_name("data") / "template_pick.npz"
NO_TEMPLATE = "none"
# combine's chihuahua_by_all: the looks' probability for a jindo-type body
# from which the call's chihuahua is given up. 0.7 and a split vote only
# until 2026-11-15 (the three dogs it moved then had 0.74, 0.81, 0.88); the
# dogs that showed the looks right below it had 0.52, 0.54 and 0.64
CHIHUAHUA_LOOKS_FLOOR = 0.5
CHIHUAHUA_NOT_FOR = ("jindo", "jindo_puppy", "retriever")
# combine's "under 60 days by the record": the age in days at or under which
# an adult retriever is not built, nor an adult jindo one sample doubted
YOUNG_DAYS = 60
# combine's "newborn": a record weight at or under this is a pup of its first
# days, which no model covers. Of 637 shelter dogs with a weight (the crawls
# of 2026-11, work/cost/fair/pool*) ten weigh this or less and all ten are
# newborns - eyes shut, lying on newspaper or held in a hand, pink skin under
# a first coat; the next two (0.58 and 0.6 kg) sit and stand with open eyes.
# A toy-breed pup of five weeks can weigh this too and is turned away with
# them: the notice can be sent again when it has grown.
NEWBORN_KG = 0.5


@lru_cache(maxsize=1)
def _weights() -> dict[str, Any]:
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"{WEIGHTS} missing; run tools/shelter_stats/train_template_pick.py")
    data = np.load(WEIGHTS, allow_pickle=False)
    if str(data["backbone"]) != BACKBONE:
        raise ValueError(f"{WEIGHTS} was fitted on {data['backbone']}, not {BACKBONE}; retrain it")
    return {k: data[k] for k in data.files}


@lru_cache(maxsize=1)
def _backbone():
    import timm
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(BACKBONE, pretrained=True, num_classes=0).to(dev).eval()
    transform = timm.data.create_transform(**timm.data.resolve_data_config({}, model=model))
    return model, transform, dev


def embed(photo: str | Path) -> np.ndarray:
    """The photograph's unit vector. The EXIF rotation is applied first, as
    for the training photos."""
    import torch
    from PIL import Image, ImageOps

    model, transform, dev = _backbone()
    image = ImageOps.exif_transpose(Image.open(photo)).convert("RGB")
    with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
        v = model(transform(image).unsqueeze(0).to(dev))
    v = v.float().cpu().numpy()[0]
    return v / np.linalg.norm(v)


def scores(vector: np.ndarray) -> dict[str, float]:
    """Template -> probability for one unit vector."""
    w = _weights()
    logits = w["coef"] @ (vector * float(w["input_scale"])) + w["intercept"]
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return {str(c): float(v) for c, v in zip(w["classes"], p)}


def pick_template(photo: str | Path, weight_kg: float | None = None) -> dict[str, Any]:
    """{"template": slug or "none", "confidence": its probability, "sure":
    confidence at or above the cut that kept the most-sure 70% in the
    held-out check, "nearest": the best real template (the one to build
    when the pick is "none"), "looks": the looks' own answer before the
    weight check, "scores": every template's probability}."""
    return decide(scores(embed(photo)), weight_kg)


def puppy_max_kg() -> float:
    """95th percentile record weight of the hand-labelled jindo-type dogs
    born this year (train_template_pick.py)."""
    return float(_weights()["puppy_max_kg"])


def decide(probs: dict[str, float], weight_kg: float | None = None) -> dict[str, Any]:
    """The pick from the probabilities. A jindo puppy heavier than
    puppy_max_kg is built as the adult jindo: the puppy mesh is a 2-month
    body (k 15.5 against the adult's 18.5, a withers 16% short), and of the
    held-out photos read as puppies above 7 kg most were adults or other
    looks (tools/shelter_stats, 2026-10-18). Its confidence is then the
    jindo-type share, puppy and adult together."""
    w = _weights()
    looks = max(probs, key=probs.get)
    nearest = max((c for c in probs if c != NO_TEMPLATE), key=probs.get)
    best, confidence = looks, probs[looks]
    heavy = weight_kg is not None and weight_kg > puppy_max_kg()
    if heavy and best == "jindo_puppy":
        best, confidence = "jindo", probs["jindo"] + probs["jindo_puppy"]
    if heavy and nearest == "jindo_puppy":
        nearest = "jindo"
    return {"template": best, "confidence": round(confidence, 3), "sure": confidence >= float(w["sure_at"]),
            "nearest": nearest, "looks": looks,
            "scores": {c: round(p, 3) for c, p in sorted(probs.items(), key=lambda kv: -kv[1])},
            "model": BACKBONE}


def _weighs_like(template: str, weight_kg: float | None) -> bool:
    """The record weight is not above what dogs of this look weigh
    (weight_range's 95th percentile); True without a weight or a range."""
    span = weight_range(template) if looks_weights_ok() else None
    return weight_kg is None or span is None or weight_kg <= span[1]


def _weighs_within(template: str, weight_kg: float | None) -> bool:
    """The record weight is inside what dogs of this look weigh
    (weight_range, 5th to 95th percentile); True without a weight or a range."""
    span = weight_range(template) if looks_weights_ok() else None
    return weight_kg is None or span is None or span[0] <= weight_kg <= span[1]


def combine(looks: dict[str, Any] | None, shape: dict[str, Any] | None,
            weight_kg: float | None = None, votes: dict[str, Any] | None = None,
            age_days: int | None = None) -> dict[str, Any]:
    """The template to build and what the verdict needs, from the two
    readings of the photograph:
      looks  this module's pick (DINOv2): its "none" - pom-type and flat
             faces - is the refusal it is good at
      shape  body_shape.read_body_shape(models=True): the vision call's
             closest model and the differences from it, which pick the
             model a person would (20 of 22 against 13 of 22 for the looks,
             tools/template_judge.py, 2026-10-18)
      votes  that call's per-sample tally (its closest_model counts)
    The vision call's model when it named one, else the looks' (its nearest
    real template for "none"); a jindo puppy above puppy_max_kg is built as
    the adult either way.

    Three places where the vision call's model is not kept, each seen on the
    64 hard shelter photos of 2026-11-11 (tools/hard_photos/pick_replay.py
    replays the pick on them and on the 30 dogs the user judged on 10-18):
      a puppy   the call names the adult jindo for jindo-type puppies (7 dogs
                of 1.7-4.4 kg, 3/3 votes for four of them): when the looks
                are sure of a jindo puppy and the dog weighs less than adult
                jindos do (weight_range's 5th percentile, 5 kg), the puppy is
                built. The user looked at five side by side (11-11) and took
                the puppy for four; the fifth, a 3 kg dog with an adult's
                grizzled face in close-up, stays the adult - the looks gave
                it 0.76, under their own sure cut, the others 0.95-0.99
      too heavy a chihuahua heavier than chihuahuas are (8-10.6 kg, 3 dogs,
                all 3/3 votes) takes the looks' model if the weight fits
                that, else the jindo
      no majority  when no model has more than half of the samples, the
                looks' model is taken if it is a real one the weight fits
                (a 28 kg labrador read jindo / jindo puppy / maltese, the
                looks retriever 1.00). A 2-1 split follows the looks only
                when they are sure and the weight is inside that look's
                range, and never off the puppy for a dog of puppy weight:
                three more hard photos right (two maltese types read
                chihuahua and jindo, a 20 kg drop-eared dog read jindo) and
                none of the user's 22 picks changed. Without those limits it
                broke two of the 22
      a chihuahua by some of the votes  for a reading model with the rule
                chihuahua_by_all (votes["read_by"], reading_model.py): a
                chihuahua the call did not name with every sample, where the
                looks name another real model at CHIHUAHUA_LOOKS_FLOOR or
                more, is built as the looks' model, whatever the weight.
                gpt-6-luna calls small short-coated dogs with pricked ears
                chihuahuas ("head large", 2 samples of 3) where gpt-4o named
                the jindo 2 and 3 of 3, and the chihuahua's bat ears and round
                eyes on a jindo-type dog are the first thing seen (user,
                2026-11-15, two of the first 19 fresh photos: 5 and 4 kg,
                looks jindo 0.81 and 0.88 - the 4 kg one under a jindo's
                weight range, which kept the split rule above off it). Of 214
                gpt-6-luna dogs on file three change, none of the judged ones
                for the worse; the chihuahuas it named with every sample and
                those the looks call "none" (pom types) stay
      a toy poodle the looks call maltese  for a reading model with the rule
                maltese_by_looks: a toy poodle the call named - with every
                sample too, which the split rule above leaves alone - is the
                maltese when the looks are sure of it and the dog weighs
                inside a maltese look's range. gpt-6-luna names the toy poodle
                for long-coated white small dogs: on the size test set (AI Hub
                71520, tools/size_eval, 2026-11-21) it built 13 of the 30
                maltese by the breed label as toy poodles, where the looks
                said maltese for all 30, and those stood 17% too tall at their
                weight (toy poodle 28 cm at 3.8 kg, maltese 23 at 3.0). Made
                on the 690 shelter dogs with a saved pick: of the 31 the call
                named toy poodle and the looks maltese, the record names the
                maltese for 13 and a poodle or bichon for 4, the looks sure of
                none of the 4. It moves 7 of the 690 (2 maltese, 2 maltipoo,
                3 mixes), none of the judged 30 or the hard 64. Checked after
                on the test set: 11 dogs move, 10 nearer their measured height
                (all 8 maltese), 1 farther (a 7 kg cocker spaniel, -14% to
                -24%); the maltese's median error 13.7% to 11.8%
      under 60 days by the record  `age_days` (fit.parse_age's days_max: the
                shelter's "60일미만") at YOUNG_DAYS or less: an adult
                retriever is built as the puppy, and an adult jindo too when
                one of the call's samples named the puppy (by "record"). The
                age is the one thing here not read off the photograph, and
                only this far. Of 62 such dogs with a body written down
                beforehand (six sets, 2026-11-15) eight were given an adult
                body and the puppy was the written body for all eight; this
                moves six of them - five drop-eared or fluffy pups of 1-3 kg
                named retriever (the looks said puppy for three) and a white
                pup named jindo by two samples of three. The user's doubt
                was the breed ("종이다르자나": the one puppy is a jindo
                puppy), so the six were rebuilt on it side by side (11-15):
                closer to the photograph for all six by my eye, the drop
                ears standing half up being the loss; then the size
                ("크기가맞으려나"): at 2 kg the adult models stand
                23.3-23.8 cm, the puppy 17.9, and a growing dog's
                height-weight slope (0.400-0.446, docs/size_data) puts a
                2 kg puppy at 18-20. A jindo named by every sample stays:
                hard photo #286, 3 kg and "under 60 days" with an adult's
                grizzled face, is the one the user kept as the adult on
                11-11 (the rule "a puppy" above), and the record's age does
                not overrule that. The small models (maltese, toypoodle,
                chihuahua) are left as picked. Not above puppy_max_kg: a
                heavier "under 60 days" is a record to doubt
      a retriever at puppy weight  the retriever is the one drop-eared,
                feathered model, so the call names it for small dogs of that
                kind too - pups, and a 4.8 kg long-coated adult that came out
                a grown retriever's long head with its teeth showing (new6
                46, user: "아직 기괴함"). At puppy_max_kg or less it is the
                looks' model when that is another real one the weight is
                inside (by "looks"), else the puppy (by "record"): of the
                nine such dogs with a body written down (2026-11-15) the
                retriever was the written one for none - maltese for one
                (the looks' maltese 0.82), the puppy for seven - and the
                three at 8-10 kg, one of them a retriever by the truth, stay
    `moved` says which of them applied.

    A newborn (weight_kg at or under NEWBORN_KG) is no model's: the pick is
    made as for any dog and marked no_template and newborn, so it is refused
    where the looks' "none" is (user, 2026-11-15: "거절하는게맞나" - the
    one built, 0.18 kg, came out a white adult jindo 10 cm tall, and on the
    puppy it would be a two-month pup 8 cm tall; neither is the animal in
    the photograph)."""
    from .reading_model import has_rule

    looks = looks or {}
    shape = shape or {}
    vision = shape.get("closest_model")
    by_vision = vision in TEMPLATES
    looks_real = bool(looks) and looks["template"] != NO_TEMPLATE
    if by_vision:
        template = vision
    elif looks:
        template = looks["nearest"] if looks["template"] == NO_TEMPLATE else looks["template"]
    else:
        template = "jindo"
    by, moved = "vision" if by_vision else "looks" if looks else "default", None
    tally = {k: int(n) for k, n in ((votes or {}).get("closest_model") or {}).items()}
    if (by_vision and tally and 2 * tally.get(vision, 0) <= sum(tally.values()) and looks_real
            and looks["template"] != template and _weighs_like(looks["template"], weight_kg)):
        template, by, moved = looks["template"], "looks", f"no majority {tally}"
    # ... and a split with a majority too, when the looks are sure of another
    # real model and the dog weighs what dogs of that look do, both ends
    # (not off the puppy for a dog of puppy weight: the looks call small
    # drop-eared puppies retrievers, an adult's long-coated body)
    elif (by_vision and tally and tally.get(vision, 0) < sum(tally.values()) and looks_real and looks.get("sure")
          and looks["template"] != template and _weighs_within(looks["template"], weight_kg)
          and not (template == "jindo_puppy" and weight_kg is not None
                   and weight_kg < (weight_range("jindo") or (0.0, 0.0))[0])):
        template, by, moved = looks["template"], "looks", f"split vote {tally}, the looks sure"
    # ... widened the same day to every chihuahua the looks do not share, all
    # samples or not, when the looks name a jindo-type body (CHIHUAHUA_NOT_FOR)
    # at CHIHUAHUA_LOOKS_FLOOR or more. Of the 16 chihuahua bodies gpt-6-luna
    # had picked on eight sets with a body written down beforehand, the four
    # the looks also called chihuahua were chihuahuas; of the eleven they did
    # not, none was one by the first written body, all samples of three for
    # five of them - a black-and-white pup of 4.2 kg with drop ears (new8 3,
    # user: "템플릿 잘못 고른 거 큰 거 같고"), a black-and-tan shiba pup, a
    # fawn dog of 6.5 kg. Where the looks named a small breed (a dachshund:
    # toy poodle 0.64) or nothing above the floor (an Italian greyhound pup:
    # retriever 0.43), the chihuahua stays
    if (has_rule((votes or {}).get("read_by"), "chihuahua_by_all") and template == "chihuahua" and by == "vision"
            and tally and looks_real and looks["template"] in CHIHUAHUA_NOT_FOR
            and looks.get("confidence", 0.0) >= CHIHUAHUA_LOOKS_FLOOR):
        template, by = looks["template"], "looks"
        moved = f"a chihuahua by {tally}, the looks {looks['template']} {looks.get('confidence')}"
    if (has_rule((votes or {}).get("read_by"), "maltese_by_looks") and template == "toypoodle" and by == "vision"
            and looks_real and looks["template"] == "maltese" and looks.get("sure")
            and _weighs_within("maltese", weight_kg)):
        template, by = "maltese", "looks"
        moved = f"a toy poodle by {tally}, the looks maltese {looks.get('confidence')}"
    if template == "chihuahua" and by == "vision" and not _weighs_like("chihuahua", weight_kg):
        fits = looks_real and looks["template"] != "chihuahua" and _weighs_like(looks["template"], weight_kg)
        template, by = (looks["template"], "looks") if fits else ("jindo", "default")
        moved = f"{weight_kg} kg is above a chihuahua's weight"
    if (template == "jindo" and looks.get("template") == "jindo_puppy" and looks.get("sure") and weight_kg is not None
            and looks_weights_ok() and weight_kg < (weight_range("jindo") or (0.0, 0.0))[0]):
        template, moved = "jindo_puppy", f"a puppy by its looks, {weight_kg} kg"
    if (template == "retriever" and looks.get("template") != NO_TEMPLATE and weight_kg is not None
            and looks_weights_ok() and weight_kg <= puppy_max_kg()):
        other = looks.get("template")
        # ... the looks' model only for a dog not recorded under 60 days: a
        # 3.5 kg cream pup of that age, retriever by all three samples, went
        # to the looks' maltese at 0.67 (new8 12, the one dog of 45 chosen for
        # their age that this rule made worse); the maltese it was right for
        # was a dog of some months (new4 23)
        young = age_days is not None and age_days <= YOUNG_DAYS
        if looks_real and other != "retriever" and _weighs_within(other, weight_kg) and not young:
            moved, by = f"{weight_kg} kg is under a retriever's weight: the looks' {other}", "looks"
        else:
            other, by = "jindo_puppy", "record"
            moved = f"{weight_kg} kg is under a retriever's weight: the puppy"
        template = other
    if (age_days is not None and age_days <= YOUNG_DAYS and looks.get("template") != NO_TEMPLATE
            and (template == "retriever" or template == "jindo" and tally.get("jindo_puppy", 0) > 0)
            and (weight_kg is None or not looks_weights_ok() or weight_kg <= puppy_max_kg())):
        moved = f"under {age_days} days by the record, not the adult {template}"
        template, by = "jindo_puppy", "record"
    # ... and when the samples were split, the adult jindo's and the
    # retriever's are counted with the puppy's: a 1.5 kg pup under 60 days
    # read jindo puppy, jindo and retriever, one sample each, went to the
    # looks' maltese by "no majority" - a curly small dog for a fluffy
    # spitz-type puppy (hard photo 320, user: "템플릿 맞는지"). All three
    # samples had named a jindo-type or drop-eared body, none a small breed
    # (with one sample for the maltese the looks' maltese stays: first fresh
    # photo 10, a maltese by the truth). Two dogs on file, the other a 0.7 kg
    # pup whose written body is the puppy
    if (age_days is not None and age_days <= YOUNG_DAYS and template != "jindo_puppy"
            and looks.get("template") != NO_TEMPLATE and tally.get("jindo_puppy", 0) > 0
            and sum(tally.get(k, 0) for k in ("jindo_puppy", "jindo", "retriever")) == sum(tally.values())
            and (weight_kg is None or not looks_weights_ok() or weight_kg <= puppy_max_kg())):
        moved = f"under {age_days} days by the record, {tally}: not the {template}"
        template, by = "jindo_puppy", "record"
    heavy = False
    if template == "jindo_puppy" and weight_kg is not None and looks_weights_ok() and weight_kg > puppy_max_kg():
        template, heavy = "jindo", True
    kept = by == "vision"   # the call's match and differences are against its own model
    newborn = weight_kg is not None and weight_kg <= NEWBORN_KG
    return {"template": template, "by": by, "moved": moved,
            "match": shape.get("model_match") if kept else None,
            "differences": list(shape.get("differences") or []) if kept else [],
            "heavy_puppy": heavy or looks.get("looks") == "jindo_puppy" and looks.get("template") == "jindo",
            "looks": looks.get("template"), "looks_confidence": looks.get("confidence"),
            "no_template": looks.get("template") == NO_TEMPLATE or newborn, "newborn": newborn}


def looks_weights_ok() -> bool:
    try:
        _weights()
    except FileNotFoundError:
        return False
    return True


def weight_range(template: str) -> tuple[float, float] | None:
    """5th-95th percentile record weight (kg) of the training dogs of this
    look, None for a look without enough of them."""
    w = _weights()
    classes = [str(c) for c in w["classes"]]
    if template not in classes:
        return None
    lo, hi = w["weight_p5_p95"][classes.index(template)]
    return None if np.isnan(lo) else (float(lo), float(hi))


def template_file(slug: str, size_class: str = "small") -> str:
    """skeleton_<slug>_<size>.glb, the name metadata.resolve_base_template gives."""
    return f"skeleton_{slug}_{size_class}.glb"


__all__ = ["BACKBONE", "NO_TEMPLATE", "TEMPLATES", "combine", "decide", "embed", "pick_template", "puppy_max_kg",
           "scores", "template_file", "weight_range"]
