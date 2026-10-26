"""Which template a dog looks like, from its photograph alone. Local, no
API call.

The photograph becomes one vector (DINOv2 ViT-L/14, image features only,
no text) and a linear layer over it scores the templates: jindo,
jindo_puppy, maltese, toypoodle, chihuahua, retriever, and "none" for the
looks no template covers (pom-type spitz, flat-faced breeds). Age is not
asked: a puppy is told by its looks (2026-10-18, user - the template and
the coat need the looks, the size needs the weight).

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


def combine(looks: dict[str, Any] | None, shape: dict[str, Any] | None,
            weight_kg: float | None = None) -> dict[str, Any]:
    """The template to build and what the verdict needs, from the two
    readings of the photograph:
      looks  this module's pick (DINOv2): its "none" - pom-type and flat
             faces - is the refusal it is good at
      shape  body_shape.read_body_shape(models=True): the vision call's
             closest model and the differences from it, which pick the
             model a person would (20 of 22 against 13 of 22 for the looks,
             tools/template_judge.py, 2026-10-18)
    The vision call's model when it named one, else the looks' (its nearest
    real template for "none"); a jindo puppy above puppy_max_kg is built as
    the adult either way."""
    looks = looks or {}
    shape = shape or {}
    vision = shape.get("closest_model")
    by_vision = vision in TEMPLATES
    if by_vision:
        template = vision
    elif looks:
        template = looks["nearest"] if looks["template"] == NO_TEMPLATE else looks["template"]
    else:
        template = "jindo"
    heavy = False
    if template == "jindo_puppy" and weight_kg is not None and looks_weights_ok() and weight_kg > puppy_max_kg():
        template, heavy = "jindo", True
    return {"template": template, "by": "vision" if by_vision else "looks" if looks else "default",
            "match": shape.get("model_match") if by_vision else None,
            "differences": list(shape.get("differences") or []) if by_vision else [],
            "heavy_puppy": heavy or looks.get("looks") == "jindo_puppy" and looks.get("template") == "jindo",
            "looks": looks.get("template"), "looks_confidence": looks.get("confidence"),
            "no_template": looks.get("template") == NO_TEMPLATE}


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
