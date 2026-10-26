"""What the model calls of one run cost: a ledger the transports write to.

Every completion and every image edit records its token counts here, keyed
by the kind of call (photo_gate, classify, face_colours, ...) so a run's log
can say where the money went. Prices are per million tokens, from the
OpenAI pricing page as of 2026-09; they are estimates for the log, not an
invoice - the usage dashboard is the invoice.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

# $ per 1M tokens: (input, cached input, output). Image models: image input,
# cached image input, image output. Unknown models cost 0 in the estimate
# and are flagged.
PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-4o": (2.50, 1.25, 10.00),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-5": (1.25, 0.125, 10.00),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-image-1": (10.00, 2.50, 40.00),
    "gpt-image-1-mini": (2.50, 0.625, 8.00),
    "gpt-image-1.5": (8.00, 2.00, 32.00),
    "gpt-image-2": (8.00, 2.00, 30.00),
}

_lock = threading.Lock()
_calls: list[dict[str, Any]] = []


def _price_for(model: str) -> tuple[float, float, float] | None:
    for name, price in sorted(PRICES.items(), key=lambda kv: -len(kv[0])):
        if model.startswith(name):
            return price
    return None


def record(kind: str, model: str, prompt_tokens: int, completion_tokens: int,
           cached_tokens: int = 0, image_tokens: int | None = None) -> None:
    """One call. `cached_tokens` is the part of `prompt_tokens` served from
    the prompt cache; `image_tokens` the part that was images, when known."""
    price = _price_for(model)
    cost = None
    if price is not None:
        uncached = max(prompt_tokens - cached_tokens, 0)
        cost = (uncached * price[0] + cached_tokens * price[1] + completion_tokens * price[2]) / 1e6
    entry = {"kind": kind, "model": model, "prompt_tokens": int(prompt_tokens),
             "cached_tokens": int(cached_tokens), "completion_tokens": int(completion_tokens),
             "image_tokens": image_tokens, "usd": None if cost is None else round(cost, 5)}
    with _lock:
        _calls.append(entry)


def reset() -> None:
    with _lock:
        _calls.clear()


def summary() -> dict[str, Any]:
    """Calls, tokens and estimated cost, in total and per kind of call."""
    with _lock:
        calls = list(_calls)
    by_kind: dict[str, dict[str, Any]] = defaultdict(lambda: {"calls": 0, "prompt_tokens": 0, "cached_tokens": 0,
                                                              "completion_tokens": 0, "usd": 0.0})
    total = {"calls": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "usd": 0.0}
    unpriced: set[str] = set()
    for c in calls:
        for bucket in (by_kind[c["kind"]], total):
            bucket["calls"] += 1
            bucket["prompt_tokens"] += c["prompt_tokens"]
            bucket["cached_tokens"] += c["cached_tokens"]
            bucket["completion_tokens"] += c["completion_tokens"]
            if c["usd"] is not None:
                bucket["usd"] += c["usd"]
            else:
                unpriced.add(c["model"])
    for bucket in (*by_kind.values(), total):
        bucket["usd"] = round(bucket["usd"], 4)
    return {"total": total, "by_kind": dict(by_kind), "models": sorted({c["model"] for c in calls}),
            "unpriced_models": sorted(unpriced), "calls": calls}


__all__ = ["record", "reset", "summary", "PRICES"]
