"""Several answers to one question, in one request when the transport can.

Every reading is asked a few times and voted. Each of those samples used to
be a request of its own, so the prompt - the photograph, the anchor renders,
the system text - was billed once per sample. A transport that has
`complete_many` is asked once for all of them: the answers are drawn
independently either way, and the prompt is billed once (measured 2026-11-12
on the face-colours call: prompt_tokens 1830 for n=1 and for n=5).
SAMPLES_IN_ONE_REQUEST=0 goes back to a request per sample.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any


def _in_one_request() -> bool:
    return os.environ.get("SAMPLES_IN_ONE_REQUEST", "1") != "0"


async def sample(transport, *, system: str, blocks: list[dict[str, Any]], schema: dict[str, Any],
                 n: int) -> list[dict[str, Any] | BaseException]:
    """`n` answers to one question. An answer that failed is its exception,
    in place - the shape asyncio.gather(return_exceptions=True) gives, so a
    caller filters them as it did. When the one request itself fails, the
    samples are asked for one by one: a transient error then costs a retry
    and not the whole reading."""
    n = max(int(n), 1)
    many = getattr(transport, "complete_many", None)
    if many is not None and n > 1 and _in_one_request():
        try:
            return list(await many(system=system, blocks=blocks, schema=schema, n=n))
        except Exception as exc:  # noqa: BLE001 - any failure of the one request falls back
            print(f"[sampling] one request for {n} answers failed ({type(exc).__name__}: {exc}); asking one by one")
    return list(await asyncio.gather(
        *(transport.complete(system=system, blocks=blocks, schema=schema) for _ in range(n)),
        return_exceptions=True))


def with_margin(blocks: list[dict[str, Any]], share: float) -> list[dict[str, Any]]:
    """The same blocks with every photograph set on a grey ground, `share`
    of its longer side wider all round. Nothing of the photograph is lost;
    the model sees it at another scale and position."""
    if share <= 0:
        return blocks
    import base64
    import io

    from PIL import Image

    out = []
    for block in blocks:
        if block.get("type") != "image":
            out.append(block)
            continue
        image = Image.open(io.BytesIO(base64.standard_b64decode(block["source"]["data"]))).convert("RGB")
        pad = round(max(image.size) * share)
        ground = Image.new("RGB", (image.width + 2 * pad, image.height + 2 * pad), (128, 128, 128))
        ground.paste(image, (pad, pad))
        buffer = io.BytesIO()
        ground.save(buffer, format="PNG")
        out.append({**block, "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.standard_b64encode(buffer.getvalue()).decode("ascii")}})
    return out


async def sample_views(transport, *, system: str, blocks: list[dict[str, Any]], schema: dict[str, Any], n: int,
                       requests: int = 1, margins: tuple[float, ...] = ()) -> list[list[dict[str, Any] | BaseException]]:
    """`requests` requests of `n` answers each, at once. With `margins`,
    request k sees the photographs with margins[k] of grey round them
    (with_margin; the first is usually 0, the photograph as it is).

    Why the views differ: gpt-6-luna gave one photograph the same answer in
    every sample of every request sent together (a tan dog's chest: unknown
    27 of 27), and another answer, as unanimous, in the build an hour
    before (cream 3 of 3) - asking the same picture again adds votes, not
    opinions. A head cropped three ways got three answers for its muzzle
    (tan 10/10, dark 4/10, dark 10/10; new shelter photo 37, 2026-11-14)."""
    requests = max(int(requests), 1)
    views = [with_margin(blocks, margins[k]) if k < len(margins) else blocks for k in range(requests)]
    return list(await asyncio.gather(*(sample(transport, system=system, blocks=view, schema=schema, n=n)
                                       for view in views)))


__all__ = ["sample", "sample_views", "with_margin"]
