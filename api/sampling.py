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


__all__ = ["sample"]
