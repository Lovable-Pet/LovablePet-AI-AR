"""An OpenAI-backed Transport, interchangeable with AnthropicTransport.

The retry policy, the schemas and the merge never learn which provider
answered: everything provider-specific is confined to this file. Two things
have to be translated at the boundary.

Images: the agents build Anthropic image blocks, which carry the base64 payload
under source/media_type. Chat Completions wants a data: URI under image_url, so
the blocks are rewritten on the way out rather than at the call sites.

Schema: OpenAI strict mode accepts a narrower JSON Schema dialect than the
Anthropic one. Numeric bounds are dropped here instead of being removed from
the shared schemas, because the Anthropic path does enforce them. What survives
the strip is still checked in Python - see _coerce - so the two providers
produce the same shape of payload.
"""

from __future__ import annotations

import json
import os
from typing import Any

import openai

from .agents import AgentRefusal

DEFAULT_MODEL = "gpt-4o"

# Keywords strict mode rejects or silently ignores. Dropped from the copy sent
# to OpenAI; the originals stay intact for Anthropic.
_UNSUPPORTED_KEYWORDS = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "default",
)


def to_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a schema into the subset strict mode guarantees.

    Strict mode also requires every property to be listed in `required` and
    every object to forbid extra properties. Our schemas already satisfy both,
    but enforcing it here means a later edit to the shared schema cannot
    quietly break only the OpenAI path.
    """
    if not isinstance(schema, dict):
        return schema

    out = {k: v for k, v in schema.items() if k not in _UNSUPPORTED_KEYWORDS}

    if "properties" in out:
        out["properties"] = {k: to_strict_schema(v) for k, v in out["properties"].items()}
        out["required"] = list(out["properties"])
        out["additionalProperties"] = False
    if "items" in out:
        out["items"] = to_strict_schema(out["items"])

    return out


def to_openai_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for block in blocks:
        if block["type"] == "image":
            source = block["source"]
            converted.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{source['media_type']};base64,{source['data']}",
                        # a block may say how much of it to look at: "low" is
                        # one 512px tile (85 tokens) and is enough for the
                        # anchor renders, which only show WHERE colours sit;
                        # the photograph and the head close-up stay "auto"
                        "detail": block.get("detail", "auto"),
                    },
                }
            )
        else:
            converted.append(block)
    return converted


def _coerce(payload: dict[str, Any]) -> dict[str, Any]:
    """Re-apply the bounds that were stripped from the schema.

    Out-of-range values are clamped rather than rejected: a confidence of 1.2
    is a formatting slip, not a reason to throw away an otherwise usable
    reading, and the retry policy already handles genuinely poor answers.
    """
    if "confidence" in payload:
        try:
            payload["confidence"] = min(1.0, max(0.0, float(payload["confidence"])))
        except (TypeError, ValueError):
            payload["confidence"] = 0.0
    if "color_count" in payload:
        try:
            payload["color_count"] = min(4, max(1, int(payload["color_count"])))
        except (TypeError, ValueError):
            payload["color_count"] = 1
    return payload


class OpenAITransport:
    """Drop-in replacement for AnthropicTransport.

    The model is configurable because account access to a given model varies;
    it only has to be one that accepts images and supports strict structured
    output.
    """

    def __init__(
        self,
        client: openai.AsyncOpenAI | None = None,
        model: str | None = None,
    ) -> None:
        self._client = client or openai.AsyncOpenAI()
        self._model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)

    async def complete(
        self, *, system: str, blocks: list[dict[str, Any]], schema: dict[str, Any]
    ) -> dict[str, Any]:
        # No temperature and no token cap: reasoning-capable models reject a
        # temperature other than the default, and these payloads are far too
        # small for a cap to be doing any work.
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": to_openai_blocks(blocks)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "appearance",
                    "strict": True,
                    "schema": to_strict_schema(schema),
                },
            },
        )

        usage = getattr(response, "usage", None)
        if usage is not None:
            from . import usage as ledger
            from .scripted import call_kind
            details = getattr(usage, "prompt_tokens_details", None)
            ledger.record(call_kind(schema), self._model,
                          int(getattr(usage, "prompt_tokens", 0) or 0),
                          int(getattr(usage, "completion_tokens", 0) or 0),
                          int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0)

        message = response.choices[0].message
        if getattr(message, "refusal", None):
            raise AgentRefusal(f"model declined the request: {message.refusal}")
        if not message.content:
            raise AgentRefusal(
                f"empty response (finish_reason={response.choices[0].finish_reason})"
            )

        return _coerce(json.loads(message.content))
