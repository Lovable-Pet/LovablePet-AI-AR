"""A Transport that answers from a script, for running the pipeline offline.

Every model call in the pipeline goes through Transport.complete(system,
blocks, schema). This one never touches the network: it looks at the schema
to tell which agent is asking and hands back the next canned answer for that
agent. The end-to-end tests use it so that every deterministic part of the
pipeline - the vote, the merge, the compositor, the critic loop, the log -
runs exactly as it would live, with the model's answers pinned.

The kind of call is read off the schema rather than the system prompt, so a
prompt edit cannot silently break the script.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

Answer = dict[str, Any] | Callable[[str, list[dict[str, Any]]], dict[str, Any]]


def call_kind(schema: dict[str, Any]) -> str:
    props = schema.get("properties", {})
    if "second_color" in props and "pattern" in props:
        return "classify"
    if "cheeks" in props and "brows" in props:
        return "face_colours"
    if "chest" in props and "belly" in props:
        return "body_colours"
    if "tail_carriage" in props:
        return "body_shape"
    if "region_colors" in props:
        return "region_colors"
    if "markings" in props:
        return "markings"
    if "patches" in props:
        return "critic"
    if "features" in props:
        return "features"
    if "add" in props and "remove" in props:
        return "simple_critic"
    if "face_box" in props:
        return "face_box"
    if "person_covers_animal" in props:
        return "photo_gate"
    if "cells" in props:
        return "region_cells"
    if "left_eye" in props:
        return "landmarks"
    if "not_visible" in props:
        return "completion"
    if "regions" in props and "pattern" in props:
        return "census"
    return "unknown"


class ScriptedTransport:
    """answers: kind -> list of answers, handed out in order and repeating the
    last one once the list runs out. An answer may be a callable taking
    (system, blocks) so a script can react to what it was sent."""

    def __init__(self, answers: dict[str, list[Answer]]) -> None:
        self._answers = {kind: list(items) for kind, items in answers.items()}
        self._cursor: dict[str, int] = defaultdict(int)
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self, *, system: str, blocks: list[dict[str, Any]], schema: dict[str, Any]
    ) -> dict[str, Any]:
        kind = call_kind(schema)
        self.calls.append({"kind": kind, "images": sum(1 for b in blocks if b.get("type") == "image")})
        items = self._answers.get(kind)
        if not items:
            raise RuntimeError(f"scripted transport has no answer for a {kind!r} call")
        index = min(self._cursor[kind], len(items) - 1)
        self._cursor[kind] += 1
        answer = items[index]
        if callable(answer):
            return answer(system, blocks)
        return dict(answer)

    def count(self, kind: str) -> int:
        return sum(1 for call in self.calls if call["kind"] == kind)


__all__ = ["ScriptedTransport", "call_kind"]
