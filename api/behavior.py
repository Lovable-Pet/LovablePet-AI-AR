"""The playback settings the AR client reads (`behavior` in the job JSON).

Every pet gets the same: normal speed, a tail wag on a tap. These used to come
from the shelter's temperament note (a model call tagged energy and shyness,
and a shy dog backed away on a tap); that call was dropped, the fields the
client reads were kept.
"""

from __future__ import annotations

from typing import Any, Iterable

DEFAULT_TAP_CLIP = "tail_wag"


def resolve_behavior(available_clips: Iterable[str]) -> dict[str, Any]:
    """The runtime settings the AR client needs, in one dict."""
    clips = {str(name).lower(): str(name) for name in available_clips}
    return {
        "playback_rate": 1.0,
        "tap_animation": clips.get(DEFAULT_TAP_CLIP, DEFAULT_TAP_CLIP),
        "tap_is_retreat": False,
    }
