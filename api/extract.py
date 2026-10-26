"""Stage 2 end to end: photographs in, a ten-region appearance out.

    classify the pattern against the anchors      (pattern_classify.py)
      -> instantiate the layout's defaults        (patterns.py)
      -> crop the face                            (face_crop.py)
      -> face call and body call, in parallel,    (agents.py)
         each a region-colour agent and a markings agent, three samples each
      -> merge, face > body > default             (merge.py)
      -> record which regions were inferred       (completion.py)

Every model answer is voted between independent samples; nothing reads a
confidence the model reported about itself. The result carries `source` on
every region entry and an `inferred_regions` list, so the compositor, the
critic and the log all know what was seen and what was assumed.
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agents import MARKINGS_AGENT, REGION_COLOR_AGENT, AgentResult, Transport, run_agent
from .completion import inferred_regions
from .face_crop import FaceCrop, crop_block, crop_face
from .merge import merge_scoped
from .pattern_classify import PatternDecision, classify_pattern
from .pattern_classify import describe as describe_decision
from .regions import BODY_REGIONS, FACE_REGIONS

MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
               ".webp": "image/webp", ".gif": "image/gif"}


def photo_blocks(photo_paths: list[str | Path]) -> list[dict[str, Any]]:
    blocks = []
    for path in photo_paths:
        path = Path(path)
        media = MEDIA_TYPES.get(path.suffix.lower(), "image/png")
        blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media,
                       "data": base64.standard_b64encode(path.read_bytes()).decode("ascii")},
        })
    return blocks


def _summary(result: AgentResult) -> dict[str, Any]:
    return {
        "scope": list(result.scope),
        "reported": [
            {k: v for k, v in item.items() if k in ("region", "color", "coverage", "type", "size",
                                                    "gradient_to", "gradient_anchor", "anchor",
                                                    "offset", "distance")}
            for item in result.payload.get(result.name) or []
        ],
        "confidence": round(result.confidence, 3),
        "agreement": result.agreement,
        "disagreed": result.disagreed,
        "calls": result.calls,
        "dropped": len(result.dropped),
        "errors": getattr(result, "errors", []),
    }


@dataclass
class ExtractResult:
    appearance: dict[str, Any]
    decision: PatternDecision
    face: FaceCrop | None
    body_regions: AgentResult
    body_markings: AgentResult
    face_regions: AgentResult | None
    face_markings: AgentResult | None
    merge_report: dict[str, Any]
    log: dict[str, Any] = field(default_factory=dict)


async def extract_appearance(
    transport: Transport,
    photo_paths: list[str | Path],
    pet_id: str,
    record_base_color: str,
    template_dir: str | Path,
    samples: int = 3,
    use_face_crop: bool = True,
    notes: list[str] | None = None,
) -> ExtractResult:
    template_dir = Path(template_dir)
    blocks = photo_blocks(photo_paths)
    if notes:
        blocks = blocks + [{"type": "text", "text": "About the photograph: " + " ".join(notes)}]

    decision = await classify_pattern(
        transport, blocks, template_dir / "anchors", record_base_color, samples=samples
    )
    defaults = decision.region_colors
    estimate = {entry["region"]: entry["color"] for entry in defaults}
    base_color = decision.base_color

    face: FaceCrop | None = None
    if use_face_crop:
        try:
            face = await crop_face(photo_paths[0], transport)
        except Exception as exc:  # noqa: BLE001 - the face call then sees the whole photo
            print(f"[extract] face crop failed ({type(exc).__name__}: {exc}); using the whole photo")
            face = None
    face_blocks = [crop_block(face)] if face is not None and face.visible else blocks

    body_regions, body_markings, face_regions, face_markings = await asyncio.gather(
        run_agent(REGION_COLOR_AGENT, transport, blocks, base_color, BODY_REGIONS, estimate, samples),
        run_agent(MARKINGS_AGENT, transport, blocks, base_color, BODY_REGIONS, estimate, samples),
        run_agent(REGION_COLOR_AGENT, transport, face_blocks, base_color, FACE_REGIONS, estimate, samples),
        run_agent(MARKINGS_AGENT, transport, face_blocks, base_color, FACE_REGIONS, estimate, samples),
    )

    merged, report = merge_scoped(
        pet_id, base_color, defaults, body_regions, face_regions, body_markings, face_markings,
        pattern=decision.pattern, overlay=decision.overlay,
    )
    merged["inferred_regions"] = inferred_regions(merged)
    merged["second_color"] = decision.second_color

    log = {
        "pattern": decision.as_dict(),
        "pattern_summary": describe_decision(decision),
        "face_crop": face.report() if face is not None else None,
        "body_regions": _summary(body_regions),
        "body_markings": _summary(body_markings),
        "face_regions": _summary(face_regions),
        "face_markings": _summary(face_markings),
        "merge": report,
    }
    return ExtractResult(
        appearance=merged, decision=decision, face=face,
        body_regions=body_regions, body_markings=body_markings,
        face_regions=face_regions, face_markings=face_markings,
        merge_report=report, log=log,
    )


__all__ = ["ExtractResult", "extract_appearance", "photo_blocks"]
