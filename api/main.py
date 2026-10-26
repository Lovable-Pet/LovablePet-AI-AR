"""Stage 2 - appearance extraction.

POST /api/pets/{pet_id}/appearance-extract

The shelter record settles everything it can on its own (size class, base
template, base color, file key). Only what a photograph can answer is sent to
the model: first which coat layout it shows (against the template's own
anchors), then, region by region, where the photograph contradicts that
layout - a face call over a head crop and a body call over the whole frame,
each a region-colour agent and a markings agent sampled twice. See
extract.py for the sequence.
"""

import json
import os
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from .agents import AgentRefusal, AnthropicTransport, Transport
from .assets import template_paths
from .extract import extract_appearance
from .metadata import UnknownSpecies, process_metadata
from .schemas import AppearanceResponse, PetMetadata

MAX_PHOTOS = 4
MAX_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_MEDIA_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
_SUFFIX = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif"}

app = FastAPI(title="Lovable Pet - appearance extraction")


PROVIDER = os.environ.get("APPEARANCE_PROVIDER", "anthropic").lower()


def get_transport() -> Transport:
    """Overridden in tests via app.dependency_overrides.

    The provider is an environment choice, not a request parameter: which
    vendor answers is an operational detail and must not be something a
    caller can flip per upload.
    """
    if PROVIDER == "openai":
        # imported lazily so the openai package is only required when it is
        # actually the configured provider
        from .openai_transport import OpenAITransport

        return OpenAITransport()
    return AnthropicTransport()


def _parse_metadata(raw: str) -> PetMetadata:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"metadata is not valid JSON: {exc}") from exc
    try:
        return PetMetadata.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors(include_url=False)) from exc


async def _save_photos(photos: list[UploadFile], directory: Path) -> list[Path]:
    if not 1 <= len(photos) <= MAX_PHOTOS:
        raise HTTPException(
            status_code=422,
            detail=f"expected 1-{MAX_PHOTOS} photos, got {len(photos)}",
        )
    paths = []
    for index, photo in enumerate(photos):
        if photo.content_type not in ALLOWED_MEDIA_TYPES:
            raise HTTPException(
                status_code=422,
                detail=f"{photo.filename}: unsupported type {photo.content_type}",
            )
        data = await photo.read()
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=422,
                detail=f"{photo.filename}: {len(data)} bytes exceeds {MAX_IMAGE_BYTES}",
            )
        path = directory / f"photo_{index}{_SUFFIX[photo.content_type]}"
        path.write_bytes(data)
        paths.append(path)
    return paths


@app.post("/api/pets/{pet_id}/appearance-extract", response_model=AppearanceResponse)
async def appearance_extract(
    pet_id: str,
    metadata: str = Form(...),
    photos: list[UploadFile] = File(...),
    transport: Transport = Depends(get_transport),
) -> AppearanceResponse:
    meta = _parse_metadata(metadata)
    if meta.pet_id != pet_id:
        raise HTTPException(
            status_code=400,
            detail=f"path petId '{pet_id}' does not match 관리번호 '{meta.pet_id}'",
        )

    try:
        resolved = process_metadata(
            pet_id=meta.pet_id,
            species=meta.species,
            weight_kg=meta.weight_kg,
            coat_color=meta.coat_color,
        )
    except UnknownSpecies as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    _, masks_dir = template_paths(resolved.base_template)
    template_dir = masks_dir.parent

    with tempfile.TemporaryDirectory(prefix="appearance_") as tmp:
        paths = await _save_photos(photos, Path(tmp))
        try:
            result = await extract_appearance(
                transport, paths, resolved.pet_id, resolved.base_color, template_dir
            )
        except AgentRefusal as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    appearance = result.appearance
    print(f"[appearance] {pet_id}: {result.log['pattern_summary']}; "
          f"inferred {appearance['inferred_regions']}")

    return AppearanceResponse(
        **{k: v for k, v in appearance.items() if k != "unobserved_regions"},
        size_class=resolved.size_class,
        base_template=resolved.base_template,
        asset_path=resolved.asset_path,
        display_name=meta.name,
        display_age=meta.age_text,
    )
