"""HTTP API for the lite pipeline: the Spring server posts a record and a
photo, gets a job id back at once, and collects the GLB when it is done.

A dog takes 3-4 minutes (vision calls, one image edit, Blender renders), so
creation is asynchronous: POST returns 202 with the job, GET polls it, and
an optional callback URL is POSTed the final job when it ends. A quick
POST /v1/fit answers "will the template suit this dog" in seconds, before
anything is spent on a coat.

    uvicorn api.server:app --host 0.0.0.0 --port 8000      (from the project root)

Environment:
    OPENAI_API_KEY        vision + image model (or .env in the project root)
    PIPELINE_API_KEY      when set, every /v1 request needs X-API-Key: <it>
    PIPELINE_CONCURRENCY  jobs run at once (default 1; Blender and numpy are heavy)
    PIPELINE_PUBLIC_URL   prefix for file URLs in responses (default: relative)
    BLENDER_EXE           renders; without Blender, jobs run with render=false
    PIPELINE_ALLOW_SCRIPTED=1  accept scripted=<case> on job creation: an offline
                          canned reading (tests/scripts.py), no API calls, no cost -
                          for wiring up the Spring side

Jobs live under output/jobs/<job_id>/ (job.json, input photo, the run's
files). A restart keeps finished jobs and marks unfinished ones failed.

See docs/API.md for the contract and the Spring client.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from .assets import BLENDER
from .pipeline import load_dotenv
from .schemas import PetMetadata

ROOT = Path(__file__).resolve().parents[1]
JOBS_DIR = ROOT / "output" / "jobs"
PHOTO_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
MAX_PHOTO_BYTES = 15 * 1024 * 1024
FINAL = ("done", "failed", "photo_rejected", "unsuitable")

os.chdir(ROOT)          # the pipeline resolves output/... and tests/... from the root
load_dotenv()

app = FastAPI(title="Lovable Pet pipeline", version="1.0")
_jobs: dict[str, dict[str, Any]] = {}
_sem = asyncio.Semaphore(int(os.environ.get("PIPELINE_CONCURRENCY", "1")))
# The photo gate's readings from /v1/fit, by the photo's SHA-256: a job on the same photo judges these
# instead of calling the gate again. Kept in memory (one worker), so a restart costs one more gate call.
_gates: dict[str, tuple[float, list[dict[str, Any]]]] = {}
GATE_KEEP_S = 24 * 3600
GATE_KEEP_MAX = 1000


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _check_key(x_api_key: str | None = Header(default=None)) -> None:
    want = os.environ.get("PIPELINE_API_KEY")
    if want and x_api_key != want:
        raise HTTPException(401, "missing or wrong X-API-Key")


def _url(job_id: str, name: str) -> str:
    return f"{os.environ.get('PIPELINE_PUBLIC_URL', '').rstrip('/')}/v1/jobs/{job_id}/files/{name}"


def _save(job: dict[str, Any]) -> None:
    d = JOBS_DIR / job["job_id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")


def _public(job: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in job.items() if not k.startswith("_")}


def _photo_digest(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _keep_gate(digest: str, readings: list[dict[str, Any]]) -> None:
    import time
    now = time.monotonic()
    for key in [k for k, (at, _) in _gates.items() if now - at > GATE_KEEP_S]:
        del _gates[key]
    while len(_gates) >= GATE_KEEP_MAX:
        del _gates[min(_gates, key=lambda k: _gates[k][0])]
    _gates[digest] = (now, readings)


def _kept_gate(digest: str) -> list[dict[str, Any]] | None:
    import time
    kept = _gates.get(digest)
    return kept[1] if kept and time.monotonic() - kept[0] <= GATE_KEEP_S else None


def _parse_record(record: str) -> PetMetadata:
    from .metadata import UnknownSpecies, process_metadata
    try:
        meta = PetMetadata.model_validate(json.loads(record))
        process_metadata(meta.pet_id, meta.species, meta.weight_kg, meta.coat_color)
        return meta
    except (json.JSONDecodeError, ValidationError, UnknownSpecies) as exc:
        raise HTTPException(422, f"record: {exc}") from exc


async def _store_photo(photo: UploadFile, dest_dir: Path) -> Path:
    ext = PHOTO_TYPES.get((photo.content_type or "").lower()) or Path(photo.filename or "").suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        raise HTTPException(415, f"photo must be jpeg, png or webp (got {photo.content_type})")
    data = await photo.read()
    if not data or len(data) > MAX_PHOTO_BYTES:
        raise HTTPException(413, f"photo must be 1 byte to {MAX_PHOTO_BYTES // 1024 // 1024} MB")
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"photo{ext}"
    path.write_bytes(data)
    return path


def _fit_view(fit: dict[str, Any] | None) -> dict[str, Any] | None:
    if not fit:
        return None
    pick = fit.get("pick") or {}
    return {"level": fit["level"], "label": fit["label"], "score": fit["score"],
            "reasons": [c["why"] for c in fit.get("checks", [])], "notes": fit.get("notes", []),
            # the template built, how it was chosen ("vision": the call compared the photo with the
            # grey models; "looks": DINOv2 alone), the parts that differ, and the DINOv2 look
            # ("none": no template covers it)
            "template": fit.get("template"), "picked_by": pick.get("by"), "match": pick.get("match"),
            "differences": pick.get("differences") or [], "looks": pick.get("looks")}


def _size_view(size: dict[str, Any] | None) -> dict[str, Any] | None:
    if not size or size.get("skipped"):
        return None
    out = {"weight_kg": size["weight_kg"], "withers_m": size["withers_target_m"], "scale": size["scale"],
           "rule": size.get("rule")}   # "weight" (measured adult) or "shape" (mesh-shape k)
    return out


def _run_job_sync(job: dict[str, Any]) -> dict[str, Any]:
    """One lite run in its own thread and event loop (the pipeline does
    blocking numpy and Blender work; the API loop stays free for polling)."""
    from .fit import FitRejected
    from .lite import run_lite
    from .prepare_photo import PhotoRejected

    run_dir = JOBS_DIR / job["job_id"] / "run"
    opts = job["_options"]
    transport = image_transport = None
    if opts.get("scripted"):
        import sys
        sys.path.insert(0, str(ROOT / "tests"))
        from scripts import CASES

        from .uv_paint import ScriptedImageTransport
        transport, image_transport = CASES[opts["scripted"]](), ScriptedImageTransport()
    try:
        result = asyncio.run(run_lite(job["_record"], [job["_photo"]], run_dir, transport=transport,
                                      image_transport=image_transport,
                                      refine=opts["refine"], render=opts["render"], fit_stop=opts["fit_stop"],
                                      gate_readings=job.get("_gate")))
    except PhotoRejected as exc:
        return {"status": "photo_rejected", "error": str(exc), "reasons": list(exc.reasons)}
    except FitRejected as exc:
        return {"status": "unsuitable", "error": str(exc), "fit": _fit_view(exc.fit)}
    except Exception as exc:  # noqa: BLE001 - reported on the job
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    log = result.log
    steps = log.get("steps", {})
    final = log.get("final", {})
    pet = result.pet_id
    files = {"model.glb": run_dir / "final" / f"{pet}.glb", "texture.png": run_dir / "final" / f"{pet}.png"}
    for kind, path in (steps.get("renders") or {}).items():
        files[f"render_{kind}.png"] = Path(path)
    files = {name: str(p) for name, p in files.items() if Path(p).exists()}
    from .glb import animation_names
    from .photo_quality import quality_view
    return {
        "status": "done",
        "fit": _fit_view(steps.get("fit")),
        "photo": quality_view((steps.get("prepare") or {}).get("quality")),
        "size": _size_view(steps.get("size")),
        "model": {"glb_url": _url(job["job_id"], "model.glb"),
                  "texture_url": _url(job["job_id"], "texture.png"),
                  "renders": {n[len("render_"):-4]: _url(job["job_id"], n) for n in files if n.startswith("render_")}},
        "animations": animation_names(files["model.glb"]) if "model.glb" in files else [],
        "behavior": final.get("behavior"),
        "appearance": {"pattern": result.appearance.get("pattern"),
                       "base_color": result.appearance.get("base_color"),
                       "second_color": result.appearance.get("second_color"),
                       "eye_color": result.appearance.get("eye_color"),
                       "features": result.appearance.get("features") or []},
        "cost_usd": round(((log.get("usage") or {}).get("total") or {}).get("usd", 0.0), 4),
        "seconds": (log.get("timings_s") or {}).get("total"),
        "_files": files,
    }


async def _callback(job: dict[str, Any]) -> None:
    url = job.get("_callback_url")
    if not url:
        return
    import httpx
    headers = {}
    if os.environ.get("PIPELINE_API_KEY"):
        headers["X-API-Key"] = os.environ["PIPELINE_API_KEY"]
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.post(url, json=_public(job), headers=headers)
            if r.status_code < 300:
                job["callback"] = {"status": r.status_code, "at": _now()}
                break
            job["callback"] = {"status": r.status_code, "at": _now()}
        except Exception as exc:  # noqa: BLE001
            job["callback"] = {"error": f"{type(exc).__name__}: {exc}", "at": _now()}
        await asyncio.sleep(2 * (attempt + 1))
    _save(job)


async def _worker(job: dict[str, Any]) -> None:
    async with _sem:
        job.update(status="running", started_at=_now())
        _save(job)
        outcome = await asyncio.to_thread(_run_job_sync, job)
        job.update(outcome, finished_at=_now())
        _save(job)
    await _callback(job)


@app.on_event("startup")
async def _load_jobs() -> None:
    if not JOBS_DIR.exists():
        return
    for f in JOBS_DIR.glob("*/job.json"):
        job = json.loads(f.read_text(encoding="utf-8"))
        if job.get("status") not in FINAL:
            job.update(status="failed", error="server restarted before the job finished", finished_at=_now())
            _save(job)
        _jobs[job["job_id"]] = job


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "blender": Path(BLENDER).exists(), "openai_key": bool(os.environ.get("OPENAI_API_KEY")),
            "running": sum(j.get("status") == "running" for j in _jobs.values()),
            "queued": sum(j.get("status") == "queued" for j in _jobs.values())}


@app.post("/v1/jobs", status_code=202, dependencies=[Depends(_check_key)])
async def create_job(record: str = Form(..., description="shelter record JSON (Korean or English keys)"),
                     photo: UploadFile = File(...),
                     callback_url: str | None = Form(None),
                     fit_stop: bool = Form(False, description="stop before the coat when the verdict is poor"),
                     refine: bool = Form(True, description="false: no image-model edit (mask build only)"),
                     render: bool | None = Form(None, description="preview renders; default when Blender exists"),
                     scripted: str | None = Form(None, description="offline case (PIPELINE_ALLOW_SCRIPTED=1 only)")
                     ) -> JSONResponse:
    meta = _parse_record(record)
    if scripted and os.environ.get("PIPELINE_ALLOW_SCRIPTED") != "1":
        raise HTTPException(403, "scripted runs are off (PIPELINE_ALLOW_SCRIPTED=1 to allow)")
    job_id = f"{meta.pet_id}-{uuid.uuid4().hex[:8]}"
    photo_path = await _store_photo(photo, JOBS_DIR / job_id / "input")
    job: dict[str, Any] = {
        "job_id": job_id, "pet_id": meta.pet_id, "status": "queued", "created_at": _now(),
        "_record": json.loads(record), "_photo": str(photo_path), "_callback_url": callback_url,
        "_options": {"fit_stop": fit_stop, "refine": refine, "scripted": scripted,
                     "render": Path(BLENDER).exists() if render is None else render},
    }
    # /v1/fit gated this photo already: the job judges those readings instead of calling again
    # (not for scripted runs, which answer the gate from their script)
    kept = None if scripted else _kept_gate(_photo_digest(photo_path))
    if kept:
        job["_gate"] = kept
    _jobs[job_id] = job
    _save(job)
    asyncio.create_task(_worker(job))
    return JSONResponse(_public(job), status_code=202, headers={"Location": f"/v1/jobs/{job_id}"})


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(_check_key)])
async def get_job(job_id: str) -> dict[str, Any]:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return _public(job)


@app.get("/v1/jobs/{job_id}/files/{name}", dependencies=[Depends(_check_key)])
async def get_file(job_id: str, name: str) -> FileResponse:
    job = _jobs.get(job_id)
    path = (job or {}).get("_files", {}).get(name)
    if not path or not Path(path).exists():
        raise HTTPException(404, "no such file (job not done, or not produced)")
    media = "model/gltf-binary" if name.endswith(".glb") else "image/png"
    return FileResponse(path, media_type=media, filename=f"{job['pet_id']}_{name}")


@app.delete("/v1/jobs/{job_id}", status_code=204, dependencies=[Depends(_check_key)])
async def delete_job(job_id: str) -> None:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    if job.get("status") not in FINAL:
        raise HTTPException(409, "job still running")
    _jobs.pop(job_id)
    shutil.rmtree(JOBS_DIR / job_id, ignore_errors=True)


@app.post("/v1/fit", dependencies=[Depends(_check_key)])
async def fit_check(record: str = Form(...), photo: UploadFile = File(...)) -> dict[str, Any]:
    """Seconds, about $0.02. The photo gate first (prepare_photo.py, about
    $0.006): a photo it refuses (a hand over the coat, two animals, text on
    the dog, ...) is answered at once as 부적합 with the gate's reasons and
    fit.photo_rejected, and nothing more is spent. The gate's readings are
    kept, so a job on the same photo does not call it again. Then the
    body-shape call on the photo with the grey model sheet (the outline and
    the closest model), the DINOv2 looks (local), the verdict and the real
    size. No coat, no pattern (the pattern only adds a caveat for
    brindle/merle/other layouts, read in the full job)."""
    from PIL import Image, ImageOps

    from .body_shape import read_body_shape
    from .extract import photo_blocks
    from .fit import assess_fit
    from .pipeline import default_transport
    from .prepare_photo import gate_photo, seen_from_above
    from .size import scale_for
    from .template_pick import combine, pick_template, template_file
    from .watermark import remove_watermarks

    from .photo_quality import quality_of_file, quality_view

    meta = _parse_record(record)
    tmp = JOBS_DIR / "_fit" / uuid.uuid4().hex[:8]
    path = await _store_photo(photo, tmp)
    try:
        transport = default_transport()
        # the gate sees what a job's gate sees: the photo upright, a known shelter logo painted out
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        image, watermarks = remove_watermarks(image)
        rejected, _, readings = await gate_photo(transport, image)
        if readings:
            _keep_gate(_photo_digest(path), readings)
        if rejected:
            return _rejected_fit(meta.pet_id, rejected)
        if watermarks:
            path = tmp / "photo_clean.png"
            image.save(path)
        # local: is the photograph big, sharp and exposed enough to read
        quality = await asyncio.to_thread(quality_of_file, path)
        try:
            looks = await asyncio.to_thread(pick_template, path, meta.weight_kg)
        except Exception:  # noqa: BLE001 - no weights or no backbone: the vision call alone
            looks = None
        # the outline and the closest grey model in one call (body_shape.py)
        answers, votes = await read_body_shape(transport, photo_blocks([path]), models=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"vision call failed: {type(exc).__name__}: {exc}") from exc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    pick = combine(looks, answers, meta.weight_kg)
    fit = assess_fit(meta.species, meta.weight_kg, meta.age_text, answers, None, votes=votes, photo_quality=quality,
                     pick=pick, from_above=seen_from_above(readings))
    from .assets import template_paths
    from .metadata import process_metadata
    resolved = process_metadata(meta.pet_id, meta.species, meta.weight_kg, meta.coat_color)
    base = template_file(fit["template"], resolved.size_class)
    glb, _ = template_paths(base)
    try:
        size = _size_view(scale_for(meta.weight_kg, glb))
    except KeyError as exc:  # a mesh without the joints to measure its shape
        size = {"skipped": True, "reason": str(exc)}
    return {"pet_id": meta.pet_id, "fit": {**_fit_view(fit), "photo_rejected": []}, "size": size,
            "shape": answers, "photo": quality_view(quality)}


def _rejected_fit(pet_id: str, reasons: list[str]) -> dict[str, Any]:
    """/v1/fit for a photo the gate refused: the shape of a normal answer,
    부적합 with the gate's Korean reasons (the job's photo_rejected ones)."""
    from .prepare_photo import REJECT_RULES
    return {"pet_id": pet_id,
            "fit": {"level": "poor", "label": "부적합", "score": 0, "reasons": [REJECT_RULES[r] for r in reasons],
                    "notes": [], "template": None, "picked_by": None, "match": None, "differences": [],
                    "looks": None, "photo_rejected": reasons},
            "size": None, "shape": None, "photo": None}


__all__ = ["app"]
