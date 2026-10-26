"""Build a whole template from a model file, rigged or not, in one go.

The template a pet is painted on is a dozen files (see README, "템플릿"):
the GLB with clips, the UV layout, sixteen source masks and ten region
masks, canonical views, the baked geometry, landmarks, fur direction,
protect map and the five pattern anchors. They come from eight scripts in
a fixed order, each reading the previous one's output. This driver runs
that order for a new model:

    python ar_pipeline/new_template.py work/new_dog/model.glb --slug jindo [--size small]
        [--height 0.83] [--joints-override work/new_dog/joints_override.json] [--rigged]
        [--from seed]

  auto_rig       (unrigged only) normalise, seam-unwrap, re-bake, rig  -> work/<slug>/<slug>_rigged.blend
  seed_regions   bone weights -> 16 source vertex groups                -> work/<slug>/<slug>_seeded.blend
  build_template GLB with idle/tail_wag, uv_layout, masks_source, masks -> output/skeleton_<slug>_<size>.glb
  canonical      four fixed cameras                                     -> output/templates/<slug>_<size>/canonical
  geometry       position/normal/valid maps                             -> output/templates/<slug>_<size>/geometry
  base_color     the template's own texture next to the masks
  refine_masks   body regions from geometry and bone chains
  landmarks      landmarks.json, fur_direction.png, protect.png
  face           skull/eyebrows/cheeks/muzzle_bridge from the eyes
  anchors        five pattern renders for the classifier

--from <step> restarts at a step (auto_rig, seed, build, canonical, geometry, refine, landmarks, face, anchors)
when an earlier one is already done. A rigged input (--rigged) skips auto_rig and must carry the
shiba rig's joint names (see auto_rig.py); the model is used as is.

Afterwards: python -m api.run --template <slug> --record ... --photo ...
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.assets import BLENDER  # noqa: E402

STEPS = ["auto_rig", "seed", "build", "canonical", "geometry", "refine", "landmarks", "face", "anchors"]


def run(cmd: list[str], label: str) -> None:
    print(f"\n[new_template] === {label}: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run([str(c) for c in cmd])
    if result.returncode != 0:
        raise SystemExit(f"[new_template] {label} failed (exit {result.returncode})")


def blender(script: str, *args: str, blend: str | None = None) -> list[str]:
    # Blender exits 0 after an uncaught Python exception unless told
    # otherwise; without this a failed auto_rig left the previous .blend in
    # place and the rest of the chain rebuilt the old rig
    cmd = [BLENDER, "--background", "--python-exit-code", "1"]
    if blend:
        cmd.append(blend)
    cmd += ["--python", f"ar_pipeline/{script}", "--", *args]
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("--slug", required=True, help="template slug: output/skeleton_<slug>_<size>.glb")
    p.add_argument("--size", default="small", choices=("small", "medium", "large"))
    p.add_argument("--height", type=float, default=0.83)
    p.add_argument("--joints-override", default=None)
    p.add_argument("--rigged", action="store_true", help="the model already carries the rig; skip auto_rig")
    p.add_argument("--from", dest="start", default=STEPS[0], choices=STEPS)
    p.add_argument("--normal-blur", type=float, default=None,
                   help="normal-map blur (texels) for the mask facing tests; default 8 for a generated "
                        "(auto-rigged) model whose surface is bumpy, 0 for a --rigged asset")
    p.add_argument("--edge-clean", type=int, default=None,
                   help="edge closing/opening radius for the body masks; default 8 generated, 4 rigged")
    args = p.parse_args()
    # a generated mesh models the chest ruff and the fur as geometry, bumps
    # 20-40 texels across; the facing tests need to look past them
    normal_blur = args.normal_blur if args.normal_blur is not None else (0.0 if args.rigged else 20.0)
    args.edge_clean = args.edge_clean if args.edge_clean is not None else (4 if args.rigged else 8)
    smoothing: list[str] = []  # the geometry step smooths normal.npy itself
    label_smooth = ["--label-smooth", "14"] if not args.rigged else []

    work = Path("work") / args.slug
    work.mkdir(parents=True, exist_ok=True)
    name = f"{args.slug}_{args.size}"
    glb = Path("output") / f"skeleton_{name}.glb"
    template = Path("output/templates") / name
    rigged = work / f"{args.slug}_rigged.blend"
    seeded = work / f"{args.slug}_seeded.blend"
    start = STEPS.index(args.start)

    def due(step: str) -> bool:
        return STEPS.index(step) >= start

    if due("auto_rig") and not args.rigged:
        extra = ["--height", str(args.height), "--debug-dir", str(work)]
        if args.joints_override:
            extra += ["--joints-override", args.joints_override]
        run(blender("auto_rig.py", args.model, str(work / f"{args.slug}_rigged.glb"), *extra), "auto_rig")
    if due("seed"):
        source = str(rigged) if not args.rigged else None
        if args.rigged:
            # seed_regions reads an open .blend: import the rigged model into one
            run(blender("prepare_regions.py", args.model, str(rigged)), "prepare (rigged input)")
            source = str(rigged)
        run(blender("seed_regions.py", str(seeded), blend=source), "seed_regions")
    if due("build"):
        run(blender("build_template.py", str(seeded), str(glb)), "build_template")
    if due("canonical"):
        run(blender("canonical_views.py", str(glb), str(template / "canonical")), "canonical_views")
    if due("geometry"):
        run(blender("bake_geometry.py", str(glb), str(template / "canonical"), str(template / "geometry")),
            "bake_geometry")
        run([sys.executable, "-c",
             "import sys; sys.path.insert(0, '.'); from api.glb import extract_base_texture; "
             f"extract_base_texture(r'{glb}').save(r'{template / 'base_color.png'}')"], "base_color")
        if normal_blur > 0:
            # the baked normals of a generated mesh carry its fur bumps; every
            # consumer that tests facing (refine_masks, the compositor's
            # chest-front and inner-leg rules, the face masks) would cut a
            # fringe from them, so the map itself is smoothed once here and
            # the raw one kept beside it
            run([sys.executable, "-c",
                 "import sys; sys.path.insert(0, 'ar_pipeline'); import numpy as np; from refine_masks import smooth_normals; "
                 f"g = r'{template / 'geometry'}'; "
                 "raw = np.load(g + '/normal.npy'); valid = np.load(g + '/valid.npy') > 0.5; "
                 "np.save(g + '/normal_raw.npy', raw); "
                 f"np.save(g + '/normal.npy', smooth_normals(raw.astype(np.float32), valid, {normal_blur}).astype(raw.dtype)); "
                 f"print('[new_template] geometry normals smoothed over {normal_blur:.0f} texels')"], "smooth normals")
    if due("refine"):
        # a source region no bone reached (eyebrows without brow joints) has no
        # file; refine_masks wants all sixteen, and the face step fills it
        run([sys.executable, "-c",
             "import sys; sys.path.insert(0, '.'); from pathlib import Path; from PIL import Image; "
             "from api.regions import SOURCE_REGIONS; "
             f"d = Path(r'{template / 'masks_source'}'); "
             "[Image.new('L', (2048, 2048), 0).save(d / f'{r}.png') for r in SOURCE_REGIONS if not (d / f'{r}.png').exists()]"],
            "blank source masks")
        run([sys.executable, "ar_pipeline/refine_masks.py", "--template", name, "--glb", str(glb),
             "--geometry", str(template / "geometry"), *smoothing, *label_smooth, "--edge-clean", str(args.edge_clean)],
            "refine_masks")
    if due("landmarks"):
        run([sys.executable, "ar_pipeline/build_landmarks.py", str(glb), str(template), str(template / "geometry")],
            "build_landmarks")
    if due("face"):
        run([sys.executable, "ar_pipeline/refine_face_masks.py", "--template", name, "--glb", str(glb),
             "--geometry", str(template / "geometry"), *smoothing], "refine_face_masks")
    if due("anchors"):
        run([sys.executable, "ar_pipeline/render_anchors.py", "--template", name], "render_anchors")
    # the template's own settings, read by the pipeline (api/lite.py): a
    # generated model gets a narrower fur blend in the refinement prompt -
    # its atlas is many small islands, and a finger-wide interleave drawn
    # island by island read as torn edges at the neck and chest
    import json
    (template / "template.json").write_text(json.dumps({
        "generated": not args.rigged, "blend_width": "narrow" if not args.rigged else "normal",
        "source": str(args.model)}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[new_template] done: {glb} and {template}. Run with: python -m api.run --template {args.slug} ...")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    main()
