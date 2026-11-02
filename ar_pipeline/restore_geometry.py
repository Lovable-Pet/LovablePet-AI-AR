"""Put a template's vertex positions and normals back from an earlier GLB,
where a later step moved them - in place.

WHY
---
`smooth_geometry --regions chest` (Taubin smoothing of the chest) was run on
every 10-12 template. It was made for the jindo, whose chest fur lumps read
as cotton wool; on the curly toy poodle it melted the curls of the chest and
throat into a smooth plate between curly legs (user, 11-02: "가슴면이 살짝
이상한데"). Later steps added vertices (fix_uv_overlaps duplicates seam
copies at the end of the buffer) and fixed normals (fix_normals), so the
earlier GLB cannot simply be copied back.

HOW
---
The vertex order up to the earlier GLB's count is unchanged, so a vertex
whose position differs from the earlier one is one the smoothing moved: it
takes the earlier position and normal. A vertex after that count is a later
seam copy: it takes the earlier position and normal of the vertex it sits
on. Then run fix_normals again (it fixes the folded flaps' normals, any
time). UVs, skin and morph targets are untouched.

    python -m ar_pipeline.restore_geometry --slug toypoodle --from work/backup_toypoodle_1012/skeleton_toypoodle_small.glb
    python -m ar_pipeline.fix_normals --slug toypoodle

Back up output/skeleton_<slug>_<size>.glb first; this edits it in place.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pygltflib import GLTF2
from scipy.spatial import cKDTree

from ar_pipeline.smooth_geometry import accessor_view

ROOT = Path(__file__).resolve().parent.parent


def _primitive(gltf: GLTF2):
    return max((p for m in gltf.meshes for p in m.primitives), key=lambda p: gltf.accessors[p.attributes.POSITION].count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slug", required=True)
    parser.add_argument("--size", default="small")
    parser.add_argument("--from", dest="source", required=True, help="the earlier GLB")
    args = parser.parse_args()
    glb = ROOT / "output" / f"skeleton_{args.slug}_{args.size}.glb"

    early = GLTF2().load(args.source)
    early_blob = bytearray(early.binary_blob())
    early_prim = _primitive(early)
    early_pos, _ = accessor_view(early, early_blob, early_prim.attributes.POSITION, np.float32, 3)
    early_nrm, _ = accessor_view(early, early_blob, early_prim.attributes.NORMAL, np.float32, 3)

    gltf = GLTF2().load(str(glb))
    blob = bytearray(gltf.binary_blob())
    prim = _primitive(gltf)
    pos, pos_off = accessor_view(gltf, blob, prim.attributes.POSITION, np.float32, 3)
    nrm, nrm_off = accessor_view(gltf, blob, prim.attributes.NORMAL, np.float32, 3)
    n = len(early_pos)
    if len(pos) < n:
        raise SystemExit(f"{glb.name} has fewer vertices ({len(pos)}) than the earlier GLB ({n})")

    moved = np.linalg.norm(pos[:n] - early_pos, axis=1) > 1e-7
    new_pos, new_nrm = pos.copy(), nrm.copy()
    new_pos[:n][moved] = early_pos[moved]
    new_nrm[:n][moved] = early_nrm[moved]
    # the later seam copies: each sits on an original vertex (its current position)
    copies = 0
    if len(pos) > n:
        distance, source = cKDTree(pos[:n]).query(pos[n:])
        on = (distance < 1e-5) & moved[source]
        rows = np.arange(n, len(pos))[on]
        new_pos[rows] = early_pos[source[on]]
        new_nrm[rows] = early_nrm[source[on]]
        copies = int(on.sum())
        if (distance >= 1e-5).any():
            print(f"[restore_geometry] {int((distance >= 1e-5).sum())} later vertices sit on no original; left as they are")
    shift = np.linalg.norm(new_pos - pos, axis=1)
    print(f"[restore_geometry] {int(moved.sum())} vertices and {copies} seam copies put back; "
          f"shift mean {shift[shift > 0].mean() * 1000:.1f} mm, max {shift.max() * 1000:.1f} mm")

    blob[pos_off:pos_off + new_pos.nbytes] = new_pos.astype(np.float32).tobytes()
    blob[nrm_off:nrm_off + new_nrm.nbytes] = new_nrm.astype(np.float32).tobytes()
    acc = gltf.accessors[prim.attributes.POSITION]
    acc.min, acc.max = new_pos.min(0).tolist(), new_pos.max(0).tolist()
    gltf.set_binary_blob(bytes(blob))
    gltf.save(str(glb))
    print(f"[restore_geometry] wrote {glb.name}")


if __name__ == "__main__":
    main()
