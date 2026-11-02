"""Put a template's vertex positions and normals back from an earlier GLB,
where a later step moved them - in place.

WHY
---
`smooth_geometry --regions chest` (Taubin smoothing of the chest) was run on
the 10-12 templates and on the 10-31 maltese. It was made for the jindo,
whose chest fur lumps read as cotton wool; on the curly toy poodle, the
shaggy maltese and the feathered retriever it melted the chest's fur into a
smooth plate between furry legs (user, 11-02: "가슴면이 살짝 이상한데").
Later steps added vertices (fix_uv_overlaps gives a triangle's overlapping
corner a vertex copy of its own, at the end of the buffer) and fixed normals
(fix_normals), so the earlier GLB cannot simply be copied back.

HOW
---
The vertex order up to the earlier GLB's count is unchanged, so a vertex
whose position differs from the earlier one is one the smoothing moved: it
takes the earlier position and normal. A vertex after that count is a later
copy: with the triangles in the same order, the corner it took over says
which vertex it was copied from, and it takes that one's earlier position
and normal. (Matched by position instead, a copy on a two-sided fur card
took the normal of the card's other face - a grey hole on the retriever's
foreleg.) Then run fix_normals again (it fixes the folded flaps' normals,
any time). UVs, skin and morph targets are untouched.

    python -m ar_pipeline.restore_geometry --slug toypoodle --from work/backup_toypoodle_1012/skeleton_toypoodle_small.glb
    python -m ar_pipeline.fix_normals --slug toypoodle

Back up output/skeleton_<slug>_<size>.glb first; this edits it in place.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pygltflib import GLTF2

from ar_pipeline.smooth_geometry import accessor_view

ROOT = Path(__file__).resolve().parent.parent


def _primitive(gltf: GLTF2):
    return max((p for m in gltf.meshes for p in m.primitives), key=lambda p: gltf.accessors[p.attributes.POSITION].count)


def _triangles(gltf: GLTF2, blob: bytearray, prim) -> np.ndarray:
    acc = gltf.accessors[prim.indices]
    dtype = {5125: np.uint32, 5123: np.uint16}[acc.componentType]
    return accessor_view(gltf, blob, prim.indices, dtype, 1)[0].reshape(-1, 3).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slug", required=True)
    parser.add_argument("--size", default="small")
    parser.add_argument("--from", dest="source", required=True, help="the earlier GLB")
    parser.add_argument("--glb", default=None, help="edit this GLB instead of the template's")
    parser.add_argument("--out", default=None, help="write here instead of in place")
    args = parser.parse_args()
    glb = Path(args.glb) if args.glb else ROOT / "output" / f"skeleton_{args.slug}_{args.size}.glb"
    out = Path(args.out) if args.out else glb

    early = GLTF2().load(args.source)
    early_blob = bytearray(early.binary_blob())
    early_prim = _primitive(early)
    early_pos, _ = accessor_view(early, early_blob, early_prim.attributes.POSITION, np.float32, 3)
    early_nrm, _ = accessor_view(early, early_blob, early_prim.attributes.NORMAL, np.float32, 3)
    early_tri = _triangles(early, early_blob, early_prim)

    gltf = GLTF2().load(str(glb))
    blob = bytearray(gltf.binary_blob())
    prim = _primitive(gltf)
    pos, pos_off = accessor_view(gltf, blob, prim.attributes.POSITION, np.float32, 3)
    nrm, nrm_off = accessor_view(gltf, blob, prim.attributes.NORMAL, np.float32, 3)
    tri = _triangles(gltf, blob, prim)
    n = len(early_pos)
    if len(pos) < n or len(tri) != len(early_tri):
        raise SystemExit(f"{glb.name}: {len(pos)} vertices / {len(tri)} triangles against the earlier "
                         f"{n} / {len(early_tri)} - not the same mesh with copies appended")

    moved = np.linalg.norm(pos[:n] - early_pos, axis=1) > 1e-7
    new_pos, new_nrm = pos.copy(), nrm.copy()
    new_pos[:n][moved] = early_pos[moved]
    new_nrm[:n][moved] = early_nrm[moved]
    # the later copies, by the triangle corner each took over
    copy_of = np.full(len(pos), -1, dtype=np.int64)
    later = tri >= n
    copy_of[tri[later]] = early_tri[later]
    rows = np.nonzero(copy_of >= 0)[0]
    sources = copy_of[rows]
    restore = moved[sources]
    new_pos[rows[restore]] = early_pos[sources[restore]]
    new_nrm[rows[restore]] = early_nrm[sources[restore]]
    unused = int((copy_of[n:] < 0).sum())
    if unused:
        print(f"[restore_geometry] {unused} later vertices are no triangle's corner; left as they are")
    shift = np.linalg.norm(new_pos - pos, axis=1)
    print(f"[restore_geometry] {int(moved.sum())} vertices and {int(restore.sum())} copies put back; "
          f"shift mean {shift[shift > 0].mean() * 1000:.1f} mm, max {shift.max() * 1000:.1f} mm")

    blob[pos_off:pos_off + new_pos.nbytes] = new_pos.astype(np.float32).tobytes()
    blob[nrm_off:nrm_off + new_nrm.nbytes] = new_nrm.astype(np.float32).tobytes()
    acc = gltf.accessors[prim.attributes.POSITION]
    acc.min, acc.max = new_pos.min(0).tolist(), new_pos.max(0).tolist()
    gltf.set_binary_blob(bytes(blob))
    gltf.save(str(out))
    print(f"[restore_geometry] wrote {out}")


if __name__ == "__main__":
    main()
