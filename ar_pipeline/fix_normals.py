"""Give the folded-back fur flaps vertex normals that face their own way.

WHY
---
The generated meshes have small fur flaps folded back over the coat - 185
triangles on the maltese, mostly on the curly chest and head. Their winding
agrees with their neighbours (the mesh is consistently oriented), but they
share their corner vertices with the coat they fold over, so the smoothed
vertex normals there point the coat's way, against the flap's own face. The
double-sided coat material then shades the flap from behind with a normal
turned into the body: small dark shards over a white chest, whatever the
texture says (a template painted one flat colour showed them all).

Turning the flaps' winding round hid the maltese's chest shards from the
front but drew new dark slivers on the poodle from other angles: the
winding was right, the normals were not.

HOW
---
A triangle whose face normal points against the mean of its three vertex
normals (dot below FLAP_DOT) is a flap. Each of its corners gets the
area-weighted mean face normal of the flaps it belongs to: a corner used
only by flaps is set in place; one the coat also uses is copied first (all
its attributes and morph-target deltas), so the coat keeps its normal and
the fold becomes a crease. Positions, UVs, skin and clips stay, so the masks
and baked maps stay valid and it can run at any point of the template
steps. Primitives that share vertex accessors (a split template's whiskers)
are done together.

Usage:
    python -m ar_pipeline.fix_normals --slug maltese [--size small] [--out x.glb]
Back up output/skeleton_<slug>_<size>.glb first; without --out it is edited in place.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from pygltflib import GLTF2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ar_pipeline.fix_uv_overlaps import append, read, repack  # noqa: E402

FLAP_DOT = 0.0


def flaps(points: np.ndarray, normals: np.ndarray, tri: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(mask of the triangles facing against their vertex normals, every
    triangle's area-weighted face normal)."""
    face = np.cross(points[tri[:, 1]] - points[tri[:, 0]], points[tri[:, 2]] - points[tri[:, 0]])
    return (face * normals[tri].mean(axis=1)).sum(axis=1) < FLAP_DOT, face


def _attribute_items(holder) -> list[tuple[str, int]]:
    items = holder.items() if isinstance(holder, dict) else vars(holder).items()
    return [(name, index) for name, index in items if isinstance(index, int)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--size", default="small")
    parser.add_argument("--out", default=None, help="write here instead of over the template GLB")
    args = parser.parse_args()
    glb = ROOT / "output" / f"skeleton_{args.slug}_{args.size}.glb"

    gltf = GLTF2().load(str(glb))
    blob = bytearray(gltf.binary_blob())
    groups: dict[tuple, list] = defaultdict(list)
    for mesh in gltf.meshes or []:
        for prim in mesh.primitives:
            if prim.indices is not None and prim.attributes.NORMAL is not None:
                key = (tuple(_attribute_items(prim.attributes)),
                       tuple(tuple(_attribute_items(t)) for t in (prim.targets or [])))
                groups[key].append(prim)

    changed = 0
    for (attributes, targets), prims in groups.items():
        data = {name: read(gltf, bytes(blob), index) for name, index in attributes}
        target_data = [{name: read(gltf, bytes(blob), index) for name, index in target} for target in targets]
        tris = [read(gltf, bytes(blob), prim.indices).reshape(-1, 3).astype(np.int64) for prim in prims]
        points, normals = data["POSITION"].astype(np.float64), data["NORMAL"]
        count = len(points)
        marks = [flaps(points, normals.astype(np.float64), tri) for tri in tris]
        if not any(bad.any() for bad, _ in marks):
            continue

        summed = np.zeros((count, 3))
        on_coat = np.zeros(count, bool)
        on_flap = np.zeros(count, bool)
        for tri, (bad, face) in zip(tris, marks):
            np.add.at(summed, tri[bad].ravel(), np.repeat(face[bad], 3, axis=0))
            on_flap[tri[bad].ravel()] = True
            on_coat[tri[~bad].ravel()] = True
        source = np.nonzero(on_flap & on_coat)[0]
        moved = np.arange(count)
        moved[source] = count + np.arange(len(source))

        def facing(rows: np.ndarray) -> np.ndarray:
            n = summed[rows]
            return (n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)).astype(normals.dtype)

        alone = np.nonzero(on_flap & ~on_coat)[0]
        new_normals = normals.copy()
        new_normals[alone] = facing(alone)
        for name, index in attributes:
            values = new_normals if name == "NORMAL" else data[name]
            extra = facing(source) if name == "NORMAL" else values[source]
            append(gltf, blob, index, np.concatenate([values, extra]).astype(values.dtype), 34962)
        for target, values_by_name in zip(targets, target_data):
            for name, index in target:
                values = values_by_name[name]
                append(gltf, blob, index, np.concatenate([values, values[source]]).astype(values.dtype), 34962)
        index_dtype = np.uint32 if count + len(source) > 65535 else np.uint16
        for prim, tri, (bad, _) in zip(prims, tris, marks):
            tri = tri.copy()
            tri[bad] = moved[tri[bad]]
            gltf.accessors[prim.indices].componentType = 5125 if index_dtype is np.uint32 else 5123
            append(gltf, blob, prim.indices, tri.reshape(-1).astype(index_dtype), 34963)
            changed += int(bad.sum())
            print(f"[normals] material {prim.material}: {int(bad.sum())} of {len(tri)} triangles are flaps")
        print(f"[normals] {len(alone)} vertices turned, {len(source)} copied")
    if not changed:
        print("[normals] no flaps")
        return
    blob = repack(gltf, blob)
    gltf.buffers[0].byteLength = len(blob)
    gltf.set_binary_blob(bytes(blob))
    out = Path(args.out) if args.out else glb
    gltf.save(str(out))
    print(f"[normals] wrote {out}")


if __name__ == "__main__":
    main()
