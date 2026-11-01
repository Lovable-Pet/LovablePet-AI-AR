"""Give a template's whisker strands their own material, so their colour is
one value per dog instead of whatever the coat paints under their UVs.

WHY
---
The generated templates model the whiskers as thin tubes welded into the
body mesh, sharing the coat material. Their UVs are collapsed onto a few
texels, most of them inside the muzzle's own chart (on the jindo puppy 91 of
the 147 whisker texels are also under skin triangles), so they took the
muzzle's colour: orange wires on a tan muzzle (shelter puppies 9 and 10),
and no texture edit could change them without spotting the muzzle.

HOW
---
A vertex within WHISKER_NEAR_HEADS nose-to-eye distances of the nose is on
a whisker when a ray from it straight inward (along -normal) meets the
surface again within WHISKER_THICK_M: the strands are ~2 mm tubes, the
muzzle is centimetres deep. Triangles whose three corners are on whiskers,
in connected strands at least WHISKER_MIN_LENGTH_M long (a lone thin flap
on the brow is not one), plus the tips and root rings the ray test misses
(see find_whiskers), move to a second primitive of the same mesh with
the material WHISKER_MATERIAL (no texture; the colour is its
baseColorFactor). The primitive shares the body's vertex accessors and
morph targets, so the skin, clips and shape keys apply to it unchanged.

The strands are also drawn in toward their axes to WHISKER_SLIM_KEEP of
their width (slim_strands): as modelled, 2 mm wide on a 5 cm muzzle, they
read as black wire once they stopped taking the muzzle's colour.

The coat under the strand roots (WHISKER_PAD_M round them, minus the
texels baked on the strands themselves) is written to whisker_pad.png in
the template folder: api.whiskers reads the finished coat there to pick
each dog's whisker colour.

Run it last, after fix_uv_overlaps and shape_keys (shape_keys wants a single
primitive, and the tools that rewrite the body's accessors would leave the
whisker primitive on the old ones). Back up the GLB first; it is edited in
place unless --out is given.

Usage:
    python -m ar_pipeline.split_whiskers --slug jindo_puppy [--size small] [--out x.glb] [--slim 0.5]
    python -m ar_pipeline.split_whiskers --slug jindo_puppy --source work/backup_whiskers_1101/skeleton_jindo_puppy_small.glb
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pygltflib import GLTF2, Accessor, Material, PbrMetallicRoughness
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.glb import WHISKER_MATERIAL, srgb_to_linear  # noqa: E402
from api.mesh import TemplateMesh  # noqa: E402
from api.whiskers import WHISKER_DARK_RGB, WHISKER_ROUGHNESS  # noqa: E402
from ar_pipeline.fix_uv_overlaps import append, read, repack  # noqa: E402

WHISKER_NEAR_HEADS = 1.6
WHISKER_THICK_M = 0.0025
WHISKER_MIN_LENGTH_M = 0.005
WHISKER_GROW_M = 0.0025
WHISKER_GROW_STEPS = 3
WHISKER_SLIM_KEEP = 0.5
WHISKER_SLIM_RADIUS_M = 0.0025
WHISKER_PAD_M = 0.008
WHISKER_CLEAR_M = 0.0015


def vertex_normals(mesh: TemplateMesh) -> np.ndarray:
    p, t = mesh.positions.astype(np.float64), mesh.triangles
    face = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
    normal = np.zeros_like(p)
    for k in range(3):
        np.add.at(normal, mesh.merged[t[:, k]], face)
    normal = normal[mesh.merged]
    return normal / (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12)


def inward_thickness(mesh: TemplateMesh, vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Distance from each vertex along -normal to the first of `triangles`
    it meets (inf if none); its own triangles are skipped."""
    p = mesh.positions.astype(np.float64)
    normal = vertex_normals(mesh)
    a, b, c = (p[mesh.triangles[triangles, k]] for k in range(3))
    e1, e2 = b - a, c - a
    corner_ids = mesh.merged[mesh.triangles[triangles]]
    out = np.full(len(vertices), np.inf)
    for start in range(0, len(vertices), 256):
        chunk = vertices[start:start + 256]
        direction = -normal[chunk][:, None, :]
        origin = p[chunk][:, None, :]
        h = np.cross(direction, e2[None])
        det = (e1[None] * h).sum(-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / det
            s = origin - a[None]
            u = inv * (s * h).sum(-1)
            q = np.cross(s, e1[None])
            v = inv * (direction * q).sum(-1)
            t = inv * (e2[None] * q).sum(-1)
        hit = (np.abs(det) > 1e-14) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-5)
        hit &= ~(corner_ids[None] == mesh.merged[chunk][:, None, None]).any(-1)
        out[start:start + len(chunk)] = np.where(hit, t, np.inf).min(1)
    return out


def find_whiskers(mesh: TemplateMesh, nose: np.ndarray, eyes: np.ndarray) -> tuple[np.ndarray, int]:
    """The triangles on whisker strands, and how many strands."""
    p = mesh.positions.astype(np.float64)
    head = float(np.linalg.norm(nose - eyes))
    near = np.nonzero(np.linalg.norm(p - nose, axis=1) < WHISKER_NEAR_HEADS * head)[0]
    targets = np.nonzero(np.linalg.norm(p[mesh.triangles].mean(1) - nose, axis=1)
                         < (WHISKER_NEAR_HEADS + 0.5) * head)[0]
    n = int(mesh.merged.max()) + 1
    thin = np.zeros(n, bool)
    thin[mesh.merged[near[inward_thickness(mesh, near, targets) < WHISKER_THICK_M]]] = True
    all_ids = mesh.merged[mesh.triangles]
    candidate = np.nonzero(thin[all_ids].all(1))[0]
    if not len(candidate):
        return candidate, 0
    ids = all_ids[candidate]
    rows = np.concatenate([ids[:, 0], ids[:, 1], ids[:, 2]])
    cols = np.concatenate([ids[:, 1], ids[:, 2], ids[:, 0]])
    _, label = connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)), directed=False)
    strand = label[ids[:, 0]]
    keep = np.zeros(len(candidate), bool)
    strands = 0
    for s in np.unique(strand):
        mine = strand == s
        corners = p[mesh.triangles[candidate[mine]].ravel()]
        if np.linalg.norm(corners.max(0) - corners.min(0)) >= WHISKER_MIN_LENGTH_M:
            keep |= mine
            strands += 1
    picked = np.zeros(len(all_ids), bool)
    picked[candidate[keep]] = True
    # the tips, the root rings and a few kinks, where the vertex normal lies
    # along the tube and the inward ray runs down it instead of across: a
    # triangle sharing an edge with a strand joins it when its other corner
    # is within WHISKER_GROW_M of the strand as first found (one tube width,
    # so the growth cannot creep over the muzzle) and it is no wider than
    # that (a skin triangle between two close roots is 6.8 mm across)
    if picked.any():
        tube = cKDTree(p[np.unique(mesh.triangles[picked].ravel())])
        corner = p[mesh.triangles]
        longest = np.max([np.linalg.norm(corner[:, a] - corner[:, b], axis=1) for a, b in ((0, 1), (1, 2), (2, 0))], 0)
        width = np.linalg.norm(np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0]), axis=1) \
            / np.maximum(longest, 1e-12)
        for _ in range(WHISKER_GROW_STEPS):
            on = np.zeros(n, bool)
            on[all_ids[picked].ravel()] = True
            edge = ~picked & (on[all_ids].sum(1) >= 2) & (width <= WHISKER_GROW_M)
            if not edge.any():
                break
            rows = np.nonzero(edge)[0]
            close = tube.query(p[mesh.triangles[rows]].reshape(-1, 3))[0].reshape(-1, 3) <= WHISKER_GROW_M
            grow = rows[close.all(1)]
            if not len(grow):
                break
            picked[grow] = True
    return np.nonzero(picked)[0], strands


def slim_strands(points: np.ndarray, tri: np.ndarray, strand: np.ndarray, keep: float) -> np.ndarray:
    """The vertex positions with each strand drawn in toward its axis to
    `keep` of its width. The axis at a vertex is the centroid of the strand
    vertices within WHISKER_SLIM_RADIUS_M (wider than the tube, so the far
    side counts). The root rings, welded to the muzzle, move too: held in
    place they stayed as barbs round a thinned strand, and the skin
    triangles they share move by under a millimetre. A vertex split off
    along a UV seam at the same spot moves with them, or the skin would
    crack open round the root."""
    used = np.unique(tri[strand].ravel())
    _, spot = np.unique(np.round(points / 1e-5).astype(np.int64), axis=0, return_inverse=True)
    spot = spot.reshape(-1)
    shift = np.zeros((int(spot.max()) + 1, 3))
    tree = cKDTree(points[used])
    for index, near in zip(used, tree.query_ball_point(points[used], WHISKER_SLIM_RADIUS_M)):
        axis = points[used[near]].mean(0)
        shift[spot[index]] = (axis - points[index]) * (1.0 - keep)
    return points + shift[spot]


def whisker_pad(mesh: TemplateMesh, whisker_tris: np.ndarray, position: np.ndarray,
                valid: np.ndarray) -> np.ndarray:
    """The texels of the muzzle round the strand roots: within WHISKER_PAD_M
    of a strand vertex that touches the rest of the mesh, and not baked on a
    strand itself."""
    p = mesh.positions.astype(np.float64)
    on = np.zeros(len(p), bool)
    on[mesh.triangles[whisker_tris].ravel()] = True
    on_ids = np.zeros(int(mesh.merged.max()) + 1, bool)
    on_ids[mesh.merged[on]] = True
    rest = np.ones(len(mesh.triangles), bool)
    rest[whisker_tris] = False
    # a root: a strand vertex also used by a triangle that stays on the body
    shared = np.zeros_like(on_ids)
    shared[mesh.merged[mesh.triangles[rest].ravel()]] = True
    roots = p[on & shared[mesh.merged]]
    rows, cols = np.nonzero(valid)
    baked = position[rows, cols].astype(np.float64)
    near_root = cKDTree(roots).query(baked, distance_upper_bound=WHISKER_PAD_M)[0] < WHISKER_PAD_M
    on_strand = cKDTree(p[on]).query(baked, distance_upper_bound=WHISKER_CLEAR_M)[0] < WHISKER_CLEAR_M
    pad = np.zeros(valid.shape, bool)
    pad[rows[near_root & ~on_strand], cols[near_root & ~on_strand]] = True
    return pad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--size", default="small")
    parser.add_argument("--out", default=None, help="write here instead of over the template GLB")
    parser.add_argument("--source", default=None, help="read this GLB instead (an unsplit backup)")
    parser.add_argument("--slim", type=float, default=WHISKER_SLIM_KEEP,
                        help="keep this share of each strand's width (1 leaves them as modelled)")
    args = parser.parse_args()
    glb = ROOT / "output" / f"skeleton_{args.slug}_{args.size}.glb"
    source = Path(args.source) if args.source else glb
    template = ROOT / "output" / "templates" / f"{args.slug}_{args.size}"

    gltf = GLTF2().load(str(source))
    if any((m.name or "") == WHISKER_MATERIAL for m in gltf.materials or []):
        raise SystemExit(f"[whiskers] {source.name} already has a {WHISKER_MATERIAL!r} material")
    points = json.loads((template / "landmarks.json").read_text(encoding="utf-8"))["points"]
    nose = np.asarray(points["nose"]["world"], dtype=np.float64)
    eyes = (np.asarray(points["left_eye"]["world"]) + np.asarray(points["right_eye"]["world"])) / 2.0

    mesh = TemplateMesh.load(source)
    whisker_tris, strands = find_whiskers(mesh, nose, eyes)
    if not strands:
        raise SystemExit("[whiskers] no whisker strands found")
    print(f"[whiskers] {strands} strands, {len(whisker_tris)} triangles")

    geometry = template / "geometry"
    position = np.load(geometry / "position.npy").astype(np.float32)
    valid = np.load(geometry / "valid.npy") > 0.5
    pad = whisker_pad(mesh, whisker_tris, position, valid)
    Image.fromarray((pad * 255).astype(np.uint8)).save(template / "whisker_pad.png")
    print(f"[whiskers] whisker_pad.png: {int(pad.sum())} texels")

    blob = bytearray(gltf.binary_blob())
    body_prim = max((p for m in gltf.meshes for p in m.primitives),
                    key=lambda p: gltf.accessors[p.attributes.POSITION].count)
    owner = next(m for m in gltf.meshes if any(p is body_prim for p in m.primitives))
    tri = read(gltf, bytes(blob), body_prim.indices).reshape(-1, 3)
    if len(tri) != len(mesh.triangles):
        raise SystemExit("[whiskers] the body primitive is not the mesh TemplateMesh read")
    strand = np.zeros(len(tri), bool)
    strand[whisker_tris] = True
    index_dtype = tri.dtype

    body_material = gltf.materials[body_prim.material]
    dark = srgb_to_linear(WHISKER_DARK_RGB)
    gltf.materials.append(Material(
        name=WHISKER_MATERIAL, doubleSided=body_material.doubleSided,
        pbrMetallicRoughness=PbrMetallicRoughness(baseColorFactor=[*dark, 1.0], metallicFactor=0.0,
                                                  roughnessFactor=WHISKER_ROUGHNESS)))
    whisker_prim = copy.deepcopy(body_prim)
    whisker_prim.material = len(gltf.materials) - 1
    # a vertex colour would multiply the whisker colour
    if getattr(whisker_prim.attributes, "COLOR_0", None) is not None:
        whisker_prim.attributes.COLOR_0 = None
    gltf.accessors.append(Accessor(componentType=gltf.accessors[body_prim.indices].componentType,
                                   type="SCALAR", count=0))
    whisker_prim.indices = len(gltf.accessors) - 1
    append(gltf, blob, whisker_prim.indices, tri[strand].reshape(-1).astype(index_dtype), 34963)
    append(gltf, blob, body_prim.indices, tri[~strand].reshape(-1).astype(index_dtype), 34963)
    owner.primitives.append(whisker_prim)
    if args.slim < 1.0:
        # both primitives read this accessor; only strand vertices move
        points = read(gltf, bytes(blob), body_prim.attributes.POSITION)
        slim = slim_strands(points.astype(np.float64), tri.astype(np.int64), strand, args.slim)
        append(gltf, blob, body_prim.attributes.POSITION, slim.astype(points.dtype), 34962)
        print(f"[whiskers] strands slimmed to {args.slim:.2f} of their width")

    blob = repack(gltf, blob)
    gltf.buffers[0].byteLength = len(blob)
    gltf.set_binary_blob(bytes(blob))
    out = Path(args.out) if args.out else glb
    gltf.save(str(out))
    print(f"[whiskers] wrote {out}")


if __name__ == "__main__":
    main()
