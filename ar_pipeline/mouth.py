"""Open a template's mouth: the lips cut apart along the lip line, the lower
jaw's skin on jaw_jnt (moved to the jaw's hinge), and a mouth inside - lips,
gums, palate, tongue and throat - so a clip can open it (pose_clips: bark).

WHY
---
user, 11-05: "진도 템플릿 짖을 수 있게 입도 움직이도록". The generated mesh
is closed at the mouth - the lip line is a crease in one surface, the head
empty inside - and auto_rig's jaw_jnt sits under the muzzle with no skin
(rig_fix gives the whole face to the head). Turned, the jaw moved nothing;
skinned without a cut it would pull the lips into a web; cut with nothing
inside, the opening would show the back faces of the head's own fur.

HOW
---
  * the lip line (lip_line): the cheapest path along the mesh's edges from
    one mouth corner round the front to the other, cheap on folded-in
    (valley) edges - the crease between the lips. The corners are the most
    folded vertices either side below the eyes (the generated mouth corner
    is a pocket), the front the most folded on the midline under the nose;
  * the cut (cut): every vertex on the line but the two corners split in
    two, the lower lip's triangles on the new copy (same position, UV,
    normal and morph deltas: the closed mouth looks as it did);
  * the hinge: jaw_jnt to the midline HINGE_BEHIND_M behind and
    HINGE_ABOVE_M above the corners, where a dog's jaw joints sit (under
    the back of the cheekbone, in front of the ear); the clips' keys on it
    and its inverse bind matrix follow;
  * the jaw's skin (jaw_share): by how much nearer a vertex lies, along the
    surface, to the lower lip than to the upper (a logistic over
    CHEEK_BLEND_M) - the chin and the underside of the jaw go with the jaw,
    the muzzle above with the head, the cheek behind the corners stretches
    between them - faded out behind the hinge (the throat) and above it;
  * the jaw deepened (deepen): its skin's underside moved down a few mm,
    the generated jaw being a thin wedge under the upper lip;
  * the mouth (mouth_mesh): its own primitive and vertex-coloured material.
    From each lip edge inward: the inside of the lip and the gum (UPPER_GUM,
    LOWER_GUM - the lower gum rising to a crest above the line of the lips,
    which gives the thin generated jaw its depth when open), then the
    palate (head) or the tongue (jaw) to the midline, darker toward the
    throat; a dog's teeth standing on each gum's crest as rounded cones
    (UPPER_TEETH, LOWER_TEETH); across the back from corner to corner a
    throat wall between palate and tongue that stretches as the jaw opens.
    Any of it the skin does not close round at rest (tuck: the generated
    lower jaw is a thin wedge) is drawn in toward the middle of the mouth,
    a tooth toward its root, until it is inside. Skinned like the lip it hangs
    from; the template's morph targets carried over from the nearest skin
    (a longer muzzle takes its mouth with it). api/glb.py repaints only the
    textured coat, so a built dog keeps the mouth's colours.

    python -m ar_pipeline.mouth --slug jindo             # work/mouth_<slug>.glb
    python -m ar_pipeline.mouth --slug jindo --install   # into the template GLB

--install backs the template up to work/backup_mouth_<slug>/ first; on a
template that already has a mouth it starts again from that backup. Run it
after rig_fix (whose --install starts from auto_rig's weights, without the
mouth) and before pose_clips (which builds bark on it).

OTHER MUZZLES (MUZZLES, --remake)
---------------------------------
Every template stands 0.83 m tall, so a toy breed's head is about 1.5x the
jindo's in the mesh and its muzzle is a different shape: the corner window
above misses, and the mouth's inside in the jindo's metres is the wrong size.
The maltese, toy poodle, jindo puppy and chihuahua came from a teammate
(2026-11-12) with the lips cut only under the nose - 8-12 cm of lip line
against the jindo's 20 - and the jaw's hinge 2 cm behind that, so the mouth
opened as a hatch in front of the muzzle with the jindo's teeth crowded into
it (user, 11-13: "토이푸들, 진도강아지, 치와와 입 자연스럽게 수정좀", then the
maltese the same way). For a template in MUZZLES:

  * the corners are looked for round a point given in nose-to-eyes units
    (how far behind the nose, how far below it), the most folded vertex on
    the outside of each cheek within CORNER_NEAR_M of it, and the front is
    the vertex nearest a point given the same way on the midline;
  * the line can be kept near the plane of its corners and front (flat_m),
    where the lips are drawn or hidden in hair and no crease leads the path;
  * the gums can stand on an arch inside the lip line (arch), where the
    muzzle's surface is hair far wider than the jaws;
  * everything measured in the jindo's metres - the hinge's offsets, the
    blend distances, the deepening, the gums, palate, tongue and teeth - is
    scaled by the length of the line the gums stand on over the jindo's
    (REFERENCE_LINE_M), so the whole dentition stands along a shorter line.

--remake takes the mouth a template already has out first (take_out: the
mouth's primitive and material dropped, the jaw's skin given back, the old
cut closed by the weld every step here works on) and keeps everything else
the template has - its clips with their jaw keys, its texture. With
--install it backs the template up to work/backup_remouth_<slug>/ once and
starts from that backup every time, so trying other corners does not stack.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from pygltflib import GLTF2, Attributes, Material, PbrMetallicRoughness, Primitive
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from scipy.special import expit

from ar_pipeline.gltf_rig import Rig, append_accessor, prune, read

ROOT = Path(__file__).resolve().parent.parent
UP = np.array([0.0, 1.0, 0.0])
MOUTH_MATERIAL = "mouth"
WELD_M = 1e-5
# the lip line: an edge costs its length times 1 + VALLEY_PREFERENCE x (1 -
# its fold / VALLEY_FULL_DEG), a fold counted only where the surface turns in
VALLEY_FULL_DEG = 30.0
VALLEY_PREFERENCE = 8.0
# where the corners are looked for: below the nose (up to CORNER_DROP_M),
# off the midline, from CORNER_BEHIND_EYES_M behind the eyes to
# CORNER_BEHIND_NOSE_M behind the nose; the front: on the midline
# (MIDLINE_M), under the nose, at most FRONT_BEHIND_NOSE_M behind it
CORNER_DROP_M = 0.04
CORNER_OFF_MIDLINE_M = 0.012
CORNER_BEHIND_EYES_M = 0.04
CORNER_BEHIND_NOSE_M = 0.05
MIDLINE_M = 0.005
FRONT_BEHIND_NOSE_M = 0.03
FRONT_UNDER_NOSE_M = 0.008
# the path stays within this of the midline, or LIP_BEYOND_CORNER_M outside
# the corners where they sit wider (a toy breed's)
LIP_HALF_WIDTH_M = 0.06
LIP_BEYOND_CORNER_M = 0.012
# templates whose lip line is placed (see OTHER MUZZLES), each read off the
# template's side and underside (the lip the texture draws, the crease the
# mesh has):
#   corner, front - where they are, in nose-to-eyes distances (behind the
#     nose, below it). A corner is the most folded vertex within
#     CORNER_NEAR_M of that point (side view) among those within
#     CORNER_OUTER_M of the outermost; the front the vertex nearest that
#     point on the midline (the crease right under a toy breed's nose folds
#     more than its lips do, and the toy poodle has no vertex on the midline
#     there);
#   flat_m - the line kept near the plane through its corners and front: an
#     edge costs 1 + (its distance from the plane / flat_m)^2 times more.
#     Where the lips are no crease in the mesh the path has nothing to
#     follow: on the chihuahua (its lips are drawn, not modelled) one side
#     ran along the lip and the other cut across under the chin, and through
#     the toy poodle's curls it zigzagged 31 cm from tuft to tuft. The jindo
#     puppy's lips are a crease that dips under its whisker pads: no plane;
#   arch - the gums and teeth stand on an arch inside the lip line that keeps
#     this share of its half-width (dental_arch), the gap between lip and
#     arch closed by a band in the coat's own material (lip_band). For a
#     muzzle whose surface is its hair: the toy poodle's lip line runs round
#     a ball of curls 13 cm wide, jagged from tuft to tuft, and its jaws are
#     half that. The maltese's lips are drawn under its nose, 6 cm wide, and
#     the line steps out from them round the moustache to the same width: its
#     arch a little narrower, to stay inside that step.
# REFERENCE_LINE_M: the jindo's lip line, the length the metres below fit
MUZZLES = {
    "maltese": {"corner": (0.88, 0.78), "front": (0.12, 0.58), "flat_m": 0.003, "arch": 0.45},
    "toypoodle": {"corner": (0.88, 0.39), "front": (0.20, 0.42), "flat_m": 0.003, "arch": 0.5},
    "jindo_puppy": {"corner": (0.93, 0.42), "front": (0.24, 0.35), "flat_m": None, "arch": None},
    "chihuahua": {"corner": (1.45, 0.57), "front": (0.31, 0.93), "flat_m": 0.004, "arch": None},
}
# the arch's front stands in from the lip line's by this share of what its
# sides do (the hair under a nose is short)
ARCH_FRONT_INSET = 0.3
CORNER_NEAR_M = 0.007
CORNER_OUTER_M = 0.012
REFERENCE_LINE_M = 0.20
# the jaw joint, from the corners (a dog's sits about a fifth of the skull's
# length behind the corners of its mouth, a little above their line)
HINGE_BEHIND_M = 0.018
HINGE_ABOVE_M = 0.012
# the jaw's share: logistic of (to the upper lip - to the lower lip) over
# CHEEK_BLEND_M; none from THROAT_FADE_M behind the hinge, all from
# JAW_FULL_AHEAD_M ahead of it; none from ABOVE_HINGE_FADE_M above it
CHEEK_BLEND_M = 0.004
THROAT_FADE_M = 0.01
JAW_FULL_AHEAD_M = 0.02
ABOVE_HINGE_FADE_M = 0.015
# the lower jaw deepened (deepen): its skin JAW_DEEPEN_M lower, by the jaw's
# share and by how far below the line of the lips it lies (all of it from
# JAW_DEEPEN_RAMP_M down; the lips' edge stays) - the generated jaw was a
# 1-1.5 cm wedge tucked under the upper lip, a sheet when open (user, 11-05:
# "밑에 턱이 너무 얇아서 이상한느낌"). A few mm keeps the template's baked
# maps good: the texture pipeline takes a texel within 2 cm of its vertex
# (api/seams.py TAP_MAX_M)
JAW_DEEPEN_M = 0.005
JAW_DEEPEN_RAMP_M = 0.008
# vertex colours (sRGB), darker than life - nothing shades the inside of the
# mouth in a viewer - and darker still by DEPTH_DARKEN toward the throat; a
# jindo's lips and the gums beside them are black
LIP_RGB = (40, 32, 34)
LIP_INNER_RGB = (58, 36, 40)
GUM_DARK_RGB = (78, 44, 50)
GUM_RGB = (140, 70, 80)
PALATE_RGB = (146, 70, 80)
TONGUE_EDGE_RGB = (150, 72, 82)
TONGUE_RGB = (192, 96, 106)
THROAT_RGB = (62, 22, 28)
TOOTH_ROOT_RGB = (212, 200, 176)
TOOTH_TIP_RGB = (244, 240, 229)
DEPTH_DARKEN = 0.55
MOUTH_ROUGHNESS = 0.45                          # wet
# the lips and gums, from each point of the lip line inward: (in, up) in m
# from the point - in across the line, up the dog's up - and the colour.
# The upper lip's inside runs up into the gap between lip and gum, the gum
# comes down to the crest the upper teeth stand on and the palate starts;
# the lower lip's inside runs down into its gap and the gum rises behind it
# to a crest above the line of the lips (at rest it sits up inside the
# upper lip, as a dog's lower teeth do): open, the jaw shows its depth - the
# generated lower jaw below the line is only a 1-1.5 cm wedge, and the lips
# turned straight in made it a sheet (user, 11-05: "밑에 턱이 너무 얇아서")
UPPER_GUM = ((0.0025, 0.0035, LIP_INNER_RGB), (0.0045, 0.0015, GUM_DARK_RGB), (0.0055, 0.0003, GUM_RGB),
             (0.0085, 0.0008, GUM_RGB), (0.0115, 0.0045, PALATE_RGB))
LOWER_GUM = ((0.0025, -0.002, LIP_INNER_RGB), (0.005, 0.0025, GUM_DARK_RGB), (0.007, 0.0045, GUM_RGB),
             (0.0095, 0.0045, GUM_RGB), (0.0115, 0.0005, TONGUE_EDGE_RGB))
CREST_ROWS = (3, 4)                             # (the lip's edge row 0) the teeth stand between these
# then INNER_ROWS rows to the midline: the palate's middle PALATE_RISE_M
# above the line of the lips, ridged; the tongue's top TONGUE_TOP_M above it.
# The front's rows end on the midline FRONT_INSET_M behind the front; the
# throat wall WALL_ROWS rows from the palate down to the tongue
INNER_ROWS = 5
PALATE_RISE_M = 0.010
PALATE_RIDGE_SHADE = 0.92
TONGUE_TOP_M = 0.004
FRONT_INSET_M = 0.02
WALL_ROWS = 4
# the teeth, each side of each jaw: (how far along the crest from the front,
# length, width along the jaw, thickness across it, the tip swept back by
# this share of the length), all m - a dog's three incisors, its canine,
# four premolars and first molar (the upper fourth premolar and the lower
# first molar the big carnassials). Each a rounded cone, TOOTH_SIDES round,
# its root TOOTH_ROOT_M into the gum, narrowing by TOOTH_PROFILE (share of
# the way to the tip, share of the root's size); none within
# TOOTH_CORNER_GAP_M of the corners. They were strips hung from every
# point of the lip line, cut to a zigzag: "이빨도 너무 삼각형조각이라 어색해"
UPPER_TEETH = ((0.0015, 0.004, 0.0026, 0.002, 0.0), (0.0043, 0.0042, 0.0026, 0.002, 0.0),
               (0.0071, 0.0045, 0.0028, 0.0022, 0.05), (0.012, 0.008, 0.005, 0.0042, 0.18),
               (0.019, 0.0025, 0.003, 0.0022, 0.1), (0.026, 0.0035, 0.005, 0.0026, 0.1),
               (0.034, 0.004, 0.006, 0.0028, 0.1), (0.044, 0.0055, 0.009, 0.0032, 0.12),
               (0.055, 0.0035, 0.008, 0.004, 0.05))
LOWER_TEETH = ((0.0015, 0.0035, 0.0024, 0.0019, 0.0), (0.0041, 0.0037, 0.0024, 0.0019, 0.0),
               (0.0067, 0.004, 0.0026, 0.002, 0.05), (0.0115, 0.0075, 0.0046, 0.004, 0.18),
               (0.018, 0.002, 0.003, 0.002, 0.1), (0.025, 0.003, 0.0045, 0.0024, 0.1),
               (0.033, 0.0035, 0.0055, 0.0026, 0.1), (0.042, 0.004, 0.0065, 0.003, 0.1),
               (0.053, 0.005, 0.01, 0.0035, 0.08))
TOOTH_SIDES = 8
TOOTH_ROOT_M = 0.001
TOOTH_PROFILE = ((0.0, 1.0), (0.4, 0.88), (0.75, 0.55))
TOOTH_CORNER_GAP_M = 0.012
# tuck: a vertex outside the skin moves TUCK_STEP of the way toward the
# middle of the mouth at the height of the lips (a tooth's toward its root),
# up to TUCK_TRIES times
TUCK_STEP = 0.25
TUCK_TRIES = 12
MAX_INFLUENCES = 4
TARGET_NEIGHBOURS = 6


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def linear(rgb: tuple[int, int, int]) -> np.ndarray:
    """An sRGB colour as the linear values glTF vertex colours take."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def weld(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(id per vertex shared by vertices at one position, position per id)."""
    _, ids = np.unique(np.round(pos / WELD_M).astype(np.int64), axis=0, return_inverse=True)
    ids = ids.ravel()
    rep = np.zeros((ids.max() + 1, 3))
    rep[ids] = pos
    return ids, rep


def edge_lengths(wt: np.ndarray, rep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The welded mesh's unique edges and their lengths."""
    pairs = np.sort(np.concatenate([wt[:, [0, 1]], wt[:, [1, 2]], wt[:, [2, 0]]]), axis=1)
    edges = np.unique(pairs, axis=0)
    return edges, np.linalg.norm(rep[edges[:, 0]] - rep[edges[:, 1]], axis=1)


def graph(edges: np.ndarray, cost: np.ndarray, n: int):
    return coo_matrix((np.r_[cost, cost], (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                      shape=(n, n)).tocsr()


def folds(wt: np.ndarray, rep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each edge two triangles share, and how far (deg) the surface folds in
    across it (0 where it folds out)."""
    fn = np.cross(rep[wt[:, 1]] - rep[wt[:, 0]], rep[wt[:, 2]] - rep[wt[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    fc = rep[wt].mean(axis=1)
    keys = np.sort(np.concatenate([wt[:, [0, 1]], wt[:, [1, 2]], wt[:, [2, 0]]]), axis=1)
    faces = np.tile(np.arange(len(wt)), 3)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    keys, faces = keys[order], faces[order]
    pair = np.flatnonzero((keys[1:] == keys[:-1]).all(axis=1))
    f0, f1 = faces[pair], faces[pair + 1]
    bend = np.degrees(np.arccos(np.clip((fn[f0] * fn[f1]).sum(axis=1), -1.0, 1.0)))
    inward = ((fc[f1] - fc[f0]) * fn[f0]).sum(axis=1) > 0
    return keys[pair], np.where(inward, bend, 0.0)


def lip_line(rep: np.ndarray, edges: np.ndarray, fold: np.ndarray, at: dict[str, np.ndarray],
             place: dict | None = None) -> list[int]:
    """Welded ids along the lip line, from the corner on the dog's left (+x)
    round the front to the corner on its right (see HOW). `place`: where the
    corners and the front are (a MUZZLES row), instead of the most folded
    vertices of the cheeks and under the nose."""
    strength = np.bincount(edges.ravel(), weights=np.repeat(fold, 2), minlength=len(rep))
    eyes = (at["L_eye_jnt"] + at["R_eye_jnt"]) / 2
    nose = at["nose_jnt"]
    x, y, z = rep.T
    cheek = (y < nose[1] + 0.005) & (y > nose[1] - CORNER_DROP_M) & \
        (z > eyes[2] - CORNER_BEHIND_EYES_M) & (z < nose[2] - CORNER_BEHIND_NOSE_M)
    if place is not None:
        muzzle = nose[2] - eyes[2]
        off = np.hypot(z - (nose[2] - place["corner"][0] * muzzle), y - (nose[1] - place["corner"][1] * muzzle))
    corners = []
    for side in (1.0, -1.0):
        if place is None:
            pool = np.flatnonzero(cheek & (side * x > CORNER_OFF_MIDLINE_M))
        else:
            pool = np.flatnonzero((off < CORNER_NEAR_M) & (side * x > CORNER_OFF_MIDLINE_M))
            if not len(pool):
                raise SystemExit(f"[mouth] no vertex within {CORNER_NEAR_M} m of the corner asked for (side {side:+.0f})")
            pool = pool[side * x[pool] > (side * x[pool]).max() - CORNER_OUTER_M]
        corners.append(int(pool[np.argmax(strength[pool])]))
    if place is None:
        pool = np.flatnonzero((np.abs(x) < MIDLINE_M) & (z > nose[2] - FRONT_BEHIND_NOSE_M) &
                              (y < nose[1] - FRONT_UNDER_NOSE_M) & (y > nose[1] - CORNER_DROP_M))
        front = int(pool[np.argmax(strength[pool])])
    else:
        front = int(np.argmin(np.linalg.norm(
            rep - [0.0, nose[1] - place["front"][1] * muzzle, nose[2] - place["front"][0] * muzzle], axis=1)))
    ends = rep[corners + [front]]
    wide = max(LIP_HALF_WIDTH_M, float(np.abs(ends[:2, 0]).max()) + LIP_BEYOND_CORNER_M)
    region = (y > ends[:, 1].min() - 0.015) & (y < ends[:, 1].max() + 0.012) & \
        (z > ends[:2, 2].min() - 0.006) & (np.abs(x) < wide)
    keep = region[edges].all(axis=1)
    length = np.linalg.norm(rep[edges[keep, 0]] - rep[edges[keep, 1]], axis=1)
    cost = length * (1.0 + VALLEY_PREFERENCE * (1.0 - np.minimum(fold[keep] / VALLEY_FULL_DEG, 1.0)))
    if place is not None and place.get("flat_m"):
        normal = np.cross(ends[0] - ends[2], ends[1] - ends[2])
        normal /= np.linalg.norm(normal)
        away = np.abs((rep[edges[keep]].mean(axis=1) - ends[2]) @ normal)
        cost *= 1.0 + (away / place["flat_m"]) ** 2
    g = graph(edges[keep], cost, len(rep))
    _, before = dijkstra(g, indices=front, return_predecessors=True)
    halves = []
    for corner in corners:
        path = [corner]
        while path[-1] != front:
            if before[path[-1]] < 0:
                raise SystemExit(f"[mouth] no path along the lips from corner {corner} to the front {front}")
            path.append(int(before[path[-1]]))
        halves.append(path)
    return halves[0] + halves[1][::-1][1:]


def cut(tri: np.ndarray, ids: np.ndarray, rep: np.ndarray, line: list[int]) -> tuple:
    """The mesh cut along `line` (see HOW): (triangles, source vertex of each
    new vertex, welded id per vertex old and new, {upper welded id: lower})."""
    inner = line[1:-1]
    wt = ids[tri]
    band = np.flatnonzero(np.isin(wt, inner).any(axis=1))
    cut_edges = {tuple(sorted(e)) for e in zip(line[:-1], line[1:])}
    parent = {f: f for f in band}

    def find(f):
        while parent[f] != f:
            parent[f] = parent[parent[f]]
            f = parent[f]
        return f

    by_edge: dict[tuple, list] = {}
    for f in band:
        a, b, c = wt[f]
        for e in ((a, b), (b, c), (c, a)):
            e = tuple(sorted(e))
            if e not in cut_edges:
                by_edge.setdefault(e, []).append(f)
    for fs in by_edge.values():
        for f in fs[1:]:
            parent[find(f)] = find(fs[0])
    groups: dict[int, list] = {}
    for f in band:
        groups.setdefault(find(f), []).append(f)
    # a side below the line: its triangles' centres lower than the line's
    # vertices they touch - of two sides, the lower one (under the jindo
    # puppy's whisker pads the upper lip's side hangs below the line too)
    height = {v: rep[v][1] for v in inner}
    below = {}
    for root, fs in groups.items():
        below[root] = -np.mean([rep[wt[f]].mean(axis=0)[1] - np.mean([height[v] for v in wt[f] if v in height])
                                for f in fs])
    if len(groups) == 2:
        lower = list(groups[max(below, key=below.get)])
    else:
        lower = [f for root, fs in groups.items() if below[root] > 0 for f in fs]
    tri = tri.copy()
    first_new = len(ids)
    new_of: dict[int, int] = {}
    lower_id = {v: len(rep) + k for k, v in enumerate(inner)}
    for f in lower:
        for c in range(3):
            if wt[f, c] in lower_id:
                o = int(tri[f, c])
                tri[f, c] = new_of.setdefault(o, first_new + len(new_of))
    src = np.array(list(new_of), dtype=np.int64)
    new_ids = np.r_[ids, [lower_id[ids[o]] for o in src]].astype(np.int64)
    return tri, src, new_ids, lower_id, len(groups)


def jaw_share(tri: np.ndarray, ids: np.ndarray, rep: np.ndarray, upper: list[int], lower: list[int],
              hinge: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """The jaw's share per welded id of the cut mesh (see HOW); its distances
    times `scale`. A vertex the surface does not reach from the lips (a
    strand of another primitive that only touches the skin) takes the gap of
    the nearest one it does reach."""
    edges, length = edge_lengths(ids[tri], rep)
    g = graph(edges, length, len(rep))
    d_up = dijkstra(g, indices=upper, min_only=True)
    d_low = dijkstra(g, indices=lower, min_only=True)
    reached = np.isfinite(d_up) & np.isfinite(d_low)
    gap = np.where(reached, d_up - d_low, 0.0)
    if not reached.all():
        _, near = cKDTree(rep[reached]).query(rep[~reached])
        gap[~reached] = gap[reached][near]
    share = expit(gap / (CHEEK_BLEND_M * scale))
    share *= smoothstep((rep[:, 2] - (hinge[2] - THROAT_FADE_M * scale)) / ((THROAT_FADE_M + JAW_FULL_AHEAD_M) * scale))
    share *= 1.0 - smoothstep((rep[:, 1] - hinge[1]) / (ABOVE_HINGE_FADE_M * scale))
    share[upper] = 0.0
    share[lower] = 1.0
    return share


def vertex_normals(pos: np.ndarray, tri: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Area-weighted normals, shared by the vertices of one welded id."""
    fn = np.cross(pos[tri[:, 1]] - pos[tri[:, 0]], pos[tri[:, 2]] - pos[tri[:, 0]])
    acc = np.zeros((ids.max() + 1, 3))
    for c in range(3):
        np.add.at(acc, ids[tri[:, c]], fn)
    acc /= np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)
    return acc[ids]


def deepen(pos: np.ndarray, normal: np.ndarray, tri: np.ndarray, ids: np.ndarray, share: np.ndarray,
           line_pos: np.ndarray, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The lower jaw's skin moved down (see JAW_DEEPEN_M, times `scale`) and
    its normals turned as its faces turn (the template's own normals kept
    otherwise): (positions, normals, how far each vertex moved)."""
    _, near = cKDTree(line_pos[:, [0, 2]]).query(pos[:, [0, 2]])
    below = line_pos[near, 1] - pos[:, 1]
    amount = JAW_DEEPEN_M * scale * share * smoothstep(below / (JAW_DEEPEN_RAMP_M * scale))
    before = vertex_normals(pos, tri, ids)
    pos = pos - UP * amount[:, None]
    turned = vertex_normals(pos, tri, ids) - before
    moved = amount > 1e-7
    normal = normal.copy()
    normal[moved] += turned[moved]
    normal[moved] /= np.maximum(np.linalg.norm(normal[moved], axis=1, keepdims=True), 1e-12)
    return pos, normal, amount


def mix(a: tuple, b: tuple, u: float) -> tuple:
    return tuple(x + (y - x) * u for x, y in zip(a, b))


def dental_arch(line_pos: np.ndarray, keep: float) -> np.ndarray:
    """A point per point of the lip line on the arch the gums stand on where
    the lips are hair (MUZZLES' arch): half an ellipse from corner to corner
    in the plane of the corners and the front, `keep` of the lip line's
    half-width wide, its front ARCH_FRONT_INSET of the sides' inset behind
    the lip line's; each point as far along it as its own is along the line."""
    along = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(line_pos, axis=0), axis=1))]
    angle = np.pi * along / along[-1]
    corners = line_pos[[0, -1]]
    centre = corners.mean(axis=0)
    wide = np.abs(corners[:, 0] - centre[0]).mean()
    front = line_pos[int(np.argmax(line_pos[:, 2]))]
    deep = front[2] - centre[2] - ARCH_FRONT_INSET * (1.0 - keep) * wide
    return np.stack([centre[0] + np.sign(corners[0, 0] - centre[0]) * keep * wide * np.cos(angle),
                     centre[1] + (front[1] - centre[1]) * np.sin(angle),
                     centre[2] + deep * np.sin(angle)], axis=1)


def mouth_mesh(line_pos: np.ndarray, scale: float = 1.0, arch: np.ndarray | None = None) -> dict:
    """The inside of the mouth (see HOW) hung on the lip line's points, its
    metres times `scale`:
    positions, colours, triangles and which way each faces (into the mouth,
    or out of a tooth), per vertex which jaw carries it (0 head ... 1 jaw)
    and where tuck draws it if it lies outside the skin, and for the
    vertices on the lips' edge the half and the point of the line they sit
    on (skinned as the lip there is; tuck leaves them). `arch`: the mouth
    hangs on these points (dental_arch) instead, each half's first row
    carried whole by its jaw ("rim": those rows' vertices, for lip_band)."""
    n = len(line_pos)
    if arch is not None:
        line_pos = arch
    xz = line_pos[:, [0, 2]]
    tangent = np.gradient(xz, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    inward = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
    flip = ((xz.mean(axis=0) - xz) * inward).sum(axis=1) < 0
    inward[flip] *= -1
    inward3 = np.stack([inward[:, 0], np.zeros(n), inward[:, 1]], axis=1)
    z_back = (line_pos[0, 2] + line_pos[-1, 2]) / 2
    inset = FRONT_INSET_M * scale
    z_front = line_pos[:, 2].max() - inset
    order = np.argsort(line_pos[:, 2])
    lip_y = lambda z: np.interp(z, line_pos[order, 2], line_pos[order, 1])
    spine_z = np.clip(line_pos[:, 2], z_back, z_front)
    spine_z[[0, -1]] = z_back
    pos, colour, carry, home, at_corner, edge = [], [], [], [], [], {}

    def add(p, rgb, jaw, to, corner=-1):
        dark = 1.0 - DEPTH_DARKEN * smoothstep((z_front + inset - p[2]) / (z_front + inset - z_back))
        pos.append(np.asarray(p, dtype=np.float64))
        colour.append(linear(rgb) * dark)
        carry.append(jaw)
        home.append(to)
        at_corner.append(corner)
        return len(pos) - 1

    grid, crest = {}, {"upper": [], "lower": []}
    for half, profile in (("upper", UPPER_GUM), ("lower", LOWER_GUM)):
        jaw = 0.0 if half == "upper" else 1.0
        for i, p in enumerate(line_pos):
            spine = np.array([0.0, lip_y(spine_z[i]), spine_z[i]])
            ring = [(p, LIP_RGB)] + [(p + (inward3[i] * d + UP * h) * scale, rgb) for d, h, rgb in profile]
            start = ring[-1][0]
            for k in range(1, INNER_ROWS + 1):
                u = k / INNER_ROWS
                q = start + (spine - start) * u
                if half == "upper":
                    q[1] = start[1] + (spine[1] + PALATE_RISE_M * scale - start[1]) * np.sin(u * np.pi / 2)
                    rgb = tuple(c * (PALATE_RIDGE_SHADE if k % 2 else 1.0) for c in PALATE_RGB)
                else:
                    q[1] = start[1] + (spine[1] + TONGUE_TOP_M * scale - start[1]) * smoothstep(u / 0.6)
                    rgb = mix(TONGUE_EDGE_RGB, TONGUE_RGB, smoothstep(u / 0.5))
                ring.append((q, rgb))
            for k, (q, rgb) in enumerate(ring):
                grid[half, i, k] = add(q, rgb, jaw, spine, {0: 0, n - 1: 1}.get(i, -1))
            if arch is None:
                edge[grid[half, i, 0]] = (half, i)
            crest[half].append((ring[CREST_ROWS[0]][0] + ring[CREST_ROWS[1]][0]) / 2)
    rows = 1 + len(UPPER_GUM) + INNER_ROWS
    tris, facing = [], []
    for half in ("upper", "lower"):
        down = -UP if half == "upper" else UP
        for i in range(n - 1):
            for k in range(rows - 1):
                a, b = grid[half, i, k], grid[half, i + 1, k]
                c, d = grid[half, i + 1, k + 1], grid[half, i, k + 1]
                tris += [(a, b, c), (a, c, d)]
                facing += [down, down]
    # the throat wall: from each corner's palate edge down to its tongue edge
    back = np.array([0.0, lip_y(z_back), z_back])
    for i in (0, n - 1):
        column = {}
        for k in range(rows):
            top, bottom = grid["upper", i, k], grid["lower", i, k]
            column[k, 0], column[k, WALL_ROWS] = top, bottom
            for m in range(1, WALL_ROWS):
                if k == 0 and arch is None:
                    column[k, m] = top
                    continue
                s = m / WALL_ROWS
                rgb = tuple(a * (1.0 - 0.3 * np.sin(s * np.pi)) for a in THROAT_RGB)
                column[k, m] = add(pos[top] + (pos[bottom] - pos[top]) * s, rgb, s, back, 0 if i == 0 else 1)
        for k in range(rows - 1):
            for m in range(WALL_ROWS):
                a, b = column[k, m], column[k + 1, m]
                c, d = column[k + 1, m + 1], column[k, m + 1]
                tris += [(a, b, c), (a, c, d)]
                facing += [np.array([0.0, 0.0, 1.0])] * 2
    # the teeth along each gum's crest, by distance from the front
    front = int(np.argmax(line_pos[:, 2]))
    angles = np.linspace(0.0, 2 * np.pi, TOOTH_SIDES, endpoint=False)
    for half, sign, teeth in (("upper", -1.0, UPPER_TEETH), ("lower", 1.0, LOWER_TEETH)):
        jaw = 0.0 if half == "upper" else 1.0
        arch = np.array(crest[half])
        along = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(arch, axis=0), axis=1))]
        for side in (1.0, -1.0):                    # toward the corner at the start of the line, the end
            for at, length, width, thick, sweep in teeth:
                at, length, width, thick = at * scale, length * scale, width * scale, thick * scale
                s = along[front] - side * at
                if not TOOTH_CORNER_GAP_M * scale < s < along[-1] - TOOTH_CORNER_GAP_M * scale:
                    continue
                root = np.array([np.interp(s, along, arch[:, c]) for c in range(3)])
                k = int(np.clip(np.searchsorted(along, s), 1, n - 1))
                run = arch[k] - arch[k - 1]
                run[1] = 0.0
                run /= max(np.linalg.norm(run), 1e-12)
                across = np.cross(UP, run)
                backward = -side * run
                axis = sign * UP
                base = root - axis * TOOTH_ROOT_M * scale
                tip = root + axis * length + backward * sweep * length
                rings = []
                for f, size in TOOTH_PROFILE:
                    centre = base + (tip - base) * f
                    rgb = mix(TOOTH_ROOT_RGB, TOOTH_TIP_RGB, f)
                    rings.append([add(centre + run * (width / 2 * size * np.cos(a)) +
                                      across * (thick / 2 * size * np.sin(a)), rgb, jaw, root) for a in angles])
                top = add(tip, TOOTH_TIP_RGB, jaw, root)
                rings.append([top] * TOOTH_SIDES)
                for lo, hi in zip(rings[:-1], rings[1:]):
                    for j in range(TOOTH_SIDES):
                        a, b, c, d = lo[j], lo[(j + 1) % TOOTH_SIDES], hi[(j + 1) % TOOTH_SIDES], hi[j]
                        out = (pos[a] + pos[b] + pos[c]) / 3 - (base + tip) / 2
                        tris += [(a, b, c), (a, c, d)]
                        facing += [out, out]
    return {"pos": np.array(pos), "colour": np.array(colour), "carry": np.array(carry), "home": np.array(home),
            "corner": np.array(at_corner),
            "tris": np.array(tris, dtype=np.int64), "facing": np.array(facing), "edge": edge,
            "rim": {half: [grid[half, i, 0] for i in range(n)] for half in ("upper", "lower")}}


def lip_band(lip_vertices: dict, rim: dict, attr: dict, dense: np.ndarray, targets: list, head_k: int,
             jaw_k: int, corner_dense: list) -> dict:
    """The band between the lips' edge and a dental arch inside it (MUZZLES'
    arch), to go in the coat's own material: from each lip vertex in to the
    arch's point for it, both jaws, and at each corner the web between the
    two when the mouth opens. An inner vertex has its lip vertex's UV (the
    band is the colour of the hair at the lip, whatever coat the dog is
    given) and morph delta, and its jaw's whole weight - at the corners the
    corner's (`corner_dense`) - as the mouth's first row there has.
    `lip_vertices`, `rim`: per half the coat's vertex and the arch's position
    at each point of the line."""
    n = len(rim["upper"])
    pos, normal, uv, weight, source = [], [], [], [], []
    index = {}
    for half, up in (("upper", -UP), ("lower", UP)):
        for i, v in enumerate(lip_vertices[half]):
            index[half, "lip", i] = len(pos)
            pos.append(attr["POSITION"][v]); normal.append(attr["NORMAL"][v]); uv.append(attr["TEXCOORD_0"][v])
            weight.append(dense[v]); source.append(v)
        for i, v in enumerate(lip_vertices[half]):
            index[half, "arch", i] = len(pos)
            rigid = np.zeros(dense.shape[1])
            rigid[head_k if half == "upper" else jaw_k] = 1.0
            if i in (0, n - 1):
                rigid = corner_dense[0 if i == 0 else 1]
            pos.append(rim[half][i]); normal.append(up); uv.append(attr["TEXCOORD_0"][v])
            weight.append(rigid); source.append(v)
    pos = np.array(pos, dtype=np.float64)
    tris, facing = [], []
    for half, up in (("upper", -UP), ("lower", UP)):
        for i in range(n - 1):
            a, b = index[half, "lip", i], index[half, "lip", i + 1]
            c, d = index[half, "arch", i + 1], index[half, "arch", i]
            tris += [(a, b, c), (a, c, d)]
            facing += [up, up]
    for i in (0, n - 1):
        tris.append((index["upper", "lip", i], index["upper", "arch", i], index["lower", "arch", i]))
        facing.append(np.array([0.0, 0.0, 1.0]))
    tris = np.array(tris, dtype=np.int64)
    fn = np.cross(pos[tris[:, 1]] - pos[tris[:, 0]], pos[tris[:, 2]] - pos[tris[:, 0]])
    flip = (fn * np.array(facing)).sum(axis=1) < 0
    tris[flip] = tris[flip][:, ::-1]
    source = np.array(source)
    return {"pos": pos, "normal": np.array(normal, dtype=np.float64), "uv": np.array(uv), "dense": np.array(weight),
            "tris": tris,
            "targets": [{key: (values[source] if key == "POSITION" else np.zeros((len(source), 3)))
                         for key, values in t.items()} for t in targets]}


def winding(points: np.ndarray, pos: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """How many times the closed surface (pos, tri) winds round each point:
    about 1 inside, 0 outside (Jacobson et al. 2013)."""
    a0, b0, c0 = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    out = np.zeros(len(points))
    for s in range(0, len(points), 64):
        q = points[s:s + 64, None, :]
        a, b, c = a0[None] - q, b0[None] - q, c0[None] - q
        la, lb, lc = (np.linalg.norm(v, axis=2) for v in (a, b, c))
        det = np.einsum("ijk,ijk->ij", a, np.cross(b, c))
        den = la * lb * lc + np.einsum("ijk,ijk->ij", a, b) * lc + np.einsum("ijk,ijk->ij", b, c) * la + \
            np.einsum("ijk,ijk->ij", c, a) * lb
        out[s:s + 64] = np.abs(2.0 * np.arctan2(det, den).sum(axis=1) / (4.0 * np.pi))
    return out


def tuck(m: dict, pos: np.ndarray, tri: np.ndarray, coat_deltas: list, mouth_deltas: list) -> int:
    """Draw the mouth's vertices that lie outside the skin (pos, tri) - at
    rest or with any morph target on full (the skin's `coat_deltas`, the
    mouth's own `mouth_deltas`; a longer muzzle pushed a canine's tip
    through the lip) - toward their homes until they are inside (see
    HOW); returns how many were."""
    tri = tri[np.linalg.norm(pos[tri].mean(axis=1) - m["pos"].mean(axis=0), axis=1) < 0.12]
    states = [(pos, np.zeros_like(m["pos"]))] + [(pos + d, own) for d, own in zip(coat_deltas, mouth_deltas)]

    def outside(idx):
        out = np.zeros(len(idx), dtype=bool)
        for skin, own in states:
            out |= winding(m["pos"][idx] + own[idx], skin, tri) < 0.5
        return idx[out]

    movable = np.ones(len(m["pos"]), dtype=bool)
    movable[list(m["edge"])] = False
    bad = outside(np.flatnonzero(movable))
    moved = len(bad)
    for _ in range(TUCK_TRIES):
        if not len(bad):
            break
        m["pos"][bad] += (m["home"][bad] - m["pos"][bad]) * TUCK_STEP
        bad = outside(bad)
    return moved


def face_normals(m: dict) -> np.ndarray:
    """Each triangle wound to face into the mouth (its `facing`), and the
    vertices' normals from them."""
    pos, tris = m["pos"], m["tris"]
    fn = np.cross(pos[tris[:, 1]] - pos[tris[:, 0]], pos[tris[:, 2]] - pos[tris[:, 0]])
    flip = (fn * m["facing"]).sum(axis=1) < 0
    tris[flip] = tris[flip][:, ::-1]
    fn[flip] *= -1
    normal = np.zeros_like(pos)
    for c in range(3):
        np.add.at(normal, tris[:, c], fn)
    return normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)


def dense_weights(joints: np.ndarray, weights: np.ndarray, n_joints: int) -> np.ndarray:
    dense = np.zeros((len(joints), n_joints))
    for k in range(joints.shape[1]):
        np.add.at(dense, (np.arange(len(joints)), joints[:, k]), weights[:, k])
    return dense / np.maximum(dense.sum(axis=1, keepdims=True), 1e-9)


def top_weights(dense: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-dense, axis=1)[:, :MAX_INFLUENCES]
    w = np.take_along_axis(dense, order, axis=1)
    return order, w / np.maximum(w.sum(axis=1, keepdims=True), 1e-9)


def move_jaw(rig: Rig, blob: bytearray, hinge: np.ndarray) -> np.ndarray:
    """jaw_jnt to `hinge` (world), its rotation kept; the inverse bind
    matrices and any clip's translation keys on it follow. Returns the
    joint's old world position."""
    gltf = rig.gltf
    jaw = rig.by_name["jaw_jnt"]
    world = rig.world()
    old = world[jaw][:3, 3].copy()
    local = (np.linalg.inv(world[rig.parent[jaw]]) @ np.r_[hinge, 1.0])[:3]
    shift = local - rig.rest_trs[jaw][0]
    gltf.nodes[jaw].translation = [float(v) for v in local]
    rig.refresh()
    world = rig.world()
    ibm = np.stack([np.linalg.inv(world[j]).T.ravel() for j in rig.joints])
    gltf.skins[0].inverseBindMatrices = append_accessor(gltf, blob, ibm, "MAT4")
    for anim in gltf.animations or []:
        for channel in anim.channels:
            if channel.target.node == jaw and channel.target.path == "translation":
                sampler = anim.samplers[channel.sampler]
                moved = read(gltf, blob, sampler.output) + shift
                anim.samplers.append(type(sampler)(input=sampler.input, interpolation=sampler.interpolation,
                                                   output=append_accessor(gltf, blob, moved, "VEC3")))
                channel.sampler = len(anim.samplers) - 1
    return old


def take_out(gltf: GLTF2, mesh) -> int:
    """The mouth `mesh` already has dropped: its primitive and its material
    (the materials after it renumbered). Returns its triangles. The jaw's
    skin is given back in apply (jaw_to_head), the old cut closed by weld."""
    index = next(i for i, m in enumerate(gltf.materials) if (m.name or "") == MOUTH_MATERIAL)
    triangles = sum(gltf.accessors[q.indices].count for q in mesh.primitives if q.material == index) // 3
    mesh.primitives = [q for q in mesh.primitives if q.material != index]
    # (and a lip band, the small primitive in the coat's material beside the coat itself)
    coat = max(mesh.primitives, key=lambda q: gltf.accessors[q.attributes.POSITION].count)
    mesh.primitives = [q for q in mesh.primitives if q is coat or q.material != coat.material]
    gltf.materials.pop(index)
    for other in gltf.meshes:
        for q in other.primitives:
            if q.material is not None and q.material > index:
                q.material -= 1
    return triangles


def jaw_to_head(dense: np.ndarray, jaw_k: int, head_k: int) -> np.ndarray:
    """Weights with the jaw's share given back: nothing on the jaw, the rest
    of each vertex's weights scaled up to one (what jaw_share's mix was made
    from); a vertex the jaw had whole goes to the head, as rig_fix gives it
    the face."""
    dense = dense.copy()
    dense[:, jaw_k] = 0.0
    whole = dense.sum(axis=1) < 0.02
    dense[whole] = 0.0
    dense[whole, head_k] = 1.0
    return dense / dense.sum(axis=1, keepdims=True)


def apply(glb: Path, out: Path, place: dict | None = None, scale: float | None = None,
          remake: bool = False) -> dict:
    """`place`, `scale`: see OTHER MUZZLES (scale None: the lip line's length
    over the jindo's where the line is placed, else 1). `remake`: take the
    mouth the GLB has out first."""
    rig = Rig(glb)
    gltf = rig.gltf
    had = any((m.name or "") == MOUTH_MATERIAL for m in gltf.materials or [])
    if had and not remake:
        raise SystemExit(f"[mouth] {glb} already has a mouth; start from the GLB without it, or --remake")
    blob = bytearray(rig.blob)
    mesh = gltf.meshes[next(n.mesh for n in gltf.nodes if n.mesh is not None and n.skin is not None)]
    taken = take_out(gltf, mesh) if had else 0
    prim = max(mesh.primitives, key=lambda p: gltf.accessors[p.attributes.POSITION].count)
    # primitives on the coat's own vertices (split_whiskers' strands)
    siblings = [q for q in mesh.primitives if q is not prim and q.attributes.POSITION == prim.attributes.POSITION]
    attr = {k: read(gltf, blob, v) for k, v in vars(prim.attributes).items() if isinstance(v, int)}
    targets = [{k: read(gltf, blob, v) for k, v in t.items()} for t in prim.targets or []]
    tri = read(gltf, blob, prim.indices).astype(np.int64).reshape(-1, 3)
    strands = [read(gltf, blob, q.indices).astype(np.int64).reshape(-1, 3) for q in siblings]
    names = [(gltf.nodes[j].name or "").split(".")[0] for j in rig.joints]
    world = rig.world()
    at = {n: world[i][:3, 3].copy() for n, i in rig.by_name.items()}

    ids, rep = weld(attr["POSITION"])
    edges, fold = folds(ids[tri], rep)
    line = lip_line(rep, edges, fold, at, place)
    lip = rep[line]
    line_m = float(np.linalg.norm(np.diff(lip, axis=0), axis=1).sum())
    arch = dental_arch(lip, place["arch"]) if place is not None and place.get("arch") else None
    if scale is None:
        stand = lip if arch is None else arch
        scale = float(np.linalg.norm(np.diff(stand, axis=0), axis=1).sum()) / REFERENCE_LINE_M if place is not None else 1.0
    tri, src, new_ids, lower_id, sides = cut(tri, ids, rep, line)
    rep2 = np.r_[rep, rep[list(lower_id)]]
    corners = rep[[line[0], line[-1]]]
    hinge = np.array([0.0, corners[:, 1].mean() + HINGE_ABOVE_M * scale, corners[:, 2].mean() - HINGE_BEHIND_M * scale])
    old_jaw = move_jaw(rig, blob, hinge)

    # the skin: every attribute and morph delta of a new vertex as its source's
    for key in attr:
        attr[key] = np.r_[attr[key], attr[key][src]]
    for t in targets:
        for key in t:
            t[key] = np.r_[t[key], t[key][src]]
    joints = attr.pop("JOINTS_0").astype(np.int64)
    w_acc = gltf.accessors[prim.attributes.WEIGHTS_0]
    weights = attr.pop("WEIGHTS_0")
    if w_acc.componentType in (5121, 5123):
        weights = weights / (255.0 if w_acc.componentType == 5121 else 65535.0)
    jaw_k = names.index("jaw_jnt")
    head_k = names.index("head_jnt")
    share = jaw_share(np.concatenate([tri] + strands), new_ids, rep2, line[1:-1], list(lower_id.values()), hinge,
                      scale)[new_ids]
    dense = dense_weights(joints, weights, len(names))
    if had:
        dense = jaw_to_head(dense, jaw_k, head_k)
    dense *= (1.0 - share)[:, None]
    dense[:, jaw_k] += share
    joints, weights = top_weights(dense)
    attr["POSITION"], attr["NORMAL"], deepened = deepen(attr["POSITION"], attr["NORMAL"], tri, new_ids, share,
                                                        lip, scale)

    # the mouth: inside on the head or the jaw, its lips' edge as the skin
    # it meets (no crack between them, however the head and jaw turn)
    m = mouth_mesh(lip, scale, arch)
    dist, near = cKDTree(attr["POSITION"][:len(ids)]).query(m["pos"], k=TARGET_NEIGHBOURS)
    idw = 1.0 / np.maximum(dist, 1e-6)
    idw /= idw.sum(axis=1, keepdims=True)
    m_targets = [np.einsum("nk,nkc->nc", idw, t["POSITION"][near]) for t in targets]
    tucked = tuck(m, attr["POSITION"], tri, [t["POSITION"] for t in targets], m_targets)
    m["normal"] = face_normals(m)
    m_dense = np.zeros((len(m["pos"]), len(names)))
    m_dense[:, head_k] = 1.0 - m["carry"]
    m_dense[:, jaw_k] = m["carry"]
    for v, (half, i) in m["edge"].items():
        wid = lower_id.get(line[i], line[i]) if half == "lower" else line[i]
        m_dense[v] = dense[np.flatnonzero(new_ids == wid)[0]]
    # the rings at the two corners (palate, tongue and the throat wall
    # between them) go as the corner's own skin does: carried whole by the
    # jaw, the lower one swung out through the cheek, whose share there is
    # half (the jindo puppy, 11-13: a black flap beside the open mouth)
    corner_dense = [dense[np.flatnonzero(new_ids == line[k])[0]] for k in (0, -1)]
    for k in (0, 1):
        m_dense[m["corner"] == k] = corner_dense[k]
    m_joints, m_weights = top_weights(m_dense)
    band = None
    if arch is not None:
        first = lambda wid: int(np.flatnonzero(new_ids == wid)[0])
        band = lip_band({"upper": [first(v) for v in line], "lower": [first(lower_id.get(v, v)) for v in line]},
                        {half: m["pos"][m["rim"][half]] for half in ("upper", "lower")}, attr, dense, targets,
                        head_k, jaw_k, corner_dense)

    # write: the coat's accessors anew (the strands' primitives on the same
    # ones), the mouth as a primitive of its own
    for key, values in attr.items():
        setattr(prim.attributes, key, append_accessor(gltf, blob, values, f"VEC{values.shape[1]}",
                                                      bounds=key == "POSITION"))
    joint_type = gltf.accessors[prim.attributes.JOINTS_0].componentType
    prim.attributes.JOINTS_0 = append_accessor(gltf, blob, joints, "VEC4", component=joint_type)
    prim.attributes.WEIGHTS_0 = append_accessor(gltf, blob, weights, "VEC4")
    index_type = 5123 if len(attr["POSITION"]) < 65535 else 5125
    prim.indices = append_accessor(gltf, blob, tri.ravel(), "SCALAR", component=index_type)
    prim.targets = [{key: append_accessor(gltf, blob, values, "VEC3", bounds=key == "POSITION")
                     for key, values in t.items()} for t in targets]
    for q in siblings:
        for key in list(attr) + ["JOINTS_0", "WEIGHTS_0"]:
            setattr(q.attributes, key, getattr(prim.attributes, key))
        q.targets = [dict(t) for t in prim.targets]
    gltf.materials.append(Material(
        name=MOUTH_MATERIAL, doubleSided=True,
        pbrMetallicRoughness=PbrMetallicRoughness(baseColorFactor=[1.0, 1.0, 1.0, 1.0], metallicFactor=0.0,
                                                  roughnessFactor=MOUTH_ROUGHNESS)))
    mouth = Primitive(
        attributes=Attributes(
            POSITION=append_accessor(gltf, blob, m["pos"], "VEC3", bounds=True),
            NORMAL=append_accessor(gltf, blob, m["normal"], "VEC3"),
            COLOR_0=append_accessor(gltf, blob, m["colour"], "VEC3"),
            JOINTS_0=append_accessor(gltf, blob, m_joints, "VEC4", component=joint_type),
            WEIGHTS_0=append_accessor(gltf, blob, m_weights, "VEC4")),
        indices=append_accessor(gltf, blob, m["tris"].ravel(), "SCALAR", component=5123),
        material=len(gltf.materials) - 1, mode=4,
        targets=[{"POSITION": append_accessor(gltf, blob, d, "VEC3", bounds=True)} for d in m_targets])
    mesh.primitives.append(mouth)
    if band is not None:
        b_joints, b_weights = top_weights(band["dense"])
        mesh.primitives.append(Primitive(
            attributes=Attributes(
                POSITION=append_accessor(gltf, blob, band["pos"], "VEC3", bounds=True),
                NORMAL=append_accessor(gltf, blob, band["normal"], "VEC3"),
                TEXCOORD_0=append_accessor(gltf, blob, band["uv"], "VEC2"),
                JOINTS_0=append_accessor(gltf, blob, b_joints, "VEC4", component=joint_type),
                WEIGHTS_0=append_accessor(gltf, blob, b_weights, "VEC4")),
            indices=append_accessor(gltf, blob, band["tris"].ravel(), "SCALAR", component=5123),
            material=prim.material, mode=4,
            targets=[{key: append_accessor(gltf, blob, values, "VEC3", bounds=key == "POSITION")
                      for key, values in t.items()} for t in band["targets"]]))
    packed = prune(gltf, bytes(blob))
    gltf.buffers[0].byteLength = len(packed)
    gltf.set_binary_blob(bytes(packed))
    out.parent.mkdir(parents=True, exist_ok=True)
    gltf.save(str(out))
    muzzle = at["nose_jnt"][2] - (at["L_eye_jnt"][2] + at["R_eye_jnt"][2]) / 2
    return {"line": len(line), "line_cm": round(line_m * 100, 1), "scale": round(float(scale), 3),
            "corners": [[round(float(v), 3) for v in c] for c in corners],
            "front": [round(float(v), 3) for v in rep[line[int(np.argmax(lip[:, 2]))]]],
            "back_of_muzzle": round(float((lip[:, 2].max() - corners[:, 2].mean()) / muzzle), 2),
            "sides": sides, "new_vertices": len(src), "taken_out": taken, "strand_primitives": len(siblings),
            "jaw": [round(float(v), 3) for v in old_jaw], "hinge": [round(float(v), 3) for v in hinge],
            "jaw_skin": int((share > 0.5).sum()), "jaw_partial": int(((share > 0.02) & (share < 0.98)).sum()),
            "deepened": int((deepened > 1e-4).sum()), "mouth": (len(m["pos"]), len(m["tris"])), "tucked": tucked,
            "lip_band": None if band is None else (len(band["pos"]), len(band["tris"]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slug", required=True)
    parser.add_argument("--size", default="small")
    parser.add_argument("--glb", default=None, help="open the mouth of this GLB instead of the template")
    parser.add_argument("--out", default=None)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--remake", action="store_true",
                        help="take the mouth the template has out first (see OTHER MUZZLES)")
    parser.add_argument("--place", nargs=6, type=float,
                        metavar=("CORNER_BACK", "CORNER_DROP", "FRONT_BACK", "FRONT_DROP", "FLAT_M", "ARCH"),
                        help="a MUZZLES row to try instead of the template's (FLAT_M 0: no plane, ARCH 0: none)")
    parser.add_argument("--scale", type=float, default=None, help="the mouth's metres times this (default: see OTHER MUZZLES)")
    args = parser.parse_args()
    template = ROOT / "output" / f"skeleton_{args.slug}_{args.size}.glb"
    place = MUZZLES.get(args.slug)
    if args.place:
        place = {"corner": tuple(args.place[:2]), "front": tuple(args.place[2:4]), "flat_m": args.place[4] or None,
                 "arch": args.place[5] or None}
    if args.remake:
        backup = ROOT / "work" / f"backup_remouth_{args.slug}" / template.name
        if args.install and not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(template, backup)
        source = Path(args.glb) if args.glb else backup if backup.exists() else template
        out = template if args.install else Path(args.out) if args.out else ROOT / "work" / f"mouth_{args.slug}.glb"
    elif args.install:
        backup = ROOT / "work" / f"backup_mouth_{args.slug}" / template.name
        has_mouth = any((m.name or "") == MOUTH_MATERIAL for m in GLTF2().load(str(template)).materials or [])
        if not has_mouth:
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(template, backup)
        elif not backup.exists():
            raise SystemExit(f"[mouth] {template} has a mouth and there is no {backup} to start again from (--remake)")
        source, out = backup, template
    else:
        source = Path(args.glb) if args.glb else template
        out = Path(args.out) if args.out else ROOT / "work" / f"mouth_{args.slug}.glb"
    report = apply(source, out, place, args.scale, args.remake)
    print(f"[mouth] {report} -> {out}")


if __name__ == "__main__":
    main()
