"""Real-world size: the dog's shoulder height from its weight, and the scale
that puts the template at that height.

The template is modelled at an arbitrary size (the jindo is 0.83 m to the
ear tips, about 0.63 m at the withers - bigger than a real jindo). AR viewers
show a GLB in metres, so every dog came out the size of a shepherd. Here the
record's weight gives a withers height, and the whole model is scaled to it
on the scene's root node: rig, clips, UVs and texture stay as they are.

Each dog is built from the template it looks like (template_pick.py), and
the scale is uniform, so the model keeps the template breed's shape at every
size; for one shape weight goes as height cubed. Each template therefore
carries the measured withers and weight of its own breed (REFERENCE), and
the height comes from them by the cube root of the weight ratio. For the
jindo:

    withers = 50.67 cm * (weight / 20.5 kg) ** (1/3)

- 50.67 cm: mean withers of the 41 males among 195 jindos of about 1.5
  years measured by calipers in Jindo County (Park et al. 2005, J. Anim.
  Sci. & Technol. (Kor.) 47(5) 891-898, "The Conformation of the Jindo -
  the trunk and limb"; 154 bitches 47.48 cm). The paper has no weights.
- 20.5 kg: the middle of the FCI standard's male range, 18-23 kg (FCI No.
  334; bitches 15-19 kg).

Checked against spitz dogs of the same build that were not used to set it
(2026-10-03): jindo bitch 17 kg/47.5 cm -> 47.6 (+0.3%), Jeju dog male
16.7 kg/46.5 cm -> 47.3 (+1.8%) and female 13.6 kg/45.8 cm -> 44.2 (-3.6%)
(Oh et al. 2016, Korean J. Vet. Serv. 39(2) 69-74, 34 dogs, Table 1), shiba male
10.4 kg/39.4 cm -> 40.5 (+2.8%), akita male 49.9 kg/68.6 cm -> 68.2 (-0.6%)
and female 36.3 kg/63.5 cm -> 61.3 (-3.5%) (AKC, typical weights). Within
4% from 10 to 50 kg.

The rule holds only for the template's shape: a 10-12 kg corgi is 25-30 cm
and this gives 41. Looks no template covers are fit.py's to turn away, not
this module's to correct.

A template without a measured reference gets its k = withers /
weight^(1/3) from its own mesh shape (rule "shape"):

    k = 23.4 * (L/H)^-0.26 * (G/H)^-0.89

H withers, L back length (collar line to tail root), G chest girth just
behind the forelegs, the mesh ratios corrected to tape-measure ratios on the
jindo. Fitted on 21 AI Hub 71520 breed groups (5,495 adults); leaving each
breed out, it predicts that breed's k within 2.6% mean, 7.1% max
(docs/size_data/README.md 1d, tools/aihub_size/validate_shape_k.py). On the
templates that have measured references it gives jindo 18.1 (measured
18.5), retriever 18.4 (18.9), toy poodle 16.7 (17.0), chihuahua 15.7
(15.8), maltese 16.9 (15.9: the long coat's fur volume, which the jindo
correction does not take out). A measured reference therefore stays first.

The jindo puppy has no measured reference, and the size needs no age (the
template is picked by looks, the size from the weight every record has):
its shape k is 15.5. Cross-checked 2026-10-18 by mesh volume (weight =
density x volume, the density that sizes the jindo mesh to its reference):
16.0 for the puppy, within 3%; on the measured templates the volume k was
off -3% to +13%, the shape k -2% to +6%, so the shape rule stays. 1.7 kg,
the median of the shelter's jindo-type pups under 60 days, gives 18.5 cm;
Kim et al. 1989 measured 22.9 cm at 2 months (no weights). Not checked
against measured puppies.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from pygltflib import GLTF2
from scipy.spatial import ConvexHull

# template name (skeleton_<name>_<size>.glb) -> (withers m, weight kg) of a
# measured adult of the breed it is modelled on
JINDO = (0.5067, 20.5)  # Park et al. 2005 male mean, FCI male weight midpoint
REFERENCE: dict[str, tuple[float, float]] = {
    "jindo": JINDO,
    "jindoshape": JINDO,  # the jindo with morph targets, same body
    # 2026-10-12 templates from the Meshy models in work/new_dog. Measured
    # dogs only (docs/size_data/README.md 1b, 1d); k = withers / weight^(1/3).
    # AI Hub 71520 adults, n=823, medians (docs/size_data/aihub_71520_k.csv), k 15.9
    "maltese": (0.23, 3.0),
    # golden retriever, GRLS same-dog vet measurements at 11-18.5 months,
    # n=3,044 medians (Simpson 2017 Table 3), k 18.9. The 26 AI Hub goldens
    # give 55.5 cm / 31.0 kg, k 18.1; labradors are heavier (k 16.7).
    "retriever": (0.572, 27.8),
    # toy + miniature poodles measured at 12 months, n=60 (Ortiz 2015), k 17.0.
    # The AI Hub "poodle" group mixes sizes (28 / 3.8, k 18.3, height CV 23%).
    "toypoodle": (0.27, 4.0),
    # AI Hub smooth-coat chihuahuas, n=39 medians, k 15.6 (long coat n=148:
    # 20 / 2.55, k 14.6). The mesh is smooth-coated.
    "chihuahua": (0.22, 2.7),
    # jindo_puppy: none (Kim 1989 has monthly withers but no weights; AI Hub
    # has no jindo under a year), so the shape rule below, k 15.5
}


def template_name(glb_path: str | Path) -> str:
    stem = Path(glb_path).stem
    return stem.removeprefix("skeleton_").rsplit("_", 1)[0]


def withers_for_weight(weight_kg: float, template: str = "jindo") -> float:
    """Withers height (m) of an adult dog of this weight shaped like the
    template's breed."""
    w = float(weight_kg)
    if not w > 0:
        raise ValueError(f"weight must be positive, got {weight_kg!r}")
    if template not in REFERENCE:
        raise KeyError(f"no measured size reference for template {template!r}; add it to size.REFERENCE")
    ref_withers, ref_weight = REFERENCE[template]
    return ref_withers * (w / ref_weight) ** (1 / 3)


def _positions(gltf: GLTF2) -> np.ndarray:
    blob = gltf.binary_blob()
    out = []
    for mesh in gltf.meshes:
        for prim in mesh.primitives:
            acc = gltf.accessors[prim.attributes.POSITION]
            view = gltf.bufferViews[acc.bufferView]
            stride = view.byteStride or 12
            start = (view.byteOffset or 0) + (acc.byteOffset or 0)
            raw = np.frombuffer(blob, dtype=np.float32, count=acc.count * stride // 4, offset=start)
            out.append(raw.reshape(acc.count, stride // 4)[:, :3])
    return np.concatenate(out)


def _joint_world(gltf: GLTF2) -> dict[str, np.ndarray]:
    """World-space rest position of every named node (TRS or matrix)."""
    def local(node) -> np.ndarray:
        if node.matrix:
            return np.array(node.matrix, dtype=float).reshape(4, 4).T
        x, y, z, w = node.rotation or [0.0, 0.0, 0.0, 1.0]
        rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        m = np.eye(4)
        m[:3, :3] = rot * np.array(node.scale or [1.0, 1.0, 1.0])
        m[:3, 3] = node.translation or [0.0, 0.0, 0.0]
        return m

    out: dict[str, np.ndarray] = {}

    def walk(i: int, parent: np.ndarray) -> None:
        m = parent @ local(gltf.nodes[i])
        if gltf.nodes[i].name:
            out[gltf.nodes[i].name] = m[:3, 3]
        for c in gltf.nodes[i].children or []:
            walk(c, m)

    for r in gltf.scenes[gltf.scene or 0].nodes:
        walk(r, np.eye(4))
    return out


# the back line rising faster than this (m per m) is the neck, not the back
NECK_SLOPE = 0.5


def template_withers(glb_path: str | Path) -> float:
    """The template's withers height (m) in its rest pose (Y up, nose +Z).

    The back line (the top of the mesh, slice by slice) runs level from the
    hips and turns up into the neck just before the shoulders; the withers
    are the last point before it turns steep. Hip and shoulder joints come
    from the auto-rig names (L_hip_jnt, L_shoulder_jnt); without them the
    front/back feet bound the trunk."""
    gltf = GLTF2().load(str(glb_path))
    v = _positions(gltf)
    y = v[:, 1] - v[:, 1].min()
    z = v[:, 2]
    joints = _joint_world(gltf)
    if "L_hip_jnt" in joints and "L_shoulder_jnt" in joints:
        z0, z1 = float(joints["L_hip_jnt"][2]), float(joints["L_shoulder_jnt"][2])
    else:
        low = y < 0.05 * y.max()
        mid = (z.min() + z.max()) / 2
        z0, z1 = float(z[low & (z < mid)].max()), float(z[low & (z > mid)].max())
    step = (z.max() - z.min()) / 40
    half = step * 0.4
    zs = np.arange(z0, z1 + step / 2, step)
    tops = np.array([y[np.abs(z - c) < half].max() if (np.abs(z - c) < half).any() else np.nan for c in zs])
    ok = ~np.isnan(tops)
    zs, tops = zs[ok], tops[ok]
    # the neck is in the front half; behind it the line can climb steeply
    # from the rump onto the back (retriever: 0.58 at the hip, 0.65 at mid-back)
    front = (zs - z0) / (z1 - z0) > 0.5
    for k in range(1, len(tops)):
        if front[k] and (tops[k] - tops[k - 1]) / step > NECK_SLOPE:
            return float(tops[k - 1])
    return float(tops[-1])


# k = SHAPE_K[0] * (L/H)^SHAPE_K[1] * (G/H)^SHAPE_K[2], fitted on AI Hub
# 71520 breed medians (docs/size_data/README.md 1d)
SHAPE_K = (23.4, -0.26, -0.89)
# mesh ratio -> AI Hub tape ratio (length, girth), set so the jindo mesh
# gives the 65 AI Hub jindos' medians L/H 1.12, G/H 1.29: the mesh carries
# the fur volume a tape presses down
MESH_TO_TAPE = (1.017, 0.937)
SLAB = 0.01       # half thickness of a cross-section slab (m)
LEG_JUMP = 1.15   # perimeter ratio between 1 cm slabs that marks the forelegs


def _perimeter(v: np.ndarray, z: float) -> float:
    slab = v[np.abs(v[:, 2] - z) < SLAB][:, :2]
    if len(slab) < 10:
        return float("nan")
    pts = slab[ConvexHull(slab).vertices]
    return float(np.sum(np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)))


def template_shape(glb_path: str | Path) -> dict[str, float]:
    """The mesh measured the way AI Hub 71520 measured its dogs (rest pose,
    Y up, nose +Z), raw mesh ratios before MESH_TO_TAPE:
      withers  template_withers
      back     top of the back above neck_base_jnt (the collar line) to
               tail_1_jnt, straight line
      girth    convex perimeter of the 2 cm slab just behind the forelegs:
               scanning forward from the trunk middle, the narrowest of the
               3 cm before the perimeter jumps by LEG_JUMP (the legs entering)
    Needs the auto-rig's neck_base_jnt and tail_1_jnt (KeyError otherwise)."""
    gltf = GLTF2().load(str(glb_path))
    v = _positions(gltf)
    joints = _joint_world(gltf)
    for name in ("neck_base_jnt", "tail_1_jnt"):
        if name not in joints:
            raise KeyError(f"{glb_path}: no {name} to measure the body shape from")
    h = template_withers(glb_path)
    z_neck = float(joints["neck_base_jnt"][2])
    neck_top = np.array([0.0, float(v[np.abs(v[:, 2] - z_neck) < SLAB][:, 1].max()), z_neck])
    tail = joints["tail_1_jnt"].astype(float)
    tail[0] = 0.0
    back = float(np.linalg.norm(neck_top - tail))
    zs = np.arange((float(joints["tail_1_jnt"][2]) + z_neck) / 2, z_neck, 0.01)
    per = np.array([_perimeter(v, z) for z in zs])
    stop = len(zs) - 1
    for i in range(1, len(zs)):
        if per[i] > LEG_JUMP * per[i - 1]:
            stop = i - 1
            break
    # the slab right before the jump already clips the legs' front fur
    lo = max(0, stop - 2)
    i = lo + int(np.nanargmin(per[lo:stop + 1]))
    return {"withers_m": h, "back_m": back, "girth_m": float(per[i]), "z_neck": z_neck, "z_chest": float(zs[i])}


def shape_k(l_over_h: float, g_over_h: float) -> float:
    """k = withers (cm) / weight (kg)^(1/3) from tape-measure body ratios."""
    a, b, c = SHAPE_K
    return a * l_over_h ** b * g_over_h ** c


@lru_cache(maxsize=32)
def _template_k(path: str, mtime: float) -> float:
    m = template_shape(path)
    return shape_k(m["back_m"] / m["withers_m"] * MESH_TO_TAPE[0], m["girth_m"] / m["withers_m"] * MESH_TO_TAPE[1])


def template_k(glb_path: str | Path) -> float:
    """The template's k from its mesh shape (cached per file version)."""
    p = Path(glb_path)
    return _template_k(str(p.resolve()), p.stat().st_mtime)


def scale_for(weight_kg: float, glb_path: str | Path) -> dict[str, float]:
    """The weight rule with the template's measured adult reference where
    it has one, else with the k from its mesh shape; KeyError when that
    mesh cannot be measured."""
    name = template_name(glb_path)
    w = float(weight_kg)
    if not w > 0:
        raise ValueError(f"weight must be positive, got {weight_kg!r}")
    k = None
    if name in REFERENCE:
        target = withers_for_weight(w, name)
        rule = "weight"
    else:
        k = template_k(glb_path)
        target = k / 100 * w ** (1 / 3)
        rule = "shape"
    current = template_withers(glb_path)
    out = {"weight_kg": w, "withers_target_m": round(target, 3),
           "withers_template_m": round(current, 3), "scale": round(target / current, 4), "rule": rule}
    if k is not None:
        out["k"] = round(k, 2)
    return out


def apply_scale(glb_path: str | Path, scale: float, out_path: str | Path | None = None) -> Path:
    """Multiply the scene's root nodes by `scale` (uniform). The roots carry
    no animation channels in our templates; a root that does is refused,
    since a scale clip would override the size."""
    gltf = GLTF2().load(str(glb_path))
    roots = gltf.scenes[gltf.scene or 0].nodes
    animated = {c.target.node for a in gltf.animations for c in a.channels if c.target.path == "scale"}
    if animated & set(roots):
        raise ValueError(f"{glb_path}: a root node has a scale animation; cannot size it")
    for i in roots:
        node = gltf.nodes[i]
        s = node.scale or [1.0, 1.0, 1.0]
        node.scale = [float(c) * scale for c in s]
    out = Path(out_path or glb_path)
    gltf.save_binary(str(out))
    return out


__all__ = ["REFERENCE", "SHAPE_K", "apply_scale", "scale_for", "shape_k", "template_k", "template_name",
           "template_shape", "template_withers", "withers_for_weight"]
