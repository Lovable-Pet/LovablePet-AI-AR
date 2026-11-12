"""A dog's body shape: default morph-target weights written into its GLB.

The template carries the targets (ar_pipeline/shape_keys.py); a dog's shape
is only which of them apply and how much - a JSON edit of mesh.weights, no
Blender and no new geometry per dog. Viewers apply the default weights
before skinning, so the clips run on the reshaped mesh.
"""

from __future__ import annotations

from pathlib import Path

from pygltflib import GLTF2

# at most one target per group is used (the targets add, and a rotation
# plus a scale of the same part do not compose by addition)
GROUPS = {
    "ears": ("ear_fold", "ear_drop"),
    "tail": ("tail_thin", "tail_down", "tail_down_thin"),
}


# a target shown by the nearest milder one: a dog read with hanging ears gets
# the folded tips. First on the jindo puppy 2, which carries ear_fold only (its
# dropped ear stuck out sideways like a wing; user, 11-01), and since 11-11 on
# the jindo too, though it carries ear_drop (body_shape.EAR_TARGETS) - so a
# reading saved before that, with ear_drop in its weights, is drawn the same
STAND_INS = {"ear_drop": "ear_fold"}


def fit_to_template(weights: dict[str, float], known: set[str]) -> dict[str, float]:
    """`weights` limited to the targets the template has, one with a
    stand-in (STAND_INS) replaced by it when the template has that."""
    fitted: dict[str, float] = {}
    for name, value in weights.items():
        if STAND_INS.get(name) in known:
            name = STAND_INS[name]
        if name in known:
            fitted[name] = value
    return fitted


def target_names(glb_path: str | Path) -> list[str]:
    gltf = GLTF2().load(str(glb_path))
    return list(((gltf.meshes[0].extras or {}).get("targetNames")) or [])


def set_shape(glb_path: str | Path, weights: dict[str, float], out_path: str | Path | None = None) -> Path:
    """Write `weights` (target name -> weight) as the body mesh's default
    weights; targets not named are 0. Raises on a name the GLB lacks or two
    targets of one group."""
    for group, names in GROUPS.items():
        used = [n for n in names if weights.get(n)]
        if len(used) > 1:
            raise ValueError(f"{group}: use one of {names}, got {used}")
    gltf = GLTF2().load(str(glb_path))
    mesh = gltf.meshes[0]
    names = list(((mesh.extras or {}).get("targetNames")) or [])
    unknown = [n for n in weights if n not in names]
    if unknown:
        raise ValueError(f"{glb_path} has no targets {unknown} (has {names})")
    # a target's weight 1 is its natural limit (shape_keys); past it the
    # outline stretches
    mesh.weights = [max(-1.0, min(1.0, float(weights.get(n, 0.0)))) for n in names]
    out = Path(out_path or glb_path)
    gltf.save_binary(str(out))
    return out


__all__ = ["GROUPS", "STAND_INS", "fit_to_template", "set_shape", "target_names"]
