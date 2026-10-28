"""Canned model answers for the three end-to-end cases.

Each script pins what the vision model would say for one animal so the
whole pipeline - vote, merge, compositor, face composite, critic loop, log -
runs offline exactly as it would live. The answers are written to be
plausible for the photograph in the case, not to flatter the pipeline: the
black shiba's script contradicts the tan_point defaults where the real dog
does (white throat, chest, belly and legs, cream cheeks).

A live run (--live in tests/test_pipeline_cases.py) replaces these with the
real transport and nothing else changes.
"""

from __future__ import annotations

import re
from typing import Any

from api.scripted import ScriptedTransport


def _shown_size(blocks: list[dict[str, Any]]) -> tuple[int, int]:
    for block in blocks:
        if block.get("type") == "text":
            found = re.search(r"(\d+) pixels wide and (\d+) pixels tall", block["text"])
            if found:
                return int(found.group(1)), int(found.group(2))
    return 1024, 1024


def face_box(rel: tuple[float, float, float, float], frontal: bool = True):
    """A head box as fractions of the image shown, resolved at call time."""
    def answer(system: str, blocks: list[dict[str, Any]]) -> dict[str, Any]:
        from api.face_photo import GRID_COLS, GRID_ROWS, _COLS
        x0, y0, x1, y1 = rel
        def label(fx: float, fy: float) -> str:
            return f"{_COLS[min(int(fx * GRID_COLS), GRID_COLS - 1)]}{min(int(fy * GRID_ROWS), GRID_ROWS - 1) + 1}"
        return {"face_box": {"top_left": label(x0, y0), "bottom_right": label(x1, y1)},
                "visible": True, "frontal": frontal}
    return answer


def landmarks(rel: dict[str, tuple[float, float]]):
    """Landmarks as fractions of the crop, answered as grid cells the way the
    live model does (api.face_photo grid referencing)."""
    def answer(system: str, blocks: list[dict[str, Any]]) -> dict[str, Any]:
        from api.face_photo import GRID_COLS, GRID_ROWS, _COLS
        out = {}
        for name, (x, y) in rel.items():
            c = min(int(x * GRID_COLS), GRID_COLS - 1)
            r = min(int(y * GRID_ROWS), GRID_ROWS - 1)
            fx, fy = x * GRID_COLS - c, y * GRID_ROWS - r
            within = ("top" if fy < 0.35 else "bottom" if fy > 0.65 else "") + \
                     ("-" if (fy < 0.35 or fy > 0.65) and (fx < 0.35 or fx > 0.65) else "") + \
                     ("left" if fx < 0.35 else "right" if fx > 0.65 else "")
            out[name] = {"col": _COLS[c], "row": r + 1, "within": within or "center", "visible": True}
        return out
    return answer


def region(region_name: str, color: str, coverage: str = "full",
           gradient_to: str = "none", gradient_anchor: str = "none", note: str = "",
           strength: str = "strong") -> dict[str, Any]:
    return {"region": region_name, "color": color, "coverage": coverage, "strength": strength,
            "gradient_to": gradient_to, "gradient_anchor": gradient_anchor, "note": note}


def marking(kind: str, region_name: str, color: str, size: str = "small",
            anchor: str = "none", offset: str = "on", distance: str = "near") -> dict[str, Any]:
    return {"type": kind, "region": region_name, "color": color, "size": size,
            "anchor": anchor, "offset": offset, "distance": distance}


def scoped(entries: list[dict[str, Any]]):
    """A region-colour answer that only returns the entries inside the call's
    scope, so one list serves both the face and the body call."""
    def answer(system: str, blocks: list[dict[str, Any]]) -> dict[str, Any]:
        text = " ".join(b["text"] for b in blocks if b.get("type") == "text")
        found = re.search(r"Regions in scope: ([^.]+)\.", text)
        scope = set(found.group(1).replace(" ", "").split(",")) if found else set()
        return {"region_colors": [e for e in entries if e["region"] in scope]}
    return answer


def scoped_markings(entries: list[dict[str, Any]], note: str = ""):
    def answer(system: str, blocks: list[dict[str, Any]]) -> dict[str, Any]:
        text = " ".join(b["text"] for b in blocks if b.get("type") == "text")
        found = re.search(r"Regions in scope: ([^.]+)\.", text)
        scope = set(found.group(1).replace(" ", "").split(",")) if found else set()
        return {"markings": [e for e in entries if e["region"] in scope], "note": note}
    return answer


def region_cells(cells: dict[str, tuple[str, int]]):
    """Which grid cell of the photograph each region's fur sits in, for the
    colour measurement. Cells outside the animal simply measure whatever is
    there, exactly as they would live."""
    return {"cells": [{"region": r, "col": c, "row": row} for r, (c, row) in cells.items()]}


# The black shiba fixture: cells that land on the dog. Body regions only -
# the measurement skips what it cannot point at, like the live call does.
BLACK_CELLS = region_cells({"back": ("E", 4), "chest": ("E", 6), "underside": ("E", 7),
                            "legs_lower": ("D", 8), "paws": ("D", 9), "skull": ("F", 3)})

PHOTO_OK = {"subject": "dog", "animals": 1, "person_covers": "none", "object_covers_animal": False,
            "bars_in_front": False, "text_on_animal": False, "clothing": False, "harness": False, "view": "front",
            "animal_cell": {"col": "E", "row": 5}, "note": ""}
PHOTO_HELD = {**PHOTO_OK, "person_covers": "large", "harness": True}


def critic(rounds: list[list[dict[str, Any]]], samples: int = 2):
    """One answer per critic round, repeated for each of the critic's samples
    so the vote is unanimous; after the list runs out, no patches."""
    answers = []
    for i, patches in enumerate(rounds):
        answers += [{"patches": patches, "note": f"round {i + 1}"}] * samples
    answers += [{"patches": [], "note": "no differences"}] * samples
    return answers


# --------------------------------------------------------------------------
# case 1: the black-and-tan shiba (tests/fixtures/image.png)
# --------------------------------------------------------------------------

def black_shiba() -> ScriptedTransport:
    regions = [
        region("cheeks", "cream", "partial", gradient_to="tan", gradient_anchor="eye",
               note="cream cheeks, tan toward the eyes"),
        region("chest", "white", "full"),
        region("underside", "white", "full"),
        region("legs_lower", "white", "partial", gradient_to="tan", gradient_anchor="paw",
               note="white with tan on the front of the lower legs"),
        region("paws", "white", "full"),
        region("ears_inner", "cream", "full"),
    ]
    return ScriptedTransport({
        "photo_gate": [PHOTO_OK],
        "region_cells": [BLACK_CELLS],
        "face_colours": [{"cheeks": "cream", "brows": "tan", "muzzle": "dark", "ears": "tan", "blaze": "none",
                          "crown": "dark",
                          "note": "cream cheeks, tan brow spots, tan ear cups"}],
        "body_colours": [{"chest": "white", "belly": "unknown", "paws": "white", "note": "white chest"}],
        "body_shape": [{"ears": "erect", "tail_build": "bushy", "tail_carriage": "up", "build": "medium",
                        "muzzle": "medium", "coat_length": "medium", "leg_length": "normal",
                        "head_size": "normal", "note": "pricked ears, curled bushy tail"}],
        "classify": [
            {"pattern": "tan_point", "base_color": "black", "second_color": "tan", "eye_color": "dark_brown",
             "white_legs": "paws", "features": ["white_chest"],
             "reason": "black coat, tan points above the eyes and on the legs"},
            {"pattern": "tan_point", "base_color": "black", "second_color": "tan", "eye_color": "dark_brown",
             "white_legs": "paws", "features": ["white_chest"],
             "reason": "tan pips over the eyes"},
        ],
        "face_box": [face_box((0.36, 0.10, 0.86, 0.50), frontal=True)],
        "region_colors": [scoped(regions)],
        "markings": [scoped_markings([])],
        # measured on tests/fixtures/image.png - eyes (222,180) and (278,158),
        # nose (275,208) - moved into the face crop (box 86,74 - 319,311,
        # 233x237) and taken as fractions of it. (image_landmarks.json marks
        # the tan brow pips, not the eyes, and is not used here.)
        "landmarks": [landmarks({"left_eye": (0.584, 0.447), "right_eye": (0.824, 0.354),
                                 "nose": (0.811, 0.565), "left_ear_base": (0.30, 0.10),
                                 "right_ear_base": (0.86, 0.02)})],
        "critic": critic([
            [{"region": "muzzle_bridge", "current": "black", "expected": "white", "kind": "color"}],
            [],
        ]),
        # simple mode
        "features": [{"features": ["white_chest", "white_socks", "tan_brows", "cream_cheeks"], "note": "black and tan"}],
        "simple_critic": [{"add": ["white_belly"], "remove": [], "note": "belly is white"}] * 2
                          + [{"add": [], "remove": [], "note": ""}] * 2,
    })


# --------------------------------------------------------------------------
# case 2: a jindo - urajiro on a fawn coat
# --------------------------------------------------------------------------

def jindo() -> ScriptedTransport:
    regions = [
        region("legs_lower", "cream", "partial", gradient_to="fawn", gradient_anchor="paw"),
        region("eyebrows", "fawn", "full"),
    ]
    return ScriptedTransport({
        "photo_gate": [PHOTO_OK],
        "region_cells": [BLACK_CELLS],
        "classify": [
            {"pattern": "urajiro", "base_color": "fawn", "second_color": "cream",
             "reason": "cream cheeks, throat and belly on a fawn coat"},
            # a split the second colour settles: cream is light, so urajiro
            {"pattern": "tan_point", "base_color": "fawn", "second_color": "cream",
             "reason": "light points"},
        ],
        "face_box": [face_box((0.30, 0.05, 0.80, 0.45), frontal=False)],
        "region_colors": [scoped(regions)],
        "markings": [scoped_markings([])],
        "landmarks": [landmarks({"left_eye": (0.5, 0.3), "right_eye": (0.6, 0.3), "nose": (0.55, 0.4)})],
        "critic": critic([[]]),
    })


# --------------------------------------------------------------------------
# case 3: a mixed-breed dog - no layout fits, every region read from the photo
# --------------------------------------------------------------------------

def mixed() -> ScriptedTransport:
    regions = [
        region("skull", "white"), region("eyebrows", "white"), region("muzzle_bridge", "white"),
        region("cheeks", "white"), region("ears_inner", "cream"), region("back", "chocolate", "partial"), region("chest", "white"),
        region("underside", "white"), region("legs_lower", "white"), region("paws", "white"),
    ]
    marks = [
        marking("patch", "skull", "chocolate", "medium", anchor="left_eye", offset="above", distance="near"),
        marking("spot", "back", "chocolate", "small", anchor="none"),
    ]
    return ScriptedTransport({
        "photo_gate": [PHOTO_OK],
        "region_cells": [BLACK_CELLS],
        # three samples a round, three rounds, never a majority: -> other
        "classify": [
            {"pattern": "other", "base_color": "white", "second_color": "chocolate", "reason": "patches"},
            {"pattern": "irish", "base_color": "white", "second_color": "chocolate", "reason": "white collar"},
            {"pattern": "solid", "base_color": "white", "second_color": "none", "reason": "mostly white"},
            {"pattern": "irish", "base_color": "white", "second_color": "chocolate", "reason": "collar"},
            {"pattern": "other", "base_color": "white", "second_color": "chocolate", "reason": "random patches"},
            {"pattern": "solid", "base_color": "white", "second_color": "none", "reason": "mostly white"},
            {"pattern": "other", "base_color": "white", "second_color": "chocolate", "reason": "patches"},
            {"pattern": "solid", "base_color": "white", "second_color": "none", "reason": "white"},
            {"pattern": "irish", "base_color": "white", "second_color": "chocolate", "reason": "collar"},
        ],
        "face_box": [face_box((0.30, 0.05, 0.80, 0.45), frontal=True)],
        "region_colors": [scoped(regions)],
        "markings": [scoped_markings(marks, "a chocolate patch over the left eye")],
        "landmarks": [landmarks({"left_eye": (0.5, 0.3), "right_eye": (0.6, 0.3), "nose": (0.55, 0.4)})],
        # a patch, then its reverse: the oscillation case
        "critic": critic([
            [{"region": "back", "current": "chocolate", "expected": "white", "kind": "color"}],
            [{"region": "back", "current": "white", "expected": "chocolate", "kind": "color"}],
        ]),
    })


def held_dog() -> ScriptedTransport:
    """The black shiba held by a person: the gate refuses it."""
    transport = black_shiba()
    transport._answers["photo_gate"] = [PHOTO_HELD]
    return transport


CASES = {"black_shiba": black_shiba, "jindo": jindo, "mixed": mixed, "held_dog": held_dog}
