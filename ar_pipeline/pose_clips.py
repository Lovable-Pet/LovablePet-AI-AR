"""Clips for a template - standing, sitting, lying, shake, rollover,
tail_wag, bark, the transitions between the held ones (sit_down, stand_up,
lie_down, get_up) and play_dead - posed on the template's own build.

WHY
---
retarget_clip (11-03) copied the retired shiba's joint rotations onto the
jindo and patched the floor contact afterwards. The shiba is a short-legged
puppy, and the jindo's torso is 1.9 times its length against legs 1.2
times: its angles crouched the sitting jindo on slanting legs, stood it on
its rump mid-roll and floated its play-dead, and each fix only chased the
puppy's shape further (user, 11-03: "애니메이션 전체적인 느낌만 비슷하고
기본적으로 우리 모델이 자연스러워야 돼").

HOW
---
Each clip is a few key poses solved on the template from its own bone
lengths and mesh, and the shiba gives only what does not depend on its
build:

  * key poses (solve_keys): stand (the template's rest pose); sit - the
    body pitched until the front legs stand near straight under the
    shoulders, the rump on the floor, the pelvis tucked under and the hind
    legs folded as dog_mocap's sitting dogs fold them (the knee up by the
    flank, the shin down to the hock on the floor just ahead of the hip,
    the paw forward), the head kept level by the neck; down - the chest and
    the forearms on the floor, the hips rolled onto one side and the hind
    legs relaxed out to the other; side - rolled onto the right side, the
    legs relaxed and lowered onto the floor, the head laid down; the legs
    tucked for a roll; a paw offered for shake. These need the leg joints
    where the mesh bends (rig_fix moves them there first);
  * timing from the shiba: when it moves from one pose to the next and how
    fast (its body's progress between the two orientations, its roll angle,
    its raised paw's height);
  * the small motion on top of a held pose - the head looking round, the
    ears, the tail, a lean of the body - as the shiba's joint rotations
    relative to its own pose at the start of the hold, put on the
    template's joints relative to the template's key pose;
  * every frame rests on the floor: the torso's lowest point on it (and
    while rolling, folded legs too), the paws a held pose stands on pinned
    there by two-bone IK, and a leg, the head or the tail under it turned up
    out of it (the tail laid along it).

The clips are written into the template GLB in place of any of the same
name, the mesh, skin and other clips untouched (no Blender round trip:
that would touch the morph targets, split primitives and fixed normals).

    python -m ar_pipeline.pose_clips --slug jindo                 # work/anim/jindo_clips.glb
    python -m ar_pipeline.pose_clips --slug jindo --install       # into the template GLB
    python -m ar_pipeline.pose_clips --slug jindo --clip sitting  # one clip
    python -m ar_pipeline.pose_clips --slug jindo --clip sit_down --clip stand_up --clip lie_down --clip get_up --install

--install backs the GLB up to work/backup_anim_<slug>/ first (once).
"""
from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pygltflib import GLTF2, Animation, AnimationChannel, AnimationChannelTarget, AnimationSampler
from scipy.spatial.transform import Rotation

from ar_pipeline import dog_mocap, video_motion
from ar_pipeline.gltf_rig import Rig, append_accessor, prune, read, sample_clip

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "archive" / "shiba" / "output" / "skeleton_shiba_small.glb"
FPS = 24.0
UP = np.array([0.0, 1.0, 0.0])
FWD = np.array([0.0, 0.0, 1.0])
LEFT = np.array([1.0, 0.0, 0.0])

LEGS = {"LB": ("L_hip_jnt", "L_knee_jnt", "L_ankle_jnt", "L_foot_jnt", "LB_toes_jnt"),
        "RB": ("R_hip_jnt", "R_knee_jnt", "R_ankle_jnt", "R_foot_jnt", "RB_toes_jnt"),
        "LF": ("L_shoulder_jnt", "L_elbow_jnt", "L_wrist_jnt", "L_hand_jnt", "LF_toes_jnt"),
        "RF": ("R_shoulder_jnt", "R_elbow_jnt", "R_wrist_jnt", "R_hand_jnt", "RF_toes_jnt")}
TORSO = ("root", "COG_jnt", "pelvis_jnt", "central_corrective", "spine_1_jnt", "spine_2_jnt", "breathe_jnt",
         "chest_jnt", "L_clavicle_jnt", "R_clavicle_jnt")
TAIL = tuple(f"tail_{k}_jnt" for k in range(1, 8))
NECK = ("neck_base_jnt", "neck_mid_jnt")
# the shiba's motion on top of a held pose is carried on these (COG_jnt as a
# world turn of the whole body, the rest relative to their parent)
SECONDARY = ("spine_1_jnt", "spine_2_jnt", "chest_jnt", "neck_base_jnt", "neck_mid_jnt", "head_jnt",
             "L_ear_base_jnt", "L_ear_tip_jnt", "R_ear_base_jnt", "R_ear_tip_jnt") + TAIL
# eased in and out over this long where a hold meets a change of pose
SECONDARY_RAMP_S = 0.3
SECONDARY_END_S = 0.6
# a turn reflected left for right (x is the dog's left)
MIRROR = np.diag([-1.0, 1.0, 1.0])

# SIT. The body pitches nose-up about the hips (which keep their place over
# the hind paws) until the shoulders stand SIT_LEG_SHARE of the front leg's
# length (shoulder to wrist) above the wrist's standing height, within
# SIT_PITCH_DEG; the pelvis turns SIT_PELVIS_DEG further (the rump tucks
# under) and each lumbar joint SIT_SPINE_DEG, and the head is brought back
# to SIT_HEAD_UP_DEG above level (shares NECK_SHARES of the turn). A real
# dog sitting (dog_mocap, built as the jindo is: hip to shoulder 1.3 times
# the shoulder's height) holds its back straight, the chest if anything a
# little steeper than the loins, 42-48 deg up, its pelvis tipped some 50
# deg past that. The lumbar joints turned with the pelvis, 22 deg each, the
# back bowed in and the chest pushed out (user, 11-03: "가슴이 왜 저렇게
# 나오고 허리가 활처럼 휘어"); turned 6 deg the other way, the back rounded
# and the chest leaned forward at 31 deg ("뭔가 살짝 어색해"). The pelvis tips
# only SIT_PELVIS_DEG past the back: the jindo's rump carries hair hanging
# 8-12 cm behind its thighs under the tail, and tipped further it tucks
# that under the rump and the dog sits on it - at 45 deg the hip stood 11
# cm up, the haunch on the folded shin ("엉덩이가 바닥에 있어야 할텐데");
# at 5 the hip is 6 cm up and the back 41 deg. (Given to the tail instead,
# the hair let the hip down to 4 cm, but the long back then stood at 45
# deg, its belly a slope to the floor and the thighs sunk into it - "전보다
# 더 이상해졌는데".)
SIT_LEG_SHARE = 0.97
SIT_PITCH_DEG = (10.0, 65.0)
SIT_PELVIS_DEG = 5.0
SIT_SPINE_DEG = 0.0
# the chest joint bent forward this far: it turns about the shoulders'
# height, so the forelegs stay put and the breastbone and the neck tip
# forward - pitched 41 deg with the body, the point of the breastbone (all
# the chest joint's) stood 5 cm out in front of the throat (user, 11-04:
# "가슴에 뾰족하게 튀어나온거")
SIT_CHEST_FLEX_DEG = 20.0
# the neck, turned with the body, leans back past upright; brought forward
# at its base to no more than SIT_NECK_PITCH_DEG above level (left at 106
# deg the chest stood out in front of it - user, 11-04: "가슴이 살짝 나온
# 느낌"; at 78 the head began to come forward over the chest)
SIT_NECK_PITCH_DEG = 82.0
SIT_HEAD_UP_DEG = 6.0
# the head brought level at the head joint, the neck left upright: shared
# down the neck, the neck bent forward and the face sank into it (user,
# 11-03: "얼굴이 목에 살짝 파묻히긴 해")
NECK_SHARES = {"neck_mid_jnt": 0.25, "head_jnt": 0.75}
# a sitting dog's hind leg (fold_hind), as a real dog sits (dog_mocap's
# captures, 18 sitting frames: the femur 45-60 deg up from the low hip to
# the knee by the flank, the tibia down to the hock on the floor 1-4 cm
# ahead of the hip, the paw 11-14 cm ahead; stifle 12-20 deg, hock 40-90):
# the femur SIT_KNEE_UP_DEG up and forward from the hip and out by
# KNEE_SPLAY, the shin down from the knee to the hock on the floor, the
# metatarsus and paw forward along the floor, turned out by META_OUT.
# (With the hock under the hip and the knee bent forward, the femur lay
# along the floor, the shin flat under it, and the haunch splayed out like
# a frog's - user, 11-03: "뒷다리도 매우 어색하고".) Stood steeper, 62 deg,
# the thigh went into the belly and the haunch lost its round.
SIT_KNEE_UP_DEG = 40.0
KNEE_SPLAY = 0.35
META_OUT = 0.2
PAW_DROP = 0.35
# a sitting dog's hocks clear of the floor, its weight on the hind paws
# (folded flat onto the hocks the generated mesh crumpled at the hock)
SIT_HOCK_LIFT_DEG = 0.0
# a sitting dog's tail swept round its left side along the floor, round the
# haunch toward the paw: each tail bone's heading, degrees from straight
# back round toward the left and the front, then the tail let down onto the
# floor (laid straight back, it read as a mat under the rump); () leaves it
# straight back
SIT_TAIL_HEADINGS: tuple = (80.0, 105.0, 130.0, 150.0, 165.0, 175.0)
# DOWN and SIDE are a real dog's (dog_mocap): built by hand they looked
# like no dog lying down (user, 11-03: "진짜 개 같지가 않아").
# BACK: the joints that carry the back's lengths, hips to chest. The
# captured down's went up 33 deg from the hips, down 23 deg and up 22 deg
# again to the chest - bent 56 and 46 deg at two joints, the waist humped
# 14 cm over the line of the chest (the shiba's back bends 15 deg a joint),
# and rolling over, the curl bent it 68 deg (user, 11-04: "허리가 난 실제로
# 가능한지 의문일 수준으로 비틀려"): laid straight (even_back).
BACK = ("COG_jnt", "spine_1_jnt", "spine_2_jnt")
# UNTWIST: the capture gives each leg bone its turn about its own length
# too, and the down's came out turned 45-77 deg at the hip, knee and hock
# (the shiba's: 15) - the knee bending out sideways 33-61 deg, the hock
# turned against it (user, 11-04: "뒷다리도 저래도 되나"); a joint within
# HINGE_STRAIGHT_DEG of straight takes the axis it had at rest
HINGE_STRAIGHT_DEG = 15.0
# a joint bent this nearly square (cosine) to its axis as the bone above
# carries it keeps the side its axis had a frame before (decided on its own
# each frame, it flipped, and the leg spun 90-170 deg in a frame; kept to
# the last frame's always, the left wrist's turned round over the clip and
# spun 180 deg where the clip came back to its start)
HINGE_SIDE_MIN = 0.3
# the least bend of each hinge (knee / elbow, hock / wrist, paw), degrees the
# way it bends at rest: playing dead, the captured dog stood with a hind leg
# straight and its hock bent 11 deg the wrong way, and as it got up an elbow
# went 10 deg past straight (user, 11-04: "죽은척도 관절, 허리 뒤틀리지
# 않게"). A wrist gives back a little (-16 in the goniometry, see TUCK_LEGS).
HINGE_MIN_DEG = {"B": (5.0, 5.0, None), "F": (0.0, -15.0, None)}
# and the most (an elbow 144 in the goniometry, a knee 138 - the captured
# dog's lie folded 160 at the joint centres, so a little more there; getting
# up, its elbow folded the forearm back onto the upper arm)
HINGE_MAX_DEG = {"B": (165.0, 145.0, None), "F": (145.0, 150.0, None)}
# limit_hinges puts right only a joint bent within this (cosine) of its own
# plane, by at most this much
HINGE_IN_PLANE = 0.7
HINGE_FIX_MAX_DEG = 25.0
# the most the bones below a knee / elbow lean out of the plane the leg bends
# in there (hinge_leg)
HINGE_SIDE_DEG = 12.0
# a paw within PAW_FLAT_M[0] of its toe's height at rest lies flat on the
# floor (flat_paw), one over PAW_FLAT_M[1] as it is, between eased
PAW_FLAT_M = (0.02, 0.05)
# (a captured clip's paw: as its capture's is within PAW_FLAT_CAPTURE_M of the floor)
PAW_FLAT_CAPTURE_M = (0.03, 0.06)
LEGS_ORDER = {"LF": 0, "RF": 1, "LB": 2, "RB": 3}                 # dog_mocap.FEET's order
# align_leg takes each bone's turn from the joint above bent this far, and
# in a clip turns it about its length toward that by at most this a frame
ALIGN_MID_DEG = 70.0
ALIGN_STEP_DEG = 6.0
ALIGN_ROOT_TWIST_DEG = 40.0
# the straightened down, propped on its folded hind legs, held its chest
# 8 cm off the floor (the capture's, on its humped back, 5): tipped forward
# about the hips, at most this far, until the chest rests on the floor too
SETTLE_FRONT_MAX_DEG = 25.0
# and its hips sat up on its folded hind legs, the rump 12 cm off the floor
# (the left knee under the belly, folded as far as a knee goes, the paws
# under the hips): rolled this much further onto the hip it lies on, as a
# dog lies at ease, the upper hind leg turned in after it so its paw stays
# down. It was 25 deg and 15 (rump 7 cm): the hips lay 39 deg round from the
# squared chest, the upper thigh stood up off them and its paw turned pads
# out, and before and after the roll the hips read as wrung off the back
# (user, 11-06: "엎드려구르기 ... 골반이 왼쪽으로 틀어져있는것같은데"). Now
# the capture's own 15 deg, the hip joint 2 cm higher than the sphinx's
# (0.7 at 25)
DOWN_HIP_ROLL_DEG = 0.0
DOWN_UPPER_HIND_IN_DEG = 0.0
# the sphinx (crouch) is squared up from the down as it lay rolled 25 deg,
# its upper hind leg in 15: the lying clip and the changes of pose to and
# from it were built on that sphinx and looked over; squared from the down
# as it now lies, its tail came out 44 deg and its right hip 12 off them,
# and the roll's first frame off the lying clip's
SPHINX_DOWN_DEG = (25.0, 15.0)
# ROLLING OVER from the down, as a dog (and the shiba's clip) does it: over
# onto the side the down already lies on, curled into a ball - the back
# rounded, the rump tucked under, the chin in toward the chest and the head
# turned to the side it goes over on, the legs folded close (TUCK). Rolled
# the other way, as a straight log, it went wrong every way at once (user,
# 11-04: "몸이 뒤틀리고 걍 이상하잖아"): the captured down lies on its left
# hip with the hind legs out to the right, and rolling onto the right side
# swung them over the top and wrung the hips round against the chest; the
# legs splayed like a dead bug's; the head went under the body upside down
# and was turned out of the floor 110 deg at the neck. (roll_side() reads the
# side off the down.)
# the curl, degrees about each joint's own left axis (+ the part ahead of it
# down): the pelvis tucked under, the back rounded, the chin in
ROLL_CURL_DEG = {"pelvis_jnt": -12.0, "spine_1_jnt": 12.0, "spine_2_jnt": 12.0,
                 "neck_base_jnt": 20.0, "head_jnt": 15.0}
# (30 deg in the 0.3 s before the roll, with the shiba's looking about laid
# on after it, the head jerked to and fro - user, 11-04: "고개가 굳이
# 까딱거리니까 바보 같은데 기분 좋아보이게")
ROLL_HEAD_TURN_DEG = 15.0
# the shiba's roll (its clip's seconds), played over the Roll segment: the
# jindo is twice its length, and at the puppy's 1.1 s the big dog spun
ROLL_SRC_S = (0.33, 1.42)
# TUCK: legs folded close while rolling over, as (degrees from straight
# down toward the front, outward share) per bone from the hip / shoulder in
# the body's frame (sagittal()) - the knees forward to the belly and the
# hocks back under them; the upper arms back along the chest, the elbows
# bent 95 deg so the forearms reach up past the chest, the wrists folded
# 105 so the paws hang back toward it, as a dog lies on its back (the
# forearms reaching forward and the hind legs out to the sides, it lay
# like a dead bug; knees in at 7 deg, the hind legs stood up off the belly
# side by side like two posts - now out 20; the upper arms forward and the
# forearms back, the elbows bent 80 deg the wrong way - user, 11-04: "앞다리,
# 뒷다리가 저래도 괜찮은지 … 실제 개가 가능한 자세인가". An elbow bends
# 15-144 deg forward, a wrist -16..148 back, a knee 18-138 back, a hock
# 16-141 forward: Jaegger et al. 2002, Labradors). Each leg its own: the
# two sides alike to the degree, the legs moved like a puppet's. The hind
# legs folded about as far as lying down folds them - the knee some 130 deg,
# the hock 100 - the paw under the thigh: folded less (the knees 90-120 deg,
# the hocks 55-65), the hind legs unfolded as the dog went over and stuck up
# off it stiff (user, 11-06: "구르기 뒷발좀 자연스럽게 … 뒤틀리지않았나").
# The hind paws hang curled from the hock, the toes 30-35 deg back from the
# hock's line (58-63 from standing, where they are 28 forward of it): carried
# forward of it as standing has them, their soles faced the dog's front and
# then the floor's, and from the front, as it came over onto its other side
# and back onto its belly, the near hind paw rose with its sole turned out
# (user, 11-06: "엎드려 구르기할때 왼쪽 발바닥 보이게 왜 드는거야자꾸?" -
# from the front-left, soles shown lifted 0.62 s on its side and 0.12 coming
# back, now 0.42 and 0.04)
TUCK_LEGS = {"LB": ((80.0, 0.35), (-50.0, 0.15), (50.0, 0.05), (15.0, 0.0)),
             "RB": ((72.0, 0.3), (-55.0, 0.15), (45.0, 0.05), (15.0, 0.0)),
             "LF": ((-20.0, 0.15), (75.0, 0.1), (-30.0, 0.0), (-60.0, 0.0)),
             "RF": ((-10.0, 0.12), (85.0, 0.1), (-15.0, 0.0), (-50.0, 0.0))}
# rolling over, the folded legs hang loose: each swung about the body's
# length toward whichever side is down, this much lying on its side (held
# in the body's frame, they stood off it like a toy's whatever way up it
# was), as the body was this long before (they trail its turn)
ROLL_LEG_FLOP_DEG = 25.0
ROLL_LEG_FLOP_LAG_S = 0.08
# the body curls over the TUCK_LEAD_S before a roll and uncurls over the
# UNTUCK_S after it; the legs fold as the body goes over, in its first
# ROLL_FOLD_DEG, and unfold in its last, so that it comes down onto them as
# the down has them (folded on the floor before and after, they propped the
# dog 23 cm up on its tucked legs, crouched as if to spring; folded within
# the roll's first and last few degrees, a tenth of a second on the shiba's
# quick roll, the knees swung 50-75 deg in a frame; in its first 100, the
# shoulder of the leg it rolls onto 28)
TUCK_LEAD_S = 0.3
UNTUCK_S = 0.7
ROLL_FOLD_DEG = 140.0
# lying before and after, the dog looks ahead at whoever asked for the roll
# (the capture's head turned 18 deg off to the right), held still: of the
# shiba's motion only its ears (its looking about, nodding and a 24 deg tilt
# of the head made the dog look simple); and it wags, the tail laid along
# the floor and swept to and fro about the vertical (amps per joint from the
# root, Hz, radians each joint behind the last - see WAG), stilled through
# the roll and FADE_S either side of it
FACE_FRONT_SHARES = {"neck_base_jnt": 0.5, "neck_mid_jnt": 0.3, "head_jnt": 0.2}
HOLD_SECONDARY = {"rollover": ("L_ear_base_jnt", "L_ear_tip_jnt", "R_ear_base_jnt", "R_ear_tip_jnt")}
HOLD_SECONDARY["lying"] = HOLD_SECONDARY["rollover"]
LYING_WAG = {"rollover": {"amps": (16.0, 9.0, 6.0, 4.0, 2.0, 1.0, 1.0), "hz": 2.0, "lag": 0.12, "fade_s": 0.3}}
LYING_WAG["lying"] = LYING_WAG["rollover"]
# within this many degrees of lying on its chest the tail is laid along the
# floor as it is lying (it went 5 cm into it as the dog tipped over); further
# round it passes under the body, where laid frame by frame it flipped 90 deg
ROLL_TAIL_DRAPE_DEG = 90.0
# SHAKE: the left front paw offered forward, in the world's frame (the sit
# is pitched), same convention as TUCK_LEGS
SHAKE_LEG = ((35.0, -0.1), (110.0, -0.1), (45.0, 0.0), (35.0, 0.0))
SHAKE_LEG_NAME = "LF"
# a paw offered in a clip read off video (build_video), same convention
# (None: the bone as the key has it): the whole leg raised from the
# shoulder, the forearm out level, the paw hanging easy, the shoulder blade
# turned OFFER_BLADE_DEG forward with it. (Lifted at the elbow alone, the
# upper arm 10 deg forward, the forearm hinged up stiffly - user, 11-04:
# "팔 더 올리지 않나? 너무 강제로 드는 느낌"; swung from the shoulder it
# dragged the breast with it until rig_fix eased the foreleg's seam.)
OFFER_LEG = {"shake": ((35.0, 0.05), (105.0, 0.05), (50.0, 0.0), (30.0, 0.0))}
VIDEO_TRUNK = ("COG_jnt", "spine_1_jnt", "spine_2_jnt", "chest_jnt")
VIDEO_TRUNK_SHARE = 0.3
OFFER_SMOOTH_FRAMES = 8.0
OFFER_RAMP_S = (0.35, 0.5)                     # up, down
OFFER_SWAY = 0.05
OFFER_SWAY_HZ = 0.6
# (turned 8 deg with the leg, the shoulder blade dragged the breast 2 cm
# further; and the upper arm at 50 deg 3 cm more than at 35 with the
# forearm raised instead - "가슴이 너무 끌려가는 느낌")
OFFER_BLADE_DEG = 0.0
# a happy dog's tail in a clip read off video: laid straight back along the
# floor and swept side to side about the vertical - joint by joint from the
# root WAG[0] deg, each WAG[3] radians behind the one before - at about WAG[2]
# Hz (a whole number of sweeps in the clip) (user, 11-04: "꼬리 좀 더
# 흔든다던지 기분 좋아 보이게"). The sweep is spread down the first joints so
# the tail bows as it goes: swung 22 deg at the root alone it hooked there,
# and with 7 deg more at each joint half a turn behind the last it wriggled
# like a snake ("꼬리가 꾸불한데"). WAG[1] is unused.
WAG = {"shake": ((14.0, 8.0, 5.0, 3.0, 2.0, 1.0, 1.0), 0.0, 2.0, 0.12)}
WAG_CLEAR_M = 0.005
# standing and wagging (build_stand_wag), in place of auto_rig's tail_wag -
# its tail twisted about its length and curled up into a hook as it swung,
# and the same 1.3 s came round and round (user, 11-04: "일어선 꼬리흔들기도
# 자연스럽게"): the tail carried `lift` deg up, swept `amps` deg joint by
# joint from the root, each `lag` radians behind, at about `hz` (a whole
# number of sweeps in `length` s), the rump swaying `hips` deg the other
# way, the sweep `vary` weaker half way through the loop
STAND_WAG = {"tail_wag": {"length": 4.0, "hz": 2.25, "amps": (22.0, 12.0, 8.0, 5.0, 3.0, 2.0, 1.0),
                          "lag": 0.12, "lift": 18.0, "hips": 3.0, "vary": 0.15}}
# barking (build_bark; user, 11-05: "짖을 수 있게 입도 움직이도록"), on a
# template whose mouth opens (ar_pipeline/mouth.py): the stand with the head
# carried `head_up` deg higher and the ears `ears` deg forward, a bout of
# barks at `barks` (s) and a pause, `length` s in all. Each bark the jaw
# opens `gape` deg in `open_s`, holds `hold_s` and shuts in `close_s` - 0.29 s
# in all, a bark lasting 0.25-0.35 s (Yin & McCowan 2004, Anim. Behav. 68:343,
# means 248-346 ms by context) and a dog's jaw opening to some 44 deg at
# most (J. Vet. Dent. 2021); 30 deg made the thin lower jaw a duck's bill,
# 28 once mouth.py had thickened it - and stays `between` deg open between
# the barks of a bout (4 bared the teeth, a growl). The body barks too: the
# head and chest moved 0.3-0.6 cm and the muzzle tipped 4 deg, and from where
# the whole dog shows only the mouth seemed to move (TODO, 11-05: "짖을 때 몸
# 동작 키우기"). So each bark is three movements, each a pulse (start from
# the bark, up, hold, down - s):
#   * `gather_s`, before it - the wind-up, against the throw: a breath in
#     (breathe_jnt `inhale` wider), the body drawn `draw_back` m back, the
#     head lowered (the neck `lower` deg forward and down, the muzzle `tuck`
#     deg down), the ears pricked `prick` deg forward;
#   * `throw_s`, the bark: the chest thrown `thrust` m forward and `dip` m
#     down onto the forelegs, which brace (the paws stay, so the legs lean
#     back from the shoulder), the neck lifted `lift` deg and the muzzle
#     tossed `toss` deg up (NECK_SHARES of it on the neck): the head goes
#     forward and up, the nose 4 cm (with the neck reaching forward instead
#     the head went down as the chest dipped and the muzzle tipped 7 deg),
#     the belly drawn in (breathe_jnt `exhale` narrower: a bark is a forced
#     breath out), the ears flicked back `flick` deg;
#   * `settle_s`, after it: the chest sinking `sink` m as the forelegs give
#     and the head nodding `nod` deg below where it was, then back.
# A bout's barks are not alike: each its own `strength` of the body's
# movements and its own `gape`, the head turned `look` deg (+ to the dog's
# left) for it - eased in over `look_s` (up, hold, down) up to the bark,
# held through it and eased back. The tail,
# carried `tail_lift` deg up, wags as tail_wag does but smaller. Keyed at
# `fps`: at FPS the jaw opened 16 deg in one frame
BARK = {"bark": {"length": 3.0, "fps": 60.0, "barks": (0.40, 1.20), "gape": (28.0, 26.0), "between": 2.0,
                 "open_s": 0.083, "hold_s": 0.042, "close_s": 0.167,
                 "strength": (1.0, 0.8), "look": (0.0, 6.0), "look_s": (0.3, 0.3, 0.6),
                 "gather_s": (-0.22, 0.17, 0.02, 0.08), "inhale": 0.06, "draw_back": 0.006,
                 "lower": 2.0, "tuck": 3.0, "prick": 5.0,
                 "throw_s": (-0.03, 0.10, 0.05, 0.30), "thrust": 0.02, "dip": 0.003, "lift": 3.0,
                 "toss": 12.0, "exhale": 0.07, "flick": 10.0,
                 "settle_s": (0.15, 0.12, 0.04, 0.32), "sink": 0.005, "nod": 3.0,
                 "head_up": 6.0, "ears": 6.0, "tail_lift": 12.0, "tail_hz": 2.0,
                 "tail_amps": (16.0, 9.0, 6.0, 4.0, 2.0, 1.0, 1.0), "tail_lag": 0.12}}
# a leg, the head or the tail this far under the floor is turned up out of it
FLOOR_SLACK_M = 0.004
LIFT_STEP_DEG = 5.0
# a leg lying along the floor (a down's forearms) sits a few millimetres in
# it; turned out of that by the shoulder, its elbow went down as its paw went
# up and the leg swung 60-110 deg in a frame - only deeper than this is lifted
LEG_SLACK_M = 0.015
HEAD_SLACK_M = 0.01
# lying, the throat may sink this far into the floor before the body is
# raised on it (held right out of it, the dog on its side floated 1-2 cm)
NECK_SLACK_M = 0.015
HEAD_LIFT_LIMIT_DEG = 45.0
# PLAYING DEAD the dog lies with its cheek on the floor, 2.5 cm into it on
# the template: turned up out of it at the neck's base, the broad flat cheek
# needed 30-40 deg, and the head lay twisted 70 deg on the neck, half up and
# looking ahead (user, 11-05: "목도 뭔가 이상하게 트는 느낌") - it is
# raised as it lies instead (lift_head_level), and the capture's neck, which
# turns only at its base and at the head (no neck_mid), bends along its
# length (even_neck: the share of the chest-to-head turn each joint is
# turned to)
HEAD_LEVEL_LIFT_DEG = 25.0
EVEN_NECK_SHARES = {"neck_base_jnt": 0.35, "neck_mid_jnt": 0.7, "head_jnt": 1.0}
# rolling over, the head comes round under the body and is turned aside
# further
ROLL_HEAD_LIFT_DEG = 110.0
RADIUS_PERCENTILE = 85.0


# --- poses and the skeleton ---------------------------------------------------------------

@dataclass
class Pose:
    """Local rotations of the joints (node -> 3x3) and the COG's local
    translation; joints left out are at rest."""
    r: dict
    t: np.ndarray

    def copy(self) -> "Pose":
        return Pose({k: v.copy() for k, v in self.r.items()}, self.t.copy())


def rotm(axis: np.ndarray, deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    return Rotation.from_rotvec(axis / max(np.linalg.norm(axis), 1e-12) * np.radians(deg)).as_matrix()


def swing(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotation taking direction `a` onto `b`."""
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    axis = np.cross(a, b)
    sin, cos = np.linalg.norm(axis), float(np.dot(a, b))
    if sin < 1e-9:
        if cos > 0:
            return np.eye(3)
        other = LEFT if abs(a[0]) < 0.9 else UP
        return Rotation.from_rotvec(np.pi * np.cross(a, other) / np.linalg.norm(np.cross(a, other))).as_matrix()
    return Rotation.from_rotvec(axis / sin * np.arctan2(sin, cos)).as_matrix()


def rot_of(m: np.ndarray) -> np.ndarray:
    r = m[:3, :3]
    return r / np.linalg.norm(r, axis=0)


def sagittal(deg: float, out: float = 0.0, side: float = 1.0) -> np.ndarray:
    """A direction `deg` from straight down toward the front, leaning
    outward (away from the midline on `side`, +1 left) by `out`."""
    a = np.radians(deg)
    v = -np.cos(a) * UP + np.sin(a) * FWD + out * side * LEFT
    return v / np.linalg.norm(v)


class Skeleton:
    """The template's joints, forward kinematics, and its skinned mesh
    (every STEP-th vertex of its largest skinned primitive) for floor
    contact."""

    STEP = 3

    def __init__(self, path: Path):
        rig = Rig(path)
        self.rig, gltf = rig, rig.gltf
        self.parent, self.order, self.joints = rig.parent, rig.order, list(rig.joints)
        self.j = rig.by_name
        self.rest_t = {i: rig.rest_trs[i][0] for i in self.order}
        self.rest_r = {i: Rotation.from_quat(rig.rest_trs[i][1]).as_matrix() for i in self.order}
        self.rest_s = {i: rig.rest_trs[i][2] for i in self.order}
        self.cog = self.j["COG_jnt"]
        self.children = {i: [c for c in gltf.nodes[i].children or []] for i in self.order}
        self.rest_pose = Pose({}, self.rest_t[self.cog].copy())
        self.rest = self.fk(self.rest_pose)
        # the skin
        blob = rig.blob
        prim = max((p for n in gltf.nodes if n.mesh is not None and n.skin is not None
                    for p in gltf.meshes[n.mesh].primitives),
                   key=lambda p: gltf.accessors[p.attributes.POSITION].count)
        pos = read(gltf, blob, prim.attributes.POSITION)[::self.STEP]
        joints = read(gltf, blob, prim.attributes.JOINTS_0)[::self.STEP].astype(np.int64)
        weights = read(gltf, blob, prim.attributes.WEIGHTS_0)[::self.STEP]
        if weights.max() > 1.5:
            weights /= 255.0 if weights.max() <= 255 else 65535.0
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-9)
        self.ibm = read(gltf, blob, gltf.skins[0].inverseBindMatrices).reshape(-1, 4, 4).transpose(0, 2, 1)
        self.verts = np.c_[pos, np.ones(len(pos))]
        self.vj, self.vw = joints, weights
        joint_nodes = np.array(self.joints)
        self.dominant = joint_nodes[joints[np.arange(len(joints)), weights.argmax(axis=1)]]
        self.torso = self.mask([self.j[n] for n in TORSO if n in self.j])
        self.all = np.ones(len(self.verts), dtype=bool)

    # -- structure
    def subtree(self, node: int) -> list[int]:
        out, stack = [], [node]
        while stack:
            i = stack.pop()
            out.append(i)
            stack.extend(self.children.get(i, []))
        return out

    def mask(self, nodes) -> np.ndarray:
        return np.isin(self.dominant, list(nodes))

    def length(self, a: int, b: int) -> float:
        return float(np.linalg.norm(self.rest[b][:3, 3] - self.rest[a][:3, 3]))

    def leg(self, name: str) -> list[int]:
        return [self.j[n] for n in LEGS[name]]

    def radius(self, joint: int, child: int) -> float:
        """How far the mesh's surface lies from the bone `joint`->`child`
        (RADIUS_PERCENTILE of the distances of the vertices it carries most):
        a bone laid this high rests its leg on the floor (at the median the
        down's forearms sank 2 cm into it)."""
        a, b = self.rest[joint][:3, 3], self.rest[child][:3, 3]
        v = self.verts[self.dominant == joint, :3]
        if not len(v):
            return 0.03
        ab = b - a
        s = np.clip((v - a) @ ab / max(ab @ ab, 1e-12), 0.0, 1.0)
        return float(np.percentile(np.linalg.norm(v - (a + s[:, None] * ab), axis=1), RADIUS_PERCENTILE))

    # -- kinematics
    def fk(self, pose: Pose) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        for i in self.order:
            m = np.eye(4)
            m[:3, :3] = pose.r.get(i, self.rest_r[i]) * self.rest_s[i]
            m[:3, 3] = pose.t if i == self.cog else self.rest_t[i]
            out[i] = out[self.parent[i]] @ m if i in self.parent else m
        return out

    def points(self, world: dict[int, np.ndarray], mask: np.ndarray) -> np.ndarray:
        mats = np.stack([world[j] @ self.ibm[k] for k, j in enumerate(self.joints)])
        v, vj, vw = self.verts[mask], self.vj[mask], self.vw[mask]
        p = np.zeros((len(v), 4))
        for k in range(vj.shape[1]):
            p += vw[:, k, None] * np.einsum("nij,nj->ni", mats[vj[:, k]], v)
        return p[:, :3]

    def low(self, world: dict[int, np.ndarray], mask: np.ndarray) -> tuple[float, np.ndarray]:
        if not mask.any():                         # a joint that carries no (sampled) vertex
            return float("inf"), np.zeros(3)
        p = self.points(world, mask)
        k = int(np.argmin(p[:, 1]))
        return float(p[k, 1]), p[k]


# --- operations on a pose ------------------------------------------------------------------

def set_world(sk: Skeleton, pose: Pose, w: dict, i: int, r_world: np.ndarray) -> None:
    p = sk.parent.get(i)
    pose.r[i] = (rot_of(w[p]).T if p is not None else np.eye(3)) @ r_world


def turn(sk: Skeleton, pose: Pose, i: int, axis: np.ndarray, deg: float) -> None:
    """Turn joint `i` (and everything it carries) by `deg` about a world axis."""
    w = sk.fk(pose)
    set_world(sk, pose, w, i, rotm(axis, deg) @ rot_of(w[i]))


def aim(sk: Skeleton, pose: Pose, i: int, child: int, direction: np.ndarray) -> None:
    """Swing joint `i` so the bone to `child` points along a world direction."""
    w = sk.fk(pose)
    have = w[child][:3, 3] - w[i][:3, 3]
    set_world(sk, pose, w, i, swing(have, direction) @ rot_of(w[i]))


def two_bone(sk: Skeleton, pose: Pose, a: int, b: int, c: int, target: np.ndarray, pole: np.ndarray) -> None:
    """`c` onto `target` (or as near as the two bones reach), `b` bent
    toward `pole`."""
    w = sk.fk(pose)
    root = w[a][:3, 3]
    l1, l2 = sk.length(a, b), sk.length(b, c)
    to = target - root
    u = to / max(np.linalg.norm(to), 1e-9)
    d = float(np.clip(np.linalg.norm(to), abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4))
    x = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    h = np.sqrt(max(l1 * l1 - x * x, 0.0))
    n = pole - u * (pole @ u)
    n = n / max(np.linalg.norm(n), 1e-9)
    knee = root + u * x + n * h
    aim(sk, pose, a, b, knee - root)
    aim(sk, pose, b, c, root + u * d - knee)


def shift(sk: Skeleton, pose: Pose, delta_world: np.ndarray) -> None:
    w = sk.fk(pose)
    parent = w[sk.parent[sk.cog]]
    pose.t = pose.t + np.linalg.inv(parent[:3, :3]) @ delta_world


def rest_on(sk: Skeleton, pose: Pose, mask: np.ndarray, soft: np.ndarray, slack: float = NECK_SLACK_M) -> None:
    """Raise or lower the body so the lowest vertex of `mask` touches the
    floor and none of `soft` is more than `slack` under it."""
    w = sk.fk(pose)
    y = min(sk.low(w, mask)[0], sk.low(w, soft)[0] + slack)
    shift(sk, pose, np.array([0.0, -y, 0.0]))


def ground(sk: Skeleton, pose: Pose, mask: np.ndarray | None = None) -> float:
    """Raise or lower the body so the lowest vertex of `mask` (the torso)
    touches the floor; returns the move."""
    y, _ = sk.low(sk.fk(pose), sk.torso if mask is None else mask)
    shift(sk, pose, np.array([0.0, -y, 0.0]))
    return -y


def lift(sk: Skeleton, pose: Pose, joint: int, mask: np.ndarray, settle: bool = False,
         limit: float = 75.0, slack: float = FLOOR_SLACK_M) -> tuple[float, np.ndarray]:
    """Turn `joint` the least that brings the lowest vertex of `mask` onto
    the floor: up out of it, or (`settle`) down onto it too. The turn is
    about the horizontal axis across the joint's reach to the middle of
    `mask` (to its lowest vertex the axis jumped from frame to frame as the
    lowest point moved between parts of a rolling leg, and the leg with
    it); returns (degrees, axis)."""
    if not mask.any():
        return 0.0, LEFT
    w = sk.fk(pose)
    pts = sk.points(w, mask)
    y = float(pts[:, 1].min())
    if (y >= -slack and not settle) or abs(y) < 0.002:
        return 0.0, LEFT
    reach = pts.mean(axis=0) - w[joint][:3, 3]
    axis = np.cross(reach, UP)
    if np.linalg.norm(axis) < 1e-6:
        axis = rot_of(w[joint]) @ LEFT
    axis = axis / np.linalg.norm(axis)
    sign = 1.0 if y < 0 else -1.0
    keep = pose.r.get(joint, sk.rest_r[joint]).copy()

    def height(deg: float) -> float:
        pose.r[joint] = keep
        turn(sk, pose, joint, axis, sign * deg)
        return sk.low(sk.fk(pose), mask)[0]

    # the first angle past which it crosses the floor, then bisected; if
    # none does (the vertex is too near the joint for a turn to move it),
    # the angle that comes nearest
    prev, best, best_y = 0.0, 0.0, y
    for deg in np.arange(LIFT_STEP_DEG, limit + 1e-6, LIFT_STEP_DEG):
        h = height(deg)
        if abs(h) < abs(best_y):
            best, best_y = deg, h
        if (h < 0) != (y < 0):
            lo, hi = prev, deg
            for _ in range(10):
                mid = 0.5 * (lo + hi)
                if (height(mid) < 0) == (y < 0):
                    lo = mid
                else:
                    hi = mid
            height(hi)
            return sign * hi, axis
        prev = deg
    height(best)
    return sign * best, axis


def body_frame(sk: Skeleton, w: dict, node: int) -> np.ndarray:
    """How far `node` has turned from rest (world), to carry rest-frame
    directions with it."""
    return rot_of(w[node]) @ rot_of(sk.rest[node]).T


def shape_leg(sk: Skeleton, pose: Pose, leg: str, bones, frame_node: int | None = None) -> None:
    """Point each bone of a leg along (degrees, outward) - see sagittal() - in
    the frame of `frame_node` (the pelvis / chest: the body's frame) or in
    the world's."""
    chain = sk.leg(leg)
    side = float(np.sign(sk.rest[chain[0]][0, 3]) or 1.0)
    for bone, a, b in zip(bones, chain, chain[1:]):
        if bone is None:                                 # left as it is
            continue
        deg, out = bone
        d = sagittal(deg, out, side)
        if frame_node is not None:
            d = body_frame(sk, sk.fk(pose), frame_node) @ d
        aim(sk, pose, a, b, d)


def leg_frame(sk: Skeleton, leg: str) -> int:
    """The torso joint a leg hangs from (its frame for TUCK_LEGS)."""
    return sk.j["pelvis_jnt"] if leg.endswith("B") else sk.j["chest_jnt"]


def plant(sk: Skeleton, pose: Pose, leg: str, target: np.ndarray, lower: tuple[np.ndarray, np.ndarray],
          pole: np.ndarray) -> None:
    """The leg's third joint (hock / wrist) onto `target` by two-bone IK,
    bent toward `pole`, and the two bones below it along `lower`."""
    a, b, c, d, e = sk.leg(leg)
    two_bone(sk, pose, a, b, c, target, pole)
    aim(sk, pose, c, d, lower[0])
    aim(sk, pose, d, e, lower[1])


def lower_dirs(sk: Skeleton, w: dict, leg: str) -> tuple[np.ndarray, np.ndarray]:
    _, _, c, d, e = sk.leg(leg)
    return w[d][:3, 3] - w[c][:3, 3], w[e][:3, 3] - w[d][:3, 3]


def pin(sk: Skeleton, pose: Pose, key: dict, legs=tuple(LEGS)) -> None:
    """Hold each leg's hock / wrist and the bones below it where `key`
    (a solved pose's fk) has them, the knee / elbow bent the way the key
    bends it (bent as it is now, a near-straight foreleg flipped its elbow
    forward and back from frame to frame)."""
    for leg in legs:
        c = sk.leg(leg)[2]
        plant(sk, pose, leg, key[c][:3, 3], lower_dirs(sk, key, leg), bend_of(sk, key, leg))


def lower_leg(sk: Skeleton, leg: str) -> np.ndarray:
    """The leg's vertices from the knee / elbow down: those by the hip or
    shoulder lie on the torso's own contact, where no turn of the leg moves
    them."""
    return sk.mask(sk.subtree(sk.leg(leg)[1]))


def lift_legs(sk: Skeleton, pose: Pose, slack: float = LEG_SLACK_M) -> None:
    for leg in LEGS:
        lift(sk, pose, sk.leg(leg)[0], lower_leg(sk, leg), slack=slack)


def lift_head(sk: Skeleton, pose: Pose, settle: bool = False, limit: float = 0.0) -> None:
    """The head out of (or, `settle`, down onto) the floor by the neck - judged
    by the head's own skin: with the neck's (rig_fix gives the throat to the
    neck joints) the lowest point was by the chest, no turn of the neck
    lifted it, and the neck swung 55-70 deg in a frame."""
    head = sk.mask(sk.subtree(sk.j["head_jnt"]))
    lift(sk, pose, sk.j["neck_base_jnt"], head, settle=settle, limit=limit or HEAD_LIFT_LIMIT_DEG, slack=HEAD_SLACK_M)


def even_neck(sk: Skeleton, pose: Pose) -> None:
    """The neck's turn from the chest to the head shared out along it
    (EVEN_NECK_SHARES), the head turned as it was."""
    w = sk.fk(pose)
    chest, head = body_frame(sk, w, sk.j["chest_jnt"]), body_frame(sk, w, sk.j["head_jnt"])
    for name, share in EVEN_NECK_SHARES.items():
        b = slerp_m(chest, head, share)
        i = sk.j[name]
        set_world(sk, pose, sk.fk(pose), i, b @ rot_of(sk.rest[i]))


def lift_head_level(sk: Skeleton, pose: Pose) -> None:
    """The head out of the floor without tipping it: the neck bent up at its
    base and the head back down by as much, so that it rises as it lies
    (HEAD_LEVEL_LIFT_DEG at most)."""
    head = sk.mask(sk.subtree(sk.j["head_jnt"]))
    neck, skull = sk.j["neck_base_jnt"], sk.j["head_jnt"]
    w = sk.fk(pose)
    if sk.low(w, head)[0] >= -HEAD_SLACK_M:
        return
    reach = sk.points(w, head).mean(axis=0) - w[neck][:3, 3]
    axis = np.cross(reach, UP)
    if np.linalg.norm(axis) < 1e-6:
        return
    axis = axis / np.linalg.norm(axis)
    keep = {i: pose.r.get(i, sk.rest_r[i]).copy() for i in (neck, skull)}

    def height(deg: float) -> float:
        for i, r in keep.items():
            pose.r[i] = r.copy()
        turn(sk, pose, neck, axis, deg)                   # (+ about cross(reach, UP) lifts what it reaches - see lift)
        turn(sk, pose, skull, axis, -deg)
        return sk.low(sk.fk(pose), head)[0]

    lo, hi = 0.0, HEAD_LEVEL_LIFT_DEG
    if height(hi) < -HEAD_SLACK_M:
        return
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        if height(mid) < -HEAD_SLACK_M:
            lo = mid
        else:
            hi = mid
    height(hi)


def drape_tail(sk: Skeleton, pose: Pose, settle: bool = False) -> None:
    """Lay the tail along the floor where it would go into it: each bone
    from the root turned up just out of the floor (`settle`: the first
    brought down onto it first)."""
    tail = [sk.j[n] for n in TAIL if n in sk.j]
    if not tail:
        return
    if settle:
        # by its outer half: the root lies on the floor with the rump, and
        # judged by the whole tail the settle found it touching already and
        # left a lying dog's tail standing up
        lift(sk, pose, tail[0], sk.mask(tail[len(tail) // 2:]), settle=True)
    for k, joint in enumerate(tail):
        lift(sk, pose, joint, sk.mask(tail[k:k + 2]))


# --- key poses ----------------------------------------------------------------------------

def fold_hind(sk: Skeleton, pose: Pose, leg: str, lift_deg: float, splay: float = 0.0, dy: float = 0.0,
              knee_up_deg: float = SIT_KNEE_UP_DEG) -> None:
    """A hind leg folded under a sitting dog: the femur `knee_up_deg` up
    and forward from the hip, a little out, putting the knee by the flank;
    the shin from there down to the hock on the floor (or `lift_deg` up the
    metatarsus off it), the metatarsus and paw forward along the floor.
    `dy` raises (or lowers) the hock and paw. (The hock set at a place ahead
    of the hip and the knee left to two-bone IK: once the rump sat right
    down, the hip came lower than the hock, the line from one to the other
    ran along the knee's pole and the knee folded back.)"""
    hip, knee, ankle, foot, toes = sk.leg(leg)
    side = float(np.sign(sk.rest[hip][0, 3]) or 1.0)
    w = sk.fk(pose)
    h = w[hip][:3, 3]
    r_hock = sk.radius(ankle, foot)
    l3 = sk.length(ankle, foot)
    paw_y = sk.rest[foot][1, 3] + dy
    lift = max(np.radians(lift_deg), np.arcsin(np.clip((r_hock - paw_y) / l3, -1.0, 1.0)))
    hock_y = paw_y + l3 * np.sin(lift)
    up = np.radians(knee_up_deg)
    femur = FWD * np.cos(up) + UP * np.sin(up) + KNEE_SPLAY * side * LEFT
    femur = femur / np.linalg.norm(femur)
    knee_at = h + sk.length(hip, knee) * femur
    # the hock under the hip's side of the body, back from the knee a shin's
    # length away (under the knee if the shin cannot reach the floor), so
    # the IK puts the knee where the femur points
    hock_x = sk.rest[ankle][0, 3] + side * splay
    dx, drop = knee_at[0] - hock_x, knee_at[1] - hock_y
    back = np.sqrt(max(sk.length(knee, ankle) ** 2 - drop * drop - dx * dx, 0.0))
    hock = np.array([hock_x, hock_y, knee_at[2] - back])
    two_bone(sk, pose, hip, knee, ankle, hock, femur)
    # the metatarsus down to the paw, turned out a little; the paw flat on
    aim(sk, pose, ankle, foot, FWD * np.cos(lift) - UP * np.sin(lift) + side * LEFT * META_OUT)
    aim(sk, pose, foot, toes, FWD - PAW_DROP * UP + side * LEFT * META_OUT)


def stand_front(sk: Skeleton, pose: Pose, leg: str, under: bool = True) -> None:
    """A front leg standing near straight: the wrist under the shoulder at
    its standing height, the pastern and paw as they stand."""
    shoulder, elbow, wrist, hand, toes = sk.leg(leg)
    w = sk.fk(pose)
    s = w[shoulder][:3, 3]
    target = np.array([sk.rest[wrist][0, 3], sk.rest[wrist][1, 3], s[2] if under else sk.rest[wrist][2, 3]])
    plant(sk, pose, leg, target, lower_dirs(sk, sk.rest, leg), -FWD)


def level_head(sk: Skeleton, pose: Pose, up_deg: float = 0.0) -> None:
    """Bring the head back to `up_deg` above its standing pitch, the turn
    shared down the neck."""
    head = sk.j["head_jnt"]
    w = sk.fk(pose)
    tilt = body_frame(sk, w, head) @ FWD
    pitch = np.degrees(np.arctan2(tilt[1], np.hypot(tilt[0], tilt[2]))) - up_deg
    for name, share in NECK_SHARES.items():
        turn(sk, pose, sk.j[name], LEFT, pitch * share)


def key_stand(sk: Skeleton) -> Pose:
    return sk.rest_pose.copy()


def key_sit(sk: Skeleton) -> tuple[Pose, float]:
    hips = [sk.j[n] for n in ("L_hip_jnt", "R_hip_jnt")]
    fronts = [sk.leg(leg) for leg in ("LF", "RF")]
    reach = np.mean([sk.length(c[0], c[1]) + sk.length(c[1], c[2]) for c in fronts])
    wrist_y = np.mean([sk.rest[c[2]][1, 3] for c in fronts])
    hip_z = np.mean([sk.rest[h][2, 3] for h in hips])
    pelvis = sk.j["pelvis_jnt"]

    def body(pitch: float) -> Pose:
        pose = sk.rest_pose.copy()
        turn(sk, pose, sk.cog, LEFT, -pitch)
        turn(sk, pose, pelvis, LEFT, -SIT_PELVIS_DEG)
        for name in ("spine_1_jnt", "spine_2_jnt"):
            turn(sk, pose, sk.j[name], LEFT, -SIT_SPINE_DEG)
        turn(sk, pose, sk.j["chest_jnt"], LEFT, SIT_CHEST_FLEX_DEG)
        w = sk.fk(pose)
        shift(sk, pose, np.array([0.0, 0.0, hip_z - np.mean([w[h][2, 3] for h in hips])]))
        ground(sk, pose)
        return pose

    def gap(pitch: float) -> float:
        w = sk.fk(body(pitch))
        return float(np.mean([w[c[0]][1, 3] for c in fronts]) - wrist_y - SIT_LEG_SHARE * reach)

    pitches = np.arange(SIT_PITCH_DEG[0], SIT_PITCH_DEG[1] + 0.5, 1.0)
    gaps = np.array([gap(p) for p in pitches])
    best = float(pitches[np.argmin(np.abs(gaps))])
    pose = body(best)
    pitch = best + 2 * SIT_SPINE_DEG                    # the chest's, for the report
    for leg in ("LB", "RB"):
        on_floor(sk, pose, leg, lambda p, dy, leg=leg: fold_hind(sk, p, leg, SIT_HOCK_LIFT_DEG, dy=dy))
    for leg in ("LF", "RF"):
        stand_front(sk, pose, leg)
    w = sk.fk(pose)
    neck = w[sk.j["neck_mid_jnt"]][:3, 3] - w[sk.j["neck_base_jnt"]][:3, 3]
    lean = np.degrees(np.arctan2(neck[1], neck[2])) - SIT_NECK_PITCH_DEG
    if lean > 0:
        turn(sk, pose, sk.j["neck_base_jnt"], LEFT, lean)
    level_head(sk, pose, SIT_HEAD_UP_DEG)
    ground(sk, pose)
    # (x is the dog's left: a turn about UP by a negative angle swings a
    # direction pointing back toward it. Turned about UP as it hung from the
    # tucked pelvis - near straight down - the tail went wherever draping it
    # lifted it, up behind the rump.)
    tail = [j for j in (sk.j.get(n) for n in TAIL) if j is not None]
    for a, b, heading in zip(tail, tail[1:], SIT_TAIL_HEADINGS):
        aim(sk, pose, a, b, rotm(UP, -heading) @ -FWD)
    drape_tail(sk, pose, settle=bool(SIT_TAIL_HEADINGS))
    return pose, pitch


def on_floor(sk: Skeleton, pose: Pose, leg: str, place) -> None:
    """`place(pose, dy)` lays a leg along the floor with its bones `dy` higher
    than their radius says; repeated until the leg's lowest vertex touches
    the floor (the mesh is not round about its bones)."""
    dy = 0.0
    for _ in range(4):
        trial = pose.copy()
        place(trial, dy)
        y, _ = sk.low(sk.fk(trial), lower_leg(sk, leg))
        if abs(y) < 0.002:
            break
        dy -= y
    pose.r, pose.t = trial.r, trial.t


def captured(sk: Skeleton, poses: dict, name: str) -> Pose:
    """Real-dog pose `name` (dog_mocap) on the template: each mapped joint
    turned in the world as the captured dog's turned from its standing
    pose, resting on the floor."""
    pose = sk.rest_pose.copy()
    turns = dog_mocap.turns(poses, name)
    for i in sk.order:
        name_i = (sk.rig.gltf.nodes[i].name or "").split(".")[0]
        if i in sk.joints and name_i in turns:
            set_world(sk, pose, sk.fk(pose), i, turns[name_i] @ rot_of(sk.rest[i]))
    ground(sk, pose, support(sk))
    return pose


def support(sk: Skeleton) -> np.ndarray:
    """What a lying or rolling body rests on: the torso and the legs from the
    knee / elbow down."""
    return sk.torso | sk.mask([j for leg in LEGS for j in sk.subtree(sk.leg(leg)[1])])


def even_back(sk: Skeleton, pose: Pose) -> None:
    """The back laid straight from the hips to the chest, its turn shared
    out evenly along it, the hips and the chest each turned as they were and
    the chest where it was (the hips let back by the length the bends took
    up) - see BACK."""
    w = sk.fk(pose)
    pelvis, chest = sk.j["pelvis_jnt"], sk.j["chest_jnt"]
    b_pelvis, b_chest = body_frame(sk, w, pelvis), body_frame(sk, w, chest)
    at = w[chest][:3, 3].copy()
    along = at - w[pelvis][:3, 3]
    for k, name in enumerate(BACK, 1):
        # (each of these carries one length of the back, along its rest FWD)
        b = slerp_m(b_pelvis, b_chest, k / (len(BACK) + 1))
        b = swing(b @ FWD, along) @ b
        set_world(sk, pose, sk.fk(pose), sk.j[name], b @ rot_of(sk.rest[sk.j[name]]))
    for i, b in ((pelvis, b_pelvis), (chest, b_chest)):
        set_world(sk, pose, sk.fk(pose), i, b @ rot_of(sk.rest[i]))
    shift(sk, pose, at - sk.fk(pose)[chest][:3, 3])


def square_hips(sk: Skeleton, pose: Pose, share: float = 1.0, untwist: float = 0.0) -> None:
    """The hips turned back in line behind the chest: the rump's swing aside
    (about the chest's up) undone `share` of the way, and its roll about the
    body's length against the chest `untwist` of the way, the back sharing
    the turn out (even_back). A turn about the up keeps every point's height
    as it was."""
    if share < 1e-6 and untwist < 1e-6:
        return
    w = sk.fk(pose)
    pelvis, chest = sk.j["pelvis_jnt"], sk.j["chest_jnt"]
    b_hips, b_chest = body_frame(sk, w, pelvis), body_frame(sk, w, chest)

    def about(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> float:
        """degrees from a to b about axis, both laid off it"""
        a, b = a - axis * (a @ axis), b - axis * (b @ axis)
        return float(np.degrees(np.arctan2(np.cross(a, b) @ axis, a @ b)))

    up = b_chest @ UP
    b_hips = rotm(up, -share * about(b_chest @ FWD, b_hips @ FWD, up)) @ b_hips
    if untwist > 1e-6:
        ahead = b_hips @ FWD
        b_hips = rotm(ahead, -untwist * about(b_chest @ LEFT, b_hips @ LEFT, ahead)) @ b_hips
    set_world(sk, pose, w, pelvis, b_hips @ rot_of(sk.rest[pelvis]))
    even_back(sk, pose)


def hinge(a: np.ndarray, b: np.ndarray, ref: np.ndarray, last: np.ndarray | None = None) -> np.ndarray:
    """The axis a joint bends about from bone direction `a` to `b`, on the
    side of `ref` - its axis as the bone above carries it (bent the other
    way, it is over-straightened, not turned round) - or, bent nearly
    square to that, of `last`, its axis a frame before; nearly straight,
    `last` or `ref`."""
    keep = ref if last is None else last
    c = np.cross(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)
    s = float(np.linalg.norm(c))
    if s < 1e-9:
        return keep
    n = c / s
    side = np.dot(n, ref) if abs(np.dot(n, ref)) > HINGE_SIDE_MIN or last is None else np.dot(n, last)
    if side < 0:
        n = -n
    u = min(s / np.sin(np.radians(HINGE_STRAIGHT_DEG)), 1.0)
    n = u * n + (1.0 - u) * keep
    return n / max(np.linalg.norm(n), 1e-12)


def bone_frame(d: np.ndarray, n: np.ndarray) -> np.ndarray:
    d = d / np.linalg.norm(d)
    n = n - np.dot(n, d) * d
    n = n / max(np.linalg.norm(n), 1e-12)
    return np.column_stack([d, n, np.cross(d, n)])


def untwist_leg(sk: Skeleton, pose: Pose, leg: str, prev: list | None = None) -> list:
    """Each bone of a leg turned about its own length so that the knee,
    hock and paw (elbow, wrist, paw) bend about one axis each, square to
    the bones they join, as they do at rest - the bones pointing as they
    were, so nothing moves (see UNTWIST). Returns the axes (in the frame of
    the joint the leg hangs from); given the last frame's as `prev`, each
    keeps to its side of it (each frame on its own, a joint near straight
    or bent the wrong way had its axis flipped from one frame to the next,
    and the leg spun 90-170 deg)."""
    chain = sk.leg(leg)
    rest = [sk.rest[j][:3, 3] for j in chain]
    d0 = [b - a for a, b in zip(rest, rest[1:])]
    n0 = [np.cross(a, b) / np.linalg.norm(np.cross(a, b)) for a, b in zip(d0, d0[1:])]
    w = sk.fk(pose)
    at = [w[j][:3, 3] for j in chain]
    d = [b - a for a, b in zip(at, at[1:])]
    root = body_frame(sk, w, sk.parent[chain[0]])
    carried = root
    turns, axes = [], [None] * len(n0)
    for k in range(len(d)):
        # the bone turned to point where it does, its bend at either end
        # about that joint's axis; between two joints, halfway
        cands = []
        for j in (k - 1, k):
            if 0 <= j < len(n0):
                if axes[j] is None:
                    axes[j] = hinge(d[j], d[j + 1], carried @ n0[j], root @ prev[j] if prev is not None else None)
                cands.append(bone_frame(d[k], axes[j]) @ bone_frame(d0[k], n0[j]).T)
        r = cands[0] if len(cands) == 1 else slerp_m(cands[0], cands[1], 0.5)
        turns.append(r)
        carried = r
    for j, r in zip(chain, turns):
        set_world(sk, pose, sk.fk(pose), j, r @ rot_of(sk.rest[j]))
    return [root.T @ n for n in axes]


def settle_front(sk: Skeleton, pose: Pose) -> float:
    """A lying body tipped forward about its hips until its front (chest,
    forelegs) rests on the floor as its back does, the head kept as it
    looked; returns the degrees."""
    sup = support(sk)
    front = sup & sk.mask(sk.subtree(sk.j["spine_2_jnt"]))
    back = sup & ~front
    neck = sk.j["neck_base_jnt"]
    keep = pose.copy()

    def gap(deg: float) -> float:
        pose.r, pose.t = {k: v.copy() for k, v in keep.r.items()}, keep.t.copy()
        turn(sk, pose, sk.cog, LEFT, deg)
        turn(sk, pose, neck, LEFT, -deg)
        ground(sk, pose, sup)
        w = sk.fk(pose)
        return sk.low(w, front)[0] - sk.low(w, back)[0]

    lo, hi = 0.0, SETTLE_FRONT_MAX_DEG
    if gap(lo) <= FLOOR_SLACK_M:
        gap(lo)
        return 0.0
    if gap(hi) > 0:
        return hi
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        if gap(mid) > 0:
            lo = mid
        else:
            hi = mid
    gap(hi)
    return hi


def face_front(sk: Skeleton, pose: Pose) -> None:
    """The head turned to look the way the chest points, level across
    (FACE_FRONT_SHARES of the turn at each neck joint), its pitch kept."""
    w = sk.fk(pose)
    ahead = body_frame(sk, w, sk.j["chest_jnt"]) @ FWD
    looks = body_frame(sk, w, sk.j["head_jnt"]) @ FWD
    yaw = np.degrees(np.arctan2(looks[0], looks[2]) - np.arctan2(ahead[0], ahead[2]))
    yaw = (yaw + 180.0) % 360.0 - 180.0
    for name, share in FACE_FRONT_SHARES.items():
        turn(sk, pose, sk.j[name], UP, -yaw * share)
    head = sk.j["head_jnt"]
    b = body_frame(sk, sk.fk(pose), head)
    tilt = np.degrees(np.arcsin(np.clip((b @ LEFT) @ UP, -1.0, 1.0)))
    turn(sk, pose, head, b @ FWD, -tilt)


def align_leg(sk: Skeleton, pose: Pose, leg: str, prev: list | None = None) -> list:
    """Each bone of a leg turned no further than it takes to point where it
    does from where the bone above carries it at rest - none turned about
    its own length - the bones pointing as they were, so nothing moves.
    (For a clip: untwist_leg finds each joint's axis afresh and, frame by
    frame, a joint near straight or bent sideways flipped it.) Given the
    last frame's turns (`prev`, in the frame of the joint the leg hangs
    from), each bone turns about its length toward that at most
    ALIGN_STEP_DEG a frame from where it was, and not at all toward one
    more than a quarter turn away (bent sideways, where 'the least turn' is
    no one way). Returns this frame's."""
    chain = sk.leg(leg)
    rest = [sk.rest[j][:3, 3] for j in chain]
    d0 = [b - a for a, b in zip(rest, rest[1:])]
    w = sk.fk(pose)
    at = [w[j][:3, 3] for j in chain]
    root = body_frame(sk, w, sk.parent[chain[0]])
    carried = root
    out = []
    for k, j in enumerate(chain[:-1]):
        ref = carried
        if k:
            # from the joint above bent ALIGN_MID_DEG, not from straight: a
            # leg folded tight points nearly opposite its straight self, and
            # the least turn there swung any way (165 deg in a frame)
            n0 = np.cross(d0[k - 1], d0[k])
            bent = np.degrees(np.arctan2(np.linalg.norm(n0), d0[k - 1] @ d0[k]))
            ref = carried @ rotm(n0, ALIGN_MID_DEG - bent)
        d = at[k + 1] - at[k]
        r = swing(ref @ d0[k], d) @ ref
        if k == 0:
            # the thigh / upper arm turned about its length (as far as a hip
            # or shoulder turns, ALIGN_ROOT_TWIST_DEG) so that the knee /
            # elbow bends in its own plane, not sideways
            hinge0 = np.cross(d0[0], d0[1])
            have = r @ (hinge0 / np.linalg.norm(hinge0))
            bend_axis = np.cross(d, at[2] - at[1])
            s = np.linalg.norm(bend_axis) / max(np.linalg.norm(d) * np.linalg.norm(at[2] - at[1]), 1e-12)
            if s > 1e-6:
                want = bend_axis / np.linalg.norm(bend_axis)
                if want @ have < 0:
                    want = -want                            # bent back: the nearer side
                u = d / np.linalg.norm(d)
                h = have - (have @ u) * u
                tw = float(np.degrees(np.arctan2(np.cross(h, want) @ u, h @ want)))
                weight = min(s / np.sin(np.radians(HINGE_STRAIGHT_DEG)), 1.0)
                r = rotm(d, weight * float(np.clip(tw, -ALIGN_ROOT_TWIST_DEG, ALIGN_ROOT_TWIST_DEG))) @ r
        if prev is not None:
            # where it was, carried on to point the way it does now
            was = root @ prev[k]
            was = swing(was @ d0[k], d) @ was
            # r and was both point the bone along d: they differ by a turn about it
            delta = Rotation.from_matrix(r @ was.T).as_rotvec()
            angle = float(np.degrees(np.linalg.norm(delta))) * float(np.sign(delta @ d) or 1.0)
            step = 0.0 if abs(angle) > 90.0 else float(np.clip(angle, -ALIGN_STEP_DEG, ALIGN_STEP_DEG))
            r = rotm(d, step) @ was
        set_world(sk, pose, sk.fk(pose), j, r @ rot_of(sk.rest[j]))
        out.append(root.T @ r)
        carried = r
    return out


def hinge_leg(sk: Skeleton, pose: Pose, leg: str, most_deg: float = HINGE_SIDE_DEG) -> None:
    """The bones below the knee / elbow swung into the plane the leg bends
    in at the knee / elbow (within `most_deg` of it), so that the hock and
    paw (wrist and paw) bend as hinges, the way the knee does - nearly
    straight there, the plane the knee's axis at rest makes with the femur /
    upper arm. Lying on its left hip or side, the captured dogs' left hind
    hock bent 40-60 deg sideways, the heel out and the foot turned in under
    the belly (user, 11-06: "왼쪽다리 뭔가 꺾이는거 ... 둘다 그런데")."""
    a, b, c, d, e = sk.leg(leg)
    w = sk.fk(pose)
    u1, u2 = w[b][:3, 3] - w[a][:3, 3], w[c][:3, 3] - w[b][:3, 3]
    n = np.cross(u1, u2)
    s = float(np.linalg.norm(n) / max(np.linalg.norm(u1) * np.linalg.norm(u2), 1e-12))
    r0 = [sk.rest[j][:3, 3] for j in (a, b, c)]
    n0 = np.cross(r0[1] - r0[0], r0[2] - r0[1])
    rest_axis = body_frame(sk, w, a) @ (n0 / np.linalg.norm(n0))
    if s < 1e-6:
        n = rest_axis
    else:
        n = n / np.linalg.norm(n)
        if n @ rest_axis < 0:
            n = -n
        k = min(s / np.sin(np.radians(HINGE_STRAIGHT_DEG)), 1.0)    # nearly straight: the rest axis
        n = k * n + (1.0 - k) * rest_axis
        n = n / np.linalg.norm(n)
    most = np.sin(np.radians(most_deg))
    for i, child in ((c, d), (d, e)):
        w = sk.fk(pose)
        v = w[child][:3, 3] - w[i][:3, 3]
        length = float(np.linalg.norm(v))
        off = float(v @ n) / max(length, 1e-12)
        if abs(off) <= most:
            continue
        inside = v - n * (v @ n)
        if np.linalg.norm(inside) < 1e-9:
            continue
        inside = inside / np.linalg.norm(inside)
        aim(sk, pose, i, child, inside * np.sqrt(1.0 - most * most) + n * np.sign(off) * most)


def flat_paw(sk: Skeleton, pose: Pose, leg: str, weight: float = 1.0) -> None:
    """The paw laid flat - turned as it is at rest, pads down, about the
    vertical as the body is (the chest / pelvis it hangs from: by the way
    the paw itself pointed, near straight down that swung 70 deg out and
    back in a third of a second) - its toe where it is: the bone
    above it (pastern / metatarsus) keeping its heading, pitched only as
    far as keeps its upper end (wrist / hock) its own radius off the floor,
    and the leg above reaching for that by two-bone IK, bent as it is;
    `weight` of the way. The paw's bone points 44 (fore) / 59 (hind) deg
    down when its pads are flat; laid near level, as the lying keys and the
    capture had it, the pads faced 45-70 deg forward and the paw looked
    flipped up or knuckled under (user, 11-06: "여전히 이상하잖아 직접봐
    왼쪽다리")."""
    if weight <= 1e-3:
        return
    a, b, c, d, e = sk.leg(leg)
    w = sk.fk(pose)
    keep = {i: pose.r.get(i, sk.rest_r[i]).copy() for i in (a, b, c, d)}
    toe = w[e][:3, 3].copy()
    paw_rest = sk.rest[e][:3, 3] - sk.rest[d][:3, 3]
    h0 = paw_rest - UP * (paw_rest @ UP)
    heading = body_frame(sk, w, leg_frame(sk, leg)) @ h0
    heading = heading - UP * (heading @ UP)
    if np.linalg.norm(heading) < 1e-6:
        heading = h0
    yaw = np.arctan2(np.cross(h0, heading) @ UP, h0 @ heading)
    spin = rotm(UP, np.degrees(yaw))
    paw = spin @ paw_rest                                   # the flat paw, toe to joint reversed
    joint = toe - paw
    pastern = w[d][:3, 3] - w[c][:3, 3]
    length = float(np.linalg.norm(pastern))
    flat_dir = pastern - UP * (pastern @ UP)
    if np.linalg.norm(flat_dir) < 1e-6:
        flat_dir = heading
    flat_dir = flat_dir / np.linalg.norm(flat_dir)
    # (pitched to keep the wrist / hock at its height - lying, the forearm
    # or the hock stays on the floor and the pastern rises to the paw - but
    # never lower than its own radius off the floor)
    lift = (joint[1] - w[c][1, 3]) / length                 # sine of its pitch, + rising toward the paw
    most = (joint[1] - sk.radius(c, d)) / length
    lift = float(np.clip(min(lift, most), -1.0, 1.0))
    pastern = (flat_dir * np.sqrt(1.0 - lift * lift) + UP * lift) * length
    plant(sk, pose, leg, joint - pastern, (pastern, paw), w[b][:3, 3] - w[a][:3, 3])
    w2 = sk.fk(pose)
    set_world(sk, pose, w2, d, spin @ rot_of(sk.rest[d]))
    if weight < 1.0:
        for i in (a, b, c, d):
            pose.r[i] = slerp_m(keep[i], pose.r[i], weight)


def near_floor(sk: Skeleton, w: dict, leg: str) -> float:
    """How fully a paw rests on the floor, by its toe's height over the
    toe's height at rest (PAW_FLAT_M): 1 on it, 0 lifted clear."""
    e = sk.leg(leg)[-1]
    up = w[e][1, 3] - sk.rest[e][1, 3]
    return float(np.clip((PAW_FLAT_M[1] - up) / (PAW_FLAT_M[1] - PAW_FLAT_M[0]), 0.0, 1.0))


def flat_paws(sk: Skeleton, pose: Pose, legs=tuple(LEGS), share: float = 1.0) -> None:
    """Each paw on the floor laid flat (flat_paw), as fully as it rests on it."""
    for leg in legs:
        flat_paw(sk, pose, leg, share * near_floor(sk, sk.fk(pose), leg))


def limit_hinges(sk: Skeleton, pose: Pose, leg: str) -> None:
    """A knee, hock, elbow or wrist bent the wrong way past HINGE_MIN_DEG
    brought back to it (degrees of bend the way the joint bends at rest -
    which at rest is the way each bends), the leg below it turned with it."""
    chain = sk.leg(leg)
    rest = [sk.rest[j][:3, 3] for j in chain]
    d0 = [b - a for a, b in zip(rest, rest[1:])]
    for j, (least, most) in enumerate(zip(HINGE_MIN_DEG[leg[-1]], HINGE_MAX_DEG[leg[-1]])):
        if least is None:
            continue
        w = sk.fk(pose)
        a, b, c = (w[chain[n]][:3, 3] for n in (j, j + 1, j + 2))
        n0 = np.cross(d0[j], d0[j + 1])
        n = body_frame(sk, w, chain[j]) @ (n0 / np.linalg.norm(n0))
        u, v = b - a, c - b
        across = np.cross(u, v)
        # only a joint bent in its own plane: bent sideways, its bend read
        # about its axis swung between -180 and 180 and put right, the leg
        # spun 160 deg in a frame
        if abs(across @ n) < HINGE_IN_PLANE * np.linalg.norm(across) and np.linalg.norm(across) > 1e-9:
            continue
        bend = np.degrees(np.arctan2(across @ n, u @ v))
        if bend < least or bend > most:
            fix = float(np.clip(bend, least, most)) - bend
            turn(sk, pose, chain[j + 1], n, float(np.clip(fix, -HINGE_FIX_MAX_DEG, HINGE_FIX_MAX_DEG)))


def key_down(sk: Skeleton, poses: dict, hip_roll_deg: float = DOWN_HIP_ROLL_DEG,
             upper_in_deg: float = DOWN_UPPER_HIND_IN_DEG, fold_hind_legs: bool = True) -> Pose:
    pose = captured(sk, poses, "down")
    side = roll_side(sk, pose)
    pelvis = sk.j["pelvis_jnt"]
    # further onto the hip it lies on, the upper hind leg in after it (DOWN_HIP_ROLL_DEG)
    turn(sk, pose, pelvis, body_frame(sk, sk.fk(pose), pelvis) @ FWD, -side * hip_roll_deg)
    even_back(sk, pose)
    upper = sk.leg("RB" if side > 0 else "LB")[0]
    turn(sk, pose, upper, body_frame(sk, sk.fk(pose), pelvis) @ FWD, side * upper_in_deg)
    for leg in LEGS:
        untwist_leg(sk, pose, leg)
    # the hind hocks and paws bending as hinges (the captured left hind foot
    # lay turned in under the belly, the heel out sideways)
    for leg in ("LB", "RB"):
        hinge_leg(sk, pose, leg)
        untwist_leg(sk, pose, leg)
    # the forequarters sternal and square, as a dog lying on one hip holds
    # its chest - the chest up straight on the forelegs alike, the head
    # straight ahead - and the back taking the turn from the rolled hips
    # (the captured chest leaned 10 deg and turned 9, the left forepaw under
    # it past the line of the body and the right out at twice the stance,
    # the head turned with the chest - user, 11-06: "엎드려 구르기할때 발이랑
    # 몸, 고개가 비틀려있는데")
    pose = square(sk, pose, sk.subtree(sk.j["chest_jnt"]))
    even_back(sk, pose)
    if fold_hind_legs:
        # the hind legs folded under it as the sphinx folds them (fold_hind:
        # the knee by the flank, the hock on the floor beside the hip, the
        # metatarsus and paw forward along it), only the hips rolled. The
        # captured ones lay out to the side it does not lie on: the one it
        # lies on had its hock opened to 118 deg and up 7 cm (the sphinx's:
        # 166, 5 cm), the shin and metatarsus standing in a Z off the paw,
        # the leg turned 42 deg about its length against the hips; the other
        # slid 10 cm out from under the rump and 5 cm up. Going from the
        # sphinx into it, just before the roll, the dog put a hind foot out
        # to its side and lifted a heel as if to spring (user, 11-06:
        # "맨처음에 걍 엎드려있다가 왼쪽 뒷다리 들고 도는데 왼쪽 뒷다리가
        # 기괴하다니까?", "구르기 시작하기 직전에 발을 왼쪽으로 잠깐 뺀다니까
        # 이거 근데 기괴하게 생겼다고")
        ground(sk, pose, support(sk))
        for leg in ("LB", "RB"):
            on_floor(sk, pose, leg, lambda p, dy, leg=leg: fold_hind(sk, p, leg, 0.0, dy=dy,
                                                                     knee_up_deg=CROUCH_KNEE_UP_DEG))
    settle_front(sk, pose)
    face_front(sk, pose)
    # the paws on the floor flat, pads down (laid near level, the forepaws'
    # pads faced 45 deg forward and out and the left hind paw's 70)
    flat_paws(sk, pose)
    ground(sk, pose, support(sk))
    drape_tail(sk, pose, settle=True)
    return pose


def key_side(sk: Skeleton, poses: dict) -> Pose:
    pose = captured(sk, poses, "side")
    lift_head(sk, pose, settle=True)
    drape_tail(sk, pose, settle=True)
    ground(sk, pose, support(sk))
    # centred where it stood
    p = sk.points(sk.fk(pose), sk.all)
    shift(sk, pose, np.array([-(p[:, 0].min() + p[:, 0].max()) / 2, 0.0, 0.0]))
    return pose


def roll_side(sk: Skeleton, down: Pose) -> float:
    """The side a dog lying in `down` rolls over onto: the one its hips lie
    on (+1 left - x is the dog's left; -1 right)."""
    w = sk.fk(down)
    return 1.0 if w[sk.j["L_hip_jnt"]][1, 3] <= w[sk.j["R_hip_jnt"]][1, 3] else -1.0


def key_curl(sk: Skeleton, down: Pose) -> Pose:
    """The down about to roll over: the head turned toward the side it goes
    over on (kept as .side)."""
    pose = down.copy()
    pose.side = roll_side(sk, down)
    neck = sk.j["neck_base_jnt"]
    turn(sk, pose, neck, body_frame(sk, sk.fk(pose), neck) @ UP, pose.side * ROLL_HEAD_TURN_DEG)
    return pose


def key_tuck(sk: Skeleton, curl: Pose) -> Pose:
    """The curl rolled into a ball (ROLL_CURL_DEG), the legs folded close
    (TUCK_LEGS). (Rounded lying on its chest, the back pitched the
    forelegs into the floor and the dog stood up on them, 29 cm.)"""
    pose = curl.copy()
    # the back straight and untwisted, hips square with the chest (the down
    # lies on one hip, 34 deg round from the chest: rolled over like that
    # its back stayed wrung 24 deg)
    w = sk.fk(pose)
    pelvis = sk.j["pelvis_jnt"]
    set_world(sk, pose, w, pelvis, body_frame(sk, w, sk.j["chest_jnt"]) @ rot_of(sk.rest[pelvis]))
    even_back(sk, pose)
    for name, deg in ROLL_CURL_DEG.items():
        i = sk.j[name]
        turn(sk, pose, i, body_frame(sk, sk.fk(pose), i) @ LEFT, deg)
    for leg in LEGS:
        shape_leg(sk, pose, leg, TUCK_LEGS[leg], leg_frame(sk, leg))
        untwist_leg(sk, pose, leg)
    return pose


def key_shake(sk: Skeleton, sit: Pose) -> Pose:
    pose = sit.copy()
    shape_leg(sk, pose, SHAKE_LEG_NAME, SHAKE_LEG)
    return pose


def solve_keys(sk: Skeleton) -> dict[str, Pose]:
    poses = dog_mocap.load()
    sit, pitch = key_sit(sk)
    down = key_down(sk, poses)
    # (rolled from the sphinx instead, square, its hind legs went from folded
    # alike under it into the tuck and back and the roll went stiff - user,
    # 11-06: "구르기 ... 너무 딱딱하고 부자연스러워 ... 뒷발이 특히 이상해짐":
    # a dog rolls from lying on one hip, over onto that side)
    curl = key_curl(sk, down)
    keys = {"stand": key_stand(sk), "sit": sit, "down": down, "side": key_side(sk, poses),
            "curl": curl, "tuck": key_tuck(sk, curl), "shake": key_shake(sk, sit),
            "sphinx": crouch(sk, key_down(sk, poses, *SPHINX_DOWN_DEG, fold_hind_legs=False))}
    keys["sit"].pitch = pitch                                   # for the report
    return keys


# --- blending -----------------------------------------------------------------------------

def slerp_m(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    """Rotation `u` of the way from a to b (u outside 0..1 runs on past)."""
    delta = Rotation.from_matrix(b @ a.T).as_rotvec()
    return Rotation.from_rotvec(delta * u).as_matrix() @ a


def blend(sk: Skeleton, a: Pose, b: Pose, u: float, nodes=None) -> Pose:
    out = a.copy()
    for i in nodes if nodes is not None else sk.joints:
        ra, rb = a.r.get(i, sk.rest_r[i]), b.r.get(i, sk.rest_r[i])
        out.r[i] = slerp_m(ra, rb, u)
    if nodes is None:
        out.t = a.t + (b.t - a.t) * u
    return out


def smooth(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


# --- the shiba: timing and secondary motion -----------------------------------------------

class Source:
    """The retired shiba's clips, sampled at FPS."""

    def __init__(self, path: Path = SOURCE):
        self.rig = Rig(path)
        self.rest = self.rig.world()
        self.cache: dict[str, tuple] = {}

    def clip(self, name: str):
        if name not in self.cache:
            times, frames = sample_clip(self.rig, name, FPS)
            worlds = [self.rig.world(f) for f in frames]
            self.cache[name] = (times, frames, worlds)
        return self.cache[name]

    def node(self, name: str) -> int | None:
        return self.rig.by_name.get(name)

    def local(self, frame: dict, node: int) -> np.ndarray:
        return Rotation.from_quat(frame.get(node, self.rig.rest_trs[node])[1]).as_matrix()

    def cog_rot(self, world: dict) -> np.ndarray:
        return rot_of(world[self.node("COG_jnt")])

    def roll(self, name: str) -> np.ndarray:
        """The body's roll about its length, unwrapped, from the clip's start (deg)."""
        times, frames, worlds = self.clip(name)
        r0 = self.cog_rot(worlds[0])
        rolls = []
        for w in worlds:
            d = self.cog_rot(w) @ r0.T
            rolls.append(Rotation.from_matrix(d).as_euler("zyx")[0])
        return np.degrees(np.unwrap(rolls))

    def progress(self, name: str, t0: float, t1: float) -> np.ndarray:
        """Per frame, how far the body has turned from its orientation at t0
        toward that at t1 (0..1, held outside)."""
        times, frames, worlds = self.clip(name)
        k0, k1 = int(round(t0 * FPS)), min(int(round(t1 * FPS)), len(times) - 1)
        ra, rb = self.cog_rot(worlds[k0]), self.cog_rot(worlds[k1])
        out = np.zeros(len(times))
        for k, w in enumerate(worlds):
            r = self.cog_rot(w)
            da = Rotation.from_matrix(r @ ra.T).magnitude()
            db = Rotation.from_matrix(r @ rb.T).magnitude()
            out[k] = da / max(da + db, 1e-9)
        out[:k0] = 0.0
        out[k1:] = 1.0
        return np.clip(out, 0.0, 1.0)

    def paw_height(self, name: str, leg: str) -> np.ndarray:
        times, frames, worlds = self.clip(name)
        toe = self.node(LEGS[leg][-1])
        return np.array([w[toe][1, 3] for w in worlds])


def partner(joint: str) -> str:
    """The joint on the other side (L_ for R_), or the joint itself on the midline."""
    for a, b in (("L_", "R_"), ("R_", "L_")):
        if joint.startswith(a):
            return b + joint[len(a):]
    return joint


def secondary(sk: Skeleton, src: Source, name: str, k: int, k_ref: int, weight: float, pose: Pose,
              mirror: bool = False, joints=SECONDARY) -> None:
    """Put the shiba's motion since frame `k_ref` on the pose: the body's
    world turn on the COG, and each SECONDARY joint's turn relative to its
    parent, carried across in world axes through the two rigs' rest frames.
    `mirror`: the motion's mirror image - each turn reflected left for
    right, a joint's taken from its partner on the other side."""
    if weight <= 0:
        return
    times, frames, worlds = src.clip(name)
    cog = src.node("COG_jnt")
    d = src.cog_rot(worlds[k]) @ src.cog_rot(worlds[k_ref]).T
    if mirror:
        d = MIRROR @ d @ MIRROR
    w = sk.fk(pose)
    d = Rotation.from_rotvec(Rotation.from_matrix(d).as_rotvec() * weight).as_matrix()
    set_world(sk, pose, w, sk.cog, d @ rot_of(w[sk.cog]))
    for joint in joints:
        i, j = sk.j.get(joint), src.node(partner(joint) if mirror else joint)
        if i is None or j is None or cog is None:
            continue
        local = src.local(frames[k], j) @ src.local(frames[k_ref], j).T
        ps, pt = rot_of(src.rest[src.rig.parent[j]]), rot_of(sk.rest[sk.parent[i]])
        turn_world = ps @ local @ ps.T
        if mirror:
            turn_world = MIRROR @ turn_world @ MIRROR
        delta = pt.T @ turn_world @ pt
        delta = Rotation.from_rotvec(Rotation.from_matrix(delta).as_rotvec() * weight).as_matrix()
        pose.r[i] = delta @ pose.r.get(i, sk.rest_r[i])


# --- the clips ----------------------------------------------------------------------------

@dataclass
class Hold:
    key: str
    t0: float
    t1: float
    # the shiba's motion read from this time of its clip on (default: t0),
    # and mirrored left for right
    src_t0: float | None = None
    mirror: bool = False


@dataclass
class Change:
    a: str
    b: str
    t0: float
    t1: float
    # eased over its own time (else as the shiba's body turns over it)
    timed: bool = False


@dataclass
class Roll:
    key: str
    t0: float
    t1: float


# segment times read off the shiba's clips (its body's orientation and its
# paws' heights over time)
# the shiba's clip each of these takes its timing and motion from (else its own name)
SOURCE_CLIPS = {"lying": "rollover"}
CLIPS = {
    # in its 12 s the shiba looks round to its right once (the viewer's left,
    # facing it) and back; played again mirrored, the dog looks to either
    # side in turn (user, 11-03: "왼쪽만 보게 되어 있는 것 같은데 오른쪽도
    # 번갈아 보게 해줘")
    "standing": [Hold("stand", 0.0, 12.0), Hold("stand", 12.0, 24.0, src_t0=0.0, mirror=True)],
    # (the shiba's sitting clip looks round the same way, once)
    "sitting": [Hold("sit", 0.0, 12.0), Hold("sit", 12.0, 24.0, src_t0=0.0, mirror=True)],
    "shake": [Hold("sit", 0.0, 6.5)],
    # from lying square (the lying clip's sphinx): the hips go over onto
    # their side, the hind legs kept folded under it (a dog rolls from lying
    # on one hip; 0.6 s; see key_down), the legs fold, it rolls over (the shiba's roll),
    # the legs unfold, and back square - held in the roll's down before and
    # after (lie_flop and lie_square to and from the lying clip), the dog lay
    # crooked for 6 s with a hind paw turned over out to the side (user,
    # 11-06: "엎드려구르기 여전히 왼쪽으로 발이 뒤틀려있고"). (The holds carry
    # the shiba's lying motion before / after its own roll.)
    "rollover": [Hold("sphinx", 0.0, 0.4, src_t0=1.42), Change("sphinx", "down", 0.4, 1.0, timed=True),
                 Hold("down", 1.0, 1.3, src_t0=0.0), Roll("down", 1.3, 2.8), Hold("down", 2.8, 3.5, src_t0=1.42),
                 Change("down", "sphinx", 3.5, 4.1, timed=True), Hold("sphinx", 4.1, 5.0, src_t0=2.12)],
    # lying down as told, square on its chest (a sphinx: crouch()), between
    # lie_down and get_up: it breathes, wags along the floor and moves its
    # ears as the roll's lying does, looking at whoever told it (without an
    # end to come back to, the dog froze on the roll's first frame lying)
    "lying": [Hold("sphinx", 0.0, 6.0, src_t0=1.42)],
}
# clips that are a real dog's motion from start to end (dog_mocap.CLIPS):
# lying down, flopping onto its side and getting up. (Standing was the
# capture's too, for a while: its weight shifts and turns on planted paws
# set the template's straight legs askew - user, 11-03: "발을 왜 굳이
# 삐딱하게" - so it is the template's own stance again, like idle. Built
# from the template's own poses instead - lying down, a flop onto its side,
# still, back and up - playing dead went stiff and lifeless: user, 11-06:
# "죽은척이랑 구르기 이전대비 너무 딱딱하고 부자연스러워".)
CAPTURED_CLIPS = ("play_dead",)
# a captured paw on the floor and still is held where it came down, while
# the dog's hips are above this share of their standing height (lying, the
# paws lie as the capture has them)
# clips read off video (video_motion), each moving from one of the
# template's key poses
VIDEO_BASED = {"shake": "sit"}
PLANT_UPRIGHT_SHARE = 0.5
HOLD_EASE_FRAMES = 5
# a captured paw within CONTACT_M of the floor is on it, still or sliding
# (the capture's own mark, planted, also wants it still), its runs cleaned
# of gaps up to CONTACT_GAP_FRAMES and of runs shorter than CONTACT_MIN_FRAMES
CONTACT_M = 0.03
CONTACT_GAP_FRAMES = 3
CONTACT_MIN_FRAMES = 3
# lying (hips below PLANT_UPRIGHT_SHARE), a paw the capture has within
# PAW_DOWN_M[0] of the floor is let down onto it, one above PAW_DOWN_M[1]
# left as it is, between eased; each joint turned at most PAW_DOWN_LIMIT_DEG
PAW_DOWN_M = (0.05, 0.09)
PAW_DOWN_LIMIT_DEG = 45.0
PAW_DOWN_EASE_FRAMES = 12
LOOP_BLEND_S = 0.5
# a captured clip's body levelled side to side while it leans less than
# CAPTURED_LEVEL_DEG[0], not at all from [1] (lying on its side), each paw on
# its own side (own_side), the corrections smoothed over
# CAPTURED_SMOOTH_FRAMES
CAPTURED_LEVEL_DEG = (15.0, 30.0)
STANCE_SHARE = (0.8, 1.5)
STANCE_KEEP = 0.35
# with its chest up, the body tipped forward (CAPTURED_LEAN_MAX_DEG at most)
# until its shoulders stand no higher than CAPTURED_SHOULDER_SHARE of the
# rest pose's - [0] with the hips down to CAPTURED_LEAN_HIPS[0] of their
# standing height, [1] from [1] up - resting on the floor by its hind end
# from CAPTURED_LEAN_GROUND_DEG of tip on; a free paw moved SIDE_FOLLOW of
# the way to its own side each frame; the legs smoothed over time over
# CAPTURED_LEG_SMOOTH_FRAMES
CAPTURED_LEAN_MAX_DEG = 40.0
CAPTURED_SHOULDER_SHARE = (0.95, 1.0)
CAPTURED_LEAN_HIPS = (0.6, 0.85)
CAPTURED_LEAN_GROUND_DEG = 8.0
SIDE_FOLLOW = 0.6
CAPTURED_LEG_SMOOTH_FRAMES = 1.6
# lying on its side (from SIDE_LYING_DEG[0] over onto it, fully from [1]),
# a captured clip's forelegs laid out from the chest as (degrees from the
# body's down toward its front, outward share) per bone, the leg underneath
# straight out and the upper one on it a little ahead, drawn in toward it -
# the captured dog lay with the upper foreleg across the lower at the wrists
# and the paws curled over, pads up (user, 11-06: "죽은척할때 중간에 앞발
# 비틀리는거만 개선")
SIDE_LYING_DEG = (50.0, 70.0)
SIDE_FORELEGS = {"lower": ((28.0, 0.0), (40.0, 0.0), (55.0, 0.0), (75.0, 0.0)),
                 "upper": ((38.0, -0.12), (50.0, -0.05), (62.0, 0.0), (80.0, 0.0))}
CAPTURED_SMOOTH_FRAMES = 2.0
# TRANSITIONS: a change of pose between two held clips, played once - from
# the first frame of the clip it leaves to the first frame of the one it goes
# into (each held clip loops back onto its first frame), so that the client
# plays loop -> transition -> loop (user, 11-05: interaction by touch and
# voice - "앉아" from standing). Built from the template's own poses: the
# captured dog's joint turns put on the template (as play_dead is) splayed
# its hind legs like a frog's sitting down, crossed its forelegs, and turned
# and tipped its body as the captured dog's (user, 11-05: "다리가 전체적으로
# 부자연스럽게 꺾여서 ... 몸도 왜 삐딱하게 비트는건데"). So the capture
# (dog_mocap.TRANSITIONS) gives only its timing - how its hips and its
# shoulders go down or up (the rump first sitting down, the forequarters
# first lying down, the forequarters first getting up) and when it moves
# each paw. The body goes from the one clip's shape to the other's square
# (no turn or lean of its own), its hips and shoulders at the capture's
# heights brought onto the two clips' (the body tipped about its hips to
# put the shoulders there), the head carried between the two clips'
# carriage. Each leg is solved as the key poses solve theirs: the paw on
# its spot, the knee / elbow bent the way the two clips bend it, the hock /
# pastern turned from the one clip's to the other's as the body goes. A paw
# stays where it is until the captured dog moves that paw, then steps
# (STEP_LIFT_M high, a forepaw curling back TRANSITION_CURL_DEG) as far as
# that move takes it of the way to the second clip's spot (the last all the
# way); a paw the capture never moves steps once the body is STEP_AFTER of
# the way. Lying down, the body goes down square onto hind legs folded
# under it alike (a sphinx, crouch()) and lies there (the lying clip), which
# the roll starts from and comes back to - lying down onto the roll's down
# straight, the hips going over at the end looked crooked (user, 11-05:
# "몸도 왜 삐딱하게"). (Going between a held clip and a lying pose that is
# not square, the body would gather from / settle onto it over
# LIE_SETTLE_S.) The ends are eased onto the clips' own frames over
# TRANSITION_IN_S / TRANSITION_OUT_S.
TRANSITIONS = {"sit_down": ("standing", "sitting"), "stand_up": ("sitting", "standing"),
               "lie_down": ("standing", "lying"), "get_up": ("lying", "standing")}
TRANSITION_IN_S = 0.2
TRANSITION_OUT_S = 0.3
LIE_SETTLE_S = 0.5
STEP_LIFT_M = 0.04
TRANSITION_CURL_DEG = 35.0
STEP_S = 0.3
STEP_MIN_S = 0.29
STEP_GAP_S = 0.08
STEP_TOLERANCE_M = 0.02
STEP_AFTER = 0.85
# a captured paw moving faster than this along the floor is stepping
STEP_SPEED_MS = 0.25
# the capture's shoulders going up or down by more than this are followed
# as they go; less (sitting, standing up), their small dip on top of the hips'
SHOULDER_CHANGE_M = 0.05
LYING_CHEST_SHARE = 0.6
KNEE_BIAS = 0.15
# the sphinx's femur from the hip, degrees up from forward (a sit's: +40)
CROUCH_KNEE_UP_DEG = -15.0
# the head carried between the two clips' carriage: the share of the turn
# each neck joint takes (the head the rest)
NECK_CARRY = {"neck_base_jnt": 0.35, "neck_mid_jnt": 0.35}
BOW_SHARE = 0.6
REACH_SHARE = 0.98
REACH_PITCH_DEG = 30.0
REACH_SMOOTH_FRAMES = 1.5
REACH_HOLD_FRAMES = 2
# breathing, as the idle clip does it (build_template): breathe_jnt's scale
# swells by BREATH_SWELL along its own y once every BREATH_S or so (fitted
# to the clip's length so it loops), in every clip but the roll and the
# transitions (a breath fitted into 1.5 s heaved)
BREATH_SWELL = 0.03
BREATH_S = 4.0
NO_BREATH = ("rollover",) + tuple(TRANSITIONS)
TAIL_SMOOTH_FRAMES = 4
NECK_SMOOTH_FRAMES = 3
# rolling over, the head is turned out of the floor frame by frame about an
# axis that swings as the body turns: smoothed over 3 frames it still wobbled
# 7.5 deg a frame from side to side ("고개가 까딱"); over 6, 3.3, the head
# into the floor 2.8 cm for a few frames (over 9, 5.4)
ROLL_NECK_SMOOTH_FRAMES = 6
# and its legs, put down on the floor and turned up out of it frame by frame
# as the side they are on comes down, jumped 24 deg in a frame
ROLL_LEG_SMOOTH_FRAMES = 3
# lying poses rest on whatever is lowest, torso or legs
LYING = ("down", "side", "sphinx")
# held poses whose paws stay where the key pose put them (not the captured
# down: re-solved by IK its folded hind legs came out twisted 60 deg from
# the capture's, and lying, its paws barely move)
PINNED = {"stand": tuple(LEGS), "sit": tuple(LEGS)}


def build_clip(sk: Skeleton, src: Source, keys: dict[str, Pose], name: str) -> dict:
    source = SOURCE_CLIPS.get(name, name)
    times, frames, worlds = src.clip(source)
    segments = CLIPS[name]
    key_world = {k: sk.fk(p) for k, p in keys.items()}
    end = segments[-1].t1
    if times[-1] < end - 1e-6:
        # longer than the shiba's clip (a hold playing its motion again)
        times = np.arange(int(round(end * FPS)) + 1) / FPS
    times = times[times <= end + 1e-6]
    # the shake's offered paw: how high the shiba holds its own, as a share
    # of its height through the hold
    offer = None
    if name == "shake":
        h = src.paw_height(name, SHAKE_LEG_NAME)
        h = h - h[0]
        held = h[(times > 1.3) & (times < 4.2)] if len(h) == len(times) else h
        offer = h / max(float(np.median(held)), 1e-6)
    roll = src.roll(source) if any(isinstance(s, Roll) for s in segments) else None
    rolls = [s for s in segments if isinstance(s, Roll)]

    def tucked(t: float) -> float:
        """How far the legs are folded for a roll at time t."""
        out = 0.0
        for r in rolls:
            if r.t0 <= t <= r.t1:
                return 1.0
            out = max(out, float(smooth(np.array((t - (r.t0 - TUCK_LEAD_S)) / TUCK_LEAD_S))) if t < r.t0 else 0.0,
                      float(smooth(np.array(1.0 - (t - r.t1) / UNTUCK_S))) if t > r.t1 else 0.0)
        return out
    rotations = {i: [] for i in sk.joints}
    translations = []
    report = {"torso": [], "legs": [], "head": [], "tail": []}
    heads = sk.mask(sk.subtree(sk.j["neck_base_jnt"]))
    tails = sk.mask([sk.j[n] for n in TAIL if n in sk.j])
    legs_mask = sk.mask([j for leg in LEGS for j in sk.subtree(sk.leg(leg)[0])])
    roll_support = support(sk)
    head_mask = sk.mask(sk.subtree(sk.j["head_jnt"]))
    neck_mask = sk.mask([sk.j[n] for n in NECK if n in sk.j])
    # lying on its chest the dog is held up on its legs this far off the
    # floor; rolling, it rests on its torso, let down from this as the legs
    # fold (switched at once, the body dropped 5 cm in a frame and the
    # hind leg it had lain on swung 30 deg)
    gap_pose = keys["curl"].copy()
    rest_on(sk, gap_pose, roll_support | head_mask, neck_mask)
    roll_gap = max(sk.low(sk.fk(gap_pose), sk.torso)[0], 0.0)
    legs_in, phi = 0.0, 0.0
    for k, t in enumerate(times):
        seg = next(s for s in segments if s.t0 - 1e-6 <= t <= s.t1 + 1e-6)
        fold = tucked(float(t))
        lying = ((isinstance(seg, Hold) and seg.key in LYING)
                 or (isinstance(seg, Change) and seg.a in LYING and seg.b in LYING))
        if isinstance(seg, Hold):
            pose = keys[seg.key].copy()
            if fold > 0:
                pose = blend(sk, pose, keys["curl"], fold)
            # the shiba's frame: from the hold's start in its clip, or src_t0
            k_ref = int(round((seg.t0 if seg.src_t0 is None else seg.src_t0) * FPS))
            k_src = min(k_ref + k - int(round(seg.t0 * FPS)), len(frames) - 1)
            # eased in after and out before a change of pose (or of the
            # shiba's frames)
            weight = 1.0
            if seg.t0 > 0:
                weight = min(weight, (t - seg.t0) / SECONDARY_RAMP_S)
            # and before the clip's end, which loops back onto the key pose
            weight = min(weight, (seg.t1 - t) / (SECONDARY_RAMP_S if seg.t1 < end - 1e-6 else SECONDARY_END_S))
            secondary(sk, src, source, k_src, k_ref, float(np.clip(weight, 0.0, 1.0)), pose, seg.mirror,
                      HOLD_SECONDARY.get(name, SECONDARY))
            if fold > 0 or seg.key in LYING:
                # (one floor rule through a lying hold and the folding
                # before a roll: switching rules moved the body and the
                # tail laid on the floor jumped 40 deg)
                ground(sk, pose, roll_support)
            else:
                if seg.key != "stand":
                    ground(sk, pose)
                pin(sk, pose, key_world[seg.key], PINNED.get(seg.key, ()))
            if offer is not None:
                chain = sk.leg(SHAKE_LEG_NAME)[:4]
                for i in chain:
                    pose.r[i] = slerp_m(pose.r.get(i, sk.rest_r[i]), keys["shake"].r.get(i, sk.rest_r[i]),
                                        float(np.clip(offer[k_src], 0.0, 1.2)))
            if seg.key not in PINNED and fold == 0:
                lift_legs(sk, pose)
        elif isinstance(seg, Change):
            if seg.timed:
                u = float(smooth(np.array((t - seg.t0) / (seg.t1 - seg.t0))))
            else:
                u = src.progress(source, seg.t0, seg.t1)[k]
            pose = blend(sk, keys[seg.a], keys[seg.b], float(u))
            # (from one way of lying to another, on the floor as a lying hold)
            ground(sk, pose, roll_support if lying else None)
            lift_legs(sk, pose)
        else:
            # the shiba's roll, its ROLL_SRC_S stretched over the segment
            s0, s1 = ROLL_SRC_S
            s = s0 + (t - seg.t0) * (s1 - s0) / (seg.t1 - seg.t0)
            at = lambda sec: float(np.interp(sec * FPS, np.arange(len(roll)), roll))
            phi = float(np.clip(at(s) - at(s0), 0.0, 360.0))
            legs_in = float(smooth(np.array(min(phi, 360.0 - phi) / ROLL_FOLD_DEG)))
            pose = blend(sk, keys["curl"], keys["tuck"], legs_in)
            # the loose legs swung toward the side that is down, as it was
            # a moment before (gravity pulls them to the body's left lying
            # on its left side: +x, a turn about FWD by a positive angle)
            lag = ROLL_LEG_FLOP_LAG_S * (s1 - s0) / (seg.t1 - seg.t0)
            phi_then = float(np.clip(at(s - lag) - at(s0), 0.0, 360.0))
            flop = keys["curl"].side * ROLL_LEG_FLOP_DEG * np.sin(np.radians(phi_then)) * legs_in
            for leg in LEGS:
                root = sk.leg(leg)[0]
                turn(sk, pose, root, body_frame(sk, sk.fk(pose), leg_frame(sk, leg)) @ FWD, flop)
            # (a turn about FWD by a positive angle lifts the left side)
            turn(sk, pose, sk.cog, FWD, -keys["curl"].side * phi)
            # the body on the floor by the torso (and below, the head), a leg
            # under it turned up out of it: unfolding as the dog comes back
            # down, the legs pass straight down under it, and resting on them
            # too it stood up on them 15-25 cm
            ground(sk, pose, sk.torso)
            lift_head(sk, pose, limit=ROLL_HEAD_LIFT_DEG)
        lift_head(sk, pose)
        if isinstance(seg, Roll) or fold > 0 or lying:
            # the head is lifted by turning the neck at its base, and the
            # throat under the turn went into the floor (8 cm rolling over):
            # lying, the body rests on the neck and head too
            rest_on(sk, pose, (sk.torso if isinstance(seg, Roll) else roll_support) | head_mask, neck_mask)
        if isinstance(seg, Roll):
            shift(sk, pose, UP * roll_gap * (1.0 - legs_in))
            lift_legs(sk, pose)
        if not isinstance(seg, Roll) or min(phi, 360.0 - phi) < ROLL_TAIL_DRAPE_DEG:
            drape_tail(sk, pose)
        w = sk.fk(pose)
        report["torso"].append(sk.low(w, sk.torso)[0])
        report["legs"].append(sk.low(w, legs_mask)[0])
        report["head"].append(sk.low(w, heads)[0])
        report["tail"].append(sk.low(w, tails)[0])
        for i in sk.joints:
            rotations[i].append(Rotation.from_matrix(pose.r.get(i, sk.rest_r[i])).as_quat())
        translations.append(pose.t.copy())
    finish(sk, rotations, neck_frames=ROLL_NECK_SMOOTH_FRAMES if rolls else NECK_SMOOTH_FRAMES,
           leg_frames=ROLL_LEG_SMOOTH_FRAMES if rolls else 0)
    if name in LYING_WAG:
        lying_wag(sk, LYING_WAG[name], times, rolls, rotations, translations, report)
    return {"times": times, "rotations": rotations, "translations": translations, "report": report}


def lying_wag(sk: Skeleton, spec: dict, times: np.ndarray, rolls: list, rotations: dict, translations: list,
              report: dict) -> None:
    """A lying dog's wag (LYING_WAG): the tail as the clip has laid it,
    swept about the vertical - spread down its first joints, each a little
    behind the one before, a whole number of sweeps over the clip - still
    while it rolls. (On the tail after finish() smoothed it: smoothed with
    the wag, a 2 Hz sweep lost 40 % of its size.)"""
    tail = [sk.j[n] for n in TAIL if n in sk.j]
    tails = sk.mask(tail)
    end = float(times[-1])
    sweeps = max(1, round(spec["hz"] * end))
    for k, t in enumerate(times):
        away = min([max(r.t0 - t, t - r.t1) for r in rolls] or [np.inf])
        strength = float(smooth(np.array(away / spec["fade_s"]))) if away > 0 else 0.0
        if strength <= 0:
            continue
        pose = Pose({i: Rotation.from_quat(rotations[i][k]).as_matrix() for i in sk.joints}, np.asarray(translations[k]))
        phase = 2 * np.pi * sweeps * t / end
        for n, i in enumerate(tail[:len(spec["amps"])]):
            turn(sk, pose, i, UP, spec["amps"][n] * strength * np.sin(phase - n * spec["lag"]))
        for i in tail:
            q = Rotation.from_matrix(pose.r[i]).as_quat()
            rotations[i][k] = q if np.dot(q, rotations[i][k]) >= 0 else -q
        report["tail"][k] = sk.low(sk.fk(pose), tails)[0]


def finish(sk: Skeleton, rotations: dict, smooth_tail: bool = True, neck_frames: int = NECK_SMOOTH_FRAMES,
           leg_frames: int = 0, loop: bool = True) -> None:
    """Quaternions on one hemisphere frame to frame, and the tail smoothed
    over time: laid along the floor frame by frame, and left alone
    mid-roll, it jumped 40-90 deg where one met the other. (`loop`: the
    clip's ends smoothed into each other; else each into itself.)"""
    for qs in rotations.values():
        for k in range(1, len(qs)):
            if np.dot(qs[k], qs[k - 1]) < 0:
                qs[k] = -qs[k]
    # (the neck too: turned out of the floor frame by frame, it jumped where
    # the turn began)
    # (not a wagging tail: smoothed over 4 frames, a 2.4 Hz wag lost half its sweep)
    smoothed_joints = [(sk.j[n], TAIL_SMOOTH_FRAMES) for n in TAIL if n in sk.j and smooth_tail] + \
        [(sk.j[n], neck_frames) for n in ("neck_base_jnt", "neck_mid_jnt", "head_jnt") if n in sk.j] + \
        [(j, leg_frames) for leg in LEGS for j in sk.leg(leg)[:4] if leg_frames]
    for joint, width in smoothed_joints:
        qs = np.asarray(rotations[joint])
        kernel = np.exp(-0.5 * (np.arange(-width, width + 1) / (width / 2)) ** 2)
        padded = (np.concatenate([qs[-width:], qs, qs[:width]]) if loop          # the clip loops
                  else np.concatenate([np.repeat(qs[:1], width, 0), qs, np.repeat(qs[-1:], width, 0)]))
        smoothed = np.stack([np.convolve(padded[:, c], kernel / kernel.sum(), mode="valid") for c in range(4)], 1)
        rotations[joint] = list(smoothed / np.linalg.norm(smoothed, axis=1, keepdims=True))


def hold_paw(sk: Skeleton, pose: Pose, leg: str, toe: np.ndarray) -> None:
    """The leg's toe onto world point `toe` by two-bone IK on its upper
    bones, the knee / elbow bent as it is and the bones below the hock /
    wrist keeping their direction."""
    a, b, c, d, e = sk.leg(leg)
    w = sk.fk(pose)
    target = toe - (w[e][:3, 3] - w[c][:3, 3])
    line = target - w[a][:3, 3]
    bend = w[b][:3, 3] - w[a][:3, 3]
    pole = bend - line * (bend @ line) / max(line @ line, 1e-12)
    if np.linalg.norm(pole) < 1e-6:
        pole = FWD if leg.endswith("B") else -FWD
    plant(sk, pose, leg, target, lower_dirs(sk, w, leg), pole)


def build_video(sk: Skeleton, keys: dict[str, Pose], video: dict, name: str) -> dict:
    """A trick read off video (video_motion) on the template: each mapped
    joint turned in the world as the video's dog turned it since the
    stretch's first frame, on top of the template's key pose (the sit:
    the dog in the video sat with its paws down at the start), the rump
    on the floor and the hind paws where the key has them."""
    motion = video[name]
    base = keys[VIDEO_BASED[name]]
    base_world = sk.fk(base)
    turns_by_frame = {ours: motion[theirs] @ motion[theirs][0].T
                      for ours, theirs in video_motion.MAP.items() if theirs in motion}
    # the body's rocking kept to VIDEO_TRUNK_SHARE (the sit's forelegs stand
    # near straight, and leaning back with the video's dog lifted the planted
    # forepaw 5 cm; rolled 5 deg with it, the hips swung the pinned hind legs
    # about 35-50 deg - user, 11-04: "팔 드니까 뒷다리 이상해지고")
    for joint in VIDEO_TRUNK:
        if joint in turns_by_frame:
            rv = Rotation.from_matrix(turns_by_frame[joint]).as_rotvec() * VIDEO_TRUNK_SHARE
            turns_by_frame[joint] = Rotation.from_rotvec(rv).as_matrix()
    names = {i: (sk.rig.gltf.nodes[i].name or "").split(".")[0] for i in sk.order if i in sk.joints}
    order = [i for i in sk.order if i in names and names[i] in turns_by_frame]
    frames = len(next(iter(motion.values())))
    rotations = {i: [] for i in sk.joints}
    translations = []
    report = {"torso": [], "legs": [], "head": [], "tail": []}
    heads = sk.mask(sk.subtree(sk.j["neck_base_jnt"]))
    tails = sk.mask([sk.j[n] for n in TAIL if n in sk.j])
    legs_mask = sk.mask([j for leg in LEGS for j in sk.subtree(sk.leg(leg)[0])])

    def follow(k: int, joints=order) -> Pose:
        pose = base.copy()
        for i in joints:
            set_world(sk, pose, sk.fk(pose), i, turns_by_frame[names[i]][k] @ rot_of(base_world[i]))
        return pose

    # the offered paw: the video gives when it goes up and down - the front
    # paw that rises most, its height as a share of its usual top - and
    # the leg takes the shape of OFFER_LEG that share of the way from the
    # key's. (Followed bone by bone, the video's leg - noisy, and built
    # unlike the template's - reached forward 15 cm off the floor with the
    # wrist hanging, a step more than a paw given: user, 11-04: "앞발이
    # 자연스럽지가 않은데".) The other forepaw stays where the key has it.
    offered, rise = None, np.zeros(frames)
    if name in OFFER_LEG:
        heights = np.array([[sk.fk(p)[sk.leg(leg)[-1]][1, 3] for leg in ("LF", "RF")]
                            for p in (follow(k) for k in range(frames))])
        heights -= heights[0]
        offered = ("LF", "RF")[int(np.argmax(np.percentile(heights, 95, axis=0)))]
        top = max(float(np.percentile(heights[:, ("LF", "RF").index(offered)], 95)), 1e-3)
        rise = np.clip(heights[:, ("LF", "RF").index(offered)] / top, 0.0, 1.0)
        # (smoothed over OFFER_SMOOTH_FRAMES: the video's paw wobbles 2-3 cm
        # three times a second, and the raised paw trembled - user, 11-04:
        # "손이 뭔가 바들바들거리는데"; a dog holds the paw out still)
        from scipy.ndimage import gaussian_filter1d
        rise = gaussian_filter1d(rise, OFFER_SMOOTH_FRAMES, mode="wrap")
        rise = rise / max(float(rise.max()), 1e-6)
        # then up quickly where it passes half way, held out with a slow
        # sway, down again (taken as it came, the paw crept up over 1.5 s:
        # "너무 강제로 드는 느낌")
        held = np.where(rise >= 0.5)[0]
        if len(held):
            t = np.arange(frames, dtype=float)
            up, down = OFFER_RAMP_S[0] * FPS, OFFER_RAMP_S[1] * FPS
            rise = smooth((t - held[0] + up / 2) / up) * (1.0 - smooth((t - held[-1] + down / 2) / down))
            rise *= 1.0 - OFFER_SWAY * (0.5 - 0.5 * np.cos(2 * np.pi * OFFER_SWAY_HZ * (t - held[0]) / FPS))
        offer = base.copy()
        shape_leg(sk, offer, offered, OFFER_LEG[name])
    planted = ("LB", "RB") + tuple(leg for leg in ("LF", "RF") if offered and leg != offered)
    # the legs that stay down and the shoulder blades keep the key's shape
    # (given the video's turns and pinned back by IK, the hind legs - folded
    # tight - swung 20-50 deg at the knee as the hips barely moved: "팔
    # 드니까 뒷다리 이상해지고"; the blades turned with the video's dragged
    # the breast along with the raised leg: "앞다리랑 가슴이랑 붙어
    # 움직이는 느낌")
    keep = {i for leg in planted for i in sk.leg(leg)[:4]} if offered else set()
    if offered:
        keep |= {sk.j[n] for n in ("L_clavicle_jnt", "R_clavicle_jnt") if n in sk.j}
        # and the tail lies as the sit has it (the video's twitched it 4-10
        # deg a frame)
        keep |= {sk.j[n] for n in TAIL if n in sk.j}
    moving = [i for i in order if i not in keep]
    front_down = tuple(leg for leg in planted if leg.endswith("F")) if offered else ("LB", "RB")
    blade = sk.j.get(f"{offered[0]}_clavicle_jnt") if offered else None
    tail = [sk.j[n] for n in TAIL if n in sk.j]
    wag = WAG.get(name)
    if wag and tail:
        # the tail laid straight back along the floor, and held just clear of
        # it so that sweeping it about the vertical never takes it in (laid
        # on the floor frame by frame as it swept, its outer joints flipped
        # 15-75 deg from frame to frame, the lowest hair changing as it went)
        # (let down from the tail's own rest, not from the sit's tail swept
        # round the haunch - turned back from there it lay rolled over on
        # itself - nor aimed level from the root, which tipped down with the
        # pelvis: bent flat there it kinked into a knot; user, 11-04: "꼬리가
        # 꾸불한데")
        laid = base.copy()
        for i in tail:
            laid.r.pop(i, None)
        drape_tail(sk, laid, settle=True)
        tails_mask = sk.mask(tail)
        for _ in range(12):
            if sk.low(sk.fk(laid), tails_mask)[0] >= WAG_CLEAR_M:
                break
            turn(sk, laid, tail[0], np.cross(UP, -FWD), -1.0)       # the root a degree up
        sweeps = max(1, round(wag[2] * (frames - 1) / FPS))
    for k in range(frames):
        pose = follow(k, moving)
        if offered:
            # the shoulder blade swings forward a little with the leg
            if blade is not None and OFFER_BLADE_DEG:
                turn(sk, pose, blade, LEFT, -OFFER_BLADE_DEG * float(rise[k]))
            for i in sk.leg(offered)[:4]:
                pose.r[i] = slerp_m(base.r.get(i, sk.rest_r[i]), offer.r.get(i, sk.rest_r[i]), float(rise[k]))
        if wag and tail:
            for i in tail:
                pose.r[i] = laid.r.get(i, sk.rest_r[i]).copy()
            phase = 2 * np.pi * sweeps * k / (frames - 1)
            for n, i in enumerate(tail[:len(wag[0])]):
                turn(sk, pose, i, UP, wag[0][n] * np.sin(phase - n * wag[3]))
        ground(sk, pose)
        pin(sk, pose, base_world, front_down)
        # (only the free legs turned out of the floor: lifted too, a pinned
        # hind leg the rolling hips dipped into it swung up 40 deg)
        for leg in LEGS:
            if leg not in planted:
                lift(sk, pose, sk.leg(leg)[0], lower_leg(sk, leg), slack=LEG_SLACK_M)
        lift_head(sk, pose)
        if not wag:
            drape_tail(sk, pose)
        w = sk.fk(pose)
        report["torso"].append(sk.low(w, sk.torso)[0])
        report["legs"].append(sk.low(w, legs_mask)[0])
        report["head"].append(sk.low(w, heads)[0])
        report["tail"].append(sk.low(w, tails)[0])
        for i in sk.joints:
            rotations[i].append(Rotation.from_matrix(pose.r.get(i, sk.rest_r[i])).as_quat())
        translations.append(pose.t.copy())
    finish(sk, rotations, smooth_tail=not wag)
    return {"times": np.arange(frames) / FPS, "rotations": rotations, "translations": translations, "report": report}


def build_stand_wag(sk: Skeleton, keys: dict[str, Pose], name: str) -> dict:
    """A happy dog standing and wagging (STAND_WAG): the stand, the tail
    carried up and swept side to side about the vertical - spread down its
    first joints so it bows as it goes, each a little behind the one
    before - the rump swaying the other way on hind legs that stay planted,
    the sweep coming and going a little over the loop."""
    spec = STAND_WAG[name]
    base = keys["stand"].copy()
    tail = [sk.j[n] for n in TAIL if n in sk.j]
    pelvis = sk.j["pelvis_jnt"]
    turn(sk, base, tail[0], LEFT, spec["lift"])
    base_world = sk.fk(base)
    frames = int(round(spec["length"] * FPS)) + 1
    sweeps = max(1, round(spec["hz"] * spec["length"]))
    rotations = {i: [] for i in sk.joints}
    translations = []
    report = {"torso": [], "legs": [], "head": [], "tail": []}
    heads = sk.mask(sk.subtree(sk.j["neck_base_jnt"]))
    tails = sk.mask(tail)
    legs_mask = sk.mask([j for leg in LEGS for j in sk.subtree(sk.leg(leg)[0])])
    for k in range(frames):
        u = k / (frames - 1)
        phase = 2 * np.pi * sweeps * u
        strength = 1.0 - spec["vary"] * (0.5 - 0.5 * np.cos(2 * np.pi * u))
        pose = base.copy()
        turn(sk, pose, pelvis, UP, -spec["hips"] * strength * np.sin(phase))
        for n, i in enumerate(tail[:len(spec["amps"])]):
            turn(sk, pose, i, UP, spec["amps"][n] * strength * np.sin(phase - n * spec["lag"]))
        pin(sk, pose, base_world, ("LB", "RB"))
        w = sk.fk(pose)
        report["torso"].append(sk.low(w, sk.torso)[0])
        report["legs"].append(sk.low(w, legs_mask)[0])
        report["head"].append(sk.low(w, heads)[0])
        report["tail"].append(sk.low(w, tails)[0])
        for i in sk.joints:
            rotations[i].append(Rotation.from_matrix(pose.r.get(i, sk.rest_r[i])).as_quat())
        translations.append(pose.t.copy())
    finish(sk, rotations, smooth_tail=False)
    return {"times": np.arange(frames) / FPS, "rotations": rotations, "translations": translations, "report": report}


def pulse(t: float, start: float, up: float, hold: float, down: float) -> float:
    """0 -> 1 over `up` s from `start`, held `hold` s, -> 0 over `down` s."""
    u = t - start
    if u < 0 or u > up + hold + down:
        return 0.0
    return float(min(smooth(np.array(u / up)), 1.0 - smooth(np.array((u - up - hold) / down))))


def build_bark(sk: Skeleton, keys: dict[str, Pose], name: str) -> dict:
    """A dog barking on its stand (BARK): a bout of barks and a pause, each
    bark the jaw dropping open about its hinge and shutting, gathered for
    with a breath in, the body drawn back and the head lowered, thrown with
    the head and neck tossed forward and up and the chest forward onto the
    braced forelegs, the belly drawn in, then settling; the ears prick and
    flick back, the
    barks of a bout differ a little; all four paws stay where the stand has
    them, the tail wags."""
    spec = BARK[name]
    base = keys["stand"].copy()
    head, jaw = sk.j["head_jnt"], sk.j["jaw_jnt"]
    neck_base = sk.j["neck_base_jnt"]
    tail = [sk.j[n] for n in TAIL if n in sk.j]
    ears = [sk.j[n] for n in ("L_ear_base_jnt", "R_ear_base_jnt") if n in sk.j]
    turn(sk, base, tail[0], LEFT, spec["tail_lift"])
    for i in ears:
        turn(sk, base, i, LEFT, spec["ears"])
    turn(sk, base, head, LEFT, -spec["head_up"])
    base_world = sk.fk(base)
    fps = spec.get("fps", FPS)
    frames = int(round(spec["length"] * fps)) + 1
    times = np.arange(frames) / fps
    sweeps = max(1, round(spec["tail_hz"] * spec["length"]))
    jaw_shape = (spec["open_s"], spec["hold_s"], spec["close_s"])
    rotations = {i: [] for i in sk.joints}
    translations, exhale = [], []
    report = {"torso": [], "legs": [], "head": [], "tail": []}
    heads = sk.mask(sk.subtree(sk.j["neck_base_jnt"]))
    tails = sk.mask(tail)
    legs_mask = sk.mask([j for leg in LEGS for j in sk.subtree(sk.leg(leg)[0])])
    barks = list(zip(spec["barks"], spec["strength"], spec["gape"], spec["look"]))
    look_up = spec["look_s"][0]

    def pulses(t: float, shape: tuple) -> float:
        """The bout's pulses of `shape` (start from the bark, up, hold,
        down) at t, each by its bark's strength."""
        return sum(s * pulse(t, t0 + shape[0], *shape[1:]) for t0, s, _, _ in barks)

    for k, t in enumerate(times):
        gape = max(g * pulse(t, t0, *jaw_shape) for t0, _, g, _ in barks)
        flick = max(pulse(t, t0, *jaw_shape) for t0, _, _, _ in barks)
        gather, throw, settle = (pulses(t, spec[key]) for key in ("gather_s", "throw_s", "settle_s"))
        look = sum(d * pulse(t, t0 - look_up, *spec["look_s"]) for t0, _, _, d in barks)
        # between the barks of a bout the mouth stays a little open
        bout = pulse(t, spec["barks"][0] + spec["open_s"], 0.05,
                     spec["barks"][-1] - spec["barks"][0] - spec["open_s"] - 0.05,
                     spec["open_s"]) if len(spec["barks"]) > 1 else 0.0
        pose = base.copy()
        pose.t = base.t + FWD * (spec["thrust"] * throw - spec["draw_back"] * gather) \
            - UP * (spec["dip"] * throw + spec["sink"] * settle)
        # the neck: lowered, then lifted (+ turns the nose down)
        turn(sk, pose, neck_base, LEFT, spec["lower"] * gather - spec["lift"] * throw)
        # the muzzle: tucked, tossed up, nodding as it settles
        tip = spec["tuck"] * gather - spec["toss"] * throw + spec["nod"] * settle
        for joint, share in NECK_SHARES.items():
            turn(sk, pose, sk.j[joint], LEFT, tip * share)
        if look:
            for joint, share in NECK_SHARES.items():
                turn(sk, pose, sk.j[joint], UP, look * share)
        for i in ears:
            turn(sk, pose, i, LEFT, spec["prick"] * gather - spec["flick"] * flick)
        phase = 2 * np.pi * sweeps * k / (frames - 1)
        for n, i in enumerate(tail[:len(spec["tail_amps"])]):
            turn(sk, pose, i, UP, spec["tail_amps"][n] * np.sin(phase - n * spec["tail_lag"]))
        w = sk.fk(pose)
        across = rot_of(w[head]) @ rot_of(sk.rest[head]).T @ LEFT
        turn(sk, pose, jaw, across, max(gape, spec["between"] * bout))
        pin(sk, pose, base_world)
        w = sk.fk(pose)
        report["torso"].append(sk.low(w, sk.torso)[0])
        report["legs"].append(sk.low(w, legs_mask)[0])
        report["head"].append(sk.low(w, heads)[0])
        report["tail"].append(sk.low(w, tails)[0])
        for i in sk.joints:
            rotations[i].append(Rotation.from_matrix(pose.r.get(i, sk.rest_r[i])).as_quat())
        translations.append(pose.t.copy())
        exhale.append(1.0 + spec["inhale"] * gather - spec["exhale"] * throw)
    # the neck smoothed over as long a time as at FPS
    finish(sk, rotations, smooth_tail=False, neck_frames=int(round(NECK_SMOOTH_FRAMES * fps / FPS)))
    breathe = breathing(times)
    breathe[:, 1] *= np.array(exhale)
    return {"times": times, "rotations": rotations, "translations": translations, "report": report,
            "breathe": breathe}


def paw_down_shares(heights: np.ndarray, lying: np.ndarray) -> np.ndarray:
    """(F, legs) how far to let each paw down (settle_paws): as near the
    floor as the captured paw is (PAW_DOWN_M), while lying, eased over
    PAW_DOWN_EASE_FRAMES (switched off as the dog rose, a hind leg swung 40
    deg in a frame)."""
    share = np.clip((PAW_DOWN_M[1] - heights) / (PAW_DOWN_M[1] - PAW_DOWN_M[0]), 0.0, 1.0) * lying[:, None]
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(share, PAW_DOWN_EASE_FRAMES, axis=0, mode="wrap")


def settle_paws(sk: Skeleton, pose: Pose, legs, shares: np.ndarray) -> None:
    """Each leg let down onto the floor `shares` of the way (paw_down_shares):
    at the hip / shoulder by its lowest part below the knee / elbow, then at
    the knee / elbow by the paw. Its joint turns put on the template's
    longer legs and thicker body, the lying dog's paws hung 10-20 cm over
    the floor where the captured dog's lay on it."""
    for leg, share in zip(legs, shares):
        share = float(share)
        if share <= 1e-3:
            continue
        chain = sk.leg(leg)
        down = pose.copy()
        lift(sk, down, chain[0], lower_leg(sk, leg), settle=True, limit=PAW_DOWN_LIMIT_DEG)
        lift(sk, down, chain[1], sk.mask(sk.subtree(chain[2])), settle=True, limit=PAW_DOWN_LIMIT_DEG)
        for i in chain[:2]:
            pose.r[i] = slerp_m(pose.r.get(i, sk.rest_r[i]), down.r.get(i, sk.rest_r[i]), share)


def cleaned(mask: np.ndarray, gap: int = CONTACT_GAP_FRAMES, least: int = CONTACT_MIN_FRAMES) -> np.ndarray:
    """A looping (F, paws) mask with each paw's gaps of up to `gap` frames
    filled and its runs shorter than `least` frames dropped."""
    F = len(mask)
    out = np.zeros_like(mask)
    for n in range(mask.shape[1]):
        m = np.tile(mask[:, n], 3)
        for a, b in runs(~m):
            if 0 < a and b < len(m) and b - a <= gap:
                m[a:b] = True
        for a, b in runs(m):
            if b - a < least:
                m[a:b] = False
        out[:, n] = m[F:2 * F]
    return out


def within_reach(sk: Skeleton, w: dict, leg: str, toe: np.ndarray) -> np.ndarray:
    """`toe` (on the floor) moved along the floor toward the hip / shoulder
    until the leg reaches it (REACH_SHARE of its two upper bones from the
    hip / shoulder to the hock / wrist), the bones below as they are."""
    a, b, c, _, e = sk.leg(leg)
    lower = w[e][:3, 3] - w[c][:3, 3]
    root = w[a][:3, 3]
    reach = REACH_SHARE * (sk.length(a, b) + sk.length(b, c))
    d = toe - lower - root
    if np.linalg.norm(d) <= reach:
        return toe
    level = d - UP * (d @ UP)
    across = float(np.sqrt(max(reach * reach - (d @ UP) ** 2, 0.0)))
    return root + UP * (d @ UP) + level / max(np.linalg.norm(level), 1e-9) * across + lower


def body_roll(sk: Skeleton, w: dict) -> float:
    """How far the body lies over onto its side (degrees, + onto its right)."""
    across = body_frame(sk, w, sk.cog) @ LEFT
    return float(np.degrees(np.arcsin(np.clip(across[1], -1.0, 1.0))))


def own_side(sk: Skeleton, w: dict, leg: str, toe: np.ndarray, keep: float = STANCE_KEEP,
             share: tuple = STANCE_SHARE) -> np.ndarray:
    """`toe` moved across the body (along the floor) toward the rest stance's
    half width from the line of the body - `keep` of the way it is off that
    kept - and on its own side, `share` of that half width (the captured dog
    stood with both forepaws to its left, one at 1.8 times the stance and the
    other at 0.6)."""
    hips = np.mean([w[sk.j[n]][:3, 3] for n in ("L_hip_jnt", "R_hip_jnt")], axis=0)
    chest = w[sk.j["chest_jnt"]][:3, 3]
    ahead = chest - hips
    ahead = ahead - UP * (ahead @ UP)
    across = np.cross(UP, ahead / max(np.linalg.norm(ahead), 1e-9))       # toward the dog's left
    side = 1.0 if leg.startswith("L") else -1.0
    half = abs(sk.rest[sk.leg(leg)[-1]][0, 3])
    off = side * float((toe - 0.5 * (hips + chest)) @ across)
    want = float(np.clip(half + keep * (off - half), share[0] * half, share[1] * half))
    return toe + side * (want - off) * across


def build_captured(sk: Skeleton, poses: dict, captured_clips: dict, name: str) -> dict:
    """A real dog's motion (dog_mocap) on the template, frame by frame:
    each mapped joint turned as the dog's turned from its standing pose,
    the body carried across the floor as the dog's hips (scaled to the
    template's height), resting on its lowest point, the paws the dog has
    planted held where they came down."""
    clip = captured_clips[name]
    turns_by_frame = {ours: clip["rot"][theirs] @ poses["stand"][theirs].T
                      for ours, theirs in dog_mocap.MAP.items() if theirs in clip["rot"]}
    order = [i for i in sk.order if i in sk.joints
             and (sk.rig.gltf.nodes[i].name or "").split(".")[0] in turns_by_frame]
    names = {i: (sk.rig.gltf.nodes[i].name or "").split(".")[0] for i in order}
    scale = sk.rest[sk.cog][1, 3] / captured_clips["stand_hips_height"]
    hips, planted = clip["hips"], clip["planted"]
    upright = hips[:, 1] > PLANT_UPRIGHT_SHARE * captured_clips["stand_hips_height"]
    legs = ("LF", "RF", "LB", "RB")                      # dog_mocap.FEET's order
    # a paw held while the captured dog has it on the floor, still or sliding
    # (held only while still, the hold let go as the captured paw slid along
    # the floor getting up, and the template's longer leg rose 5 cm off it -
    # user, 11-06: "왼발이 내려가다가 튀는느낌"); and how firmly, eased in and
    # out over twice HOLD_EASE_FRAMES: switched on and off at once, a paw that
    # had drifted from where the template's own leg would put it swung the
    # leg 50-80 deg in a frame
    contact = cleaned(clip["feet"][:, :, 1] < CONTACT_M)
    hold = (contact & upright[:, None]).astype(float)
    toe_rest_y = {leg: sk.rest[sk.leg(leg)[-1]][1, 3] for leg in legs}
    held: dict[str, np.ndarray] = {}
    floor = support(sk)
    necks = sk.mask([sk.j[n] for n in NECK if n in sk.j])
    down = paw_down_shares(clip["feet"][:, :, 1], ~upright)
    rotations = {i: [] for i in sk.joints}
    translations = []
    report = {"torso": [], "legs": [], "head": [], "tail": []}
    heads = sk.mask(sk.subtree(sk.j["neck_base_jnt"]))
    tails = sk.mask([sk.j[n] for n in TAIL if n in sk.j])
    legs_mask = sk.mask([j for leg in LEGS for j in sk.subtree(sk.leg(leg)[0])])
    chest, chest_rest_y = sk.j["chest_jnt"], sk.rest[sk.j["chest_jnt"]][1, 3]
    hindmost = sk.torso | sk.mask([j for leg in ("LB", "RB") for j in sk.subtree(sk.leg(leg)[1])])

    def base(k: int, bones: dict, upright: float | None = None) -> Pose:
        """The captured dog's pose at frame k on the template, on the floor;
        its hips squared `upright` of the way (pose.upright; by default as
        far as the frame's own lean asks)."""
        pose = sk.rest_pose.copy()
        for i in order:
            set_world(sk, pose, sk.fk(pose), i, turns_by_frame[names[i]][k] @ rot_of(sk.rest[i]))
        # the back laid even and the legs bent as legs bend, as the down
        # (bent up to 70 and 55 deg at two joints of the back, the leg bones
        # turned up to 107 deg about their length)
        even_back(sk, pose)
        # the hips in line behind the chest, not swung aside or wrung, until
        # the dog goes over onto its side (CAPTURED_LEVEL_DEG, as the body is
        # levelled, and smoothed over time as that is): the captured dog stood
        # and lay down with its rump swung up to 18 deg to its left and the
        # hips wrung up to 12 against the chest (user, 11-06: "죽은척 초반에
        # 뒷다리 그냥 골반이 왼쪽으로 틀어져있는것같은데"); going over, the
        # hips go first, as they did
        if upright is None:
            lo, hi = CAPTURED_LEVEL_DEG
            upright = float(np.clip((hi - abs(body_roll(sk, sk.fk(pose)))) / (hi - lo), 0.0, 1.0))
        square_hips(sk, pose, upright, upright)
        pose.upright = upright
        for leg in LEGS:
            bones[leg] = align_leg(sk, pose, leg, bones.get(leg))
            # (and the hock / wrist and paw bending as hinges: lying on its
            # left side, the left hind heel stuck out sideways and the foot
            # turned in under the belly - hinge_leg)
            hinge_leg(sk, pose, leg)
            limit_hinges(sk, pose, leg)
        even_neck(sk, pose)
        w = sk.fk(pose)
        at = w[sk.cog][:3, 3]
        shift(sk, pose, np.array([hips[k, 0] * scale - at[0], 0.0, hips[k, 2] * scale - at[2]]))
        ground(sk, pose, floor)
        return pose

    def on_floor_tipped(pose: Pose, deg: float) -> None:
        """Rest the body on the floor - tipped forward `deg`, by all but the
        forelegs, which the hold puts on it, more so the further it tips
        (CAPTURED_LEAN_GROUND_DEG): by the forelegs as the capture swings
        them, the tipped body was lifted back up and the forepaws were no
        nearer the floor; switched from one to the other at once, the body
        dropped 4-5 cm in a frame."""
        w = sk.fk(pose)
        share = float(np.clip(deg / CAPTURED_LEAN_GROUND_DEG, 0.0, 1.0))
        y = (1.0 - share) * sk.low(w, floor)[0] + share * sk.low(w, hindmost)[0]
        shift(sk, pose, np.array([0.0, -y, 0.0]))

    def levelled(pose: Pose, deg: float) -> Pose:
        """`pose` turned `deg` about the body's length, back on the floor."""
        if abs(deg) > 1e-6:
            ahead = body_frame(sk, sk.fk(pose), sk.cog) @ FWD
            turn(sk, pose, sk.cog, ahead - UP * (ahead @ UP), deg)
            ground(sk, pose, floor)
        return pose

    # the body level side to side until it lies on its side (CAPTURED_LEVEL_DEG),
    # and while its chest is up, tipped forward (CAPTURED_LEAN_MAX_DEG at most:
    # more and the sit became a crouch) until the forepaws the captured dog
    # has down reach the floor - the captured dog stood and sat leaning 10-15
    # deg onto its right side, and sitting down and getting up held its
    # chest higher than the template's long back lets its forepaws reach the
    # floor: they hung 8-15 cm up (user, 11-06: "죽은척할때도 다리가 살짝
    # 부자연스럽게 비틀리는데")
    level, lean, bones0 = np.zeros(len(hips)), np.zeros(len(hips)), {}
    squared = np.zeros(len(hips))                        # how far the hips are squared (base)
    # and how firmly each paw is put on its own side of the body: less as the
    # body goes over onto its side, not at all lying on it; a hind paw also
    # less as the chest comes down (lying on one hip a dog has both hind legs
    # out to one side) - eased over time (switched on at once as the dog came
    # up off its side, a foreleg swung 110-146 deg in a frame)
    fore, hind, flat = np.zeros(len(hips)), np.zeros(len(hips)), np.zeros(len(hips))
    under = np.zeros(len(hips))                          # +1: lying on its left side, -1 on its right
    lo, hi = CAPTURED_LEVEL_DEG
    for k in range(len(hips)):
        pose = base(k, bones0)
        squared[k] = pose.upright
        roll = body_roll(sk, sk.fk(pose))
        level[k] = -roll * float(np.clip((hi - abs(roll)) / (hi - lo), 0.0, 1.0))
        pose = levelled(pose, level[k])
        w = sk.fk(pose)
        roll = body_roll(sk, w)
        fore[k] = float(np.clip((hi - abs(roll)) / (hi - lo), 0.0, 1.0))
        flat[k] = float(np.clip((abs(roll) - SIDE_LYING_DEG[0]) / (SIDE_LYING_DEG[1] - SIDE_LYING_DEG[0]), 0.0, 1.0))
        under[k] = 1.0 if roll < 0 else -1.0
        hind[k] = fore[k] * float(np.clip((w[chest][1, 3] / chest_rest_y - LYING_CHEST_SHARE) / 0.1, 0.0, 1.0))
        if abs(body_roll(sk, w)) < CAPTURED_LEVEL_DEG[1] and w[chest][1, 3] > LYING_CHEST_SHARE * chest_rest_y:
            goals = []
            for n, leg in enumerate(legs[:2]):
                if clip["feet"][k, n, 1] < PAW_DOWN_M[0]:
                    c, e = sk.leg(leg)[2], sk.leg(leg)[-1]
                    toe = w[e][:3, 3]
                    goals.append((leg, np.array([toe[0], toe_rest_y[leg], toe[2]]), w[e][:3, 3] - w[c][:3, 3]))
            need = pitch_to_reach(sk, pose, goals)
            if goals:
                # and the shoulders no higher than a straight foreleg stands
                # them, a little lower while the hips are down
                # (CAPTURED_SHOULDER_SHARE, from the hips' share of their
                # standing height CAPTURED_LEAN_HIPS): sitting down and
                # getting up the template's shoulders rose 4-6 cm over that,
                # and at that height the forelegs still stood straight, their
                # paws 2-4 cm up
                low = float(np.clip((CAPTURED_LEAN_HIPS[1] - hips[k, 1] / captured_clips["stand_hips_height"])
                                    / (CAPTURED_LEAN_HIPS[1] - CAPTURED_LEAN_HIPS[0]), 0.0, 1.0))
                share = CAPTURED_SHOULDER_SHARE[1] + low * (CAPTURED_SHOULDER_SHARE[0] - CAPTURED_SHOULDER_SHARE[1])
                top = share * sk.rest[sk.j["L_shoulder_jnt"]][1, 3]
                for deg in np.arange(max(need, 0.0), CAPTURED_LEAN_MAX_DEG + 1e-6, 1.0):
                    tipped = pitched(sk, pose, float(deg))
                    on_floor_tipped(tipped, float(deg))
                    wp = sk.fk(tipped)
                    if max(wp[sk.j[nm]][1, 3] for nm in ("L_shoulder_jnt", "R_shoulder_jnt")) <= top:
                        need = float(deg)
                        break
                else:
                    need = CAPTURED_LEAN_MAX_DEG
            lean[k] = min(need, CAPTURED_LEAN_MAX_DEG)
    from scipy.ndimage import gaussian_filter1d
    level = gaussian_filter1d(level, CAPTURED_SMOOTH_FRAMES, mode="wrap")
    squared = gaussian_filter1d(squared, CAPTURED_SMOOTH_FRAMES, mode="wrap")
    lean = gaussian_filter1d(lean, CAPTURED_SMOOTH_FRAMES, mode="wrap")
    fore = gaussian_filter1d(fore, 2 * CAPTURED_SMOOTH_FRAMES, mode="wrap")
    hind = gaussian_filter1d(hind, 2 * CAPTURED_SMOOTH_FRAMES, mode="wrap")
    flat = gaussian_filter1d(flat, 2 * CAPTURED_SMOOTH_FRAMES, mode="wrap")
    # a paw the dog has down held, too, while it sits down, sits and gets up
    # - a forepaw unless the dog lies on its side, a hind paw while the chest
    # is up - not only while it stands: let go, the left hind paw touched
    # down sitting and rose 4-5 cm off the floor again at once, and the left
    # forepaw hung 5-6 cm up getting up (user, 11-06: "왼발이 내려가다가
    # 튀는느낌")
    # how flat each paw is laid (flat_paw): as near the floor as the captured
    # paw is (PAW_FLAT_CAPTURE_M), not lying on its side, eased over time
    # (by the template's own paw height frame by frame, it came and went as
    # the paw stepped and the leg jerked)
    near = np.clip((PAW_FLAT_CAPTURE_M[1] - clip["feet"][:, :, 1]) / (PAW_FLAT_CAPTURE_M[1] - PAW_FLAT_CAPTURE_M[0]),
                   0.0, 1.0) * (1.0 - flat[:, None])
    flatness = np.clip(gaussian_filter1d(near, 2 * CAPTURED_SMOOTH_FRAMES, axis=0, mode="wrap"), 0.0, 1.0)
    hold[:, :2] = np.maximum(hold[:, :2], contact[:, :2] & (fore[:, None] > 0.5))
    hold[:, 2:] = np.maximum(hold[:, 2:], contact[:, 2:] & (hind[:, None] > 0.5))
    box = np.ones(HOLD_EASE_FRAMES) / HOLD_EASE_FRAMES
    for _ in range(2):                              # (twice: eased in and out, not at a steady rate)
        hold = np.stack([np.convolve(np.r_[hold[-HOLD_EASE_FRAMES:, n], hold[:, n], hold[:HOLD_EASE_FRAMES, n]], box,
                                     mode="same")[HOLD_EASE_FRAMES:-HOLD_EASE_FRAMES] for n in range(len(legs))], 1)
    bones: dict[str, list] = {}
    follow = {leg: np.zeros(3) for leg in legs}
    spots: list[dict] = [{} for _ in range(len(hips))]           # where each held paw is held, frame by frame
    head = sk.j["head_jnt"]
    for k in range(len(hips)):
        pose = levelled(base(k, bones, float(squared[k])), level[k])
        looking = rot_of(sk.fk(pose)[head])
        pose = pitched(sk, pose, lean[k])
        if lean[k] > 0.5:
            # (the head carried as the dog carries it, not tipped down with the
            # body toward the floor)
            carry_head(sk, pose, looking)
        on_floor_tipped(pose, float(lean[k]))
        firmly = {"LF": fore[k], "RF": fore[k], "LB": hind[k], "RB": hind[k]}
        for n, leg in enumerate(legs):
            if hold[k, n] < 1e-3:
                held.pop(leg, None)
                continue
            if leg not in held:
                w = sk.fk(pose)
                toe = w[sk.leg(leg)[-1]][:3, 3]
                held[leg] = np.array([toe[0], toe_rest_y[leg], toe[2]])
                held[leg] = held[leg] + firmly[leg] * (own_side(sk, w, leg, held[leg]) - held[leg])
            elif not planted[k, n]:
                # (the captured paw sliding along the floor, or yet to come
                # down or just up as the hold eases in or out: the spot
                # moved with it)
                step = (clip["feet"][k, n] - clip["feet"][k - 1, n]) * scale
                held[leg] = held[leg] + np.array([step[0], 0.0, step[2]])
            # the spot not across the line of the body, frame by frame
            # (standing, the captured dog slid its right hind paw 7 cm over
            # under it; moved there for good, a paw the capture then slid out
            # went on out to 18 cm)
            w = sk.fk(pose)
            inside = own_side(sk, w, leg, held[leg], keep=1.0, share=(STANCE_SHARE[0], np.inf))
            spot = held[leg] + firmly[leg] * (inside - held[leg])
            # (a spot out of the leg's reach - the template's long back holds
            # its shoulders higher than the captured dog's sitting down and
            # getting up - drawn in along the floor until the leg reaches it:
            # held out there, the leg reached for it straight and lifted the
            # paw 2-3 cm, and let go it dropped back - user, 11-06: "왼발이
            # 내려가다가 튀는느낌")
            spot = within_reach(sk, w, leg, spot)
            firm = pose.copy()
            hold_paw(sk, firm, leg, spot)
            for i in sk.leg(leg)[:4]:
                pose.r[i] = slerp_m(pose.r.get(i, sk.rest_r[i]), firm.r.get(i, sk.rest_r[i]), float(hold[k, n]))
            spots[k][leg] = spot
        lift_head_level(sk, pose)
        if max(fore[k], hind[k]) > 0.5:
            # not lying on its side: a free leg the tipped body put into the
            # floor lifted out of it at its hip / shoulder, and the body only
            # raised, by its torso and neck (rested on all its lower legs,
            # the body rose 3-6 cm over a foreleg stepping under it as it got
            # up, and the held forepaw with it - user, 11-06: "왼발이
            # 내려가다가 튀는느낌")
            for n, leg in enumerate(legs):
                if hold[k, n] < 0.5:
                    lift(sk, pose, sk.leg(leg)[0], lower_leg(sk, leg), slack=LEG_SLACK_M)
            w = sk.fk(pose)
            y = min(sk.low(w, sk.torso)[0], sk.low(w, necks)[0] + NECK_SLACK_M)
            if y < 0:
                shift(sk, pose, np.array([0.0, -y, 0.0]))
        else:
            # (the head lifted by the neck's base pushed the throat and the
            # breast 3-4 cm into the floor while the dog lay on its side)
            rest_on(sk, pose, floor, necks)
        if flat[k] > 1e-3:
            laid = pose.copy()
            for leg in ("LF", "RF"):
                role = "lower" if (leg == "LF") == (under[k] > 0) else "upper"
                shape_leg(sk, laid, leg, SIDE_FORELEGS[role], chest)
                untwist_leg(sk, laid, leg)
                lift(sk, laid, sk.leg(leg)[0], lower_leg(sk, leg), slack=LEG_SLACK_M)
                for i in sk.leg(leg):
                    pose.r[i] = slerp_m(pose.r.get(i, sk.rest_r[i]), laid.r.get(i, sk.rest_r[i]), float(flat[k]))
        # (the foreleg on top less as it lies on its side - it rests on the
        # one underneath; that one fully: let down only as the laid forelegs
        # gave way, it rose 6 cm off the floor rolling up off its side)
        below = "LF" if under[k] > 0 else "RF"
        fore_share = [1.0 if leg == below else 1.0 - flat[k] for leg in ("LF", "RF")]
        settle_paws(sk, pose, legs, down[k] * np.array(fore_share + [1.0, 1.0]))
        # (the paws held and let down by turns that keep no axis; and held
        # by the knee, a hind paw stood on a hock bent the wrong way again)
        for leg in LEGS:
            bones[leg] = align_leg(sk, pose, leg, bones[leg])
            # (a paw on the floor flat, pads down, unless the dog lies on its
            # side - there its paws lie on their sides: see flat_paw)
            flat_paw(sk, pose, leg, flatness[k, LEGS_ORDER[leg]])
            hinge_leg(sk, pose, leg)
            limit_hinges(sk, pose, leg)
        # a paw off the floor kept on its own side too (stepping as it got
        # up, the captured dog crossed its forepaws and a hind paw under it),
        # the move followed a share SIDE_FOLLOW of the way each frame (set
        # anew each frame, it came and went from one frame to the next and
        # the legs twitched - user, 11-06: "중간에 경련와서 부자연스러움")
        w = sk.fk(pose)
        for n, leg in enumerate(legs):
            toe = w[sk.leg(leg)[-1]][:3, 3]
            want = (own_side(sk, w, leg, toe) - toe) * firmly[leg] * (1.0 - hold[k, n])
            follow[leg] = follow[leg] + SIDE_FOLLOW * (want - follow[leg])
            if np.linalg.norm(follow[leg]) > 1e-4:
                hold_paw(sk, pose, leg, toe + follow[leg])
                w = sk.fk(pose)
        drape_tail(sk, pose)
        w = sk.fk(pose)
        report["torso"].append(sk.low(w, sk.torso)[0])
        report["legs"].append(sk.low(w, legs_mask)[0])
        report["head"].append(sk.low(w, heads)[0])
        report["tail"].append(sk.low(w, tails)[0])
        for i in sk.joints:
            rotations[i].append(Rotation.from_matrix(pose.r.get(i, sk.rest_r[i])).as_quat())
        translations.append(pose.t.copy())
    # the last LOOP_BLEND_S onto the first frame once more, on the template:
    # the capture loops, but the paws held at its end are not those held at
    # its start (an 11 deg jump where the clip came round)
    m = int(round(LOOP_BLEND_S * FPS))
    for i in sk.joints:
        first = rotations[i][0]
        for n in range(m):
            k = len(rotations[i]) - m + n
            u = float(smooth(np.array((n + 1) / m)))
            rotations[i][k] = Rotation.from_matrix(slerp_m(Rotation.from_quat(rotations[i][k]).as_matrix(),
                                                           Rotation.from_quat(first).as_matrix(), u)).as_quat()
    for n in range(m):
        k = len(translations) - m + n
        u = float(smooth(np.array((n + 1) / m)))
        translations[k] = translations[k] + (translations[0] - translations[k]) * u
    finish(sk, rotations)
    # the legs smoothed over time (CAPTURED_LEG_SMOOTH_FRAMES) and the held
    # paws put back on their spots after: turned frame by frame by IK, the
    # holds, the side corrections and the hinge limits jittered and the legs
    # twitched (user, 11-06: "중간에 경련와서 부자연스러움")
    from scipy.ndimage import gaussian_filter1d
    for leg in LEGS:
        for i in sk.leg(leg)[:4]:
            qs = gaussian_filter1d(np.asarray(rotations[i]), CAPTURED_LEG_SMOOTH_FRAMES, axis=0, mode="wrap")
            rotations[i] = list(qs / np.linalg.norm(qs, axis=1, keepdims=True))
    # (the body's height smoothed a little first: its lowest point moves
    # between parts of the body from frame to frame)
    t = np.asarray(translations)
    kernel = np.exp(-0.5 * (np.arange(-2, 3) / 1.0) ** 2)
    padded = np.concatenate([t[-2:, 1], t[:, 1], t[:2, 1]])
    t[:, 1] = np.convolve(padded, kernel / kernel.sum(), mode="valid")
    F = len(hips)
    for k in range(F):
        # (over the blend back onto the start, the spots and the holds
        # blended as the frames are: re-held up to it and not over it, the
        # paws jumped 1-2 cm where it began)
        u = float(smooth(np.array((k - (F - m) + 1) / m))) if k >= F - m else 0.0
        targets = {}
        for n, leg in enumerate(legs):
            a, b = spots[k].get(leg), spots[0].get(leg)
            weight = (1.0 - u) * (hold[k, n] if a is not None else 0.0) + u * (hold[0, n] if b is not None else 0.0)
            if weight > 1e-3:
                targets[n] = (b if a is None else a if b is None else (1.0 - u) * a + u * b, weight)
        if not targets:
            continue
        pose = Pose({i: Rotation.from_quat(rotations[i][k]).as_matrix() for i in sk.joints}, t[k])
        for n, (spot, weight) in targets.items():
            leg = legs[n]
            firm = pose.copy()
            hold_paw(sk, firm, leg, spot)
            for i in sk.leg(leg)[:4]:
                r = slerp_m(pose.r[i], firm.r[i], float(weight))
                q = Rotation.from_matrix(r).as_quat()
                rotations[i][k] = q if np.dot(q, rotations[i][k]) >= 0 else -q
    return {"times": np.arange(len(hips)) / FPS, "rotations": rotations, "translations": list(t), "report": report,
            "corrections": {"hold": hold, "lean": lean, "level": level, "fore": fore, "hind": hind, "flat": flat}}


# --- transitions --------------------------------------------------------------------------

def first_frame(sk: Skeleton, built: dict, name: str) -> Pose:
    """Clip `name`'s first frame as a pose: this run's, if it built the clip,
    else the GLB's."""
    if name in built:
        result = built[name]
        r = {i: Rotation.from_quat(result["rotations"][i][0]).as_matrix() for i in sk.joints}
        return Pose(r, np.asarray(result["translations"][0], dtype=np.float64).copy())
    _, frames = sample_clip(sk.rig, name, FPS)
    rest = sk.rig.rest_trs
    r = {i: Rotation.from_quat(frames[0].get(i, rest[i])[1]).as_matrix() for i in sk.joints}
    return Pose(r, np.asarray(frames[0].get(sk.cog, rest[sk.cog])[0], dtype=np.float64).copy())


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of each run of True in `mask`."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2]))


def reach_short(sk: Skeleton, pose: Pose, goals: list) -> float:
    """How far (m) the farthest of `goals` [(leg, toe point, the vector of
    the bones below the hock / wrist)] lies beyond its leg's reach - the hip /
    shoulder to the hock / wrist, REACH_SHARE of the two bones' length."""
    w = sk.fk(pose)
    out = -np.inf
    for leg, toe, lower in goals:
        a, b, c, _, _ = sk.leg(leg)
        out = max(out, float(np.linalg.norm(toe - lower - w[a][:3, 3]))
                  - REACH_SHARE * (sk.length(a, b) + sk.length(b, c)))
    return out


def pitch_to_reach(sk: Skeleton, pose: Pose, goals: list) -> float:
    """The least turn of the body nose-up or nose-down about its hips
    (degrees, + nose-down) that brings every paw of `goals` within reach;
    or, if none does, the nearest."""
    if not goals or reach_short(sk, pose, goals) <= 0:
        return 0.0
    best, best_short = 0.0, reach_short(sk, pose, goals)
    for deg in sorted(np.arange(-REACH_PITCH_DEG, REACH_PITCH_DEG + 1e-6, 1.0), key=abs):
        short = reach_short(sk, pitched(sk, pose, deg), goals)
        if short <= 0:
            return float(deg)
        if short < best_short:
            best, best_short = float(deg), short
    return best


def hips_line(sk: Skeleton, w: dict) -> tuple[np.ndarray, np.ndarray]:
    """(the point between the hips, the point between the shoulders)."""
    hips = np.mean([w[sk.j[n]][:3, 3] for n in ("L_hip_jnt", "R_hip_jnt")], axis=0)
    shoulders = np.mean([w[sk.j[n]][:3, 3] for n in ("L_shoulder_jnt", "R_shoulder_jnt")], axis=0)
    return hips, shoulders


def pitched(sk: Skeleton, pose: Pose, deg: float) -> Pose:
    """`pose` turned `deg` nose-down about the line through its hips."""
    out = pose.copy()
    if abs(deg) < 1e-9:
        return out
    w = sk.fk(out)
    hips = [sk.j[n] for n in ("L_hip_jnt", "R_hip_jnt")]
    pivot = np.mean([w[h][:3, 3] for h in hips], axis=0)
    across = w[hips[0]][:3, 3] - w[hips[1]][:3, 3]               # toward the dog's left
    across = across - UP * (across @ UP)
    turn(sk, out, sk.cog, across / max(np.linalg.norm(across), 1e-9), deg)
    shift(sk, out, pivot - np.mean([sk.fk(out)[h][:3, 3] for h in hips], axis=0))
    return out


def bend_of(sk: Skeleton, w: dict, leg: str) -> np.ndarray:
    """Which way the leg's knee / elbow stands off the line from the hip /
    shoulder to the hock / wrist (unit; pin's pole)."""
    a, b, c, _, _ = sk.leg(leg)
    line = w[c][:3, 3] - w[a][:3, 3]
    bend = w[b][:3, 3] - w[a][:3, 3]
    pole = bend - line * (bend @ line) / max(line @ line, 1e-12)
    if np.linalg.norm(pole) < 1e-3 * np.linalg.norm(line):
        pole = FWD if leg.endswith("B") else -FWD
    return pole / np.linalg.norm(pole)


def unit(v: np.ndarray) -> np.ndarray:
    return v / max(np.linalg.norm(v), 1e-12)


def toward(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    """The direction `u` of the way from direction `a` to direction `b`."""
    a = a / np.linalg.norm(a)
    return slerp_m(np.eye(3), swing(a, b), float(np.clip(u, 0.0, 1.0))) @ a


def lies(sk: Skeleton, pose: Pose) -> bool:
    """Lying: the chest below LYING_CHEST_SHARE of its standing height."""
    chest = sk.j["chest_jnt"]
    return bool(sk.fk(pose)[chest][1, 3] < LYING_CHEST_SHARE * sk.rest[chest][1, 3])


def other_side(sk: Skeleton, i: int) -> int:
    """The joint on the other side of the body (the joint itself on the midline)."""
    name = (sk.rig.gltf.nodes[i].name or "").split(".")[0]
    for a, b in (("LB_", "RB_"), ("RB_", "LB_"), ("LF_", "RF_"), ("RF_", "LF_"), ("L_", "R_"), ("R_", "L_")):
        if name.startswith(a):
            return sk.j.get(b + name[len(a):], i)
    return i


def square(sk: Skeleton, pose: Pose, joints) -> Pose:
    """`pose` the same on both sides for `joints`: each turned (in the world,
    from rest) half way between its own turn and its partner's reflected
    left for right - the body's lean and its turn about the vertical gone -
    and the body moved onto the midline."""
    w = sk.fk(pose)
    turns = {i: rot_of(w[i]) @ rot_of(sk.rest[i]).T for i in sk.joints}
    out = pose.copy()
    for i in sk.order:
        if i in joints:
            even = slerp_m(turns[i], MIRROR @ turns[other_side(sk, i)] @ MIRROR, 0.5)
            set_world(sk, out, sk.fk(out), i, even @ rot_of(sk.rest[i]))
    shift(sk, out, np.array([-sk.fk(out)[sk.cog][0, 3], 0.0, 0.0]))
    return out


def crouch(sk: Skeleton, down: Pose) -> Pose:
    """The roll's down squared up into a sphinx: the hips level with the
    chest instead of rolled onto their side, the back straight, the hind legs
    folded under it alike - as a sit folds them, the femur CROUCH_KNEE_UP_DEG
    from forward (the knee by the belly, the hock on the floor beside the
    hip, the paw forward)."""
    hind = {j for leg in ("LB", "RB") for j in sk.leg(leg)}
    tail = {sk.j[n] for n in TAIL if n in sk.j}
    pose = square(sk, down, [i for i in sk.joints if i not in hind and i not in tail])
    ground(sk, pose)
    for leg in ("LB", "RB"):
        on_floor(sk, pose, leg, lambda p, dy, leg=leg: fold_hind(sk, p, leg, 0.0, dy=dy,
                                                                 knee_up_deg=CROUCH_KNEE_UP_DEG))
    # (the paws flat, pads down: see flat_paw)
    flat_paws(sk, pose)
    ground(sk, pose, support(sk))
    drape_tail(sk, pose)
    return pose


def progress(y: np.ndarray) -> np.ndarray:
    """How far `y` has gone from its first value to its last (0..1, never back)."""
    if abs(y[-1] - y[0]) < 1e-6:
        return np.zeros(len(y))
    return np.maximum.accumulate(np.clip((y - y[0]) / (y[-1] - y[0]), 0.0, 1.0))


def paw_moves(feet: np.ndarray) -> list[list[tuple[int, int]]]:
    """Per paw, the [start, end) frames over which the captured dog moves it
    along the floor faster than STEP_SPEED_MS (pauses of two frames bridged),
    a frame either side."""
    from scipy.ndimage import uniform_filter1d
    speed = uniform_filter1d(np.linalg.norm(np.gradient(feet[..., [0, 2]], axis=0), axis=-1) * FPS, 3,
                             axis=0, mode="nearest")
    out = []
    for n in range(feet.shape[1]):
        merged: list[tuple[int, int]] = []
        for s, e in runs(speed[:, n] > STEP_SPEED_MS):
            if merged and s - merged[-1][1] <= 2:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        out.append([(max(s - 1, 0), min(e + 1, len(speed))) for s, e in merged if e - s >= 2])
    return out


def slow(steps: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Steps [start, end) lengthened to STEP_MIN_S, half each way, as far as
    the steps before and after leave room (a paw the capture whisked along
    the floor in 0.1-0.15 s stepped too fast to read)."""
    need = int(round(STEP_MIN_S * FPS))
    out = []
    for k, (s, e) in enumerate(steps):
        grow = need - (e - s)
        if grow > 0:
            before = out[-1][1] if out else -10 ** 6
            after = steps[k + 1][0] if k + 1 < len(steps) else 10 ** 6
            s = max(s - grow // 2, before, 0)
            e = min(s + need, after)
        out.append((s, e))
    return out


def unit_lower(sk: Skeleton, w: dict, leg: str) -> tuple[np.ndarray, np.ndarray]:
    """The directions of the two bones below the hock / wrist."""
    d1, d2 = lower_dirs(sk, w, leg)
    return d1 / np.linalg.norm(d1), d2 / np.linalg.norm(d2)


def carry_head(sk: Skeleton, pose: Pose, head_world: np.ndarray) -> None:
    """The head turned to `head_world` (its rotation in the world), the
    neck taking NECK_CARRY of the turn joint by joint and the head the rest."""
    head = sk.j["head_jnt"]
    fix = Rotation.from_matrix(head_world @ rot_of(sk.fk(pose)[head]).T).as_rotvec()
    angle = float(np.linalg.norm(fix))
    if angle > 1e-9:
        for joint, share in NECK_CARRY.items():
            turn(sk, pose, sk.j[joint], fix, np.degrees(angle) * share)
    set_world(sk, pose, sk.fk(pose), head, head_world)


def orthonormal(m: np.ndarray) -> np.ndarray:
    """The rotation nearest `m`."""
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def local_of(sk: Skeleton, w: dict, i: int) -> np.ndarray:
    """Joint i's rotation from its parent in the pose whose fk is `w`."""
    return rot_of(w[sk.parent[i]]).T @ rot_of(w[i])


def aim_tail(sk: Skeleton, pose: Pose, tail: list, wa: dict, wb: dict, u: float) -> None:
    """Each tail bone turned toward its direction `u` of the way from wa's to
    wb's (each in the pelvis's frame, so that it goes with the body) by way
    of laid straight back (sitting, swept round the haunch straight from the
    stand's carriage, it went through the thigh): from where it is, the
    least turn, so that it does not twist about its length; the last joint
    (no bone below it) turned from wa's to wb's."""
    pelvis = sk.j["pelvis_jnt"]
    w = sk.fk(pose)
    frame = rot_of(w[pelvis])
    ahead = body_frame(sk, w, sk.cog) @ FWD
    level = ahead - UP * (ahead @ UP)
    back = frame.T @ -(level / max(np.linalg.norm(level), 1e-9))
    for i, j in zip(tail, tail[1:]):
        da = rot_of(wa[pelvis]).T @ (wa[j][:3, 3] - wa[i][:3, 3])
        db = rot_of(wb[pelvis]).T @ (wb[j][:3, 3] - wb[i][:3, 3])
        d = toward(da, back, u / 0.5) if u < 0.5 else toward(back, db, (u - 0.5) / 0.5)
        aim(sk, pose, i, j, frame @ d)
    pose.r[tail[-1]] = slerp_m(local_of(sk, wa, tail[-1]), local_of(sk, wb, tail[-1]), u)


def turned_from(sk: Skeleton, a: Pose, b: Pose) -> float:
    """The most any joint is turned from pose a to pose b (degrees)."""
    return max(np.degrees(Rotation.from_matrix(a.r.get(i, sk.rest_r[i]) @ b.r.get(i, sk.rest_r[i]).T).magnitude())
               for i in sk.joints)


def exact_ends(sk: Skeleton, rotations: dict, start: Pose, end: Pose) -> None:
    """The first and last frames exactly `start` and `end` (smoothing moved them)."""
    for k, pose in ((0, start), (len(rotations[sk.joints[0]]) - 1, end)):
        for i in sk.joints:
            q = Rotation.from_matrix(pose.r.get(i, sk.rest_r[i])).as_quat()
            rotations[i][k] = q if np.dot(q, rotations[i][k]) >= 0 else -q


def build_transition(sk: Skeleton, captured_clips: dict, name: str, start: Pose, end: Pose) -> dict:
    """A change of pose from `start` (a held clip's first frame) to `end`
    (another's), on the template's own poses, timed by the captured dog's
    (dog_mocap.TRANSITIONS) - see TRANSITIONS."""
    clip = captured_clips[name]
    legs = ("LF", "RF", "LB", "RB")                      # dog_mocap.FEET's order
    a = crouch(sk, start) if lies(sk, start) else start  # the poses the body goes between
    b = crouch(sk, end) if lies(sk, end) else end
    # a lying end not square already (the roll's down) gathered from / settled
    # onto over LIE_SETTLE_S
    lie_a, lie_b = turned_from(sk, a, start) > 3.0, turned_from(sk, b, end) > 3.0
    wa, wb = sk.fk(a), sk.fk(b)
    toes = {leg: sk.leg(leg)[-1] for leg in legs}
    tail = [sk.j[n] for n in TAIL if n in sk.j]
    leg_joints = {j for leg in LEGS for j in sk.leg(leg)}
    body = [i for i in sk.joints if i not in leg_joints and i not in tail]
    # -- the capture's timing, over the stretch from just before anything
    # moves to just after the last of it
    uh = progress(clip["hips"][:, 1])
    shoulders = clip["shoulders"][:, 1]
    follow = abs(shoulders[-1] - shoulders[0]) > SHOULDER_CHANGE_M
    us = progress(shoulders) if follow else uh
    scale = sk.rest[sk.cog][1, 3] / captured_clips["stand_hips_height"]
    dip = np.zeros(len(uh)) if follow else \
        (shoulders - (shoulders[0] + (shoulders[-1] - shoulders[0]) * uh)) * scale
    moves = paw_moves(clip["feet"])
    busy = (np.gradient(uh) > 1e-3) | (np.gradient(us) > 1e-3)
    for paw_moved in moves:
        for s, e in paw_moved:
            busy[s:e] = True
    first = max(int(np.argmax(busy)) - 2, 0)
    last = min(len(busy) - int(np.argmax(busy[::-1])) + 2, len(busy) - 1)
    span = last - first + 1
    # -- the paws: each stays on its spot until the captured dog moves that
    # paw, then steps as far as that move takes it of the way to b's spot
    spot_a = {leg: wa[toes[leg]][:3, 3].copy() for leg in legs}
    spot_b = {leg: wb[toes[leg]][:3, 3].copy() for leg in legs}
    feet = clip["feet"]
    plans, late = {}, []
    for n, leg in enumerate(legs):
        plans[leg] = []
        if np.linalg.norm(spot_b[leg] - spot_a[leg]) <= STEP_TOLERANCE_M:
            continue
        mine = [(max(s, first) - first, min(e, last + 1) - first) for s, e in moves[n] if e > first and s <= last]
        mine = [(s, e) for s, e in mine if e - s >= 2]
        gone = [float(feet[first + e - 1, n, 2] - feet[first + s, n, 2]) for s, e in mine]
        total = sum(gone)
        if mine and abs(total) > 0.02 and all(g * total >= 0 for g in gone):
            share = np.cumsum(gone) / total
            share[-1] = 1.0
            plans[leg] = [(s, e, float(f)) for (s, e), f in zip(slow(mine), share)]
        else:
            late.append(leg)
    # a paw the capture never moves steps once the body is STEP_AFTER of the
    # way, one after another, hind paws first, and not while the paw on its
    # own side is off the floor (the two right paws up at once, the dog
    # rocked onto its left)
    went = 0.5 * (uh[first:last + 1] + us[first:last + 1])
    k0 = int(np.searchsorted(went, STEP_AFTER))
    for leg in sorted(late, key=lambda x: (x.endswith("F"), x)):
        mate = {"LF": "LB", "LB": "LF", "RF": "RB", "RB": "RF"}[leg]
        length = int(round(STEP_S * FPS))
        for s0, e0, _ in plans[mate]:
            if s0 < k0 + length and k0 < e0:
                k0 = e0
        plans[leg] = [(k0, k0 + length, 1.0)]
        k0 += length + int(round(STEP_GAP_S * FPS))
    main_frames = max(span, max((s[1] for p in plans.values() for s in p), default=0) + 2)

    def paw(leg: str, m: int) -> tuple[np.ndarray, float]:
        """Where the paw is at main frame m, and how far into a step it is
        (0: standing on its spot)."""
        here = spot_a[leg]
        for s, e, f in plans[leg]:
            there = spot_a[leg] + (spot_b[leg] - spot_a[leg]) * f
            if m < s:
                return here, 0.0
            if m < e:
                r = (m - s + 1) / (e - s + 1)
                # (lifted as sin, the paw left the floor at full speed and
                # the leg jerked as the step began)
                return here + (there - here) * float(smooth(np.array(r))) + UP * STEP_LIFT_M * np.sin(np.pi * r) ** 2, r
            here = there
        return here, 0.0

    # lying down, the forequarters go down as the forepaws walk out (a dog
    # lowers its chest as it reaches): following the captured dog's
    # shoulders alone, the jindo stood tall on two straight forelegs reaching
    # far ahead before it dropped (user, 11-06: "너가 볼때 아쉬운부분
    # 개선"), so the shoulders go at least BOW_SHARE of the way down that the
    # forepaws are out
    us_main = us[first:last + 1].copy()
    if follow and shoulders[-1] < shoulders[0]:
        out = np.array([np.mean([np.linalg.norm(paw(leg, m)[0] - spot_a[leg])
                                 / max(np.linalg.norm(spot_b[leg] - spot_a[leg]), 1e-9) for leg in ("LF", "RF")])
                        for m in range(span)])
        us_main = np.maximum(us_main, BOW_SHARE * np.clip(out, 0.0, 1.0))

    # -- the trunk frame by frame: its shape from a's to b's, its hips and
    # shoulders at the capture's heights brought onto a's and b's, the head
    # carried from a's carriage to b's
    hip_a, sh_a = hips_line(sk, wa)
    hip_b, sh_b = hips_line(sk, wb)
    head = sk.j["head_jnt"]
    trunks = []
    for m in range(span):
        c = first + m
        u = float(0.5 * (uh[c] + us_main[m]))
        pose = blend(sk, a, b, u, nodes=body)
        pose.t = a.t + (b.t - a.t) * u
        hip_y = hip_a[1] + (hip_b[1] - hip_a[1]) * uh[c]
        sh_y = sh_a[1] + (sh_b[1] - sh_a[1]) * us_main[m] + dip[c]
        hip_at = hip_a + (hip_b - hip_a) * uh[c]
        hip, sh = hips_line(sk, sk.fk(pose))
        length = np.linalg.norm(sh - hip)
        have = np.degrees(np.arcsin(np.clip((sh[1] - hip[1]) / length, -1.0, 1.0)))
        want = np.degrees(np.arcsin(np.clip((sh_y - hip_y) / length, -1.0, 1.0)))
        pose = pitched(sk, pose, have - want)
        hip, _ = hips_line(sk, sk.fk(pose))
        shift(sk, pose, np.array([hip_at[0] - hip[0], hip_y - hip[1], hip_at[2] - hip[2]]))
        low = sk.low(sk.fk(pose), sk.torso)[0]
        if low < 0:
            shift(sk, pose, UP * -low)
        carry_head(sk, pose, slerp_m(rot_of(wa[head]), rot_of(wb[head]), u))
        trunks.append((pose, float(uh[c]), float(us_main[m])))
    # a knee put as near as it goes to where the femur points, that turned
    # from a's to b's: the sit's knee stands out by the flank off the line
    # from the hip to the hock, and that sideways bend turned to from the
    # stand's forward one splayed the half-sat dog's knees out like a frog's
    # (user, 11-05: "다리가 ... 부자연스럽게 꺾여서"); nearly straight, the leg
    # bends the way a knee does (forward). An elbow bends the way the two
    # clips bend theirs, from the one to the other (put by the upper arm, it
    # flipped forward 90-110 deg in a frame as the forepaw reached out ahead)
    upper_a = {leg: unit(wa[sk.leg(leg)[1]][:3, 3] - wa[sk.leg(leg)[0]][:3, 3]) for leg in legs}
    upper_b = {leg: unit(wb[sk.leg(leg)[1]][:3, 3] - wb[sk.leg(leg)[0]][:3, 3]) for leg in legs}
    bend_a = {leg: bend_of(sk, wa, leg) for leg in legs}
    bend_b = {leg: bend_of(sk, wb, leg) for leg in legs}
    low_a = {leg: unit_lower(sk, wa, leg) for leg in legs}
    low_b = {leg: unit_lower(sk, wb, leg) for leg in legs}
    poses, lean = [], []
    for m in range(main_frames):
        pose, u_h, u_s = trunks[min(m, span - 1)]
        pose = pose.copy()
        across = rot_of(sk.fk(pose)[sk.cog]) @ LEFT
        wants, goals = {}, []
        for leg in legs:
            u = u_h if leg.endswith("B") else u_s
            toe, r = paw(leg, m)
            d1, d2 = toward(low_a[leg][0], low_b[leg][0], u), toward(low_a[leg][1], low_b[leg][1], u)
            if leg.endswith("F") and 0 < r < 1:
                # the forepaw curls back off the floor as it steps
                curl = rotm(across, TRANSITION_CURL_DEG * np.sin(np.pi * r) ** 2)
                d1, d2 = curl @ d1, curl @ d2
            c, d, e = sk.leg(leg)[2:]
            lower = d1 * sk.length(c, d) + d2 * sk.length(d, e)
            pole = (toward(upper_a[leg], upper_b[leg], u) + KNEE_BIAS * FWD if leg.endswith("B")
                    else toward(bend_a[leg], bend_b[leg], u))
            wants[leg] = (toe, r, (d1, d2), lower, pole)
            # (a stepping paw too: left out, the body rose as the paw came
            # off the floor and the leg reached for its step straight)
            goals.append((leg, toe, lower))
        lean.append(pitch_to_reach(sk, pose, goals))
        poses.append((pose, wants, 0.5 * (u_h + u_s)))
    from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d
    # the lean each frame needs held over the frames about it, then smoothed
    # (smoothed alone, a frame's need was halved and the leg snapped straight
    # reaching for its spot, and back - user, 11-06: "중간에 경련와서")
    lean = np.array(lean)
    width = 2 * REACH_HOLD_FRAMES + 1
    lean = (maximum_filter1d(np.maximum(lean, 0.0), width, mode="nearest")
            + minimum_filter1d(np.minimum(lean, 0.0), width, mode="nearest"))
    lean = gaussian_filter1d(lean, REACH_SMOOTH_FRAMES, mode="nearest")
    # -- the timeline: getting up, the gathering first; lying down, the
    # settling after; else the ends eased onto the clips' frames
    settle = int(round(LIE_SETTLE_S * FPS))
    overlap = 3
    begin = settle - overlap if lie_a else 0             # the main motion's first frame
    frames = begin + main_frames + (settle - overlap if lie_b else 0)
    times = np.arange(frames) / FPS
    rotations = {i: [] for i in sk.joints}
    translations = []
    report = {"torso": [], "legs": [], "head": [], "tail": [], "slip": []}
    heads = sk.mask(sk.subtree(sk.j["neck_base_jnt"]))
    tails = sk.mask(tail)
    legs_mask = sk.mask([j for leg in LEGS for j in sk.subtree(sk.leg(leg)[0])])
    tail_now = {i: a.r.get(i, sk.rest_r[i]).copy() for i in tail}
    for k in range(frames):
        m = int(np.clip(k - begin, 0, main_frames - 1))
        pose, wants, went_m = poses[m]
        pose = pitched(sk, pose, float(lean[m]))
        # (tipped onto the floor as it lies down, raised back out of it: the
        # chest went 3 cm into it)
        low = sk.low(sk.fk(pose), sk.torso)[0]
        if low < 0:
            shift(sk, pose, UP * -low)
        slip = np.zeros(len(legs))
        for n, leg in enumerate(legs):
            toe, r, lower_dirs_, lower, pole = wants[leg]
            plant(sk, pose, leg, toe - lower, lower_dirs_, pole)
            if r == 0:
                slip[n] = float(np.linalg.norm(sk.fk(pose)[toes[leg]][:3, 3] - toe))
        for i in tail:
            pose.r[i] = tail_now[i]
        aim_tail(sk, pose, tail, wa, wb, float(smooth(np.array(went_m))))
        # (carried from frame to frame, the rotations drifted off true until
        # one would not read as a rotation)
        tail_now = {i: Rotation.from_matrix(orthonormal(pose.r[i])).as_matrix() for i in tail}
        drape_tail(sk, pose)
        # the ends onto the clips' own frames
        if lie_a:
            if k < settle:
                pose = blend(sk, start, pose, float(smooth(np.array(k / settle))))
        else:
            pose = blend(sk, start, pose, float(smooth(np.array(k / (TRANSITION_IN_S * FPS)))))
        if lie_b:
            ks = begin + main_frames - overlap
            if k >= ks:
                pose = blend(sk, pose, end, float(smooth(np.array((k - ks) / max(frames - 1 - ks, 1)))))
        else:
            out = TRANSITION_OUT_S * FPS
            pose = blend(sk, pose, end, float(smooth(np.array((k - (frames - 1 - out)) / out))))
        if lie_a or lie_b:
            lift_legs(sk, pose)
        if k == 0:
            pose = start.copy()
        elif k == frames - 1:
            pose = end.copy()
        w = sk.fk(pose)
        report["torso"].append(sk.low(w, sk.torso)[0])
        report["legs"].append(sk.low(w, legs_mask)[0])
        report["head"].append(sk.low(w, heads)[0])
        report["tail"].append(sk.low(w, tails)[0])
        report["slip"].append(slip)
        for i in sk.joints:
            rotations[i].append(Rotation.from_matrix(pose.r.get(i, sk.rest_r[i])).as_quat())
        translations.append(pose.t.copy())
    finish(sk, rotations, loop=False)
    exact_ends(sk, rotations, start, end)
    return {"times": times, "rotations": rotations, "translations": translations, "report": report,
            "lean": lean, "plans": plans}


# --- writing ------------------------------------------------------------------------------

def breathing(times: np.ndarray) -> np.ndarray:
    """breathe_jnt's scale over `times`, a whole number of breaths."""
    length = float(times[-1] - times[0]) if len(times) > 1 else BREATH_S
    period = length / max(1, round(length / BREATH_S))
    swell = 0.5 - 0.5 * np.cos(2 * np.pi * (times - times[0]) / period)
    return np.stack([np.ones_like(times), 1.0 + BREATH_SWELL * swell, np.ones_like(times)], axis=1)


def write(glb: Path, clips: dict[str, dict], cog: int, out: Path, breathe: int | None = None, drop=()) -> None:
    gltf = GLTF2().load(str(glb))
    names = set(clips) | set(drop)
    gltf.animations = [a for a in gltf.animations or [] if a.name not in names]
    blob = prune(gltf, gltf.binary_blob())
    for name, result in clips.items():
        times = np.asarray(result["times"], dtype=np.float32)
        time_acc = append_accessor(gltf, blob, times, "SCALAR", bounds=True)
        anim = Animation(name=name, samplers=[], channels=[])
        for node, quats in result["rotations"].items():
            acc = append_accessor(gltf, blob, np.asarray(quats, dtype=np.float32), "VEC4")
            anim.samplers.append(AnimationSampler(input=time_acc, output=acc, interpolation="LINEAR"))
            anim.channels.append(AnimationChannel(sampler=len(anim.samplers) - 1,
                                                  target=AnimationChannelTarget(node=node, path="rotation")))
        acc = append_accessor(gltf, blob, np.asarray(result["translations"], dtype=np.float32), "VEC3")
        anim.samplers.append(AnimationSampler(input=time_acc, output=acc, interpolation="LINEAR"))
        anim.channels.append(AnimationChannel(sampler=len(anim.samplers) - 1,
                                              target=AnimationChannelTarget(node=cog, path="translation")))
        if breathe is not None and name not in NO_BREATH:
            scales = result["breathe"] if "breathe" in result else breathing(times.astype(np.float64))
            acc = append_accessor(gltf, blob, np.asarray(scales, dtype=np.float32), "VEC3")
            anim.samplers.append(AnimationSampler(input=time_acc, output=acc, interpolation="LINEAR"))
            anim.channels.append(AnimationChannel(sampler=len(anim.samplers) - 1,
                                                  target=AnimationChannelTarget(node=breathe, path="scale")))
        gltf.animations.append(anim)
    gltf.buffers[0].byteLength = len(blob)
    gltf.set_binary_blob(bytes(blob))
    out.parent.mkdir(parents=True, exist_ok=True)
    gltf.save(str(out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slug", required=True)
    parser.add_argument("--size", default="small")
    parser.add_argument("--clip", action="append", help="only these clips (default: all)")
    parser.add_argument("--glb", default=None, help="pose this GLB (a built dog) instead of the template")
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--out", default=None)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--drop", action="append", default=[], help="clips to take out of the GLB")
    args = parser.parse_args()
    glb = Path(args.glb) if args.glb else ROOT / "output" / f"skeleton_{args.slug}_{args.size}.glb"
    sk = Skeleton(glb)
    src = Source(Path(args.source))
    keys = solve_keys(sk)
    print(f"[pose_clips] sit pitched {keys['sit'].pitch:.0f} deg")
    clips = {}
    captured_clips = dog_mocap.load_clips()
    poses = dog_mocap.load()
    video = video_motion.load()
    # (the transitions last: they start and end on the held clips' first
    # frames, this run's if it builds them)
    for name in args.clip or list(CLIPS) + list(CAPTURED_CLIPS) + list(STAND_WAG) + list(BARK) + list(TRANSITIONS):
        if name in TRANSITIONS:
            a, b = TRANSITIONS[name]
            result = build_transition(sk, captured_clips, name, first_frame(sk, clips, a), first_frame(sk, clips, b))
            print(f"[pose_clips] {name}: {a} -> {b}, a held paw off its spot by at most "
                  f"{np.max(result['report'].pop('slip')) * 100:.1f} cm")
        elif name in CAPTURED_CLIPS:
            result = build_captured(sk, poses, captured_clips, name)
        elif name in STAND_WAG:
            result = build_stand_wag(sk, keys, name)
        elif name in BARK:
            if "jaw_jnt" not in sk.j or not sk.mask([sk.j["jaw_jnt"]]).any():
                print(f"[pose_clips] {name}: skipped - the jaw carries no skin (ar_pipeline/mouth.py first)")
                continue
            result = build_bark(sk, keys, name)
        elif name in VIDEO_BASED and name in video:
            result = build_video(sk, keys, video, name)
        else:
            result = build_clip(sk, src, keys, name)
        rep = {k: np.asarray(v) * 100 for k, v in result["report"].items()}
        print(f"[pose_clips] {name}: {len(result['times'])} frames; lowest (cm) torso "
              f"{rep['torso'].min():+.1f}..{rep['torso'].max():+.1f}, legs {rep['legs'].min():+.1f}, "
              f"head {rep['head'].min():+.1f}, tail {rep['tail'].min():+.1f}")
        clips[name] = result
    if args.install:
        backup = ROOT / "work" / f"backup_anim_{args.slug}"
        if not backup.exists():
            backup.mkdir(parents=True)
            shutil.copy(glb, backup / glb.name)
        out = glb
    else:
        out = Path(args.out) if args.out else ROOT / "work" / "anim" / f"{args.slug}_clips.glb"
    write(glb, clips, sk.cog, out, sk.j.get("breathe_jnt"), drop=args.drop)
    print(f"[pose_clips] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
