#!/usr/bin/env bash
# A dog video -> MoCapAnything BVH on the Truebones "Dog" skeleton, in WSL.
#   wsl -d Ubuntu-24.04 -- bash ar_pipeline/mocapanything/run_video.sh <name> </mnt/c/.../clip.mp4> <out dir (/mnt/c/...)>
# Then: python -m ar_pipeline.video_motion --clip <clip> --bvh <out dir>/Dog_Dog_rot6d_pred.bvh
# Best input: one dog, whole body in view, from the side, camera still, cropped to the dog.
set -euo pipefail
NAME="$1"; SRC="$2"; OUT="$3"
source ~/mca/bin/activate
cd ~/MocapAnything
mkdir -p ours/videos "$OUT"
find ours/videos -name '*.mp4' -delete
cp "$SRC" "ours/videos/Dog#${NAME}.mp4"
python - <<'PY'
import yaml
c = yaml.safe_load(open("demo/configs/demo_wild.yaml"))
c["data"]["video_roots"] = ["./ours/videos"]
c["data"]["frames_tmp_root"] = "./demo_outputs/_frames_ours"
c["output"]["output_tag"] = "ours"
c["output"]["save_dir"] = "./demo_outputs/ours"
c["output"]["blender_path"] = ""
yaml.safe_dump(c, open("ours/config.yaml", "w"))
PY
export PYTHONPATH="$PWD:$PWD/TripoSG"
python inference/video2pose2rot.py --config ours/config.yaml > "$OUT/inference.log" 2>&1
cp "demo_outputs/ours/ours/videos/Dog#${NAME}/"*.bvh "demo_outputs/ours/ours/videos/Dog#${NAME}/"*side.mp4 "$OUT/"
ls "$OUT"
