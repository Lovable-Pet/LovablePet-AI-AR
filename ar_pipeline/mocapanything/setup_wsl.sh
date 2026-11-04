#!/usr/bin/env bash
# One-time setup of MoCapAnything (https://github.com/animotionlab26/MocapAnything, MIT) in WSL
# Ubuntu, without sudo: uv for the Python env, the end-to-end weights and the demo data
# (which holds the Truebones "Dog" reference skeleton video_motion maps from).
#   wsl -d Ubuntu-24.04 -- bash ar_pipeline/mocapanything/setup_wsl.sh   (from Git Bash: MSYS_NO_PATHCONV=1)
set -euo pipefail
mkdir -p ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
    cd /tmp && curl -sL -o uv.tgz https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz
    tar xzf uv.tgz && cp uv-x86_64-unknown-linux-gnu/uv ~/.local/bin/
fi
cd ~
[ -d MocapAnything ] || git clone -q --depth 1 https://github.com/animotionlab26/MocapAnything.git
[ -x ~/mca/bin/python ] || uv venv -q ~/mca --python 3.12
source ~/mca/bin/activate
uv pip install -q torch torchvision numpy opencv-python-headless pillow matplotlib scipy scikit-image \
    trimesh roma pyyaml tqdm huggingface_hub transformers gradio imageio-ffmpeg einops easydict
cd ~/MocapAnything
hf download kehong/MoCapAnythingV2-weights --local-dir ./checkpoints >/dev/null
hf download kehong/MoCapAnythingV2-data-sample --repo-type dataset --local-dir ./demo/data >/dev/null
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
