#!/usr/bin/env bash
# One-time setup on a fresh Ubuntu 24.04 EC2 instance. Clone the repo to
# ~/pipeline, put .env next to it, then from ~/pipeline:
#   bash deploy/setup_server.sh
# Safe to run again: it skips what is already there.
set -euo pipefail
cd "$(dirname "$0")/.."
[ "$PWD" = "$HOME/pipeline" ] || { echo "clone the repo to ~/pipeline (deploy/pet-pipeline.service expects it)"; exit 1; }

sudo apt-get update
sudo apt-get install -y gh libgl1

# Python 3.13 + venv via uv
command -v ~/.local/bin/uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
UV=~/.local/bin/uv
[ -d .venv ] || $UV venv --python 3.13 .venv
source .venv/bin/activate
$UV pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
$UV pip install -r requirements.txt

# Templates (not in git): the assets-v1 release of this repo
if [ ! -d output/templates/jindo_small ]; then
  gh auth status >/dev/null 2>&1 || gh auth login
  gh release download assets-v1 -D /tmp/pet-assets --clobber
  for f in /tmp/pet-assets/*.tar.gz; do tar -xzf "$f"; done
  rm -rf /tmp/pet-assets
fi

[ -f .env ] || { echo "write ~/pipeline/.env (OPENAI_API_KEY, PIPELINE_API_KEY) and run this again"; exit 1; }

sudo cp deploy/pet-pipeline.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now pet-pipeline
sudo systemctl restart pet-pipeline
sleep 3
curl -fsS localhost:8000/health && echo && echo "pipeline is up on :8000"
