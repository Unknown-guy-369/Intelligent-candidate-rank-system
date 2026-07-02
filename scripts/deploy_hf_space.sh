#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: scripts/deploy_hf_space.sh <namespace/space-name>" >&2
  echo "Example: scripts/deploy_hf_space.sh your-hf-username/redrob-candidate-ranker-sandbox" >&2
  exit 2
fi

SPACE_ID="$1"
HF_BIN="${HF_BIN:-.venv-hf/bin/hf}"

"$HF_BIN" auth whoami >/dev/null
"$HF_BIN" repos create "$SPACE_ID" --type space --space-sdk docker --exist-ok
"$HF_BIN" upload "$SPACE_ID" . --type space \
  --include Dockerfile \
  --include .dockerignore \
  --include sandbox_app.py \
  --include "app/**" \
  --include requirements.txt \
  --include requirements-sandbox.txt \
  --include README.md \
  --exclude "data/**" \
  --exclude ".git/**" \
  --exclude ".venv*/**" \
  --exclude "__pycache__/**" \
  --exclude "*/__pycache__/**" \
  --commit-message "Deploy Redrob candidate ranker sandbox"

echo "Deployed: https://huggingface.co/spaces/$SPACE_ID"
