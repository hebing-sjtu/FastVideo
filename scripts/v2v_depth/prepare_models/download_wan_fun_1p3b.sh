#!/usr/bin/env bash
# Download Wan2.1-Fun-1.3B-InP for V2V + depth FT (debug / smoke track).
#
# Local first, then optional copy to /data — same pattern as download_wan_i2v_14b.sh.
#
# Usage:
#   bash scripts/v2v_depth/prepare_models/download_wan_fun_1p3b.sh
#   LOCAL_DIR=/workspace/models DEST_DIR=/data/models bash scripts/v2v_depth/prepare_models/download_wan_fun_1p3b.sh
#   SKIP_COPY=1 bash scripts/v2v_depth/prepare_models/download_wan_fun_1p3b.sh

set -euo pipefail

REPO_ID="${REPO_ID:-weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers}"
DIR_NAME="${DIR_NAME:-Wan2.1-Fun-1.3B-InP-Diffusers}"
LOCAL_DIR="${LOCAL_DIR:-/tmp/models/${DIR_NAME}}"
DEST_DIR="${DEST_DIR:-/data/models/${DIR_NAME}}"
SKIP_COPY="${SKIP_COPY:-0}"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

echo "=== Wan2.1 Fun 1.3B InP download ==="
echo "Repo:       ${REPO_ID}"
echo "Local dir:  ${LOCAL_DIR}"
echo "Dest dir:   ${DEST_DIR}"
echo "=============================="

mkdir -p "$(dirname "${LOCAL_DIR}")"
rm -rf "${LOCAL_DIR}.incomplete" "${LOCAL_DIR}/.cache" 2>/dev/null || true

if command -v hf >/dev/null 2>&1; then
  hf download "${REPO_ID}" --local-dir "${LOCAL_DIR}"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "${REPO_ID}" --local-dir "${LOCAL_DIR}"
else
  python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="${REPO_ID}", local_dir="${LOCAL_DIR}", local_dir_use_symlinks=False, resume_download=True)
PY
fi

for need in model_index.json transformer vae text_encoder tokenizer; do
  if [[ ! -e "${LOCAL_DIR}/${need}" ]]; then
    echo "ERROR: missing ${LOCAL_DIR}/${need} — download incomplete" >&2
    exit 1
  fi
done
echo "Local download OK: ${LOCAL_DIR}"

if [[ "${SKIP_COPY}" == "1" ]]; then
  echo "SKIP_COPY=1 — not copying to ${DEST_DIR}"
  echo "Use init_from: ${LOCAL_DIR}"
  exit 0
fi

if [[ ! -d "$(dirname "${DEST_DIR}")" ]]; then
  echo "Parent of DEST_DIR does not exist: $(dirname "${DEST_DIR}")" >&2
  exit 1
fi

echo "Copying to ${DEST_DIR}..."
mkdir -p "$(dirname "${DEST_DIR}")"
rm -rf "${DEST_DIR}"
cp -a "${LOCAL_DIR}" "${DEST_DIR}"

for need in model_index.json transformer vae text_encoder tokenizer; do
  if [[ ! -e "${DEST_DIR}/${need}" ]]; then
    echo "ERROR: copy missing ${DEST_DIR}/${need}" >&2
    exit 1
  fi
done

echo "Done. init_from: ${DEST_DIR}"
