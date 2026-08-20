#!/usr/bin/env bash
# Download Wan2.1 I2V 14B for V2V + depth FT.
#
# Writes to a LOCAL directory first, then optionally copies into /data (GCS fuse).
# Do not hf-download straight onto gcsfuse at ~1GB/s — that leaves *.incomplete
# and intermittent Errno 107.
#
# Usage:
#   bash scripts/v2v_depth/prepare_models/download_wan_i2v_14b.sh
#   LOCAL_DIR=/workspace/models DEST_DIR=/data/models bash scripts/v2v_depth/prepare_models/download_wan_i2v_14b.sh
#   RESOLUTION=720p bash scripts/v2v_depth/prepare_models/download_wan_i2v_14b.sh
#   SKIP_COPY=1 bash scripts/v2v_depth/prepare_models/download_wan_i2v_14b.sh   # local only

set -euo pipefail

RESOLUTION="${RESOLUTION:-480p}"   # 480p | 720p
case "${RESOLUTION}" in
  480p|480P)
    REPO_ID="${REPO_ID:-Wan-AI/Wan2.1-I2V-14B-480P-Diffusers}"
    DIR_NAME="${DIR_NAME:-Wan2.1-I2V-14B-480P-Diffusers}"
    ;;
  720p|720P)
    REPO_ID="${REPO_ID:-Wan-AI/Wan2.1-I2V-14B-720P-Diffusers}"
    DIR_NAME="${DIR_NAME:-Wan2.1-I2V-14B-720P-Diffusers}"
    ;;
  *)
    echo "RESOLUTION must be 480p or 720p, got: ${RESOLUTION}" >&2
    exit 1
    ;;
esac

LOCAL_DIR="${LOCAL_DIR:-/tmp/models/${DIR_NAME}}"
DEST_DIR="${DEST_DIR:-/data/models/${DIR_NAME}}"
SKIP_COPY="${SKIP_COPY:-0}"

# Prefer stable hub downloads over hf_transfer bursts onto slow/fuse targets.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

echo "=== Wan2.1 I2V 14B download ==="
echo "Repo:       ${REPO_ID}"
echo "Local dir:  ${LOCAL_DIR}"
echo "Dest dir:   ${DEST_DIR}"
echo "Skip copy:  ${SKIP_COPY}"
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

# Sanity: Diffusers layout required by FastVideo loader.
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
  echo "Set DEST_DIR or SKIP_COPY=1" >&2
  exit 1
fi

echo "Copying to ${DEST_DIR} (this may take a while on gcsfuse)..."
mkdir -p "$(dirname "${DEST_DIR}")"
rm -rf "${DEST_DIR}"
# Archive copy is gentler on fuse than millions of tiny random writes from hf.
cp -a "${LOCAL_DIR}" "${DEST_DIR}"

for need in model_index.json transformer vae text_encoder tokenizer; do
  if [[ ! -e "${DEST_DIR}/${need}" ]]; then
    echo "ERROR: copy missing ${DEST_DIR}/${need}" >&2
    exit 1
  fi
done

echo "Done."
echo "  local:  ${LOCAL_DIR}"
echo "  gcs:    ${DEST_DIR}"
echo "YAML: init_from: ${DEST_DIR}"
