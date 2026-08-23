#!/usr/bin/env bash
# Idempotent per-node setup for the official FastVideo image.
#
# Mental model: the Docker image is the runtime; this repo is the code.
# Maintain the repo on your laptop, push, then on each training node:
#
#   First time (deploy key or PAT must already work). Use bash, not sh:
#     exec bash
#     git clone git@github.com:hebing-sjtu/FastVideo.git /workspace/FastVideo
#     bash /workspace/FastVideo/scripts/bootstrap_node.sh
#
#   Every later run (both nodes, same command):
#     bash /workspace/FastVideo/scripts/bootstrap_node.sh
#
# This script pulls, reinstalls the editable package (no torch rebuild),
# creates local data dirs, and hooks the env into ~/.bashrc.
# It does not start training and does not configure laptop SSH.
#
# Usage:
#   bash scripts/bootstrap_node.sh [--no-pull] [--no-install] [--no-bashrc]
#
# Env overrides:
#   FV_REPO_DIR     default /workspace/FastVideo
#   FV_DATA_ROOT    default /data
#   FV_VENV         default /opt/venv
#   FV_GIT_REMOTE   default origin
#   FV_GIT_BRANCH   default current branch (usually main)

set -euo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
    echo "ERROR: this script must run under bash, not sh." >&2
    echo "  exec bash" >&2
    echo "  bash $0 $*" >&2
    exit 1
fi

DO_PULL=1
DO_INSTALL=1
DO_BASHRC=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-pull) DO_PULL=0; shift ;;
        --no-install) DO_INSTALL=0; shift ;;
        --no-bashrc) DO_BASHRC=0; shift ;;
        -h|--help)
            sed -n '2,30p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown arg: $1" >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FV_REPO_DIR="${FV_REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export FV_DATA_ROOT="${FV_DATA_ROOT:-/data}"
export FV_VENV="${FV_VENV:-/opt/venv}"
export FV_GIT_REMOTE="${FV_GIT_REMOTE:-origin}"

# shellcheck source=/dev/null
. "${SCRIPT_DIR}/node_env.sh"

echo "=== FastVideo node bootstrap ==="
echo "Host:       $(hostname)"
echo "Repo:       ${FV_REPO_DIR}"
echo "Data root:  ${FV_DATA_ROOT}"
echo "Venv:       ${FV_VENV}"
echo "================================"

if [[ ! -f "${FV_VENV}/bin/activate" ]]; then
    echo "ERROR: ${FV_VENV}/bin/activate not found." >&2
    echo "This script expects ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:py3.12-latest" >&2
    exit 1
fi

mkdir -p \
    "${FV_LOCAL_ROOT}/hf" \
    "${FV_LOCAL_ROOT}/models" \
    "${FV_DATA_ROOT}/ckpt" \
    "${FV_DATA_ROOT}/raw" \
    "${FV_DATA_ROOT}/preprocessed" \
    "${FV_REPO_DIR}/logs"

cd "${FV_REPO_DIR}"

if [[ ! -d .git ]]; then
    echo "ERROR: ${FV_REPO_DIR} is not a git checkout." >&2
    echo "On this node run:" >&2
    echo "  git clone git@github.com:hebing-sjtu/FastVideo.git ${FV_REPO_DIR}" >&2
    exit 1
fi

if [[ "${DO_PULL}" -eq 1 ]]; then
    if [[ -n "$(git status --porcelain)" ]]; then
        echo "ERROR: working tree is dirty; refusing to pull." >&2
        echo "Commit on your laptop and push, or stash/discard on the node." >&2
        git status -sb
        exit 1
    fi
    branch="${FV_GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
    echo "Pulling ${FV_GIT_REMOTE}/${branch} ..."
    git fetch "${FV_GIT_REMOTE}"
    git pull --ff-only "${FV_GIT_REMOTE}" "${branch}"
fi

echo "Git HEAD: $(git rev-parse --short HEAD)  $(git log -1 --pretty=%s)"
echo "Remote:   $(git remote get-url "${FV_GIT_REMOTE}")"

if [[ "${DO_INSTALL}" -eq 1 ]]; then
    echo "Editable install (no dependency resolve) ..."
    if command -v uv >/dev/null 2>&1; then
        uv pip install -e ".[dev]" --no-deps
    elif [[ -x "${FV_VENV}/bin/pip" ]]; then
        echo "uv not on PATH; falling back to ${FV_VENV}/bin/pip"
        "${FV_VENV}/bin/pip" install -e ".[dev]" --no-deps
    else
        echo "ERROR: uv not found. On the official image it is /root/.local/bin/uv" >&2
        echo "  echo \$0          # should be bash, not sh" >&2
        echo "  ls /root/.local/bin/uv /opt/venv/bin/python" >&2
        echo "  export PATH=\"/root/.local/bin:/opt/venv/bin:\$PATH\"" >&2
        exit 1
    fi
fi

if [[ "${DO_BASHRC}" -eq 1 ]]; then
    bashrc="${HOME}/.bashrc"
    marker_begin="# >>> fastvideo-node >>>"
    marker_end="# <<< fastvideo-node <<<"
    block=$(cat <<EOF
${marker_begin}
export FV_REPO_DIR="${FV_REPO_DIR}"
export FV_DATA_ROOT="${FV_DATA_ROOT}"
export FV_VENV="${FV_VENV}"
. "${SCRIPT_DIR}/node_env.sh"
${marker_end}
EOF
)
    if [[ -f "${bashrc}" ]] && grep -q "${marker_begin}" "${bashrc}"; then
        tmp="$(mktemp)"
        awk -v b="${marker_begin}" -v e="${marker_end}" '
            $0 == b { skip=1; next }
            $0 == e { skip=0; next }
            !skip { print }
        ' "${bashrc}" > "${tmp}"
        printf '\n%s\n' "${block}" >> "${tmp}"
        mv "${tmp}" "${bashrc}"
    else
        printf '\n%s\n' "${block}" >> "${bashrc}"
    fi
    echo "Hooked env into ${bashrc}"
fi

echo
echo "--- sanity ---"
echo "uv: $({ command -v uv || echo missing; })"
"${FV_VENV}/bin/python" - <<'PY'
import sys
print("python:", sys.executable)
try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.version.cuda, "gpus:", torch.cuda.device_count())
except Exception as exc:
    print("torch: UNAVAILABLE", exc)
    sys.exit(1)
try:
    import fastvideo
    print("fastvideo:", getattr(fastvideo, "__file__", "ok"))
except Exception as exc:
    print("fastvideo: UNAVAILABLE", exc)
    sys.exit(1)
PY

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L
fi

echo
echo "Bootstrap done. Next (do not start from this script):"
echo "  tmux new -s work   # or: tmux attach -t work"
echo "  # compare HEAD on the other node: git rev-parse HEAD"
echo "  # single node:"
echo "  NUM_GPUS=8 bash examples/train/run.sh <config.yaml>"
echo "  # two nodes: same MASTER_ADDR (node0 IP), NODE_RANK=0 and 1"
echo "  NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0> NUM_GPUS=8 bash examples/train/run.sh <config.yaml>"
