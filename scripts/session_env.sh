# Restore a shell to the environment a FastVideo run expects, and verify it rather than assume it.
#
#   . /workspace/FastVideo/scripts/session_env.sh
#
# Must be sourced. Running it as a child process sets variables that die with the child, which is
# the whole reason this file exists separately from bootstrap_node.sh.
#
# What it is for. `node_env.sh` sets the paths and activates /opt/venv, and ~/.bashrc already
# sources it -- but it assumes it is the first thing to touch the shell. Three things break that:
#
#   * another venv is active. `uv run`, `uv venv` or a hand-rolled venv prepends its own bin to
#     PATH and sets VIRTUAL_ENV, and activating /opt/venv on top does not undo either -- the second
#     activate clobbers the _OLD_VIRTUAL_PATH the first one saved, so `deactivate` can no longer
#     unwind it. The state that produces is a shell where `python` is right and `pip`, `torchrun`
#     or a subprocess is not.
#   * the NCCL launch block is documented in NODE_ENVIRONMENT.md but lives in no script, so it is
#     re-typed per session and a missing `unset` is invisible: a stale NCCL_NET_PLUGIN=none sends a
#     16-rank job over TCP at a fraction of the bandwidth with no error at all.
#   * a stale /FastVideo exists on the image. A run importing that one ignores every local edit and
#     reports nothing unusual.
#
# So this deactivates whatever is active, applies both blocks, and then checks. Everything it sets
# is idempotent; sourcing it twice is a no-op.

# --- locate the checkout, under bash or zsh ------------------------------------------------------
# Web consoles land in either shell. zsh's %x expansion is a bash parse error, so it goes through
# eval where bash never sees it.
_fv_self=""
if [ -n "${BASH_SOURCE:-}" ]; then
    _fv_self="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
    eval '_fv_self="${(%):-%x}"'
fi
if [ -n "${_fv_self}" ] && [ -f "${_fv_self}" ]; then
    FV_REPO_DIR="$(cd "$(dirname "${_fv_self}")/.." && pwd)"
else
    FV_REPO_DIR="${FV_REPO_DIR:-/workspace/FastVideo}"
fi
export FV_REPO_DIR
: "${FV_VENV:=/opt/venv}"
export FV_VENV
unset _fv_self

_fv_problems=0
_fv_note() { printf '  %s\n' "$*"; }
_fv_fail() { printf '  PROBLEM: %s\n' "$*"; _fv_problems=$((_fv_problems + 1)); }

echo "=== session env: ${FV_REPO_DIR} ==="

# --- 1. leave any foreign venv -------------------------------------------------------------------
# `deactivate` first, because it is the only thing that can restore the PATH the venv saved. It is a
# shell function, so it exists only if some activate script ran in this shell.
_fv_active=""
[ -n "${VIRTUAL_ENV:-}" ] && _fv_active="$(cd "${VIRTUAL_ENV}" 2>/dev/null && pwd -P || printf '%s' "${VIRTUAL_ENV}")"
_fv_want="$(cd "${FV_VENV}" 2>/dev/null && pwd -P || printf '%s' "${FV_VENV}")"
# Resolved before comparing: an alias or a trailing slash would otherwise make this "leave" the very
# venv it is about to re-enter, which works but reports a move that did not happen.
if [ -n "${_fv_active}" ] && [ "${_fv_active}" != "${_fv_want}" ]; then
    _fv_note "leaving venv ${VIRTUAL_ENV}"
    if command -v deactivate >/dev/null 2>&1; then
        deactivate 2>/dev/null || true
    fi
    # deactivate may be absent (a bare PATH edit) or may have been clobbered by a second activate,
    # so strip the directory by hand as well. Both are safe when the other already worked.
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        _fv_stale="${VIRTUAL_ENV}/bin"
        PATH="$(printf '%s' "${PATH}" | tr ':' '\n' | grep -vxF "${_fv_stale}" | paste -sd: -)"
        export PATH
        unset VIRTUAL_ENV _fv_stale
    fi
fi
unset _fv_active _fv_want
# uv reads these to decide which environment to use, and they outlive a deactivate.
unset UV_PROJECT_ENVIRONMENT UV_PYTHON CONDA_PREFIX PYTHONHOME

# `uv run` prefers a project .venv over anything active, so it would quietly reintroduce the
# interpreter this script just left. Worth naming rather than deleting: it may be deliberate.
if [ -d "${FV_REPO_DIR}/.venv" ]; then
    _fv_note "note: ${FV_REPO_DIR}/.venv exists, and \`uv run\` prefers it over ${FV_VENV}."
    _fv_note "      use plain \`python\` / \`torchrun\`, or \`uv run --no-project\`, or remove it."
fi

# --- 2. paths, caches, and the image venv --------------------------------------------------------
if [ -f "${FV_REPO_DIR}/scripts/node_env.sh" ]; then
    # shellcheck source=/dev/null
    . "${FV_REPO_DIR}/scripts/node_env.sh"
else
    _fv_fail "${FV_REPO_DIR}/scripts/node_env.sh not found; is FV_REPO_DIR right?"
fi

# --- 3. the launch environment (NODE_ENVIRONMENT.md) ---------------------------------------------
# gIB is how NCCL crosses the two nodes. Sourcing this sets NCCL_CONF_FILE, whose first line is
# NCCL_NET=gIB -- a forced network with no fallback, which is why the libibverbs check below is not
# optional.
if [ -f /usr/local/gib/scripts/set_nccl_env.sh ]; then
    # shellcheck source=/dev/null
    . /usr/local/gib/scripts/set_nccl_env.sh >/dev/null 2>&1 || true
    export LD_LIBRARY_PATH="/usr/local/gib/lib64:${LD_LIBRARY_PATH:-}"
fi
# Keeps the watchdog from killing a rank during the long first load.
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false
# AOT/inductor is not torch.compile; these are the switches PyTorch actually reads.
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
# Each of these looked like a fix for the gIB failure and is not; see NODE_ENVIRONMENT.md. Leaving
# one set costs bandwidth or NVLink silently.
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_NET_PLUGIN FASTVIDEO_NCCL_SO_PATH LD_PRELOAD

cd "${FV_REPO_DIR}" 2>/dev/null || _fv_fail "cannot cd to ${FV_REPO_DIR}"

# --- 4. verify ------------------------------------------------------------------------------------
_fv_python="$(command -v python 2>/dev/null)"
# Resolved on both sides, for the same reason the Python-side check below is: an alias anywhere in
# the path must not be reported as the wrong interpreter.
_fv_python_dir="$(cd "$(dirname "${_fv_python:-/nonexistent}")" 2>/dev/null && pwd -P)"
_fv_venv_dir="$(cd "${FV_VENV}/bin" 2>/dev/null && pwd -P)"
if [ -z "${_fv_python}" ] || [ "${_fv_python_dir}" != "${_fv_venv_dir}" ]; then
    _fv_fail "python is ${_fv_python:-missing}, expected ${FV_VENV}/bin/python"
else
    _fv_note "python          ${_fv_python}"
fi
unset _fv_python_dir _fv_venv_dir

# One interpreter call for all three, because the interesting failure is the combination: a torch
# that imports from somewhere other than the venv, or a fastvideo that is not this checkout.
_fv_report="$(python - "${FV_REPO_DIR}" "${FV_VENV}" <<'PY' 2>&1
import os, sys

# Both sides go through realpath. A check meant to catch a silently wrong import is worse than
# nothing if it also fires whenever a symlink sits anywhere in the path: /workspace is a real
# directory on these nodes, but FV_REPO_DIR can be handed in via one, and a false alarm here
# teaches you to skip reading the line that matters.
def under(path: str, root: str) -> bool:
    path, root = os.path.realpath(path), os.path.realpath(root)
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)

repo, venv = sys.argv[1], sys.argv[2]
lines, bad = [], 0
try:
    import torch
    lines.append(f"torch           {torch.__version__}  cuda {torch.version.cuda}  gpus {torch.cuda.device_count()}")
    if not under(torch.__file__, venv):
        lines.append(f"PROBLEM: torch imports from {torch.__file__}, outside {venv}")
        bad += 1
except Exception as error:
    lines.append(f"PROBLEM: import torch failed: {error}")
    bad += 1
try:
    import fastvideo
    path = fastvideo.__file__ or ""
    lines.append(f"fastvideo       {path}")
    # The image carries a second copy at /FastVideo. Importing it is silent and ignores every edit.
    if not under(path, repo):
        lines.append(f"PROBLEM: fastvideo is not this checkout; expected under {repo}")
        bad += 1
except Exception as error:
    lines.append(f"PROBLEM: import fastvideo failed: {error}")
    bad += 1
print("\n".join(lines))
sys.exit(1 if bad else 0)
PY
)"
printf '%s\n' "${_fv_report}" | sed 's/^/  /'
# Count the lines, not whether any matched: "torch is missing" and "fastvideo is the wrong copy"
# are two separate things to fix and the summary should say so.
_fv_problems=$((_fv_problems + $(printf '%s\n' "${_fv_report}" | grep -c 'PROBLEM')))
unset _fv_report _fv_python

# gIB dlopens libibverbs.so.1 and the image does not ship it. With NCCL_NET forced to gIB there is
# no fallback, and every rank dies in ncclCommInitRank naming neither the plugin nor the library.
if [ -n "${NCCL_CONF_FILE:-}" ] && grep -qs "NCCL_NET=gIB" "${NCCL_CONF_FILE}"; then
    if ldconfig -p 2>/dev/null | grep -q 'libibverbs\.so\.1'; then
        _fv_note "gIB             NCCL_NET=gIB, libibverbs.so.1 present"
    else
        _fv_fail "NCCL_NET=gIB is forced but libibverbs.so.1 is missing. On BOTH nodes:"
        _fv_note "    apt-get update && apt-get install -y libibverbs1 ibverbs-providers ibverbs-utils && ldconfig"
    fi
fi

# Rendezvous is assigned by the platform, not chosen. A login shell does not always carry it, and
# torchrun does not read PET_* as its own flags -- a bare torchrun is a one-node job whatever the
# config asks for.
if [ -n "${PET_MASTER_ADDR:-}" ] && [ -n "${PET_NODE_RANK:-}" ]; then
    _fv_note "rendezvous      node_rank ${PET_NODE_RANK} -> ${PET_MASTER_ADDR}:${PET_MASTER_PORT:-?}"
else
    _fv_note "rendezvous      PET_* not set: single-node (--standalone) only"
fi

if [ "${_fv_problems}" -eq 0 ]; then
    echo "  ok. Two nodes: pass --nnodes 2 --node_rank \$PET_NODE_RANK --master_addr \$PET_MASTER_ADDR"
else
    echo "  ${_fv_problems} problem(s) above. See NODE_ENVIRONMENT.md."
fi
unset _fv_problems
unset -f _fv_note _fv_fail 2>/dev/null || true
