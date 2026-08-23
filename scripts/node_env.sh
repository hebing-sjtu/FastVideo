# Sourced by bootstrap_node.sh and ~/.bashrc on training nodes.
# Override any variable before sourcing, or export it in the shell.
#
# uv lives in ~/.local/bin on the official image and is NOT on PATH until
# .bashrc is loaded. Web consoles often skip that, so we put it on PATH here.

: "${FV_REPO_DIR:=/workspace/FastVideo}"
: "${FV_DATA_ROOT:=/data}"
: "${FV_LOCAL_ROOT:=/workspace}"
: "${FV_VENV:=/opt/venv}"

export FV_REPO_DIR FV_DATA_ROOT FV_LOCAL_ROOT FV_VENV

# Weights must stay on local disk. safetensors loads via mmap, and mmap over
# the gcsfuse mount at FV_DATA_ROOT turns one load into thousands of random
# GCS reads — it hangs, and a mount hiccup faults the process with SIGBUS.
export HF_HOME="${HF_HOME:-${FV_LOCAL_ROOT}/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${FV_REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Official image: uv installer → $HOME/.local/bin; venv → /opt/venv.
export PATH="${HOME}/.local/bin:/root/.local/bin:${FV_VENV}/bin:${PATH}"

if [ -f "${HOME}/.local/bin/env" ]; then
    # shellcheck source=/dev/null
    . "${HOME}/.local/bin/env"
elif [ -f /root/.local/bin/env ]; then
    # shellcheck source=/dev/null
    . /root/.local/bin/env
fi

if [ -f "${FV_VENV}/bin/activate" ]; then
    # shellcheck source=/dev/null
    . "${FV_VENV}/bin/activate"
fi
