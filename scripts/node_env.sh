# Sourced by bootstrap_node.sh and ~/.bashrc on training nodes.
# Override any variable before sourcing, or export it in the shell.

: "${FV_REPO_DIR:=/workspace/FastVideo}"
: "${FV_DATA_ROOT:=/data}"
: "${FV_VENV:=/opt/venv}"

export FV_REPO_DIR FV_DATA_ROOT FV_VENV
export HF_HOME="${HF_HOME:-${FV_DATA_ROOT}/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${FV_REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -f "${FV_VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${FV_VENV}/bin/activate"
fi
