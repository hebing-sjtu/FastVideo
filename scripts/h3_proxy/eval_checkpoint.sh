#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Sample one checkpoint's validation set and stop. No training step runs.
#
# This exists because the eval-only recipe is four settings that must agree, and every one of them
# fails quietly:
#
#   * `resume_from_checkpoint` absent -> the job starts at step 0 with a zero LoRA, runs the
#     validation, writes videos, and exits cleanly. The videos are the base model, so comparing them
#     against a step-0 eval shows no difference and invites the conclusion that nothing was learned.
#     Only the step baked into the filename says otherwise, and that is read after the sampling is
#     paid for.
#   * `max_train_steps` not equal to the resumed step -> the loop is not empty and the job trains.
#   * `every_steps` not dividing the resumed step -> `on_validation_begin` returns early and the job
#     finishes having logged nothing.
#   * the target canvas, proxy grid or anchor short edge not matching the cache -> conditioning on a
#     token grid the checkpoint never trained on, which reads as a bad checkpoint.
#
# The geometry comes from the cache via `describe_cache.py --emit-flags` rather than from arguments,
# so it cannot drift from what was encoded. Everything else is derived from --step.
#
# Usage::
#
#     scripts/h3_proxy/eval_checkpoint.sh \
#         --step 150 \
#         --run /data/binghe/h3_proxy/runs/gta_v2_cwm_lr4e4/checkpoints \
#         --cache /data/binghe/h3_proxy/cache/gta_v2_cwm \
#         --val-json /data/binghe/h3_proxy/gta_v2_validation_val6.json
#
# `--step 0` is the base-model baseline and is the one case that legitimately has no checkpoint.
#
# Anything after a bare `--` is passed through to the trainer as extra overrides, and recorded in the
# manifest so two runs can still be compared. That is how a regime the default config does not
# describe gets sampled -- notably CWM's wn contract, which the default w0 config would otherwise
# sample with one given frame against a prompt promising thirty-four::
#
#     scripts/h3_proxy/eval_checkpoint.sh --step 0 \
#         --cache /data/binghe/h3_proxy/cache/gta_v2_cwm_wn \
#         --val-json /data/binghe/h3_proxy/gta_v2_validation.json \
#         -- --models.student.num_given_latent_frames 10 \
#            --callbacks.validation.num_given_latent_frames 10 \
#            --callbacks.validation.cwm_system_prompt wn
#
# Defaults to one node of 8 with `--standalone`. On a two-node allocation --standalone would start
# two unrelated 8-GPU jobs writing one output directory, so pass `--nnodes 2` and run the same line
# on both nodes; the rendezvous comes from PET_* (NODE_ENVIRONMENT.md) because this cluster assigns
# it rather than letting the job choose:
#
#     scripts/h3_proxy/eval_checkpoint.sh --step 600 --nnodes 2 --sp-size 8 ...
#
# `--sp-size` defaults to one SP group per node. It is worth setting deliberately for two reasons:
# `num_gpus / sp_size` SP groups each sample a share of the validation set, so a smaller value
# finishes sooner; and a control trunk shards its heads across the SP group, so the value has to
# divide `controlnet_num_heads`. The 16 a two-node world would suggest divides neither this
# scenario's 8 heads nor anything useful, which is why sp_size is not simply the world size.

set -euo pipefail

CONFIG=examples/train/scenario/h3_proxy/proxy_bd_finetune.yaml
CONFIG_GIVEN=""
CONFIG_SOURCE="$CONFIG"
NPROC=8
# Two nodes cannot use --standalone, and on this cluster the rendezvous is handed to the job in
# PET_* rather than chosen. See NODE_ENVIRONMENT.md.
NNODES=1
# Decoupled from the world size, because it is not a free knob: a control trunk shards its heads
# across the SP group, so sp_size has to divide `controlnet_num_heads` -- 8 for this scenario, which
# rules out the 16 a two-node world would otherwise suggest. Empty means "one SP group per node".
SP_SIZE=""
STEP=""
RUN=""
CACHE=""
CACHE_FROM=""
VAL_JSON=""
VAL_JSON_FROM=""
OUT=""
TAG=""
LIST=""
RUNS_ROOT=/data/binghe/h3_proxy/runs
# Trainer overrides after a bare `--`. Last on the torchrun line, so they win over what this script
# derives -- the geometry included, which is worth knowing before overriding it.
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --step) STEP="$2"; shift 2 ;;
        --run) RUN="$2"; shift 2 ;;
        --cache) CACHE="$2"; shift 2 ;;
        --val-json) VAL_JSON="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --config) CONFIG="$2"; CONFIG_GIVEN=1; CONFIG_SOURCE="$2"; shift 2 ;;
        --nproc) NPROC="$2"; shift 2 ;;
        --nnodes) NNODES="$2"; shift 2 ;;
        --sp-size) SP_SIZE="$2"; shift 2 ;;
        --list) LIST=1; shift ;;
        --runs-root) RUNS_ROOT="$2"; shift 2 ;;
        --) shift; EXTRA=("$@"); break ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

# Which run is which is not answerable from directory names, which drift, but every checkpoint
# records the cache it trained on, and that is what separates one experiment from another.
if [[ -n "$LIST" ]]; then
    if [[ -n "$RUN" ]]; then
        echo "$RUN"
        for candidate in $(ls -d "$RUN"/checkpoint-* 2>/dev/null | sort -V); do
            if [[ -f "$candidate/dcp/.metadata" ]]; then
                printf '  %-18s  %s  loadable\n' "$(basename "$candidate")" \
                    "$(date -r "$candidate" '+%m-%d %H:%M' 2>/dev/null || echo '     ')"
            else
                printf '  %-18s  %s  incomplete\n' "$(basename "$candidate")" \
                    "$(date -r "$candidate" '+%m-%d %H:%M' 2>/dev/null || echo '     ')"
            fi
        done
        exit 0
    fi
    shopt -s nullglob
    for run in "$RUNS_ROOT"/*/checkpoints; do
        last=""
        for candidate in $(ls -d "$run"/checkpoint-* 2>/dev/null | sort -V); do
            [[ -f "$candidate/dcp/.metadata" ]] && last="$candidate"
        done
        [[ -n "$last" ]] || continue
        cache="$(python scripts/h3_proxy/probe_resume.py "$last" --emit training.data.data_path 2>/dev/null \
                 || echo '<unrecorded>')"
        printf '%-56s  last=%-18s  trained on %s\n' "${run#"$RUNS_ROOT"/}" "$(basename "$last")" "$cache"
    done
    echo
    echo "Then: $0 --list --run $RUNS_ROOT/<run>/checkpoints"
    exit 0
fi

[[ -n "$STEP" ]] || { echo "--step is required." >&2; exit 2; }
if [[ ! "$STEP" =~ ^[0-9]+$ ]]; then
    echo "--step must be a non-negative integer, got '$STEP'." >&2
    exit 2
fi
if [[ "$STEP" -gt 0 && -z "$RUN" ]]; then
    echo "--run is required for --step $STEP; only --step 0 has no checkpoint to resume." >&2
    exit 2
fi
# Step 0 has no checkpoint to read the run's own settings back out of, so it must be told them.
if [[ "$STEP" -eq 0 ]]; then
    [[ -n "$CACHE" ]] || { echo "--cache is required for --step 0, which has no checkpoint to read it from." >&2; exit 2; }
    [[ -n "$VAL_JSON" ]] || { echo "--val-json is required for --step 0." >&2; exit 2; }
fi

# Refuse before loading a 134 GB snapshot, rather than after.
CKPT=""
if [[ "$STEP" -gt 0 ]]; then
    CKPT="$RUN/checkpoint-$STEP"
    if [[ ! -d "$CKPT/dcp" ]]; then
        echo "No loadable checkpoint at $CKPT (needs dcp/). On disk:" >&2
        found=0
        for candidate in $(ls -d "$RUN"/checkpoint-* 2>/dev/null | sort -V); do
            found=1
            if [[ -f "$candidate/dcp/.metadata" ]]; then
                echo "  $(basename "$candidate")  loadable" >&2
            else
                echo "  $(basename "$candidate")  incomplete, a save that did not finish" >&2
            fi
        done
        [[ "$found" -eq 1 ]] || echo "  none" >&2
        exit 2
    fi
fi

# The checkpoint stores the whole training config, so what the run trained on is a fact on disk
# rather than something to remember. Evaluating against that config makes every setting that shapes
# a state-dict key agree by construction, which is the failure probe_resume.py otherwise only warns
# about, and picks up the cache and validation set the run actually used.
if [[ -n "$CKPT" ]]; then
    if [[ -z "$CONFIG_GIVEN" ]]; then
        # GNU mktemp wants the X's last, so name the file inside a temp dir rather than templating
        # a suffix onto it.
        CONFIG_DIR="$(mktemp -d)"
        trap 'rm -rf "$CONFIG_DIR"' EXIT
        CONFIG="$CONFIG_DIR/from_checkpoint_$STEP.yaml"
        python scripts/h3_proxy/probe_resume.py "$CKPT" --dump-config "$CONFIG" >/dev/null
        CONFIG_SOURCE="the config saved in checkpoint-$STEP"
    fi
    if [[ -z "$CACHE" ]]; then
        CACHE="$(python scripts/h3_proxy/probe_resume.py "$CKPT" --emit training.data.data_path)"
        CACHE_FROM="checkpoint-$STEP"
    fi
    if [[ -z "$VAL_JSON" ]]; then
        VAL_JSON="$(python scripts/h3_proxy/probe_resume.py "$CKPT" --emit callbacks.validation.dataset_file)"
        VAL_JSON_FROM="checkpoint-$STEP"
    fi
fi

# Everything below overrides `callbacks.validation.*`, and an override cannot bring a callback into
# being -- on its own it would construct one out of whichever fields happened to be overridden. A
# run that trained with the block deleted rather than merely switched off saves a config with no
# validation at all, so the config is the one thing here that can be unusable while the checkpoint
# is fine. Settle it before the GPUs, and name the way out.
if ! python -c '
import sys, yaml
config = yaml.safe_load(open(sys.argv[1])) or {}
sys.exit(0 if isinstance((config.get("callbacks") or {}).get("validation"), dict) else 1)
' "$CONFIG" 2>/dev/null; then
    echo "$CONFIG_SOURCE configures no validation callback, and this script samples by turning one" >&2
    echo "on. Point it at a scenario that declares the block instead:" >&2
    echo >&2
    echo "  --config examples/train/scenario/h3_proxy/proxy_controlnet_finetune.yaml" >&2
    echo >&2
    echo "Geometry still comes from the cache, and probe_resume.py still checks the state-dict key" >&2
    echo "names against whichever config is used, so this costs none of what the saved config gave." >&2
    exit 2
fi

if [[ ! -d "$CACHE" ]]; then
    echo "cache directory not found: $CACHE${CACHE_FROM:+  (from $CACHE_FROM)}" >&2
    echo "Pass --cache. Caches on disk:" >&2
    ls -d /data/binghe/h3_proxy/cache/*/ 2>/dev/null | sed 's/^/  /' >&2 || echo "  none" >&2
    exit 2
fi
if [[ ! -f "$VAL_JSON" ]]; then
    echo "validation set not found: $VAL_JSON${VAL_JSON_FROM:+  (from $VAL_JSON_FROM)}" >&2
    if [[ -n "$VAL_JSON_FROM" ]]; then
        # A run that kept validation off never read this field, so its saved value can be a default
        # that was never true. Everything else in the config was exercised by training; this one
        # alone can be fiction.
        echo "That is what the run configured, but a run with validation off never reads it, so the" >&2
        echo "value can be an untouched default. Pass --val-json. Validation sets on disk:" >&2
    else
        echo "Pass an existing --val-json. Validation sets on disk:" >&2
    fi
    ls /data/binghe/h3_proxy/*validation*.json 2>/dev/null | sed 's/^/  /' >&2 || echo "  none" >&2
    echo "  (build one with scripts/h3_proxy/prepare_data/seg_dir_to_validation_json.py)" >&2
    exit 2
fi

# A live training directory prunes by highest step, so writing a final checkpoint into it would
# delete the early ones this eval exists to inspect.
[[ -n "$OUT" ]] || OUT="/data/binghe/h3_proxy/runs/eval_$(basename "$CACHE")_step${STEP}${TAG:+_$TAG}/checkpoints"
if [[ -n "$RUN" && "$(cd "$(dirname "$OUT")" 2>/dev/null && pwd || echo "$OUT")" == "$(cd "$RUN" 2>/dev/null && pwd || echo "$RUN")" ]]; then
    echo "--out must not be the training run's own checkpoints dir; it would prune the early checkpoints." >&2
    exit 2
fi

# Any positive value divides 0, so step 0 needs no special case beyond avoiding a zero divisor.
# gIB dlopens libibverbs.so.1, and the container image does not ship it. set_nccl_env.sh pins
# NCCL_NET=gIB, and a forced network with no usable devices has no fallback, so this fails on one
# node as readily as on two -- as `ncclCommInitRank: internal error` from init_world_group, which
# names neither gIB nor libibverbs. The real cause prints at INFO, three lines above the first WARN,
# so NCCL_DEBUG=WARN hides it. It has to be reinstalled after every container restart, which makes
# it worth two seconds here. See NODE_ENVIRONMENT.md.
if [[ -n "${NCCL_CONF_FILE:-}" ]] && grep -qs "NCCL_NET=gIB" "${NCCL_CONF_FILE:-/dev/null}"; then
    if ! ldconfig -p 2>/dev/null | grep -q "libibverbs\.so\.1"; then
        echo "NCCL_CONF_FILE forces NCCL_NET=gIB, but libibverbs.so.1 is not installed. gIB dlopens it" >&2
        echo "at runtime and a forced network with no devices has no fallback, so every rank will die in" >&2
        echo "ncclCommInitRank with 'internal error' and name neither. On BOTH nodes:" >&2
        echo "  apt-get update && apt-get install -y libibverbs1 ibverbs-providers ibverbs-utils && ldconfig" >&2
        echo "  ibv_devinfo | grep -E 'hca_id|state'    # expect mlx5_0..7, PORT_ACTIVE" >&2
        echo "ibverbs-providers is not optional: without libmlx5 the library opens and enumerates zero" >&2
        echo "devices, which fails identically." >&2
        exit 2
    fi
fi

EVERY=$(( STEP > 0 ? STEP : 1 ))

# --- mesh ---------------------------------------------------------------------------------------
#
# `nnodes * nproc_per_node == num_gpus == hsdp_replicate_dim * hsdp_shard_dim`. sp_size is the one
# of these that is not determined by the hardware, and it is not a pure throughput knob either:
# `num_gpus / sp_size` SP groups each sample a share of the validation set, so lowering it finishes
# sooner, while the trunk's head split puts a floor under how low it can go.
WORLD=$(( NNODES * NPROC ))
[[ -n "$SP_SIZE" ]] || SP_SIZE="$NPROC"
if (( WORLD % SP_SIZE )); then
    echo "--sp-size $SP_SIZE does not divide the $WORLD GPUs ($NNODES nodes x $NPROC)." >&2
    exit 2
fi

# The trunk scatters its heads across the SP group, so a world size that divides the backbone's 56
# heads can still leave the trunk unable to split its own -- and `sp_size: 16` on two nodes is
# exactly that case for this scenario's 8. The model raises with a list of valid sizes, but only
# after the snapshot is on the GPUs, so settle it against the config instead.
CONTROL_HEADS="$(python -c '
import sys, yaml
student = ((yaml.safe_load(open(sys.argv[1])) or {}).get("models") or {}).get("student") or {}
print(int(student.get("controlnet_num_heads", 8)) if student.get("enable_camera_controlnet") else 0)
' "$CONFIG" 2>/dev/null || echo 0)"
if [[ "$CONTROL_HEADS" -gt 0 ]] && (( CONTROL_HEADS % SP_SIZE )); then
    echo "This checkpoint carries a control trunk with controlnet_num_heads=$CONTROL_HEADS, and the trunk" >&2
    echo "shards its heads across the SP group, so --sp-size must divide $CONTROL_HEADS -- not $SP_SIZE." >&2
    printf '  valid: ' >&2
    for candidate in $(seq 1 "$CONTROL_HEADS"); do
        (( CONTROL_HEADS % candidate )) || (( WORLD % candidate )) || printf '%s ' "$candidate" >&2
    done
    echo >&2
    echo "  $WORLD GPUs / sp_size = the number of SP groups, each sampling a share of the validation set." >&2
    exit 2
fi

# Both nodes run the same command and only PET_NODE_RANK differs; --standalone would instead start
# two unrelated 8-GPU jobs writing one output directory.
LAUNCH=(--standalone --nproc_per_node "$NPROC")
if [[ "$NNODES" -gt 1 ]]; then
    for variable in PET_MASTER_ADDR PET_MASTER_PORT PET_NODE_RANK; do
        if [[ -z "${!variable:-}" ]]; then
            echo "--nnodes $NNODES needs \$$variable, which this cluster sets per node and a login shell may not" >&2
            echo "carry. Check: echo \$PET_MASTER_ADDR \$PET_MASTER_PORT \$PET_NODE_RANK   (see NODE_ENVIRONMENT.md)" >&2
            exit 2
        fi
    done
    LAUNCH=(--nnodes "$NNODES" --nproc_per_node "$NPROC"
            --node_rank "$PET_NODE_RANK"
            --master_addr "$PET_MASTER_ADDR" --master_port "$PET_MASTER_PORT")
fi

GEOM="$(python scripts/h3_proxy/describe_cache.py "$CACHE" --emit-flags)"

# A resume whose key names disagree with this config loads nothing and says nothing, so settle that
# before the GPUs are touched. When the config came from the checkpoint this can only agree, and
# saying so is still worth the two seconds: it also reports how many LoRA weights are in the save.
if [[ -n "$CKPT" ]]; then
    python scripts/h3_proxy/probe_resume.py "$CKPT" --config "$CONFIG"
    echo
fi

echo "eval-only:"
echo "  step            $STEP"
echo "  checkpoint      ${CKPT:-<none: base model, LoRA is zero>}"
echo "  config          $CONFIG_SOURCE"
echo "  cache           $CACHE"
echo "  geometry        $GEOM"
echo "  validation set  $VAL_JSON"
echo "  output          $OUT"
echo "  every_steps     $EVERY   (must divide $STEP)"
echo "  mesh            $NNODES node(s) x $NPROC = $WORLD GPUs, sp_size $SP_SIZE -> $(( WORLD / SP_SIZE )) SP group(s)"
echo
echo "  Watch for 'lora_B norm 0 -> <nonzero>' from the resume; that line is the proof the"
echo "  checkpoint's weights reached the model that samples. It raises rather than logs if the"
echo "  norm is still zero."
echo

RESUME=()
if [[ -n "$CKPT" ]]; then
    RESUME=(--training.checkpoint.resume_from_checkpoint "$CKPT")
fi

# Two eval runs are only comparable if everything except the checkpoint was the same, and an mp4
# does not record what produced it. Without this, "the videos differ" cannot be told apart from
# "the runs sampled different clips at a different geometry", which is the mistake that made the
# step-0/step-138 comparison meaningless in the first place.
mkdir -p "$OUT"
# Values travel as argv rather than interpolated into the source, so a path holding a quote writes
# a manifest instead of a syntax error. Only the rendezvous leader writes: on two nodes both run
# this same line, and two processes truncating one file is a race whose loser leaves it empty.
if [[ "${PET_NODE_RANK:-0}" -eq 0 ]]; then
python - "$OUT/eval_manifest.json" \
    "$STEP" "$CKPT" "$CONFIG_SOURCE" "$CACHE" "$GEOM" "$VAL_JSON" "$EVERY" "$NPROC" "$NNODES" "$SP_SIZE" \
    ${EXTRA[@]+"${EXTRA[@]}"} <<'PY'
import json, sys

out, step, ckpt, config_source, cache, geometry, val_json, every, nproc, nnodes, sp_size = sys.argv[1:12]
with open(out, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "step": int(step),
            "checkpoint": ckpt,
            "config_source": config_source,
            "cache": cache,
            "geometry": geometry,
            "val_json": val_json,
            "every_steps": int(every),
            "nproc": int(nproc),
            "nnodes": int(nnodes),
            # Recorded for the same reason as the geometry: sequence parallelism changes where the
            # attention is split, so two evals at different sp_size are not bit-comparable -- and
            # compare_eval_predictions.py reads pixels.
            "sp_size": int(sp_size),
            # Recorded because they change what was sampled. A wn baseline and a w0 baseline are
            # both "step 0" and are not the same picture, and the mp4 does not say which it is.
            "extra_overrides": sys.argv[12:],
        },
        handle,
        indent=2,
        sort_keys=True,
    )
PY
fi

set -x
torchrun "${LAUNCH[@]}" \
    -m fastvideo.train.entrypoint.train \
    --config "$CONFIG" \
    --training.data.data_path "$CACHE" \
    --training.distributed.num_gpus "$WORLD" \
    --training.distributed.sp_size "$SP_SIZE" \
    --training.distributed.hsdp_shard_dim "$WORLD" \
    "${RESUME[@]}" \
    --training.loop.max_train_steps "$STEP" \
    --callbacks.validation.dataset_file "$VAL_JSON" \
    --callbacks.validation.every_steps "$EVERY" \
    --callbacks.validation.run_at_start true \
    --training.checkpoint.output_dir "$OUT" \
    --training.tracker.run_name "eval_$(basename "$CACHE")_step${STEP}${TAG:+_$TAG}" \
    $GEOM ${EXTRA[@]+"${EXTRA[@]}"}
