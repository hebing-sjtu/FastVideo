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

set -euo pipefail

CONFIG=examples/train/scenario/h3_proxy/proxy_bd_finetune.yaml
CONFIG_GIVEN=""
CONFIG_SOURCE="$CONFIG"
NPROC=8
STEP=""
RUN=""
CACHE=""
CACHE_FROM=""
VAL_JSON=""
VAL_JSON_FROM=""
OUT=""
TAG=""

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
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

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
# a manifest instead of a syntax error.
python - "$OUT/eval_manifest.json" \
    "$STEP" "$CKPT" "$CONFIG_SOURCE" "$CACHE" "$GEOM" "$VAL_JSON" "$EVERY" "$NPROC" <<'PY'
import json, sys

out, step, ckpt, config_source, cache, geometry, val_json, every, nproc = sys.argv[1:10]
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
        },
        handle,
        indent=2,
        sort_keys=True,
    )
PY

set -x
torchrun --standalone --nproc_per_node "$NPROC" \
    -m fastvideo.train.entrypoint.train \
    --config "$CONFIG" \
    --training.data.data_path "$CACHE" \
    --training.distributed.num_gpus "$NPROC" \
    --training.distributed.sp_size "$NPROC" \
    --training.distributed.hsdp_shard_dim "$NPROC" \
    "${RESUME[@]}" \
    --training.loop.max_train_steps "$STEP" \
    --callbacks.validation.dataset_file "$VAL_JSON" \
    --callbacks.validation.every_steps "$EVERY" \
    --callbacks.validation.run_at_start true \
    --training.checkpoint.output_dir "$OUT" \
    --training.tracker.run_name "eval_$(basename "$CACHE")_step${STEP}${TAG:+_$TAG}" \
    $GEOM
