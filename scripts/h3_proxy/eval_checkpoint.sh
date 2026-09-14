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
NPROC=8
STEP=""
RUN=""
CACHE=""
VAL_JSON=""
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
        --config) CONFIG="$2"; shift 2 ;;
        --nproc) NPROC="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$STEP" ]] || { echo "--step is required." >&2; exit 2; }
[[ -n "$CACHE" ]] || { echo "--cache is required." >&2; exit 2; }
[[ -n "$VAL_JSON" ]] || { echo "--val-json is required." >&2; exit 2; }
if [[ ! "$STEP" =~ ^[0-9]+$ ]]; then
    echo "--step must be a non-negative integer, got '$STEP'." >&2
    exit 2
fi
if [[ "$STEP" -gt 0 && -z "$RUN" ]]; then
    echo "--run is required for --step $STEP; only --step 0 has no checkpoint to resume." >&2
    exit 2
fi
if [[ ! -f "$VAL_JSON" ]]; then
    echo "--val-json not found: $VAL_JSON" >&2
    exit 2
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

# A live training directory prunes by highest step, so writing a final checkpoint into it would
# delete the early ones this eval exists to inspect.
[[ -n "$OUT" ]] || OUT="/data/binghe/h3_proxy/runs/eval_$(basename "$CACHE")_step${STEP}${TAG:+_$TAG}/checkpoints"
if [[ -n "$RUN" && "$(cd "$(dirname "$OUT")" 2>/dev/null && pwd || echo "$OUT")" == "$(cd "$RUN" 2>/dev/null && pwd || echo "$RUN")" ]]; then
    echo "--out must not be the training run's own checkpoints dir; it would prune the early checkpoints." >&2
    exit 2
fi

# Any positive value divides 0, so step 0 needs no special case beyond avoiding a zero divisor.
EVERY=$(( STEP > 0 ? STEP : 1 ))

GEOM="$(python scripts/h3_proxy/describe_cache.py "$CACHE" --emit-flags)"

echo "eval-only:"
echo "  step            $STEP"
echo "  checkpoint      ${CKPT:-<none: base model, LoRA is zero>}"
echo "  cache           $CACHE"
echo "  geometry        $GEOM"
echo "  validation set  $VAL_JSON"
echo "  output          $OUT"
echo "  every_steps     $EVERY   (must divide $STEP)"
echo

RESUME=()
if [[ -n "$CKPT" ]]; then
    RESUME=(--training.checkpoint.resume_from_checkpoint "$CKPT")
fi

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
