#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Fan one encode manifest across the GPUs on this node, one process per GPU.
#
# `encode_proxy_samples.py` already has the two properties this needs. It shards deterministically
# -- shard i takes entries[i::n] -- and it skips a clip whose `.pt` already exists, so re-running
# this after an interruption resumes instead of redoing the work. Nothing here tracks progress.
#
# What the encoder does not do is place itself on a GPU. `--device cuda` resolves to cuda:0 in
# every process, so the placement has to come from CUDA_VISIBLE_DEVICES out here; each shard then
# sees one GPU and the default `--device cuda` is already right.
#
# Every argument is forwarded to the encoder untouched, except `--shard-index` and `--num-shards`,
# which this script owns.
#
#   scripts/h3_proxy/prepare_data/encode_proxy_shards.sh \
#       --manifest /data/binghe/h3_proxy/abot_train.jsonl \
#       --root /data/binghe/datasets/ABot-sub-2000-clips \
#       --output /data/binghe/h3_proxy/cache/abot_train \
#       --model-path /data/models/MiniMax-H3 \
#       --anchor-short-edge 2048 --proxy-height 192 --proxy-width 336
#
# Environment:
#   NUM_SHARDS   processes to start. Defaults to every visible GPU.
#   STAGGER_SEC  delay between starts, default 45. Each shard deserialises its own copy of a ~64 GB
#                bf16 Qwen3-VL text encoder, so eight simultaneous starts are one ~500 GB read
#                burst and a matching host-RAM spike. A few staggered minutes is cheap against a
#                multi-hour job, and the GPUs are idle during that window anyway.
#   LOG_DIR      per-shard logs. Defaults to `<output>/../encode_logs`.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
encoder="$script_dir/encode_proxy_samples.py"
if [[ ! -f "$encoder" ]]; then
    echo "Cannot find encode_proxy_samples.py beside this script ($encoder)." >&2
    exit 1
fi

# Read --manifest and --output back out of the forwarded arguments, for the log directory default
# and the completion count. Both spellings, since either is a reasonable thing to type.
manifest=""
output=""
previous=""
for argument in "$@"; do
    case "$previous" in
        --manifest) manifest="$argument" ;;
        --output) output="$argument" ;;
    esac
    case "$argument" in
        --manifest=*) manifest="${argument#*=}" ;;
        --output=*) output="${argument#*=}" ;;
    esac
    previous="$argument"
done
if [[ -z "$output" ]]; then
    echo "--output is required so the shards agree on a cache directory." >&2
    exit 1
fi

if [[ -z "${NUM_SHARDS:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        NUM_SHARDS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
    else
        NUM_SHARDS=1
    fi
fi
if [[ "$NUM_SHARDS" -lt 1 ]]; then
    echo "Found no GPUs to shard across. Set NUM_SHARDS explicitly to override." >&2
    exit 1
fi
STAGGER_SEC="${STAGGER_SEC:-45}"
LOG_DIR="${LOG_DIR:-$(dirname "$output")/encode_logs}"
mkdir -p "$LOG_DIR" "$output"

echo "Encoding into $output with $NUM_SHARDS shard(s), ${STAGGER_SEC}s apart. Logs: $LOG_DIR"

pids=()
trap 'echo; echo "Interrupted; stopping shards."; kill "${pids[@]}" 2>/dev/null; exit 130' INT TERM

for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    log="$LOG_DIR/shard_${shard}.log"
    CUDA_VISIBLE_DEVICES="$shard" python "$encoder" \
        --shard-index "$shard" --num-shards "$NUM_SHARDS" "$@" >"$log" 2>&1 &
    pid=$!
    pids+=("$pid")
    echo "  shard $shard -> GPU $shard, pid $pid, log $log"
    if ((shard + 1 < NUM_SHARDS)); then
        sleep "$STAGGER_SEC"
    fi
done

echo
echo "All shards started. Follow one with: tail -f $LOG_DIR/shard_0.log"
echo

# Every shard is waited on even after one fails. The others are doing useful work, and a shard that
# dies has left its finished clips on disk for the next run to skip.
status=0
for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    set +e
    wait "${pids[shard]}"
    code=$?
    set -e
    if [[ "$code" -eq 0 ]]; then
        echo "shard $shard: ok"
    else
        status=1
        echo "shard $shard: FAILED (exit $code), tail of $LOG_DIR/shard_${shard}.log:"
        tail -n 15 "$LOG_DIR/shard_${shard}.log" | sed 's/^/    /'
    fi
done

written=$(find "$output" -maxdepth 1 -name '*.pt' | wc -l | tr -d ' ')
echo
echo "Cache holds $written sample(s)."
if [[ -n "$manifest" && -f "$manifest" ]]; then
    rows=$(awk 'NF && $0 !~ /^#/' "$manifest" | wc -l | tr -d ' ')
    echo "Manifest has $rows row(s)."
    if [[ "$written" -lt "$rows" ]]; then
        # Re-running is the fix and it is safe: finished clips are skipped, so a second pass only
        # retries what is missing.
        echo "$((rows - written)) missing. Re-run this same command to retry only those, or grep the"
        echo "logs for FAILED to see which clips the encoder rejected and why."
        status=1
    fi
fi

exit "$status"
