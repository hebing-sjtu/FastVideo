#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Fan one encode manifest across one or more GPU nodes, one process per local GPU.
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
#   NUM_SHARDS   processes to start on this node. Defaults to every visible GPU. Every node in a
#                multi-node run must use the same value.
#   NODE_COUNT   nodes sharing the manifest and output directory (default 1).
#   NODE_RANK    this node's zero-based rank in NODE_COUNT (default 0). The global shard index is
#                NODE_RANK * NUM_SHARDS + local GPU index, so two 8-GPU nodes cover 0..15 once.
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
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
if ! [[ "$NODE_COUNT" =~ ^[0-9]+$ ]] || ((NODE_COUNT < 1)); then
    echo "NODE_COUNT must be a positive integer, got '$NODE_COUNT'." >&2
    exit 1
fi
if ! [[ "$NODE_RANK" =~ ^[0-9]+$ ]] || ((NODE_RANK < 0 || NODE_RANK >= NODE_COUNT)); then
    echo "NODE_RANK must be in [0, NODE_COUNT), got '$NODE_RANK' for NODE_COUNT=$NODE_COUNT." >&2
    exit 1
fi
GLOBAL_SHARDS=$((NODE_COUNT * NUM_SHARDS))
SHARD_BASE=$((NODE_RANK * NUM_SHARDS))
STAGGER_SEC="${STAGGER_SEC:-45}"
LOG_DIR="${LOG_DIR:-$(dirname "$output")/encode_logs}"
mkdir -p "$LOG_DIR" "$output"

echo "Encoding into $output on node $NODE_RANK/$NODE_COUNT with $NUM_SHARDS local process(es)."
echo "Global shards: $SHARD_BASE..$((SHARD_BASE + NUM_SHARDS - 1)) of $GLOBAL_SHARDS; " \
     "${STAGGER_SEC}s apart. Logs: $LOG_DIR"

pids=()
global_shards=()
trap 'echo; echo "Interrupted; stopping shards."; kill "${pids[@]}" 2>/dev/null; exit 130' INT TERM

for ((local_shard = 0; local_shard < NUM_SHARDS; local_shard++)); do
    global_shard=$((SHARD_BASE + local_shard))
    log="$LOG_DIR/shard_${global_shard}.log"
    CUDA_VISIBLE_DEVICES="$local_shard" python "$encoder" \
        --shard-index "$global_shard" --num-shards "$GLOBAL_SHARDS" "$@" >"$log" 2>&1 &
    pid=$!
    pids+=("$pid")
    global_shards+=("$global_shard")
    echo "  global shard $global_shard -> local GPU $local_shard, pid $pid, log $log"
    if ((local_shard + 1 < NUM_SHARDS)); then
        sleep "$STAGGER_SEC"
    fi
done

echo
echo "All local shards started. Follow one with: tail -f $LOG_DIR/shard_${SHARD_BASE}.log"
echo

# Every shard is waited on even after one fails. The others are doing useful work, and a shard that
# dies has left its finished clips on disk for the next run to skip.
status=0
for ((local_shard = 0; local_shard < NUM_SHARDS; local_shard++)); do
    global_shard="${global_shards[local_shard]}"
    set +e
    wait "${pids[local_shard]}"
    code=$?
    set -e
    if [[ "$code" -eq 0 ]]; then
        echo "global shard $global_shard: ok"
    else
        status=1
        echo "global shard $global_shard: FAILED (exit $code), tail of $LOG_DIR/shard_${global_shard}.log:"
        tail -n 15 "$LOG_DIR/shard_${global_shard}.log" | sed 's/^/    /'
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
        if ((NODE_COUNT == 1)); then
            echo "$((rows - written)) missing. Re-run this same command to retry only those, or grep the"
            echo "logs for FAILED to see which clips the encoder rejected and why."
            status=1
        else
            echo "$((rows - written)) not present yet. Other nodes may still be writing; after every node"
            echo "finishes, compare the cache against the manifest with describe_cache.py."
        fi
    fi
fi

exit "$status"
