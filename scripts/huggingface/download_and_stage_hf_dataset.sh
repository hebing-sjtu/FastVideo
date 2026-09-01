#!/usr/bin/env bash
# Download a large HF dataset onto fast local scratch, then fan out a parallel
# rsync to its final home.
#
# The two stages are deliberate: on nodes where download bandwidth outruns what
# ceph-fuse can absorb, writing the download stream straight to the shared mount
# drops the connection. Local scratch absorbs the burst; rsync then moves bytes
# at a rate the shared mount can sustain.

set -euo pipefail

REPO="acvlab/ABot-World-Explorer-500h"
REPO_TYPE="dataset"
STAGE_DIR="/tmp/ABot-World-Explorer-500h"
DEST_DIR="/data/binghe/datasets/ABot-World-Explorer-500h"
JOBS=128
DOWNLOAD_WORKERS=16
MAX_ATTEMPTS=20
RUN_DOWNLOAD=1
RUN_COPY=1
INSTALL_DEPS=0
SKIP_SPACE_CHECK=0
MIN_HUB_VERSION="0.24.0"

usage() {
    cat <<'EOF'
Usage: download_and_stage_hf_dataset.sh [options]

  --repo ID              HF repo id            (default: acvlab/ABot-World-Explorer-500h)
  --repo-type TYPE       dataset | model       (default: dataset)
  --stage-dir PATH       local scratch dir     (default: /tmp/ABot-World-Explorer-500h)
  --dest-dir PATH        final destination     (default: /data/binghe/datasets/ABot-World-Explorer-500h)
  --jobs N               parallel rsync procs  (default: 128)
  --download-workers N   hf download workers   (default: 16)
  --max-attempts N       download retries      (default: 20)
  --download-only        stage the download, skip the rsync
  --copy-only            skip the download, only rsync stage -> dest
  --install-deps         pip install/upgrade huggingface_hub + hf_transfer
  --skip-space-check     do not refuse to start on a tight filesystem
  -h, --help             show this message

The download stage is resumable: rerunning skips files already present in the
stage dir, so an interrupted transfer just needs the same command again.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)             REPO="$2"; shift 2 ;;
        --repo-type)        REPO_TYPE="$2"; shift 2 ;;
        --stage-dir)        STAGE_DIR="$2"; shift 2 ;;
        --dest-dir)         DEST_DIR="$2"; shift 2 ;;
        --jobs)             JOBS="$2"; shift 2 ;;
        --download-workers) DOWNLOAD_WORKERS="$2"; shift 2 ;;
        --max-attempts)     MAX_ATTEMPTS="$2"; shift 2 ;;
        --download-only)    RUN_COPY=0; shift ;;
        --copy-only)        RUN_DOWNLOAD=0; shift ;;
        --install-deps)     INSTALL_DEPS=1; shift ;;
        --skip-space-check) SKIP_SPACE_CHECK=1; shift ;;
        -h|--help)          usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { printf '[%s] ERROR: %s\n' "$(date '+%F %T')" "$*" >&2; exit 1; }

WORKLIST=""
KEEP_WORKLIST=0
cleanup() {
    if [[ -n "$WORKLIST" && "$KEEP_WORKLIST" -eq 0 ]]; then
        rm -rf "$WORKLIST"
    fi
}
trap cleanup EXIT

PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python
command -v "$PYTHON" >/dev/null 2>&1 || die "no python interpreter found"

# Walk up to the nearest existing ancestor so df works on a path we intend to create.
existing_ancestor() {
    local p="$1"
    while [[ ! -d "$p" && "$p" != "/" ]]; do p="$(dirname "$p")"; done
    printf '%s' "$p"
}

avail_bytes() {
    local target out
    target="$(existing_ancestor "$1")"
    out="$(df -B1 --output=avail "$target" 2>/dev/null | tail -1 | tr -d '[:space:]')" || out=""
    if [[ ! "$out" =~ ^[0-9]+$ ]]; then
        out="$(df -k "$target" | awk 'NR==2 {printf "%.0f", $4 * 1024}')"
    fi
    printf '%s' "${out:-0}"
}

dir_bytes() {
    du -sb --exclude=.cache "$1" 2>/dev/null | awk '{print $1}' || echo 0
}

human() {
    "$PYTHON" - "$1" <<'PY'
import sys
n = float(sys.argv[1])
for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
    if abs(n) < 1024 or unit == "PiB":
        print(f"{n:.1f} {unit}")
        break
    n /= 1024
PY
}

# Ask the Hub how big the snapshot is so we can refuse to start a doomed transfer.
remote_size_bytes() {
    local url="https://huggingface.co/api/${REPO_TYPE}s/${REPO}/treesize/main"
    curl -sfL --max-time 30 "$url" 2>/dev/null \
        | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin).get("size", 0))' 2>/dev/null \
        || echo 0
}

check_space() {
    local path="$1" need="$2" label="$3"
    if [[ ! "$need" =~ ^[0-9]+$ ]] || [[ "$need" -le 0 ]]; then
        log "unknown required size, skipping $label space check"
        return 0
    fi
    local free
    free="$(avail_bytes "$path")"
    [[ "$free" =~ ^[0-9]+$ ]] || { log "could not read free space for $path, skipping check"; return 0; }
    log "$label: need ~$(human "$need"), have $(human "$free") free on $(existing_ancestor "$path")"
    if [[ "$SKIP_SPACE_CHECK" -eq 0 && "$free" -lt "$need" ]]; then
        die "not enough space for $label at $path (need $(human "$need"), have $(human "$free")). Pick another path or pass --skip-space-check."
    fi
}

ensure_hub() {
    if [[ "$INSTALL_DEPS" -eq 1 ]]; then
        log "installing/upgrading huggingface_hub and hf_transfer"
        "$PYTHON" -m pip install -q --upgrade "huggingface_hub>=${MIN_HUB_VERSION}" hf_transfer
    fi
    "$PYTHON" - "$MIN_HUB_VERSION" <<'PY' || die "huggingface_hub is missing or too old; rerun with --install-deps"
import sys
try:
    import huggingface_hub
except ImportError:
    sys.exit(1)

def parts(v):
    out = []
    for chunk in v.split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        out.append(int(digits) if digits else 0)
    return out

have, want = huggingface_hub.__version__, sys.argv[1]
print(f"huggingface_hub {have}")
sys.exit(0 if parts(have) >= parts(want) else 1)
PY
}

download() {
    ensure_hub
    mkdir -p "$STAGE_DIR"

    # Keep every byte and all cache metadata on the scratch filesystem so the
    # home directory cannot silently fill up during a multi-TB pull.
    export HF_HOME="${HF_HOME:-${STAGE_DIR%/}.hf-home}"
    export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
    mkdir -p "$HF_HOME"

    if ! "$PYTHON" -c 'import hf_transfer' >/dev/null 2>&1; then
        log "hf_transfer unavailable, falling back to the standard downloader"
        export HF_HUB_ENABLE_HF_TRANSFER=0
    fi

    local cli=()
    if command -v hf >/dev/null 2>&1; then
        cli=(hf download)
    elif command -v huggingface-cli >/dev/null 2>&1; then
        cli=(huggingface-cli download)
    fi

    local attempt=1
    while (( attempt <= MAX_ATTEMPTS )); do
        log "download attempt ${attempt}/${MAX_ATTEMPTS} -> $STAGE_DIR"
        local rc=0
        if [[ ${#cli[@]} -gt 0 ]]; then
            "${cli[@]}" "$REPO" \
                --repo-type "$REPO_TYPE" \
                --local-dir "$STAGE_DIR" \
                --max-workers "$DOWNLOAD_WORKERS" || rc=$?
        else
            REPO="$REPO" REPO_TYPE="$REPO_TYPE" STAGE_DIR="$STAGE_DIR" \
            DOWNLOAD_WORKERS="$DOWNLOAD_WORKERS" "$PYTHON" - <<'PY' || rc=$?
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["REPO"],
    repo_type=os.environ["REPO_TYPE"],
    local_dir=os.environ["STAGE_DIR"],
    max_workers=int(os.environ["DOWNLOAD_WORKERS"]),
)
PY
        fi
        if [[ $rc -eq 0 ]]; then
            log "download complete"
            return 0
        fi
        log "attempt $attempt failed (exit $rc); retrying in 30s"
        sleep 30
        attempt=$(( attempt + 1 ))
    done
    die "download did not finish after $MAX_ATTEMPTS attempts"
}

# rsync is single-threaded, so parallelism comes from splitting the file list
# round-robin across $JOBS concurrent rsync processes.
parallel_copy() {
    [[ -d "$STAGE_DIR" ]] || die "stage dir $STAGE_DIR does not exist"
    mkdir -p "$DEST_DIR"

    local worklist logdir
    WORKLIST="$(mktemp -d "${TMPDIR:-/tmp}/rsync-chunks.XXXXXX")"
    worklist="$WORKLIST"
    logdir="$worklist/logs"
    mkdir -p "$logdir"

    # Recreate directories first (including empty ones), but never the Hub's
    # local bookkeeping dir. Filter order matters: first match wins.
    log "replicating directory skeleton"
    rsync -a -f'- /.cache/' -f'+ */' -f'- *' "$STAGE_DIR/" "$DEST_DIR/"

    log "enumerating files under $STAGE_DIR"
    ( cd "$STAGE_DIR" && find . -type f -printf '%P\n' ) \
        | grep -v '^\.cache/huggingface/' > "$worklist/all.txt" || true

    local total
    total="$(wc -l < "$worklist/all.txt" | tr -d '[:space:]')"
    [[ "$total" -gt 0 ]] || { log "nothing to copy"; return 0; }
    log "$total files to copy using $JOBS parallel rsync processes"

    # Round-robin split keeps each worker's byte load comparable without having
    # to stat every file up front.
    split -n "r/$JOBS" -d -a 4 "$worklist/all.txt" "$worklist/chunk."

    local pids=() chunk
    for chunk in "$worklist"/chunk.*; do
        [[ -s "$chunk" ]] || continue
        rsync -a --files-from="$chunk" "$STAGE_DIR/" "$DEST_DIR/" \
            > "$logdir/$(basename "$chunk").log" 2>&1 &
        pids+=("$!")
    done

    local failed=0 pid
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=$(( failed + 1 ))
    done
    if [[ "$failed" -gt 0 ]]; then
        # Leave the chunk lists and logs in place so the failure can be diagnosed.
        KEEP_WORKLIST=1
        log "$failed rsync worker(s) reported errors; logs in $logdir"
        die "parallel copy incomplete, inspect $logdir and rerun with --copy-only"
    fi
    log "parallel copy finished"
}

verify() {
    local src_files dst_files src_bytes dst_bytes
    src_files="$( ( cd "$STAGE_DIR" && find . -type f -printf '%P\n' ) | grep -vc '^\.cache/huggingface/' || true )"
    dst_files="$( ( cd "$DEST_DIR" && find . -type f | wc -l ) | tr -d '[:space:]' )"
    src_bytes="$(dir_bytes "$STAGE_DIR")"
    dst_bytes="$(dir_bytes "$DEST_DIR")"
    log "source: $src_files files, $(human "$src_bytes")"
    log "dest:   $dst_files files, $(human "$dst_bytes")"
    if [[ "$src_files" == "$dst_files" && "$src_bytes" == "$dst_bytes" ]]; then
        log "verification OK"
    else
        log "WARNING: source and destination differ; rerun with --copy-only to reconcile"
    fi
}

log "repo=$REPO stage=$STAGE_DIR dest=$DEST_DIR jobs=$JOBS"

NEED=0
if [[ "$RUN_DOWNLOAD" -eq 1 ]]; then
    NEED="$(remote_size_bytes)"
    [[ "$NEED" =~ ^[0-9]+$ ]] || NEED=0
    check_space "$STAGE_DIR" "$NEED" "download stage"
fi
if [[ "$RUN_COPY" -eq 1 ]]; then
    if [[ "$NEED" == "0" && -d "$STAGE_DIR" ]]; then
        NEED="$(dir_bytes "$STAGE_DIR")"
    fi
    check_space "$DEST_DIR" "$NEED" "destination"
fi

if [[ "$RUN_DOWNLOAD" -eq 1 ]]; then
    download
fi
if [[ "$RUN_COPY" -eq 1 ]]; then
    parallel_copy
    verify
fi

log "done"
