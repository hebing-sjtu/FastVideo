#!/usr/bin/env python3
"""Download an evenly spread subset of ABot-World-Explorer-500h.

The full dataset is ~2.5 TiB across 30,969 samples, which is impractical for a
quick validation run. This picks N samples spaced evenly across the sorted
sample_id space and fetches only those, reproducing the upstream
``data/<shard>/<sample_id>/`` layout so code written against the full dataset
works unchanged against the subset.

Sample ids are content hashes, so even spacing over the sorted id space is an
unbiased sample and also covers all 256 shards roughly in proportion to their
size.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_REPO = "acvlab/ABot-World-Explorer-500h"
METADATA_NAME = "metadata.jsonl"
PER_SAMPLE_FILES = ("video.mp4", "annotations.tar")


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] ERROR: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download an evenly spread subset of ABot-World-Explorer-500h",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument(
        "--revision",
        default=None,
        help="pin a commit sha for reproducibility; defaults to the revision "
        "referenced by metadata.jsonl",
    )
    p.add_argument("--num-videos", type=int, default=2000)
    p.add_argument(
        "--out-dir",
        default="/tmp/ABot-World-Explorer-subset2000",
        help="subset root; upstream data/<shard>/<sample_id>/ layout is recreated inside",
    )
    p.add_argument(
        "--strategy",
        choices=("stride", "random"),
        default="stride",
        help="stride is deterministic and evenly spaced; random needs --seed",
    )
    p.add_argument("--seed", type=int, default=0, help="only used by --strategy random")
    p.add_argument("--workers", type=int, default=16, help="concurrent file downloads")
    p.add_argument("--max-attempts", type=int, default=8, help="retries per file")
    p.add_argument(
        "--metadata",
        default=None,
        help="reuse a local metadata.jsonl instead of downloading it",
    )
    p.add_argument(
        "--videos-only",
        action="store_true",
        help="skip annotations.tar and fetch only video.mp4",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="report the selection and its exact size, download nothing",
    )
    return p.parse_args()


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def load_metadata(args: argparse.Namespace) -> list[dict]:
    if args.metadata:
        path = Path(args.metadata)
        if not path.is_file():
            die(f"--metadata {path} not found")
        log(f"reading local metadata {path}")
        raw = path.read_text()
    else:
        from huggingface_hub import hf_hub_download

        log(f"fetching {METADATA_NAME} from {args.repo}")
        cached = hf_hub_download(
            repo_id=args.repo,
            filename=METADATA_NAME,
            repo_type="dataset",
            revision=args.revision,
        )
        raw = Path(cached).read_text()

    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not records:
        die("metadata.jsonl is empty")
    log(f"{len(records)} samples in the full dataset")
    return records


def infer_revision(records: list[dict]) -> str | None:
    """Pull the commit sha out of the hf:// URI so a subset is reproducible."""
    path = records[0].get("video", {}).get("path", "")
    if "@" in path:
        tail = path.split("@", 1)[1]
        sha = tail.split("/", 1)[0]
        if len(sha) == 40:
            return sha
    return None


def select(records: list[dict], n: int, strategy: str, seed: int) -> list[dict]:
    ordered = sorted(records, key=lambda r: r["sample_id"])
    total = len(ordered)
    if n >= total:
        log(f"requested {n} >= dataset size {total}; taking everything")
        return ordered

    if strategy == "random":
        rng = random.Random(seed)
        picked = sorted(rng.sample(range(total), n))
    else:
        # Evenly spaced indices; the round() keeps spacing balanced at both ends
        # rather than letting truncation bunch samples toward the start.
        picked = sorted({min(total - 1, round(i * total / n)) for i in range(n)})
        # Deduplication can leave us a hair short on pathological ratios.
        cursor = 0
        while len(picked) < n and cursor < total:
            if cursor not in picked:
                picked.append(cursor)
                picked.sort()
            cursor += 1

    return [ordered[i] for i in picked[:n]]


def shard_of(sample_id: str) -> str:
    return sample_id[:2]


def rel_paths(sample_id: str, videos_only: bool) -> list[str]:
    names = ("video.mp4",) if videos_only else PER_SAMPLE_FILES
    return [f"data/{shard_of(sample_id)}/{sample_id}/{name}" for name in names]


def report_spread(selected: list[dict], records: list[dict]) -> None:
    from collections import Counter

    sel = Counter(shard_of(r["sample_id"]) for r in selected)
    full = Counter(shard_of(r["sample_id"]) for r in records)
    counts = sorted(sel.values())
    log(
        f"selection covers {len(sel)}/{len(full)} shards; "
        f"per-shard min/median/max = {counts[0]}/{counts[len(counts) // 2]}/{counts[-1]}"
    )


def exact_size(repo: str, revision: str | None, paths: list[str]) -> int | None:
    """Ask the Hub for real blob sizes so a dry run can be trusted."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        total = 0
        batch = 400
        for i in range(0, len(paths), batch):
            infos = api.get_paths_info(
                repo_id=repo,
                paths=paths[i : i + batch],
                repo_type="dataset",
                revision=revision,
            )
            for info in infos:
                total += getattr(info, "size", 0) or 0
        return total
    except Exception as exc:  # noqa: BLE001 - size reporting must never be fatal
        log(f"could not query exact sizes ({exc}); skipping size report")
        return None


def write_manifest(
    out_dir: Path, args: argparse.Namespace, revision: str | None, selected: list[dict]
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mirror the upstream filename so downstream loaders find it where they expect.
    meta_path = out_dir / METADATA_NAME
    with meta_path.open("w") as fh:
        for rec in selected:
            fh.write(json.dumps(rec) + "\n")

    manifest = {
        "source_repo": args.repo,
        "revision": revision,
        "num_videos": len(selected),
        "strategy": args.strategy,
        "seed": args.seed if args.strategy == "random" else None,
        "videos_only": args.videos_only,
        "sample_ids": [r["sample_id"] for r in selected],
    }
    (out_dir / "subset_manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"wrote {meta_path.name} and subset_manifest.json to {out_dir}")


def download_all(
    repo: str,
    revision: str | None,
    paths: list[str],
    out_dir: Path,
    workers: int,
    max_attempts: int,
) -> int:
    from huggingface_hub import hf_hub_download

    done = 0
    failed: list[str] = []
    lock = threading.Lock()
    total = len(paths)

    def fetch(rel: str) -> tuple[str, bool]:
        for attempt in range(1, max_attempts + 1):
            try:
                hf_hub_download(
                    repo_id=repo,
                    filename=rel,
                    repo_type="dataset",
                    revision=revision,
                    local_dir=str(out_dir),
                )
                return rel, True
            except Exception as exc:  # noqa: BLE001 - retry any transport error
                if attempt == max_attempts:
                    log(f"giving up on {rel}: {exc}")
                    return rel, False
                time.sleep(min(30, 2**attempt))
        return rel, False

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, rel) for rel in paths]
        for fut in as_completed(futures):
            rel, ok = fut.result()
            with lock:
                done += 1
                if not ok:
                    failed.append(rel)
                if done % 100 == 0 or done == total:
                    log(f"{done}/{total} files ({len(failed)} failed)")

    if failed:
        log(f"{len(failed)} file(s) failed; rerun the same command to retry them")
        for rel in failed[:10]:
            log(f"  failed: {rel}")
    return len(failed)


def main() -> int:
    args = parse_args()

    if args.num_videos <= 0:
        die("--num-videos must be positive")

    records = load_metadata(args)
    revision = args.revision or infer_revision(records)
    if revision:
        log(f"pinned revision {revision}")
    else:
        log("no revision pinned; using main (subset will not be reproducible)")

    selected = select(records, args.num_videos, args.strategy, args.seed)
    log(f"selected {len(selected)} samples via {args.strategy}")
    report_spread(selected, records)

    paths: list[str] = []
    for rec in selected:
        paths.extend(rel_paths(rec["sample_id"], args.videos_only))

    out_dir = Path(args.out_dir)
    size = exact_size(args.repo, revision, paths)
    if size:
        log(f"{len(paths)} files, {human(size)} to download")

    if args.dry_run:
        log("dry run, nothing downloaded")
        log(f"first 3 paths: {paths[:3]}")
        return 0

    if not os.environ.get("HF_HUB_ENABLE_HF_TRANSFER"):
        try:
            import hf_transfer  # noqa: F401

            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            log("enabled hf_transfer")
        except ImportError:
            log("hf_transfer not installed; downloads will use a single connection per file")

    write_manifest(out_dir, args, revision, selected)
    failed = download_all(
        args.repo, revision, paths, out_dir, args.workers, args.max_attempts
    )

    got = sum(1 for rel in paths if (out_dir / rel).is_file())
    log(f"{got}/{len(paths)} files present under {out_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
