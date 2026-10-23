"""Queue every held-out sample under several noise seeds for the SolarWM serve workers.

Phase 0 of the reward study: the same model and prompt, only the seed differs, so the spread
of a reward across seeds is the headroom a reward-driven fine-tune could exploit. Jobs are
seed-major, so a partial run still has whole seeds.

    python phase0_enqueue.py --data heldout32_oc.txt --spool /workspace/rl0/spool --seeds 42,1,2,3,4

``--long`` groups the ``<take>-sNN.pt`` clips of ``split_proxy_take.py`` takes into one chained
job per take, the first clip ``w0_omni`` and the rest its continuation role ``wn_omni``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

JOB_SCHEMA = "h3-proxy-serve-job.v1"
CONTINUATION = {"w0_omni": "wn_omni", "w0": "wn"}


def job_id(seed: int, index: int, name: str) -> str:
    return f"s{seed:04d}-{index:03d}-{name}"


def groups(samples: list[Path], long: bool) -> list[tuple[str, list[Path]]]:
    if not long:
        return [(pt.stem, [pt]) for pt in samples]
    takes: dict[str, list[Path]] = {}
    for pt in samples:
        match = re.fullmatch(r"(.+)-s(\d\d)", pt.stem)
        if match is None:
            raise SystemExit(f"{pt} is not a <take>-sNN clip")
        takes.setdefault(match.group(1), []).append(pt)
    return [(name, sorted(clips, key=lambda p: p.stem)) for name, clips in sorted(takes.items())]


def jobs(samples: list[Path], seeds: list[int], role: str, long: bool = False) -> list[tuple[str, dict]]:
    out = []
    for seed in seeds:
        for i, (name, clips) in enumerate(groups(samples, long)):
            segments = [{"name": pt.stem, "pt": str(pt), "cwm_system": role if k == 0 else CONTINUATION[role]}
                        for k, pt in enumerate(clips)]
            out.append((job_id(seed, i, name), {"schema": JOB_SCHEMA, "seed": seed, "fps": 24,
                                                 "segments": segments, "phase0": {"sample": name, "index": i}}))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True, help="manifest with one cached .pt per line")
    p.add_argument("--spool", required=True)
    p.add_argument("--seeds", default="42,1,2,3,4")
    p.add_argument("--cwm-system", default="w0_omni")
    p.add_argument("--long", action="store_true", help="one chained job per <take>-sNN group")
    args = p.parse_args()
    samples = [Path(line.split()[0]) for line in Path(args.data).read_text().splitlines() if line.strip()]
    spool = Path(args.spool)
    (spool / "queue").mkdir(parents=True, exist_ok=True)
    known = {path.stem for folder in ("queue", "running", "done") for path in (spool / folder).glob("*.json")}
    queued = 0
    for name, job in jobs(samples, [int(s) for s in args.seeds.split(",")], args.cwm_system, args.long):
        if name in known:
            continue
        scratch = spool / f".{name}.json"
        scratch.write_text(json.dumps(job, indent=2), encoding="utf-8")
        os.replace(scratch, spool / "queue" / f"{name}.json")
        queued += 1
        time.sleep(0.01)  # claim order is mtime, then name
    print(f"queued {queued} job(s); {len(known)} already known", flush=True)


if __name__ == "__main__":
    main()
