"""Queue every held-out sample under several noise seeds for the SolarWM serve workers.

Phase 0 of the reward study: the same model and prompt, only the seed differs, so the spread
of a reward across seeds is the headroom a reward-driven fine-tune could exploit. Jobs are
seed-major, so a partial run still has whole seeds.

    python phase0_enqueue.py --data heldout32_oc.txt --spool /workspace/rl0/spool --seeds 42,1,2,3,4
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

JOB_SCHEMA = "h3-proxy-serve-job.v1"


def job_id(seed: int, index: int, pt: Path) -> str:
    return f"s{seed:04d}-{index:03d}-{pt.stem}"


def jobs(samples: list[Path], seeds: list[int], role: str) -> list[tuple[str, dict]]:
    return [(job_id(seed, i, pt), {"schema": JOB_SCHEMA, "seed": seed, "fps": 24,
                                   "segments": [{"name": pt.stem, "pt": str(pt), "cwm_system": role}],
                                   "phase0": {"sample": pt.stem, "index": i}})
            for seed in seeds for i, pt in enumerate(samples)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True, help="manifest with one cached .pt per line")
    p.add_argument("--spool", required=True)
    p.add_argument("--seeds", default="42,1,2,3,4")
    p.add_argument("--cwm-system", default="w0_omni")
    args = p.parse_args()
    samples = [Path(line.split()[0]) for line in Path(args.data).read_text().splitlines() if line.strip()]
    spool = Path(args.spool)
    (spool / "queue").mkdir(parents=True, exist_ok=True)
    known = {path.stem for folder in ("queue", "running", "done") for path in (spool / folder).glob("*.json")}
    queued = 0
    for name, job in jobs(samples, [int(s) for s in args.seeds.split(",")], args.cwm_system):
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
