#!/usr/bin/env python3
"""Share one serve front end's queue with the workers of a second pod, through the GCS bucket.

``proxy_service.py`` and its own ``proxy_loop serve`` workers run on the front-end pod; a second
pod runs more workers on an identical ``/workspace/serve`` layout, so every path in a job and in
its result stays valid on both sides. ``out`` runs on the front end: it hands a queued job over
only while no local worker is idle and the other pod has an idle worker that no handed-over job is
already waiting for, mirrors that job's progress into the local ``running/`` and lands its result
in the local ``out/`` and ``done/``, where the UI and the gallery pick it up. ``in`` runs on the
worker pod: it queues what arrives, sends progress and results back, and publishes the pod's
worker counts, which ``out`` also copies to ``spool/remote/`` for the UI's status line.

The bucket only takes whole-file writes and a long-lived reader can miss files another node wrote
later, so every marker is a separate file and a directory is listed before a file in it is read.
Usage::

    spool_relay.py out --node 144951     # front-end pod
    spool_relay.py in --node 144951      # worker pod
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

RELAY = Path("/data/binghe/h3_proxy/serve-relay/service")
STATUS_FRESH_S = 120
# A handed-over job that never started on a pod whose status went silent comes back here.
ABANDON_S = 600
MARKERS = {"READY", "CANCEL", "progress.json", "done.json", "RESULT_DONE", "out"}


def log(message: str) -> None:
    print(f"[relay {time.strftime('%H:%M:%S')}] {message}", flush=True)


def listing(directory: Path) -> set[str]:
    try:
        return set(os.listdir(directory))
    except FileNotFoundError:
        return set()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_local(path: Path, payload: Any) -> None:
    """Atomic on the pod's own disk, where the UI may read the file at any moment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(f".{path.name}.tmp")
    scratch.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(scratch, path)


def write_bucket(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def alive(beat: dict[str, Any], now: float) -> bool:
    """An idle worker beats every poll; a busy one is silent until its job ends, so ask its pid."""
    if beat.get("state") == "idle":
        return now - float(beat.get("updated", 0)) < 30
    if beat.get("state") != "busy":
        return False
    try:
        os.kill(int(beat.get("pid") or 0), 0)
    except (OSError, ValueError):
        return False
    return True


def local_workers(spool: Path) -> list[dict[str, Any]]:
    now, rows = time.time(), []
    for path in spool.glob("workers/*.json"):
        try:
            beat = read_json(path)
        except (OSError, ValueError):
            continue
        if alive(beat, now):
            rows.append(beat)
    return rows


def queued(spool: Path) -> list[Path]:
    rows = []
    for path in spool.glob("queue/*.json"):
        try:
            rows.append((path.stat().st_mtime, path.name, path))
        except FileNotFoundError:
            continue
    return [path for _, _, path in sorted(rows)]


class State:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.jobs: dict[str, dict[str, Any]] = read_json(path) if path.is_file() else {}

    def save(self) -> None:
        write_local(self.path, self.jobs)


# front end ------------------------------------------------------------


def remote_status(node: str) -> dict[str, Any] | None:
    directory = RELAY / "status"
    if f"{node}.json" not in listing(directory):
        return None
    try:
        status = read_json(directory / f"{node}.json")
    except (OSError, ValueError):
        return None
    return status if time.time() - float(status.get("updated", 0)) < STATUS_FRESH_S else None


def ship(spool: Path, jobs: Path, path: Path, node: str, state: State) -> None:
    job_id = path.stem
    running = spool / "running" / path.name
    try:
        os.replace(path, running)
    except FileNotFoundError:
        return  # a local worker claimed it first
    payload = read_json(running)
    target = RELAY / "jobs" / job_id
    shutil.copytree(jobs / job_id, target, dirs_exist_ok=True)
    write_bucket(target / "job.json", payload)
    (target / "READY").write_text(str(time.time()), encoding="utf-8")
    total = len(payload.get("segments", []))
    write_local(spool / "running" / f"{job_id}.progress.json",
                {"done": 0, "total": total, "updated": time.time(), "node": node})
    state.jobs[job_id] = {"node": node, "phase": "shipped", "shipped": time.time()}
    state.save()
    log(f"shipped {job_id} ({total} segment(s)) to {node}")


def follow(spool: Path, job_id: str, entry: dict[str, Any], state: State, status: dict[str, Any] | None) -> bool:
    """Mirror one handed-over job; True while it has not started on the other pod."""
    target = RELAY / "jobs" / job_id
    names = listing(target)
    if "RESULT_DONE" in names:
        shutil.copytree(target / "out", spool / "out" / job_id, dirs_exist_ok=True)
        write_local(spool / "done" / f"{job_id}.json", read_json(target / "done.json"))
        for name in (f"{job_id}.json", f"{job_id}.progress.json"):
            (spool / "running" / name).unlink(missing_ok=True)
        entry["phase"] = "landed"
        state.save()
        log(f"landed {job_id} from {entry['node']}")
        return False
    if "progress.json" in names:
        try:
            progress = read_json(target / "progress.json")
        except (OSError, ValueError):
            return False
        write_local(spool / "running" / f"{job_id}.progress.json", {**progress, "node": entry["node"]})
        return False
    if status is None and time.time() - float(entry["shipped"]) > ABANDON_S:
        (target / "CANCEL").write_text(str(time.time()), encoding="utf-8")
        (spool / "running" / f"{job_id}.progress.json").unlink(missing_ok=True)
        os.replace(spool / "running" / f"{job_id}.json", spool / "queue" / f"{job_id}.json")
        entry["phase"] = "returned"
        state.save()
        log(f"{entry['node']} went silent; {job_id} is back in the local queue")
        return False
    return True


def run_out(args: argparse.Namespace) -> None:
    serve = Path(args.serve)
    spool, jobs = serve / "spool", serve / "jobs"
    state = State(serve / "relay_service_out.json")
    log(f"out: front end {serve} -> {args.node} via {RELAY}")
    while True:
        try:
            status = remote_status(args.node)
            mirror = spool / "remote" / f"{args.node}.json"
            if status is not None:
                write_local(mirror, status)
            else:
                mirror.unlink(missing_ok=True)
            waiting = sum(follow(spool, job_id, entry, state, status)
                          for job_id, entry in list(state.jobs.items()) if entry["phase"] == "shipped")
            if status is not None:
                free = int(status.get("idle", 0)) - waiting
                local_idle = sum(beat.get("state") == "idle" for beat in local_workers(spool))
                for path in queued(spool)[:max(free, 0)] if local_idle == 0 else []:
                    ship(spool, jobs, path, args.node, state)
        except Exception:  # keep relaying; the next pass retries
            traceback.print_exc()
        time.sleep(args.poll)


# worker pod -----------------------------------------------------------


def receive(spool: Path, jobs: Path, job_id: str, state: State) -> None:
    source = RELAY / "jobs" / job_id
    shutil.copytree(source, jobs / job_id, dirs_exist_ok=True,
                    ignore=lambda directory, names: [n for n in names if Path(directory) == source and n in MARKERS])
    scratch = spool / "queue" / f".{job_id}.tmp"
    shutil.copyfile(source / "job.json", scratch)
    os.replace(scratch, spool / "queue" / f"{job_id}.json")
    state.jobs[job_id] = {"phase": "queued", "received": time.time()}
    state.save()
    log(f"queued {job_id}")


def report(spool: Path, job_id: str, entry: dict[str, Any], state: State) -> None:
    target = RELAY / "jobs" / job_id
    done = spool / "done" / f"{job_id}.json"
    if done.is_file():
        if (spool / "out" / job_id).is_dir():
            shutil.copytree(spool / "out" / job_id, target / "out", dirs_exist_ok=True)
        shutil.copyfile(done, target / "done.json")
        (target / "RESULT_DONE").write_text(str(time.time()), encoding="utf-8")
        entry["phase"] = "done"
        state.save()
        log(f"returned {job_id}")
        return
    progress = spool / "running" / f"{job_id}.progress.json"
    try:
        text = progress.read_text(encoding="utf-8")
    except FileNotFoundError:
        return
    if text != entry.get("progress"):
        (target / "progress.json").write_text(text, encoding="utf-8")
        entry["progress"] = text


def run_in(args: argparse.Namespace) -> None:
    serve = Path(args.serve)
    spool, jobs = serve / "spool", serve / "jobs"
    for name in ("queue", "running", "out", "done"):
        (spool / name).mkdir(parents=True, exist_ok=True)
    state = State(serve / "relay_service_in.json")
    log(f"in: {RELAY} -> worker pod {serve} as {args.node}")
    while True:
        try:
            for job_id in sorted(listing(RELAY / "jobs") - set(state.jobs)):
                names = listing(RELAY / "jobs" / job_id)
                if "READY" in names and "CANCEL" not in names:
                    receive(spool, jobs, job_id, state)
            for job_id, entry in list(state.jobs.items()):
                if entry["phase"] == "queued":
                    report(spool, job_id, entry, state)
            workers = local_workers(spool)
            write_bucket(RELAY / "status" / f"{args.node}.json", {
                "node": args.node,
                "workers": len(workers),
                "idle": sum(beat.get("state") == "idle" for beat in workers),
                "queued": len(queued(spool)),
                "weights_id": next((beat.get("weights_id") for beat in workers), ""),
                "updated": time.time(),
            })
        except Exception:  # keep relaying; the next pass retries
            traceback.print_exc()
        time.sleep(args.poll)


def main() -> None:
    global RELAY
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("out", "in"))
    parser.add_argument("--node", required=True, help="name of the worker pod, as the status line shows it")
    parser.add_argument("--serve", default="/workspace/serve", help="holds spool/ and jobs/ on both pods")
    parser.add_argument("--relay", type=Path, default=RELAY, help="bucket directory both pods mount")
    parser.add_argument("--poll", type=float, default=5.0)
    args = parser.parse_args()
    RELAY = args.relay
    (RELAY / "jobs").mkdir(parents=True, exist_ok=True)
    run_out(args) if args.mode == "out" else run_in(args)


if __name__ == "__main__":
    main()
