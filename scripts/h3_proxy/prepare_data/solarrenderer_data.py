#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A CARLA + Magpie subset of ``Enderfga/SolarRenderer-Data`` as extra omni proxy pairs.

SolarRenderer-Data pairs an engine structure video with the engine render of the same shot: 24 fps,
158 frames, 1344x768. Two of its sources match the omni proxy LoRA's references:

==========  ========================================  =============================================
source      structure video                           omni modality
==========  ========================================  =============================================
CARLA       CityScapes-palette semantic segmentation  ``semantic``, read through ``semantic.mp4``
Magpie      untextured grey block-out render          ``style``, the RGB video as is
==========  ========================================  =============================================

Neither carries depth, so neither yields ``depth`` or ``duv`` rows. UnrealZoo is CC BY-NC and left out.

Steps, each resumable::

    select     pick clips from train/index.jsonl, write selection.jsonl
    fetch      download only the picked clips, by HTTP range into the ~2 GB shards
    semantic   CARLA palette RGB -> lossless class-id semantic.mp4 + semantic.json
    captions   recaption the first 124 frames of each render in the gta_record prose
    manifest   omni encode rows for encode_proxy_samples.py

Layout under ``--root``::

    selection.jsonl
    <source>/<clip>/render.mp4          target, the engine render
    <source>/<clip>/condition.mp4       structure video as released (CRF 15, yuv420p)
    <source>/<clip>/first_frame.png
    <source>/<clip>/meta.json           the index row plus split and provenance
    carla/<clip>/semantic.mp4           CARLA tag in B, R = G = 0, libx264rgb crf 0
    carla/<clip>/semantic.json          tag names, ``"taxonomy": "carla"``
    <source>/<clip>/captions/prompt.json

The ``heldout`` split is the release's own ``benchmark/heldout`` CARLA and Magpie clips, which the
release holds out by map, scene or recording; the train pick excludes them.

Usage::

    python solarrenderer_data.py select   --root /workspace/solarrenderer --per-source 2000
    python solarrenderer_data.py fetch    --root /workspace/solarrenderer --workers 32
    python solarrenderer_data.py semantic --root /workspace/solarrenderer --workers 64
    python solarrenderer_data.py captions --root /workspace/solarrenderer --env-file .../vertex.env
    python solarrenderer_data.py manifest --root /workspace/solarrenderer --split train --out train.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gta_record import CARLA_TAGS  # noqa: E402

REPO = "Enderfga/SolarRenderer-Data"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
SOURCES = ("carla", "magpie")
TRAINING_INPUT = "structure + first frame + text"
MEMBERS = ("render.mp4", "condition.mp4", "first_frame.png", "json")
NUM_FRAMES = 124
FPS = 24.0
SELECTION_NAME = "selection.jsonl"
SEED = "solarrenderer_subset_v1"

# A lossy-coded palette pixel within this RGB distance of a tag colour is taken as that tag; farther
# ones are boundary blends and take the majority of their certain neighbours.
CERTAIN_DISTANCE = 24.0
# A blend can also land on a third palette colour (road + sky averages to wall), which then shows as a
# one-pixel line of a class neither side has. A tag held by fewer than this many pixels of its own 5x5
# window is treated as such a blend; a real object of about 3x3 pixels keeps its tag.
MIN_SUPPORT = 7


def stable_key(*parts: str) -> str:
    return hashlib.sha1("/".join((SEED, *parts)).encode()).hexdigest()


def clip_name(clip_id: str) -> str:
    return clip_id.split("/", 1)[1]


def clip_dir(root: Path, source: str, name: str) -> Path:
    return root / source / name


def fetched_clips(root: Path, source: str) -> list[Path]:
    """Clip directories ``fetch`` has finished; its ``.part`` staging directories are left out."""
    directory = root / source
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir()
                  if not path.name.endswith(".part") and (path / "meta.json").is_file())


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(temporary, path)


# ----------------------------------------------------------------------------- select


def pick_spread(rows: list[dict], count: int) -> list[dict]:
    """``count`` rows evenly spaced through ``rows`` in clip-id order (CARLA ids run through its maps)."""
    ordered = sorted(rows, key=lambda row: row["clip_id"])
    if count >= len(ordered):
        return ordered
    return [ordered[int((index + 0.5) * len(ordered) / count)] for index in range(count)]


def magpie_group(clip_id: str) -> str:
    """``scene04_player2`` of ``magpie/scene04_player2_012250``: one recording."""
    return "_".join(clip_name(clip_id).split("_")[:2])


def pick_round_robin(rows: list[dict], count: int) -> list[dict]:
    """Round-robin over recordings, each recording in a seeded order, so no recording dominates."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[magpie_group(row["clip_id"])].append(row)
    queues = [sorted(groups[key], key=lambda row: stable_key(row["clip_id"])) for key in sorted(groups)]
    picked: list[dict] = []
    depth = 0
    while len(picked) < count and any(depth < len(queue) for queue in queues):
        for queue in queues:
            if depth < len(queue) and len(picked) < count:
                picked.append(queue[depth])
        depth += 1
    return sorted(picked, key=lambda row: row["clip_id"])


def fetch_text(path: str) -> str:
    import requests

    response = requests.get(BASE_URL + path, timeout=300)
    response.raise_for_status()
    return response.text


def cmd_select(args: argparse.Namespace) -> None:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "index.jsonl"
    if not index_path.is_file():
        index_path.write_text(fetch_text("train/index.jsonl"), encoding="utf-8")
    heldout_path = root / "heldout_cases.jsonl"
    if not heldout_path.is_file():
        heldout_path.write_text(fetch_text("benchmark/heldout/cases.jsonl"), encoding="utf-8")
    index = read_jsonl(index_path)
    heldout = [row for row in read_jsonl(heldout_path) if row["clip_id"].split("/")[0] in SOURCES]
    excluded = {row["clip_id"] for row in heldout}

    selection = []
    for source in SOURCES:
        pool = [row for row in index if row["source"] == source and row["training_input"] == TRAINING_INPUT
                and row["clip_id"] not in excluded]
        picked = pick_spread(pool, args.per_source) if source == "carla" else pick_round_robin(pool, args.per_source)
        selection += [{**row, "split": "train"} for row in picked]
        print(f"{source}: {len(picked)} of {len(pool)} eligible train clips, {len({r['shard'] for r in picked})} shards")
    for row in heldout:
        source = row["clip_id"].split("/")[0]
        selection.append({**row, "source": source, "split": "heldout",
                          "benchmark_dir": f"benchmark/heldout/clips/{row['clip_id']}"})
    print(f"heldout: {Counter(row['source'] for row in selection if row['split'] == 'heldout')}")
    with open(root / SELECTION_NAME, "w", encoding="utf-8") as handle:
        for row in selection:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {root / SELECTION_NAME}: {len(selection)} clips")


# ----------------------------------------------------------------------------- fetch


class RangeReader:
    """Byte ranges of one shard over a kept-alive session, re-resolving the CDN URL when it lapses."""

    def __init__(self, path: str) -> None:
        import requests

        self.path = path
        self.session = requests.Session()
        self.url = ""
        self.lock = threading.Lock()

    def _resolve(self) -> None:
        response = self.session.get(BASE_URL + self.path, allow_redirects=False, timeout=60)
        if response.status_code >= 400:
            response.raise_for_status()
        self.url = response.headers.get("location") or BASE_URL + self.path

    def read(self, start: int, stop: int) -> bytes:
        """``[start, stop)``."""
        for attempt in range(6):
            try:
                with self.lock:
                    if not self.url:
                        self._resolve()
                    url = self.url
                response = self.session.get(url, headers={"Range": f"bytes={start}-{stop - 1}"}, timeout=300)
                if response.status_code in (401, 403):
                    with self.lock:
                        self.url = ""
                    raise RuntimeError(f"{response.status_code} on {self.path}")
                response.raise_for_status()
                data = response.content
                if len(data) != stop - start:
                    raise RuntimeError(f"{self.path} [{start}, {stop}) returned {len(data)} bytes")
                return data
            except Exception:  # noqa: BLE001 - network: retry, then raise the last error
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        raise AssertionError("unreachable")


def _octal(field: bytes) -> int:
    text = field.rstrip(b"\0 ").strip()
    return int(text, 8) if text else 0


def _pax_fields(data: bytes) -> dict[str, str]:
    fields = {}
    while data:
        length = int(data.split(b" ", 1)[0])
        record, data = data[:length], data[length:]
        key, _, value = record.split(b" ", 1)[1].rstrip(b"\n").partition(b"=")
        fields[key.decode()] = value.decode()
    return fields


def index_shard(reader: RangeReader, cache: Path) -> dict[str, dict[str, tuple[int, int]]]:
    """``key -> {member suffix: (data offset, size)}`` by walking the tar headers with range reads."""
    if cache.is_file():
        return {key: {suffix: tuple(span) for suffix, span in members.items()}
                for key, members in json.loads(cache.read_text()).items()}
    members: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
    offset, pending_name, pending_size = 0, None, None
    while True:
        header = reader.read(offset, offset + 512)
        if header == b"\0" * 512:
            break
        size = _octal(header[124:136])
        kind = header[156:157]
        data_offset = offset + 512
        offset = data_offset + (size + 511) // 512 * 512
        if kind in (b"x", b"L"):
            payload = reader.read(data_offset, data_offset + size)
            if kind == b"L":
                pending_name = payload.rstrip(b"\0").decode()
            else:
                fields = _pax_fields(payload)
                pending_name = fields.get("path", pending_name)
                pending_size = int(fields["size"]) if "size" in fields else pending_size
            continue
        if kind == b"g":
            continue
        name = pending_name
        if not name:
            prefix = header[345:500].rstrip(b"\0").decode()
            name = (prefix + "/" if prefix else "") + header[0:100].rstrip(b"\0").decode()
        if pending_size is not None:
            size = pending_size
            offset = data_offset + (size + 511) // 512 * 512
        pending_name, pending_size = None, None
        if kind in (b"0", b"\0"):
            key, _, suffix = Path(name).name.partition(".")
            members[key][suffix] = (data_offset, size)
    cache.parent.mkdir(parents=True, exist_ok=True)
    write_json(cache, members)
    return dict(members)


def fetch_train_clip(reader: RangeReader, row: dict, members: dict[str, tuple[int, int]], target: Path) -> None:
    missing = [suffix for suffix in MEMBERS if suffix not in members]
    if missing:
        raise RuntimeError(f"{row['clip_id']} lacks {missing} in {row['shard']}")
    spans = [members[suffix] for suffix in MEMBERS]
    start = min(offset for offset, _ in spans)
    stop = max(offset + size for offset, size in spans)
    blob = reader.read(start, stop)
    staging = target.with_name(target.name + ".part")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for suffix in MEMBERS:
        offset, size = members[suffix]
        name = "meta.json" if suffix == "json" else suffix
        (staging / name).write_bytes(blob[offset - start:offset - start + size])
    meta = json.loads((staging / "meta.json").read_text(encoding="utf-8"))
    meta.update({"split": row["split"], "shard": row["shard"], "repo": REPO})
    write_json(staging / "meta.json", meta)
    os.replace(staging, target)


def fetch_heldout_clip(row: dict, target: Path) -> None:
    import requests

    staging = target.with_name(target.name + ".part")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for name in ("render.mp4", "condition.mp4", "first_frame.png"):
        response = requests.get(BASE_URL + f"{row['benchmark_dir']}/{name}", timeout=300)
        response.raise_for_status()
        (staging / name).write_bytes(response.content)
    meta = {key: value for key, value in row.items() if key != "benchmark_dir"}
    meta.update({"repo": REPO, "benchmark_dir": row["benchmark_dir"]})
    write_json(staging / "meta.json", meta)
    os.replace(staging, target)


def cmd_fetch(args: argparse.Namespace) -> None:
    root = Path(args.root)
    selection = read_jsonl(root / SELECTION_NAME)
    todo = [row for row in selection
            if not (clip_dir(root, row["source"], clip_name(row["clip_id"])) / "meta.json").is_file()]
    print(f"{len(selection) - len(todo)} clips already fetched, {len(todo)} to go")
    by_shard: dict[str, list[dict]] = defaultdict(list)
    for row in todo:
        if row["split"] == "train":
            by_shard[row["shard"]].append(row)
    heldout = [row for row in todo if row["split"] == "heldout"]
    failures: list[str] = []
    done = finished = 0

    def run_shard(shard: str, rows: list[dict]) -> int:
        reader = RangeReader(shard)
        members = index_shard(reader, root / "shard_index" / (shard.replace("/", "__") + ".json"))
        for row in rows:
            key = clip_name(row["clip_id"])
            fetch_train_clip(reader, row, members.get(key, {}), clip_dir(root, row["source"], key))
        return len(rows)

    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run_shard, shard, rows): shard for shard, rows in sorted(by_shard.items())}
        futures.update({pool.submit(fetch_heldout_clip, row, clip_dir(root, row["source"], clip_name(row["clip_id"]))):
                        row["clip_id"] for row in heldout})
        for future in as_completed(futures):
            finished += 1
            try:
                done += future.result() or 1
            except Exception as error:  # noqa: BLE001 - one shard or clip; the rest go on
                failures.append(f"{futures[future]}: {error}")
                print(f"FAILED {futures[future]}: {error}", flush=True)
            if finished % 10 == 0:
                print(f"fetched {done}/{len(todo)} clips, {finished}/{len(futures)} shards", flush=True)
    print(f"fetched {done} clips, {len(failures)} failures")
    if failures:
        raise SystemExit(1)


# ----------------------------------------------------------------------------- semantic


def tag_lut() -> tuple[np.ndarray, np.ndarray]:
    """Nearest CARLA tag and its distance for every 6-bit RGB cell, ``[64**3]`` each."""
    palette = np.array([colour for _, colour, _ in CARLA_TAGS], dtype=np.float32)
    centres = (np.arange(64, dtype=np.float32) * 4 + 1.5)
    grid = np.stack(np.meshgrid(centres, centres, centres, indexing="ij"), axis=-1).reshape(-1, 1, 3)
    distance = np.sqrt(((grid - palette[None]) ** 2).sum(-1))
    return distance.argmin(1).astype(np.uint8), distance.min(1).astype(np.float32)


def palette_to_tags(frame: np.ndarray, labels: np.ndarray, distances: np.ndarray,
                    certain: float = CERTAIN_DISTANCE) -> tuple[np.ndarray, float]:
    """One ``[H, W, 3]`` palette frame -> ``[H, W]`` tags, and the share of pixels that were blends.

    A blend -- far from every tag colour, or a tag too thin to be an object -- takes the most common
    tag among the certain pixels of its 5x5 window; one with no certain neighbour keeps its nearest
    colour.
    """
    from scipy.ndimage import uniform_filter

    q = frame.astype(np.int32) >> 2
    cells = (q[..., 0] << 12) | (q[..., 1] << 6) | q[..., 2]
    tags = labels[cells]
    unsure = distances[cells] > certain
    present = np.unique(tags)
    for tag in present:
        mask = tags == tag
        support = uniform_filter(mask.astype(np.float32), size=5, mode="nearest") * 25.0
        unsure |= mask & (support < MIN_SUPPORT - 0.5)
    share = float(unsure.mean())
    if not unsure.any():
        return tags, share
    best = np.zeros(tags.shape, dtype=np.float32)
    choice = tags.copy()
    for tag in present:
        votes = uniform_filter(((tags == tag) & ~unsure).astype(np.float32), size=5, mode="nearest")
        better = unsure & (votes > best)
        choice[better] = tag
        best = np.where(better, votes, best)
    return np.where(unsure, choice, tags).astype(np.uint8), share


def semantic_json() -> dict:
    return {
        "taxonomy": "carla",
        "source": "SolarRenderer-Data CARLA condition.mp4, nearest CARLA 0.9.14+ palette colour",
        "classes": {str(index): {"id": index, "name": name, "color": list(colour)}
                    for index, (name, colour, _) in enumerate(CARLA_TAGS)},
    }


def convert_semantic(directory: str) -> dict:
    import av

    from compose_gta_duv import write_duv

    clip = Path(directory)
    labels, distances = tag_lut()
    frames, shares = [], []
    with av.open(str(clip / "condition.mp4")) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            tags, share = palette_to_tags(frame.to_ndarray(format="rgb24"), labels, distances)
            coded = np.zeros((*tags.shape, 3), dtype=np.uint8)
            coded[..., 2] = tags
            frames.append(coded)
            shares.append(share)
    write_duv(clip / "semantic.mp4", frames, FPS)
    write_json(clip / "semantic.json", semantic_json())
    histogram = np.bincount(np.stack([f[..., 2] for f in frames[::12]]).ravel(), minlength=len(CARLA_TAGS))
    report = {"frames": len(frames), "blend_share_mean": round(float(np.mean(shares)), 4),
              "blend_share_max": round(float(np.max(shares)), 4),
              "tag_share": {CARLA_TAGS[i][0]: round(float(c) / histogram.sum(), 4)
                            for i, c in enumerate(histogram) if c}}
    meta_path = clip / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["semantic"] = report
    write_json(meta_path, meta)
    return report


def cmd_semantic(args: argparse.Namespace) -> None:
    root = Path(args.root)
    clips = [path for path in fetched_clips(root, "carla")
             if args.overwrite or not (path / "semantic.json").is_file()]
    print(f"{len(clips)} CARLA clips to convert")
    failures = 0
    blends = []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(convert_semantic, str(path)): path.name for path in clips}
        for count, future in enumerate(as_completed(futures), 1):
            try:
                blends.append(future.result()["blend_share_mean"])
            except Exception as error:  # noqa: BLE001
                failures += 1
                print(f"FAILED {futures[future]}: {error}", flush=True)
            if count % 100 == 0:
                print(f"converted {count}/{len(clips)}, mean blend share {np.mean(blends):.4f}", flush=True)
    print(f"converted {len(clips) - failures}, {failures} failures"
          + (f", blend share mean {np.mean(blends):.4f} max {np.max(blends):.4f}" if blends else ""))
    if failures:
        raise SystemExit(1)


# ----------------------------------------------------------------------------- captions

CAPTION_SYSTEM = """You write the training caption of a short video clip for a video generation model.

Describe only what is visible in this clip, start to end, in one paragraph of 60 to 110 words, in this
order: one short phrase naming the medium (for example "Realistic 3D video game animation" or "3D CG
game render"), the setting, the lighting, then what the main subject and other moving things do, in
the order it happens, including how the camera moves. Name concrete looks (clothing, colours, vehicle
types). Present tense. No speculation, no sound, no camera jargon beyond plain words like "the camera
follows behind", and never mention references, structure videos, segmentation, grey boxes or first frames.

A longer description written for a longer cut of the same shot is given as a hint. It may describe
events after this clip ends; leave out anything you cannot see in the clip.

Examples of the style:
- Video game animation captures an industrial city street lined with weathered brick buildings leading
  toward a concrete overpass under bright daytime sunlight. A dark grey sedan rolls steadily forward along
  the cracked asphalt lane, gently decelerating until it comes to a stop near the curb. The front door
  swings open, and a man wearing a white jacket and dark pants steps out of the sedan onto the pavement.
- Realistic 3D video game animation shows a wide asphalt city avenue framed by commercial buildings leading
  toward an arched bridge under a hazy sky. Warm low sunlight glints against the haze, casting long shadows
  across the road. A grey van cruises along the street and eases to a halt near the center line.

Reply with only a JSON object: {"caption": "..."}"""

CAPTION_NAME = "captions/prompt.json"
CAPTION_VERSION = "solarrenderer_data.captions.v1"
FORBIDDEN = ("<video", "<picture", "first frame", "grey box", "gray box", "segmentation", "block-out")


def hint_text(meta: dict) -> str:
    prompt = str(meta.get("prompt") or "")
    if "detailed_description:" in prompt:
        prompt = prompt.split("detailed_description:", 1)[1].split("overall_soundscape:", 1)[0]
    return " ".join(prompt.split())[:3000]


def validate_caption(text: str) -> str:
    caption = " ".join(str(text or "").split())
    words = len(caption.split())
    if not 45 <= words <= 140:
        raise ValueError(f"caption has {words} words; 60 to 110 were asked for")
    lowered = caption.lower()
    hit = [term for term in FORBIDDEN if term in lowered]
    if hit:
        raise ValueError(f"caption mentions {hit}")
    return caption


def trimmed_preview(render: Path, out: Path) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(render), "-frames:v", str(NUM_FRAMES), "-vf",
                    "scale=640:-2", "-c:v", "libx264", "-crf", "23", "-an", str(out)], check=True)


def caption_clip(client: Any, model: str, clip: Path, sample_fps: float) -> dict:
    from clip_prompts.llm import ChatRequest, Message, Text, Video, parse_json_object

    meta = json.loads((clip / "meta.json").read_text(encoding="utf-8"))
    duration = round(NUM_FRAMES / FPS, 3)
    with tempfile.TemporaryDirectory() as scratch:
        preview = Path(scratch) / "preview.mp4"
        trimmed_preview(clip / "render.mp4", preview)
        messages = [
            Message.system(Text(CAPTION_SYSTEM)),
            Message.user(Text(f"Clip length: {duration:.2f} s.\nHint: {hint_text(meta)}"),
                         Video(preview, fps=sample_fps)),
        ]
        error: Exception | None = None
        for attempt in range(1, 4):
            response = client.complete(ChatRequest(model=model, messages=tuple(messages), temperature=0.2,
                                                   json_object=True))
            try:
                caption = validate_caption(parse_json_object(response.text).get("caption"))
                break
            except (ValueError, RuntimeError) as exc:
                error = exc
                messages += [Message("assistant", (Text(response.text),)),
                             Message.user(Text(f"That reply is invalid: {exc}. Reply again with only the JSON object."))]
        else:
            raise RuntimeError(f"no valid caption after 3 attempts: {error}")
    payload = {"prompt": caption, "duration": duration, "frames": NUM_FRAMES, "fps": FPS, "version": CAPTION_VERSION,
               "source": f"vlm:target:{model}", "attempts": attempt,
               "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (clip / "captions").mkdir(exist_ok=True)
    write_json(clip / CAPTION_NAME, payload)
    return payload


def cmd_captions(args: argparse.Namespace) -> None:
    from clip_prompts import observe
    from clip_prompts.llm import load_dotenv

    if not Path(args.env_file).is_file():
        raise SystemExit(f"--env-file {args.env_file} does not exist")
    load_dotenv(Path(args.env_file))
    client = observe.build_client(args.backend)
    root = Path(args.root)
    clips = [path for source in SOURCES for path in fetched_clips(root, source)
             if args.overwrite or not (path / CAPTION_NAME).is_file()]
    if args.limit:
        clips = clips[:args.limit]
    print(f"{len(clips)} clips to caption with {args.model}", flush=True)
    failures = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(caption_clip, client, args.model, path, args.sample_fps): path for path in clips}
        for count, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as error:  # noqa: BLE001
                failures += 1
                print(f"FAILED {futures[future].parent.name}/{futures[future].name}: {error}", flush=True)
            if count % 50 == 0:
                print(f"captioned {count}/{len(clips)}, {failures} failures", flush=True)
    print(f"captioned {len(clips) - failures}, {failures} failures")
    if failures and not args.keep_going:
        raise SystemExit(1)


# ----------------------------------------------------------------------------- manifest


def manifest_row(source: str, clip: Path, prompt: str) -> dict:
    relative = f"{source}/{clip.name}"
    row = {
        "name": f"solar_{source}__{clip.name}",
        "id": relative,
        "target": f"{relative}/render.mp4",
        "prompt": prompt,
        "cwm_system": "w0_omni",
        "modality_prompt": "solarrenderer",
        "target_kind": "engine",
        "target_style": source,
    }
    if source == "carla":
        row.update({"proxy_modality": "semantic", "proxy_semantic_video": f"{relative}/semantic.mp4"})
    else:
        row.update({"proxy_modality": "style", "proxy": f"{relative}/condition.mp4", "proxy_style": "greybox"})
    return row


def cmd_manifest(args: argparse.Namespace) -> None:
    from seg_dir_to_encode_manifest import extract_prompt_text

    root = Path(args.root)
    rows, skipped = [], Counter()
    for source in SOURCES:
        for clip in fetched_clips(root, source):
            if json.loads((clip / "meta.json").read_text(encoding="utf-8")).get("split") != args.split:
                continue
            caption = clip / CAPTION_NAME
            if not caption.is_file():
                skipped[f"{source} without caption"] += 1
                continue
            if source == "carla" and not (clip / "semantic.mp4").is_file():
                skipped["carla without semantic.mp4"] += 1
                continue
            prompt = extract_prompt_text(json.loads(caption.read_text(encoding="utf-8")))
            rows.append(manifest_row(source, clip, prompt))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {out}: {len(rows)} rows {dict(Counter(row['proxy_modality'] for row in rows))}")
    for reason, count in sorted(skipped.items()):
        print(f"  skipped {count}: {reason}")


# ----------------------------------------------------------------------------- cli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select")
    select.add_argument("--root", required=True)
    select.add_argument("--per-source", type=int, default=2000)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--root", required=True)
    fetch.add_argument("--workers", type=int, default=32)
    semantic = sub.add_parser("semantic")
    semantic.add_argument("--root", required=True)
    semantic.add_argument("--workers", type=int, default=32)
    semantic.add_argument("--overwrite", action="store_true")
    captions = sub.add_parser("captions")
    captions.add_argument("--root", required=True)
    captions.add_argument("--env-file", required=True)
    captions.add_argument("--backend", default="vertex")
    captions.add_argument("--model", default="gemini-3.8-flash")
    captions.add_argument("--workers", type=int, default=16)
    captions.add_argument("--sample-fps", type=float, default=4.0)
    captions.add_argument("--limit", type=int)
    captions.add_argument("--overwrite", action="store_true")
    captions.add_argument("--keep-going", action="store_true")
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--root", required=True)
    manifest.add_argument("--split", choices=("train", "heldout"), required=True)
    manifest.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    {"select": cmd_select, "fetch": cmd_fetch, "semantic": cmd_semantic, "captions": cmd_captions,
     "manifest": cmd_manifest}[args.command](args)


if __name__ == "__main__":
    main()
