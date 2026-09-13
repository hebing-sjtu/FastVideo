#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Scan a flat ``seg_*/`` dataset into an ``encode_proxy_samples`` manifest.

Two layouts are accepted. The flat one::

    <root>/
      seg_0000/
        video_src.mp4        low-poly render -> the Ref2VA video reference ("proxy")
        video_target.mp4     high-quality clip -> the denoising target
        prompt.txt           the only text that enters training
        metadata.json        identity; read only for cross-checking the id
        .minimax_h3/         teacher provenance, ignored here

And the nested GTA / native-proxy one (``gta_web_0902``)::

    <root>/
      seg_0000/
        prompt.json          structured caption; compiled into the training prompt
        prompt.txt           preferred when present
        metadata.json        may name the target file
        minimax_h3/output.mp4
        minimax_h3/image_1.png
        proxy/duv.mp4        native game DUV, passed through as proxy_duv_video
      manifests/
        <prefix>_train.jsonl
        <prefix>_val.jsonl

A directory is treated as nested when the flat trio is absent and ``proxy/duv.mp4`` plus a
target exist. ``minimax_h3/prompt.txt`` is the Ref2VA edit instruction and needs
``--allow-teacher-prompt``; a structured ``prompt.json`` compiles to a window-stamped caption.

Corpus gate
-----------
When the root holds a ``vlm_filter.json``, only clips whose judge scores all clear
``--min-vlm-score`` are emitted. A clip that follows its prompt but not its proxy teaches the model
that the proxy is ignorable, so the scores gate the corpus rather than annotate it. Pass
``--vlm-filter none`` to keep everything.

The seg directories are authoritative for *what exists*; the manifests are authoritative for
*which split a clip belongs to*. So this scans the directories and, unless ``--split all``, keeps
only the ids the split manifest names. Ids are matched by the ``seg_\\d+`` pattern anywhere in each
manifest row, which avoids depending on that file's field names.

Every emitted row is checked for its three required files. A seg directory missing any of them is
reported and skipped rather than left to fail one-by-one inside the encoder.

Frame budget
------------
The encoder resamples to H3's fixed 24 fps *before* trimming, so a clip's usable length is
``floor(num_source_frames * 24 / source_fps)``, not its raw frame count. A 124-frame 30-fps clip
yields 99 frames and would fail ``--num-frames 124`` — every clip in the set, after the model is
loaded. This probes container metadata (no decoding) on a sample of clips and prints the largest
valid ``--num-frames`` the set supports.

Usage::

    python scripts/h3_proxy/prepare_data/seg_dir_to_encode_manifest.py \\
      --root /data/tmp --split train \\
      --out /workspace/h3_train.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vlm_filter import add_filter_arguments, load_vlm_filter  # noqa: E402

# The H3 causal VAE consumes frames in groups of 17 after a 5-frame head.
FRAME_MULTIPLE = 17
FRAME_OFFSET = 5
MINIMAX_H3_FPS = 24

SEG_PATTERN = re.compile(r"seg_\d+")

# The window stamp CWM puts in front of a caption, e.g. "[0.00s-5.17s] ". A prompt carrying it is
# known to describe one window; one without it could be an episode summary, so the builders audit
# for it and the contract compiler adds it.
WINDOW_MARKER_PATTERN = re.compile(r"^\[\d+\.\d{2}s-\d+\.\d{2}s\] ")

# Prose variants a `contract`/`version 3` prompt.json compiles. `rich` is the CWM-shaped one:
# medium, environment, lighting, then subject appearance and motion. `lean` drops the scene detail
# down to a few words, which is not what the reference captions look like.
CONTRACT_PROSE_STYLES = ("rich", "lean")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Dataset root containing seg_*/ and manifests/.")
    p.add_argument("--split", choices=("train", "val", "all"), default="train")
    p.add_argument("--out", required=True, help="Output encode manifest jsonl.")
    # `proxy` for encode_proxy_samples (H3), `source` for encode_v2v_depth_samples (Wan V2V). The
    # same video_src.mp4 either way; only the field name differs.
    p.add_argument("--source-key", choices=("proxy", "source"), default="proxy")
    p.add_argument("--proxy-stream",
                   choices=("duv", "color", "auto"),
                   default="auto",
                   help="Nested layout: 'duv' writes proxy_duv_video (native game DUV, no resample). "
                   "'color' writes a regular RGB proxy. 'auto' uses duv.mp4 when present.")
    p.add_argument("--contract-prose",
                   choices=CONTRACT_PROSE_STYLES,
                   default="rich",
                   help="Which compiled prose to take from a structured prompt.json (default rich).")
    p.add_argument("--allow-teacher-prompt",
                   action="store_true",
                   help="Fall back to minimax_h3/prompt.txt when no other text exists. Off by default: "
                   "that file is a Ref2VA edit instruction describing a keyframe task this pipeline "
                   "does not pack, so training on it teaches the wrong contract.")
    p.add_argument("--probe-limit",
                   type=int,
                   default=32,
                   help="Clips to probe for the frame budget; 0 probes all, which is slower on FUSE mounts.")
    add_filter_arguments(p)
    return p.parse_args()


def largest_valid_num_frames(usable: int) -> int:
    """Largest ``n <= usable`` with ``n %% 17 == 5``, or 0 if none fits."""
    if usable < FRAME_OFFSET:
        return 0
    return (usable - FRAME_OFFSET) // FRAME_MULTIPLE * FRAME_MULTIPLE + FRAME_OFFSET


def read_split(root: Path, split: str) -> tuple[set[str], list[Path]]:
    """Ids mentioned by one split's manifests, and the files they came from."""
    directory = root / "manifests"
    if not directory.is_dir():
        raise SystemExit(f"--split {split} needs {directory}, which does not exist. Use --split all to take every "
                         "seg directory.")
    matches = sorted(directory.glob(f"*_{split}.jsonl"))
    ids: set[str] = set()
    for path in matches:
        with open(path, encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                # Match against the raw line rather than parsed fields: the split file's schema is
                # not part of this contract, only the seg ids it mentions.
                ids.update(SEG_PATTERN.findall(line))
    return ids, matches


def split_ids(root: Path, split: str) -> set[str] | None:
    """Ids named by the split manifest, or None to accept every seg directory.

    Also cross-checks the sibling split. Two splits that share ids are not a situation any caller
    asked for, and it is invisible downstream: training would simply include the held-out clips and
    report a validation loss on data it had already fitted.
    """
    if split == "all":
        return None
    ids, matches = read_split(root, split)
    if not matches:
        directory = root / "manifests"
        listing = ", ".join(path.name for path in sorted(directory.iterdir())) or "(empty)"
        raise SystemExit(f"No '*_{split}.jsonl' in {directory}. Found: {listing}")
    if not ids:
        raise SystemExit(f"{', '.join(str(path) for path in matches)} mention no 'seg_NNNN' ids.")
    print(f"Split '{split}': {len(ids)} ids from {', '.join(path.name for path in matches)}")

    sibling = "val" if split == "train" else "train"
    other, other_matches = read_split(root, sibling)
    if other_matches:
        shared = ids & other
        if shared:
            print(f"  WARNING: {len(shared)} of these ids are also in '{sibling}' "
                  f"({', '.join(sorted(shared)[:8])}{' ...' if len(shared) > 8 else ''}).")
            print("  train and val are not disjoint, so every shared clip would be both trained on and "
                  "validated on. Fix the split files before encoding.")
    return ids


def probe_usable_frames(path: Path) -> int | None:
    """Frames this clip contributes at 24 fps, from container metadata alone."""
    try:
        import av
    except ImportError:
        return None
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                return None
            stream = container.streams.video[0]
            rate = stream.average_rate or getattr(stream, "guessed_rate", None)
            if not rate:
                return None
            count = int(stream.frames or 0)
            if count <= 0:
                # Containers written without a frame count still carry a duration.
                if stream.duration and stream.time_base:
                    count = int(float(stream.duration * stream.time_base) * float(rate))
                if count <= 0:
                    return None
            return int(count * MINIMAX_H3_FPS / float(rate))
    except (OSError, ValueError, StopIteration):
        return None


def _first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def _string_field(payload: object, *keys: str) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def window_marker(duration: object) -> str:
    """The ``[0.00s-5.17s] `` stamp for a window of ``duration`` seconds, or "" if unknown."""
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
        return ""
    return f"[0.00s-{float(duration):.2f}s] "


def extract_prompt_text(payload: object, *, prose_style: str = "rich") -> str:
    """The one sentence that enters training, window-stamped when the payload dates it.

    A structured ``prompt.json`` carries both a ``rich`` and a ``lean`` compilation plus the
    window's ``duration``. Stamping the chosen prose keeps these prompts indistinguishable from a
    ``captions-export`` one, which is what the manifest audit downstream checks for.
    """
    if isinstance(payload, str):
        return payload.strip()
    if not isinstance(payload, dict):
        return ""
    text = _compiled_prose(payload, prose_style=prose_style)
    if not text:
        return ""
    if WINDOW_MARKER_PATTERN.match(text):
        return text
    return window_marker(payload.get("duration")) + text


def _compiled_prose(payload: dict, *, prose_style: str) -> str:
    direct = _string_field(payload, "prompt", "caption", "text", "user")
    if direct:
        return direct
    compiled = payload.get("compiled")
    if not isinstance(compiled, dict):
        return ""
    cwm = compiled.get("cwm")
    if isinstance(cwm, dict):
        user = _string_field(cwm, "user")
        if user:
            return user
    # Requested style first, then the other one: a contract that compiled only one of them still
    # yields text rather than falling through to the teacher instruction.
    ordered = (prose_style, ) + tuple(style for style in CONTRACT_PROSE_STYLES if style != prose_style)
    for style in ordered:
        variant = compiled.get(style)
        if isinstance(variant, dict):
            prose = _string_field(variant, "global")
            if prose:
                return prose
    return _string_field(compiled, "global")


def read_seg_prompt(seg: Path, *, prose_style: str = "rich", allow_teacher: bool = False) -> tuple[str, str]:
    """Training prompt. A root ``prompt.txt`` wins, then the structured ``prompt.json``.

    ``minimax_h3/prompt.txt`` is a Ref2VA edit instruction that describes a source video and a
    keyframe-completion task, neither of which this pipeline packs. It is only consulted when
    ``allow_teacher`` says to, so a dataset that lost its captions fails loudly instead of
    training on the wrong contract.
    """
    root_txt = seg / "prompt.txt"
    if root_txt.is_file():
        text = root_txt.read_text(encoding="utf-8").strip()
        if text:
            return text, "prompt.txt"
    prompt_json = seg / "prompt.json"
    if prompt_json.is_file():
        try:
            text = extract_prompt_text(json.loads(prompt_json.read_text(encoding="utf-8")), prose_style=prose_style)
        except (OSError, json.JSONDecodeError):
            text = ""
        if text:
            return text, "prompt.json"
    if allow_teacher:
        teacher = seg / "minimax_h3" / "prompt.txt"
        if teacher.is_file():
            text = teacher.read_text(encoding="utf-8").strip()
            if text:
                return text, "minimax_h3/prompt.txt"
    return "", "none"


def read_nested_target(seg: Path) -> Path | None:
    named: list[Path] = []
    meta_file = seg / "metadata.json"
    if meta_file.is_file():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
        raw = _string_field(meta, "output", "target", "video", "target_video", "output_video")
        if not raw and isinstance(meta.get("files"), dict):
            raw = _string_field(meta["files"], "output", "target", "video")
        if raw:
            named.append(seg / raw if not Path(raw).is_absolute() else Path(raw))
    return _first_existing(*named, seg / "minimax_h3" / "output.mp4", seg / "video_target.mp4")


def resolve_seg_media(
    seg: Path,
    *,
    proxy_stream: str = "auto",
    prose_style: str = "rich",
    allow_teacher: bool = False,
) -> tuple[dict, str] | tuple[None, str]:
    """Return a relative-path row fragment, or (None, reason)."""
    read = lambda: read_seg_prompt(seg, prose_style=prose_style, allow_teacher=allow_teacher)  # noqa: E731
    flat_target = seg / "video_target.mp4"
    flat_proxy = seg / "video_src.mp4"
    if flat_target.is_file() and flat_proxy.is_file():
        prompt, source = read()
        if not prompt:
            return None, "no prompt (no prompt.txt / prompt.json)"
        return {
            "target": flat_target,
            "proxy": flat_proxy,
            "anchor": _first_existing(seg / "minimax_h3" / "image_1.png", seg / "anchor.png"),
            "prompt": prompt,
            "prompt_source": source,
            "layout": "flat",
        }, ""

    target = read_nested_target(seg)
    duv = seg / "proxy" / "duv.mp4"
    color = seg / "proxy" / "color.mp4"
    if proxy_stream == "duv":
        proxy, proxy_kind = (duv, "duv")
    elif proxy_stream == "color":
        proxy, proxy_kind = (color, "color")
    elif duv.is_file():
        proxy, proxy_kind = (duv, "duv")
    else:
        proxy, proxy_kind = (color, "color")

    missing = []
    if target is None:
        missing.append("minimax_h3/output.mp4")
    if not proxy.is_file():
        missing.append(f"proxy/{proxy.name}")
    if missing:
        return None, "missing " + ", ".join(missing)
    prompt, source = read()
    if not prompt:
        return None, ("no prompt (no prompt.txt / prompt.json"
                      f"{'' if allow_teacher else '; minimax_h3/prompt.txt needs --allow-teacher-prompt'})")
    return {
        "target": target,
        "proxy": proxy,
        "anchor": _first_existing(seg / "minimax_h3" / "image_1.png"),
        "prompt": prompt,
        "prompt_source": source,
        "layout": "nested",
        "proxy_kind": proxy_kind,
    }, ""


def report_frame_budget(rows: list[dict], root: Path, limit: int) -> None:
    sample = rows if limit <= 0 else rows[::max(1, len(rows) // limit)][:limit]
    budgets: list[tuple[str, int]] = []
    for row in sample:
        for key in ("target", "proxy", "source", "proxy_duv_video"):
            relative = row.get(key)
            if not relative:
                continue
            usable = probe_usable_frames(root / str(relative))
            if usable is not None:
                budgets.append((f"{row['name']}/{key}", usable))
    if not budgets:
        print("Frame budget: could not probe any clip (PyAV missing or metadata absent). Verify --num-frames by "
              "encoding a few clips before launching the full set.")
        return
    name, worst = min(budgets, key=lambda item: item[1])
    print(f"Frame budget over {len(budgets)} probed streams: shortest is {worst} frames at 24 fps ({name}).")
    recommended = largest_valid_num_frames(worst)
    if recommended:
        print(f"  Largest valid --num-frames for this sample: {recommended}")
    else:
        print(f"  No valid --num-frames fits {worst} frames; the shortest clip is unusable.")


def report_window_scope(rows: list[dict]) -> None:
    """Say how many prompts are scoped to one window, and complain if any are not."""
    prompts = [str(row.get("prompt", "")) for row in rows]
    windowed = [prompt for prompt in prompts if WINDOW_MARKER_PATTERN.match(prompt)]
    print(f"  window-scoped prompts: {len(windowed)}/{len(prompts)}")
    if len(windowed) == len(prompts):
        return
    offenders = [prompt for prompt in prompts if not WINDOW_MARKER_PATTERN.match(prompt)]
    longest = max(offenders, key=len)
    head = longest if len(longest) <= 140 else longest[:137] + "..."
    print(f"    WARNING: {len(offenders)} prompt(s) carry no '[0.00s-5.17s] ' stamp, so nothing says they "
          "describe only this window. Longest:")
    print(f"      {head}")


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"--root does not exist: {root}")
    keep = split_ids(root, args.split)

    seg_dirs = sorted(path for path in root.glob("seg_*") if path.is_dir())
    if not seg_dirs:
        raise SystemExit(f"No 'seg_*' directories under {root}")

    vlm = load_vlm_filter(root, args)
    if vlm is not None:
        vlm.report(on_disk={path.name for path in seg_dirs})

    rows: list[dict] = []
    skipped_split = 0
    rejected_vlm: list[str] = []
    incomplete: list[str] = []
    layouts: dict[str, int] = {}
    prompt_sources: dict[str, int] = {}
    for seg in seg_dirs:
        if keep is not None and seg.name not in keep:
            skipped_split += 1
            continue
        if vlm is not None:
            accepted, why = vlm.verdict(seg.name)
            if not accepted:
                rejected_vlm.append(f"{seg.name}: {why}")
                continue
        media, reason = resolve_seg_media(
            seg,
            proxy_stream=args.proxy_stream,
            prose_style=args.contract_prose,
            allow_teacher=args.allow_teacher_prompt,
        )
        if media is None:
            incomplete.append(f"{seg.name}: {reason}")
            continue
        layouts[str(media["layout"])] = layouts.get(str(media["layout"]), 0) + 1
        prompt_sources[str(media["prompt_source"])] = prompt_sources.get(str(media["prompt_source"]), 0) + 1
        row = {
            "name": seg.name,
            "target": str(Path(media["target"]).relative_to(root)),
            "prompt": media["prompt"],
            "id": seg.name,
        }
        if media.get("proxy_kind") == "duv" or (
                media["layout"] == "nested" and Path(media["proxy"]).name == "duv.mp4"):
            row["proxy_duv_video"] = str(Path(media["proxy"]).relative_to(root))
        else:
            row[args.source_key] = str(Path(media["proxy"]).relative_to(root))
        if media.get("anchor") is not None:
            row["anchor"] = str(Path(media["anchor"]).relative_to(root))
        rows.append(row)

    if not rows:
        raise SystemExit(f"No complete seg directories survived (scanned {len(seg_dirs)}, "
                         f"{len(incomplete)} incomplete, {len(rejected_vlm)} below the VLM threshold, "
                         f"{skipped_split} out of split).")

    missing_from_disk = sorted(keep - {row["name"] for row in rows}) if keep is not None else []

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows -> {out}")
    # State the total explicitly: without it the reader has to add the lines below to notice that
    # the split files reference more clips than the directory tree holds.
    print(f"  {len(seg_dirs)} seg directories on disk, {len(rows)} of them usable in split '{args.split}'")
    if layouts:
        print(f"  layouts: {', '.join(f'{key}={value}' for key, value in sorted(layouts.items()))}")
    if prompt_sources:
        print(f"  prompt sources: {', '.join(f'{key}={value}' for key, value in sorted(prompt_sources.items()))}")
    report_window_scope(rows)
    if rejected_vlm:
        print(f"  {len(rejected_vlm)} rejected by the VLM filter:")
        for line in rejected_vlm[:10]:
            print(f"    {line}")
        if len(rejected_vlm) > 10:
            print(f"    ... and {len(rejected_vlm) - 10} more")
    if skipped_split:
        print(f"  {skipped_split} seg directories are not in split '{args.split}'")
    if incomplete:
        print(f"  {len(incomplete)} incomplete, skipped:")
        for line in incomplete[:10]:
            print(f"    {line}")
        if len(incomplete) > 10:
            print(f"    ... and {len(incomplete) - 10} more")
    if missing_from_disk:
        print(f"  {len(missing_from_disk)} ids in the split manifest have no usable seg directory: "
              f"{', '.join(missing_from_disk[:10])}{' ...' if len(missing_from_disk) > 10 else ''}")
    report_frame_budget(rows, root, args.probe_limit)


if __name__ == "__main__":
    main()
