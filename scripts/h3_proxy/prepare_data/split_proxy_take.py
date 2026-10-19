#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Split a long take into the overlapping clips a SolarWM proxy rollout generates in turn.

The rollout generates a take one 124-frame clip at a time. Clip 0 is an ordinary ``w0_omni``
sample. Every later clip starts 85 frames after the previous one and re-uses its last 39 frames
(two latent groups of 17 frames and the 5-frame head, 12 latents); the first 10 of those latents
are given, so a continuation clip denoises 27 latents and the stitched take grows by 85 frames
per clip::

    clip 0   [0, 124)       w0_omni, given 1 latent (the anchor frame)
    clip k   [85k, 85k+124)  wn_omni, given 10 latents = frames [85k, 85k+34) of clip k-1's output
    take     124 + 85 (n - 1) frames: 9 clips are 804 frames, 33.5 s at 24 fps

Every clip keeps the take's one ``<Picture 1>`` (``anchor.png``, the take's first frame unless
``--anchor`` names another) and its own caption. Outputs, under ``--out``::

    clips/sNN/{target,duv,depth,semantic,style}.mp4   lossless cuts, re-read bit-exact
    anchor.png, manifest.jsonl                        the encode manifest, root = --out
    take.json                                         h3-proxy-take.v1, read by solarwm infer

Then encode with the flags of the training cache (``--cwm-system w0_omni``; the rows carry
``wn_omni`` and ``take_anchor`` themselves) and give ``take.json`` to SolarWM, which looks for
``<name>-sNN.pt`` next to it or under ``inference.proxy_take_cache``.

Sources. Either long videos (``--target`` or a ``--anchor``-only placeholder target, plus the
references of ``--modality``), or one ``gta_record`` parent recording (``--gta-root --gta-parent``),
whose enter/drive/exit segs overlap and are stitched on their ``track.json`` source frames; there
each clip takes the caption of the seg covering its middle frame. Usage::

    python scripts/h3_proxy/prepare_data/split_proxy_take.py --name gta0022-depth \\
      --gta-root /data/binghe/datasets/gta_record_0930-v0003 --gta-parent 0022 --modality depth \\
      --out /workspace/takes/gta0022-depth

    python scripts/h3_proxy/prepare_data/split_proxy_take.py --name city-duv --modality duv \\
      --proxy-duv long/duv.mp4 --anchor look.png --captions captions.json --out /workspace/takes/city-duv
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

TAKE_SCHEMA = "h3-proxy-take.v1"
FPS = 24
SEGMENT_FRAMES = 124
OVERLAP_FRAMES = 39
GIVEN_FRAMES = 34
STRIDE_FRAMES = SEGMENT_FRAMES - OVERLAP_FRAMES
MODALITIES = ("duv", "depth", "semantic", "style")
# Decoded layout of each stream, as encode_proxy_samples reads it back.
LAYOUTS = {"target": "rgb24", "duv": "rgb24", "depth": "gray", "semantic": "rgb24", "style": "rgb24"}
SIDECARS = ("track.json", "semantic.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="Take id; clips are named <name>-sNN.")
    p.add_argument("--out", required=True)
    p.add_argument("--modality", choices=MODALITIES, required=True, help="The <Video 1> reference of every clip.")
    p.add_argument("--segments", type=int, default=0, help="Clips to cut; 0 = as many as the source holds.")
    p.add_argument("--target", default="", help="Long target video; a real target is compared in the rollout.")
    p.add_argument("--anchor", default="", help="<Picture 1> image. Without --target it is also a still "
                   "placeholder target, which the rollout never compares against.")
    p.add_argument("--proxy-duv", default="")
    p.add_argument("--proxy-depth", default="", help="Log-z grey depth video; track.json is read beside it.")
    p.add_argument("--proxy-semantic", default="", help="Class-id-in-B video; semantic.json is read beside it.")
    p.add_argument("--proxy-style", default="", help="A restyled (e.g. low-poly) render of the whole take.")
    p.add_argument("--caption", default="", help="One caption for every clip.")
    p.add_argument("--captions", default="", help="JSON list with one caption per clip.")
    p.add_argument("--gta-root", default="")
    p.add_argument("--gta-parent", default="")
    p.add_argument("--contract-prose", choices=("rich", "lean"), default="rich")
    p.add_argument("--oc", default="", help="Identity sheet given to every clip as <Picture 2>.")
    p.add_argument("--oc-subject", default="")
    args = p.parse_args(argv)
    if bool(args.gta_root) != bool(args.gta_parent):
        p.error("--gta-root and --gta-parent go together")
    if args.gta_root:
        if args.modality == "style":
            p.error("gta_record renders are per seg, so a parent has no take-long style reference")
        if args.target or args.proxy_duv or args.proxy_depth or args.proxy_semantic or args.proxy_style:
            p.error("--gta-parent supplies the target and references itself")
    else:
        if not (args.target or args.anchor):
            p.error("give --target, or --anchor for a placeholder target")
        if not (args.caption or args.captions):
            p.error("give --caption or --captions")
        needed = {"duv": ("proxy_duv", ), "depth": ("proxy_depth", "proxy_semantic"),
                  "semantic": ("proxy_depth", "proxy_semantic"), "style": ("proxy_style", )}[args.modality]
        missing = [f"--{key.replace('_', '-')}" for key in needed if not getattr(args, key)]
        if missing:
            p.error(f"--modality {args.modality} needs {' '.join(missing)}")
    if bool(args.oc) != bool(args.oc_subject):
        p.error("--oc and --oc-subject go together")
    return args


def take_frames(segments: int) -> int:
    return SEGMENT_FRAMES + STRIDE_FRAMES * (segments - 1)


def segment_count(frames: int) -> int:
    return 0 if frames < SEGMENT_FRAMES else (frames - SEGMENT_FRAMES) // STRIDE_FRAMES + 1


def resample_index(count: int, fps: float) -> np.ndarray:
    """Source frame of each 24-fps slot: the encoder's nearest resample, as indices."""
    if fps == FPS:
        return np.arange(count)
    scale = FPS / fps
    slots = np.floor(np.arange(count) * scale + 0.5).astype(np.int64)
    repeats = np.diff(slots, append=math.floor(count * scale + 0.5))
    return np.repeat(np.arange(count), repeats)


def decode(path: Path, layout: str) -> np.ndarray:
    """Every frame of ``path`` on the 24-fps timeline, ``[T, H, W]`` grey or ``[T, H, W, 3]`` RGB."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rate = float(stream.average_rate or stream.guessed_rate)
        frames = np.stack([frame.to_ndarray(format=layout) for frame in container.decode(stream)])
    return frames[resample_index(len(frames), rate)]


def write_lossless(path: Path, frames: np.ndarray, layout: str) -> None:
    """Write a clip that decodes back to exactly ``frames``; checked, since a cut must not drift."""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264rgb" if layout == "rgb24" else "libx264", rate=FPS)
        stream.height, stream.width = frames.shape[1:3]
        stream.pix_fmt = layout
        stream.options = {"qp": "0", "preset": "veryfast"}
        for array in frames:
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(array), format=layout)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    if not np.array_equal(decode(path, layout), frames):
        raise RuntimeError(f"{path} does not decode back to the frames written; the cut is not lossless")


def gta_parent_sources(root: Path, parent: str, prose: str) -> tuple[dict[str, np.ndarray], list[dict], Path]:
    """Stitch a parent's segs on their source frames; returns streams, caption spans, a proxy dir."""
    from gta_record import read_semantic_map
    from seg_dir_to_encode_manifest import extract_prompt_text

    segs = []
    for meta_path in sorted(root.glob("seg_*/metadata.json")):
        if str(json.loads(meta_path.read_text(encoding="utf-8")).get("parent")) != parent:
            continue
        proxy = meta_path.parent / "proxy"
        track = json.loads((proxy / "track.json").read_text(encoding="utf-8"))
        segs.append((int(track["cut"]["source_start"]), proxy, track))
    if not segs:
        raise SystemExit(f"no seg of parent {parent!r} under {root}")
    segs.sort(key=lambda item: item[0])
    first = segs[0][2]
    for _, proxy, track in segs[1:]:
        if track.get("depth") != first.get("depth"):
            raise ValueError(f"{proxy} quantizes depth unlike {segs[0][1]}; the stitched take would mix scales")
        if read_semantic_map(proxy) != read_semantic_map(segs[0][1]):
            raise ValueError(f"{proxy} numbers semantic classes unlike {segs[0][1]}")

    streams: dict[str, list[np.ndarray]] = {key: [] for key in ("target", "duv", "depth", "semantic")}
    files = {"target": "color.mp4", "duv": "duv.mp4", "depth": "depth.mp4", "semantic": "semantic.mp4"}
    spans, end = [], 0
    for start, proxy, _ in segs:
        if start > end:
            raise ValueError(f"parent {parent} has a gap: {proxy} starts at source frame {start}, cover ends at {end}")
        clip = {key: decode(proxy / name, LAYOUTS[key]) for key, name in files.items()}
        length = len(clip["target"])
        skip = end - start
        for key in ("duv", "depth", "semantic"):
            if skip and streams[key] and not np.array_equal(clip[key][:skip], np.concatenate(streams[key])[start:end]):
                raise ValueError(f"{proxy}/{files[key]} disagrees with the previous seg on their shared frames")
        for key, frames in clip.items():
            streams[key].append(frames[skip:])
        text = extract_prompt_text(json.loads((proxy / "captions/prompt.json").read_text(encoding="utf-8")),
                                   prose_style=prose)
        spans.append({"start": start, "end": start + length, "caption": text, "seg": proxy.parent.name})
        end = start + length
    return {key: np.concatenate(parts) for key, parts in streams.items()}, spans, segs[0][1]


def caption_at(spans: list[dict], frame: int) -> str:
    """Caption of the seg whose span holds ``frame``, the latest-starting one where segs overlap."""
    holding = [span for span in spans if span["start"] <= frame < span["end"]]
    return (holding[-1] if holding else spans[-1])["caption"]


def build(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    modality = args.modality
    plane_streams = ("depth", "semantic") if modality in ("depth", "semantic") else (modality, )
    sidecar_dir: Path | None = None
    spans: list[dict] = []
    if args.gta_root:
        streams, spans, sidecar_dir = gta_parent_sources(Path(args.gta_root), args.gta_parent, args.contract_prose)
        streams = {key: streams[key] for key in ("target", *plane_streams)}
        source = {"kind": "gta_record", "root": args.gta_root, "parent": args.gta_parent,
                  "segs": [span["seg"] for span in spans]}
    else:
        paths = {"target": args.target, "duv": args.proxy_duv, "depth": args.proxy_depth,
                 "semantic": args.proxy_semantic, "style": args.proxy_style}
        streams = {key: decode(Path(paths[key]), LAYOUTS[key]) for key in plane_streams}
        if args.target:
            streams["target"] = decode(Path(args.target), "rgb24")
        if modality in ("depth", "semantic"):
            sidecar_dir = Path(args.proxy_depth).parent
        source = {"kind": "videos", **{key: value for key, value in paths.items() if value}}

    available = min(len(frames) for frames in streams.values())
    count = args.segments or segment_count(available)
    if count < 1 or take_frames(count) > available:
        raise SystemExit(f"{count} clips need {take_frames(max(count, 1))} frames; the sources hold {available}")
    captions = json.loads(Path(args.captions).read_text(encoding="utf-8")) if args.captions else None
    if captions is not None and (not isinstance(captions, list) or len(captions) < count):
        raise SystemExit(f"--captions must list at least {count} captions")

    if args.anchor:
        anchor = Image.open(args.anchor).convert("RGB")
    elif "target" in streams:
        anchor = Image.fromarray(streams["target"][0])
    anchor.save(out / "anchor.png")
    real = "target" in streams
    if not real:
        height, width = next(iter(streams.values())).shape[1:3]
        still = np.asarray(anchor.resize((width, height), Image.BICUBIC))
        write_lossless(out / "clips/placeholder_target.mp4", np.repeat(still[None], SEGMENT_FRAMES, 0), "rgb24")
    if args.oc:
        shutil.copyfile(args.oc, out / f"oc{Path(args.oc).suffix}")

    rows, segments = [], []
    for index in range(count):
        start = index * STRIDE_FRAMES
        clip_dir = Path("clips") / f"s{index:02d}"
        for key, frames in streams.items():
            write_lossless(out / clip_dir / f"{key}.mp4", frames[start:start + SEGMENT_FRAMES], LAYOUTS[key])
        if sidecar_dir is not None:
            for name in SIDECARS:
                if (sidecar_dir / name).is_file():
                    shutil.copyfile(sidecar_dir / name, out / clip_dir / name)
        if spans:
            text = caption_at(spans, start + SEGMENT_FRAMES // 2)
        else:
            text = str(captions[index]) if captions is not None else args.caption
        name = f"{args.name}-s{index:02d}"
        role = "w0_omni" if index == 0 else "wn_omni"
        row = {
            "name": name,
            "target": str(clip_dir / "target.mp4") if real else "clips/placeholder_target.mp4",
            "prompt": text,
            "proxy_modality": modality,
            "cwm_system": role,
            "take_anchor": "anchor.png",
            "take": args.name,
            "take_segment": index,
            "target_kind": "color" if real else "placeholder",
        }
        if modality == "duv":
            row["proxy_duv_video"] = str(clip_dir / "duv.mp4")
        elif modality == "style":
            row["proxy"] = str(clip_dir / "style.mp4")
        else:
            row["proxy_depth_video"] = str(clip_dir / "depth.mp4")
            row["proxy_semantic_video"] = str(clip_dir / "semantic.mp4")
        if args.oc:
            row["oc"], row["oc_subject"] = f"oc{Path(args.oc).suffix}", args.oc_subject
        rows.append(row)
        segments.append({"name": name, "start": start, "cwm_system": role, "caption": text})

    with (out / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    take = {
        "schema": TAKE_SCHEMA,
        "name": args.name,
        "fps": FPS,
        "segment_frames": SEGMENT_FRAMES,
        "overlap_frames": OVERLAP_FRAMES,
        "given_frames": GIVEN_FRAMES,
        "stride_frames": STRIDE_FRAMES,
        "frames": take_frames(count),
        "target": "real" if real else "placeholder",
        "modality": modality,
        "source": source,
        "segments": segments,
    }
    (out / "take.json").write_text(json.dumps(take, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return take


def main(argv: list[str] | None = None) -> None:
    take = build(parse_args(argv))
    print(f"[take] {take['name']}: {len(take['segments'])} clips, {take['frames']} frames "
          f"({take['frames'] / FPS:.1f} s), target={take['target']}")


if __name__ == "__main__":
    main()
