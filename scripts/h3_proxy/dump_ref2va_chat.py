#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Print the Ref2VA string Qwen is tokenized from (chat + Picture/Video + caption).

This is a structural dump. It does not load the 32B encoder. Vision pads are
collapsed; pass ``--model-path`` with ``--anchor`` and ``--proxy`` to fill counts.

Usage::

    # which records are in there, and which one --index picks
    python scripts/h3_proxy/dump_ref2va_chat.py \\
        --validation-json /data/binghe/h3_proxy/abot_validation_val6.json --list

    python scripts/h3_proxy/dump_ref2va_chat.py \\
        --validation-json /data/binghe/h3_proxy/abot_validation_val6.json \\
        --cwm-system w0

    # same record, plus its anchor / proxy / target pushed to W&B so the text and
    # the pixels can be read side by side from a laptop
    python scripts/h3_proxy/dump_ref2va_chat.py \\
        --validation-json /data/binghe/h3_proxy/abot_validation_val6.json \\
        --cwm-system w0 --wandb

    python scripts/h3_proxy/dump_ref2va_chat.py \\
        --caption '[0.00s-5.17s] A man walks forward along the road.' \\
        --cwm-system none
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any

from fastvideo.pipelines.basic.minimax_h3.cwm_presentation import (
    canonical_caption,
    load_cwm_system_prompt,
    resolve_cwm_system_role,
)
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_FPS,
    MINIMAX_H3_IMAGE_PAD_TOKEN,
    MINIMAX_H3_VIDEO_PAD_TOKEN,
    MINIMAX_H3_VISION_END_TOKEN,
    MINIMAX_H3_VISION_START_TOKEN,
)
from fastvideo.pipelines.basic.minimax_h3.reference import (
    MINIMAX_H3_QWEN_TEMPORAL_PATCH,
    MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS,
)

QWEN_CHAT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--caption", help="User caption only (the validation JSON 'caption' field).")
    source.add_argument("--validation-json", help="Validation file written by clip_dir_to_validation_json.py.")
    parser.add_argument("--index", type=int, default=0, help="Record index inside --validation-json.")
    parser.add_argument("--cwm-system", default="w0", choices=("w0", "wn", "none"))
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument(
        "--image-pads",
        type=int,
        default=None,
        help="Collapse <|image_pad|> to this count. Omit to print ×?",
    )
    parser.add_argument(
        "--video-pads",
        type=int,
        default=None,
        help="Collapse <|video_pad|> per block to this count. Omit to print ×?",
    )
    parser.add_argument(
        "--model-path",
        help="Optional MiniMax-H3 snapshot. Uses its processor.apply_chat_template for the wrap.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List every record's index, id, and caption head, then exit. Needs --validation-json.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Also upload this record's anchor, proxy, and target to W&B alongside the rendered text. "
        "Needs --validation-json, because --caption carries no media.",
    )
    parser.add_argument("--wandb-project", default="fastvideo_h3_proxy")
    parser.add_argument("--wandb-run-name", default=None, help="Defaults to h3_case_<id>.")
    args = parser.parse_args()
    if args.list and not args.validation_json:
        raise SystemExit("--list reads records from --validation-json")
    if args.wandb and not args.validation_json:
        raise SystemExit("--wandb uploads the record's media, which only --validation-json carries")
    return args


def qwen_block_timestamps(num_frames: int) -> list[float]:
    """Same 2 fps / temporal-patch-2 blocks as sample_reference_video_frames."""
    stride = MINIMAX_H3_FPS / MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS
    indices: list[int] = []
    cursor = 0.0
    while round(cursor) < num_frames:
        if not indices or round(cursor) > indices[-1]:
            indices.append(round(cursor))
        cursor += stride
    timestamps = [index / MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS for index in range(len(indices))]
    timestamps += [timestamps[-1]] * (-len(timestamps) % MINIMAX_H3_QWEN_TEMPORAL_PATCH)
    return [(timestamps[index] + timestamps[index + MINIMAX_H3_QWEN_TEMPORAL_PATCH - 1]) / 2
            for index in range(0, len(timestamps), MINIMAX_H3_QWEN_TEMPORAL_PATCH)]


def _vision_span(pad_token: str, count: int | None) -> str:
    # ASCII only: some terminals render ⟨⟩ / × as underscores and make "?" look like corruption.
    n = str(count) if count is not None else "?"
    return (f"{MINIMAX_H3_VISION_START_TOKEN}[{pad_token} x{n}]"
            f"{MINIMAX_H3_VISION_END_TOKEN}")


def user_body(caption: str, *, num_frames: int, image_pads: int | None, video_pads: int | None) -> str:
    parts = [f"<Picture 1>: {_vision_span(MINIMAX_H3_IMAGE_PAD_TOKEN, image_pads)}"]
    parts.append("<Video 1>: ")
    for timestamp in qwen_block_timestamps(num_frames):
        parts.append(f"<{timestamp:.1f} seconds>")
        parts.append(_vision_span(MINIMAX_H3_VIDEO_PAD_TOKEN, video_pads))
    parts.append(canonical_caption(caption) if caption.strip() else caption)
    return "".join(parts)


def wrap_chat(user: str, role: str | None, *, model_path: str | None) -> str:
    if role is None:
        return user
    system = load_cwm_system_prompt(role)
    if model_path:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(str(Path(model_path).expanduser() / "processor"), trust_remote_code=True)
        formatted = processor.apply_chat_template(
            [{
                "role": "system",
                "content": system
            }, {
                "role": "user",
                "content": user
            }],
            tokenize=False,
            add_generation_prompt=False,
        )
        if not isinstance(formatted, str):
            raise TypeError("apply_chat_template(..., tokenize=False) must return a string")
        return formatted
    return QWEN_CHAT_TEMPLATE.format(system=system, user=user)


def load_records(validation_json: str) -> tuple[Path, list[dict[str, Any]]]:
    path = Path(validation_json).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
    if not records:
        raise SystemExit(f"{path} has no records")
    return path, list(records)


def record_caption(record: dict[str, Any]) -> str:
    caption = record.get("caption") or record.get("prompt")
    return caption if isinstance(caption, str) else ""


def list_records(args: argparse.Namespace) -> None:
    """Show which record each --index picks, so the dump can be pointed at a case on purpose."""
    path, records = load_records(args.validation_json)
    print(f"{path}: {len(records)} record(s)")
    for index, record in enumerate(records):
        caption = record_caption(record).replace("\n", " ")
        head = caption if len(caption) <= 96 else caption[:93] + "..."
        marker = "->" if index == args.index else "  "
        print(f"{marker} [{index}] id={record.get('id', '?')}  {head}")


def load_caption(args: argparse.Namespace) -> tuple[str, str, dict[str, Any] | None]:
    if args.caption is not None:
        return args.caption, "(--caption)", None
    path, records = load_records(args.validation_json)
    if not 0 <= args.index < len(records):
        raise SystemExit(f"--index {args.index} out of range for {len(records)} records in {path}")
    record = records[args.index]
    caption = record_caption(record)
    if not caption:
        raise SystemExit(f"record {args.index} has no caption/prompt")
    label = f"{path} [{args.index}/{len(records)}] id={record.get('id', '?')}"
    return caption, label, record


def _readable(record: dict[str, Any], key: str) -> str | None:
    """A path the record carries and that exists, or None. Mirrors the validation callback."""
    value = record.get(key)
    if not isinstance(value, str) or not value:
        return None
    return value if os.path.isfile(value) else None


def _anchor_image(record: dict[str, Any]) -> Any:
    """The anchor, falling back to the target's frame 0 the way the encoder and callback do."""
    from PIL import Image

    anchor = _readable(record, "anchor_path")
    if anchor is not None:
        return Image.open(anchor).convert("RGB")
    target = _readable(record, "target_path")
    if target is None:
        return None
    from fastvideo.pipelines.basic.minimax_h3.reference import decode_reference_video

    frames, _, _ = decode_reference_video(target)
    return Image.fromarray(frames[0])


def upload_to_wandb(args: argparse.Namespace, record: dict[str, Any], *, rendered: str, role: str | None) -> None:
    """Push one case's conditioning bundle -- text, anchor, proxy, target -- to one W&B run.

    The point is to read the words and the pixels together: a caption that names a 60-second episode
    is only obviously wrong next to the 5 seconds the proxy actually covers.
    """
    import wandb

    case_id = str(record.get("id", f"index{args.index}"))
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"h3_case_{case_id}",
        job_type="inspect_conditioning",
        config={
            "case_id": case_id,
            "index": args.index,
            "validation_json": args.validation_json,
            "cwm_system": role or "none",
            "num_frames": args.num_frames,
            "qwen_video_blocks": len(qwen_block_timestamps(args.num_frames)),
        },
    )
    try:
        media: dict[str, Any] = {
            # <pre> rather than a plain string: the chat wrap's newlines and the caption's CRLF are
            # exactly what is under inspection, and W&B renders markdown out of bare text.
            "qwen_input": wandb.Html(f"<pre>{html.escape(rendered)}</pre>"),
            "caption": wandb.Html(f"<pre>{html.escape(record_caption(record))}</pre>"),
        }
        anchor = _anchor_image(record)
        if anchor is not None:
            media["anchor"] = wandb.Image(anchor, caption=f"{case_id} anchor / target frame 0")
        for key, label in (("proxy_path", "proxy"), ("target_path", "target")):
            path = _readable(record, key)
            if path is not None:
                media[label] = wandb.Video(path, caption=f"{case_id} {label}", format="mp4")
            else:
                print(f"note: record has no readable {key}, skipping the {label} panel")
        run.log(media)
        run.summary["case_id"] = case_id
        print(f"uploaded to {run.url}")
    finally:
        run.finish()


def main() -> None:
    args = parse_args()
    if args.list:
        list_records(args)
        return
    caption, source, record = load_caption(args)
    role = resolve_cwm_system_role(args.cwm_system)
    body = user_body(
        caption,
        num_frames=args.num_frames,
        image_pads=args.image_pads,
        video_pads=args.video_pads,
    )
    rendered = wrap_chat(body, role, model_path=args.model_path)
    print(f"source: {source}")
    print(f"cwm_system: {role or 'none'}")
    print(f"num_frames: {args.num_frames} -> {len(qwen_block_timestamps(args.num_frames))} Qwen video blocks")
    print("--- Qwen input (vision pads collapsed) ---")
    print(rendered)
    if args.wandb:
        assert record is not None  # parse_args requires --validation-json alongside --wandb
        upload_to_wandb(args, record, rendered=rendered, role=role)


if __name__ == "__main__":
    main()
