#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Print the Ref2VA string Qwen is tokenized from (chat + Picture/Video + caption).

This is a structural dump. It does not load the 32B encoder. Vision pads are
collapsed; pass ``--model-path`` with ``--anchor`` and ``--proxy`` to fill counts.

Usage::

    python scripts/h3_proxy/dump_ref2va_chat.py \\
        --validation-json /data/binghe/h3_proxy/abot_validation_val6.json \\
        --cwm-system w0

    python scripts/h3_proxy/dump_ref2va_chat.py \\
        --caption '[0.00s-5.17s] A man walks forward along the road.' \\
        --cwm-system none
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    return parser.parse_args()


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
    inner = f"{pad_token} ×{count}" if count is not None else f"{pad_token} ×?"
    return f"{MINIMAX_H3_VISION_START_TOKEN}⟨{inner}⟩{MINIMAX_H3_VISION_END_TOKEN}"


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


def load_caption(args: argparse.Namespace) -> tuple[str, str]:
    if args.caption is not None:
        return args.caption, "(--caption)"
    path = Path(args.validation_json).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
    if not records:
        raise SystemExit(f"{path} has no records")
    if not 0 <= args.index < len(records):
        raise SystemExit(f"--index {args.index} out of range for {len(records)} records in {path}")
    record = records[args.index]
    caption = record.get("caption") or record.get("prompt")
    if not isinstance(caption, str) or not caption:
        raise SystemExit(f"record {args.index} has no caption/prompt")
    label = f"{path} [{args.index}/{len(records)}] id={record.get('id', '?')}"
    return caption, label


def main() -> None:
    args = parse_args()
    caption, source = load_caption(args)
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


if __name__ == "__main__":
    main()
