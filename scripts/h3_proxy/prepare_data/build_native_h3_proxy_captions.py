#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build DUV-aware, single-anchor H3 Ref2VA captions for a validation set.

The GTA corpus already carries a VLM-written native H3 edit prompt at
``seg_*/minimax_h3/prompt.txt``. It has the useful, clip-specific parts that cannot be templated:
subject definitions and a detailed RGB-target motion/layout narration. Its reference contract is
wrong for this pipeline, though: it describes a regular source video and two keyframe pictures,
while proxy-to-video presents one false-colour DUV video and one RGB anchor.

This tool keeps the VLM observations and deterministically replaces only that reference contract:

* ``<Video 1>`` becomes the DUV authority for camera, geometry, positions and timing;
* ``<Picture 1>`` becomes the sole appearance reference;
* every ``<Picture 2>`` block is removed;
* the edit summary and retention analysis are rewritten to match those two inputs.

The output is the ``{clip_id: caption}`` mapping consumed by ``set_validation_captions.py``. It is
deliberately a separate file: the source VLM prompt is provenance and must not be overwritten.

Usage::

    python scripts/h3_proxy/prepare_data/build_native_h3_proxy_captions.py \\
      --val-json /data/binghe/h3_proxy/gta_v2_cwm_validation_val6.json \\
      --root /data/binghe/datasets/gta_web_0902_v2/gta_web_0902 \\
      --out /data/binghe/h3_proxy/native_h3_captions_val6.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

SECTION_NAMES = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)
SECTION_PATTERN = re.compile(rf"(?m)^({'|'.join(SECTION_NAMES)}):[ \t]*$")
SEGMENT_ID_PATTERN = re.compile(r"seg_\d+")

VIDEO_DEFINITION = (
    "<Video 1> is the colored proxy/src of this same shot (semantic class fills plus blocky 3D "
    "volumes: magenta/purple road, green sidewalk/ground, cyan far field, purple trees, colored "
    "subject or vehicle). It is the sole authority for camera path, framing, silhouettes, volumes, "
    "subject positions, action order and timing across 124 frames at 24 fps (5.17 seconds). Rebuild "
    "as photoreal matching <Picture 1>. Never copy the proxy false-color look."
)
PICTURE_DEFINITION = (
    "<Picture 1> is LOOK-only for the opening of [Shot 1]: the first frame of the photoreal target "
    "(tgt). Materials, lighting, identity, and world look follow this picture for the whole shot. It "
    "does not override <Video 1> camera, framing, action, or scene layout."
)
SUMMARY = (
    "[video editing + keyframe completion] The target video is an edited version of <Video 1>. "
    "Camera, facing/yaw, pose, and timing follow <Video 1>; LOOK stays photoreal matching <Picture 1> "
    "for the full duration. No invented locomotion."
)
VIDEO_RETENTION = (
    "<Video 1>: partially_preserved - camera, framing, silhouettes, volumes, and action timing stay "
    "with <Video 1>; proxy false-color gives way to the photoreal look of <Picture 1>."
)
PICTURE_RETENTION = (
    "<Picture 1>: partially_preserved - photoreal materials, lighting, and identity match this "
    "picture for the whole shot. Camera, framing, subject screen position, and action stay with "
    "<Video 1>."
)
DETAIL_PREFIX = (
    "Decode <Video 1> as a colored semantic/layout proxy with 3D volumes, not the final look. "
    "Reconstruct each class volume as photoreal matching <Picture 1>. Do not copy purple, green, or "
    "cyan fills. The shot begins from <Picture 1>."
)
NARRATION_MARKER = "TGT RGB motion/layout narration"


def parse_sections(text: str) -> dict[str, str]:
    """Split one native H3 prompt without interpreting its clip-specific prose."""
    matches = list(SECTION_PATTERN.finditer(text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group(1)] = text[match.end():stop].strip()
    missing = [name for name in SECTION_NAMES[:4] if not sections.get(name)]
    if missing:
        raise ValueError(f"native H3 prompt is missing required section(s): {', '.join(missing)}")
    return sections


def reference_blocks(text: str) -> list[str]:
    """Paragraphs headed by ``<Video/Picture/Subject N>``."""
    starts = list(re.finditer(r"(?m)^<[^>\n]+>", text))
    blocks = []
    for index, start in enumerate(starts):
        stop = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        blocks.append(text[start.start():stop].strip())
    return blocks


def adapt_native_h3_prompt(text: str) -> str:
    """Change a two-keyframe native H3 prompt into the successful DUV + one-anchor contract."""
    sections = parse_sections(text)
    subject_definitions = [
        block for block in reference_blocks(sections["subject_definitions"]) if block.startswith("<Subject ")
    ]
    subject_retention = [
        block for block in reference_blocks(sections["retention_analysis"]) if block.startswith("<Subject ")
    ]
    if not subject_definitions:
        raise ValueError("native H3 prompt defines no <Subject N> to preserve")

    detail = sections["detailed_description"]
    marker = detail.find(NARRATION_MARKER)
    if marker < 0:
        raise ValueError(f"native H3 detailed_description has no {NARRATION_MARKER!r}")
    narration = detail[marker:].strip()

    output = [
        "subject_definitions:",
        VIDEO_DEFINITION,
        PICTURE_DEFINITION,
        *subject_definitions,
        "",
        "summary:",
        SUMMARY,
        "",
        "retention_analysis:",
        VIDEO_RETENTION,
        PICTURE_RETENTION,
        *subject_retention,
        "",
        "detailed_description:",
        DETAIL_PREFIX,
        "",
        narration,
        "",
        "overall_soundscape:",
        sections.get("overall_soundscape") or "N/A",
        "",
        "non_diegetic_music:",
        sections.get("non_diegetic_music") or "N/A",
    ]
    result = "\n".join(output).strip()
    if "<Picture 2>" in result:
        raise AssertionError("adapted prompt still refers to an input the pipeline does not present")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val-json", required=True, help="Validation JSON whose clip ids select prompts.")
    parser.add_argument("--root", required=True, help="Dataset root containing seg_*/minimax_h3/prompt.txt.")
    parser.add_argument("--out", required=True, help="Output {clip_id: caption} JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(Path(args.val_json).expanduser(), encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise SystemExit(f"{args.val_json} holds no validation records under 'data'.")

    root = Path(args.root).expanduser()
    captions: dict[str, str] = {}
    for record in records:
        clip_id = str(record.get("id", ""))
        if not SEGMENT_ID_PATTERN.fullmatch(clip_id):
            raise SystemExit(f"invalid or missing validation clip id: {clip_id!r}")
        source = root / clip_id / "minimax_h3" / "prompt.txt"
        if not source.is_file():
            raise SystemExit(f"{clip_id}: native H3 VLM prompt is missing: {source}")
        try:
            captions[clip_id] = adapt_native_h3_prompt(source.read_text(encoding="utf-8"))
        except ValueError as error:
            raise SystemExit(f"{clip_id}: {error}") from error

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(captions, handle, ensure_ascii=False, indent=2)
    print(f"Wrote {len(captions)} native H3 DUV captions -> {out}")


if __name__ == "__main__":
    main()
