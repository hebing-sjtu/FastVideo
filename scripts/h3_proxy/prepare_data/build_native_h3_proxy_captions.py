#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build DUV-aware, single-anchor H3 Ref2VA captions for a validation set.

The preferred source is a ``low_high_pipeline`` job. Its VLM watches each HIGH RGB clip and writes
``high_motion/seg_*.txt`` plus ``seg_*_subject.txt``. A packed six-section prompt under
``pre_data/seg_*/prompt.txt`` and the legacy corpus path ``seg_*/minimax_h3/prompt.txt`` are also
accepted. The clip-specific subject and RGB-target motion narration cannot be templated, while the
reference contract can: proxy-to-video presents one packed DUV video and one RGB anchor.

This tool keeps the VLM observations and deterministically replaces only that reference contract:

* ``<Video 1>`` becomes the DUV authority for camera, geometry, positions and timing;
* ``<Picture 1>`` becomes the sole appearance reference;
* every ``<Picture 2>`` reference is redirected to the sole ``<Picture 1>`` anchor;
* the edit summary and retention analysis are rewritten to match those two inputs.

The output is the ``{clip_id: caption}`` mapping consumed by ``set_validation_captions.py``. It is
deliberately a separate file: the source VLM prompt is provenance and must not be overwritten.

Usage::

    python scripts/h3_proxy/prepare_data/build_native_h3_proxy_captions.py \\
      --val-json /data/binghe/h3_proxy/gta_v2_cwm_validation_val6.json \\
      --root /data/binghe/low_high_job \\
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

TECHNICAL_VIDEO_DEFINITION = (
    "<Video 1> is the packed DUV proxy of this same shot: R is inverse-log depth (nearer surfaces "
    "are brighter; invalid depth and sky are 0), while G and B together encode the semantic class. "
    "It is the sole authority for camera path, framing, silhouettes, depth volumes, class regions, "
    "subject positions, action order and timing across 124 frames at 24 fps (5.17 seconds). Rebuild "
    "as photoreal matching <Picture 1>. Never copy the DUV false-color look."
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
TECHNICAL_DETAIL_PREFIX = """\
Decode <Video 1> as packed DUV, not a photoreal plate and not an ordinary RGB label map.
- R: inverse-log depth over 0.3-256 m; near is bright, far/invalid/sky is 0.
- G,B: read the pair as one semantic code; do not interpret either channel as display colour.
(G,B) classes:
- (96, 43): sky
- (32, 213): followed player
- (96, 213): other pedestrian
- (160, 213): vehicle
- (160, 128): building
- (32, 128): road
- (224, 128): ground / infrastructure
- (96, 128): vegetation
- (224, 43): terrain
- (160, 43): water
- (224, 213): prop
Use R for occlusion and camera distance and (G,B) for class regions. Reconstruct those regions as
photoreal matching <Picture 1>; never copy the DUV channel colours. The shot begins from
<Picture 1>."""
COLLEAGUE_VIDEO_DEFINITION = (
    "<Video 1> is the colored proxy/src of this same shot (semantic class fills plus blocky 3D "
    "volumes: magenta/purple road, green sidewalk/ground, cyan far field, purple trees, colored "
    "subject or vehicle). It is the sole authority for camera path, framing, silhouettes, volumes, "
    "subject positions, action order and timing across 124 frames at 24 fps (5.17 seconds). Rebuild "
    "as photoreal matching <Picture 1>. Never copy the proxy false-color look."
)
COLLEAGUE_DETAIL_PREFIX = (
    "Decode <Video 1> as a colored semantic/layout proxy with 3D volumes, not the final look. "
    "Reconstruct each class volume as photoreal matching <Picture 1>. Do not copy purple, green, "
    "or cyan fills. The shot begins from <Picture 1>."
)
PROMPT_STYLES = {
    "technical": (TECHNICAL_VIDEO_DEFINITION, TECHNICAL_DETAIL_PREFIX),
    "colleague": (COLLEAGUE_VIDEO_DEFINITION, COLLEAGUE_DETAIL_PREFIX),
}
NARRATION_MARKERS = (
    "TGT RGB motion/layout narration",
    "RGB HIGH-src motion/layout narration",
    "RGB target motion/layout narration",
)
GENERATED_SUFFIXES = ("BODY UPGRADE:", "FACE LOCK:", "END LOCK:", "FALL RELAX:")


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


def shot_narration(detail: str) -> str:
    """Extract the VLM observation, excluding the source pipeline's reference instructions."""
    start = -1
    for marker in NARRATION_MARKERS:
        marker_at = detail.find(marker)
        if marker_at >= 0:
            start = detail.find("[Shot 1]", marker_at)
            if start < 0:
                start = marker_at
            break
    if start < 0:
        start = detail.find("[Shot 1]")
    narration = detail[start:] if start >= 0 else detail
    stops = [narration.find(f"\n\n{suffix}") for suffix in GENERATED_SUFFIXES]
    stops = [stop for stop in stops if stop >= 0]
    if stops:
        narration = narration[:min(stops)]
    return narration.replace("<Picture 2>", "<Picture 1>").strip()


def render_caption(
    subject_definitions: list[str],
    subject_retention: list[str],
    narration: str,
    *,
    prompt_style: str = "technical",
) -> str:
    if not subject_definitions:
        raise ValueError("source prompt defines no <Subject N> to preserve")
    if not narration:
        raise ValueError("source prompt has no generated RGB motion/layout narration")
    try:
        video_definition, detail_prefix = PROMPT_STYLES[prompt_style]
    except KeyError:
        raise ValueError(f"unknown prompt style: {prompt_style!r}") from None
    output = [
        "subject_definitions:",
        video_definition,
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
        detail_prefix,
        "",
        "RGB target motion/layout narration (VLM watched the matching HIGH RGB clip; use it to name "
        "the scene, facing and gait, not to override <Video 1> depth, silhouettes or class regions):",
        narration,
        "",
        "overall_soundscape:",
        "N/A",
        "",
        "non_diegetic_music:",
        "N/A",
    ]
    return "\n".join(output).strip().replace("<Picture 2>", "<Picture 1>")


def adapt_native_h3_prompt(text: str, *, prompt_style: str = "technical") -> str:
    """Retain a packed prompt's VLM observation under the DUV + one-anchor contract."""
    sections = parse_sections(text)
    subject_definitions = [
        block for block in reference_blocks(sections["subject_definitions"]) if block.startswith("<Subject ")
    ]
    subject_retention = [
        block for block in reference_blocks(sections["retention_analysis"]) if block.startswith("<Subject ")
    ]
    return render_caption(
        subject_definitions,
        subject_retention,
        shot_narration(sections["detailed_description"]),
        prompt_style=prompt_style,
    )


def caption_from_low_high_observation(
    motion: Path,
    subject: Path,
    *,
    prompt_style: str = "technical",
) -> str:
    """Build directly from ``write_high_motion.py`` outputs, with no legacy prose involved."""
    subject_text = subject.read_text(encoding="utf-8").strip()
    prefix = "<Subject 1> is "
    if subject_text.startswith(prefix):
        subject_text = subject_text[len(prefix):].strip()
    return render_caption(
        [f"<Subject 1> is {subject_text}"],
        [
            "<Subject 1>: fully_preserved - facing/yaw, pose, scale, and screen placement follow "
            "<Video 1>; appearance matches <Picture 1>."
        ],
        shot_narration(motion.read_text(encoding="utf-8")),
        prompt_style=prompt_style,
    )


def caption_for_clip(root: Path, clip_id: str, *, prompt_style: str = "technical") -> tuple[str, str]:
    """Prefer raw low_high VLM observations, then packed low_high and legacy prompts."""
    motion = root / "high_motion" / f"{clip_id}.txt"
    subject = root / "high_motion" / f"{clip_id}_subject.txt"
    if motion.is_file() and subject.is_file():
        return (
            caption_from_low_high_observation(motion, subject, prompt_style=prompt_style),
            "low_high high_motion",
        )

    candidates = (
        (root / "pre_data" / clip_id / "prompt.txt", "low_high pre_data"),
        (root / clip_id / "prompt.txt", "low_high segment"),
        (root / clip_id / "minimax_h3" / "prompt.txt", "legacy minimax_h3"),
    )
    for source, label in candidates:
        if source.is_file():
            return (
                adapt_native_h3_prompt(
                    source.read_text(encoding="utf-8"),
                    prompt_style=prompt_style,
                ),
                label,
            )
    searched = ", ".join(str(path) for path, _ in candidates)
    raise FileNotFoundError(
        f"no prompt source for {clip_id}; expected {motion} + {subject}, or one of: {searched}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val-json", required=True, help="Validation JSON whose clip ids select prompts.")
    parser.add_argument(
        "--root",
        required=True,
        help="low_high_pipeline job root (preferred), pre_data root, or legacy dataset root.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=tuple(PROMPT_STYLES),
        default="technical",
        help="'colleague' uses the successful natural visual proxy wording; 'technical' writes the "
        "numeric DUV channel contract.",
    )
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
    source_counts: dict[str, int] = {}
    for record in records:
        clip_id = str(record.get("id", ""))
        if not SEGMENT_ID_PATTERN.fullmatch(clip_id):
            raise SystemExit(f"invalid or missing validation clip id: {clip_id!r}")
        try:
            captions[clip_id], source_label = caption_for_clip(
                root,
                clip_id,
                prompt_style=args.prompt_style,
            )
            source_counts[source_label] = source_counts.get(source_label, 0) + 1
        except (FileNotFoundError, ValueError) as error:
            raise SystemExit(f"{clip_id}: {error}") from error

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(captions, handle, ensure_ascii=False, indent=2)
    print(f"Wrote {len(captions)} native H3 DUV captions -> {out}")
    print(f"Prompt style: {args.prompt_style}")
    print("Sources: " + ", ".join(f"{name}={count}" for name, count in sorted(source_counts.items())))


if __name__ == "__main__":
    main()
