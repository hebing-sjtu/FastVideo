#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Ask Gemini to write low_high_pipeline-compatible motion text for a validation set.

Each validation record already names its photoreal target video. Gemini watches that HIGH RGB clip
and writes only the two observational fields used by ``low_high_pipeline/write_high_motion.py``:

* ``high_motion/seg_XXXX.txt`` -- a ``[Shot 1]`` camera/action/layout narration;
* ``high_motion/seg_XXXX_subject.txt`` -- one visible-subject description.

The script is resume-safe and never accepts an API key on the command line. Set ``GEMINI_API_KEY``
in the environment so it does not land in shell history.

Usage::

    export GEMINI_API_KEY='...'
    python scripts/h3_proxy/prepare_data/generate_low_high_motion.py \
      --val-json /data/binghe/h3_proxy/gta_v2_cwm_validation_val6.json \
      --out-root /data/binghe/h3_proxy/low_high_val6
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

SYSTEM = """\
You write MiniMax-H3 detailed_description motion bodies.
Return JSON only: {"detailed_description": "...", "subject_one_line": "..."}.
No markdown fences."""

USER = """\
The attached video is the HIGH-FIDELITY RGB target gameplay clip: 124 frames at 24 fps,
approximately 5.17 seconds. It is the motion twin of a packed DUV proxy that will be labeled
<Video 1> in MiniMax-H3. Watch this HIGH RGB video, not the DUV proxy.

Write:
1) detailed_description -- start with "[Shot 1]" and describe one continuous take. Cover the
actual environment, camera class and path, camera height, framing, subject path, opening and later
screen-space facing/yaw, gait/action order, visible vehicles or pedestrians, and major scene
events. Cite <Video 1> as the authority for motion and layout. Observe only: do not invent turns,
collisions, locomotion, people, vehicles, or destruction. "Facing forward" is ambiguous: explicitly
say back/nape, face-to-camera, left_profile, or right_profile. Do not describe DUV colours, depth
encoding, restyling, or prompt mechanics. Do not use clock timestamps. Use 180-320 words.
2) subject_one_line -- one sentence defining <Subject 1> from the visible target: person/vehicle
type, hair, wardrobe, carried objects, and other stable appearance cues. Do not add restyle language.

Return JSON only."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val-json", required=True)
    parser.add_argument("--out-root", required=True, help="Writes <out-root>/high_motion/.")
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=3.0)
    return parser.parse_args()


def records_from(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise SystemExit(f"{path} holds no validation records under 'data'.")
    return records


def target_video(record: dict[str, Any]) -> Path:
    raw = record.get("target_path") or record.get("ref_video")
    if not isinstance(raw, str) or not raw:
        raise ValueError("record has neither target_path nor ref_video")
    path = Path(raw).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def parse_response(text: str, clip_id: str) -> tuple[str, str]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, stop = text.find("{"), text.rfind("}")
        if start < 0 or stop <= start:
            raise ValueError(f"{clip_id}: Gemini response is not JSON: {text[:200]}") from None
        payload = json.loads(text[start:stop + 1])
    detail = str(payload.get("detailed_description") or "").strip()
    subject = str(payload.get("subject_one_line") or "").strip()
    if not detail.startswith("[Shot 1]"):
        raise ValueError(f"{clip_id}: detailed_description must start with [Shot 1]: {detail[:160]}")
    if not subject:
        raise ValueError(f"{clip_id}: Gemini returned an empty subject_one_line")
    return detail, subject


def wait_until_ready(client: Any, uploaded: Any, poll_seconds: float) -> Any:
    while True:
        state = str(getattr(getattr(uploaded, "state", None), "name", "")).upper()
        if state in {"", "ACTIVE", "SUCCEEDED"}:
            return uploaded
        if state in {"FAILED", "ERROR"}:
            raise RuntimeError(f"Gemini file processing failed: {uploaded}")
        time.sleep(poll_seconds)
        uploaded = client.files.get(name=uploaded.name)


def generate(client: Any, model: str, video: Path, poll_seconds: float) -> tuple[str, str]:
    from google.genai import types

    uploaded = wait_until_ready(client, client.files.upload(file=str(video)), poll_seconds)
    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=USER),
                        types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type),
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM,
                response_mime_type="application/json",
                temperature=0.2,
                max_output_tokens=4000,
            ),
        )
        return parse_response(str(response.text or ""), video.stem)
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception as error:  # noqa: BLE001
            print(f"WARNING: could not delete uploaded Gemini file {uploaded.name}: {error}")


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY in the environment; do not pass credentials in chat or argv.")
    try:
        from google import genai
    except ImportError:
        raise SystemExit("Missing google-genai. Install it with: python -m pip install 'google-genai>=1.0'") from None

    records = records_from(Path(args.val_json).expanduser())
    out_dir = Path(args.out_root).expanduser() / "high_motion"
    out_dir.mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=api_key)
    generated = reused = 0
    for record in records:
        clip_id = str(record.get("id") or "")
        if not clip_id.startswith("seg_"):
            raise SystemExit(f"invalid validation id: {clip_id!r}")
        detail_path = out_dir / f"{clip_id}.txt"
        subject_path = out_dir / f"{clip_id}_subject.txt"
        if not args.overwrite and detail_path.is_file() and subject_path.is_file():
            print(f"{clip_id}: reuse")
            reused += 1
            continue
        video = target_video(record)
        print(f"{clip_id}: Gemini {args.model} <- {video}", flush=True)
        detail, subject = generate(client, args.model, video, args.poll_seconds)
        detail_path.write_text(detail + "\n", encoding="utf-8")
        subject_path.write_text(subject + "\n", encoding="utf-8")
        generated += 1
    print(f"Done: generated={generated}, reused={reused} -> {out_dir}")


if __name__ == "__main__":
    main()
