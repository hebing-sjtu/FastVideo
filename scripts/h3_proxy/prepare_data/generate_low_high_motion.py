#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Ask Gemini to write low_high_pipeline-compatible motion text for a validation set.

Each validation record already names its photoreal target video. Gemini watches that HIGH RGB clip
and writes only the two observational fields used by ``low_high_pipeline/write_high_motion.py``:

* ``high_motion/seg_XXXX.txt`` -- a ``[Shot 1]`` camera/action/layout narration;
* ``high_motion/seg_XXXX_subject.txt`` -- one visible-subject description.

The default backend matches ``low_high_pipeline``: Vertex Gemini authenticated by a service
account. Set ``VERTEX_SA_JSON`` / ``GOOGLE_APPLICATION_CREDENTIALS``, or the split
``VERTEX_PROJECT``, ``VERTEX_CLIENT_EMAIL`` and ``VERTEX_PRIVATE_KEY`` variables. The optional
``developer`` backend accepts ``GEMINI_API_KEY`` for AI Studio, but it is not the aligned default.

Usage::

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
    parser.add_argument("--backend", choices=("vertex", "developer"), default="vertex")
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"))
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


def generate_developer(client: Any, model: str, video: Path, poll_seconds: float) -> tuple[str, str]:
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


def vertex_credentials() -> tuple[Any, str, str]:
    """Load the same service-account forms accepted by low_high_pipeline."""
    import google.auth
    from google.oauth2 import service_account

    scope = "https://www.googleapis.com/auth/cloud-platform"
    location = os.environ.get("VERTEX_LOCATION") or os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"
    json_path = os.environ.get("VERTEX_SA_JSON") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if json_path:
        path = Path(json_path).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"Vertex service-account JSON does not exist: {path}")
        credentials = service_account.Credentials.from_service_account_file(str(path), scopes=[scope])
        project = os.environ.get("VERTEX_PROJECT") or os.environ.get("VERTEX_PROJECT_ID")
        project = project or credentials.project_id
        return credentials, str(project), location

    project = os.environ.get("VERTEX_PROJECT") or os.environ.get("VERTEX_PROJECT_ID")
    email = os.environ.get("VERTEX_CLIENT_EMAIL")
    private_key = os.environ.get("VERTEX_PRIVATE_KEY") or os.environ.get("VERTEXT_KEY")
    if project and email and private_key:
        info = {
            "type": "service_account",
            "project_id": project,
            "client_email": email,
            "private_key": private_key.strip().strip("\"'").replace("\\n", "\n"),
            "private_key_id": (
                os.environ.get("VERTEX_PRIVATE_KEY_ID")
                or os.environ.get("VERTEXT_KEY_ID")
                or ""
            ),
            "token_uri": os.environ.get("VERTEX_TOKEN_URI", "https://oauth2.googleapis.com/token"),
        }
        credentials = service_account.Credentials.from_service_account_info(info, scopes=[scope])
        return credentials, project, location

    try:
        credentials, adc_project = google.auth.default(scopes=[scope])
    except google.auth.exceptions.DefaultCredentialsError as error:
        raise SystemExit(
            "Vertex credentials are missing. Set VERTEX_SA_JSON/GOOGLE_APPLICATION_CREDENTIALS, "
            "or VERTEX_PROJECT + VERTEX_CLIENT_EMAIL + VERTEX_PRIVATE_KEY."
        ) from error
    project = project or adc_project
    if not project:
        raise SystemExit("Vertex authentication succeeded but no project id was found; set VERTEX_PROJECT.")
    return credentials, str(project), location


def generate_vertex(client: Any, model: str, video: Path) -> tuple[str, str]:
    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=USER),
                    types.Part.from_bytes(data=video.read_bytes(), mime_type="video/mp4"),
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


def build_client(backend: str) -> Any:
    try:
        from google import genai
    except ImportError:
        raise SystemExit("Missing google-genai. Install it with: python -m pip install 'google-genai>=1.0'") from None
    if backend == "developer":
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("The developer backend needs GEMINI_API_KEY in the environment.")
        return genai.Client(api_key=api_key)
    credentials, project, location = vertex_credentials()
    print(f"Vertex project={project} location={location}", flush=True)
    return genai.Client(vertexai=True, project=project, location=location, credentials=credentials)


def main() -> None:
    args = parse_args()
    records = records_from(Path(args.val_json).expanduser())
    out_dir = Path(args.out_root).expanduser() / "high_motion"
    out_dir.mkdir(parents=True, exist_ok=True)
    client = build_client(args.backend)
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
        print(f"{clip_id}: Vertex Gemini {args.model} <- {video}", flush=True)
        if args.backend == "vertex":
            detail, subject = generate_vertex(client, args.model, video)
        else:
            detail, subject = generate_developer(client, args.model, video, args.poll_seconds)
        detail_path.write_text(detail + "\n", encoding="utf-8")
        subject_path.write_text(subject + "\n", encoding="utf-8")
        generated += 1
    print(f"Done: generated={generated}, reused={reused} -> {out_dir}")


if __name__ == "__main__":
    main()
