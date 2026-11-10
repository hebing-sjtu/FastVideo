#!/usr/bin/env python3
"""Qwen3-VL judge of one generated long-take segment against its layout proxy.

A long take is generated segment by segment, and a segment can drift from the game: an extra
training dummy, a door in a wall the proxy shows as plain, a new arena look. The judge sees the
take's opening frame (and the OC sheet), then a few moments of the segment, each a side-by-side
of the proxy (left, the game state) and the generated frame (right), and answers per failure kind.
The model is the H3 text encoder's own Qwen3-VL with its LM head, as ``proxy_service.py`` uses it.

  serve: an HTTP endpoint for SolarWM's ``proxy_serve`` gate (POST /judge, JSON in and out)
  audit: judge the segments of finished takes (``segments/seg-XX/compare.mp4``) into a JSONL

Usage::

    CUDA_VISIBLE_DEVICES=7 /opt/venv/bin/python segment_judge.py serve --port 7870
    CUDA_VISIBLE_DEVICES=7 /opt/venv/bin/python segment_judge.py audit --manifest takes.json \\
        --out audit.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

MODEL = Path("/data/models/MiniMax-H3")
KINDS = ("dummies", "structures", "scene", "character", "broken")
GIVEN_FRAMES = 34  # a continuation segment's first frames only repeat the previous tail
PANEL_WIDTH = 640

PROMPT = """You are checking one segment of a generated video game clip for consistency failures.
Image 1 is the opening frame of the clip: it fixes the arena style, its walls, floor, props, training dummies and lighting.{oc}
Each of the last {count} images (Images {first} to {last}) is one moment of a later segment, in time order. Its LEFT half is the layout proxy, a plain grey render of the game state: it is the ground truth for where the walls, crates, training dummies and the character are. Its RIGHT half is the generated frame at the same moment.
For every one of these images fill one row:
- openings: any door, doorway, gate, arch, window, staircase or building front in the RIGHT half at a spot where the LEFT half shows a plain wall or open floor, e.g. "wooden door in the left wall"; "" if none. Decorative patterns that belong to the arena style of Image 1, and towers, roofs, trees or sky far behind and above the wall, are not openings.
Then judge the whole segment:
- scene: the arena changes look compared with Image 1 (wall material or colour, floor, time of day);
- character: the main character's outfit, hair or body differs from Image 1{oc_ref}, the character vanishes, or there are two copies of it;
- broken: melting, smeared or garbled regions, or a frame that is mostly blur.
Glowing spell effects, beams, fire, lightning, sparks, rings, smoke, motion blur and camera movement are expected and are never failures.
Answer with JSON only, no other text:
{{"rows": [{{"image": {first}, "openings": ""}}, ...one row per image...], "scene": false, "character": false, "broken": false, "score": <integer 0-10 for how well the right halves follow the proxy and Image 1>, "problems": "<one short sentence naming the image numbers, or empty>"}}"""

# Counted in calls that see only one side: shown both, the VLM copies the proxy's count.
COUNT_PROMPT = """The {count} images are frames of {what}.
{guide}
For every image count the training dummies that stand upright and the ones lying knocked over on the ground. Count a dummy that is only partly in the frame too. Answer with JSON only, no other text:
{{"rows": [{{"image": 1, "standing": 0, "fallen": 0}}, ...one row per image...]}}"""
COUNT_GUIDES = {
    "proxy": ("a plain grey render of a game", "A training dummy is a rounded capsule body with short arm stubs on a round base. "
              "The jointed humanoid figure is the player character, not a dummy; boxes and wall blocks are not dummies."),
    "generated": ("a video game", "A training dummy is a practice figure of wood, straw, metal or cloth on a post or "
                  "a round base. The player character, other people, crates, barrels, posts and pillars are not dummies."),
}

# VLM counts of small far dummies are off by one now and then, and a grey dummy next to the
# grey player reads as none, so a moment whose proxy count is 0 is skipped and a surplus must
# hold in every other moment (at least EXTRA_MOMENTS of them). On the 2026-11-10 long takes this
# flagged 3 of 58 segments, each with an extra dummy in every moment.
EXTRA_MOMENTS = 3
MISSING_BY, MISSING_MOMENTS = 2, 2


def thumbnail(image: Image.Image, width: int = PANEL_WIDTH) -> Image.Image:
    image = image.convert("RGB")
    return image.resize((width, max(2, round(image.height * width / image.width))), Image.BICUBIC)


def pair(proxy: Image.Image, generated: Image.Image, label: str) -> Image.Image:
    left, right = thumbnail(proxy), thumbnail(generated)
    canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    draw = ImageDraw.Draw(canvas)
    draw.line([(left.width, 0), (left.width, canvas.height)], fill=(255, 255, 255), width=3)
    draw.text((8, 6), f"{label} proxy", fill=(255, 255, 0))
    draw.text((left.width + 8, 6), f"{label} generated", fill=(255, 255, 0))
    return canvas


def _count(row: dict[str, Any], key: str) -> int:
    try:
        return max(0, int(row.get(key) or 0))
    except (TypeError, ValueError):
        return 0


def _json(raw: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", raw, flags=re.S)
    try:
        answer = json.loads(match.group(0)) if match else {}
    except ValueError:
        answer = {}
    return answer if isinstance(answer, dict) else {}


def parse_counts(raw: str, count: int) -> list[int] | None:
    """Total dummies per image, or None unless the answer has one row per image."""

    rows = [row for row in _json(raw).get("rows") or [] if isinstance(row, dict)]
    if len(rows) != count:
        return None
    return [_count(row, "standing") + _count(row, "fallen") for row in rows]


def parse(raw: str, proxy_counts: list[int] | None = None,
          generated_counts: list[int] | None = None) -> dict[str, Any]:
    """The VLM's rows and segment flags, with ``dummies`` and ``structures`` decided here.

    dummies: every moment with dummies in the proxy shows more in the generated frame (at least
    ``EXTRA_MOMENTS`` such moments), or ``MISSING_BY`` fewer in ``MISSING_MOMENTS``; structures:
    any moment names an opening the proxy does not have.
    """

    answer = _json(raw)
    rows = [row for row in answer.get("rows") or [] if isinstance(row, dict)]
    pairs = (list(zip(proxy_counts, generated_counts))
             if proxy_counts is not None and generated_counts is not None else [])
    misses = [g - p for p, g in pairs]
    openings = [f"image {row.get('image')}: {row['openings']}" for row in rows
                if str(row.get("openings") or "").strip().lower() not in ("", "none", "no")]
    seen = [g - p for p, g in pairs if p > 0]
    surplus = len(seen) >= EXTRA_MOMENTS and all(m >= 1 for m in seen)
    shortfall = sum(m <= -MISSING_BY for m in misses) >= MISSING_MOMENTS
    verdict: dict[str, Any] = {
        "dummies": surplus or shortfall,
        "structures": bool(openings),
        **{kind: bool(answer.get(kind)) for kind in ("scene", "character", "broken")},
    }
    try:
        verdict["score"] = max(0, min(10, int(answer.get("score"))))
    except (TypeError, ValueError):
        verdict["score"] = None
    verdict["dummy_counts"] = {"proxy": proxy_counts, "generated": generated_counts}
    verdict["dummy_misses"] = misses
    verdict["openings"] = openings
    verdict["problems"] = str(answer.get("problems") or "")
    verdict["rows"] = rows
    verdict["parsed"] = bool(rows)
    verdict["raw"] = raw
    return verdict


class Judge:
    def __init__(self, model_path: Path = MODEL, device: str = "cuda:0", work: Path = Path("/tmp/segment-judge")):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        merged = work / "vlm-processor"
        merged.mkdir(parents=True, exist_ok=True)
        for folder in ("tokenizer", "processor"):
            for source in (model_path / folder).iterdir():
                link = merged / source.name
                if not link.exists():
                    link.symlink_to(source)
        chat = model_path / "text_encoder" / "chat_template.json"
        if chat.is_file() and not (merged / "chat_template.json").exists():
            (merged / "chat_template.json").symlink_to(chat)
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(str(merged))
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path / "text_encoder"), dtype=torch.bfloat16
        ).to(self.device).eval()
        self.lock = threading.Lock()

    @torch.no_grad()
    def ask(self, images: list[Image.Image], text: str, max_new_tokens: int = 200) -> str:
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]
        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        with self.lock:
            inputs = self.processor(text=[prompt], images=images, return_tensors="pt").to(self.device)
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        answer = self.processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return re.sub(r"<think>.*?</think>", "", answer, flags=re.S).strip()

    def judge(self, anchor: Image.Image, oc: Image.Image | None,
              pairs: list[tuple[Image.Image, Image.Image]]) -> dict[str, Any]:
        images = [thumbnail(anchor)]
        if oc is not None:
            images.append(thumbnail(oc, 384))
        first = len(images) + 1
        images += [pair(p, g, f"Image {first + i}:") for i, (p, g) in enumerate(pairs)]
        text = PROMPT.format(
            oc="\nImage 2 is the identity sheet of the main character." if oc is not None else "",
            oc_ref=" and Image 2" if oc is not None else "",
            count=len(pairs),
            first=first,
            last=len(images),
        )
        started = time.time()
        counts = {}
        for side, index in (("proxy", 0), ("generated", 1)):
            what, guide = COUNT_GUIDES[side]
            frames = [thumbnail(p[index]) for p in pairs]
            prompt = COUNT_PROMPT.format(count=len(frames), what=what, guide=guide)
            counts[side] = parse_counts(self.ask(frames, prompt, max_new_tokens=300), len(frames))
        verdict = parse(self.ask(images, text, max_new_tokens=500), counts["proxy"], counts["generated"])
        verdict["failures"] = [kind for kind in KINDS if verdict[kind]]
        verdict["seconds"] = round(time.time() - started, 1)
        return verdict


def open_image(path: str | None) -> Image.Image | None:
    return Image.open(path).convert("RGB") if path else None


# ----------------------------------------------------------------------
# serve
# ----------------------------------------------------------------------


def serve(args: argparse.Namespace) -> int:
    judge = Judge(device=args.device)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:
            pass

        def _reply(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            self._reply(200, {"ok": True})

        def do_POST(self) -> None:
            try:
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                pairs = [(Image.open(p).convert("RGB"), Image.open(g).convert("RGB")) for p, g in request["pairs"]]
                verdict = judge.judge(open_image(request["anchor"]), open_image(request.get("oc")), pairs)
                print(f"[judge] {request.get('tag', '')} score={verdict['score']} "
                      f"failures={verdict['failures']} {verdict['problems'][:160]} ({verdict['seconds']}s)", flush=True)
                self._reply(200, verdict)
            except Exception as exc:  # the gate treats an unusable verdict as a pass
                print(f"[judge] error: {type(exc).__name__}: {exc}", flush=True)
                self._reply(500, {"error": f"{type(exc).__name__}: {exc}"})

    # A replacement loads while the old judge still serves, then takes the port the moment it frees.
    print(f"[judge] loaded; binding {args.host}:{args.port}", flush=True)
    while True:
        try:
            server = ThreadingHTTPServer((args.host, args.port), Handler)
            break
        except OSError:
            time.sleep(0.2)
    print(f"[judge] serving on {args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


# ----------------------------------------------------------------------
# audit
# ----------------------------------------------------------------------


def read_video(path: Path) -> np.ndarray:
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(path), format="ffmpeg")
    try:
        return np.stack([np.asarray(frame) for frame in reader])
    finally:
        reader.close()


def sample_pairs(compare: np.ndarray, first: int, count: int) -> list[tuple[Image.Image, Image.Image]]:
    """``compare`` frames are proxy | generated side by side; ``count`` moments from ``first`` on."""

    width = compare.shape[2] // 2
    picks = np.linspace(first, len(compare) - 1, count).round().astype(int)
    return [(Image.fromarray(compare[i, :, :width]), Image.fromarray(compare[i, :, width:2 * width])) for i in picks]


def audit(args: argparse.Namespace) -> int:
    """Judge every segment of the takes a manifest lists: [{"name", "take", "anchor", "oc"}, ...]."""

    judge = Judge(device=args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as sink:
        for entry in json.loads(Path(args.manifest).read_text(encoding="utf-8")):
            anchor, oc = open_image(entry["anchor"]), open_image(entry.get("oc"))
            for folder in sorted(Path(entry["take"]).glob("segments/seg-*")):
                index = int(folder.name.split("-")[1])
                compare = read_video(folder / "compare.mp4")
                pairs = sample_pairs(compare, 0 if index == 0 else GIVEN_FRAMES, args.samples)
                verdict = judge.judge(anchor, oc, pairs)
                row = {"take": entry["name"], "segment": index, **verdict}
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
                print(f"[audit] {row['take']} seg{index:02d} score={verdict['score']} "
                      f"failures={verdict['failures']} {verdict['problems'][:160]}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("serve")
    run.add_argument("--host", default="0.0.0.0")
    run.add_argument("--port", type=int, default=7870)
    check = commands.add_parser("audit")
    check.add_argument("--manifest", required=True,
                       help='JSON list of {"name", "take": out dir with segments/seg-XX/compare.mp4, "anchor", "oc"}')
    check.add_argument("--samples", type=int, default=6)
    check.add_argument("--out", required=True)
    for child in (run, check):
        child.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    return serve(args) if args.command == "serve" else audit(args)


if __name__ == "__main__":
    sys.exit(main())
