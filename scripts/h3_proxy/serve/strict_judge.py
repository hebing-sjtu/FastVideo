#!/usr/bin/env python3
"""A strict Qwen3.6 judge of long-take segments and of painted first frames, against the game proxy.

``segment_judge.py`` (the H3 text encoder's 8B Qwen3-VL) let through a dummy that turned into a
person, a character that turned to face the camera and a camera pushed in close; this judge is a
27B model asked about each of those failure kinds per moment, plus one look at the segment's motion.

A segment is judged on: the take's opening frame (and the OC sheet); ``samples`` moments, each a
side-by-side of the proxy (left, the game state) and the generated frame (right); per-side dummy
counts made in separate calls (shown both sides, a VLM copies the proxy's count); and, when the gate
sends a clip, contact sheets of consecutive frames for flicker, morphing and objects popping. A first
frame is judged on the proxy opening, the painting and the painting with the proxy's edges drawn on.

  serve:  HTTP for SolarWM's ``proxy_gate`` (POST /judge) and for first frames (POST /first_frame)
  audit:  judge the segments of finished takes (``segments/seg-XX/compare.mp4``) into a JSONL
  frames: judge painted first frames listed in a JSON manifest into a JSONL

Usage::

    CUDA_VISIBLE_DEVICES=7 /opt/fv-venv/bin/python strict_judge.py serve --port 7871
    CUDA_VISIBLE_DEVICES=7 /opt/fv-venv/bin/python strict_judge.py audit --manifest takes.json --out a.jsonl
"""

from __future__ import annotations

import argparse
import base64
import io
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

MODEL = Path("/workspace/models/Qwen3.6-27B")
GIVEN_FRAMES = 34  # a continuation segment's first frames only repeat the previous tail
PANEL_WIDTH = 640

# Moments (out of ``samples``) a per-moment flag must hold in before it fails the segment: one
# moment is enough for a door in a plain wall, two for the rest, which a VLM now and then reads into
# a single frame.
MOMENTS = {"structures": 1, "objects": 2, "dummy_shape": 2, "facing": 2, "artifacts": 2, "framing": 3}
EXTRA_MOMENTS = 3  # a dummy surplus must hold in every moment with dummies, and in at least this many
MISSING_BY, MISSING_MOMENTS = 2, 2
KINDS = ("dummies", "structures", "objects", "dummy_shape", "facing", "framing", "artifacts",
         "scene", "character", "motion", "score")

LAYOUT_PROMPT = """You are a strict quality inspector for a generated video game clip. Reject anything a viewer would notice.
The "opening frame" fixes the arena: its style, walls, floor, props, training dummies and lighting.{oc}
Each "moment" image is one instant of a later segment, in time order. Its LEFT half is the layout proxy, a plain grey render of the game: it is the ground truth for where the walls, pillars, crates, training dummies and the character are, which way the character faces, and how far the camera is. Its RIGHT half is the generated frame at the same instant.
For every moment fill one row, comparing RIGHT with LEFT:
- proxy_facing / facing: which way the main character faces the camera in LEFT and in RIGHT: "back" (we see the back), "side", "front" (we see the face and chest) or "absent".
- dummy_shape: "" if every training dummy in RIGHT is a plain practice dummy on a stand; otherwise name the change, e.g. "near-left dummy became a person", "dummy is a cross-shaped rack", "dummy melted".
- structures: a door, doorway, gate, arch, window, staircase or building front in RIGHT where LEFT shows plain wall or open floor; "" if none. Decorative patterns of the arena style and towers, roofs, trees or sky far behind and above the walls do not count.
- objects: a large object inside the arena in RIGHT with no counterpart in LEFT: an extra pillar, wall, statue, person, creature or vehicle; "" if none. Small props near the walls do not count.
- framing: "match" if the camera distance and height in RIGHT look like LEFT; "closer" or "farther" if the character is clearly (about 1.5x or more) larger or smaller than in LEFT.
- artifacts: melting, smeared or garbled regions, a duplicated or missing limb, an extra head, a body merged with an object, or a frame that is mostly blur; "" if none. Glowing spell effects, beams, fire, lightning, sparks, rings, smoke and motion blur of a fast dash are expected and are never artifacts.
Then judge the segment as a whole:
- scene: true if the arena changes look compared with the opening frame (wall material or colour, floor, time of day, weather).
- character: true if the main character's outfit, hair or body differs from the opening frame{oc_ref}, or there are two copies of it.
- score: an integer 0-10, strict: 9-10 only if the RIGHT halves are clean and follow LEFT closely; 6 or lower if anything above is wrong.
Answer with JSON only, no other text:
{{"rows": [{{"moment": 1, "proxy_facing": "back", "facing": "back", "dummy_shape": "", "structures": "", "objects": "", "framing": "match", "artifacts": ""}}, ...one row per moment...], "scene": false, "character": false, "score": 0, "problems": "<one short sentence naming the moments, or empty>"}}"""

MOTION_PROMPT = """These contact sheets show consecutive frames of one generated video game clip, left to right and top to bottom, with the time printed on each frame. Watch for temporal failures a viewer would notice:
- flicker: the image, lighting or colours jump between neighbouring frames;
- morphing: an object, a training dummy, the character or the arena changes shape or identity over time;
- popping: an object or person appears or disappears suddenly;
- teleport: the character jumps to another place between neighbouring frames (a fast dash with a light trail is expected and is not a teleport).
Glowing spell effects that appear and fade, and camera movement, are expected.
Answer with JSON only, no other text:
{"flicker": "", "morphing": "", "popping": "", "teleport": "", "score": <integer 0-10 for temporal coherence, strict>}
Each field is "" when there is no such failure, otherwise a short description with the frame times."""

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

FIRST_FRAME_PROMPT = """You are a strict art director checking the first frame of a video game shot before it is animated.
Image 1, "proxy", is a plain grey render of the game: the ground truth layout. It shows the walls, pillars, crates, training dummies (rounded capsule bodies on round bases), the player character (the jointed grey figure, seen from behind) and the camera.
Image 2, "painting", is the finished game screenshot that must start the shot. Image 3 is the painting with the proxy's object outlines drawn on it in red.{oc}
The painting passes only if it can replace the proxy frame exactly. Check:
- dummies: the number of training dummies, and that each sits where its grey dummy is, at the same size, on a stand; none extra, none missing, none turned into a person, statue or rack;
- layout: walls, wall tops, pillars, gateways and crates follow the red outlines; no extra pillar, wall, doorway, building or large object inside the arena;
- player: one player character, at the grey figure's place and size, standing on the ground, seen from behind{oc_ref};
- plausibility: the scene is physically coherent (nothing floating, cut in half or merged, perspective consistent, no duplicated objects, no text or watermark);
- style: one consistent finished look matching this description: {theme}
Answer with JSON only, no other text:
{{"proxy_dummies": 0, "painting_dummies": 0, "dummies_ok": true, "layout_ok": true, "player_ok": true, "plausible": true, "style_ok": true, "score": <integer 0-10, strict>, "problems": "<short list of every problem, or empty>"}}"""

FF_DIFF_PROMPT = """The image shows two views of the same game frame side by side. LEFT is a plain grey render: the ground-truth layout. RIGHT is the finished painting, which must match it object for object.
Go through LEFT object by object (every wall segment, wall corner, pillar, crate, training dummy and the player figure, including objects cut by the image border) and find its counterpart in RIGHT at the same place and size. Then search RIGHT for anything LEFT does not have.
Answer with JSON only, no other text:
{"missing": "<objects of LEFT with no counterpart in RIGHT, saying where, or empty>", "extra": "<objects or structures in RIGHT with no counterpart in LEFT, saying where: crates, barrels, benches, racks, statues, extra pillars or walls, doors, doorways, arches, gates, windows, towers, roofs, trees or buildings rising above the walls, people or creatures; or empty>", "moved": "<objects of RIGHT clearly shifted or resized from their place in LEFT, or empty>"}
Decoration on the wall surfaces (ivy, banners, shields, lanterns, carvings, cracks, trim) and leaves, snow or puddles on the floor are not objects."""

FF_COUNT_PROMPT = """The {count} images are views of game arenas. For every image count:
- dummies: training dummies (a body on a post or on a round base; in a grey render, rounded capsules with arm stubs on round bases);
- crates: boxes and crates on the floor (count each box of a stack);
- pillars: tall square pillars or columns;
- openings: doors, doorways, gates, arches, windows or gaps in the walls;
- people: human figures, the player included.
Count objects that are only partly in the image too. Answer with JSON only, no other text:
{{"rows": [{{"image": 1, "dummies": 0, "crates": 0, "pillars": 0, "openings": 0, "people": 0}}, ...one row per image...]}}"""
FF_COUNTED = ("dummies", "crates", "pillars", "openings", "people")


def thumbnail(image: Image.Image, width: int = PANEL_WIDTH) -> Image.Image:
    image = image.convert("RGB")
    return image.resize((width, max(2, round(image.height * width / image.width))), Image.BICUBIC)


def pair(proxy: Image.Image, generated: Image.Image, width: int = PANEL_WIDTH) -> Image.Image:
    left, right = thumbnail(proxy, width), thumbnail(generated, width)
    canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    draw = ImageDraw.Draw(canvas)
    draw.line([(left.width, 0), (left.width, canvas.height)], fill=(255, 255, 255), width=3)
    draw.text((8, 6), "LEFT proxy", fill=(255, 255, 0))
    draw.text((left.width + 8, 6), "RIGHT generated", fill=(255, 255, 0))
    return canvas


def contact_sheets(frames: list[Image.Image], times: list[float], cols: int = 3, rows: int = 2,
                   width: int = 480) -> list[Image.Image]:
    tiles = [thumbnail(f, width) for f in frames]
    h = tiles[0].height
    sheets = []
    for start in range(0, len(tiles), cols * rows):
        sheet = Image.new("RGB", (cols * width, rows * h))
        draw = ImageDraw.Draw(sheet)
        for k, (tile, t) in enumerate(zip(tiles[start:start + cols * rows], times[start:start + cols * rows])):
            x, y = (k % cols) * width, (k // cols) * h
            sheet.paste(tile, (x, y))
            draw.rectangle((x, y, x + 74, y + 18), fill=(0, 0, 0))
            draw.text((x + 4, y + 3), f"t={t:.2f}s", fill=(255, 255, 0))
        sheets.append(sheet)
    return sheets


def _json(raw: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", raw, flags=re.S)
    try:
        answer = json.loads(match.group(0)) if match else {}
    except ValueError:
        answer = {}
    return answer if isinstance(answer, dict) else {}


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _flag(value: Any) -> str:
    text = str(value or "").strip()
    empty = ("", "none", "no", "false", "n/a", "match", "ok", "empty", "nothing", "-", "null")
    return "" if text.lower().strip(" .<>") in empty else text


def parse_counts(raw: str, count: int) -> list[int] | None:
    rows = [row for row in _json(raw).get("rows") or [] if isinstance(row, dict)]
    if len(rows) != count:
        return None
    return [max(0, _int(r.get("standing")) or 0) + max(0, _int(r.get("fallen")) or 0) for r in rows]


def decide(layout: dict[str, Any], count: int, proxy_counts: list[int] | None, generated_counts: list[int] | None,
           motion: dict[str, Any] | None, min_score: float) -> dict[str, Any]:
    """The verdict: which failure kinds hold, from the VLM's per-moment rows and segment answers."""

    rows = [row for row in layout.get("rows") or [] if isinstance(row, dict)]
    found: dict[str, list[str]] = {kind: [] for kind in MOMENTS}
    for index, row in enumerate(rows, start=1):
        for kind in ("dummy_shape", "structures", "objects", "artifacts"):
            if _flag(row.get(kind)):
                found[kind].append(f"moment {index}: {_flag(row.get(kind))}")
        facing, proxy_facing = str(row.get("facing") or "").lower(), str(row.get("proxy_facing") or "").lower()
        if facing == "front" and proxy_facing in ("back", "side"):
            found["facing"].append(f"moment {index}: faces the camera, proxy {proxy_facing}")
        framing = str(row.get("framing") or "match").lower()
        if framing in ("closer", "farther"):
            found["framing"].append(f"moment {index}: camera {framing}")
    failures = {kind: notes for kind, notes in found.items() if len(notes) >= MOMENTS[kind]}
    if proxy_counts is not None and generated_counts is not None:
        pairs = list(zip(proxy_counts, generated_counts))
        seen = [g - p for p, g in pairs if p > 0]
        surplus = len(seen) >= EXTRA_MOMENTS and all(m >= 1 for m in seen)
        shortfall = sum(g - p <= -MISSING_BY for p, g in pairs) >= MISSING_MOMENTS
        if surplus or shortfall:
            failures["dummies"] = [f"proxy {proxy_counts} vs generated {generated_counts}"]
    for kind in ("scene", "character"):
        if layout.get(kind) is True or str(layout.get(kind)).lower() == "true":
            failures[kind] = [kind]
    motion_notes = [f"{k}: {_flag(motion.get(k))}" for k in ("flicker", "morphing", "popping", "teleport")
                    if motion and _flag(motion.get(k))]
    motion_score = _int((motion or {}).get("score"))
    if motion_notes or (motion_score is not None and motion_score < min_score):
        failures["motion"] = motion_notes or [f"motion score {motion_score}"]
    score = _int(layout.get("score"))
    score = None if score is None else max(0, min(10, score))
    if score is not None and score < min_score:
        failures["score"] = [f"score {score} < {min_score:g}"]
    return {
        "parsed": len(rows) == count and score is not None,
        "score": score,
        "motion_score": motion_score,
        "failures": [kind for kind in KINDS if kind in failures],
        "notes": failures,
        "dummy_counts": {"proxy": proxy_counts, "generated": generated_counts},
        "problems": "; ".join(n for notes in failures.values() for n in notes)[:600],
        "vlm_problems": str(layout.get("problems") or ""),
        "rows": rows,
        "motion": motion or {},
    }


class Judge:
    def __init__(self, model_path: Path = MODEL, device: str = "cuda:0", min_score: float = 7.0) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(str(model_path))
        # loaded on the CPU and moved: a device_map needs accelerate, which the venv does not have
        self.model = AutoModelForImageTextToText.from_pretrained(str(model_path), dtype=torch.bfloat16)
        self.model = self.model.to(self.device).eval()
        self.min_score = min_score
        self.lock = threading.Lock()

    @torch.no_grad()
    def ask(self, parts: list[str | Image.Image], max_new_tokens: int = 600) -> str:
        content = [{"type": "image", "image": p} if isinstance(p, Image.Image) else {"type": "text", "text": p}
                   for p in parts]
        with self.lock:
            inputs = self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", enable_thinking=False,
            ).to(self.device)
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        answer = self.processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return re.sub(r"<think>.*?</think>", "", answer, flags=re.S).strip()

    def judge(self, anchor: Image.Image, oc: Image.Image | None, pairs: list[tuple[Image.Image, Image.Image]],
              clip: list[Image.Image] | None = None, clip_times: list[float] | None = None) -> dict[str, Any]:
        started = time.time()
        parts: list[str | Image.Image] = ["Opening frame:", thumbnail(anchor)]
        if oc is not None:
            parts += ["Identity sheet of the main character:", thumbnail(oc, 448)]
        for index, (p, g) in enumerate(pairs, start=1):
            parts += [f"Moment {index}:", pair(p, g)]
        parts.append(LAYOUT_PROMPT.format(
            oc="\nThe \"identity sheet\" shows the main character." if oc is not None else "",
            oc_ref=" and the identity sheet" if oc is not None else ""))
        raw = self.ask(parts, max_new_tokens=900)
        counts = {}
        for side, index in (("proxy", 0), ("generated", 1)):
            what, guide = COUNT_GUIDES[side]
            frames: list[str | Image.Image] = []
            for k, p in enumerate(pairs, start=1):
                frames += [f"Image {k}:", thumbnail(p[index])]
            frames.append(COUNT_PROMPT.format(count=len(pairs), what=what, guide=guide))
            counts[side] = parse_counts(self.ask(frames, max_new_tokens=400), len(pairs))
        motion = None
        if clip:
            sheets = contact_sheets(clip, clip_times or [i / 24 for i in range(len(clip))])
            parts = []
            for k, sheet in enumerate(sheets, start=1):
                parts += [f"Contact sheet {k}:", sheet]
            motion = _json(self.ask(parts + [MOTION_PROMPT], max_new_tokens=300))
        verdict = decide(_json(raw), len(pairs), counts["proxy"], counts["generated"], motion, self.min_score)
        verdict["raw"] = raw
        verdict["seconds"] = round(time.time() - started, 1)
        return verdict

    def count(self, images: list[Image.Image], side: str = "generated") -> list[int] | None:
        """Training dummies (standing + fallen) per image, one image per call so none are mixed up."""
        what, guide = COUNT_GUIDES[side]
        out = []
        for image in images:
            got = parse_counts(self.ask(["Image 1:", thumbnail(image, 1344),
                                         COUNT_PROMPT.format(count=1, what=what, guide=guide)], max_new_tokens=100), 1)
            if got is None:
                return None
            out += got
        return out

    def detect(self, image: Image.Image, what: str) -> list[list[float]]:
        """Boxes ``[x0, y0, x1, y1]`` (fractions of the image) of every ``what``, from Qwen's grounding."""
        raw = self.ask(["Image:", thumbnail(image, 1344),
                        f"Locate every {what} in the image, including small far ones and ones cut by the border. "
                        'Answer with JSON only: [{"bbox_2d": [x1, y1, x2, y2]}, ...], coordinates in 0-1000.'],
                       max_new_tokens=600)
        match = re.search(r"\[.*\]", raw, flags=re.S)
        try:
            rows = json.loads(match.group(0)) if match else []
        except ValueError:
            rows = []
        boxes = []
        for row in rows if isinstance(rows, list) else []:
            box = row.get("bbox_2d") if isinstance(row, dict) else row
            if isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
                boxes.append([v / 1000.0 for v in box])
        return boxes

    def first_frame(self, proxy: Image.Image, painting: Image.Image, overlay: Image.Image,
                    oc: Image.Image | None, theme: str) -> dict[str, Any]:
        started = time.time()
        parts: list[str | Image.Image] = ["Image 1, proxy:", thumbnail(proxy, 960), "Image 2, painting:",
                                          thumbnail(painting, 960), "Image 3, painting with the proxy outlines in red:",
                                          thumbnail(overlay, 960)]
        if oc is not None:
            parts += ["Image 4, identity sheet of the player character:", thumbnail(oc, 448)]
        parts.append(FIRST_FRAME_PROMPT.format(
            theme=theme or "a finished third-person video game screenshot",
            oc="\nImage 4 is the identity sheet of the player character." if oc is not None else "",
            oc_ref=", and matching the identity sheet (Image 4)" if oc is not None else ""))
        raw = self.ask(parts, max_new_tokens=500)
        answer = _json(raw)
        score = _int(answer.get("score"))
        checks = ("dummies_ok", "layout_ok", "player_ok", "plausible", "style_ok")
        failed = [c for c in checks if answer.get(c) is not True]
        count_gap = (_int(answer.get("proxy_dummies")) or 0) != (_int(answer.get("painting_dummies")) or 0)
        if count_gap and "dummies_ok" not in failed:
            failed.append("dummies_ok")
        # the overall look above is lenient on layout: a side-by-side difference and separate counts
        diff = _json(self.ask([pair(proxy, painting, 800), FF_DIFF_PROMPT], max_new_tokens=400))
        differences = {k: _flag(diff.get(k)) for k in ("missing", "extra", "moved") if _flag(diff.get(k))}
        if differences:
            failed.append("differences")
        rows = _json(self.ask(["Image 1:", thumbnail(proxy, 960), "Image 2:", thumbnail(painting, 960),
                               FF_COUNT_PROMPT.format(count=2)], max_new_tokens=300)).get("rows") or []
        counts = {}
        if len(rows) == 2 and all(isinstance(r, dict) for r in rows):
            counts = {k: [_int(rows[0].get(k)), _int(rows[1].get(k))] for k in FF_COUNTED}
            gaps = [k for k, (p, g) in counts.items() if p is not None and g is not None and p != g]
            if gaps:
                failed.append("counts:" + ",".join(gaps))
        problems = "; ".join([str(answer.get("problems") or "")] + [f"{k}: {v}" for k, v in differences.items()])
        return {
            "parsed": bool(answer) and score is not None,
            "pass": bool(answer) and not failed and score is not None and score >= self.min_score,
            "score": score,
            "failed": failed,
            "counts": counts,
            "differences": differences,
            **{k: answer.get(k) for k in ("proxy_dummies", "painting_dummies")},
            "problems": problems.strip("; "),
            "raw": raw,
            "seconds": round(time.time() - started, 1),
        }


def encode_image(image: Image.Image | None) -> str | None:
    if image is None:
        return None
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode()


class Remote:
    """A running ``serve`` judge behind the same two calls as :class:`Judge`."""

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        import urllib.request

        request = urllib.request.Request(self.url + path, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=1800) as response:
            return json.loads(response.read())

    def judge(self, anchor, oc, pairs, clip=None, clip_times=None) -> dict[str, Any]:
        return self._post("/judge", {
            "anchor_b64": encode_image(anchor), "oc_b64": encode_image(oc),
            "pairs_b64": [[encode_image(p), encode_image(g)] for p, g in pairs],
            "clip_b64": [encode_image(c) for c in clip or []], "clip_fps": 1.0,
            "clip_frames": list(clip_times or [])})

    def detect(self, image, what) -> list[list[float]]:
        return self._post("/detect", {"image_b64": encode_image(image), "what": what})["boxes"]

    def count(self, images, side="generated") -> list[int] | None:
        return self._post("/count", {"images_b64": [encode_image(i) for i in images], "side": side})["counts"]

    def first_frame(self, proxy, painting, overlay, oc, theme) -> dict[str, Any]:
        return self._post("/first_frame", {"proxy_b64": encode_image(proxy), "painting_b64": encode_image(painting),
                                           "overlay_b64": encode_image(overlay), "oc_b64": encode_image(oc),
                                           "theme": theme})


def decode_image(data: str | None) -> Image.Image | None:
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB") if data else None


def open_image(path: str | None) -> Image.Image | None:
    return Image.open(path).convert("RGB") if path else None


# ----------------------------------------------------------------------
# serve
# ----------------------------------------------------------------------


def serve(args: argparse.Namespace) -> int:
    judge = Judge(Path(args.model), args.device, args.min_score)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:
            pass

        def _reply(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            self._reply(200, {"ok": True, "model": str(args.model)})

        def do_POST(self) -> None:
            try:
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path.rstrip("/").endswith("detect"):
                    self._reply(200, {"boxes": judge.detect(decode_image(request["image_b64"]), request["what"])})
                    return
                if self.path.rstrip("/").endswith("count"):
                    images = [decode_image(i) for i in request["images_b64"]]
                    self._reply(200, {"counts": judge.count(images, request.get("side", "generated"))})
                    return
                if self.path.rstrip("/").endswith("first_frame"):
                    verdict = judge.first_frame(decode_image(request["proxy_b64"]), decode_image(request["painting_b64"]),
                                                decode_image(request["overlay_b64"]), decode_image(request.get("oc_b64")),
                                                request.get("theme", ""))
                    print(f"[judge] first frame {request.get('tag', '')} pass={verdict['pass']} score={verdict['score']} "
                          f"failed={verdict['failed']} {str(verdict.get('problems'))[:200]} ({verdict['seconds']}s)",
                          flush=True)
                else:
                    pairs = [(decode_image(p), decode_image(g)) for p, g in request["pairs_b64"]]
                    clip = [decode_image(c) for c in request.get("clip_b64") or []]
                    fps = float(request.get("clip_fps") or 24)
                    times = [i / fps for i in request.get("clip_frames") or range(len(clip))]
                    verdict = judge.judge(decode_image(request["anchor_b64"]), decode_image(request.get("oc_b64")),
                                          pairs, clip, times)
                    print(f"[judge] {request.get('tag', '')} score={verdict['score']} motion={verdict['motion_score']} "
                          f"failures={verdict['failures']} {verdict['problems'][:240]} ({verdict['seconds']}s)", flush=True)
                self._reply(200, verdict)
            except Exception as exc:  # the gate treats an unusable verdict as a pass
                print(f"[judge] error: {type(exc).__name__}: {exc}", flush=True)
                self._reply(500, {"error": f"{type(exc).__name__}: {exc}"})

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[judge] {args.model} serving on {args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


# ----------------------------------------------------------------------
# audit / frames
# ----------------------------------------------------------------------


def read_video(path: Path) -> np.ndarray:
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(path), format="ffmpeg")
    try:
        return np.stack([np.asarray(frame) for frame in reader])
    finally:
        reader.close()


def audit(args: argparse.Namespace) -> int:
    """Judge every segment of the takes a manifest lists: [{"name", "take", "anchor", "oc"}, ...]."""

    judge = Remote(args.url) if args.url else Judge(Path(args.model), args.device, args.min_score)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as sink:
        for entry in json.loads(Path(args.manifest).read_text(encoding="utf-8")):
            anchor, oc = open_image(entry["anchor"]), open_image(entry.get("oc"))
            for folder in sorted(Path(entry["take"]).glob("segments/seg-*")):
                index = int(folder.name.split("-")[1])
                if entry.get("segments") is not None and index not in entry["segments"]:
                    continue
                compare = read_video(folder / "compare.mp4")
                width = compare.shape[2] // 2
                first = 0 if index == 0 else GIVEN_FRAMES
                picks = np.linspace(first, len(compare) - 1, args.samples).round().astype(int)
                pairs = [(Image.fromarray(compare[i, :, :width]), Image.fromarray(compare[i, :, width:2 * width]))
                         for i in picks]
                clip_ids = np.linspace(first, len(compare) - 1, args.clip).round().astype(int)
                clip = [Image.fromarray(compare[i, :, width:2 * width]) for i in clip_ids]
                verdict = judge.judge(anchor, oc, pairs, clip, [i / 24 for i in clip_ids])
                row = {"take": entry["name"], "segment": index, **verdict}
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
                print(f"[audit] {row['take']} seg{index:02d} score={verdict['score']} motion={verdict['motion_score']} "
                      f"failures={verdict['failures']} {verdict['problems'][:200]} ({verdict['seconds']}s)", flush=True)
    return 0


def frames(args: argparse.Namespace) -> int:
    """Judge painted first frames: [{"name", "proxy", "painting", "overlay", "oc", "theme"}, ...]."""

    judge = Remote(args.url) if args.url else Judge(Path(args.model), args.device, args.min_score)
    with Path(args.out).open("a", encoding="utf-8") as sink:
        for entry in json.loads(Path(args.manifest).read_text(encoding="utf-8")):
            verdict = judge.first_frame(open_image(entry["proxy"]), open_image(entry["painting"]),
                                        open_image(entry["overlay"]), open_image(entry.get("oc")), entry.get("theme", ""))
            sink.write(json.dumps({"name": entry["name"], **verdict}, ensure_ascii=False) + "\n")
            sink.flush()
            print(f"[frames] {entry['name']} pass={verdict['pass']} score={verdict['score']} failed={verdict['failed']} "
                  f"{str(verdict.get('problems'))[:240]}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("serve")
    run.add_argument("--host", default="0.0.0.0")
    run.add_argument("--port", type=int, default=7871)
    check = commands.add_parser("audit")
    check.add_argument("--manifest", required=True,
                       help='JSON list of {"name", "take": out dir with segments/seg-XX/compare.mp4, "anchor", "oc"}')
    check.add_argument("--samples", type=int, default=8)
    check.add_argument("--clip", type=int, default=12)
    check.add_argument("--out", required=True)
    paint = commands.add_parser("frames")
    paint.add_argument("--manifest", required=True)
    paint.add_argument("--out", required=True)
    for child in (check, paint):
        child.add_argument("--url", help="a running serve judge (e.g. http://127.0.0.1:7871) instead of loading one")
    for child in (run, check, paint):
        child.add_argument("--model", default=str(MODEL))
        child.add_argument("--device", default="cuda:0")
        child.add_argument("--min-score", type=float, default=7.0)
    args = parser.parse_args(argv)
    return {"serve": serve, "audit": audit, "frames": frames}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
