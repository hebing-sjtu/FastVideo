#!/usr/bin/env python3
"""Intranet conversion service for the H3 Ref2VA proxy LoRA.

A request is an arbitrary proxy video (a DUV, depth or semantic visualisation, or a restyled
RGB render), a first frame (``<Picture 1>``), optionally an OC identity sheet (``<Picture 2>``)
and free text. Qwen3-VL -- the H3 text encoder's own weights, run with their language-model head
-- names the proxy modality and writes a caption in the training captions' style. The request is
then encoded into 124-frame cache samples exactly as ``encode_proxy_samples.py`` encodes an omni
training row, and spooled to SolarWM's resident ``proxy_loop serve`` workers; a proxy longer than
one segment becomes a take of segments overlapping by 39 frames.

Two GPUs: ``cuda:0`` holds the cache encoders (VAE + FastVideo Qwen3-VL), ``cuda:1`` the HF
Qwen3-VL used for generation. Usage::

    CUDA_VISIBLE_DEVICES=6,7 PYTHONPATH=/workspace/FastVideo-oc /opt/fv-venv/bin/python \\
      scripts/h3_proxy/serve/proxy_service.py --spool /workspace/serve/spool \\
      --jobs /workspace/serve/jobs --port 7860
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prepare_data"))

from encode_proxy_samples import (  # noqa: E402
    Encoders, crop_to_aspect, fit_frames, init_single_process_distributed, read_anchor_image, scale_to_short_edge,
)
from oc_sheets import OC_PROMPTS  # noqa: E402

FPS = 24
SEGMENT_FRAMES = 124
STRIDE_FRAMES = 85
HEIGHT, WIDTH = 704, 1280
PROXY_HEIGHT, PROXY_WIDTH = 352, 640
ANCHOR_SHORT_EDGE = 2048
OC_SHORT_EDGE = 1024
QWEN_VIDEO_FPS = 2.0
MODALITIES = ("duv", "depth", "semantic", "style")
OPENING_ROLE, CONTINUATION_ROLE = "w0_omni", "wn_omni"
JOB_SCHEMA = "h3-proxy-serve-job.v1"
SEGMENT_SPAN = f"[0.00s-{SEGMENT_FRAMES / FPS:.2f}s]"

MODALITY_GUIDE = {
    "depth": "a grey depth map: all three colour channels equal, smooth gradients and no texture",
    "semantic": "flat, untextured colour regions marking people, vehicles and props on one uniform "
    "background colour, with hard edges and only a handful of distinct colours",
    "duv": "a packed geometry video: the red channel is a smooth depth gradient while green and blue "
    "hold flat class colours, so it looks like a red-tinted depth map with flat-coloured silhouettes",
    "style": "an ordinary RGB rendering with shading and texture (game footage, a low-poly or grey-box "
    "render, a cartoon, or real footage)",
}

CLASSIFY_PROMPT = """The {count} images are frames sampled in order from one conditioning video for a video generator. Decide which kind of video it is:
- depth: {depth}; usually bright is near and dark is far.
- semantic: {semantic}.
- duv: {duv}.
- style: {style}.
Answer with JSON only, no other text: {{"modality": "depth|semantic|duv|style", "near_is_bright": true|false|null, "reason": "<one short sentence>"}}. near_is_bright is null unless the modality is depth or duv."""

CAPTION_EXAMPLES = (
    "A 3D video game captures an urban asphalt street intersection surrounded by buildings and glowing signs at "
    "night. Dim night lighting emanates from overhead streetlamps and neon signs. Inside sits a man in a light grey "
    "jacket and jeans within a black SUV. The black SUV rolls forward slowly along the lane before easing to a halt "
    "at the junction. Then, the man opens the front driver door, steps down onto the dark pavement, and stands "
    "outside beside the SUV looking forward.",
    "This realistic 3D video game animation depicts a broad city street intersection lined with multistory "
    "commercial buildings and billboards. Bright midday sunlight illuminates the paved street and casts distinct "
    "shadows on the ground. A black SUV drives forward along the street toward the red intersection markings, then "
    "slows to a halt. From within the cabin, a man in a light grey sweatshirt and dark pants pushes the driver's door "
    "open, steps out onto the asphalt road, and stands upright facing forward beside the SUV.",
    "A 3D video game animation depicts a paved sidewalk and parking entrance beside multistory city buildings under "
    "bright daylight. A man in a white jacket and dark jeans starts at a parking barrier and walks forward along the "
    "sidewalk toward an intersection. As he approaches another man in a purple t-shirt and red trousers standing on "
    "the pavement, the man winds up and punches the other man directly in the face, and the other man collapses onto "
    "the concrete. The man stands in place over the fallen figure looking down at the pavement.",
)

CAPTION_PROMPT = """Image 1 is the first frame of a video to generate; it fixes the look: the medium, the scene, the lighting and how the characters appear.{oc}
The last {count} images are frames sampled in order from a {modality} conditioning video ({guide}). They fix the layout, the camera and the motion over {seconds:.2f} seconds.{user}
Write the caption of the generated video in exactly the style of these examples:
{examples}
Rules: one paragraph of 60 to 110 words in English. Start with the medium as Image 1 shows it (for example "A 3D video game captures ..." or "A live-action film captures ..."), then the setting and the lighting, then the main subject's appearance, then what happens, in order, following the motion in the conditioning frames. Describe the result as an ordinary video: never mention depth maps, segmentation, conditioning, frames or images. Output only the caption."""


# ----------------------------------------------------------------------
# Media
# ----------------------------------------------------------------------


def read_proxy_video(path: Path) -> np.ndarray:
    """``[T, H, W, 3]`` uint8 frames resampled to 24 fps by whole-frame selection."""
    from fastvideo.pipelines.basic.minimax_h3.reference import decode_reference_video, resample_reference_frames

    frames, source_fps, _ = decode_reference_video(path)
    frames = resample_reference_frames(frames, source_fps)
    if frames.shape[0] < 1:
        raise ValueError("the proxy video has no frames")
    return frames


def write_mp4(path: Path, frames: np.ndarray, fps: int = FPS) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = width - width % 2, height - height % 2
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "movflags": "+faststart"}
        for frame in frames:
            image = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame[:stream.height, :stream.width]),
                                               format="rgb24")
            container.mux(stream.encode(image))
        container.mux(stream.encode())


def pixel_guess(frames: np.ndarray) -> dict[str, Any]:
    """A colour-statistics opinion on the modality, shown beside the VLM's."""
    picks = frames[np.linspace(0, len(frames) - 1, num=min(8, len(frames))).astype(int)]
    small = picks[:, ::4, ::4].astype(np.int16)
    chroma = float(np.abs(small[..., 0] - small[..., 1]).mean() + np.abs(small[..., 1] - small[..., 2]).mean())
    colours = int(np.median([len(np.unique(f.reshape(-1, 3), axis=0)) for f in small]))
    gb = int(np.median([len(np.unique(f[..., 1:].reshape(-1, 2), axis=0)) for f in small]))
    reds = int(np.median([len(np.unique(f[..., 0])) for f in small]))
    if chroma < 4.0:
        modality = "depth"
    elif gb <= 64 and reds > 64:
        modality = "duv"
    elif colours <= 64:
        modality = "semantic"
    else:
        modality = "style"
    return {"modality": modality, "chroma": round(chroma, 2), "colours": colours, "gb_pairs": gb, "reds": reds}


def to_reference(frames: np.ndarray, modality: str, near_is_bright: bool) -> np.ndarray:
    """The user's proxy as the ``[T, 352, 640, 3]`` uint8 reference the LoRA was trained on."""
    if modality == "depth":
        grey = (frames[..., 0] * 0.299 + frames[..., 1] * 0.587 + frames[..., 2] * 0.114).round()
        grey = grey.clip(0, 255).astype(np.uint8)
        if not near_is_bright:
            grey = 255 - grey
        grey = fit_frames(grey, PROXY_HEIGHT, PROXY_WIDTH, "center-crop", codes=False, what="proxy")
        return np.repeat(grey[..., None], 3, axis=-1)
    if modality in {"semantic", "duv"}:
        return fit_frames(frames, PROXY_HEIGHT, PROXY_WIDTH, "center-crop", codes=True, what="proxy",
                          code_resize="nearest")
    return fit_frames(frames, PROXY_HEIGHT, PROXY_WIDTH, "center-crop", codes=False, what="proxy")


def plan_segments(frames: int, max_segments: int) -> tuple[int, int]:
    """``(segments, output frames)`` for a proxy of ``frames`` frames."""
    count = 1 if frames <= SEGMENT_FRAMES else 1 + math.ceil((frames - SEGMENT_FRAMES) / STRIDE_FRAMES)
    count = max(1, min(count, max_segments))
    return count, min(frames, SEGMENT_FRAMES + STRIDE_FRAMES * (count - 1))


def segment_windows(reference: np.ndarray, count: int) -> list[np.ndarray]:
    """Each segment's 124 frames, the proxy held on its last frame past its end."""
    total = SEGMENT_FRAMES + STRIDE_FRAMES * (count - 1)
    if reference.shape[0] < total:
        tail = np.repeat(reference[-1:], total - reference.shape[0], axis=0)
        reference = np.concatenate([reference, tail])
    return [reference[i * STRIDE_FRAMES:i * STRIDE_FRAMES + SEGMENT_FRAMES] for i in range(count)]


def thumbnail(image: Image.Image, short_edge: int = 448) -> Image.Image:
    scale = short_edge / min(image.size)
    return image.resize((max(28, round(image.size[0] * scale)), max(28, round(image.size[1] * scale))),
                        Image.Resampling.BICUBIC)


def sample_frames(frames: np.ndarray, count: int) -> list[Image.Image]:
    picks = np.linspace(0, len(frames) - 1, num=min(count, len(frames))).round().astype(int)
    return [thumbnail(Image.fromarray(frames[i]), 352) for i in picks]


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------


class Vlm:
    """The H3 text encoder's Qwen3-VL with its LM head, for classification and captions."""

    def __init__(self, model_path: Path, device: str, work: Path) -> None:
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
        # No device_map: that path needs `accelerate`, which the FastVideo venv does not carry.
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(str(model_path / "text_encoder"),
                                                                     dtype=torch.bfloat16).to(self.device).eval()

    @torch.no_grad()
    def ask(self, images: list[Image.Image], text: str, max_new_tokens: int = 320) -> str:
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]
        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=images or None, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        answer = self.processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return re.sub(r"<think>.*?</think>", "", answer, flags=re.S).strip()

    def classify(self, frames: np.ndarray) -> dict[str, Any]:
        images = sample_frames(frames, 6)
        text = CLASSIFY_PROMPT.format(count=len(images), **MODALITY_GUIDE)
        raw = self.ask(images, text, max_new_tokens=120)
        match = re.search(r"\{.*\}", raw, flags=re.S)
        try:
            answer = json.loads(match.group(0)) if match else {}
        except ValueError:
            answer = {}
        modality = str(answer.get("modality") or "").strip().lower()
        return {
            "modality": modality if modality in MODALITIES else None,
            "near_is_bright": answer.get("near_is_bright"),
            "reason": str(answer.get("reason") or ""),
            "raw": raw,
        }

    def caption(self, anchor: Image.Image, oc: Image.Image | None, window: np.ndarray, modality: str,
                user_text: str) -> str:
        images = [thumbnail(anchor)]
        oc_line = ""
        if oc is not None:
            images.append(thumbnail(oc))
            oc_line = (" Image 2 is the identity sheet of the main subject; describe the subject as it looks "
                       "there.")
        proxies = sample_frames(window, 8)
        images += proxies
        user = (f'\nThe user describes the video as: "{user_text.strip()}". Follow it wherever the images do '
                "not contradict it." if user_text.strip() else "")
        text = CAPTION_PROMPT.format(oc=oc_line, count=len(proxies), modality=modality,
                                     guide=MODALITY_GUIDE[modality], seconds=len(window) / FPS, user=user,
                                     examples="\n".join(f"- {example}" for example in CAPTION_EXAMPLES))
        caption = " ".join(self.ask(images, text).split())
        return f"{SEGMENT_SPAN} {caption.strip(chr(34))}"


class CacheEncoder:
    """Build serve samples with the training cache's encoders and document layout."""

    def __init__(self, model_path: Path, device: str) -> None:
        self.encoders = Encoders(model_path, device, cwm_system=OPENING_ROLE, qwen_video_fps=QWEN_VIDEO_FPS)

    def still_target(self, anchor_path: Path) -> torch.Tensor:
        """The placeholder target: the first frame held for 124 frames, so latent 0 is the anchor."""
        with Image.open(anchor_path) as raw:
            still = crop_to_aspect(raw.convert("RGB"), HEIGHT, WIDTH).resize((WIDTH, HEIGHT), Image.Resampling.BICUBIC)
        frames = np.repeat(np.asarray(still)[None], SEGMENT_FRAMES, axis=0)
        pixels = torch.from_numpy(frames).permute(3, 0, 1, 2)[None].float().div_(255.0)
        return self.encoders.encode_pixels(pixels)

    def sample(self, *, window: np.ndarray, modality: str, role: str, caption: str, anchor: Image.Image,
               target: torch.Tensor, anchor_latent: torch.Tensor, oc: tuple[Image.Image, torch.Tensor, str] | None,
               info: dict[str, Any]) -> dict[str, Any]:
        enc = self.encoders
        pixels = torch.from_numpy(window.copy()).permute(3, 0, 1, 2)[None].float().div_(255.0)
        document = f"Reference modality: {modality} video.\n{caption}"
        embedding, tags = enc.encode_text(document, anchor, [window], cwm_system=role)
        sample: dict[str, Any] = {
            "vae_latent": target,
            "proxy_latents": torch.stack([enc.encode_pixels(pixels)]),
            "anchor_latent": anchor_latent,
            "text_embedding": embedding,
            "text_token_tags": tags,
            "info": {
                "num_frames": SEGMENT_FRAMES,
                "pixel_size": (HEIGHT, WIDTH),
                "qwen_video_fps": QWEN_VIDEO_FPS,
                "prompt": caption,
                "conditioning_prompt": document,
                "anchor_source": "take_anchor",
                "cwm_system": role,
                "fit": "center-crop",
                "proxy_references": [modality],
                "proxy_modality": modality,
                "proxy_variants": list(MODALITIES),
                "code_resize": "nearest",
                "target_kind": "service",
                **info,
            },
        }
        if oc is not None:
            sheet, oc_latent, subject = oc
            sheeted = f"Reference modality: {modality} video.\n{OC_PROMPTS[subject]}\n{caption}"
            oc_embedding, oc_tags = enc.encode_text(sheeted, anchor, [window], cwm_system=role, pictures=(sheet, ))
            sample.update({
                "oc_latent": oc_latent,
                "text_embedding": oc_embedding.to(torch.bfloat16),
                "text_token_tags": oc_tags,
                "text_embedding_no_oc": embedding.to(torch.bfloat16),
                "text_token_tags_no_oc": tags,
            })
            sample["info"].update({"oc_subject": subject, "oc_short_edge": OC_SHORT_EDGE,
                                   "oc_conditioning_prompt": sheeted})
        return sample


# ----------------------------------------------------------------------
# Service
# ----------------------------------------------------------------------


class Service:

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.spool = Path(args.spool)
        self.jobs = Path(args.jobs)
        self.jobs.mkdir(parents=True, exist_ok=True)
        (self.spool / "queue").mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        init_single_process_distributed()
        model = Path(args.model)
        print("[serve] loading cache encoders on cuda:0", flush=True)
        self.cache = CacheEncoder(model, "cuda:0")
        print("[serve] loading VLM on cuda:1", flush=True)
        self.vlm = Vlm(model, "cuda:1", self.jobs.parent)
        print("[serve] models ready", flush=True)

    # analysis ---------------------------------------------------------

    def analyze(self, job: Path, proxy_path: str, anchor_path: str | None, oc_path: str | None, text: str,
                modality_choice: str, depth_choice: str, verbatim: bool, max_seconds: float) -> dict[str, Any]:
        if not proxy_path:
            raise ValueError("请上传 proxy 视频")
        if not anchor_path:
            raise ValueError("请上传首图（<Picture 1>）")
        frames = read_proxy_video(Path(proxy_path))
        guess = pixel_guess(frames)
        max_segments = max(1, 1 + int((max_seconds * FPS - SEGMENT_FRAMES) // STRIDE_FRAMES))
        count, out_frames = plan_segments(len(frames), min(max_segments, self.args.max_segments))
        with self.lock:
            verdict = self.vlm.classify(frames) if modality_choice == "auto" else {}
            modality = modality_choice if modality_choice != "auto" else (verdict.get("modality") or guess["modality"])
            if depth_choice == "auto":
                near = verdict.get("near_is_bright")
                near_is_bright = True if near is None else bool(near)
            else:
                near_is_bright = depth_choice == "near_bright"
            reference = to_reference(frames, modality, near_is_bright)
            windows = segment_windows(reference, count)
            anchor = Image.open(anchor_path).convert("RGB")
            oc = Image.open(oc_path).convert("RGB") if oc_path else None
            if verbatim and text.strip():
                body = text.strip()
                captions = [body if body.startswith("[") else f"{SEGMENT_SPAN} {body}"] * count
            else:
                captions = [self.vlm.caption(anchor, oc, window, modality, text) for window in windows]
        write_mp4(job / "proxy_reference.mp4", reference[:out_frames])
        analysis = {
            "modality": modality,
            "modality_source": "user" if modality_choice != "auto" else ("vlm" if verdict.get("modality") else
                                                                         "pixel-statistics"),
            "near_is_bright": near_is_bright,
            "vlm": verdict,
            "pixel_guess": guess,
            "proxy_frames": int(len(frames)),
            "segments": count,
            "output_frames": out_frames,
            "captions": captions,
        }
        (job / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
        return analysis

    # encoding and spooling --------------------------------------------

    def submit(self, job: Path, proxy_path: str, anchor_path: str, oc_path: str | None, oc_subject: str,
               analysis: dict[str, Any], captions: list[str], steps: int, seed: int) -> str:
        modality = analysis["modality"]
        frames = read_proxy_video(Path(proxy_path))
        reference = to_reference(frames, modality, bool(analysis["near_is_bright"]))
        windows = segment_windows(reference, int(analysis["segments"]))
        if len(captions) == 1 and len(windows) > 1:
            captions = captions * len(windows)
        if len(captions) != len(windows):
            raise ValueError(f"{len(windows)} 段需要 {len(windows)} 条 caption（空行分隔），收到 {len(captions)} 条")
        job_id = job.name
        with self.lock:
            anchor = read_anchor_image(Path(anchor_path), np.zeros((1, HEIGHT, WIDTH, 3), np.uint8), ANCHOR_SHORT_EDGE,
                                       aspect=(HEIGHT, WIDTH))
            target = self.cache.still_target(Path(anchor_path))
            anchor_latent = self.cache.encoders.encode_keyframe(anchor)
            oc = None
            if oc_path:
                with Image.open(oc_path) as raw:
                    sheet = scale_to_short_edge(raw.convert("RGB"), OC_SHORT_EDGE)
                oc = (sheet, self.cache.encoders.encode_keyframe(sheet), oc_subject)
            segments = []
            for index, (window, caption) in enumerate(zip(windows, captions, strict=True)):
                role = OPENING_ROLE if index == 0 else CONTINUATION_ROLE
                sample = self.cache.sample(window=window, modality=modality, role=role, caption=caption,
                                           anchor=anchor, target=target, anchor_latent=anchor_latent, oc=oc,
                                           info={"take": job_id, "take_segment": index, "take_anchor": "anchor"})
                path = job / f"{job_id}-s{index:02d}.pt"
                torch.save(sample, path)
                segments.append({"name": path.stem, "pt": str(path), "cwm_system": role})
            torch.cuda.empty_cache()
        payload = {"schema": JOB_SCHEMA, "segments": segments, "fps": FPS, "frames": int(analysis["output_frames"]),
                   "steps": int(steps), "seed": int(seed)}
        (job / "job.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        scratch = self.spool / "queue" / f".{job_id}.tmp"
        scratch.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(scratch, self.spool / "queue" / f"{job_id}.json")
        return job_id

    def status(self, job_id: str) -> tuple[str, dict[str, Any] | None]:
        done = self.spool / "done" / f"{job_id}.json"
        if done.is_file():
            return "done", json.loads(done.read_text(encoding="utf-8"))
        progress = self.spool / "running" / f"{job_id}.progress.json"
        if (self.spool / "running" / f"{job_id}.json").is_file():
            state = json.loads(progress.read_text(encoding="utf-8")) if progress.is_file() else {"done": 0}
            return "running", state
        queued = sorted(self.spool.glob("queue/*.json"), key=lambda p: p.stat().st_mtime)
        names = [p.stem for p in queued]
        return "queued", {"position": names.index(job_id) + 1 if job_id in names else 0, "queued": len(names)}

    def workers(self) -> list[dict[str, Any]]:
        rows = []
        for path in sorted(self.spool.glob("workers/*.json")):
            try:
                beat = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            beat["age_s"] = round(time.time() - float(beat.get("updated", 0)), 1)
            rows.append(beat)
        return rows

    def new_job(self) -> Path:
        job = self.jobs / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        job.mkdir(parents=True)
        return job


def split_captions(text: str) -> list[str]:
    return [" ".join(part.split()) for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]


def stage_inputs(job: Path, proxy: str, anchor: str | None, oc: str | None) -> tuple[str, str | None, str | None]:
    """Copy the uploads into the job directory, so the job keeps its own inputs."""
    import shutil

    def copy(source: str | None, stem: str) -> str | None:
        if not source:
            return None
        target = job / f"{stem}{Path(source).suffix.lower() or '.bin'}"
        shutil.copyfile(source, target)
        return str(target)

    return copy(proxy, "proxy"), copy(anchor, "anchor"), copy(oc, "oc")


def workers_markdown(service: Service) -> str:
    rows = service.workers()
    if not rows:
        return "**推理 worker：** 尚未就绪"
    live = [r for r in rows if r["age_s"] < 30 and r.get("state") != "stopped"]
    busy = [r for r in live if r.get("state") == "busy"]
    queued = len(list(service.spool.glob("queue/*.json")))
    weights = rows[0].get("weights_id", "")
    return (f"**推理 worker：** {len(live)} 在线 / {len(busy)} 忙碌，排队 {queued} 个 · 模型 `{weights}`")


def build_ui(service: Service) -> Any:
    import gradio as gr

    def analyze(proxy, anchor, oc, oc_subject, text, modality, depth, verbatim, max_seconds, state):
        job = service.new_job()
        proxy, anchor, oc = stage_inputs(job, proxy, anchor, oc)
        analysis = service.analyze(job, proxy, anchor, oc, text or "", modality, depth, verbatim, max_seconds)
        state = {"job": str(job), "proxy": proxy, "anchor": anchor, "oc": oc, "analysis": analysis}
        return (analysis_markdown(analysis), "\n\n".join(analysis["captions"]), str(job / "proxy_reference.mp4"),
                state)

    def generate(proxy, anchor, oc, oc_subject, text, modality, depth, verbatim, max_seconds, captions, steps, seed,
                 state):
        try:
            if not state or not captions.strip():
                yield "VLM 分析中…", None, None, None, captions, None, state
                md, captions, preview, state = analyze(proxy, anchor, oc, oc_subject, text, modality, depth, verbatim,
                                                       max_seconds, state)
                yield md, None, None, None, captions, preview, state
            job = Path(state["job"])
            yield "编码条件（Qwen3-VL + VAE）…", None, None, None, captions, gr.update(), state
            job_id = service.submit(job, state["proxy"], state["anchor"], state["oc"], oc_subject, state["analysis"],
                                    split_captions(captions), int(steps), int(seed))
            started = time.time()
            while True:
                phase, info = service.status(job_id)
                elapsed = int(time.time() - started)
                if phase == "done":
                    break
                if phase == "running":
                    message = f"生成中：第 {info.get('done', 0)}/{state['analysis']['segments']} 段完成 · {elapsed}s"
                else:
                    message = f"排队中：第 {info['position']}/{info['queued']} 位 · {elapsed}s"
                yield message, None, None, None, captions, gr.update(), state
                time.sleep(3)
            if info.get("error"):
                yield f"生成失败：{info['error']}", None, None, info, captions, gr.update(), state
                return
            summary = (f"完成：{info['frames']} 帧（{info['frames'] / FPS:.1f}s），{len(info.get('segments', []))} 段，"
                       f"用时 {info['seconds']}s，worker `{info.get('worker')}`，任务 `{job_id}`")
            yield summary, info["take"], info["compare"], info, captions, gr.update(), state
        except Exception as exc:  # surfaced in the UI
            yield f"出错：{type(exc).__name__}: {exc}", None, None, None, captions, gr.update(), state

    with gr.Blocks(title="H3 Proxy → Video") as demo:
        gr.Markdown("## H3 Ref2VA proxy 转视频（BD + OC：R5-B@600）\n"
                    "上传任意 proxy 视频（DUV / depth / semantic 可视化，或风格化渲染）、首图、可选的人物设定图（OC）"
                    "和文字描述。VLM 先判断 proxy 种类并写标准化 caption（可编辑），再生成 704×1280、24 fps 的视频；"
                    "长于 5.17 s 的 proxy 按 124 帧一段、重叠 39 帧接续生成。")
        status_bar = gr.Markdown(workers_markdown(service))
        state = gr.State(None)
        with gr.Row():
            with gr.Column():
                proxy = gr.File(label="Proxy 视频（原样读取，不转码）", file_types=["video"], type="filepath")
                with gr.Row():
                    anchor = gr.Image(label="首图 <Picture 1>", type="filepath")
                    oc = gr.Image(label="OC 人物设定图 <Picture 2>（可选）", type="filepath")
                oc_subject = gr.Radio(["protagonist", "vehicle"], value="protagonist", label="OC 主体")
                text = gr.Textbox(label="文字描述（可选，中英文均可；VLM 会据此写标准化 caption）", lines=3)
                with gr.Row():
                    modality = gr.Dropdown(["auto", *MODALITIES], value="auto", label="Proxy 种类")
                    depth = gr.Dropdown([("自动", "auto"), ("近处亮", "near_bright"), ("近处暗", "near_dark")],
                                        value="auto", label="Depth 明暗")
                verbatim = gr.Checkbox(label="直接把我的文字当 caption（不经 VLM 改写）", value=False)
                with gr.Row():
                    max_seconds = gr.Slider(5, 60, value=15, step=0.5, label="最长输出秒数")
                    steps = gr.Slider(10, 50, value=30, step=1, label="采样步数")
                    seed = gr.Number(value=42, precision=0, label="种子")
                with gr.Row():
                    analyze_btn = gr.Button("① VLM 分析")
                    generate_btn = gr.Button("② 生成", variant="primary")
            with gr.Column():
                analysis_md = gr.Markdown()
                captions = gr.Textbox(label="标准化 caption（每段一条，空行分隔；可编辑后再生成）", lines=8)
                preview = gr.Video(label="模型看到的 proxy 参考（352×640）")
                status = gr.Markdown()
                result = gr.Video(label="生成结果")
                compare = gr.Video(label="对比：proxy | 生成")
                details = gr.JSON(label="任务详情")
        inputs = [proxy, anchor, oc, oc_subject, text, modality, depth, verbatim, max_seconds]
        analyze_btn.click(analyze, inputs=[*inputs, state], outputs=[analysis_md, captions, preview, state])
        generate_btn.click(generate, inputs=[*inputs, captions, steps, seed, state],
                           outputs=[status, result, compare, details, captions, preview, state])
        for component in (proxy, anchor, oc, modality, depth, verbatim, max_seconds):
            component.change(lambda: (None, ""), outputs=[state, captions])
        gr.Timer(10).tick(lambda: workers_markdown(service), outputs=status_bar)
    return demo


def analysis_markdown(analysis: dict[str, Any]) -> str:
    vlm = analysis.get("vlm") or {}
    guess = analysis["pixel_guess"]
    lines = [f"**Proxy 种类：`{analysis['modality']}`**（来源：{analysis['modality_source']}）"]
    if vlm:
        lines.append(f"- VLM：`{vlm.get('modality')}` — {vlm.get('reason', '')}")
    lines.append(f"- 像素统计：`{guess['modality']}`（色差 {guess['chroma']}，颜色数 {guess['colours']}）")
    if vlm.get("modality") and vlm["modality"] != guess["modality"]:
        lines.append("- ⚠ VLM 与像素统计不一致，必要时手动指定种类")
    if analysis["modality"] == "depth":
        lines.append(f"- Depth：{'近处亮' if analysis['near_is_bright'] else '近处暗（已反相为近处亮）'}")
    lines.append(f"- {analysis['proxy_frames']} 帧 @24fps → {analysis['segments']} 段，输出 {analysis['output_frames']} 帧"
                 f"（{analysis['output_frames'] / FPS:.1f}s）")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="/data/models/MiniMax-H3")
    p.add_argument("--spool", required=True, help="the spool the SolarWM serve workers poll")
    p.add_argument("--jobs", required=True, help="per-request inputs and encoded samples")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--max-segments", type=int, default=12)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    service = Service(args)
    demo = build_ui(service)
    demo.queue(default_concurrency_limit=16).launch(
        server_name=args.host, server_port=args.port, allowed_paths=[str(Path(args.spool)), str(Path(args.jobs))],
        show_error=True)


if __name__ == "__main__":
    main()
