# SPDX-License-Identifier: Apache-2.0
"""Encode the H3 Ref2VA omni (+OC) corpus into Wan2.2 Ref2VA cache samples for SolarWM.

Same rows, modalities, crops and OC rule as the MiniMax-H3 cache (``encode_proxy_samples.py``),
re-encoded with the Wan VAE and umT5 at a Wan family's native clip:

``wan22_ti2v_5b``   121 frames of the 24 fps timeline, 704x1280, 48-channel Wan2.2 VAE (16x).
``wan22_i2v_a14b``  81 frames at 16 fps -- timeline frames ``round(1.5 i)`` -- 480x832,
                    16-channel Wan2.1 VAE (8x), plus the official I2V first-frame latent.

Each sample (schema ``solarwm.wan22-ref2va-sample.v1``, read by
``solarwm.backends.wan22.ref2va.data``) holds ``target_latent``, ``proxy_latent`` (the row's one
modality), ``anchor_latent`` (``take_anchor`` or the target's first frame, cropped to the target
aspect and scaled to ``--anchor-short-edge``), ``oc_latent`` for rows with an ``oc`` sheet,
``i2v_latent`` for A14B, and umT5 rows: ``prompt_embeds`` of the document with the OC line and
``prompt_embeds_no_oc`` without it (only the first when the row has no sheet). The document is
the modality line, the OC line when present, then the caption; umT5 reads at most 512 tokens.

Run in the FastVideo venv (media helpers) with SolarWM's source on the path (Wan codecs)::

    /opt/fv-venv/bin/python scripts/h3_proxy/prepare_data/encode_wan_proxy_samples.py \\
      --family wan22_ti2v_5b --solarwm-src /workspace/SolarWM-wan/src \\
      --wan-root /data/models/Wan2.2-TI2V-5B \\
      --manifest /data/binghe/h3_proxy/manifests/gta_record_0930_omni_train_oc.jsonl \\
      --include /data/binghe/h3_proxy/manifests/gta_record_0930_omni_704p_ref640_clean_oc.txt \\
      --root /data/binghe/datasets/gta_record_0930-v0003 \\
      --output /workspace/wan_proxy/cache/gta0930_omni_5b_121f_oc \\
      --list-prefix /data/binghe/wan_proxy/cache/gta0930_omni_5b_121f_oc \\
      --shard-index 0 --num-shards 8

Write to pod-local disk and copy the finished directory to ``/data`` afterwards; the GCS mount
does not support the atomic rename each sample is written with. Each shard writes
``samples-shardNNN-of-MMM.txt`` naming its samples under ``--list-prefix``; concatenate them
into the training list.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

SCHEMA = "solarwm.wan22-ref2va-sample.v1"
FAMILIES = {
    "wan22_ti2v_5b": {
        "pixel_frames": 121, "fps": 24, "stride": 1.0, "height": 704, "width": 1280,
        "proxy_height": 352, "proxy_width": 640, "anchor_short_edge": 2048, "oc_short_edge": 1024,
        "compression": 16, "vae": "Wan2.2_VAE.pth",
    },
    "wan22_i2v_a14b": {
        "pixel_frames": 81, "fps": 16, "stride": 1.5, "height": 480, "width": 832,
        "proxy_height": 240, "proxy_width": 416, "anchor_short_edge": 960, "oc_short_edge": 704,
        "compression": 8, "vae": "Wan2.1_VAE.pth",
    },
}
TIME_PREFIX = re.compile(r"\[\d+(?:\.\d+)?s\s*-\s*\d+(?:\.\d+)?s\]\s*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", required=True, choices=sorted(FAMILIES))
    parser.add_argument("--solarwm-src", required=True, help="SolarWM checkout's src/ (Wan VAE and umT5).")
    parser.add_argument("--wan-root", required=True, help="Official Wan2.2 directory with VAE, umT5 and tokenizer.")
    parser.add_argument("--manifest", required=True, help="Omni (+OC) JSONL, one row per clip.")
    parser.add_argument("--include", default="", help="Optional list of .pt paths or names to keep (e.g. the H3 clean list).")
    parser.add_argument("--root", required=True, help="Dataset root the manifest paths are relative to.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--list-prefix", default="",
                        help="Directory the shard's sample list names instead of --output (the /data copy).")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--proxy-variants", nargs="+", default=["duv", "depth", "semantic", "style"])
    parser.add_argument("--modality-prompt", choices=("label", "gta-record"), default="label")
    parser.add_argument("--strip-time-prefix", action="store_true",
                        help="Drop '[0.00s-5.17s]' markers; they time the 124-frame H3 clip.")
    for key in ("height", "width", "proxy_height", "proxy_width", "anchor_short_edge", "oc_short_edge"):
        parser.add_argument(f"--{key.replace('_', '-')}", type=int, default=0, help="Override the family default.")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must lie in [0, --num-shards)")
    profile = FAMILIES[args.family]
    for key in ("height", "width", "proxy_height", "proxy_width", "anchor_short_edge", "oc_short_edge"):
        if not getattr(args, key):
            setattr(args, key, int(profile[key]))
    multiple = 2 * int(profile["compression"])
    for key in ("height", "width", "proxy_height", "proxy_width"):
        if getattr(args, key) % multiple:
            parser.error(f"--{key.replace('_', '-')} must be a multiple of {multiple} for {args.family}")
    # Attributes the H3 media helpers read.
    args.num_frames = timeline_frames(args.family)
    args.fit = "center-crop"
    args.code_resize = "nearest"
    args.proxy_variants = tuple(args.proxy_variants)
    args.proxy_references = ()
    return args


def timeline_frames(family: str) -> int:
    profile = FAMILIES[family]
    return int(round((profile["pixel_frames"] - 1) * profile["stride"])) + 1


def timeline_indices(family: str) -> list[int]:
    profile = FAMILIES[family]
    return [int(round(i * profile["stride"])) for i in range(int(profile["pixel_frames"]))]


def to_vae_pixels(frames_thwc: np.ndarray) -> torch.Tensor:
    """``[T, H, W, 3]`` uint8 to ``[1, 3, T, H, W]`` float in ``[-1, 1]``."""
    return torch.from_numpy(np.ascontiguousarray(frames_thwc)).permute(3, 0, 1, 2)[None].float().div_(127.5).sub_(1.0)


def image_pixels(image: Image.Image) -> torch.Tensor:
    return to_vae_pixels(np.array(image.convert("RGB"))[None])


class WanCodec:

    def __init__(self, family: str, wan_root: Path, device: str) -> None:
        from solarwm.backends.wan22.runtime.components import Wan5BVAE, WanA14BVAE, WanTextEncoder

        self.family = family
        self.device = torch.device(device)
        vae_path = wan_root / FAMILIES[family]["vae"]
        self.vae = (Wan5BVAE(vae_path) if family == "wan22_ti2v_5b" else WanA14BVAE(vae_path)).to(self.device)
        self.text = WanTextEncoder(wan_root / "models_t5_umt5-xxl-enc-bf16.pth", wan_root / "google" / "umt5-xxl")
        self.text.to(self.device)

    @torch.no_grad()
    def encode(self, pixels_bcthw: torch.Tensor) -> torch.Tensor:
        """``[1, 3, T, H, W]`` pixels to a ``[C, T', h, w]`` bf16 latent."""
        latent = self.vae.encode(pixels_bcthw.to(self.device))  # [1, T', C, h, w]
        return latent[0].permute(1, 0, 2, 3).to(torch.bfloat16).cpu().contiguous()

    @torch.no_grad()
    def prompts(self, documents: list[str]) -> list[torch.Tensor]:
        """umT5 rows of each document, trimmed to its token count."""
        _, mask = self.text.tokenizer(documents, return_mask=True, add_special_tokens=True)
        lengths = mask.gt(0).sum(dim=1).tolist()
        embeds = self.text(documents)["prompt_embeds"]
        rows = []
        for document, row, length in zip(documents, embeds, lengths, strict=True):
            if int(length) >= 512:
                print(f"  warning: document reaches umT5's 512-token limit and is truncated: {document[:80]!r}...")
            rows.append(row[:int(length)].to(torch.bfloat16).cpu().contiguous())
        return rows


def documents_for(entry: dict[str, Any], modality: str, args: argparse.Namespace) -> tuple[str, str | None]:
    """The umT5 documents ``(with OC line, without)``; the second is ``None`` for rows without a sheet."""
    from encode_proxy_samples import GTA_RECORD_MODALITY_PROMPTS
    from oc_sheets import OC_PROMPTS

    caption = str(entry["prompt"]).strip()
    if args.strip_time_prefix:
        caption = TIME_PREFIX.sub("", caption).strip()
    if args.modality_prompt == "gta-record":
        modality_line = GTA_RECORD_MODALITY_PROMPTS[modality]
    else:
        modality_line = f"Reference modality: {modality} video."
    plain = f"{modality_line}\n{caption}"
    if not entry.get("oc"):
        return plain, None
    subject = str(entry.get("oc_subject") or "")
    if subject not in OC_PROMPTS:
        raise ValueError(f"oc_subject must be one of {sorted(OC_PROMPTS)}, got {subject!r}")
    return f"{modality_line}\n{OC_PROMPTS[subject]}\n{caption}", plain


def encode_entry(entry: dict[str, Any], codec: WanCodec, args: argparse.Namespace, root: Path) -> dict[str, Any]:
    from encode_proxy_samples import (anchor_aspect, entry_proxy_references, load_proxy_clips, read_anchor_image,
                                      read_video_frames, scale_to_short_edge, take_anchor_path)

    family = args.family
    indices = timeline_indices(family)
    modality = entry_proxy_references(entry, args)[0]
    target_frames = read_video_frames(root / str(entry["target"]), args.num_frames, args.height, args.width, args.fit,
                                      crop=entry.get("target_crop"))
    target = target_frames[indices]
    proxies = load_proxy_clips(entry, args, root)
    if len(proxies) != 1:
        raise ValueError(f"expected one proxy reference, got {len(proxies)}")
    proxy_pixels = proxies[0][0][:, :, indices].mul(2.0).sub(1.0)
    anchor = read_anchor_image(take_anchor_path(entry, root), target_frames, args.anchor_short_edge,
                               aspect=anchor_aspect(args))
    with_oc, without_oc = documents_for(entry, modality, args)
    documents = [with_oc] if without_oc is None else [with_oc, without_oc]
    prompts = codec.prompts(documents)

    sample: dict[str, Any] = {
        "schema": SCHEMA,
        "family": family,
        "target_latent": codec.encode(to_vae_pixels(target)),
        "proxy_latent": codec.encode(proxy_pixels),
        "anchor_latent": codec.encode(image_pixels(anchor)),
        "prompt_embeds": prompts[0],
    }
    if family == "wan22_i2v_a14b":
        first = to_vae_pixels(target)
        first[:, :, 1:] = 0.0
        sample["i2v_latent"] = codec.encode(first)
    info: dict[str, Any] = {
        "name": str(entry.get("name") or Path(str(entry["target"])).stem),
        "prompt": str(entry["prompt"]),
        "conditioning_prompt": without_oc or with_oc,
        "proxy_modality": modality,
        "pixel_frames": int(FAMILIES[family]["pixel_frames"]),
        "fps": int(FAMILIES[family]["fps"]),
        "timeline_fps": 24,
        "timeline_indices": indices,
        "pixel_size": (int(args.height), int(args.width)),
        "proxy_pixel_size": (int(args.proxy_height), int(args.proxy_width)),
        "anchor_pixel_size": (int(anchor.size[1]), int(anchor.size[0])),
        "anchor_source": "take_anchor" if entry.get("take_anchor") else "target_first_frame",
        "fit": args.fit,
        "code_resize": args.code_resize,
        "modality_prompt": args.modality_prompt,
        "strip_time_prefix": bool(args.strip_time_prefix),
        "prompt_tokens": int(prompts[0].shape[0]),
        **{key: entry[key] for key in ("target_kind", "target_style", "proxy_style", "take", "take_segment")
           if entry.get(key)},
    }
    if without_oc is not None:
        with Image.open(root / str(entry["oc"])) as raw:
            sheet = scale_to_short_edge(raw.convert("RGB"), args.oc_short_edge)
        sample["oc_latent"] = codec.encode(image_pixels(sheet))
        sample["prompt_embeds_no_oc"] = prompts[1]
        info.update({
            "oc": str(entry["oc"]),
            "oc_subject": str(entry["oc_subject"]),
            "oc_short_edge": int(args.oc_short_edge),
            "oc_pixel_size": (int(sheet.size[1]), int(sheet.size[0])),
            "oc_conditioning_prompt": with_oc,
            "prompt_tokens_no_oc": int(prompts[1].shape[0]),
        })
    sample["info"] = info
    return sample


def included_names(path: str) -> set[str] | None:
    if not path:
        return None
    names = set()
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if value and not value.startswith("#"):
            names.add(Path(value).stem if value.endswith(".pt") else value)
    return names


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.solarwm_src).expanduser().resolve()))
    from encode_proxy_samples import assign_mixed_modalities, preflight_media

    root = Path(args.root).expanduser()
    output = Path(args.output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, encoding="utf-8") as handle:
        entries = [json.loads(line) for line in handle if line.strip() and not line.startswith("#")]
    entries = assign_mixed_modalities(entries, tuple(args.proxy_variants))
    keep = included_names(args.include)
    if keep is not None:
        entries = [entry for entry in entries
                   if str(entry.get("name") or Path(str(entry["target"])).stem).replace("/", "__") in keep]
        print(f"--include keeps {len(entries)} of the manifest rows ({len(keep)} names listed)")
    entries = entries[args.shard_index::args.num_shards]
    if args.limit:
        entries = entries[:args.limit]
    if not entries:
        raise SystemExit(f"Shard {args.shard_index}/{args.num_shards} is empty")
    preflight_media(entries, root, args)
    codec = WanCodec(args.family, Path(args.wan_root).expanduser(), args.device)

    written = skipped = failed = 0
    listing = output / f"samples-shard{args.shard_index:03d}-of-{args.num_shards:03d}.txt"
    paths = []
    for index, entry in enumerate(entries):
        name = str(entry.get("name") or Path(str(entry["target"])).stem).replace("/", "__")
        out_path = output / f"{name}.pt"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            paths.append(out_path)
            continue
        started = time.monotonic()
        try:
            sample = encode_entry(entry, codec, args, root)
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            failed += 1
            print(f"[{index + 1}/{len(entries)}] FAILED {name}: {error}", flush=True)
            continue
        temporary = out_path.with_suffix(".pt.tmp")
        torch.save(sample, temporary)
        os.replace(temporary, out_path)
        paths.append(out_path)
        written += 1
        if written % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
        print(f"[{index + 1}/{len(entries)}] {name}: target {tuple(sample['target_latent'].shape)} "
              f"proxy {tuple(sample['proxy_latent'].shape)} anchor {tuple(sample['anchor_latent'].shape)} "
              f"oc {tuple(sample['oc_latent'].shape) if 'oc_latent' in sample else None} "
              f"text {sample['info']['prompt_tokens']} tok, {time.monotonic() - started:.1f}s", flush=True)
    listed = Path(args.list_prefix) if args.list_prefix else output
    listing.write_text("".join(f"{listed / path.name}\n" for path in paths), encoding="utf-8")
    print(f"Done: {written} written, {skipped} skipped, {failed} failed -> {output} (list {listing})")


if __name__ == "__main__":
    main()
