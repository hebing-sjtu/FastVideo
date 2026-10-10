#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Scan a ``gta_record_*`` dataset into a mixed-single-reference omni encode manifest.

Every row is one target with one ``<Video 1>`` reference, the target's first frame as
``<Picture 1>``, and the target's own caption:

========================  ==================================================================
target                    ``color``: the seg's native ``proxy/color.mp4``;
                          ``high``: each high-tier MiniMax-H3 render's ``output.mp4``
``<Video 1>`` modality    ``duv`` (``proxy/duv.mp4``), ``depth`` / ``semantic`` (from
                          ``proxy/depth.mp4`` + ``proxy/semantic.mp4``), or ``style`` -- a
                          low-tier (low-poly) render of the same seg
text                      ``captions/prompt.json`` beside the target, compiled prose
========================  ==================================================================

Renders are 1344x768 restyles of the 1280x720 source with about six invented rows above and below,
so render targets and style references carry a crop back to the source's field of view (see
``gta_record.render_crop_box``). Every reference and target then shares one frame after the
encoder's center crop.

One modality per target, assigned round-robin within each (target kind, event tag) group in a
seeded order, so the four modalities and both target kinds stay balanced without storing a target
latent more than once. Style rows rotate through the seg's low-poly styles. The split is the one
``metadata.json`` declares (by parent recording). ``--heldout-per-cell N`` additionally writes N rows
per (target kind, modality) cell drawn from distinct segs, for a held-out evaluation set.

Usage::

    python scripts/h3_proxy/prepare_data/record_dir_to_omni_manifest.py \\
      --root /data/binghe/datasets/gta_record_0930-v0003 --split train \\
      --out /data/binghe/h3_proxy/manifests/gta_record_0930_omni_train.jsonl

    python scripts/h3_proxy/prepare_data/record_dir_to_omni_manifest.py \\
      --root /data/binghe/datasets/gta_record_0930-v0003 --split val --heldout-per-cell 4 \\
      --out /data/binghe/h3_proxy/manifests/gta_record_0930_omni_val.jsonl \\
      --heldout-out /data/binghe/h3_proxy/manifests/gta_record_0930_omni_heldout32.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gta_record import render_crop_box  # noqa: E402
from seg_dir_to_encode_manifest import CONTRACT_PROSE_STYLES, extract_prompt_text  # noqa: E402

VARIANTS = ("duv", "depth", "semantic", "style")
TARGET_KINDS = ("color", "high")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Dataset root holding dataset.json and seg_*/.")
    p.add_argument("--split", choices=("train", "val"), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--targets", nargs="+", choices=TARGET_KINDS, default=list(TARGET_KINDS))
    p.add_argument("--contract-prose", choices=CONTRACT_PROSE_STYLES, default="rich")
    p.add_argument("--seed", default="gta_record_omni_v1", help="Salt of the deterministic row order.")
    p.add_argument("--heldout-per-cell", type=int, default=0)
    p.add_argument("--heldout-out", default="")
    p.add_argument("--allow-unscored",
                   action="store_true",
                   help="Keep renders whose provenance.json has no VLM 'pass' decision (13 on v0003).")
    args = p.parse_args()
    if args.heldout_per_cell and not args.heldout_out:
        p.error("--heldout-per-cell needs --heldout-out")
    return args


def stable_key(seed: str, *parts: str) -> str:
    return hashlib.sha1("/".join((seed, *parts)).encode()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def render_passed(render_dir: Path) -> bool:
    path = render_dir / "provenance.json"
    if not path.is_file():
        return False
    return (read_json(path).get("evaluation") or {}).get("decision") == "pass"


def caption(path: Path, prose: str) -> str:
    return extract_prompt_text(read_json(path), prose_style=prose) if path.is_file() else ""


def collect_targets(root: Path, split: str, *, kinds: tuple[str, ...], prose: str,
                    allow_unscored: bool) -> tuple[list[dict], Counter]:
    """Target candidates of one split, each with the seg's usable low-poly renders."""
    tiers = {name: entry["tier"] for name, entry in read_json(root / "dataset.json")["styles"].items()}
    skipped: Counter = Counter()
    targets: list[dict] = []
    for seg in sorted(path for path in root.glob("seg_*") if path.is_dir()):
        meta = read_json(seg / "metadata.json")
        if meta.get("split") != split:
            continue
        nodes = {node["id"]: node for node in meta["nodes"]}
        proxy = nodes.get("proxy")
        if proxy is None:
            skipped["no proxy node"] += 1
            continue
        source_size = (int(proxy["video"]["width"]), int(proxy["video"]["height"]))
        renders = []
        for node in meta["nodes"]:
            if node["id"] == "proxy":
                continue
            render_dir = seg / node["id"]
            if not allow_unscored and not render_passed(render_dir):
                skipped["render without VLM pass"] += 1
                continue
            size = (int(node["video"]["width"]), int(node["video"]["height"]))
            renders.append({
                "uid": node["id"],
                "style": node["style"],
                "tier": tiers[node["style"]],
                "video": f"{seg.name}/{node['id']}/{node['files']['video']}",
                "caption": render_dir / node.get("caption", "captions/prompt.json"),
                "crop": list(render_crop_box(size, source_size)) if size != source_size else None,
            })
        lows = sorted((render for render in renders if render["tier"] == "low"), key=lambda r: r["style"])
        base = {
            "seg": seg.name,
            "parent": str(meta.get("parent")),
            "tag": str(meta.get("tag")),
            "lows": lows,
            "proxy_dir": f"{seg.name}/proxy",
        }
        if "color" in kinds:
            text = caption(seg / "proxy" / proxy.get("caption", "captions/prompt.json"), prose)
            if text:
                targets.append({**base, "kind": "color", "name": f"{seg.name}__color", "target":
                                f"{seg.name}/proxy/{proxy['files']['video']}", "crop": None, "style": "proxy",
                                "prompt": text})
            else:
                skipped["color target without caption"] += 1
        if "high" in kinds:
            for render in renders:
                if render["tier"] != "high":
                    continue
                text = caption(render["caption"], prose)
                if not text:
                    skipped["high target without caption"] += 1
                    continue
                targets.append({**base, "kind": "high", "name": f"{seg.name}__{render['uid']}", "target":
                                render["video"], "crop": render["crop"], "style": render["style"], "prompt": text})
    return targets, skipped


def assign(targets: list[dict], variants: tuple[str, ...], seed: str) -> list[dict]:
    """One modality per target, balanced per (kind, tag); style rotates the seg's low-poly renders."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for target in targets:
        groups[(target["kind"], target["tag"])].append(target)
    style_turn: Counter = Counter()
    rows = []
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda t: stable_key(seed, t["name"]))
        offset = int(stable_key(seed, *key), 16) % len(variants)
        for index, target in enumerate(ordered):
            choices = [variants[(offset + index + step) % len(variants)] for step in range(len(variants))]
            modality = next((kind for kind in choices if kind != "style" or target["lows"]), None)
            if modality is None:
                continue
            rows.append(build_row(target, modality, style_turn, seed))
    return sorted(rows, key=lambda row: row["name"])


def build_row(target: dict, modality: str, style_turn: Counter, seed: str) -> dict:
    row = {
        "name": target["name"],
        "id": target["seg"],
        "target": target["target"],
        "prompt": target["prompt"],
        "proxy_modality": modality,
        "cwm_system": "w0_omni",
        "target_kind": target["kind"],
        "target_style": target["style"],
        "tag": target["tag"],
        "parent": target["parent"],
    }
    if target["crop"]:
        row["target_crop"] = target["crop"]
    proxy_dir = target["proxy_dir"]
    if modality == "duv":
        row["proxy_duv_video"] = f"{proxy_dir}/duv.mp4"
    elif modality in ("depth", "semantic"):
        row["proxy_depth_video"] = f"{proxy_dir}/depth.mp4"
        row["proxy_semantic_video"] = f"{proxy_dir}/semantic.mp4"
    else:
        lows = target["lows"]
        start = int(stable_key(seed, target["seg"], "style"), 16) % len(lows)
        low = lows[(start + style_turn[target["seg"]]) % len(lows)]
        style_turn[target["seg"]] += 1
        row["proxy"] = low["video"]
        row["proxy_style"] = low["style"]
        if low["crop"]:
            row["proxy_crop"] = low["crop"]
    return row


def pick_heldout(rows: list[dict], per_cell: int, kinds: tuple[str, ...], variants: tuple[str, ...],
                 seed: str) -> list[dict]:
    """``per_cell`` rows per (kind, modality), each from a seg no other picked row uses."""
    used: set[str] = set()
    picked = []
    for kind in kinds:
        for modality in variants:
            pool = sorted((row for row in rows if row["target_kind"] == kind and row["proxy_modality"] == modality),
                          key=lambda row: stable_key(seed, "heldout", row["name"]))
            taken = 0
            for row in pool:
                if taken == per_cell:
                    break
                if row["id"] in used:
                    continue
                used.add(row["id"])
                picked.append(row)
                taken += 1
            if taken < per_cell:
                raise SystemExit(f"held-out cell ({kind}, {modality}) has only {taken} rows from unused segs")
    return picked


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(label: str, rows: list[dict]) -> None:
    cells = Counter((row["target_kind"], row["proxy_modality"]) for row in rows)
    print(f"{label}: {len(rows)} rows from {len({row['id'] for row in rows})} segs")
    for kind in TARGET_KINDS:
        line = ", ".join(f"{modality}={cells[(kind, modality)]}" for modality in VARIANTS if cells[(kind, modality)])
        if line:
            print(f"  {kind:5s} {line}")
    styles = Counter(row["proxy_style"] for row in rows if row.get("proxy_style"))
    if styles:
        print(f"  low-poly style references: {dict(sorted(styles.items()))}")


def check_media(root: Path, rows: list[dict]) -> None:
    keys = ("target", "proxy", "proxy_duv_video", "proxy_depth_video", "proxy_semantic_video")
    missing = [f"{row['name']}:{row[key]}" for row in rows for key in keys if row.get(key) and not (root / row[key]).is_file()]
    if missing:
        duv = sum("duv.mp4" in item for item in missing)
        hint = " Compose proxy/duv.mp4 with compose_gta_duv.py first." if duv else ""
        raise SystemExit(f"{len(missing)} referenced files are missing, e.g. {missing[:3]}.{hint}")


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    kinds, variants = tuple(args.targets), tuple(args.variants)
    targets, skipped = collect_targets(root, args.split, kinds=kinds, prose=args.contract_prose,
                                       allow_unscored=args.allow_unscored)
    if not targets:
        raise SystemExit(f"No {args.split} targets under {root}")
    rows = assign(targets, variants, args.seed)
    check_media(root, rows)
    write_rows(Path(args.out).expanduser(), rows)
    summarize(f"Wrote {args.out}", rows)
    for reason, count in sorted(skipped.items()):
        print(f"  skipped {count}: {reason}")
    if args.heldout_per_cell:
        heldout = pick_heldout(rows, args.heldout_per_cell, kinds, variants, args.seed)
        write_rows(Path(args.heldout_out).expanduser(), heldout)
        summarize(f"Wrote {args.heldout_out}", heldout)


if __name__ == "__main__":
    main()
