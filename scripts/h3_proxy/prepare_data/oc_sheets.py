# SPDX-License-Identifier: Apache-2.0
"""OC identity sheets (``<Picture 2>``) of gta_record high-tier renders.

Every high-tier MiniMax-H3 render of a gta_record dataset was generated from three conditions:
the semantic proxy as ``<Video 1>``, a style first frame as ``<Picture 1>`` and an OC (original
character) identity sheet -- front / left profile / back of the followed protagonist or vehicle --
as ``<Picture 2>``. The render keeps them under ``<render>/reference/``: ``image_2.png`` is the
sheet and ``prompt.txt`` the generation prompt, whose ``<Picture 2>`` line says what it shows.
Because the target video's subject was drawn from exactly this sheet, it is a supervised identity
reference for that target. Native ``color`` targets have none.

Add the sheets to an existing encode manifest::

    python scripts/h3_proxy/prepare_data/oc_sheets.py \\
      --root /data/binghe/datasets/gta_record_0930-v0003 \\
      --manifest /data/binghe/h3_proxy/manifests/gta_record_0930_omni_train.jsonl \\
      --out /data/binghe/h3_proxy/manifests/gta_record_0930_omni_train_oc.jsonl

``--swap`` instead gives every sheeted row the sheet of another row with the same subject kind and
a different style, for an evaluation that checks the identity follows the sheet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

OC_SHEET = "reference/image_2.png"
OC_SUBJECTS = ("protagonist", "vehicle")
# The ``<Picture 2>`` line Qwen reads ahead of the caption, phrased like the generation prompts.
OC_PROMPTS = {
    "protagonist":
    "<Picture 2> is the protagonist identity sheet (front / left profile / back). The character the "
    "camera follows is the <Picture 2> character in every frame, also while turning or entering and "
    "leaving the frame. Identity only: camera, pose, motion and framing come from <Video 1>.",
    "vehicle":
    "<Picture 2> is the identity sheet of the enclosed vehicle the player drives; the driver stays hidden "
    "inside. The followed vehicle is the <Picture 2> vehicle in every frame, also while turning or entering "
    "and leaving the frame. Identity only: camera, pose, motion and framing come from <Video 1>.",
}


def sheet_subject(prompt: str) -> str | None:
    """What the render's ``<Picture 2>`` shows, from the first line of its prompt that names it."""
    line = next((line.lower() for line in prompt.splitlines() if line.startswith("<Picture 2>")), "")
    named = re.search(r"supplies the (\S+) identity", line)
    kind = named.group(1) if named else line
    if "vehicle" in kind:
        return "vehicle"
    if any(word in kind for word in ("hero", "protagonist", "character")):
        return "protagonist"
    return None


def render_sheet(root: Path, render_dir: str) -> dict[str, str] | None:
    """``{"oc", "oc_subject"}`` for a render directory relative to ``root``, or ``None``."""
    reference = root / render_dir / "reference"
    if not (reference / "image_2.png").is_file() or not (reference / "prompt.txt").is_file():
        return None
    subject = sheet_subject((reference / "prompt.txt").read_text(encoding="utf-8"))
    if subject is None:
        return None
    return {"oc": f"{render_dir}/{OC_SHEET}", "oc_subject": subject}


def add_sheets(root: Path, rows: list[dict]) -> list[dict]:
    """Each high-tier row with its render's sheet; other rows unchanged."""
    out = []
    for row in rows:
        row = {key: value for key, value in row.items() if key not in ("oc", "oc_subject")}
        if row.get("target_kind") == "high":
            row.update(render_sheet(root, str(Path(row["target"]).parent)) or {})
        out.append(row)
    return out


def swap_sheets(rows: list[dict], seed: str = "oc-swap") -> list[dict]:
    """Give each sheeted row a sheet of the same subject kind from a row of another style."""
    sheeted = [row for row in rows if row.get("oc")]
    out = []
    for row in rows:
        if not row.get("oc"):
            out.append(row)
            continue
        donors = sorted(
            (other for other in sheeted
             if other["oc_subject"] == row["oc_subject"] and other.get("target_style") != row.get("target_style")),
            key=lambda other: hashlib.sha1(f"{seed}/{row['name']}/{other['name']}".encode()).hexdigest())
        if not donors:
            raise SystemExit(f"{row['name']}: no {row['oc_subject']} sheet of another style to swap in")
        out.append({**row, "oc": donors[0]["oc"], "oc_donor": donors[0]["name"]})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="Dataset root holding seg_*/.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--swap", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).expanduser()
    with open(args.manifest, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows = add_sheets(root, rows)
    if args.swap:
        rows = swap_sheets(rows)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    high = sum(row.get("target_kind") == "high" for row in rows)
    subjects = {kind: sum(row.get("oc_subject") == kind for row in rows) for kind in OC_SUBJECTS}
    print(f"Wrote {out}: {len(rows)} rows, {sum(subjects.values())}/{high} high rows with a sheet {subjects}")


if __name__ == "__main__":
    main()
