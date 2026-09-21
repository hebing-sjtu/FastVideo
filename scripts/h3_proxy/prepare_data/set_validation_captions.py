# SPDX-License-Identifier: Apache-2.0
"""Swap the captions of a validation JSON, keeping every media path as it was.

A validation record pairs a clip's proxy, anchor and reference video with the text the sampler
will encode. Only the text is in question when comparing caption schemas, so this rewrites that
one field and leaves the rest byte-for-byte: the two files then differ by exactly the variable
under test, which is what makes the two panels comparable.

Captions arrive as ``{clip_id: caption}``. Every record must be covered -- a partial mapping would
sample some clips under the new schema and some under the old one, and the panel gives no hint
which is which.

Usage::

    python scripts/h3_proxy/prepare_data/set_validation_captions.py \\
        --val-json /data/binghe/h3_proxy/gta_v2_cwm_validation_val6.json \\
        --captions /data/binghe/h3_proxy/native_captions.json \\
        --out /data/binghe/h3_proxy/gta_v2_native_validation_val6.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# What `clip_dir_to_validation_json.py` writes. ValidationDataset reads `caption`; the rest is
# media the sampler resolves by path.
CAPTION_KEY = "caption"
ID_KEY = "id"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val-json", required=True, help="Validation JSON to read.")
    parser.add_argument("--captions", required=True, help='JSON mapping {clip_id: caption}.')
    parser.add_argument("--out", required=True, help="Where to write the rewritten validation JSON.")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Keep a record's existing caption when the mapping has no entry for it. Off by default: a "
        "half-rewritten file samples two schemas at once and the panel cannot say which produced what.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    payload = load_json(Path(args.val_json).expanduser())
    records = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise SystemExit(f"{args.val_json} holds no validation records under 'data'.")

    captions = load_json(Path(args.captions).expanduser())
    if not isinstance(captions, dict):
        raise SystemExit(f"{args.captions} must be a JSON object mapping clip id to caption.")

    rewritten, missing = 0, []
    for record in records:
        clip_id = str(record.get(ID_KEY, ""))
        caption = captions.get(clip_id)
        if caption is None:
            missing.append(clip_id)
            continue
        if not isinstance(caption, str) or not caption.strip():
            raise SystemExit(f"caption for {clip_id!r} is empty; an empty prompt encodes to nothing.")
        record[CAPTION_KEY] = caption
        rewritten += 1

    if missing and not args.allow_missing:
        raise SystemExit(f"{len(missing)} record(s) have no caption in the mapping: {', '.join(missing[:10])}"
                         f"{' ...' if len(missing) > 10 else ''}. Supply them, or pass --allow-missing.")

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"data": records}, handle, ensure_ascii=False, indent=2)

    unused = sorted(set(captions) - {str(record.get(ID_KEY, "")) for record in records})
    print(f"Wrote {len(records)} records ({rewritten} recaptioned) -> {out}")
    if missing:
        print(f"  kept the original caption on {len(missing)}: {', '.join(missing[:10])}")
    if unused:
        # Usually a clip-id convention mismatch between the caption source and the validation set,
        # which otherwise shows up as a silently unchanged file.
        print(f"  {len(unused)} caption(s) matched no record: {', '.join(unused[:10])}")


if __name__ == "__main__":
    main()
