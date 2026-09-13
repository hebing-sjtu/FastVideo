#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decide which seg ids a dataset's ``vlm_filter.json`` considers trainable.

A judge model scored every clip of ``gta_web_0902_v2`` against its source on several
observable-correspondence rubrics. A clip whose camera trajectory or layout does not follow its
proxy teaches the model that the proxy can be ignored, which is the failure this stage is being
trained against -- so the scores have to gate the corpus rather than sit next to it.

Joining by directory name
-------------------------
The obvious key would be ``uid``, which both sides carry. It does not work: on
``gta_web_0902_v2`` the 996 filter uids and the 996 ``metadata.json`` uids overlap in **zero**
entries, because each export hashes its own absolute paths. The directory name overlaps 996/996,
``tag`` agrees on all 996 under that join, and both sides trace the same clip back to
``gta_web_0831/seg_NNNN``. So the name is the only usable key, and it is corroborated rather than
assumed. ``report`` prints the agreement it can still check so a future dataset that breaks this
does not filter silently against the wrong rows.

Thresholding
------------
Scores are quantised -- ``score_scale`` on these records reads ``discrete_0_1_step_0.2`` -- so a
threshold between two rungs behaves like the rung above it. ``report`` states the effective cut it
observed in the data instead of trusting the requested number, because ``>= 0.7`` silently meaning
``>= 0.8`` is the kind of thing that is only noticed after a training run.

A clip with no record, or with any of the required metrics missing or non-numeric, is rejected.
"Not scored" is not "passed": the caller asked for clips the judge approved.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

FILTER_NAME = "vlm_filter.json"
DEFAULT_METRICS = ("preservation", "camera_trajectory", "layout", "quality")
DEFAULT_THRESHOLD = 0.7


def add_filter_arguments(parser) -> None:
    """Register the shared ``--vlm-filter`` trio on a manifest builder."""
    parser.add_argument(
        "--vlm-filter",
        default="auto",
        help=f"Path to a {FILTER_NAME}, 'auto' to use <root>/{FILTER_NAME} when it exists, or 'none' "
        "to keep every clip. Filtering is on by default because an unfiltered corpus is the "
        "thing being corrected; the report says how many clips it removed.",
    )
    parser.add_argument(
        "--min-vlm-score",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Every metric must be >= this (default {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--vlm-metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metric names that must all clear the threshold "
        f"(default {','.join(DEFAULT_METRICS)}).",
    )


@dataclass(frozen=True, slots=True)
class VlmFilter:
    """Per-clip verdicts from one ``vlm_filter.json``."""

    path: Path
    table: dict[str, dict]
    metrics: tuple[str, ...]
    threshold: float

    def verdict(self, name: str) -> tuple[bool, str]:
        """Whether ``name`` is trainable, and why not when it is not."""
        record = self.table.get(name)
        if record is None:
            return False, f"no {self.path.name} record"
        below: list[str] = []
        for metric in self.metrics:
            value = record.get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False, f"{metric} is {value!r} in {self.path.name}"
            if float(value) < self.threshold:
                below.append(f"{metric}={float(value):g}")
        if below:
            return False, f"below {self.threshold:g}: {', '.join(below)}"
        return True, ""

    def passing(self) -> set[str]:
        return {name for name in self.table if self.verdict(name)[0]}

    def report(self, *, on_disk: set[str] | None = None) -> None:
        """Print the score distribution, the effective cut, and the join's health."""
        print(f"VLM filter: {self.path}")
        scales = {str(record.get("score_scale")) for record in self.table.values() if record.get("score_scale")}
        rubrics = {str(record.get("rubric_version")) for record in self.table.values() if record.get("rubric_version")}
        print(f"  {len(self.table)} records, requiring {', '.join(self.metrics)} >= {self.threshold:g}"
              f"{'  [' + ', '.join(sorted(scales | rubrics)) + ']' if scales or rubrics else ''}")

        observed: set[float] = set()
        for metric in self.metrics:
            counts = Counter(record.get(metric) for record in self.table.values())
            observed.update(value for value in counts if isinstance(value, (int, float)) and not isinstance(value, bool))
            rendered = ", ".join(f"{key if key is not None else 'missing'}:{count}"
                                for key, count in sorted(counts.items(), key=lambda kv: (kv[0] is None, kv[0])))
            print(f"    {metric}: {rendered}")

        kept = sorted(value for value in observed if value >= self.threshold)
        dropped = sorted(value for value in observed if value < self.threshold)
        if kept and dropped and dropped[-1] < self.threshold <= kept[0]:
            print(f"  effective cut: keeps scores >= {kept[0]:g}, drops <= {dropped[-1]:g}. The scale is "
                  f"quantised, so --min-vlm-score {self.threshold:g} behaves as {kept[0]:g}.")

        passing = self.passing()
        print(f"  {len(passing)}/{len(self.table)} records pass")

        if on_disk is None:
            return
        # The join is by name, so say out loud how well the two sides line up. A dataset whose
        # filter was written for a different export would show up here as a small intersection
        # instead of as a quietly mis-scored corpus.
        shared = on_disk & set(self.table)
        print(f"  name join: {len(shared)}/{len(on_disk)} seg directories have a record")
        if len(shared) != len(on_disk):
            orphans = sorted(on_disk - set(self.table))
            print(f"    WARNING: {len(orphans)} directories are unscored and will be dropped: "
                  f"{', '.join(orphans[:8])}{' ...' if len(orphans) > 8 else ''}")


def resolve_filter_path(root: Path, requested: str) -> Path | None:
    """The filter file to use, or None when filtering is off."""
    value = (requested or "").strip()
    if value.lower() in {"", "none", "off", "false"}:
        return None
    if value.lower() == "auto":
        candidate = root / FILTER_NAME
        return candidate if candidate.is_file() else None
    path = Path(value).expanduser()
    if not path.is_file():
        raise SystemExit(f"--vlm-filter {path} does not exist. Pass 'none' to keep every clip.")
    return path


def load_vlm_filter(root: Path, args) -> VlmFilter | None:
    """Build a filter from ``--vlm-filter`` / ``--min-vlm-score`` / ``--vlm-metrics``."""
    path = resolve_filter_path(root, getattr(args, "vlm_filter", "auto"))
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read {path}: {error}") from error
    table = _as_table(payload, path)
    metrics = tuple(name.strip() for name in str(getattr(args, "vlm_metrics", "")).split(",") if name.strip())
    if not metrics:
        raise SystemExit("--vlm-metrics named no metrics.")
    unknown = [name for name in metrics if not any(name in record for record in table.values())]
    if unknown:
        available = sorted({key for record in table.values() for key, value in record.items()
                            if isinstance(value, (int, float)) and not isinstance(value, bool)})
        raise SystemExit(f"{path.name} has no metric named {', '.join(unknown)}. Numeric fields present: "
                         f"{', '.join(available)}")
    return VlmFilter(path=path, table=table, metrics=metrics, threshold=float(getattr(args, "min_vlm_score",
                                                                                     DEFAULT_THRESHOLD)))


def _as_table(payload: object, path: Path) -> dict[str, dict]:
    """Accept either a name-keyed object or a list of records carrying their own id."""
    if isinstance(payload, dict):
        records = payload.get("clips") if isinstance(payload.get("clips"), (dict, list)) else payload
    else:
        records = payload
    if isinstance(records, dict):
        table = {str(key): value for key, value in records.items() if isinstance(value, dict)}
        if table:
            return table
    if isinstance(records, list):
        table = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            for key in ("clip", "id", "name", "seg"):
                value = record.get(key)
                if isinstance(value, str) and value.strip():
                    table[Path(value.strip()).name] = record
                    break
        if table:
            return table
    raise SystemExit(f"{path} is not a name-keyed object or a list of records with a 'clip'/'id' field.")


__all__ = [
    "DEFAULT_METRICS",
    "DEFAULT_THRESHOLD",
    "FILTER_NAME",
    "VlmFilter",
    "add_filter_arguments",
    "load_vlm_filter",
    "resolve_filter_path",
]
