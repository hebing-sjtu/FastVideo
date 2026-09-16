#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Did training change the prediction, and did it change it *toward* the target?

``compare_eval_outputs.py`` answers a weaker question than it looks like it does. Differing bytes
prove the adapter reached the sampling model and nothing more: a weight delta of 1e-4 changes every
byte of an mp4 while looking identical, and so does a delta that made the output worse. "The videos
differ but I cannot see an improvement" is exactly the outcome that hashing cannot interpret.

So this reads the pixels. The comparison panel is ``proxy | prediction | target`` by construction,
which means each eval run already carries its own ground truth, and two runs of the same validation
set are the same scene under the same motion. That makes two separate measurements available:

  * **drift** -- how far the later checkpoint's prediction moved from the baseline's. Answers
    "is the adapter doing anything", in pixels rather than in bytes.
  * **error** -- how far each prediction sits from its own target. Answers "did it move in the
    right direction", which drift alone cannot distinguish from moving in any direction.

Both are reported per frame, and the frame axis matters more than the average under CWM's wn
regime. wn hands the model a real-footage prefix, so the opening frames are the same clean target
pixels in *both* runs and cannot differ no matter how well training went. An average over the whole
clip dilutes the region under judgement with a region that is identical by construction -- and the
prefix is roughly a quarter of the clip, which is enough to make a real improvement look small.

The prefix boundary is detected from the drift curve rather than assumed, which doubles as an
end-to-end check on the given-frames mechanism: if the flat-then-rising step lands where the prefix
should end, the prefix is genuinely being handed over.

Pixel error against a target is a coarse instrument for a generative model -- a plausible render
that differs from the take scores badly. It is used here only because the paired target *is* the
same scene under the same motion, so a model that tracks the proxy better cannot score worse. Read
a large change as meaningful and a small one as nothing.

Usage::

    python scripts/h3_proxy/compare_eval_predictions.py \\
        /data/binghe/h3_proxy/runs/eval_gta_v2_cwm_wn_step0_wn_base/checkpoints \\
        /data/binghe/h3_proxy/runs/eval_gta_v2_cwm_wn_step525_wn_base/checkpoints
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from compare_eval_outputs import compare_manifests, index_videos, load_manifest

# The panel is composed at the prediction's height with this gutter between columns, and the width
# of each column follows from the cache geometry. Both are `MiniMaxH3ProxyValidationCallback`
# defaults; a run that overrode them needs them passed here too.
DEFAULT_SEPARATOR_PX = 4

GEOMETRY_RE = {
    "target_height": re.compile(r"--training\.data\.num_height\s+(\d+)"),
    "target_width": re.compile(r"--training\.data\.num_width\s+(\d+)"),
    "proxy_height": re.compile(r"--callbacks\.validation\.proxy_height\s+(\d+)"),
    "proxy_width": re.compile(r"--callbacks\.validation\.proxy_width\s+(\d+)"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("left", help="The baseline eval output directory, normally the step-0 one.")
    parser.add_argument("right", help="The later checkpoint's eval output directory.")
    parser.add_argument("--separator-px",
                        type=int,
                        default=DEFAULT_SEPARATOR_PX,
                        help="Must match the run's callbacks.validation.panel_separator_px.")
    parser.add_argument("--prefix-frames",
                        type=int,
                        default=None,
                        help="Force the prefix boundary instead of detecting it from the drift curve.")
    parser.add_argument("--force",
                        action="store_true",
                        help="Compare even when the manifests disagree. The numbers are then about the videos only "
                        "and say nothing about the checkpoints.")
    return parser.parse_args()


def parse_geometry(manifest: dict[str, Any] | None) -> dict[str, int]:
    """The panel's column widths are a consequence of the cache geometry, which the manifest records.

    Guessing the split from the file width instead would need the columns to be equally wide, and
    they are only equally wide when the proxy's aspect ratio happens to match the target's. A
    704x1280 target with a 192x336 proxy does not, so the arithmetic has to be done properly.
    """
    geometry = (manifest or {}).get("geometry") or ""
    found: dict[str, int] = {}
    for name, pattern in GEOMETRY_RE.items():
        match = pattern.search(geometry)
        if match:
            found[name] = int(match.group(1))
    missing = [name for name in GEOMETRY_RE if name not in found]
    if missing:
        raise SystemExit(f"The manifest's geometry string is missing {', '.join(missing)}, so the panel columns "
                         f"cannot be located. Got: {geometry!r}")
    return found


@dataclass(frozen=True)
class PanelLayout:
    """Where the prediction and target columns sit inside a composed panel frame."""

    band_height: int
    columns: dict[str, tuple[int, int]]

    def crop(self, frame: np.ndarray, name: str) -> np.ndarray:
        start, end = self.columns[name]
        return frame[self.band_height:, start:end]


def solve_layout(frame_shape: tuple[int, ...], geometry: dict[str, int], separator_px: int) -> PanelLayout:
    """Locate the columns by reconstructing the composition, then checking it against the file.

    Searching the frame for dark gutter columns would be the obvious alternative and is worse: a
    night-time clip has plenty of columns that are uniformly near the separator's colour, and H.264
    has already moved every exact value. Reconstructing instead gives an answer that is either
    provably right -- the reconstructed width matches the file's to the pixel -- or refused.
    """
    height, width = int(frame_shape[0]), int(frame_shape[1])
    canvas = geometry["target_height"]
    proxy_width = max(1, round(geometry["proxy_width"] * canvas / geometry["proxy_height"]))
    target_width = geometry["target_width"]

    # `compose_comparison_video` pads odd dimensions up by a pixel for H.264 4:2:0, and the label
    # band is present unless the run turned it off, so both are candidates rather than givens.
    for names, widths in (
        (("proxy", "prediction", "target"), (proxy_width, target_width, target_width)),
        (("prediction", "target"), (target_width, target_width)),
        (("proxy", "prediction"), (proxy_width, target_width)),
    ):
        composed_width = sum(widths) + separator_px * (len(widths) - 1)
        for band_height in (max(18, canvas // 22), 0):
            composed_height = canvas + band_height
            if (width - composed_width) in (0, 1) and (height - composed_height) in (0, 1):
                columns: dict[str, tuple[int, int]] = {}
                offset = 0
                for name, column_width in zip(names, widths, strict=True):
                    columns[name] = (offset, offset + column_width)
                    offset += column_width + separator_px
                return PanelLayout(band_height=band_height, columns=columns)

    raise SystemExit(f"A {height}x{width} panel matches no layout of a {canvas}-tall canvas with a "
                     f"{geometry['target_width']}px target column and a {proxy_width}px proxy column at "
                     f"{separator_px}px separators. Pass --separator-px if the run overrode it; otherwise the "
                     "manifest's geometry does not describe these videos.")


def read_frames(path: Path) -> list[np.ndarray]:
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(path))
    try:
        return [np.asarray(frame, dtype=np.uint8) for frame in reader]
    finally:
        reader.close()


def mean_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    """Mean absolute difference on the 0-255 scale, in float64 to keep uint8 from wrapping."""
    return float(np.abs(left.astype(np.float64) - right.astype(np.float64)).mean())


def detect_prefix(drift: np.ndarray) -> int | None:
    """The leading run of frames that barely moved, or None if the curve has no such step.

    Under wn this recovers the given-frame prefix: the same clean target pixels went into both runs,
    so their only difference there is the encoder's own noise. Returning None is a real answer --
    it means the curve is flat, and a flat curve at *any* height has no prefix to separate out.
    """
    if drift.size < 8:
        return None
    ordered = np.sort(drift)
    floor = float(np.median(ordered[:max(1, drift.size // 5)]))
    ceiling = float(np.median(ordered[-max(1, drift.size // 5):]))
    # A curve whose top fifth is not clearly above its bottom fifth is flat, and the 1.0 keeps a
    # clip whose drift is uniformly near zero from reading as a huge relative step.
    if ceiling < 2.0 * floor + 1.0:
        return None
    threshold = floor + 0.25 * (ceiling - floor)
    above = np.flatnonzero(drift > threshold)
    if above.size == 0 or above[0] == 0:
        return None
    return int(above[0])


def curve(values: np.ndarray, width: int = 48) -> str:
    """A fixed-width bar per frame bucket, so the shape of the curve survives a terminal."""
    if values.size == 0:
        return ""
    buckets = np.array_split(values, min(width, values.size))
    heights = np.array([float(bucket.mean()) for bucket in buckets])
    top = float(heights.max()) or 1.0
    blocks = " ▁▂▃▄▅▆▇█"
    return "".join(blocks[min(len(blocks) - 1, int(round(height / top * (len(blocks) - 1))))] for height in heights)


def main() -> None:
    args = parse_args()
    left_dir, right_dir = Path(args.left).expanduser(), Path(args.right).expanduser()
    for directory in (left_dir, right_dir):
        if not directory.is_dir():
            raise SystemExit(f"Not a directory: {directory}")

    left_manifest, right_manifest = load_manifest(left_dir), load_manifest(right_dir)
    comparable = compare_manifests(left_manifest, right_manifest)
    if not comparable and not args.force:
        raise SystemExit(1)
    geometry = parse_geometry(left_manifest or right_manifest)

    # Only the panels: the standalone prediction mp4 carries no target to score against.
    left_videos = {key[0]: path for key, path in index_videos(left_dir).items() if key[1] == "_compare"}
    right_videos = {key[0]: path for key, path in index_videos(right_dir).items() if key[1] == "_compare"}
    shared = sorted(set(left_videos) & set(right_videos), key=int)
    if not shared:
        raise SystemExit(f"No comparison panels in common ({len(left_videos)} vs {len(right_videos)} found). This "
                         "reads the `_compare` panels, which exist only if the run logged them.")

    drifts: list[np.ndarray] = []
    left_errors: list[np.ndarray] = []
    right_errors: list[np.ndarray] = []
    print(f"\n{len(shared)} panels in common:")
    for index in shared:
        left_frames, right_frames = read_frames(left_videos[index]), read_frames(right_videos[index])
        if not left_frames or not right_frames:
            print(f"  video_{index}: unreadable, skipped")
            continue
        layout = solve_layout(left_frames[0].shape, geometry, args.separator_px)
        if "target" not in layout.columns:
            raise SystemExit("These panels have no target column, so there is nothing to measure error against. "
                             "The run logged proxy and prediction only.")
        # Truncated rather than padded: a held frame would score as the model diverging at the tail.
        count = min(len(left_frames), len(right_frames))
        drift = np.empty(count)
        left_error = np.empty(count)
        right_error = np.empty(count)
        for frame in range(count):
            left_prediction = layout.crop(left_frames[frame], "prediction")
            right_prediction = layout.crop(right_frames[frame], "prediction")
            target = layout.crop(left_frames[frame], "target")
            drift[frame] = mean_abs_diff(left_prediction, right_prediction)
            left_error[frame] = mean_abs_diff(left_prediction, target)
            right_error[frame] = mean_abs_diff(right_prediction, target)
        drifts.append(drift)
        left_errors.append(left_error)
        right_errors.append(right_error)
        print(f"  video_{index}: {count} frames, drift {drift.mean():6.2f}, "
              f"error {left_error.mean():6.2f} -> {right_error.mean():6.2f}")

    if not drifts:
        raise SystemExit("Nothing was readable.")

    count = min(drift.size for drift in drifts)
    drift = np.mean([d[:count] for d in drifts], axis=0)
    left_error = np.mean([e[:count] for e in left_errors], axis=0)
    right_error = np.mean([e[:count] for e in right_errors], axis=0)

    boundary = args.prefix_frames if args.prefix_frames is not None else detect_prefix(drift)
    print(f"\nper-frame curves over {count} frames, averaged across panels (0-255 scale):")
    print(f"  drift  |right-left|   {curve(drift)}   max {drift.max():.2f}")
    print(f"  error  left->target   {curve(left_error)}   mean {left_error.mean():.2f}")
    print(f"  error  right->target  {curve(right_error)}   mean {right_error.mean():.2f}")

    if boundary is None:
        print("\nNo flat-then-rising step in the drift curve, so there is no prefix to separate out. Under wn that\n"
              "is itself a finding: the given frames should pin the opening of both runs to the same real footage,\n"
              "which would show as near-zero drift up to the boundary.")
        judged = slice(0, count)
    else:
        source = "forced" if args.prefix_frames is not None else "detected"
        print(f"\nprefix boundary {boundary} ({source}); drift before it {drift[:boundary].mean():.2f}, "
              f"after {drift[boundary:].mean():.2f}")
        if args.prefix_frames is None:
            print("  A boundary near 34 frames is CWM's wn prefix arriving as advertised. One far from it means\n"
                  "  the step is something else -- read the curve before trusting the split.")
        judged = slice(boundary, count)

    before, after = left_error[judged].mean(), right_error[judged].mean()
    moved = drift[judged].mean()
    print(f"\nover the {count - (judged.start or 0)} generated frames:")
    print(f"  prediction moved      {moved:.2f}")
    print(f"  error to target       {before:.2f} -> {after:.2f}  ({(after - before) / before * 100:+.1f}%)")

    print()
    # Two independent thresholds, because the failures they separate are unrelated. A drift under a
    # quantisation step is not a small effect, it is no effect. A 2% error change is inside the gap
    # between two samplings of the same checkpoint at different seeds.
    if moved < 1.0:
        print("The prediction barely moved, which is not a weak-training result -- training that did anything at\n"
              "all perturbs a diffusion trajectory visibly. Suspect the adapter instead: check the resume's\n"
              "'lora_B norm 0 -> ...' line, and that the step in these filenames is the step you asked for.")
    elif after < before * 0.98:
        print("The prediction moved and moved toward the target, so the LoRA is learning the task. Whether it has\n"
              "learned enough is a question about how much further the error can fall, not about whether training\n"
              "is working -- compare a third checkpoint to see if the trend is still going.")
    elif after > before * 1.02:
        print("The prediction moved *away* from the target. The adapter is training on something, and it is not\n"
              "this. A conditioning signal the model reads differently at sampling time than at training time\n"
              "does exactly this, so check that the sampled regime matches the cache's: given-frame count,\n"
              "system prompt, proxy grid.")
    else:
        print("The prediction moved but not measurably toward the target. The adapter is reaching the model and\n"
              "the optimiser is doing something, so this is not a plumbing failure -- it is the substantive\n"
              "outcome that the run is not learning to track. Weight-space drift is the next thing to read:\n"
              "compare_lora_checkpoints.py says whether the gradient has a consistent direction or is circling.")


if __name__ == "__main__":
    main()
