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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from compare_eval_outputs import compare_manifests, index_videos, load_manifest

# `MiniMaxH3ProxyValidationCallback` defaults; a run that overrode panel_separator_px needs it
# passed here too.
DEFAULT_SEPARATOR_PX = 4

# `compose_comparison_video` fills the gutters -- and any odd-dimension padding -- with this, and
# fills them flat. Flatness is what makes a gutter findable: a column of real footage that is this
# dark for all 384 rows of the canvas, to within compression error, essentially does not occur.
GUTTER_VALUE = 24
GUTTER_TOLERANCE = 14.0

# The label band's background. It is a different value from the gutter, which is the only reason
# the band's height can be read off the picture at all: the band spans the gutters, so both are
# flat and dark there and any threshold wide enough to accept one accepts the other.
LABEL_BACKGROUND_VALUE = 16

# The panel's own columns cannot be derived from the cache geometry, which is why they are found
# instead. `_read_panel` decodes the proxy and target clips from *disk at their native resolution*
# and scales them to the panel height; the cache's proxy grid is the encoder's reference size and
# says nothing about the mp4 the panel was built from. Nor are the columns equal: a 704x1280
# prediction next to a proxy of any other aspect ratio gives three different widths.
PANEL_ORDER = ("proxy", "prediction", "target")


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
    parser.add_argument("--columns",
                        default=None,
                        help="Name the panel's columns left to right, e.g. 'prediction,target', when the panel is "
                        f"not the default {','.join(PANEL_ORDER)}.")
    parser.add_argument("--force",
                        action="store_true",
                        help="Compare even when the manifests disagree. The numbers are then about the videos only "
                        "and say nothing about the checkpoints.")
    return parser.parse_args()


@dataclass(frozen=True)
class PanelLayout:
    """Where the prediction and target columns sit inside a composed panel frame."""

    band_height: int
    columns: dict[str, tuple[int, int]]

    def crop(self, frame: np.ndarray, name: str) -> np.ndarray:
        start, end = self.columns[name]
        return frame[self.band_height:, start:end]

    def describe(self) -> str:
        return ", ".join(f"{name} {start}:{end} ({end - start}px)" for name, (start, end) in self.columns.items())


def _flat_gutter_columns(frame: np.ndarray, band_probe: int) -> np.ndarray:
    """Columns that are the gutter colour in *every* probed row, not merely on average.

    The distinction carries the whole method. A mean would accept a column of dark footage with a
    bright pixel in it; requiring every row to be within tolerance means one bright pixel
    disqualifies the column, and a real column that is uniform to +-14 over hundreds of rows is
    not something game footage produces.
    """
    probed = frame[band_probe:, :, :].astype(np.float64)
    return np.abs(probed - GUTTER_VALUE).max(axis=(0, 2)) < GUTTER_TOLERANCE


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True spans of ``mask`` as half-open [start, end) intervals."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(end)) for start, end in zip(edges[::2], edges[1::2], strict=True)]


def _band_height(frame: np.ndarray, gutter: tuple[int, int]) -> int:
    """How many rows the label band occupies, or 0 if the run drew no labels.

    Measured down a gutter column, where the band's 16 sits directly above the gutter's 24 with no
    picture in between. Classifying each row by which of the two it is nearer finds that boundary;
    a darkness threshold cannot, because it accepts both.
    """
    column = frame[:, (gutter[0] + gutter[1]) // 2, :].astype(np.float64).mean(axis=1)
    is_band = np.abs(column - LABEL_BACKGROUND_VALUE) < np.abs(column - GUTTER_VALUE)
    # Four consecutive rows, so a label long enough to overflow its panel and cross the gutter
    # costs a row of evidence rather than the whole measurement.
    for row in range(is_band.size - 3):
        if not is_band[row:row + 4].any():
            return row
    return 0


def solve_layout(frame: np.ndarray, separator_px: int, names: list[str] | None = None) -> PanelLayout:
    """Find the columns in the panel itself, because nothing else records them.

    The obvious alternative -- reconstructing the composition from the cache geometry -- cannot
    work: the proxy and target columns come from clips decoded off disk at whatever resolution they
    were written at, while the geometry records the encoder's reference grid. Those are different
    numbers, and a 402x2062 panel from a 704x1280 cache is what that mistake looks like.
    """
    height, width = frame.shape[0], frame.shape[1]
    # Well below any plausible label band (``max(18, canvas // 22)``) and well above zero, so the
    # probe sees canvas rows whether or not labels were drawn.
    band_probe = max(1, height // 8)
    gutters = [
        (start, end) for start, end in _runs(_flat_gutter_columns(frame, band_probe))
        # A gutter is exactly `separator_px` wide before encoding; compression bleeds its edges into
        # the neighbouring image columns, so the core can come back narrower. The padding column
        # `_pad_to_even` may add is 1px and is excluded here, then trimmed off the last panel below.
        if 2 <= (end - start) <= separator_px + 2
    ]
    if not 1 <= len(gutters) <= 2:
        raise SystemExit(f"Found {len(gutters)} separator columns in a {height}x{width} panel, expected 1 or 2 "
                         f"(3 panels means 2 gutters). Pass --separator-px if the run overrode "
                         f"panel_separator_px. Candidates at: {gutters}")

    band_height = _band_height(frame, gutters[0])

    bounds = [0]
    for start, end in gutters:
        bounds.extend((start, end))
    bounds.append(width)
    spans = [(bounds[index], bounds[index + 1]) for index in range(0, len(bounds), 2)]

    # `_pad_to_even` grows an odd width by a gutter-coloured column, which belongs to no panel.
    last_start, last_end = spans[-1]
    flat = _flat_gutter_columns(frame, band_probe)
    while last_end > last_start + 1 and flat[last_end - 1]:
        last_end -= 1
    spans[-1] = (last_start, last_end)

    if names is None:
        if len(spans) != len(PANEL_ORDER):
            raise SystemExit(f"Found {len(spans)} panels, and only a {len(PANEL_ORDER)}-panel "
                             f"{'|'.join(PANEL_ORDER)} layout can be named by position. Widths are "
                             f"{[end - start for start, end in spans]}; pass --columns to say which is which.")
        names = list(PANEL_ORDER)
    elif len(names) != len(spans):
        raise SystemExit(f"--columns names {len(names)} columns but {len(spans)} were found, with widths "
                         f"{[end - start for start, end in spans]}.")

    return PanelLayout(band_height=band_height, columns=dict(zip(names, spans, strict=True)))


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


def match_width(frame: np.ndarray, width: int) -> np.ndarray:
    """Resample to ``width``, for scoring a target column against a prediction column.

    The two are the same height and rarely the same width: the panel scales each source to the
    panel height, and a clip captured at 1080x1920 does not land on the width a 704x1280 sampling
    canvas does. Only the target is ever moved, so neither run's prediction is touched and the
    resampling contributes the same blur to both sides of the comparison.
    """
    if frame.shape[1] == width:
        return frame
    from PIL import Image

    return np.asarray(Image.fromarray(frame).resize((width, frame.shape[0]), Image.Resampling.LANCZOS),
                      dtype=np.uint8)


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
    """One digit per frame bucket, scaled 0-9 against the row's own maximum.

    Digits rather than block-drawing characters: those need a font that has them and a paste that
    preserves them, and when either fails every bucket collapses to the same glyph and the curve
    reads as flat -- which is one of the conclusions this is supposed to distinguish.
    """
    if values.size == 0:
        return ""
    buckets = np.array_split(values, min(width, values.size))
    heights = np.array([float(bucket.mean()) for bucket in buckets])
    top = float(heights.max()) or 1.0
    return "".join(str(min(9, int(round(height / top * 9)))) for height in heights)


# Frame differences are correlated at a quarter resolution. The point is where the change lands,
# which survives it, and it makes the off-time null affordable: that needs every frame's difference
# kept, then correlated against several time offsets.
DELTA_DOWNSAMPLE = 4

# Offsets for the off-time null, spread and coprime with nothing in particular so that a clip with
# periodic motion cannot line up with all of them.
NULL_LAGS = (5, 17, 41)


def downsample(frame: np.ndarray, factor: int = DELTA_DOWNSAMPLE) -> np.ndarray:
    """Block-mean to 1/factor in each axis, trimming the remainder rather than padding it."""
    height = frame.shape[0] - frame.shape[0] % factor
    width = frame.shape[1] - frame.shape[1] % factor
    trimmed = frame[:height, :width].astype(np.float32)
    return trimmed.reshape(height // factor, factor, width // factor, factor, -1).mean(axis=(1, 3))


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Pearson correlation, or nan when either side is constant and has none to report."""
    if left.size < 2 or left.std() == 0 or right.std() == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def lagged_null(store: list[dict[str, np.ndarray]], first: int) -> dict[str, float]:
    """What tracking scores when the prediction is held against the take at the wrong time.

    Without this the on-time score has no scale. 0.076 is a real number only relative to what the
    same arithmetic returns for frames that cannot correspond, and for a clip whose motion barely
    changes over its length that floor is not near zero: a steady pan matches a steady pan whenever
    you sample it. The gap between the two is the part that says the prediction is locked to *this*
    take rather than merely moving like it.
    """
    scores: dict[str, list[float]] = {"left": [], "right": []}
    for entry in store:
        frames = entry["target"].shape[0]
        # ``first`` is a frame index and the stored differences begin at frame 1, so it shifts by
        # one. Getting this wrong would fold given frames into the null and flatter it.
        indices = np.arange(max(0, first - 1), frames)
        if indices.size == 0:
            continue
        for lag in NULL_LAGS:
            if lag >= frames:
                continue
            shifted = (indices + lag) % frames
            for name in scores:
                scores[name].append(
                    float(
                        np.nanmean([
                            correlation(entry[name][at].ravel(), entry["target"][to].ravel())
                            for at, to in zip(indices, shifted, strict=True)
                        ])))
    return {name: float(np.nanmean(values)) if values else float("nan") for name, values in scores.items()}


def verdict(*,
            moved: float,
            before: float,
            after: float,
            rates: dict[str, float],
            tracks: dict[str, float],
            null: dict[str, float] | None = None,
            achievable: float = float("nan")) -> str:
    """Which of the five outcomes this is, given the three numbers that separate them.

    The thresholds are independent because the failures are unrelated. A drift below a quantisation
    step is not a small effect but no effect: training that did anything at all perturbs a diffusion
    trajectory. A 2% error change is inside the spread between two samplings of one checkpoint.
    """
    if moved < 1.0:
        return ("The prediction barely moved, which is not a weak-training result -- training that did anything at\n"
                "all perturbs a diffusion trajectory visibly. Suspect the adapter instead: check the resume's\n"
                "'lora_B norm 0 -> ...' line, and that the step in these filenames is the step you asked for.")
    if after < before * 0.98:
        return ("The prediction moved and moved toward the target, so the LoRA is learning the task. Whether it has\n"
                "learned enough is a question about how much further the error can fall, not about whether training\n"
                "is working -- compare a third checkpoint to see if the trend is still going.")
    # Before the error-worsened verdict, not after it. When these disagree the motion measurements
    # are the ones the run is being asked about, and the error is the coarser instrument: at a mean
    # of tens of levels it is measuring appearance, where a few percent is not evidence about
    # alignment. Ordering these the other way calls a large improvement in tracking a failure.
    if tracks["right"] - tracks["left"] > 0.02:
        return (f"Tracking rose from {tracks['left']:+.3f} to {tracks['right']:+.3f}: the prediction's frame-to-frame\n"
                "change is landing more where the take's does. That is the claim 'it follows the picture' makes,\n"
                "and it is direction-sensitive in a way neither the appearance error nor the rate is. Read it as\n"
                "the result even if the error did not move.")
    if abs(rates["left"] - 1.0) - abs(rates["right"] - 1.0) > 0.05:
        direction = "flat" if after <= before * 1.02 else f"up {(after - before) / before * 100:.1f}%"
        return (f"The prediction's rate of change moved toward the take's while pixel error stayed {direction}.\n"
                "Those are not in conflict: at a mean error of tens of levels the distance to the target is an\n"
                "appearance measurement, too coarse to register an alignment that improved. Rate is the axis the\n"
                "run is being asked about, so read it as the result and the error as uninformative here.")
    if after > before * 1.02:
        return ("The prediction moved *away* from the target, and its rate of change did not improve either. The\n"
                "adapter is training on something, and it is not this. A conditioning signal the model reads\n"
                "differently at sampling time than at training time does exactly this, so check that the sampled\n"
                "regime matches the cache's: given-frame count, system prompt, proxy grid.")
    lines = [
        "The prediction moved, and none of the three measurements improved: appearance error, rate, and",
        f"tracking ({tracks['left']:+.3f} -> {tracks['right']:+.3f}) all sat still. The adapter is reaching the "
        "model and the",
        "optimiser is doing something, so this is not a plumbing failure -- it is the substantive outcome that",
        "the run is not learning to track.",
    ]
    if null:
        # Against the off-time floor rather than against zero. Whether the prediction is locked to
        # this take at all is a different question from whether training improved the locking, and
        # it is the one that decides if the conditioning is reaching the model in any form.
        margin = tracks["right"] - null["right"]
        if margin > 0.02:
            lines += [
                "",
                f"It is locked to this take, though: tracking {tracks['right']:+.3f} against an off-time floor of "
                f"{null['right']:+.3f}",
                "means the change does land where the take's does, more than it would for frames that cannot",
                "correspond. The conditioning is reaching the model. What did not move is how well it is used.",
            ]
            if achievable == achievable and achievable > 0.1:
                lines += [
                    "",
                    f"How well is {margin / achievable * 100:.0f}% of the {achievable:+.3f} an exact match scores on "
                    "the given prefix. So there is",
                    "headroom and the signal to climb it is present -- what is missing is a gradient that can",
                    "attribute the difference, which is a property of the conditioning pathway and not of how",
                    "long it trained.",
                ]
        else:
            lines += [
                "",
                f"And it is not locked to this take at all: tracking {tracks['right']:+.3f} against an off-time "
                f"floor of {null['right']:+.3f}",
                "is no gap. The prediction moves as much as the take and at the same moments, and in places the",
                "take does not. That is a conditioning signal arriving as a global statistic rather than as a",
                "correspondence, which is what a pathway with no frame or token registration would produce.",
            ]
    lines += [
        "",
        "Weight-space drift is the next thing to read: compare_lora_checkpoints.py says whether the gradient",
        "has a consistent direction or is circling, and a random walk there confirms this rather than adding",
        "to it.",
    ]
    return "\n".join(lines)


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
    names = [name.strip() for name in args.columns.split(",")] if args.columns else None

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
    motions: dict[str, list[np.ndarray]] = {"left": [], "right": [], "target": []}
    trackings: dict[str, list[np.ndarray]] = {"left": [], "right": [], "proxy": []}
    delta_store: list[dict[str, np.ndarray]] = []
    layout_note: str | None = None
    print(f"\n{len(shared)} panels in common:")
    for index in shared:
        left_frames, right_frames = read_frames(left_videos[index]), read_frames(right_videos[index])
        if not left_frames or not right_frames:
            print(f"  video_{index}: unreadable, skipped")
            continue
        layout = solve_layout(left_frames[0], args.separator_px, names)
        if layout_note is None:
            layout_note = layout.describe()
            print(f"  panel columns: {layout_note}; label band {layout.band_height}px")
        for required in ("prediction", "target"):
            if required not in layout.columns:
                raise SystemExit(f"No {required} column in the panel, so there is nothing to measure. Columns "
                                 f"found: {layout.describe()}")
        # Truncated rather than padded: a held frame would score as the model diverging at the tail.
        count = min(len(left_frames), len(right_frames))
        drift = np.empty(count)
        left_error = np.empty(count)
        right_error = np.empty(count)
        if left_frames[0].shape != right_frames[0].shape:
            raise SystemExit(f"video_{index} is {left_frames[0].shape} on the left and {right_frames[0].shape} on "
                             "the right, so one panel's columns cannot locate the other's. The runs composed at "
                             "different panel_height or included different columns, which makes them "
                             "incomparable however the checkpoints did.")
        # Frame-to-frame change, which is what "the camera moves at the wrong speed" is a statement
        # about. Distance to the target is dominated by appearance -- a mean of 32/255 is nowhere
        # near a near-miss on alignment -- so it can sit flat while tracking improves underneath it.
        motion = {"left": np.zeros(count), "right": np.zeros(count), "target": np.zeros(count)}
        # Rate is a magnitude and blind to direction: a prediction churning in the wrong place can
        # change by exactly as much per frame as the take does. Correlating the two frame
        # differences pixel by pixel asks whether the change happens in the same places and the
        # same sense, which is what "it does not follow the picture" actually claims.
        tracking = {"left": np.zeros(count), "right": np.zeros(count), "proxy": np.zeros(count)}
        # Kept for the off-time null, which cannot be computed until the prefix boundary is known
        # and that comes from the drift curve averaged over every panel. A quarter-resolution
        # difference is ~50k floats, so all six clips together are tens of megabytes.
        deltas: dict[str, list[np.ndarray]] = {"left": [], "right": [], "target": []}
        previous: tuple[np.ndarray, ...] | None = None
        previous_proxy: np.ndarray | None = None
        for frame in range(count):
            left_prediction = layout.crop(left_frames[frame], "prediction")
            right_prediction = layout.crop(right_frames[frame], "prediction")
            target = match_width(layout.crop(left_frames[frame], "target"), left_prediction.shape[1])
            proxy = (match_width(layout.crop(left_frames[frame], "proxy"), left_prediction.shape[1])
                     if "proxy" in layout.columns else None)
            drift[frame] = mean_abs_diff(left_prediction, right_prediction)
            left_error[frame] = mean_abs_diff(left_prediction, target)
            right_error[frame] = mean_abs_diff(right_prediction, target)
            current = (left_prediction, right_prediction, target)
            if previous is not None:
                for name, now, before in zip(motion, current, previous, strict=True):
                    motion[name][frame] = mean_abs_diff(now, before)
                target_delta = downsample(target) - downsample(previous[2])
                deltas["target"].append(target_delta)
                for name, now, before in (("left", left_prediction, previous[0]), ("right", right_prediction,
                                                                                   previous[1])):
                    delta = downsample(now) - downsample(before)
                    deltas[name].append(delta)
                    tracking[name][frame] = correlation(delta.ravel(), target_delta.ravel())
                # The proxy on the same metric. Not a ceiling: DUV's semantic channels are codes,
                # piecewise constant over a road or a wall, so a camera pan across one produces no
                # difference at all where RGB produces a large one. The two signals have different
                # spatial support, and correlating them says nothing about what DUV determines.
                if proxy is not None and previous_proxy is not None:
                    proxy_delta = downsample(proxy) - downsample(previous_proxy)
                    tracking["proxy"][frame] = correlation(proxy_delta.ravel(), target_delta.ravel())
            previous = current
            previous_proxy = proxy
        drifts.append(drift)
        left_errors.append(left_error)
        right_errors.append(right_error)
        for name, values in motion.items():
            motions[name].append(values)
        for name, values in tracking.items():
            trackings[name].append(values)
        # The first frame has no difference, so the stored deltas start at frame 1 and the null's
        # indices have to line up with that.
        delta_store.append({name: np.stack(values) for name, values in deltas.items() if values})
        print(f"  video_{index}: {count} frames, drift {drift.mean():6.2f}, "
              f"error {left_error.mean():6.2f} -> {right_error.mean():6.2f}")

    if not drifts:
        raise SystemExit("Nothing was readable.")

    count = min(drift.size for drift in drifts)
    drift = np.mean([d[:count] for d in drifts], axis=0)
    left_error = np.mean([e[:count] for e in left_errors], axis=0)
    right_error = np.mean([e[:count] for e in right_errors], axis=0)
    motion = {name: np.mean([values[:count] for values in curves], axis=0) for name, curves in motions.items()}
    track = {name: np.mean([values[:count] for values in curves], axis=0) for name, curves in trackings.items()}

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

    # The rate question, asked separately from the appearance question. `rate` is how fast the
    # picture changes relative to the take: below 1 the prediction is sluggish, above 1 it churns.
    # `phase` is whether it speeds up and slows down when the take does, which a prediction at the
    # right average rate can still get wrong.
    reference = motion["target"][judged]
    # Phase is only answerable if the take's own rate varies. A clip shot at a constant speed has
    # no accelerations to match, and correlating two near-constant curves reports noise.
    phased = reference.std() > 0.05 * reference.mean() if reference.mean() else False
    print(f"\nframe-to-frame motion over the same frames (target {reference.mean():.2f}):")
    null = lagged_null(delta_store, judged.start or 0)
    rates = {}
    tracks = {}
    for name, label in (("left", "baseline "), ("right", "checkpoint")):
        observed = motion[name][judged]
        rates[name] = observed.mean() / reference.mean() if reference.mean() else float("nan")
        tracks[name] = float(np.nanmean(track[name][judged]))
        phase = f"phase {correlation(observed, reference):+.2f}" if phased else "phase n/a"
        print(f"  {label}            {observed.mean():.2f}   rate {rates[name]:.2f}x   {phase}   "
              f"tracking {tracks[name]:+.3f}  (off-time {null[name]:+.3f})")
    if not phased:
        print("  (the take's own rate barely varies over these frames, so there are no accelerations to match)")
    # The given prefix is an exact match by construction -- the prediction there *is* the decoded
    # target footage -- so its score is what this metric returns for a prediction that tracks
    # perfectly, through the same VAE roundtrip, panel resampling and encoder. Unlike the proxy's
    # cross-modal number this is a real ceiling, and it turns "+0.100, is that good" into a
    # fraction. Frame 0 has no difference, hence the 1.
    achievable = float("nan")
    if judged.start:
        prefix = np.concatenate([track["left"][1:judged.start], track["right"][1:judged.start]])
        achievable = float(np.nanmean(prefix)) if prefix.size else float("nan")
        print(f"  given prefix, both runs                                  tracking {achievable:+.3f}   "
              "<- an exact match")

    cross_modal = float(np.nanmean(track["proxy"][judged]))
    if cross_modal:
        print(f"  proxy vs target                                          tracking {cross_modal:+.3f}")
    print("\n  rate is a magnitude. tracking is the pixelwise correlation of the two frame differences, so it is\n"
          "  the one that says whether the change lands where the take's does -- but only against its own\n"
          "  off-time score, which is the same arithmetic on frames that cannot correspond. A steady pan\n"
          "  matches a steady pan at any offset, so the floor is not zero and the gap is the whole signal.\n"
          "  The prefix bounds it from above: there the prediction is the given footage, so that is what\n"
          "  tracking returns for an exact match under the same roundtrip and compression.")
    if cross_modal:
        print("  The proxy's number is not a ceiling. DUV's semantic channels are codes, constant across a road\n"
              "  or a wall, so a pan over one changes nothing where RGB changes a lot: the two have different\n"
              "  spatial support and correlating them says nothing about what the DUV determines.")

    print()
    print(
        verdict(moved=moved,
                before=before,
                after=after,
                rates=rates,
                tracks=tracks,
                null=null,
                achievable=achievable))


if __name__ == "__main__":
    main()
