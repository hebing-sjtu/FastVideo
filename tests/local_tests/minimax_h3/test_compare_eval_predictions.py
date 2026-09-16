# SPDX-License-Identifier: Apache-2.0
"""The eval-panel reader's one load-bearing step: locating the prediction and target columns.

Everything this script reports is a statistic over two crops. A split that is off by a column
still produces plausible-looking numbers -- drift and error both stay finite and neither looks
wrong -- so the split has to be checked against a panel the callback actually composed rather
than against the arithmetic that produced it.

The geometries here are the two that matter: a 704x1280 target whose 192x336 proxy has a
*different* aspect ratio, so the columns are unequal and the naive "split into thirds" is wrong,
and CWM's released 768x1344, where the ratios match and thirds would have worked by accident.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "scripts" / "h3_proxy"
# By path rather than by import: `compose_comparison_video` needs numpy and Pillow, while
# `import fastvideo.train.utils.video_panels` runs the package __init__ and so needs torch and a
# CUDA probe. The composition is the thing under test and it has no business requiring either.
PANELS = REPO / "fastvideo" / "train" / "utils" / "video_panels.py"


def _load(name: str, path: Path | None = None) -> types.ModuleType:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, path or SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _noise(frames: int, height: int, width: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (height, width, 3), dtype=np.uint8) for _ in range(frames)]


@pytest.mark.parametrize("labels", [True, False])
@pytest.mark.parametrize("panel_height", [384, 704])
def test_the_split_recovers_the_exact_columns_the_callback_composed(labels, panel_height):
    """Three sources at three different aspect ratios, which is the case in practice.

    The proxy and target columns come from clips decoded off disk, so their widths follow those
    files rather than the cache. Nothing in the run records them, and they are not equal: any
    method that assumes equal thirds, or derives a width from the cache's proxy grid, gets a
    plausible-looking crop of the wrong pixels.
    """
    panels_module = _load("video_panels", PANELS)
    module = _load("compare_eval_predictions")

    proxy = _noise(3, 192, 336, seed=3)  # 1.750
    prediction = _noise(3, 704, 1280, seed=1)  # 1.818, the sampling canvas
    ground_truth = _noise(3, 1080, 1920, seed=2)  # 1.778, the clip as captured
    composed = panels_module.compose_comparison_video(
        [proxy, prediction, ground_truth],
        labels=["proxy (src)", "prediction", "target (tgt)"] if labels else None,
        separator_px=4,
        height=panel_height,
    )
    layout = module.solve_layout(composed[0], 4)

    # The panel scales each source to the panel height, so the crop is compared against the scaled
    # source rather than the original -- and then exactly. These arrays went in uncompressed, so a
    # crop that is right is bit-for-bit right and anything less means the columns are misplaced.
    for name, source in (("prediction", prediction), ("target", ground_truth)):
        expected = panels_module.resize_frames(source, panel_height)
        assert np.array_equal(layout.crop(composed[0], name), expected[0]), name


def test_a_panel_with_no_findable_gutters_is_refused():
    """Better a refusal than a statistic over two arbitrary crops, which is what guessing gives."""
    module = _load("compare_eval_predictions")
    with pytest.raises(SystemExit, match="Found 0 separator columns"):
        module.solve_layout(_noise(1, 402, 2062, seed=5)[0], 4)


def test_dark_footage_does_not_read_as_a_gutter():
    """The failure mode of looking for dark columns, which is why flatness is what is required.

    A night clip supplies plenty of columns as dark as the gutter. None of them is *uniform* down
    the whole canvas, and demanding that every row be within tolerance is what separates them.
    """
    panels_module = _load("video_panels", PANELS)
    module = _load("compare_eval_predictions")

    rng = np.random.default_rng(11)
    night = [rng.integers(0, 40, (200, 360, 3), dtype=np.uint8) for _ in range(2)]
    composed = panels_module.compose_comparison_video([night, night, night], labels=None, separator_px=4, height=200)
    layout = module.solve_layout(composed[0], 4)
    assert len(layout.columns) == 3
    assert np.array_equal(layout.crop(composed[0], "prediction"), night[0])


def test_two_columns_must_be_named():
    """A two-panel layout is genuinely ambiguous: proxy|prediction and prediction|target both occur."""
    panels_module = _load("video_panels", PANELS)
    module = _load("compare_eval_predictions")

    prediction = _noise(2, 704, 1280, seed=1)
    ground_truth = _noise(2, 1080, 1920, seed=2)
    composed = panels_module.compose_comparison_video([prediction, ground_truth], separator_px=4, height=384)
    with pytest.raises(SystemExit, match="pass --columns"):
        module.solve_layout(composed[0], 4)

    layout = module.solve_layout(composed[0], 4, ["prediction", "target"])
    expected = panels_module.resize_frames(ground_truth, 384)
    assert np.array_equal(layout.crop(composed[0], "target"), expected[0])


def test_a_flat_drift_curve_has_no_prefix_boundary():
    """The answer that matters most: a run whose prediction never moved has no step to find.

    Reporting a boundary anyway would split a flat curve at an arbitrary frame and then compare
    two halves of the same number, which reads as a clean result.
    """
    module = _load("compare_eval_predictions")
    rng = np.random.default_rng(0)
    assert module.detect_prefix(rng.normal(8.0, 0.2, 60)) is None
    assert module.detect_prefix(np.zeros(60)) is None


def test_a_wn_prefix_shows_up_as_the_boundary():
    """Near-zero drift over the given frames, then real drift: the shape wn should produce."""
    module = _load("compare_eval_predictions")
    drift = np.concatenate([np.full(34, 0.3), np.full(26, 9.0)])
    assert module.detect_prefix(drift) == 34


def test_a_flat_error_with_an_improved_rate_is_not_reported_as_a_failure():
    """The two measurements can disagree, and the rate is the one the run is being asked about.

    At a mean error of tens of levels, distance to the target is an appearance measurement. It can
    sit still while the prediction's rate of change moves onto the take's -- and calling that a
    failure to learn would be the wrong conclusion from the right numbers.
    """
    module = _load("compare_eval_predictions")
    sluggish = module.verdict(moved=11.0, before=32.16, after=32.36, rates={"left": 0.55, "right": 0.80}, tracks={"left": 0.30, "right": 0.30})
    assert "rate of change moved toward" in sluggish

    stuck = module.verdict(moved=11.0, before=32.16, after=32.36, rates={"left": 0.55, "right": 0.56}, tracks={"left": 0.30, "right": 0.30})
    assert "not learning to track" in stuck

    # And the rate has to be consulted *before* the error-worsened verdict, or a large improvement
    # in tracking gets reported as training on the wrong thing whenever appearance error ticks up.
    risen = module.verdict(moved=56.0, before=51.06, after=53.33, rates={"left": 0.25, "right": 0.83}, tracks={"left": 0.30, "right": 0.30})
    assert "rate of change moved toward" in risen and "up 4.4%" in risen


@pytest.mark.parametrize(
    ("moved", "before", "after", "expected"),
    [
        (0.2, 32.0, 32.0, "barely moved"),
        (11.0, 32.0, 28.0, "moved toward the target"),
        (11.0, 32.0, 36.0, "moved *away*"),
    ],
)
def test_the_verdict_separates_outcomes_that_look_alike(moved, before, after, expected):
    """Each of these has a different cause and a different next action, and a resume that loaded
    nothing must never be confused with a run that trained and did not learn."""
    module = _load("compare_eval_predictions")
    rates = {"left": 1.0, "right": 1.0}
    tracks = {"left": 0.30, "right": 0.30}
    assert expected in module.verdict(moved=moved, before=before, after=after, rates=rates, tracks=tracks)


def test_tracking_outranks_the_rate_and_the_error():
    """Direction beats magnitude beats appearance, because that is the order of their resolution.

    Rate is blind to direction -- a prediction churning in the wrong place changes by as much per
    frame as the take does -- so a rate that did not move says nothing once tracking has.
    """
    module = _load("compare_eval_predictions")
    tracked = module.verdict(moved=11.0,
                             before=32.16,
                             after=32.36,
                             rates={"left": 1.02, "right": 1.02},
                             tracks={"left": 0.21, "right": 0.34})
    assert "Tracking rose" in tracked

    # And the real gta_v2_wn numbers: everything still, including a rate that was already 1.02x in
    # the base model, so the sluggishness hypothesis was never the one to test.
    stalled = module.verdict(moved=11.0,
                             before=32.16,
                             after=32.36,
                             rates={"left": 1.02, "right": 1.02},
                             tracks={"left": 0.21, "right": 0.21})
    assert "none of the three measurements improved" in stalled


def test_a_tracking_score_is_read_against_its_own_off_time_floor():
    """The same tracking score means opposite things at different floors.

    A steady pan correlates with a steady pan whenever you sample it, so the floor is not zero and
    an on-time score above it is the only evidence that the prediction follows *this* take rather
    than merely moving like it.
    """
    module = _load("compare_eval_predictions")
    stalled = dict(moved=11.0,
                   before=32.16,
                   after=32.36,
                   rates={"left": 1.02, "right": 1.02},
                   tracks={"left": 0.076, "right": 0.072})

    locked = module.verdict(**stalled, null={"left": 0.010, "right": 0.011})
    assert "locked to this take, though" in locked

    adrift = module.verdict(**stalled, null={"left": 0.070, "right": 0.069})
    assert "not locked to this take at all" in adrift
    assert "global statistic rather than as a" in adrift

    # No null computed: neither claim is available, and neither is made.
    silent = module.verdict(**stalled)
    assert "locked to this take" not in silent


def _wn_panels(module, tmp_path, monkeypatch, *, improvement: float, reseed: bool = False):
    """Two eval directories whose panels share a prefix and diverge after it, as wn produces.

    ``improvement`` scales how much of the baseline's error the later run removes over the
    generated frames. ``reseed`` gives the later run a *different* error of the same size, which
    is the case that separates "moved toward the target" from merely "moved": without it, an
    improvement of zero makes the two predictions identical and tests the wrong branch.
    """
    compose_comparison_video = _load("video_panels", PANELS).compose_comparison_video
    geometry = ("--training.data.num_height 704 --training.data.num_width 1280 "
                "--callbacks.validation.anchor_short_edge 2048 "
                "--callbacks.validation.proxy_height 192 --callbacks.validation.proxy_width 336")
    frames, prefix = 60, 34
    rng = np.random.default_rng(7)
    # The real shapes: a 384-tall panel, a proxy and a target decoded off disk at their own
    # resolutions, and a prediction on the sampling canvas. No two columns come out the same width,
    # which is what the target-to-prediction scoring has to survive.
    target = [rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8) for _ in range(frames)]
    proxy = [rng.integers(0, 256, (192, 336, 3), dtype=np.uint8) for _ in range(frames)]
    error = [rng.integers(0, 90, (704, 1280, 3), dtype=np.uint8) for _ in range(frames)]
    other = [rng.integers(0, 90, (704, 1280, 3), dtype=np.uint8) for _ in range(frames)]
    # The prefix is the target resampled onto the prediction's canvas, which is what a decoded
    # given-frame prefix is: the same pixels the target panel shows, at the sampling resolution.
    from PIL import Image

    given = [
        np.asarray(Image.fromarray(frame).resize((1280, 704), Image.Resampling.LANCZOS), dtype=np.uint8)
        for frame in target
    ]

    def prediction(scale: float, deviation: list[np.ndarray]) -> list[np.ndarray]:
        # The prefix is the given footage in both runs, which is what "already given" means.
        return [
            given[t] if t < prefix else np.clip(given[t].astype(np.int32) + deviation[t] * scale, 0,
                                                255).astype(np.uint8) for t in range(frames)
        ]

    directories = []
    for name, scale in (("left", 1.0), ("right", 1.0 - improvement)):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "eval_manifest.json").write_text(
            json.dumps({
                "step": 0 if name == "left" else 525,
                "cache": "/cache/gta_v2_cwm_wn",
                "geometry": geometry,
                "val_json": "/val.json",
                "nproc": 8,
            }),
            encoding="utf-8",
        )
        deviation = other if (reseed and name == "right") else error
        panel = compose_comparison_video([proxy, prediction(scale, deviation), target],
                                         labels=["proxy (src)", "prediction", "target (tgt)"],
                                         separator_px=4,
                                         height=384)
        directories.append((directory, panel))

    saved = {directory: panel for directory, panel in directories}
    monkeypatch.setattr(module, "read_frames", lambda path: saved[path.parent])
    monkeypatch.setattr(
        module, "index_videos", lambda directory: {("0", "_compare"): directory / "validation_step_0_video_0.mp4"})
    return [directory for directory, _ in directories]


def test_a_run_that_moved_toward_the_target_reads_as_learning(tmp_path, monkeypatch, capsys):
    module = _load("compare_eval_predictions")
    left, right = _wn_panels(module, tmp_path, monkeypatch, improvement=0.5)
    monkeypatch.setattr(sys, "argv", ["compare_eval_predictions.py", str(left), str(right)])
    module.main()
    out = capsys.readouterr().out
    assert "moved toward the target" in out
    # The boundary has to be found within a frame or two of the real one. Exactness is too much to
    # ask of a curve read off lossy-free arrays here but compressed in practice.
    assert "prefix boundary 34" in out


def test_a_run_that_moved_without_improving_reads_as_not_learning(tmp_path, monkeypatch, capsys):
    """The substantive outcome, and the one that must not be reported as a plumbing failure."""
    module = _load("compare_eval_predictions")
    # A different error of the same size: the prediction moves, its distance to the target does not.
    left, right = _wn_panels(module, tmp_path, monkeypatch, improvement=0.0, reseed=True)
    monkeypatch.setattr(sys, "argv", ["compare_eval_predictions.py", str(left), str(right)])
    module.main()
    out = capsys.readouterr().out
    assert "not learning to track" in out or "moved *away*" in out
    assert "barely moved" not in out
