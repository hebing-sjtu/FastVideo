# SPDX-License-Identifier: Apache-2.0
"""Long takes cut into the overlapping clips of a proxy rollout."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts/h3_proxy/prepare_data"
H, W = 32, 64


@pytest.fixture(scope="module")
def split():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("split_proxy_take", SCRIPTS / "split_proxy_take.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _streams(frames: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    semantic = np.zeros((frames, H, W, 3), dtype=np.uint8)
    semantic[..., 2] = rng.choice(np.array([0, 1, 3], dtype=np.uint8), size=(frames, H, W))
    return {
        "target": rng.integers(0, 256, size=(frames, H, W, 3), dtype=np.uint8),
        "depth": rng.integers(1, 256, size=(frames, H, W), dtype=np.uint8),
        "semantic": semantic,
    }


def _sidecars(directory: Path) -> None:
    (directory / "track.json").write_text(json.dumps({"depth": {"encoding": "h264-logz-gray8"}}))
    (directory / "semantic.json").write_text(json.dumps({"classes": {"0": "static world", "1": "player",
                                                                     "3": "vehicle"}}))


def test_the_geometry_matches_the_rollout(split):
    assert split.STRIDE_FRAMES == 85 and split.GIVEN_FRAMES == 34
    assert [split.take_frames(n) for n in (1, 3, 9)] == [124, 294, 804]
    assert [split.segment_count(n) for n in (123, 124, 208, 209, 322)] == [0, 1, 1, 2, 3]
    assert len(split.resample_index(30, 30.0)) == 24


def test_long_videos_cut_into_lossless_overlapping_clips(split, tmp_path):
    source = _streams(215)
    for key, frames in source.items():
        split.write_lossless(tmp_path / f"{key}.mp4", frames, split.LAYOUTS[key])
    _sidecars(tmp_path)
    out = tmp_path / "take"
    split.main(["--name", "demo", "--out", str(out), "--modality", "depth", "--target", str(tmp_path / "target.mp4"),
                "--proxy-depth", str(tmp_path / "depth.mp4"), "--proxy-semantic", str(tmp_path / "semantic.mp4"),
                "--caption", "a car drives"])

    take = json.loads((out / "take.json").read_text())
    assert take["schema"] == "h3-proxy-take.v1" and take["frames"] == 209 and take["target"] == "real"
    assert [(s["name"], s["start"], s["cwm_system"]) for s in take["segments"]] == [
        ("demo-s00", 0, "w0_omni"), ("demo-s01", 85, "wn_omni")]
    rows = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
    assert rows[1]["proxy_depth_video"] == "clips/s01/depth.mp4" and rows[1]["take_anchor"] == "anchor.png"
    assert (rows[1]["cwm_system"], rows[1]["take_segment"], rows[1]["target_kind"]) == ("wn_omni", 1, "color")
    assert np.array_equal(split.decode(out / "clips/s01/target.mp4", "rgb24"), source["target"][85:209])
    assert np.array_equal(split.decode(out / "clips/s01/depth.mp4", "gray"), source["depth"][85:209])
    assert (out / "clips/s01/track.json").is_file() and (out / "clips/s01/semantic.json").is_file()
    assert np.array_equal(np.asarray(Image.open(out / "anchor.png")), source["target"][0])


def test_an_anchor_alone_gives_a_placeholder_target(split, tmp_path):
    source = _streams(124)
    split.write_lossless(tmp_path / "duv.mp4", source["target"], "rgb24")
    Image.fromarray(source["target"][5]).save(tmp_path / "look.png")
    out = tmp_path / "take"
    split.main(["--name", "p", "--out", str(out), "--modality", "duv", "--proxy-duv", str(tmp_path / "duv.mp4"),
                "--anchor", str(tmp_path / "look.png"), "--caption", "x"])
    take = json.loads((out / "take.json").read_text())
    row = json.loads((out / "manifest.jsonl").read_text())
    assert take["target"] == "placeholder" and row["target"] == "clips/placeholder_target.mp4"
    still = split.decode(out / row["target"], "rgb24")
    assert len(still) == 124 and np.array_equal(still[0], source["target"][5])


def _gta_seg(root: Path, name: str, parent: str, start: int, frames: dict[str, np.ndarray], split) -> None:
    proxy = root / name / "proxy"
    (proxy / "captions").mkdir(parents=True)
    (root / name / "metadata.json").write_text(json.dumps({"parent": parent, "split": "val"}))
    _sidecars(proxy)
    track = json.loads((proxy / "track.json").read_text())
    (proxy / "track.json").write_text(json.dumps({**track, "cut": {"source_start": start}}))
    (proxy / "captions/prompt.json").write_text(json.dumps({"text": name}))
    window = {key: value[start:start + 124] for key, value in frames.items()}
    window["duv"] = window["target"]
    files = {"target": "color.mp4", "duv": "duv.mp4", "depth": "depth.mp4", "semantic": "semantic.mp4"}
    for key, filename in files.items():
        split.write_lossless(proxy / filename, window[key], split.LAYOUTS[key])


def test_a_gta_parent_stitches_its_segs_on_source_frames(split, tmp_path, monkeypatch):
    import seg_dir_to_encode_manifest

    monkeypatch.setattr(seg_dir_to_encode_manifest, "extract_prompt_text",
                        lambda payload, prose_style="rich": payload["text"])
    source = _streams(230)
    _gta_seg(tmp_path, "seg_0007_enter", "0007", 0, source, split)
    _gta_seg(tmp_path, "seg_0007_drive", "0007", 106, source, split)
    out = tmp_path / "take"
    split.main(["--name", "g", "--out", str(out), "--modality", "semantic", "--gta-root", str(tmp_path),
                "--gta-parent", "0007"])
    take = json.loads((out / "take.json").read_text())
    assert take["frames"] == 209 and take["source"]["segs"] == ["seg_0007_enter", "seg_0007_drive"]
    assert [s["caption"] for s in take["segments"]] == ["seg_0007_enter", "seg_0007_drive"]
    assert np.array_equal(split.decode(out / "clips/s01/semantic.mp4", "rgb24"), source["semantic"][85:209])

    tampered = _streams(230, seed=1)
    _gta_seg(tmp_path / "bad", "seg_0008_enter", "0008", 0, source, split)
    _gta_seg(tmp_path / "bad", "seg_0008_drive", "0008", 106, tampered, split)
    with pytest.raises(ValueError, match="shared frames"):
        split.main(["--name", "b", "--out", str(tmp_path / "b"), "--modality", "depth", "--gta-root",
                    str(tmp_path / "bad"), "--gta-parent", "0008"])
