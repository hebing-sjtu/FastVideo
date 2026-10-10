# SPDX-License-Identifier: Apache-2.0
"""gta_record omni pairs: video-borne depth/semantic, render crops and the manifest builder."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts/h3_proxy/prepare_data"


def _load(name: str):
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def encode_script():
    return _load("encode_proxy_samples")


@pytest.fixture(scope="module")
def builder():
    return _load("record_dir_to_omni_manifest")


def _write_lossless(path: Path, frames: np.ndarray) -> None:
    """Grey as matroska rawvideo, RGB as lossless libx264rgb -- both decode bit-exact."""
    import av

    grey = frames.ndim == 3
    with av.open(str(path), mode="w", format="matroska" if grey else "mp4") as container:
        stream = container.add_stream("rawvideo" if grey else "libx264rgb", rate=24)
        stream.height, stream.width = frames.shape[1:3]
        stream.pix_fmt = "gray" if grey else "rgb24"
        if not grey:
            stream.options = {"qp": "0", "preset": "ultrafast"}
        layout = "gray" if grey else "rgb24"
        for array in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(array), format=layout)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _record_proxy(directory: Path, frames: int = 2, height: int = 36, width: int = 64) -> tuple[np.ndarray, np.ndarray]:
    directory.mkdir(parents=True)
    rng = np.random.default_rng(0)
    grey = rng.integers(1, 256, size=(frames, height, width)).astype(np.uint8)
    grey[:, 0, 0] = 0
    ids = rng.choice(np.array([0, 1, 2, 3, 10], dtype=np.uint8), size=(frames, height, width))
    semantic = np.zeros((frames, height, width, 3), dtype=np.uint8)
    semantic[..., 2] = ids
    _write_lossless(directory / "depth.mp4", grey)
    _write_lossless(directory / "semantic.mp4", semantic)
    (directory / "track.json").write_text(json.dumps({"depth": {
        "encoding": "h264-logz-gray8",
        "quantization": {"clipNear": 0.1, "clipFar": 256.0, "invalidCode": 0},
    }}))
    names = {0: "static world", 1: "player", 2: "ped", 3: "vehicle", 10: "prop"}
    (directory / "semantic.json").write_text(json.dumps(
        {"classes": {str(key): {"id": key, "name": name} for key, name in names.items()}}))
    return grey, ids


def test_record_videos_decode_to_the_planes_a_duv_directory_holds(tmp_path):
    record = _load("gta_record")
    grey, ids = _record_proxy(tmp_path / "proxy")
    depth, semantic = record.read_proxy_planes_from_videos(tmp_path / "proxy/depth.mp4",
                                                           tmp_path / "proxy/semantic.mp4", 2)
    quantization = record.read_depth_quantization(tmp_path / "proxy")
    assert np.array_equal(depth, record.decode_depth_codes(grey, quantization))
    assert depth[0, 0, 0] == 0.0
    lut = record.proxy_class_lut(record.read_semantic_map(tmp_path / "proxy"))
    assert np.array_equal(semantic, lut[ids])


class _FakeEncoders:

    def __init__(self):
        self.previews = None
        self.prompt = None

    def encode_pixels(self, pixels):
        _, _, frames, height, width = pixels.shape
        return torch.zeros(24, frames, height // 16, width // 16)

    def encode_keyframe(self, image):
        return torch.zeros(24, 1, image.size[1] // 16, image.size[0] // 16)

    def encode_text(self, prompt, anchor, previews, *, cwm_system=None):
        self.prompt = prompt
        self.previews = previews
        return torch.zeros(5, 8), torch.ones(5, dtype=torch.long)


def _mixed_args(**overrides):
    values = dict(num_frames=2, height=32, width=64, proxy_height=16, proxy_width=32, fit="center-crop",
                  code_resize="nearest", proxy_references=("duv", ),
                  proxy_variants=("duv", "depth", "semantic", "style"), cwm_system="w0_omni",
                  anchor_short_edge=64, qwen_video_fps=2.0)
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize("modality", ["duv", "depth", "semantic"])
def test_a_record_row_builds_every_plane_modality_from_videos(encode_script, tmp_path, monkeypatch, modality):
    from fastvideo.pipelines.basic.minimax_h3.proxy import proxy_reference_clip

    record = _load("gta_record")
    _record_proxy(tmp_path / "seg/proxy")
    crops = []

    def fake_read(path, frames, height, width, fit="resize", crop=None):
        crops.append(crop)
        return np.zeros((frames, height, width, 3), dtype=np.uint8)

    monkeypatch.setattr(encode_script, "read_video_frames", fake_read)
    encoders = _FakeEncoders()
    entry = {
        "target": "seg/render/output.mp4",
        "target_crop": [0, 2, 64, 34],
        "proxy_depth_video": "seg/proxy/depth.mp4",
        "proxy_semantic_video": "seg/proxy/semantic.mp4",
        "proxy_modality": modality,
        "prompt": "a car drives",
        "target_kind": "high",
        "target_style": "biopunk",
    }
    sample = encode_script.encode_entry(entry, encoders, _mixed_args(), tmp_path)
    assert crops == [[0, 2, 64, 34]]
    assert sample["proxy_latents"].shape == (1, 24, 2, 1, 2)
    assert sample["info"]["proxy_modality"] == modality
    assert sample["info"]["target_kind"] == "high" and sample["info"]["target_style"] == "biopunk"
    assert encoders.prompt.startswith(f"Reference modality: {modality} video.\n")

    depth, semantic = record.read_proxy_planes_from_videos(tmp_path / "seg/proxy/depth.mp4",
                                                           tmp_path / "seg/proxy/semantic.mp4", 2)
    fit = lambda planes: encode_script.fit_frames(planes, 16, 32, "center-crop", codes=True, what="test",  # noqa: E731
                                                  code_resize="nearest")
    expected = proxy_reference_clip(modality, fit(depth), fit(semantic))
    assert np.array_equal(encoders.previews[0], encode_script.pixels_to_preview(expected))


def test_a_composed_duv_video_may_be_nearest_resized_but_never_blended(encode_script, tmp_path):
    rng = np.random.default_rng(1)
    codes = rng.choice(np.array([0, 64, 128, 255], dtype=np.uint8), size=(2, 36, 64, 3))
    _write_lossless(tmp_path / "duv.mp4", codes)
    pixels, preview = encode_script.read_duv_video_clip(tmp_path / "duv.mp4", 2, 16, 32, "center-crop", "nearest")
    assert preview.shape == (2, 16, 32, 3)
    assert set(np.unique(preview)) <= {0, 64, 128, 255}
    with pytest.raises(ValueError):
        encode_script.read_duv_video_clip(tmp_path / "duv.mp4", 2, 16, 32, "center-crop")


def test_style_rows_carry_their_crop_and_rows_without_a_source_are_refused(encode_script, tmp_path, monkeypatch):
    crops = []

    def fake_read(path, frames, height, width, fit="resize", crop=None):
        crops.append((Path(path).name, crop))
        return np.zeros((frames, height, width, 3), dtype=np.uint8)

    monkeypatch.setattr(encode_script, "read_video_frames", fake_read)
    entry = {"target": "seg/proxy/color.mp4", "proxy": "seg/low/output.mp4", "proxy_crop": [0, 2, 64, 34],
             "proxy_modality": "style", "prompt": "a car"}
    encode_script.encode_entry(entry, _FakeEncoders(), _mixed_args(), tmp_path)
    assert crops == [("color.mp4", None), ("output.mp4", [0, 2, 64, 34])]
    with pytest.raises(KeyError, match="no source"):
        encode_script.encode_entry({**entry, "proxy_modality": "depth"}, _FakeEncoders(), _mixed_args(), tmp_path)
    rows = encode_script.assign_mixed_modalities(
        [{"name": "a", "proxy_depth_video": "d.mp4", "proxy_semantic_video": "s.mp4"}], ("depth", ))
    assert rows[0]["proxy_modality"] == "depth"


def _record_dataset(root: Path, segs: int = 8) -> None:
    root.mkdir(parents=True)
    styles = {"proxy": {"tier": "proxy"}, "lowpoly": {"tier": "low"}, "voxel": {"tier": "low"},
              "biopunk": {"tier": "high"}, "desert": {"tier": "high"}}
    (root / "dataset.json").write_text(json.dumps({"styles": styles}))
    caption = {"duration": 5.167, "compiled": {"rich": {"global": "A car drives forward."}}}
    for index in range(segs):
        seg = root / f"seg_{index:04d}_{'drive' if index % 2 else 'exit'}"
        (seg / "proxy/captions").mkdir(parents=True)
        for name in ("color.mp4", "depth.mp4", "semantic.mp4", "duv.mp4"):
            (seg / "proxy" / name).write_bytes(b"x")
        (seg / "proxy/captions/prompt.json").write_text(json.dumps(caption))
        nodes = [{"id": "proxy", "style": "proxy", "caption": "captions/prompt.json",
                  "files": {"video": "color.mp4"}, "video": {"width": 1280, "height": 720}}]
        for style in ("lowpoly", "voxel", "biopunk", "desert"):
            uid = f"{style}{index}"
            (seg / uid / "captions").mkdir(parents=True)
            (seg / uid / "output.mp4").write_bytes(b"x")
            (seg / uid / "captions/prompt.json").write_text(json.dumps(caption))
            decision = None if (index, style) == (0, "desert") else "pass"
            (seg / uid / "provenance.json").write_text(json.dumps({"evaluation": {"decision": decision}}))
            nodes.append({"id": uid, "style": style, "caption": "captions/prompt.json",
                          "files": {"video": "output.mp4"}, "video": {"width": 1344, "height": 768}})
        (seg / "metadata.json").write_text(json.dumps({
            "split": "val" if index >= segs - 2 else "train",
            "parent": f"{index // 2:04d}",
            "tag": "drive_vehicle" if index % 2 else "exit_vehicle",
            "nodes": nodes,
        }))


def test_the_builder_pairs_both_targets_with_balanced_single_references(builder, tmp_path):
    _record_dataset(tmp_path / "ds")
    targets, skipped = builder.collect_targets(tmp_path / "ds", "train", kinds=("color", "high"), prose="rich",
                                               allow_unscored=False)
    assert skipped["render without VLM pass"] == 1
    assert Counter(target["kind"] for target in targets) == {"color": 6, "high": 11}
    rows = builder.assign(targets, builder.VARIANTS, "seed")
    assert len(rows) == len(targets) == len({row["name"] for row in rows})

    for kind in ("color", "high"):
        counts = Counter(row["proxy_modality"] for row in rows if row["target_kind"] == kind)
        assert max(counts.values()) - min(counts.values()) <= 2, counts

    for row in rows:
        assert row["prompt"].startswith("[0.00s-5.17s] ")
        assert row["cwm_system"] == "w0_omni"
        if row["target_kind"] == "color":
            assert row["target"].endswith("proxy/color.mp4") and "target_crop" not in row
        else:
            assert row["target_crop"] == [0, 6, 1344, 762]
            assert row["target_style"] in {"biopunk", "desert"}
        if row["proxy_modality"] == "style":
            assert row["proxy_style"] in {"lowpoly", "voxel"} and row["proxy_crop"] == [0, 6, 1344, 762]
            assert row["proxy"].split("/")[0] == row["id"], "low-poly reference from the same seg"
        elif row["proxy_modality"] == "duv":
            assert row["proxy_duv_video"] == f"{row['id']}/proxy/duv.mp4"
        else:
            assert row["proxy_depth_video"] == f"{row['id']}/proxy/depth.mp4"
    assert builder.assign(targets, builder.VARIANTS, "seed") == rows, "deterministic"


def test_heldout_cells_draw_from_distinct_segs_of_the_val_split(builder, tmp_path):
    _record_dataset(tmp_path / "ds", segs=40)
    targets, _ = builder.collect_targets(tmp_path / "ds", "val", kinds=("color", "high"), prose="rich",
                                         allow_unscored=False)
    assert {target["seg"] for target in targets} == {"seg_0038_exit", "seg_0039_drive"}

    targets, _ = builder.collect_targets(tmp_path / "ds", "train", kinds=("color", "high"), prose="rich",
                                         allow_unscored=False)
    rows = builder.assign(targets, builder.VARIANTS, "seed")
    picked = builder.pick_heldout(rows, 1, builder.TARGET_KINDS, builder.VARIANTS, "seed")
    assert len(picked) == 8 and len({row["id"] for row in picked}) == 8
    assert Counter((row["target_kind"], row["proxy_modality"]) for row in picked) == {
        (kind, modality): 1 for kind in builder.TARGET_KINDS for modality in builder.VARIANTS
    }
    with pytest.raises(SystemExit, match="only"):
        builder.pick_heldout(rows, 10, builder.TARGET_KINDS, builder.VARIANTS, "seed")
