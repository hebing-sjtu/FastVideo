# SPDX-License-Identifier: Apache-2.0
"""SolarRenderer-Data subset: CARLA tags, semantic-only rows, per-row modality prompts, manifests."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile

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
def solar():
    return _load("solarrenderer_data")


@pytest.fixture(scope="module")
def encode_script():
    return _load("encode_proxy_samples")


@pytest.fixture(scope="module")
def record():
    return _load("gta_record")


def _write_lossless(path: Path, frames: np.ndarray) -> None:
    import av

    with av.open(str(path), mode="w", format="mp4") as container:
        stream = container.add_stream("libx264rgb", rate=24)
        stream.height, stream.width = frames.shape[1:3]
        stream.pix_fmt = "rgb24"
        stream.options = {"qp": "0", "preset": "ultrafast"}
        for array in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(array), format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _carla_clip(directory: Path, solar, frames: int = 2, height: int = 36, width: int = 64) -> np.ndarray:
    directory.mkdir(parents=True)
    rng = np.random.default_rng(0)
    tags = rng.choice(np.array([1, 3, 11, 12, 14, 24], dtype=np.uint8), size=(frames, height, width))
    coded = np.zeros((frames, height, width, 3), dtype=np.uint8)
    coded[..., 2] = tags
    _write_lossless(directory / "semantic.mp4", coded)
    (directory / "semantic.json").write_text(json.dumps(solar.semantic_json()))
    return tags


def test_carla_tags_map_through_their_own_table(record, solar, tmp_path):
    _carla_clip(tmp_path / "clip", solar)
    assert record.read_semantic_taxonomy(tmp_path / "clip") == "carla"
    lut = record.proxy_class_lut(record.read_semantic_map(tmp_path / "clip"), "carla")
    proxy = record.PROXY_CLASSES
    names = [name for name, _, _ in record.CARLA_TAGS]
    assert proxy[lut[names.index("road")]] == "road_paved"
    assert proxy[lut[names.index("pedestrian")]] == "human"
    assert proxy[lut[names.index("car")]] == "vehicle"
    assert proxy[lut[names.index("ground")]] == "road_paved"
    assert set(record.CARLA_CLASS_TO_PROXY.values()) <= set(proxy)
    # The gta_record table keeps its own reading of the shared name.
    assert record.SOURCE_CLASS_TO_PROXY["ground"] == "infrastructure"
    gta_dir = tmp_path / "gta"
    gta_dir.mkdir()
    (gta_dir / "semantic.json").write_text(json.dumps({"classes": {"0": "static world"}}))
    assert record.read_semantic_taxonomy(gta_dir) == "gta_record"
    (gta_dir / "semantic.json").write_text(json.dumps({"taxonomy": "ade", "classes": {"0": "x"}}))
    with pytest.raises(ValueError, match="unknown taxonomy"):
        record.read_semantic_taxonomy(gta_dir)


def test_lossy_palette_pixels_snap_back_to_their_tags(solar):
    palette = {name: np.array(colour) for name, colour, _ in solar.CARLA_TAGS}
    names = [name for name, _, _ in solar.CARLA_TAGS]
    frame = np.zeros((40, 60, 3), dtype=np.int32)
    frame[:, :30] = palette["road"]
    frame[:, 30:] = palette["sky"]
    truth = np.where(np.arange(60)[None, :] < 30, names.index("road"), names.index("sky")).repeat(40, 0)
    # The road/sky blend sits within a few levels of the wall colour: a third class neither side has.
    frame[:, 30] = (palette["road"] + palette["sky"]) // 2
    frame[10:13, 10:13] = palette["pedestrian"]
    truth[10:13, 10:13] = names.index("pedestrian")
    noisy = np.clip(frame + np.random.default_rng(1).integers(-6, 7, frame.shape), 0, 255).astype(np.uint8)
    labels, distances = solar.tag_lut()
    assert labels[(99 >> 2) << 12 | (97 >> 2) << 6 | (154 >> 2)] == names.index("wall")
    tags, share = solar.palette_to_tags(noisy, labels, distances)
    assert 0.0 < share < 0.1
    assert np.array_equal(np.delete(tags, 30, axis=1), np.delete(truth, 30, axis=1))
    assert set(np.unique(tags[:, 30])) <= {names.index("road"), names.index("sky")}


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
                  anchor_short_edge=64, qwen_video_fps=2.0, modality_prompt="gta-record")
    values.update(overrides)
    return argparse.Namespace(**values)


def _fake_frames(monkeypatch, encode_script):
    monkeypatch.setattr(encode_script, "read_video_frames",
                        lambda path, frames, height, width, fit="resize", crop=None: np.zeros(
                            (frames, height, width, 3), dtype=np.uint8))


def test_a_semantic_video_without_depth_builds_the_semantic_reference(encode_script, record, solar, tmp_path,
                                                                     monkeypatch):
    from fastvideo.pipelines.basic.minimax_h3.proxy import semantic_frame_pixels

    tags = _carla_clip(tmp_path / "carla/c0", solar)
    _fake_frames(monkeypatch, encode_script)
    entry = solar.manifest_row("carla", tmp_path / "carla/c0", "[0.00s-5.17s] cars cross a junction")
    assert "proxy_depth_video" not in entry
    encoders = _FakeEncoders()
    sample = encode_script.encode_entry(entry, encoders, _mixed_args(), tmp_path)
    assert sample["proxy_latents"].shape == (1, 24, 2, 1, 2)
    assert sample["info"]["proxy_modality"] == "semantic"
    assert sample["info"]["proxy_variants"] == ["duv", "depth", "semantic", "style"]
    assert sample["info"]["modality_prompt"] == "solarrenderer"
    assert sample["info"]["target_kind"] == "engine" and sample["info"]["target_style"] == "carla"
    description = encode_script.SOLARRENDERER_MODALITY_PROMPTS["semantic"]
    assert encoders.prompt == f"{description}\n[0.00s-5.17s] cars cross a junction"

    lut = record.proxy_class_lut(record.read_semantic_map(tmp_path / "carla/c0"), "carla")
    planes = encode_script.fit_frames(lut[tags], 16, 32, "center-crop", codes=True, what="test",
                                      code_resize="nearest")
    expected = np.stack([semantic_frame_pixels(ids) for ids in planes])
    assert np.array_equal(encoders.previews[0], (expected * 255).round().astype(np.uint8))


def test_grey_box_rows_are_style_rows_with_their_own_description(encode_script, solar, tmp_path, monkeypatch):
    _fake_frames(monkeypatch, encode_script)
    entry = solar.manifest_row("magpie", tmp_path / "magpie/scene04_player2_000500", "a woman runs")
    assert entry["proxy"] == "magpie/scene04_player2_000500/condition.mp4" and entry["proxy_style"] == "greybox"
    encoders = _FakeEncoders()
    sample = encode_script.encode_entry(entry, encoders, _mixed_args(), tmp_path)
    assert encoders.prompt == f"{encode_script.SOLARRENDERER_MODALITY_PROMPTS['style']}\na woman runs"
    assert sample["info"]["proxy_style"] == "greybox"
    # Without a row override the process-wide description still applies.
    plain = {key: value for key, value in entry.items() if key != "modality_prompt"}
    encode_script.encode_entry(plain, encoders, _mixed_args(), tmp_path)
    assert encoders.prompt.startswith(encode_script.GTA_RECORD_MODALITY_PROMPTS["style"])
    with pytest.raises(ValueError, match="does not describe"):
        encode_script.conditioning_prompt_for(_mixed_args(), ("depth", ), "x", {"modality_prompt": "solarrenderer"})


def test_labelled_rows_may_cover_a_subset_of_the_variants(encode_script):
    variants = ("duv", "depth", "semantic", "style")
    rows = [{"name": "a", "proxy_semantic_video": "s.mp4", "proxy_modality": "semantic"},
            {"name": "b", "proxy": "g.mp4", "proxy_modality": "style"}]
    assert [row["proxy_modality"] for row in encode_script.assign_mixed_modalities(rows, variants)] == [
        "semantic", "style"]
    with pytest.raises(SystemExit, match="No manifest row was assigned"):
        encode_script.assign_mixed_modalities([{"name": "c", "proxy": "g.mp4"}], variants)
    assert encode_script.modality_source_keys({"proxy_semantic_video": "s.mp4"}, "depth") == ()


def test_the_manifest_takes_finished_captioned_clips_of_one_split(solar, tmp_path):
    for source, name, split in (("carla", "cc1", "train"), ("carla", "cc2", "heldout"), ("magpie", "s0_p1_5", "train"),
                                ("magpie", "s0_p1_6", "train"), ("carla", "cc3.part", "train")):
        clip = tmp_path / source / name
        (clip / "captions").mkdir(parents=True)
        (clip / "meta.json").write_text(json.dumps({"split": split}))
        if source == "carla":
            (clip / "semantic.mp4").write_bytes(b"x")
        if name != "s0_p1_6":
            (clip / "captions/prompt.json").write_text(json.dumps({"prompt": "A car turns.", "duration": 5.167}))
    out = tmp_path / "train.jsonl"
    solar.cmd_manifest(argparse.Namespace(root=str(tmp_path), split="train", out=str(out)))
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [row["name"] for row in rows] == ["solar_carla__cc1", "solar_magpie__s0_p1_5"]
    assert all(row["prompt"] == "[0.00s-5.17s] A car turns." for row in rows)
    assert rows[0]["proxy_semantic_video"] == "carla/cc1/semantic.mp4" and rows[0]["proxy_modality"] == "semantic"
    assert rows[1]["proxy"] == "magpie/s0_p1_5/condition.mp4" and rows[1]["proxy_modality"] == "style"


def test_picks_spread_over_ids_and_recordings(solar):
    carla = [{"clip_id": f"carla/cc{index:09d}"} for index in range(100)]
    picked = solar.pick_spread(carla, 10)
    assert [row["clip_id"] for row in picked] == [f"carla/cc{index:09d}" for index in range(5, 100, 10)]
    magpie = [{"clip_id": f"magpie/scene{s:02d}_player{p}_{t:06d}"}
              for s in range(3) for p in range(2) for t in range(10 if s else 30)]
    chosen = solar.pick_round_robin(magpie, 12)
    groups = Counter(solar.magpie_group(row["clip_id"]) for row in chosen)
    assert len(groups) == 6 and set(groups.values()) == {2}
    assert solar.pick_round_robin(magpie, 12) == chosen


class _BytesReader:

    def __init__(self, data: bytes):
        self.data = data
        self.calls = 0

    def read(self, start: int, stop: int) -> bytes:
        self.calls += 1
        return self.data[start:stop]


def test_the_shard_index_finds_every_member_by_header_hops(solar, tmp_path):
    buffer = io.BytesIO()
    payloads = {}
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for key in ("cc0", "cc1", "x" * 120):
            for suffix in solar.MEMBERS:
                data = f"{key}.{suffix}".encode() * (7 if suffix == "render.mp4" else 1)
                if suffix == "json":
                    data = json.dumps({"clip_id": f"carla/{key}"}).encode()
                info = tarfile.TarInfo(f"{key}.{suffix}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
                payloads[(key, suffix)] = data
    reader = _BytesReader(buffer.getvalue())
    members = solar.index_shard(reader, tmp_path / "index.json")
    assert set(members) == {"cc0", "cc1", "x" * 120}
    for (key, suffix), data in payloads.items():
        offset, size = members[key][suffix]
        assert reader.data[offset:offset + size] == data
    assert solar.index_shard(_BytesReader(b""), tmp_path / "index.json") == members, "cached"

    out = tmp_path / "carla/cc1"
    out.parent.mkdir()
    solar.fetch_train_clip(reader, {"clip_id": "carla/cc1", "split": "train", "shard": "s.tar"}, members["cc1"], out)
    assert (out / "render.mp4").read_bytes() == payloads[("cc1", "render.mp4")]
    meta = json.loads((out / "meta.json").read_text())
    assert meta["clip_id"] == "carla/cc1" and meta["split"] == "train" and meta["shard"] == "s.tar"
    assert not (tmp_path / "carla/cc1.part").exists()


def test_captions_are_checked_for_length_and_leaked_inputs(solar):
    good = " ".join(["word"] * 80)
    assert solar.validate_caption(good) == good
    with pytest.raises(ValueError, match="words"):
        solar.validate_caption("too short")
    with pytest.raises(ValueError, match="mentions"):
        solar.validate_caption(good + " shown in the grey box")
    hint = solar.hint_text({"prompt": "summary:\nx\ndetailed_description:\nA man walks.\n\noverall_soundscape:\nrain"})
    assert hint == "A man walks."
