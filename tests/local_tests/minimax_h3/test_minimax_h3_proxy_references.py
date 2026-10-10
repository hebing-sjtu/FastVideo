# SPDX-License-Identifier: Apache-2.0
"""Separate depth / semantic proxy references and the center-crop fit of the proxy encoder."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from fastvideo.pipelines.basic.minimax_h3.cwm_presentation import (
    CWM_SYSTEM_ROLES,
    load_cwm_system_prompt,
    resolve_cwm_system_role,
)
from fastvideo.pipelines.basic.minimax_h3.proxy import (
    PROXY_REFERENCE_KINDS,
    PROXY_VARIANT_KINDS,
    PROXY_SEMANTIC_NUM_CLASSES,
    center_crop_box,
    encode_depth,
    pack_duv_clip,
    proxy_reference_clip,
    semantic_palette,
)

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts/h3_proxy/prepare_data/encode_proxy_samples.py"


@pytest.fixture(scope="module")
def encode_script():
    spec = importlib.util.spec_from_file_location("encode_proxy_samples", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _planes(frames: int, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.5, 300.0, size=(frames, height, width)).astype(np.float32)
    depth[:, 0, 0] = 0.0
    semantic = rng.integers(0, PROXY_SEMANTIC_NUM_CLASSES, size=(frames, height, width)).astype(np.uint8)
    return depth, semantic


def _write_planes(directory: Path, depth: np.ndarray, semantic: np.ndarray) -> None:
    directory.mkdir(parents=True)
    for ordinal, (metres, ids) in enumerate(zip(depth, semantic, strict=True)):
        (directory / f"{ordinal:06d}.depth.f32").write_bytes(metres.astype("<f4").tobytes())
        Image.fromarray(ids, mode="L").save(directory / f"{ordinal:06d}.semantic_id.png")


def test_the_depth_semantic_role_is_hash_locked_and_names_two_videos():
    assert "w0_depth_semantic" in CWM_SYSTEM_ROLES
    assert resolve_cwm_system_role("W0_Depth_Semantic") == "w0_depth_semantic"
    text = load_cwm_system_prompt("w0_depth_semantic")
    assert text.startswith("AWM_PROXY_CONTROL.")
    assert "<Video 1> is the depth video" in text
    assert "<Video 2> is the semantic video" in text
    assert "very beginning of the take" in text


def test_the_mixed_omni_role_is_hash_locked_and_expects_a_modality_label():
    assert resolve_cwm_system_role("W0_OMNI") == "w0_omni"
    text = load_cwm_system_prompt("w0_omni")
    assert text.startswith("AWM_PROXY_CONTROL.")
    assert "modality explicitly named at the start of the user text" in text
    assert "DUV geometry, metric depth" in text


def test_the_semantic_palette_gives_every_class_its_own_interior_colour():
    palette = semantic_palette()
    assert palette.shape == (PROXY_SEMANTIC_NUM_CLASSES, 3)
    assert len({tuple(colour) for colour in palette}) == PROXY_SEMANTIC_NUM_CLASSES
    assert palette.min() > 0 and palette.max() < 255


def test_separate_references_split_the_duv_halves_across_full_channels():
    depth, semantic = _planes(3, 4, 6)
    duv = pack_duv_clip(depth, semantic)
    grey = proxy_reference_clip("depth", depth, semantic)
    colour = proxy_reference_clip("semantic", depth, semantic)
    assert grey.shape == colour.shape == duv.shape == (1, 3, 3, 4, 6)
    for channel in range(3):
        torch.testing.assert_close(grey[0, channel], duv[0, 0])
    torch.testing.assert_close(grey[0, 0, 0], torch.from_numpy(encode_depth(depth[0])))
    expected = semantic_palette()[semantic.astype(np.int64)].astype(np.float32) / 255.0
    torch.testing.assert_close(colour[0].permute(1, 2, 3, 0), torch.from_numpy(expected))
    torch.testing.assert_close(proxy_reference_clip("duv", depth, semantic), duv)
    assert set(PROXY_REFERENCE_KINDS) == {"duv", "depth", "semantic"}
    assert set(PROXY_VARIANT_KINDS) == {"duv", "depth", "semantic", "style"}
    with pytest.raises(ValueError, match="Unknown proxy reference"):
        proxy_reference_clip("normals", depth, semantic)


def test_center_crop_box_centres_and_refuses_to_grow():
    assert center_crop_box((720, 1280), (704, 1280)) == (8, 0)
    assert center_crop_box((704, 1280), (704, 1280)) == (0, 0)
    with pytest.raises(ValueError, match="Cannot center-crop"):
        center_crop_box((704, 1280), (720, 1280))


def test_center_crop_cuts_720p_to_704_and_scales_1080p_first(encode_script):
    rows = np.arange(720, dtype=np.uint8)[None, :, None, None].repeat(1280, axis=2).repeat(3, axis=3)
    cropped = encode_script.fit_frames(rows, 704, 1280, "center-crop", codes=False, what="clip")
    assert cropped.shape == (1, 704, 1280, 3)
    np.testing.assert_array_equal(cropped[0, :, 0, 0], np.arange(8, 712, dtype=np.uint8))
    assert encode_script.cover_size((1080, 1920), (704, 1280)) == (720, 1280)
    full_hd = np.zeros((1, 1080, 1920, 3), dtype=np.uint8)
    assert encode_script.fit_frames(full_hd, 704, 1280, "center-crop", codes=False, what="clip").shape == (1, 704,
                                                                                                          1280, 3)


def test_code_planes_are_cropped_but_never_scaled(encode_script):
    planes = np.zeros((1, 720, 1280), dtype=np.uint8)
    assert encode_script.fit_frames(planes, 704, 1280, "center-crop", codes=True, what="planes").shape == (1, 704,
                                                                                                          1280)
    with pytest.raises(ValueError, match="cannot be scaled"):
        encode_script.fit_frames(np.zeros((1, 1080, 1920), np.uint8), 704, 1280, "center-crop", codes=True, what="p")
    with pytest.raises(ValueError, match="resizing would blend"):
        encode_script.fit_frames(planes, 704, 1280, "resize", codes=True, what="planes")


def test_code_planes_can_crop_then_nearest_downsample_without_blending(encode_script):
    rows = np.arange(36, dtype=np.float32)[None, :, None].repeat(64, axis=2)
    fitted = encode_script.fit_frames(
        rows,
        16,
        32,
        "center-crop",
        codes=True,
        what="depth",
        code_resize="nearest",
    )
    assert fitted.shape == (1, 16, 32)
    assert fitted.dtype == np.float32
    assert set(np.unique(fitted)).issubset(set(np.arange(2, 34, dtype=np.float32)))


def test_native_planes_become_aligned_depth_and_semantic_clips(encode_script, tmp_path):
    depth, semantic = _planes(2, 36, 64)
    _write_planes(tmp_path / "duv", depth, semantic)
    clips = encode_script.read_proxy_reference_clips(tmp_path / "duv", 2, 32, 64, "center-crop",
                                                     ("depth", "semantic"))
    assert [pixels.shape for pixels, _ in clips] == [(1, 3, 2, 32, 64)] * 2
    assert [preview.shape for _, preview in clips] == [(2, 32, 64, 3)] * 2
    torch.testing.assert_close(clips[1][0], proxy_reference_clip("semantic", depth[:, 2:34], semantic[:, 2:34]))
    torch.testing.assert_close(clips[0][0], proxy_reference_clip("depth", depth[:, 2:34], semantic[:, 2:34]))
    legacy_pixels, _ = encode_script.read_duv_clip(tmp_path / "duv", 2, 36, 64)
    torch.testing.assert_close(legacy_pixels, pack_duv_clip(depth, semantic))


def test_the_anchor_is_cropped_to_the_targets_framing(encode_script):
    anchor = Image.new("RGB", (1280, 720))
    assert encode_script.crop_to_aspect(anchor, 704, 1280).size == (1280, 704)
    scaled = encode_script.read_anchor_image(None, np.asarray(anchor)[None], 2048, aspect=(704, 1280))
    assert scaled.size == (3712, 2048)


def test_a_chat_role_must_describe_the_references_it_wraps(encode_script):
    encode_script.check_role_references("w0_depth_semantic", ("depth", "semantic"))
    encode_script.check_role_references("w0", ("duv", ))
    encode_script.check_role_references("none", ("depth", "semantic"))
    with pytest.raises(ValueError, match="describes one proxy video"):
        encode_script.check_role_references("w0", ("depth", "semantic"))
    with pytest.raises(ValueError, match="in that order"):
        encode_script.check_role_references("w0_depth_semantic", ("semantic", "depth"))


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


def test_an_omni_entry_caches_one_latent_per_reference(encode_script, tmp_path, monkeypatch):
    depth, semantic = _planes(2, 36, 64)
    _write_planes(tmp_path / "clip/duv", depth, semantic)
    target = np.zeros((2, 36, 64, 3), dtype=np.uint8)
    monkeypatch.setattr(
        encode_script, "read_video_frames", lambda path, frames, height, width, fit="resize", crop=None: encode_script.fit_frames(
            target, height, width, fit, codes=False, what=str(path)))
    args = argparse.Namespace(num_frames=2,
                              height=32,
                              width=64,
                              proxy_height=32,
                              proxy_width=64,
                              fit="center-crop",
                              proxy_references=("depth", "semantic"),
                              cwm_system="w0_depth_semantic",
                              anchor_short_edge=64,
                              qwen_video_fps=2.0)
    encoders = _FakeEncoders()
    entry = {"target": "clip/rgb.mp4", "proxy_duv": "clip/duv", "prompt": "a road"}
    sample = encode_script.encode_entry(entry, encoders, args, tmp_path)
    assert "proxy_latent" not in sample
    assert sample["proxy_latents"].shape == (2, 24, 2, 2, 4)
    assert sample["vae_latent"].shape == (24, 2, 2, 4)
    assert sample["info"]["proxy_references"] == ["depth", "semantic"]
    assert sample["info"]["fit"] == "center-crop"
    assert sample["info"]["cwm_system"] == "w0_depth_semantic"
    assert len(encoders.previews) == 2
    with pytest.raises(KeyError, match="needs 'proxy_duv'"):
        encode_script.encode_entry({**entry, "proxy_duv": None, "proxy_duv_video": "clip/duv.mp4"}, encoders, args,
                                   tmp_path)
    with pytest.raises(ValueError, match="describes one proxy video"):
        encode_script.encode_entry({**entry, "cwm_system": "w0"}, encoders, args, tmp_path)


def test_legacy_caches_keep_their_keys_and_metadata(encode_script):
    legacy = argparse.Namespace(fit="resize", proxy_references=("duv", ), height=768, width=1344)
    omni = argparse.Namespace(fit="center-crop", proxy_references=("depth", "semantic"), height=704, width=1280)
    assert encode_script.is_legacy(legacy) and not encode_script.is_legacy(omni)
    assert encode_script.reference_info(legacy) == {}
    assert encode_script.reference_info(omni) == {"fit": "center-crop", "proxy_references": ["depth", "semantic"]}
    assert encode_script.anchor_aspect(legacy) is None
    assert encode_script.anchor_aspect(omni) == (704, 1280)


def test_mixed_omni_entry_caches_one_typed_reference(encode_script, tmp_path, monkeypatch):
    depth, semantic = _planes(2, 36, 64)
    _write_planes(tmp_path / "clip/duv", depth, semantic)
    target = np.zeros((2, 36, 64, 3), dtype=np.uint8)
    monkeypatch.setattr(
        encode_script,
        "read_video_frames",
        lambda path, frames, height, width, fit="resize", crop=None: encode_script.fit_frames(
            target, height, width, fit, codes=False, what=str(path)
        ),
    )
    args = argparse.Namespace(
        num_frames=2,
        height=32,
        width=64,
        proxy_height=16,
        proxy_width=32,
        fit="center-crop",
        code_resize="nearest",
        proxy_references=("duv",),
        proxy_variants=("duv", "depth", "semantic", "style"),
        cwm_system="w0_omni",
        anchor_short_edge=64,
        qwen_video_fps=2.0,
    )
    encoders = _FakeEncoders()
    entry = {
        "target": "clip/rgb.mp4",
        "proxy_duv": "clip/duv",
        "proxy_modality": "depth",
        "prompt": "a car turns left",
    }
    sample = encode_script.encode_entry(entry, encoders, args, tmp_path)
    assert "proxy_latent" not in sample
    assert sample["proxy_latents"].shape == (1, 24, 2, 1, 2)
    assert sample["info"]["proxy_modality"] == "depth"
    assert sample["info"]["proxy_references"] == ["depth"]
    assert sample["info"]["proxy_variants"] == ["duv", "depth", "semantic", "style"]
    assert sample["info"]["prompt"] == "a car turns left"
    assert encoders.prompt == "Reference modality: depth video.\na car turns left"
    assert len(encoders.previews) == 1


def test_unlabelled_mixed_rows_are_balanced_without_duplicating_targets(encode_script):
    rows = [
        {"name": f"clip-{index}", "proxy_duv": f"duv/{index}", "proxy": f"style/{index}.mp4"}
        for index in range(5)
    ]
    assigned = encode_script.assign_mixed_modalities(
        rows, ("duv", "depth", "semantic", "style")
    )
    assert len(assigned) == len(rows)
    assert [row["proxy_modality"] for row in assigned] == [
        "duv",
        "depth",
        "semantic",
        "style",
        "duv",
    ]
    assert all("proxy_modality" not in row for row in rows)
