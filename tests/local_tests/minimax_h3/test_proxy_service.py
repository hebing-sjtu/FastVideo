import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SERVE = Path(__file__).resolve().parents[3] / "scripts" / "h3_proxy" / "serve" / "proxy_service.py"


@pytest.fixture(scope="module")
def service():
    spec = importlib.util.spec_from_file_location("proxy_service", SERVE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["proxy_service"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("frames", "cap", "expected"), [
    (60, 12, (1, 60)),
    (124, 12, (1, 124)),
    (125, 12, (2, 125)),
    (209, 12, (2, 209)),
    (210, 12, (3, 210)),
    (2000, 3, (3, 294)),
])
def test_plan_segments_cover_the_proxy_on_the_take_stride(service, frames, cap, expected):
    assert service.plan_segments(frames, cap) == expected


def test_segment_windows_hold_the_last_frame_past_the_end(service):
    reference = np.arange(130, dtype=np.uint8)[:, None, None, None].repeat(2, 1).repeat(2, 2).repeat(3, 3)
    windows = service.segment_windows(reference, 2)
    assert [w.shape[0] for w in windows] == [124, 124]
    assert windows[1][0, 0, 0, 0] == 85
    assert windows[1][-1, 0, 0, 0] == 129


def test_depth_reference_is_grey_and_flips_a_near_dark_map(service):
    frames = np.zeros((2, 352, 640, 3), dtype=np.uint8)
    frames[..., 0] = 200
    bright = service.to_reference(frames, "depth", near_is_bright=True)
    dark = service.to_reference(frames, "depth", near_is_bright=False)
    assert bright.shape == (2, 352, 640, 3)
    assert (bright[..., 0] == bright[..., 2]).all()
    assert int(bright[0, 0, 0, 0]) + int(dark[0, 0, 0, 0]) == 255


def test_pixel_guess_separates_grey_flat_and_textured_video(service):
    rng = np.random.default_rng(0)
    grey = np.repeat(np.linspace(0, 255, 640, dtype=np.uint8)[None, None, :, None], 352, 1).repeat(3, 3)
    grey = np.repeat(grey, 4, 0)
    flat = np.zeros((4, 352, 640, 3), dtype=np.uint8)
    flat[:, :, 320:] = (255, 0, 128)
    textured = rng.integers(0, 256, size=(4, 352, 640, 3), dtype=np.uint8)
    assert service.pixel_guess(grey)["modality"] == "depth"
    assert service.pixel_guess(flat)["modality"] == "semantic"
    assert service.pixel_guess(textured)["modality"] == "style"


def test_gallery_archives_finished_jobs_with_their_inputs(service, tmp_path):
    import json

    spool, jobs = tmp_path / "spool", tmp_path / "jobs"
    job, out = jobs / "20261021-000000-abc123", spool / "out" / "20261021-000000-abc123"
    for folder in (job, out, spool / "done"):
        folder.mkdir(parents=True)
    frames = np.zeros((6, 64, 96, 3), dtype=np.uint8)
    for name in ("take.mp4", "take_compare.mp4"):
        service.write_mp4(out / name, frames)
    service.write_mp4(job / "proxy_reference.mp4", frames)
    service.Image.fromarray(frames[0]).save(job / "anchor.png")
    (job / "proxy.mp4").write_bytes((out / "take.mp4").read_bytes())
    (job / "job.json").write_text("{}")
    (job / "analysis.json").write_text(json.dumps({"modality": "depth", "segments": 1, "captions": ["[0.00s-5.17s] x"]}))
    (job / "request.json").write_text(json.dumps({"submitter": "binghe", "submitted": 1.0}))
    (spool / "done" / f"{job.name}.json").write_text(json.dumps(
        {"take": str(out / "take.mp4"), "compare": str(out / "take_compare.mp4"), "frames": 124, "seconds": 9.0}))
    (spool / "done" / "failed.json").write_text(json.dumps({"error": "boom"}))

    gallery = service.Gallery(spool, jobs, tmp_path / "local", tmp_path / "remote")
    assert gallery.pending() == ["20261021-000000-abc123", "failed"]
    assert gallery.archive(job.name) is True
    assert gallery.archive("failed") is False
    assert gallery.pending() == []
    [entry] = gallery.entries()
    assert entry["submitter"] == "binghe" and entry["modality"] == "depth" and not entry["oc"]
    assert {"take.mp4", "compare.mp4", "proxy_reference.mp4", "anchor.png", "proxy_input.mp4", "thumb.jpg",
            "meta.json"} <= set(entry["files"])
    assert service.gallery_file(entry, "anchor").endswith("anchor.png")
    assert "binghe" in service.gallery_label(entry)

    assert gallery.card(entry).endswith("thumb.jpg")
    gallery.build_cards()
    card = service.Image.open(gallery.card(entry))
    assert gallery.card(entry).endswith(f"cards/{job.name}.jpg")
    assert card.size == (640, 2 * round(64 * 640 / 48) + 4)
