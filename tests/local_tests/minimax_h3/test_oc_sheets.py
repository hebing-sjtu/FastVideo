# SPDX-License-Identifier: Apache-2.0
"""OC identity sheets: which renders have one, what it shows, and the swapped evaluation set."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "h3_proxy" / "prepare_data"
sys.path.insert(0, str(SCRIPTS))

import oc_sheets  # noqa: E402


@pytest.mark.parametrize(
    ("line", "subject"),
    [
        ("<Picture 2> supplies the vehicle_closed identity from the matching view: stone pod.", "vehicle"),
        ("<Picture 2> supplies the hero identity from the matching view: a ranger.", "protagonist"),
        ("<Picture 2> is the protagonist identity sheet (front / left profile / back).", "protagonist"),
        ("<Picture 2> is the ENCLOSED vehicle of this world. A person sits inside.", "vehicle"),
        ("<Picture 2> supplies the lighting.", None),
    ],
)
def test_the_first_picture_2_line_names_the_subject(line: str, subject: str | None) -> None:
    prompt = f"<Video 1> fixes the camera.\n{line}\n<Picture 2>: identity only; a hero vehicle."
    assert oc_sheets.sheet_subject(prompt) == subject


def _render(root: Path, seg: str, uid: str, line: str | None) -> str:
    reference = root / seg / uid / "reference"
    reference.mkdir(parents=True)
    if line is not None:
        (reference / "image_2.png").write_bytes(b"png")
        (reference / "prompt.txt").write_text(f"<Video 1> camera.\n{line}\n", encoding="utf-8")
    return f"{seg}/{uid}/output.mp4"


def test_only_high_rows_with_a_sheet_get_one(tmp_path: Path) -> None:
    hero = "<Picture 2> supplies the hero identity from the matching view."
    rows = [
        {"name": "a", "target_kind": "high", "target": _render(tmp_path, "seg_0", "r1", hero)},
        {"name": "b", "target_kind": "high", "target": _render(tmp_path, "seg_0", "r2", None)},
        {"name": "c", "target_kind": "color", "target": "seg_0/proxy/color.mp4", "oc": "stale"},
    ]
    out = oc_sheets.add_sheets(tmp_path, rows)
    assert out[0]["oc"] == "seg_0/r1/reference/image_2.png"
    assert out[0]["oc_subject"] == "protagonist"
    assert "oc" not in out[1] and "oc" not in out[2]


def test_swapped_sheets_keep_the_subject_and_change_the_style() -> None:
    rows = [
        {"name": "a", "oc": "a.png", "oc_subject": "protagonist", "target_style": "jungle"},
        {"name": "b", "oc": "b.png", "oc_subject": "protagonist", "target_style": "jungle"},
        {"name": "c", "oc": "c.png", "oc_subject": "protagonist", "target_style": "desert"},
        {"name": "d", "oc": "d.png", "oc_subject": "vehicle", "target_style": "desert"},
        {"name": "e", "oc": "e.png", "oc_subject": "vehicle", "target_style": "snow"},
        {"name": "f", "target_style": "proxy"},
    ]
    swapped = {row["name"]: row for row in oc_sheets.swap_sheets(rows)}
    assert swapped["a"]["oc"] == swapped["b"]["oc"] == "c.png"
    assert swapped["c"]["oc"] in {"a.png", "b.png"}
    assert (swapped["d"]["oc"], swapped["e"]["oc"]) == ("e.png", "d.png")
    assert "oc" not in swapped["f"]
    with pytest.raises(SystemExit, match="no vehicle sheet"):
        oc_sheets.swap_sheets(rows[3:4])
