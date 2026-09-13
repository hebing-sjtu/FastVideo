# SPDX-License-Identifier: Apache-2.0
"""The seg manifest builders' corpus gate: VLM scores, contract prose, semantic codes.

These three decide *which clips* and *what text* reach the encoder, and all three fail silently
when they are wrong -- a mis-joined score table trains on clips the judge rejected, a prompt that
fell through to the teacher instruction describes a task the pipeline does not pack, and a
collapsing semantic palette makes road and sky the same symbol. None of that raises.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "h3_proxy" / "prepare_data"


def _load(name: str) -> types.ModuleType:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before exec because ``@dataclass(slots=True)`` rebuilds the class and resolves its
    # annotations through ``sys.modules[cls.__module__]``.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    defaults = {"vlm_filter": "auto", "min_vlm_score": 0.7, "vlm_metrics": "preservation,camera_trajectory"}
    return types.SimpleNamespace(**(defaults | overrides))


def test_a_clip_without_a_score_is_rejected_rather_than_waved_through(tmp_path: Path) -> None:
    """"Not scored" is not "passed": the caller asked for clips the judge approved."""
    vlm_filter = _load("vlm_filter")
    (tmp_path / "vlm_filter.json").write_text(json.dumps({"seg_0000": {"preservation": 1.0, "camera_trajectory": 1.0}}))

    table = vlm_filter.load_vlm_filter(tmp_path, _args())
    assert table is not None
    assert table.verdict("seg_0000") == (True, "")
    accepted, why = table.verdict("seg_0001")
    assert not accepted
    assert "no vlm_filter.json record" in why


def test_a_missing_metric_cannot_pass_by_omission(tmp_path: Path) -> None:
    """One unscored rubric on an otherwise perfect clip still rejects it: the four are an AND."""
    vlm_filter = _load("vlm_filter")
    (tmp_path / "vlm_filter.json").write_text(
        json.dumps({
            "seg_0000": {
                "preservation": 1.0,
                "camera_trajectory": 1.0
            },
            "seg_0001": {
                "preservation": 1.0
            },
        }))

    table = vlm_filter.load_vlm_filter(tmp_path, _args())
    assert table is not None
    assert table.verdict("seg_0000")[0]
    accepted, why = table.verdict("seg_0001")
    assert not accepted
    assert "camera_trajectory" in why


def test_a_metric_absent_from_the_whole_table_is_a_typo_not_a_rejection(tmp_path: Path) -> None:
    """Rejecting every clip would look like a strict corpus; a misspelled rubric has to be loud."""
    vlm_filter = _load("vlm_filter")
    (tmp_path / "vlm_filter.json").write_text(json.dumps({"seg_0000": {"preservation": 1.0}}))

    try:
        vlm_filter.load_vlm_filter(tmp_path, _args(vlm_metrics="preservation,camera_trajectry"))
    except SystemExit as error:
        assert "camera_trajectry" in str(error)
        assert "preservation" in str(error)
    else:
        raise AssertionError("a metric name absent from every record should not be accepted")


def test_contract_v3_prose_is_window_stamped_and_prefers_the_rich_compilation(tmp_path: Path) -> None:
    """The stamp is what the manifest audit uses to tell a window caption from an episode summary.

    ``rich`` is the CWM-shaped variant; ``lean`` compresses the scene to a few words. Training on
    ``lean`` while the reference captions are ``rich`` is a distribution shift nothing reports.
    """
    builder = _load("seg_dir_to_encode_manifest")
    payload = {
        "contract": "scene",
        "version": 3,
        "duration": 5.167,
        "compiled": {
            "lean": {
                "global": "Video game. City street."
            },
            "rich": {
                "global": "Third-person action video game. Broad urban boulevard."
            },
        },
    }

    rich = builder.extract_prompt_text(payload, prose_style="rich")
    lean = builder.extract_prompt_text(payload, prose_style="lean")
    assert rich == "[0.00s-5.17s] Third-person action video game. Broad urban boulevard."
    assert lean == "[0.00s-5.17s] Video game. City street."
    assert builder.WINDOW_MARKER_PATTERN.match(rich)
    # Stamping twice would move the window, so an already-stamped caption is left alone.
    assert builder.extract_prompt_text({"prompt": rich, "duration": 5.167}) == rich


def test_the_teacher_edit_instruction_needs_an_explicit_opt_in(tmp_path: Path) -> None:
    builder = _load("seg_dir_to_encode_manifest")
    seg = tmp_path / "seg_0000"
    (seg / "minimax_h3").mkdir(parents=True)
    (seg / "minimax_h3" / "prompt.txt").write_text("<Video 1> is the source video for the target video edit")

    assert builder.read_seg_prompt(seg) == ("", "none")
    text, source = builder.read_seg_prompt(seg, allow_teacher=True)
    assert source == "minimax_h3/prompt.txt"
    assert text.startswith("<Video 1>")


def test_the_cwm_palette_gives_every_gta_class_its_own_code() -> None:
    """``standard11`` collapses eleven labels onto six codes; ``cwm12`` does not collapse at all."""
    compose = _load("compose_gta_duv")
    class_ids = list(range(11))

    legacy = compose.build_palette("standard11", class_ids)
    injective = compose.build_palette("cwm12", class_ids)

    assert len(set(legacy.values())) == 6
    assert legacy[0] == legacy[5]  # sky and road are the same symbol
    assert len(set(injective.values())) == len(class_ids)
    assert set(injective.values()) <= {(u, v) for u in compose.CWM_SEMANTIC_U for v in compose.CWM_SEMANTIC_V}
