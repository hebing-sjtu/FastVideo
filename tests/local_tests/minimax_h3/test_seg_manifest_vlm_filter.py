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


def test_the_semantic_code_indexes_u_fastest_like_cwm_does() -> None:
    """``(U[label % 4], V[label // 4])``, not the cartesian product in the other order.

    Both orderings produce twelve distinct pairs, so an injectivity check passes either way. This
    is the assertion that catches the wrong twelve, which is a silent mismatch against everything
    the checkpoint saw in pretraining.
    """
    compose = _load("compose_gta_duv")
    u, v = compose.CWM_SEMANTIC_U, compose.CWM_SEMANTIC_V

    assert compose.cwm_code(0) == (u[0], v[0])
    assert compose.cwm_code(1) == (u[1], v[0])  # u moves first
    assert compose.cwm_code(4) == (u[0], v[1])  # v only after four labels
    assert compose.cwm_code(11) == (u[3], v[2])
    assert len({compose.cwm_code(label) for label in range(12)}) == 12


def test_the_cwm_palette_maps_by_meaning_and_keeps_all_eleven_distinct() -> None:
    """Road has to land on ``road_paved``. Mapping by number would put it on ``vegetation``."""
    compose = _load("compose_gta_duv")
    class_ids = list(range(11))

    legacy = compose.build_palette("abot", class_ids)
    cwm = compose.build_palette("cwm", class_ids)

    assert len(set(legacy.values())) == 6
    assert legacy[0] == legacy[5], "the legacy table's sky and road really are the same symbol"

    assert cwm[5] == compose.cwm_code(4), "GTA road -> CWM road_paved"
    assert cwm[0] == compose.cwm_code(1), "GTA sky -> CWM sky"
    assert cwm[7] == compose.cwm_code(5), "GTA vegetation -> CWM vegetation"
    assert cwm[0] != cwm[5], "sky and road are now separable without leaning on R"
    assert len(set(cwm.values())) == 11, "every label distinct, including ego vs NPC"


def test_ego_and_npc_never_share_a_code_under_either_player_slot() -> None:
    """Telling the camera's own avatar from a passer-by is the point of separating them.

    Which one keeps ``human`` is a judgement call, so both settings are pinned: the default leaves
    ``human`` on the NPCs and puts the camera-locked player on ``void_unknown``, the weakest prior
    available, and the alternative swaps that at the cost of calling NPCs ``animal``.
    """
    compose = _load("compose_gta_duv")
    class_ids = list(range(11))
    human, void, animal = compose.cwm_code(8), compose.cwm_code(0), compose.cwm_code(9)

    default = compose.build_palette("cwm", class_ids)
    assert default[1] == void and default[2] == human

    swapped = compose.build_palette("cwm", class_ids, player_slot="human")
    assert swapped[1] == human and swapped[2] == animal

    for palette in (default, swapped):
        assert palette[1] != palette[2]
        assert len(set(palette.values())) == 11

    try:
        compose.build_palette("cwm", class_ids, player_slot="ped")
    except SystemExit as error:
        assert "player_slot" in str(error)
    else:
        raise AssertionError("an unknown player_slot was accepted")


def test_a_class_the_mapping_has_no_entry_for_is_an_error_not_a_neighbouring_code() -> None:
    """A twelfth GTA label must stop the run. Falling through would silently relabel it."""
    compose = _load("compose_gta_duv")
    try:
        compose.build_palette("cwm", [*range(11), 11])
    except SystemExit as error:
        assert "GTA_TO_CWM" in str(error)
    else:
        raise AssertionError("an unmapped class id was accepted")


def test_the_depth_plane_is_bright_near_and_zero_where_there_is_no_hit() -> None:
    """CWM's polarity and its sentinel, both opposite to ABot's.

    Under ABot the sky was 255 and the near field was 0. Feeding that to a model pretrained on the
    reverse is not a degradation, it is an inversion: every surface reads as sky and the sky reads
    as touching the lens.
    """
    import numpy as np

    compose = _load("compose_gta_duv")
    metres = np.array([[0.0, compose.CWM_NEAR, 1.0, compose.CWM_FAR, 1000.0]], dtype=np.float32)
    red = compose.encode_depth_cwm(metres)

    assert red[0, 0] == 0, "no hit"
    assert red[0, 1] == 255, "near plane is the bright end"
    assert red[0, 3] == 0, "far plane shares the sentinel, as it does in cwm_h3_inference"
    assert red[0, 4] == 0, "beyond far is clipped to far"
    assert 0 < red[0, 2] < 255
    # Monotone decreasing in distance, which is what makes it readable as a disparity field.
    finite = compose.encode_depth_cwm(np.array([[0.3, 1.0, 10.0, 100.0, 256.0]], dtype=np.float32))
    assert list(finite[0]) == sorted(finite[0], reverse=True)


def test_the_decode_range_comes_from_metadata_when_the_seg_declares_one(tmp_path: Path) -> None:
    """A wrong decode range is a monotone relabelling of a depth map, so it looks correct."""
    compose = _load("compose_gta_duv")
    seg = tmp_path / "seg_0000"
    seg.mkdir()

    near, far, origin = compose.read_source_depth_range(seg, 0.1, 256.0)
    assert (near, far) == (0.1, 256.0)
    assert "no metadata.json" in origin

    (seg / "metadata.json").write_text(json.dumps({"proxy": {"depth_near_m": 0.25, "depth_far_m": 512.0}}))
    near, far, origin = compose.read_source_depth_range(seg, 0.1, 256.0)
    assert (near, far) == (0.25, 512.0)
    assert origin == "metadata.json"

    (seg / "metadata.json").write_text(json.dumps({"fps": 24, "frames": 124}))
    near, far, origin = compose.read_source_depth_range(seg, 0.1, 256.0)
    assert (near, far) == (0.1, 256.0)
    assert "declares no depth near/far" in origin
