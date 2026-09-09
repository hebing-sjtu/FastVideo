# SPDX-License-Identifier: Apache-2.0
"""CWM system-chat wrap: hashes, CRLF, and chat prefix/suffix around a Ref2VA body."""

from fastvideo.pipelines.basic.minimax_h3.cwm_presentation import (
    canonical_caption,
    load_cwm_system_prompt,
    resolve_cwm_system_role,
    wrap_ref2va_chat,
)
from fastvideo.pipelines.basic.minimax_h3.packing import MINIMAX_H3_TEXT_TAG, MINIMAX_H3_VIDEO_TAG


class _FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) % 97 + 1 for char in text]}


class _FakeProcessor:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        assert add_generation_prompt is False
        system = messages[0]["content"]
        user = messages[1]["content"]
        return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n")


def test_packaged_system_prompts_match_the_cwm_release():
    w0 = load_cwm_system_prompt("w0")
    wn = load_cwm_system_prompt("wn")
    assert w0.startswith("AWM_PROXY_CONTROL.")
    assert wn.startswith("AWM_PROXY_CONTROL.")
    assert "very beginning of the take" in w0
    assert "continues a take already in progress" in wn
    assert w0 != wn


def test_canonical_caption_uses_crlf():
    assert canonical_caption("a\nb") == "a\r\nb"
    assert canonical_caption("a\r\nb") == "a\r\nb"


def test_resolve_cwm_system_role():
    assert resolve_cwm_system_role("w0") == "w0"
    assert resolve_cwm_system_role("WN") == "wn"
    assert resolve_cwm_system_role("none") is None
    assert resolve_cwm_system_role("") is None


def test_wrap_ref2va_chat_keeps_vision_tags_and_adds_text_affixes():
    user_ids = [10, 20, 30]
    user_tags = [MINIMAX_H3_TEXT_TAG, MINIMAX_H3_VIDEO_TAG, MINIMAX_H3_TEXT_TAG]
    caption = "[0.00s-5.17s] A man walks forward along the road."
    token_ids, token_tags = wrap_ref2va_chat(
        _FakeTokenizer(),
        _FakeProcessor(),
        load_cwm_system_prompt("w0"),
        user_ids,
        user_tags,
        caption=caption,
    )
    start = next(index for index in range(len(token_ids) - 2) if token_ids[index:index + 3] == user_ids)
    assert token_tags[start:start + 3] == user_tags
    assert token_tags[0] == MINIMAX_H3_TEXT_TAG
    assert token_tags[-1] == MINIMAX_H3_TEXT_TAG
    assert len(token_ids) > len(user_ids)
    assert len(token_ids) == len(token_tags)
