# SPDX-License-Identifier: Apache-2.0
"""CWM Ref2VA chat wrapper: system role plus the existing Picture/Video user body.

CWM inference runs Qwen through ``apply_chat_template`` with a frozen
``AWM_PROXY_CONTROL`` system prompt and the user caption in the user turn.
FastVideo previously tokenized only the user body. Wrapping here is what makes
a newly encoded ``text_embedding`` consume the same roles CWM does.

The user *body* (``<Picture 1>`` / ``<Video 1>`` labels, vision pads, caption)
is still built by :func:`build_ref2va_presentation`. This module only adds the
chat prefix/suffix as text-tagged tokens so vision pad ids stay in place.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from fastvideo.pipelines.basic.minimax_h3.packing import MINIMAX_H3_TEXT_TAG

_PROMPT_ROOT = Path(__file__).with_name("prompts")
_PROMPT_PATHS = {
    "w0": _PROMPT_ROOT / "system_w0.txt",
    "wn": _PROMPT_ROOT / "system_wn.txt",
}
_PROMPT_SHA256 = {
    "w0": "d488897872a5b190ff8d56b6acc255b66a78f67ae862c12582fcffc8a5dd4ddc",
    "wn": "cd018def9793b4f73cd1c9da9d8ca9ca508dd94a260a19d915aaf9f80b97b6b2",
}

# A marker that cannot appear in the packaged system prompts. The chat template
# is asked to place it as the entire user content; we split on it and splice
# the already-tokenized Ref2VA body in between.
_USER_BODY_SENTINEL = "<<<FASTVIDEO_H3_REF2VA_USER_BODY>>>"

CWM_SYSTEM_PROMPT_KEY = "minimax_h3_cwm_system"


def canonical_caption(value: str) -> str:
    """Match ``cwm_h3_inference.config._canonical_caption`` so CRLF vs LF is not a token drift."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("caption must be a non-empty string")
    normalized = value.strip().replace("\r\n", "\n").replace("\r", "\n")
    return normalized.replace("\n", "\r\n")


def load_cwm_system_prompt(role: str) -> str:
    """Return the packaged CWM system text for ``w0`` or ``wn``.

    Bytes are hashed against the CWM release so a silent edit of the prompt
    files cannot ship a different instruction than inference.
    """
    if role not in _PROMPT_PATHS:
        raise ValueError(f"CWM system role must be 'w0' or 'wn', got {role!r}")
    path = _PROMPT_PATHS[role]
    if not path.is_file():
        raise FileNotFoundError(f"packaged CWM system prompt is missing: {path}")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != _PROMPT_SHA256[role]:
        raise ValueError(f"packaged CWM system prompt does not match this release: {path.name}")
    prompt = payload.decode("utf-8").strip()
    if not prompt:
        raise ValueError(f"packaged CWM system prompt is empty: {path.name}")
    return prompt


def resolve_cwm_system_role(value: object) -> str | None:
    """Normalize a role flag. ``None`` / ``""`` / ``"none"`` disable the wrapper."""
    if value is None:
        return None
    role = str(value).strip().lower()
    if not role or role == "none":
        return None
    if role in {"w0", "wn"}:
        return role
    raise ValueError(f"CWM system role must be 'w0', 'wn', or 'none', got {value!r}")


def wrap_ref2va_chat(
    tokenizer: Any,
    processor: Any,
    system_prompt: str,
    user_token_ids: list[int],
    user_token_tags: list[int],
    *,
    caption: str,
) -> tuple[list[int], list[int]]:
    """Prefix/suffix the Ref2VA user body with Qwen's system+user chat template."""
    if len(user_token_ids) != len(user_token_tags):
        raise ValueError("user token ids and tags must be the same length")
    apply = getattr(processor, "apply_chat_template", None)
    if not callable(apply):
        raise TypeError("Ref2VA CWM wrap requires processor.apply_chat_template")

    formatted = apply(
        [{
            "role": "system",
            "content": system_prompt
        }, {
            "role": "user",
            "content": _USER_BODY_SENTINEL
        }],
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(formatted, str):
        raise TypeError("apply_chat_template(..., tokenize=False) must return a string")
    if formatted.count(system_prompt) != 1:
        raise ValueError("chat template must contain the system prompt exactly once")
    if formatted.count("<|im_start|>system\n") != 1 or formatted.count("<|im_start|>user\n") != 1:
        raise ValueError("chat template must contain exactly one system block and one user block")
    if "<|im_start|>assistant" in formatted:
        raise ValueError("chat template must not add an assistant block")
    if formatted.count(_USER_BODY_SENTINEL) != 1:
        raise ValueError("chat template must keep the user-body sentinel exactly once")
    if len(caption) >= 16 and formatted.count(caption) != 0:
        raise ValueError("caption must not appear in the chat wrapper; it lives in the user body")

    prefix, suffix = formatted.split(_USER_BODY_SENTINEL, 1)
    prefix_ids = _token_ids(tokenizer, prefix)
    suffix_ids = _token_ids(tokenizer, suffix)
    token_ids = prefix_ids + list(user_token_ids) + suffix_ids
    token_tags = ([MINIMAX_H3_TEXT_TAG] * len(prefix_ids) + list(user_token_tags) +
                  [MINIMAX_H3_TEXT_TAG] * len(suffix_ids))
    return token_ids, token_tags


def _token_ids(tokenizer: Any, value: str) -> list[int]:
    if not value:
        return []
    tokenized = tokenizer(value, add_special_tokens=False)
    input_ids = tokenized["input_ids"] if isinstance(tokenized, dict) else tokenized.input_ids
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError("CWM chat wrap tokenization must produce exactly one sequence.")
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


__all__ = [
    "CWM_SYSTEM_PROMPT_KEY",
    "canonical_caption",
    "load_cwm_system_prompt",
    "resolve_cwm_system_role",
    "wrap_ref2va_chat",
]
