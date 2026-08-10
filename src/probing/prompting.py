from __future__ import annotations

from typing import Any

import torch

from .adapters import ModelAdapter
from .contracts import ModelRequest, PromptPairInput
from .domain import TokenizedPrompt


def prepare_pair_condition(
    adapter: ModelAdapter,
    *,
    pair: PromptPairInput,
    model: ModelRequest,
    condition: str,
) -> tuple[torch.Tensor, TokenizedPrompt]:
    if condition not in {"original", "perturbed"}:
        raise ValueError(f"unknown prompt-pair condition {condition!r}")
    text = pair.original if condition == "original" else pair.perturbed
    source_messages = (
        pair.original_messages
        if condition == "original"
        else pair.perturbed_messages
    )
    messages: tuple[dict[str, Any], ...] = tuple(
        item.model_dump(mode="json", exclude_none=True) for item in source_messages
    )
    return adapter.prepare_prompt(
        text=text,
        messages=messages,
        tools=pair.tools,
        chat_template=model.chat_template,
        enable_thinking=model.enable_thinking,
    )


__all__ = ["prepare_pair_condition"]
