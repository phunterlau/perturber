from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal
from typing import Any

import torch

from ..domain import ModelMetadata, ResolvedObservable, TokenizedPrompt


@dataclass(frozen=True)
class ForwardCapture:
    logits: torch.Tensor
    activations: tuple[torch.Tensor, ...]
    last_layer_input: torch.Tensor
    last_ffn_output: torch.Tensor
    tokenized: TokenizedPrompt


@dataclass(frozen=True)
class LayerResidualCheckpoints:
    """Residual-stream states around one decoder block at one token position."""

    layer: int
    block_input: torch.Tensor
    post_attention: torch.Tensor
    post_ffn: torch.Tensor


@dataclass(frozen=True)
class TrajectoryForwardCapture:
    logits: torch.Tensor
    checkpoints: tuple[LayerResidualCheckpoints, ...]
    tokenized: TokenizedPrompt


@dataclass(frozen=True)
class ActivationEdit:
    layer: int
    neurons: tuple[int, ...]
    operation: Literal["scale", "mix"]
    strength: float
    source_values: torch.Tensor | None = None


@dataclass(frozen=True)
class GeneratedSequence:
    text: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class ResidualEdit:
    layer: int
    direction: torch.Tensor
    beta: float
    normalization: Literal["raw", "residual_norm"] = "residual_norm"


@dataclass(frozen=True)
class AttentionMetadata:
    layer_count: int
    output_head_count: int
    key_value_head_count: int
    head_dim: int

    def key_value_head(self, output_head: int) -> int:
        if not 0 <= output_head < self.output_head_count:
            raise ValueError(f"output head {output_head} is out of range")
        if self.output_head_count % self.key_value_head_count:
            raise ValueError("output heads must be divisible by key/value heads")
        return output_head // (self.output_head_count // self.key_value_head_count)


@dataclass(frozen=True)
class AttentionHeadEdit:
    layer: int
    heads: tuple[int, ...]
    operation: Literal["scale", "mix"]
    strength: float
    # Positions are target-prompt positions. Negative positions use ordinary
    # Python indexing. A source tensor for mixing is [position, head, head_dim],
    # [head, head_dim] for one position, or a full [sequence, all_heads, head_dim].
    positions: tuple[int, ...] = (-1,)
    source_values: torch.Tensor | None = None


@dataclass(frozen=True)
class AttentionForwardCapture:
    logits: torch.Tensor
    # Layer-indexed CPU float32 tensors. Captured entries have shape
    # [sequence, output_head, head_dim]; omitted layers are None.
    head_outputs: tuple[torch.Tensor | None, ...]
    # When requested, [output_head, source_position] at capture_position.
    attention_weights: tuple[torch.Tensor | None, ...]
    # When requested, [source_position, key_value_head, head_dim].
    values: tuple[torch.Tensor | None, ...]
    tokenized: TokenizedPrompt


class ModelAdapter(ABC):
    tokenizer: object

    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata: ...

    @abstractmethod
    def prepare_input(
        self,
        text: str,
        *,
        chat_template: bool,
        enable_thinking: bool,
    ) -> tuple[torch.Tensor, TokenizedPrompt]: ...

    def prepare_chat_input(
        self,
        messages: tuple[dict[str, Any], ...],
        *,
        tools: tuple[dict[str, Any], ...],
        enable_thinking: bool,
    ) -> tuple[torch.Tensor, TokenizedPrompt]:
        if (
            len(messages) == 1
            and messages[0].get("role") == "user"
            and isinstance(messages[0].get("content"), str)
            and not tools
        ):
            return self.prepare_input(
                messages[0]["content"],
                chat_template=True,
                enable_thinking=enable_thinking,
            )
        raise NotImplementedError("the adapter does not support structured chat prompts")

    def prepare_prompt(
        self,
        *,
        text: str | None,
        messages: tuple[dict[str, Any], ...],
        tools: tuple[dict[str, Any], ...],
        chat_template: bool,
        enable_thinking: bool,
    ) -> tuple[torch.Tensor, TokenizedPrompt]:
        if messages:
            if not chat_template:
                raise ValueError("structured messages require chat_template=true")
            return self.prepare_chat_input(
                messages,
                tools=tools,
                enable_thinking=enable_thinking,
            )
        if text is None:
            raise ValueError("text prompt is missing")
        return self.prepare_input(
            text,
            chat_template=chat_template,
            enable_thinking=enable_thinking,
        )

    @abstractmethod
    def forward_capture(
        self,
        input_ids: torch.Tensor,
        tokenized: TokenizedPrompt,
        capture_position: int,
    ) -> ForwardCapture: ...

    @abstractmethod
    def behavioral_direction(self, observable: ResolvedObservable) -> torch.Tensor: ...

    @abstractmethod
    def structural_couplings(
        self,
        direction: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]: ...

    def forward_trajectory_capture(
        self,
        input_ids: torch.Tensor,
        tokenized: TokenizedPrompt,
        capture_position: int,
    ) -> TrajectoryForwardCapture:
        raise NotImplementedError("the adapter does not support trajectory probing")

    def decode_residual(self, residual: torch.Tensor) -> torch.Tensor:
        """Decode one residual vector with the model's native final readout."""

        raise NotImplementedError("the adapter does not support residual decoding")

    def forward_intervened(
        self,
        input_ids: torch.Tensor,
        tokenized: TokenizedPrompt,
        capture_position: int,
        edits: tuple[ActivationEdit, ...],
    ) -> ForwardCapture:
        raise NotImplementedError("the adapter does not support FFN interventions")

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        seed: int,
        edits: tuple[ActivationEdit, ...] = (),
    ) -> GeneratedSequence:
        raise NotImplementedError("the adapter does not support generation")

    def forward_residual_intervened(
        self,
        input_ids: torch.Tensor,
        tokenized: TokenizedPrompt,
        capture_position: int,
        edits: tuple[ResidualEdit, ...],
    ) -> ForwardCapture:
        raise NotImplementedError("the adapter does not support residual intervention")

    def generate_residual_intervened(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        seed: int,
        edits: tuple[ResidualEdit, ...],
    ) -> GeneratedSequence:
        raise NotImplementedError("the adapter does not support residual generation")

    def attention_metadata(self) -> AttentionMetadata:
        raise NotImplementedError("the adapter does not support attention probing")

    def attention_output_couplings(
        self, direction: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Return dF^T W_O blocks as [output_head, head_dim] per layer."""
        raise NotImplementedError("the adapter does not support attention probing")

    def forward_attention_capture(
        self,
        input_ids: torch.Tensor,
        tokenized: TokenizedPrompt,
        capture_position: int,
        *,
        layers: tuple[int, ...] = (),
        edits: tuple[AttentionHeadEdit, ...] = (),
        include_attention_weights: bool = False,
    ) -> AttentionForwardCapture:
        raise NotImplementedError("the adapter does not support attention probing")
