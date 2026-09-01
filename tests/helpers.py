from __future__ import annotations

import torch

from probing.adapters.base import (
    ActivationEdit,
    AttentionForwardCapture,
    AttentionHeadEdit,
    AttentionMetadata,
    ForwardCapture,
    GeneratedSequence,
    LayerResidualCheckpoints,
    ModelAdapter,
    ResidualEdit,
    ResidualGradientCapture,
    TrajectoryForwardCapture,
)
from probing.domain import (
    ModelMetadata,
    ProbeSpec,
    PromptPair,
    ObservableSpec,
    ResolvedObservable,
    TokenizedPrompt,
)
from probing.engine import ProbeEngine


class FakeTokenizer:
    mapping = {"No": 0, "Yes": 1, "Maybe": 2, "multi": [2, 3]}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        value = self.mapping[text]
        return value if isinstance(value, list) else [value]

    def decode(self, token_ids: int | list[int]) -> str:
        token_id = token_ids if isinstance(token_ids, int) else token_ids[0]
        return {0: "No", 1: "Yes", 2: "Maybe", 3: "token"}[token_id]


class FakeAdapter(ModelAdapter):
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.forward_calls = 0

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id="fake/qwen3",
            resolved_revision="fixture",
            model_type="qwen3",
            adapter="fake",
            device="cpu",
            dtype="float32",
            parameter_count=10,
            layer_count=1,
        )

    def prepare_input(
        self,
        text: str,
        *,
        chat_template: bool,
        enable_thinking: bool,
    ) -> tuple[torch.Tensor, TokenizedPrompt]:
        wrong_markers = ("London", "equals five", "50 degrees Celsius")
        value = 1 if any(marker in text for marker in wrong_markers) else 0
        tokenized = TokenizedPrompt(
            text=text,
            input_ids=(value,),
            decoded_tokens=(str(value),),
        )
        return torch.tensor([[value]]), tokenized

    def forward_capture(
        self,
        input_ids: torch.Tensor,
        tokenized: TokenizedPrompt,
        capture_position: int,
    ) -> ForwardCapture:
        self.forward_calls += 1
        perturbed = bool(int(input_ids[0, 0]))
        return ForwardCapture(
            logits=(
                torch.tensor([3.0, 0.0, -1.0])
                if perturbed
                else torch.tensor([0.0, 2.0, -1.0])
            ),
            activations=(
                torch.tensor([3.0, 1.0])
                if perturbed
                else torch.tensor([1.0, 2.0]),
            ),
            last_layer_input=torch.tensor([2.0, 0.0])
            if perturbed
            else torch.tensor([4.0, 0.0]),
            last_ffn_output=torch.tensor([1.0, 0.0])
            if perturbed
            else torch.tensor([2.0, 0.0]),
            tokenized=tokenized,
        )

    def behavioral_direction(self, observable: ResolvedObservable) -> torch.Tensor:
        return torch.tensor([1.0, 0.0])

    def structural_couplings(
        self,
        direction: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return (torch.tensor([2.0, -1.0]),)

    def forward_trajectory_capture(
        self,
        input_ids: torch.Tensor,
        tokenized: TokenizedPrompt,
        capture_position: int,
    ) -> TrajectoryForwardCapture:
        baseline = self.forward_capture(input_ids, tokenized, capture_position)
        perturbed = bool(int(input_ids[0, 0]))
        block_input = (
            torch.tensor([1.8, 0.2, -1.0])
            if perturbed
            else torch.tensor([0.3, 1.4, -1.0])
        )
        post_attention = (
            torch.tensor([2.4, 0.1, -1.0])
            if perturbed
            else torch.tensor([0.2, 1.8, -1.0])
        )
        return TrajectoryForwardCapture(
            logits=baseline.logits,
            checkpoints=(
                LayerResidualCheckpoints(
                    layer=0,
                    block_input=block_input,
                    post_attention=post_attention,
                    post_ffn=baseline.logits,
                ),
            ),
            tokenized=tokenized,
        )

    def decode_residual(self, residual: torch.Tensor) -> torch.Tensor:
        return residual.detach().float().cpu()

    def forward_residual_gradients(
        self,
        input_ids: torch.Tensor,
        tokenized: TokenizedPrompt,
        capture_position: int,
        observable: ResolvedObservable,
    ) -> ResidualGradientCapture:
        baseline = self.forward_capture(input_ids, tokenized, capture_position)
        perturbed = bool(int(input_ids[0, 0]))
        return ResidualGradientCapture(
            logits=baseline.logits,
            residuals=(torch.tensor([1.0, 0.5]) if perturbed else torch.tensor([0.5, 1.0]),),
            gradients=(torch.tensor([1.0, 0.25]) if perturbed else torch.tensor([0.5, 0.5]),),
            tokenized=tokenized,
        )

    def native_residual_gradient(
        self,
        residual: torch.Tensor,
        observable: ResolvedObservable,
    ) -> torch.Tensor:
        return torch.tensor([0.75, 0.25])

    def layer_couplings(
        self,
        directions: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        assert len(directions) == 1
        value = float(directions[0][0].item())
        return (torch.tensor([2.0 * value, -1.0 * value]),)

    @staticmethod
    def _edited_activation(
        activation: torch.Tensor, edits: tuple[ActivationEdit, ...]
    ) -> torch.Tensor:
        result = activation.clone()
        for edit in edits:
            assert edit.layer == 0
            indices = torch.tensor(edit.neurons, dtype=torch.long)
            current = result.index_select(0, indices)
            if edit.operation == "scale":
                replacement = current * edit.strength
            else:
                assert edit.source_values is not None
                source = edit.source_values.flatten()
                if source.numel() != len(edit.neurons):
                    source = source.index_select(0, indices)
                replacement = current * (1 - edit.strength) + source * edit.strength
            result.index_copy_(0, indices, replacement)
        return result

    def forward_intervened(
        self,
        input_ids: torch.Tensor,
        tokenized: TokenizedPrompt,
        capture_position: int,
        edits: tuple[ActivationEdit, ...],
    ) -> ForwardCapture:
        baseline = self.forward_capture(input_ids, tokenized, capture_position)
        edited = self._edited_activation(baseline.activations[0], edits)
        gap_effect = float(torch.dot(torch.tensor([2.0, -1.0]), edited - baseline.activations[0]).item())
        logits = baseline.logits.clone()
        logits[0] += gap_effect
        return ForwardCapture(
            logits=logits,
            activations=(edited,),
            last_layer_input=baseline.last_layer_input,
            last_ffn_output=baseline.last_ffn_output,
            tokenized=tokenized,
        )

    def forward_trajectory_intervened(
        self,
        input_ids: torch.Tensor,
        tokenized: TokenizedPrompt,
        capture_position: int,
        edits: tuple[ActivationEdit, ...],
    ) -> TrajectoryForwardCapture:
        edited = self.forward_intervened(
            input_ids, tokenized, capture_position, edits
        )
        baseline = self.forward_trajectory_capture(
            input_ids, tokenized, capture_position
        )
        return TrajectoryForwardCapture(
            logits=edited.logits,
            checkpoints=tuple(
                LayerResidualCheckpoints(
                    layer=item.layer,
                    block_input=item.block_input,
                    post_attention=item.post_attention,
                    post_ffn=edited.logits,
                )
                for item in baseline.checkpoints
            ),
            tokenized=tokenized,
        )

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
        tokenized = TokenizedPrompt(
            text="fixture",
            input_ids=tuple(int(value) for value in input_ids[0].tolist()),
            decoded_tokens=(),
        )
        capture = (
            self.forward_intervened(input_ids, tokenized, -1, edits)
            if edits
            else self.forward_capture(input_ids, tokenized, -1)
        )
        token_id = int(torch.argmax(capture.logits).item())
        return GeneratedSequence(
            text=self.tokenizer.decode([token_id]),
            token_ids=(token_id,),
        )

    def forward_residual_intervened(
        self,
        input_ids: torch.Tensor,
        tokenized: TokenizedPrompt,
        capture_position: int,
        edits: tuple[ResidualEdit, ...],
    ) -> ForwardCapture:
        baseline = self.forward_capture(input_ids, tokenized, capture_position)
        effect = sum(float(edit.direction[0].item()) * edit.beta * 2 for edit in edits)
        logits = baseline.logits.clone()
        logits[0] += effect
        return ForwardCapture(
            logits=logits,
            activations=baseline.activations,
            last_layer_input=baseline.last_layer_input,
            last_ffn_output=baseline.last_ffn_output,
            tokenized=tokenized,
        )

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
        tokenized = TokenizedPrompt(
            text="fixture",
            input_ids=tuple(int(value) for value in input_ids[0].tolist()),
            decoded_tokens=(),
        )
        capture = self.forward_residual_intervened(
            input_ids, tokenized, -1, edits
        )
        token_id = int(torch.argmax(capture.logits).item())
        return GeneratedSequence(
            text=self.tokenizer.decode([token_id]), token_ids=(token_id,)
        )


class FakeAttentionAdapter(FakeAdapter):
    """Two-layer deterministic attention circuit used for workflow tests."""

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id="fake/qwen3",
            resolved_revision="fixture",
            model_type="qwen3",
            adapter="fake-attention",
            device="cpu",
            dtype="float32",
            parameter_count=20,
            layer_count=2,
        )

    def attention_metadata(self) -> AttentionMetadata:
        return AttentionMetadata(
            layer_count=2,
            output_head_count=2,
            key_value_head_count=2,
            head_dim=1,
        )

    def attention_output_couplings(
        self, direction: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        return (
            torch.tensor([[2.0], [-1.0]]),
            torch.tensor([[1.5], [-0.5]]),
        )

    @staticmethod
    def _apply_head_edits(
        values: torch.Tensor,
        edits: tuple[AttentionHeadEdit, ...],
        layer: int,
    ) -> torch.Tensor:
        result = values.clone()
        for edit in edits:
            if edit.layer != layer:
                continue
            for source_index, raw_position in enumerate(edit.positions):
                position = raw_position if raw_position >= 0 else len(result) + raw_position
                indices = torch.tensor(edit.heads, dtype=torch.long)
                current = result[position].index_select(0, indices)
                if edit.operation == "scale":
                    replacement = current * edit.strength
                else:
                    assert edit.source_values is not None
                    source = edit.source_values
                    if source.ndim == 2:
                        source = source.unsqueeze(0)
                    replacement = (
                        current * (1 - edit.strength)
                        + source[source_index] * edit.strength
                    )
                result[position].index_copy_(0, indices, replacement)
        return result

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
        self.forward_calls += 1
        perturbed = bool(int(input_ids[0, 0]))
        base_logits = (
            torch.tensor([3.0, 0.0, -1.0])
            if perturbed
            else torch.tensor([0.0, 2.0, -1.0])
        )
        base0 = (
            torch.tensor([[[3.0], [1.0]]])
            if perturbed
            else torch.tensor([[[1.0], [2.0]]])
        )
        base1 = (
            torch.tensor([[[4.0], [0.0]]])
            if perturbed
            else torch.tensor([[[2.0], [1.0]]])
        )
        layer0 = self._apply_head_edits(base0, edits, 0)
        delta0 = layer0 - base0
        propagated = torch.empty_like(base1)
        propagated[:, 0, :] = 0.5 * delta0[:, 0, :] + 0.25 * delta0[:, 1, :]
        propagated[:, 1, :] = -0.25 * delta0[:, 0, :] + 0.5 * delta0[:, 1, :]
        layer1_before_edit = base1 + propagated
        layer1 = self._apply_head_edits(layer1_before_edit, edits, 1)
        couplings = self.attention_output_couplings(torch.tensor([1.0]))
        gap_effect = float(
            (couplings[0] * (layer0[-1] - base0[-1])).sum().item()
            + (couplings[1] * (layer1[-1] - base1[-1])).sum().item()
        )
        logits = base_logits.clone()
        logits[0] += gap_effect
        selected = set(layers or (0, 1))
        outputs = tuple(
            value if index in selected else None
            for index, value in enumerate((layer0, layer1))
        )
        if include_attention_weights:
            weights = tuple(
                torch.ones((2, 1)) if index in selected else None
                for index in range(2)
            )
            value_tensors = tuple(
                value if index in selected else None
                for index, value in enumerate((layer0, layer1))
            )
        else:
            weights = (None, None)
            value_tensors = (None, None)
        return AttentionForwardCapture(
            logits=logits,
            head_outputs=outputs,
            attention_weights=weights,
            values=value_tensors,
            tokenized=tokenized,
        )


def fake_spec() -> ProbeSpec:
    return ProbeSpec(
        model_id="fake/qwen3",
        pair=PromptPair(
            original="The capital of France is Paris",
            perturbed="The capital of France is London",
        ),
        observable=ObservableSpec(
            name="agreement",
            target_tokens=("No",),
            control_tokens=("Yes",),
        ),
        top_k=2,
    )


def fake_result():
    return ProbeEngine(FakeAdapter()).analyze(fake_spec())
