from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ALGORITHM_VERSION = "perturbation-probing-mvp-v1"
ATTENTION_ALGORITHM_VERSION = "attention-path-probing-mvp-v1"


@dataclass(frozen=True)
class PromptPair:
    original: str | None
    perturbed: str | None
    original_messages: tuple[dict[str, Any], ...] = ()
    perturbed_messages: tuple[dict[str, Any], ...] = ()
    tools: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class ObservableSpec:
    name: str
    target_tokens: tuple[str, ...]
    control_tokens: tuple[str, ...]


@dataclass(frozen=True)
class ProbeSpec:
    model_id: str
    pair: PromptPair
    observable: ObservableSpec
    revision: str | None = None
    chat_template: bool = True
    enable_thinking: bool = False
    capture_position: int = -1
    top_k: int = 500


@dataclass(frozen=True)
class ResolvedToken:
    text: str
    token_id: int
    decoded: str


@dataclass(frozen=True)
class ResolvedObservable:
    name: str
    target: tuple[ResolvedToken, ...]
    control: tuple[ResolvedToken, ...]

    @property
    def target_ids(self) -> tuple[int, ...]:
        return tuple(item.token_id for item in self.target)

    @property
    def control_ids(self) -> tuple[int, ...]:
        return tuple(item.token_id for item in self.control)


@dataclass(frozen=True)
class TokenizedPrompt:
    text: str
    input_ids: tuple[int, ...]
    decoded_tokens: tuple[str, ...]


@dataclass(frozen=True)
class NextTokenPrediction:
    token_id: int
    decoded: str
    logit: float
    probability: float


@dataclass(frozen=True)
class ModelMetadata:
    model_id: str
    resolved_revision: str | None
    model_type: str
    adapter: str
    device: str
    dtype: str
    parameter_count: int
    layer_count: int


@dataclass(frozen=True)
class NeuronScore:
    rank: int
    layer_rank: int
    layer: int
    neuron: int
    coupling: float
    original_activation: float
    perturbed_activation: float
    activation_delta: float
    importance: float


@dataclass(frozen=True)
class LayerSummary:
    layer: int
    signed_sum: float
    absolute_sum: float
    positive_mass: float
    negative_mass: float
    top_10_share: float
    maximum_absolute: float
    top_neuron: int
    activation_delta_norm: float


@dataclass(frozen=True)
class ProbeResult:
    algorithm_version: str
    created_at: str
    elapsed_seconds: float
    spec: ProbeSpec
    model: ModelMetadata
    observable: ResolvedObservable
    original: TokenizedPrompt
    perturbed: TokenizedPrompt
    original_prediction: NextTokenPrediction
    perturbed_prediction: NextTokenPrediction
    original_gap: float
    perturbed_gap: float
    measured_delta: float
    predicted_delta: float
    ffn_skip_original: float | None
    ffn_skip_perturbed: float | None
    ffn_skip_mean: float | None
    circuit_regime: str
    layers: tuple[LayerSummary, ...]
    neurons: tuple[NeuronScore, ...]
    total_neuron_count: int
    warnings: tuple[str, ...] = field(default_factory=tuple)
    runtime: dict[str, Any] = field(default_factory=dict)
