from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    field_validator,
    model_validator,
)


RANK_SCHEMA_VERSION = "probe.rank/v1"
QUALIFICATION_SCHEMA_VERSION = "probe.qualify/v1"
INTERVENTION_SCHEMA_VERSION = "probe.intervention/v1"
DIRECTION_SCHEMA_VERSION = "probe.direction/v1"
ATTENTION_RANK_SCHEMA_VERSION = "probe.attention-rank/v1"
ATTENTION_INTERVENTION_SCHEMA_VERSION = "probe.attention-intervention/v1"
ATTENTION_TRACE_SCHEMA_VERSION = "probe.attention-trace/v1"
TRAJECTORY_SCHEMA_VERSION = "probe.trajectory/v1"
FFN_COUPLING_SCHEMA_VERSION = "probe.ffn-coupling/v1"
EVENT_SCHEMA_VERSION = "probe.event/v1"
MANIFEST_SCHEMA_VERSION = "probe.run/v1"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ModelRequest(StrictModel):
    id: str = "Qwen/Qwen3-0.6B"
    revision: str | None = None
    adapter: Literal["auto", "qwen3"] = "auto"
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    chat_template: bool = True
    enable_thinking: bool = False

    @field_validator("id")
    @classmethod
    def validate_model_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model id must not be blank")
        return value


class ChatMessage(StrictModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[dict[str, Any], ...] = ()

    @model_validator(mode="after")
    def validate_message_payload(self) -> "ChatMessage":
        if self.content is None and not self.tool_calls:
            raise ValueError("chat message requires content or tool_calls")
        if self.content is not None and not self.content.strip() and not self.tool_calls:
            raise ValueError("chat message content must not be blank")
        return self


class PromptPairInput(StrictModel):
    id: str
    original: str | None = None
    perturbed: str | None = None
    original_messages: tuple[ChatMessage, ...] = ()
    perturbed_messages: tuple[ChatMessage, ...] = ()
    tools: tuple[dict[str, Any], ...] = ()
    split: Literal["discovery", "validation", "heldout"] = "discovery"
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @field_validator("id", "original", "perturbed")
    @classmethod
    def validate_nonblank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("pair id and prompts must not be blank")
        return value

    @model_validator(mode="after")
    def validate_prompt_forms(self) -> "PromptPairInput":
        for condition, text, messages in (
            ("original", self.original, self.original_messages),
            ("perturbed", self.perturbed, self.perturbed_messages),
        ):
            if (text is None) == (not messages):
                raise ValueError(
                    f"{condition} must provide exactly one of text or structured messages"
                )
        if self.tools and not (self.original_messages and self.perturbed_messages):
            raise ValueError("tool definitions require structured messages for both conditions")
        return self


class ExperimentSet(StrictModel):
    schema_version: Literal["probe.experiment-set/v1"] = "probe.experiment-set/v1"
    name: str
    description: str | None = None
    pairs: tuple[PromptPairInput, ...] = Field(min_length=1)
    tags: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_experiment_set(self) -> "ExperimentSet":
        if not self.name.strip():
            raise ValueError("experiment set name must not be blank")
        ids = [pair.id for pair in self.pairs]
        if len(ids) != len(set(ids)):
            raise ValueError("experiment set pair IDs must be unique")
        return self


class PerturbationCase(StrictModel):
    id: str
    values: dict[str, str | int | float | bool]
    split: Literal["discovery", "validation", "heldout"] = "discovery"
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class PerturbationTemplate(StrictModel):
    schema_version: Literal["probe.perturbation-template/v1"] = (
        "probe.perturbation-template/v1"
    )
    name: str
    target_factor: str
    original_template: str
    perturbed_template: str
    cases: tuple[PerturbationCase, ...] = Field(min_length=1)
    tags: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_template(self) -> "PerturbationTemplate":
        if not self.name.strip() or not self.target_factor.strip():
            raise ValueError("perturbation template name and target factor must not be blank")
        if not self.original_template.strip() or not self.perturbed_template.strip():
            raise ValueError("perturbation prompt templates must not be blank")
        ids = [item.id for item in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("perturbation case IDs must be unique")
        return self


class PerturbationDiagnostic(StrictModel):
    pair_id: str
    changed_word_count: int = Field(ge=0)
    shared_word_fraction: float = Field(ge=0, le=1)
    length_ratio: float = Field(gt=0)
    warnings: tuple[str, ...] = ()


class PerturbationCompilation(StrictModel):
    schema_version: Literal["probe.perturbation-compilation/v1"] = (
        "probe.perturbation-compilation/v1"
    )
    target_factor: str
    experiment_set: ExperimentSet
    diagnostics: tuple[PerturbationDiagnostic, ...]
    warnings: tuple[str, ...] = ()


class ObservableRequest(StrictModel):
    name: str
    target_tokens: tuple[str, ...] = Field(min_length=1)
    control_tokens: tuple[str, ...] = Field(min_length=1)
    reduction: Literal["mean_logit_gap"] = "mean_logit_gap"
    decision_position: Literal[0] = 0

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("observable name must not be blank")
        return value

    @field_validator("target_tokens", "control_tokens")
    @classmethod
    def validate_token_strings(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(value == "" for value in values):
            raise ValueError("observable token strings must not be empty")
        return values


class CaptureRequest(StrictModel):
    activation: Literal["post_swiglu"] = "post_swiglu"
    # The v1 coupling is defined for the last prompt position feeding the first
    # generated-token decision. Earlier prompt positions need a propagation
    # model through later-token attention and are therefore capability-gated out.
    position: Literal[-1] = -1
    layers: Literal["all"] = "all"


class RankingRequest(StrictModel):
    top_k: PositiveInt = 500
    select_by: Literal["absolute_importance"] = "absolute_importance"
    pair_aggregation: Literal["single_pair", "rms"] = "single_pair"


class ExecutionLimits(StrictModel):
    max_forward_passes: PositiveInt
    max_wall_seconds: float | None = Field(default=None, gt=0)
    max_artifact_bytes: PositiveInt
    allow_download: bool = False
    max_download_bytes: PositiveInt | None = None
    trust_remote_code: bool = False
    seed: int = 0

    @model_validator(mode="after")
    def validate_download_budget(self) -> "ExecutionLimits":
        if self.allow_download and self.max_download_bytes is None:
            raise ValueError(
                "max_download_bytes is required when allow_download is true"
            )
        return self


class RankSpec(StrictModel):
    schema_version: Literal["probe.rank/v1"] = RANK_SCHEMA_VERSION
    kind: Literal["rank"] = "rank"
    name: str = "unnamed-ranking"
    description: str | None = None
    model: ModelRequest = Field(default_factory=ModelRequest)
    pairs: tuple[PromptPairInput, ...] = Field(min_length=1)
    observable: ObservableRequest
    capture: CaptureRequest = Field(default_factory=CaptureRequest)
    ranking: RankingRequest = Field(default_factory=RankingRequest)
    execution: ExecutionLimits
    tags: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_pairs_and_aggregation(self) -> "RankSpec":
        pair_ids = [pair.id for pair in self.pairs]
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError("pair IDs must be unique")
        if len(self.pairs) == 1 and self.ranking.pair_aggregation != "single_pair":
            raise ValueError("one pair must use pair_aggregation='single_pair'")
        if len(self.pairs) > 1 and self.ranking.pair_aggregation != "rms":
            raise ValueError("multiple pairs must use pair_aggregation='rms'")
        return self


class TrajectorySpec(StrictModel):
    schema_version: Literal["probe.trajectory/v1"] = TRAJECTORY_SCHEMA_VERSION
    kind: Literal["trajectory"] = "trajectory"
    name: str = "unnamed-trajectory"
    description: str | None = None
    parent_run_id: str
    pair_ids: tuple[str, ...] = ()
    position: Literal[-1] = -1
    checkpoints: tuple[Literal["block_input", "post_attention", "post_ffn"], ...] = (
        "block_input",
        "post_attention",
        "post_ffn",
    )
    top_k: PositiveInt = Field(default=10, le=100)
    transition_limit: PositiveInt = Field(default=5, le=50)
    execution: ExecutionLimits
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("name", "parent_run_id")
    @classmethod
    def validate_trajectory_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("trajectory name and parent run ID must not be blank")
        return value

    @model_validator(mode="after")
    def validate_trajectory(self) -> "TrajectorySpec":
        if not self.checkpoints:
            raise ValueError("trajectory requires at least one checkpoint")
        if len(self.checkpoints) != len(set(self.checkpoints)):
            raise ValueError("trajectory checkpoints must be unique")
        if len(self.pair_ids) != len(set(self.pair_ids)):
            raise ValueError("trajectory pair IDs must be unique")
        return self


class FFNCouplingSpec(StrictModel):
    schema_version: Literal["probe.ffn-coupling/v1"] = FFN_COUPLING_SCHEMA_VERSION
    kind: Literal["ffn_coupling"] = "ffn_coupling"
    name: str = "unnamed-ffn-coupling"
    description: str | None = None
    parent_run_id: str
    trajectory_run_id: str | None = None
    pair_ids: tuple[str, ...] = ()
    layers: tuple[int, ...] = ()
    position: Literal[-1] = -1
    methods: tuple[
        Literal["native_local_readout", "downstream_endpoint_gradient"], ...
    ] = ("native_local_readout", "downstream_endpoint_gradient")
    top_k: PositiveInt = Field(default=500, le=10_000)
    max_backward_passes: PositiveInt
    execution: ExecutionLimits
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("name", "parent_run_id")
    @classmethod
    def validate_ffn_coupling_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("FFN coupling name and parent run ID must not be blank")
        return value

    @model_validator(mode="after")
    def validate_ffn_coupling(self) -> "FFNCouplingSpec":
        if not self.methods:
            raise ValueError("FFN coupling requires at least one method")
        if len(self.methods) != len(set(self.methods)):
            raise ValueError("FFN coupling methods must be unique")
        if "downstream_endpoint_gradient" not in self.methods:
            raise ValueError(
                "FFN coupling v1 requires downstream_endpoint_gradient"
            )
        if len(self.layers) != len(set(self.layers)) or any(
            layer < 0 for layer in self.layers
        ):
            raise ValueError("FFN coupling layers must be unique and non-negative")
        if len(self.pair_ids) != len(set(self.pair_ids)):
            raise ValueError("FFN coupling pair IDs must be unique")
        return self


class GenerationRequest(StrictModel):
    max_new_tokens: PositiveInt = 32
    do_sample: bool = False
    temperature: float = Field(default=1.0, gt=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    seed: int = 0


class BehaviorEvaluatorRequest(StrictModel):
    kind: Literal[
        "token_set", "contains", "regex", "exact", "unicode_script"
    ] = "token_set"
    target_values: tuple[str, ...] = ()
    control_values: tuple[str, ...] = ()
    case_sensitive: bool = False

    @model_validator(mode="after")
    def validate_values(self) -> "BehaviorEvaluatorRequest":
        if self.kind != "token_set" and not (
            self.target_values and self.control_values
        ):
            raise ValueError(
                "non-token-set evaluators require target_values and control_values"
            )
        if any(value == "" for value in self.target_values + self.control_values):
            raise ValueError("evaluator values must not be empty")
        if self.kind == "unicode_script":
            supported = {"han", "latin"}
            unknown = set(self.target_values + self.control_values) - supported
            if unknown:
                raise ValueError(
                    f"unsupported unicode scripts {sorted(unknown)}; supported: {sorted(supported)}"
                )
        return self


class QualificationSpec(StrictModel):
    schema_version: Literal["probe.qualify/v1"] = QUALIFICATION_SCHEMA_VERSION
    kind: Literal["qualify"] = "qualify"
    name: str = "unnamed-qualification"
    description: str | None = None
    parent_run_id: str
    pair_ids: tuple[str, ...] = ()
    generation: GenerationRequest = Field(default_factory=GenerationRequest)
    evaluator: BehaviorEvaluatorRequest = Field(
        default_factory=BehaviorEvaluatorRequest
    )
    execution: ExecutionLimits
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("name", "parent_run_id")
    @classmethod
    def validate_qualification_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("qualification name and parent run ID must not be blank")
        return value


class NeuronReference(StrictModel):
    layer: int = Field(ge=0)
    neuron: int = Field(ge=0)


class NeuronSelectionRequest(StrictModel):
    strategy: Literal["ranked_top_k", "explicit"] = "ranked_top_k"
    candidate_method: Literal[
        "parent_ranking", "direct_downstream_overlap"
    ] = "parent_ranking"
    overlap_pool_size: PositiveInt | None = None
    top_k: PositiveInt | None = 20
    explicit: tuple[NeuronReference, ...] = ()
    layers: tuple[int, ...] = ()
    sign: Literal["any", "positive", "negative"] = "any"
    min_sign_consistency: float = Field(default=0.0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_strategy(self) -> "NeuronSelectionRequest":
        if self.strategy == "ranked_top_k":
            if self.top_k is None:
                raise ValueError("ranked_top_k selection requires top_k")
            if self.explicit:
                raise ValueError("ranked_top_k selection cannot include explicit neurons")
            if self.candidate_method == "direct_downstream_overlap":
                if self.overlap_pool_size is None:
                    raise ValueError(
                        "direct_downstream_overlap requires overlap_pool_size"
                    )
                if self.top_k is not None and self.overlap_pool_size < self.top_k:
                    raise ValueError("overlap_pool_size must be at least top_k")
            elif self.overlap_pool_size is not None:
                raise ValueError(
                    "overlap_pool_size is only valid for direct_downstream_overlap"
                )
        else:
            if not self.explicit:
                raise ValueError("explicit selection requires at least one neuron")
            if self.top_k is not None:
                raise ValueError("explicit selection requires top_k=null")
            if self.candidate_method != "parent_ranking":
                raise ValueError("explicit selection cannot change candidate_method")
            if self.overlap_pool_size is not None:
                raise ValueError("explicit selection cannot set overlap_pool_size")
            identities = [(item.layer, item.neuron) for item in self.explicit]
            if len(identities) != len(set(identities)):
                raise ValueError("explicit neuron identities must be unique")
        if any(layer < 0 for layer in self.layers):
            raise ValueError("selection layers must be non-negative")
        if len(self.layers) != len(set(self.layers)):
            raise ValueError("selection layers must be unique")
        return self


class InterventionOperationRequest(StrictModel):
    mode: Literal["ablate", "amplify", "patch", "restore"]
    condition: Literal["auto", "original", "perturbed", "both"] = "auto"
    apply_during_generation: bool = False

    @model_validator(mode="after")
    def validate_generation_semantics(self) -> "InterventionOperationRequest":
        if self.mode in {"patch", "restore"} and self.apply_during_generation:
            raise ValueError(
                "patch/restore generation is unsupported because source activations "
                "are defined only at the first-token decision"
            )
        return self


class DoseSweepRequest(StrictModel):
    neuron_counts: tuple[PositiveInt, ...] = (20,)
    strengths: tuple[float, ...] = (0.0,)

    @model_validator(mode="after")
    def validate_unique_doses(self) -> "DoseSweepRequest":
        if not self.neuron_counts or not self.strengths:
            raise ValueError("dose sweep requires neuron_counts and strengths")
        if len(self.neuron_counts) != len(set(self.neuron_counts)):
            raise ValueError("neuron_counts must be unique")
        if len(self.strengths) != len(set(self.strengths)):
            raise ValueError("strengths must be unique")
        return self


class RandomControlRequest(StrictModel):
    samples: int = Field(default=3, ge=0, le=100)
    same_layer: Literal[True] = True


class AdditivityRequest(StrictModel):
    top_n: int = Field(default=0, ge=0, le=8)


class InterventionSpec(StrictModel):
    schema_version: Literal["probe.intervention/v1"] = INTERVENTION_SCHEMA_VERSION
    kind: Literal["intervention"] = "intervention"
    name: str = "unnamed-intervention"
    description: str | None = None
    parent_run_id: str
    qualification_run_id: str | None = None
    trajectory_run_id: str | None = None
    pair_ids: tuple[str, ...] = ()
    include_weak_pairs: bool = False
    selection: NeuronSelectionRequest = Field(default_factory=NeuronSelectionRequest)
    operation: InterventionOperationRequest
    sweep: DoseSweepRequest = Field(default_factory=DoseSweepRequest)
    controls: RandomControlRequest = Field(default_factory=RandomControlRequest)
    additivity: AdditivityRequest = Field(default_factory=AdditivityRequest)
    generation: GenerationRequest | None = None
    evaluator: BehaviorEvaluatorRequest | None = None
    collateral_observables: tuple[ObservableRequest, ...] = ()
    execution: ExecutionLimits
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator(
        "name", "parent_run_id", "qualification_run_id", "trajectory_run_id"
    )
    @classmethod
    def validate_intervention_identity(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.strip():
            raise ValueError("intervention identifiers must not be blank")
        return value

    @model_validator(mode="after")
    def validate_dose_semantics(self) -> "InterventionSpec":
        strengths = self.sweep.strengths
        if self.operation.mode == "ablate" and any(
            value < 0 or value > 1 for value in strengths
        ):
            raise ValueError("ablation strengths are activation scales in [0, 1]")
        if self.operation.mode == "amplify" and any(value < 1 for value in strengths):
            raise ValueError("amplification strengths must be at least 1")
        if self.operation.mode in {"patch", "restore"} and any(
            value < 0 or value > 1 for value in strengths
        ):
            raise ValueError("patch/restore strengths are mixing fractions in [0, 1]")
        names = [item.name for item in self.collateral_observables]
        if len(names) != len(set(names)):
            raise ValueError("collateral observable names must be unique")
        if self.evaluator is not None and (
            self.generation is None or not self.operation.apply_during_generation
        ):
            raise ValueError(
                "an intervention evaluator requires generation and apply_during_generation=true"
            )
        return self


class DirectionInjectionSpec(StrictModel):
    schema_version: Literal["probe.direction/v1"] = DIRECTION_SCHEMA_VERSION
    kind: Literal["direction"] = "direction"
    name: str = "unnamed-direction-sweep"
    description: str | None = None
    parent_run_id: str
    qualification_run_id: str | None = None
    pair_ids: tuple[str, ...] = ()
    include_weak_pairs: bool = False
    layers: tuple[int, ...] = Field(min_length=1)
    betas: tuple[float, ...] = Field(min_length=1)
    condition: Literal["original", "perturbed", "both"] = "both"
    normalization: Literal["raw", "residual_norm"] = "residual_norm"
    random_direction_samples: int = Field(default=3, ge=0, le=100)
    generation: GenerationRequest | None = None
    evaluator: BehaviorEvaluatorRequest | None = None
    collateral_observables: tuple[ObservableRequest, ...] = ()
    execution: ExecutionLimits
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("name", "parent_run_id", "qualification_run_id")
    @classmethod
    def validate_direction_identity(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.strip():
            raise ValueError("direction sweep identifiers must not be blank")
        return value

    @model_validator(mode="after")
    def validate_direction_sweep(self) -> "DirectionInjectionSpec":
        if len(self.layers) != len(set(self.layers)) or any(
            layer < 0 for layer in self.layers
        ):
            raise ValueError("direction layers must be unique and non-negative")
        if len(self.betas) != len(set(self.betas)):
            raise ValueError("direction betas must be unique")
        if all(beta == 0 for beta in self.betas):
            raise ValueError("direction sweep requires at least one non-zero beta")
        if self.evaluator is not None and self.generation is None:
            raise ValueError("a direction evaluator requires generation settings")
        names = [item.name for item in self.collateral_observables]
        if len(names) != len(set(names)):
            raise ValueError("collateral observable names must be unique")
        return self


class AttentionHeadReference(StrictModel):
    layer: int = Field(ge=0)
    head: int = Field(ge=0)


class AttentionRankingRequest(StrictModel):
    top_k: PositiveInt = 64
    pair_aggregation: Literal["single_pair", "rms"] = "single_pair"


class AttentionHeadRankSpec(StrictModel):
    """Observational output-head attribution at the first-token decision."""

    schema_version: Literal["probe.attention-rank/v1"] = (
        ATTENTION_RANK_SCHEMA_VERSION
    )
    kind: Literal["attention_rank"] = "attention_rank"
    name: str = "unnamed-attention-ranking"
    description: str | None = None
    parent_run_id: str
    qualification_run_id: str | None = None
    pair_ids: tuple[str, ...] = ()
    include_weak_pairs: bool = False
    ranking: AttentionRankingRequest = Field(default_factory=AttentionRankingRequest)
    execution: ExecutionLimits
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("name", "parent_run_id", "qualification_run_id")
    @classmethod
    def validate_attention_rank_identity(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("attention ranking identifiers must not be blank")
        return value


class AttentionHeadSelectionRequest(StrictModel):
    strategy: Literal["ranked_top_k", "explicit"] = "ranked_top_k"
    top_k: PositiveInt | None = 16
    explicit: tuple[AttentionHeadReference, ...] = ()
    layers: tuple[int, ...] = ()
    sign: Literal["any", "positive", "negative"] = "any"
    min_sign_consistency: float = Field(default=0.0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_attention_head_strategy(self) -> "AttentionHeadSelectionRequest":
        if self.strategy == "ranked_top_k":
            if self.top_k is None:
                raise ValueError("ranked_top_k selection requires top_k")
            if self.explicit:
                raise ValueError("ranked_top_k selection cannot include explicit heads")
        else:
            if not self.explicit:
                raise ValueError("explicit selection requires at least one head")
            if self.top_k is not None:
                raise ValueError("explicit selection requires top_k=null")
            identities = [(item.layer, item.head) for item in self.explicit]
            if len(identities) != len(set(identities)):
                raise ValueError("explicit attention head identities must be unique")
        if any(layer < 0 for layer in self.layers):
            raise ValueError("selection layers must be non-negative")
        if len(self.layers) != len(set(self.layers)):
            raise ValueError("selection layers must be unique")
        return self


class AttentionInterventionOperationRequest(StrictModel):
    mode: Literal["ablate", "amplify", "patch", "restore"]
    condition: Literal["auto", "original", "perturbed", "both"] = "auto"


class AttentionDoseSweepRequest(StrictModel):
    head_counts: tuple[PositiveInt, ...] = (1, 2, 4, 8, 16)
    strengths: tuple[float, ...] = (0.0,)

    @model_validator(mode="after")
    def validate_attention_doses(self) -> "AttentionDoseSweepRequest":
        if not self.head_counts or not self.strengths:
            raise ValueError("attention dose sweep requires head_counts and strengths")
        if len(self.head_counts) != len(set(self.head_counts)):
            raise ValueError("head_counts must be unique")
        if len(self.strengths) != len(set(self.strengths)):
            raise ValueError("strengths must be unique")
        return self


class AttentionHeadInterventionSpec(StrictModel):
    schema_version: Literal["probe.attention-intervention/v1"] = (
        ATTENTION_INTERVENTION_SCHEMA_VERSION
    )
    kind: Literal["attention_intervention"] = "attention_intervention"
    name: str = "unnamed-attention-intervention"
    description: str | None = None
    parent_run_id: str
    qualification_run_id: str | None = None
    pair_ids: tuple[str, ...] = ()
    include_weak_pairs: bool = False
    selection: AttentionHeadSelectionRequest = Field(
        default_factory=AttentionHeadSelectionRequest
    )
    operation: AttentionInterventionOperationRequest
    sweep: AttentionDoseSweepRequest = Field(default_factory=AttentionDoseSweepRequest)
    controls: RandomControlRequest = Field(
        default_factory=lambda: RandomControlRequest(samples=5)
    )
    execution: ExecutionLimits
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("name", "parent_run_id", "qualification_run_id")
    @classmethod
    def validate_attention_intervention_identity(
        cls, value: str | None
    ) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("attention intervention identifiers must not be blank")
        return value

    @model_validator(mode="after")
    def validate_attention_intervention_doses(
        self,
    ) -> "AttentionHeadInterventionSpec":
        strengths = self.sweep.strengths
        if self.operation.mode == "ablate" and any(
            value < 0 or value > 1 for value in strengths
        ):
            raise ValueError("head ablation strengths are output scales in [0, 1]")
        if self.operation.mode == "amplify" and any(value < 1 for value in strengths):
            raise ValueError("head amplification strengths must be at least 1")
        if self.operation.mode in {"patch", "restore"} and any(
            value < 0 or value > 1 for value in strengths
        ):
            raise ValueError("head patch/restore strengths are mixing fractions in [0, 1]")
        return self


class AlignedTokenPosition(StrictModel):
    original: int = Field(ge=0)
    perturbed: int = Field(ge=0)


class TokenAlignmentRequest(StrictModel):
    pair_id: str
    mode: Literal["identity", "explicit"] = "identity"
    positions: tuple[AlignedTokenPosition, ...] = ()

    @model_validator(mode="after")
    def validate_alignment(self) -> "TokenAlignmentRequest":
        if not self.pair_id.strip():
            raise ValueError("alignment pair_id must not be blank")
        if self.mode == "identity" and self.positions:
            raise ValueError("identity alignment cannot include explicit positions")
        if self.mode == "explicit":
            if not self.positions:
                raise ValueError("explicit alignment requires token positions")
            original = [item.original for item in self.positions]
            perturbed = [item.perturbed for item in self.positions]
            if len(original) != len(set(original)) or len(perturbed) != len(set(perturbed)):
                raise ValueError("explicit alignment must be one-to-one")
        return self


class AttentionTraceSpec(StrictModel):
    schema_version: Literal["probe.attention-trace/v1"] = ATTENTION_TRACE_SCHEMA_VERSION
    kind: Literal["attention_trace"] = "attention_trace"
    name: str = "unnamed-attention-trace"
    description: str | None = None
    trace_kind: Literal["token_edges", "head_paths"]
    parent_run_id: str
    parent_intervention_run_id: str | None = None
    pair_ids: tuple[str, ...] = ()
    include_weak_pairs: bool = False
    heads: tuple[AttentionHeadReference, ...] = ()
    senders: tuple[AttentionHeadReference, ...] = ()
    receivers: tuple[AttentionHeadReference, ...] = ()
    operation: Literal["patch", "restore"] = "patch"
    alignments: tuple[TokenAlignmentRequest, ...] = ()
    max_token_edges: PositiveInt = 256
    max_paths: PositiveInt = Field(default=64, le=64)
    controls: RandomControlRequest = Field(
        default_factory=lambda: RandomControlRequest(samples=5)
    )
    execution: ExecutionLimits
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("name", "parent_run_id", "parent_intervention_run_id")
    @classmethod
    def validate_attention_trace_identity(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("attention trace identifiers must not be blank")
        return value

    @model_validator(mode="after")
    def validate_trace_shape(self) -> "AttentionTraceSpec":
        def unique(items: tuple[AttentionHeadReference, ...], label: str) -> None:
            identities = [(item.layer, item.head) for item in items]
            if len(identities) != len(set(identities)):
                raise ValueError(f"{label} attention heads must be unique")

        unique(self.heads, "trace")
        unique(self.senders, "sender")
        unique(self.receivers, "receiver")
        alignment_ids = [item.pair_id for item in self.alignments]
        if len(alignment_ids) != len(set(alignment_ids)):
            raise ValueError("token alignment pair IDs must be unique")
        if self.trace_kind == "token_edges":
            if not self.heads:
                raise ValueError("token_edges tracing requires heads")
            if self.senders or self.receivers or self.parent_intervention_run_id:
                raise ValueError(
                    "token_edges tracing cannot include path-only fields"
                )
        else:
            if not self.senders or not self.receivers:
                raise ValueError("head_paths tracing requires senders and receivers")
            if self.heads:
                raise ValueError("head_paths tracing uses senders/receivers, not heads")
            if self.parent_intervention_run_id is None:
                raise ValueError(
                    "head_paths tracing requires parent_intervention_run_id"
                )
            paths = len(self.senders) * len(self.receivers)
            if paths > self.max_paths:
                raise ValueError(
                    f"requested {paths} sender-receiver paths exceeds max_paths={self.max_paths}"
                )
            invalid = [
                (sender.layer, receiver.layer)
                for sender in self.senders
                for receiver in self.receivers
                if sender.layer >= receiver.layer
            ]
            if invalid:
                raise ValueError("each sender layer must precede each receiver layer")
            if not self.alignments:
                raise ValueError("head_paths tracing requires explicit alignment declarations")
        return self


ExperimentSpec = (
    RankSpec
    | TrajectorySpec
    | FFNCouplingSpec
    | QualificationSpec
    | InterventionSpec
    | DirectionInjectionSpec
    | AttentionHeadRankSpec
    | AttentionHeadInterventionSpec
    | AttentionTraceSpec
)


class ResearchWorkflowSpec(StrictModel):
    """A replayable multi-stage driver with symbolic lineage references."""

    schema_version: Literal["probe.workflow/v1"] = "probe.workflow/v1"
    name: str
    description: str | None = None
    rank: RankSpec
    qualification: QualificationSpec | None = None
    trajectory: TrajectorySpec | None = None
    ffn_coupling: FFNCouplingSpec | None = None
    interventions: tuple[InterventionSpec, ...] = ()
    directions: tuple[DirectionInjectionSpec, ...] = ()
    attention_rank: AttentionHeadRankSpec | None = None
    attention_interventions: tuple[AttentionHeadInterventionSpec, ...] = ()
    attention_traces: tuple[AttentionTraceSpec, ...] = ()

    @field_validator("name")
    @classmethod
    def validate_workflow_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("workflow name must not be blank")
        return value

    @model_validator(mode="after")
    def validate_symbolic_lineage(self) -> "ResearchWorkflowSpec":
        if self.qualification is not None and self.qualification.parent_run_id != "$rank":
            raise ValueError("workflow qualification parent_run_id must be '$rank'")
        if self.trajectory is not None and self.trajectory.parent_run_id != "$rank":
            raise ValueError("workflow trajectory parent_run_id must be '$rank'")
        if self.ffn_coupling is not None:
            if self.ffn_coupling.parent_run_id != "$rank":
                raise ValueError("workflow FFN coupling parent_run_id must be '$rank'")
            if self.ffn_coupling.trajectory_run_id not in {None, "$trajectory"}:
                raise ValueError(
                    "workflow FFN coupling trajectory_run_id must be '$trajectory' or null"
                )
            if (
                self.ffn_coupling.trajectory_run_id == "$trajectory"
                and self.trajectory is None
            ):
                raise ValueError(
                    "workflow FFN coupling references '$trajectory' but no trajectory stage exists"
                )
        for child in self.interventions:
            if child.parent_run_id not in {"$rank", "$ffn_coupling"}:
                raise ValueError(
                    "workflow intervention parent_run_id must be '$rank' or '$ffn_coupling'"
                )
            if child.parent_run_id == "$ffn_coupling" and self.ffn_coupling is None:
                raise ValueError(
                    "workflow intervention references '$ffn_coupling' but no FFN coupling stage exists"
                )
            if child.qualification_run_id not in {None, "$qualification"}:
                raise ValueError(
                    "workflow qualification_run_id must be '$qualification' or null"
                )
            if child.qualification_run_id == "$qualification" and self.qualification is None:
                raise ValueError(
                    "workflow causal stage references '$qualification' but no qualification stage exists"
                )
            if child.trajectory_run_id not in {None, "$trajectory"}:
                raise ValueError(
                    "workflow intervention trajectory_run_id must be '$trajectory' or null"
                )
            if child.trajectory_run_id == "$trajectory" and self.trajectory is None:
                raise ValueError(
                    "workflow intervention references '$trajectory' but no trajectory stage exists"
                )
        for child in self.directions:
            if child.parent_run_id != "$rank":
                raise ValueError("workflow direction parent_run_id must be '$rank'")
            if child.qualification_run_id not in {None, "$qualification"}:
                raise ValueError(
                    "workflow qualification_run_id must be '$qualification' or null"
                )
            if child.qualification_run_id == "$qualification" and self.qualification is None:
                raise ValueError(
                    "workflow causal stage references '$qualification' but no qualification stage exists"
                )
        if self.attention_rank is not None:
            if self.attention_rank.parent_run_id != "$rank":
                raise ValueError("workflow attention rank parent_run_id must be '$rank'")
            if self.attention_rank.qualification_run_id not in {None, "$qualification"}:
                raise ValueError(
                    "workflow attention rank qualification_run_id must be '$qualification' or null"
                )
            if (
                self.attention_rank.qualification_run_id == "$qualification"
                and self.qualification is None
            ):
                raise ValueError(
                    "workflow attention rank references '$qualification' but no qualification stage exists"
                )
        elif self.attention_interventions or self.attention_traces:
            raise ValueError("workflow attention child stages require attention_rank")
        for child in self.attention_interventions:
            if child.parent_run_id != "$attention_rank":
                raise ValueError(
                    "workflow attention intervention parent_run_id must be '$attention_rank'"
                )
            if child.qualification_run_id not in {None, "$qualification"}:
                raise ValueError(
                    "workflow attention intervention qualification_run_id must be '$qualification' or null"
                )
        for child in self.attention_traces:
            if child.parent_run_id != "$attention_rank":
                raise ValueError(
                    "workflow attention trace parent_run_id must be '$attention_rank'"
                )
            if child.parent_intervention_run_id not in {
                None,
                "$attention_intervention",
            }:
                raise ValueError(
                    "workflow trace parent_intervention_run_id must be '$attention_intervention' or null"
                )
            if (
                child.parent_intervention_run_id == "$attention_intervention"
                and len(self.attention_interventions) != 1
            ):
                raise ValueError(
                    "'$attention_intervention' requires exactly one attention intervention stage"
                )
        return self


class ExperimentPlan(StrictModel):
    schema_version: Literal["probe.plan/v1"] = "probe.plan/v1"
    science_hash: str
    request_hash: str
    kind: Literal[
        "rank",
        "trajectory",
        "ffn_coupling",
        "qualify",
        "intervention",
        "direction",
        "attention_rank",
        "attention_intervention",
        "attention_trace",
    ]
    pair_count: PositiveInt
    forward_passes: PositiveInt
    backward_passes: int = Field(default=0, ge=0)
    within_budget: bool
    model_cached: bool
    resolved_device: str
    warnings: tuple[str, ...] = ()


class CapabilityReport(StrictModel):
    schema_version: Literal["probe.capabilities/v1"] = "probe.capabilities/v1"
    model_id: str
    supported: bool
    adapter: str
    device: str
    dtype: str
    activation: str
    ranking: bool = True
    interventions: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


class PreflightReport(StrictModel):
    schema_version: Literal["probe.preflight/v1"] = "probe.preflight/v1"
    valid: Literal[True] = True
    science_hash: str
    request_hash: str
    executable: bool
    model_ready: bool
    acquisition_required: bool
    plan: ExperimentPlan
    capabilities: CapabilityReport
    warnings: tuple[str, ...] = ()


class ClaimRecord(StrictModel):
    claim_id: str
    claim_type: Literal[
        "observable_validity",
        "candidate_ranking",
        "replicated_ranking",
        "causal_effect",
        "necessity",
        "sufficiency",
        "restoration",
        "generalization",
        "attention_routing",
        "causal_path",
    ]
    status: Literal["supported", "exploratory", "blocked", "not_supported"]
    evidence_run_ids: tuple[str, ...] = ()
    statement: str
    limitations: tuple[str, ...] = ()


class ResearchIntent(StrictModel):
    hypothesis: str
    intended_perturbation: str
    invariants: tuple[str, ...] = ()
    falsifying_outcome: str

    @field_validator("hypothesis", "intended_perturbation", "falsifying_outcome")
    @classmethod
    def validate_research_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("research intent fields must not be blank")
        return value.strip()


class ResearchCaseCreate(StrictModel):
    schema_version: Literal["probe.research-case-create/v1"] = (
        "probe.research-case-create/v1"
    )
    intent: ResearchIntent
    workflow: ResearchWorkflowSpec
    rank_run_id: str | None = None


class ResearchCaseUpdate(StrictModel):
    schema_version: Literal["probe.research-case-update/v1"] = (
        "probe.research-case-update/v1"
    )
    revision: PositiveInt
    intent: ResearchIntent
    workflow: ResearchWorkflowSpec


class ResearchCaseStage(StrictModel):
    key: str = Field(pattern=r"^[a-z][a-z0-9-]{0,127}$")
    kind: Literal[
        "rank",
        "trajectory",
        "ffn_coupling",
        "qualify",
        "intervention",
        "direction",
        "attention_rank",
        "attention_intervention",
        "attention_trace",
    ]
    name: str
    trace_kind: Literal["token_edges", "head_paths"] | None = None
    spec_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal[
        "not_configured", "ready", "running", "failed", "gate_failed", "verified"
    ]
    job_id: str | None = None
    run_id: str | None = None
    parent_run_ids: tuple[str, ...] = ()
    verification_failures: tuple[str, ...] = ()
    error: ErrorDetail | None = None
    claims: tuple[ClaimRecord, ...] = ()
    warnings: tuple[str, ...] = ()


class ResearchCase(StrictModel):
    schema_version: Literal["probe.research-case/v1"] = "probe.research-case/v1"
    case_id: str
    revision: PositiveInt
    created_at: datetime
    updated_at: datetime
    intent: ResearchIntent
    workflow: ResearchWorkflowSpec
    evidence_label: Literal[
        "observational", "behaviorally_qualified", "locally_causal", "heldout_replicated"
    ] = "observational"
    stages: tuple[ResearchCaseStage, ...]
    warnings: tuple[str, ...] = ()


class ResearchCaseStagePlan(StrictModel):
    key: str
    status: str
    plan: ExperimentPlan | None = None
    blocked_reason: str | None = None


class ResearchCasePlan(StrictModel):
    schema_version: Literal["probe.research-case-plan/v1"] = (
        "probe.research-case-plan/v1"
    )
    case_id: str
    evidence_label: str
    stages: tuple[ResearchCaseStagePlan, ...]
    total_forward_passes: int = Field(ge=0)
    warnings: tuple[str, ...] = ()


class ExecutionReceipt(StrictModel):
    """Agent-facing handoff from one completed typed experiment stage."""

    schema_version: Literal["probe.execution-receipt/v1"] = (
        "probe.execution-receipt/v1"
    )
    run_id: str
    run_kind: Literal[
        "rank",
        "trajectory",
        "ffn_coupling",
        "qualify",
        "intervention",
        "direction",
        "attention_rank",
        "attention_intervention",
        "attention_trace",
    ]
    evidence_stage: str
    logical_forward_passes: PositiveInt
    result: dict[str, Any]


class PairQualification(StrictModel):
    status: Literal["informative", "weak", "invalid"]
    original_decision: Literal["target", "control", "other", "tie"]
    perturbed_decision: Literal["target", "control", "other", "tie"]
    decision_crossing: bool
    predictions_in_observable: bool
    original_target_probability: float = Field(ge=0, le=1)
    original_control_probability: float = Field(ge=0, le=1)
    perturbed_target_probability: float = Field(ge=0, le=1)
    perturbed_control_probability: float = Field(ge=0, le=1)
    absolute_movement: float = Field(ge=0)
    reasons: tuple[str, ...] = ()


class QualificationAggregate(StrictModel):
    informative_pairs: int = Field(ge=0)
    weak_pairs: int = Field(ge=0)
    invalid_pairs: int = Field(ge=0)
    informative_pair_ids: tuple[str, ...] = ()
    claim_eligible: bool
    signal_concentration_label: str


class PairResultSummary(StrictModel):
    pair_id: str
    split: Literal["discovery", "validation", "heldout"] = "discovery"
    original_gap: float
    perturbed_gap: float
    measured_delta: float
    predicted_delta: float
    original_prediction: str
    perturbed_prediction: str
    ffn_skip_mean: float | None
    circuit_regime: str
    elapsed_seconds: float
    qualification: PairQualification | None = None
    warnings: tuple[str, ...] = ()


class AggregateNeuronScore(StrictModel):
    rank: PositiveInt
    layer: int = Field(ge=0)
    neuron: int = Field(ge=0)
    coupling: float
    original_activation_mean: float
    perturbed_activation_mean: float
    activation_delta_mean: float
    importance_mean: float
    importance_rms: float = Field(ge=0)
    sign_consistency: float = Field(ge=0, le=1)


class AggregateLayerSummary(StrictModel):
    layer: int = Field(ge=0)
    signed_mean_sum: float
    rms_mass: float = Field(ge=0)
    positive_mean_mass: float = Field(ge=0)
    negative_mean_mass: float = Field(ge=0)
    top_10_rms_share: float = Field(ge=0, le=1)
    maximum_rms: float = Field(ge=0)
    top_neuron: int = Field(ge=0)
    activation_delta_norm_mean: float = Field(ge=0)


class RankRunSummary(StrictModel):
    schema_version: Literal["probe.rank-result/v1"] = "probe.rank-result/v1"
    science_hash: str
    pair_count: PositiveInt
    split_counts: dict[str, int] = Field(default_factory=dict)
    logical_forward_passes: PositiveInt
    model: dict[str, Any]
    observable: dict[str, Any]
    pairs: tuple[PairResultSummary, ...]
    layers: tuple[AggregateLayerSummary, ...]
    neurons: tuple[AggregateNeuronScore, ...]
    total_neuron_count: PositiveInt
    measured_delta_mean: float
    predicted_delta_mean: float
    ffn_skip_mean: float | None
    evidence_stage: Literal["exploratory_pair", "replicated_ranking"]
    qualification: QualificationAggregate | None = None
    claims: tuple[ClaimRecord, ...] = ()
    warnings: tuple[str, ...] = ()


class TrajectoryCheckpointSummary(StrictModel):
    layer: int = Field(ge=0)
    checkpoint: Literal["block_input", "post_attention", "post_ffn"]
    original_gap: float
    perturbed_gap: float
    pair_delta: float
    original_target_probability: float = Field(ge=0, le=1)
    perturbed_target_probability: float = Field(ge=0, le=1)
    original_control_probability: float = Field(ge=0, le=1)
    perturbed_control_probability: float = Field(ge=0, le=1)
    original_entropy: float = Field(ge=0)
    perturbed_entropy: float = Field(ge=0)
    original_target_rank: PositiveInt
    perturbed_target_rank: PositiveInt
    original_forward_kl_to_final: float = Field(ge=0)
    perturbed_forward_kl_to_final: float = Field(ge=0)
    paired_js: float = Field(ge=0)
    paired_total_variation: float = Field(ge=0, le=1)


class TrajectoryTransitionSuggestion(StrictModel):
    rank: PositiveInt
    layer: int = Field(ge=0)
    checkpoint: Literal["block_input", "post_attention", "post_ffn"]
    pair_delta_change: float
    absolute_change: float = Field(ge=0)
    reason: Literal["largest_pair_delta_change"] = "largest_pair_delta_change"


class PairTrajectorySummary(StrictModel):
    pair_id: str
    split: Literal["discovery", "validation", "heldout"] = "discovery"
    checkpoints: tuple[TrajectoryCheckpointSummary, ...]
    transitions: tuple[TrajectoryTransitionSuggestion, ...]
    final_pair_delta: float
    warnings: tuple[str, ...] = ()


class TrajectoryRunSummary(StrictModel):
    schema_version: Literal["probe.trajectory-result/v1"] = (
        "probe.trajectory-result/v1"
    )
    science_hash: str
    parent_run_id: str
    model: dict[str, Any]
    observable: dict[str, Any]
    pair_count: PositiveInt
    logical_forward_passes: PositiveInt
    pairs: tuple[PairTrajectorySummary, ...]
    evidence_stage: Literal["observational_trajectory"] = "observational_trajectory"
    claims: tuple[ClaimRecord, ...] = ()
    warnings: tuple[str, ...] = ()


class FFNCouplingNeuronScore(StrictModel):
    rank: PositiveInt
    layer: int = Field(ge=0)
    neuron: int = Field(ge=0)
    activation_delta_mean: float
    direct_coupling: float
    direct_importance_rms: float = Field(ge=0)
    native_coupling_mean: float | None = None
    native_importance_mean: float | None = None
    native_importance_rms: float | None = Field(default=None, ge=0)
    downstream_coupling_mean: float
    downstream_importance_mean: float
    downstream_importance_rms: float = Field(ge=0)
    downstream_sign_consistency: float = Field(ge=0, le=1)
    direct_downstream_sign_agreement: float = Field(ge=0, le=1)


class FFNCouplingLayerSummary(StrictModel):
    layer: int = Field(ge=0)
    downstream_rms_mass: float = Field(ge=0)
    native_rms_mass: float | None = Field(default=None, ge=0)
    direct_rms_mass: float = Field(ge=0)
    top_neuron: int = Field(ge=0)


class FFNCouplingPairSummary(StrictModel):
    pair_id: str
    split: Literal["discovery", "validation", "heldout"] = "discovery"
    original_gradient_norm_mean: float = Field(ge=0)
    perturbed_gradient_norm_mean: float = Field(ge=0)


class FFNCouplingRunSummary(StrictModel):
    schema_version: Literal["probe.ffn-coupling-result/v1"] = (
        "probe.ffn-coupling-result/v1"
    )
    science_hash: str
    parent_run_id: str
    trajectory_run_id: str | None = None
    model: dict[str, Any]
    observable: dict[str, Any]
    pair_count: PositiveInt
    logical_forward_passes: PositiveInt
    logical_backward_passes: PositiveInt
    methods: tuple[str, ...]
    pairs: tuple[FFNCouplingPairSummary, ...]
    layers: tuple[FFNCouplingLayerSummary, ...]
    neurons: tuple[FFNCouplingNeuronScore, ...]
    total_neuron_count: PositiveInt
    evidence_stage: Literal["observational_ffn_coupling"] = (
        "observational_ffn_coupling"
    )
    claims: tuple[ClaimRecord, ...] = ()
    warnings: tuple[str, ...] = ()


class NeuronCandidate(AggregateNeuronScore):
    observable_effect: Literal["toward_target", "toward_control", "neutral"]


class RunOverview(StrictModel):
    schema_version: Literal["probe.run-overview/v1"] = "probe.run-overview/v1"
    run_id: str
    science_hash: str
    evidence_stage: Literal["exploratory_pair", "replicated_ranking"]
    model: dict[str, Any]
    observable: dict[str, Any]
    pair_count: PositiveInt
    logical_forward_passes: PositiveInt
    pairs: tuple[PairResultSummary, ...]
    measured_delta_mean: float
    predicted_delta_mean: float
    ffn_skip_mean: float | None
    total_neuron_count: PositiveInt
    top_layers: tuple[AggregateLayerSummary, ...]
    top_neurons: tuple[NeuronCandidate, ...]
    qualification: QualificationAggregate | None = None
    claims: tuple[ClaimRecord, ...] = ()
    warnings: tuple[str, ...] = ()


class GeneratedBehavior(StrictModel):
    condition: Literal["original", "perturbed"]
    text: str
    token_ids: tuple[int, ...]
    observable_decision: Literal["target", "control", "other", "tie"]
    behavior_decision: Literal["target", "control", "other", "ambiguous"]
    agrees_with_observable: bool


class QualifiedPairResult(StrictModel):
    pair_id: str
    split: Literal["discovery", "validation", "heldout"] = "discovery"
    first_token: PairQualification
    generated: tuple[GeneratedBehavior, ...]
    status: Literal["informative", "weak", "invalid"]
    reasons: tuple[str, ...] = ()


class QualificationRunSummary(StrictModel):
    schema_version: Literal["probe.qualification-result/v1"] = (
        "probe.qualification-result/v1"
    )
    science_hash: str
    parent_run_id: str
    model: dict[str, Any]
    evaluator: BehaviorEvaluatorRequest
    generation: GenerationRequest
    pairs: tuple[QualifiedPairResult, ...]
    aggregate: QualificationAggregate
    logical_forward_passes: PositiveInt
    evidence_stage: Literal["qualified_observable"] = "qualified_observable"
    claims: tuple[ClaimRecord, ...] = ()
    warnings: tuple[str, ...] = ()


class SelectedNeuron(StrictModel):
    rank: PositiveInt | None = None
    layer: int = Field(ge=0)
    neuron: int = Field(ge=0)
    importance_mean: float | None = None
    importance_rms: float | None = Field(default=None, ge=0)
    sign_consistency: float | None = Field(default=None, ge=0, le=1)
    score_method: Literal[
        "direct_structural",
        "downstream_endpoint_gradient",
        "direct_downstream_overlap",
    ] = "direct_structural"


class InterventionObservation(StrictModel):
    pair_id: str
    split: Literal["discovery", "validation", "heldout"] = "discovery"
    arm: Literal["selected", "matched_random", "additivity_single", "additivity_pair"]
    control_sample: int | None = Field(default=None, ge=0)
    condition: Literal["original", "perturbed"]
    mode: Literal["ablate", "amplify", "patch", "restore"]
    neuron_count: PositiveInt
    strength: float
    baseline_gap: float
    source_gap: float | None = None
    intervention_gap: float
    gap_effect: float
    normalized_source_progress: float | None = None
    baseline_prediction: str
    intervention_prediction: str
    baseline_generated_text: str | None = None
    generated_text: str | None = None
    baseline_behavior_decision: Literal[
        "target", "control", "other", "ambiguous"
    ] | None = None
    intervention_behavior_decision: Literal[
        "target", "control", "other", "ambiguous"
    ] | None = None
    collateral_gap_effects: dict[str, float] = Field(default_factory=dict)


class InterventionTrajectoryCheckpoint(StrictModel):
    pair_id: str
    split: Literal["discovery", "validation", "heldout"] = "discovery"
    arm: Literal["selected", "matched_random"]
    control_sample: int | None = Field(default=None, ge=0)
    condition: Literal["original", "perturbed"]
    mode: Literal["ablate", "amplify", "patch", "restore"]
    neuron_count: PositiveInt
    strength: float
    layer: int = Field(ge=0)
    checkpoint: Literal["block_input", "post_attention", "post_ffn"]
    baseline_gap: float
    intervention_gap: float
    gap_effect: float
    normalized_source_progress: float | None = None


class InterventionDoseSummary(StrictModel):
    split: Literal["discovery", "validation", "heldout"] = "discovery"
    condition: Literal["original", "perturbed"]
    neuron_count: PositiveInt
    strength: float
    selected_effect_mean: float
    selected_absolute_effect_mean: float = Field(ge=0)
    random_absolute_effect_mean: float | None = Field(default=None, ge=0)
    controlled_absolute_effect: float | None = None
    bootstrap_low: float | None = None
    bootstrap_high: float | None = None
    pair_count: PositiveInt
    random_observation_count: int = Field(ge=0)


class AdditivityViolation(StrictModel):
    pair_id: str
    condition: Literal["original", "perturbed"]
    first: NeuronReference
    second: NeuronReference
    first_effect: float
    second_effect: float
    joint_effect: float
    epsilon: float


class CausalWidthEstimate(StrictModel):
    split: Literal["discovery", "validation", "heldout"] = "discovery"
    condition: Literal["original", "perturbed"]
    strength: float
    saturation_effect: float = Field(ge=0)
    width_at_90_percent: PositiveInt
    monotonic: bool


class InterventionRunSummary(StrictModel):
    schema_version: Literal["probe.intervention-result/v1"] = (
        "probe.intervention-result/v1"
    )
    science_hash: str
    parent_run_id: str
    rank_run_id: str | None = None
    candidate_score_method: Literal[
        "direct_structural",
        "downstream_endpoint_gradient",
        "direct_downstream_overlap",
    ] = "direct_structural"
    qualification_run_id: str | None = None
    trajectory_run_id: str | None = None
    model: dict[str, Any]
    observable: dict[str, Any]
    operation: InterventionOperationRequest
    selection: NeuronSelectionRequest
    selected_neurons: tuple[SelectedNeuron, ...]
    pairs: tuple[str, ...]
    split_counts: dict[str, int] = Field(default_factory=dict)
    observations: tuple[InterventionObservation, ...]
    trajectory_overlays: tuple[InterventionTrajectoryCheckpoint, ...] = ()
    doses: tuple[InterventionDoseSummary, ...]
    additivity: tuple[AdditivityViolation, ...] = ()
    causal_width: tuple[CausalWidthEstimate, ...] = ()
    logical_forward_passes: PositiveInt
    evidence_stage: Literal["causal_intervention"] = "causal_intervention"
    claims: tuple[ClaimRecord, ...] = ()
    warnings: tuple[str, ...] = ()


class DirectionObservation(StrictModel):
    pair_id: str
    split: Literal["discovery", "validation", "heldout"] = "discovery"
    arm: Literal["behavioral_direction", "matched_random_direction"]
    control_sample: int | None = Field(default=None, ge=0)
    condition: Literal["original", "perturbed"]
    layer: int = Field(ge=0)
    beta: float
    normalization: Literal["raw", "residual_norm"]
    baseline_gap: float
    intervention_gap: float
    gap_effect: float
    baseline_prediction: str
    intervention_prediction: str
    baseline_generated_text: str | None = None
    generated_text: str | None = None
    baseline_behavior_decision: Literal[
        "target", "control", "other", "ambiguous"
    ] | None = None
    intervention_behavior_decision: Literal[
        "target", "control", "other", "ambiguous"
    ] | None = None
    collateral_gap_effects: dict[str, float] = Field(default_factory=dict)


class DirectionDoseSummary(StrictModel):
    split: Literal["discovery", "validation", "heldout"] = "discovery"
    condition: Literal["original", "perturbed"]
    layer: int = Field(ge=0)
    beta: float
    selected_effect_mean: float
    selected_absolute_effect_mean: float = Field(ge=0)
    random_absolute_effect_mean: float | None = Field(default=None, ge=0)
    controlled_absolute_effect: float | None = None
    bootstrap_low: float | None = None
    bootstrap_high: float | None = None
    pair_count: PositiveInt
    random_observation_count: int = Field(ge=0)


class DirectionInjectionRunSummary(StrictModel):
    schema_version: Literal["probe.direction-result/v1"] = (
        "probe.direction-result/v1"
    )
    science_hash: str
    parent_run_id: str
    qualification_run_id: str | None = None
    model: dict[str, Any]
    observable: dict[str, Any]
    layers: tuple[int, ...]
    betas: tuple[float, ...]
    normalization: Literal["raw", "residual_norm"]
    behavioral_direction_norm: float = Field(gt=0)
    ffn_skip_mean: float | None = None
    observations: tuple[DirectionObservation, ...]
    doses: tuple[DirectionDoseSummary, ...]
    pairs: tuple[str, ...]
    split_counts: dict[str, int] = Field(default_factory=dict)
    logical_forward_passes: PositiveInt
    evidence_stage: Literal["causal_intervention"] = "causal_intervention"
    claims: tuple[ClaimRecord, ...] = ()
    warnings: tuple[str, ...] = ()


class AttentionHeadScore(StrictModel):
    rank: PositiveInt
    layer: int = Field(ge=0)
    head: int = Field(ge=0)
    direct_effect_mean: float
    direct_effect_rms: float = Field(ge=0)
    sign_consistency: float = Field(ge=0, le=1)
    original_output_norm_mean: float = Field(ge=0)
    perturbed_output_norm_mean: float = Field(ge=0)
    output_delta_norm_mean: float = Field(ge=0)


class AttentionLayerSummary(StrictModel):
    layer: int = Field(ge=0)
    signed_effect_sum: float
    rms_mass: float = Field(ge=0)
    positive_mean_mass: float = Field(ge=0)
    negative_mean_mass: float = Field(ge=0)
    top_head: int = Field(ge=0)
    maximum_head_rms: float = Field(ge=0)


class AttentionPairSummary(StrictModel):
    pair_id: str
    split: Literal["discovery", "validation", "heldout"] = "discovery"
    original_gap: float
    perturbed_gap: float
    measured_delta: float
    predicted_attention_delta: float
    original_token_count: PositiveInt
    perturbed_token_count: PositiveInt


class AttentionHeadRankRunSummary(StrictModel):
    schema_version: Literal["probe.attention-rank-result/v1"] = (
        "probe.attention-rank-result/v1"
    )
    science_hash: str
    parent_run_id: str
    qualification_run_id: str | None = None
    model: dict[str, Any]
    observable: dict[str, Any]
    pair_count: PositiveInt
    pairs: tuple[AttentionPairSummary, ...]
    layers: tuple[AttentionLayerSummary, ...]
    heads: tuple[AttentionHeadScore, ...]
    total_head_count: PositiveInt
    output_head_count: PositiveInt
    key_value_head_count: PositiveInt
    head_dim: PositiveInt
    logical_forward_passes: PositiveInt
    evidence_stage: Literal["attention_hypothesis"] = "attention_hypothesis"
    claims: tuple[ClaimRecord, ...] = ()
    warnings: tuple[str, ...] = ()


class SelectedAttentionHead(StrictModel):
    rank: PositiveInt | None = None
    layer: int = Field(ge=0)
    head: int = Field(ge=0)
    direct_effect_mean: float | None = None
    direct_effect_rms: float | None = Field(default=None, ge=0)
    sign_consistency: float | None = Field(default=None, ge=0, le=1)


class AttentionInterventionObservation(StrictModel):
    pair_id: str
    split: Literal["discovery", "validation", "heldout"] = "discovery"
    arm: Literal["selected", "matched_random"]
    control_sample: int | None = Field(default=None, ge=0)
    condition: Literal["original", "perturbed"]
    mode: Literal["ablate", "amplify", "patch", "restore"]
    head_count: PositiveInt
    strength: float
    baseline_gap: float
    source_gap: float | None = None
    intervention_gap: float
    gap_effect: float
    normalized_source_progress: float | None = None
    baseline_prediction: str
    intervention_prediction: str


class AttentionInterventionDoseSummary(StrictModel):
    split: Literal["discovery", "validation", "heldout"] = "discovery"
    condition: Literal["original", "perturbed"]
    head_count: PositiveInt
    strength: float
    selected_effect_mean: float
    selected_absolute_effect_mean: float = Field(ge=0)
    random_absolute_effect_mean: float | None = Field(default=None, ge=0)
    controlled_absolute_effect: float | None = None
    pair_count: PositiveInt
    random_observation_count: int = Field(ge=0)


class AttentionHeadInterventionRunSummary(StrictModel):
    schema_version: Literal["probe.attention-intervention-result/v1"] = (
        "probe.attention-intervention-result/v1"
    )
    science_hash: str
    parent_run_id: str
    rank_run_id: str
    qualification_run_id: str | None = None
    model: dict[str, Any]
    observable: dict[str, Any]
    operation: AttentionInterventionOperationRequest
    selection: AttentionHeadSelectionRequest
    selected_heads: tuple[SelectedAttentionHead, ...]
    pairs: tuple[str, ...]
    split_counts: dict[str, int] = Field(default_factory=dict)
    observations: tuple[AttentionInterventionObservation, ...]
    doses: tuple[AttentionInterventionDoseSummary, ...]
    logical_forward_passes: PositiveInt
    evidence_stage: Literal["attention_causal_heads"] = "attention_causal_heads"
    claims: tuple[ClaimRecord, ...] = ()
    warnings: tuple[str, ...] = ()


class AttentionTokenEdge(StrictModel):
    pair_id: str
    condition: Literal["original", "perturbed"]
    layer: int = Field(ge=0)
    head: int = Field(ge=0)
    key_value_head: int = Field(ge=0)
    source_position: int = Field(ge=0)
    source_token_id: int = Field(ge=0)
    source_token: str
    attention_weight: float = Field(ge=0, le=1)
    direct_effect: float
    output_norm: float = Field(ge=0)


class AttentionPathObservation(StrictModel):
    pair_id: str
    split: Literal["discovery", "validation", "heldout"] = "discovery"
    arm: Literal["selected_path", "matched_random_path"]
    control_sample: int | None = Field(default=None, ge=0)
    operation: Literal["patch", "restore"]
    sender: AttentionHeadReference
    receiver: AttentionHeadReference
    baseline_gap: float
    source_gap: float
    sender_patched_gap: float
    path_patched_gap: float
    sender_total_effect: float
    path_specific_effect: float
    normalized_source_progress: float | None = None
    alignment_mode: Literal["identity", "explicit"]


class AttentionTraceRunSummary(StrictModel):
    schema_version: Literal["probe.attention-trace-result/v1"] = (
        "probe.attention-trace-result/v1"
    )
    science_hash: str
    parent_run_id: str
    rank_run_id: str
    parent_intervention_run_id: str | None = None
    model: dict[str, Any]
    observable: dict[str, Any]
    trace_kind: Literal["token_edges", "head_paths"]
    pairs: tuple[str, ...]
    token_edges: tuple[AttentionTokenEdge, ...] = ()
    paths: tuple[AttentionPathObservation, ...] = ()
    logical_forward_passes: PositiveInt
    evidence_stage: Literal["attention_hypothesis", "attention_causal_paths"]
    claims: tuple[ClaimRecord, ...] = ()
    warnings: tuple[str, ...] = ()


class RunComparison(StrictModel):
    run_id: str
    top_n: PositiveInt
    overlap_fraction: float = Field(ge=0, le=1)
    sign_agreement: float = Field(ge=0, le=1)
    mean_rank_displacement: float = Field(ge=0)
    ffn_skip_difference: float | None = None
    changed_factors: tuple[str, ...] = ()


class ComparisonReport(StrictModel):
    schema_version: Literal["probe.comparison/v1"] = "probe.comparison/v1"
    reference_run_id: str
    comparisons: tuple[RunComparison, ...]
    scientific_replication: bool
    warnings: tuple[str, ...] = ()


class StabilityReport(StrictModel):
    schema_version: Literal["probe.stability/v1"] = "probe.stability/v1"
    run_id: str
    pair_count: PositiveInt
    top_n: PositiveInt
    split_count: int = Field(ge=0)
    mean_top_n_overlap: float | None = Field(default=None, ge=0, le=1)
    minimum_top_n_overlap: float | None = Field(default=None, ge=0, le=1)
    mean_sign_agreement: float | None = Field(default=None, ge=0, le=1)
    bootstrap_iterations: int = Field(ge=0)
    neuron_intervals: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()


class SensitivityComparison(StrictModel):
    first_group: str
    second_group: str
    first_pair_count: PositiveInt
    second_pair_count: PositiveInt
    top_n_overlap: float = Field(ge=0, le=1)
    sign_agreement: float = Field(ge=0, le=1)


class SensitivityReport(StrictModel):
    schema_version: Literal["probe.sensitivity/v1"] = "probe.sensitivity/v1"
    run_id: str
    metadata_key: str
    groups: dict[str, tuple[str, ...]]
    comparisons: tuple[SensitivityComparison, ...]
    top_n: PositiveInt
    warnings: tuple[str, ...] = ()


class QueryEnvelope(StrictModel):
    schema_version: Literal["probe.query/v1"] = "probe.query/v1"
    run_id: str
    query: Literal["layers", "neurons", "files"]
    parameters: dict[str, Any] = Field(default_factory=dict)
    sort: str | None = None
    source_count: int = Field(ge=0)
    matched_count: int = Field(ge=0)
    returned_count: int = Field(ge=0)
    items: tuple[dict[str, Any], ...]


class VerificationReport(StrictModel):
    schema_version: Literal["probe.verification/v1"] = "probe.verification/v1"
    run_id: str
    valid: bool
    failures: tuple[str, ...] = ()


def _validate_relative_bundle_path(value: str) -> str:
    candidate = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or candidate.is_absolute()
        or str(candidate) != value
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError("replay paths must be normalized relative POSIX paths")
    return value


class ReplayExecutionRecord(StrictModel):
    torch_seed: int
    python_seed: int | None = None
    numpy_seed: int | None = None
    deterministic_algorithms: bool = False
    model_eval: Literal[True] = True
    inference_mode: Literal[True] = True
    use_cache: Literal[False] = False
    generation: Literal["none"] = "none"
    resolved_model_revision: str
    adapter: str
    device: str
    model_dtype: str
    capture_dtype: Literal["float32"] = "float32"
    expected_environment: dict[str, str] = Field(default_factory=dict)


class ReplayComparisonPolicy(StrictModel):
    scalar_absolute_tolerance: float = Field(default=1e-4, ge=0)
    scalar_relative_tolerance: float = Field(default=1e-4, ge=0)
    ranking_top_n: PositiveInt = 50
    minimum_top_n_overlap: float = Field(default=0.9, ge=0, le=1)
    minimum_sign_agreement: float = Field(default=0.95, ge=0, le=1)
    maximum_mean_rank_displacement: float = Field(default=5.0, ge=0)
    require_exact_predictions: bool = True
    artifact_hashes: Literal["report", "require"] = "report"


class ReplayDriver(StrictModel):
    schema_version: Literal["probe.replay-driver/v1"] = "probe.replay-driver/v1"
    name: str
    description: str | None = None
    spec: str
    baseline: str = "baseline.json"
    report_directory: str = "reports"
    reproducibility: ReplayExecutionRecord
    comparison: ReplayComparisonPolicy = Field(default_factory=ReplayComparisonPolicy)

    @field_validator("name")
    @classmethod
    def validate_replay_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("replay name must not be blank")
        return value

    @field_validator("spec", "baseline", "report_directory")
    @classmethod
    def validate_replay_paths(cls, value: str) -> str:
        return _validate_relative_bundle_path(value)


class ReplayBaseline(StrictModel):
    schema_version: Literal["probe.replay-baseline/v1"] = "probe.replay-baseline/v1"
    driver_name: str
    recorded_at: datetime
    source_run_id: str
    science_hash: str
    request_hash: str
    run_fingerprint: str
    algorithm_version: str
    resolved_model: dict[str, Any]
    environment: dict[str, str]
    artifact_hashes: dict[str, str]
    summary: dict[str, Any]


class ReplayCheck(StrictModel):
    name: str
    passed: bool
    required: bool = True
    baseline: Any = None
    replay: Any = None
    detail: str | None = None


class ReplayReport(StrictModel):
    schema_version: Literal["probe.replay-report/v1"] = "probe.replay-report/v1"
    driver_name: str
    baseline_run_id: str
    replay_run_id: str
    created_at: datetime
    verdict: Literal["passed", "failed"]
    checks: tuple[ReplayCheck, ...]
    ranking: dict[str, Any]
    numeric: dict[str, Any]
    artifact_hashes: dict[str, Any]
    report_files: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class ReplayOutcome(StrictModel):
    schema_version: Literal["probe.replay-outcome/v1"] = "probe.replay-outcome/v1"
    driver_name: str
    baseline_run_id: str
    replay_run_id: str
    verdict: Literal["passed", "failed"]
    required_checks_passed: int = Field(ge=0)
    required_checks_total: int = Field(ge=0)
    numeric_metrics_passed: int = Field(ge=0)
    numeric_metrics_total: int = Field(ge=0)
    neuron_metrics_passed: int = Field(ge=0)
    neuron_metrics_total: int = Field(ge=0)
    maximum_absolute_difference: float = Field(ge=0)
    top_n: int = Field(ge=0)
    top_n_overlap: float = Field(ge=0, le=1)
    sign_agreement: float = Field(ge=0, le=1)
    mean_rank_displacement: float = Field(ge=0)
    stable_artifact_hashes_matched: bool
    report_files: tuple[str, ...]
    warnings: tuple[str, ...] = ()


class ReplayIdentity(StrictModel):
    schema_version: Literal["probe.replay-identity/v1"] = "probe.replay-identity/v1"
    driver_name: str
    driver: str
    spec: str
    baseline: str
    baseline_exists: bool
    science_hash: str
    request_hash: str
    reproducibility: ReplayExecutionRecord
    comparison: ReplayComparisonPolicy


class ReplayRecordReceipt(StrictModel):
    schema_version: Literal["probe.replay-record/v1"] = "probe.replay-record/v1"
    driver_name: str
    baseline: str
    source_run_id: str
    science_hash: str
    request_hash: str
    ranking_top_n: PositiveInt


class WorkflowStageOutcome(StrictModel):
    kind: Literal[
        "rank",
        "trajectory",
        "ffn_coupling",
        "qualify",
        "intervention",
        "direction",
        "attention_rank",
        "attention_intervention",
        "attention_trace",
    ]
    name: str
    run_id: str
    parent_run_ids: tuple[str, ...] = ()
    evidence_stage: str
    logical_forward_passes: PositiveInt
    claims: tuple[ClaimRecord, ...] = ()
    warnings: tuple[str, ...] = ()


class ResearchWorkflowOutcome(StrictModel):
    schema_version: Literal["probe.workflow-outcome/v1"] = (
        "probe.workflow-outcome/v1"
    )
    workflow_id: str
    name: str
    stages: tuple[WorkflowStageOutcome, ...]
    rank_run_id: str
    qualification_run_id: str | None = None
    trajectory_run_id: str | None = None
    ffn_coupling_run_id: str | None = None
    intervention_run_ids: tuple[str, ...] = ()
    direction_run_ids: tuple[str, ...] = ()
    attention_rank_run_id: str | None = None
    attention_intervention_run_ids: tuple[str, ...] = ()
    attention_trace_run_ids: tuple[str, ...] = ()
    logical_forward_passes: PositiveInt
    claims: tuple[ClaimRecord, ...] = ()
    warnings: tuple[str, ...] = ()


class ResearchReport(StrictModel):
    schema_version: Literal["probe.research-report/v1"] = (
        "probe.research-report/v1"
    )
    run_id: str
    run_kind: Literal[
        "rank",
        "trajectory",
        "ffn_coupling",
        "qualify",
        "intervention",
        "direction",
        "attention_rank",
        "attention_intervention",
        "attention_trace",
    ]
    evidence_stage: str
    parent_run_ids: tuple[str, ...] = ()
    headline: str
    key_results: tuple[str, ...]
    claims: tuple[ClaimRecord, ...] = ()
    limitations: tuple[str, ...] = ()
    recommended_next_steps: tuple[str, ...] = ()


class ReportReceipt(StrictModel):
    schema_version: Literal["probe.report-receipt/v1"] = "probe.report-receipt/v1"
    run_id: str
    json_path: str
    markdown_path: str
    json_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    markdown_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ErrorDetail(StrictModel):
    code: str
    message: str
    retryable: bool = False
    hint: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(StrictModel):
    schema_version: Literal["probe.error/v1"] = "probe.error/v1"
    error: ErrorDetail


class JobEvent(StrictModel):
    schema_version: Literal["probe.event/v1"] = EVENT_SCHEMA_VERSION
    event: str
    sequence: int = Field(ge=0)
    timestamp: datetime
    job_id: str
    request_id: str
    science_hash: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ArtifactRef(StrictModel):
    path: str
    media_type: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)

    @field_validator("path")
    @classmethod
    def validate_portable_relative_path(cls, value: str) -> str:
        candidate = PurePosixPath(value)
        if (
            not value
            or "\\" in value
            or candidate.is_absolute()
            or str(candidate) != value
            or any(part in {"", ".", ".."} for part in candidate.parts)
        ):
            raise ValueError("artifact path must be a normalized relative POSIX path")
        return value


class RunManifest(StrictModel):
    schema_version: Literal["probe.run/v1"] = MANIFEST_SCHEMA_VERSION
    run_id: str
    job_id: str
    request_id: str
    science_hash: str
    run_fingerprint: str
    created_at: datetime
    completed_at: datetime
    evidence_stage: Literal[
        "exploratory_pair",
        "replicated_ranking",
        "observational_trajectory",
        "observational_ffn_coupling",
        "qualified_observable",
        "causal_intervention",
        "attention_hypothesis",
        "attention_causal_heads",
        "attention_causal_paths",
        "generalized",
    ]
    run_kind: Literal[
        "rank",
        "trajectory",
        "ffn_coupling",
        "qualify",
        "intervention",
        "direction",
        "attention_rank",
        "attention_intervention",
        "attention_trace",
    ] = "rank"
    parent_run_ids: tuple[str, ...] = ()
    algorithm_version: str
    requested_model: ModelRequest
    resolved_model: dict[str, Any]
    environment: dict[str, str] = Field(default_factory=dict)
    pair_count: PositiveInt
    artifacts: tuple[ArtifactRef, ...]
    warnings: tuple[str, ...] = ()


class JobStatus(StrictModel):
    schema_version: Literal["probe.job/v1"] = "probe.job/v1"
    job_id: str
    request_id: str
    request_hash: str | None = None
    science_hash: str
    state: Literal["queued", "running", "completed", "failed", "cancelled"]
    created_at: datetime
    updated_at: datetime
    run_id: str | None = None
    error: ErrorDetail | None = None
