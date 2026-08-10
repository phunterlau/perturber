from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import random
from statistics import fmean

import torch

from .adapters import AttentionHeadEdit
from .contracts import (
    AttentionHeadInterventionRunSummary,
    AttentionHeadInterventionSpec,
    AttentionHeadRankRunSummary,
    AttentionHeadRankSpec,
    AttentionHeadScore,
    AttentionInterventionDoseSummary,
    AttentionInterventionObservation,
    AttentionLayerSummary,
    AttentionPairSummary,
    ClaimRecord,
    PairResultSummary,
    RankRunSummary,
    RankSpec,
    SelectedAttentionHead,
)
from .domain import ObservableSpec
from .engine import ProbeEngine
from .observables import resolve_observable
from .prompting import prepare_pair_condition
from .scoring import logit_gap


@dataclass(frozen=True)
class AttentionRankComputation:
    summary: AttentionHeadRankRunSummary
    tensors: dict[str, torch.Tensor]


def _prediction(engine: ProbeEngine, logits: torch.Tensor) -> str:
    token_id = int(torch.argmax(logits.detach().float().cpu()).item())
    return engine.adapter.tokenizer.decode([token_id])


def _eligible_parent_pairs(
    *,
    parent_spec: RankSpec,
    parent_summary: RankRunSummary,
    pair_ids: tuple[str, ...],
    include_weak_pairs: bool,
    qualification_statuses: dict[str, str] | None,
    stage: str,
) -> tuple[tuple[int, object, PairResultSummary], ...]:
    summary_by_id = {item.pair_id: item for item in parent_summary.pairs}
    spec_by_id = {
        item.id: (index, item) for index, item in enumerate(parent_spec.pairs)
    }
    requested = pair_ids or tuple(spec_by_id)
    if len(requested) != len(set(requested)):
        raise ValueError(f"{stage} pair IDs must be unique")
    unknown = sorted(set(requested) - set(spec_by_id))
    if unknown:
        raise ValueError(f"{stage} pair IDs were not found in parent run: {unknown}")
    selected: list[tuple[int, object, PairResultSummary]] = []
    for pair_id in requested:
        index, pair = spec_by_id[pair_id]
        pair_summary = summary_by_id[pair_id]
        qualification = pair_summary.qualification
        status = (
            qualification_statuses.get(pair_id)
            if qualification_statuses is not None
            else qualification.status if qualification is not None else None
        )
        eligible = status is None or status == "informative"
        if include_weak_pairs and status is not None:
            eligible = status in {"informative", "weak"}
        if eligible:
            selected.append((index, pair, pair_summary))
    if not selected:
        raise ValueError(
            f"no requested pairs are eligible for {stage}; qualify the observable or "
            "set include_weak_pairs=true for an explicitly exploratory run"
        )
    return tuple(selected)


def attention_rank_plan_counts(
    *,
    parent_spec: RankSpec,
    parent_summary: RankRunSummary,
    spec: AttentionHeadRankSpec,
    qualification_statuses: dict[str, str] | None = None,
) -> tuple[int, int]:
    pairs = _eligible_parent_pairs(
        parent_spec=parent_spec,
        parent_summary=parent_summary,
        pair_ids=spec.pair_ids,
        include_weak_pairs=spec.include_weak_pairs,
        qualification_statuses=qualification_statuses,
        stage="attention ranking",
    )
    discovery_count = sum(pair.split == "discovery" for _, pair, _ in pairs)
    if discovery_count == 0:
        raise ValueError("attention ranking requires at least one discovery pair")
    expected = "single_pair" if discovery_count == 1 else "rms"
    if spec.ranking.pair_aggregation != expected:
        raise ValueError(
            f"{discovery_count} discovery pair(s) require "
            f"pair_aggregation={expected!r}"
        )
    return len(pairs), 2 * len(pairs)


def run_attention_rank(
    *,
    engine: ProbeEngine,
    parent_spec: RankSpec,
    parent_summary: RankRunSummary,
    spec: AttentionHeadRankSpec,
    science_hash: str,
    qualification_statuses: dict[str, str] | None = None,
) -> AttentionRankComputation:
    pairs = _eligible_parent_pairs(
        parent_spec=parent_spec,
        parent_summary=parent_summary,
        pair_ids=spec.pair_ids,
        include_weak_pairs=spec.include_weak_pairs,
        qualification_statuses=qualification_statuses,
        stage="attention ranking",
    )
    discovery_indices = tuple(
        index for index, (_, pair, _) in enumerate(pairs) if pair.split == "discovery"
    )
    if not discovery_indices:
        raise ValueError("attention ranking requires at least one discovery pair")
    expected = "single_pair" if len(discovery_indices) == 1 else "rms"
    if spec.ranking.pair_aggregation != expected:
        raise ValueError(
            f"{len(discovery_indices)} discovery pair(s) require "
            f"pair_aggregation={expected!r}"
        )

    observable = resolve_observable(
        engine.adapter.tokenizer,
        ObservableSpec(
            name=parent_spec.observable.name,
            target_tokens=parent_spec.observable.target_tokens,
            control_tokens=parent_spec.observable.control_tokens,
        ),
    )
    direction = engine.adapter.behavioral_direction(observable)
    couplings = engine.adapter.attention_output_couplings(direction)
    metadata = engine.adapter.attention_metadata()
    if len(couplings) != metadata.layer_count:
        raise RuntimeError("attention coupling layer count does not match metadata")

    tensors: dict[str, torch.Tensor] = {}
    importance_by_pair: list[tuple[torch.Tensor, ...]] = []
    original_by_pair: list[tuple[torch.Tensor, ...]] = []
    perturbed_by_pair: list[tuple[torch.Tensor, ...]] = []
    pair_summaries: list[AttentionPairSummary] = []

    for parent_index, pair, parent_pair_summary in pairs:
        original_ids, original_tokens = prepare_pair_condition(
            engine.adapter,
            pair=pair,
            model=parent_spec.model,
            condition="original",
        )
        perturbed_ids, perturbed_tokens = prepare_pair_condition(
            engine.adapter,
            pair=pair,
            model=parent_spec.model,
            condition="perturbed",
        )
        original = engine.adapter.forward_attention_capture(
            original_ids, original_tokens, parent_spec.capture.position
        )
        perturbed = engine.adapter.forward_attention_capture(
            perturbed_ids, perturbed_tokens, parent_spec.capture.position
        )
        local_importance: list[torch.Tensor] = []
        local_original: list[torch.Tensor] = []
        local_perturbed: list[torch.Tensor] = []
        for layer in range(metadata.layer_count):
            original_full = original.head_outputs[layer]
            perturbed_full = perturbed.head_outputs[layer]
            if original_full is None or perturbed_full is None:
                raise RuntimeError(f"missing attention output for layer {layer}")
            original_head = original_full[-1].float()
            perturbed_head = perturbed_full[-1].float()
            if original_head.shape != (
                metadata.output_head_count,
                metadata.head_dim,
            ) or perturbed_head.shape != original_head.shape:
                raise RuntimeError(f"invalid attention output shape at layer {layer}")
            importance = (
                couplings[layer].float() * (perturbed_head - original_head)
            ).sum(dim=-1)
            local_importance.append(importance)
            local_original.append(original_head)
            local_perturbed.append(perturbed_head)
            prefix = f"pair_{parent_index}.layer_{layer}"
            tensors[f"attention_importance.{prefix}"] = importance.contiguous()
            tensors[f"head_output_original.{prefix}"] = original_head.contiguous()
            tensors[f"head_output_perturbed.{prefix}"] = perturbed_head.contiguous()
            tensors[f"attention_coupling.layer_{layer}"] = couplings[
                layer
            ].float().contiguous()
        original_gap = logit_gap(
            original.logits, observable.target_ids, observable.control_ids
        )
        perturbed_gap = logit_gap(
            perturbed.logits, observable.target_ids, observable.control_ids
        )
        pair_summaries.append(
            AttentionPairSummary(
                pair_id=pair.id,
                split=pair.split,
                original_gap=original_gap,
                perturbed_gap=perturbed_gap,
                measured_delta=perturbed_gap - original_gap,
                predicted_attention_delta=float(
                    sum(value.sum().item() for value in local_importance)
                ),
                original_token_count=len(original.tokenized.input_ids),
                perturbed_token_count=len(perturbed.tokenized.input_ids),
            )
        )
        importance_by_pair.append(tuple(local_importance))
        original_by_pair.append(tuple(local_original))
        perturbed_by_pair.append(tuple(local_perturbed))

    mean_by_layer: list[torch.Tensor] = []
    rms_by_layer: list[torch.Tensor] = []
    consistency_by_layer: list[torch.Tensor] = []
    original_norm_by_layer: list[torch.Tensor] = []
    perturbed_norm_by_layer: list[torch.Tensor] = []
    delta_norm_by_layer: list[torch.Tensor] = []
    layer_summaries: list[AttentionLayerSummary] = []
    for layer in range(metadata.layer_count):
        importance = torch.stack(
            [importance_by_pair[index][layer] for index in discovery_indices]
        )
        original = torch.stack(
            [original_by_pair[index][layer] for index in discovery_indices]
        )
        perturbed = torch.stack(
            [perturbed_by_pair[index][layer] for index in discovery_indices]
        )
        mean = importance.mean(dim=0)
        rms = torch.sqrt(importance.square().mean(dim=0))
        mean_sign = torch.sign(mean)
        consistency = (torch.sign(importance) == mean_sign.unsqueeze(0)).float().mean(0)
        consistency = torch.where(
            mean_sign == 0, torch.zeros_like(consistency), consistency
        )
        original_norm = torch.linalg.vector_norm(original, dim=-1).mean(dim=0)
        perturbed_norm = torch.linalg.vector_norm(perturbed, dim=-1).mean(dim=0)
        delta_norm = torch.linalg.vector_norm(perturbed - original, dim=-1).mean(dim=0)
        top_head = int(torch.argmax(rms).item())
        layer_summaries.append(
            AttentionLayerSummary(
                layer=layer,
                signed_effect_sum=float(mean.sum().item()),
                rms_mass=float(rms.sum().item()),
                positive_mean_mass=float(mean.clamp_min(0).sum().item()),
                negative_mean_mass=float((-mean.clamp_max(0)).sum().item()),
                top_head=top_head,
                maximum_head_rms=float(rms[top_head].item()),
            )
        )
        mean_by_layer.append(mean)
        rms_by_layer.append(rms)
        consistency_by_layer.append(consistency)
        original_norm_by_layer.append(original_norm)
        perturbed_norm_by_layer.append(perturbed_norm)
        delta_norm_by_layer.append(delta_norm)

    flattened = torch.cat(rms_by_layer)
    chosen = min(spec.ranking.top_k, flattened.numel())
    order = torch.argsort(flattened, descending=True, stable=True)[:chosen].tolist()
    head_scores: list[AttentionHeadScore] = []
    for rank, flat_index in enumerate(order, start=1):
        layer = flat_index // metadata.output_head_count
        head = flat_index % metadata.output_head_count
        head_scores.append(
            AttentionHeadScore(
                rank=rank,
                layer=layer,
                head=head,
                direct_effect_mean=float(mean_by_layer[layer][head].item()),
                direct_effect_rms=float(rms_by_layer[layer][head].item()),
                sign_consistency=float(consistency_by_layer[layer][head].item()),
                original_output_norm_mean=float(
                    original_norm_by_layer[layer][head].item()
                ),
                perturbed_output_norm_mean=float(
                    perturbed_norm_by_layer[layer][head].item()
                ),
                output_delta_norm_mean=float(delta_norm_by_layer[layer][head].item()),
            )
        )

    warnings = (
        "Attention head ranking is direct-logit observational attribution; it is not a causal path claim.",
        "Head output differences can include upstream effects and are not independent across layers.",
        "Only the first generated-token decision is analyzed.",
    )
    summary = AttentionHeadRankRunSummary(
        science_hash=science_hash,
        parent_run_id=spec.parent_run_id,
        qualification_run_id=spec.qualification_run_id,
        model=asdict(engine.adapter.metadata),
        observable=parent_summary.observable,
        pair_count=len(pairs),
        pairs=tuple(pair_summaries),
        layers=tuple(layer_summaries),
        heads=tuple(head_scores),
        total_head_count=metadata.layer_count * metadata.output_head_count,
        output_head_count=metadata.output_head_count,
        key_value_head_count=metadata.key_value_head_count,
        head_dim=metadata.head_dim,
        logical_forward_passes=2 * len(pairs),
        claims=(
            ClaimRecord(
                claim_id="attention-head-hypothesis",
                claim_type="attention_routing",
                status="exploratory",
                statement=(
                    "Direct-logit head scores identify output heads for controlled intervention."
                ),
                limitations=warnings,
            ),
        ),
        warnings=warnings,
    )
    return AttentionRankComputation(summary=summary, tensors=tensors)


def _select_heads(
    summary: AttentionHeadRankRunSummary,
    spec: AttentionHeadInterventionSpec,
) -> tuple[SelectedAttentionHead, ...]:
    ranked = {(item.layer, item.head): item for item in summary.heads}
    request = spec.selection
    if request.strategy == "explicit":
        return tuple(
            SelectedAttentionHead(
                rank=(source.rank if source is not None else None),
                layer=item.layer,
                head=item.head,
                direct_effect_mean=(
                    source.direct_effect_mean if source is not None else None
                ),
                direct_effect_rms=(
                    source.direct_effect_rms if source is not None else None
                ),
                sign_consistency=(
                    source.sign_consistency if source is not None else None
                ),
            )
            for item in request.explicit
            for source in (ranked.get((item.layer, item.head)),)
        )
    candidates = list(summary.heads)
    if request.layers:
        allowed = set(request.layers)
        candidates = [item for item in candidates if item.layer in allowed]
    if request.sign == "positive":
        candidates = [item for item in candidates if item.direct_effect_mean > 0]
    elif request.sign == "negative":
        candidates = [item for item in candidates if item.direct_effect_mean < 0]
    candidates = [
        item
        for item in candidates
        if item.sign_consistency >= request.min_sign_consistency
    ]
    assert request.top_k is not None
    if len(candidates) < request.top_k:
        raise ValueError(
            f"selection requested {request.top_k} heads but only {len(candidates)} "
            "satisfy the filters"
        )
    return tuple(
        SelectedAttentionHead(
            rank=item.rank,
            layer=item.layer,
            head=item.head,
            direct_effect_mean=item.direct_effect_mean,
            direct_effect_rms=item.direct_effect_rms,
            sign_consistency=item.sign_consistency,
        )
        for item in candidates[: request.top_k]
    )


def _attention_conditions(spec: AttentionHeadInterventionSpec) -> tuple[str, ...]:
    condition = spec.operation.condition
    if condition == "auto":
        if spec.operation.mode == "patch":
            return ("original",)
        if spec.operation.mode == "restore":
            return ("perturbed",)
        return ("original", "perturbed")
    if condition == "both":
        return ("original", "perturbed")
    return (condition,)


def attention_intervention_plan_counts(
    *,
    rank_spec: RankSpec,
    rank_summary: RankRunSummary,
    attention_summary: AttentionHeadRankRunSummary,
    spec: AttentionHeadInterventionSpec,
    qualification_statuses: dict[str, str] | None = None,
) -> tuple[int, int]:
    pairs = _eligible_parent_pairs(
        parent_spec=rank_spec,
        parent_summary=rank_summary,
        pair_ids=spec.pair_ids,
        include_weak_pairs=spec.include_weak_pairs,
        qualification_statuses=qualification_statuses,
        stage="attention intervention",
    )
    selected = _select_heads(attention_summary, spec)
    if max(spec.sweep.head_counts) > len(selected):
        raise ValueError("largest head dose exceeds the selected head population")
    selected_calls = (
        len(pairs)
        * len(_attention_conditions(spec))
        * len(spec.sweep.head_counts)
        * len(spec.sweep.strengths)
    )
    return len(pairs), selected_calls * (1 + spec.controls.samples)


def _heads_by_layer(
    heads: tuple[SelectedAttentionHead, ...],
) -> dict[int, tuple[int, ...]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for item in heads:
        grouped[item.layer].append(item.head)
    return {layer: tuple(values) for layer, values in grouped.items()}


def _head_edits(
    *,
    selected: tuple[SelectedAttentionHead, ...],
    mode: str,
    strength: float,
    tensors: dict[str, torch.Tensor],
    parent_pair_index: int,
) -> tuple[AttentionHeadEdit, ...]:
    edits: list[AttentionHeadEdit] = []
    for layer, heads in _heads_by_layer(selected).items():
        if mode in {"ablate", "amplify"}:
            edits.append(
                AttentionHeadEdit(
                    layer=layer,
                    heads=heads,
                    operation="scale",
                    strength=strength,
                )
            )
            continue
        source_condition = "perturbed" if mode == "patch" else "original"
        key = (
            f"head_output_{source_condition}.pair_{parent_pair_index}.layer_{layer}"
        )
        if key not in tensors:
            raise ValueError(f"attention rank tensor artifact is missing {key!r}")
        head_indices = torch.tensor(heads, dtype=torch.long)
        source = tensors[key].index_select(0, head_indices).unsqueeze(0)
        edits.append(
            AttentionHeadEdit(
                layer=layer,
                heads=heads,
                operation="mix",
                strength=strength,
                source_values=source,
            )
        )
    return tuple(edits)


def _matched_random_heads(
    *,
    selected: tuple[SelectedAttentionHead, ...],
    excluded: set[tuple[int, int]],
    output_head_count: int,
    rng: random.Random,
) -> tuple[SelectedAttentionHead, ...]:
    result: list[SelectedAttentionHead] = []
    for layer, heads in _heads_by_layer(selected).items():
        candidates = [
            head
            for head in range(output_head_count)
            if (layer, head) not in excluded
        ]
        if len(candidates) < len(heads):
            raise ValueError(f"layer {layer} has too few heads for matched controls")
        result.extend(
            SelectedAttentionHead(layer=layer, head=head)
            for head in sorted(rng.sample(candidates, len(heads)))
        )
    return tuple(result)


def run_attention_intervention(
    *,
    engine: ProbeEngine,
    rank_spec: RankSpec,
    rank_summary: RankRunSummary,
    attention_summary: AttentionHeadRankRunSummary,
    attention_tensors: dict[str, torch.Tensor],
    spec: AttentionHeadInterventionSpec,
    science_hash: str,
    qualification_statuses: dict[str, str] | None = None,
) -> AttentionHeadInterventionRunSummary:
    selected_all = _select_heads(attention_summary, spec)
    if max(spec.sweep.head_counts) > len(selected_all):
        raise ValueError("largest head dose exceeds the selected head population")
    pairs = _eligible_parent_pairs(
        parent_spec=rank_spec,
        parent_summary=rank_summary,
        pair_ids=spec.pair_ids,
        include_weak_pairs=spec.include_weak_pairs,
        qualification_statuses=qualification_statuses,
        stage="attention intervention",
    )
    conditions = _attention_conditions(spec)
    observable = resolve_observable(
        engine.adapter.tokenizer,
        ObservableSpec(
            name=rank_spec.observable.name,
            target_tokens=rank_spec.observable.target_tokens,
            control_tokens=rank_spec.observable.control_tokens,
        ),
    )
    metadata = engine.adapter.attention_metadata()
    excluded = {(item.layer, item.head) for item in selected_all}
    rng = random.Random(spec.execution.seed)
    observations: list[AttentionInterventionObservation] = []
    model_calls = 0

    def run_one(
        *,
        parent_pair_index: int,
        pair: object,
        parent_pair_summary: PairResultSummary,
        condition: str,
        heads: tuple[SelectedAttentionHead, ...],
        strength: float,
        arm: str,
        control_sample: int | None,
    ) -> AttentionInterventionObservation:
        nonlocal model_calls
        baseline_gap = (
            parent_pair_summary.original_gap
            if condition == "original"
            else parent_pair_summary.perturbed_gap
        )
        baseline_prediction = (
            parent_pair_summary.original_prediction
            if condition == "original"
            else parent_pair_summary.perturbed_prediction
        )
        source_gap = None
        if spec.operation.mode in {"patch", "restore"}:
            source_gap = (
                parent_pair_summary.perturbed_gap
                if spec.operation.mode == "patch"
                else parent_pair_summary.original_gap
            )
        input_ids, tokenized = prepare_pair_condition(
            engine.adapter,
            pair=pair,
            model=rank_spec.model,
            condition=condition,
        )
        edits = _head_edits(
            selected=heads,
            mode=spec.operation.mode,
            strength=strength,
            tensors=attention_tensors,
            parent_pair_index=parent_pair_index,
        )
        capture = engine.adapter.forward_attention_capture(
            input_ids,
            tokenized,
            rank_spec.capture.position,
            layers=tuple(sorted({item.layer for item in heads})),
            edits=edits,
        )
        model_calls += 1
        intervention_gap = logit_gap(
            capture.logits, observable.target_ids, observable.control_ids
        )
        progress = None
        if source_gap is not None and abs(source_gap - baseline_gap) > 1e-8:
            progress = (intervention_gap - baseline_gap) / (
                source_gap - baseline_gap
            )
        return AttentionInterventionObservation(
            pair_id=pair.id,
            split=pair.split,
            arm=arm,
            control_sample=control_sample,
            condition=condition,
            mode=spec.operation.mode,
            head_count=len(heads),
            strength=strength,
            baseline_gap=baseline_gap,
            source_gap=source_gap,
            intervention_gap=intervention_gap,
            gap_effect=intervention_gap - baseline_gap,
            normalized_source_progress=progress,
            baseline_prediction=baseline_prediction,
            intervention_prediction=_prediction(engine, capture.logits),
        )

    for head_count in spec.sweep.head_counts:
        selected = selected_all[:head_count]
        for strength in spec.sweep.strengths:
            for parent_index, pair, parent_pair_summary in pairs:
                for condition in conditions:
                    observations.append(
                        run_one(
                            parent_pair_index=parent_index,
                            pair=pair,
                            parent_pair_summary=parent_pair_summary,
                            condition=condition,
                            heads=selected,
                            strength=strength,
                            arm="selected",
                            control_sample=None,
                        )
                    )
                    for sample in range(spec.controls.samples):
                        controls = _matched_random_heads(
                            selected=selected,
                            excluded=excluded,
                            output_head_count=metadata.output_head_count,
                            rng=rng,
                        )
                        observations.append(
                            run_one(
                                parent_pair_index=parent_index,
                                pair=pair,
                                parent_pair_summary=parent_pair_summary,
                                condition=condition,
                                heads=controls,
                                strength=strength,
                                arm="matched_random",
                                control_sample=sample,
                            )
                        )

    doses: list[AttentionInterventionDoseSummary] = []
    for split in ("discovery", "validation", "heldout"):
        for condition in conditions:
            for head_count in spec.sweep.head_counts:
                for strength in spec.sweep.strengths:
                    chosen = [
                        item
                        for item in observations
                        if item.split == split
                        and item.condition == condition
                        and item.head_count == head_count
                        and item.strength == strength
                    ]
                    selected_values = [
                        item.gap_effect for item in chosen if item.arm == "selected"
                    ]
                    if not selected_values:
                        continue
                    random_values = [
                        item.gap_effect
                        for item in chosen
                        if item.arm == "matched_random"
                    ]
                    selected_abs = fmean(abs(value) for value in selected_values)
                    random_abs = (
                        fmean(abs(value) for value in random_values)
                        if random_values
                        else None
                    )
                    doses.append(
                        AttentionInterventionDoseSummary(
                            split=split,
                            condition=condition,
                            head_count=head_count,
                            strength=strength,
                            selected_effect_mean=fmean(selected_values),
                            selected_absolute_effect_mean=selected_abs,
                            random_absolute_effect_mean=random_abs,
                            controlled_absolute_effect=(
                                selected_abs - random_abs
                                if random_abs is not None
                                else None
                            ),
                            pair_count=len(selected_values),
                            random_observation_count=len(random_values),
                        )
                    )

    controlled = [
        item.controlled_absolute_effect
        for item in doses
        if item.controlled_absolute_effect is not None
    ]
    supported = bool(controlled and max(controlled) > 0)
    selected_statuses = []
    for _parent_index, pair, pair_summary in pairs:
        status = (
            qualification_statuses.get(pair.id)
            if qualification_statuses is not None
            else (
                pair_summary.qualification.status
                if pair_summary.qualification is not None
                else None
            )
        )
        selected_statuses.append(status)
    weak_evidence = any(status not in {None, "informative"} for status in selected_statuses)
    warning_values = [
        "Head-output interventions are local to the selected prompts, first-token observable, and doses.",
        "A head can be causally influential without being a unique or semantically stable circuit component.",
    ]
    if weak_evidence:
        warning_values.append(
            "At least one included pair did not pass the informative-observable gate; causal-head results are exploratory."
        )
    warnings = tuple(warning_values)
    return AttentionHeadInterventionRunSummary(
        science_hash=science_hash,
        parent_run_id=spec.parent_run_id,
        rank_run_id=attention_summary.parent_run_id,
        qualification_run_id=spec.qualification_run_id,
        model=asdict(engine.adapter.metadata),
        observable=rank_summary.observable,
        operation=spec.operation,
        selection=spec.selection,
        selected_heads=selected_all,
        pairs=tuple(pair.id for _, pair, _ in pairs),
        split_counts={
            split: sum(pair.split == split for _, pair, _ in pairs)
            for split in ("discovery", "validation", "heldout")
            if any(pair.split == split for _, pair, _ in pairs)
        },
        observations=tuple(observations),
        doses=tuple(doses),
        logical_forward_passes=model_calls,
        claims=(
            ClaimRecord(
                claim_id="controlled-attention-head-effect",
                claim_type="attention_routing",
                status=(
                    "exploratory"
                    if supported and weak_evidence
                    else "supported"
                    if supported
                    else "not_supported"
                ),
                statement=(
                    "Selected attention output heads were compared with same-layer random controls."
                ),
                limitations=warnings,
            ),
        ),
        warnings=warnings,
    )


__all__ = [
    "AttentionRankComputation",
    "attention_intervention_plan_counts",
    "attention_rank_plan_counts",
    "run_attention_intervention",
    "run_attention_rank",
]
