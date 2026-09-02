from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

import torch

from .contracts import (
    AggregateLayerSummary,
    AggregateNeuronScore,
    ClaimRecord,
    PairResultSummary,
    QualificationAggregate,
    RankingObjective,
    RankRunSummary,
)
from .engine import ProbeAnalysis
from .qualification import qualify_pair_logits
from .scoring import classify_circuit


@dataclass(frozen=True)
class AggregateComputation:
    summary: RankRunSummary
    tensors: dict[str, torch.Tensor]


def aggregate_analyses(
    *,
    science_hash: str,
    pair_ids: tuple[str, ...],
    analyses: tuple[ProbeAnalysis, ...],
    top_k: int,
    pair_splits: tuple[str, ...] | None = None,
    pair_aggregation: str = "signed_mean",
) -> AggregateComputation:
    if not analyses or len(analyses) != len(pair_ids):
        raise ValueError("pair IDs and analyses must be non-empty and aligned")
    pair_splits = pair_splits or tuple("discovery" for _ in pair_ids)
    if len(pair_splits) != len(pair_ids):
        raise ValueError("pair splits must align with pair IDs")
    ranking_indices = tuple(
        index for index, split in enumerate(pair_splits) if split == "discovery"
    )
    if not ranking_indices:
        raise ValueError("at least one discovery pair is required for neuron ranking")
    ranking_analyses = tuple(analyses[index] for index in ranking_indices)
    if pair_aggregation not in {"single_pair", "signed_mean", "rms"}:
        raise ValueError(f"unsupported pair aggregation {pair_aggregation!r}")
    if len(ranking_analyses) > 1 and pair_aggregation == "single_pair":
        raise ValueError("multiple discovery pairs require signed_mean or rms aggregation")
    ranking_objective: RankingObjective = (
        "shared_direction" if pair_aggregation == "signed_mean" else "effect_magnitude"
    )

    layer_count = len(analyses[0].importance_by_layer)
    first = analyses[0]
    if layer_count == 0:
        raise ValueError("pair analyses must contain at least one layer")
    for pair_index, item in enumerate(analyses):
        counts = {
            len(item.importance_by_layer),
            len(item.coupling_by_layer),
            len(item.original_activation_by_layer),
            len(item.perturbed_activation_by_layer),
        }
        if counts != {layer_count}:
            raise ValueError(
                f"pair analysis {pair_index} has inconsistent layer counts"
            )
        if item.result.model != first.result.model:
            raise ValueError("all pair analyses must use the same resolved model")
        if item.result.observable != first.result.observable:
            raise ValueError("all pair analyses must use the same resolved observable")
        for layer in range(layer_count):
            values = (
                item.importance_by_layer[layer].detach().float().cpu().flatten(),
                item.coupling_by_layer[layer].detach().float().cpu().flatten(),
                item.original_activation_by_layer[layer].detach().float().cpu().flatten(),
                item.perturbed_activation_by_layer[layer].detach().float().cpu().flatten(),
            )
            if values[0].numel() == 0 or len({value.shape for value in values}) != 1:
                raise ValueError(
                    f"pair analysis {pair_index} layer {layer} has incompatible shapes"
                )
            if not all(torch.isfinite(value).all().item() for value in values):
                raise ValueError(
                    f"pair analysis {pair_index} layer {layer} contains non-finite values"
                )
            reference_coupling = (
                first.coupling_by_layer[layer].detach().float().cpu().flatten()
            )
            if not torch.equal(values[1], reference_coupling):
                raise ValueError(
                    f"pair analysis {pair_index} layer {layer} has different couplings"
                )

    mean_by_layer: list[torch.Tensor] = []
    rms_by_layer: list[torch.Tensor] = []
    consistency_by_layer: list[torch.Tensor] = []
    coupling_by_layer: list[torch.Tensor] = []
    original_mean_by_layer: list[torch.Tensor] = []
    perturbed_mean_by_layer: list[torch.Tensor] = []
    delta_mean_by_layer: list[torch.Tensor] = []
    layers: list[AggregateLayerSummary] = []
    tensors: dict[str, torch.Tensor] = {}

    for layer in range(layer_count):
        importance_all = torch.stack(
            [item.importance_by_layer[layer].float().flatten() for item in analyses]
        )
        original_all = torch.stack(
            [item.original_activation_by_layer[layer].float().flatten() for item in analyses]
        )
        perturbed_all = torch.stack(
            [item.perturbed_activation_by_layer[layer].float().flatten() for item in analyses]
        )
        importance = importance_all[list(ranking_indices)]
        original = original_all[list(ranking_indices)]
        perturbed = perturbed_all[list(ranking_indices)]
        coupling = analyses[0].coupling_by_layer[layer].float().flatten()
        delta = perturbed - original

        importance_mean = importance.mean(dim=0)
        importance_rms = torch.sqrt(torch.mean(importance.square(), dim=0))
        importance_coherence = torch.where(
            importance_rms > 0,
            importance_mean.abs() / importance_rms,
            torch.zeros_like(importance_rms),
        ).clamp(max=1.0)
        mean_sign = torch.sign(importance_mean)
        matching = (torch.sign(importance) == mean_sign.unsqueeze(0)).float()
        consistency = matching.mean(dim=0)
        consistency = torch.where(mean_sign == 0, torch.zeros_like(consistency), consistency)
        original_mean = original.mean(dim=0)
        perturbed_mean = perturbed.mean(dim=0)
        delta_mean = delta.mean(dim=0)

        rms_mass = float(importance_rms.sum().item())
        absolute_mean = importance_mean.abs()
        absolute_mean_mass = float(absolute_mean.sum().item())
        top_count = min(10, importance_rms.numel())
        top_mass = float(torch.topk(importance_rms, top_count).values.sum().item())
        top_neuron = int(torch.argmax(importance_rms).item())
        top_mean_mass = float(torch.topk(absolute_mean, top_count).values.sum().item())
        top_mean_neuron = int(torch.argmax(absolute_mean).item())
        layers.append(
            AggregateLayerSummary(
                layer=layer,
                signed_mean_sum=float(importance_mean.sum().item()),
                rms_mass=rms_mass,
                positive_mean_mass=float(importance_mean.clamp_min(0).sum().item()),
                negative_mean_mass=float((-importance_mean.clamp_max(0)).sum().item()),
                top_10_rms_share=top_mass / rms_mass if rms_mass else 0.0,
                maximum_rms=float(importance_rms[top_neuron].item()),
                top_neuron=top_neuron,
                activation_delta_norm_mean=fmean(
                    float(torch.linalg.vector_norm(row).item()) for row in delta
                ),
                absolute_mean_mass=absolute_mean_mass,
                top_10_mean_share=(
                    top_mean_mass / absolute_mean_mass if absolute_mean_mass else 0.0
                ),
                maximum_absolute_mean=float(absolute_mean[top_mean_neuron].item()),
                top_mean_neuron=top_mean_neuron,
            )
        )

        mean_by_layer.append(importance_mean)
        rms_by_layer.append(importance_rms)
        consistency_by_layer.append(consistency)
        coupling_by_layer.append(coupling)
        original_mean_by_layer.append(original_mean)
        perturbed_mean_by_layer.append(perturbed_mean)
        delta_mean_by_layer.append(delta_mean)
        tensors[f"importance_mean.layer_{layer}"] = importance_mean.contiguous()
        tensors[f"importance_rms.layer_{layer}"] = importance_rms.contiguous()
        tensors[f"importance_coherence.layer_{layer}"] = importance_coherence.contiguous()
        tensors[f"sign_consistency.layer_{layer}"] = consistency.contiguous()
        tensors[f"coupling.layer_{layer}"] = coupling.contiguous()
        tensors[f"activation_delta_mean.layer_{layer}"] = delta_mean.contiguous()
        for pair_index in range(len(analyses)):
            prefix = f"pair_{pair_index}.layer_{layer}"
            pair_delta = perturbed_all[pair_index] - original_all[pair_index]
            tensors[f"importance.{prefix}"] = importance_all[pair_index].contiguous()
            tensors[f"activation_original.{prefix}"] = original_all[pair_index].contiguous()
            tensors[f"activation_perturbed.{prefix}"] = perturbed_all[pair_index].contiguous()
            tensors[f"activation_delta.{prefix}"] = pair_delta.contiguous()

    offsets: list[int] = []
    total = 0
    for values in rms_by_layer:
        offsets.append(total)
        total += values.numel()
    chosen = min(max(1, top_k), total)
    shared_indices = torch.argsort(
        torch.cat(mean_by_layer).abs(), descending=True, stable=True
    )[:chosen].tolist()
    magnitude_indices = torch.argsort(
        torch.cat(rms_by_layer), descending=True, stable=True
    )[:chosen].tolist()

    def build_neurons(flat_indices: list[int]) -> tuple[AggregateNeuronScore, ...]:
        neurons: list[AggregateNeuronScore] = []
        for rank, flat_index in enumerate(flat_indices, start=1):
            layer = max(
                index for index, offset in enumerate(offsets) if offset <= flat_index
            )
            neuron = flat_index - offsets[layer]
            mean = float(mean_by_layer[layer][neuron].item())
            rms = float(rms_by_layer[layer][neuron].item())
            neurons.append(
                AggregateNeuronScore(
                    rank=rank,
                    layer=layer,
                    neuron=neuron,
                    coupling=float(coupling_by_layer[layer][neuron].item()),
                    original_activation_mean=float(
                        original_mean_by_layer[layer][neuron].item()
                    ),
                    perturbed_activation_mean=float(
                        perturbed_mean_by_layer[layer][neuron].item()
                    ),
                    activation_delta_mean=float(
                        delta_mean_by_layer[layer][neuron].item()
                    ),
                    importance_mean=mean,
                    importance_rms=rms,
                    sign_consistency=float(
                        consistency_by_layer[layer][neuron].item()
                    ),
                    importance_coherence=(min(1.0, abs(mean) / rms) if rms else 0.0),
                )
            )
        return tuple(neurons)

    shared_direction_neurons = build_neurons(shared_indices)
    effect_magnitude_neurons = build_neurons(magnitude_indices)
    neurons = (
        effect_magnitude_neurons
        if ranking_objective == "effect_magnitude"
        else shared_direction_neurons
    )

    qualifications = tuple(
        qualify_pair_logits(
            original_logits=analysis.original_logits,
            perturbed_logits=analysis.perturbed_logits,
            observable=analysis.result.observable,
        )
        for analysis in analyses
    )
    pair_summaries = tuple(
        PairResultSummary(
            pair_id=pair_id,
            split=split,
            original_gap=analysis.result.original_gap,
            perturbed_gap=analysis.result.perturbed_gap,
            measured_delta=analysis.result.measured_delta,
            predicted_delta=analysis.result.predicted_delta,
            original_prediction=analysis.result.original_prediction.decoded,
            perturbed_prediction=analysis.result.perturbed_prediction.decoded,
            ffn_skip_mean=analysis.result.ffn_skip_mean,
            circuit_regime=analysis.result.circuit_regime,
            elapsed_seconds=analysis.result.elapsed_seconds,
            qualification=qualification,
            warnings=analysis.result.warnings,
        )
        for pair_id, split, analysis, qualification in zip(
            pair_ids, pair_splits, analyses, qualifications, strict=True
        )
    )
    finite_ratios = [
        item.result.ffn_skip_mean
        for item in ranking_analyses
        if item.result.ffn_skip_mean is not None
    ]
    warning_values = list(
        dict.fromkeys(warning for item in analyses for warning in item.result.warnings)
    )
    if len(ranking_analyses) > 1:
        warning_values = [
            warning
            for warning in warning_values
            if not warning.startswith("Exploratory result from one prompt pair")
        ]
        warning_values.insert(
            0,
            f"Replicated ranking from {len(ranking_analyses)} discovery prompt pairs remains "
            "observational; causal claims require intervention and broader validation.",
        )
    if len(ranking_analyses) != len(analyses):
        warning_values.append(
            "Neuron ranking uses discovery pairs only; validation and held-out pairs are retained for separate evaluation."
        )
    warning_values.append(
        "Candidate order uses absolute signed-mean importance across discovery pairs; "
        "RMS remains a prompt-conditional magnitude diagnostic."
        if ranking_objective == "shared_direction"
        else "Candidate order uses RMS effect magnitude and can prioritize sign-varying "
        "prompt-conditional neurons; use signed_mean for the paper-faithful shared circuit."
    )
    warnings = tuple(warning_values)
    first_result = analyses[0].result
    ffn_skip_mean = fmean(finite_ratios) if finite_ratios else None
    informative_pair_ids = tuple(
        pair_id
        for pair_id, qualification in zip(pair_ids, qualifications, strict=True)
        if qualification.status == "informative"
    )
    qualification_aggregate = QualificationAggregate(
        informative_pairs=len(informative_pair_ids),
        weak_pairs=sum(item.status == "weak" for item in qualifications),
        invalid_pairs=sum(item.status == "invalid" for item in qualifications),
        informative_pair_ids=informative_pair_ids,
        claim_eligible=any(
            qualifications[index].status == "informative"
            for index in ranking_indices
        ),
        signal_concentration_label=classify_circuit(ffn_skip_mean),
    )
    discovery_qualifications = tuple(qualifications[index] for index in ranking_indices)
    discovery_informative = sum(
        item.status == "informative" for item in discovery_qualifications
    )
    if len(ranking_analyses) == 1:
        ranking_claim = ClaimRecord(
            claim_id="candidate-ranking",
            claim_type="candidate_ranking",
            status=(
                "exploratory"
                if discovery_qualifications[0].status == "informative"
                else "blocked"
            ),
            statement="Signed neuron scores identify candidates for controlled intervention.",
            limitations=(
                "One prompt pair is not a replication.",
                "Ranking alone is not causal evidence.",
            ),
        )
    else:
        ranking_claim = ClaimRecord(
            claim_id="replicated-ranking",
            claim_type="replicated_ranking",
            status=("supported" if discovery_informative >= 2 else "blocked"),
            statement=(
                "Neurons were ranked by absolute signed-mean importance across prompt pairs."
                if ranking_objective == "shared_direction"
                else "Neurons were ranked by RMS effect magnitude across prompt pairs."
            ),
            limitations=(
                "Replication is observational until controlled intervention.",
                "Only informative pairs support the aggregate claim.",
            ),
        )
    observable_claim = ClaimRecord(
        claim_id="first-token-observable",
        claim_type="observable_validity",
        status=(
            "supported"
            if len(informative_pair_ids) == len(analyses)
            else "exploratory"
            if informative_pair_ids
            else "blocked"
        ),
        statement="The first-token observable was checked for binary decision crossing and argmax membership.",
        limitations=(
            "Generated continuation behavior has not yet been evaluated.",
        ),
    )
    summary = RankRunSummary(
        science_hash=science_hash,
        pair_count=len(analyses),
        split_counts={
            split: pair_splits.count(split)
            for split in ("discovery", "validation", "heldout")
            if split in pair_splits
        },
        logical_forward_passes=2 * len(analyses),
        model=first_result.model.__dict__,
        observable={
            "name": first_result.observable.name,
            "target": [item.__dict__ for item in first_result.observable.target],
            "control": [item.__dict__ for item in first_result.observable.control],
        },
        pairs=pair_summaries,
        layers=tuple(layers),
        neurons=neurons,
        ranking_objective=ranking_objective,
        shared_direction_neurons=shared_direction_neurons,
        effect_magnitude_neurons=effect_magnitude_neurons,
        total_neuron_count=total,
        measured_delta_mean=fmean(
            item.result.measured_delta for item in ranking_analyses
        ),
        predicted_delta_mean=fmean(
            item.result.predicted_delta for item in ranking_analyses
        ),
        ffn_skip_mean=ffn_skip_mean,
        evidence_stage=(
            "exploratory_pair"
            if len(ranking_analyses) == 1
            else "replicated_ranking"
        ),
        qualification=qualification_aggregate,
        claims=(observable_claim, ranking_claim),
        warnings=warnings,
    )
    return AggregateComputation(summary=summary, tensors=tensors)
