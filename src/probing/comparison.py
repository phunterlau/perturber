from __future__ import annotations

import random
from statistics import fmean

import torch

from .contracts import (
    ComparisonReport,
    RankRunSummary,
    RankSpec,
    RunComparison,
    StabilityReport,
)


def compare_rank_runs(
    *,
    reference_run_id: str,
    reference: RankRunSummary,
    candidates: tuple[tuple[str, RankRunSummary], ...],
    top_n: int,
    reference_spec: RankSpec | None = None,
    candidate_specs: tuple[RankSpec | None, ...] | None = None,
) -> ComparisonReport:
    if top_n < 1:
        raise ValueError("top_n must be positive")
    reference_values = reference.neurons[:top_n]
    reference_ranks = {
        (item.layer, item.neuron): item.rank for item in reference_values
    }
    reference_signs = {
        (item.layer, item.neuron): 1 if item.importance_mean > 0 else -1
        if item.importance_mean < 0
        else 0
        for item in reference_values
    }
    comparisons = []
    warnings = []
    candidate_specs = candidate_specs or tuple(None for _ in candidates)
    if len(candidate_specs) != len(candidates):
        raise ValueError("candidate specs must align with candidate summaries")
    for (run_id, summary), candidate_spec in zip(
        candidates, candidate_specs, strict=True
    ):
        values = summary.neurons[:top_n]
        ranks = {(item.layer, item.neuron): item.rank for item in values}
        signs = {
            (item.layer, item.neuron): 1 if item.importance_mean > 0 else -1
            if item.importance_mean < 0
            else 0
            for item in values
        }
        shared = set(reference_ranks) & set(ranks)
        denominator = max(1, min(top_n, len(reference_ranks), len(ranks)))
        overlap = len(shared) / denominator
        sign_agreement = (
            sum(reference_signs[key] == signs[key] for key in shared) / len(shared)
            if shared
            else 0.0
        )
        displacement = (
            fmean(abs(reference_ranks[key] - ranks[key]) for key in shared)
            if shared
            else float(top_n)
        )
        ffn_difference = None
        if reference.ffn_skip_mean is not None and summary.ffn_skip_mean is not None:
            ffn_difference = summary.ffn_skip_mean - reference.ffn_skip_mean
        changed_factors = []
        if reference_spec is not None and candidate_spec is not None:
            if reference_spec.model != candidate_spec.model:
                changed_factors.append("model")
            if reference_spec.observable != candidate_spec.observable:
                changed_factors.append("observable_token_set")
            reference_prompts = tuple(
                item.model_dump(mode="json", exclude={"metadata", "split"})
                for item in reference_spec.pairs
            )
            candidate_prompts = tuple(
                item.model_dump(mode="json", exclude={"metadata", "split"})
                for item in candidate_spec.pairs
            )
            if reference_prompts != candidate_prompts:
                changed_factors.append("prompt_or_perturbation")
            if tuple(item.split for item in reference_spec.pairs) != tuple(
                item.split for item in candidate_spec.pairs
            ):
                changed_factors.append("split_assignment")
            if tuple(item.metadata for item in reference_spec.pairs) != tuple(
                item.metadata for item in candidate_spec.pairs
            ):
                changed_factors.append("pair_metadata")
        comparisons.append(
            RunComparison(
                run_id=run_id,
                top_n=denominator,
                overlap_fraction=overlap,
                sign_agreement=sign_agreement,
                mean_rank_displacement=displacement,
                ffn_skip_difference=ffn_difference,
                changed_factors=tuple(changed_factors),
            )
        )
    scientific = any(summary.science_hash != reference.science_hash for _, summary in candidates)
    if not scientific:
        warnings.append(
            "All science hashes match; this is computational rerun comparison, not scientific replication."
        )
    return ComparisonReport(
        reference_run_id=reference_run_id,
        comparisons=tuple(comparisons),
        scientific_replication=scientific,
        warnings=tuple(warnings),
    )


def _pair_importance_matrix(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    layer_ids = sorted(
        int(key.rsplit("_", 1)[1])
        for key in tensors
        if key.startswith("coupling.layer_")
    )
    pair_ids = sorted(
        {
            int(key.split(".")[1].split("_")[1])
            for key in tensors
            if key.startswith("importance.pair_")
        }
    )
    if not layer_ids or not pair_ids:
        raise ValueError("rank tensors do not contain per-pair importance values")
    rows = []
    for pair_id in pair_ids:
        values = []
        for layer in layer_ids:
            key = f"importance.pair_{pair_id}.layer_{layer}"
            if key not in tensors:
                raise ValueError(f"rank tensors are missing {key!r}")
            values.append(tensors[key].detach().float().cpu().flatten())
        rows.append(torch.cat(values))
    return torch.stack(rows)


def rank_stability(
    *,
    run_id: str,
    summary: RankRunSummary,
    tensors: dict[str, torch.Tensor],
    top_n: int,
    splits: int = 100,
    bootstrap_iterations: int = 1000,
    seed: int = 0,
) -> StabilityReport:
    matrix = _pair_importance_matrix(tensors)
    pair_count = int(matrix.shape[0])
    effective_top_n = min(top_n, int(matrix.shape[1]))
    if effective_top_n < 1:
        raise ValueError("top_n must be positive")
    warnings = []
    overlaps: list[float] = []
    sign_agreements: list[float] = []
    rng = random.Random(seed)
    if pair_count >= 2:
        identities = tuple(range(pair_count))
        seen: set[tuple[int, ...]] = set()
        attempts = 0
        while len(seen) < splits and attempts < splits * 20:
            attempts += 1
            shuffled = list(identities)
            rng.shuffle(shuffled)
            first_size = max(1, pair_count // 2)
            first = tuple(sorted(shuffled[:first_size]))
            second = tuple(index for index in identities if index not in first)
            if not second or first in seen:
                continue
            seen.add(first)
            first_values = matrix[list(first)]
            second_values = matrix[list(second)]
            first_mean = first_values.mean(dim=0)
            second_mean = second_values.mean(dim=0)
            first_score = (
                first_mean.abs()
                if summary.ranking_objective == "shared_direction"
                else torch.sqrt(first_values.square().mean(dim=0))
            )
            second_score = (
                second_mean.abs()
                if summary.ranking_objective == "shared_direction"
                else torch.sqrt(second_values.square().mean(dim=0))
            )
            first_top = set(
                torch.argsort(first_score, descending=True, stable=True)[:effective_top_n].tolist()
            )
            second_top = set(
                torch.argsort(second_score, descending=True, stable=True)[:effective_top_n].tolist()
            )
            shared = first_top & second_top
            overlaps.append(len(shared) / effective_top_n)
            if shared:
                sign_agreements.append(
                    sum(
                        int(torch.sign(first_mean[index]).item())
                        == int(torch.sign(second_mean[index]).item())
                        for index in shared
                    )
                    / len(shared)
                )
            else:
                sign_agreements.append(0.0)
    else:
        warnings.append("Split-half stability requires at least two prompt pairs.")
    warnings.append(
        f"Split-half top-k sets use the run's {summary.ranking_objective} objective."
    )

    layer_offsets: dict[int, int] = {}
    offset = 0
    for layer in sorted(item.layer for item in summary.layers):
        layer_offsets[layer] = offset
        offset += int(tensors[f"coupling.layer_{layer}"].numel())
    intervals = []
    if bootstrap_iterations > 0:
        bootstrap_rng = random.Random(seed + 1)
        for neuron in summary.neurons[:effective_top_n]:
            flat_index = layer_offsets[neuron.layer] + neuron.neuron
            values = matrix[:, flat_index]
            means = []
            rms_values = []
            for _ in range(bootstrap_iterations):
                indices = [bootstrap_rng.randrange(pair_count) for _ in range(pair_count)]
                sample = values[indices]
                means.append(float(sample.mean().item()))
                rms_values.append(float(torch.sqrt(sample.square().mean()).item()))
            means.sort()
            rms_values.sort()
            low_index = max(0, int(0.025 * bootstrap_iterations) - 1)
            high_index = min(bootstrap_iterations - 1, int(0.975 * bootstrap_iterations))
            intervals.append(
                {
                    "layer": neuron.layer,
                    "neuron": neuron.neuron,
                    "rank": neuron.rank,
                    "importance_mean_low": means[low_index],
                    "importance_mean_high": means[high_index],
                    "importance_rms_low": rms_values[low_index],
                    "importance_rms_high": rms_values[high_index],
                }
            )
    return StabilityReport(
        run_id=run_id,
        pair_count=pair_count,
        top_n=effective_top_n,
        split_count=len(overlaps),
        mean_top_n_overlap=fmean(overlaps) if overlaps else None,
        minimum_top_n_overlap=min(overlaps) if overlaps else None,
        mean_sign_agreement=fmean(sign_agreements) if sign_agreements else None,
        bootstrap_iterations=bootstrap_iterations,
        neuron_intervals=tuple(intervals),
        warnings=tuple(warnings),
    )


__all__ = ["compare_rank_runs", "rank_stability"]
