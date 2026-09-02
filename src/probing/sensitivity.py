from __future__ import annotations

import itertools

import torch

from .comparison import _pair_importance_matrix
from .contracts import (
    RankRunSummary,
    RankSpec,
    SensitivityComparison,
    SensitivityReport,
)


def perturbation_sensitivity(
    *,
    run_id: str,
    spec: RankSpec,
    summary: RankRunSummary,
    tensors: dict[str, torch.Tensor],
    metadata_key: str,
    top_n: int,
) -> SensitivityReport:
    if not metadata_key.strip():
        raise ValueError("metadata_key must not be blank")
    matrix = _pair_importance_matrix(tensors)
    summary_ids = tuple(item.pair_id for item in summary.pairs)
    spec_ids = tuple(item.id for item in spec.pairs)
    if summary_ids != spec_ids or int(matrix.shape[0]) != len(spec.pairs):
        raise ValueError("rank spec, summary, and tensor pair order do not align")
    effective_top_n = min(top_n, int(matrix.shape[1]))
    if effective_top_n < 1:
        raise ValueError("top_n must be positive")

    grouped_indices: dict[str, list[int]] = {}
    for index, pair in enumerate(spec.pairs):
        if metadata_key == "split":
            value = pair.split
        else:
            value = pair.metadata.get(metadata_key, "__missing__")
        group = str(value)
        grouped_indices.setdefault(group, []).append(index)
    warnings = []
    if len(grouped_indices) < 2:
        warnings.append(
            "Sensitivity comparison requires at least two metadata groups."
        )
    warnings.append(
        f"Group top-k sets use the run's {summary.ranking_objective} objective."
    )

    group_scores: dict[str, tuple[torch.Tensor, torch.Tensor, set[int]]] = {}
    for group, indices in grouped_indices.items():
        values = matrix[indices]
        rms = torch.sqrt(values.square().mean(dim=0))
        mean = values.mean(dim=0)
        score = mean.abs() if summary.ranking_objective == "shared_direction" else rms
        top = set(
            torch.argsort(score, descending=True, stable=True)[:effective_top_n].tolist()
        )
        group_scores[group] = (rms, mean, top)

    comparisons = []
    for first, second in itertools.combinations(sorted(group_scores), 2):
        _first_rms, first_mean, first_top = group_scores[first]
        _second_rms, second_mean, second_top = group_scores[second]
        shared = first_top & second_top
        sign_agreement = (
            sum(
                int(torch.sign(first_mean[index]).item())
                == int(torch.sign(second_mean[index]).item())
                for index in shared
            )
            / len(shared)
            if shared
            else 0.0
        )
        comparisons.append(
            SensitivityComparison(
                first_group=first,
                second_group=second,
                first_pair_count=len(grouped_indices[first]),
                second_pair_count=len(grouped_indices[second]),
                top_n_overlap=len(shared) / effective_top_n,
                sign_agreement=sign_agreement,
            )
        )
    return SensitivityReport(
        run_id=run_id,
        metadata_key=metadata_key,
        groups={
            group: tuple(spec.pairs[index].id for index in indices)
            for group, indices in grouped_indices.items()
        },
        comparisons=tuple(comparisons),
        top_n=effective_top_n,
        warnings=tuple(warnings),
    )


__all__ = ["perturbation_sensitivity"]
