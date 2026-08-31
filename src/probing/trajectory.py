from __future__ import annotations

from dataclasses import asdict
import math

import torch

from .contracts import (
    PairTrajectorySummary,
    RankRunSummary,
    RankSpec,
    TrajectoryCheckpointSummary,
    TrajectoryRunSummary,
    TrajectorySpec,
    TrajectoryTransitionSuggestion,
)
from .engine import ProbeEngine
from .observables import resolve_observable
from .scoring import logit_gap


_CHECKPOINT_ORDER = {"block_input": 0, "post_attention": 1, "post_ffn": 2}


def trajectory_plan_counts(
    *, parent_summary: RankRunSummary, spec: TrajectorySpec
) -> tuple[int, int]:
    available = {pair.pair_id for pair in parent_summary.pairs}
    requested = spec.pair_ids or tuple(pair.pair_id for pair in parent_summary.pairs)
    unknown = sorted(set(requested) - available)
    if unknown:
        raise ValueError(f"trajectory references unknown parent pairs: {unknown}")
    return len(requested), 2 * len(requested)


def _distribution_metrics(
    logits: torch.Tensor,
    *,
    final_logits: torch.Tensor,
    target_ids: tuple[int, ...],
    control_ids: tuple[int, ...],
) -> dict[str, float | int | torch.Tensor]:
    values = logits.detach().float().cpu()
    final_values = final_logits.detach().float().cpu()
    log_probabilities = torch.log_softmax(values, dim=0)
    probabilities = log_probabilities.exp()
    final_log_probabilities = torch.log_softmax(final_values, dim=0)
    target_logit = values[list(target_ids)].mean()
    target_rank = 1 + int((values > target_logit).sum().item())
    return {
        "probabilities": probabilities,
        "log_probabilities": log_probabilities,
        "gap": logit_gap(values, target_ids, control_ids),
        "target_probability": float(probabilities[list(target_ids)].mean().item()),
        "control_probability": float(probabilities[list(control_ids)].mean().item()),
        "entropy": float((-(probabilities * log_probabilities).sum()).item()),
        "target_rank": target_rank,
        "forward_kl_to_final": max(
            0.0,
            float(
                (
                    probabilities
                    * (log_probabilities - final_log_probabilities)
                ).sum().item()
            ),
        ),
    }


def _paired_divergence(
    original: dict[str, float | int | torch.Tensor],
    perturbed: dict[str, float | int | torch.Tensor],
) -> tuple[float, float]:
    p = original["probabilities"]
    q = perturbed["probabilities"]
    assert isinstance(p, torch.Tensor) and isinstance(q, torch.Tensor)
    midpoint = 0.5 * (p + q)
    log_midpoint = midpoint.clamp_min(torch.finfo(midpoint.dtype).tiny).log()
    log_p = original["log_probabilities"]
    log_q = perturbed["log_probabilities"]
    assert isinstance(log_p, torch.Tensor) and isinstance(log_q, torch.Tensor)
    js = 0.5 * (
        (p * (log_p - log_midpoint)).sum()
        + (q * (log_q - log_midpoint)).sum()
    )
    tv = 0.5 * (p - q).abs().sum()
    return max(0.0, float(js.item())), min(1.0, max(0.0, float(tv.item())))


def _transition_suggestions(
    checkpoints: tuple[TrajectoryCheckpointSummary, ...], limit: int
) -> tuple[TrajectoryTransitionSuggestion, ...]:
    candidates: list[tuple[float, int, str, float]] = []
    previous = 0.0
    for item in checkpoints:
        change = item.pair_delta - previous
        candidates.append((abs(change), item.layer, item.checkpoint, change))
        previous = item.pair_delta
    candidates.sort(key=lambda value: (-value[0], value[1], _CHECKPOINT_ORDER[value[2]]))
    return tuple(
        TrajectoryTransitionSuggestion(
            rank=rank,
            layer=layer,
            checkpoint=checkpoint,
            pair_delta_change=change,
            absolute_change=magnitude,
        )
        for rank, (magnitude, layer, checkpoint, change) in enumerate(
            candidates[:limit], start=1
        )
    )


def run_trajectory(
    *,
    engine: ProbeEngine,
    parent_spec: RankSpec,
    parent_summary: RankRunSummary,
    spec: TrajectorySpec,
    science_hash: str,
) -> TrajectoryRunSummary:
    observable = resolve_observable(engine.adapter.tokenizer, parent_spec.observable)
    selected = set(spec.pair_ids) if spec.pair_ids else None
    pairs_by_id = {pair.id: pair for pair in parent_spec.pairs}
    parent_pairs = [
        item
        for item in parent_summary.pairs
        if selected is None or item.pair_id in selected
    ]
    pair_results: list[PairTrajectorySummary] = []
    warnings: list[str] = [
        "Native trajectory values are observational decodability evidence, not causal use."
    ]

    for parent_pair in parent_pairs:
        pair = pairs_by_id[parent_pair.pair_id]
        original_ids, original_tokens = engine.adapter.prepare_prompt(
            text=pair.original,
            messages=tuple(item.model_dump(mode="json") for item in pair.original_messages),
            tools=pair.tools,
            chat_template=parent_spec.model.chat_template,
            enable_thinking=parent_spec.model.enable_thinking,
        )
        perturbed_ids, perturbed_tokens = engine.adapter.prepare_prompt(
            text=pair.perturbed,
            messages=tuple(item.model_dump(mode="json") for item in pair.perturbed_messages),
            tools=pair.tools,
            chat_template=parent_spec.model.chat_template,
            enable_thinking=parent_spec.model.enable_thinking,
        )
        original = engine.adapter.forward_trajectory_capture(
            original_ids, original_tokens, spec.position
        )
        perturbed = engine.adapter.forward_trajectory_capture(
            perturbed_ids, perturbed_tokens, spec.position
        )
        if len(original.checkpoints) != len(perturbed.checkpoints):
            raise RuntimeError("paired trajectory layer counts do not match")

        final_original_decoded = engine.adapter.decode_residual(
            original.checkpoints[-1].post_ffn
        )
        final_perturbed_decoded = engine.adapter.decode_residual(
            perturbed.checkpoints[-1].post_ffn
        )
        pair_warnings: list[str] = []
        for condition, decoded, ordinary in (
            ("original", final_original_decoded, original.logits),
            ("perturbed", final_perturbed_decoded, perturbed.logits),
        ):
            if not torch.allclose(decoded, ordinary, rtol=2e-3, atol=2e-3):
                maximum = float((decoded - ordinary).abs().max().item())
                raise RuntimeError(
                    f"{condition} final native checkpoint does not match ordinary logits; "
                    f"max_abs_error={maximum}"
                )

        checkpoint_results: list[TrajectoryCheckpointSummary] = []
        for original_layer, perturbed_layer in zip(
            original.checkpoints, perturbed.checkpoints, strict=True
        ):
            if original_layer.layer != perturbed_layer.layer:
                raise RuntimeError("paired trajectory layer identities do not match")
            for checkpoint in spec.checkpoints:
                original_residual = getattr(original_layer, checkpoint)
                perturbed_residual = getattr(perturbed_layer, checkpoint)
                original_logits = engine.adapter.decode_residual(original_residual)
                perturbed_logits = engine.adapter.decode_residual(perturbed_residual)
                original_metrics = _distribution_metrics(
                    original_logits,
                    final_logits=original.logits,
                    target_ids=observable.target_ids,
                    control_ids=observable.control_ids,
                )
                perturbed_metrics = _distribution_metrics(
                    perturbed_logits,
                    final_logits=perturbed.logits,
                    target_ids=observable.target_ids,
                    control_ids=observable.control_ids,
                )
                js, tv = _paired_divergence(original_metrics, perturbed_metrics)
                original_gap = float(original_metrics["gap"])
                perturbed_gap = float(perturbed_metrics["gap"])
                values = TrajectoryCheckpointSummary(
                    layer=original_layer.layer,
                    checkpoint=checkpoint,
                    original_gap=original_gap,
                    perturbed_gap=perturbed_gap,
                    pair_delta=perturbed_gap - original_gap,
                    original_target_probability=float(
                        original_metrics["target_probability"]
                    ),
                    perturbed_target_probability=float(
                        perturbed_metrics["target_probability"]
                    ),
                    original_control_probability=float(
                        original_metrics["control_probability"]
                    ),
                    perturbed_control_probability=float(
                        perturbed_metrics["control_probability"]
                    ),
                    original_entropy=float(original_metrics["entropy"]),
                    perturbed_entropy=float(perturbed_metrics["entropy"]),
                    original_target_rank=int(original_metrics["target_rank"]),
                    perturbed_target_rank=int(perturbed_metrics["target_rank"]),
                    original_forward_kl_to_final=float(
                        original_metrics["forward_kl_to_final"]
                    ),
                    perturbed_forward_kl_to_final=float(
                        perturbed_metrics["forward_kl_to_final"]
                    ),
                    paired_js=js,
                    paired_total_variation=tv,
                )
                if not all(
                    math.isfinite(value)
                    for value in (
                        values.original_gap,
                        values.perturbed_gap,
                        values.original_entropy,
                        values.perturbed_entropy,
                        values.paired_js,
                        values.paired_total_variation,
                    )
                ):
                    raise RuntimeError("trajectory produced non-finite metrics")
                checkpoint_results.append(values)

        checkpoint_tuple = tuple(checkpoint_results)
        pair_results.append(
            PairTrajectorySummary(
                pair_id=pair.id,
                split=pair.split,
                checkpoints=checkpoint_tuple,
                transitions=_transition_suggestions(
                    checkpoint_tuple, spec.transition_limit
                ),
                final_pair_delta=checkpoint_tuple[-1].pair_delta,
                warnings=tuple(pair_warnings),
            )
        )

    return TrajectoryRunSummary(
        science_hash=science_hash,
        parent_run_id=spec.parent_run_id,
        model=asdict(engine.adapter.metadata),
        observable={
            "name": observable.name,
            "target_tokens": [item.text for item in observable.target],
            "control_tokens": [item.text for item in observable.control],
        },
        pair_count=len(pair_results),
        logical_forward_passes=2 * len(pair_results),
        pairs=tuple(pair_results),
        warnings=tuple(warnings),
    )
