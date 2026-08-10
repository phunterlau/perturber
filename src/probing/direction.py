from __future__ import annotations

import random
from statistics import fmean

import torch

from .adapters import ResidualEdit
from .contracts import (
    ClaimRecord,
    DirectionDoseSummary,
    DirectionInjectionRunSummary,
    DirectionInjectionSpec,
    DirectionObservation,
    PairResultSummary,
    RankRunSummary,
    RankSpec,
)
from .domain import ObservableSpec
from .engine import ProbeEngine
from .observables import resolve_observable
from .qualification import evaluate_generated_behavior
from .scoring import logit_gap
from .prompting import prepare_pair_condition


def _conditions(spec: DirectionInjectionSpec) -> tuple[str, ...]:
    return ("original", "perturbed") if spec.condition == "both" else (spec.condition,)


def _eligible_pairs(
    parent_spec: RankSpec,
    summary: RankRunSummary,
    spec: DirectionInjectionSpec,
    qualification_statuses: dict[str, str] | None = None,
) -> tuple[tuple[object, PairResultSummary], ...]:
    pairs = {item.id: item for item in parent_spec.pairs}
    summaries = {item.pair_id: item for item in summary.pairs}
    pair_ids = spec.pair_ids or tuple(pairs)
    unknown = sorted(set(pair_ids) - set(pairs))
    if unknown:
        raise ValueError(f"direction sweep pair IDs were not found in parent run: {unknown}")
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("direction sweep pair IDs must be unique")
    selected = []
    for pair_id in pair_ids:
        pair_summary = summaries[pair_id]
        qualification = pair_summary.qualification
        status = (
            qualification_statuses.get(pair_id)
            if qualification_statuses is not None
            else qualification.status if qualification is not None else None
        )
        eligible = status is None or status == "informative"
        if spec.include_weak_pairs and status is not None:
            eligible = status in {"informative", "weak"}
        if eligible:
            selected.append((pairs[pair_id], pair_summary))
    if not selected:
        raise ValueError(
            "no requested pairs are eligible for direction injection; qualify the "
            "observable or explicitly include weak pairs"
        )
    return tuple(selected)


def direction_plan_counts(
    *,
    parent_spec: RankSpec,
    parent_summary: RankRunSummary,
    spec: DirectionInjectionSpec,
    qualification_statuses: dict[str, str] | None = None,
) -> tuple[int, int]:
    pair_count = len(
        _eligible_pairs(
            parent_spec,
            parent_summary,
            spec,
            qualification_statuses,
        )
    )
    condition_count = len(_conditions(spec))
    selected = pair_count * condition_count * len(spec.layers) * len(spec.betas)
    controls = selected * spec.random_direction_samples
    generation = 0
    if spec.generation is not None:
        generation = selected + pair_count * condition_count
    collateral_baselines = (
        pair_count * condition_count if spec.collateral_observables else 0
    )
    return pair_count, selected + controls + generation + collateral_baselines


def _random_orthogonal_direction(
    direction: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    reference = direction.detach().float().cpu().flatten()
    unit = reference / reference.norm()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    candidate = torch.randn(reference.shape, generator=generator)
    candidate = candidate - torch.dot(candidate, unit) * unit
    norm = candidate.norm()
    if float(norm.item()) <= 1e-12:
        raise RuntimeError("failed to sample a non-zero orthogonal control direction")
    return candidate / norm * reference.norm()


def _interval(values: list[float], seed: int) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    draws = sorted(fmean(rng.choice(values) for _ in values) for _ in range(1000))
    return draws[24], draws[974]


def run_direction_injection(
    *,
    engine: ProbeEngine,
    parent_spec: RankSpec,
    parent_summary: RankRunSummary,
    spec: DirectionInjectionSpec,
    science_hash: str,
    qualification_statuses: dict[str, str] | None = None,
) -> DirectionInjectionRunSummary:
    layer_count = int(parent_summary.model["layer_count"])
    invalid_layers = [layer for layer in spec.layers if layer >= layer_count]
    if invalid_layers:
        raise ValueError(
            f"direction layers are outside model range 0..{layer_count - 1}: {invalid_layers}"
        )
    pairs = _eligible_pairs(
        parent_spec,
        parent_summary,
        spec,
        qualification_statuses,
    )
    conditions = _conditions(spec)
    observable = resolve_observable(
        engine.adapter.tokenizer,
        ObservableSpec(
            name=parent_spec.observable.name,
            target_tokens=parent_spec.observable.target_tokens,
            control_tokens=parent_spec.observable.control_tokens,
        ),
    )
    collateral = tuple(
        resolve_observable(
            engine.adapter.tokenizer,
            ObservableSpec(
                name=request.name,
                target_tokens=request.target_tokens,
                control_tokens=request.control_tokens,
            ),
        )
        for request in spec.collateral_observables
    )
    direction = engine.adapter.behavioral_direction(observable).detach().float().cpu()
    direction_norm = float(direction.norm().item())
    if direction_norm <= 1e-12:
        raise ValueError("behavioral direction has zero norm")
    random_directions = tuple(
        _random_orthogonal_direction(
            direction, seed=spec.execution.seed + index + 1
        )
        for index in range(spec.random_direction_samples)
    )
    observations: list[DirectionObservation] = []
    baseline_generation: dict[tuple[str, str], tuple[str, str | None]] = {}
    baseline_collateral: dict[tuple[str, str], dict[str, float]] = {}
    model_calls = 0

    def run_one(
        *,
        pair: object,
        pair_summary: PairResultSummary,
        condition: str,
        layer: int,
        beta: float,
        local_direction: torch.Tensor,
        arm: str,
        control_sample: int | None,
        generate: bool,
    ) -> DirectionObservation:
        nonlocal model_calls
        baseline_gap = (
            pair_summary.original_gap
            if condition == "original"
            else pair_summary.perturbed_gap
        )
        baseline_prediction = (
            pair_summary.original_prediction
            if condition == "original"
            else pair_summary.perturbed_prediction
        )
        input_ids, tokenized = prepare_pair_condition(
            engine.adapter,
            pair=pair,
            model=parent_spec.model,
            condition=condition,
        )
        cache_key = (pair.id, condition)
        if collateral and cache_key not in baseline_collateral:
            baseline_capture = engine.adapter.forward_capture(
                input_ids, tokenized, parent_spec.capture.position
            )
            model_calls += 1
            baseline_collateral[cache_key] = {
                item.name: logit_gap(
                    baseline_capture.logits, item.target_ids, item.control_ids
                )
                for item in collateral
            }
        edits = (
            ResidualEdit(
                layer=layer,
                direction=local_direction,
                beta=beta,
                normalization=spec.normalization,
            ),
        )
        capture = engine.adapter.forward_residual_intervened(
            input_ids, tokenized, parent_spec.capture.position, edits
        )
        model_calls += 1
        intervention_gap = logit_gap(
            capture.logits, observable.target_ids, observable.control_ids
        )
        baseline_text = None
        generated_text = None
        baseline_behavior = None
        intervention_behavior = None
        if generate and spec.generation is not None:
            if cache_key not in baseline_generation:
                baseline = engine.adapter.generate(
                    input_ids,
                    max_new_tokens=spec.generation.max_new_tokens,
                    do_sample=spec.generation.do_sample,
                    temperature=spec.generation.temperature,
                    top_p=spec.generation.top_p,
                    seed=spec.generation.seed + len(baseline_generation),
                )
                baseline_decision = (
                    evaluate_generated_behavior(
                        evaluator=spec.evaluator,
                        text=baseline.text,
                        token_ids=baseline.token_ids,
                        observable=observable,
                    )
                    if spec.evaluator is not None
                    else None
                )
                baseline_generation[cache_key] = (baseline.text, baseline_decision)
                model_calls += 1
            baseline_text, baseline_behavior = baseline_generation[cache_key]
            generated = engine.adapter.generate_residual_intervened(
                input_ids,
                max_new_tokens=spec.generation.max_new_tokens,
                do_sample=spec.generation.do_sample,
                temperature=spec.generation.temperature,
                top_p=spec.generation.top_p,
                seed=spec.generation.seed + len(observations) + 10_000,
                edits=edits,
            )
            generated_text = generated.text
            intervention_behavior = (
                evaluate_generated_behavior(
                    evaluator=spec.evaluator,
                    text=generated.text,
                    token_ids=generated.token_ids,
                    observable=observable,
                )
                if spec.evaluator is not None
                else None
            )
            model_calls += 1
        token_id = int(torch.argmax(capture.logits).item())
        return DirectionObservation(
            pair_id=pair.id,
            split=pair.split,
            arm=arm,
            control_sample=control_sample,
            condition=condition,
            layer=layer,
            beta=beta,
            normalization=spec.normalization,
            baseline_gap=baseline_gap,
            intervention_gap=intervention_gap,
            gap_effect=intervention_gap - baseline_gap,
            baseline_prediction=baseline_prediction,
            intervention_prediction=engine.adapter.tokenizer.decode([token_id]),
            baseline_generated_text=baseline_text,
            generated_text=generated_text,
            baseline_behavior_decision=baseline_behavior,
            intervention_behavior_decision=intervention_behavior,
            collateral_gap_effects={
                item.name: (
                    logit_gap(capture.logits, item.target_ids, item.control_ids)
                    - baseline_collateral[cache_key][item.name]
                )
                for item in collateral
            },
        )

    for layer in spec.layers:
        for beta in spec.betas:
            for pair, pair_summary in pairs:
                for condition in conditions:
                    observations.append(
                        run_one(
                            pair=pair,
                            pair_summary=pair_summary,
                            condition=condition,
                            layer=layer,
                            beta=beta,
                            local_direction=direction,
                            arm="behavioral_direction",
                            control_sample=None,
                            generate=spec.generation is not None,
                        )
                    )
                    for sample, random_direction in enumerate(random_directions):
                        observations.append(
                            run_one(
                                pair=pair,
                                pair_summary=pair_summary,
                                condition=condition,
                                layer=layer,
                                beta=beta,
                                local_direction=random_direction,
                                arm="matched_random_direction",
                                control_sample=sample,
                                generate=False,
                            )
                        )

    doses = []
    split_values = tuple(dict.fromkeys(pair.split for pair, _summary in pairs))
    for split in split_values:
        for condition in conditions:
            for layer in spec.layers:
                for beta in spec.betas:
                    selected_effects = [
                        item.gap_effect
                        for item in observations
                        if item.arm == "behavioral_direction"
                        and item.split == split
                        and item.condition == condition
                        and item.layer == layer
                        and item.beta == beta
                    ]
                    random_effects = [
                        abs(item.gap_effect)
                        for item in observations
                        if item.arm == "matched_random_direction"
                        and item.split == split
                        and item.condition == condition
                        and item.layer == layer
                        and item.beta == beta
                    ]
                    if not selected_effects:
                        continue
                    low, high = _interval(
                        selected_effects, spec.execution.seed + layer
                    )
                    selected_absolute = fmean(abs(value) for value in selected_effects)
                    random_absolute = fmean(random_effects) if random_effects else None
                    doses.append(
                        DirectionDoseSummary(
                            split=split,
                            condition=condition,
                            layer=layer,
                            beta=beta,
                            selected_effect_mean=fmean(selected_effects),
                            selected_absolute_effect_mean=selected_absolute,
                            random_absolute_effect_mean=random_absolute,
                            controlled_absolute_effect=(
                                selected_absolute - random_absolute
                                if random_absolute is not None
                                else None
                            ),
                            bootstrap_low=low,
                            bootstrap_high=high,
                            pair_count=len(selected_effects),
                            random_observation_count=len(random_effects),
                        )
                    )

    controlled = any(
        item.controlled_absolute_effect is not None
        and item.controlled_absolute_effect > 0
        for item in doses
    )
    directional_interval = any(
        item.controlled_absolute_effect is not None
        and item.controlled_absolute_effect > 0
        and item.bootstrap_low is not None
        and item.bootstrap_high is not None
        and (item.bootstrap_low > 0 or item.bootstrap_high < 0)
        for item in doses
    )
    warnings = [
        "Direction injection tests linear controllability, not localization to an FFN circuit."
    ]
    if parent_summary.ffn_skip_mean is not None and parent_summary.ffn_skip_mean < 0.2:
        warnings.append(
            "Parent FFN/Skip is in the paper's low-concentration readout-compatible range; successful injection remains a separate empirical question."
        )
    if spec.generation is not None and spec.evaluator is None:
        warnings.append(
            "Generated continuations were recorded without a behavior evaluator."
        )
    if spec.include_weak_pairs:
        warnings.append("Weak pairs were explicitly included in the direction sweep.")
    if 0 < spec.random_direction_samples < 3:
        warnings.append(
            "Fewer than three random-direction draws were requested; causal claims remain exploratory."
        )
    claim = ClaimRecord(
        claim_id="direction-controllability",
        claim_type="causal_effect",
        status=(
            "supported"
            if (
                controlled
                and directional_interval
                and len(pairs) >= 2
                and spec.random_direction_samples >= 3
            )
            else "exploratory"
            if controlled
            else "not_supported"
        ),
        evidence_run_ids=(spec.parent_run_id,),
        statement="Behavioral-direction injection was compared with norm-matched orthogonal random directions.",
        limitations=(
            "A logit-gap response does not by itself establish generated-behavior control.",
            "Direction effects can alter unrelated behaviors and require collateral evaluation.",
            "Supported status requires at least three random-direction draws and a non-zero directional bootstrap interval; it is not a population-level significance test.",
        ),
    )
    return DirectionInjectionRunSummary(
        science_hash=science_hash,
        parent_run_id=spec.parent_run_id,
        qualification_run_id=spec.qualification_run_id,
        model=parent_summary.model,
        observable=parent_summary.observable,
        layers=spec.layers,
        betas=spec.betas,
        normalization=spec.normalization,
        behavioral_direction_norm=direction_norm,
        ffn_skip_mean=parent_summary.ffn_skip_mean,
        observations=tuple(observations),
        doses=tuple(doses),
        pairs=tuple(pair.id for pair, _summary in pairs),
        split_counts={
            split: sum(pair.split == split for pair, _summary in pairs)
            for split in split_values
        },
        logical_forward_passes=model_calls,
        claims=(claim,),
        warnings=tuple(warnings),
    )


__all__ = ["direction_plan_counts", "run_direction_injection"]
