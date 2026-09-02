from __future__ import annotations

from collections import defaultdict
import itertools
import random
from statistics import fmean
from typing import Iterable

import torch

from .adapters import ActivationEdit
from .contracts import (
    AdditivityViolation,
    CausalWidthEstimate,
    ClaimRecord,
    FFNCouplingRunSummary,
    InterventionDoseSummary,
    InterventionObservation,
    InterventionRunSummary,
    InterventionSpec,
    InterventionTrajectoryCheckpoint,
    PairResultSummary,
    RankRunSummary,
    RankSpec,
    SelectedNeuron,
    TrajectoryRunSummary,
)
from .domain import ObservableSpec
from .engine import ProbeEngine
from .observables import resolve_observable
from .scoring import logit_gap
from .prompting import prepare_pair_condition
from .qualification import evaluate_generated_behavior


def _prediction(engine: ProbeEngine, logits: torch.Tensor) -> str:
    token_id = int(torch.argmax(logits.detach().float().cpu()).item())
    return engine.adapter.tokenizer.decode([token_id])


def _select_neurons(
    summary: RankRunSummary,
    spec: InterventionSpec,
    candidate_summary: FFNCouplingRunSummary | None = None,
) -> tuple[SelectedNeuron, ...]:
    if candidate_summary is not None:
        return _select_coupling_neurons(candidate_summary, spec)
    if spec.selection.candidate_method != "parent_ranking":
        raise ValueError(
            "direct/downstream overlap selection requires an FFN coupling parent"
        )
    request = spec.selection
    ranking_objective = (
        summary.ranking_objective
        if request.ranking_objective == "parent"
        else request.ranking_objective
    )
    if ranking_objective == "shared_direction":
        candidates = list(summary.shared_direction_neurons)
    else:
        candidates = list(summary.effect_magnitude_neurons)
    if not candidates:
        if ranking_objective != summary.ranking_objective:
            raise ValueError(
                f"parent rank run does not contain {ranking_objective} candidates"
            )
        candidates = list(summary.neurons)
    ranked = {(item.layer, item.neuron): item for item in candidates}
    if request.strategy == "explicit":
        selected = []
        for reference in request.explicit:
            source = ranked.get((reference.layer, reference.neuron))
            selected.append(
                SelectedNeuron(
                    rank=source.rank if source is not None else None,
                    layer=reference.layer,
                    neuron=reference.neuron,
                    importance_mean=(source.importance_mean if source is not None else None),
                    importance_rms=(source.importance_rms if source is not None else None),
                    sign_consistency=(source.sign_consistency if source is not None else None),
                    score_method="direct_structural",
                    ranking_objective=ranking_objective,
                )
            )
        return tuple(selected)

    if request.layers:
        allowed_layers = set(request.layers)
        candidates = [item for item in candidates if item.layer in allowed_layers]
    if request.sign == "positive":
        candidates = [item for item in candidates if item.importance_mean > 0]
    elif request.sign == "negative":
        candidates = [item for item in candidates if item.importance_mean < 0]
    candidates = [
        item
        for item in candidates
        if item.sign_consistency >= request.min_sign_consistency
    ]
    assert request.top_k is not None
    if len(candidates) < request.top_k:
        raise ValueError(
            f"selection requested {request.top_k} ranked neurons but only "
            f"{len(candidates)} satisfy the filters"
        )
    return tuple(
        SelectedNeuron(
            rank=item.rank,
            layer=item.layer,
            neuron=item.neuron,
            importance_mean=item.importance_mean,
            importance_rms=item.importance_rms,
            sign_consistency=item.sign_consistency,
            score_method="direct_structural",
            ranking_objective=ranking_objective,
        )
        for item in candidates[: request.top_k]
    )


def _select_coupling_neurons(
    summary: FFNCouplingRunSummary,
    spec: InterventionSpec,
) -> tuple[SelectedNeuron, ...]:
    request = spec.selection
    ranking_objective = (
        summary.ranking_objective
        if request.ranking_objective == "parent"
        else request.ranking_objective
    )
    if ranking_objective == "shared_direction":
        candidates = list(summary.shared_direction_neurons)
    else:
        candidates = list(summary.effect_magnitude_neurons)
    if not candidates:
        if ranking_objective != summary.ranking_objective:
            raise ValueError(
                f"FFN coupling run does not contain {ranking_objective} candidates"
            )
        candidates = list(summary.neurons)
    ranked = {(item.layer, item.neuron): item for item in candidates}

    def selected(item, *, score_method: str = "downstream_endpoint_gradient") -> SelectedNeuron:
        return SelectedNeuron(
            rank=item.rank,
            layer=item.layer,
            neuron=item.neuron,
            importance_mean=item.downstream_importance_mean,
            importance_rms=item.downstream_importance_rms,
            sign_consistency=item.downstream_sign_consistency,
            score_method=score_method,
            ranking_objective=ranking_objective,
        )

    if request.strategy == "explicit":
        values = []
        for reference in request.explicit:
            source = ranked.get((reference.layer, reference.neuron))
            values.append(
                selected(source)
                if source is not None
                else SelectedNeuron(
                    layer=reference.layer,
                    neuron=reference.neuron,
                    score_method="downstream_endpoint_gradient",
                    ranking_objective=ranking_objective,
                )
            )
        return tuple(values)

    if request.layers:
        allowed_layers = set(request.layers)
        candidates = [item for item in candidates if item.layer in allowed_layers]
    if request.sign == "positive":
        candidates = [item for item in candidates if item.downstream_importance_mean > 0]
    elif request.sign == "negative":
        candidates = [item for item in candidates if item.downstream_importance_mean < 0]
    candidates = [
        item
        for item in candidates
        if item.downstream_sign_consistency >= request.min_sign_consistency
    ]
    assert request.top_k is not None
    if request.candidate_method == "direct_downstream_overlap":
        assert request.overlap_pool_size is not None
        downstream_pool = candidates[: request.overlap_pool_size]
        direct_pool = sorted(
            candidates,
            key=lambda item: (
                -(
                    abs(item.direct_importance_mean)
                    if ranking_objective == "shared_direction"
                    else item.direct_importance_rms
                ),
                item.layer,
                item.neuron,
            ),
        )[: request.overlap_pool_size]
        direct_keys = {(item.layer, item.neuron) for item in direct_pool}
        candidates = [
            item
            for item in downstream_pool
            if (item.layer, item.neuron) in direct_keys
        ]
    if len(candidates) < request.top_k:
        raise ValueError(
            f"selection requested {request.top_k} coupling candidates but only "
            f"{len(candidates)} satisfy the filters"
        )
    method = (
        "direct_downstream_overlap"
        if request.candidate_method == "direct_downstream_overlap"
        else "downstream_endpoint_gradient"
    )
    return tuple(
        selected(item, score_method=method)
        for item in candidates[: request.top_k]
    )


def _conditions(spec: InterventionSpec) -> tuple[str, ...]:
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


def _eligible_pairs(
    parent_spec: RankSpec,
    summary: RankRunSummary,
    spec: InterventionSpec,
    qualification_statuses: dict[str, str] | None = None,
) -> tuple[tuple[int, object, PairResultSummary], ...]:
    summary_by_id = {item.pair_id: item for item in summary.pairs}
    spec_by_id = {item.id: (index, item) for index, item in enumerate(parent_spec.pairs)}
    pair_ids = spec.pair_ids or tuple(spec_by_id)
    unknown = sorted(set(pair_ids) - set(spec_by_id))
    if unknown:
        raise ValueError(f"intervention pair IDs were not found in parent run: {unknown}")
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("intervention pair IDs must be unique")

    selected = []
    excluded = []
    for pair_id in pair_ids:
        index, pair = spec_by_id[pair_id]
        pair_summary = summary_by_id[pair_id]
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
            selected.append((index, pair, pair_summary))
        else:
            excluded.append(pair_id)
    if not selected:
        raise ValueError(
            "no requested pairs are eligible for intervention; use a qualified pair "
            "or set include_weak_pairs=true for an explicitly exploratory run"
        )
    return tuple(selected)


def _by_layer(neurons: Iterable[SelectedNeuron]) -> dict[int, tuple[int, ...]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for item in neurons:
        grouped[item.layer].append(item.neuron)
    return {layer: tuple(values) for layer, values in grouped.items()}


def _activation_edits(
    *,
    selected: tuple[SelectedNeuron, ...],
    mode: str,
    strength: float,
    tensors: dict[str, torch.Tensor],
    parent_pair_index: int,
) -> tuple[ActivationEdit, ...]:
    edits: list[ActivationEdit] = []
    for layer, neurons in _by_layer(selected).items():
        if mode in {"ablate", "amplify"}:
            edits.append(
                ActivationEdit(
                    layer=layer,
                    neurons=neurons,
                    operation="scale",
                    strength=strength,
                )
            )
            continue
        source_condition = "perturbed" if mode == "patch" else "original"
        key = f"activation_{source_condition}.pair_{parent_pair_index}.layer_{layer}"
        try:
            source = tensors[key]
        except KeyError as exc:
            raise ValueError(f"parent tensor artifact is missing {key!r}") from exc
        edits.append(
            ActivationEdit(
                layer=layer,
                neurons=neurons,
                operation="mix",
                strength=strength,
                source_values=source,
            )
        )
    return tuple(edits)


def _matched_random_neurons(
    *,
    selected: tuple[SelectedNeuron, ...],
    all_selected: set[tuple[int, int]],
    tensors: dict[str, torch.Tensor],
    rng: random.Random,
) -> tuple[SelectedNeuron, ...]:
    random_values: list[SelectedNeuron] = []
    for layer, neurons in _by_layer(selected).items():
        key = f"coupling.layer_{layer}"
        if key not in tensors:
            raise ValueError(f"parent tensor artifact is missing {key!r}")
        width = int(tensors[key].numel())
        candidates = [
            neuron
            for neuron in range(width)
            if (layer, neuron) not in all_selected
        ]
        if len(candidates) < len(neurons):
            raise ValueError(f"layer {layer} has too few neurons for matched controls")
        random_values.extend(
            SelectedNeuron(
                layer=layer,
                neuron=neuron,
                ranking_objective=selected[0].ranking_objective,
            )
            for neuron in sorted(rng.sample(candidates, len(neurons)))
        )
    return tuple(random_values)


def _percentile_interval(values: list[float], *, seed: int) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    draws = sorted(
        fmean(rng.choice(values) for _ in values)
        for _ in range(1000)
    )
    return draws[24], draws[974]


def intervention_plan_counts(
    *,
    parent_spec: RankSpec,
    parent_summary: RankRunSummary,
    spec: InterventionSpec,
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
    dose_count = len(spec.sweep.neuron_counts) * len(spec.sweep.strengths)
    selected_forwards = pair_count * condition_count * dose_count
    control_forwards = selected_forwards * spec.controls.samples
    generation_calls = 0
    if spec.operation.apply_during_generation and spec.generation is not None:
        generation_calls = selected_forwards + pair_count * condition_count
    additivity_calls = 0
    if spec.additivity.top_n:
        n = spec.additivity.top_n
        additivity_calls = pair_count * condition_count * (n + n * (n - 1) // 2)
    collateral_baselines = (
        pair_count * condition_count if spec.collateral_observables else 0
    )
    return (
        pair_count,
        selected_forwards
        + control_forwards
        + generation_calls
        + additivity_calls
        + collateral_baselines,
    )


def run_intervention(
    *,
    engine: ProbeEngine,
    parent_spec: RankSpec,
    parent_summary: RankRunSummary,
    parent_tensors: dict[str, torch.Tensor],
    spec: InterventionSpec,
    science_hash: str,
    qualification_statuses: dict[str, str] | None = None,
    candidate_summary: FFNCouplingRunSummary | None = None,
    trajectory_summary: TrajectoryRunSummary | None = None,
) -> InterventionRunSummary:
    selected_all = _select_neurons(parent_summary, spec, candidate_summary)
    maximum_count = max(spec.sweep.neuron_counts)
    if maximum_count > len(selected_all):
        raise ValueError(
            f"largest neuron dose {maximum_count} exceeds selected population "
            f"{len(selected_all)}"
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
    observations: list[InterventionObservation] = []
    trajectory_overlays: list[InterventionTrajectoryCheckpoint] = []
    random_generator = random.Random(spec.execution.seed)
    all_selected = {(item.layer, item.neuron) for item in selected_all}
    model_calls = 0
    baseline_generation: dict[tuple[str, str], tuple[str, str | None]] = {}
    baseline_collateral: dict[tuple[str, str], dict[str, float]] = {}
    trajectory_pairs = (
        {item.pair_id: item for item in trajectory_summary.pairs}
        if trajectory_summary is not None
        else {}
    )
    missing_trajectory_pairs = (
        sorted(
            {pair.id for _index, pair, _summary in pairs} - set(trajectory_pairs)
        )
        if trajectory_summary is not None
        else []
    )
    if missing_trajectory_pairs:
        raise ValueError(
            "trajectory overlay is missing intervention pairs: "
            f"{missing_trajectory_pairs}"
        )

    def run_one(
        *,
        parent_pair_index: int,
        pair: object,
        pair_summary: PairResultSummary,
        condition: str,
        neurons: tuple[SelectedNeuron, ...],
        strength: float,
        arm: str,
        control_sample: int | None,
        generate: bool,
    ) -> InterventionObservation:
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
        source_gap = None
        if spec.operation.mode in {"patch", "restore"}:
            source_gap = (
                pair_summary.perturbed_gap
                if spec.operation.mode == "patch"
                else pair_summary.original_gap
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
        edits = _activation_edits(
            selected=neurons,
            mode=spec.operation.mode,
            strength=strength,
            tensors=parent_tensors,
            parent_pair_index=parent_pair_index,
        )
        trajectory_capture = None
        if trajectory_summary is not None and arm in {"selected", "matched_random"}:
            trajectory_capture = engine.adapter.forward_trajectory_intervened(
                input_ids,
                tokenized,
                parent_spec.capture.position,
                edits,
            )
            capture = trajectory_capture
        else:
            capture = engine.adapter.forward_intervened(
                input_ids,
                tokenized,
                parent_spec.capture.position,
                edits,
            )
        model_calls += 1
        intervention_gap = logit_gap(
            capture.logits, observable.target_ids, observable.control_ids
        )
        progress = None
        if source_gap is not None and abs(source_gap - baseline_gap) > 1e-8:
            progress = (intervention_gap - baseline_gap) / (source_gap - baseline_gap)

        if trajectory_capture is not None:
            pair_trajectory = trajectory_pairs[pair.id]
            baseline_by_key = {
                (item.layer, item.checkpoint): item
                for item in pair_trajectory.checkpoints
            }
            source_condition = (
                "perturbed" if spec.operation.mode == "patch" else "original"
            )
            for layer in trajectory_capture.checkpoints:
                for checkpoint_name in (
                    "block_input",
                    "post_attention",
                    "post_ffn",
                ):
                    baseline_checkpoint = baseline_by_key.get(
                        (layer.layer, checkpoint_name)
                    )
                    if baseline_checkpoint is None:
                        continue
                    baseline_checkpoint_gap = float(
                        getattr(baseline_checkpoint, f"{condition}_gap")
                    )
                    checkpoint_logits = engine.adapter.decode_residual(
                        getattr(layer, checkpoint_name)
                    )
                    checkpoint_gap = logit_gap(
                        checkpoint_logits,
                        observable.target_ids,
                        observable.control_ids,
                    )
                    checkpoint_progress = None
                    if spec.operation.mode in {"patch", "restore"}:
                        source_checkpoint_gap = float(
                            getattr(
                                baseline_checkpoint,
                                f"{source_condition}_gap",
                            )
                        )
                        denominator = source_checkpoint_gap - baseline_checkpoint_gap
                        if abs(denominator) > 1e-8:
                            checkpoint_progress = (
                                checkpoint_gap - baseline_checkpoint_gap
                            ) / denominator
                    trajectory_overlays.append(
                        InterventionTrajectoryCheckpoint(
                            pair_id=pair.id,
                            split=pair.split,
                            arm=arm,
                            control_sample=control_sample,
                            condition=condition,
                            mode=spec.operation.mode,
                            neuron_count=len(neurons),
                            strength=strength,
                            layer=layer.layer,
                            checkpoint=checkpoint_name,
                            baseline_gap=baseline_checkpoint_gap,
                            intervention_gap=checkpoint_gap,
                            gap_effect=checkpoint_gap - baseline_checkpoint_gap,
                            normalized_source_progress=checkpoint_progress,
                        )
                    )

        generated_text = None
        baseline_text = None
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
            generated = engine.adapter.generate(
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
        return InterventionObservation(
            pair_id=pair.id,
            split=pair.split,
            arm=arm,
            control_sample=control_sample,
            condition=condition,
            mode=spec.operation.mode,
            neuron_count=len(neurons),
            strength=strength,
            baseline_gap=baseline_gap,
            source_gap=source_gap,
            intervention_gap=intervention_gap,
            gap_effect=intervention_gap - baseline_gap,
            normalized_source_progress=progress,
            baseline_prediction=baseline_prediction,
            intervention_prediction=_prediction(engine, capture.logits),
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

    for neuron_count in spec.sweep.neuron_counts:
        selected = selected_all[:neuron_count]
        for strength in spec.sweep.strengths:
            for parent_pair_index, pair, pair_summary in pairs:
                for condition in conditions:
                    observations.append(
                        run_one(
                            parent_pair_index=parent_pair_index,
                            pair=pair,
                            pair_summary=pair_summary,
                            condition=condition,
                            neurons=selected,
                            strength=strength,
                            arm="selected",
                            control_sample=None,
                            generate=spec.operation.apply_during_generation,
                        )
                    )
                    for control_sample in range(spec.controls.samples):
                        control_neurons = _matched_random_neurons(
                            selected=selected,
                            all_selected=all_selected,
                            tensors=parent_tensors,
                            rng=random_generator,
                        )
                        observations.append(
                            run_one(
                                parent_pair_index=parent_pair_index,
                                pair=pair,
                                pair_summary=pair_summary,
                                condition=condition,
                                neurons=control_neurons,
                                strength=strength,
                                arm="matched_random",
                                control_sample=control_sample,
                                generate=False,
                            )
                        )

    additivity: list[AdditivityViolation] = []
    if spec.additivity.top_n:
        additivity_neurons = selected_all[: spec.additivity.top_n]
        if len(additivity_neurons) < 2:
            raise ValueError("additivity requires at least two selected neurons")
        if spec.operation.mode == "amplify":
            full_strength = max(spec.sweep.strengths)
        elif spec.operation.mode == "ablate":
            full_strength = min(spec.sweep.strengths)
        else:
            full_strength = max(spec.sweep.strengths)
        for parent_pair_index, pair, pair_summary in pairs:
            for condition in conditions:
                single_effects: dict[tuple[int, int], float] = {}
                for neuron in additivity_neurons:
                    observation = run_one(
                        parent_pair_index=parent_pair_index,
                        pair=pair,
                        pair_summary=pair_summary,
                        condition=condition,
                        neurons=(neuron,),
                        strength=full_strength,
                        arm="additivity_single",
                        control_sample=None,
                        generate=False,
                    )
                    observations.append(observation)
                    single_effects[(neuron.layer, neuron.neuron)] = observation.gap_effect
                for first, second in itertools.combinations(additivity_neurons, 2):
                    joint = run_one(
                        parent_pair_index=parent_pair_index,
                        pair=pair,
                        pair_summary=pair_summary,
                        condition=condition,
                        neurons=(first, second),
                        strength=full_strength,
                        arm="additivity_pair",
                        control_sample=None,
                        generate=False,
                    )
                    observations.append(joint)
                    first_effect = single_effects[(first.layer, first.neuron)]
                    second_effect = single_effects[(second.layer, second.neuron)]
                    additivity.append(
                        AdditivityViolation(
                            pair_id=pair.id,
                            condition=condition,
                            first={"layer": first.layer, "neuron": first.neuron},
                            second={"layer": second.layer, "neuron": second.neuron},
                            first_effect=first_effect,
                            second_effect=second_effect,
                            joint_effect=joint.gap_effect,
                            epsilon=joint.gap_effect - first_effect - second_effect,
                        )
                    )

    dose_summaries: list[InterventionDoseSummary] = []
    split_values = tuple(dict.fromkeys(pair.split for _index, pair, _summary in pairs))
    for split in split_values:
        for condition in conditions:
            for neuron_count in spec.sweep.neuron_counts:
                for strength in spec.sweep.strengths:
                    selected_effects = [
                        item.gap_effect
                        for item in observations
                        if item.arm == "selected"
                        and item.split == split
                        and item.condition == condition
                        and item.neuron_count == neuron_count
                        and item.strength == strength
                    ]
                    random_effects = [
                        abs(item.gap_effect)
                        for item in observations
                        if item.arm == "matched_random"
                        and item.split == split
                        and item.condition == condition
                        and item.neuron_count == neuron_count
                        and item.strength == strength
                    ]
                    if not selected_effects:
                        continue
                    low, high = _percentile_interval(
                        selected_effects,
                        seed=spec.execution.seed + neuron_count,
                    )
                    selected_absolute = fmean(abs(value) for value in selected_effects)
                    random_absolute = fmean(random_effects) if random_effects else None
                    dose_summaries.append(
                        InterventionDoseSummary(
                            split=split,
                            condition=condition,
                            neuron_count=neuron_count,
                            strength=strength,
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

    controlled_positive = any(
        item.controlled_absolute_effect is not None
        and item.controlled_absolute_effect > 0
        for item in dose_summaries
    )
    directional_interval = any(
        item.controlled_absolute_effect is not None
        and item.controlled_absolute_effect > 0
        and item.bootstrap_low is not None
        and item.bootstrap_high is not None
        and (item.bootstrap_low > 0 or item.bootstrap_high < 0)
        for item in dose_summaries
    )
    causal_width: list[CausalWidthEstimate] = []
    if len(spec.sweep.neuron_counts) >= 2:
        for split in split_values:
            for condition in conditions:
                for strength in spec.sweep.strengths:
                    rows = sorted(
                        (
                            item
                            for item in dose_summaries
                            if item.split == split
                            and item.condition == condition
                            and item.strength == strength
                        ),
                        key=lambda item: item.neuron_count,
                    )
                    if len(rows) < 2:
                        continue
                    effects = [
                        max(
                            0.0,
                            item.controlled_absolute_effect
                            if item.controlled_absolute_effect is not None
                            else item.selected_absolute_effect_mean,
                        )
                        for item in rows
                    ]
                    saturation = max(effects)
                    if saturation <= 0:
                        continue
                    threshold = 0.9 * saturation
                    width = next(
                        item.neuron_count
                        for item, effect in zip(rows, effects, strict=True)
                        if effect >= threshold
                    )
                    causal_width.append(
                        CausalWidthEstimate(
                            split=split,
                            condition=condition,
                            strength=strength,
                            saturation_effect=saturation,
                            width_at_90_percent=width,
                            monotonic=all(
                                second + 1e-8 >= first
                                for first, second in zip(effects, effects[1:])
                            ),
                        )
                    )
    replicated = len(pairs) >= 2
    has_controls = spec.controls.samples > 0
    robust_controls = spec.controls.samples >= 3
    selected_directions = {
        1 if (item.importance_mean or 0) > 0 else -1
        for item in selected_all
        if item.importance_mean not in {None, 0}
    }
    claim_type = {
        "ablate": "necessity",
        "amplify": "causal_effect",
        "patch": "sufficiency",
        "restore": "restoration",
    }[spec.operation.mode]
    if spec.operation.mode == "ablate" and len(selected_directions) != 1:
        claim_type = "causal_effect"
    claims = [
        ClaimRecord(
            claim_id=f"{spec.operation.mode}-effect",
            claim_type=claim_type,
            status=(
                "supported"
                if (
                    controlled_positive
                    and directional_interval
                    and replicated
                    and robust_controls
                )
                else "exploratory"
                if controlled_positive
                else "not_supported"
            ),
            evidence_run_ids=(spec.parent_run_id,),
            statement=(
                f"The {spec.operation.mode} sweep measured selected-neuron effects "
                "against same-layer matched-random controls."
            ),
            limitations=(
                "Causal interpretation is local to the declared prompts, observable, model revision, and doses.",
                "Generated behavior is evidence only when generation was enabled and evaluated separately.",
                "Supported status requires at least three random-control draws and a non-zero directional bootstrap interval; it is not a population-level significance test.",
            ),
        )
    ]
    heldout_doses = [
        item for item in dose_summaries if item.split == "heldout"
    ]
    if heldout_doses:
        heldout_supported = any(
            item.controlled_absolute_effect is not None
            and item.controlled_absolute_effect > 0
            and item.bootstrap_low is not None
            and item.bootstrap_high is not None
            and (item.bootstrap_low > 0 or item.bootstrap_high < 0)
            for item in heldout_doses
        ) and robust_controls
        claims.append(
            ClaimRecord(
                claim_id="heldout-generalization",
                claim_type="generalization",
                status="supported" if heldout_supported else "not_supported",
                evidence_run_ids=(spec.parent_run_id,),
                statement="The declared intervention was evaluated on held-out prompt pairs.",
                limitations=(
                    "Held-out evidence is limited to the declared experiment set and evaluator.",
                ),
            )
        )
    warnings = []
    if not has_controls:
        warnings.append("No matched-random controls were requested.")
    elif not robust_controls:
        warnings.append(
            "Fewer than three matched-random draws were requested; causal claims remain exploratory."
        )
    if not replicated:
        warnings.append("The intervention uses fewer than two eligible prompt pairs.")
    if spec.include_weak_pairs:
        warnings.append("Weak pairs were explicitly included; treat aggregate claims as exploratory.")
    if spec.operation.mode == "ablate" and len(selected_directions) != 1:
        warnings.append(
            "Ablation selection mixes positive and negative observable directions; interpret it as a generic causal effect, not necessity for one direction."
        )
    if any(not item.monotonic for item in causal_width):
        warnings.append(
            "At least one dose response is non-monotonic; its 90% width is descriptive rather than a stable circuit-size estimate."
        )
    if not spec.operation.apply_during_generation and spec.generation is not None:
        warnings.append("Generation settings were ignored because apply_during_generation=false.")
    if spec.operation.apply_during_generation and spec.generation is not None and spec.evaluator is None:
        warnings.append("Generated continuations were recorded without a behavior evaluator.")
    if trajectory_summary is not None:
        warnings.append(
            "Intervened trajectory overlays localize where a controlled effect becomes decodable; claim status still follows qualification, matched controls, and replication."
        )
    return InterventionRunSummary(
        science_hash=science_hash,
        parent_run_id=spec.parent_run_id,
        rank_run_id=(
            candidate_summary.parent_run_id
            if candidate_summary is not None
            else spec.parent_run_id
        ),
        candidate_score_method=(
            selected_all[0].score_method
            if candidate_summary is not None and selected_all
            else "direct_structural"
        ),
        candidate_ranking_objective=(
            selected_all[0].ranking_objective
            if selected_all
            else (
                candidate_summary.ranking_objective
                if candidate_summary is not None
                else parent_summary.ranking_objective
            )
        ),
        qualification_run_id=spec.qualification_run_id,
        trajectory_run_id=spec.trajectory_run_id,
        model=parent_summary.model,
        observable=parent_summary.observable,
        operation=spec.operation,
        selection=spec.selection,
        selected_neurons=selected_all,
        pairs=tuple(pair.id for _index, pair, _summary in pairs),
        split_counts={
            split: sum(pair.split == split for _index, pair, _summary in pairs)
            for split in split_values
        },
        observations=tuple(observations),
        trajectory_overlays=tuple(trajectory_overlays),
        doses=tuple(dose_summaries),
        additivity=tuple(additivity),
        causal_width=tuple(causal_width),
        logical_forward_passes=model_calls,
        claims=tuple(claims),
        warnings=tuple(warnings),
    )


__all__ = ["intervention_plan_counts", "run_intervention"]
