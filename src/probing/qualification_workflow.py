from __future__ import annotations

from .contracts import (
    ClaimRecord,
    GeneratedBehavior,
    PairQualification,
    QualificationAggregate,
    QualificationRunSummary,
    QualificationSpec,
    QualifiedPairResult,
    RankRunSummary,
    RankSpec,
)
from .engine import ProbeEngine
from .qualification import evaluate_generated_behavior, observable_decision
from .observables import resolve_observable
from .domain import ObservableSpec
from .prompting import prepare_pair_condition


def _selected_pairs(
    parent_spec: RankSpec,
    parent_summary: RankRunSummary,
    requested: tuple[str, ...],
) -> tuple[tuple[object, object], ...]:
    spec_by_id = {item.id: item for item in parent_spec.pairs}
    summary_by_id = {item.pair_id: item for item in parent_summary.pairs}
    pair_ids = requested or tuple(spec_by_id)
    unknown = sorted(set(pair_ids) - set(spec_by_id))
    if unknown:
        raise ValueError(f"qualification pair IDs were not found in parent run: {unknown}")
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("qualification pair IDs must be unique")
    return tuple((spec_by_id[pair_id], summary_by_id[pair_id]) for pair_id in pair_ids)


def run_qualification(
    *,
    engine: ProbeEngine,
    parent_spec: RankSpec,
    parent_summary: RankRunSummary,
    spec: QualificationSpec,
    science_hash: str,
) -> QualificationRunSummary:
    observable = resolve_observable(
        engine.adapter.tokenizer,
        ObservableSpec(
            name=parent_spec.observable.name,
            target_tokens=parent_spec.observable.target_tokens,
            control_tokens=parent_spec.observable.control_tokens,
        ),
    )
    selected = _selected_pairs(parent_spec, parent_summary, spec.pair_ids)
    results: list[QualifiedPairResult] = []
    warnings: list[str] = []

    for pair_index, (pair, pair_summary) in enumerate(selected):
        first_token = pair_summary.qualification
        if first_token is None:
            first_token = PairQualification(
                status="weak",
                original_decision=observable_decision(pair_summary.original_gap),
                perturbed_decision=observable_decision(pair_summary.perturbed_gap),
                decision_crossing=(
                    {
                        observable_decision(pair_summary.original_gap),
                        observable_decision(pair_summary.perturbed_gap),
                    }
                    == {"target", "control"}
                ),
                predictions_in_observable=False,
                original_target_probability=0,
                original_control_probability=0,
                perturbed_target_probability=0,
                perturbed_control_probability=0,
                absolute_movement=abs(pair_summary.measured_delta),
                reasons=("parent run predates first-token qualification metrics",),
            )

        generated_results: list[GeneratedBehavior] = []
        for condition_index, (condition, gap) in enumerate(
            (
                ("original", pair_summary.original_gap),
                ("perturbed", pair_summary.perturbed_gap),
            )
        ):
            input_ids, _tokenized = prepare_pair_condition(
                engine.adapter,
                pair=pair,
                model=parent_spec.model,
                condition=condition,
            )
            generated = engine.adapter.generate(
                input_ids,
                max_new_tokens=spec.generation.max_new_tokens,
                do_sample=spec.generation.do_sample,
                temperature=spec.generation.temperature,
                top_p=spec.generation.top_p,
                seed=spec.generation.seed + pair_index * 2 + condition_index,
            )
            behavior_decision = evaluate_generated_behavior(
                evaluator=spec.evaluator,
                text=generated.text,
                token_ids=generated.token_ids,
                observable=observable,
            )
            first_decision = observable_decision(gap)
            generated_results.append(
                GeneratedBehavior(
                    condition=condition,
                    text=generated.text,
                    token_ids=generated.token_ids,
                    observable_decision=first_decision,
                    behavior_decision=behavior_decision,
                    agrees_with_observable=behavior_decision == first_decision,
                )
            )

        generated_crossing = {
            item.behavior_decision for item in generated_results
        } == {"target", "control"}
        agreements = sum(item.agrees_with_observable for item in generated_results)
        reasons = list(first_token.reasons)
        if not generated_crossing:
            reasons.append("generated behaviors do not form the target/control contrast")
        if agreements != 2:
            reasons.append(
                f"generated behavior agrees with the first-token observable for {agreements}/2 conditions"
            )
        # Generated behavior is the stronger semantic validity check. It may
        # qualify a sparse token-set gap whose argmax falls on an unlisted but
        # behaviorally equivalent token, provided the gap itself crosses and
        # both generated outcomes agree with its declared direction.
        if first_token.decision_crossing and generated_crossing and agreements == 2:
            status = "informative"
        elif first_token.status != "invalid" and agreements >= 1:
            status = "weak"
        else:
            status = "invalid"
        results.append(
            QualifiedPairResult(
                pair_id=pair.id,
                split=pair.split,
                first_token=first_token,
                generated=tuple(generated_results),
                status=status,
                reasons=tuple(dict.fromkeys(reasons)),
            )
        )

    informative_ids = tuple(item.pair_id for item in results if item.status == "informative")
    aggregate = QualificationAggregate(
        informative_pairs=len(informative_ids),
        weak_pairs=sum(item.status == "weak" for item in results),
        invalid_pairs=sum(item.status == "invalid" for item in results),
        informative_pair_ids=informative_ids,
        claim_eligible=bool(informative_ids),
        signal_concentration_label=(
            parent_summary.qualification.signal_concentration_label
            if parent_summary.qualification is not None
            else "FFN signal concentration not available in parent run"
        ),
    )
    if not informative_ids:
        warnings.append(
            "No pair passed generated-behavior qualification; causal intervention is blocked by default."
        )
    elif len(informative_ids) != len(results):
        warnings.append(
            "Only informative pairs should be included in default causal summaries."
        )
    claim = ClaimRecord(
        claim_id="generated-observable-validity",
        claim_type="observable_validity",
        status=(
            "supported"
            if len(informative_ids) == len(results)
            else "exploratory"
            if informative_ids
            else "blocked"
        ),
        evidence_run_ids=(spec.parent_run_id,),
        statement="Generated behavior was compared with the declared first-token observable.",
        limitations=(
            "Evaluator validity is task-dependent and recorded in the qualification spec.",
        ),
    )
    return QualificationRunSummary(
        science_hash=science_hash,
        parent_run_id=spec.parent_run_id,
        model=parent_summary.model,
        evaluator=spec.evaluator,
        generation=spec.generation,
        pairs=tuple(results),
        aggregate=aggregate,
        logical_forward_passes=2 * len(results),
        claims=(claim,),
        warnings=tuple(warnings),
    )


__all__ = ["run_qualification"]
