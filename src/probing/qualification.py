from __future__ import annotations

import re
import unicodedata

import torch

from .contracts import BehaviorEvaluatorRequest, PairQualification
from .domain import ResolvedObservable
from .scoring import logit_gap


DECISION_EPSILON = 1e-6
MOVEMENT_EPSILON = 1e-4


def observable_decision(gap: float) -> str:
    if gap > DECISION_EPSILON:
        return "target"
    if gap < -DECISION_EPSILON:
        return "control"
    return "tie"


def _probability_mass(probabilities: torch.Tensor, token_ids: tuple[int, ...]) -> float:
    return float(probabilities[list(token_ids)].sum().item())


def qualify_pair_logits(
    *,
    original_logits: torch.Tensor,
    perturbed_logits: torch.Tensor,
    observable: ResolvedObservable,
) -> PairQualification:
    original = original_logits.detach().float().cpu()
    perturbed = perturbed_logits.detach().float().cpu()
    original_gap = logit_gap(original, observable.target_ids, observable.control_ids)
    perturbed_gap = logit_gap(perturbed, observable.target_ids, observable.control_ids)
    original_decision = observable_decision(original_gap)
    perturbed_decision = observable_decision(perturbed_gap)
    decision_crossing = {original_decision, perturbed_decision} == {
        "target",
        "control",
    }

    original_prediction = int(torch.argmax(original).item())
    perturbed_prediction = int(torch.argmax(perturbed).item())
    observable_ids = set(observable.target_ids) | set(observable.control_ids)
    predictions_in_observable = (
        original_prediction in observable_ids and perturbed_prediction in observable_ids
    )
    original_probabilities = torch.softmax(original, dim=0)
    perturbed_probabilities = torch.softmax(perturbed, dim=0)
    absolute_movement = abs(perturbed_gap - original_gap)

    reasons: list[str] = []
    if not decision_crossing:
        reasons.append("observable gap does not cross the binary decision boundary")
    if not predictions_in_observable:
        reasons.append("one or both argmax predictions fall outside the observable token sets")
    if absolute_movement < MOVEMENT_EPSILON:
        reasons.append("observable movement is negligible")

    if decision_crossing and predictions_in_observable:
        status = "informative"
    elif decision_crossing or predictions_in_observable or absolute_movement >= MOVEMENT_EPSILON:
        status = "weak"
    else:
        status = "invalid"

    return PairQualification(
        status=status,
        original_decision=original_decision,
        perturbed_decision=perturbed_decision,
        decision_crossing=decision_crossing,
        predictions_in_observable=predictions_in_observable,
        original_target_probability=_probability_mass(
            original_probabilities, observable.target_ids
        ),
        original_control_probability=_probability_mass(
            original_probabilities, observable.control_ids
        ),
        perturbed_target_probability=_probability_mass(
            perturbed_probabilities, observable.target_ids
        ),
        perturbed_control_probability=_probability_mass(
            perturbed_probabilities, observable.control_ids
        ),
        absolute_movement=absolute_movement,
        reasons=tuple(reasons),
    )


def evaluate_generated_behavior(
    *,
    evaluator: BehaviorEvaluatorRequest,
    text: str,
    token_ids: tuple[int, ...],
    observable: ResolvedObservable,
) -> str:
    if evaluator.kind == "token_set":
        if not token_ids:
            return "other"
        first = token_ids[0]
        if first in observable.target_ids:
            return "target"
        if first in observable.control_ids:
            return "control"
        return "other"

    if evaluator.kind == "unicode_script":
        def script_count(script: str) -> int:
            count = 0
            for character in text:
                if not character.isalpha():
                    continue
                name = unicodedata.name(character, "")
                if script == "han" and (
                    "CJK UNIFIED IDEOGRAPH" in name
                    or "CJK COMPATIBILITY IDEOGRAPH" in name
                ):
                    count += 1
                elif script == "latin" and "LATIN" in name:
                    count += 1
            return count

        target = sum(script_count(script) for script in evaluator.target_values)
        control = sum(script_count(script) for script in evaluator.control_values)
        if target == control == 0:
            return "other"
        if target == control:
            return "ambiguous"
        return "target" if target > control else "control"

    candidate = text if evaluator.case_sensitive else text.casefold()
    target_values = (
        evaluator.target_values
        if evaluator.case_sensitive
        else tuple(value.casefold() for value in evaluator.target_values)
    )
    control_values = (
        evaluator.control_values
        if evaluator.case_sensitive
        else tuple(value.casefold() for value in evaluator.control_values)
    )

    if evaluator.kind == "contains":
        target = any(value in candidate for value in target_values)
        control = any(value in candidate for value in control_values)
    elif evaluator.kind == "exact":
        normalized = candidate.strip()
        target = normalized in {value.strip() for value in target_values}
        control = normalized in {value.strip() for value in control_values}
    else:
        flags = 0 if evaluator.case_sensitive else re.IGNORECASE
        target = any(re.search(value, text, flags=flags) is not None for value in evaluator.target_values)
        control = any(re.search(value, text, flags=flags) is not None for value in evaluator.control_values)

    if target and control:
        return "ambiguous"
    if target:
        return "target"
    if control:
        return "control"
    return "other"


__all__ = [
    "evaluate_generated_behavior",
    "observable_decision",
    "qualify_pair_logits",
]
