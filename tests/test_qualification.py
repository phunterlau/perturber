import pytest
import torch

from probing.contracts import BehaviorEvaluatorRequest
from probing.domain import ResolvedObservable, ResolvedToken
from probing.qualification import evaluate_generated_behavior, qualify_pair_logits


def _observable() -> ResolvedObservable:
    return ResolvedObservable(
        name="binary",
        target=(ResolvedToken(text="No", token_id=0, decoded="No"),),
        control=(ResolvedToken(text="Yes", token_id=1, decoded="Yes"),),
    )


def test_qualification_requires_crossing_and_argmax_membership() -> None:
    result = qualify_pair_logits(
        original_logits=torch.tensor([0.0, 2.0, -1.0]),
        perturbed_logits=torch.tensor([3.0, 0.0, -1.0]),
        observable=_observable(),
    )

    assert result.status == "informative"
    assert result.decision_crossing is True
    assert result.predictions_in_observable is True
    assert result.original_decision == "control"
    assert result.perturbed_decision == "target"
    assert result.absolute_movement == pytest.approx(5.0)


def test_qualification_retains_non_crossing_pair_as_weak() -> None:
    result = qualify_pair_logits(
        original_logits=torch.tensor([0.0, 2.0, -1.0]),
        perturbed_logits=torch.tensor([1.0, 2.0, -1.0]),
        observable=_observable(),
    )

    assert result.status == "weak"
    assert result.decision_crossing is False
    assert "does not cross" in result.reasons[0]


def test_generated_evaluator_classifies_dominant_unicode_script() -> None:
    evaluator = BehaviorEvaluatorRequest(
        kind="unicode_script",
        target_values=("han",),
        control_values=("latin",),
    )

    assert evaluate_generated_behavior(
        evaluator=evaluator,
        text="法国的首都是巴黎。",
        token_ids=(2,),
        observable=_observable(),
    ) == "target"
    assert evaluate_generated_behavior(
        evaluator=evaluator,
        text="The capital is Paris.",
        token_ids=(1,),
        observable=_observable(),
    ) == "control"
    assert evaluate_generated_behavior(
        evaluator=evaluator,
        text="纯水在 0°C 结冰",
        token_ids=(2,),
        observable=_observable(),
    ) == "target"
