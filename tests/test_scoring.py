import pytest
import torch

from probing.scoring import classify_circuit, ffn_skip_ratio, logit_gap, rank_neurons


def test_logit_gap_is_target_minus_control_mean() -> None:
    logits = torch.tensor([5.0, 1.0, 3.0, -1.0])
    assert logit_gap(logits, (0, 2), (1, 3)) == pytest.approx(4.0)


def test_signed_ranking_and_layer_summary() -> None:
    ranked = rank_neurons(
        original_activations=(torch.tensor([1.0, 2.0]),),
        perturbed_activations=(torch.tensor([3.0, 1.0]),),
        couplings=(torch.tensor([2.0, -1.0]),),
        top_k=2,
    )

    assert ranked.predicted_delta == pytest.approx(5.0)
    assert ranked.total_neuron_count == 2
    assert ranked.neurons[0].neuron == 0
    assert ranked.neurons[0].importance == pytest.approx(4.0)
    assert ranked.neurons[1].importance == pytest.approx(1.0)
    assert ranked.layers[0].signed_sum == pytest.approx(5.0)
    assert ranked.layers[0].positive_mass == pytest.approx(5.0)
    assert ranked.layers[0].negative_mass == pytest.approx(0.0)


def test_ffn_skip_and_regimes() -> None:
    ratio = ffn_skip_ratio(
        torch.tensor([1.0, 0.0]),
        torch.tensor([4.0, 2.0]),
        torch.tensor([2.0, 8.0]),
    )
    assert ratio == pytest.approx(0.5)
    assert classify_circuit(ratio).startswith("high FFN signal concentration")
    assert classify_circuit(0.1).startswith("low FFN signal concentration")
    assert classify_circuit(0.25).startswith("intermediate FFN signal concentration")


def test_equal_scores_have_deterministic_lower_index_tie_break() -> None:
    ranked = rank_neurons(
        original_activations=(torch.zeros(3),),
        perturbed_activations=(torch.ones(3),),
        couplings=(torch.tensor([1.0, -1.0, 1.0]),),
        top_k=3,
    )

    assert [item.neuron for item in ranked.neurons] == [0, 1, 2]
    assert [item.layer_rank for item in ranked.neurons] == [1, 2, 3]


def test_non_finite_scores_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        rank_neurons(
            original_activations=(torch.tensor([0.0, float("nan")]),),
            perturbed_activations=(torch.ones(2),),
            couplings=(torch.ones(2),),
            top_k=2,
        )
