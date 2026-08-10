from dataclasses import replace

import pytest
import torch

from probing.aggregation import aggregate_analyses
from probing.engine import ProbeEngine
from helpers import FakeAdapter, fake_spec


def test_aggregation_rejects_pair_dependent_structural_couplings() -> None:
    engine = ProbeEngine(FakeAdapter())
    first = engine.analyze_details(fake_spec())
    second = replace(first, coupling_by_layer=(torch.tensor([9.0, -1.0]),))

    with pytest.raises(ValueError, match="different couplings"):
        aggregate_analyses(
            science_hash="0" * 64,
            pair_ids=("first", "second"),
            analyses=(first, second),
            top_k=2,
        )


def test_aggregate_equal_rms_scores_use_layer_then_neuron_order() -> None:
    engine = ProbeEngine(FakeAdapter())
    first = engine.analyze_details(fake_spec())
    tied = replace(
        first,
        importance_by_layer=(torch.tensor([1.0, -1.0]),),
        coupling_by_layer=(torch.tensor([1.0, 1.0]),),
        original_activation_by_layer=(torch.zeros(2),),
        perturbed_activation_by_layer=(torch.tensor([1.0, -1.0]),),
    )

    aggregate = aggregate_analyses(
        science_hash="0" * 64,
        pair_ids=("only",),
        analyses=(tied,),
        top_k=2,
    )

    assert [item.neuron for item in aggregate.summary.neurons] == [0, 1]


def test_heldout_pairs_are_saved_but_do_not_select_ranked_neurons() -> None:
    engine = ProbeEngine(FakeAdapter())
    discovery = engine.analyze_details(fake_spec())
    heldout = replace(
        discovery,
        importance_by_layer=(torch.tensor([0.0, 1000.0]),),
        original_activation_by_layer=(torch.zeros(2),),
        perturbed_activation_by_layer=(torch.tensor([0.0, -1000.0]),),
    )

    aggregate = aggregate_analyses(
        science_hash="0" * 64,
        pair_ids=("discovery", "heldout"),
        pair_splits=("discovery", "heldout"),
        analyses=(discovery, heldout),
        top_k=2,
    )

    assert aggregate.summary.neurons[0].neuron == 0
    assert aggregate.summary.neurons[0].importance_rms == pytest.approx(4.0)
    assert aggregate.summary.split_counts == {"discovery": 1, "heldout": 1}
    assert aggregate.summary.pairs[1].split == "heldout"
    assert "importance.pair_1.layer_0" in aggregate.tensors
