import pytest

from probing.engine import ProbeEngine
from helpers import FakeAdapter, fake_spec


def test_engine_runs_exact_two_pass_pair_and_reports_diagnostics() -> None:
    result = ProbeEngine(FakeAdapter()).analyze(fake_spec())

    assert result.original_gap == pytest.approx(-2.0)
    assert result.perturbed_gap == pytest.approx(3.0)
    assert result.measured_delta == pytest.approx(5.0)
    assert result.predicted_delta == pytest.approx(5.0)
    assert result.original_prediction.decoded == "Yes"
    assert result.perturbed_prediction.decoded == "No"
    assert result.ffn_skip_original == pytest.approx(0.5)
    assert result.ffn_skip_perturbed == pytest.approx(0.5)
    assert result.ffn_skip_mean == pytest.approx(0.5)
    assert result.runtime["logical_forward_passes"] == 2
    assert result.total_neuron_count == 2
    assert "Exploratory" in result.warnings[0]
