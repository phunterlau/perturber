import pytest

from probing.contracts import PerturbationTemplate
from probing.perturbations import compile_perturbations


def test_perturbation_template_compiles_splits_and_confound_diagnostics() -> None:
    template = PerturbationTemplate.model_validate(
        {
            "name": "capital-swap",
            "target_factor": "premise correctness",
            "original_template": "The capital of {country} is {correct}. Answer Yes or No.",
            "perturbed_template": "The capital of {country} is {wrong}. Answer Yes or No.",
            "cases": [
                {
                    "id": "france",
                    "values": {
                        "country": "France",
                        "correct": "Paris",
                        "wrong": "London",
                    },
                    "split": "heldout",
                }
            ],
        }
    )

    compiled = compile_perturbations(template)

    pair = compiled.experiment_set.pairs[0]
    assert pair.split == "heldout"
    assert pair.metadata["target_factor"] == "premise correctness"
    assert compiled.diagnostics[0].changed_word_count == 1
    assert compiled.diagnostics[0].shared_word_fraction > 0.8


def test_perturbation_template_rejects_missing_case_value() -> None:
    template = PerturbationTemplate.model_validate(
        {
            "name": "missing",
            "target_factor": "entity",
            "original_template": "{entity}",
            "perturbed_template": "{replacement}",
            "cases": [{"id": "one", "values": {"entity": "Paris"}}],
        }
    )

    with pytest.raises(ValueError, match="replacement"):
        compile_perturbations(template)
