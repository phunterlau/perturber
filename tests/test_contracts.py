import io
from pathlib import Path

import pytest
from pydantic import ValidationError

from probing.contracts import ArtifactRef, ExperimentSet
from probing.specs import (
    example_rank_spec,
    example_replication_spec,
    load_spec,
    science_hash,
)


def test_example_is_strict_and_has_stable_science_hash() -> None:
    spec = example_rank_spec()
    assert spec.execution.max_forward_passes == 2
    assert spec.execution.max_artifact_bytes == 50_000_000
    assert len(science_hash(spec)) == 64

    with pytest.raises(ValidationError):
        type(spec).model_validate({**spec.model_dump(), "invented": True})


def test_yaml_duplicate_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate YAML key"):
        load_spec(
            "-",
            stdin=io.StringIO(
                "kind: rank\nkind: rank\nschema_version: probe.rank/v1\n"
            ),
        )


def test_json_duplicate_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key: 'kind'"):
        load_spec(
            "-",
            stdin=io.StringIO(
                '{"schema_version":"probe.rank/v1","kind":"rank","kind":"rank"}'
            ),
        )


def test_replication_example_has_matching_aggregation_and_budget() -> None:
    spec = example_replication_spec()

    assert len(spec.pairs) == 3
    assert spec.ranking.pair_aggregation == "signed_mean"
    assert spec.execution.max_forward_passes == 6


@pytest.mark.parametrize(
    "filename",
    [
        "paper-safety-bpe.yaml",
        "paper-language-en-zh.yaml",
        "paper-factual-entity.yaml",
        "paper-code-vs-explain.yaml",
        "paper-cot-complex-simple.yaml",
    ],
)
def test_paper_case_smoke_specs_are_strict_bounded_replications(filename) -> None:
    path = Path(__file__).parents[1] / "examples" / filename

    spec = load_spec(str(path))

    assert len(spec.pairs) == 3
    assert spec.ranking.pair_aggregation == "signed_mean"
    assert spec.execution.max_forward_passes == 6
    assert spec.execution.allow_download is False
    assert spec.tags["fidelity"] in {"representative-smoke", "capability-smoke"}


@pytest.mark.parametrize("path", ["../secret", "/absolute", "a/../b", "a\\b", "./a"])
def test_artifact_paths_must_be_portable_relative_paths(path) -> None:
    with pytest.raises(ValidationError, match="normalized relative POSIX"):
        ArtifactRef(
            path=path,
            media_type="application/octet-stream",
            sha256="0" * 64,
            size_bytes=0,
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("model", "id"), "   "),
        (("pairs", 0, "id"), "\t"),
        (("pairs", 0, "original"), "\n"),
        (("observable", "name"), " "),
    ],
)
def test_scientific_identity_fields_reject_blank_values(path, value) -> None:
    data = example_rank_spec().model_dump(mode="json")
    target = data
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value

    with pytest.raises(ValidationError, match="must not be blank"):
        type(example_rank_spec()).model_validate(data)


def test_science_hash_excludes_execution_metadata_but_tracks_prompt_changes() -> None:
    spec = example_rank_spec()
    presentation_change = spec.model_copy(
        update={
            "name": "renamed",
            "tags": {"owner": "agent"},
            "execution": spec.execution.model_copy(update={"seed": 99}),
        }
    )
    changed_pair = spec.model_copy(
        update={
            "pairs": (
                spec.pairs[0].model_copy(update={"perturbed": "A different premise"}),
            )
        }
    )

    assert science_hash(presentation_change) == science_hash(spec)
    assert science_hash(changed_pair) != science_hash(spec)


def test_v1_restricts_capture_to_the_first_generation_decision_position() -> None:
    data = example_rank_spec().model_dump(mode="json")
    data["capture"]["position"] = 0

    with pytest.raises(ValidationError, match="Input should be -1"):
        type(example_rank_spec()).model_validate(data)


def test_empty_observable_token_and_non_finite_metadata_are_rejected() -> None:
    empty_token = example_rank_spec().model_dump(mode="json")
    empty_token["observable"]["target_tokens"] = [""]
    with pytest.raises(ValidationError, match="token strings must not be empty"):
        type(example_rank_spec()).model_validate(empty_token)

    non_finite = example_rank_spec().model_dump(mode="json")
    non_finite["pairs"][0]["metadata"] = {"score": float("nan")}
    with pytest.raises(ValidationError, match="finite number"):
        type(example_rank_spec()).model_validate(non_finite)


def test_experiment_set_supports_splits_and_structured_tool_prompts() -> None:
    value = ExperimentSet.model_validate(
        {
            "name": "tool-routing",
            "pairs": [
                {
                    "id": "weather",
                    "split": "heldout",
                    "original_messages": [
                        {"role": "user", "content": "Check the weather in Paris."}
                    ],
                    "perturbed_messages": [
                        {"role": "user", "content": "Write a poem about Paris."}
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "weather",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                }
            ],
        }
    )

    assert value.pairs[0].split == "heldout"
    assert value.pairs[0].original is None
    assert value.pairs[0].tools[0]["function"]["name"] == "weather"
