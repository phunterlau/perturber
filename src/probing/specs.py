from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, TextIO

import yaml

from .contracts import (
    AttentionHeadInterventionSpec,
    AttentionHeadRankSpec,
    AttentionTraceSpec,
    DirectionInjectionSpec,
    ExperimentSpec,
    InterventionSpec,
    QualificationSpec,
    RankSpec,
)


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _load_json(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_construct_json_object)


def _construct_mapping(
    loader: UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def hash_value(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _hash_payload(spec: ExperimentSpec, *, science: bool) -> dict[str, Any]:
    exclude = {"execution", "name", "description", "tags"} if science else set()
    payload = spec.model_dump(mode="json", exclude=exclude)
    # Keep v1 plain-text rank hashes stable after structured prompt/split support.
    # Non-default structured fields remain part of scientific and request identity.
    if isinstance(spec, RankSpec):
        for pair in payload.get("pairs", []):
            if not pair.get("original_messages"):
                pair.pop("original_messages", None)
            if not pair.get("perturbed_messages"):
                pair.pop("perturbed_messages", None)
            if not pair.get("tools"):
                pair.pop("tools", None)
            if pair.get("split") == "discovery":
                pair.pop("split", None)
    return payload


def science_hash(spec: ExperimentSpec) -> str:
    payload = _hash_payload(spec, science=True)
    return hash_value(payload)


def request_hash(spec: ExperimentSpec) -> str:
    return hash_value(_hash_payload(spec, science=False))


def parse_spec_data(data: Any) -> ExperimentSpec:
    if not isinstance(data, dict):
        raise ValueError("experiment spec must be an object")
    kind = data.get("kind")
    models = {
        "rank": RankSpec,
        "qualify": QualificationSpec,
        "intervention": InterventionSpec,
        "direction": DirectionInjectionSpec,
        "attention_rank": AttentionHeadRankSpec,
        "attention_intervention": AttentionHeadInterventionSpec,
        "attention_trace": AttentionTraceSpec,
    }
    try:
        model = models[kind]
    except KeyError as exc:
        raise ValueError(
            f"unsupported experiment kind {kind!r}; supported: {sorted(models)}"
        ) from exc
    return model.model_validate(data)


def load_document(path: str | Path, stdin: TextIO | None = None) -> Any:
    source = str(path)
    if source == "-":
        text = (stdin or sys.stdin).read()
        suffix = ""
    else:
        file_path = Path(path)
        text = file_path.read_text(encoding="utf-8")
        suffix = file_path.suffix.lower()

    if suffix == ".json":
        data = _load_json(text)
    elif suffix in {".yaml", ".yml"}:
        data = yaml.load(text, Loader=UniqueKeyLoader)
    else:
        try:
            data = _load_json(text)
        except json.JSONDecodeError:
            data = yaml.load(text, Loader=UniqueKeyLoader)
    return data


def load_spec(path: str | Path, stdin: TextIO | None = None) -> ExperimentSpec:
    return parse_spec_data(load_document(path, stdin=stdin))


def example_rank_spec() -> RankSpec:
    return RankSpec.model_validate(
        {
            "schema_version": "probe.rank/v1",
            "kind": "rank",
            "name": "agreement-capital",
            "description": "Correct versus wrong premise agreement probe.",
            "model": {"id": "Qwen/Qwen3-0.6B"},
            "pairs": [
                {
                    "id": "capital",
                    "original": "The capital of France is Paris, right? Answer only Yes or No, with no explanation.",
                    "perturbed": "The capital of France is London, right? Answer only Yes or No, with no explanation.",
                }
            ],
            "observable": {
                "name": "agreement:no-minus-yes",
                "target_tokens": ["No"],
                "control_tokens": ["Yes"],
            },
            "execution": {
                "max_forward_passes": 2,
                "max_artifact_bytes": 50_000_000,
            },
        }
    )


def example_replication_spec() -> RankSpec:
    value = example_rank_spec().model_dump(mode="json")
    value.update(
        {
            "name": "agreement-replication",
            "description": (
                "Three benign correct-to-wrong premise pairs for a replicated "
                "ranking smoke experiment."
            ),
            "pairs": [
                value["pairs"][0],
                {
                    "id": "arithmetic",
                    "original": (
                        "Two plus two equals four, right? Answer only Yes or No, "
                        "with no explanation."
                    ),
                    "perturbed": (
                        "Two plus two equals five, right? Answer only Yes or No, "
                        "with no explanation."
                    ),
                },
                {
                    "id": "science",
                    "original": (
                        "Pure water freezes at 0 degrees Celsius, right? Answer only "
                        "Yes or No, with no explanation."
                    ),
                    "perturbed": (
                        "Pure water freezes at 50 degrees Celsius, right? Answer only "
                        "Yes or No, with no explanation."
                    ),
                },
            ],
            "ranking": {
                **value["ranking"],
                "pair_aggregation": "rms",
            },
            "execution": {
                **value["execution"],
                "max_forward_passes": 6,
                "max_artifact_bytes": 100_000_000,
            },
            "tags": {
                "family": "agreement",
                "source": "paper-derived-smoke",
            },
        }
    )
    return RankSpec.model_validate(value)
