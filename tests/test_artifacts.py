import csv
import json

from probing.artifacts import export_result
from helpers import fake_result


def test_export_writes_versioned_manifest_and_tables(tmp_path) -> None:
    result = fake_result()
    destination = export_result(result, tmp_path)

    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["algorithm_version"] == "perturbation-probing-mvp-v1"
    assert manifest["model"]["resolved_revision"] == "fixture"
    assert manifest["observable"]["target"][0]["token_id"] == 0
    assert manifest["original_prediction"]["decoded"] == "Yes"

    with (destination / "neurons.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["importance"] == "4.0"

    with (destination / "layers.csv").open(newline="") as handle:
        layers = list(csv.DictReader(handle))
    assert len(layers) == 1
