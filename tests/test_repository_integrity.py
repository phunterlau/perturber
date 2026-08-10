import json

import pytest

from probing.errors import ArtifactError
from test_service import fake_rank_spec, make_service


def test_verify_detects_content_tampering_missing_and_untracked_files(tmp_path) -> None:
    service = make_service(tmp_path)
    outcome = service.execute(fake_rank_spec())
    directory = outcome.run_directory

    neurons = directory / "neurons.csv"
    original = neurons.read_bytes()
    neurons.write_bytes(original.replace(b"rank", b"rAnk", 1))
    (directory / "layers.csv").unlink()
    (directory / "unexpected.txt").write_text("not in manifest", encoding="utf-8")

    failures = service.repository.verify(outcome.manifest.run_id)

    assert "sha256:neurons.csv" in failures
    assert "missing:layers.csv" in failures
    assert "untracked:unexpected.txt" in failures


def test_corrupt_manifest_cannot_escape_the_run_directory(tmp_path) -> None:
    service = make_service(tmp_path)
    outcome = service.execute(fake_rank_spec())
    path = outcome.run_directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["path"] = "../outside.txt"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactError, match="invalid manifest metadata"):
        service.repository.verify(outcome.manifest.run_id)
