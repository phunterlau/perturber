from types import SimpleNamespace

import pytest

from probing.contracts import ModelRequest
from probing.errors import ModelPolicyError
from probing.models import ModelManager


def test_unknown_remote_file_size_rejects_download_budget(monkeypatch, tmp_path) -> None:
    info = SimpleNamespace(
        sha="revision",
        siblings=[
            SimpleNamespace(rfilename="config.json", size=100),
            SimpleNamespace(rfilename="weights.safetensors", size=None),
        ],
    )
    monkeypatch.setattr(
        "probing.models.HfApi",
        lambda: SimpleNamespace(model_info=lambda *_args, **_kwargs: info),
    )
    manager = ModelManager(tmp_path / "cache")

    with pytest.raises(ModelPolicyError, match="unknown sizes") as captured:
        manager.fetch(ModelRequest(id="fake/model"), max_download_bytes=1_000_000)

    assert captured.value.details["unknown_size_files"] == ["weights.safetensors"]


def test_fetch_pins_the_budgeted_revision_and_records_symbolic_ref(
    monkeypatch, tmp_path
) -> None:
    manager = ModelManager(tmp_path / "cache")
    request = ModelRequest(id="fake/model")
    revision = "a" * 40
    snapshot = tmp_path / "cache" / "models--fake--model" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    calls = []

    monkeypatch.setattr(manager, "remote_size", lambda _request: (100, revision))

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr("probing.models.snapshot_download", fake_download)

    result = manager.fetch(request, max_download_bytes=100)

    assert result == snapshot
    assert calls[0]["revision"] == revision
    assert (
        tmp_path / "cache" / "models--fake--model" / "refs" / "main"
    ).read_text(encoding="utf-8") == revision


def test_manager_repairs_legacy_newline_reference_for_offline_loading(tmp_path) -> None:
    cache = tmp_path / "cache"
    repository = cache / "models--fake--model"
    revision = "b" * 40
    snapshot = repository / "snapshots" / revision
    snapshot.mkdir(parents=True)
    reference = repository / "refs" / "main"
    reference.parent.mkdir(parents=True)
    reference.write_text(revision + "\n", encoding="utf-8")

    manager = ModelManager(cache)

    assert reference.read_text(encoding="utf-8") == revision
    assert manager.resolve_cached_snapshot(ModelRequest(id="fake/model")) == snapshot
