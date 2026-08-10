from probing import legacy as cli


def test_tui_preloads_engine_before_app_run(monkeypatch, tmp_path) -> None:
    events: list[str] = []
    sentinel = object()

    def fake_from_pretrained(*_args, **_kwargs):
        events.append("load")
        return sentinel

    class FakeWorkbench:
        def __init__(self, **kwargs) -> None:
            events.append("app-init")
            assert kwargs["engine_factory"]() is sentinel

        def run(self) -> None:
            events.append("app-run")

    monkeypatch.setattr(cli.ProbeEngine, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(cli, "ProbeWorkbench", FakeWorkbench)

    cli.main(
        [
            "--local-files-only",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output",
            str(tmp_path / "runs"),
        ]
    )

    assert events == ["load", "app-init", "app-run"]
