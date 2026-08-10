import pytest

from probing.errors import EndpointError
from probing.server_process import start_server


def test_managed_server_rejects_occupied_port_before_spawning(
    monkeypatch, tmp_path
) -> None:
    class OccupiedSocket:
        socket_option = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def setsockopt(self, level, option, value) -> None:
            self.socket_option = (level, option, value)

        def bind(self, _address) -> None:
            raise OSError(48, "Address already in use")

    monkeypatch.setattr(
        "probing.server_process.socket.socket",
        lambda *_args, **_kwargs: OccupiedSocket(),
    )

    with pytest.raises(EndpointError, match="cannot start probe server"):
        start_server(tmp_path / "workspace", tmp_path / "cache", port=8765)
