from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any

import httpx

from .errors import EndpointError


def _runtime_file(workspace: Path) -> Path:
    return workspace.resolve() / "server.json"


def server_status(workspace: Path) -> dict[str, Any]:
    path = _runtime_file(workspace)
    if not path.is_file():
        return {"running": False, "runtime_file": str(path)}
    try:
        runtime = json.loads(path.read_text(encoding="utf-8"))
        pid = int(runtime["pid"])
        os.kill(pid, 0)
        response = httpx.get(f"{runtime['endpoint']}/api/v1/health", timeout=1.0)
        healthy = response.status_code == 200
    except Exception:
        return {"running": False, "runtime_file": str(path)}
    return {"running": True, "healthy": healthy, **runtime}


def start_server(workspace: Path, cache_dir: Path, *, port: int) -> dict[str, Any]:
    current = server_status(workspace)
    if current.get("running"):
        return current
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_check:
        # Match the HTTP server's restart behavior. Without SO_REUSEADDR, a
        # recently stopped daemon can leave 8765 unavailable during TIME_WAIT
        # even though no process is listening on it.
        port_check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            port_check.bind(("127.0.0.1", port))
        except OSError as exc:
            raise EndpointError(
                f"cannot start probe server on 127.0.0.1:{port}: {exc}",
                hint="Choose a free loopback port with '--port'.",
            ) from exc
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    log_path = workspace / "server.log"
    command = [
        sys.executable,
        "-m",
        "probing",
        "--workspace",
        str(workspace),
        "--cache-dir",
        str(cache_dir.resolve()),
        "serve",
        "--port",
        str(port),
    ]
    with log_path.open("ab") as log:
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
            close_fds=True,
        )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        status = server_status(workspace)
        if status.get("running") and status.get("healthy"):
            return status
        time.sleep(0.1)
    raise EndpointError(
        "server did not become healthy within 15 seconds",
        hint=f"Inspect {log_path}",
    )


def stop_server(workspace: Path) -> dict[str, Any]:
    status = server_status(workspace)
    if not status.get("running"):
        return status
    pid = int(status["pid"])
    command = subprocess.run(
        ["/bin/ps", "-p", str(pid), "-o", "command="],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    if "probing" not in command or "serve" not in command:
        raise EndpointError(
            f"refusing to signal PID {pid}; it does not look like a probe server"
        )
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 5
    stopped = False
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            stopped = True
            break
        time.sleep(0.05)
    if not stopped:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            stopped = True
    if not stopped:
        raise EndpointError(
            f"probe server PID {pid} did not stop within 5 seconds",
            hint="Inspect the server log and retry; runtime metadata was preserved.",
        )
    path = _runtime_file(workspace)
    if path.is_file():
        path.unlink()
    token_path = workspace.resolve() / "server.token"
    if token_path.is_file():
        token_path.unlink()
    return {"running": False, "pid": pid}
