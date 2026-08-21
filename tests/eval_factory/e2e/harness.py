from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from urllib.parse import urlsplit

from playwright.sync_api import Browser, Page, Playwright

_MAX_LOG_BYTES = 64 * 1024
_MAX_EVENTS = 500


class BrowserE2EPreflightError(RuntimeError):
    pass


class BrowserE2EServerError(RuntimeError):
    pass


def require_chromium(playwright: Playwright) -> Path:
    executable = Path(playwright.chromium.executable_path)
    if not executable.is_file() or not os.access(
        executable,
        os.X_OK,
    ):
        raise BrowserE2EPreflightError(
            "CHROMIUM_NOT_INSTALLED: run `uv run python -m playwright install chromium`"
        )
    return executable


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@dataclass(slots=True)
class ManagedServer:
    command: tuple[str, ...]
    port: int
    log_path: Path
    cwd: Path
    environment: dict[str, str] = field(default_factory=dict)
    process: subprocess.Popen[bytes] | None = None
    _log_handle: object | None = None

    def start(self) -> None:
        if self.process is not None:
            raise BrowserE2EServerError("SERVER_ALREADY_STARTED")
        self.log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        handle = self.log_path.open("wb")
        self._log_handle = handle
        self.process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=os.environ | self.environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def wait_ready(
        self,
        path: str,
        *,
        timeout_seconds: float = 30,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        url = f"http://127.0.0.1:{self.port}{path}"
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise BrowserE2EServerError("SERVER_EXITED_BEFORE_READY")
            try:
                with urllib.request.urlopen(
                    url,
                    timeout=1,
                ) as response:
                    if response.status < 500:
                        return
            except (
                OSError,
                urllib.error.URLError,
            ):
                time.sleep(0.1)
        raise BrowserE2EServerError("SERVER_READINESS_TIMEOUT")

    def stop(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        if self._log_handle is not None:
            self._log_handle.close()  # type: ignore[union-attr]
        self.process = None
        self._log_handle = None

    def __enter__(self) -> ManagedServer:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.stop()


@dataclass(slots=True)
class BrowserDiagnostics:
    root: Path
    console: list[dict[str, str]] = field(default_factory=list)
    network: list[dict[str, object]] = field(default_factory=list)

    def attach(self, page: Page) -> None:
        page.on(
            "console",
            lambda message: self._append_console(message.type),
        )
        page.on(
            "response",
            lambda response: self._append_network(
                response.request.method,
                response.url,
                response.status,
            ),
        )

    def write_failure(
        self,
        page: Page,
        *,
        api_log: Path,
        web_log: Path,
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        page.screenshot(
            path=str(self.root / "screenshot.png"),
            full_page=True,
        )
        _write_bounded(
            self.root / "page.html",
            page.content().encode(),
        )
        _write_jsonl(
            self.root / "browser-console.jsonl",
            self.console,
        )
        _write_jsonl(
            self.root / "network.jsonl",
            self.network,
        )
        _copy_redacted_log(
            api_log,
            self.root / "api.log",
        )
        _copy_redacted_log(
            web_log,
            self.root / "vite.log",
        )

    def _append_console(self, level: str) -> None:
        if len(self.console) < _MAX_EVENTS:
            self.console.append({"level": level})

    def _append_network(
        self,
        method: str,
        url: str,
        status: int,
    ) -> None:
        if len(self.network) >= _MAX_EVENTS:
            return
        parsed = urlsplit(url)
        self.network.append(
            {
                "method": method,
                "path": parsed.path,
                "status": status,
            }
        )


@dataclass(slots=True)
class BrowserSession:
    browser: Browser
    api: ManagedServer
    web: ManagedServer
    web_url: str
    api_url: str

    def restart_api(self) -> None:
        self.api.stop()
        self.api.start()
        self.api.wait_ready("/api/plan-reviews/contract")

    def close(self) -> None:
        self.browser.close()
        self.web.stop()
        self.api.stop()


def _write_bounded(path: Path, payload: bytes) -> None:
    path.write_bytes(payload[:_MAX_LOG_BYTES])


def _write_jsonl(
    path: Path,
    values: list[dict[str, object]] | list[dict[str, str]],
) -> None:
    encoded = b"".join(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
        for value in values[:_MAX_EVENTS]
    )
    _write_bounded(path, encoded)


def _copy_redacted_log(
    source: Path,
    destination: Path,
) -> None:
    if not source.exists():
        destination.write_text(
            "log unavailable\n",
            encoding="utf-8",
        )
        return
    payload = source.read_bytes()[-_MAX_LOG_BYTES:]
    text = payload.decode(errors="replace")
    for marker in (
        "private-reference",
        "grader-rule",
        "hidden-condition",
        "credential",
        "raw-trace",
    ):
        text = text.replace(marker, "<redacted>")
    _write_bounded(destination, text.encode())


__all__ = [
    "BrowserDiagnostics",
    "BrowserE2EPreflightError",
    "BrowserE2EServerError",
    "BrowserSession",
    "ManagedServer",
    "free_port",
    "require_chromium",
]
