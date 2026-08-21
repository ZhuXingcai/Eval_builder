from __future__ import annotations

import asyncio
import os
import re
import signal
from pathlib import Path

SECRET_PATTERN = re.compile(r"(?i)(api[_-]?key|authorization|token|secret)([\"'\s:=]+)([^\s\"']+)")


async def drain_stream(
    stream: asyncio.StreamReader | None,
    *,
    limit_bytes: int = 2 * 1024 * 1024,
) -> str:
    if stream is None:
        return ""
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await stream.read(64 * 1024)
        if not chunk:
            break
        if size < limit_bytes:
            remaining = limit_bytes - size
            chunks.append(chunk[:remaining])
            size += min(len(chunk), remaining)
    return redact_text(b"".join(chunks).decode(errors="replace").strip())


def redact_text(value: str) -> str:
    return SECRET_PATTERN.sub(r"\1\2[REDACTED]", value)


async def terminate_process(process: asyncio.subprocess.Process, timeout: float = 5.0) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout)
    except TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
        await process.wait()


def isolated_environment(workspace: Path) -> dict[str, str]:
    environment = os.environ.copy()
    home = workspace / ".runtime-home"
    cache = workspace / ".runtime-cache"
    tmp = workspace / ".runtime-tmp"
    for directory in (home, cache, tmp):
        directory.mkdir(parents=True, exist_ok=True)
    environment.update(
        {
            "HOME": str(home),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "XDG_CACHE_HOME": str(cache),
            "TMPDIR": str(tmp),
            "PYTHONIOENCODING": "utf-8",
        }
    )
    return environment
