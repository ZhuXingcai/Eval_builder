from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlparse

import httpx

from env_mock_agent.schemas import FetchResult

ALLOWED_SCHEMES = {"http", "https"}


class HttpFetchProvider:
    name = "http"

    def __init__(
        self,
        download_root: Path,
        *,
        timeout_seconds: float = 30,
        max_bytes: int = 50 * 1024 * 1024,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.download_root = download_root.expanduser().resolve()
        self.timeout_seconds = timeout_seconds
        self.max_bytes = max_bytes
        self.transport = transport

    async def fetch(self, url: str) -> FetchResult:
        parsed = urlparse(url)
        if parsed.scheme not in ALLOWED_SCHEMES:
            raise ValueError(f"unsupported fetch scheme: {parsed.scheme}")
        self.download_root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        temporary = self.download_root / f".download-{hashlib.sha256(url.encode()).hexdigest()}"
        bytes_count = 0
        try:
            async with (
                httpx.AsyncClient(
                    follow_redirects=True,
                    timeout=self.timeout_seconds,
                    transport=self.transport,
                ) as client,
                client.stream("GET", url) as response,
            ):
                response.raise_for_status()
                with temporary.open("wb") as output:
                    async for chunk in response.aiter_bytes():
                        bytes_count += len(chunk)
                        if bytes_count > self.max_bytes:
                            raise ValueError(f"download exceeds {self.max_bytes} bytes: {url}")
                        digest.update(chunk)
                        output.write(chunk)
                content_hash = digest.hexdigest()
                suffix = Path(urlparse(str(response.url)).path).suffix[:16]
                target = self.download_root / f"{content_hash}{suffix}"
                temporary.replace(target)
                return FetchResult(
                    url=str(response.url),
                    status_code=response.status_code,
                    media_type=response.headers.get("content-type", "").split(";")[0] or None,
                    sha256=content_hash,
                    content_path=str(target),
                    bytes_count=bytes_count,
                    headers={
                        key.lower(): value
                        for key, value in response.headers.items()
                        if key.lower() in {"content-type", "content-length", "last-modified", "etag"}
                    },
                )
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
