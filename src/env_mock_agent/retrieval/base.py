from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from env_mock_agent.schemas import FetchResult, SearchHit


class SearchProvider(Protocol):
    name: str

    async def search(self, query: str, *, limit: int = 10) -> Sequence[SearchHit]: ...


class FetchProvider(Protocol):
    name: str

    async def fetch(self, url: str) -> FetchResult: ...
