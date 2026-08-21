from __future__ import annotations

from env_mock_agent.retrieval.base import FetchProvider, SearchProvider


class RetrievalRegistry:
    def __init__(self) -> None:
        self.search_providers: dict[str, SearchProvider] = {}
        self.fetch_providers: dict[str, FetchProvider] = {}

    def register_search(self, provider: SearchProvider) -> None:
        self.search_providers[provider.name] = provider

    def register_fetch(self, provider: FetchProvider) -> None:
        self.fetch_providers[provider.name] = provider

    def search(self, name: str) -> SearchProvider:
        try:
            return self.search_providers[name]
        except KeyError as exc:
            raise KeyError(f"search provider not registered: {name}") from exc

    def fetch(self, name: str) -> FetchProvider:
        try:
            return self.fetch_providers[name]
        except KeyError as exc:
            raise KeyError(f"fetch provider not registered: {name}") from exc
