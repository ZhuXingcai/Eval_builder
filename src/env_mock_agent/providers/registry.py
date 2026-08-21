from __future__ import annotations

from env_mock_agent.providers.base import ArtifactProvider, ProviderCapability


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, ArtifactProvider] = {}
        self._asset_types: dict[str, str] = {}

    def register(self, provider: ArtifactProvider) -> None:
        self._providers[provider.name] = provider
        for asset_type in provider.asset_types:
            self._asset_types[asset_type.lower()] = provider.name

    def get(self, name: str) -> ArtifactProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            raise KeyError(f"provider not registered: {name}") from exc

    def for_asset_type(self, asset_type: str) -> ArtifactProvider | None:
        normalized = asset_type.lower().removeprefix("file_").lstrip(".")
        provider_name = self._asset_types.get(normalized)
        return self._providers.get(provider_name) if provider_name else None

    def capabilities(self) -> list[ProviderCapability]:
        return [provider.probe() for provider in self._providers.values()]

    @classmethod
    def default(cls) -> ProviderRegistry:
        from env_mock_agent.providers.code_project import CodeProjectProvider
        from env_mock_agent.providers.docx import DocxProvider
        from env_mock_agent.providers.pdf import PdfProvider
        from env_mock_agent.providers.structured import StructuredDataProvider
        from env_mock_agent.providers.text import TextProvider
        from env_mock_agent.providers.xlsx import XlsxProvider

        registry = cls()
        for provider in (
            TextProvider(),
            StructuredDataProvider(),
            DocxProvider(),
            XlsxProvider(),
            PdfProvider(),
            CodeProjectProvider(),
        ):
            registry.register(provider)
        return registry
