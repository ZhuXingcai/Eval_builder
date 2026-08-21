from __future__ import annotations

from env_mock_agent.models.anthropic_client import AnthropicStructuredClient
from env_mock_agent.models.ark_client import ArkStructuredClient
from env_mock_agent.models.base import StructuredModelClient


def get_structured_client(provider: str, model: str | None = None) -> StructuredModelClient:
    normalized = provider.strip().lower()
    if normalized == "anthropic":
        return AnthropicStructuredClient(model or "claude-opus-4-7")
    if normalized == "ark":
        return ArkStructuredClient(model or "glm-5-2-260617")
    raise KeyError(f"structured model provider is not configured: {provider}")
