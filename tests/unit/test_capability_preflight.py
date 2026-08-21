from __future__ import annotations

from env_mock_agent.providers import ProviderRegistry, preflight_dependencies
from env_mock_agent.schemas import DependencySpec


def test_preflight_blocks_unsupported_format_without_silent_fallback() -> None:
    decisions = preflight_dependencies(
        [
            DependencySpec(
                dependency_id="D-001",
                path="input.pptx",
                asset_type="pptx",
            ),
            DependencySpec(
                dependency_id="D-002",
                path="input.xlsx",
                asset_type="xlsx",
            ),
        ],
        ProviderRegistry.default(),
    )
    assert decisions[0].available is False
    assert decisions[0].provider is None
    assert "no deterministic provider" in decisions[0].reason
    assert decisions[1].available is True
    assert decisions[1].provider == "xlsx"
