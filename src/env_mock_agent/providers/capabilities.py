from __future__ import annotations

from pydantic import BaseModel

from env_mock_agent.providers.registry import ProviderRegistry
from env_mock_agent.schemas import DependencySpec


class CapabilityDecision(BaseModel):
    dependency_id: str
    asset_type: str
    available: bool
    provider: str | None = None
    reason: str = ""


def preflight_dependencies(
    dependencies: list[DependencySpec],
    registry: ProviderRegistry,
) -> list[CapabilityDecision]:
    decisions: list[CapabilityDecision] = []
    for dependency in dependencies:
        provider = registry.for_asset_type(dependency.asset_type)
        if provider is None:
            decisions.append(
                CapabilityDecision(
                    dependency_id=dependency.dependency_id,
                    asset_type=dependency.asset_type,
                    available=False,
                    reason=f"no deterministic provider for {dependency.asset_type}",
                )
            )
            continue
        capability = provider.probe()
        decisions.append(
            CapabilityDecision(
                dependency_id=dependency.dependency_id,
                asset_type=dependency.asset_type,
                available=capability.available,
                provider=provider.name if capability.available else None,
                reason=capability.reason,
            )
        )
    return decisions
