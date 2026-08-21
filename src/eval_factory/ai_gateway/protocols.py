from __future__ import annotations

from typing import Protocol

from eval_factory.contracts.ai_gateway_v2 import (
    GatewayInvocationRequestV2,
    GatewayInvocationResultV2,
    ModelCapabilityProfileV2,
    ModelRouteDecisionV2,
    ModelRouteRequestV2,
)
from eval_factory.contracts.core import ObjectRef


class AIGateway(Protocol):
    def route(self, request: ModelRouteRequestV2) -> ModelRouteDecisionV2: ...

    def get_route(self, reference: ObjectRef) -> ModelRouteDecisionV2: ...

    async def invoke(
        self,
        request: GatewayInvocationRequestV2,
        *,
        route: ModelRouteDecisionV2,
    ) -> GatewayInvocationResultV2: ...


class GatewayProvider[ProviderResultT](Protocol):
    async def invoke(
        self,
        request: GatewayInvocationRequestV2,
        *,
        model_profile: ModelCapabilityProfileV2,
    ) -> ProviderResultT: ...


__all__ = ["AIGateway", "GatewayProvider"]
