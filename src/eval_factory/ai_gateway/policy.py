from __future__ import annotations

from eval_factory.contracts.ai_gateway_v2 import (
    GatewayInvocationRequestV2,
    ModelRouteDecisionV2,
    PromptTemplateV2,
)


class GatewayPolicyError(RuntimeError):
    pass


class GatewaySecurityPolicy:
    @staticmethod
    def validate_invocation(
        request: GatewayInvocationRequestV2,
        *,
        route: ModelRouteDecisionV2,
        prompt: PromptTemplateV2,
    ) -> None:
        if request.route_decision_ref != route.to_ref():
            raise GatewayPolicyError("invocation does not bind persisted route")
        if request.prompt_template_ref != prompt.to_ref():
            raise GatewayPolicyError("invocation does not bind current prompt template")
        if request.output_schema_ref != prompt.output_schema_ref:
            raise GatewayPolicyError("invocation output schema differs from prompt template")


__all__ = ["GatewayPolicyError", "GatewaySecurityPolicy"]
