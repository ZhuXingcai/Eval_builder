from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from eval_factory.ai_gateway.model_catalog import ModelCatalog
from eval_factory.ai_gateway.policy import GatewaySecurityPolicy
from eval_factory.ai_gateway.prompt_registry import PromptRegistry
from eval_factory.ai_gateway.protocols import GatewayProvider
from eval_factory.ai_gateway.receipts import GatewayRecordStore
from eval_factory.ai_gateway.routing import ModelRouteError, ModelRouter
from eval_factory.contracts.ai_gateway_v2 import (
    GatewayInvocationRequestV2,
    GatewayInvocationResultV2,
    GatewayInvocationStatusV2,
    GatewayReceiptV2,
    GatewayUsageV2,
    ModelCapabilityProfileV2,
    ModelRouteDecisionV2,
    ModelRouteRequestV2,
)
from eval_factory.contracts.core import ObjectRef


@dataclass(frozen=True, slots=True)
class ProviderInvocationResult:
    status: GatewayInvocationStatusV2
    response_body_ref: ObjectRef | None
    output_ref: ObjectRef | None
    usage: GatewayUsageV2
    failure_code: str | None

    def __post_init__(self) -> None:
        if self.status is GatewayInvocationStatusV2.SUCCEEDED:
            if self.response_body_ref is None or self.output_ref is None or self.failure_code is not None:
                raise ValueError("successful provider result requires body and output refs")
        elif self.response_body_ref is not None or self.output_ref is not None or self.failure_code is None:
            raise ValueError("failed provider result requires a failure code only")


class EmbeddedAIGateway:
    def __init__(
        self,
        *,
        router: ModelRouter,
        catalog: ModelCatalog,
        prompts: PromptRegistry,
        records: GatewayRecordStore,
        provider: GatewayProvider[ProviderInvocationResult],
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._router = router
        self._catalog = catalog
        self._prompts = prompts
        self._records = records
        self._provider = provider
        self._clock = clock or (lambda: datetime.now(UTC))

    def route(self, request: ModelRouteRequestV2) -> ModelRouteDecisionV2:
        prompt = self._prompts.get_by_ref(request.prompt_template_ref)
        if prompt.task_kind != request.task_kind:
            raise ModelRouteError("route task kind differs from prompt template")
        if not set(prompt.required_model_capabilities).issubset(request.required_capabilities):
            raise ModelRouteError("route capabilities do not satisfy prompt template")
        predecessor = None
        if request.predecessor_route_ref is not None:
            predecessor = self._records.get_route(request.predecessor_route_ref)
            if request.failed_receipt_ref is None:
                raise ModelRouteError("successor route requires a failed receipt")
            receipt = self._records.get_receipt(request.failed_receipt_ref)
            if (
                receipt.status is GatewayInvocationStatusV2.SUCCEEDED
                or receipt.route_decision_ref != predecessor.to_ref()
            ):
                raise ModelRouteError("successor route failure receipt is not authoritative")
        decision = self._router.route(request, predecessor=predecessor)
        return self._records.commit_route(request, decision)

    def get_route(
        self,
        reference: ObjectRef,
    ) -> ModelRouteDecisionV2:
        return self._records.get_route(reference)

    async def invoke(
        self,
        request: GatewayInvocationRequestV2,
        *,
        route: ModelRouteDecisionV2,
    ) -> GatewayInvocationResultV2:
        replay = self._records.get_invocation(request.to_ref())
        if replay is not None:
            if replay[0] != request or route.to_ref() != request.route_decision_ref:
                raise ValueError("gateway invocation replay authority changed")
            return replay[2]
        persisted_route = self._records.get_route(route.to_ref())
        if persisted_route != route or request.route_decision_ref != route.to_ref():
            raise ValueError("gateway invocation route is not persisted authority")
        prompt = self._prompts.get_by_ref(request.prompt_template_ref)
        GatewaySecurityPolicy.validate_invocation(request, route=route, prompt=prompt)
        model_profile = self._catalog.get_profile(route.selected_model_profile_ref)
        provider_result = await self._invoke_provider(request, model_profile=model_profile)
        receipt = GatewayReceiptV2.create(
            receipt_id=f"gateway-receipt://{request.invocation_request_id.rsplit('://', 1)[-1]}",
            invocation_request_ref=request.to_ref(),
            route_decision_ref=route.to_ref(),
            status=provider_result.status,
            response_body_ref=provider_result.response_body_ref,
            usage=provider_result.usage,
            failure_code=provider_result.failure_code,
            completed_at=self._clock(),
            audit=request.audit,
        )
        result = GatewayInvocationResultV2.create(
            invocation_result_id=(
                f"gateway-invocation-result://{request.invocation_request_id.rsplit('://', 1)[-1]}"
            ),
            invocation_request_ref=request.to_ref(),
            route_decision_ref=route.to_ref(),
            receipt_ref=receipt.to_ref(),
            status=provider_result.status,
            output_ref=provider_result.output_ref,
            failure_code=provider_result.failure_code,
            audit=request.audit,
        )
        return self._records.commit_invocation(request, receipt, result)

    async def _invoke_provider(
        self,
        request: GatewayInvocationRequestV2,
        *,
        model_profile: ModelCapabilityProfileV2,
    ) -> ProviderInvocationResult:
        try:
            return await self._provider.invoke(
                request,
                model_profile=model_profile,
            )
        except Exception:
            return ProviderInvocationResult(
                status=GatewayInvocationStatusV2.FAILED,
                response_body_ref=None,
                output_ref=None,
                usage=GatewayUsageV2(
                    input_tokens=0,
                    output_tokens=0,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                    charged_tokens=0,
                    reported_cost_micro_usd=None,
                ),
                failure_code="MODEL_PROVIDER_FAILED",
            )


__all__ = ["EmbeddedAIGateway", "ProviderInvocationResult"]
