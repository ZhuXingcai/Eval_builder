from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from eval_factory.ai_gateway.invocation import (
    EmbeddedAIGateway,
    ProviderInvocationResult,
)
from eval_factory.ai_gateway.model_catalog import ModelCatalog
from eval_factory.ai_gateway.prompt_registry import PromptRegistry
from eval_factory.ai_gateway.receipts import GatewayRecordStore
from eval_factory.ai_gateway.routing import ModelRouter
from eval_factory.contracts.ai_gateway_v2 import (
    GatewayInvocationRequestV2,
    GatewayInvocationStatusV2,
    GatewayUsageV2,
    ModelCapabilityProfileV2,
    ModelHealthSnapshotV2,
    ModelPriceScheduleV2,
    ModelQualityBaselineV2,
    ModelRoutingPolicyV2,
    PromptTemplateV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding

HASH = "a" * 64
NOW = datetime(2026, 8, 6, tzinfo=UTC)


def ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v2",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}/{version}",
        object_version=version,
        object_sha256=HASH,
    )


def audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="attachment-gateway-test",
        governing_versions=(
            VersionBinding(
                component="graph-engineered-eval-factory",
                version="attachment-gateway-v1",
                sha256=HASH,
            ),
        ),
    )


def prompt(
    *,
    task_kind: str,
    agent_role: str,
    input_type: str,
    output_type: str,
) -> PromptTemplateV2:
    return PromptTemplateV2.create(
        prompt_template_id=f"prompt-template://{task_kind}/v1",
        agent_role=agent_role,
        task_kind=task_kind,
        system_template_ref=ref(
            "prompt-template-content",
            f"{task_kind}-system",
        ),
        instruction_template_ref=ref(
            "prompt-template-content",
            f"{task_kind}-instruction",
        ),
        required_input_object_types=(input_type,),
        output_schema_ref=ref("json-schema", output_type),
        allowed_tool_ids=(),
        required_model_capabilities=("reasoning", "structured-output"),
        injection_policy_ref=ref(
            "prompt-injection-policy",
            task_kind,
        ),
        template_version=1,
        audit=audit(),
    )


def profile(
    name: str,
    *,
    capabilities: tuple[str, ...] = (
        "reasoning",
        "structured-output",
    ),
    data_classifications: tuple[str, ...] = ("RESTRICTED_TRACE_DERIVED",),
) -> ModelCapabilityProfileV2:
    return ModelCapabilityProfileV2.create(
        model_profile_id=f"model-profile://{name}",
        provider_id="provider://offline",
        model_id=f"model://{name}",
        model_version="2026-08",
        capabilities=capabilities,
        supported_data_classifications=data_classifications,
        supported_residencies=("LOCAL",),
        context_limit_tokens=100_000,
        output_limit_tokens=16_000,
        availability="AVAILABLE",
        audit=audit(),
    )


class DeterministicAttachmentProvider:
    def __init__(
        self,
        outputs: dict[tuple[str, str, str, str], ObjectRef],
        *,
        failed_prompt_refs: tuple[ObjectRef, ...] = (),
    ) -> None:
        self.outputs = outputs
        self.failed_prompt_refs = {_ref_key(reference) for reference in failed_prompt_refs}
        self.calls: list[tuple[GatewayInvocationRequestV2, ModelCapabilityProfileV2]] = []

    async def invoke(
        self,
        request: GatewayInvocationRequestV2,
        *,
        model_profile: ModelCapabilityProfileV2,
    ) -> ProviderInvocationResult:
        self.calls.append((request, model_profile))
        if _ref_key(request.prompt_template_ref) in self.failed_prompt_refs:
            return ProviderInvocationResult(
                status=GatewayInvocationStatusV2.FAILED,
                response_body_ref=None,
                output_ref=None,
                usage=_usage(),
                failure_code="MODEL_PROVIDER_FAILED",
            )
        output_ref = self.outputs[_ref_key(request.prompt_template_ref)]
        return ProviderInvocationResult(
            status=GatewayInvocationStatusV2.SUCCEEDED,
            response_body_ref=ref(
                "model-response-content",
                request.invocation_request_id,
            ),
            output_ref=output_ref,
            usage=_usage(),
            failure_code=None,
        )


def build_gateway(
    tmp_path: Path,
    *,
    prompts: tuple[PromptTemplateV2, ...],
    profiles: tuple[ModelCapabilityProfileV2, ...],
    agent_definition_refs: tuple[ObjectRef, ...],
    provider: DeterministicAttachmentProvider,
) -> EmbeddedAIGateway:
    baselines = tuple(
        ModelQualityBaselineV2.create(
            baseline_id=(
                "model-quality-baseline://"
                f"{profile_value.model_profile_id.rsplit('://', 1)[-1]}"
                f"/{prompt_value.task_kind}"
            ),
            model_profile_ref=profile_value.to_ref(),
            task_kind=prompt_value.task_kind,
            benchmark_ref=ref(
                "model-quality-benchmark",
                (f"{profile_value.model_profile_id.rsplit('://', 1)[-1]}-{prompt_value.task_kind}"),
            ),
            quality_basis_points=9200,
            minimum_sample_count=100,
            audit=audit(),
        )
        for profile_value in profiles
        for prompt_value in prompts
    )
    health = tuple(
        ModelHealthSnapshotV2.create(
            health_snapshot_id=(f"model-health-snapshot://{value.model_profile_id.rsplit('://', 1)[-1]}"),
            model_profile_ref=value.to_ref(),
            observed_at=NOW,
            availability="HEALTHY",
            success_basis_points=9900,
            latency_milliseconds=500,
            audit=audit(),
        )
        for value in profiles
    )
    prices = tuple(
        ModelPriceScheduleV2.create(
            price_schedule_id=(f"model-price-schedule://{value.model_profile_id.rsplit('://', 1)[-1]}"),
            model_profile_ref=value.to_ref(),
            input_micro_usd_per_million_tokens=1,
            output_micro_usd_per_million_tokens=1,
            effective_from=NOW,
            audit=audit(),
        )
        for value in profiles
    )
    catalog = ModelCatalog(
        profiles=profiles,
        quality_baselines=baselines,
        health_snapshots=health,
        price_schedules=prices,
    )
    routing_policy = ModelRoutingPolicyV2.create(
        routing_policy_id="model-routing-policy://attachment-specialists",
        allowed_provider_ids=("provider://offline",),
        allowed_agent_definition_refs=tuple(sorted(agent_definition_refs, key=_ref_key)),
        allowed_model_profile_refs=tuple(
            sorted(
                (value.to_ref() for value in profiles),
                key=_ref_key,
            )
        ),
        quality_weight=100,
        health_weight=10,
        latency_weight=1,
        cost_weight=1,
        audit=audit(),
    )
    return EmbeddedAIGateway(
        router=ModelRouter(catalog, routing_policy),
        catalog=catalog,
        prompts=PromptRegistry(prompts),
        records=GatewayRecordStore(tmp_path / "gateway-records.sqlite3"),
        provider=provider,
        clock=lambda: NOW,
    )


def _usage() -> GatewayUsageV2:
    return GatewayUsageV2(
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        charged_tokens=150,
        reported_cost_micro_usd=1,
    )


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


__all__ = [
    "DeterministicAttachmentProvider",
    "audit",
    "build_gateway",
    "profile",
    "prompt",
    "ref",
]
