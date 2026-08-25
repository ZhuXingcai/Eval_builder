from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from eval_factory.agent_system.private_store import (
    FactoryPrivateObjectStore,
)
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
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.harness.requirement_agent import (
    GatewayRequirementAgentConfig,
    GatewayRequirementAgentLoop,
)
from eval_factory.harness.runtime_models import (
    ProviderEvidenceClassV1,
    RequirementInterpretationProposalV1,
)
from eval_factory.harness.session_service import HarnessSessionService
from eval_factory.harness.session_store import HarnessSessionStore


class FixtureRequirementAgentConfigV1(ContractModelV2):
    schema_version: Literal["eval-harness/fixture-requirement-agent-config/v1"] = (
        "eval-harness/fixture-requirement-agent-config/v1"
    )
    claim_scope: Literal["DEVELOPMENT_FIXTURE_ONLY"] = "DEVELOPMENT_FIXTURE_ONLY"
    model_profile: ModelCapabilityProfileV2
    quality_baseline: ModelQualityBaselineV2
    health_snapshot: ModelHealthSnapshotV2
    price_schedule: ModelPriceScheduleV2
    routing_policy: ModelRoutingPolicyV2
    prompt: PromptTemplateV2
    agent_definition_ref: ObjectRef
    budget_reservation_ref: ObjectRef
    proposal: RequirementInterpretationProposalV1
    usage: GatewayUsageV2
    data_classification: str = Field(min_length=1, max_length=128)
    residency: str = Field(min_length=1, max_length=128)
    input_token_budget: int = Field(ge=1, le=10_000_000_000)
    output_token_budget: int = Field(ge=1, le=10_000_000_000)
    max_cost_micro_usd: int = Field(ge=0, le=10_000_000_000_000)
    minimum_quality_basis_points: int = Field(ge=0, le=10_000)
    assistant_chunk_characters: int = Field(default=512, ge=64, le=4_096)

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        profile_ref = self.model_profile.to_ref()
        if (
            self.prompt.task_kind != "harness-requirement-interpretation"
            or self.quality_baseline.task_kind != self.prompt.task_kind
            or self.quality_baseline.model_profile_ref != profile_ref
            or self.health_snapshot.model_profile_ref != profile_ref
            or self.price_schedule.model_profile_ref != profile_ref
            or self.agent_definition_ref not in self.routing_policy.allowed_agent_definition_refs
            or profile_ref not in self.routing_policy.allowed_model_profile_refs
            or self.model_profile.provider_id not in self.routing_policy.allowed_provider_ids
            or not set(self.prompt.required_model_capabilities).issubset(self.model_profile.capabilities)
        ):
            raise ValueError(
                "fixture requirement Agent authority is inconsistent",
            )
        if (
            self.budget_reservation_ref.object_type != "work-model-reservation"
            or self.budget_reservation_ref.object_version != "v2"
        ):
            raise ValueError(
                "fixture requirement Agent budget ref is invalid",
            )
        if (
            self.data_classification not in (self.model_profile.supported_data_classifications)
            or self.residency not in self.model_profile.supported_residencies
        ):
            raise ValueError(
                "fixture requirement Agent data scope is unsupported",
            )
        return self


class FixtureRequirementGatewayProvider:
    """Deterministic provider boundary for local Agent Shell development."""

    def __init__(
        self,
        *,
        private_store: FactoryPrivateObjectStore,
        config: FixtureRequirementAgentConfigV1,
    ) -> None:
        self.private_store = private_store
        self.config = config
        self.invocation_count = 0

    async def invoke(
        self,
        request: GatewayInvocationRequestV2,
        *,
        model_profile: ModelCapabilityProfileV2,
    ) -> ProviderInvocationResult:
        if (
            model_profile.to_ref() != self.config.model_profile.to_ref()
            or request.output_schema_ref != self.config.prompt.output_schema_ref
        ):
            raise ValueError(
                "fixture requirement invocation authority differs",
            )
        self.invocation_count += 1
        response_body_ref = self.private_store.put_bytes(
            object_type="model-response-content",
            payload=self.config.proposal.model_dump_json().encode(),
        )
        output_ref = self.private_store.put_model(
            object_type="requirement-proposal",
            value=self.config.proposal,
        )
        return ProviderInvocationResult(
            status=GatewayInvocationStatusV2.SUCCEEDED,
            response_body_ref=response_body_ref,
            output_ref=output_ref,
            usage=self.config.usage,
            failure_code=None,
        )


def build_fixture_requirement_session_service(
    *,
    harness_store_path: Path,
    private_store_path: Path,
    gateway_store_path: Path,
    config: FixtureRequirementAgentConfigV1,
) -> HarnessSessionService:
    private_store = FactoryPrivateObjectStore(
        private_store_path,
    )
    catalog = ModelCatalog(
        profiles=(config.model_profile,),
        quality_baselines=(config.quality_baseline,),
        health_snapshots=(config.health_snapshot,),
        price_schedules=(config.price_schedule,),
    )
    records = GatewayRecordStore(
        gateway_store_path,
    )
    gateway = EmbeddedAIGateway(
        router=ModelRouter(
            catalog,
            config.routing_policy,
        ),
        catalog=catalog,
        prompts=PromptRegistry((config.prompt,)),
        records=records,
        provider=FixtureRequirementGatewayProvider(
            private_store=private_store,
            config=config,
        ),
    )
    session_store = HarnessSessionStore(
        harness_store_path,
    )
    return HarnessSessionService(
        store=session_store,
        requirement_agent=GatewayRequirementAgentLoop(
            gateway=gateway,
            lookup=records,
            private_store=private_store,
            session_store=session_store,
            config=GatewayRequirementAgentConfig(
                prompt=config.prompt,
                agent_definition_ref=config.agent_definition_ref,
                allowed_model_profile_refs=(config.model_profile.to_ref(),),
                budget_reservation_ref=(config.budget_reservation_ref),
                data_classification=config.data_classification,
                residency=config.residency,
                input_token_budget=config.input_token_budget,
                output_token_budget=config.output_token_budget,
                max_cost_micro_usd=config.max_cost_micro_usd,
                minimum_quality_basis_points=(config.minimum_quality_basis_points),
                evidence_class=(ProviderEvidenceClassV1.MECHANISM_FIXTURE),
                assistant_chunk_characters=(config.assistant_chunk_characters),
            ),
        ),
    )


__all__ = [
    "FixtureRequirementAgentConfigV1",
    "FixtureRequirementGatewayProvider",
    "build_fixture_requirement_session_service",
]
