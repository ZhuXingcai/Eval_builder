from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, Self

from pydantic import Field, model_validator

from eval_factory.agent_system.attachment_planning import (
    AttachmentGenerationPlanCompiler,
)
from eval_factory.agent_system.private_store import (
    FactoryPrivateObjectStore,
)
from eval_factory.ai_gateway.protocols import AIGateway
from eval_factory.contracts.agent_system_v2 import (
    AttachmentGenerationPlanV2,
    AttachmentMockWorkV2,
    CompiledAttachmentGenerationPlanV2,
    FactoryRunPolicyV2,
)
from eval_factory.contracts.ai_gateway_v2 import (
    GatewayInvocationRequestV2,
    GatewayInvocationStatusV2,
    ModelRouteDecisionV2,
    ModelRouteRequestV2,
    PromptTemplateV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2


class AttachmentPlanningAgentError(RuntimeError):
    pass


class AttachmentPlanningInputV1(ContractModelV2):
    schema_version: Literal["eval-factory/private-attachment-planning-input/v1"] = (
        "eval-factory/private-attachment-planning-input/v1"
    )
    run_ref: ObjectRef
    producer_task_view_ref: ObjectRef
    evidence_bundle_ref: ObjectRef
    attachment_planning_context_ref: ObjectRef
    work_templates: tuple[AttachmentMockWorkV2, ...] = Field(
        min_length=1,
        max_length=100_000,
    )
    quality_policy_ref: ObjectRef
    solvability_policy_ref: ObjectRef

    @model_validator(mode="after")
    def validate_input(self) -> Self:
        expected = (
            (self.run_ref, "factory-run", "v2"),
            (
                self.producer_task_view_ref,
                "producer-task-view",
                "v2",
            ),
            (self.evidence_bundle_ref, "evidence-bundle", "v1"),
            (
                self.attachment_planning_context_ref,
                "attachment-planning-context",
                "v2",
            ),
        )
        if any(
            ref.object_type != object_type or ref.object_version != version
            for ref, object_type, version in expected
        ):
            raise ValueError("attachment planning input has invalid source refs")
        return self


@dataclass(frozen=True, slots=True)
class AttachmentPlanningAgentConfig:
    prompt: PromptTemplateV2
    agent_definition_ref: ObjectRef
    allowed_model_profile_refs: tuple[ObjectRef, ...]
    budget_reservation_ref: ObjectRef


@dataclass(frozen=True, slots=True)
class AttachmentPlanningAgentResult:
    plan: AttachmentGenerationPlanV2
    compiled_plan: CompiledAttachmentGenerationPlanV2
    route: ModelRouteDecisionV2
    invocation_result_ref: ObjectRef


class GatewayAttachmentPlanningAgent:
    def __init__(
        self,
        *,
        gateway: AIGateway,
        private_store: FactoryPrivateObjectStore,
        compiler: AttachmentGenerationPlanCompiler,
        config: AttachmentPlanningAgentConfig,
    ) -> None:
        self.gateway = gateway
        self.private_store = private_store
        self.compiler = compiler
        self.config = config

    async def propose(
        self,
        *,
        task_ref: ObjectRef,
        run_ref: ObjectRef,
        producer_task_view_ref: ObjectRef,
        evidence_bundle_ref: ObjectRef,
        attachment_planning_context_ref: ObjectRef,
        work_templates: tuple[AttachmentMockWorkV2, ...],
        quality_policy_ref: ObjectRef,
        solvability_policy_ref: ObjectRef,
        policy: FactoryRunPolicyV2,
        audit: ContractAudit,
    ) -> AttachmentPlanningAgentResult:
        planning_input = AttachmentPlanningInputV1(
            run_ref=run_ref,
            producer_task_view_ref=producer_task_view_ref,
            evidence_bundle_ref=evidence_bundle_ref,
            attachment_planning_context_ref=(attachment_planning_context_ref),
            work_templates=tuple(
                sorted(
                    work_templates,
                    key=lambda value: value.work_key,
                )
            ),
            quality_policy_ref=quality_policy_ref,
            solvability_policy_ref=solvability_policy_ref,
        )
        rendering_ref = self.private_store.put_model(
            object_type="prompt-rendering",
            value=planning_input,
        )
        route = self.gateway.route(
            self._route_request(
                task_ref=task_ref,
                rendering_ref=rendering_ref,
                audit=audit,
            )
        )
        invocation = GatewayInvocationRequestV2.create(
            invocation_request_id=(
                f"gateway-invocation-request://attachment-planning/{_suffix(rendering_ref)}"
            ),
            agent_task_ref=task_ref,
            route_decision_ref=route.to_ref(),
            prompt_template_ref=self.config.prompt.to_ref(),
            prompt_rendering_ref=rendering_ref,
            output_schema_ref=self.config.prompt.output_schema_ref,
            rag_result_refs=(),
            idempotency_key=(f"gateway-invoke-attachment-planning-{_suffix(rendering_ref)}"),
            audit=audit,
        )
        result = await self.gateway.invoke(
            invocation,
            route=route,
        )
        if result.status is not GatewayInvocationStatusV2.SUCCEEDED or result.output_ref is None:
            raise AttachmentPlanningAgentError("attachment Planner Gateway invocation did not succeed")
        plan = self.private_store.get_model(
            result.output_ref,
            AttachmentGenerationPlanV2,
        )
        self._validate_output(planning_input, plan)
        compiled = self.compiler.compile(
            plan=plan,
            policy=policy,
            audit=audit,
        )
        return AttachmentPlanningAgentResult(
            plan=plan,
            compiled_plan=compiled,
            route=route,
            invocation_result_ref=result.to_ref(),
        )

    def _route_request(
        self,
        *,
        task_ref: ObjectRef,
        rendering_ref: ObjectRef,
        audit: ContractAudit,
    ) -> ModelRouteRequestV2:
        return ModelRouteRequestV2.create(
            route_request_id=(f"model-route-request://attachment-planning/{_suffix(rendering_ref)}"),
            agent_task_ref=task_ref,
            agent_definition_ref=(self.config.agent_definition_ref),
            task_kind="attachment-planning",
            prompt_template_ref=self.config.prompt.to_ref(),
            required_capabilities=(self.config.prompt.required_model_capabilities),
            data_classification="RESTRICTED_TRACE_DERIVED",
            residency="LOCAL",
            input_token_budget=16_000,
            output_token_budget=8_000,
            max_cost_micro_usd=1_000_000,
            minimum_quality_basis_points=8000,
            allowed_model_profile_refs=(self.config.allowed_model_profile_refs),
            budget_reservation_ref=(self.config.budget_reservation_ref),
            route_version=1,
            predecessor_route_ref=None,
            failed_receipt_ref=None,
            generator_model_profile_ref=None,
            audit=audit,
        )

    @staticmethod
    def _validate_output(
        planning_input: AttachmentPlanningInputV1,
        plan: AttachmentGenerationPlanV2,
    ) -> None:
        if (
            plan.run_ref != planning_input.run_ref
            or plan.producer_task_view_ref != planning_input.producer_task_view_ref
            or plan.evidence_bundle_ref != planning_input.evidence_bundle_ref
            or plan.attachment_planning_context_ref != planning_input.attachment_planning_context_ref
            or plan.quality_policy_ref != planning_input.quality_policy_ref
            or plan.solvability_policy_ref != planning_input.solvability_policy_ref
        ):
            raise AttachmentPlanningAgentError("attachment Planner output widens source authority")
        expected = {work.artifact_group_ref: set(work.artifact_ids) for work in planning_input.work_templates}
        observed = {work.artifact_group_ref: set(work.artifact_ids) for work in plan.works}
        if observed != expected:
            raise AttachmentPlanningAgentError("attachment Planner output changes artifact ownership")


def _suffix(value: ObjectRef) -> str:
    return hashlib.sha256(value.object_id.encode()).hexdigest()[:32]


__all__ = [
    "AttachmentPlanningAgentConfig",
    "AttachmentPlanningAgentError",
    "AttachmentPlanningAgentResult",
    "AttachmentPlanningInputV1",
    "GatewayAttachmentPlanningAgent",
]
