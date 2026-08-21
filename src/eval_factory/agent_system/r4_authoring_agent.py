from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from eval_factory.agent_system.private_store import (
    FactoryPrivateObjectStore,
)
from eval_factory.ai_gateway.protocols import AIGateway
from eval_factory.contracts.ai_gateway_v2 import (
    GatewayInvocationRequestV2,
    GatewayInvocationStatusV2,
    ModelRouteRequestV2,
    PromptTemplateV2,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ContractModel,
    ObjectRef,
)
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.task_authoring.draft_models import (
    TaskDraftAuthoringProposal,
    TaskDraftAuthoringRequest,
)
from eval_factory.task_authoring.models import (
    TaskEpisodeGroupingProposal,
    TaskEpisodeGroupingRequest,
)
from eval_factory.task_authoring.prompt_safety_models import (
    TaskPromptSafetyProposal,
    TaskPromptSafetyRequest,
)
from eval_factory.task_authoring.rubric_models import (
    RubricCandidateProposal,
    RubricCandidateRequest,
)


class R4AuthoringAgentError(RuntimeError):
    pass


class TaskEpisodeAgentInputV1(ContractModelV2):
    schema_version: Literal["eval-factory/private-task-episode-agent-input/v1"] = (
        "eval-factory/private-task-episode-agent-input/v1"
    )
    request: TaskEpisodeGroupingRequest


class TaskDraftAgentInputV1(ContractModelV2):
    schema_version: Literal["eval-factory/private-task-draft-agent-input/v1"] = (
        "eval-factory/private-task-draft-agent-input/v1"
    )
    request: TaskDraftAuthoringRequest
    rewritten_prompt: str
    task_intent: str
    evaluation_claim: str


class TaskPromptSafetyAgentInputV1(ContractModelV2):
    schema_version: Literal["eval-factory/private-task-prompt-safety-agent-input/v1"] = (
        "eval-factory/private-task-prompt-safety-agent-input/v1"
    )
    request: TaskPromptSafetyRequest


class RubricAgentInputV1(ContractModelV2):
    schema_version: Literal["eval-factory/private-rubric-agent-input/v1"] = (
        "eval-factory/private-rubric-agent-input/v1"
    )
    request: RubricCandidateRequest


@dataclass(frozen=True, slots=True)
class R4AuthoringAgentConfig:
    episode_prompt: PromptTemplateV2
    draft_prompt: PromptTemplateV2
    safety_prompt: PromptTemplateV2
    rubric_prompt: PromptTemplateV2
    episode_agent_definition_ref: ObjectRef
    draft_agent_definition_ref: ObjectRef
    safety_agent_definition_ref: ObjectRef
    rubric_agent_definition_ref: ObjectRef
    allowed_model_profile_refs: tuple[ObjectRef, ...]
    episode_budget_reservation_ref: ObjectRef
    draft_budget_reservation_ref: ObjectRef
    safety_budget_reservation_ref: ObjectRef
    rubric_budget_reservation_ref: ObjectRef


class GatewayR4AuthoringAgent:
    def __init__(
        self,
        *,
        gateway: AIGateway,
        private_store: FactoryPrivateObjectStore,
        config: R4AuthoringAgentConfig,
    ) -> None:
        self.gateway = gateway
        self.private_store = private_store
        self.config = config

    async def group_episode(
        self,
        *,
        task_ref: ObjectRef,
        request: TaskEpisodeGroupingRequest,
        audit: ContractAudit,
    ) -> TaskEpisodeGroupingProposal:
        return await self._invoke(
            task_ref=task_ref,
            prompt=self.config.episode_prompt,
            agent_definition_ref=(self.config.episode_agent_definition_ref),
            budget_reservation_ref=(self.config.episode_budget_reservation_ref),
            rendering=TaskEpisodeAgentInputV1(
                request=request,
            ),
            output_type=TaskEpisodeGroupingProposal,
            audit=audit,
        )

    async def author_draft(
        self,
        *,
        task_ref: ObjectRef,
        request: TaskDraftAuthoringRequest,
        rewritten_prompt: str,
        task_intent: str,
        evaluation_claim: str,
        audit: ContractAudit,
    ) -> TaskDraftAuthoringProposal:
        return await self._invoke(
            task_ref=task_ref,
            prompt=self.config.draft_prompt,
            agent_definition_ref=(self.config.draft_agent_definition_ref),
            budget_reservation_ref=(self.config.draft_budget_reservation_ref),
            rendering=TaskDraftAgentInputV1(
                request=request,
                rewritten_prompt=rewritten_prompt,
                task_intent=task_intent,
                evaluation_claim=evaluation_claim,
            ),
            output_type=TaskDraftAuthoringProposal,
            audit=audit,
        )

    async def review_prompt(
        self,
        *,
        task_ref: ObjectRef,
        request: TaskPromptSafetyRequest,
        audit: ContractAudit,
    ) -> TaskPromptSafetyProposal:
        return await self._invoke(
            task_ref=task_ref,
            prompt=self.config.safety_prompt,
            agent_definition_ref=(self.config.safety_agent_definition_ref),
            budget_reservation_ref=(self.config.safety_budget_reservation_ref),
            rendering=TaskPromptSafetyAgentInputV1(
                request=request,
            ),
            output_type=TaskPromptSafetyProposal,
            audit=audit,
        )

    async def author_rubric(
        self,
        *,
        task_ref: ObjectRef,
        request: RubricCandidateRequest,
        audit: ContractAudit,
    ) -> RubricCandidateProposal:
        return await self._invoke(
            task_ref=task_ref,
            prompt=self.config.rubric_prompt,
            agent_definition_ref=(self.config.rubric_agent_definition_ref),
            budget_reservation_ref=(self.config.rubric_budget_reservation_ref),
            rendering=RubricAgentInputV1(request=request),
            output_type=RubricCandidateProposal,
            audit=audit,
        )

    async def _invoke[OutputT: ContractModel](
        self,
        *,
        task_ref: ObjectRef,
        prompt: PromptTemplateV2,
        agent_definition_ref: ObjectRef,
        budget_reservation_ref: ObjectRef,
        rendering: ContractModelV2,
        output_type: type[OutputT],
        audit: ContractAudit,
    ) -> OutputT:
        rendering_ref = self.private_store.put_model(
            object_type="prompt-rendering",
            value=rendering,
        )
        suffix = hashlib.sha256(rendering_ref.object_id.encode()).hexdigest()[:32]
        route = self.gateway.route(
            ModelRouteRequestV2.create(
                route_request_id=(f"model-route-request://{prompt.task_kind}/{suffix}"),
                agent_task_ref=task_ref,
                agent_definition_ref=agent_definition_ref,
                task_kind=prompt.task_kind,
                prompt_template_ref=prompt.to_ref(),
                required_capabilities=(prompt.required_model_capabilities),
                data_classification=("RESTRICTED_TRACE_DERIVED"),
                residency="LOCAL",
                input_token_budget=32_000,
                output_token_budget=8_000,
                max_cost_micro_usd=2_000_000,
                minimum_quality_basis_points=8_000,
                allowed_model_profile_refs=tuple(
                    sorted(
                        set(self.config.allowed_model_profile_refs),
                        key=_ref_key,
                    )
                ),
                budget_reservation_ref=(budget_reservation_ref),
                route_version=1,
                predecessor_route_ref=None,
                failed_receipt_ref=None,
                generator_model_profile_ref=None,
                audit=audit,
            )
        )
        invocation = GatewayInvocationRequestV2.create(
            invocation_request_id=(f"gateway-invocation-request://{prompt.task_kind}/{suffix}"),
            agent_task_ref=task_ref,
            route_decision_ref=route.to_ref(),
            prompt_template_ref=prompt.to_ref(),
            prompt_rendering_ref=rendering_ref,
            output_schema_ref=prompt.output_schema_ref,
            rag_result_refs=(),
            idempotency_key=(f"gateway-invoke-{prompt.task_kind}-{suffix}"),
            audit=audit,
        )
        result = await self.gateway.invoke(
            invocation,
            route=route,
        )
        if result.status is not GatewayInvocationStatusV2.SUCCEEDED or result.output_ref is None:
            raise R4AuthoringAgentError(f"{prompt.task_kind} Gateway invocation failed")
        return self.private_store.get_model(
            result.output_ref,
            output_type,
        )


def _ref_key(
    value: ObjectRef,
) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


__all__ = [
    "GatewayR4AuthoringAgent",
    "R4AuthoringAgentConfig",
    "R4AuthoringAgentError",
    "RubricAgentInputV1",
    "TaskDraftAgentInputV1",
    "TaskEpisodeAgentInputV1",
    "TaskPromptSafetyAgentInputV1",
]
