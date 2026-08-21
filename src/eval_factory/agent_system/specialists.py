from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from eval_factory.agent_system.private_store import FactoryPrivateObjectStore
from eval_factory.ai_gateway.protocols import AIGateway
from eval_factory.contracts.agent_system_v2 import (
    DatasetBuildPlanV2,
    EvaluationRequirementSpecV2,
    ExtractedUserPromptV2,
    InferredUserIntentV2,
    TaskRewriteCandidateV2,
    TaskRewritePlanV2,
)
from eval_factory.contracts.ai_gateway_v2 import (
    GatewayInvocationRequestV2,
    GatewayInvocationStatusV2,
    ModelRouteDecisionV2,
    ModelRouteRequestV2,
    PromptTemplateV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, Sha256
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.memory.models import (
    MemoryAccessContextV1,
    MemoryKindV1,
    MemoryRecallQueryV1,
    MemoryRetrievalModeV1,
    MemorySensitivityV1,
)
from eval_factory.memory.service import AgentMemoryService


class CoreSemanticOperationV1(StrEnum):
    INFER_INTENT = "INFER_INTENT"
    REWRITE_TASK = "REWRITE_TASK"


class CoreSemanticInputV1(ContractModelV2):
    schema_version: Literal["eval-factory/private-core-semantic-input/v1"] = (
        "eval-factory/private-core-semantic-input/v1"
    )
    operation: CoreSemanticOperationV1
    extracted_prompt_ref: ObjectRef
    prompt_text: str = Field(min_length=1, max_length=1_000_000)
    inferred_intent_ref: ObjectRef | None = None
    rewrite_plan_ref: ObjectRef | None = None

    @model_validator(mode="after")
    def validate_operation(self) -> Self:
        if self.operation is CoreSemanticOperationV1.INFER_INTENT:
            if self.inferred_intent_ref is not None or self.rewrite_plan_ref is not None:
                raise ValueError("intent input cannot bind rewrite authority")
        elif self.inferred_intent_ref is None or self.rewrite_plan_ref is None:
            raise ValueError("rewrite input requires intent and rewrite plan refs")
        return self


class RequirementPlanningInputV1(ContractModelV2):
    schema_version: Literal["eval-factory/private-requirement-planning-input/v1"] = (
        "eval-factory/private-requirement-planning-input/v1"
    )
    requirement_spec_ref: ObjectRef
    run_ref: ObjectRef
    goals: tuple[str, ...] = Field(min_length=1, max_length=128)
    constraints: tuple[str, ...] = Field(default=(), max_length=256)
    assumptions: tuple[str, ...] = Field(default=(), max_length=256)
    open_questions: tuple[str, ...] = Field(default=(), max_length=256)

    @model_validator(mode="after")
    def validate_authority(self) -> Self:
        if (
            self.requirement_spec_ref.object_type != "evaluation-requirement-spec"
            or self.requirement_spec_ref.object_version != "v2"
        ):
            raise ValueError("planning input requires a v2 requirement spec")
        if self.run_ref.object_type != "factory-run" or self.run_ref.object_version != "v2":
            raise ValueError("planning input requires a v2 Factory run")
        return self


class RequirementPlanningMemoryV2(ContractModelV2):
    schema_version: Literal["eval-factory/private-requirement-planning-memory/v2"] = (
        "eval-factory/private-requirement-planning-memory/v2"
    )
    memory_record_ref: ObjectRef
    admission_approval_ref: ObjectRef
    kind: MemoryKindV1
    content: str = Field(min_length=1, max_length=32_768)
    content_sha256: Sha256
    total_score_basis_points: int = Field(ge=0, le=10_000)

    @model_validator(mode="after")
    def validate_memory(self) -> Self:
        if (
            self.memory_record_ref.object_type != "agent-memory-record"
            or self.memory_record_ref.object_version != "private-v1"
        ):
            raise ValueError("planning memory requires a private-v1 Agent memory record")
        if self.admission_approval_ref.object_type != "memory-admission-approval":
            raise ValueError("planning memory requires explicit admission approval")
        if self.kind is MemoryKindV1.PROCEDURAL:
            raise ValueError("procedural memory cannot be injected into planning")
        if hashlib.sha256(self.content.encode("utf-8")).hexdigest() != self.content_sha256:
            raise ValueError("planning memory content hash is stale")
        return self


class RequirementPlanningInputV2(ContractModelV2):
    schema_version: Literal["eval-factory/private-requirement-planning-input/v2"] = (
        "eval-factory/private-requirement-planning-input/v2"
    )
    requirement_spec_ref: ObjectRef
    run_ref: ObjectRef
    goals: tuple[str, ...] = Field(min_length=1, max_length=128)
    constraints: tuple[str, ...] = Field(default=(), max_length=256)
    assumptions: tuple[str, ...] = Field(default=(), max_length=256)
    open_questions: tuple[str, ...] = Field(default=(), max_length=256)
    memory_context: tuple[RequirementPlanningMemoryV2, ...] = Field(
        default=(),
        max_length=16,
    )

    @model_validator(mode="after")
    def validate_authority(self) -> Self:
        if (
            self.requirement_spec_ref.object_type != "evaluation-requirement-spec"
            or self.requirement_spec_ref.object_version != "v2"
        ):
            raise ValueError("planning input requires a v2 requirement spec")
        if self.run_ref.object_type != "factory-run" or self.run_ref.object_version != "v2":
            raise ValueError("planning input requires a v2 Factory run")
        memory_ids = tuple(item.memory_record_ref.object_id for item in self.memory_context)
        if len(memory_ids) != len(set(memory_ids)):
            raise ValueError("planning memory refs must be unique")
        return self


@dataclass(frozen=True, slots=True)
class RequirementPlannerConfig:
    prompt: PromptTemplateV2
    agent_definition_ref: ObjectRef
    allowed_model_profile_refs: tuple[ObjectRef, ...]
    budget_reservation_ref: ObjectRef


@dataclass(frozen=True, slots=True)
class RequirementPlanningMemoryProvider:
    service: AgentMemoryService
    access: MemoryAccessContextV1
    limit: int = 8

    def __post_init__(self) -> None:
        if self.limit < 1 or self.limit > 16:
            raise ValueError("planning memory limit must be between 1 and 16")

    def recall(
        self,
        *,
        requirement: EvaluationRequirementSpecV2,
        audit: ContractAudit,
    ) -> tuple[RequirementPlanningMemoryV2, ...]:
        query_text = "\n".join(
            (
                *requirement.goals,
                *requirement.constraints,
                *requirement.assumptions,
                *requirement.open_questions,
            )
        )[:4_096]
        result = self.service.recall(
            MemoryRecallQueryV1(
                access=self.access,
                query_text=query_text,
                mode=MemoryRetrievalModeV1.LEXICAL,
                limit=self.limit,
                evaluated_at=audit.created_at,
            )
        )
        return tuple(
            RequirementPlanningMemoryV2(
                memory_record_ref=match.memory.record.to_ref(),
                admission_approval_ref=match.memory.record.approval_ref,
                kind=match.memory.record.kind,
                content=match.memory.content,
                content_sha256=match.memory.record.content_sha256,
                total_score_basis_points=match.total_score_basis_points,
            )
            for match in result.matches
            if match.memory.record.sensitivity is MemorySensitivityV1.RESTRICTED
            and match.memory.record.approval_ref is not None
            and match.memory.record.kind is not MemoryKindV1.PROCEDURAL
        )


@dataclass(frozen=True, slots=True)
class RequirementPlanResult:
    plan: DatasetBuildPlanV2
    route: ModelRouteDecisionV2
    invocation_result_ref: ObjectRef


@dataclass(frozen=True, slots=True)
class SemanticAgentConfig:
    intent_prompt: PromptTemplateV2
    rewrite_prompt: PromptTemplateV2
    intent_agent_definition_ref: ObjectRef
    rewrite_agent_definition_ref: ObjectRef
    allowed_model_profile_refs: tuple[ObjectRef, ...]
    intent_budget_reservation_ref: ObjectRef
    rewrite_budget_reservation_ref: ObjectRef


@dataclass(frozen=True, slots=True)
class IntentAgentResult:
    intent: InferredUserIntentV2
    route: ModelRouteDecisionV2
    invocation_result_ref: ObjectRef


@dataclass(frozen=True, slots=True)
class RewriteAgentResult:
    plan: TaskRewritePlanV2
    candidate: TaskRewriteCandidateV2
    route: ModelRouteDecisionV2
    invocation_result_ref: ObjectRef


class GatewayRequirementPlannerAgent:
    def __init__(
        self,
        *,
        gateway: AIGateway,
        private_store: FactoryPrivateObjectStore,
        config: RequirementPlannerConfig,
        memory_provider: RequirementPlanningMemoryProvider | None = None,
    ) -> None:
        self.gateway = gateway
        self.private_store = private_store
        self.config = config
        self.memory_provider = memory_provider

    async def propose(
        self,
        *,
        task_ref: ObjectRef,
        run_ref: ObjectRef,
        requirement: EvaluationRequirementSpecV2,
        audit: ContractAudit,
    ) -> RequirementPlanResult:
        if self.memory_provider is None:
            planning_input: RequirementPlanningInputV1 | RequirementPlanningInputV2 = (
                RequirementPlanningInputV1(
                    requirement_spec_ref=requirement.to_ref(),
                    run_ref=run_ref,
                    goals=requirement.goals,
                    constraints=requirement.constraints,
                    assumptions=requirement.assumptions,
                    open_questions=requirement.open_questions,
                )
            )
        else:
            planning_input = RequirementPlanningInputV2(
                requirement_spec_ref=requirement.to_ref(),
                run_ref=run_ref,
                goals=requirement.goals,
                constraints=requirement.constraints,
                assumptions=requirement.assumptions,
                open_questions=requirement.open_questions,
                memory_context=self.memory_provider.recall(
                    requirement=requirement,
                    audit=audit,
                ),
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
        invocation = GatewayIntentRewriteAgent._invocation_request(
            task_ref=task_ref,
            prompt=self.config.prompt,
            rendering_ref=rendering_ref,
            route=route,
            audit=audit,
        )
        result = await self.gateway.invoke(invocation, route=route)
        if result.status is not GatewayInvocationStatusV2.SUCCEEDED or result.output_ref is None:
            raise RuntimeError("requirement Planner Gateway invocation did not succeed")
        plan = self.private_store.get_model(result.output_ref, DatasetBuildPlanV2)
        if plan.run_ref != run_ref:
            raise RuntimeError("requirement Planner output does not bind Factory run")
        if (
            not set(requirement.goals).issubset(plan.goals)
            or not set(requirement.constraints).issubset(plan.user_constraints)
            or not set(requirement.assumptions).issubset(plan.assumptions)
            or not set(requirement.open_questions).issubset(plan.unresolved_questions)
        ):
            raise RuntimeError("requirement Planner output drops requirement authority")
        return RequirementPlanResult(
            plan=plan,
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
        suffix = _suffix(rendering_ref.object_id)
        return ModelRouteRequestV2.create(
            route_request_id=f"model-route-request://planning/{suffix}",
            agent_task_ref=task_ref,
            agent_definition_ref=self.config.agent_definition_ref,
            task_kind="planning",
            prompt_template_ref=self.config.prompt.to_ref(),
            required_capabilities=self.config.prompt.required_model_capabilities,
            data_classification=(
                "RESTRICTED_TRACE_DERIVED" if self.memory_provider is not None else "INTERNAL_DERIVED"
            ),
            residency="LOCAL",
            input_token_budget=16_000,
            output_token_budget=8_000,
            max_cost_micro_usd=2_000_000,
            minimum_quality_basis_points=8000,
            allowed_model_profile_refs=self.config.allowed_model_profile_refs,
            budget_reservation_ref=self.config.budget_reservation_ref,
            route_version=1,
            predecessor_route_ref=None,
            failed_receipt_ref=None,
            generator_model_profile_ref=None,
            audit=audit,
        )


class GatewayIntentRewriteAgent:
    def __init__(
        self,
        *,
        gateway: AIGateway,
        private_store: FactoryPrivateObjectStore,
        config: SemanticAgentConfig,
    ) -> None:
        self.gateway = gateway
        self.private_store = private_store
        self.config = config

    async def infer_intent(
        self,
        *,
        task_ref: ObjectRef,
        extracted_prompt: ExtractedUserPromptV2,
        prompt_text: str,
        audit: ContractAudit,
    ) -> IntentAgentResult:
        semantic_input = CoreSemanticInputV1(
            operation=CoreSemanticOperationV1.INFER_INTENT,
            extracted_prompt_ref=extracted_prompt.to_ref(),
            prompt_text=prompt_text,
        )
        rendering_ref = self.private_store.put_model(
            object_type="prompt-rendering",
            value=semantic_input,
        )
        route = self.gateway.route(
            self._route_request(
                task_ref=task_ref,
                task_kind="extraction",
                prompt=self.config.intent_prompt,
                agent_definition_ref=self.config.intent_agent_definition_ref,
                budget_reservation_ref=self.config.intent_budget_reservation_ref,
                rendering_ref=rendering_ref,
                audit=audit,
            )
        )
        invocation = self._invocation_request(
            task_ref=task_ref,
            prompt=self.config.intent_prompt,
            rendering_ref=rendering_ref,
            route=route,
            audit=audit,
        )
        result = await self.gateway.invoke(invocation, route=route)
        if result.status is not GatewayInvocationStatusV2.SUCCEEDED or result.output_ref is None:
            raise RuntimeError("intent Agent Gateway invocation did not succeed")
        intent = self.private_store.get_model(
            result.output_ref,
            InferredUserIntentV2,
        )
        if intent.extracted_prompt_ref != extracted_prompt.to_ref():
            raise RuntimeError("intent Agent output does not bind extracted prompt")
        return IntentAgentResult(
            intent=intent,
            route=route,
            invocation_result_ref=result.to_ref(),
        )

    async def rewrite(
        self,
        *,
        task_ref: ObjectRef,
        extracted_prompt: ExtractedUserPromptV2,
        intent: InferredUserIntentV2,
        prompt_text: str,
        audit: ContractAudit,
    ) -> RewriteAgentResult:
        plan = TaskRewritePlanV2.create(
            rewrite_plan_id=f"task-rewrite-plan://{_suffix(extracted_prompt.object_id)}",
            extracted_prompt_ref=extracted_prompt.to_ref(),
            inferred_intent_ref=intent.to_ref(),
            rewrite_policy_ref=_stable_ref("task-rewrite-policy", extracted_prompt.object_id),
            target_capabilities=("instruction-following",),
            fidelity_constraints=("preserve-source-grounded-user-request",),
            forbidden_transformations=("replace-original-prompt-with-inferred-intent",),
            acceptance_check_refs=(_stable_ref("acceptance-check", f"{extracted_prompt.object_id}:rewrite"),),
            audit=audit,
        )
        semantic_input = CoreSemanticInputV1(
            operation=CoreSemanticOperationV1.REWRITE_TASK,
            extracted_prompt_ref=extracted_prompt.to_ref(),
            prompt_text=prompt_text,
            inferred_intent_ref=intent.to_ref(),
            rewrite_plan_ref=plan.to_ref(),
        )
        rendering_ref = self.private_store.put_model(
            object_type="prompt-rendering",
            value=semantic_input,
        )
        route = self.gateway.route(
            self._route_request(
                task_ref=task_ref,
                task_kind="rewrite",
                prompt=self.config.rewrite_prompt,
                agent_definition_ref=self.config.rewrite_agent_definition_ref,
                budget_reservation_ref=self.config.rewrite_budget_reservation_ref,
                rendering_ref=rendering_ref,
                audit=audit,
            )
        )
        invocation = self._invocation_request(
            task_ref=task_ref,
            prompt=self.config.rewrite_prompt,
            rendering_ref=rendering_ref,
            route=route,
            audit=audit,
        )
        result = await self.gateway.invoke(invocation, route=route)
        if result.status is not GatewayInvocationStatusV2.SUCCEEDED or result.output_ref is None:
            raise RuntimeError("rewrite Agent Gateway invocation did not succeed")
        candidate = self.private_store.get_model(
            result.output_ref,
            TaskRewriteCandidateV2,
        )
        if (
            candidate.extracted_prompt_ref != extracted_prompt.to_ref()
            or candidate.inferred_intent_ref != intent.to_ref()
            or candidate.rewrite_plan_ref != plan.to_ref()
        ):
            raise RuntimeError("rewrite Agent output does not bind source authorities")
        return RewriteAgentResult(
            plan=plan,
            candidate=candidate,
            route=route,
            invocation_result_ref=result.to_ref(),
        )

    def _route_request(
        self,
        *,
        task_ref: ObjectRef,
        task_kind: str,
        prompt: PromptTemplateV2,
        agent_definition_ref: ObjectRef,
        budget_reservation_ref: ObjectRef,
        rendering_ref: ObjectRef,
        audit: ContractAudit,
    ) -> ModelRouteRequestV2:
        suffix = _suffix(rendering_ref.object_id)
        return ModelRouteRequestV2.create(
            route_request_id=f"model-route-request://{task_kind}/{suffix}",
            agent_task_ref=task_ref,
            agent_definition_ref=agent_definition_ref,
            task_kind=task_kind,
            prompt_template_ref=prompt.to_ref(),
            required_capabilities=prompt.required_model_capabilities,
            data_classification="RESTRICTED_TRACE_DERIVED",
            residency="LOCAL",
            input_token_budget=16_000,
            output_token_budget=4_000,
            max_cost_micro_usd=1_000_000,
            minimum_quality_basis_points=8000,
            allowed_model_profile_refs=self.config.allowed_model_profile_refs,
            budget_reservation_ref=budget_reservation_ref,
            route_version=1,
            predecessor_route_ref=None,
            failed_receipt_ref=None,
            generator_model_profile_ref=None,
            audit=audit,
        )

    @staticmethod
    def _invocation_request(
        *,
        task_ref: ObjectRef,
        prompt: PromptTemplateV2,
        rendering_ref: ObjectRef,
        route: ModelRouteDecisionV2,
        audit: ContractAudit,
    ) -> GatewayInvocationRequestV2:
        suffix = _suffix(rendering_ref.object_id)
        return GatewayInvocationRequestV2.create(
            invocation_request_id=f"gateway-invocation-request://{prompt.task_kind}/{suffix}",
            agent_task_ref=task_ref,
            route_decision_ref=route.to_ref(),
            prompt_template_ref=prompt.to_ref(),
            prompt_rendering_ref=rendering_ref,
            output_schema_ref=prompt.output_schema_ref,
            rag_result_refs=(),
            idempotency_key=f"gateway-invoke-{prompt.task_kind}-{suffix}",
            audit=audit,
        )


def _stable_ref(object_type: str, seed: str) -> ObjectRef:
    digest = hashlib.sha256(seed.encode()).hexdigest()
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://sha256/{digest}",
        object_version="v2",
        object_sha256=digest,
    )


def _suffix(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:32]


__all__ = [
    "CoreSemanticInputV1",
    "CoreSemanticOperationV1",
    "GatewayIntentRewriteAgent",
    "GatewayRequirementPlannerAgent",
    "IntentAgentResult",
    "RequirementPlanResult",
    "RequirementPlannerConfig",
    "RequirementPlanningInputV1",
    "RequirementPlanningInputV2",
    "RequirementPlanningMemoryProvider",
    "RequirementPlanningMemoryV2",
    "RewriteAgentResult",
    "SemanticAgentConfig",
]
