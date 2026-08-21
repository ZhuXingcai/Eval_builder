from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from eval_factory.agent_system.grading_design import (
    GradingDesignPolicyError,
    JudgeDesignCompilation,
    JudgeDesignCompiler,
    JudgeDesignProposalOutcomeV1,
    JudgeDesignProposalV1,
    JudgeDesignValidator,
)
from eval_factory.agent_system.grading_material_store import (
    GradingDesignMaterialStore,
)
from eval_factory.agent_system.grading_planning import (
    GradingDesignPlanCompiler,
)
from eval_factory.agent_system.private_store import (
    FactoryPrivateObjectError,
    FactoryPrivateObjectStore,
)
from eval_factory.ai_gateway.protocols import AIGateway
from eval_factory.ai_gateway.routing import (
    ModelRouteBlockedError,
)
from eval_factory.contracts.agent_system_v2 import (
    CompiledGradingDesignPlanV2,
    CriteriaRubricResultV2,
    FactoryRunPolicyV2,
    GradingDesignOutcomeV2,
    GradingDesignPlanV2,
    GradingDesignResultV2,
    JudgeDesignSpecV2,
    JudgeDesignValidationOutcomeV2,
    JudgeDesignValidationV2,
)
from eval_factory.contracts.ai_gateway_v2 import (
    GatewayInvocationRequestV2,
    GatewayInvocationStatusV2,
    ModelRouteDecisionV2,
    ModelRouteRejectionCodeV2,
    ModelRouteRequestV2,
    PromptTemplateV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.task_v2 import (
    EvaluatorReferenceGrantV2,
    EvaluatorSpecV2,
    ReferencePolicyV2,
    RubricSetV2,
    ToolPolicyV2,
    evaluator_reference_grant_ref,
)
from eval_factory.task_authoring import (
    VerifiedModelDomainAuthorization,
)


class GradingDesignAgentError(RuntimeError):
    pass


class GradingDesignPlanningInputV1(ContractModelV2):
    schema_version: Literal["eval-factory/private-grading-design-planning-input/v1"] = (
        "eval-factory/private-grading-design-planning-input/v1"
    )
    plan_template: GradingDesignPlanV2


class GradingDesignAgentInputV1(ContractModelV2):
    schema_version: Literal["eval-factory/private-grading-design-agent-input/v1"] = (
        "eval-factory/private-grading-design-agent-input/v1"
    )
    plan_ref: ObjectRef
    criteria_result_ref: ObjectRef
    rubric_set: RubricSetV2
    evaluator_spec: EvaluatorSpecV2
    reference_policy: ReferencePolicyV2
    tool_policy: ToolPolicyV2


@dataclass(frozen=True, slots=True)
class GradingDesignPlanningAgentConfig:
    prompt: PromptTemplateV2
    agent_definition_ref: ObjectRef
    allowed_model_profile_refs: tuple[ObjectRef, ...]
    budget_reservation_ref: ObjectRef


@dataclass(frozen=True, slots=True)
class GradingDesignPlanningAgentResult:
    plan: GradingDesignPlanV2
    compiled_plan: CompiledGradingDesignPlanV2
    route: ModelRouteDecisionV2
    invocation_result_ref: ObjectRef


@dataclass(frozen=True, slots=True)
class GradingDesignAgentConfig:
    prompt: PromptTemplateV2
    agent_definition_ref: ObjectRef
    allowed_model_profile_refs: tuple[ObjectRef, ...]
    budget_reservation_ref: ObjectRef


@dataclass(frozen=True, slots=True)
class GradingDesignExecution:
    result: GradingDesignResultV2
    route: ModelRouteDecisionV2 | None
    invocation_result_ref: ObjectRef | None
    design_spec: JudgeDesignSpecV2 | None
    validation: JudgeDesignValidationV2
    reference_grants: tuple[EvaluatorReferenceGrantV2, ...]


class GatewayGradingDesignPlanningAgent:
    def __init__(
        self,
        *,
        gateway: AIGateway,
        private_store: FactoryPrivateObjectStore,
        compiler: GradingDesignPlanCompiler,
        config: GradingDesignPlanningAgentConfig,
    ) -> None:
        self.gateway = gateway
        self.private_store = private_store
        self.compiler = compiler
        self.config = config

    async def propose(
        self,
        *,
        task_ref: ObjectRef,
        plan_template: GradingDesignPlanV2,
        policy: FactoryRunPolicyV2,
        audit: ContractAudit,
    ) -> GradingDesignPlanningAgentResult:
        rendering_ref = self.private_store.put_model(
            object_type="prompt-rendering",
            value=GradingDesignPlanningInputV1(plan_template=plan_template),
        )
        route = self.gateway.route(
            _route_request(
                task_ref=task_ref,
                rendering_ref=rendering_ref,
                prompt=self.config.prompt,
                agent_definition_ref=(self.config.agent_definition_ref),
                allowed_model_profile_refs=(self.config.allowed_model_profile_refs),
                budget_reservation_ref=(self.config.budget_reservation_ref),
                task_kind="grading-design-planning",
                generator_model_profile_ref=(plan_template.generator_model_profile_ref),
                max_model_tokens=(plan_template.max_model_tokens),
                max_cost_micro_usd=(plan_template.max_cost_micro_usd),
                audit=audit,
            )
        )
        invocation = _invocation_request(
            task_ref=task_ref,
            route=route,
            prompt=self.config.prompt,
            rendering_ref=rendering_ref,
            task_kind="grading-design-planning",
            audit=audit,
        )
        result = await self.gateway.invoke(
            invocation,
            route=route,
        )
        if result.status is not GatewayInvocationStatusV2.SUCCEEDED or result.output_ref is None:
            raise GradingDesignAgentError("grading planning invocation did not succeed")
        try:
            plan = self.private_store.get_model(
                result.output_ref,
                GradingDesignPlanV2,
            )
        except FactoryPrivateObjectError as exc:
            raise GradingDesignAgentError("grading planning output is malformed") from exc
        self._validate_output(plan_template, plan)
        return GradingDesignPlanningAgentResult(
            plan=plan,
            compiled_plan=self.compiler.compile(
                plan=plan,
                policy=policy,
                audit=audit,
            ),
            route=route,
            invocation_result_ref=result.to_ref(),
        )

    @staticmethod
    def _validate_output(
        template: GradingDesignPlanV2,
        plan: GradingDesignPlanV2,
    ) -> None:
        fixed_fields = (
            "run_ref",
            "plan_version",
            "predecessor_plan_ref",
            "criteria_rubric_result_ref",
            "rubric_set_ref",
            "evaluator_spec_ref",
            "reference_policy_ref",
            "tool_policy_ref",
            "generator_model_profile_ref",
            "judge_input_schema_ref",
            "judge_output_schema_ref",
            "required_output_fields",
            "allowed_judge_model_profile_refs",
            "judge_prompt_template_ref",
            "agent_role",
            "required_capability_ids",
            "specialist_tool_ids",
            "data_purpose",
            "data_classifications",
            "model_policy_ref",
            "acceptance_check_refs",
            "max_attempts",
            "max_model_requests",
            "max_model_tokens",
            "max_cost_micro_usd",
        )
        if any(getattr(template, field_name) != getattr(plan, field_name) for field_name in fixed_fields):
            raise GradingDesignAgentError("grading planning output widens fixed authority")


class GatewayGradingDesignAgent:
    def __init__(
        self,
        *,
        gateway: AIGateway,
        private_store: FactoryPrivateObjectStore,
        material_store: GradingDesignMaterialStore,
        config: GradingDesignAgentConfig,
    ) -> None:
        self.gateway = gateway
        self.private_store = private_store
        self.material_store = material_store
        self.config = config

    async def author(
        self,
        *,
        task_ref: ObjectRef,
        plan: GradingDesignPlanV2,
        criteria_result: CriteriaRubricResultV2,
        criteria_route: ModelRouteDecisionV2,
        rubric_set: RubricSetV2,
        evaluator_spec: EvaluatorSpecV2,
        reference_policy: ReferencePolicyV2,
        tool_policy: ToolPolicyV2,
        model_authorizations: tuple[VerifiedModelDomainAuthorization, ...],
        evaluated_at: datetime,
        audit: ContractAudit,
    ) -> GradingDesignExecution:
        self.validate_sources(
            plan=plan,
            criteria_result=criteria_result,
            criteria_route=criteria_route,
            rubric_set=rubric_set,
            evaluator_spec=evaluator_spec,
            reference_policy=reference_policy,
            tool_policy=tool_policy,
        )
        rendering_ref = self.private_store.put_model(
            object_type="prompt-rendering",
            value=GradingDesignAgentInputV1(
                plan_ref=plan.to_ref(),
                criteria_result_ref=criteria_result.to_ref(),
                rubric_set=rubric_set,
                evaluator_spec=evaluator_spec,
                reference_policy=reference_policy,
                tool_policy=tool_policy,
            ),
        )
        return await self._author_after_preflight(
            task_ref=task_ref,
            plan=plan,
            criteria_result=criteria_result,
            criteria_route=criteria_route,
            rubric_set=rubric_set,
            evaluator_spec=evaluator_spec,
            reference_policy=reference_policy,
            tool_policy=tool_policy,
            model_authorizations=model_authorizations,
            evaluated_at=evaluated_at,
            audit=audit,
            rendering_ref=rendering_ref,
        )

    def validate_sources(
        self,
        *,
        plan: GradingDesignPlanV2,
        criteria_result: CriteriaRubricResultV2,
        criteria_route: ModelRouteDecisionV2,
        rubric_set: RubricSetV2,
        evaluator_spec: EvaluatorSpecV2,
        reference_policy: ReferencePolicyV2,
        tool_policy: ToolPolicyV2,
    ) -> None:
        if self.config.prompt.to_ref() != plan.judge_prompt_template_ref or not set(
            plan.allowed_judge_model_profile_refs
        ).issubset(self.config.allowed_model_profile_refs):
            raise GradingDesignAgentError("grading design Agent configuration is stale")
        try:
            JudgeDesignCompiler.validate_sources(
                plan=plan,
                criteria_result=criteria_result,
                criteria_route=criteria_route,
                rubric_set=rubric_set,
                evaluator_spec=evaluator_spec,
                reference_policy=reference_policy,
                tool_policy=tool_policy,
            )
        except GradingDesignPolicyError as exc:
            raise GradingDesignAgentError("grading design source authority is stale") from exc

    async def _author_after_preflight(
        self,
        *,
        task_ref: ObjectRef,
        plan: GradingDesignPlanV2,
        criteria_result: CriteriaRubricResultV2,
        criteria_route: ModelRouteDecisionV2,
        rubric_set: RubricSetV2,
        evaluator_spec: EvaluatorSpecV2,
        reference_policy: ReferencePolicyV2,
        tool_policy: ToolPolicyV2,
        model_authorizations: tuple[VerifiedModelDomainAuthorization, ...],
        evaluated_at: datetime,
        audit: ContractAudit,
        rendering_ref: ObjectRef,
    ) -> GradingDesignExecution:
        try:
            route = self.gateway.route(
                _route_request(
                    task_ref=task_ref,
                    rendering_ref=rendering_ref,
                    prompt=self.config.prompt,
                    agent_definition_ref=(self.config.agent_definition_ref),
                    allowed_model_profile_refs=(plan.allowed_judge_model_profile_refs),
                    budget_reservation_ref=(self.config.budget_reservation_ref),
                    task_kind="grading-design",
                    generator_model_profile_ref=(plan.generator_model_profile_ref),
                    max_model_tokens=plan.max_model_tokens,
                    max_cost_micro_usd=(plan.max_cost_micro_usd),
                    audit=audit,
                )
            )
        except ModelRouteBlockedError as exc:
            reasons = {code for candidate in exc.candidates for code in candidate.rejection_codes}
            reason = (
                "GENERATOR_JUDGE_COLLISION"
                if reasons == {ModelRouteRejectionCodeV2.GENERATOR_JUDGE_COLLISION}
                else "NO_ELIGIBLE_JUDGE_MODEL"
            )
            capability_reasons = {
                ModelRouteRejectionCodeV2.CAPABILITY_MISMATCH,
                ModelRouteRejectionCodeV2.CONTEXT_LIMIT,
                ModelRouteRejectionCodeV2.OUTPUT_LIMIT,
                ModelRouteRejectionCodeV2.PROVIDER_UNAVAILABLE,
                ModelRouteRejectionCodeV2.QUALITY_BASELINE_MISSING,
                ModelRouteRejectionCodeV2.QUALITY_BELOW_MINIMUM,
                ModelRouteRejectionCodeV2.HEALTH_SNAPSHOT_MISSING,
                ModelRouteRejectionCodeV2.PRICE_SCHEDULE_MISSING,
            }
            blocked_capability = bool(reasons) and reasons.issubset(capability_reasons)
            return self._non_success(
                plan=plan,
                criteria_result=criteria_result,
                route=None,
                invocation_result_ref=None,
                gateway_receipt_ref=None,
                proposal_ref=None,
                validation_outcome=(
                    JudgeDesignValidationOutcomeV2.BLOCKED_CAPABILITY
                    if blocked_capability
                    else JudgeDesignValidationOutcomeV2.BLOCKED_POLICY
                ),
                result_outcome=(
                    GradingDesignOutcomeV2.BLOCKED_CAPABILITY
                    if blocked_capability
                    else GradingDesignOutcomeV2.BLOCKED_POLICY
                ),
                reason_codes=(reason,),
                audit=audit,
            )
        invocation = _invocation_request(
            task_ref=task_ref,
            route=route,
            prompt=self.config.prompt,
            rendering_ref=rendering_ref,
            task_kind="grading-design",
            audit=audit,
        )
        gateway_result = await self.gateway.invoke(
            invocation,
            route=route,
        )
        if (
            gateway_result.status is not GatewayInvocationStatusV2.SUCCEEDED
            or gateway_result.output_ref is None
        ):
            return self._non_success(
                plan=plan,
                criteria_result=criteria_result,
                route=route,
                invocation_result_ref=gateway_result.to_ref(),
                gateway_receipt_ref=gateway_result.receipt_ref,
                proposal_ref=None,
                validation_outcome=(JudgeDesignValidationOutcomeV2.BLOCKED_CAPABILITY),
                result_outcome=(GradingDesignOutcomeV2.BLOCKED_CAPABILITY),
                reason_codes=(gateway_result.failure_code or "MODEL_PROVIDER_FAILED",),
                audit=audit,
            )
        try:
            proposal = self.private_store.get_model(
                gateway_result.output_ref,
                JudgeDesignProposalV1,
            )
        except FactoryPrivateObjectError as exc:
            raise GradingDesignAgentError("grading provider output is malformed") from exc
        if proposal.outcome is not (JudgeDesignProposalOutcomeV1.PROPOSED):
            validation_outcome = (
                JudgeDesignValidationOutcomeV2.ABSTAIN_UNJUDGEABLE
                if proposal.outcome is JudgeDesignProposalOutcomeV1.ABSTAIN
                else JudgeDesignValidationOutcomeV2.ESCALATE_REFERENCE_REQUIRED
            )
            result_outcome = (
                GradingDesignOutcomeV2.ABSTAINED
                if proposal.outcome is JudgeDesignProposalOutcomeV1.ABSTAIN
                else GradingDesignOutcomeV2.ESCALATED
            )
            return self._non_success(
                plan=plan,
                criteria_result=criteria_result,
                route=route,
                invocation_result_ref=gateway_result.to_ref(),
                gateway_receipt_ref=gateway_result.receipt_ref,
                proposal_ref=gateway_result.output_ref,
                validation_outcome=validation_outcome,
                result_outcome=result_outcome,
                reason_codes=proposal.reason_codes,
                audit=audit,
            )
        try:
            candidate = JudgeDesignCompiler().compile(
                plan=plan,
                criteria_result=criteria_result,
                criteria_route=criteria_route,
                rubric_set=rubric_set,
                evaluator_spec=evaluator_spec,
                reference_policy=reference_policy,
                tool_policy=tool_policy,
                proposal=proposal,
                prompt_rendering_ref=rendering_ref,
                route=route,
                gateway_result=gateway_result,
            )
            compilation = JudgeDesignValidator().validate(
                candidate=candidate,
                model_authorizations=model_authorizations,
                evaluated_at=evaluated_at,
                audit=audit,
            )
        except GradingDesignPolicyError as exc:
            raise GradingDesignAgentError("grading design failed deterministic validation") from exc
        self._persist_compilation(compilation)
        result = GradingDesignResultV2.create(
            result_id=(f"grading-design-result://{plan.object_sha256}"),
            plan_ref=plan.to_ref(),
            criteria_rubric_result_ref=criteria_result.to_ref(),
            route_decision_ref=route.to_ref(),
            gateway_receipt_ref=gateway_result.receipt_ref,
            gateway_invocation_result_ref=(gateway_result.to_ref()),
            proposal_ref=gateway_result.output_ref,
            judge_design_spec_ref=(
                compilation.design_spec.to_ref() if compilation.design_spec is not None else None
            ),
            validation_ref=compilation.validation.to_ref(),
            outcome=compilation.outcome,
            reason_codes=compilation.reason_codes,
            audit=audit,
        )
        return GradingDesignExecution(
            result=result,
            route=route,
            invocation_result_ref=gateway_result.to_ref(),
            design_spec=compilation.design_spec,
            validation=compilation.validation,
            reference_grants=compilation.reference_grants,
        )

    def _persist_compilation(
        self,
        compilation: JudgeDesignCompilation,
    ) -> None:
        for value in (
            *compilation.reference_grants,
            compilation.validation,
            *((compilation.design_spec,) if compilation.design_spec is not None else ()),
        ):
            reference = (
                evaluator_reference_grant_ref(value)
                if isinstance(value, EvaluatorReferenceGrantV2)
                else value.to_ref()
            )
            self.material_store.put_model(
                behavior_ref=reference,
                value=value,
            )

    def _non_success(
        self,
        *,
        plan: GradingDesignPlanV2,
        criteria_result: CriteriaRubricResultV2,
        route: ModelRouteDecisionV2 | None,
        invocation_result_ref: ObjectRef | None,
        gateway_receipt_ref: ObjectRef | None,
        proposal_ref: ObjectRef | None,
        validation_outcome: JudgeDesignValidationOutcomeV2,
        result_outcome: GradingDesignOutcomeV2,
        reason_codes: tuple[str, ...],
        audit: ContractAudit,
    ) -> GradingDesignExecution:
        validation = JudgeDesignValidationV2.create(
            validation_id=(f"judge-design-validation://{plan.object_sha256}/{validation_outcome.value}"),
            plan_ref=plan.to_ref(),
            design_spec_ref=None,
            criterion_ids=(),
            validated_output_fields=(),
            reference_access_result_refs=(),
            outcome=validation_outcome,
            reason_codes=reason_codes,
            audit=audit,
        )
        self.material_store.put_model(
            behavior_ref=validation.to_ref(),
            value=validation,
        )
        result = GradingDesignResultV2.create(
            result_id=(f"grading-design-result://{plan.object_sha256}"),
            plan_ref=plan.to_ref(),
            criteria_rubric_result_ref=criteria_result.to_ref(),
            route_decision_ref=(route.to_ref() if route is not None else None),
            gateway_receipt_ref=(gateway_receipt_ref),
            gateway_invocation_result_ref=(invocation_result_ref),
            proposal_ref=proposal_ref,
            judge_design_spec_ref=None,
            validation_ref=validation.to_ref(),
            outcome=result_outcome,
            reason_codes=reason_codes,
            audit=audit,
        )
        return GradingDesignExecution(
            result=result,
            route=route,
            invocation_result_ref=invocation_result_ref,
            design_spec=None,
            validation=validation,
            reference_grants=(),
        )


def _route_request(
    *,
    task_ref: ObjectRef,
    rendering_ref: ObjectRef,
    prompt: PromptTemplateV2,
    agent_definition_ref: ObjectRef,
    allowed_model_profile_refs: tuple[ObjectRef, ...],
    budget_reservation_ref: ObjectRef,
    task_kind: str,
    generator_model_profile_ref: ObjectRef,
    max_model_tokens: int,
    max_cost_micro_usd: int,
    audit: ContractAudit,
) -> ModelRouteRequestV2:
    return ModelRouteRequestV2.create(
        route_request_id=(f"model-route-request://{task_kind}/{_suffix(rendering_ref)}"),
        agent_task_ref=task_ref,
        agent_definition_ref=agent_definition_ref,
        task_kind=task_kind,
        prompt_template_ref=prompt.to_ref(),
        required_capabilities=(prompt.required_model_capabilities),
        data_classification="RESTRICTED_EVALUATOR_CONTROL",
        residency="LOCAL",
        input_token_budget=max_model_tokens,
        output_token_budget=min(8000, max_model_tokens),
        max_cost_micro_usd=max_cost_micro_usd,
        minimum_quality_basis_points=8500,
        allowed_model_profile_refs=(allowed_model_profile_refs),
        budget_reservation_ref=budget_reservation_ref,
        route_version=1,
        predecessor_route_ref=None,
        failed_receipt_ref=None,
        generator_model_profile_ref=generator_model_profile_ref,
        audit=audit,
    )


def _invocation_request(
    *,
    task_ref: ObjectRef,
    route: ModelRouteDecisionV2,
    prompt: PromptTemplateV2,
    rendering_ref: ObjectRef,
    task_kind: str,
    audit: ContractAudit,
) -> GatewayInvocationRequestV2:
    return GatewayInvocationRequestV2.create(
        invocation_request_id=(f"gateway-invocation-request://{task_kind}/{_suffix(rendering_ref)}"),
        agent_task_ref=task_ref,
        route_decision_ref=route.to_ref(),
        prompt_template_ref=prompt.to_ref(),
        prompt_rendering_ref=rendering_ref,
        output_schema_ref=prompt.output_schema_ref,
        rag_result_refs=(),
        idempotency_key=(f"gateway-invoke-{task_kind}-{_suffix(rendering_ref)}"),
        audit=audit,
    )


def _suffix(value: ObjectRef) -> str:
    return hashlib.sha256(value.object_id.encode()).hexdigest()[:32]


__all__ = [
    "GatewayGradingDesignAgent",
    "GatewayGradingDesignPlanningAgent",
    "GradingDesignAgentConfig",
    "GradingDesignAgentError",
    "GradingDesignAgentInputV1",
    "GradingDesignExecution",
    "GradingDesignPlanningAgentConfig",
    "GradingDesignPlanningAgentResult",
    "GradingDesignPlanningInputV1",
]
