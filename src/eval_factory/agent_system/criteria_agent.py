from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, Self

from pydantic import Field, model_validator

from eval_factory.agent_system.criteria_material_store import (
    CriteriaRubricMaterialStore,
)
from eval_factory.agent_system.criteria_planning import (
    CriteriaRubricPlanCompiler,
)
from eval_factory.agent_system.private_store import (
    FactoryPrivateObjectError,
    FactoryPrivateObjectStore,
)
from eval_factory.ai_gateway.protocols import AIGateway
from eval_factory.contracts.agent_system_v2 import (
    AttachmentQualityAssessmentV2,
    AttachmentQualityOutcomeV2,
    CompiledCriteriaRubricPlanV2,
    CriteriaRubricGoalV2,
    CriteriaRubricOutcomeV2,
    CriteriaRubricPlanV2,
    CriteriaRubricResultV2,
    FactoryRunPolicyV2,
    SolvabilityAssessmentV2,
    SolvabilityOutcomeV2,
)
from eval_factory.contracts.ai_gateway_v2 import (
    GatewayInvocationRequestV2,
    GatewayInvocationResultV2,
    GatewayInvocationStatusV2,
    ModelRouteDecisionV2,
    ModelRouteRequestV2,
    PromptTemplateV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.task import RubricVisibility
from eval_factory.contracts.task_v2 import (
    ContestantToolPolicyV2,
    EvaluatorSpecV2,
    ReferencePolicyV2,
    RubricSetV2,
    TaskDraftV2,
    ToolPolicyV2,
    contestant_tool_policy_ref,
    evaluator_spec_ref,
    reference_policy_ref,
    rubric_set_ref,
    task_draft_ref,
    tool_policy_ref,
)
from eval_factory.task_authoring import (
    EvaluationContractCompiler,
    EvaluationContractOutcome,
    EvaluatorBindingDefinition,
    RubricAuthoringOutcome,
    RubricAuthoringPolicyError,
    RubricAuthoringReason,
    RubricCandidateProposal,
    RubricCandidateRequest,
    RubricCandidateRequestBuilder,
    RubricCriterionSelection,
    RubricGeneratedSelectionAdapter,
    RubricSetCompiler,
    ToolCapabilityCatalog,
    ToolPolicyCompilationOutcome,
    ToolPolicyCompiler,
    tool_capability_catalog_ref,
)


class CriteriaRubricAgentError(RuntimeError):
    pass


class CriteriaRubricPlanningInputV1(ContractModelV2):
    schema_version: Literal["eval-factory/private-criteria-rubric-planning-input/v1"] = (
        "eval-factory/private-criteria-rubric-planning-input/v1"
    )
    plan_template: CriteriaRubricPlanV2


class CriteriaRubricAgentInputV1(ContractModelV2):
    schema_version: Literal["eval-factory/private-criteria-rubric-agent-input/v1"] = (
        "eval-factory/private-criteria-rubric-agent-input/v1"
    )
    plan_ref: ObjectRef
    task_draft_ref: ObjectRef
    attachment_quality_ref: ObjectRef
    solvability_ref: ObjectRef
    rubric_request: RubricCandidateRequest
    criterion_goals: tuple[CriteriaRubricGoalV2, ...] = Field(
        min_length=1,
        max_length=10_000,
    )

    @model_validator(mode="after")
    def validate_input(self) -> Self:
        expected = (
            (self.plan_ref, "criteria-rubric-plan"),
            (self.task_draft_ref, "task-draft"),
            (
                self.attachment_quality_ref,
                "attachment-quality-assessment",
            ),
            (self.solvability_ref, "solvability-assessment"),
        )
        if any(ref.object_type != object_type or ref.object_version != "v2" for ref, object_type in expected):
            raise ValueError("criteria/rubric Agent input refs are invalid")
        if self.rubric_request.task_draft_ref != self.task_draft_ref:
            raise ValueError("criteria/rubric Agent request binds another TaskDraft")
        return self


class CriteriaRubricProposalV1(ContractModelV2):
    schema_version: Literal["eval-factory/private-criteria-rubric-proposal/v1"] = (
        "eval-factory/private-criteria-rubric-proposal/v1"
    )
    outcome: RubricAuthoringOutcome
    criteria: tuple[RubricCriterionSelection, ...] = Field(
        default=(),
        max_length=10_000,
    )
    unresolved_reasons: frozenset[RubricAuthoringReason] = frozenset()
    confidence_basis_points: int = Field(ge=0, le=10_000)

    @model_validator(mode="after")
    def validate_proposal(self) -> Self:
        if self.outcome is RubricAuthoringOutcome.COMPILED:
            if not self.criteria or self.unresolved_reasons:
                raise ValueError("compiled criteria proposal requires criteria only")
            if self.confidence_basis_points < 8000:
                raise ValueError("compiled criteria proposal requires sufficient confidence")
        else:
            if self.criteria or not self.unresolved_reasons:
                raise ValueError("non-compiled criteria proposal requires reasons only")
        return self


@dataclass(frozen=True, slots=True)
class CriteriaRubricPlanningAgentConfig:
    prompt: PromptTemplateV2
    agent_definition_ref: ObjectRef
    allowed_model_profile_refs: tuple[ObjectRef, ...]
    budget_reservation_ref: ObjectRef


@dataclass(frozen=True, slots=True)
class CriteriaRubricPlanningAgentResult:
    plan: CriteriaRubricPlanV2
    compiled_plan: CompiledCriteriaRubricPlanV2
    route: ModelRouteDecisionV2
    invocation_result_ref: ObjectRef


@dataclass(frozen=True, slots=True)
class CriteriaRubricAgentConfig:
    prompt: PromptTemplateV2
    agent_definition_ref: ObjectRef
    allowed_model_profile_refs: tuple[ObjectRef, ...]
    budget_reservation_ref: ObjectRef


@dataclass(frozen=True, slots=True)
class CriteriaRubricExecution:
    result: CriteriaRubricResultV2
    route: ModelRouteDecisionV2
    invocation_result_ref: ObjectRef
    rubric_set: RubricSetV2 | None
    evaluator_spec: EvaluatorSpecV2 | None
    reference_policy: ReferencePolicyV2 | None
    tool_policy: ToolPolicyV2 | None
    contestant_tool_policy: ContestantToolPolicyV2 | None


class GatewayCriteriaRubricPlanningAgent:
    def __init__(
        self,
        *,
        gateway: AIGateway,
        private_store: FactoryPrivateObjectStore,
        compiler: CriteriaRubricPlanCompiler,
        config: CriteriaRubricPlanningAgentConfig,
    ) -> None:
        self.gateway = gateway
        self.private_store = private_store
        self.compiler = compiler
        self.config = config

    async def propose(
        self,
        *,
        task_ref: ObjectRef,
        plan_template: CriteriaRubricPlanV2,
        policy: FactoryRunPolicyV2,
        audit: ContractAudit,
    ) -> CriteriaRubricPlanningAgentResult:
        planning_input = CriteriaRubricPlanningInputV1(
            plan_template=plan_template,
        )
        rendering_ref = self.private_store.put_model(
            object_type="prompt-rendering",
            value=planning_input,
        )
        route = self.gateway.route(
            _route_request(
                task_ref=task_ref,
                rendering_ref=rendering_ref,
                prompt=self.config.prompt,
                agent_definition_ref=(self.config.agent_definition_ref),
                allowed_model_profile_refs=(self.config.allowed_model_profile_refs),
                budget_reservation_ref=(self.config.budget_reservation_ref),
                task_kind="criteria-rubric-planning",
                audit=audit,
            )
        )
        invocation = _invocation_request(
            task_ref=task_ref,
            route=route,
            prompt=self.config.prompt,
            rendering_ref=rendering_ref,
            task_kind="criteria-rubric-planning",
            audit=audit,
        )
        result = await self.gateway.invoke(invocation, route=route)
        if result.status is not GatewayInvocationStatusV2.SUCCEEDED or result.output_ref is None:
            raise CriteriaRubricAgentError("criteria/rubric planning invocation did not succeed")
        try:
            plan = self.private_store.get_model(
                result.output_ref,
                CriteriaRubricPlanV2,
            )
        except FactoryPrivateObjectError as exc:
            raise CriteriaRubricAgentError("criteria/rubric planning output is malformed") from exc
        self._validate_output(plan_template, plan)
        compiled = self.compiler.compile(
            plan=plan,
            policy=policy,
            audit=audit,
        )
        return CriteriaRubricPlanningAgentResult(
            plan=plan,
            compiled_plan=compiled,
            route=route,
            invocation_result_ref=result.to_ref(),
        )

    @staticmethod
    def _validate_output(
        template: CriteriaRubricPlanV2,
        plan: CriteriaRubricPlanV2,
    ) -> None:
        fixed_fields = (
            "run_ref",
            "plan_version",
            "predecessor_plan_ref",
            "task_draft_ref",
            "attachment_quality_ref",
            "solvability_ref",
            "allowed_prompt_requirement_ids",
            "required_prompt_requirement_ids",
            "allowed_attachment_dependency_ids",
            "required_attachment_dependency_ids",
            "allowed_task_tool_ids",
            "required_task_tool_ids",
            "allowed_evaluator_binding_ids",
            "allowed_reference_modes",
            "tool_catalog_ref",
            "agent_role",
            "required_capability_ids",
            "specialist_tool_ids",
            "data_purpose",
            "data_classifications",
            "prompt_template_ref",
            "model_policy_ref",
            "acceptance_check_refs",
            "max_attempts",
            "max_model_requests",
            "max_model_tokens",
            "max_cost_micro_usd",
        )
        if any(getattr(template, field_name) != getattr(plan, field_name) for field_name in fixed_fields):
            raise CriteriaRubricAgentError("criteria/rubric planning output widens fixed authority")


class GatewayCriteriaRubricAgent:
    def __init__(
        self,
        *,
        gateway: AIGateway,
        private_store: FactoryPrivateObjectStore,
        material_store: CriteriaRubricMaterialStore,
        config: CriteriaRubricAgentConfig,
    ) -> None:
        self.gateway = gateway
        self.private_store = private_store
        self.material_store = material_store
        self.config = config

    async def author(
        self,
        *,
        task_ref: ObjectRef,
        plan: CriteriaRubricPlanV2,
        task_draft: TaskDraftV2,
        attachment_quality: AttachmentQualityAssessmentV2,
        solvability: SolvabilityAssessmentV2,
        binding_definitions: tuple[EvaluatorBindingDefinition, ...],
        tool_catalog: ToolCapabilityCatalog,
        audit: ContractAudit,
    ) -> CriteriaRubricExecution:
        rubric_request = self.validate_sources(
            plan=plan,
            task_draft=task_draft,
            attachment_quality=attachment_quality,
            solvability=solvability,
            tool_catalog=tool_catalog,
            audit=audit,
        )
        agent_input = CriteriaRubricAgentInputV1(
            plan_ref=plan.to_ref(),
            task_draft_ref=task_draft_ref(task_draft),
            attachment_quality_ref=attachment_quality.to_ref(),
            solvability_ref=solvability.to_ref(),
            rubric_request=rubric_request,
            criterion_goals=plan.criterion_goals,
        )
        rendering_ref = self.private_store.put_model(
            object_type="prompt-rendering",
            value=agent_input,
        )
        route = self.gateway.route(
            _route_request(
                task_ref=task_ref,
                rendering_ref=rendering_ref,
                prompt=self.config.prompt,
                agent_definition_ref=(self.config.agent_definition_ref),
                allowed_model_profile_refs=(self.config.allowed_model_profile_refs),
                budget_reservation_ref=(self.config.budget_reservation_ref),
                task_kind="criteria-rubric",
                audit=audit,
            )
        )
        invocation = _invocation_request(
            task_ref=task_ref,
            route=route,
            prompt=self.config.prompt,
            rendering_ref=rendering_ref,
            task_kind="criteria-rubric",
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
            return self._commit_blocked(
                plan=plan,
                route=route,
                gateway_result=gateway_result,
                proposal_ref=None,
                outcome=CriteriaRubricOutcomeV2.BLOCKED_CAPABILITY,
                reason_codes=(gateway_result.failure_code or "MODEL_PROVIDER_FAILED",),
                audit=audit,
            )
        try:
            proposal = self.private_store.get_model(
                gateway_result.output_ref,
                CriteriaRubricProposalV1,
            )
        except FactoryPrivateObjectError as exc:
            raise CriteriaRubricAgentError("criteria/rubric provider output is malformed") from exc
        self._validate_proposal(plan, proposal)
        if route.selected_model_profile_ref is None:
            raise CriteriaRubricAgentError("criteria/rubric route lacks a selected model")
        r4_proposal = RubricGeneratedSelectionAdapter().adapt(
            rubric_request,
            outcome=proposal.outcome,
            criteria=proposal.criteria,
            unresolved_reasons=proposal.unresolved_reasons,
            model_profile=route.selected_model_profile_ref.object_id,
            prompt_version=(
                f"{self.config.prompt.prompt_template_id}/v{self.config.prompt.template_version}"
            ),
            audit=audit,
        )
        rubric_result = RubricSetCompiler().compile(
            request=rubric_request,
            proposal=r4_proposal,
            task_draft=task_draft,
            audit=audit,
        )
        if rubric_result.outcome is not RubricAuthoringOutcome.COMPILED:
            return self._commit_blocked(
                plan=plan,
                route=route,
                gateway_result=gateway_result,
                proposal_ref=gateway_result.output_ref,
                outcome=_rubric_outcome(rubric_result.outcome),
                reason_codes=tuple(
                    value.value
                    for value in sorted(
                        rubric_result.unresolved_reasons,
                        key=lambda value: value.value,
                    )
                ),
                audit=audit,
            )
        if rubric_result.rubric_set is None:
            raise CriteriaRubricAgentError("compiled rubric result lacks RubricSet authority")
        rubric_set = rubric_result.rubric_set
        evaluation_result = EvaluationContractCompiler().compile(
            rubric_set=rubric_set,
            binding_definitions=binding_definitions,
            reference_mode=plan.selected_reference_mode,
            audit=audit,
        )
        if evaluation_result.outcome is not EvaluationContractOutcome.COMPILED:
            return self._commit_blocked(
                plan=plan,
                route=route,
                gateway_result=gateway_result,
                proposal_ref=gateway_result.output_ref,
                outcome=(
                    CriteriaRubricOutcomeV2.BLOCKED_BINDING
                    if evaluation_result.outcome is EvaluationContractOutcome.BLOCKED_BINDING
                    else CriteriaRubricOutcomeV2.BLOCKED_POLICY
                ),
                reason_codes=tuple(
                    value.value
                    for value in sorted(
                        evaluation_result.unresolved_reasons,
                        key=lambda value: value.value,
                    )
                ),
                audit=audit,
            )
        if evaluation_result.evaluator_spec is None or evaluation_result.reference_policy is None:
            raise CriteriaRubricAgentError("compiled evaluation result lacks complete authority")
        evaluator_spec = evaluation_result.evaluator_spec
        reference_policy = evaluation_result.reference_policy
        tool_result = ToolPolicyCompiler().compile(
            task_draft=task_draft,
            rubric_set=rubric_set,
            evaluator_spec=evaluator_spec,
            catalog=tool_catalog,
            audit=audit,
        )
        if tool_result.outcome is not ToolPolicyCompilationOutcome.COMPILED:
            return self._commit_blocked(
                plan=plan,
                route=route,
                gateway_result=gateway_result,
                proposal_ref=gateway_result.output_ref,
                outcome=(
                    CriteriaRubricOutcomeV2.BLOCKED_CAPABILITY
                    if tool_result.outcome is ToolPolicyCompilationOutcome.BLOCKED_CAPABILITY
                    else CriteriaRubricOutcomeV2.BLOCKED_POLICY
                ),
                reason_codes=tuple(
                    value.value
                    for value in sorted(
                        tool_result.unresolved_reasons,
                        key=lambda value: value.value,
                    )
                ),
                audit=audit,
            )
        if tool_result.tool_policy is None or tool_result.contestant_projection is None:
            raise CriteriaRubricAgentError("compiled tool result lacks complete authority")
        tool_policy = tool_result.tool_policy
        contestant_projection = tool_result.contestant_projection
        self._validate_current_chain(
            rubric_request=rubric_request,
            r4_proposal=r4_proposal,
            task_draft=task_draft,
            rubric_set=rubric_set,
            binding_definitions=binding_definitions,
            evaluator_spec=evaluator_spec,
            reference_policy=reference_policy,
            tool_catalog=tool_catalog,
            tool_policy=tool_policy,
            contestant_projection=contestant_projection,
            audit=audit,
        )
        self._persist_private_chain(
            rubric_set=rubric_set,
            evaluator_spec=evaluator_spec,
            reference_policy=reference_policy,
            tool_policy=tool_policy,
            contestant_projection=contestant_projection,
        )
        result = CriteriaRubricResultV2.create(
            result_id=(f"criteria-rubric-result://{plan.object_sha256}"),
            plan_ref=plan.to_ref(),
            task_draft_ref=task_draft_ref(task_draft),
            attachment_quality_ref=attachment_quality.to_ref(),
            solvability_ref=solvability.to_ref(),
            route_decision_ref=route.to_ref(),
            gateway_receipt_ref=gateway_result.receipt_ref,
            gateway_invocation_result_ref=gateway_result.to_ref(),
            proposal_ref=gateway_result.output_ref,
            rubric_set_ref=rubric_set_ref(rubric_set),
            evaluator_spec_ref=evaluator_spec_ref(evaluator_spec),
            reference_policy_ref=reference_policy_ref(reference_policy),
            tool_policy_ref=tool_policy_ref(tool_policy),
            contestant_tool_policy_ref=contestant_tool_policy_ref(contestant_projection),
            outcome=CriteriaRubricOutcomeV2.SUCCEEDED,
            reason_codes=(),
            audit=audit,
        )
        return CriteriaRubricExecution(
            result=result,
            route=route,
            invocation_result_ref=gateway_result.to_ref(),
            rubric_set=rubric_set,
            evaluator_spec=evaluator_spec,
            reference_policy=reference_policy,
            tool_policy=tool_policy,
            contestant_tool_policy=contestant_projection,
        )

    def validate_sources(
        self,
        *,
        plan: CriteriaRubricPlanV2,
        task_draft: TaskDraftV2,
        attachment_quality: AttachmentQualityAssessmentV2,
        solvability: SolvabilityAssessmentV2,
        tool_catalog: ToolCapabilityCatalog,
        audit: ContractAudit,
    ) -> RubricCandidateRequest:
        self._validate_source_refs(
            plan=plan,
            task_draft=task_draft,
            attachment_quality=attachment_quality,
            solvability=solvability,
            tool_catalog=tool_catalog,
        )
        try:
            return RubricCandidateRequestBuilder().build(
                task_draft=task_draft,
                allowed_evaluator_bindings=(plan.allowed_evaluator_binding_ids),
                audit=audit,
            )
        except RubricAuthoringPolicyError as exc:
            raise CriteriaRubricAgentError("criteria/rubric TaskDraft authority is stale") from exc

    @staticmethod
    def _validate_source_refs(
        *,
        plan: CriteriaRubricPlanV2,
        task_draft: TaskDraftV2,
        attachment_quality: AttachmentQualityAssessmentV2,
        solvability: SolvabilityAssessmentV2,
        tool_catalog: ToolCapabilityCatalog,
    ) -> None:
        if (
            plan.task_draft_ref != task_draft_ref(task_draft)
            or plan.attachment_quality_ref != attachment_quality.to_ref()
            or plan.solvability_ref != solvability.to_ref()
            or plan.tool_catalog_ref != tool_capability_catalog_ref(tool_catalog)
        ):
            raise CriteriaRubricAgentError("criteria/rubric upstream authority is stale")
        if (
            attachment_quality.outcome is not AttachmentQualityOutcomeV2.PASSED
            or solvability.outcome is not SolvabilityOutcomeV2.SOLVABLE
            or solvability.quality_assessment_ref != attachment_quality.to_ref()
        ):
            raise CriteriaRubricAgentError("criteria/rubric requires passing attachment authority")
        if set(plan.allowed_prompt_requirement_ids) != set(task_draft.prompt_requirement_ids):
            raise CriteriaRubricAgentError("criteria/rubric prompt requirement scope is stale")
        if set(plan.allowed_attachment_dependency_ids) != {
            value.dependency_id for value in task_draft.attachment_dependencies
        }:
            raise CriteriaRubricAgentError("criteria/rubric attachment dependency scope is stale")
        if set(plan.allowed_task_tool_ids) != set(task_draft.allowed_tools):
            raise CriteriaRubricAgentError("criteria/rubric task tool scope is stale")

    @staticmethod
    def _validate_proposal(
        plan: CriteriaRubricPlanV2,
        proposal: CriteriaRubricProposalV1,
    ) -> None:
        if proposal.outcome is not RubricAuthoringOutcome.COMPILED:
            return
        goals = {value.goal_id: value for value in plan.criterion_goals}
        selections = {value.selection_id: value for value in proposal.criteria}
        if set(goals) != set(selections):
            raise CriteriaRubricAgentError("criteria proposal does not exactly cover plan goals")
        for goal_id, goal in goals.items():
            selection = selections[goal_id]
            if (
                selection.judged_object_kind is not goal.judged_object_kind
                or selection.prompt_requirement_ids != goal.prompt_requirement_ids
                or selection.attachment_dependency_ids != goal.attachment_dependency_ids
                or selection.allowed_tool_ids != goal.allowed_tool_ids
                or selection.evaluator_binding != goal.evaluator_binding_id
                or selection.visibility is not RubricVisibility.EVALUATOR_ONLY
                or Decimal(str(selection.weight)) * 10_000 != Decimal(goal.weight_basis_points)
            ):
                raise CriteriaRubricAgentError("criteria proposal widens reviewed goal authority")

    @staticmethod
    def _validate_current_chain(
        *,
        rubric_request: RubricCandidateRequest,
        r4_proposal: RubricCandidateProposal,
        task_draft: TaskDraftV2,
        rubric_set: RubricSetV2,
        binding_definitions: tuple[EvaluatorBindingDefinition, ...],
        evaluator_spec: EvaluatorSpecV2,
        reference_policy: ReferencePolicyV2,
        tool_catalog: ToolCapabilityCatalog,
        tool_policy: ToolPolicyV2,
        contestant_projection: ContestantToolPolicyV2,
        audit: ContractAudit,
    ) -> None:
        rubric_rebuilt = RubricSetCompiler().compile(
            request=rubric_request,
            proposal=r4_proposal,
            task_draft=task_draft,
            audit=audit,
        )
        if rubric_rebuilt.rubric_set != rubric_set:
            raise CriteriaRubricAgentError("current RubricSet differs from deterministic replay")
        evaluation_rebuilt = EvaluationContractCompiler().compile(
            rubric_set=rubric_set,
            binding_definitions=binding_definitions,
            reference_mode=reference_policy.mode,
            audit=audit,
        )
        if (
            evaluation_rebuilt.evaluator_spec != evaluator_spec
            or evaluation_rebuilt.reference_policy != reference_policy
        ):
            raise CriteriaRubricAgentError("current evaluation contracts differ from deterministic replay")
        ToolPolicyCompiler().validate_current(
            task_draft=task_draft,
            rubric_set=rubric_set,
            evaluator_spec=evaluator_spec,
            catalog=tool_catalog,
            tool_policy=tool_policy,
            contestant_projection=contestant_projection,
        )

    def _persist_private_chain(
        self,
        *,
        rubric_set: RubricSetV2,
        evaluator_spec: EvaluatorSpecV2,
        reference_policy: ReferencePolicyV2,
        tool_policy: ToolPolicyV2,
        contestant_projection: ContestantToolPolicyV2,
    ) -> None:
        for behavior_ref, value in (
            (rubric_set_ref(rubric_set), rubric_set),
            (evaluator_spec_ref(evaluator_spec), evaluator_spec),
            (
                reference_policy_ref(reference_policy),
                reference_policy,
            ),
            (tool_policy_ref(tool_policy), tool_policy),
            (
                contestant_tool_policy_ref(contestant_projection),
                contestant_projection,
            ),
        ):
            self.material_store.put_model(
                behavior_ref=behavior_ref,
                value=value,
            )

    def _commit_blocked(
        self,
        *,
        plan: CriteriaRubricPlanV2,
        route: ModelRouteDecisionV2,
        gateway_result: GatewayInvocationResultV2,
        proposal_ref: ObjectRef | None,
        outcome: CriteriaRubricOutcomeV2,
        reason_codes: tuple[str, ...],
        audit: ContractAudit,
    ) -> CriteriaRubricExecution:
        result = CriteriaRubricResultV2.create(
            result_id=(f"criteria-rubric-result://{plan.object_sha256}"),
            plan_ref=plan.to_ref(),
            task_draft_ref=plan.task_draft_ref,
            attachment_quality_ref=plan.attachment_quality_ref,
            solvability_ref=plan.solvability_ref,
            route_decision_ref=route.to_ref(),
            gateway_receipt_ref=gateway_result.receipt_ref,
            gateway_invocation_result_ref=gateway_result.to_ref(),
            proposal_ref=proposal_ref,
            rubric_set_ref=None,
            evaluator_spec_ref=None,
            reference_policy_ref=None,
            tool_policy_ref=None,
            contestant_tool_policy_ref=None,
            outcome=outcome,
            reason_codes=reason_codes,
            audit=audit,
        )
        return CriteriaRubricExecution(
            result=result,
            route=route,
            invocation_result_ref=gateway_result.to_ref(),
            rubric_set=None,
            evaluator_spec=None,
            reference_policy=None,
            tool_policy=None,
            contestant_tool_policy=None,
        )


def _rubric_outcome(
    outcome: RubricAuthoringOutcome,
) -> CriteriaRubricOutcomeV2:
    return {
        RubricAuthoringOutcome.ABSTAIN: (CriteriaRubricOutcomeV2.ABSTAINED),
        RubricAuthoringOutcome.BLOCKED_REACHABILITY: (CriteriaRubricOutcomeV2.BLOCKED_REACHABILITY),
        RubricAuthoringOutcome.BLOCKED_CAPABILITY: (CriteriaRubricOutcomeV2.BLOCKED_CAPABILITY),
    }[outcome]


def _route_request(
    *,
    task_ref: ObjectRef,
    rendering_ref: ObjectRef,
    prompt: PromptTemplateV2,
    agent_definition_ref: ObjectRef,
    allowed_model_profile_refs: tuple[ObjectRef, ...],
    budget_reservation_ref: ObjectRef,
    task_kind: str,
    audit: ContractAudit,
) -> ModelRouteRequestV2:
    return ModelRouteRequestV2.create(
        route_request_id=(f"model-route-request://{task_kind}/{_suffix(rendering_ref)}"),
        agent_task_ref=task_ref,
        agent_definition_ref=agent_definition_ref,
        task_kind=task_kind,
        prompt_template_ref=prompt.to_ref(),
        required_capabilities=prompt.required_model_capabilities,
        data_classification="RESTRICTED_EVALUATOR_CONTROL",
        residency="LOCAL",
        input_token_budget=16_000,
        output_token_budget=8_000,
        max_cost_micro_usd=500_000,
        minimum_quality_basis_points=8500,
        allowed_model_profile_refs=allowed_model_profile_refs,
        budget_reservation_ref=budget_reservation_ref,
        route_version=1,
        predecessor_route_ref=None,
        failed_receipt_ref=None,
        generator_model_profile_ref=None,
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
    "CriteriaRubricAgentConfig",
    "CriteriaRubricAgentError",
    "CriteriaRubricAgentInputV1",
    "CriteriaRubricExecution",
    "CriteriaRubricPlanningAgentConfig",
    "CriteriaRubricPlanningAgentResult",
    "CriteriaRubricPlanningInputV1",
    "CriteriaRubricProposalV1",
    "GatewayCriteriaRubricAgent",
    "GatewayCriteriaRubricPlanningAgent",
]
