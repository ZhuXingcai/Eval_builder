from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

from eval_factory.agent_system.attachment_quality_runtime import (
    FactoryAttachmentQualityContext,
    FactoryAttachmentQualityView,
)
from eval_factory.agent_system.criteria_agent import (
    CriteriaRubricExecution,
    GatewayCriteriaRubricPlanningAgent,
)
from eval_factory.agent_system.criteria_authority_material import (
    FactoryCriteriaAuthorityMaterialStore,
    FactoryCriteriaAuthorityResult,
)
from eval_factory.agent_system.criteria_material_store import (
    CriteriaRubricMaterialStore,
)
from eval_factory.agent_system.criteria_subgraph import (
    CriteriaRubricSupervisedExecution,
    SupervisedCriteriaRubricRunner,
)
from eval_factory.agent_system.plan_review import (
    PlanReviewError,
    PlanReviewService,
)
from eval_factory.agent_system.registry import AgentRegistry
from eval_factory.agent_system.store import (
    FactoryControlNotFoundError,
    FactoryControlStore,
)
from eval_factory.attachment_planning import ItemQualityCompiler
from eval_factory.contracts.agent_system_v2 import (
    CompiledCriteriaRubricPlanV2,
    CriteriaRubricGoalV2,
    CriteriaRubricOutcomeV2,
    CriteriaRubricPlanV2,
    CriteriaRubricResultV2,
    FactoryRunPolicyV2,
    PlanKindV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.core_v2 import canonical_value_v2
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryItemRunBindingV2,
    FactoryItemStageHeadV2,
    FactoryItemStageOutcomeV2,
    FactoryItemStageV2,
)
from eval_factory.contracts.task import ReferenceMode
from eval_factory.contracts.task_v2 import (
    ContestantToolPolicyV2,
    EvaluatorSpecV2,
    R4TaskContractSetV2,
    ReferencePolicyV2,
    RubricJudgedObjectKindV2,
    RubricSetV2,
    TaskDraftV2,
    ToolPolicyV2,
    contestant_tool_policy_ref,
    evaluator_spec_ref,
    r4_task_contract_set_carried_sha256,
    reference_policy_ref,
    rubric_set_ref,
    task_draft_ref,
    tool_policy_ref,
)
from eval_factory.task_authoring import (
    EvaluatorBindingDefinition,
    ToolCapabilityCatalog,
    tool_capability_catalog_ref,
)


class FactoryCriteriaItemRuntimeError(RuntimeError):
    pass


class FactoryCriteriaItemState(StrEnum):
    WAITING_REVIEW = "WAITING_REVIEW"
    READY_TO_EXECUTE = "READY_TO_EXECUTE"
    EXECUTED = "EXECUTED"


@dataclass(frozen=True, slots=True)
class FactoryCriteriaItemView:
    item_id: str
    plan_ref: ObjectRef
    compiled_plan_ref: ObjectRef
    review_ref: ObjectRef
    stage_head_ref: ObjectRef | None
    material_ref: ObjectRef | None
    authority: FactoryCriteriaAuthorityResult | None
    state: FactoryCriteriaItemState


class FactoryCriteriaPlanTemplateBuilder:
    def __init__(self, registry: AgentRegistry) -> None:
        self.registry = registry

    def build(
        self,
        *,
        binding: FactoryItemRunBindingV2,
        run_ref: ObjectRef,
        task_draft: TaskDraftV2,
        quality: FactoryAttachmentQualityView,
        binding_definitions: tuple[
            EvaluatorBindingDefinition,
            ...,
        ],
        tool_catalog: ToolCapabilityCatalog,
        policy: FactoryRunPolicyV2,
        audit: ContractAudit,
    ) -> CriteriaRubricPlanV2:
        if not binding_definitions:
            raise FactoryCriteriaItemRuntimeError("criteria planning requires evaluator bindings")
        definition = self.registry.resolve(
            "criteria-rubric-agent",
            "criteria-rubric",
        )
        capabilities = self.registry.capabilities_for(definition)
        evaluator_ids = tuple(sorted(value.evaluator_binding_id for value in binding_definitions))
        prompt_ids = task_draft.prompt_requirement_ids
        dependency_ids = tuple(sorted(value.dependency_id for value in task_draft.attachment_dependencies))
        tool_ids = task_draft.allowed_tools
        if dependency_ids:
            judged_kind = RubricJudgedObjectKindV2.WORKSPACE_STATE
        elif tool_ids:
            judged_kind = RubricJudgedObjectKindV2.TOOL_BEHAVIOR
        else:
            judged_kind = RubricJudgedObjectKindV2.CONTESTANT_RESPONSE
        goal_digest = hashlib.sha256(
            (f"{binding.object_id}|{quality.item_quality_head_ref.object_id}").encode()
        ).hexdigest()
        goal = CriteriaRubricGoalV2(
            goal_id=f"criteria-goal://sha256/{goal_digest}",
            goal_summary=(
                "Judge the visible response and current input workspace against all task requirements."
            ),
            judged_object_kind=judged_kind,
            prompt_requirement_ids=prompt_ids,
            attachment_dependency_ids=dependency_ids,
            allowed_tool_ids=tool_ids,
            evaluator_binding_id=evaluator_ids[0],
            weight_basis_points=10_000,
        )
        finalization = quality.result.finalization
        return CriteriaRubricPlanV2.create(
            plan_id=(f"criteria-rubric-plan://factory-item/{binding.object_sha256}"),
            run_ref=run_ref,
            plan_version=1,
            predecessor_plan_ref=None,
            task_draft_ref=task_draft_ref(task_draft),
            attachment_quality_ref=(finalization.attachment_quality.to_ref()),
            solvability_ref=(quality.result.solvability.assessment.to_ref()),
            allowed_prompt_requirement_ids=prompt_ids,
            required_prompt_requirement_ids=prompt_ids,
            allowed_attachment_dependency_ids=dependency_ids,
            required_attachment_dependency_ids=dependency_ids,
            allowed_task_tool_ids=tool_ids,
            required_task_tool_ids=tool_ids,
            criterion_goals=(goal,),
            allowed_evaluator_binding_ids=evaluator_ids,
            allowed_reference_modes=(ReferenceMode.NONE,),
            selected_reference_mode=ReferenceMode.NONE,
            tool_catalog_ref=tool_capability_catalog_ref(tool_catalog),
            agent_role=definition.agent_role,
            required_capability_ids=tuple(sorted(value.capability_id for value in capabilities)),
            specialist_tool_ids=definition.tool_ids,
            data_classifications=(definition.allowed_data_classifications),
            prompt_template_ref=(definition.prompt_template_ref),
            model_policy_ref=definition.model_policy_ref,
            acceptance_check_refs=definition.validator_refs,
            max_attempts=min(
                definition.max_attempts,
                policy.max_agent_attempts,
            ),
            max_model_requests=min(
                definition.max_model_requests,
                policy.max_model_requests,
            ),
            max_model_tokens=min(
                definition.max_model_tokens,
                policy.max_model_tokens,
            ),
            max_cost_micro_usd=min(
                definition.max_cost_micro_usd,
                policy.max_cost_micro_usd,
            ),
            audit=audit,
        )


class FactoryCriteriaItemRuntime:
    def __init__(
        self,
        *,
        store: FactoryControlStore,
        plan_reviews: PlanReviewService,
        planning_agent: GatewayCriteriaRubricPlanningAgent,
        runner: SupervisedCriteriaRubricRunner,
        template_builder: FactoryCriteriaPlanTemplateBuilder,
        criteria_materials: CriteriaRubricMaterialStore,
        authority_materials: (FactoryCriteriaAuthorityMaterialStore),
        requested_by: str,
    ) -> None:
        self.store = store
        self.plan_reviews = plan_reviews
        self.planning_agent = planning_agent
        self.runner = runner
        self.template_builder = template_builder
        self.criteria_materials = criteria_materials
        self.authority_materials = authority_materials
        self.requested_by = requested_by

    async def advance(
        self,
        *,
        binding: FactoryItemRunBindingV2,
        policy: FactoryRunPolicyV2,
        task_draft: TaskDraftV2,
        task_contract_set: R4TaskContractSetV2,
        quality: FactoryAttachmentQualityView,
        quality_context: FactoryAttachmentQualityContext,
        binding_definitions: tuple[
            EvaluatorBindingDefinition,
            ...,
        ],
        tool_catalog: ToolCapabilityCatalog,
        audit: ContractAudit,
    ) -> FactoryCriteriaItemView:
        review = await self.prepare_review(
            binding=binding,
            policy=policy,
            task_draft=task_draft,
            task_contract_set=task_contract_set,
            quality=quality,
            quality_context=quality_context,
            binding_definitions=binding_definitions,
            tool_catalog=tool_catalog,
            audit=audit,
        )
        if review.state is FactoryCriteriaItemState.READY_TO_EXECUTE:
            return await self.execute_reviewed(
                binding=binding,
                policy=policy,
                task_draft=task_draft,
                task_contract_set=task_contract_set,
                quality=quality,
                quality_context=quality_context,
                binding_definitions=binding_definitions,
                tool_catalog=tool_catalog,
                audit=audit,
            )
        return review

    async def prepare_review(
        self,
        *,
        binding: FactoryItemRunBindingV2,
        policy: FactoryRunPolicyV2,
        task_draft: TaskDraftV2,
        task_contract_set: R4TaskContractSetV2,
        quality: FactoryAttachmentQualityView,
        quality_context: FactoryAttachmentQualityContext,
        binding_definitions: tuple[
            EvaluatorBindingDefinition,
            ...,
        ],
        tool_catalog: ToolCapabilityCatalog,
        audit: ContractAudit,
    ) -> FactoryCriteriaItemView:
        del task_contract_set, quality_context
        item_run = self.store.get_item_run(binding)
        if item_run.policy_ref != policy.to_ref():
            raise FactoryCriteriaItemRuntimeError("criteria item run policy authority changed")
        quality_head = self.store.get_item_stage_head(
            binding.item_id,
            FactoryItemStageV2.ITEM_QUALITY,
        )
        if (
            quality_head.to_ref() != quality.item_quality_head_ref
            or quality_head.outcome is not FactoryItemStageOutcomeV2.SUCCEEDED
        ):
            raise FactoryCriteriaItemRuntimeError(
                "criteria planning requires current successful item quality"
            )
        try:
            material = self.store.get_domain_plan(
                item_run.run_id,
                PlanKindV2.CRITERIA_RUBRIC,
            )
        except FactoryControlNotFoundError:
            template = self.template_builder.build(
                binding=binding,
                run_ref=item_run.to_ref(),
                task_draft=task_draft,
                quality=quality,
                binding_definitions=binding_definitions,
                tool_catalog=tool_catalog,
                policy=policy,
                audit=audit,
            )
            planning = await self.planning_agent.propose(
                task_ref=_stable_ref(
                    "agent-task",
                    f"{binding.object_id}:criteria-planning",
                ),
                plan_template=template,
                policy=policy,
                audit=audit,
            )
            self.store.commit_domain_plan(
                run_id=item_run.run_id,
                expected_run_version=item_run.run_version,
                plan_kind=PlanKindV2.CRITERIA_RUBRIC,
                plan=planning.plan,
                compiled_plan=planning.compiled_plan,
                audit=audit,
                idempotency_key=(f"commit-factory-criteria-plan-{planning.plan.object_sha256}"),
                material_ref=quality.material_ref,
            )
            plan = planning.plan
            compiled = planning.compiled_plan
        else:
            plan = CriteriaRubricPlanV2.model_validate_json(material.plan_record_json)
            compiled = CompiledCriteriaRubricPlanV2.model_validate_json(material.compiled_plan_record_json)
            if (
                self.store.get_domain_plan_material_ref(
                    run_id=item_run.run_id,
                    plan_kind=PlanKindV2.CRITERIA_RUBRIC,
                )
                != quality.material_ref
            ):
                raise FactoryCriteriaItemRuntimeError("criteria plan quality material drifted")
        try:
            review = self.plan_reviews.require_resumed_plan(
                run_id=item_run.run_id,
                plan_kind=PlanKindV2.CRITERIA_RUBRIC,
                plan_ref=plan.to_ref(),
            )
        except PlanReviewError:
            review = self.plan_reviews.open_plan(
                run_id=item_run.run_id,
                plan_kind=PlanKindV2.CRITERIA_RUBRIC,
                requested_by=self.requested_by,
                idempotency_key=(f"open-factory-criteria-review-{plan.object_sha256}"),
                audit=audit,
            )
            return FactoryCriteriaItemView(
                item_id=binding.item_id,
                plan_ref=plan.to_ref(),
                compiled_plan_ref=compiled.to_ref(),
                review_ref=review.request.to_ref(),
                stage_head_ref=None,
                material_ref=None,
                authority=None,
                state=FactoryCriteriaItemState.WAITING_REVIEW,
            )
        return FactoryCriteriaItemView(
            item_id=binding.item_id,
            plan_ref=plan.to_ref(),
            compiled_plan_ref=compiled.to_ref(),
            review_ref=review.request.to_ref(),
            stage_head_ref=None,
            material_ref=None,
            authority=None,
            state=FactoryCriteriaItemState.READY_TO_EXECUTE,
        )

    async def execute_reviewed(
        self,
        *,
        binding: FactoryItemRunBindingV2,
        policy: FactoryRunPolicyV2,
        task_draft: TaskDraftV2,
        task_contract_set: R4TaskContractSetV2,
        quality: FactoryAttachmentQualityView,
        quality_context: FactoryAttachmentQualityContext,
        binding_definitions: tuple[
            EvaluatorBindingDefinition,
            ...,
        ],
        tool_catalog: ToolCapabilityCatalog,
        audit: ContractAudit,
    ) -> FactoryCriteriaItemView:
        item_run = self.store.get_item_run(binding)
        if item_run.policy_ref != policy.to_ref():
            raise FactoryCriteriaItemRuntimeError(
                "criteria item run policy authority changed",
            )
        quality_head = self.store.get_item_stage_head(
            binding.item_id,
            FactoryItemStageV2.ITEM_QUALITY,
        )
        if (
            quality_head.to_ref() != quality.item_quality_head_ref
            or quality_head.outcome is not FactoryItemStageOutcomeV2.SUCCEEDED
        ):
            raise FactoryCriteriaItemRuntimeError(
                "criteria execution requires current successful item quality",
            )
        material = self.store.get_domain_plan(
            item_run.run_id,
            PlanKindV2.CRITERIA_RUBRIC,
        )
        plan = CriteriaRubricPlanV2.model_validate_json(
            material.plan_record_json,
        )
        compiled = CompiledCriteriaRubricPlanV2.model_validate_json(
            material.compiled_plan_record_json,
        )
        if (
            self.store.get_domain_plan_material_ref(
                run_id=item_run.run_id,
                plan_kind=PlanKindV2.CRITERIA_RUBRIC,
            )
            != quality.material_ref
        ):
            raise FactoryCriteriaItemRuntimeError(
                "criteria plan quality material drifted",
            )
        review = self.plan_reviews.require_resumed_plan(
            run_id=item_run.run_id,
            plan_kind=PlanKindV2.CRITERIA_RUBRIC,
            plan_ref=plan.to_ref(),
        )
        supervised: CriteriaRubricSupervisedExecution | None
        try:
            self.store.get_item_stage_head(
                binding.item_id,
                FactoryItemStageV2.CRITERIA_RUBRIC,
            )
        except FactoryControlNotFoundError:
            supervised = await self.runner.run(
                run_id=item_run.run_id,
                plan=plan,
                compiled_plan=compiled,
                task_draft=task_draft,
                attachment_quality=(quality.result.finalization.attachment_quality),
                solvability=(quality.result.solvability.assessment),
                binding_definitions=binding_definitions,
                tool_catalog=tool_catalog,
                audit=audit,
            )
        else:
            supervised = None
        return self.commit_execution(
            binding=binding,
            plan=plan,
            compiled=compiled,
            review_ref=review.request.to_ref(),
            supervised=supervised,
            task_draft=task_draft,
            task_contract_set=task_contract_set,
            quality=quality,
            quality_context=quality_context,
            tool_catalog=tool_catalog,
            audit=audit,
        )

    def commit_execution(
        self,
        *,
        binding: FactoryItemRunBindingV2,
        plan: CriteriaRubricPlanV2,
        compiled: CompiledCriteriaRubricPlanV2,
        review_ref: ObjectRef,
        supervised: CriteriaRubricSupervisedExecution | None,
        task_draft: TaskDraftV2,
        task_contract_set: R4TaskContractSetV2,
        quality: FactoryAttachmentQualityView,
        quality_context: FactoryAttachmentQualityContext,
        tool_catalog: ToolCapabilityCatalog,
        audit: ContractAudit,
    ) -> FactoryCriteriaItemView:
        quality_head = self.store.get_item_stage_head(
            binding.item_id,
            FactoryItemStageV2.ITEM_QUALITY,
        )
        try:
            stage_head = self.store.get_item_stage_head(
                binding.item_id,
                FactoryItemStageV2.CRITERIA_RUBRIC,
            )
        except FactoryControlNotFoundError:
            if supervised is None:
                raise FactoryCriteriaItemRuntimeError(
                    "criteria execution result is unavailable",
                ) from None
            authority = self._build_authority(
                supervised,
                task_draft=task_draft,
                task_contract_set=task_contract_set,
                quality=quality,
                quality_context=quality_context,
                audit=audit,
            )
            authority_ref = self.authority_materials.put(authority)
            outcome, reasons = _stage_outcome(supervised.result)
            stage_head = FactoryItemStageHeadV2.create(
                item_binding_ref=binding.to_ref(),
                item_run_ref=binding.item_run_ref,
                stage=FactoryItemStageV2.CRITERIA_RUBRIC,
                stage_version=1,
                predecessor_head_ref=None,
                dependency_result_refs=(quality_head.result_ref,),
                result_ref=supervised.result.to_ref(),
                outcome=outcome,
                reason_codes=reasons,
                audit=audit,
            )
            stage_head = self.store.commit_item_stage_head(
                stage_head,
                idempotency_key=(f"commit-factory-criteria-head-{stage_head.object_sha256}"),
                material_ref=authority_ref,
            )
        authority_ref = self.store.get_item_stage_material_ref(stage_head.to_ref())
        authority = self.authority_materials.get(authority_ref)
        self._validate_current(
            binding=binding,
            quality_head=quality_head,
            stage_head=stage_head,
            authority=authority,
            plan=plan,
            task_draft=task_draft,
            quality=quality,
            task_contract_set=task_contract_set,
            quality_context=quality_context,
            tool_catalog=tool_catalog,
            audit=audit,
        )
        return FactoryCriteriaItemView(
            item_id=binding.item_id,
            plan_ref=plan.to_ref(),
            compiled_plan_ref=compiled.to_ref(),
            review_ref=review_ref,
            stage_head_ref=stage_head.to_ref(),
            material_ref=authority_ref,
            authority=authority,
            state=FactoryCriteriaItemState.EXECUTED,
        )

    def _build_authority(
        self,
        supervised: CriteriaRubricSupervisedExecution,
        *,
        task_draft: TaskDraftV2,
        task_contract_set: R4TaskContractSetV2,
        quality: FactoryAttachmentQualityView,
        quality_context: FactoryAttachmentQualityContext,
        audit: ContractAudit,
    ) -> FactoryCriteriaAuthorityResult:
        result = supervised.result
        route = self.runner.agent.gateway.get_route(result.route_decision_ref)
        values: tuple[
            RubricSetV2 | None,
            EvaluatorSpecV2 | None,
            ReferencePolicyV2 | None,
            ToolPolicyV2 | None,
            ContestantToolPolicyV2 | None,
        ]
        if result.outcome is CriteriaRubricOutcomeV2.SUCCEEDED:
            if (
                result.rubric_set_ref is None
                or result.evaluator_spec_ref is None
                or result.reference_policy_ref is None
                or result.tool_policy_ref is None
                or result.contestant_tool_policy_ref is None
            ):
                raise FactoryCriteriaItemRuntimeError("successful criteria result has no complete chain")
            values = (
                self.criteria_materials.get_model(
                    result.rubric_set_ref,
                    RubricSetV2,
                ),
                self.criteria_materials.get_model(
                    result.evaluator_spec_ref,
                    EvaluatorSpecV2,
                ),
                self.criteria_materials.get_model(
                    result.reference_policy_ref,
                    ReferencePolicyV2,
                ),
                self.criteria_materials.get_model(
                    result.tool_policy_ref,
                    ToolPolicyV2,
                ),
                self.criteria_materials.get_model(
                    result.contestant_tool_policy_ref,
                    ContestantToolPolicyV2,
                ),
            )
        else:
            values = (None, None, None, None, None)
        execution = CriteriaRubricExecution(
            result=result,
            route=route,
            invocation_result_ref=(result.gateway_invocation_result_ref),
            rubric_set=values[0],
            evaluator_spec=values[1],
            reference_policy=values[2],
            tool_policy=values[3],
            contestant_tool_policy=values[4],
        )
        reviewed_contract_set: R4TaskContractSetV2 | None = None
        reviewed_item_quality = None
        if result.outcome is CriteriaRubricOutcomeV2.SUCCEEDED:
            (
                rubric_set,
                evaluator_spec,
                reference_policy,
                tool_policy,
                contestant_tool_policy,
            ) = values
            if (
                rubric_set is None
                or evaluator_spec is None
                or reference_policy is None
                or tool_policy is None
                or contestant_tool_policy is None
            ):
                raise FactoryCriteriaItemRuntimeError("successful criteria result has no reviewed R4 chain")
            reviewed_contract_set = _reviewed_contract_set(
                base=task_contract_set,
                task_draft=task_draft,
                result=result,
                audit=audit,
            )
            finalization = quality.result.finalization
            reviewed_item_quality = ItemQualityCompiler().compile(
                task_contract_set=reviewed_contract_set,
                review_policy=quality_context.review_policy,
                semantic_workflow=finalization.semantic_workflow,
                candidate_revision=finalization.candidate_revision,
                deterministic_validation=(finalization.deterministic_validation),
                source_deterministic_validation=(finalization.source_validation),
                job_store=quality_context.job_store,
                job_id=quality_context.job_id,
                item_id=quality_context.item_id,
                audit=audit,
            )
        return FactoryCriteriaAuthorityResult(
            supervised=supervised,
            execution=execution,
            reviewed_task_contract_set=reviewed_contract_set,
            reviewed_item_quality=reviewed_item_quality,
        )

    def _validate_current(
        self,
        *,
        binding: FactoryItemRunBindingV2,
        quality_head: FactoryItemStageHeadV2,
        stage_head: FactoryItemStageHeadV2,
        authority: FactoryCriteriaAuthorityResult,
        plan: CriteriaRubricPlanV2,
        task_draft: TaskDraftV2,
        quality: FactoryAttachmentQualityView,
        task_contract_set: R4TaskContractSetV2,
        quality_context: FactoryAttachmentQualityContext,
        tool_catalog: ToolCapabilityCatalog,
        audit: ContractAudit,
    ) -> None:
        result = authority.execution.result
        current_domain = self.store.get_domain_result(
            self.store.get_item_run(binding).run_id,
            PlanKindV2.CRITERIA_RUBRIC,
        )
        if (
            authority.supervised.result != result
            or stage_head.item_binding_ref != binding.to_ref()
            or stage_head.dependency_result_refs != (quality_head.result_ref,)
            or stage_head.result_ref != result.to_ref()
            or current_domain.result_ref != result.to_ref()
            or result.plan_ref != plan.to_ref()
            or self.runner.agent.gateway.get_route(result.route_decision_ref) != authority.execution.route
        ):
            raise FactoryCriteriaItemRuntimeError("criteria item authority drifted")
        self.runner.agent.validate_sources(
            plan=plan,
            task_draft=task_draft,
            attachment_quality=(quality.result.finalization.attachment_quality),
            solvability=quality.result.solvability.assessment,
            tool_catalog=tool_catalog,
            audit=audit,
        )
        if result.outcome is CriteriaRubricOutcomeV2.SUCCEEDED:
            execution = authority.execution
            if (
                execution.rubric_set is None
                or execution.evaluator_spec is None
                or execution.reference_policy is None
                or execution.tool_policy is None
                or execution.contestant_tool_policy is None
                or rubric_set_ref(execution.rubric_set) != result.rubric_set_ref
                or evaluator_spec_ref(execution.evaluator_spec) != result.evaluator_spec_ref
                or reference_policy_ref(execution.reference_policy) != result.reference_policy_ref
                or tool_policy_ref(execution.tool_policy) != result.tool_policy_ref
                or contestant_tool_policy_ref(execution.contestant_tool_policy)
                != result.contestant_tool_policy_ref
                or authority.reviewed_task_contract_set is None
                or authority.reviewed_item_quality is None
            ):
                raise FactoryCriteriaItemRuntimeError("criteria private chain drifted")
            expected_contract_set = _reviewed_contract_set(
                base=task_contract_set,
                task_draft=task_draft,
                result=result,
                audit=authority.reviewed_task_contract_set.audit,
            )
            if canonical_value_v2(expected_contract_set) != canonical_value_v2(
                authority.reviewed_task_contract_set
            ):
                raise FactoryCriteriaItemRuntimeError("reviewed R4 contract authority drifted")
            finalization = quality.result.finalization
            ItemQualityCompiler().validate_current(
                authority.reviewed_item_quality,
                task_contract_set=(authority.reviewed_task_contract_set),
                review_policy=quality_context.review_policy,
                semantic_workflow=finalization.semantic_workflow,
                candidate_revision=finalization.candidate_revision,
                deterministic_validation=(finalization.deterministic_validation),
                source_deterministic_validation=(finalization.source_validation),
                job_store=quality_context.job_store,
                job_id=quality_context.job_id,
                item_id=quality_context.item_id,
            )
        elif authority.reviewed_task_contract_set is not None or authority.reviewed_item_quality is not None:
            raise FactoryCriteriaItemRuntimeError("non-success criteria retained reviewed authority")


def _stage_outcome(
    result: CriteriaRubricResultV2,
) -> tuple[
    FactoryItemStageOutcomeV2,
    tuple[str, ...],
]:
    if result.outcome is CriteriaRubricOutcomeV2.SUCCEEDED:
        return FactoryItemStageOutcomeV2.SUCCEEDED, ()
    if result.outcome is CriteriaRubricOutcomeV2.BLOCKED_CAPABILITY:
        return (
            FactoryItemStageOutcomeV2.BLOCKED_CAPABILITY,
            result.reason_codes,
        )
    if result.outcome is CriteriaRubricOutcomeV2.ABSTAINED:
        return FactoryItemStageOutcomeV2.FAILED, (result.reason_codes or ("CRITERIA_RUBRIC_ABSTAINED",))
    return (
        FactoryItemStageOutcomeV2.BLOCKED_POLICY,
        result.reason_codes,
    )


def _reviewed_contract_set(
    *,
    base: R4TaskContractSetV2,
    task_draft: TaskDraftV2,
    result: CriteriaRubricResultV2,
    audit: ContractAudit,
) -> R4TaskContractSetV2:
    reviewed_refs = (
        result.rubric_set_ref,
        result.evaluator_spec_ref,
        result.reference_policy_ref,
        result.tool_policy_ref,
        result.contestant_tool_policy_ref,
    )
    if (
        result.outcome is not CriteriaRubricOutcomeV2.SUCCEEDED
        or any(reference is None for reference in reviewed_refs)
        or base.task_draft_ref != task_draft_ref(task_draft)
    ):
        raise FactoryCriteriaItemRuntimeError("reviewed R4 contract inputs are incomplete")
    (
        rubric_ref,
        evaluator_ref,
        reference_ref,
        tool_ref,
        contestant_tool_ref,
    ) = reviewed_refs
    assert rubric_ref is not None
    assert evaluator_ref is not None
    assert reference_ref is not None
    assert tool_ref is not None
    assert contestant_tool_ref is not None
    refs = (
        base.task_draft_ref,
        base.task_prompt_safety_gate_ref,
        rubric_ref,
        evaluator_ref,
        reference_ref,
        tool_ref,
        contestant_tool_ref,
        base.producer_storage_authorization_ref,
        base.producer_task_view_ref,
    )
    value = base.model_copy(
        update={
            "contract_set_id": "r4-task-contract-set://pending",
            "rubric_set_ref": rubric_ref,
            "evaluator_spec_ref": evaluator_ref,
            "reference_policy_ref": reference_ref,
            "tool_policy_ref": tool_ref,
            "contestant_tool_policy_ref": contestant_tool_ref,
            "contract_set_sha256": "0" * 64,
            "audit": ContractAudit(
                created_at=audit.created_at,
                created_by=audit.created_by,
                governing_versions=audit.governing_versions,
                input_refs=tuple(
                    sorted(
                        refs,
                        key=lambda reference: (
                            reference.object_type,
                            reference.object_id,
                            reference.object_version,
                            reference.object_sha256,
                        ),
                    )
                ),
            ),
        }
    )
    digest = r4_task_contract_set_carried_sha256(value)
    return value.model_copy(
        update={
            "contract_set_id": (f"r4-task-contract-set://sha256/{digest}"),
            "contract_set_sha256": digest,
        }
    )


def _stable_ref(
    object_type: str,
    seed: str,
) -> ObjectRef:
    digest = hashlib.sha256(seed.encode()).hexdigest()
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://sha256/{digest}",
        object_version="v2",
        object_sha256=digest,
    )


__all__ = [
    "FactoryCriteriaItemRuntime",
    "FactoryCriteriaItemRuntimeError",
    "FactoryCriteriaItemState",
    "FactoryCriteriaItemView",
    "FactoryCriteriaPlanTemplateBuilder",
]
