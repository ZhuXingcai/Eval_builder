from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from eval_factory.agent_system.criteria_item_runtime import (
    FactoryCriteriaItemView,
)
from eval_factory.agent_system.grading_agent import (
    GatewayGradingDesignPlanningAgent,
    GradingDesignExecution,
)
from eval_factory.agent_system.grading_authority_material import (
    FactoryGradingAuthorityMaterialStore,
    FactoryGradingAuthorityResult,
)
from eval_factory.agent_system.grading_material_store import (
    GradingDesignMaterialStore,
)
from eval_factory.agent_system.grading_subgraph import (
    GradingDesignSupervisedExecution,
    SupervisedGradingDesignRunner,
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
from eval_factory.contracts.agent_system_v2 import (
    CompiledGradingDesignPlanV2,
    FactoryRunPolicyV2,
    GradingDesignOutcomeV2,
    GradingDesignPlanV2,
    GradingDesignResultV2,
    JudgeAggregationModeV2,
    JudgeDesignSpecV2,
    JudgeDesignValidationV2,
    JudgeTaskMappingV2,
    PlanKindV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryItemRunBindingV2,
    FactoryItemStageHeadV2,
    FactoryItemStageOutcomeV2,
    FactoryItemStageV2,
)
from eval_factory.contracts.task_v2 import (
    evaluator_spec_ref,
    reference_policy_ref,
    rubric_set_ref,
    tool_policy_ref,
)
from eval_factory.task_authoring import (
    VerifiedModelDomainAuthorization,
)


class FactoryGradingItemRuntimeError(RuntimeError):
    pass


class FactoryGradingItemState(StrEnum):
    WAITING_REVIEW = "WAITING_REVIEW"
    READY_TO_EXECUTE = "READY_TO_EXECUTE"
    EXECUTED = "EXECUTED"


@dataclass(frozen=True, slots=True)
class FactoryGradingPlanTemplateConfig:
    allowed_judge_model_profile_refs: tuple[ObjectRef, ...]
    judge_input_schema_ref: ObjectRef
    judge_output_schema_ref: ObjectRef
    passing_score_basis_points: int = 8_000
    minimum_confidence_basis_points: int = 8_000
    escalate_on_reference_unavailable: bool = True


@dataclass(frozen=True, slots=True)
class FactoryGradingItemView:
    item_id: str
    plan_ref: ObjectRef
    compiled_plan_ref: ObjectRef
    review_ref: ObjectRef
    stage_head_ref: ObjectRef | None
    material_ref: ObjectRef | None
    authority: FactoryGradingAuthorityResult | None
    state: FactoryGradingItemState


class FactoryGradingPlanTemplateBuilder:
    def __init__(
        self,
        registry: AgentRegistry,
        config: FactoryGradingPlanTemplateConfig,
    ) -> None:
        self.registry = registry
        self.config = config

    def build(
        self,
        *,
        binding: FactoryItemRunBindingV2,
        run_ref: ObjectRef,
        criteria: FactoryCriteriaItemView,
        policy: FactoryRunPolicyV2,
        audit: ContractAudit,
    ) -> GradingDesignPlanV2:
        authority = criteria.authority
        if authority is None:
            raise FactoryGradingItemRuntimeError("grading planning requires complete criteria authority")
        criteria_execution = authority.execution
        rubric_set = criteria_execution.rubric_set
        evaluator_spec = criteria_execution.evaluator_spec
        reference_policy = criteria_execution.reference_policy
        tool_policy = criteria_execution.tool_policy
        if rubric_set is None or evaluator_spec is None or reference_policy is None or tool_policy is None:
            raise FactoryGradingItemRuntimeError("grading planning requires complete criteria authority")
        criteria_result = criteria_execution.result
        generator_ref = criteria_execution.route.selected_model_profile_ref
        definition = self.registry.resolve(
            "grading-design-agent",
            "grading-design",
        )
        capabilities = self.registry.capabilities_for(definition)
        groups: dict[str, list[tuple[str, int]]] = {}
        for criterion in rubric_set.criteria:
            groups.setdefault(
                criterion.evaluator_binding,
                [],
            ).append(
                (
                    criterion.criterion_id,
                    int(criterion.weight * 10_000),
                )
            )
        if not groups:
            raise FactoryGradingItemRuntimeError("grading planning requires rubric criteria")
        grouped = tuple(sorted(groups.items()))
        raw_weights = tuple(sum(weight for _, weight in values) for _, values in grouped)
        task_weights = list(raw_weights)
        task_weights[-1] += 10_000 - sum(task_weights)
        judge_tasks = tuple(
            JudgeTaskMappingV2(
                task_key=(f"judge-task://sha256/{hashlib.sha256(binding_id.encode()).hexdigest()}"),
                criterion_ids=tuple(sorted(criterion_id for criterion_id, _ in values)),
                evaluator_binding_id=binding_id,
                score_weight_basis_points=task_weights[index],
            )
            for index, (binding_id, values) in enumerate(grouped)
        )
        return GradingDesignPlanV2.create(
            plan_id=(f"grading-design-plan://factory-item/{binding.object_sha256}"),
            run_ref=run_ref,
            plan_version=1,
            predecessor_plan_ref=None,
            criteria_rubric_result_ref=(criteria_result.to_ref()),
            rubric_set_ref=rubric_set_ref(rubric_set),
            evaluator_spec_ref=evaluator_spec_ref(evaluator_spec),
            reference_policy_ref=reference_policy_ref(reference_policy),
            tool_policy_ref=tool_policy_ref(tool_policy),
            generator_model_profile_ref=generator_ref,
            judge_tasks=judge_tasks,
            aggregation_mode=JudgeAggregationModeV2.WEIGHTED_SUM,
            passing_score_basis_points=(self.config.passing_score_basis_points),
            minimum_confidence_basis_points=(self.config.minimum_confidence_basis_points),
            escalate_on_reference_unavailable=(self.config.escalate_on_reference_unavailable),
            judge_input_schema_ref=(self.config.judge_input_schema_ref),
            judge_output_schema_ref=(self.config.judge_output_schema_ref),
            required_output_fields=(
                "ABSTAIN",
                "EVIDENCE_IDS",
                "FAILURE_CLASS",
                "SCORE_BASIS_POINTS",
            ),
            allowed_judge_model_profile_refs=(self.config.allowed_judge_model_profile_refs),
            judge_prompt_template_ref=(definition.prompt_template_ref),
            agent_role=definition.agent_role,
            required_capability_ids=tuple(sorted(value.capability_id for value in capabilities)),
            specialist_tool_ids=definition.tool_ids,
            data_classifications=(definition.allowed_data_classifications),
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


class FactoryGradingItemRuntime:
    def __init__(
        self,
        *,
        store: FactoryControlStore,
        plan_reviews: PlanReviewService,
        planning_agent: GatewayGradingDesignPlanningAgent,
        runner: SupervisedGradingDesignRunner,
        template_builder: FactoryGradingPlanTemplateBuilder,
        grading_materials: GradingDesignMaterialStore,
        authority_materials: (FactoryGradingAuthorityMaterialStore),
        requested_by: str,
    ) -> None:
        self.store = store
        self.plan_reviews = plan_reviews
        self.planning_agent = planning_agent
        self.runner = runner
        self.template_builder = template_builder
        self.grading_materials = grading_materials
        self.authority_materials = authority_materials
        self.requested_by = requested_by

    async def advance(
        self,
        *,
        binding: FactoryItemRunBindingV2,
        policy: FactoryRunPolicyV2,
        criteria: FactoryCriteriaItemView,
        model_authorizations: tuple[
            VerifiedModelDomainAuthorization,
            ...,
        ],
        evaluated_at: datetime,
        audit: ContractAudit,
    ) -> FactoryGradingItemView:
        review = await self.prepare_review(
            binding=binding,
            policy=policy,
            criteria=criteria,
            model_authorizations=model_authorizations,
            evaluated_at=evaluated_at,
            audit=audit,
        )
        if review.state is FactoryGradingItemState.READY_TO_EXECUTE:
            return await self.execute_reviewed(
                binding=binding,
                policy=policy,
                criteria=criteria,
                model_authorizations=model_authorizations,
                evaluated_at=evaluated_at,
                audit=audit,
            )
        return review

    async def prepare_review(
        self,
        *,
        binding: FactoryItemRunBindingV2,
        policy: FactoryRunPolicyV2,
        criteria: FactoryCriteriaItemView,
        model_authorizations: tuple[
            VerifiedModelDomainAuthorization,
            ...,
        ],
        evaluated_at: datetime,
        audit: ContractAudit,
    ) -> FactoryGradingItemView:
        del model_authorizations, evaluated_at
        if (
            criteria.state.value != "EXECUTED"
            or criteria.stage_head_ref is None
            or criteria.material_ref is None
            or criteria.authority is None
        ):
            raise FactoryGradingItemRuntimeError("grading requires current executed criteria")
        criteria_head = self.store.get_item_stage_head(
            binding.item_id,
            FactoryItemStageV2.CRITERIA_RUBRIC,
        )
        if (
            criteria_head.to_ref() != criteria.stage_head_ref
            or criteria_head.outcome is not FactoryItemStageOutcomeV2.SUCCEEDED
        ):
            raise FactoryGradingItemRuntimeError("grading requires successful criteria authority")
        item_run = self.store.get_item_run(binding)
        if item_run.policy_ref != policy.to_ref():
            raise FactoryGradingItemRuntimeError("grading item run policy authority changed")
        try:
            material = self.store.get_domain_plan(
                item_run.run_id,
                PlanKindV2.GRADING_DESIGN,
            )
        except FactoryControlNotFoundError:
            template = self.template_builder.build(
                binding=binding,
                run_ref=item_run.to_ref(),
                criteria=criteria,
                policy=policy,
                audit=audit,
            )
            planning = await self.planning_agent.propose(
                task_ref=_stable_ref(
                    "agent-task",
                    f"{binding.object_id}:grading-planning",
                ),
                plan_template=template,
                policy=policy,
                audit=audit,
            )
            self.store.commit_domain_plan(
                run_id=item_run.run_id,
                expected_run_version=item_run.run_version,
                plan_kind=PlanKindV2.GRADING_DESIGN,
                plan=planning.plan,
                compiled_plan=planning.compiled_plan,
                audit=audit,
                idempotency_key=(f"commit-factory-grading-plan-{planning.plan.object_sha256}"),
                material_ref=criteria.material_ref,
            )
            plan = planning.plan
            compiled = planning.compiled_plan
        else:
            plan = GradingDesignPlanV2.model_validate_json(material.plan_record_json)
            compiled = CompiledGradingDesignPlanV2.model_validate_json(material.compiled_plan_record_json)
            if (
                self.store.get_domain_plan_material_ref(
                    run_id=item_run.run_id,
                    plan_kind=PlanKindV2.GRADING_DESIGN,
                )
                != criteria.material_ref
            ):
                raise FactoryGradingItemRuntimeError("grading plan criteria material drifted")
        try:
            review = self.plan_reviews.require_resumed_plan(
                run_id=item_run.run_id,
                plan_kind=PlanKindV2.GRADING_DESIGN,
                plan_ref=plan.to_ref(),
            )
        except PlanReviewError:
            review = self.plan_reviews.open_plan(
                run_id=item_run.run_id,
                plan_kind=PlanKindV2.GRADING_DESIGN,
                requested_by=self.requested_by,
                idempotency_key=(f"open-factory-grading-review-{plan.object_sha256}"),
                audit=audit,
            )
            return FactoryGradingItemView(
                item_id=binding.item_id,
                plan_ref=plan.to_ref(),
                compiled_plan_ref=compiled.to_ref(),
                review_ref=review.request.to_ref(),
                stage_head_ref=None,
                material_ref=None,
                authority=None,
                state=FactoryGradingItemState.WAITING_REVIEW,
            )
        return FactoryGradingItemView(
            item_id=binding.item_id,
            plan_ref=plan.to_ref(),
            compiled_plan_ref=compiled.to_ref(),
            review_ref=review.request.to_ref(),
            stage_head_ref=None,
            material_ref=None,
            authority=None,
            state=FactoryGradingItemState.READY_TO_EXECUTE,
        )

    async def execute_reviewed(
        self,
        *,
        binding: FactoryItemRunBindingV2,
        policy: FactoryRunPolicyV2,
        criteria: FactoryCriteriaItemView,
        model_authorizations: tuple[
            VerifiedModelDomainAuthorization,
            ...,
        ],
        evaluated_at: datetime,
        audit: ContractAudit,
    ) -> FactoryGradingItemView:
        if (
            criteria.state.value != "EXECUTED"
            or criteria.stage_head_ref is None
            or criteria.material_ref is None
            or criteria.authority is None
        ):
            raise FactoryGradingItemRuntimeError(
                "grading requires current executed criteria",
            )
        criteria_head = self.store.get_item_stage_head(
            binding.item_id,
            FactoryItemStageV2.CRITERIA_RUBRIC,
        )
        if (
            criteria_head.to_ref() != criteria.stage_head_ref
            or criteria_head.outcome is not FactoryItemStageOutcomeV2.SUCCEEDED
        ):
            raise FactoryGradingItemRuntimeError(
                "grading requires successful criteria authority",
            )
        item_run = self.store.get_item_run(binding)
        if item_run.policy_ref != policy.to_ref():
            raise FactoryGradingItemRuntimeError(
                "grading item run policy authority changed",
            )
        material = self.store.get_domain_plan(
            item_run.run_id,
            PlanKindV2.GRADING_DESIGN,
        )
        plan = GradingDesignPlanV2.model_validate_json(
            material.plan_record_json,
        )
        compiled = CompiledGradingDesignPlanV2.model_validate_json(
            material.compiled_plan_record_json,
        )
        if (
            self.store.get_domain_plan_material_ref(
                run_id=item_run.run_id,
                plan_kind=PlanKindV2.GRADING_DESIGN,
            )
            != criteria.material_ref
        ):
            raise FactoryGradingItemRuntimeError(
                "grading plan criteria material drifted",
            )
        review = self.plan_reviews.require_resumed_plan(
            run_id=item_run.run_id,
            plan_kind=PlanKindV2.GRADING_DESIGN,
            plan_ref=plan.to_ref(),
        )
        supervised: GradingDesignSupervisedExecution | None
        try:
            self.store.get_item_stage_head(
                binding.item_id,
                FactoryItemStageV2.GRADING_DESIGN,
            )
        except FactoryControlNotFoundError:
            execution = criteria.authority.execution
            rubric_set = execution.rubric_set
            evaluator_spec = execution.evaluator_spec
            reference_policy = execution.reference_policy
            tool_policy = execution.tool_policy
            if (
                rubric_set is None
                or evaluator_spec is None
                or reference_policy is None
                or tool_policy is None
            ):
                raise FactoryGradingItemRuntimeError("grading criteria private chain is incomplete") from None
            supervised = await self.runner.run(
                run_id=item_run.run_id,
                plan=plan,
                compiled_plan=compiled,
                criteria_result=execution.result,
                criteria_route=execution.route,
                rubric_set=rubric_set,
                evaluator_spec=evaluator_spec,
                reference_policy=reference_policy,
                tool_policy=tool_policy,
                model_authorizations=model_authorizations,
                evaluated_at=evaluated_at,
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
            criteria=criteria,
            audit=audit,
        )

    def commit_execution(
        self,
        *,
        binding: FactoryItemRunBindingV2,
        plan: GradingDesignPlanV2,
        compiled: CompiledGradingDesignPlanV2,
        review_ref: ObjectRef,
        supervised: GradingDesignSupervisedExecution | None,
        criteria: FactoryCriteriaItemView,
        audit: ContractAudit,
    ) -> FactoryGradingItemView:
        criteria_head = self.store.get_item_stage_head(
            binding.item_id,
            FactoryItemStageV2.CRITERIA_RUBRIC,
        )
        try:
            stage_head = self.store.get_item_stage_head(
                binding.item_id,
                FactoryItemStageV2.GRADING_DESIGN,
            )
        except FactoryControlNotFoundError:
            if supervised is None:
                raise FactoryGradingItemRuntimeError(
                    "grading execution result is unavailable",
                ) from None
            authority = self._build_authority(supervised)
            authority_ref = self.authority_materials.put(authority)
            outcome, reasons = _stage_outcome(supervised.result)
            stage_head = FactoryItemStageHeadV2.create(
                item_binding_ref=binding.to_ref(),
                item_run_ref=binding.item_run_ref,
                stage=FactoryItemStageV2.GRADING_DESIGN,
                stage_version=1,
                predecessor_head_ref=None,
                dependency_result_refs=(criteria_head.result_ref,),
                result_ref=supervised.result.to_ref(),
                outcome=outcome,
                reason_codes=reasons,
                audit=audit,
            )
            stage_head = self.store.commit_item_stage_head(
                stage_head,
                idempotency_key=(f"commit-factory-grading-head-{stage_head.object_sha256}"),
                material_ref=authority_ref,
            )
        authority_ref = self.store.get_item_stage_material_ref(stage_head.to_ref())
        authority = self.authority_materials.get(authority_ref)
        self._validate_current(
            binding=binding,
            criteria_head=criteria_head,
            stage_head=stage_head,
            authority=authority,
            plan=plan,
            criteria=criteria,
        )
        return FactoryGradingItemView(
            item_id=binding.item_id,
            plan_ref=plan.to_ref(),
            compiled_plan_ref=compiled.to_ref(),
            review_ref=review_ref,
            stage_head_ref=stage_head.to_ref(),
            material_ref=authority_ref,
            authority=authority,
            state=FactoryGradingItemState.EXECUTED,
        )

    def _build_authority(
        self,
        supervised: GradingDesignSupervisedExecution,
    ) -> FactoryGradingAuthorityResult:
        result = supervised.result
        route = (
            self.runner.agent.gateway.get_route(result.route_decision_ref)
            if result.route_decision_ref is not None
            else None
        )
        validation = self.grading_materials.get_model(
            result.validation_ref,
            JudgeDesignValidationV2,
        )
        design_spec = (
            self.grading_materials.get_model(
                result.judge_design_spec_ref,
                JudgeDesignSpecV2,
            )
            if result.judge_design_spec_ref is not None
            else None
        )
        execution = GradingDesignExecution(
            result=result,
            route=route,
            invocation_result_ref=(result.gateway_invocation_result_ref),
            design_spec=design_spec,
            validation=validation,
            reference_grants=(),
        )
        return FactoryGradingAuthorityResult(
            supervised=supervised,
            execution=execution,
        )

    def _validate_current(
        self,
        *,
        binding: FactoryItemRunBindingV2,
        criteria_head: FactoryItemStageHeadV2,
        stage_head: FactoryItemStageHeadV2,
        authority: FactoryGradingAuthorityResult,
        plan: GradingDesignPlanV2,
        criteria: FactoryCriteriaItemView,
    ) -> None:
        result = authority.execution.result
        current_domain = self.store.get_domain_result(
            self.store.get_item_run(binding).run_id,
            PlanKindV2.GRADING_DESIGN,
        )
        if (
            authority.supervised.result != result
            or stage_head.item_binding_ref != binding.to_ref()
            or stage_head.dependency_result_refs != (criteria_head.result_ref,)
            or stage_head.result_ref != result.to_ref()
            or current_domain.result_ref != result.to_ref()
            or result.plan_ref != plan.to_ref()
            or result.criteria_rubric_result_ref != criteria_head.result_ref
            or authority.execution.validation.to_ref() != result.validation_ref
            or (
                authority.execution.design_spec.to_ref()
                if authority.execution.design_spec is not None
                else None
            )
            != result.judge_design_spec_ref
        ):
            raise FactoryGradingItemRuntimeError("grading item authority drifted")
        if criteria.authority is None:
            raise FactoryGradingItemRuntimeError("grading criteria authority is unavailable")
        criteria_execution = criteria.authority.execution
        if (
            criteria_execution.rubric_set is None
            or criteria_execution.evaluator_spec is None
            or criteria_execution.reference_policy is None
            or criteria_execution.tool_policy is None
        ):
            raise FactoryGradingItemRuntimeError("grading criteria private chain drifted")
        self.runner.agent.validate_sources(
            plan=plan,
            criteria_result=criteria_execution.result,
            criteria_route=criteria_execution.route,
            rubric_set=criteria_execution.rubric_set,
            evaluator_spec=criteria_execution.evaluator_spec,
            reference_policy=criteria_execution.reference_policy,
            tool_policy=criteria_execution.tool_policy,
        )
        if result.route_decision_ref is not None and (
            authority.execution.route is None
            or self.runner.agent.gateway.get_route(result.route_decision_ref) != authority.execution.route
        ):
            raise FactoryGradingItemRuntimeError("grading route authority drifted")


def _stage_outcome(
    result: GradingDesignResultV2,
) -> tuple[
    FactoryItemStageOutcomeV2,
    tuple[str, ...],
]:
    if result.outcome is GradingDesignOutcomeV2.SUCCEEDED:
        return FactoryItemStageOutcomeV2.SUCCEEDED, ()
    if result.outcome is GradingDesignOutcomeV2.BLOCKED_CAPABILITY:
        return (
            FactoryItemStageOutcomeV2.BLOCKED_CAPABILITY,
            result.reason_codes,
        )
    if result.outcome is GradingDesignOutcomeV2.ABSTAINED:
        return FactoryItemStageOutcomeV2.FAILED, (result.reason_codes or ("GRADING_DESIGN_ABSTAINED",))
    return (
        FactoryItemStageOutcomeV2.BLOCKED_POLICY,
        result.reason_codes,
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
    "FactoryGradingItemRuntime",
    "FactoryGradingItemRuntimeError",
    "FactoryGradingItemState",
    "FactoryGradingItemView",
    "FactoryGradingPlanTemplateBuilder",
    "FactoryGradingPlanTemplateConfig",
]
