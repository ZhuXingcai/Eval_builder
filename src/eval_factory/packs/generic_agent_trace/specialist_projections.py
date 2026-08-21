from __future__ import annotations

from eval_factory.agent_system.criteria_item_runtime import (
    FactoryCriteriaItemRuntime,
)
from eval_factory.agent_system.criteria_subgraph import (
    CriteriaRubricSupervisedExecution,
)
from eval_factory.agent_system.grading_item_runtime import (
    FactoryGradingItemRuntime,
)
from eval_factory.agent_system.grading_subgraph import (
    GradingDesignSupervisedExecution,
)
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.contracts.agent_system_v2 import (
    CompiledCriteriaRubricPlanV2,
    CompiledGradingDesignPlanV2,
    CriteriaRubricPlanV2,
    GradingDesignPlanV2,
    PlanKindV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryItemRunBindingV2,
)
from eval_factory.contracts.task_v2 import (
    evaluator_spec_ref,
    reference_policy_ref,
    rubric_set_ref,
    task_draft_ref,
    tool_policy_ref,
)
from eval_factory.harness.contracts import sorted_refs
from eval_factory.packs.generic_agent_trace.capability_contracts import (
    CriteriaRubricCapabilityRequestV1,
    GradingDesignCapabilityRequestV1,
)
from eval_factory.packs.generic_agent_trace.owner_refs import (
    generic_agent_tool_catalog_ref,
)
from eval_factory.packs.generic_agent_trace.plan_review_preparation import (
    GenericAgentItemReviewMaterialSource,
)


class GenericAgentSpecialistProjectionError(RuntimeError):
    pass


def _binding_for_run(
    store: FactoryControlStore,
    *,
    dataset_run_id: str,
    item_run_id: str,
) -> FactoryItemRunBindingV2:
    matches = tuple(
        binding
        for binding in store.list_item_bindings(dataset_run_id)
        if store.get_item_run(binding).run_id == item_run_id
    )
    if len(matches) != 1:
        raise GenericAgentSpecialistProjectionError(
            "specialist item binding is not unique",
        )
    return matches[0]


class GenericAgentCriteriaProjection:
    def __init__(
        self,
        *,
        dataset_run_id: str,
        store: FactoryControlStore,
        runtime: FactoryCriteriaItemRuntime,
        materials: GenericAgentItemReviewMaterialSource,
    ) -> None:
        self.dataset_run_id = dataset_run_id
        self.store = store
        self.runtime = runtime
        self.materials = materials

    def commit(
        self,
        *,
        request: CriteriaRubricCapabilityRequestV1,
        execution: CriteriaRubricSupervisedExecution,
        audit: ContractAudit,
    ) -> tuple[ObjectRef, ...]:
        binding = _binding_for_run(
            self.store,
            dataset_run_id=self.dataset_run_id,
            item_run_id=request.run_id,
        )
        source = self.materials.criteria(binding.item_id)
        if (
            task_draft_ref(source.task_draft) != request.task_draft_ref
            or source.quality.result.finalization.attachment_quality.to_ref()
            != request.attachment_quality_ref
            or source.quality.result.solvability.assessment.to_ref() != request.solvability_ref
            or generic_agent_tool_catalog_ref(
                source.tool_catalog,
            )
            != request.tool_catalog_ref
        ):
            raise GenericAgentSpecialistProjectionError(
                "Criteria request differs from owner material",
            )
        plan_material = self.store.get_domain_plan(
            request.run_id,
            PlanKindV2.CRITERIA_RUBRIC,
        )
        plan = CriteriaRubricPlanV2.model_validate_json(
            plan_material.plan_record_json,
        )
        compiled = CompiledCriteriaRubricPlanV2.model_validate_json(
            plan_material.compiled_plan_record_json,
        )
        review = self.runtime.plan_reviews.require_resumed_plan(
            run_id=request.run_id,
            plan_kind=PlanKindV2.CRITERIA_RUBRIC,
            plan_ref=plan.to_ref(),
        )
        view = self.runtime.commit_execution(
            binding=binding,
            plan=plan,
            compiled=compiled,
            review_ref=review.request.to_ref(),
            supervised=execution,
            task_draft=source.task_draft,
            task_contract_set=source.task_contract_set,
            quality=source.quality,
            quality_context=source.quality_context,
            tool_catalog=source.tool_catalog,
            audit=audit,
        )
        if view.stage_head_ref is None or view.material_ref is None:
            raise GenericAgentSpecialistProjectionError(
                "Criteria projection has no committed authority",
            )
        return sorted_refs((view.stage_head_ref, view.material_ref))


class GenericAgentGradingProjection:
    def __init__(
        self,
        *,
        dataset_run_id: str,
        store: FactoryControlStore,
        runtime: FactoryGradingItemRuntime,
        materials: GenericAgentItemReviewMaterialSource,
    ) -> None:
        self.dataset_run_id = dataset_run_id
        self.store = store
        self.runtime = runtime
        self.materials = materials

    def commit(
        self,
        *,
        request: GradingDesignCapabilityRequestV1,
        execution: GradingDesignSupervisedExecution,
        audit: ContractAudit,
    ) -> tuple[ObjectRef, ...]:
        binding = _binding_for_run(
            self.store,
            dataset_run_id=self.dataset_run_id,
            item_run_id=request.run_id,
        )
        source = self.materials.grading(binding.item_id)
        criteria_authority = source.criteria.authority
        if criteria_authority is None:
            raise GenericAgentSpecialistProjectionError(
                "Grading projection requires Criteria authority",
            )
        criteria = criteria_authority.execution
        if (
            criteria.rubric_set is None
            or criteria.evaluator_spec is None
            or criteria.reference_policy is None
            or criteria.tool_policy is None
            or criteria.result.to_ref() != request.criteria_result_ref
            or criteria.route.to_ref() != request.criteria_route_ref
            or rubric_set_ref(criteria.rubric_set) != request.rubric_set_ref
            or evaluator_spec_ref(criteria.evaluator_spec) != request.evaluator_spec_ref
            or reference_policy_ref(criteria.reference_policy) != request.reference_policy_ref
            or tool_policy_ref(criteria.tool_policy) != request.tool_policy_ref
        ):
            raise GenericAgentSpecialistProjectionError(
                "Grading request differs from Criteria authority",
            )
        plan_material = self.store.get_domain_plan(
            request.run_id,
            PlanKindV2.GRADING_DESIGN,
        )
        plan = GradingDesignPlanV2.model_validate_json(
            plan_material.plan_record_json,
        )
        compiled = CompiledGradingDesignPlanV2.model_validate_json(
            plan_material.compiled_plan_record_json,
        )
        review = self.runtime.plan_reviews.require_resumed_plan(
            run_id=request.run_id,
            plan_kind=PlanKindV2.GRADING_DESIGN,
            plan_ref=plan.to_ref(),
        )
        view = self.runtime.commit_execution(
            binding=binding,
            plan=plan,
            compiled=compiled,
            review_ref=review.request.to_ref(),
            supervised=execution,
            criteria=source.criteria,
            audit=audit,
        )
        if view.stage_head_ref is None or view.material_ref is None:
            raise GenericAgentSpecialistProjectionError(
                "Grading projection has no committed authority",
            )
        return sorted_refs((view.stage_head_ref, view.material_ref))


__all__ = [
    "GenericAgentCriteriaProjection",
    "GenericAgentGradingProjection",
    "GenericAgentSpecialistProjectionError",
]
