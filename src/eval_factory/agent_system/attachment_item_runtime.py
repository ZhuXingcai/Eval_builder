from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

from env_mock_agent.facade import (
    AttachmentExecutionFacade,
    AttachmentRoutingFacade,
)
from eval_factory.agent_system.attachment_planner import (
    GatewayAttachmentPlanningAgent,
)
from eval_factory.agent_system.attachment_preparation_builder import (
    FactoryAttachmentPreparationBuilder,
)
from eval_factory.agent_system.attachment_preparation_material import (
    FactoryAttachmentPreparationMaterialStore,
)
from eval_factory.agent_system.attachment_subgraph import (
    AttachmentSupervisedExecution,
    SupervisedAttachmentR5Runner,
)
from eval_factory.agent_system.plan_review import (
    PlanReviewError,
    PlanReviewService,
)
from eval_factory.agent_system.store import (
    FactoryControlNotFoundError,
    FactoryControlStore,
)
from eval_factory.agent_system.task_authoring_bridge import (
    FactoryTaskAuthoringBridgeResult,
)
from eval_factory.contracts.agent_system_v2 import (
    AttachmentGenerationPlanV2,
    AttachmentMockWorkV2,
    AttachmentSubgraphOutcomeV2,
    CompiledAttachmentGenerationPlanV2,
    FactoryRunPolicyV2,
    FactoryRunStatusV2,
    PlanKindV2,
)
from eval_factory.contracts.attachment_v2 import (
    ArtifactExecutionGroupV2,
    artifact_execution_group_ref,
    attachment_planning_context_ref,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryItemRunBindingV2,
    FactoryItemStageHeadV2,
    FactoryItemStageOutcomeV2,
    FactoryItemStageV2,
)
from eval_factory.contracts.task_v2 import (
    producer_task_view_ref,
)
from eval_factory.provenance.bundles import evidence_bundle_ref


class FactoryAttachmentItemRuntimeError(RuntimeError):
    pass


class FactoryAttachmentItemState(StrEnum):
    WAITING_REVIEW = "WAITING_REVIEW"
    READY_TO_EXECUTE = "READY_TO_EXECUTE"
    EXECUTED = "EXECUTED"


@dataclass(frozen=True, slots=True)
class FactoryAttachmentItemView:
    item_id: str
    item_run_ref: ObjectRef
    plan_ref: ObjectRef
    compiled_plan_ref: ObjectRef
    review_ref: ObjectRef
    stage_head_ref: ObjectRef | None
    supervised_execution: AttachmentSupervisedExecution | None
    state: FactoryAttachmentItemState


class FactoryAttachmentItemRuntime:
    def __init__(
        self,
        *,
        store: FactoryControlStore,
        plan_reviews: PlanReviewService,
        planner: GatewayAttachmentPlanningAgent,
        preparation_builder: FactoryAttachmentPreparationBuilder,
        preparation_materials: (FactoryAttachmentPreparationMaterialStore),
        routing_facade: AttachmentRoutingFacade,
        execution_facade: AttachmentExecutionFacade,
        requested_by: str,
        runner: SupervisedAttachmentR5Runner | None = None,
    ) -> None:
        self.store = store
        self.plan_reviews = plan_reviews
        self.planner = planner
        self.preparation_builder = preparation_builder
        self.preparation_materials = preparation_materials
        self.routing_facade = routing_facade
        self.execution_facade = execution_facade
        self.requested_by = requested_by
        self.runner = runner

    async def advance(
        self,
        *,
        binding: FactoryItemRunBindingV2,
        task_authoring: FactoryTaskAuthoringBridgeResult,
        policy: FactoryRunPolicyV2,
        audit: ContractAudit,
    ) -> FactoryAttachmentItemView:
        review = await self.prepare_review(
            binding=binding,
            task_authoring=task_authoring,
            policy=policy,
            audit=audit,
        )
        if review.state is FactoryAttachmentItemState.READY_TO_EXECUTE and self.runner is not None:
            return await self.execute_reviewed(
                binding=binding,
                policy=policy,
                audit=audit,
            )
        return review

    async def prepare_review(
        self,
        *,
        binding: FactoryItemRunBindingV2,
        task_authoring: FactoryTaskAuthoringBridgeResult,
        policy: FactoryRunPolicyV2,
        audit: ContractAudit,
    ) -> FactoryAttachmentItemView:
        item_run = self.store.get_item_run(binding)
        if item_run.policy_ref != policy.to_ref():
            raise FactoryAttachmentItemRuntimeError("item run policy authority changed")
        if item_run.status is FactoryRunStatusV2.CREATED:
            item_run = self.store.begin_item_planning(
                binding,
                idempotency_key=(f"begin-factory-item-planning-{binding.object_sha256}"),
                audit=audit,
            )
        try:
            material = self.store.get_domain_plan(
                item_run.run_id,
                PlanKindV2.ATTACHMENT_GENERATION,
            )
        except FactoryControlNotFoundError:
            preparation = await self.preparation_builder.build(
                task_authoring=task_authoring,
                routing_facade=self.routing_facade,
                execution_facade=self.execution_facade,
                audit=audit,
            )
            material_ref = self.preparation_materials.put(preparation)
            execution_plan = preparation.execution_result.execution_plan
            if execution_plan is None:
                raise FactoryAttachmentItemRuntimeError(
                    "attachment preparation has no execution plan"
                ) from None
            planning = await self.planner.propose(
                task_ref=_stable_ref(
                    "agent-task",
                    f"{binding.object_id}:attachment-planning",
                ),
                run_ref=item_run.to_ref(),
                producer_task_view_ref=producer_task_view_ref(preparation.producer_task_view),
                evidence_bundle_ref=evidence_bundle_ref(preparation.evidence_bundle),
                attachment_planning_context_ref=(
                    attachment_planning_context_ref(preparation.attachment_planning_context)
                ),
                work_templates=self._work_templates(
                    execution_plan.groups,
                ),
                quality_policy_ref=_stable_ref(
                    "attachment-quality-policy",
                    f"{binding.object_id}:quality",
                ),
                solvability_policy_ref=_stable_ref(
                    "solvability-policy",
                    f"{binding.object_id}:solvability",
                ),
                policy=policy,
                audit=audit,
            )
            self.store.commit_domain_plan(
                run_id=item_run.run_id,
                expected_run_version=item_run.run_version,
                plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
                plan=planning.plan,
                compiled_plan=planning.compiled_plan,
                audit=audit,
                idempotency_key=(f"commit-factory-attachment-plan-{planning.plan.object_sha256}"),
                material_ref=material_ref,
            )
            plan = planning.plan
            compiled = planning.compiled_plan
        else:
            plan = AttachmentGenerationPlanV2.model_validate_json(material.plan_record_json)
            compiled = CompiledAttachmentGenerationPlanV2.model_validate_json(
                material.compiled_plan_record_json
            )
            material_ref = self.store.get_domain_plan_material_ref(
                run_id=item_run.run_id,
                plan_kind=(PlanKindV2.ATTACHMENT_GENERATION),
            )
            preparation = self.preparation_materials.get(material_ref)
        try:
            review = self.plan_reviews.require_resumed_plan(
                run_id=item_run.run_id,
                plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
                plan_ref=plan.to_ref(),
            )
            state = FactoryAttachmentItemState.READY_TO_EXECUTE
        except PlanReviewError:
            review = self.plan_reviews.open_plan(
                run_id=item_run.run_id,
                plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
                requested_by=self.requested_by,
                idempotency_key=(f"open-factory-attachment-review-{plan.object_sha256}"),
                audit=audit,
            )
            state = FactoryAttachmentItemState.WAITING_REVIEW
        return FactoryAttachmentItemView(
            item_id=binding.item_id,
            item_run_ref=self.store.get_run(item_run.run_id).to_ref(),
            plan_ref=plan.to_ref(),
            compiled_plan_ref=compiled.to_ref(),
            review_ref=review.request.to_ref(),
            stage_head_ref=None,
            supervised_execution=None,
            state=state,
        )

    async def execute_reviewed(
        self,
        *,
        binding: FactoryItemRunBindingV2,
        policy: FactoryRunPolicyV2,
        audit: ContractAudit,
    ) -> FactoryAttachmentItemView:
        if self.runner is None:
            raise FactoryAttachmentItemRuntimeError(
                "attachment execution capability is unavailable",
            )
        item_run = self.store.get_item_run(binding)
        if item_run.policy_ref != policy.to_ref():
            raise FactoryAttachmentItemRuntimeError(
                "item run policy authority changed",
            )
        material = self.store.get_domain_plan(
            item_run.run_id,
            PlanKindV2.ATTACHMENT_GENERATION,
        )
        plan = AttachmentGenerationPlanV2.model_validate_json(
            material.plan_record_json,
        )
        compiled = CompiledAttachmentGenerationPlanV2.model_validate_json(
            material.compiled_plan_record_json,
        )
        material_ref = self.store.get_domain_plan_material_ref(
            run_id=item_run.run_id,
            plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
        )
        preparation = self.preparation_materials.get(material_ref)
        review = self.plan_reviews.require_resumed_plan(
            run_id=item_run.run_id,
            plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
            plan_ref=plan.to_ref(),
        )
        supervised_execution = await self.runner.run(
            run_id=item_run.run_id,
            plan=plan,
            compiled_plan=compiled,
            preparation=preparation,
            facade=self.execution_facade,
            audit=audit,
        )
        task_head = self.store.get_item_stage_head(
            binding.item_id,
            FactoryItemStageV2.TASK_AUTHORING,
        )
        outcome, reason_codes = _stage_outcome(
            supervised_execution.subgraph_result.outcome,
            supervised_execution.subgraph_result.reason_codes,
        )
        stage_head = FactoryItemStageHeadV2.create(
            item_binding_ref=binding.to_ref(),
            item_run_ref=binding.item_run_ref,
            stage=FactoryItemStageV2.ATTACHMENT,
            stage_version=1,
            predecessor_head_ref=None,
            dependency_result_refs=(task_head.result_ref,),
            result_ref=(supervised_execution.subgraph_result.to_ref()),
            outcome=outcome,
            reason_codes=reason_codes,
            audit=audit,
        )
        stage_head = self.store.commit_item_stage_head(
            stage_head,
            idempotency_key=(f"commit-factory-attachment-head-{stage_head.object_sha256}"),
        )
        return FactoryAttachmentItemView(
            item_id=binding.item_id,
            item_run_ref=self.store.get_run(item_run.run_id).to_ref(),
            plan_ref=plan.to_ref(),
            compiled_plan_ref=compiled.to_ref(),
            review_ref=review.request.to_ref(),
            stage_head_ref=stage_head.to_ref(),
            supervised_execution=supervised_execution,
            state=FactoryAttachmentItemState.EXECUTED,
        )

    @staticmethod
    def _work_templates(
        groups: tuple[ArtifactExecutionGroupV2, ...],
    ) -> tuple[AttachmentMockWorkV2, ...]:
        values: list[AttachmentMockWorkV2] = []
        for index, group in enumerate(groups):
            group_ref = artifact_execution_group_ref(group)
            values.append(
                AttachmentMockWorkV2(
                    work_key=(f"attachment-{index:04d}-{group_ref.object_sha256[:12]}"),
                    artifact_group_ref=group_ref,
                    artifact_ids=tuple(unit.artifact_id for unit in group.units),
                    agent_role="attachment-mock-agent",
                    dependency_work_keys=(),
                    input_object_types=("attachment-planning-context",),
                    output_object_types=("attachment-group-result",),
                    required_capability_ids=("agent-capability://attachment-mock",),
                    allowed_tool_ids=("attachment-execution",),
                    data_purposes=("attachment-production",),
                    data_classifications=("RESTRICTED_TRACE_DERIVED",),
                    workspace_policy_ref=_stable_ref(
                        "agent-workspace-policy",
                        group_ref.object_id,
                    ),
                    acceptance_check_refs=(
                        _stable_ref(
                            "acceptance-check",
                            group_ref.object_id,
                        ),
                    ),
                    max_attempts=2,
                    max_model_requests=0,
                    max_model_tokens=0,
                    max_cost_micro_usd=0,
                )
            )
        return tuple(values)


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


def _stage_outcome(
    outcome: AttachmentSubgraphOutcomeV2,
    reason_codes: tuple[str, ...],
) -> tuple[
    FactoryItemStageOutcomeV2,
    tuple[str, ...],
]:
    if outcome is AttachmentSubgraphOutcomeV2.SUCCEEDED:
        return FactoryItemStageOutcomeV2.SUCCEEDED, ()
    return (
        FactoryItemStageOutcomeV2.FAILED,
        reason_codes or ("ATTACHMENT_SUBGRAPH_FAILED",),
    )


__all__ = [
    "FactoryAttachmentItemRuntime",
    "FactoryAttachmentItemRuntimeError",
    "FactoryAttachmentItemState",
    "FactoryAttachmentItemView",
]
