from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass

from eval_factory.agent_system.candidate_projection_runtime import (
    FactoryCandidateProjectionInput,
)
from eval_factory.agent_system.store import (
    FactoryControlStore,
)
from eval_factory.approval.application_persistence import (
    UserPlanApplicationConflictError,
    UserPlanApplicationPersistenceService,
)
from eval_factory.approval.persistence import (
    UserDecisionPersistenceService,
)
from eval_factory.contracts.approval import (
    ApprovalCheckpoint,
    UserApprovalPolicy,
)
from eval_factory.contracts.approval_application_v2 import (
    DirectedRevalidationReportV2,
    UserPlanApplicationV2,
    directed_revalidation_report_v2_ref,
    user_plan_application_v2_ref,
)
from eval_factory.contracts.approval_decision_v2 import (
    UserDecisionCommitOutcomeV2,
    UserDecisionCommitResultV2,
    user_decision_commit_result_v2_ref,
)
from eval_factory.contracts.approval_v2 import (
    user_approval_policy_ref,
)
from eval_factory.contracts.attachment_v2 import (
    AttachmentReconstructionResultV2,
)
from eval_factory.contracts.batch_quality_v2 import (
    BatchQualityReportV2,
    batch_quality_report_v2_ref,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetAggregateResultV2,
    FactoryItemRunBindingV2,
    FactoryItemStageOutcomeV2,
    FactoryItemStageV2,
)
from eval_factory.contracts.labeling_v2 import (
    LabelDecisionV2,
)
from eval_factory.contracts.orchestration_v2 import (
    DatasetJobSpecV2,
    ResolvedJobWorkGraphV2,
    resolved_job_work_graph_v2_ref,
)
from eval_factory.contracts.quality_v2 import (
    ItemQualityCompilationResultV2,
    item_quality_compilation_result_ref,
)
from eval_factory.contracts.task_v2 import (
    R4TaskContractSetV2,
    TaskDraftV2,
    TaskPromptSafetyGateV2,
    r4_task_contract_set_ref,
)
from eval_factory.dataset.release import (
    EvaluationItemReleaseSource,
)
from eval_factory.orchestration.job_store import (
    JobStore,
    RecordNotFoundError,
)


class FactoryReleaseSourceBuilderError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FactoryReleaseSourceItem:
    binding: FactoryItemRunBindingV2
    source_trace_ref: ObjectRef
    label_decisions: tuple[LabelDecisionV2, ...]
    task_draft: TaskDraftV2
    task_prompt_safety_gate: TaskPromptSafetyGateV2
    base_task_contract_set: R4TaskContractSetV2
    task_contract_set: R4TaskContractSetV2
    attachment_result: AttachmentReconstructionResultV2
    attachment_item_quality_ref: ObjectRef
    item_quality: ItemQualityCompilationResultV2
    criteria_result_ref: ObjectRef
    criteria_material_ref: ObjectRef
    grading_result_ref: ObjectRef


class FactoryReleaseSourceBuilder:
    def __init__(
        self,
        *,
        store: FactoryControlStore,
        job_store: JobStore,
    ) -> None:
        self.store = store
        self.job_store = job_store

    def build(
        self,
        *,
        dataset_run_id: str,
        aggregate: FactoryDatasetAggregateResultV2,
        items: tuple[FactoryReleaseSourceItem, ...],
        batch_quality: BatchQualityReportV2,
        approval_policy: UserApprovalPolicy,
        not_required_checkpoints: tuple[
            ApprovalCheckpoint,
            ...,
        ] = (),
    ) -> tuple[FactoryCandidateProjectionInput, ...]:
        current_aggregate = self.store.get_dataset_aggregate(
            dataset_run_id,
        )
        if current_aggregate != aggregate:
            raise FactoryReleaseSourceBuilderError("release source aggregate is not current")
        batch_ref = batch_quality_report_v2_ref(batch_quality)
        if aggregate.batch_quality_ref != batch_ref or not aggregate.candidate_binding_refs:
            raise FactoryReleaseSourceBuilderError("release source requires current candidate batch quality")
        ordered = tuple(
            sorted(
                items,
                key=lambda value: _ref_key(value.binding.to_ref()),
            )
        )
        if tuple(value.binding.to_ref() for value in ordered) != aggregate.candidate_binding_refs:
            raise FactoryReleaseSourceBuilderError("release source Item partition is not exact")
        try:
            first_item = self.job_store.get_item(ordered[0].binding.item_id)
            graph = self.job_store.get_job_work_graph(first_item.job_id)
            job_spec = self.job_store.get_job_spec(graph.job_id)
        except RecordNotFoundError as exc:
            raise FactoryReleaseSourceBuilderError(
                "release source JobStore authority is unavailable"
            ) from exc
        if (
            resolved_job_work_graph_v2_ref(graph) != batch_quality.resolved_job_work_graph_ref
            or user_approval_policy_ref(approval_policy) != job_spec.approval_policy_ref
        ):
            raise FactoryReleaseSourceBuilderError(
                "release source JobStore or approval policy authority differs"
            )
        item_quality_refs = dict(
            zip(
                batch_quality.item_ids,
                batch_quality.item_quality_result_refs,
                strict=True,
            )
        )
        if set(item_quality_refs) != {value.binding.item_id for value in ordered}:
            raise FactoryReleaseSourceBuilderError("release source Batch quality inventory differs")
        decisions = tuple(
            sorted(
                UserDecisionPersistenceService(self.job_store).list_job_commits(graph.job_id),
                key=lambda value: _ref_key(user_decision_commit_result_v2_ref(value)),
            )
        )
        applications, reports = self._revalidation_authority(
            job_id=graph.job_id,
            item_ids=tuple(value.binding.item_id for value in ordered),
            decisions=decisions,
        )
        current_heads = UserPlanApplicationPersistenceService(self.job_store).list_current_heads(graph.job_id)
        checkpoint_order = {value: index for index, value in enumerate(ApprovalCheckpoint)}
        not_required = tuple(
            sorted(
                set(not_required_checkpoints),
                key=lambda value: checkpoint_order[value],
            )
        )
        sources = tuple(
            self._build_item(
                dataset_run_id=dataset_run_id,
                value=value,
                job_spec=job_spec,
                graph=graph,
                batch_quality=batch_quality,
                expected_item_quality_ref=(item_quality_refs[value.binding.item_id]),
                approval_policy=approval_policy,
                decisions=decisions,
                applications=applications,
                reports=reports,
                current_heads=current_heads,
                not_required_checkpoints=not_required,
            )
            for value in ordered
        )
        return tuple(
            FactoryCandidateProjectionInput(
                binding=value.binding,
                source=source,
                task_authoring_result_ref=(r4_task_contract_set_ref(value.base_task_contract_set)),
                attachment_item_quality_ref=(value.attachment_item_quality_ref),
                criteria_result_ref=value.criteria_result_ref,
                criteria_material_ref=value.criteria_material_ref,
                grading_result_ref=value.grading_result_ref,
            )
            for value, source in zip(
                ordered,
                sources,
                strict=True,
            )
        )

    def _build_item(
        self,
        *,
        dataset_run_id: str,
        value: FactoryReleaseSourceItem,
        job_spec: DatasetJobSpecV2,
        graph: ResolvedJobWorkGraphV2,
        batch_quality: BatchQualityReportV2,
        expected_item_quality_ref: ObjectRef,
        approval_policy: UserApprovalPolicy,
        decisions: tuple[
            UserDecisionCommitResultV2,
            ...,
        ],
        applications: tuple[UserPlanApplicationV2, ...],
        reports: tuple[DirectedRevalidationReportV2, ...],
        current_heads: tuple[ObjectRef, ...],
        not_required_checkpoints: tuple[
            ApprovalCheckpoint,
            ...,
        ],
    ) -> EvaluationItemReleaseSource:
        binding = self.store.get_item_binding(
            dataset_run_id,
            value.binding.item_id,
        )
        if binding != value.binding:
            raise FactoryReleaseSourceBuilderError("release source binding is not current")
        try:
            item = self.job_store.get_item(binding.item_id)
        except RecordNotFoundError as exc:
            raise FactoryReleaseSourceBuilderError("release source JobStore Item is unavailable") from exc
        expected_source = dict(
            zip(
                graph.item_ids,
                graph.source_trace_refs,
                strict=True,
            )
        ).get(binding.item_id)
        if (
            item.job_id != graph.job_id
            or expected_source != value.source_trace_ref
            or binding.item_id not in batch_quality.item_ids
        ):
            raise FactoryReleaseSourceBuilderError("release source JobStore Item or trace authority differs")
        task_head = self.store.get_item_stage_head(
            binding.item_id,
            FactoryItemStageV2.TASK_AUTHORING,
        )
        quality_head = self.store.get_item_stage_head(
            binding.item_id,
            FactoryItemStageV2.ITEM_QUALITY,
        )
        criteria_head = self.store.get_item_stage_head(
            binding.item_id,
            FactoryItemStageV2.CRITERIA_RUBRIC,
        )
        grading_head = self.store.get_item_stage_head(
            binding.item_id,
            FactoryItemStageV2.GRADING_DESIGN,
        )
        if (
            task_head.result_ref != r4_task_contract_set_ref(value.base_task_contract_set)
            or quality_head.result_ref != value.attachment_item_quality_ref
            or item_quality_compilation_result_ref(value.item_quality) != expected_item_quality_ref
            or value.item_quality.quality_report.r4_task_contract_set_ref
            != r4_task_contract_set_ref(value.task_contract_set)
            or criteria_head.result_ref != value.criteria_result_ref
            or self.store.get_item_stage_material_ref(
                criteria_head.to_ref(),
            )
            != value.criteria_material_ref
            or grading_head.result_ref != value.grading_result_ref
            or criteria_head.outcome is not FactoryItemStageOutcomeV2.SUCCEEDED
            or grading_head.outcome is not FactoryItemStageOutcomeV2.SUCCEEDED
        ):
            raise FactoryReleaseSourceBuilderError(
                "release source task, quality, or grading authority differs"
            )
        relevant_applications = tuple(
            application for application in applications if binding.item_id in application.affected_item_ids
        )
        relevant_refs = tuple(user_plan_application_v2_ref(value) for value in relevant_applications)
        relevant_reports = tuple(report for report in reports if report.application_ref in relevant_refs)
        return EvaluationItemReleaseSource(
            job_spec=job_spec,
            item=item,
            resolved_job_work_graph=graph,
            source_trace_refs=(value.source_trace_ref,),
            label_decisions=value.label_decisions,
            task_draft=value.task_draft,
            task_prompt_safety_gate=(value.task_prompt_safety_gate),
            task_contract_set=value.task_contract_set,
            attachment_result=value.attachment_result,
            item_quality=value.item_quality,
            batch_quality=batch_quality,
            approval_policy=approval_policy,
            decision_commits=decisions,
            relevant_revalidation_application_refs=(relevant_refs),
            revalidation_reports=relevant_reports,
            current_head_refs=current_heads,
            not_required_checkpoints=(not_required_checkpoints),
        )

    def _revalidation_authority(
        self,
        *,
        job_id: str,
        item_ids: tuple[str, ...],
        decisions: tuple[
            UserDecisionCommitResultV2,
            ...,
        ],
    ) -> tuple[
        tuple[UserPlanApplicationV2, ...],
        tuple[DirectedRevalidationReportV2, ...],
    ]:
        service = UserPlanApplicationPersistenceService(self.job_store)
        applications = []
        reports = []
        for decision in decisions:
            if decision.outcome is not UserDecisionCommitOutcomeV2.ADJUSTMENT_ACCEPTED:
                continue
            try:
                application = service.get_application_for_decision(decision.commit_id)
            except UserPlanApplicationConflictError as exc:
                raise FactoryReleaseSourceBuilderError(
                    "release source adjustment lacks an application"
                ) from exc
            if application.job_id != job_id or not set(application.affected_item_ids) & set(item_ids):
                continue
            applications.append(application)
            with suppress(RecordNotFoundError):
                reports.append(service.get_report(application.application_id))
        return (
            tuple(
                sorted(
                    applications,
                    key=lambda value: _ref_key(user_plan_application_v2_ref(value)),
                )
            ),
            tuple(
                sorted(
                    reports,
                    key=lambda value: _ref_key(directed_revalidation_report_v2_ref(value)),
                )
            ),
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
    "FactoryReleaseSourceBuilder",
    "FactoryReleaseSourceBuilderError",
    "FactoryReleaseSourceItem",
]
