from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from eval_factory.agent_system.private_store import (
    FactoryPrivateObjectError,
    FactoryPrivateObjectStore,
)
from eval_factory.contracts.approval import (
    ApprovalCheckpoint,
    UserApprovalPolicy,
)
from eval_factory.contracts.approval_application_v2 import (
    DirectedRevalidationReportV2,
)
from eval_factory.contracts.approval_decision_v2 import (
    UserDecisionCommitResultV2,
)
from eval_factory.contracts.attachment_v2 import (
    AttachmentReconstructionResultV2,
)
from eval_factory.contracts.batch_quality_v2 import (
    BatchQualityReportV2,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.labeling_v2 import (
    LabelDecisionV2,
)
from eval_factory.contracts.orchestration_v2 import (
    DatasetJobSpecV2,
    ResolvedJobWorkGraphV2,
)
from eval_factory.contracts.quality_v2 import (
    ItemQualityCompilationResultV2,
)
from eval_factory.contracts.release_projection_v2 import (
    ReleaseProjectionResultV2,
)
from eval_factory.contracts.task_v2 import (
    R4TaskContractSetV2,
    TaskDraftV2,
    TaskPromptSafetyGateV2,
)
from eval_factory.dataset.release import (
    EvaluationItemReleaseSource,
)
from eval_factory.orchestration.models import ItemRecord


class FactoryCandidateProjectionMaterialError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FactoryCandidateProjectionResult:
    source: EvaluationItemReleaseSource
    projection: ReleaseProjectionResultV2
    task_authoring_result_ref: ObjectRef
    attachment_item_quality_ref: ObjectRef
    criteria_result_ref: ObjectRef
    criteria_material_ref: ObjectRef
    grading_result_ref: ObjectRef


class _FactoryCandidateProjectionMaterialV1(ContractModelV2):
    schema_version: Literal["eval-factory/private-candidate-projection-material/v1"] = (
        "eval-factory/private-candidate-projection-material/v1"
    )

    job_spec: DatasetJobSpecV2
    item: ItemRecord
    resolved_job_work_graph: ResolvedJobWorkGraphV2
    source_trace_refs: tuple[ObjectRef, ...]
    label_decisions: tuple[LabelDecisionV2, ...]
    task_draft: TaskDraftV2
    task_prompt_safety_gate: TaskPromptSafetyGateV2
    task_contract_set: R4TaskContractSetV2
    attachment_result: AttachmentReconstructionResultV2
    item_quality: ItemQualityCompilationResultV2
    batch_quality: BatchQualityReportV2
    approval_policy: UserApprovalPolicy
    decision_commits: tuple[UserDecisionCommitResultV2, ...]
    relevant_revalidation_application_refs: tuple[ObjectRef, ...]
    revalidation_reports: tuple[DirectedRevalidationReportV2, ...]
    current_head_refs: tuple[ObjectRef, ...]
    not_required_checkpoints: tuple[ApprovalCheckpoint, ...]
    projection: ReleaseProjectionResultV2
    task_authoring_result_ref: ObjectRef
    attachment_item_quality_ref: ObjectRef
    criteria_result_ref: ObjectRef
    criteria_material_ref: ObjectRef
    grading_result_ref: ObjectRef

    @classmethod
    def from_result(
        cls,
        value: FactoryCandidateProjectionResult,
    ) -> _FactoryCandidateProjectionMaterialV1:
        source = value.source
        return cls(
            job_spec=source.job_spec,
            item=source.item,
            resolved_job_work_graph=(source.resolved_job_work_graph),
            source_trace_refs=source.source_trace_refs,
            label_decisions=source.label_decisions,
            task_draft=source.task_draft,
            task_prompt_safety_gate=(source.task_prompt_safety_gate),
            task_contract_set=source.task_contract_set,
            attachment_result=source.attachment_result,
            item_quality=source.item_quality,
            batch_quality=source.batch_quality,
            approval_policy=source.approval_policy,
            decision_commits=source.decision_commits,
            relevant_revalidation_application_refs=(source.relevant_revalidation_application_refs),
            revalidation_reports=source.revalidation_reports,
            current_head_refs=source.current_head_refs,
            not_required_checkpoints=(source.not_required_checkpoints),
            projection=value.projection,
            task_authoring_result_ref=(value.task_authoring_result_ref),
            attachment_item_quality_ref=(value.attachment_item_quality_ref),
            criteria_result_ref=value.criteria_result_ref,
            criteria_material_ref=value.criteria_material_ref,
            grading_result_ref=value.grading_result_ref,
        )

    def to_result(self) -> FactoryCandidateProjectionResult:
        source = EvaluationItemReleaseSource(
            job_spec=self.job_spec,
            item=self.item,
            resolved_job_work_graph=(self.resolved_job_work_graph),
            source_trace_refs=self.source_trace_refs,
            label_decisions=self.label_decisions,
            task_draft=self.task_draft,
            task_prompt_safety_gate=(self.task_prompt_safety_gate),
            task_contract_set=self.task_contract_set,
            attachment_result=self.attachment_result,
            item_quality=self.item_quality,
            batch_quality=self.batch_quality,
            approval_policy=self.approval_policy,
            decision_commits=self.decision_commits,
            relevant_revalidation_application_refs=(self.relevant_revalidation_application_refs),
            revalidation_reports=self.revalidation_reports,
            current_head_refs=self.current_head_refs,
            not_required_checkpoints=(self.not_required_checkpoints),
        )
        return FactoryCandidateProjectionResult(
            source=source,
            projection=self.projection,
            task_authoring_result_ref=(self.task_authoring_result_ref),
            attachment_item_quality_ref=(self.attachment_item_quality_ref),
            criteria_result_ref=self.criteria_result_ref,
            criteria_material_ref=self.criteria_material_ref,
            grading_result_ref=self.grading_result_ref,
        )


class FactoryCandidateProjectionMaterialStore:
    def __init__(
        self,
        private_store: FactoryPrivateObjectStore,
    ) -> None:
        self.private_store = private_store

    def put(
        self,
        value: FactoryCandidateProjectionResult,
    ) -> ObjectRef:
        return self.private_store.put_model(
            object_type="candidate-projection-material",
            value=(_FactoryCandidateProjectionMaterialV1.from_result(value)),
        )

    def get(
        self,
        reference: ObjectRef,
    ) -> FactoryCandidateProjectionResult:
        if reference.object_type != "candidate-projection-material" or reference.object_version != "v2":
            raise FactoryCandidateProjectionMaterialError(
                "candidate projection material reference type is invalid"
            )
        try:
            value = self.private_store.get_model(
                reference,
                _FactoryCandidateProjectionMaterialV1,
            )
        except FactoryPrivateObjectError as exc:
            raise FactoryCandidateProjectionMaterialError(
                "candidate projection material is unavailable"
            ) from exc
        return value.to_result()


__all__ = [
    "FactoryCandidateProjectionMaterialError",
    "FactoryCandidateProjectionMaterialStore",
    "FactoryCandidateProjectionResult",
]
