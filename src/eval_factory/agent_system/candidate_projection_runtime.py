from __future__ import annotations

from dataclasses import dataclass

from eval_factory.agent_system.candidate_projection_material import (
    FactoryCandidateProjectionMaterialStore,
    FactoryCandidateProjectionResult,
)
from eval_factory.agent_system.store import (
    FactoryControlNotFoundError,
    FactoryControlStore,
)
from eval_factory.contracts.batch_quality_v2 import (
    batch_quality_report_v2_ref,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetAggregateResultV2,
    FactoryItemRunBindingV2,
    FactoryItemStageHeadV2,
    FactoryItemStageOutcomeV2,
    FactoryItemStageV2,
)
from eval_factory.contracts.quality_v2 import (
    item_quality_compilation_result_ref,
)
from eval_factory.contracts.release_projection_v2 import (
    ReleaseProjectionPolicyV2,
    ReleaseProjectionResultV2,
)
from eval_factory.dataset.persistence import (
    ReleaseProjectionPersistenceService,
)
from eval_factory.dataset.release import (
    EvaluationItemReleaseSource,
)
from eval_factory.orchestration.job_store import JobStore


class FactoryCandidateProjectionRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FactoryCandidateProjectionInput:
    binding: FactoryItemRunBindingV2
    source: EvaluationItemReleaseSource
    task_authoring_result_ref: ObjectRef
    attachment_item_quality_ref: ObjectRef
    criteria_result_ref: ObjectRef
    criteria_material_ref: ObjectRef
    grading_result_ref: ObjectRef


@dataclass(frozen=True, slots=True)
class FactoryCandidateProjectionView:
    item_id: str
    stage_head_ref: ObjectRef
    material_ref: ObjectRef
    projection: ReleaseProjectionResultV2


class FactoryCandidateProjectionRuntime:
    def __init__(
        self,
        *,
        store: FactoryControlStore,
        job_store: JobStore,
        materials: FactoryCandidateProjectionMaterialStore,
    ) -> None:
        self.store = store
        self.persistence = ReleaseProjectionPersistenceService(job_store)
        self.materials = materials

    def advance(
        self,
        *,
        dataset_run_id: str,
        aggregate: FactoryDatasetAggregateResultV2,
        inputs: tuple[FactoryCandidateProjectionInput, ...],
        policy: ReleaseProjectionPolicyV2,
        audit: ContractAudit,
    ) -> tuple[FactoryCandidateProjectionView, ...]:
        current_aggregate = self.store.get_dataset_aggregate(dataset_run_id)
        if current_aggregate != aggregate:
            raise FactoryCandidateProjectionRuntimeError("candidate projection aggregate is not current")
        if aggregate.batch_quality_ref is None or not aggregate.candidate_binding_refs:
            raise FactoryCandidateProjectionRuntimeError("candidate projection requires batch-approved items")
        input_refs = tuple(
            sorted(
                (value.binding.to_ref() for value in inputs),
                key=_ref_key,
            )
        )
        if input_refs != aggregate.candidate_binding_refs:
            raise FactoryCandidateProjectionRuntimeError("candidate projection input partition is not exact")
        views = tuple(
            self._advance_item(
                dataset_run_id=dataset_run_id,
                aggregate=aggregate,
                value=value,
                policy=policy,
                audit=audit,
            )
            for value in sorted(
                inputs,
                key=lambda item: item.binding.item_id,
            )
        )
        return views

    def _advance_item(
        self,
        *,
        dataset_run_id: str,
        aggregate: FactoryDatasetAggregateResultV2,
        value: FactoryCandidateProjectionInput,
        policy: ReleaseProjectionPolicyV2,
        audit: ContractAudit,
    ) -> FactoryCandidateProjectionView:
        binding = self.store.get_item_binding(
            dataset_run_id,
            value.binding.item_id,
        )
        if binding != value.binding:
            raise FactoryCandidateProjectionRuntimeError("candidate projection binding is not current")
        source = value.source
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
            source.item.item_id != binding.item_id
            or task_head.result_ref != value.task_authoring_result_ref
            or quality_head.result_ref != value.attachment_item_quality_ref
            or criteria_head.result_ref != value.criteria_result_ref
            or self.store.get_item_stage_material_ref(
                criteria_head.to_ref(),
            )
            != value.criteria_material_ref
            or grading_head.result_ref != value.grading_result_ref
            or source.batch_quality is None
            or batch_quality_report_v2_ref(source.batch_quality) != aggregate.batch_quality_ref
            or criteria_head.outcome is not FactoryItemStageOutcomeV2.SUCCEEDED
            or grading_head.outcome is not FactoryItemStageOutcomeV2.SUCCEEDED
        ):
            raise FactoryCandidateProjectionRuntimeError(
                "candidate projection source differs from current item authority"
            )
        try:
            stage_head = self.store.get_item_stage_head(
                binding.item_id,
                FactoryItemStageV2.RELEASE_CANDIDATE,
            )
        except FactoryControlNotFoundError:
            projection = self.persistence.request_release(
                source=source,
                policy=policy,
                idempotency_key=(f"request-factory-release-{binding.object_sha256}"),
                audit=audit,
            )
            result = FactoryCandidateProjectionResult(
                source=source,
                projection=projection,
                task_authoring_result_ref=(value.task_authoring_result_ref),
                attachment_item_quality_ref=(value.attachment_item_quality_ref),
                criteria_result_ref=value.criteria_result_ref,
                criteria_material_ref=value.criteria_material_ref,
                grading_result_ref=value.grading_result_ref,
            )
            material_ref = self.materials.put(result)
            stage_head = FactoryItemStageHeadV2.create(
                item_binding_ref=binding.to_ref(),
                item_run_ref=binding.item_run_ref,
                stage=FactoryItemStageV2.RELEASE_CANDIDATE,
                stage_version=1,
                predecessor_head_ref=None,
                dependency_result_refs=(
                    aggregate.batch_quality_ref,
                    item_quality_compilation_result_ref(source.item_quality),
                    value.criteria_result_ref,
                    value.grading_result_ref,
                ),
                result_ref=projection.to_ref(),
                outcome=FactoryItemStageOutcomeV2.SUCCEEDED,
                reason_codes=(),
                audit=audit,
            )
            stage_head = self.store.commit_item_stage_head(
                stage_head,
                idempotency_key=(f"commit-factory-release-head-{stage_head.object_sha256}"),
                material_ref=material_ref,
            )
        material_ref = self.store.get_item_stage_material_ref(stage_head.to_ref())
        result = self.materials.get(material_ref)
        persisted = self.persistence.request_release(
            source=result.source,
            policy=policy,
            idempotency_key=(f"request-factory-release-{binding.object_sha256}"),
            audit=result.projection.audit,
        )
        if persisted != result.projection:
            raise FactoryCandidateProjectionRuntimeError("candidate JobStore projection authority drifted")
        self.persistence.compiler.validate_current(
            result.projection,
            source=result.source,
            policy=policy,
            previous_result=None,
            audit=result.projection.audit,
        )
        if (
            result.source != source
            or result.task_authoring_result_ref != value.task_authoring_result_ref
            or result.attachment_item_quality_ref != value.attachment_item_quality_ref
            or result.criteria_result_ref != value.criteria_result_ref
            or result.criteria_material_ref != value.criteria_material_ref
            or result.grading_result_ref != value.grading_result_ref
            or stage_head.item_binding_ref != binding.to_ref()
            or stage_head.dependency_result_refs
            != tuple(
                sorted(
                    (
                        aggregate.batch_quality_ref,
                        item_quality_compilation_result_ref(source.item_quality),
                        value.criteria_result_ref,
                        value.grading_result_ref,
                    ),
                    key=_ref_key,
                )
            )
            or stage_head.result_ref != result.projection.to_ref()
        ):
            raise FactoryCandidateProjectionRuntimeError("candidate projection stage authority drifted")
        return FactoryCandidateProjectionView(
            item_id=binding.item_id,
            stage_head_ref=stage_head.to_ref(),
            material_ref=material_ref,
            projection=result.projection,
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
    "FactoryCandidateProjectionInput",
    "FactoryCandidateProjectionRuntime",
    "FactoryCandidateProjectionRuntimeError",
    "FactoryCandidateProjectionView",
]
