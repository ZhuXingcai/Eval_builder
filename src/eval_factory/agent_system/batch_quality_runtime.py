from __future__ import annotations

from dataclasses import dataclass

from env_mock_agent.facade import (
    AttachmentCrossItemSafetyScanFacade,
    AttachmentDuplicateFingerprintFacade,
)
from eval_factory.agent_system.batch_quality_material import (
    FactoryBatchQualityMaterialStore,
    FactoryBatchQualityResult,
    FactoryBatchRuntimeResult,
    FactoryBatchTerminalPartitionResult,
)
from eval_factory.agent_system.store import (
    FactoryControlNotFoundError,
    FactoryControlStore,
)
from eval_factory.batch_quality.cross_item_safety import (
    CrossItemSafetyCompiler,
    CrossItemSafetyItemSource,
)
from eval_factory.batch_quality.duplicates import (
    DuplicateDetectionCompiler,
    DuplicateDetectionItemSource,
)
from eval_factory.batch_quality.reports import (
    BatchQualityCompiler,
    BatchQualityItemSource,
)
from eval_factory.contracts.batch_quality_v2 import (
    BatchQualityPolicyV2,
    batch_quality_report_v2_ref,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.cross_item_safety_v2 import (
    CrossItemSafetyPolicyV2,
)
from eval_factory.contracts.dataset_runtime_v2 import (
    CandidateDatasetOutcomeV2,
    FactoryDatasetAggregateResultV2,
)
from eval_factory.contracts.duplicate_v2 import (
    DuplicateDetectionPolicyV2,
)
from eval_factory.contracts.orchestration_v2 import (
    ResolvedJobWorkGraphV2,
)


class FactoryBatchQualityRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FactoryBatchQualityContext:
    resolved_job_work_graph: ResolvedJobWorkGraphV2
    duplicate_policy: DuplicateDetectionPolicyV2
    duplicate_facade: AttachmentDuplicateFingerprintFacade
    cross_item_policy: CrossItemSafetyPolicyV2
    cross_item_facade: AttachmentCrossItemSafetyScanFacade
    batch_policy: BatchQualityPolicyV2


@dataclass(frozen=True, slots=True)
class FactoryBatchQualityView:
    aggregate: FactoryDatasetAggregateResultV2
    material_ref: ObjectRef
    result: FactoryBatchRuntimeResult


class FactoryBatchQualityRuntime:
    def __init__(
        self,
        *,
        store: FactoryControlStore,
        materials: FactoryBatchQualityMaterialStore,
    ) -> None:
        self.store = store
        self.materials = materials

    async def advance(
        self,
        *,
        dataset_run_id: str,
        core_vertical_result_ref: ObjectRef,
        sources: tuple[BatchQualityItemSource, ...],
        rejected_binding_refs: tuple[ObjectRef, ...] = (),
        blocked_binding_refs: tuple[ObjectRef, ...] = (),
        context: FactoryBatchQualityContext,
        audit: ContractAudit,
    ) -> FactoryBatchQualityView:
        if not sources:
            return self._advance_terminal_partition(
                dataset_run_id=dataset_run_id,
                core_vertical_result_ref=(core_vertical_result_ref),
                rejected_binding_refs=rejected_binding_refs,
                blocked_binding_refs=blocked_binding_refs,
                audit=audit,
            )
        if len(sources) == 1:
            binding_ref = self.store.get_item_binding(
                dataset_run_id,
                sources[0].item_id,
            ).to_ref()
            if binding_ref in {
                *rejected_binding_refs,
                *blocked_binding_refs,
            }:
                raise FactoryBatchQualityRuntimeError(
                    "single Batch source also appears in a terminal partition",
                )
            return self._advance_terminal_partition(
                dataset_run_id=dataset_run_id,
                core_vertical_result_ref=(core_vertical_result_ref),
                rejected_binding_refs=rejected_binding_refs,
                blocked_binding_refs=tuple(
                    sorted(
                        (*blocked_binding_refs, binding_ref),
                        key=_ref_key,
                    ),
                ),
                blocked_reason="INSUFFICIENT_BATCH_CANDIDATES",
                audit=audit,
            )
        try:
            aggregate = self.store.get_dataset_aggregate(dataset_run_id)
        except FactoryControlNotFoundError:
            duplicate_sources = tuple(
                DuplicateDetectionItemSource(
                    item_id=value.item_id,
                    task_draft=value.task_draft,
                    task_contract_set=value.task_contract_set,
                    item_quality=value.item_quality,
                )
                for value in sources
            )
            duplicate_result = await DuplicateDetectionCompiler().compile(
                resolved_job_work_graph=(context.resolved_job_work_graph),
                policy=context.duplicate_policy,
                sources=duplicate_sources,
                facade=context.duplicate_facade,
                audit=audit,
            )
            cross_sources = tuple(
                CrossItemSafetyItemSource(
                    item_id=value.item_id,
                    task_draft=value.task_draft,
                    task_prompt_safety_gate=(value.task_prompt_safety_gate),
                    leakage_reference_set=(value.leakage_reference_set),
                    task_contract_set=value.task_contract_set,
                    item_quality=value.item_quality,
                )
                for value in sources
            )
            cross_result = await CrossItemSafetyCompiler().compile(
                resolved_job_work_graph=(context.resolved_job_work_graph),
                policy=context.cross_item_policy,
                sources=cross_sources,
                facade=context.cross_item_facade,
                audit=audit,
            )
            report = BatchQualityCompiler().compile(
                resolved_job_work_graph=(context.resolved_job_work_graph),
                policy=context.batch_policy,
                sources=sources,
                duplicate_policy=context.duplicate_policy,
                duplicate_result=duplicate_result,
                cross_item_policy=context.cross_item_policy,
                cross_item_result=cross_result,
                audit=audit,
            )
            result = FactoryBatchQualityResult(
                resolved_job_work_graph=(context.resolved_job_work_graph),
                sources=sources,
                duplicate_policy=context.duplicate_policy,
                duplicate_result=duplicate_result,
                cross_item_policy=context.cross_item_policy,
                cross_item_result=cross_result,
                batch_policy=context.batch_policy,
                report=report,
            )
            material_ref = self.materials.put(result)
            bindings = {
                value.item_id: self.store.get_item_binding(
                    dataset_run_id,
                    value.item_id,
                )
                for value in sources
            }
            explicit_blocked = tuple(
                sorted(
                    blocked_binding_refs,
                    key=_ref_key,
                )
            )
            if len(explicit_blocked) != len(set(explicit_blocked)) or not set(explicit_blocked).issubset(
                {binding.to_ref() for binding in bindings.values()}
            ):
                raise FactoryBatchQualityRuntimeError(
                    "explicit blocked bindings differ from Batch sources"
                ) from None
            blocked_ref_set = set(explicit_blocked)
            candidate_refs = tuple(
                bindings[value.item_id].to_ref()
                for value in sources
                if (
                    value.item_id not in report.blocked_item_ids
                    and bindings[value.item_id].to_ref() not in blocked_ref_set
                )
            )
            blocked = tuple(
                sorted(
                    {
                        *explicit_blocked,
                        *(bindings[item_id].to_ref() for item_id in report.blocked_item_ids),
                    },
                    key=_ref_key,
                )
            )
            rejected = tuple(
                sorted(
                    rejected_binding_refs,
                    key=_ref_key,
                )
            )
            if len(rejected) != len(set(rejected)) or set(rejected) & set(blocked):
                raise FactoryBatchQualityRuntimeError(
                    "Batch terminal bindings are duplicate or overlapping"
                ) from None
            all_refs = tuple(
                sorted(
                    (
                        *candidate_refs,
                        *rejected,
                        *blocked,
                    ),
                    key=_ref_key,
                )
            )
            dataset_run = self.store.get_run(dataset_run_id)
            reasons: tuple[str, ...]
            if candidate_refs and (rejected or blocked):
                outcome = CandidateDatasetOutcomeV2.PARTIAL
                reasons = ("PARTIAL_CANDIDATE_SET",)
            elif candidate_refs:
                outcome = CandidateDatasetOutcomeV2.COMPLETE
                reasons = ()
            elif blocked and not rejected:
                outcome = CandidateDatasetOutcomeV2.BLOCKED
                reasons = ("BATCH_QUALITY_BLOCKED",)
            else:
                outcome = CandidateDatasetOutcomeV2.NO_ELIGIBLE_ITEMS
                reasons = ("NO_ELIGIBLE_ITEMS",)
            aggregate = FactoryDatasetAggregateResultV2.create(
                dataset_run_ref=dataset_run.to_ref(),
                core_vertical_result_ref=(core_vertical_result_ref),
                item_binding_refs=all_refs,
                candidate_binding_refs=candidate_refs,
                rejected_binding_refs=rejected,
                blocked_binding_refs=blocked,
                batch_quality_ref=(batch_quality_report_v2_ref(report) if candidate_refs else None),
                outcome=outcome,
                reason_codes=reasons,
                audit=audit,
            )
            aggregate = self.store.commit_dataset_aggregate(
                aggregate,
                material_ref=material_ref,
                idempotency_key=(f"commit-factory-batch-quality-{aggregate.object_sha256}"),
            )
        material_ref = self.store.get_dataset_aggregate_material_ref(dataset_run_id)
        stored_result = self.materials.get(material_ref)
        if not isinstance(
            stored_result,
            FactoryBatchQualityResult,
        ):
            raise FactoryBatchQualityRuntimeError("batch quality material type drifted")
        self._validate_current(
            aggregate=aggregate,
            result=stored_result,
            core_vertical_result_ref=core_vertical_result_ref,
            sources=sources,
        )
        return FactoryBatchQualityView(
            aggregate=aggregate,
            material_ref=material_ref,
            result=stored_result,
        )

    def _advance_terminal_partition(
        self,
        *,
        dataset_run_id: str,
        core_vertical_result_ref: ObjectRef,
        rejected_binding_refs: tuple[ObjectRef, ...],
        blocked_binding_refs: tuple[ObjectRef, ...],
        audit: ContractAudit,
        blocked_reason: str = "BATCH_QUALITY_BLOCKED",
    ) -> FactoryBatchQualityView:
        rejected = tuple(
            sorted(
                rejected_binding_refs,
                key=_ref_key,
            )
        )
        blocked = tuple(
            sorted(
                blocked_binding_refs,
                key=_ref_key,
            )
        )
        if set(rejected) & set(blocked):
            raise FactoryBatchQualityRuntimeError("terminal item partitions overlap")
        if blocked and not rejected:
            outcome = CandidateDatasetOutcomeV2.BLOCKED
            reasons = (blocked_reason,)
        else:
            outcome = CandidateDatasetOutcomeV2.NO_ELIGIBLE_ITEMS
            reasons = ("NO_ELIGIBLE_ITEMS",)
        expected = FactoryBatchTerminalPartitionResult(
            core_vertical_result_ref=(core_vertical_result_ref),
            rejected_binding_refs=rejected,
            blocked_binding_refs=blocked,
            outcome=outcome,
            reason_codes=reasons,
        )
        try:
            aggregate = self.store.get_dataset_aggregate(dataset_run_id)
        except FactoryControlNotFoundError:
            material_ref = self.materials.put(expected)
            dataset_run = self.store.get_run(dataset_run_id)
            aggregate = FactoryDatasetAggregateResultV2.create(
                dataset_run_ref=dataset_run.to_ref(),
                core_vertical_result_ref=(core_vertical_result_ref),
                item_binding_refs=tuple(
                    sorted(
                        (
                            *rejected,
                            *blocked,
                        ),
                        key=_ref_key,
                    )
                ),
                candidate_binding_refs=(),
                rejected_binding_refs=rejected,
                blocked_binding_refs=blocked,
                batch_quality_ref=None,
                outcome=outcome,
                reason_codes=reasons,
                audit=audit,
            )
            aggregate = self.store.commit_dataset_aggregate(
                aggregate,
                material_ref=material_ref,
                idempotency_key=(f"commit-factory-terminal-partition-{aggregate.object_sha256}"),
            )
        material_ref = self.store.get_dataset_aggregate_material_ref(dataset_run_id)
        result = self.materials.get(material_ref)
        if (
            not isinstance(
                result,
                FactoryBatchTerminalPartitionResult,
            )
            or result != expected
            or aggregate.core_vertical_result_ref != core_vertical_result_ref
            or aggregate.rejected_binding_refs != rejected
            or aggregate.blocked_binding_refs != blocked
            or aggregate.candidate_binding_refs
            or aggregate.batch_quality_ref is not None
        ):
            raise FactoryBatchQualityRuntimeError("terminal partition authority drifted")
        return FactoryBatchQualityView(
            aggregate=aggregate,
            material_ref=material_ref,
            result=result,
        )

    @staticmethod
    def _validate_current(
        *,
        aggregate: FactoryDatasetAggregateResultV2,
        result: FactoryBatchQualityResult,
        core_vertical_result_ref: ObjectRef,
        sources: tuple[BatchQualityItemSource, ...],
    ) -> None:
        DuplicateDetectionCompiler().validate_current(
            result.duplicate_result,
            resolved_job_work_graph=(result.resolved_job_work_graph),
            policy=result.duplicate_policy,
            sources=tuple(
                DuplicateDetectionItemSource(
                    item_id=value.item_id,
                    task_draft=value.task_draft,
                    task_contract_set=value.task_contract_set,
                    item_quality=value.item_quality,
                )
                for value in result.sources
            ),
        )
        CrossItemSafetyCompiler().validate_current(
            result.cross_item_result,
            resolved_job_work_graph=(result.resolved_job_work_graph),
            policy=result.cross_item_policy,
            sources=tuple(
                CrossItemSafetyItemSource(
                    item_id=value.item_id,
                    task_draft=value.task_draft,
                    task_prompt_safety_gate=(value.task_prompt_safety_gate),
                    leakage_reference_set=(value.leakage_reference_set),
                    task_contract_set=value.task_contract_set,
                    item_quality=value.item_quality,
                )
                for value in result.sources
            ),
        )
        BatchQualityCompiler().validate_current(
            result.report,
            resolved_job_work_graph=(result.resolved_job_work_graph),
            policy=result.batch_policy,
            sources=result.sources,
            duplicate_policy=result.duplicate_policy,
            duplicate_result=result.duplicate_result,
            cross_item_policy=result.cross_item_policy,
            cross_item_result=result.cross_item_result,
        )
        if (
            aggregate.core_vertical_result_ref != core_vertical_result_ref
            or aggregate.batch_quality_ref
            != (batch_quality_report_v2_ref(result.report) if aggregate.candidate_binding_refs else None)
            or sources != result.sources
        ):
            raise FactoryBatchQualityRuntimeError("batch quality aggregate authority drifted")


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
    "FactoryBatchQualityContext",
    "FactoryBatchQualityRuntime",
    "FactoryBatchQualityRuntimeError",
    "FactoryBatchQualityView",
]
