from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from eval_factory.agent_system.private_store import (
    FactoryPrivateObjectError,
    FactoryPrivateObjectStore,
)
from eval_factory.batch_quality.reports import (
    BatchQualityItemSource,
)
from eval_factory.contracts.batch_quality_v2 import (
    BatchQualityPolicyV2,
    BatchQualityReportV2,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.cross_item_safety_v2 import (
    CrossItemSafetyPolicyV2,
    CrossItemSafetyResultV2,
)
from eval_factory.contracts.dataset_runtime_v2 import (
    CandidateDatasetOutcomeV2,
)
from eval_factory.contracts.duplicate_v2 import (
    DuplicateDetectionPolicyV2,
    DuplicateDetectionResultV2,
)
from eval_factory.contracts.orchestration_v2 import (
    ResolvedJobWorkGraphV2,
)


class FactoryBatchQualityMaterialError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FactoryBatchQualityResult:
    resolved_job_work_graph: ResolvedJobWorkGraphV2
    sources: tuple[BatchQualityItemSource, ...]
    duplicate_policy: DuplicateDetectionPolicyV2
    duplicate_result: DuplicateDetectionResultV2
    cross_item_policy: CrossItemSafetyPolicyV2
    cross_item_result: CrossItemSafetyResultV2
    batch_policy: BatchQualityPolicyV2
    report: BatchQualityReportV2


@dataclass(frozen=True, slots=True)
class FactoryBatchTerminalPartitionResult:
    core_vertical_result_ref: ObjectRef
    rejected_binding_refs: tuple[ObjectRef, ...]
    blocked_binding_refs: tuple[ObjectRef, ...]
    outcome: CandidateDatasetOutcomeV2
    reason_codes: tuple[str, ...]


class _FactoryBatchQualityMaterialV1(ContractModelV2):
    schema_version: Literal["eval-factory/private-batch-quality-material/v1"] = (
        "eval-factory/private-batch-quality-material/v1"
    )

    resolved_job_work_graph: ResolvedJobWorkGraphV2
    sources: tuple[BatchQualityItemSource, ...]
    duplicate_policy: DuplicateDetectionPolicyV2
    duplicate_result: DuplicateDetectionResultV2
    cross_item_policy: CrossItemSafetyPolicyV2
    cross_item_result: CrossItemSafetyResultV2
    batch_policy: BatchQualityPolicyV2
    report: BatchQualityReportV2

    @classmethod
    def from_result(
        cls,
        value: FactoryBatchQualityResult,
    ) -> _FactoryBatchQualityMaterialV1:
        return cls(
            resolved_job_work_graph=(value.resolved_job_work_graph),
            sources=value.sources,
            duplicate_policy=value.duplicate_policy,
            duplicate_result=value.duplicate_result,
            cross_item_policy=value.cross_item_policy,
            cross_item_result=value.cross_item_result,
            batch_policy=value.batch_policy,
            report=value.report,
        )

    def to_result(self) -> FactoryBatchQualityResult:
        return FactoryBatchQualityResult(
            resolved_job_work_graph=self.resolved_job_work_graph,
            sources=self.sources,
            duplicate_policy=self.duplicate_policy,
            duplicate_result=self.duplicate_result,
            cross_item_policy=self.cross_item_policy,
            cross_item_result=self.cross_item_result,
            batch_policy=self.batch_policy,
            report=self.report,
        )


class _FactoryBatchTerminalPartitionMaterialV1(ContractModelV2):
    schema_version: Literal["eval-factory/private-batch-terminal-partition-material/v1"] = (
        "eval-factory/private-batch-terminal-partition-material/v1"
    )

    core_vertical_result_ref: ObjectRef
    rejected_binding_refs: tuple[ObjectRef, ...]
    blocked_binding_refs: tuple[ObjectRef, ...]
    outcome: CandidateDatasetOutcomeV2
    reason_codes: tuple[str, ...]

    @classmethod
    def from_result(
        cls,
        value: FactoryBatchTerminalPartitionResult,
    ) -> _FactoryBatchTerminalPartitionMaterialV1:
        return cls(
            core_vertical_result_ref=(value.core_vertical_result_ref),
            rejected_binding_refs=value.rejected_binding_refs,
            blocked_binding_refs=value.blocked_binding_refs,
            outcome=value.outcome,
            reason_codes=value.reason_codes,
        )

    def to_result(
        self,
    ) -> FactoryBatchTerminalPartitionResult:
        return FactoryBatchTerminalPartitionResult(
            core_vertical_result_ref=(self.core_vertical_result_ref),
            rejected_binding_refs=self.rejected_binding_refs,
            blocked_binding_refs=self.blocked_binding_refs,
            outcome=self.outcome,
            reason_codes=self.reason_codes,
        )


FactoryBatchRuntimeResult = FactoryBatchQualityResult | FactoryBatchTerminalPartitionResult


class FactoryBatchQualityMaterialStore:
    def __init__(
        self,
        private_store: FactoryPrivateObjectStore,
    ) -> None:
        self.private_store = private_store

    def put(
        self,
        value: FactoryBatchRuntimeResult,
    ) -> ObjectRef:
        material: ContractModelV2
        if isinstance(
            value,
            FactoryBatchTerminalPartitionResult,
        ):
            material = _FactoryBatchTerminalPartitionMaterialV1.from_result(value)
        else:
            material = _FactoryBatchQualityMaterialV1.from_result(value)
        return self.private_store.put_model(
            object_type="batch-quality-material",
            value=material,
        )

    def get(
        self,
        reference: ObjectRef,
    ) -> FactoryBatchRuntimeResult:
        if reference.object_type != "batch-quality-material" or reference.object_version != "v2":
            raise FactoryBatchQualityMaterialError("batch quality material reference type is invalid")
        try:
            value = self.private_store.get_model(
                reference,
                _FactoryBatchQualityMaterialV1,
            )
        except FactoryPrivateObjectError:
            try:
                terminal = self.private_store.get_model(
                    reference,
                    _FactoryBatchTerminalPartitionMaterialV1,
                )
            except FactoryPrivateObjectError as exc:
                raise FactoryBatchQualityMaterialError("batch quality material is unavailable") from exc
            return terminal.to_result()
        else:
            return value.to_result()


__all__ = [
    "FactoryBatchQualityMaterialError",
    "FactoryBatchQualityMaterialStore",
    "FactoryBatchQualityResult",
    "FactoryBatchRuntimeResult",
    "FactoryBatchTerminalPartitionResult",
]
