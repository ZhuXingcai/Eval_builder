from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from eval_factory.contracts.approval import (
    ALL_CHECKPOINTS,
    PLAN_CHECKPOINTS,
    ApprovalCheckpoint,
    ApprovalMode,
)
from eval_factory.contracts.core import (
    ContractAudit,
    FailureRecord,
    Identifier,
    ObjectRef,
    Sha256,
)
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.contracts.orchestration import (
    ConcurrencyLimit,
    ExportTarget,
    ResourceBudget,
    TraceSourceRef,
)
from eval_factory.contracts.review_v2 import SemanticReviewRoundV2


class StageNameV2(StrEnum):
    TRACE_INDEX = "trace_index"
    SAFETY = "safety"
    LABEL_PLAN = "label_plan"
    LABEL = "label"
    TASK_REWRITE_PLAN = "task_rewrite_plan"
    TASK_AUTHORING = "task_authoring"
    ENVIRONMENT_STRATEGY = "environment_strategy"
    ATTACHMENT = "attachment"
    ITEM_QUALITY = "item_quality"
    BATCH_QUALITY = "batch_quality"
    FINAL_DATASET_REVIEW = "final_dataset_review"
    RELEASE = "release"


CHECKPOINT_STAGES = {
    ApprovalCheckpoint.LABEL_PLAN: StageNameV2.LABEL_PLAN,
    ApprovalCheckpoint.TASK_REWRITE_PLAN: StageNameV2.TASK_REWRITE_PLAN,
    ApprovalCheckpoint.ENVIRONMENT_STRATEGY: StageNameV2.ENVIRONMENT_STRATEGY,
    ApprovalCheckpoint.FINAL_DATASET_REVIEW: StageNameV2.FINAL_DATASET_REVIEW,
}


class DatasetJobSpecV2(ContractModelV2):
    schema_version: Literal["eval-factory/dataset-job-spec/v2"] = "eval-factory/dataset-job-spec/v2"
    job_id: Identifier
    traces: tuple[TraceSourceRef, ...] = Field(min_length=1)
    requested_stages: tuple[StageNameV2, ...] = Field(min_length=1)
    privacy_profile: Identifier
    model_profiles: tuple[Identifier, ...] = ()
    budget: ResourceBudget
    concurrency: ConcurrencyLimit
    selection_spec_ref: ObjectRef | None = None
    approval_policy_ref: ObjectRef
    approval_mode: ApprovalMode
    enabled_checkpoints: frozenset[ApprovalCheckpoint]
    export_target: ExportTarget
    idempotency_key: Identifier
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_plan(self) -> DatasetJobSpecV2:
        if len(set(self.requested_stages)) != len(self.requested_stages):
            raise ValueError("requested_stages must be unique")
        if self.requested_stages[0] is not StageNameV2.TRACE_INDEX:
            raise ValueError("requested_stages must begin with trace_index")
        canonical = list(StageNameV2)
        positions = [canonical.index(stage) for stage in self.requested_stages]
        if positions != sorted(positions):
            raise ValueError("requested_stages must follow canonical v2 stage order")

        expected = {
            ApprovalMode.NONE: frozenset(),
            ApprovalMode.PLAN_GATES: PLAN_CHECKPOINTS,
            ApprovalMode.FINAL_ONLY: frozenset({ApprovalCheckpoint.FINAL_DATASET_REVIEW}),
            ApprovalMode.PLAN_AND_FINAL: ALL_CHECKPOINTS,
        }
        if self.approval_mode in expected and self.enabled_checkpoints != expected[self.approval_mode]:
            raise ValueError("approval_mode and enabled_checkpoints disagree")
        if self.approval_mode is ApprovalMode.CUSTOM and not self.enabled_checkpoints:
            raise ValueError("CUSTOM approval mode requires enabled checkpoints")

        stages = set(self.requested_stages)
        for checkpoint, stage in CHECKPOINT_STAGES.items():
            if (checkpoint in self.enabled_checkpoints) != (stage in stages):
                raise ValueError(f"{stage} presence must match {checkpoint} enablement")
        return self


class ResolvedStageNodeV2(ContractModelV2):
    schema_version: Literal["eval-factory/resolved-stage-node/v2"] = "eval-factory/resolved-stage-node/v2"
    stage: StageNameV2
    depends_on: tuple[StageNameV2, ...] = ()

    @model_validator(mode="after")
    def validate_dependencies(self) -> Self:
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("resolved stage dependencies must be unique")
        if self.stage in self.depends_on:
            raise ValueError("resolved stage cannot depend on itself")
        if self.stage is StageNameV2.TRACE_INDEX and self.depends_on:
            raise ValueError("trace_index must be the only resolved stage root")
        if self.stage is not StageNameV2.TRACE_INDEX and not self.depends_on:
            raise ValueError("non-root resolved stage requires a dependency")
        return self


class ResolvedDatasetJobPlanV2(ContractModelV2):
    schema_version: Literal["eval-factory/resolved-dataset-job-plan/v2"] = (
        "eval-factory/resolved-dataset-job-plan/v2"
    )
    resolved_job_plan_id: Identifier
    dataset_job_spec_ref: ObjectRef
    requested_stages: tuple[StageNameV2, ...] = Field(min_length=1)
    resolved_stages: tuple[StageNameV2, ...] = Field(min_length=1)
    auto_added_stages: tuple[StageNameV2, ...] = ()
    nodes: tuple[ResolvedStageNodeV2, ...] = Field(min_length=1)
    policy_version: Identifier
    resolved_job_plan_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        _require_dataset_job_spec_ref(self.dataset_job_spec_ref)
        _require_canonical_unique_stages("requested_stages", self.requested_stages)
        _require_canonical_unique_stages("resolved_stages", self.resolved_stages)
        if self.requested_stages[0] is not StageNameV2.TRACE_INDEX:
            raise ValueError("requested stages must begin with trace_index")
        if self.resolved_stages[0] is not StageNameV2.TRACE_INDEX:
            raise ValueError("resolved stages must begin with trace_index")

        resolved_set = set(self.resolved_stages)
        if any(stage not in resolved_set for stage in self.requested_stages):
            raise ValueError("requested stages must be a subset of resolved stages")
        if self.resolved_stages[-1] is not self.requested_stages[-1]:
            raise ValueError("resolved stages must end at the highest requested stage")
        expected_added = tuple(
            stage for stage in self.resolved_stages if stage not in set(self.requested_stages)
        )
        if self.auto_added_stages != expected_added:
            raise ValueError("auto-added stages must equal the ordered resolved/requested difference")
        if tuple(node.stage for node in self.nodes) != self.resolved_stages:
            raise ValueError("resolved plan nodes must exactly cover resolved stages in order")

        for index, node in enumerate(self.nodes):
            expected_dependencies = () if index == 0 else (self.resolved_stages[index - 1],)
            if node.depends_on != expected_dependencies:
                raise ValueError("resolved plan edges must form the canonical direct stage chain")

        if self.audit.input_refs != (self.dataset_job_spec_ref,):
            raise ValueError("resolved plan audit refs are incomplete")
        policy_bindings = tuple(
            binding
            for binding in self.audit.governing_versions
            if binding.component == "dataset-job-stage-policy"
        )
        if len(policy_bindings) != 1 or policy_bindings[0].version != self.policy_version:
            raise ValueError("resolved plan audit is missing the current stage policy")
        validate_resolved_dataset_job_plan_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        dataset_job_spec_ref: ObjectRef,
        requested_stages: tuple[StageNameV2, ...],
        resolved_stages: tuple[StageNameV2, ...],
        auto_added_stages: tuple[StageNameV2, ...],
        nodes: tuple[ResolvedStageNodeV2, ...],
        policy_version: Identifier,
        audit: ContractAudit,
    ) -> ResolvedDatasetJobPlanV2:
        value = cls(
            resolved_job_plan_id="resolved-job-plan://pending",
            dataset_job_spec_ref=dataset_job_spec_ref,
            requested_stages=requested_stages,
            resolved_stages=resolved_stages,
            auto_added_stages=auto_added_stages,
            nodes=nodes,
            policy_version=policy_version,
            resolved_job_plan_sha256="0" * 64,
            audit=audit,
        )
        digest = resolved_dataset_job_plan_v2_carried_sha256(value)
        return value.model_copy(
            update={
                "resolved_job_plan_id": f"resolved-job-plan://sha256/{digest}",
                "resolved_job_plan_sha256": digest,
            }
        )


def dataset_job_spec_v2_ref(spec: DatasetJobSpecV2) -> ObjectRef:
    return ObjectRef(
        object_type="dataset-job-spec",
        object_id=spec.job_id,
        object_version="v2",
        object_sha256=spec.canonical_sha256(),
    )


def resolved_dataset_job_plan_v2_carried_sha256(
    plan: ResolvedDatasetJobPlanV2,
) -> str:
    payload = {
        "schema_version": plan.schema_version,
        "dataset_job_spec_ref": plan.dataset_job_spec_ref,
        "requested_stages": plan.requested_stages,
        "resolved_stages": plan.resolved_stages,
        "auto_added_stages": plan.auto_added_stages,
        "nodes": plan.nodes,
        "policy_version": plan.policy_version,
    }
    encoded = json.dumps(
        canonical_value_v2(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def resolved_dataset_job_plan_v2_ref(
    plan: ResolvedDatasetJobPlanV2,
) -> ObjectRef:
    validate_resolved_dataset_job_plan_v2_identity(plan)
    return ObjectRef(
        object_type="resolved-dataset-job-plan",
        object_id=plan.resolved_job_plan_id,
        object_version="v2",
        object_sha256=plan.resolved_job_plan_sha256,
    )


def validate_resolved_dataset_job_plan_v2_identity(
    plan: ResolvedDatasetJobPlanV2,
) -> None:
    if (
        plan.resolved_job_plan_id == "resolved-job-plan://pending"
        and plan.resolved_job_plan_sha256 == "0" * 64
    ):
        return
    observed = resolved_dataset_job_plan_v2_carried_sha256(plan)
    expected_id = f"resolved-job-plan://sha256/{observed}"
    if plan.resolved_job_plan_sha256 != observed or plan.resolved_job_plan_id != expected_id:
        raise ValueError("resolved job plan identity is stale")


def _require_dataset_job_spec_ref(ref: ObjectRef) -> None:
    if ref.object_type != "dataset-job-spec" or ref.object_version != "v2":
        raise ValueError("dataset_job_spec_ref must reference dataset-job-spec v2")


def _require_canonical_unique_stages(
    label: str,
    stages: tuple[StageNameV2, ...],
) -> None:
    if len(set(stages)) != len(stages):
        raise ValueError(f"{label} must be unique")
    positions = [list(StageNameV2).index(stage) for stage in stages]
    if positions != sorted(positions):
        raise ValueError(f"{label} must follow canonical v2 stage order")


R6_WORK_GRAPH_POLICY_VERSION: Literal["dataset-work-graph/r6-02-v1"] = "dataset-work-graph/r6-02-v1"


class WorkUnitScopeV2(StrEnum):
    JOB = "JOB"
    ITEM = "ITEM"
    ARTIFACT_GROUP = "ARTIFACT_GROUP"


class WorkDependencyJoinModeV2(StrEnum):
    ALL_SUCCEEDED = "ALL_SUCCEEDED"
    ALL_TERMINAL = "ALL_TERMINAL"


class WorkReadinessV2(StrEnum):
    READY = "READY"
    WAITING = "WAITING"
    BLOCKED_DEPENDENCY = "BLOCKED_DEPENDENCY"
    SKIPPED_ITEM = "SKIPPED_ITEM"


R6_WORK_CONTROL_POLICY_VERSION: Literal["work-control/r6-05-v1"] = "work-control/r6-05-v1"
R6_WORK_CONTROL_FAILURE_MESSAGE: Literal["Controlled work completed with a typed non-success outcome."] = (
    "Controlled work completed with a typed non-success outcome."
)


class WorkLeaseEventKindV2(StrEnum):
    HEARTBEAT = "HEARTBEAT"
    SUCCEEDED = "SUCCEEDED"
    RETRYABLE_FAILURE = "RETRYABLE_FAILURE"
    TERMINAL_FAILURE = "TERMINAL_FAILURE"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class WorkLeaseStateV2(StrEnum):
    ACTIVE = "ACTIVE"
    SUCCEEDED = "SUCCEEDED"
    RETRYABLE_FAILURE = "RETRYABLE_FAILURE"
    TERMINAL_FAILURE = "TERMINAL_FAILURE"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class WorkRetryDecisionKindV2(StrEnum):
    RETRY_SCHEDULED = "RETRY_SCHEDULED"
    EXHAUSTED = "EXHAUSTED"


class ResolvedWorkUnitV2(ContractModelV2):
    schema_version: Literal["eval-factory/resolved-work-unit/v2"] = "eval-factory/resolved-work-unit/v2"
    resolved_work_unit_id: Identifier
    scope: WorkUnitScopeV2
    stage: StageNameV2
    job_id: Identifier
    item_id: Identifier | None = None
    source_trace_ref: ObjectRef | None = None
    artifact_execution_group_ref: ObjectRef | None = None
    semantic_review_round: SemanticReviewRoundV2 | None = None
    depends_on_work_unit_refs: tuple[ObjectRef, ...] = ()
    join_mode: WorkDependencyJoinModeV2
    policy_version: Literal["dataset-work-graph/r6-02-v1"] = R6_WORK_GRAPH_POLICY_VERSION
    resolved_work_unit_sha256: Sha256

    @model_validator(mode="after")
    def validate_unit(self) -> Self:
        if self.semantic_review_round is not None and (
            self.scope is not WorkUnitScopeV2.ITEM or self.stage is not StageNameV2.ITEM_QUALITY
        ):
            raise ValueError("semantic review round requires ITEM ITEM_QUALITY work")
        if self.scope is WorkUnitScopeV2.JOB:
            if (
                self.item_id is not None
                or self.source_trace_ref is not None
                or self.artifact_execution_group_ref is not None
            ):
                raise ValueError("JOB work unit cannot carry item, trace, or artifact group")
        elif self.scope is WorkUnitScopeV2.ITEM:
            if (
                self.item_id is None
                or self.source_trace_ref is None
                or self.artifact_execution_group_ref is not None
            ):
                raise ValueError("ITEM work unit requires item and trace only")
            _require_ref_version(
                self.source_trace_ref,
                "trace-source",
                self.source_trace_ref.object_version,
                "source_trace_ref",
            )
        else:
            if (
                self.item_id is None
                or self.source_trace_ref is not None
                or self.artifact_execution_group_ref is None
                or self.stage is not StageNameV2.ATTACHMENT
            ):
                raise ValueError("ARTIFACT_GROUP work unit requires item, attachment stage, and group ref")
            _require_ref_version(
                self.artifact_execution_group_ref,
                "artifact-execution-group",
                "v2",
                "artifact_execution_group_ref",
            )
        _require_sorted_unique_refs(
            "work unit dependencies",
            self.depends_on_work_unit_refs,
        )
        for ref in self.depends_on_work_unit_refs:
            _require_ref_version(
                ref,
                "resolved-work-unit",
                "v2",
                "depends_on_work_unit_refs",
            )
            if ref.object_id == self.resolved_work_unit_id:
                raise ValueError("work unit cannot depend on itself")
        validate_resolved_work_unit_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        scope: WorkUnitScopeV2,
        stage: StageNameV2,
        job_id: Identifier,
        item_id: Identifier | None,
        source_trace_ref: ObjectRef | None,
        artifact_execution_group_ref: ObjectRef | None,
        depends_on_work_unit_refs: tuple[ObjectRef, ...],
        join_mode: WorkDependencyJoinModeV2,
        semantic_review_round: SemanticReviewRoundV2 | None = None,
    ) -> ResolvedWorkUnitV2:
        value = cls(
            resolved_work_unit_id="resolved-work-unit://pending",
            scope=scope,
            stage=stage,
            job_id=job_id,
            item_id=item_id,
            source_trace_ref=source_trace_ref,
            artifact_execution_group_ref=artifact_execution_group_ref,
            semantic_review_round=semantic_review_round,
            depends_on_work_unit_refs=_sorted_refs(depends_on_work_unit_refs),
            join_mode=join_mode,
            resolved_work_unit_sha256="0" * 64,
        )
        digest = resolved_work_unit_v2_carried_sha256(value)
        return value.model_copy(
            update={
                "resolved_work_unit_id": (f"resolved-work-unit://sha256/{digest}"),
                "resolved_work_unit_sha256": digest,
            }
        )


class ResolvedJobWorkGraphV2(ContractModelV2):
    schema_version: Literal["eval-factory/resolved-job-work-graph/v2"] = (
        "eval-factory/resolved-job-work-graph/v2"
    )
    resolved_job_work_graph_id: Identifier
    job_id: Identifier
    dataset_job_spec_ref: ObjectRef
    resolved_job_plan_ref: ObjectRef
    item_ids: tuple[Identifier, ...] = Field(min_length=1)
    source_trace_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    work_units: tuple[ResolvedWorkUnitV2, ...] = Field(min_length=1)
    policy_version: Literal["dataset-work-graph/r6-02-v1"] = R6_WORK_GRAPH_POLICY_VERSION
    resolved_job_work_graph_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        _require_dataset_job_spec_ref(self.dataset_job_spec_ref)
        _require_ref_version(
            self.resolved_job_plan_ref,
            "resolved-dataset-job-plan",
            "v2",
            "resolved_job_plan_ref",
        )
        if self.dataset_job_spec_ref.object_id != self.job_id or len(self.item_ids) != len(
            self.source_trace_refs
        ):
            raise ValueError("work graph job/item/source identity is inconsistent")
        _require_unique_values("work graph item IDs", self.item_ids)
        _require_sorted_unique_refs(
            "work graph source trace refs",
            self.source_trace_refs,
        )
        expected_items = tuple(dataset_item_id_v2(self.job_id, ref) for ref in self.source_trace_refs)
        if self.item_ids != expected_items:
            raise ValueError("work graph item IDs must match exact source traces")
        seen_refs: set[tuple[str, str, str, str]] = set()
        for unit in self.work_units:
            validate_resolved_work_unit_v2_identity(unit)
            if unit.job_id != self.job_id or unit.scope is WorkUnitScopeV2.ARTIFACT_GROUP:
                raise ValueError("work graph contains a cross-job or artifact unit")
            ref = resolved_work_unit_v2_ref(unit)
            key = _ref_key(ref)
            if key in seen_refs:
                raise ValueError("work graph unit refs must be unique")
            for dependency in unit.depends_on_work_unit_refs:
                if _ref_key(dependency) not in seen_refs:
                    raise ValueError("work graph dependencies must reference preceding units")
            seen_refs.add(key)
        expected_audit = _sorted_refs((self.dataset_job_spec_ref, self.resolved_job_plan_ref))
        if self.audit.input_refs != expected_audit:
            raise ValueError("work graph audit refs are incomplete")
        validate_resolved_job_work_graph_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        job_id: Identifier,
        dataset_job_spec_ref: ObjectRef,
        resolved_job_plan_ref: ObjectRef,
        item_ids: tuple[Identifier, ...],
        source_trace_refs: tuple[ObjectRef, ...],
        work_units: tuple[ResolvedWorkUnitV2, ...],
        audit: ContractAudit,
    ) -> ResolvedJobWorkGraphV2:
        value = cls(
            resolved_job_work_graph_id="resolved-job-work-graph://pending",
            job_id=job_id,
            dataset_job_spec_ref=dataset_job_spec_ref,
            resolved_job_plan_ref=resolved_job_plan_ref,
            item_ids=item_ids,
            source_trace_refs=source_trace_refs,
            work_units=work_units,
            resolved_job_work_graph_sha256="0" * 64,
            audit=audit,
        )
        digest = resolved_job_work_graph_v2_carried_sha256(value)
        return value.model_copy(
            update={
                "resolved_job_work_graph_id": (f"resolved-job-work-graph://sha256/{digest}"),
                "resolved_job_work_graph_sha256": digest,
            }
        )


class ArtifactGroupFanoutV2(ContractModelV2):
    schema_version: Literal["eval-factory/artifact-group-fanout/v2"] = "eval-factory/artifact-group-fanout/v2"
    artifact_group_fanout_id: Identifier
    resolved_job_work_graph_ref: ObjectRef
    parent_attachment_work_unit_ref: ObjectRef
    artifact_execution_plan_ref: ObjectRef
    group_work_units: tuple[ResolvedWorkUnitV2, ...] = Field(min_length=1)
    policy_version: Literal["dataset-work-graph/r6-02-v1"] = R6_WORK_GRAPH_POLICY_VERSION
    artifact_group_fanout_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_fanout(self) -> Self:
        for ref, object_type, field_name in (
            (
                self.resolved_job_work_graph_ref,
                "resolved-job-work-graph",
                "resolved_job_work_graph_ref",
            ),
            (
                self.parent_attachment_work_unit_ref,
                "resolved-work-unit",
                "parent_attachment_work_unit_ref",
            ),
            (
                self.artifact_execution_plan_ref,
                "artifact-execution-plan",
                "artifact_execution_plan_ref",
            ),
        ):
            _require_ref_version(ref, object_type, "v2", field_name)
        group_refs: list[ObjectRef] = []
        unit_refs: list[ObjectRef] = []
        for unit in self.group_work_units:
            if unit.scope is not WorkUnitScopeV2.ARTIFACT_GROUP:
                raise ValueError("artifact fanout contains a non-group work unit")
            if unit.depends_on_work_unit_refs:
                raise ValueError("artifact group work units cannot redefine R5 edges")
            group_refs.append(unit.artifact_execution_group_ref)  # type: ignore[arg-type]
            unit_refs.append(resolved_work_unit_v2_ref(unit))
        _require_sorted_unique_refs("artifact fanout group refs", tuple(group_refs))
        if len(unit_refs) != len(set(unit_refs)):
            raise ValueError("artifact fanout unit refs must be unique")
        expected_audit = _sorted_refs(
            (
                self.resolved_job_work_graph_ref,
                self.parent_attachment_work_unit_ref,
                self.artifact_execution_plan_ref,
                *tuple(group_refs),
            )
        )
        if self.audit.input_refs != expected_audit:
            raise ValueError("artifact fanout audit refs are incomplete")
        validate_artifact_group_fanout_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        resolved_job_work_graph_ref: ObjectRef,
        parent_attachment_work_unit_ref: ObjectRef,
        artifact_execution_plan_ref: ObjectRef,
        group_work_units: tuple[ResolvedWorkUnitV2, ...],
        audit: ContractAudit,
    ) -> ArtifactGroupFanoutV2:
        value = cls(
            artifact_group_fanout_id="artifact-group-fanout://pending",
            resolved_job_work_graph_ref=resolved_job_work_graph_ref,
            parent_attachment_work_unit_ref=parent_attachment_work_unit_ref,
            artifact_execution_plan_ref=artifact_execution_plan_ref,
            group_work_units=group_work_units,
            artifact_group_fanout_sha256="0" * 64,
            audit=audit,
        )
        digest = artifact_group_fanout_v2_carried_sha256(value)
        return value.model_copy(
            update={
                "artifact_group_fanout_id": (f"artifact-group-fanout://sha256/{digest}"),
                "artifact_group_fanout_sha256": digest,
            }
        )


class WorkReadinessSnapshotV2(ContractModelV2):
    schema_version: Literal["eval-factory/work-readiness-snapshot/v2"] = (
        "eval-factory/work-readiness-snapshot/v2"
    )
    work_readiness_snapshot_id: Identifier
    resolved_job_work_graph_ref: ObjectRef
    artifact_group_fanout_ref: ObjectRef | None = None
    semantic_review_fanout_ref: ObjectRef | None = None
    work_unit_ref: ObjectRef
    readiness: WorkReadinessV2
    succeeded_dependency_result_refs: tuple[ObjectRef, ...] = ()
    retryable_dependency_result_refs: tuple[ObjectRef, ...] = ()
    terminal_non_success_result_refs: tuple[ObjectRef, ...] = ()
    waiting_dependency_work_unit_refs: tuple[ObjectRef, ...] = ()
    skipped_item_refs: tuple[ObjectRef, ...] = ()
    policy_version: Literal["dataset-work-graph/r6-02-v1"] = R6_WORK_GRAPH_POLICY_VERSION
    work_readiness_snapshot_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        _require_ref_version(
            self.resolved_job_work_graph_ref,
            "resolved-job-work-graph",
            "v2",
            "resolved_job_work_graph_ref",
        )
        _require_ref_version(
            self.work_unit_ref,
            "resolved-work-unit",
            "v2",
            "work_unit_ref",
        )
        if self.artifact_group_fanout_ref is not None:
            _require_ref_version(
                self.artifact_group_fanout_ref,
                "artifact-group-fanout",
                "v2",
                "artifact_group_fanout_ref",
            )
        if self.semantic_review_fanout_ref is not None:
            _require_ref_version(
                self.semantic_review_fanout_ref,
                "semantic-review-fanout",
                "v2",
                "semantic_review_fanout_ref",
            )
        if self.artifact_group_fanout_ref is not None and self.semantic_review_fanout_ref is not None:
            raise ValueError("work readiness cannot bind two fanout sidecars")
        partitions = (
            self.succeeded_dependency_result_refs,
            self.retryable_dependency_result_refs,
            self.terminal_non_success_result_refs,
            self.waiting_dependency_work_unit_refs,
            self.skipped_item_refs,
        )
        keys: list[tuple[str, str, str, str]] = []
        for label, refs in zip(
            (
                "succeeded dependency refs",
                "retryable dependency refs",
                "terminal dependency refs",
                "waiting dependency refs",
                "skipped item refs",
            ),
            partitions,
            strict=True,
        ):
            _require_sorted_unique_refs(label, refs)
            keys.extend(_ref_key(ref) for ref in refs)
        for ref in self.succeeded_dependency_result_refs:
            if not _is_work_result_ref(ref):
                raise ValueError("readiness result partitions contain an invalid ref")
        for ref in self.retryable_dependency_result_refs:
            if not _is_work_result_ref(ref) and not (
                ref.object_type == "work-retry-decision" and ref.object_version == "v2"
            ):
                raise ValueError("readiness result partitions contain an invalid ref")
        for ref in self.terminal_non_success_result_refs:
            if not _is_work_result_ref(ref) and not (
                ref.object_type
                in {
                    "work-lease-event",
                    "work-retry-decision",
                }
                and ref.object_version == "v2"
            ):
                raise ValueError("readiness result partitions contain an invalid ref")
        for ref in self.waiting_dependency_work_unit_refs:
            _require_ref_version(
                ref,
                "resolved-work-unit",
                "v2",
                "waiting_dependency_work_unit_refs",
            )
        for ref in self.skipped_item_refs:
            _require_ref_version(
                ref,
                "item-record",
                "record/v1",
                "skipped_item_refs",
            )
        if len(keys) != len(set(keys)):
            raise ValueError("readiness partitions must be disjoint")
        has_waiting = bool(self.retryable_dependency_result_refs or self.waiting_dependency_work_unit_refs)
        if self.readiness is WorkReadinessV2.WAITING and not has_waiting:
            raise ValueError("WAITING readiness requires waiting evidence")
        if (
            self.readiness is WorkReadinessV2.BLOCKED_DEPENDENCY
            and not self.terminal_non_success_result_refs
            and not self.skipped_item_refs
        ):
            raise ValueError("dependency block requires terminal evidence")
        if self.readiness is WorkReadinessV2.SKIPPED_ITEM and (
            not self.skipped_item_refs or any(partitions[index] for index in range(4))
        ):
            raise ValueError("SKIPPED_ITEM requires only skipped item refs")
        if self.readiness is WorkReadinessV2.READY and has_waiting:
            raise ValueError("READY readiness cannot contain waiting evidence")
        expected_audit = _sorted_refs(
            (
                self.resolved_job_work_graph_ref,
                *((self.artifact_group_fanout_ref,) if self.artifact_group_fanout_ref else ()),
                *((self.semantic_review_fanout_ref,) if self.semantic_review_fanout_ref else ()),
                self.work_unit_ref,
                *self.succeeded_dependency_result_refs,
                *self.retryable_dependency_result_refs,
                *self.terminal_non_success_result_refs,
                *self.waiting_dependency_work_unit_refs,
                *self.skipped_item_refs,
            )
        )
        if self.audit.input_refs != expected_audit:
            raise ValueError("work readiness audit refs are incomplete")
        validate_work_readiness_snapshot_v2_identity(self)
        return self


class WorkControlPolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/work-control-policy/v2"] = "eval-factory/work-control-policy/v2"
    work_control_policy_id: Identifier
    resolved_job_work_graph_ref: ObjectRef
    lease_duration_seconds: int = Field(gt=0)
    heartbeat_extension_seconds: int = Field(gt=0)
    max_attempts: int = Field(ge=1)
    retry_delay_seconds: tuple[int, ...]
    retry_lease_expiry: bool
    policy_version: Literal["work-control/r6-05-v1"] = R6_WORK_CONTROL_POLICY_VERSION
    work_control_policy_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        _require_ref_version(
            self.resolved_job_work_graph_ref,
            "resolved-job-work-graph",
            "v2",
            "resolved_job_work_graph_ref",
        )
        if len(self.retry_delay_seconds) != self.max_attempts - 1:
            raise ValueError("retry_delay_seconds must contain max_attempts - 1 values")
        if any(delay < 0 for delay in self.retry_delay_seconds):
            raise ValueError("retry_delay_seconds cannot contain a negative value")
        _require_control_audit(
            self.audit,
            (self.resolved_job_work_graph_ref,),
            self.policy_version,
        )
        validate_work_control_policy_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        resolved_job_work_graph_ref: ObjectRef,
        lease_duration_seconds: int,
        heartbeat_extension_seconds: int,
        max_attempts: int,
        retry_delay_seconds: tuple[int, ...],
        retry_lease_expiry: bool,
        audit: ContractAudit,
    ) -> WorkControlPolicyV2:
        value = cls(
            work_control_policy_id="work-control-policy://pending",
            resolved_job_work_graph_ref=resolved_job_work_graph_ref,
            lease_duration_seconds=lease_duration_seconds,
            heartbeat_extension_seconds=heartbeat_extension_seconds,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
            retry_lease_expiry=retry_lease_expiry,
            work_control_policy_sha256="0" * 64,
            audit=audit,
        )
        digest = work_control_policy_v2_carried_sha256(value)
        return value.model_copy(
            update={
                "work_control_policy_id": f"work-control-policy://sha256/{digest}",
                "work_control_policy_sha256": digest,
            }
        )


class WorkDispatchDecisionV2(ContractModelV2):
    schema_version: Literal["eval-factory/work-dispatch-decision/v2"] = (
        "eval-factory/work-dispatch-decision/v2"
    )
    work_dispatch_decision_id: Identifier
    resolved_job_work_graph_ref: ObjectRef
    work_unit_ref: ObjectRef
    work_readiness_snapshot_ref: ObjectRef
    work_control_policy_ref: ObjectRef
    retry_decision_ref: ObjectRef | None = None
    attempt: int = Field(ge=1)
    eligible_at: datetime
    policy_version: Literal["work-control/r6-05-v1"] = R6_WORK_CONTROL_POLICY_VERSION
    work_dispatch_decision_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        for ref, object_type, field_name in (
            (
                self.resolved_job_work_graph_ref,
                "resolved-job-work-graph",
                "resolved_job_work_graph_ref",
            ),
            (self.work_unit_ref, "resolved-work-unit", "work_unit_ref"),
            (
                self.work_readiness_snapshot_ref,
                "work-readiness-snapshot",
                "work_readiness_snapshot_ref",
            ),
            (
                self.work_control_policy_ref,
                "work-control-policy",
                "work_control_policy_ref",
            ),
        ):
            _require_ref_version(ref, object_type, "v2", field_name)
        if self.attempt == 1 and self.retry_decision_ref is not None:
            raise ValueError("initial dispatch cannot bind a retry decision")
        if self.attempt > 1:
            if self.retry_decision_ref is None:
                raise ValueError("retry dispatch requires a retry decision")
            _require_ref_version(
                self.retry_decision_ref,
                "work-retry-decision",
                "v2",
                "retry_decision_ref",
            )
        refs = (
            self.resolved_job_work_graph_ref,
            self.work_unit_ref,
            self.work_readiness_snapshot_ref,
            self.work_control_policy_ref,
            *((self.retry_decision_ref,) if self.retry_decision_ref else ()),
        )
        _require_control_audit(self.audit, refs, self.policy_version)
        validate_work_dispatch_decision_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        resolved_job_work_graph_ref: ObjectRef,
        work_unit_ref: ObjectRef,
        work_readiness_snapshot_ref: ObjectRef,
        work_control_policy_ref: ObjectRef,
        retry_decision_ref: ObjectRef | None,
        attempt: int,
        eligible_at: datetime,
        audit: ContractAudit,
    ) -> WorkDispatchDecisionV2:
        value = cls(
            work_dispatch_decision_id="work-dispatch-decision://pending",
            resolved_job_work_graph_ref=resolved_job_work_graph_ref,
            work_unit_ref=work_unit_ref,
            work_readiness_snapshot_ref=work_readiness_snapshot_ref,
            work_control_policy_ref=work_control_policy_ref,
            retry_decision_ref=retry_decision_ref,
            attempt=attempt,
            eligible_at=eligible_at,
            work_dispatch_decision_sha256="0" * 64,
            audit=audit,
        )
        digest = work_dispatch_decision_v2_carried_sha256(value)
        return value.model_copy(
            update={
                "work_dispatch_decision_id": f"work-dispatch-decision://sha256/{digest}",
                "work_dispatch_decision_sha256": digest,
            }
        )


class WorkLeaseV2(ContractModelV2):
    schema_version: Literal["eval-factory/work-lease/v2"] = "eval-factory/work-lease/v2"
    work_lease_id: Identifier
    work_dispatch_decision_ref: ObjectRef
    work_unit_ref: ObjectRef
    work_control_policy_ref: ObjectRef
    holder_ref: ObjectRef
    fencing_token: int = Field(ge=1)
    attempt: int = Field(ge=1)
    stage_run_ref: ObjectRef | None = None
    acquired_at: datetime
    expires_at: datetime
    policy_version: Literal["work-control/r6-05-v1"] = R6_WORK_CONTROL_POLICY_VERSION
    work_lease_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_lease(self) -> Self:
        for ref, object_type, object_version, field_name in (
            (
                self.work_dispatch_decision_ref,
                "work-dispatch-decision",
                "v2",
                "work_dispatch_decision_ref",
            ),
            (self.work_unit_ref, "resolved-work-unit", "v2", "work_unit_ref"),
            (
                self.work_control_policy_ref,
                "work-control-policy",
                "v2",
                "work_control_policy_ref",
            ),
            (self.holder_ref, "worker-principal", "v1", "holder_ref"),
        ):
            _require_ref_version(ref, object_type, object_version, field_name)
        if self.stage_run_ref is not None:
            _require_ref_version(
                self.stage_run_ref,
                "stage-run",
                "identity/v1",
                "stage_run_ref",
            )
        if self.expires_at <= self.acquired_at:
            raise ValueError("expires_at must be after acquired_at")
        refs = (
            self.work_dispatch_decision_ref,
            self.work_unit_ref,
            self.work_control_policy_ref,
            self.holder_ref,
            *((self.stage_run_ref,) if self.stage_run_ref else ()),
        )
        _require_control_audit(self.audit, refs, self.policy_version)
        validate_work_lease_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        work_dispatch_decision_ref: ObjectRef,
        work_unit_ref: ObjectRef,
        work_control_policy_ref: ObjectRef,
        holder_ref: ObjectRef,
        fencing_token: int,
        attempt: int,
        stage_run_ref: ObjectRef | None,
        acquired_at: datetime,
        expires_at: datetime,
        audit: ContractAudit,
    ) -> WorkLeaseV2:
        value = cls(
            work_lease_id="work-lease://pending",
            work_dispatch_decision_ref=work_dispatch_decision_ref,
            work_unit_ref=work_unit_ref,
            work_control_policy_ref=work_control_policy_ref,
            holder_ref=holder_ref,
            fencing_token=fencing_token,
            attempt=attempt,
            stage_run_ref=stage_run_ref,
            acquired_at=acquired_at,
            expires_at=expires_at,
            work_lease_sha256="0" * 64,
            audit=audit,
        )
        digest = work_lease_v2_carried_sha256(value)
        return value.model_copy(
            update={
                "work_lease_id": f"work-lease://sha256/{digest}",
                "work_lease_sha256": digest,
            }
        )


class WorkLeaseEventV2(ContractModelV2):
    schema_version: Literal["eval-factory/work-lease-event/v2"] = "eval-factory/work-lease-event/v2"
    work_lease_event_id: Identifier
    work_lease_ref: ObjectRef
    work_unit_ref: ObjectRef
    event_kind: WorkLeaseEventKindV2
    lease_version: int = Field(ge=1)
    fencing_token: int = Field(ge=1)
    effective_expires_at: datetime | None = None
    result_refs: tuple[ObjectRef, ...] = ()
    failure: FailureRecord | None = None
    work_control_policy_ref: ObjectRef
    policy_version: Literal["work-control/r6-05-v1"] = R6_WORK_CONTROL_POLICY_VERSION
    work_lease_event_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_event(self) -> Self:
        _require_ref_version(self.work_lease_ref, "work-lease", "v2", "work_lease_ref")
        _require_ref_version(self.work_unit_ref, "resolved-work-unit", "v2", "work_unit_ref")
        _require_ref_version(
            self.work_control_policy_ref,
            "work-control-policy",
            "v2",
            "work_control_policy_ref",
        )
        _require_sorted_unique_refs("work lease event result refs", self.result_refs)
        for ref in self.result_refs:
            if not _is_work_result_ref(ref):
                raise ValueError("work lease event result_refs contain an invalid ref")
        if self.event_kind is WorkLeaseEventKindV2.HEARTBEAT:
            if self.effective_expires_at is None or self.result_refs or self.failure is not None:
                raise ValueError("heartbeat cannot carry result or failure")
        elif self.event_kind is WorkLeaseEventKindV2.SUCCEEDED:
            if self.effective_expires_at is not None or not self.result_refs or self.failure is not None:
                raise ValueError("successful lease event requires only result refs")
        else:
            if self.effective_expires_at is not None or self.failure is None:
                raise ValueError("non-success lease event requires a failure")
            if self.event_kind is WorkLeaseEventKindV2.RETRYABLE_FAILURE and not self.failure.retryable:
                raise ValueError("retryable lease event requires retryable failure")
            if (
                self.failure.message != R6_WORK_CONTROL_FAILURE_MESSAGE
                or self.failure.evidence_refs
                or self.failure.detail
            ):
                raise ValueError("work lease event failure must be content-free")
        refs = (
            self.work_lease_ref,
            self.work_unit_ref,
            self.work_control_policy_ref,
            *self.result_refs,
        )
        _require_control_audit(self.audit, refs, self.policy_version)
        validate_work_lease_event_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        work_lease_ref: ObjectRef,
        work_unit_ref: ObjectRef,
        event_kind: WorkLeaseEventKindV2,
        lease_version: int,
        fencing_token: int,
        effective_expires_at: datetime | None,
        result_refs: tuple[ObjectRef, ...],
        failure: FailureRecord | None,
        work_control_policy_ref: ObjectRef,
        audit: ContractAudit,
    ) -> WorkLeaseEventV2:
        value = cls(
            work_lease_event_id="work-lease-event://pending",
            work_lease_ref=work_lease_ref,
            work_unit_ref=work_unit_ref,
            event_kind=event_kind,
            lease_version=lease_version,
            fencing_token=fencing_token,
            effective_expires_at=effective_expires_at,
            result_refs=_sorted_refs(result_refs),
            failure=failure,
            work_control_policy_ref=work_control_policy_ref,
            work_lease_event_sha256="0" * 64,
            audit=audit,
        )
        digest = work_lease_event_v2_carried_sha256(value)
        return value.model_copy(
            update={
                "work_lease_event_id": f"work-lease-event://sha256/{digest}",
                "work_lease_event_sha256": digest,
            }
        )


class WorkRetryDecisionV2(ContractModelV2):
    schema_version: Literal["eval-factory/work-retry-decision/v2"] = "eval-factory/work-retry-decision/v2"
    work_retry_decision_id: Identifier
    work_unit_ref: ObjectRef
    prior_lease_event_ref: ObjectRef
    prior_result_refs: tuple[ObjectRef, ...] = ()
    work_control_policy_ref: ObjectRef
    decision: WorkRetryDecisionKindV2
    completed_attempt: int = Field(ge=1)
    next_attempt: int | None = Field(default=None, ge=2)
    eligible_at: datetime | None = None
    policy_version: Literal["work-control/r6-05-v1"] = R6_WORK_CONTROL_POLICY_VERSION
    work_retry_decision_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        _require_ref_version(self.work_unit_ref, "resolved-work-unit", "v2", "work_unit_ref")
        _require_ref_version(
            self.prior_lease_event_ref,
            "work-lease-event",
            "v2",
            "prior_lease_event_ref",
        )
        _require_ref_version(
            self.work_control_policy_ref,
            "work-control-policy",
            "v2",
            "work_control_policy_ref",
        )
        _require_sorted_unique_refs("retry decision prior result refs", self.prior_result_refs)
        for ref in self.prior_result_refs:
            if not _is_work_result_ref(ref):
                raise ValueError("retry decision prior_result_refs contain an invalid ref")
        if self.decision is WorkRetryDecisionKindV2.RETRY_SCHEDULED:
            if self.next_attempt != self.completed_attempt + 1 or self.eligible_at is None:
                raise ValueError("RETRY_SCHEDULED requires the next attempt and eligible_at")
        elif self.next_attempt is not None or self.eligible_at is not None:
            raise ValueError("EXHAUSTED cannot carry next attempt or eligible_at")
        refs = (
            self.work_unit_ref,
            self.prior_lease_event_ref,
            self.work_control_policy_ref,
            *self.prior_result_refs,
        )
        _require_control_audit(self.audit, refs, self.policy_version)
        validate_work_retry_decision_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        work_unit_ref: ObjectRef,
        prior_lease_event_ref: ObjectRef,
        prior_result_refs: tuple[ObjectRef, ...],
        work_control_policy_ref: ObjectRef,
        decision: WorkRetryDecisionKindV2,
        completed_attempt: int,
        next_attempt: int | None,
        eligible_at: datetime | None,
        audit: ContractAudit,
    ) -> WorkRetryDecisionV2:
        value = cls(
            work_retry_decision_id="work-retry-decision://pending",
            work_unit_ref=work_unit_ref,
            prior_lease_event_ref=prior_lease_event_ref,
            prior_result_refs=_sorted_refs(prior_result_refs),
            work_control_policy_ref=work_control_policy_ref,
            decision=decision,
            completed_attempt=completed_attempt,
            next_attempt=next_attempt,
            eligible_at=eligible_at,
            work_retry_decision_sha256="0" * 64,
            audit=audit,
        )
        digest = work_retry_decision_v2_carried_sha256(value)
        return value.model_copy(
            update={
                "work_retry_decision_id": f"work-retry-decision://sha256/{digest}",
                "work_retry_decision_sha256": digest,
            }
        )


class WorkCancellationRecordV2(ContractModelV2):
    schema_version: Literal["eval-factory/work-cancellation-record/v2"] = (
        "eval-factory/work-cancellation-record/v2"
    )
    work_cancellation_record_id: Identifier
    resolved_job_work_graph_ref: ObjectRef
    work_control_policy_ref: ObjectRef
    reason_code: Identifier
    cancelled_lease_event_refs: tuple[ObjectRef, ...] = ()
    preserved_result_refs: tuple[ObjectRef, ...] = ()
    policy_version: Literal["work-control/r6-05-v1"] = R6_WORK_CONTROL_POLICY_VERSION
    work_cancellation_record_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        _require_ref_version(
            self.resolved_job_work_graph_ref,
            "resolved-job-work-graph",
            "v2",
            "resolved_job_work_graph_ref",
        )
        _require_ref_version(
            self.work_control_policy_ref,
            "work-control-policy",
            "v2",
            "work_control_policy_ref",
        )
        _require_sorted_unique_refs(
            "cancelled lease event refs",
            self.cancelled_lease_event_refs,
        )
        _require_sorted_unique_refs("preserved result refs", self.preserved_result_refs)
        for ref in self.cancelled_lease_event_refs:
            _require_ref_version(
                ref,
                "work-lease-event",
                "v2",
                "cancelled_lease_event_refs",
            )
        for ref in self.preserved_result_refs:
            if not _is_work_result_ref(ref):
                raise ValueError("preserved_result_refs contain an invalid ref")
        refs = (
            self.resolved_job_work_graph_ref,
            self.work_control_policy_ref,
            *self.cancelled_lease_event_refs,
            *self.preserved_result_refs,
        )
        _require_control_audit(self.audit, refs, self.policy_version)
        validate_work_cancellation_record_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        resolved_job_work_graph_ref: ObjectRef,
        work_control_policy_ref: ObjectRef,
        reason_code: Identifier,
        cancelled_lease_event_refs: tuple[ObjectRef, ...],
        preserved_result_refs: tuple[ObjectRef, ...],
        audit: ContractAudit,
    ) -> WorkCancellationRecordV2:
        value = cls(
            work_cancellation_record_id="work-cancellation-record://pending",
            resolved_job_work_graph_ref=resolved_job_work_graph_ref,
            work_control_policy_ref=work_control_policy_ref,
            reason_code=reason_code,
            cancelled_lease_event_refs=_sorted_refs(cancelled_lease_event_refs),
            preserved_result_refs=_sorted_refs(preserved_result_refs),
            work_cancellation_record_sha256="0" * 64,
            audit=audit,
        )
        digest = work_cancellation_record_v2_carried_sha256(value)
        return value.model_copy(
            update={
                "work_cancellation_record_id": (f"work-cancellation-record://sha256/{digest}"),
                "work_cancellation_record_sha256": digest,
            }
        )


def trace_source_v2_ref(trace: TraceSourceRef) -> ObjectRef:
    return ObjectRef(
        object_type="trace-source",
        object_id=trace.source_trace_id,
        object_version=trace.adapter_version,
        object_sha256=trace.raw_sha256,
    )


def dataset_item_id_v2(job_id: str, source_trace_ref: ObjectRef) -> str:
    digest = _payload_sha256(
        {
            "job_id": job_id,
            "source_trace_ref": _ref_payload(source_trace_ref),
        }
    )
    return f"item://r6-02/sha256/{digest}"


def resolved_work_unit_v2_carried_sha256(unit: ResolvedWorkUnitV2) -> str:
    excluded = {
        "resolved_work_unit_id",
        "resolved_work_unit_sha256",
    }
    if unit.semantic_review_round is None:
        excluded.add("semantic_review_round")
    return _payload_sha256(
        unit.model_dump(
            mode="python",
            exclude=excluded,
        )
    )


def resolved_work_unit_v2_ref(unit: ResolvedWorkUnitV2) -> ObjectRef:
    validate_resolved_work_unit_v2_identity(unit)
    return ObjectRef(
        object_type="resolved-work-unit",
        object_id=unit.resolved_work_unit_id,
        object_version="v2",
        object_sha256=unit.resolved_work_unit_sha256,
    )


def validate_resolved_work_unit_v2_identity(unit: ResolvedWorkUnitV2) -> None:
    _validate_carried_identity(
        object_id=unit.resolved_work_unit_id,
        object_sha256=unit.resolved_work_unit_sha256,
        prefix="resolved-work-unit",
        observed=resolved_work_unit_v2_carried_sha256(unit),
    )


def resolved_job_work_graph_v2_carried_sha256(
    graph: ResolvedJobWorkGraphV2,
) -> str:
    payload = graph.model_dump(
        mode="python",
        exclude={
            "resolved_job_work_graph_id",
            "resolved_job_work_graph_sha256",
            "audit",
        },
    )
    return _payload_sha256(_strip_empty_semantic_review_rounds(payload))


def resolved_job_work_graph_v2_ref(graph: ResolvedJobWorkGraphV2) -> ObjectRef:
    validate_resolved_job_work_graph_v2_identity(graph)
    return ObjectRef(
        object_type="resolved-job-work-graph",
        object_id=graph.resolved_job_work_graph_id,
        object_version="v2",
        object_sha256=graph.resolved_job_work_graph_sha256,
    )


def validate_resolved_job_work_graph_v2_identity(
    graph: ResolvedJobWorkGraphV2,
) -> None:
    _validate_carried_identity(
        object_id=graph.resolved_job_work_graph_id,
        object_sha256=graph.resolved_job_work_graph_sha256,
        prefix="resolved-job-work-graph",
        observed=resolved_job_work_graph_v2_carried_sha256(graph),
    )


def artifact_group_fanout_v2_carried_sha256(
    fanout: ArtifactGroupFanoutV2,
) -> str:
    payload = fanout.model_dump(
        mode="python",
        exclude={
            "artifact_group_fanout_id",
            "artifact_group_fanout_sha256",
            "audit",
        },
    )
    return _payload_sha256(_strip_empty_semantic_review_rounds(payload))


def artifact_group_fanout_v2_ref(fanout: ArtifactGroupFanoutV2) -> ObjectRef:
    validate_artifact_group_fanout_v2_identity(fanout)
    return ObjectRef(
        object_type="artifact-group-fanout",
        object_id=fanout.artifact_group_fanout_id,
        object_version="v2",
        object_sha256=fanout.artifact_group_fanout_sha256,
    )


def validate_artifact_group_fanout_v2_identity(
    fanout: ArtifactGroupFanoutV2,
) -> None:
    _validate_carried_identity(
        object_id=fanout.artifact_group_fanout_id,
        object_sha256=fanout.artifact_group_fanout_sha256,
        prefix="artifact-group-fanout",
        observed=artifact_group_fanout_v2_carried_sha256(fanout),
    )


def work_readiness_snapshot_v2_carried_sha256(
    snapshot: WorkReadinessSnapshotV2,
) -> str:
    excluded = {
        "work_readiness_snapshot_id",
        "work_readiness_snapshot_sha256",
        "audit",
    }
    if snapshot.semantic_review_fanout_ref is None:
        excluded.add("semantic_review_fanout_ref")
    return _payload_sha256(
        snapshot.model_dump(
            mode="python",
            exclude=excluded,
        )
    )


def work_readiness_snapshot_v2_ref(
    snapshot: WorkReadinessSnapshotV2,
) -> ObjectRef:
    validate_work_readiness_snapshot_v2_identity(snapshot)
    return ObjectRef(
        object_type="work-readiness-snapshot",
        object_id=snapshot.work_readiness_snapshot_id,
        object_version="v2",
        object_sha256=snapshot.work_readiness_snapshot_sha256,
    )


def validate_work_readiness_snapshot_v2_identity(
    snapshot: WorkReadinessSnapshotV2,
) -> None:
    _validate_carried_identity(
        object_id=snapshot.work_readiness_snapshot_id,
        object_sha256=snapshot.work_readiness_snapshot_sha256,
        prefix="work-readiness-snapshot",
        observed=work_readiness_snapshot_v2_carried_sha256(snapshot),
    )


def work_control_policy_v2_carried_sha256(
    policy: WorkControlPolicyV2,
) -> str:
    return _control_carried_sha256(
        policy,
        "work_control_policy_id",
        "work_control_policy_sha256",
    )


def work_control_policy_v2_ref(policy: WorkControlPolicyV2) -> ObjectRef:
    validate_work_control_policy_v2_identity(policy)
    return ObjectRef(
        object_type="work-control-policy",
        object_id=policy.work_control_policy_id,
        object_version="v2",
        object_sha256=policy.work_control_policy_sha256,
    )


def validate_work_control_policy_v2_identity(policy: WorkControlPolicyV2) -> None:
    _validate_carried_identity(
        object_id=policy.work_control_policy_id,
        object_sha256=policy.work_control_policy_sha256,
        prefix="work-control-policy",
        observed=work_control_policy_v2_carried_sha256(policy),
    )


def work_dispatch_decision_v2_carried_sha256(
    decision: WorkDispatchDecisionV2,
) -> str:
    return _control_carried_sha256(
        decision,
        "work_dispatch_decision_id",
        "work_dispatch_decision_sha256",
    )


def work_dispatch_decision_v2_ref(
    decision: WorkDispatchDecisionV2,
) -> ObjectRef:
    validate_work_dispatch_decision_v2_identity(decision)
    return ObjectRef(
        object_type="work-dispatch-decision",
        object_id=decision.work_dispatch_decision_id,
        object_version="v2",
        object_sha256=decision.work_dispatch_decision_sha256,
    )


def validate_work_dispatch_decision_v2_identity(
    decision: WorkDispatchDecisionV2,
) -> None:
    _validate_carried_identity(
        object_id=decision.work_dispatch_decision_id,
        object_sha256=decision.work_dispatch_decision_sha256,
        prefix="work-dispatch-decision",
        observed=work_dispatch_decision_v2_carried_sha256(decision),
    )


def work_lease_v2_carried_sha256(lease: WorkLeaseV2) -> str:
    return _control_carried_sha256(
        lease,
        "work_lease_id",
        "work_lease_sha256",
    )


def work_lease_v2_ref(lease: WorkLeaseV2) -> ObjectRef:
    validate_work_lease_v2_identity(lease)
    return ObjectRef(
        object_type="work-lease",
        object_id=lease.work_lease_id,
        object_version="v2",
        object_sha256=lease.work_lease_sha256,
    )


def validate_work_lease_v2_identity(lease: WorkLeaseV2) -> None:
    _validate_carried_identity(
        object_id=lease.work_lease_id,
        object_sha256=lease.work_lease_sha256,
        prefix="work-lease",
        observed=work_lease_v2_carried_sha256(lease),
    )


def work_lease_event_v2_carried_sha256(event: WorkLeaseEventV2) -> str:
    return _control_carried_sha256(
        event,
        "work_lease_event_id",
        "work_lease_event_sha256",
    )


def work_lease_event_v2_ref(event: WorkLeaseEventV2) -> ObjectRef:
    validate_work_lease_event_v2_identity(event)
    return ObjectRef(
        object_type="work-lease-event",
        object_id=event.work_lease_event_id,
        object_version="v2",
        object_sha256=event.work_lease_event_sha256,
    )


def validate_work_lease_event_v2_identity(event: WorkLeaseEventV2) -> None:
    _validate_carried_identity(
        object_id=event.work_lease_event_id,
        object_sha256=event.work_lease_event_sha256,
        prefix="work-lease-event",
        observed=work_lease_event_v2_carried_sha256(event),
    )


def work_retry_decision_v2_carried_sha256(
    decision: WorkRetryDecisionV2,
) -> str:
    return _control_carried_sha256(
        decision,
        "work_retry_decision_id",
        "work_retry_decision_sha256",
    )


def work_retry_decision_v2_ref(
    decision: WorkRetryDecisionV2,
) -> ObjectRef:
    validate_work_retry_decision_v2_identity(decision)
    return ObjectRef(
        object_type="work-retry-decision",
        object_id=decision.work_retry_decision_id,
        object_version="v2",
        object_sha256=decision.work_retry_decision_sha256,
    )


def validate_work_retry_decision_v2_identity(
    decision: WorkRetryDecisionV2,
) -> None:
    _validate_carried_identity(
        object_id=decision.work_retry_decision_id,
        object_sha256=decision.work_retry_decision_sha256,
        prefix="work-retry-decision",
        observed=work_retry_decision_v2_carried_sha256(decision),
    )


def work_cancellation_record_v2_carried_sha256(
    record: WorkCancellationRecordV2,
) -> str:
    return _control_carried_sha256(
        record,
        "work_cancellation_record_id",
        "work_cancellation_record_sha256",
    )


def work_cancellation_record_v2_ref(
    record: WorkCancellationRecordV2,
) -> ObjectRef:
    validate_work_cancellation_record_v2_identity(record)
    return ObjectRef(
        object_type="work-cancellation-record",
        object_id=record.work_cancellation_record_id,
        object_version="v2",
        object_sha256=record.work_cancellation_record_sha256,
    )


def validate_work_cancellation_record_v2_identity(
    record: WorkCancellationRecordV2,
) -> None:
    _validate_carried_identity(
        object_id=record.work_cancellation_record_id,
        object_sha256=record.work_cancellation_record_sha256,
        prefix="work-cancellation-record",
        observed=work_cancellation_record_v2_carried_sha256(record),
    )


def _control_carried_sha256(
    value: ContractModelV2,
    id_field: str,
    sha_field: str,
) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="python",
            exclude={id_field, sha_field, "audit"},
        )
    )


def _strip_empty_semantic_review_rounds(
    value: object,
) -> object:
    if isinstance(value, dict):
        return {
            key: _strip_empty_semantic_review_rounds(item)
            for key, item in value.items()
            if not (key == "semantic_review_round" and item is None)
        }
    if isinstance(value, (list, tuple)):
        return [_strip_empty_semantic_review_rounds(item) for item in value]
    return value


def _payload_sha256(value: object) -> str:
    encoded = json.dumps(
        canonical_value_v2(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _ref_payload(ref: ObjectRef) -> dict[str, object]:
    return ref.model_dump(mode="python", exclude_none=False)


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _sorted_refs(refs: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    return tuple(sorted(refs, key=_ref_key))


def _require_sorted_unique_refs(label: str, refs: tuple[ObjectRef, ...]) -> None:
    if refs != _sorted_refs(refs) or len(refs) != len(set(refs)):
        raise ValueError(f"{label} must be sorted and unique")


def _require_unique_values(label: str, values: tuple[str, ...]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


def _require_ref_version(
    ref: ObjectRef,
    object_type: str,
    object_version: str,
    field_name: str,
) -> None:
    if ref.object_type != object_type or ref.object_version != object_version:
        raise ValueError(f"{field_name} must reference {object_type} {object_version}")


def _is_work_result_ref(ref: ObjectRef) -> bool:
    return (ref.object_type == "stage-result" and ref.object_version == "record/v1") or (
        ref.object_type
        in {
            "artifact-execution-batch",
            "artifact-execution-receipt",
            "model-admission-decision",
            "model-control-receipt",
            "resource-admission-decision",
        }
        and ref.object_version == "v2"
    )


def _require_control_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
    policy_version: str,
) -> None:
    if audit.input_refs != _sorted_refs(refs):
        raise ValueError("work control audit refs are incomplete")
    bindings = tuple(binding for binding in audit.governing_versions if binding.component == "work-control")
    if len(bindings) != 1 or bindings[0].version != policy_version:
        raise ValueError("work control audit is missing the current policy")


def _validate_carried_identity(
    *,
    object_id: str,
    object_sha256: str,
    prefix: str,
    observed: str,
) -> None:
    if object_id == f"{prefix}://pending" and object_sha256 == "0" * 64:
        return
    if object_sha256 != observed or object_id != f"{prefix}://sha256/{observed}":
        raise ValueError(f"{prefix.replace('-', ' ')} identity is stale")
