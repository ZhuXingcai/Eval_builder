from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, ClassVar, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from eval_factory.contracts.agent_system_v2 import FactoryRunStatusV2
from eval_factory.contracts.core import (
    ContractAudit,
    Identifier,
    ObjectRef,
    Sha256,
)
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2

SafeCode = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Z][A-Z0-9_]{1,127}$"),
]

_DENIED_REF_MARKERS = (
    "credential",
    "final-answer",
    "final-output",
    "grader-rule",
    "hidden-condition",
    "model-output-body",
    "private-reference",
    "prompt-body",
    "rag-text",
    "raw-trace",
    "raw-traj",
    "retrieved-text",
    "runtime-transcript",
    "secret",
    "store-path",
)


class CandidateDatasetOutcomeV2(StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    NO_ELIGIBLE_ITEMS = "NO_ELIGIBLE_ITEMS"
    BLOCKED = "BLOCKED"


class FactoryItemStageV2(StrEnum):
    CORE_SELECTION = "CORE_SELECTION"
    TASK_AUTHORING = "TASK_AUTHORING"
    ATTACHMENT = "ATTACHMENT"
    ITEM_QUALITY = "ITEM_QUALITY"
    CRITERIA_RUBRIC = "CRITERIA_RUBRIC"
    GRADING_DESIGN = "GRADING_DESIGN"
    RELEASE_CANDIDATE = "RELEASE_CANDIDATE"


class FactoryItemStageOutcomeV2(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    REJECTED = "REJECTED"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"
    BLOCKED_POLICY = "BLOCKED_POLICY"
    FAILED = "FAILED"


class FactoryDatasetNextActionV2(StrEnum):
    ADVANCE = "ADVANCE"
    REVIEW_PLAN = "REVIEW_PLAN"
    INSPECT_OUTPUT = "INSPECT_OUTPUT"
    NONE = "NONE"


class _DatasetRuntimeObjectV2(ContractModelV2):
    object_id: Identifier
    object_sha256: Sha256
    audit: ContractAudit

    OBJECT_TYPE: ClassVar[str]
    OBJECT_VERSION: ClassVar[str] = "v2"

    @classmethod
    def _create(
        cls,
        *,
        audit: ContractAudit,
        values: dict[str, object],
    ) -> Self:
        provisional = cls.model_construct(
            object_id=f"{cls.OBJECT_TYPE}://pending",
            object_sha256="0" * 64,
            audit=audit,
            **values,  # type: ignore[arg-type]
        )
        refs = _collect_refs(provisional)
        safe_audit = _audit_with_refs(audit, refs)
        provisional = cls.model_construct(
            object_id=f"{cls.OBJECT_TYPE}://pending",
            object_sha256="0" * 64,
            audit=safe_audit,
            **values,  # type: ignore[arg-type]
        )
        digest = _carried_sha256(provisional)
        return cls(
            object_id=f"{cls.OBJECT_TYPE}://sha256/{digest}",
            object_sha256=digest,
            audit=safe_audit,
            **values,
        )

    @model_validator(mode="after")
    def validate_derived_identity(self) -> Self:
        digest = _carried_sha256(self)
        if self.object_sha256 != digest or self.object_id != f"{self.OBJECT_TYPE}://sha256/{digest}":
            raise ValueError("dataset runtime object identity is stale")
        refs = _collect_refs(self)
        _require_safe_refs(refs)
        if self.audit.input_refs != refs:
            raise ValueError("dataset runtime audit refs differ from object refs")
        return self

    def to_ref(self) -> ObjectRef:
        return ObjectRef(
            object_type=self.OBJECT_TYPE,
            object_id=self.object_id,
            object_version=self.OBJECT_VERSION,
            object_sha256=self.object_sha256,
        )


class FactoryDatasetRunRequestV2(_DatasetRuntimeObjectV2):
    schema_version: Literal["eval-factory/factory-dataset-run-request/v2"] = (
        "eval-factory/factory-dataset-run-request/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "factory-dataset-run-request"

    dataset_run_id: Identifier
    requirement_spec_ref: ObjectRef
    manifest_ref: ObjectRef
    source_authorization_ref: ObjectRef
    factory_policy_ref: ObjectRef
    pipeline_policy_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=128,
    )
    gateway_registry_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=32,
    )
    output_target_ref: ObjectRef
    idempotency_key: Identifier
    max_transitions: int = Field(ge=1, le=100_000)

    @classmethod
    def create(
        cls,
        *,
        dataset_run_id: str,
        requirement_spec_ref: ObjectRef,
        manifest_ref: ObjectRef,
        source_authorization_ref: ObjectRef,
        factory_policy_ref: ObjectRef,
        pipeline_policy_refs: tuple[ObjectRef, ...],
        gateway_registry_refs: tuple[ObjectRef, ...],
        output_target_ref: ObjectRef,
        idempotency_key: str,
        max_transitions: int,
        audit: ContractAudit,
    ) -> FactoryDatasetRunRequestV2:
        return cls._create(
            audit=audit,
            values={
                "dataset_run_id": dataset_run_id,
                "requirement_spec_ref": requirement_spec_ref,
                "manifest_ref": manifest_ref,
                "source_authorization_ref": source_authorization_ref,
                "factory_policy_ref": factory_policy_ref,
                "pipeline_policy_refs": _sorted_refs(
                    pipeline_policy_refs,
                ),
                "gateway_registry_refs": _sorted_refs(
                    gateway_registry_refs,
                ),
                "output_target_ref": output_target_ref,
                "idempotency_key": idempotency_key,
                "max_transitions": max_transitions,
            },
        )

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        _require_ref(
            self.requirement_spec_ref,
            "evaluation-requirement-spec",
            "requirement_spec_ref",
        )
        _require_ref(
            self.manifest_ref,
            "trace-manifest",
            "manifest_ref",
        )
        _require_ref(
            self.source_authorization_ref,
            "trace-source-authorization",
            "source_authorization_ref",
        )
        _require_ref(
            self.factory_policy_ref,
            "factory-run-policy",
            "factory_policy_ref",
        )
        _require_sorted_unique_refs(
            self.pipeline_policy_refs,
            "pipeline_policy_refs",
        )
        if any(not ref.object_type.endswith("-policy") for ref in self.pipeline_policy_refs):
            raise ValueError("pipeline policy refs require policy object types")
        _require_sorted_unique_refs(
            self.gateway_registry_refs,
            "gateway_registry_refs",
        )
        allowed_registries = {
            "agent-registry",
            "model-catalog",
            "prompt-registry",
            "rag-registry",
        }
        if any(ref.object_type not in allowed_registries for ref in self.gateway_registry_refs):
            raise ValueError("gateway registry refs contain an unsupported type")
        _require_ref(
            self.output_target_ref,
            "candidate-output-target",
            "output_target_ref",
        )
        return self


class FactoryDatasetPlanningAuthorityV2(_DatasetRuntimeObjectV2):
    schema_version: Literal["eval-factory/factory-dataset-planning-authority/v2"] = (
        "eval-factory/factory-dataset-planning-authority/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "factory-dataset-planning-authority"

    dataset_run_ref: ObjectRef
    planning_version: int = Field(ge=1, le=1_000_000)
    predecessor_authority_ref: ObjectRef | None = None
    plan_ref: ObjectRef
    route_decision_ref: ObjectRef
    invocation_result_ref: ObjectRef

    @classmethod
    def create(
        cls,
        *,
        dataset_run_ref: ObjectRef,
        planning_version: int,
        predecessor_authority_ref: ObjectRef | None,
        plan_ref: ObjectRef,
        route_decision_ref: ObjectRef,
        invocation_result_ref: ObjectRef,
        audit: ContractAudit,
    ) -> FactoryDatasetPlanningAuthorityV2:
        return cls._create(
            audit=audit,
            values={
                "dataset_run_ref": dataset_run_ref,
                "planning_version": planning_version,
                "predecessor_authority_ref": (predecessor_authority_ref),
                "plan_ref": plan_ref,
                "route_decision_ref": route_decision_ref,
                "invocation_result_ref": invocation_result_ref,
            },
        )

    @model_validator(mode="after")
    def validate_authority(self) -> Self:
        for ref, object_type, label in (
            (
                self.dataset_run_ref,
                "factory-run",
                "dataset_run_ref",
            ),
            (
                self.plan_ref,
                "dataset-build-plan",
                "plan_ref",
            ),
            (
                self.route_decision_ref,
                "model-route-decision",
                "route_decision_ref",
            ),
            (
                self.invocation_result_ref,
                "gateway-invocation-result",
                "invocation_result_ref",
            ),
        ):
            _require_ref(ref, object_type, label)
        if self.planning_version == 1:
            if self.predecessor_authority_ref is not None:
                raise ValueError("first planning authority has no predecessor")
        else:
            if self.predecessor_authority_ref is None:
                raise ValueError("planning successor requires predecessor")
            _require_ref(
                self.predecessor_authority_ref,
                self.OBJECT_TYPE,
                "predecessor_authority_ref",
            )
        return self


class FactoryItemRunBindingV2(_DatasetRuntimeObjectV2):
    schema_version: Literal["eval-factory/factory-item-run-binding/v2"] = (
        "eval-factory/factory-item-run-binding/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "factory-item-run-binding"

    dataset_run_ref: ObjectRef
    item_run_ref: ObjectRef
    item_id: Identifier
    binding_version: int = Field(ge=1, le=1_000_000)
    predecessor_binding_ref: ObjectRef | None = None
    trace_candidate_decision_ref: ObjectRef
    extracted_prompt_ref: ObjectRef
    inferred_intent_ref: ObjectRef
    rewrite_candidate_ref: ObjectRef

    @classmethod
    def create(
        cls,
        *,
        dataset_run_ref: ObjectRef,
        item_run_ref: ObjectRef,
        item_id: str,
        binding_version: int,
        predecessor_binding_ref: ObjectRef | None,
        trace_candidate_decision_ref: ObjectRef,
        extracted_prompt_ref: ObjectRef,
        inferred_intent_ref: ObjectRef,
        rewrite_candidate_ref: ObjectRef,
        audit: ContractAudit,
    ) -> FactoryItemRunBindingV2:
        return cls._create(
            audit=audit,
            values={
                "dataset_run_ref": dataset_run_ref,
                "item_run_ref": item_run_ref,
                "item_id": item_id,
                "binding_version": binding_version,
                "predecessor_binding_ref": (predecessor_binding_ref),
                "trace_candidate_decision_ref": (trace_candidate_decision_ref),
                "extracted_prompt_ref": extracted_prompt_ref,
                "inferred_intent_ref": inferred_intent_ref,
                "rewrite_candidate_ref": rewrite_candidate_ref,
            },
        )

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        _require_ref(
            self.dataset_run_ref,
            "factory-run",
            "dataset_run_ref",
        )
        _require_ref(
            self.item_run_ref,
            "factory-run",
            "item_run_ref",
        )
        if self.dataset_run_ref == self.item_run_ref:
            raise ValueError("dataset and item runs must be distinct")
        if self.binding_version == 1:
            if self.predecessor_binding_ref is not None:
                raise ValueError("binding version one cannot have a predecessor")
        else:
            if self.predecessor_binding_ref is None:
                raise ValueError("binding successor requires predecessor")
            _require_ref(
                self.predecessor_binding_ref,
                self.OBJECT_TYPE,
                "predecessor_binding_ref",
            )
        for ref, object_type, label in (
            (
                self.trace_candidate_decision_ref,
                "trace-candidate-decision",
                "trace_candidate_decision_ref",
            ),
            (
                self.extracted_prompt_ref,
                "extracted-user-prompt",
                "extracted_prompt_ref",
            ),
            (
                self.inferred_intent_ref,
                "inferred-user-intent",
                "inferred_intent_ref",
            ),
            (
                self.rewrite_candidate_ref,
                "task-rewrite-candidate",
                "rewrite_candidate_ref",
            ),
        ):
            _require_ref(ref, object_type, label)
        return self


class FactoryItemStageHeadV2(_DatasetRuntimeObjectV2):
    schema_version: Literal["eval-factory/factory-item-stage-head/v2"] = (
        "eval-factory/factory-item-stage-head/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "factory-item-stage-head"

    item_binding_ref: ObjectRef
    item_run_ref: ObjectRef
    stage: FactoryItemStageV2
    stage_version: int = Field(ge=1, le=1_000_000)
    predecessor_head_ref: ObjectRef | None = None
    dependency_result_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=10_000,
    )
    result_ref: ObjectRef
    outcome: FactoryItemStageOutcomeV2
    reason_codes: tuple[SafeCode, ...] = Field(
        default=(),
        max_length=256,
    )

    @classmethod
    def create(
        cls,
        *,
        item_binding_ref: ObjectRef,
        item_run_ref: ObjectRef,
        stage: FactoryItemStageV2,
        stage_version: int,
        predecessor_head_ref: ObjectRef | None,
        dependency_result_refs: tuple[ObjectRef, ...],
        result_ref: ObjectRef,
        outcome: FactoryItemStageOutcomeV2,
        reason_codes: tuple[str, ...],
        audit: ContractAudit,
    ) -> FactoryItemStageHeadV2:
        return cls._create(
            audit=audit,
            values={
                "item_binding_ref": item_binding_ref,
                "item_run_ref": item_run_ref,
                "stage": stage,
                "stage_version": stage_version,
                "predecessor_head_ref": predecessor_head_ref,
                "dependency_result_refs": _sorted_refs(
                    dependency_result_refs,
                ),
                "result_ref": result_ref,
                "outcome": outcome,
                "reason_codes": tuple(sorted(reason_codes)),
            },
        )

    @model_validator(mode="after")
    def validate_head(self) -> Self:
        _require_ref(
            self.item_binding_ref,
            "factory-item-run-binding",
            "item_binding_ref",
        )
        _require_ref(
            self.item_run_ref,
            "factory-run",
            "item_run_ref",
        )
        if self.stage_version == 1:
            if self.predecessor_head_ref is not None:
                raise ValueError("stage version one cannot have a predecessor")
        else:
            if self.predecessor_head_ref is None:
                raise ValueError("stage successor requires predecessor")
            _require_ref(
                self.predecessor_head_ref,
                self.OBJECT_TYPE,
                "predecessor_head_ref",
            )
        _require_sorted_unique_refs(
            self.dependency_result_refs,
            "dependency_result_refs",
        )
        expected_result_type = {
            FactoryItemStageV2.CORE_SELECTION: ("task-rewrite-candidate"),
            FactoryItemStageV2.TASK_AUTHORING: ("r4-task-contract-set"),
            FactoryItemStageV2.ATTACHMENT: ("attachment-subgraph-result"),
            FactoryItemStageV2.ITEM_QUALITY: ("item-quality-compilation-result"),
            FactoryItemStageV2.CRITERIA_RUBRIC: ("criteria-rubric-result"),
            FactoryItemStageV2.GRADING_DESIGN: ("grading-design-result"),
            FactoryItemStageV2.RELEASE_CANDIDATE: ("release-projection-result"),
        }[self.stage]
        if self.result_ref.object_type != expected_result_type or self.result_ref.object_version != "v2":
            raise ValueError(f"{self.stage.value} requires {expected_result_type}/v2")
        _require_sorted_unique(
            self.reason_codes,
            "reason_codes",
        )
        if self.outcome is FactoryItemStageOutcomeV2.SUCCEEDED:
            if self.reason_codes:
                raise ValueError("successful item stage cannot retain reasons")
        elif not self.reason_codes:
            raise ValueError("non-success item stage requires reasons")
        return self


class FactoryDatasetAggregateResultV2(_DatasetRuntimeObjectV2):
    schema_version: Literal["eval-factory/factory-dataset-aggregate-result/v2"] = (
        "eval-factory/factory-dataset-aggregate-result/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "factory-dataset-aggregate-result"

    dataset_run_ref: ObjectRef
    core_vertical_result_ref: ObjectRef
    item_binding_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=1_000_000,
    )
    candidate_binding_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=1_000_000,
    )
    rejected_binding_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=1_000_000,
    )
    blocked_binding_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=1_000_000,
    )
    batch_quality_ref: ObjectRef | None = None
    outcome: CandidateDatasetOutcomeV2
    reason_codes: tuple[SafeCode, ...] = Field(
        default=(),
        max_length=256,
    )
    item_count: int = Field(ge=0, le=1_000_000)
    candidate_count: int = Field(ge=0, le=1_000_000)
    rejected_count: int = Field(ge=0, le=1_000_000)
    blocked_count: int = Field(ge=0, le=1_000_000)

    @classmethod
    def create(
        cls,
        *,
        dataset_run_ref: ObjectRef,
        core_vertical_result_ref: ObjectRef,
        item_binding_refs: tuple[ObjectRef, ...],
        candidate_binding_refs: tuple[ObjectRef, ...],
        rejected_binding_refs: tuple[ObjectRef, ...],
        blocked_binding_refs: tuple[ObjectRef, ...],
        batch_quality_ref: ObjectRef | None,
        outcome: CandidateDatasetOutcomeV2,
        reason_codes: tuple[str, ...],
        audit: ContractAudit,
    ) -> FactoryDatasetAggregateResultV2:
        items = _sorted_refs(item_binding_refs)
        candidates = _sorted_refs(candidate_binding_refs)
        rejected = _sorted_refs(rejected_binding_refs)
        blocked = _sorted_refs(blocked_binding_refs)
        return cls._create(
            audit=audit,
            values={
                "dataset_run_ref": dataset_run_ref,
                "core_vertical_result_ref": (core_vertical_result_ref),
                "item_binding_refs": items,
                "candidate_binding_refs": candidates,
                "rejected_binding_refs": rejected,
                "blocked_binding_refs": blocked,
                "batch_quality_ref": batch_quality_ref,
                "outcome": outcome,
                "reason_codes": tuple(sorted(reason_codes)),
                "item_count": len(items),
                "candidate_count": len(candidates),
                "rejected_count": len(rejected),
                "blocked_count": len(blocked),
            },
        )

    @model_validator(mode="after")
    def validate_aggregate(self) -> Self:
        _require_ref(
            self.dataset_run_ref,
            "factory-run",
            "dataset_run_ref",
        )
        _require_ref(
            self.core_vertical_result_ref,
            "core-vertical-result",
            "core_vertical_result_ref",
        )
        partitions = (
            self.candidate_binding_refs,
            self.rejected_binding_refs,
            self.blocked_binding_refs,
        )
        for values, label in (
            (self.item_binding_refs, "item_binding_refs"),
            (
                self.candidate_binding_refs,
                "candidate_binding_refs",
            ),
            (
                self.rejected_binding_refs,
                "rejected_binding_refs",
            ),
            (
                self.blocked_binding_refs,
                "blocked_binding_refs",
            ),
        ):
            _require_sorted_unique_refs(values, label)
            if any(
                ref.object_type != "factory-item-run-binding" or ref.object_version != "v2" for ref in values
            ):
                raise ValueError(f"{label} requires factory item bindings")
        if _sets_overlap(*(set(values) for values in partitions)):
            raise ValueError("aggregate item partitions overlap")
        if set(self.item_binding_refs) != set().union(*(set(values) for values in partitions)):
            raise ValueError("aggregate requires an exact partition")
        expected_counts = (
            len(self.item_binding_refs),
            len(self.candidate_binding_refs),
            len(self.rejected_binding_refs),
            len(self.blocked_binding_refs),
        )
        if expected_counts != (
            self.item_count,
            self.candidate_count,
            self.rejected_count,
            self.blocked_count,
        ):
            raise ValueError("aggregate item counts are stale")
        _require_sorted_unique(
            self.reason_codes,
            "reason_codes",
        )
        if self.candidate_binding_refs:
            if self.batch_quality_ref is None:
                raise ValueError("candidate aggregate requires batch quality")
            _require_ref(
                self.batch_quality_ref,
                "batch-quality-report",
                "batch_quality_ref",
            )
        elif self.batch_quality_ref is not None:
            raise ValueError("aggregate without candidates cannot carry batch quality")
        if self.outcome is CandidateDatasetOutcomeV2.COMPLETE:
            if (
                not self.candidate_binding_refs
                or self.rejected_binding_refs
                or self.blocked_binding_refs
                or self.reason_codes
            ):
                raise ValueError("complete aggregate must contain candidates only")
        elif self.outcome is CandidateDatasetOutcomeV2.PARTIAL:
            if (
                not self.candidate_binding_refs
                or not (self.rejected_binding_refs or self.blocked_binding_refs)
                or not self.reason_codes
            ):
                raise ValueError("partial aggregate requires mixed partitions")
        elif self.outcome is (CandidateDatasetOutcomeV2.NO_ELIGIBLE_ITEMS):
            if self.candidate_binding_refs or not self.reason_codes:
                raise ValueError("no-eligible aggregate forbids candidates")
        elif self.candidate_binding_refs or not self.blocked_binding_refs or not self.reason_codes:
            raise ValueError("blocked aggregate requires blocked items only")
        return self


class DatasetDeliveryPlanV2(_DatasetRuntimeObjectV2):
    schema_version: Literal["eval-factory/dataset-delivery-plan/v2"] = "eval-factory/dataset-delivery-plan/v2"
    OBJECT_TYPE: ClassVar[str] = "dataset-delivery-plan"

    plan_id: Identifier
    run_ref: ObjectRef
    plan_version: int = Field(ge=1, le=1_000_000)
    predecessor_plan_ref: ObjectRef | None
    aggregate_result_ref: ObjectRef
    candidate_item_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=1_000_000,
    )
    candidate_projection_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=1_000_000,
    )
    candidate_stage_head_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=1_000_000,
    )
    rejected_binding_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=1_000_000,
    )
    blocked_binding_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=1_000_000,
    )
    output_target_ref: ObjectRef
    max_files: int = Field(ge=1, le=10_000_000)
    max_total_bytes: int = Field(ge=1, le=10_000_000_000_000)
    production_release_allowed: Literal[False] = False

    @classmethod
    def create(
        cls,
        *,
        plan_id: str,
        run_ref: ObjectRef,
        plan_version: int,
        predecessor_plan_ref: ObjectRef | None,
        aggregate_result_ref: ObjectRef,
        candidate_item_refs: tuple[ObjectRef, ...],
        candidate_projection_refs: tuple[ObjectRef, ...],
        candidate_stage_head_refs: tuple[ObjectRef, ...],
        rejected_binding_refs: tuple[ObjectRef, ...],
        blocked_binding_refs: tuple[ObjectRef, ...],
        output_target_ref: ObjectRef,
        max_files: int,
        max_total_bytes: int,
        audit: ContractAudit,
    ) -> DatasetDeliveryPlanV2:
        return cls._create(
            audit=audit,
            values={
                "plan_id": plan_id,
                "run_ref": run_ref,
                "plan_version": plan_version,
                "predecessor_plan_ref": predecessor_plan_ref,
                "aggregate_result_ref": aggregate_result_ref,
                "candidate_item_refs": _sorted_refs(candidate_item_refs),
                "candidate_projection_refs": _sorted_refs(candidate_projection_refs),
                "candidate_stage_head_refs": _sorted_refs(
                    candidate_stage_head_refs,
                ),
                "rejected_binding_refs": _sorted_refs(rejected_binding_refs),
                "blocked_binding_refs": _sorted_refs(blocked_binding_refs),
                "output_target_ref": output_target_ref,
                "max_files": max_files,
                "max_total_bytes": max_total_bytes,
            },
        )

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        for ref, object_type, label in (
            (self.run_ref, "factory-run", "run_ref"),
            (
                self.aggregate_result_ref,
                "factory-dataset-aggregate-result",
                "aggregate_result_ref",
            ),
            (
                self.output_target_ref,
                "candidate-output-target",
                "output_target_ref",
            ),
        ):
            _require_ref(ref, object_type, label)
        if self.plan_version == 1:
            if self.predecessor_plan_ref is not None:
                raise ValueError("delivery plan version one has no predecessor")
        elif self.predecessor_plan_ref is None:
            raise ValueError("delivery plan successor requires predecessor")
        else:
            _require_ref(
                self.predecessor_plan_ref,
                self.OBJECT_TYPE,
                "predecessor_plan_ref",
            )
        for values, label, object_type in (
            (
                self.candidate_item_refs,
                "candidate_item_refs",
                "evaluation-item",
            ),
            (
                self.candidate_projection_refs,
                "candidate_projection_refs",
                "release-projection-result",
            ),
            (
                self.candidate_stage_head_refs,
                "candidate_stage_head_refs",
                "factory-item-stage-head",
            ),
            (
                self.rejected_binding_refs,
                "rejected_binding_refs",
                "factory-item-run-binding",
            ),
            (
                self.blocked_binding_refs,
                "blocked_binding_refs",
                "factory-item-run-binding",
            ),
        ):
            _require_sorted_unique_refs(values, label)
            if any(ref.object_type != object_type or ref.object_version != "v2" for ref in values):
                raise ValueError(f"{label} requires {object_type}/v2")
        if not (
            len(self.candidate_item_refs)
            == len(self.candidate_projection_refs)
            == len(self.candidate_stage_head_refs)
        ):
            raise ValueError("delivery plan candidate inventories differ")
        if set(self.rejected_binding_refs) & set(self.blocked_binding_refs):
            raise ValueError("delivery plan terminal partitions overlap")
        return self


class CompiledDatasetDeliveryPlanV2(_DatasetRuntimeObjectV2):
    schema_version: Literal["eval-factory/compiled-dataset-delivery-plan/v2"] = (
        "eval-factory/compiled-dataset-delivery-plan/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "compiled-dataset-delivery-plan"

    compiled_plan_id: Identifier
    source_plan_ref: ObjectRef
    policy_ref: ObjectRef
    aggregate_result_ref: ObjectRef
    output_target_ref: ObjectRef
    candidate_count: int = Field(ge=1, le=1_000_000)
    compiler_version: Literal["dataset-delivery-plan-compiler/v1"] = "dataset-delivery-plan-compiler/v1"
    production_release_allowed: Literal[False] = False

    @classmethod
    def create(
        cls,
        *,
        compiled_plan_id: str,
        source_plan_ref: ObjectRef,
        policy_ref: ObjectRef,
        aggregate_result_ref: ObjectRef,
        output_target_ref: ObjectRef,
        candidate_count: int,
        audit: ContractAudit,
    ) -> CompiledDatasetDeliveryPlanV2:
        return cls._create(
            audit=audit,
            values={
                "compiled_plan_id": compiled_plan_id,
                "source_plan_ref": source_plan_ref,
                "policy_ref": policy_ref,
                "aggregate_result_ref": aggregate_result_ref,
                "output_target_ref": output_target_ref,
                "candidate_count": candidate_count,
            },
        )

    @model_validator(mode="after")
    def validate_compiled(self) -> Self:
        for ref, object_type, label in (
            (
                self.source_plan_ref,
                "dataset-delivery-plan",
                "source_plan_ref",
            ),
            (
                self.policy_ref,
                "factory-run-policy",
                "policy_ref",
            ),
            (
                self.aggregate_result_ref,
                "factory-dataset-aggregate-result",
                "aggregate_result_ref",
            ),
            (
                self.output_target_ref,
                "candidate-output-target",
                "output_target_ref",
            ),
        ):
            _require_ref(ref, object_type, label)
        return self


class CandidateDatasetDeliveryManifestV2(_DatasetRuntimeObjectV2):
    schema_version: Literal["eval-factory/candidate-dataset-delivery-manifest/v2"] = (
        "eval-factory/candidate-dataset-delivery-manifest/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "candidate-dataset-delivery-manifest"

    dataset_run_ref: ObjectRef
    aggregate_result_ref: ObjectRef
    batch_quality_ref: ObjectRef
    candidate_item_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=1_000_000,
    )
    candidate_projection_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=1_000_000,
    )
    candidate_stage_head_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=1_000_000,
    )
    rejected_binding_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=1_000_000,
    )
    blocked_binding_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=1_000_000,
    )
    inventory_ref: ObjectRef
    provenance_manifest_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=1_000_000,
    )
    item_count: int = Field(ge=1, le=1_000_000)

    @classmethod
    def create(
        cls,
        *,
        dataset_run_ref: ObjectRef,
        aggregate_result_ref: ObjectRef,
        batch_quality_ref: ObjectRef,
        candidate_item_refs: tuple[ObjectRef, ...],
        candidate_projection_refs: tuple[ObjectRef, ...],
        candidate_stage_head_refs: tuple[ObjectRef, ...],
        rejected_binding_refs: tuple[ObjectRef, ...],
        blocked_binding_refs: tuple[ObjectRef, ...],
        inventory_ref: ObjectRef,
        provenance_manifest_refs: tuple[ObjectRef, ...],
        audit: ContractAudit,
    ) -> CandidateDatasetDeliveryManifestV2:
        items = _sorted_refs(candidate_item_refs)
        return cls._create(
            audit=audit,
            values={
                "dataset_run_ref": dataset_run_ref,
                "aggregate_result_ref": aggregate_result_ref,
                "batch_quality_ref": batch_quality_ref,
                "candidate_item_refs": items,
                "candidate_projection_refs": _sorted_refs(
                    candidate_projection_refs,
                ),
                "candidate_stage_head_refs": _sorted_refs(
                    candidate_stage_head_refs,
                ),
                "rejected_binding_refs": _sorted_refs(
                    rejected_binding_refs,
                ),
                "blocked_binding_refs": _sorted_refs(
                    blocked_binding_refs,
                ),
                "inventory_ref": inventory_ref,
                "provenance_manifest_refs": _sorted_refs(
                    provenance_manifest_refs,
                ),
                "item_count": len(items),
            },
        )

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        for ref, object_type, label in (
            (
                self.dataset_run_ref,
                "factory-run",
                "dataset_run_ref",
            ),
            (
                self.aggregate_result_ref,
                "factory-dataset-aggregate-result",
                "aggregate_result_ref",
            ),
            (
                self.batch_quality_ref,
                "batch-quality-report",
                "batch_quality_ref",
            ),
            (
                self.inventory_ref,
                "candidate-dataset-inventory",
                "inventory_ref",
            ),
        ):
            _require_ref(ref, object_type, label)
        for values, label, object_type in (
            (
                self.candidate_item_refs,
                "candidate_item_refs",
                "evaluation-item",
            ),
            (
                self.candidate_projection_refs,
                "candidate_projection_refs",
                "release-projection-result",
            ),
            (
                self.candidate_stage_head_refs,
                "candidate_stage_head_refs",
                "factory-item-stage-head",
            ),
            (
                self.rejected_binding_refs,
                "rejected_binding_refs",
                "factory-item-run-binding",
            ),
            (
                self.blocked_binding_refs,
                "blocked_binding_refs",
                "factory-item-run-binding",
            ),
            (
                self.provenance_manifest_refs,
                "provenance_manifest_refs",
                "provenance-manifest",
            ),
        ):
            _require_sorted_unique_refs(values, label)
            if any(ref.object_type != object_type or ref.object_version != "v2" for ref in values):
                raise ValueError(f"{label} requires {object_type}/v2")
        if set(self.rejected_binding_refs) & set(self.blocked_binding_refs):
            raise ValueError("delivery rejected and blocked partitions overlap")
        if not (
            self.item_count
            == len(self.candidate_item_refs)
            == len(self.candidate_projection_refs)
            == len(self.candidate_stage_head_refs)
            == len(self.provenance_manifest_refs)
        ):
            raise ValueError("delivery requires exact candidate coverage")
        return self


class FactoryDatasetRunViewV2(_DatasetRuntimeObjectV2):
    schema_version: Literal["eval-factory/factory-dataset-run-view/v2"] = (
        "eval-factory/factory-dataset-run-view/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "factory-dataset-run-view"

    request_ref: ObjectRef
    dataset_run_ref: ObjectRef
    status: FactoryRunStatusV2
    pending_review_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=10_000,
    )
    item_binding_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=1_000_000,
    )
    candidate_count: int = Field(ge=0, le=1_000_000)
    rejected_count: int = Field(ge=0, le=1_000_000)
    blocked_count: int = Field(ge=0, le=1_000_000)
    incomplete_count: int = Field(ge=0, le=1_000_000)
    aggregate_result_ref: ObjectRef | None = None
    delivery_manifest_ref: ObjectRef | None = None
    next_action: FactoryDatasetNextActionV2

    @classmethod
    def create(
        cls,
        *,
        request_ref: ObjectRef,
        dataset_run_ref: ObjectRef,
        status: FactoryRunStatusV2,
        pending_review_refs: tuple[ObjectRef, ...],
        item_binding_refs: tuple[ObjectRef, ...],
        candidate_count: int,
        rejected_count: int,
        blocked_count: int,
        incomplete_count: int,
        aggregate_result_ref: ObjectRef | None,
        delivery_manifest_ref: ObjectRef | None,
        next_action: FactoryDatasetNextActionV2,
        audit: ContractAudit,
    ) -> FactoryDatasetRunViewV2:
        return cls._create(
            audit=audit,
            values={
                "request_ref": request_ref,
                "dataset_run_ref": dataset_run_ref,
                "status": status,
                "pending_review_refs": _sorted_refs(
                    pending_review_refs,
                ),
                "item_binding_refs": _sorted_refs(
                    item_binding_refs,
                ),
                "candidate_count": candidate_count,
                "rejected_count": rejected_count,
                "blocked_count": blocked_count,
                "incomplete_count": incomplete_count,
                "aggregate_result_ref": aggregate_result_ref,
                "delivery_manifest_ref": delivery_manifest_ref,
                "next_action": next_action,
            },
        )

    @model_validator(mode="after")
    def validate_view(self) -> Self:
        _require_ref(
            self.request_ref,
            "factory-dataset-run-request",
            "request_ref",
        )
        _require_ref(
            self.dataset_run_ref,
            "factory-run",
            "dataset_run_ref",
        )
        _require_sorted_unique_refs(
            self.pending_review_refs,
            "pending_review_refs",
        )
        if any(
            ref.object_type != "plan-review-request" or ref.object_version != "v2"
            for ref in self.pending_review_refs
        ):
            raise ValueError("pending reviews require plan-review-request/v2")
        _require_sorted_unique_refs(
            self.item_binding_refs,
            "item_binding_refs",
        )
        if any(
            ref.object_type != "factory-item-run-binding" or ref.object_version != "v2"
            for ref in self.item_binding_refs
        ):
            raise ValueError("runtime view requires item binding refs")
        if self.candidate_count + self.rejected_count + self.blocked_count + self.incomplete_count != len(
            self.item_binding_refs
        ):
            raise ValueError("runtime view item counts are not exact")
        if self.aggregate_result_ref is not None:
            _require_ref(
                self.aggregate_result_ref,
                "factory-dataset-aggregate-result",
                "aggregate_result_ref",
            )
        if self.delivery_manifest_ref is not None:
            _require_ref(
                self.delivery_manifest_ref,
                "candidate-dataset-delivery-manifest",
                "delivery_manifest_ref",
            )
        if self.status is FactoryRunStatusV2.WAITING_REVIEW:
            if not self.pending_review_refs or self.next_action is not FactoryDatasetNextActionV2.REVIEW_PLAN:
                raise ValueError("waiting runtime view requires pending review")
        elif self.pending_review_refs:
            raise ValueError("non-waiting runtime view cannot carry pending reviews")
        if self.status is FactoryRunStatusV2.COMPLETED:
            if (
                self.aggregate_result_ref is None
                or self.delivery_manifest_ref is None
                or self.next_action is not FactoryDatasetNextActionV2.INSPECT_OUTPUT
            ):
                raise ValueError("completed runtime view requires delivery authority")
        elif self.delivery_manifest_ref is not None:
            raise ValueError("non-completed runtime view cannot carry delivery")
        if (
            self.status
            in {
                FactoryRunStatusV2.BLOCKED,
                FactoryRunStatusV2.FAILED,
                FactoryRunStatusV2.CANCELLED,
            }
            and self.next_action is not FactoryDatasetNextActionV2.NONE
        ):
            raise ValueError("terminal non-success runtime view has no next action")
        return self


def _carried_sha256(
    value: _DatasetRuntimeObjectV2,
) -> str:
    payload = value.model_dump(
        mode="python",
        exclude={
            "schema_version",
            "object_id",
            "object_sha256",
            "audit",
        },
    )
    return hashlib.sha256(
        json.dumps(
            canonical_value_v2(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _collect_refs(
    value: ContractModelV2,
) -> tuple[ObjectRef, ...]:
    refs: list[ObjectRef] = []

    def visit(item: object) -> None:
        if isinstance(item, ObjectRef):
            refs.append(item)
        elif isinstance(item, ContractModelV2):
            for field_name in type(item).model_fields:
                if field_name in {
                    "audit",
                    "object_id",
                    "object_sha256",
                }:
                    continue
                visit(getattr(item, field_name))
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, tuple | list | set | frozenset):
            for child in item:
                visit(child)

    visit(value)
    return _sorted_refs(tuple(refs))


def _audit_with_refs(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    return ContractAudit(
        created_at=audit.created_at,
        created_by=audit.created_by,
        governing_versions=audit.governing_versions,
        input_refs=refs,
    )


def _require_ref(
    value: ObjectRef,
    object_type: str,
    label: str,
) -> None:
    if value.object_type != object_type or value.object_version != "v2":
        raise ValueError(f"{label} must reference {object_type}/v2")


def _require_safe_refs(
    refs: tuple[ObjectRef, ...],
) -> None:
    for ref in refs:
        rendered = (f"{ref.object_type} {ref.object_id}").casefold()
        if any(marker in rendered for marker in _DENIED_REF_MARKERS):
            raise ValueError("dataset runtime refs contain sensitive material")


def _require_sorted_unique_refs(
    values: tuple[ObjectRef, ...],
    label: str,
) -> None:
    if values != _sorted_refs(values):
        raise ValueError(f"{label} must be sorted and unique")


def _require_sorted_unique(
    values: tuple[str, ...],
    label: str,
) -> None:
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{label} must be sorted and unique")


def _sorted_refs(
    values: tuple[ObjectRef, ...],
) -> tuple[ObjectRef, ...]:
    return tuple(sorted(set(values), key=_ref_key))


def _ref_key(
    value: ObjectRef,
) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _sets_overlap(
    *values: set[ObjectRef],
) -> bool:
    observed: set[ObjectRef] = set()
    for current in values:
        if observed & current:
            return True
        observed.update(current)
    return False


__all__ = [
    "CandidateDatasetDeliveryManifestV2",
    "CandidateDatasetOutcomeV2",
    "CompiledDatasetDeliveryPlanV2",
    "DatasetDeliveryPlanV2",
    "FactoryDatasetAggregateResultV2",
    "FactoryDatasetNextActionV2",
    "FactoryDatasetPlanningAuthorityV2",
    "FactoryDatasetRunRequestV2",
    "FactoryDatasetRunViewV2",
    "FactoryItemRunBindingV2",
    "FactoryItemStageHeadV2",
    "FactoryItemStageOutcomeV2",
    "FactoryItemStageV2",
]
