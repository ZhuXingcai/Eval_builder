from __future__ import annotations

from datetime import datetime
from typing import ClassVar, Literal, Self

from pydantic import Field, model_validator

from eval_factory.contracts.agent_system_v2 import (
    PlanKindV2,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.harness.contracts import (
    HarnessObjectV1,
    require_ref,
    require_sorted_unique_refs,
)


class RequirementPlanningCapabilityRequestV1(HarnessObjectV1):
    schema_version: Literal["generic-agent-trace/requirement-planning-request/v1"] = (
        "generic-agent-trace/requirement-planning-request/v1"
    )
    OBJECT_TYPE: ClassVar[str] = "generic-requirement-planning-request"

    plan_ref: ObjectRef
    policy_ref: ObjectRef

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        _require_v2_ref(self.plan_ref, "dataset-build-plan", "plan_ref")
        _require_v2_ref(self.policy_ref, "factory-run-policy", "policy_ref")
        return self


class TraceIngestionCapabilityRequestV1(HarnessObjectV1):
    schema_version: Literal["generic-agent-trace/trace-ingestion-request/v1"] = (
        "generic-agent-trace/trace-ingestion-request/v1"
    )
    OBJECT_TYPE: ClassVar[str] = "generic-trace-ingestion-request"

    trace_source_ref: ObjectRef
    job_id: str = Field(min_length=1, max_length=512)
    attempt: int = Field(ge=1, le=1_000_000)

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        _require_v2_ref(
            self.trace_source_ref,
            "trace-source",
            "trace_source_ref",
        )
        return self


class TaskAuthoringCapabilityRequestV1(HarnessObjectV1):
    schema_version: Literal["generic-agent-trace/task-authoring-request/v1"] = (
        "generic-agent-trace/task-authoring-request/v1"
    )
    OBJECT_TYPE: ClassVar[str] = "generic-task-authoring-request"

    decision_ref: ObjectRef
    extracted_prompt_ref: ObjectRef
    intent_ref: ObjectRef
    rewrite_ref: ObjectRef
    requirement_ref: ObjectRef

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        for reference, object_type, label in (
            (
                self.decision_ref,
                "trace-candidate-decision",
                "decision_ref",
            ),
            (
                self.extracted_prompt_ref,
                "extracted-user-prompt",
                "extracted_prompt_ref",
            ),
            (
                self.intent_ref,
                "inferred-user-intent",
                "intent_ref",
            ),
            (
                self.rewrite_ref,
                "task-rewrite-candidate",
                "rewrite_ref",
            ),
            (
                self.requirement_ref,
                "evaluation-requirement-spec",
                "requirement_ref",
            ),
        ):
            _require_v2_ref(reference, object_type, label)
        return self


class AttachmentReconstructionCapabilityRequestV1(HarnessObjectV1):
    schema_version: Literal["generic-agent-trace/attachment-reconstruction-request/v1"] = (
        "generic-agent-trace/attachment-reconstruction-request/v1"
    )
    OBJECT_TYPE: ClassVar[str] = "generic-attachment-reconstruction-request"

    run_id: str = Field(min_length=1, max_length=512)
    plan_ref: ObjectRef
    compiled_plan_ref: ObjectRef
    preparation_ref: ObjectRef
    prior_batch_ref: ObjectRef | None = None
    lease_duration_seconds: int = Field(default=600, ge=1, le=86_400)

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        _require_v2_ref(
            self.plan_ref,
            "attachment-generation-plan",
            "plan_ref",
        )
        _require_v2_ref(
            self.compiled_plan_ref,
            "compiled-attachment-generation-plan",
            "compiled_plan_ref",
        )
        _require_v2_ref(
            self.preparation_ref,
            "attachment-r5-preparation",
            "preparation_ref",
        )
        if self.prior_batch_ref is not None:
            _require_v2_ref(
                self.prior_batch_ref,
                "artifact-execution-batch",
                "prior_batch_ref",
            )
        return self


class CriteriaRubricCapabilityRequestV1(HarnessObjectV1):
    schema_version: Literal["generic-agent-trace/criteria-rubric-request/v1"] = (
        "generic-agent-trace/criteria-rubric-request/v1"
    )
    OBJECT_TYPE: ClassVar[str] = "generic-criteria-rubric-request"

    run_id: str = Field(min_length=1, max_length=512)
    plan_ref: ObjectRef
    compiled_plan_ref: ObjectRef
    task_draft_ref: ObjectRef
    attachment_quality_ref: ObjectRef
    solvability_ref: ObjectRef
    binding_definitions_ref: ObjectRef
    tool_catalog_ref: ObjectRef
    lease_duration_seconds: int = Field(default=600, ge=1, le=86_400)

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        for reference, object_type, label in (
            (self.plan_ref, "criteria-rubric-plan", "plan_ref"),
            (
                self.compiled_plan_ref,
                "compiled-criteria-rubric-plan",
                "compiled_plan_ref",
            ),
            (self.task_draft_ref, "task-draft", "task_draft_ref"),
            (
                self.attachment_quality_ref,
                "attachment-quality-assessment",
                "attachment_quality_ref",
            ),
            (
                self.solvability_ref,
                "solvability-assessment",
                "solvability_ref",
            ),
            (
                self.binding_definitions_ref,
                "evaluator-binding-definitions",
                "binding_definitions_ref",
            ),
            (
                self.tool_catalog_ref,
                "tool-capability-catalog",
                "tool_catalog_ref",
            ),
        ):
            _require_v2_ref(reference, object_type, label)
        return self


class GradingDesignCapabilityRequestV1(HarnessObjectV1):
    schema_version: Literal["generic-agent-trace/grading-design-request/v1"] = (
        "generic-agent-trace/grading-design-request/v1"
    )
    OBJECT_TYPE: ClassVar[str] = "generic-grading-design-request"

    run_id: str = Field(min_length=1, max_length=512)
    plan_ref: ObjectRef
    compiled_plan_ref: ObjectRef
    criteria_result_ref: ObjectRef
    criteria_route_ref: ObjectRef
    rubric_set_ref: ObjectRef
    evaluator_spec_ref: ObjectRef
    reference_policy_ref: ObjectRef
    tool_policy_ref: ObjectRef
    model_authorizations_ref: ObjectRef
    evaluated_at: datetime
    lease_duration_seconds: int = Field(default=600, ge=1, le=86_400)

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if self.evaluated_at.tzinfo is None:
            raise ValueError("evaluated_at must be timezone-aware")
        for reference, object_type, label in (
            (self.plan_ref, "grading-design-plan", "plan_ref"),
            (
                self.compiled_plan_ref,
                "compiled-grading-design-plan",
                "compiled_plan_ref",
            ),
            (
                self.criteria_result_ref,
                "criteria-rubric-result",
                "criteria_result_ref",
            ),
            (
                self.criteria_route_ref,
                "model-route-decision",
                "criteria_route_ref",
            ),
            (self.rubric_set_ref, "rubric-set", "rubric_set_ref"),
            (
                self.evaluator_spec_ref,
                "evaluator-spec",
                "evaluator_spec_ref",
            ),
            (
                self.reference_policy_ref,
                "reference-policy",
                "reference_policy_ref",
            ),
            (self.tool_policy_ref, "tool-policy", "tool_policy_ref"),
            (
                self.model_authorizations_ref,
                "model-domain-authorizations",
                "model_authorizations_ref",
            ),
        ):
            _require_v2_ref(reference, object_type, label)
        return self


class AttachmentQualityCapabilityRequestV1(HarnessObjectV1):
    schema_version: Literal["generic-agent-trace/attachment-quality-request/v1"] = (
        "generic-agent-trace/attachment-quality-request/v1"
    )
    OBJECT_TYPE: ClassVar[str] = "generic-attachment-quality-request"

    subgraph_result_ref: ObjectRef
    group_result_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=100_000,
    )
    work_envelope_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=100_000,
    )
    source_validation_ref: ObjectRef
    item_quality_ref: ObjectRef

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        _require_v2_ref(
            self.subgraph_result_ref,
            "attachment-subgraph-result",
            "subgraph_result_ref",
        )
        require_sorted_unique_refs(
            self.group_result_refs,
            "group_result_refs",
        )
        require_sorted_unique_refs(
            self.work_envelope_refs,
            "work_envelope_refs",
        )
        if any(
            value.object_type != "attachment-group-result" or value.object_version != "v2"
            for value in self.group_result_refs
        ):
            raise ValueError("group_result_refs contain an invalid type")
        if any(
            value.object_type != "agent-result-envelope" or value.object_version != "v2"
            for value in self.work_envelope_refs
        ):
            raise ValueError("work_envelope_refs contain an invalid type")
        if len(self.group_result_refs) != len(self.work_envelope_refs):
            raise ValueError("quality request group and envelope counts differ")
        _require_v2_ref(
            self.source_validation_ref,
            "deterministic-item-validation-result",
            "source_validation_ref",
        )
        _require_v2_ref(
            self.item_quality_ref,
            "item-quality-compilation-result",
            "item_quality_ref",
        )
        return self


class AttachmentQualityCapabilityRequestV2(HarnessObjectV1):
    schema_version: Literal["generic-agent-trace/attachment-quality-request/v2"] = (
        "generic-agent-trace/attachment-quality-request/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "generic-attachment-quality-request"

    item_binding_ref: ObjectRef
    task_authoring_material_ref: ObjectRef
    attachment_execution_material_ref: ObjectRef

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        for reference, object_type, label in (
            (
                self.item_binding_ref,
                "factory-item-run-binding",
                "item_binding_ref",
            ),
            (
                self.task_authoring_material_ref,
                "task-authoring-material",
                "task_authoring_material_ref",
            ),
            (
                self.attachment_execution_material_ref,
                "attachment-execution-material",
                "attachment_execution_material_ref",
            ),
        ):
            _require_v2_ref(reference, object_type, label)
        return self


class BatchQualityCapabilityRequestV1(HarnessObjectV1):
    schema_version: Literal["generic-agent-trace/batch-quality-request/v1"] = (
        "generic-agent-trace/batch-quality-request/v1"
    )
    OBJECT_TYPE: ClassVar[str] = "generic-batch-quality-request"

    dataset_run_id: str = Field(min_length=1, max_length=512)
    core_vertical_result_ref: ObjectRef
    sources_ref: ObjectRef
    rejected_binding_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=1_000_000,
    )
    blocked_binding_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=1_000_000,
    )
    resolved_job_work_graph_ref: ObjectRef
    duplicate_policy_ref: ObjectRef
    cross_item_policy_ref: ObjectRef
    batch_policy_ref: ObjectRef

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        require_sorted_unique_refs(
            self.rejected_binding_refs,
            "rejected_binding_refs",
        )
        require_sorted_unique_refs(
            self.blocked_binding_refs,
            "blocked_binding_refs",
        )
        if set(self.rejected_binding_refs) & set(self.blocked_binding_refs):
            raise ValueError("batch terminal binding partitions overlap")
        for reference, object_type, label in (
            (self.sources_ref, "batch-quality-sources", "sources_ref"),
            (
                self.resolved_job_work_graph_ref,
                "resolved-job-work-graph",
                "resolved_job_work_graph_ref",
            ),
            (
                self.duplicate_policy_ref,
                "duplicate-detection-policy",
                "duplicate_policy_ref",
            ),
            (
                self.cross_item_policy_ref,
                "cross-item-safety-policy",
                "cross_item_policy_ref",
            ),
            (
                self.batch_policy_ref,
                "batch-quality-policy",
                "batch_policy_ref",
            ),
        ):
            _require_v2_ref(reference, object_type, label)
        return self


class PlanReviewCapabilityRequestV1(HarnessObjectV1):
    schema_version: Literal["generic-agent-trace/plan-review-request/v1"] = (
        "generic-agent-trace/plan-review-request/v1"
    )
    OBJECT_TYPE: ClassVar[str] = "generic-plan-review-request"

    run_id: str = Field(min_length=1, max_length=512)
    plan_kind: PlanKindV2
    plan_ref: ObjectRef

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if self.plan_ref.object_version != "v2":
            raise ValueError("plan_ref must use object version v2")
        return self


class DeliveryCapabilityRequestV1(HarnessObjectV1):
    schema_version: Literal["generic-agent-trace/delivery-request/v1"] = (
        "generic-agent-trace/delivery-request/v1"
    )
    OBJECT_TYPE: ClassVar[str] = "generic-delivery-request"

    dataset_run_id: str = Field(min_length=1, max_length=512)
    aggregate_ref: ObjectRef
    candidate_projection_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=1_000_000,
    )
    export_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=1_000_000,
    )
    policy_ref: ObjectRef

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        require_sorted_unique_refs(
            self.candidate_projection_refs,
            "candidate_projection_refs",
        )
        require_sorted_unique_refs(self.export_refs, "export_refs")
        _require_v2_ref(
            self.aggregate_ref,
            "factory-dataset-aggregate-result",
            "aggregate_ref",
        )
        _require_v2_ref(
            self.policy_ref,
            "factory-run-policy",
            "policy_ref",
        )
        if any(
            value.object_type != "release-projection-result" or value.object_version != "v2"
            for value in self.candidate_projection_refs
        ):
            raise ValueError("candidate projection refs contain an invalid type")
        if any(
            value.object_type != "authorized-candidate-export" or value.object_version != "v2"
            for value in self.export_refs
        ):
            raise ValueError("candidate export refs contain an invalid type")
        return self


def _require_v2_ref(
    reference: ObjectRef,
    object_type: str,
    label: str,
) -> None:
    require_ref(
        reference,
        object_type,
        label,
        object_version="v2",
    )


CapabilityRequestV1 = (
    RequirementPlanningCapabilityRequestV1
    | TraceIngestionCapabilityRequestV1
    | TaskAuthoringCapabilityRequestV1
    | AttachmentReconstructionCapabilityRequestV1
    | CriteriaRubricCapabilityRequestV1
    | GradingDesignCapabilityRequestV1
    | AttachmentQualityCapabilityRequestV1
    | AttachmentQualityCapabilityRequestV2
    | BatchQualityCapabilityRequestV1
    | PlanReviewCapabilityRequestV1
    | DeliveryCapabilityRequestV1
)


__all__ = [
    "AttachmentQualityCapabilityRequestV1",
    "AttachmentQualityCapabilityRequestV2",
    "AttachmentReconstructionCapabilityRequestV1",
    "BatchQualityCapabilityRequestV1",
    "CapabilityRequestV1",
    "CriteriaRubricCapabilityRequestV1",
    "DeliveryCapabilityRequestV1",
    "GradingDesignCapabilityRequestV1",
    "PlanReviewCapabilityRequestV1",
    "RequirementPlanningCapabilityRequestV1",
    "TaskAuthoringCapabilityRequestV1",
    "TraceIngestionCapabilityRequestV1",
]
