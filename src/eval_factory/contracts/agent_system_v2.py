from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Annotated, ClassVar, Literal, Self

from pydantic import BaseModel, Field, StringConstraints, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    Identifier,
    ObjectRef,
    Sha256,
    SourceSpanRef,
    TypedAttribute,
)
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.contracts.task import ReferenceMode
from eval_factory.contracts.task_v2 import RubricJudgedObjectKindV2

SafeSummary = Annotated[str, StringConstraints(min_length=1, max_length=1_000)]
SafeCode = Annotated[str, StringConstraints(pattern=r"^[A-Z][A-Z0-9_]{1,127}$")]

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
)


class FactoryRunStatusV2(StrEnum):
    CREATED = "CREATED"
    PLANNING = "PLANNING"
    WAITING_REVIEW = "WAITING_REVIEW"
    RUNNING = "RUNNING"
    BLOCKED = "BLOCKED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class PlanKindV2(StrEnum):
    GLOBAL_BUILD = "GLOBAL_BUILD"
    TRACE_CLEANING = "TRACE_CLEANING"
    TASK_REWRITE = "TASK_REWRITE"
    ATTACHMENT_GENERATION = "ATTACHMENT_GENERATION"
    CRITERIA_RUBRIC = "CRITERIA_RUBRIC"
    GRADING_DESIGN = "GRADING_DESIGN"
    FINAL_DELIVERY = "FINAL_DELIVERY"


class AgentTaskStatusV2(StrEnum):
    PENDING = "PENDING"
    READY = "READY"
    LEASED = "LEASED"
    TERMINAL = "TERMINAL"
    CANCELLED = "CANCELLED"


class AgentTaskOutcomeV2(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    ABSTAINED = "ABSTAINED"
    BLOCKED = "BLOCKED"
    RETRYABLE_FAILURE = "RETRYABLE_FAILURE"
    FAILED = "FAILED"


class AgentWorkspaceStateV2(StrEnum):
    ACTIVE = "ACTIVE"
    COMMITTED = "COMMITTED"
    QUARANTINED = "QUARANTINED"


class TraceCandidateDispositionV2(StrEnum):
    CANDIDATE = "CANDIDATE"
    NON_CANDIDATE = "NON_CANDIDATE"
    BLOCKED = "BLOCKED"


class AttachmentSubgraphOutcomeV2(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    PARTIAL = "PARTIAL"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"


class AttachmentQualityOutcomeV2(StrEnum):
    PASSED = "PASSED"
    REPAIR_REQUIRED = "REPAIR_REQUIRED"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"


class SolvabilityOutcomeV2(StrEnum):
    SOLVABLE = "SOLVABLE"
    UNSOLVABLE = "UNSOLVABLE"
    ABSTAINED = "ABSTAINED"
    BLOCKED = "BLOCKED"


class CriteriaRubricOutcomeV2(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    ABSTAINED = "ABSTAINED"
    BLOCKED_REACHABILITY = "BLOCKED_REACHABILITY"
    BLOCKED_BINDING = "BLOCKED_BINDING"
    BLOCKED_POLICY = "BLOCKED_POLICY"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"


class JudgeAggregationModeV2(StrEnum):
    WEIGHTED_SUM = "WEIGHTED_SUM"
    ALL_CRITICAL = "ALL_CRITICAL"


class JudgeDesignValidationOutcomeV2(StrEnum):
    VALID = "VALID"
    ABSTAIN_UNJUDGEABLE = "ABSTAIN_UNJUDGEABLE"
    ESCALATE_REFERENCE_REQUIRED = "ESCALATE_REFERENCE_REQUIRED"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"
    BLOCKED_POLICY = "BLOCKED_POLICY"
    INVALID_DESIGN = "INVALID_DESIGN"


class GradingDesignOutcomeV2(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    ABSTAINED = "ABSTAINED"
    ESCALATED = "ESCALATED"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"
    BLOCKED_POLICY = "BLOCKED_POLICY"
    INVALID_DESIGN = "INVALID_DESIGN"


class PlannerAssessmentActionV2(StrEnum):
    CONTINUE = "CONTINUE"
    RETRY = "RETRY"
    REPLAN = "REPLAN"
    ESCALATE = "ESCALATE"
    FINISH = "FINISH"


class FactoryCompletionOutcomeV2(StrEnum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    CAPABILITY_BLOCKED = "CAPABILITY_BLOCKED"
    QUALITY_FAILED = "QUALITY_FAILED"


class PlanReviewStateV2(StrEnum):
    PENDING_REVIEW = "PENDING_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    DEFERRED = "DEFERRED"
    REVISION_REQUESTED = "REVISION_REQUESTED"
    SUPERSEDED = "SUPERSEDED"
    RESUMED = "RESUMED"


class PlanDecisionKindV2(StrEnum):
    APPROVE = "APPROVE"
    EDIT = "EDIT"
    REJECT = "REJECT"
    DEFER = "DEFER"
    REQUEST_MORE = "REQUEST_MORE"


class _FactoryObjectV2(ContractModelV2):
    object_id: Identifier
    object_sha256: Sha256
    audit: ContractAudit

    OBJECT_TYPE: ClassVar[str]
    OBJECT_VERSION: ClassVar[str] = "v2"

    @classmethod
    def create(cls, *, audit: ContractAudit, **values: object) -> Self:
        return cls._create(audit=audit, values=values)

    @classmethod
    def _create(cls, *, audit: ContractAudit, values: dict[str, object]) -> Self:
        provisional = cls.model_construct(
            object_id=f"{cls.OBJECT_TYPE}://pending",
            object_sha256="0" * 64,
            audit=audit,
            **values,  # type: ignore[arg-type]
        )
        safe_audit = _audit_with_refs(audit, _collect_refs(provisional))
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
        expected_sha256 = _carried_sha256(self)
        expected_id = f"{self.OBJECT_TYPE}://sha256/{expected_sha256}"
        if self.object_sha256 != expected_sha256 or self.object_id != expected_id:
            raise ValueError("object fields do not match the derived identity")
        refs = _collect_refs(self)
        _require_safe_refs(refs)
        if self.audit.input_refs != refs:
            raise ValueError("audit input refs do not match the canonical object refs")
        return self

    def to_ref(self) -> ObjectRef:
        return ObjectRef(
            object_type=self.OBJECT_TYPE,
            object_id=self.object_id,
            object_version=self.OBJECT_VERSION,
            object_sha256=self.object_sha256,
        )


class FactoryRunPolicyV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/factory-run-policy/v2"] = "eval-factory/factory-run-policy/v2"
    OBJECT_TYPE: ClassVar[str] = "factory-run-policy"

    policy_id: Identifier
    allowed_task_kinds: tuple[Identifier, ...] = Field(min_length=1, max_length=256)
    max_transitions: int = Field(ge=1, le=100_000)
    max_plan_revisions: int = Field(ge=0, le=1_000)
    max_agent_attempts: int = Field(ge=1, le=100)
    max_model_requests: int = Field(ge=0, le=10_000_000)
    max_model_tokens: int = Field(ge=0, le=10_000_000_000)
    max_cost_micro_usd: int = Field(ge=0, le=10_000_000_000_000)
    production_release_allowed: Literal[False] = False
    policy_version: Literal["factory-control/phase-a-v1"] = "factory-control/phase-a-v1"

    @classmethod
    def create(  # type: ignore[override]
        cls,
        *,
        policy_id: str,
        allowed_task_kinds: tuple[str, ...],
        max_transitions: int,
        max_plan_revisions: int,
        max_agent_attempts: int,
        max_model_requests: int,
        max_model_tokens: int,
        max_cost_micro_usd: int,
        audit: ContractAudit,
    ) -> FactoryRunPolicyV2:
        return cls._create(
            audit=audit,
            values={
                "policy_id": policy_id,
                "allowed_task_kinds": tuple(sorted(allowed_task_kinds)),
                "max_transitions": max_transitions,
                "max_plan_revisions": max_plan_revisions,
                "max_agent_attempts": max_agent_attempts,
                "max_model_requests": max_model_requests,
                "max_model_tokens": max_model_tokens,
                "max_cost_micro_usd": max_cost_micro_usd,
            },
        )

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        _require_sorted_unique(self.allowed_task_kinds, "allowed_task_kinds")
        return self


class EvaluationRequirementSpecV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/evaluation-requirement-spec/v2"] = (
        "eval-factory/evaluation-requirement-spec/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "evaluation-requirement-spec"

    requirement_spec_id: Identifier
    run_id: Identifier
    source_ref: ObjectRef
    goals: tuple[SafeSummary, ...] = Field(min_length=1, max_length=128)
    constraints: tuple[SafeSummary, ...] = Field(default=(), max_length=256)
    assumptions: tuple[SafeSummary, ...] = Field(default=(), max_length=256)
    open_questions: tuple[SafeSummary, ...] = Field(default=(), max_length=256)
    requirement_version: int = Field(ge=1, le=1_000_000)

    @model_validator(mode="after")
    def validate_requirement(self) -> Self:
        _require_ref(self.source_ref, "evaluation-requirement-source", "source_ref")
        for values, label in (
            (self.goals, "goals"),
            (self.constraints, "constraints"),
            (self.assumptions, "assumptions"),
            (self.open_questions, "open_questions"),
        ):
            _require_unique(values, label)
        return self


class FactoryRunV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/factory-run/v2"] = "eval-factory/factory-run/v2"
    OBJECT_TYPE: ClassVar[str] = "factory-run"

    run_id: Identifier
    run_version: int = Field(ge=0, le=100_000_000)
    status: FactoryRunStatusV2
    policy_ref: ObjectRef
    requirement_spec_ref: ObjectRef
    current_plan_ref: ObjectRef | None
    compiled_plan_ref: ObjectRef | None
    active_task_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=100_000)
    result_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=100_000)
    pending_review_ref: ObjectRef | None
    planner_assessment_ref: ObjectRef | None
    completion_ref: ObjectRef | None
    delivery_manifest_ref: ObjectRef | None
    transition_count: int = Field(ge=0, le=100_000_000)
    model_requests_used: int = Field(ge=0, le=10_000_000)
    model_tokens_used: int = Field(ge=0, le=10_000_000_000)
    cost_micro_usd_used: int = Field(ge=0, le=10_000_000_000_000)

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        _require_ref(self.policy_ref, "factory-run-policy", "policy_ref")
        _require_ref(
            self.requirement_spec_ref,
            "evaluation-requirement-spec",
            "requirement_spec_ref",
        )
        _require_sorted_unique_refs(self.active_task_refs, "active_task_refs")
        _require_sorted_unique_refs(self.result_refs, "result_refs")
        if (self.current_plan_ref is None) != (self.compiled_plan_ref is None):
            raise ValueError("current and compiled plan refs must appear together")
        if self.status is FactoryRunStatusV2.COMPLETED:
            if self.completion_ref is None or self.delivery_manifest_ref is None:
                raise ValueError("completed run requires completion and delivery refs")
        elif self.delivery_manifest_ref is not None:
            raise ValueError("non-completed run cannot publish a delivery manifest")
        if self.status is FactoryRunStatusV2.WAITING_REVIEW and self.pending_review_ref is None:
            raise ValueError("waiting run requires a pending review ref")
        return self


class DatasetBuildPlanTaskV2(ContractModelV2):
    schema_version: Literal["eval-factory/dataset-build-plan-task/v2"] = (
        "eval-factory/dataset-build-plan-task/v2"
    )
    task_key: Identifier
    stage: Identifier
    task_kind: Identifier
    agent_role: Identifier
    dependency_task_keys: tuple[Identifier, ...] = Field(default=(), max_length=1_000)
    input_object_types: tuple[Identifier, ...] = Field(default=(), max_length=256)
    output_object_types: tuple[Identifier, ...] = Field(min_length=1, max_length=256)
    required_capability_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=256)
    acceptance_check_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=256)
    plan_review_kind: PlanKindV2 | None = None
    max_attempts: int = Field(ge=1, le=100)
    max_model_requests: int = Field(ge=0, le=1_000_000)
    max_model_tokens: int = Field(ge=0, le=1_000_000_000)
    max_cost_micro_usd: int = Field(ge=0, le=1_000_000_000_000)

    @model_validator(mode="after")
    def validate_task(self) -> Self:
        for values, label in (
            (self.dependency_task_keys, "dependency_task_keys"),
            (self.input_object_types, "input_object_types"),
            (self.output_object_types, "output_object_types"),
            (self.required_capability_ids, "required_capability_ids"),
        ):
            _require_sorted_unique(values, label)
        _require_sorted_unique_refs(self.acceptance_check_refs, "acceptance_check_refs")
        if self.task_key in self.dependency_task_keys:
            raise ValueError("plan task cannot depend on itself")
        return self


class DatasetBuildPlanV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/dataset-build-plan/v2"] = "eval-factory/dataset-build-plan/v2"
    OBJECT_TYPE: ClassVar[str] = "dataset-build-plan"

    plan_id: Identifier
    run_ref: ObjectRef
    plan_version: int = Field(ge=1, le=1_000_000)
    predecessor_plan_ref: ObjectRef | None
    goals: tuple[SafeSummary, ...] = Field(min_length=1, max_length=128)
    user_constraints: tuple[SafeSummary, ...] = Field(default=(), max_length=256)
    assumptions: tuple[SafeSummary, ...] = Field(default=(), max_length=256)
    unresolved_questions: tuple[SafeSummary, ...] = Field(default=(), max_length=256)
    stage_order: tuple[Identifier, ...] = Field(min_length=1, max_length=64)
    tasks: tuple[DatasetBuildPlanTaskV2, ...] = Field(min_length=1, max_length=100_000)
    required_review_kinds: tuple[PlanKindV2, ...] = Field(default=(), max_length=7)
    total_model_requests: int = Field(ge=0, le=10_000_000)
    total_model_tokens: int = Field(ge=0, le=10_000_000_000)
    total_cost_micro_usd: int = Field(ge=0, le=10_000_000_000_000)

    @classmethod
    def create(  # type: ignore[override]
        cls,
        *,
        plan_id: str,
        run_ref: ObjectRef,
        plan_version: int,
        predecessor_plan_ref: ObjectRef | None,
        goals: tuple[str, ...],
        user_constraints: tuple[str, ...],
        assumptions: tuple[str, ...],
        unresolved_questions: tuple[str, ...],
        stage_order: tuple[str, ...],
        tasks: tuple[DatasetBuildPlanTaskV2, ...],
        required_review_kinds: tuple[PlanKindV2, ...],
        total_model_requests: int,
        total_model_tokens: int,
        total_cost_micro_usd: int,
        audit: ContractAudit,
    ) -> DatasetBuildPlanV2:
        return cls._create(
            audit=audit,
            values={
                "plan_id": plan_id,
                "run_ref": run_ref,
                "plan_version": plan_version,
                "predecessor_plan_ref": predecessor_plan_ref,
                "goals": tuple(sorted(goals)),
                "user_constraints": tuple(sorted(user_constraints)),
                "assumptions": tuple(sorted(assumptions)),
                "unresolved_questions": tuple(sorted(unresolved_questions)),
                "stage_order": stage_order,
                "tasks": tuple(sorted(tasks, key=lambda task: task.task_key)),
                "required_review_kinds": tuple(sorted(required_review_kinds, key=lambda kind: kind.value)),
                "total_model_requests": total_model_requests,
                "total_model_tokens": total_model_tokens,
                "total_cost_micro_usd": total_cost_micro_usd,
            },
        )

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        _require_ref(self.run_ref, "factory-run", "run_ref")
        if self.plan_version == 1 and self.predecessor_plan_ref is not None:
            raise ValueError("first plan version cannot have a predecessor")
        if self.plan_version > 1:
            if self.predecessor_plan_ref is None:
                raise ValueError("successor plan requires a predecessor")
            _require_ref(
                self.predecessor_plan_ref,
                "dataset-build-plan",
                "predecessor_plan_ref",
            )
        _require_unique(self.stage_order, "stage_order")
        _require_unique(self.goals, "goals")
        _require_unique(self.user_constraints, "user_constraints")
        _require_unique(self.assumptions, "assumptions")
        _require_unique(self.unresolved_questions, "unresolved_questions")
        _require_sorted_unique(
            tuple(kind.value for kind in self.required_review_kinds),
            "required_review_kinds",
        )
        task_keys = tuple(task.task_key for task in self.tasks)
        _require_sorted_unique(task_keys, "plan task keys")
        stage_set = set(self.stage_order)
        task_set = set(task_keys)
        for task in self.tasks:
            if task.stage not in stage_set:
                raise ValueError(f"task {task.task_key} uses an unknown stage")
            missing = set(task.dependency_task_keys) - task_set
            if missing:
                raise ValueError(f"task {task.task_key} has an unknown dependency")
        _require_acyclic(self.tasks)
        if self.total_model_requests != sum(task.max_model_requests for task in self.tasks):
            raise ValueError("total model requests differ from task budgets")
        if self.total_model_tokens != sum(task.max_model_tokens for task in self.tasks):
            raise ValueError("total model tokens differ from task budgets")
        if self.total_cost_micro_usd != sum(task.max_cost_micro_usd for task in self.tasks):
            raise ValueError("total model cost differs from task budgets")
        return self


class CompiledDatasetBuildPlanV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/compiled-dataset-build-plan/v2"] = (
        "eval-factory/compiled-dataset-build-plan/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "compiled-dataset-build-plan"

    compiled_plan_id: Identifier
    source_plan_ref: ObjectRef
    policy_ref: ObjectRef
    tasks: tuple[DatasetBuildPlanTaskV2, ...] = Field(min_length=1, max_length=100_000)
    topological_task_keys: tuple[Identifier, ...] = Field(min_length=1, max_length=100_000)
    compiler_version: Literal["dataset-build-plan-compiler/v1"] = "dataset-build-plan-compiler/v1"
    production_release_allowed: Literal[False] = False

    @model_validator(mode="after")
    def validate_compiled_plan(self) -> Self:
        _require_ref(self.source_plan_ref, "dataset-build-plan", "source_plan_ref")
        _require_ref(self.policy_ref, "factory-run-policy", "policy_ref")
        task_keys = tuple(task.task_key for task in self.tasks)
        _require_sorted_unique(task_keys, "compiled task keys")
        _require_unique(self.topological_task_keys, "topological_task_keys")
        if set(task_keys) != set(self.topological_task_keys):
            raise ValueError("topological task keys must cover every compiled task")
        positions = {task_key: index for index, task_key in enumerate(self.topological_task_keys)}
        for task in self.tasks:
            if any(
                positions[dependency] >= positions[task.task_key] for dependency in task.dependency_task_keys
            ):
                raise ValueError("compiled task order violates a dependency")
        return self


class AttachmentMockWorkV2(ContractModelV2):
    schema_version: Literal["eval-factory/attachment-mock-work/v2"] = "eval-factory/attachment-mock-work/v2"

    work_key: Identifier
    artifact_group_ref: ObjectRef
    artifact_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=100_000)
    agent_role: Identifier
    dependency_work_keys: tuple[Identifier, ...] = Field(default=(), max_length=1_000)
    input_object_types: tuple[Identifier, ...] = Field(min_length=1, max_length=256)
    output_object_types: tuple[Identifier, ...] = Field(min_length=1, max_length=256)
    required_capability_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=256)
    allowed_tool_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=256)
    data_purposes: tuple[Identifier, ...] = Field(min_length=1, max_length=64)
    data_classifications: tuple[Identifier, ...] = Field(min_length=1, max_length=64)
    workspace_policy_ref: ObjectRef
    acceptance_check_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=256)
    max_attempts: int = Field(ge=1, le=100)
    max_model_requests: int = Field(ge=0, le=1_000_000)
    max_model_tokens: int = Field(ge=0, le=1_000_000_000)
    max_cost_micro_usd: int = Field(ge=0, le=1_000_000_000_000)

    @model_validator(mode="after")
    def validate_work(self) -> Self:
        _require_ref(
            self.artifact_group_ref,
            "artifact-execution-group",
            "artifact_group_ref",
        )
        _require_ref(
            self.workspace_policy_ref,
            "agent-workspace-policy",
            "workspace_policy_ref",
        )
        for values, label in (
            (self.artifact_ids, "artifact_ids"),
            (self.dependency_work_keys, "dependency_work_keys"),
            (self.input_object_types, "input_object_types"),
            (self.output_object_types, "output_object_types"),
            (self.required_capability_ids, "required_capability_ids"),
            (self.allowed_tool_ids, "allowed_tool_ids"),
            (self.data_purposes, "data_purposes"),
            (self.data_classifications, "data_classifications"),
        ):
            _require_sorted_unique(values, label)
        _require_sorted_unique_refs(
            self.acceptance_check_refs,
            "acceptance_check_refs",
        )
        if self.work_key in self.dependency_work_keys:
            raise ValueError("attachment work cannot depend on itself")
        return self


class AttachmentGenerationPlanV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/attachment-generation-plan/v2"] = (
        "eval-factory/attachment-generation-plan/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "attachment-generation-plan"

    plan_id: Identifier
    run_ref: ObjectRef
    plan_version: int = Field(ge=1, le=1_000_000)
    predecessor_plan_ref: ObjectRef | None
    producer_task_view_ref: ObjectRef
    evidence_bundle_ref: ObjectRef
    attachment_planning_context_ref: ObjectRef
    works: tuple[AttachmentMockWorkV2, ...] = Field(min_length=1, max_length=100_000)
    max_parallel_groups: int = Field(ge=1, le=10_000)
    quality_policy_ref: ObjectRef
    solvability_policy_ref: ObjectRef
    total_model_requests: int = Field(ge=0, le=10_000_000)
    total_model_tokens: int = Field(ge=0, le=10_000_000_000)
    total_cost_micro_usd: int = Field(ge=0, le=10_000_000_000_000)

    @classmethod
    def create(  # type: ignore[override]
        cls,
        *,
        plan_id: str,
        run_ref: ObjectRef,
        plan_version: int,
        predecessor_plan_ref: ObjectRef | None,
        producer_task_view_ref: ObjectRef,
        evidence_bundle_ref: ObjectRef,
        attachment_planning_context_ref: ObjectRef,
        works: tuple[AttachmentMockWorkV2, ...],
        max_parallel_groups: int,
        quality_policy_ref: ObjectRef,
        solvability_policy_ref: ObjectRef,
        total_model_requests: int,
        total_model_tokens: int,
        total_cost_micro_usd: int,
        audit: ContractAudit,
    ) -> AttachmentGenerationPlanV2:
        return cls._create(
            audit=audit,
            values={
                "plan_id": plan_id,
                "run_ref": run_ref,
                "plan_version": plan_version,
                "predecessor_plan_ref": predecessor_plan_ref,
                "producer_task_view_ref": producer_task_view_ref,
                "evidence_bundle_ref": evidence_bundle_ref,
                "attachment_planning_context_ref": (attachment_planning_context_ref),
                "works": tuple(sorted(works, key=lambda value: value.work_key)),
                "max_parallel_groups": max_parallel_groups,
                "quality_policy_ref": quality_policy_ref,
                "solvability_policy_ref": solvability_policy_ref,
                "total_model_requests": total_model_requests,
                "total_model_tokens": total_model_tokens,
                "total_cost_micro_usd": total_cost_micro_usd,
            },
        )

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        _require_ref(self.run_ref, "factory-run", "run_ref")
        _require_ref(
            self.producer_task_view_ref,
            "producer-task-view",
            "producer_task_view_ref",
        )
        if (
            self.evidence_bundle_ref.object_type != "evidence-bundle"
            or self.evidence_bundle_ref.object_version != "v1"
        ):
            raise ValueError("evidence_bundle_ref must reference evidence-bundle/v1")
        _require_ref(
            self.attachment_planning_context_ref,
            "attachment-planning-context",
            "attachment_planning_context_ref",
        )
        _require_ref(
            self.quality_policy_ref,
            "attachment-quality-policy",
            "quality_policy_ref",
        )
        _require_ref(
            self.solvability_policy_ref,
            "solvability-policy",
            "solvability_policy_ref",
        )
        if self.plan_version == 1 and self.predecessor_plan_ref is not None:
            raise ValueError("first attachment plan version cannot have a predecessor")
        if self.plan_version > 1:
            if self.predecessor_plan_ref is None:
                raise ValueError("successor attachment plan requires a predecessor")
            _require_ref(
                self.predecessor_plan_ref,
                "attachment-generation-plan",
                "predecessor_plan_ref",
            )
        work_keys = tuple(work.work_key for work in self.works)
        _require_sorted_unique(work_keys, "attachment work keys")
        known = set(work_keys)
        seen_artifacts: set[str] = set()
        for work in self.works:
            missing = set(work.dependency_work_keys) - known
            if missing:
                raise ValueError("attachment work has an unknown dependency")
            overlap = seen_artifacts.intersection(work.artifact_ids)
            if overlap:
                raise ValueError("every artifact must belong to exactly one attachment work")
            seen_artifacts.update(work.artifact_ids)
        _require_attachment_acyclic(self.works)
        if self.max_parallel_groups > len(self.works):
            raise ValueError("attachment parallelism exceeds work count")
        if self.total_model_requests != sum(work.max_model_requests for work in self.works):
            raise ValueError("attachment model request total differs from work budgets")
        if self.total_model_tokens != sum(work.max_model_tokens for work in self.works):
            raise ValueError("attachment model token total differs from work budgets")
        if self.total_cost_micro_usd != sum(work.max_cost_micro_usd for work in self.works):
            raise ValueError("attachment model cost total differs from work budgets")
        return self


class AttachmentWorkAssignmentV2(ContractModelV2):
    schema_version: Literal["eval-factory/attachment-work-assignment/v2"] = (
        "eval-factory/attachment-work-assignment/v2"
    )

    work_key: Identifier
    agent_definition_ref: ObjectRef
    capability_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_assignment(self) -> Self:
        _require_ref(
            self.agent_definition_ref,
            "agent-definition",
            "agent_definition_ref",
        )
        _require_sorted_unique_refs(self.capability_refs, "capability_refs")
        if any(ref.object_type != "agent-capability" for ref in self.capability_refs):
            raise ValueError("attachment assignment requires Agent capability refs")
        return self


class CompiledAttachmentGenerationPlanV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/compiled-attachment-generation-plan/v2"] = (
        "eval-factory/compiled-attachment-generation-plan/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "compiled-attachment-generation-plan"

    compiled_plan_id: Identifier
    source_plan_ref: ObjectRef
    policy_ref: ObjectRef
    works: tuple[AttachmentMockWorkV2, ...] = Field(min_length=1, max_length=100_000)
    assignments: tuple[AttachmentWorkAssignmentV2, ...] = Field(
        min_length=1,
        max_length=100_000,
    )
    topological_work_keys: tuple[Identifier, ...] = Field(
        min_length=1,
        max_length=100_000,
    )
    max_parallel_groups: int = Field(ge=1, le=10_000)
    compiler_version: Literal["attachment-generation-plan-compiler/v1"] = (
        "attachment-generation-plan-compiler/v1"
    )
    production_release_allowed: Literal[False] = False

    @model_validator(mode="after")
    def validate_compiled(self) -> Self:
        _require_ref(
            self.source_plan_ref,
            "attachment-generation-plan",
            "source_plan_ref",
        )
        _require_ref(self.policy_ref, "factory-run-policy", "policy_ref")
        work_keys = tuple(work.work_key for work in self.works)
        assignment_keys = tuple(value.work_key for value in self.assignments)
        _require_sorted_unique(work_keys, "compiled attachment work keys")
        _require_sorted_unique(
            assignment_keys,
            "compiled attachment assignment keys",
        )
        _require_unique(self.topological_work_keys, "topological_work_keys")
        if set(work_keys) != set(assignment_keys) or set(work_keys) != set(self.topological_work_keys):
            raise ValueError("compiled attachment inventories must exactly match")
        positions = {work_key: index for index, work_key in enumerate(self.topological_work_keys)}
        for work in self.works:
            if any(
                positions[dependency] >= positions[work.work_key] for dependency in work.dependency_work_keys
            ):
                raise ValueError("compiled attachment order violates a dependency")
        if self.max_parallel_groups > len(self.works):
            raise ValueError("compiled attachment parallelism exceeds work count")
        return self


class CriteriaRubricGoalV2(ContractModelV2):
    schema_version: Literal["eval-factory/criteria-rubric-goal/v2"] = "eval-factory/criteria-rubric-goal/v2"

    goal_id: Identifier
    goal_summary: SafeSummary
    judged_object_kind: RubricJudgedObjectKindV2
    prompt_requirement_ids: tuple[Identifier, ...] = Field(
        default=(),
        max_length=10_000,
    )
    attachment_dependency_ids: tuple[Identifier, ...] = Field(
        default=(),
        max_length=10_000,
    )
    allowed_tool_ids: tuple[Identifier, ...] = Field(
        default=(),
        max_length=1_000,
    )
    evaluator_binding_id: Identifier
    weight_basis_points: int = Field(ge=1, le=10_000)
    visibility: Literal["EVALUATOR_ONLY"] = "EVALUATOR_ONLY"

    @model_validator(mode="after")
    def validate_goal(self) -> Self:
        for values, label in (
            (self.prompt_requirement_ids, "prompt_requirement_ids"),
            (
                self.attachment_dependency_ids,
                "attachment_dependency_ids",
            ),
            (self.allowed_tool_ids, "allowed_tool_ids"),
        ):
            _require_sorted_unique(values, label)
        if (
            self.judged_object_kind is RubricJudgedObjectKindV2.WORKSPACE_STATE
            and not self.attachment_dependency_ids
        ):
            raise ValueError("workspace-state criteria require an attachment dependency")
        if self.judged_object_kind is RubricJudgedObjectKindV2.TOOL_BEHAVIOR and not self.allowed_tool_ids:
            raise ValueError("tool-behavior criteria require an allowed tool")
        return self


class CriteriaRubricPlanV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/criteria-rubric-plan/v2"] = "eval-factory/criteria-rubric-plan/v2"
    OBJECT_TYPE: ClassVar[str] = "criteria-rubric-plan"

    plan_id: Identifier
    run_ref: ObjectRef
    plan_version: int = Field(ge=1, le=1_000_000)
    predecessor_plan_ref: ObjectRef | None
    task_draft_ref: ObjectRef
    attachment_quality_ref: ObjectRef
    solvability_ref: ObjectRef
    allowed_prompt_requirement_ids: tuple[Identifier, ...] = Field(
        min_length=1,
        max_length=10_000,
    )
    required_prompt_requirement_ids: tuple[Identifier, ...] = Field(
        min_length=1,
        max_length=10_000,
    )
    allowed_attachment_dependency_ids: tuple[Identifier, ...] = Field(
        default=(),
        max_length=10_000,
    )
    required_attachment_dependency_ids: tuple[Identifier, ...] = Field(
        default=(),
        max_length=10_000,
    )
    allowed_task_tool_ids: tuple[Identifier, ...] = Field(
        default=(),
        max_length=1_000,
    )
    required_task_tool_ids: tuple[Identifier, ...] = Field(
        default=(),
        max_length=1_000,
    )
    criterion_goals: tuple[CriteriaRubricGoalV2, ...] = Field(
        min_length=1,
        max_length=10_000,
    )
    allowed_evaluator_binding_ids: tuple[Identifier, ...] = Field(
        min_length=1,
        max_length=1_000,
    )
    allowed_reference_modes: tuple[ReferenceMode, ...] = Field(
        min_length=1,
        max_length=5,
    )
    selected_reference_mode: ReferenceMode
    tool_catalog_ref: ObjectRef
    agent_role: Identifier
    required_capability_ids: tuple[Identifier, ...] = Field(
        min_length=1,
        max_length=256,
    )
    specialist_tool_ids: tuple[Identifier, ...] = Field(
        min_length=1,
        max_length=256,
    )
    data_purpose: Literal["criteria-rubric-authoring"] = "criteria-rubric-authoring"
    data_classifications: tuple[Identifier, ...] = Field(
        min_length=1,
        max_length=64,
    )
    prompt_template_ref: ObjectRef
    model_policy_ref: ObjectRef
    acceptance_check_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=256,
    )
    max_attempts: int = Field(ge=1, le=100)
    max_model_requests: int = Field(ge=1, le=1_000_000)
    max_model_tokens: int = Field(ge=1, le=1_000_000_000)
    max_cost_micro_usd: int = Field(ge=1, le=1_000_000_000_000)

    @classmethod
    def create(  # type: ignore[override]
        cls,
        *,
        plan_id: str,
        run_ref: ObjectRef,
        plan_version: int,
        predecessor_plan_ref: ObjectRef | None,
        task_draft_ref: ObjectRef,
        attachment_quality_ref: ObjectRef,
        solvability_ref: ObjectRef,
        allowed_prompt_requirement_ids: tuple[str, ...],
        required_prompt_requirement_ids: tuple[str, ...],
        allowed_attachment_dependency_ids: tuple[str, ...],
        required_attachment_dependency_ids: tuple[str, ...],
        allowed_task_tool_ids: tuple[str, ...],
        required_task_tool_ids: tuple[str, ...],
        criterion_goals: tuple[CriteriaRubricGoalV2, ...],
        allowed_evaluator_binding_ids: tuple[str, ...],
        allowed_reference_modes: tuple[ReferenceMode, ...],
        selected_reference_mode: ReferenceMode,
        tool_catalog_ref: ObjectRef,
        agent_role: str,
        required_capability_ids: tuple[str, ...],
        specialist_tool_ids: tuple[str, ...],
        data_classifications: tuple[str, ...],
        prompt_template_ref: ObjectRef,
        model_policy_ref: ObjectRef,
        acceptance_check_refs: tuple[ObjectRef, ...],
        max_attempts: int,
        max_model_requests: int,
        max_model_tokens: int,
        max_cost_micro_usd: int,
        audit: ContractAudit,
    ) -> CriteriaRubricPlanV2:
        return cls._create(
            audit=audit,
            values={
                "plan_id": plan_id,
                "run_ref": run_ref,
                "plan_version": plan_version,
                "predecessor_plan_ref": predecessor_plan_ref,
                "task_draft_ref": task_draft_ref,
                "attachment_quality_ref": attachment_quality_ref,
                "solvability_ref": solvability_ref,
                "allowed_prompt_requirement_ids": tuple(sorted(allowed_prompt_requirement_ids)),
                "required_prompt_requirement_ids": tuple(sorted(required_prompt_requirement_ids)),
                "allowed_attachment_dependency_ids": tuple(sorted(allowed_attachment_dependency_ids)),
                "required_attachment_dependency_ids": tuple(sorted(required_attachment_dependency_ids)),
                "allowed_task_tool_ids": tuple(sorted(allowed_task_tool_ids)),
                "required_task_tool_ids": tuple(sorted(required_task_tool_ids)),
                "criterion_goals": tuple(
                    sorted(
                        criterion_goals,
                        key=lambda value: value.goal_id,
                    )
                ),
                "allowed_evaluator_binding_ids": tuple(sorted(allowed_evaluator_binding_ids)),
                "allowed_reference_modes": tuple(
                    sorted(
                        allowed_reference_modes,
                        key=lambda value: value.value,
                    )
                ),
                "selected_reference_mode": selected_reference_mode,
                "tool_catalog_ref": tool_catalog_ref,
                "agent_role": agent_role,
                "required_capability_ids": tuple(sorted(required_capability_ids)),
                "specialist_tool_ids": tuple(sorted(specialist_tool_ids)),
                "data_classifications": tuple(sorted(data_classifications)),
                "prompt_template_ref": prompt_template_ref,
                "model_policy_ref": model_policy_ref,
                "acceptance_check_refs": tuple(sorted(acceptance_check_refs, key=_ref_key)),
                "max_attempts": max_attempts,
                "max_model_requests": max_model_requests,
                "max_model_tokens": max_model_tokens,
                "max_cost_micro_usd": max_cost_micro_usd,
            },
        )

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        for ref, object_type, label in (
            (self.run_ref, "factory-run", "run_ref"),
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
                self.prompt_template_ref,
                "prompt-template",
                "prompt_template_ref",
            ),
            (
                self.model_policy_ref,
                "model-routing-policy",
                "model_policy_ref",
            ),
        ):
            _require_ref(ref, object_type, label)
        if (
            self.tool_catalog_ref.object_type != "tool-capability-catalog"
            or self.tool_catalog_ref.object_version != "tool-capability-catalog/r4-07-v1"
        ):
            raise ValueError("tool_catalog_ref must reference the current R4-07 catalog")
        if self.plan_version == 1 and self.predecessor_plan_ref is not None:
            raise ValueError("first criteria/rubric plan cannot have a predecessor")
        if self.plan_version > 1:
            if self.predecessor_plan_ref is None:
                raise ValueError("criteria/rubric successor requires a predecessor")
            _require_ref(
                self.predecessor_plan_ref,
                "criteria-rubric-plan",
                "predecessor_plan_ref",
            )
        for values, label in (
            (
                self.allowed_prompt_requirement_ids,
                "allowed_prompt_requirement_ids",
            ),
            (
                self.required_prompt_requirement_ids,
                "required_prompt_requirement_ids",
            ),
            (
                self.allowed_attachment_dependency_ids,
                "allowed_attachment_dependency_ids",
            ),
            (
                self.required_attachment_dependency_ids,
                "required_attachment_dependency_ids",
            ),
            (self.allowed_task_tool_ids, "allowed_task_tool_ids"),
            (self.required_task_tool_ids, "required_task_tool_ids"),
            (
                self.allowed_evaluator_binding_ids,
                "allowed_evaluator_binding_ids",
            ),
            (
                self.required_capability_ids,
                "required_capability_ids",
            ),
            (self.specialist_tool_ids, "specialist_tool_ids"),
            (self.data_classifications, "data_classifications"),
        ):
            _require_sorted_unique(values, label)
        _require_sorted_unique(
            tuple(value.value for value in self.allowed_reference_modes),
            "allowed_reference_modes",
        )
        _require_sorted_unique_refs(
            self.acceptance_check_refs,
            "acceptance_check_refs",
        )
        if not set(self.required_prompt_requirement_ids).issubset(self.allowed_prompt_requirement_ids):
            raise ValueError("required prompt requirements exceed allowed scope")
        if not set(self.required_attachment_dependency_ids).issubset(self.allowed_attachment_dependency_ids):
            raise ValueError("required attachment dependencies exceed allowed scope")
        if not set(self.required_task_tool_ids).issubset(self.allowed_task_tool_ids):
            raise ValueError("required task tools exceed allowed scope")
        if self.selected_reference_mode not in self.allowed_reference_modes:
            raise ValueError("selected reference mode is outside the allowlist")
        goal_ids = tuple(value.goal_id for value in self.criterion_goals)
        _require_sorted_unique(goal_ids, "criterion goal IDs")
        if sum(value.weight_basis_points for value in self.criterion_goals) != 10_000:
            raise ValueError("criteria weights must total 10000 basis points")
        covered_prompts: set[str] = set()
        covered_dependencies: set[str] = set()
        covered_tools: set[str] = set()
        for goal in self.criterion_goals:
            if not set(goal.prompt_requirement_ids).issubset(self.allowed_prompt_requirement_ids):
                raise ValueError("criterion goal widens prompt requirement scope")
            if not set(goal.attachment_dependency_ids).issubset(self.allowed_attachment_dependency_ids):
                raise ValueError("criterion goal widens attachment dependency scope")
            if not set(goal.allowed_tool_ids).issubset(self.allowed_task_tool_ids):
                raise ValueError("criterion goal widens task tool scope")
            if goal.evaluator_binding_id not in self.allowed_evaluator_binding_ids:
                raise ValueError("criterion goal uses an unapproved evaluator binding")
            covered_prompts.update(goal.prompt_requirement_ids)
            covered_dependencies.update(goal.attachment_dependency_ids)
            covered_tools.update(goal.allowed_tool_ids)
        if not set(self.required_prompt_requirement_ids).issubset(covered_prompts):
            raise ValueError("criterion goals do not cover required prompt requirements")
        if not set(self.required_attachment_dependency_ids).issubset(covered_dependencies):
            raise ValueError("criterion goals do not cover required attachments")
        if not set(self.required_task_tool_ids).issubset(covered_tools):
            raise ValueError("criterion goals do not cover required task tools")
        _require_safe_refs(_collect_refs(self))
        return self


class CompiledCriteriaRubricPlanV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/compiled-criteria-rubric-plan/v2"] = (
        "eval-factory/compiled-criteria-rubric-plan/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "compiled-criteria-rubric-plan"

    compiled_plan_id: Identifier
    source_plan_ref: ObjectRef
    policy_ref: ObjectRef
    agent_definition_ref: ObjectRef
    capability_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=256,
    )
    task_draft_ref: ObjectRef
    attachment_quality_ref: ObjectRef
    solvability_ref: ObjectRef
    tool_catalog_ref: ObjectRef
    prompt_template_ref: ObjectRef
    model_policy_ref: ObjectRef
    acceptance_check_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=256,
    )
    compiler_version: Literal["criteria-rubric-plan-compiler/v1"] = "criteria-rubric-plan-compiler/v1"
    production_release_allowed: Literal[False] = False

    @model_validator(mode="after")
    def validate_compiled(self) -> Self:
        for ref, object_type, label in (
            (
                self.source_plan_ref,
                "criteria-rubric-plan",
                "source_plan_ref",
            ),
            (self.policy_ref, "factory-run-policy", "policy_ref"),
            (
                self.agent_definition_ref,
                "agent-definition",
                "agent_definition_ref",
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
                self.prompt_template_ref,
                "prompt-template",
                "prompt_template_ref",
            ),
            (
                self.model_policy_ref,
                "model-routing-policy",
                "model_policy_ref",
            ),
        ):
            _require_ref(ref, object_type, label)
        if (
            self.tool_catalog_ref.object_type != "tool-capability-catalog"
            or self.tool_catalog_ref.object_version != "tool-capability-catalog/r4-07-v1"
        ):
            raise ValueError("tool_catalog_ref must reference the current R4-07 catalog")
        _require_sorted_unique_refs(self.capability_refs, "capability_refs")
        if any(ref.object_type != "agent-capability" for ref in self.capability_refs):
            raise ValueError("criteria/rubric compilation requires Agent capabilities")
        _require_sorted_unique_refs(
            self.acceptance_check_refs,
            "acceptance_check_refs",
        )
        _require_safe_refs(_collect_refs(self))
        return self


class CriteriaRubricResultV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/criteria-rubric-result/v2"] = (
        "eval-factory/criteria-rubric-result/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "criteria-rubric-result"

    result_id: Identifier
    plan_ref: ObjectRef
    task_draft_ref: ObjectRef
    attachment_quality_ref: ObjectRef
    solvability_ref: ObjectRef
    route_decision_ref: ObjectRef
    gateway_receipt_ref: ObjectRef
    gateway_invocation_result_ref: ObjectRef
    proposal_ref: ObjectRef | None
    rubric_set_ref: ObjectRef | None
    evaluator_spec_ref: ObjectRef | None
    reference_policy_ref: ObjectRef | None
    tool_policy_ref: ObjectRef | None
    contestant_tool_policy_ref: ObjectRef | None
    outcome: CriteriaRubricOutcomeV2
    reason_codes: tuple[SafeCode, ...] = Field(
        default=(),
        max_length=256,
    )

    @classmethod
    def create(  # type: ignore[override]
        cls,
        *,
        result_id: str,
        plan_ref: ObjectRef,
        task_draft_ref: ObjectRef,
        attachment_quality_ref: ObjectRef,
        solvability_ref: ObjectRef,
        route_decision_ref: ObjectRef,
        gateway_receipt_ref: ObjectRef,
        gateway_invocation_result_ref: ObjectRef,
        proposal_ref: ObjectRef | None,
        rubric_set_ref: ObjectRef | None,
        evaluator_spec_ref: ObjectRef | None,
        reference_policy_ref: ObjectRef | None,
        tool_policy_ref: ObjectRef | None,
        contestant_tool_policy_ref: ObjectRef | None,
        outcome: CriteriaRubricOutcomeV2,
        reason_codes: tuple[str, ...],
        audit: ContractAudit,
    ) -> CriteriaRubricResultV2:
        return cls._create(
            audit=audit,
            values={
                "result_id": result_id,
                "plan_ref": plan_ref,
                "task_draft_ref": task_draft_ref,
                "attachment_quality_ref": attachment_quality_ref,
                "solvability_ref": solvability_ref,
                "route_decision_ref": route_decision_ref,
                "gateway_receipt_ref": gateway_receipt_ref,
                "gateway_invocation_result_ref": (gateway_invocation_result_ref),
                "proposal_ref": proposal_ref,
                "rubric_set_ref": rubric_set_ref,
                "evaluator_spec_ref": evaluator_spec_ref,
                "reference_policy_ref": reference_policy_ref,
                "tool_policy_ref": tool_policy_ref,
                "contestant_tool_policy_ref": contestant_tool_policy_ref,
                "outcome": outcome,
                "reason_codes": tuple(sorted(reason_codes)),
            },
        )

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        for ref, object_type, label in (
            (self.plan_ref, "criteria-rubric-plan", "plan_ref"),
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
                self.route_decision_ref,
                "model-route-decision",
                "route_decision_ref",
            ),
            (
                self.gateway_receipt_ref,
                "gateway-receipt",
                "gateway_receipt_ref",
            ),
            (
                self.gateway_invocation_result_ref,
                "gateway-invocation-result",
                "gateway_invocation_result_ref",
            ),
        ):
            _require_ref(ref, object_type, label)
        optional_refs = (
            (self.proposal_ref, "criteria-rubric-proposal", "proposal_ref"),
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
                self.contestant_tool_policy_ref,
                "contestant-tool-policy",
                "contestant_tool_policy_ref",
            ),
        )
        for optional_ref, object_type, label in optional_refs:
            if optional_ref is not None:
                _require_ref(optional_ref, object_type, label)
        chain_refs = (
            self.rubric_set_ref,
            self.evaluator_spec_ref,
            self.reference_policy_ref,
            self.tool_policy_ref,
            self.contestant_tool_policy_ref,
        )
        _require_sorted_unique(self.reason_codes, "reason_codes")
        if self.outcome is CriteriaRubricOutcomeV2.SUCCEEDED:
            if self.proposal_ref is None or any(ref is None for ref in chain_refs):
                raise ValueError("successful criteria/rubric result requires a complete chain")
            if self.reason_codes:
                raise ValueError("successful criteria/rubric result cannot retain reasons")
        else:
            if any(ref is not None for ref in chain_refs):
                raise ValueError("non-successful criteria/rubric result cannot publish a partial chain")
            if not self.reason_codes:
                raise ValueError("non-successful criteria/rubric result requires reasons")
        _require_safe_refs(_collect_refs(self))
        return self


class JudgeTaskMappingV2(ContractModelV2):
    schema_version: Literal["eval-factory/judge-task-mapping/v2"] = "eval-factory/judge-task-mapping/v2"

    task_key: Identifier
    criterion_ids: tuple[Identifier, ...] = Field(
        min_length=1,
        max_length=10_000,
    )
    evaluator_binding_id: Identifier
    score_weight_basis_points: int = Field(ge=1, le=10_000)

    @model_validator(mode="after")
    def validate_mapping(self) -> Self:
        _require_sorted_unique(
            self.criterion_ids,
            "judge task criterion IDs",
        )
        return self


class GradingDesignPlanV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/grading-design-plan/v2"] = "eval-factory/grading-design-plan/v2"
    OBJECT_TYPE: ClassVar[str] = "grading-design-plan"

    plan_id: Identifier
    run_ref: ObjectRef
    plan_version: int = Field(ge=1, le=1_000_000)
    predecessor_plan_ref: ObjectRef | None
    criteria_rubric_result_ref: ObjectRef
    rubric_set_ref: ObjectRef
    evaluator_spec_ref: ObjectRef
    reference_policy_ref: ObjectRef
    tool_policy_ref: ObjectRef
    generator_model_profile_ref: ObjectRef
    judge_tasks: tuple[JudgeTaskMappingV2, ...] = Field(
        min_length=1,
        max_length=10_000,
    )
    aggregation_mode: JudgeAggregationModeV2
    passing_score_basis_points: int = Field(ge=1, le=10_000)
    minimum_confidence_basis_points: int = Field(ge=0, le=10_000)
    escalate_on_reference_unavailable: bool
    judge_input_schema_ref: ObjectRef
    judge_output_schema_ref: ObjectRef
    required_output_fields: tuple[SafeCode, ...] = Field(
        min_length=4,
        max_length=16,
    )
    allowed_judge_model_profile_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=1_000,
    )
    judge_prompt_template_ref: ObjectRef
    agent_role: Identifier
    required_capability_ids: tuple[Identifier, ...] = Field(
        min_length=1,
        max_length=256,
    )
    specialist_tool_ids: tuple[Identifier, ...] = Field(
        min_length=1,
        max_length=256,
    )
    data_purpose: Literal["grading-design-authoring"] = "grading-design-authoring"
    data_classifications: tuple[Identifier, ...] = Field(
        min_length=1,
        max_length=64,
    )
    model_policy_ref: ObjectRef
    acceptance_check_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=256,
    )
    max_attempts: int = Field(ge=1, le=100)
    max_model_requests: int = Field(ge=1, le=1_000_000)
    max_model_tokens: int = Field(ge=1, le=1_000_000_000)
    max_cost_micro_usd: int = Field(ge=1, le=1_000_000_000_000)

    @classmethod
    def create(  # type: ignore[override]
        cls,
        *,
        plan_id: str,
        run_ref: ObjectRef,
        plan_version: int,
        predecessor_plan_ref: ObjectRef | None,
        criteria_rubric_result_ref: ObjectRef,
        rubric_set_ref: ObjectRef,
        evaluator_spec_ref: ObjectRef,
        reference_policy_ref: ObjectRef,
        tool_policy_ref: ObjectRef,
        generator_model_profile_ref: ObjectRef,
        judge_tasks: tuple[JudgeTaskMappingV2, ...],
        aggregation_mode: JudgeAggregationModeV2,
        passing_score_basis_points: int,
        minimum_confidence_basis_points: int,
        escalate_on_reference_unavailable: bool,
        judge_input_schema_ref: ObjectRef,
        judge_output_schema_ref: ObjectRef,
        required_output_fields: tuple[str, ...],
        allowed_judge_model_profile_refs: tuple[ObjectRef, ...],
        judge_prompt_template_ref: ObjectRef,
        agent_role: str,
        required_capability_ids: tuple[str, ...],
        specialist_tool_ids: tuple[str, ...],
        data_classifications: tuple[str, ...],
        model_policy_ref: ObjectRef,
        acceptance_check_refs: tuple[ObjectRef, ...],
        max_attempts: int,
        max_model_requests: int,
        max_model_tokens: int,
        max_cost_micro_usd: int,
        audit: ContractAudit,
    ) -> GradingDesignPlanV2:
        return cls._create(
            audit=audit,
            values={
                "plan_id": plan_id,
                "run_ref": run_ref,
                "plan_version": plan_version,
                "predecessor_plan_ref": predecessor_plan_ref,
                "criteria_rubric_result_ref": (criteria_rubric_result_ref),
                "rubric_set_ref": rubric_set_ref,
                "evaluator_spec_ref": evaluator_spec_ref,
                "reference_policy_ref": reference_policy_ref,
                "tool_policy_ref": tool_policy_ref,
                "generator_model_profile_ref": (generator_model_profile_ref),
                "judge_tasks": tuple(
                    sorted(
                        judge_tasks,
                        key=lambda value: value.task_key,
                    )
                ),
                "aggregation_mode": aggregation_mode,
                "passing_score_basis_points": (passing_score_basis_points),
                "minimum_confidence_basis_points": (minimum_confidence_basis_points),
                "escalate_on_reference_unavailable": (escalate_on_reference_unavailable),
                "judge_input_schema_ref": judge_input_schema_ref,
                "judge_output_schema_ref": judge_output_schema_ref,
                "required_output_fields": tuple(sorted(required_output_fields)),
                "allowed_judge_model_profile_refs": tuple(
                    sorted(
                        allowed_judge_model_profile_refs,
                        key=_ref_key,
                    )
                ),
                "judge_prompt_template_ref": (judge_prompt_template_ref),
                "agent_role": agent_role,
                "required_capability_ids": tuple(sorted(required_capability_ids)),
                "specialist_tool_ids": tuple(sorted(specialist_tool_ids)),
                "data_classifications": tuple(sorted(data_classifications)),
                "model_policy_ref": model_policy_ref,
                "acceptance_check_refs": tuple(
                    sorted(
                        acceptance_check_refs,
                        key=_ref_key,
                    )
                ),
                "max_attempts": max_attempts,
                "max_model_requests": max_model_requests,
                "max_model_tokens": max_model_tokens,
                "max_cost_micro_usd": max_cost_micro_usd,
            },
        )

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        for ref, object_type, label in (
            (self.run_ref, "factory-run", "run_ref"),
            (
                self.criteria_rubric_result_ref,
                "criteria-rubric-result",
                "criteria_rubric_result_ref",
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
                self.generator_model_profile_ref,
                "model-capability-profile",
                "generator_model_profile_ref",
            ),
            (
                self.judge_input_schema_ref,
                "json-schema",
                "judge_input_schema_ref",
            ),
            (
                self.judge_output_schema_ref,
                "json-schema",
                "judge_output_schema_ref",
            ),
            (
                self.judge_prompt_template_ref,
                "prompt-template",
                "judge_prompt_template_ref",
            ),
            (
                self.model_policy_ref,
                "model-routing-policy",
                "model_policy_ref",
            ),
        ):
            _require_ref(ref, object_type, label)
        if self.plan_version == 1 and self.predecessor_plan_ref is not None:
            raise ValueError("first grading plan cannot have a predecessor")
        if self.plan_version > 1:
            if self.predecessor_plan_ref is None:
                raise ValueError("grading successor requires a predecessor")
            _require_ref(
                self.predecessor_plan_ref,
                "grading-design-plan",
                "predecessor_plan_ref",
            )
        task_keys = tuple(value.task_key for value in self.judge_tasks)
        _require_sorted_unique(task_keys, "judge task keys")
        criterion_ids = tuple(
            criterion_id for task in self.judge_tasks for criterion_id in task.criterion_ids
        )
        _require_unique(
            criterion_ids,
            "criterion IDs across judge tasks",
        )
        if sum(value.score_weight_basis_points for value in self.judge_tasks) != 10_000:
            raise ValueError("judge task weights must total 10000")
        expected_fields = (
            "ABSTAIN",
            "EVIDENCE_IDS",
            "FAILURE_CLASS",
            "SCORE_BASIS_POINTS",
        )
        if self.required_output_fields != expected_fields:
            raise ValueError("judge output fields must cover score, evidence, abstain and failure class")
        _require_sorted_unique_refs(
            self.allowed_judge_model_profile_refs,
            "allowed judge model refs",
        )
        if any(
            ref.object_type != "model-capability-profile" for ref in self.allowed_judge_model_profile_refs
        ):
            raise ValueError("allowed judge models must use capability profiles")
        for values, label in (
            (
                self.required_capability_ids,
                "required_capability_ids",
            ),
            (self.specialist_tool_ids, "specialist_tool_ids"),
            (
                self.data_classifications,
                "data_classifications",
            ),
        ):
            _require_sorted_unique(values, label)
        _require_sorted_unique_refs(
            self.acceptance_check_refs,
            "acceptance_check_refs",
        )
        _require_safe_refs(_collect_refs(self))
        return self


class CompiledGradingDesignPlanV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/compiled-grading-design-plan/v2"] = (
        "eval-factory/compiled-grading-design-plan/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "compiled-grading-design-plan"

    compiled_plan_id: Identifier
    source_plan_ref: ObjectRef
    policy_ref: ObjectRef
    agent_definition_ref: ObjectRef
    capability_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=256,
    )
    criteria_rubric_result_ref: ObjectRef
    rubric_set_ref: ObjectRef
    evaluator_spec_ref: ObjectRef
    reference_policy_ref: ObjectRef
    tool_policy_ref: ObjectRef
    generator_model_profile_ref: ObjectRef
    judge_prompt_template_ref: ObjectRef
    model_policy_ref: ObjectRef
    acceptance_check_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=256,
    )
    compiler_version: Literal["grading-design-plan-compiler/v1"] = "grading-design-plan-compiler/v1"
    production_release_allowed: Literal[False] = False

    @model_validator(mode="after")
    def validate_compiled(self) -> Self:
        for ref, object_type, label in (
            (
                self.source_plan_ref,
                "grading-design-plan",
                "source_plan_ref",
            ),
            (self.policy_ref, "factory-run-policy", "policy_ref"),
            (
                self.agent_definition_ref,
                "agent-definition",
                "agent_definition_ref",
            ),
            (
                self.criteria_rubric_result_ref,
                "criteria-rubric-result",
                "criteria_rubric_result_ref",
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
                self.generator_model_profile_ref,
                "model-capability-profile",
                "generator_model_profile_ref",
            ),
            (
                self.judge_prompt_template_ref,
                "prompt-template",
                "judge_prompt_template_ref",
            ),
            (
                self.model_policy_ref,
                "model-routing-policy",
                "model_policy_ref",
            ),
        ):
            _require_ref(ref, object_type, label)
        _require_sorted_unique_refs(
            self.capability_refs,
            "capability_refs",
        )
        if any(ref.object_type != "agent-capability" for ref in self.capability_refs):
            raise ValueError("grading compilation requires Agent capabilities")
        _require_sorted_unique_refs(
            self.acceptance_check_refs,
            "acceptance_check_refs",
        )
        _require_safe_refs(_collect_refs(self))
        return self


class JudgeDesignSpecV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/judge-design-spec/v2"] = "eval-factory/judge-design-spec/v2"
    OBJECT_TYPE: ClassVar[str] = "judge-design-spec"

    design_spec_id: Identifier
    plan_ref: ObjectRef
    criteria_rubric_result_ref: ObjectRef
    rubric_set_ref: ObjectRef
    evaluator_spec_ref: ObjectRef
    reference_policy_ref: ObjectRef
    tool_policy_ref: ObjectRef
    prompt_template_ref: ObjectRef
    prompt_rendering_ref: ObjectRef
    route_decision_ref: ObjectRef
    gateway_receipt_ref: ObjectRef
    gateway_invocation_result_ref: ObjectRef
    proposal_ref: ObjectRef
    selected_judge_model_profile_ref: ObjectRef
    generator_model_profile_ref: ObjectRef
    judge_tasks: tuple[JudgeTaskMappingV2, ...] = Field(
        min_length=1,
        max_length=10_000,
    )
    aggregation_mode: JudgeAggregationModeV2
    passing_score_basis_points: int = Field(ge=1, le=10_000)
    minimum_confidence_basis_points: int = Field(ge=0, le=10_000)
    escalate_on_reference_unavailable: bool
    judge_input_schema_ref: ObjectRef
    judge_output_schema_ref: ObjectRef
    required_output_fields: tuple[SafeCode, ...] = Field(
        min_length=4,
        max_length=16,
    )
    reference_grant_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=10_000,
    )
    private_material_inventory_sha256: Sha256
    policy_version: Literal["judge-design/v1"] = "judge-design/v1"

    @model_validator(mode="after")
    def validate_spec(self) -> Self:
        expected = (
            (self.plan_ref, "grading-design-plan"),
            (
                self.criteria_rubric_result_ref,
                "criteria-rubric-result",
            ),
            (self.rubric_set_ref, "rubric-set"),
            (self.evaluator_spec_ref, "evaluator-spec"),
            (self.reference_policy_ref, "reference-policy"),
            (self.tool_policy_ref, "tool-policy"),
            (self.prompt_template_ref, "prompt-template"),
            (self.prompt_rendering_ref, "prompt-rendering"),
            (self.route_decision_ref, "model-route-decision"),
            (self.gateway_receipt_ref, "gateway-receipt"),
            (
                self.gateway_invocation_result_ref,
                "gateway-invocation-result",
            ),
            (self.proposal_ref, "judge-design-proposal"),
            (
                self.selected_judge_model_profile_ref,
                "model-capability-profile",
            ),
            (
                self.generator_model_profile_ref,
                "model-capability-profile",
            ),
            (self.judge_input_schema_ref, "json-schema"),
            (self.judge_output_schema_ref, "json-schema"),
        )
        for ref, object_type in expected:
            _require_ref(ref, object_type, object_type)
        if self.selected_judge_model_profile_ref == self.generator_model_profile_ref:
            raise ValueError("judge model must differ from generator model")
        task_keys = tuple(value.task_key for value in self.judge_tasks)
        _require_sorted_unique(task_keys, "judge task keys")
        criterion_ids = tuple(
            criterion_id for task in self.judge_tasks for criterion_id in task.criterion_ids
        )
        _require_unique(
            criterion_ids,
            "criterion IDs across judge tasks",
        )
        _require_sorted_unique(
            self.required_output_fields,
            "required_output_fields",
        )
        _require_sorted_unique_refs(
            self.reference_grant_refs,
            "reference_grant_refs",
        )
        if any(ref.object_type != "evaluator-reference-grant" for ref in self.reference_grant_refs):
            raise ValueError("reference grants must use evaluator-reference-grant refs")
        _require_safe_refs(_collect_refs(self))
        return self


class JudgeDesignValidationV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/judge-design-validation/v2"] = (
        "eval-factory/judge-design-validation/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "judge-design-validation"

    validation_id: Identifier
    plan_ref: ObjectRef
    design_spec_ref: ObjectRef | None
    criterion_ids: tuple[Identifier, ...] = Field(
        default=(),
        max_length=10_000,
    )
    validated_output_fields: tuple[SafeCode, ...] = Field(
        default=(),
        max_length=16,
    )
    reference_access_result_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=10_000,
    )
    outcome: JudgeDesignValidationOutcomeV2
    reason_codes: tuple[SafeCode, ...] = Field(
        default=(),
        max_length=256,
    )

    @model_validator(mode="after")
    def validate_validation(self) -> Self:
        _require_ref(
            self.plan_ref,
            "grading-design-plan",
            "plan_ref",
        )
        if self.design_spec_ref is not None:
            _require_ref(
                self.design_spec_ref,
                "judge-design-spec",
                "design_spec_ref",
            )
        _require_sorted_unique(
            self.criterion_ids,
            "criterion_ids",
        )
        _require_sorted_unique(
            self.validated_output_fields,
            "validated_output_fields",
        )
        _require_sorted_unique_refs(
            self.reference_access_result_refs,
            "reference_access_result_refs",
        )
        if any(
            ref.object_type != "evaluator-reference-access-result"
            for ref in self.reference_access_result_refs
        ):
            raise ValueError("reference access results use an invalid ref type")
        _require_sorted_unique(self.reason_codes, "reason_codes")
        if self.outcome is JudgeDesignValidationOutcomeV2.VALID:
            if (
                self.design_spec_ref is None
                or not self.criterion_ids
                or not self.validated_output_fields
                or self.reason_codes
            ):
                raise ValueError("valid judge design requires complete evidence and no reasons")
        elif not self.reason_codes:
            raise ValueError("non-valid judge design requires closed reasons")
        _require_safe_refs(_collect_refs(self))
        return self


class GradingDesignResultV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/grading-design-result/v2"] = "eval-factory/grading-design-result/v2"
    OBJECT_TYPE: ClassVar[str] = "grading-design-result"

    result_id: Identifier
    plan_ref: ObjectRef
    criteria_rubric_result_ref: ObjectRef
    route_decision_ref: ObjectRef | None
    gateway_receipt_ref: ObjectRef | None
    gateway_invocation_result_ref: ObjectRef | None
    proposal_ref: ObjectRef | None
    judge_design_spec_ref: ObjectRef | None
    validation_ref: ObjectRef
    outcome: GradingDesignOutcomeV2
    reason_codes: tuple[SafeCode, ...] = Field(
        default=(),
        max_length=256,
    )

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        _require_ref(
            self.plan_ref,
            "grading-design-plan",
            "plan_ref",
        )
        _require_ref(
            self.criteria_rubric_result_ref,
            "criteria-rubric-result",
            "criteria_rubric_result_ref",
        )
        gateway_refs = (
            self.route_decision_ref,
            self.gateway_receipt_ref,
            self.gateway_invocation_result_ref,
        )
        for ref, object_type, label in (
            (
                self.route_decision_ref,
                "model-route-decision",
                "route_decision_ref",
            ),
            (
                self.gateway_receipt_ref,
                "gateway-receipt",
                "gateway_receipt_ref",
            ),
            (
                self.gateway_invocation_result_ref,
                "gateway-invocation-result",
                "gateway_invocation_result_ref",
            ),
            (
                self.proposal_ref,
                "judge-design-proposal",
                "proposal_ref",
            ),
            (
                self.judge_design_spec_ref,
                "judge-design-spec",
                "judge_design_spec_ref",
            ),
        ):
            if ref is not None:
                _require_ref(ref, object_type, label)
        _require_ref(
            self.validation_ref,
            "judge-design-validation",
            "validation_ref",
        )
        _require_sorted_unique(self.reason_codes, "reason_codes")
        if any(ref is None for ref in gateway_refs) and any(ref is not None for ref in gateway_refs):
            raise ValueError("grading Gateway authority must be all present or all absent")
        if self.outcome is GradingDesignOutcomeV2.SUCCEEDED:
            if (
                any(ref is None for ref in gateway_refs)
                or self.proposal_ref is None
                or self.judge_design_spec_ref is None
                or self.reason_codes
            ):
                raise ValueError("successful grading design requires complete authority")
        else:
            if self.judge_design_spec_ref is not None or not self.reason_codes:
                raise ValueError("non-success grading result cannot publish a current spec")
        _require_safe_refs(_collect_refs(self))
        return self


class AgentCapabilityV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/agent-capability/v2"] = "eval-factory/agent-capability/v2"
    OBJECT_TYPE: ClassVar[str] = "agent-capability"

    capability_id: Identifier
    task_kinds: tuple[Identifier, ...] = Field(min_length=1, max_length=256)
    input_object_types: tuple[Identifier, ...] = Field(default=(), max_length=256)
    output_object_types: tuple[Identifier, ...] = Field(min_length=1, max_length=256)
    model_capabilities: tuple[Identifier, ...] = Field(default=(), max_length=256)
    tool_ids: tuple[Identifier, ...] = Field(default=(), max_length=256)
    data_purposes: tuple[Identifier, ...] = Field(min_length=1, max_length=64)
    data_classifications: tuple[Identifier, ...] = Field(min_length=1, max_length=64)

    @classmethod
    def create(  # type: ignore[override]
        cls,
        *,
        capability_id: str,
        task_kinds: tuple[str, ...],
        input_object_types: tuple[str, ...],
        output_object_types: tuple[str, ...],
        model_capabilities: tuple[str, ...],
        tool_ids: tuple[str, ...],
        data_purposes: tuple[str, ...],
        data_classifications: tuple[str, ...],
        audit: ContractAudit,
    ) -> AgentCapabilityV2:
        return cls._create(
            audit=audit,
            values={
                "capability_id": capability_id,
                "task_kinds": tuple(sorted(task_kinds)),
                "input_object_types": tuple(sorted(input_object_types)),
                "output_object_types": tuple(sorted(output_object_types)),
                "model_capabilities": tuple(sorted(model_capabilities)),
                "tool_ids": tuple(sorted(tool_ids)),
                "data_purposes": tuple(sorted(data_purposes)),
                "data_classifications": tuple(sorted(data_classifications)),
            },
        )

    @model_validator(mode="after")
    def validate_capability(self) -> Self:
        for values, label in (
            (self.task_kinds, "task_kinds"),
            (self.input_object_types, "input_object_types"),
            (self.output_object_types, "output_object_types"),
            (self.model_capabilities, "model_capabilities"),
            (self.tool_ids, "tool_ids"),
            (self.data_purposes, "data_purposes"),
            (self.data_classifications, "data_classifications"),
        ):
            _require_sorted_unique(values, label)
        return self


class AgentDefinitionV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/agent-definition/v2"] = "eval-factory/agent-definition/v2"
    OBJECT_TYPE: ClassVar[str] = "agent-definition"

    agent_definition_id: Identifier
    agent_role: Identifier
    agent_version: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    capability_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=256)
    prompt_template_ref: ObjectRef
    model_policy_ref: ObjectRef
    tool_ids: tuple[Identifier, ...] = Field(default=(), max_length=256)
    data_purpose: Identifier
    allowed_data_classifications: tuple[Identifier, ...] = Field(min_length=1, max_length=64)
    validator_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=256)
    max_attempts: int = Field(ge=1, le=100)
    max_model_requests: int = Field(ge=0, le=1_000_000)
    max_model_tokens: int = Field(ge=0, le=1_000_000_000)
    max_cost_micro_usd: int = Field(ge=0, le=1_000_000_000_000)
    workspace_isolated: Literal[True] = True
    network_allowed: bool

    @model_validator(mode="after")
    def validate_definition(self) -> Self:
        _require_sorted_unique_refs(self.capability_refs, "capability_refs")
        _require_ref(self.prompt_template_ref, "prompt-template", "prompt_template_ref")
        _require_ref(self.model_policy_ref, "model-routing-policy", "model_policy_ref")
        _require_sorted_unique(self.tool_ids, "tool_ids")
        _require_sorted_unique(
            self.allowed_data_classifications,
            "allowed_data_classifications",
        )
        _require_sorted_unique_refs(self.validator_refs, "validator_refs")
        return self


class AgentTaskV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/agent-task/v2"] = "eval-factory/agent-task/v2"
    OBJECT_TYPE: ClassVar[str] = "agent-task"

    agent_task_id: Identifier
    run_ref: ObjectRef
    compiled_plan_ref: ObjectRef
    plan_task_key: Identifier
    task_kind: Identifier
    agent_definition_ref: ObjectRef
    input_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=10_000)
    expected_output_object_types: tuple[Identifier, ...] = Field(min_length=1, max_length=256)
    required_capability_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=256)
    status: AgentTaskStatusV2
    attempt: int = Field(ge=0, le=100)
    max_attempts: int = Field(ge=1, le=100)

    @model_validator(mode="after")
    def validate_agent_task(self) -> Self:
        _require_ref(self.run_ref, "factory-run", "run_ref")
        if (
            self.compiled_plan_ref.object_type
            not in {
                "compiled-dataset-build-plan",
                "compiled-attachment-generation-plan",
                "compiled-criteria-rubric-plan",
                "compiled-grading-design-plan",
            }
            or self.compiled_plan_ref.object_version != "v2"
        ):
            raise ValueError("compiled_plan_ref must reference a supported compiled plan/v2")
        _require_ref(
            self.agent_definition_ref,
            "agent-definition",
            "agent_definition_ref",
        )
        _require_sorted_unique_refs(self.input_refs, "input_refs")
        _require_sorted_unique(
            self.expected_output_object_types,
            "expected_output_object_types",
        )
        _require_sorted_unique_refs(
            self.required_capability_refs,
            "required_capability_refs",
        )
        if self.attempt > self.max_attempts:
            raise ValueError("agent task attempt exceeds maximum attempts")
        if self.status is AgentTaskStatusV2.PENDING and self.attempt != 0:
            raise ValueError("pending agent task cannot have an attempt")
        return self


class AgentWorkspaceReceiptV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/agent-workspace-receipt/v2"] = (
        "eval-factory/agent-workspace-receipt/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "agent-workspace-receipt"

    workspace_receipt_id: Identifier
    run_ref: ObjectRef
    task_ref: ObjectRef
    agent_definition_ref: ObjectRef
    attempt: int = Field(ge=1, le=100)
    fencing_token: int = Field(ge=1)
    namespace_sha256: Sha256
    state: AgentWorkspaceStateV2
    predecessor_workspace_receipt_ref: ObjectRef | None
    inventory_sha256: Sha256 | None
    reason_code: SafeCode | None

    @model_validator(mode="after")
    def validate_receipt(self) -> Self:
        _require_ref(self.run_ref, "factory-run", "run_ref")
        _require_ref(self.task_ref, "agent-task", "task_ref")
        _require_ref(
            self.agent_definition_ref,
            "agent-definition",
            "agent_definition_ref",
        )
        if self.state is AgentWorkspaceStateV2.ACTIVE:
            if any(
                (
                    self.predecessor_workspace_receipt_ref,
                    self.inventory_sha256,
                    self.reason_code,
                )
            ):
                raise ValueError("active workspace receipt cannot be terminal")
            return self
        if self.predecessor_workspace_receipt_ref is None:
            raise ValueError("terminal workspace receipt requires its active predecessor")
        _require_ref(
            self.predecessor_workspace_receipt_ref,
            "agent-workspace-receipt",
            "predecessor_workspace_receipt_ref",
        )
        if self.state is AgentWorkspaceStateV2.COMMITTED:
            if self.inventory_sha256 is None or self.reason_code is not None:
                raise ValueError("committed workspace requires inventory and no reason")
        elif self.reason_code is None or self.inventory_sha256 is not None:
            raise ValueError("quarantined workspace requires reason and no inventory")
        return self


class AgentResultEnvelopeV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/agent-result-envelope/v2"] = "eval-factory/agent-result-envelope/v2"
    OBJECT_TYPE: ClassVar[str] = "agent-result-envelope"

    result_id: Identifier
    task_ref: ObjectRef
    agent_definition_ref: ObjectRef
    attempt: int = Field(ge=1, le=100)
    lease_id: Identifier
    fencing_token: int = Field(ge=1)
    workspace_receipt_ref: ObjectRef
    outcome: AgentTaskOutcomeV2
    output_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=10_000)
    delegated_dataset_job_result_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=10_000,
    )
    gateway_receipt_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=10_000)
    validator_result_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=10_000)
    failure_code: SafeCode | None = None
    safe_metrics: tuple[TypedAttribute, ...] = Field(default=(), max_length=256)

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        _require_ref(self.task_ref, "agent-task", "task_ref")
        _require_ref(
            self.agent_definition_ref,
            "agent-definition",
            "agent_definition_ref",
        )
        _require_ref(
            self.workspace_receipt_ref,
            "agent-workspace-receipt",
            "workspace_receipt_ref",
        )
        _require_sorted_unique_refs(self.output_refs, "output_refs")
        _require_sorted_unique_refs(
            self.delegated_dataset_job_result_refs,
            "delegated_dataset_job_result_refs",
        )
        allowed_delegations = {
            "dataset-job-result",
            "r6-canary-dataset-result",
            "stage-result",
        }
        if any(ref.object_type not in allowed_delegations for ref in self.delegated_dataset_job_result_refs):
            raise ValueError("delegated JobStore refs contain an unsupported result type")
        _require_sorted_unique_refs(self.gateway_receipt_refs, "gateway_receipt_refs")
        _require_sorted_unique_refs(self.validator_result_refs, "validator_result_refs")
        if self.outcome is AgentTaskOutcomeV2.SUCCEEDED:
            if not self.output_refs or self.failure_code is not None:
                raise ValueError("successful result requires output and no failure code")
        elif self.failure_code is None:
            raise ValueError("non-success result requires a closed failure code")
        return self


class PlannerAssessmentV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/planner-assessment/v2"] = "eval-factory/planner-assessment/v2"
    OBJECT_TYPE: ClassVar[str] = "planner-assessment"

    assessment_id: Identifier
    run_ref: ObjectRef
    compiled_plan_ref: ObjectRef
    validator_result_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=10_000)
    proposed_action: PlannerAssessmentActionV2
    affected_task_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=10_000)
    rationale_codes: tuple[SafeCode, ...] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_assessment(self) -> Self:
        _require_ref(self.run_ref, "factory-run", "run_ref")
        _require_ref(
            self.compiled_plan_ref,
            "compiled-dataset-build-plan",
            "compiled_plan_ref",
        )
        _require_sorted_unique_refs(self.validator_result_refs, "validator_result_refs")
        _require_sorted_unique_refs(self.affected_task_refs, "affected_task_refs")
        _require_sorted_unique(self.rationale_codes, "rationale_codes")
        if self.proposed_action is PlannerAssessmentActionV2.FINISH and self.affected_task_refs:
            raise ValueError("finish assessment cannot target retry work")
        return self


class FactoryRunCompletionV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/factory-run-completion/v2"] = (
        "eval-factory/factory-run-completion/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "factory-run-completion"

    completion_id: Identifier
    run_ref: ObjectRef
    compiled_plan_ref: ObjectRef
    outcome: FactoryCompletionOutcomeV2
    completed_task_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=100_000)
    incomplete_task_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=100_000)
    blocked_task_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=100_000)
    validator_result_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=100_000)
    reason_codes: tuple[SafeCode, ...] = Field(default=(), max_length=256)

    @model_validator(mode="after")
    def validate_completion(self) -> Self:
        _require_ref(self.run_ref, "factory-run", "run_ref")
        _require_ref(
            self.compiled_plan_ref,
            "compiled-dataset-build-plan",
            "compiled_plan_ref",
        )
        partitions = (
            self.completed_task_refs,
            self.incomplete_task_refs,
            self.blocked_task_refs,
        )
        for values, label in zip(
            partitions,
            ("completed_task_refs", "incomplete_task_refs", "blocked_task_refs"),
            strict=True,
        ):
            _require_sorted_unique_refs(values, label)
        if _sets_overlap(*(set(values) for values in partitions)):
            raise ValueError("completion task partitions must be disjoint")
        _require_sorted_unique_refs(self.validator_result_refs, "validator_result_refs")
        _require_sorted_unique(self.reason_codes, "reason_codes")
        if self.outcome is FactoryCompletionOutcomeV2.COMPLETE:
            if self.incomplete_task_refs or self.blocked_task_refs or self.reason_codes:
                raise ValueError("complete outcome cannot retain incomplete work")
        elif not self.reason_codes:
            raise ValueError("non-complete outcome requires reason codes")
        return self


class PlanReviewRequestV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/plan-review-request/v2"] = "eval-factory/plan-review-request/v2"
    OBJECT_TYPE: ClassVar[str] = "plan-review-request"

    review_request_id: Identifier
    run_ref: ObjectRef
    plan_ref: ObjectRef
    plan_kind: PlanKindV2
    plan_version: int = Field(ge=1, le=1_000_000)
    presentation_ref: ObjectRef
    requested_by: Identifier
    state: Literal[PlanReviewStateV2.PENDING_REVIEW] = PlanReviewStateV2.PENDING_REVIEW

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        _require_ref(self.run_ref, "factory-run", "run_ref")
        _require_ref(self.presentation_ref, "plan-review-presentation", "presentation_ref")
        if self.plan_ref.object_type not in {
            "dataset-build-plan",
            "task-rewrite-plan",
            "trace-cleaning-plan",
            "attachment-generation-plan",
            "criteria-rubric-plan",
            "grading-design-plan",
            "dataset-delivery-plan",
        }:
            raise ValueError("plan review request has an unsupported plan ref")
        return self


class PlanReviewPresentationV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/plan-review-presentation/v2"] = (
        "eval-factory/plan-review-presentation/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "plan-review-presentation"

    presentation_id: Identifier
    plan_ref: ObjectRef
    plan_kind: PlanKindV2
    title: SafeSummary
    summary_lines: tuple[SafeSummary, ...] = Field(min_length=1, max_length=256)
    editable_paths: tuple[Annotated[str, StringConstraints(min_length=1, max_length=256)], ...] = Field(
        default=(),
        max_length=256,
    )
    warning_codes: tuple[SafeCode, ...] = Field(default=(), max_length=256)

    @model_validator(mode="after")
    def validate_presentation(self) -> Self:
        _require_unique(self.summary_lines, "summary_lines")
        _require_sorted_unique(self.editable_paths, "editable_paths")
        _require_sorted_unique(self.warning_codes, "warning_codes")
        return self


class PlanRevisionV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/plan-revision/v2"] = "eval-factory/plan-revision/v2"
    OBJECT_TYPE: ClassVar[str] = "plan-revision"

    revision_id: Identifier
    review_request_ref: ObjectRef
    base_plan_ref: ObjectRef
    successor_plan_ref: ObjectRef
    base_plan_version: int = Field(ge=1, le=1_000_000)
    successor_plan_version: int = Field(ge=2, le=1_000_000)
    changed_paths: tuple[Annotated[str, StringConstraints(min_length=1, max_length=256)], ...] = Field(
        min_length=1,
        max_length=256,
    )
    invalidated_object_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=100_000)
    revised_by: Identifier

    @model_validator(mode="after")
    def validate_revision(self) -> Self:
        _require_ref(
            self.review_request_ref,
            "plan-review-request",
            "review_request_ref",
        )
        if self.base_plan_ref.object_type != self.successor_plan_ref.object_type:
            raise ValueError("plan revision cannot change plan type")
        if self.successor_plan_version != self.base_plan_version + 1:
            raise ValueError("successor plan version must be contiguous")
        _require_sorted_unique(self.changed_paths, "changed_paths")
        _require_sorted_unique_refs(
            self.invalidated_object_refs,
            "invalidated_object_refs",
        )
        return self


class PlanDecisionV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/plan-decision/v2"] = "eval-factory/plan-decision/v2"
    OBJECT_TYPE: ClassVar[str] = "plan-decision"

    decision_id: Identifier
    review_request_ref: ObjectRef
    expected_plan_version: int = Field(ge=1, le=1_000_000)
    decision: PlanDecisionKindV2
    revision_ref: ObjectRef | None
    decided_by: Identifier
    reason_code: SafeCode
    idempotency_key: Identifier
    decided_at: datetime

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        _require_ref(
            self.review_request_ref,
            "plan-review-request",
            "review_request_ref",
        )
        if self.decision is PlanDecisionKindV2.EDIT:
            if self.revision_ref is None:
                raise ValueError("edit decision requires an immutable revision ref")
            _require_ref(self.revision_ref, "plan-revision", "revision_ref")
        elif self.revision_ref is not None:
            raise ValueError("only an edit decision can bind a revision ref")
        if self.decided_at.utcoffset() is None:
            raise ValueError("plan decision timestamp must be timezone-aware")
        return self


class PlanReviewResultV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/plan-review-result/v2"] = "eval-factory/plan-review-result/v2"
    OBJECT_TYPE: ClassVar[str] = "plan-review-result"

    result_id: Identifier
    review_request_ref: ObjectRef
    plan_ref: ObjectRef
    plan_version: int = Field(ge=1, le=1_000_000)
    state: PlanReviewStateV2
    decision_ref: ObjectRef | None
    successor_plan_ref: ObjectRef | None
    resume_token_ref: ObjectRef | None

    @model_validator(mode="after")
    def validate_review_result(self) -> Self:
        _require_ref(
            self.review_request_ref,
            "plan-review-request",
            "review_request_ref",
        )
        if self.state is PlanReviewStateV2.PENDING_REVIEW:
            if any((self.decision_ref, self.successor_plan_ref, self.resume_token_ref)):
                raise ValueError("pending review cannot have terminal refs")
        elif self.decision_ref is None:
            raise ValueError("terminal review result requires a decision ref")
        if self.successor_plan_ref is not None and self.state is not PlanReviewStateV2.REVISION_REQUESTED:
            raise ValueError("only a revision result can bind a successor plan")
        return self


class IntentClaimV2(ContractModelV2):
    schema_version: Literal["eval-factory/intent-claim/v2"] = "eval-factory/intent-claim/v2"
    claim_id: Identifier
    summary: SafeSummary
    evidence_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=256)
    confidence_basis_points: int = Field(ge=0, le=10_000)
    uncertain: bool

    @model_validator(mode="after")
    def validate_claim(self) -> Self:
        _require_sorted_unique_refs(self.evidence_refs, "evidence_refs")
        if not self.uncertain and self.confidence_basis_points == 0:
            raise ValueError("certain intent claim requires positive confidence")
        return self


class ExtractedUserPromptV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/extracted-user-prompt/v2"] = "eval-factory/extracted-user-prompt/v2"
    OBJECT_TYPE: ClassVar[str] = "extracted-user-prompt"

    extracted_prompt_id: Identifier
    trace_ref: ObjectRef
    interaction_segment_ref: ObjectRef
    content_ref: ObjectRef
    source_spans: tuple[SourceSpanRef, ...] = Field(min_length=1, max_length=10_000)
    context_segment_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=1_000)

    @model_validator(mode="after")
    def validate_prompt(self) -> Self:
        _require_ref(self.trace_ref, "trace-ir", "trace_ref")
        _require_ref(
            self.interaction_segment_ref,
            "interaction-segment",
            "interaction_segment_ref",
        )
        _require_ref(self.content_ref, "user-prompt-content", "content_ref")
        _require_sorted_unique_refs(self.context_segment_refs, "context_segment_refs")
        _require_unique(
            tuple(span.span_id for span in self.source_spans),
            "source_spans",
        )
        if any(
            span.source_trace_id != self.source_spans[0].source_trace_id
            or span.raw_sha256 != self.source_spans[0].raw_sha256
            for span in self.source_spans
        ):
            raise ValueError("extracted prompt source spans must share one source")
        return self


class InferredUserIntentV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/inferred-user-intent/v2"] = "eval-factory/inferred-user-intent/v2"
    OBJECT_TYPE: ClassVar[str] = "inferred-user-intent"

    inferred_intent_id: Identifier
    extracted_prompt_ref: ObjectRef
    claims: tuple[IntentClaimV2, ...] = Field(default=(), max_length=256)
    unresolved_requirements: tuple[SafeSummary, ...] = Field(default=(), max_length=256)
    abstained: bool

    @model_validator(mode="after")
    def validate_intent(self) -> Self:
        _require_ref(
            self.extracted_prompt_ref,
            "extracted-user-prompt",
            "extracted_prompt_ref",
        )
        _require_sorted_unique(
            tuple(claim.claim_id for claim in self.claims),
            "intent claim IDs",
        )
        _require_unique(self.unresolved_requirements, "unresolved_requirements")
        if self.abstained:
            if self.claims or not self.unresolved_requirements:
                raise ValueError("abstained intent requires unresolved facts and no claims")
        elif not self.claims:
            raise ValueError("non-abstained intent requires evidence-bound claims")
        return self


class TaskRewritePlanV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/task-rewrite-plan/v2"] = "eval-factory/task-rewrite-plan/v2"
    OBJECT_TYPE: ClassVar[str] = "task-rewrite-plan"

    rewrite_plan_id: Identifier
    extracted_prompt_ref: ObjectRef
    inferred_intent_ref: ObjectRef
    rewrite_policy_ref: ObjectRef
    target_capabilities: tuple[Identifier, ...] = Field(min_length=1, max_length=256)
    fidelity_constraints: tuple[SafeSummary, ...] = Field(min_length=1, max_length=256)
    forbidden_transformations: tuple[SafeSummary, ...] = Field(min_length=1, max_length=256)
    acceptance_check_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_rewrite_plan(self) -> Self:
        _require_ref(
            self.extracted_prompt_ref,
            "extracted-user-prompt",
            "extracted_prompt_ref",
        )
        _require_ref(
            self.inferred_intent_ref,
            "inferred-user-intent",
            "inferred_intent_ref",
        )
        _require_ref(self.rewrite_policy_ref, "task-rewrite-policy", "rewrite_policy_ref")
        _require_sorted_unique(self.target_capabilities, "target_capabilities")
        _require_unique(self.fidelity_constraints, "fidelity_constraints")
        _require_unique(self.forbidden_transformations, "forbidden_transformations")
        _require_sorted_unique_refs(self.acceptance_check_refs, "acceptance_check_refs")
        return self


class TraceCandidateDecisionV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/trace-candidate-decision/v2"] = (
        "eval-factory/trace-candidate-decision/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "trace-candidate-decision"

    decision_id: Identifier
    source_trace_id: Identifier
    source_ref: ObjectRef
    disposition: TraceCandidateDispositionV2
    reason_codes: tuple[SafeCode, ...] = Field(default=(), max_length=64)
    cleaned_trace_ref: ObjectRef | None

    @model_validator(mode="after")
    def validate_candidate_decision(self) -> Self:
        _require_ref(self.source_ref, "trace-source", "source_ref")
        _require_sorted_unique(self.reason_codes, "reason_codes")
        if self.disposition is TraceCandidateDispositionV2.CANDIDATE:
            if self.cleaned_trace_ref is None or self.reason_codes:
                raise ValueError("candidate trace requires cleaned trace ref and no reasons")
            _require_ref(self.cleaned_trace_ref, "trace-ir", "cleaned_trace_ref")
        elif self.cleaned_trace_ref is not None or not self.reason_codes:
            raise ValueError("non-candidate trace requires reasons and no cleaned trace ref")
        return self


class TaskRewriteCandidateV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/task-rewrite-candidate/v2"] = (
        "eval-factory/task-rewrite-candidate/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "task-rewrite-candidate"

    candidate_id: Identifier
    extracted_prompt_ref: ObjectRef
    inferred_intent_ref: ObjectRef
    rewrite_plan_ref: ObjectRef
    rewritten_prompt_ref: ObjectRef
    evidence_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=256)
    status: Literal["REVIEW_READY"] = "REVIEW_READY"

    @model_validator(mode="after")
    def validate_rewrite_candidate(self) -> Self:
        for ref, object_type, label in (
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
            (self.rewrite_plan_ref, "task-rewrite-plan", "rewrite_plan_ref"),
            (
                self.rewritten_prompt_ref,
                "rewritten-prompt-content",
                "rewritten_prompt_ref",
            ),
        ):
            _require_ref(ref, object_type, label)
        _require_sorted_unique_refs(self.evidence_refs, "evidence_refs")
        return self


class AttachmentGroupResultV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/attachment-group-result/v2"] = (
        "eval-factory/attachment-group-result/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "attachment-group-result"

    result_id: Identifier
    plan_ref: ObjectRef
    work_key: Identifier
    artifact_group_ref: ObjectRef
    execution_batch_ref: ObjectRef
    reconstruction_result_ref: ObjectRef
    execution_receipt_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=100_000,
    )
    artifact_ids: tuple[Identifier, ...] = Field(
        min_length=1,
        max_length=100_000,
    )
    outcome: AgentTaskOutcomeV2
    reason_codes: tuple[SafeCode, ...] = Field(
        default=(),
        max_length=256,
    )

    @classmethod
    def create(  # type: ignore[override]
        cls,
        *,
        result_id: str,
        plan_ref: ObjectRef,
        work_key: str,
        artifact_group_ref: ObjectRef,
        execution_batch_ref: ObjectRef,
        reconstruction_result_ref: ObjectRef,
        execution_receipt_refs: tuple[ObjectRef, ...],
        artifact_ids: tuple[str, ...],
        outcome: AgentTaskOutcomeV2,
        reason_codes: tuple[str, ...],
        audit: ContractAudit,
    ) -> AttachmentGroupResultV2:
        return cls._create(
            audit=audit,
            values={
                "result_id": result_id,
                "plan_ref": plan_ref,
                "work_key": work_key,
                "artifact_group_ref": artifact_group_ref,
                "execution_batch_ref": execution_batch_ref,
                "reconstruction_result_ref": (reconstruction_result_ref),
                "execution_receipt_refs": tuple(
                    sorted(
                        execution_receipt_refs,
                        key=_ref_key,
                    )
                ),
                "artifact_ids": tuple(sorted(artifact_ids)),
                "outcome": outcome,
                "reason_codes": tuple(sorted(reason_codes)),
            },
        )

    @model_validator(mode="after")
    def validate_group_result(self) -> Self:
        _require_ref(
            self.plan_ref,
            "attachment-generation-plan",
            "plan_ref",
        )
        _require_ref(
            self.artifact_group_ref,
            "artifact-execution-group",
            "artifact_group_ref",
        )
        _require_ref(
            self.execution_batch_ref,
            "artifact-execution-batch",
            "execution_batch_ref",
        )
        _require_ref(
            self.reconstruction_result_ref,
            "attachment-reconstruction-result",
            "reconstruction_result_ref",
        )
        _require_sorted_unique_refs(
            self.execution_receipt_refs,
            "execution_receipt_refs",
        )
        if any(
            reference.object_type != "artifact-execution-receipt" for reference in self.execution_receipt_refs
        ):
            raise ValueError("attachment group result requires execution receipts")
        _require_sorted_unique(
            self.artifact_ids,
            "artifact_ids",
        )
        if len(self.execution_receipt_refs) != len(self.artifact_ids):
            raise ValueError("attachment group result receipt inventory is incomplete")
        _require_sorted_unique(
            self.reason_codes,
            "reason_codes",
        )
        if self.outcome is AgentTaskOutcomeV2.SUCCEEDED:
            if self.reason_codes:
                raise ValueError("successful attachment group cannot retain reasons")
        elif not self.reason_codes:
            raise ValueError("non-successful attachment group requires reasons")
        return self


class AttachmentSubgraphResultV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/attachment-subgraph-result/v2"] = (
        "eval-factory/attachment-subgraph-result/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "attachment-subgraph-result"

    result_id: Identifier
    plan_ref: ObjectRef
    work_result_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=100_000)
    succeeded_work_result_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=100_000,
    )
    retryable_work_result_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=100_000,
    )
    blocked_work_result_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=100_000,
    )
    cancelled_work_result_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=100_000,
    )
    outcome: AttachmentSubgraphOutcomeV2
    reason_codes: tuple[SafeCode, ...] = Field(default=(), max_length=256)

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        _require_ref(
            self.plan_ref,
            "attachment-generation-plan",
            "plan_ref",
        )
        partitions = (
            self.succeeded_work_result_refs,
            self.retryable_work_result_refs,
            self.blocked_work_result_refs,
            self.cancelled_work_result_refs,
        )
        _require_sorted_unique_refs(self.work_result_refs, "work_result_refs")
        if any(ref.object_type != "agent-result-envelope" for ref in self.work_result_refs):
            raise ValueError("attachment work results must be Agent result envelopes")
        for values, label in zip(
            partitions,
            (
                "succeeded_work_result_refs",
                "retryable_work_result_refs",
                "blocked_work_result_refs",
                "cancelled_work_result_refs",
            ),
            strict=True,
        ):
            _require_sorted_unique_refs(values, label)
        if _sets_overlap(*(set(values) for values in partitions)) or set(
            self.work_result_refs
        ) != set().union(*(set(values) for values in partitions)):
            raise ValueError("attachment work results require an exact partition")
        _require_sorted_unique(self.reason_codes, "reason_codes")
        if self.outcome is AttachmentSubgraphOutcomeV2.SUCCEEDED:
            if len(self.succeeded_work_result_refs) != len(self.work_result_refs) or self.reason_codes:
                raise ValueError("successful attachment subgraph must fully succeed")
        elif not self.reason_codes:
            raise ValueError("non-successful attachment subgraph requires reasons")
        if self.outcome is AttachmentSubgraphOutcomeV2.BLOCKED and self.succeeded_work_result_refs:
            raise ValueError("blocked attachment subgraph cannot contain success")
        if self.outcome is AttachmentSubgraphOutcomeV2.CANCELLED and len(
            self.cancelled_work_result_refs
        ) != len(self.work_result_refs):
            raise ValueError("cancelled attachment subgraph must fully cancel")
        return self


class AttachmentQualityAssessmentV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/attachment-quality-assessment/v2"] = (
        "eval-factory/attachment-quality-assessment/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "attachment-quality-assessment"

    assessment_id: Identifier
    attachment_subgraph_result_ref: ObjectRef
    item_quality_result_ref: ObjectRef
    validator_result_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=100_000,
    )
    outcome: AttachmentQualityOutcomeV2
    nonwaivable: bool
    reason_codes: tuple[SafeCode, ...] = Field(default=(), max_length=256)

    @model_validator(mode="after")
    def validate_assessment(self) -> Self:
        _require_ref(
            self.attachment_subgraph_result_ref,
            "attachment-subgraph-result",
            "attachment_subgraph_result_ref",
        )
        _require_ref(
            self.item_quality_result_ref,
            "item-quality-compilation-result",
            "item_quality_result_ref",
        )
        _require_sorted_unique_refs(
            self.validator_result_refs,
            "validator_result_refs",
        )
        _require_sorted_unique(self.reason_codes, "reason_codes")
        if self.outcome is AttachmentQualityOutcomeV2.PASSED:
            if self.reason_codes or self.nonwaivable:
                raise ValueError("passing attachment quality cannot retain blockers")
        elif not self.reason_codes:
            raise ValueError("non-passing attachment quality requires reasons")
        if self.nonwaivable and self.outcome is not AttachmentQualityOutcomeV2.REJECTED:
            raise ValueError("non-waivable attachment quality must be rejected")
        return self


class SolvabilityAssessmentV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/solvability-assessment/v2"] = (
        "eval-factory/solvability-assessment/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "solvability-assessment"

    assessment_id: Identifier
    attachment_subgraph_result_ref: ObjectRef
    quality_assessment_ref: ObjectRef
    route_decision_ref: ObjectRef
    gateway_receipt_ref: ObjectRef
    evidence_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=100_000)
    outcome: SolvabilityOutcomeV2
    reason_codes: tuple[SafeCode, ...] = Field(default=(), max_length=256)

    @model_validator(mode="after")
    def validate_assessment(self) -> Self:
        _require_ref(
            self.attachment_subgraph_result_ref,
            "attachment-subgraph-result",
            "attachment_subgraph_result_ref",
        )
        _require_ref(
            self.quality_assessment_ref,
            "attachment-quality-assessment",
            "quality_assessment_ref",
        )
        _require_ref(
            self.route_decision_ref,
            "model-route-decision",
            "route_decision_ref",
        )
        _require_ref(
            self.gateway_receipt_ref,
            "gateway-receipt",
            "gateway_receipt_ref",
        )
        _require_sorted_unique_refs(self.evidence_refs, "evidence_refs")
        _require_sorted_unique(self.reason_codes, "reason_codes")
        if self.outcome is SolvabilityOutcomeV2.SOLVABLE:
            if self.reason_codes:
                raise ValueError("solvable assessment cannot retain reasons")
        elif not self.reason_codes:
            raise ValueError("non-solvable assessment requires reasons")
        return self


class CoreVerticalResultV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/core-vertical-result/v2"] = "eval-factory/core-vertical-result/v2"
    OBJECT_TYPE: ClassVar[str] = "core-vertical-result"

    result_id: Identifier
    manifest_ref: ObjectRef
    requirement_spec_ref: ObjectRef
    candidate_decision_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=1_000_000)
    non_candidate_decision_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=1_000_000)
    blocked_decision_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=1_000_000)
    extracted_prompt_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=1_000_000)
    inferred_intent_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=1_000_000)
    rewrite_candidate_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=1_000_000)
    route_decision_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=1_000_000)
    source_count: int = Field(ge=1, le=1_000_000)

    @model_validator(mode="after")
    def validate_core_vertical(self) -> Self:
        _require_ref(self.manifest_ref, "trace-manifest", "manifest_ref")
        _require_ref(
            self.requirement_spec_ref,
            "evaluation-requirement-spec",
            "requirement_spec_ref",
        )
        partitions = (
            self.candidate_decision_refs,
            self.non_candidate_decision_refs,
            self.blocked_decision_refs,
        )
        for values, label in zip(
            partitions,
            (
                "candidate_decision_refs",
                "non_candidate_decision_refs",
                "blocked_decision_refs",
            ),
            strict=True,
        ):
            _require_sorted_unique_refs(values, label)
            if any(ref.object_type != "trace-candidate-decision" for ref in values):
                raise ValueError("core vertical partition contains an invalid decision ref")
        if _sets_overlap(*(set(values) for values in partitions)):
            raise ValueError("core vertical trace partitions must be disjoint")
        if self.source_count != sum(len(values) for values in partitions):
            raise ValueError("core vertical partitions do not cover source count")
        for values, label, object_type in (
            (
                self.extracted_prompt_refs,
                "extracted_prompt_refs",
                "extracted-user-prompt",
            ),
            (
                self.inferred_intent_refs,
                "inferred_intent_refs",
                "inferred-user-intent",
            ),
            (
                self.rewrite_candidate_refs,
                "rewrite_candidate_refs",
                "task-rewrite-candidate",
            ),
            (
                self.route_decision_refs,
                "route_decision_refs",
                "model-route-decision",
            ),
        ):
            _require_sorted_unique_refs(values, label)
            if any(ref.object_type != object_type for ref in values):
                raise ValueError(f"{label} contains an invalid object type")
        candidate_count = len(self.candidate_decision_refs)
        if not (
            len(self.extracted_prompt_refs)
            == len(self.inferred_intent_refs)
            == len(self.rewrite_candidate_refs)
            == candidate_count
        ):
            raise ValueError("core vertical candidate artifacts do not have exact coverage")
        return self


class DatasetDeliveryManifestV2(_FactoryObjectV2):
    schema_version: Literal["eval-factory/dataset-delivery-manifest/v2"] = (
        "eval-factory/dataset-delivery-manifest/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "dataset-delivery-manifest"

    delivery_manifest_id: Identifier
    run_ref: ObjectRef
    completion_ref: ObjectRef
    requirement_summary_ref: ObjectRef
    trace_candidate_table_ref: ObjectRef
    extracted_prompt_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=100_000)
    inferred_intent_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=100_000)
    rewrite_candidate_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=100_000)
    unresolved_report_ref: ObjectRef
    audit_ref: ObjectRef
    item_count: int = Field(ge=0, le=1_000_000)

    @model_validator(mode="after")
    def validate_delivery(self) -> Self:
        _require_ref(self.run_ref, "factory-run", "run_ref")
        _require_ref(self.completion_ref, "factory-run-completion", "completion_ref")
        for values, label, object_type in (
            (self.extracted_prompt_refs, "extracted_prompt_refs", "extracted-user-prompt"),
            (self.inferred_intent_refs, "inferred_intent_refs", "inferred-user-intent"),
            (self.rewrite_candidate_refs, "rewrite_candidate_refs", "task-rewrite-candidate"),
        ):
            _require_sorted_unique_refs(values, label)
            if any(ref.object_type != object_type for ref in values):
                raise ValueError(f"{label} contains an invalid object type")
        if len(self.extracted_prompt_refs) != len(self.inferred_intent_refs):
            raise ValueError("delivery prompt and intent counts must match")
        if self.item_count != len(self.rewrite_candidate_refs):
            raise ValueError("delivery item count differs from rewrite candidates")
        return self


def _carried_sha256(value: _FactoryObjectV2) -> str:
    payload = value.model_dump(
        mode="python",
        exclude={"schema_version", "object_id", "object_sha256", "audit"},
    )
    encoded = json.dumps(
        canonical_value_v2(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _collect_refs(value: BaseModel) -> tuple[ObjectRef, ...]:
    refs: list[ObjectRef] = []

    def visit(item: object) -> None:
        if isinstance(item, ObjectRef):
            refs.append(item)
        elif isinstance(item, BaseModel):
            for field_name in type(item).model_fields:
                if field_name in {"audit", "object_id", "object_sha256"}:
                    continue
                visit(getattr(item, field_name))
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, tuple | list | set | frozenset):
            for child in item:
                visit(child)

    visit(value)
    unique: list[ObjectRef] = []
    seen: set[tuple[str, str, str, str]] = set()
    for ref in refs:
        key = _ref_key(ref)
        if key not in seen:
            seen.add(key)
            unique.append(ref)
    return tuple(unique)


def _audit_with_refs(audit: ContractAudit, refs: tuple[ObjectRef, ...]) -> ContractAudit:
    return ContractAudit(
        created_at=audit.created_at,
        created_by=audit.created_by,
        governing_versions=audit.governing_versions,
        input_refs=refs,
    )


def _require_ref(value: ObjectRef, object_type: str, label: str) -> None:
    if value.object_type != object_type or value.object_version != "v2":
        raise ValueError(f"{label} must reference {object_type}/v2")


def _require_safe_refs(refs: tuple[ObjectRef, ...]) -> None:
    for ref in refs:
        rendered = f"{ref.object_type} {ref.object_id}".casefold()
        if any(marker in rendered for marker in _DENIED_REF_MARKERS):
            raise ValueError("control records cannot reference sensitive object classes")


def _require_unique(values: tuple[object, ...], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


def _require_sorted_unique(values: tuple[str, ...], label: str) -> None:
    if tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise ValueError(f"{label} must be sorted and unique")


def _require_sorted_unique_refs(values: tuple[ObjectRef, ...], label: str) -> None:
    keys = tuple(_ref_key(value) for value in values)
    if tuple(sorted(keys)) != keys or len(set(keys)) != len(keys):
        raise ValueError(f"{label} must be sorted and unique")


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _require_acyclic(tasks: tuple[DatasetBuildPlanTaskV2, ...]) -> None:
    dependencies = {task.task_key: set(task.dependency_task_keys) for task in tasks}
    ready = sorted(task_key for task_key, values in dependencies.items() if not values)
    visited: list[str] = []
    while ready:
        current = ready.pop(0)
        visited.append(current)
        for task_key in sorted(dependencies):
            if current in dependencies[task_key]:
                dependencies[task_key].remove(current)
                if not dependencies[task_key] and task_key not in visited and task_key not in ready:
                    ready.append(task_key)
                    ready.sort()
    if len(visited) != len(tasks):
        raise ValueError("dataset build plan task graph contains a cycle")


def _require_attachment_acyclic(
    works: tuple[AttachmentMockWorkV2, ...],
) -> None:
    dependencies = {work.work_key: set(work.dependency_work_keys) for work in works}
    ready = sorted(work_key for work_key, values in dependencies.items() if not values)
    visited: list[str] = []
    while ready:
        current = ready.pop(0)
        visited.append(current)
        for work_key in sorted(dependencies):
            if current in dependencies[work_key]:
                dependencies[work_key].remove(current)
                if not dependencies[work_key] and work_key not in visited and work_key not in ready:
                    ready.append(work_key)
                    ready.sort()
    if len(visited) != len(works):
        raise ValueError("attachment work graph contains a cycle")


def _sets_overlap(*values: set[ObjectRef]) -> bool:
    seen: set[ObjectRef] = set()
    for value in values:
        if seen & value:
            return True
        seen.update(value)
    return False


__all__ = [
    "AgentCapabilityV2",
    "AgentDefinitionV2",
    "AgentResultEnvelopeV2",
    "AgentTaskOutcomeV2",
    "AgentTaskStatusV2",
    "AgentTaskV2",
    "AgentWorkspaceReceiptV2",
    "AgentWorkspaceStateV2",
    "AttachmentGenerationPlanV2",
    "AttachmentMockWorkV2",
    "AttachmentQualityAssessmentV2",
    "AttachmentQualityOutcomeV2",
    "AttachmentSubgraphOutcomeV2",
    "AttachmentSubgraphResultV2",
    "AttachmentWorkAssignmentV2",
    "CompiledAttachmentGenerationPlanV2",
    "CompiledCriteriaRubricPlanV2",
    "CompiledDatasetBuildPlanV2",
    "CompiledGradingDesignPlanV2",
    "CoreVerticalResultV2",
    "CriteriaRubricGoalV2",
    "CriteriaRubricOutcomeV2",
    "CriteriaRubricPlanV2",
    "CriteriaRubricResultV2",
    "DatasetBuildPlanTaskV2",
    "DatasetBuildPlanV2",
    "DatasetDeliveryManifestV2",
    "EvaluationRequirementSpecV2",
    "ExtractedUserPromptV2",
    "FactoryCompletionOutcomeV2",
    "FactoryRunCompletionV2",
    "FactoryRunPolicyV2",
    "FactoryRunStatusV2",
    "FactoryRunV2",
    "GradingDesignOutcomeV2",
    "GradingDesignPlanV2",
    "GradingDesignResultV2",
    "InferredUserIntentV2",
    "IntentClaimV2",
    "JudgeAggregationModeV2",
    "JudgeDesignSpecV2",
    "JudgeDesignValidationOutcomeV2",
    "JudgeDesignValidationV2",
    "JudgeTaskMappingV2",
    "PlanDecisionKindV2",
    "PlanDecisionV2",
    "PlanKindV2",
    "PlanReviewPresentationV2",
    "PlanReviewRequestV2",
    "PlanReviewResultV2",
    "PlanReviewStateV2",
    "PlanRevisionV2",
    "PlannerAssessmentActionV2",
    "PlannerAssessmentV2",
    "SolvabilityAssessmentV2",
    "SolvabilityOutcomeV2",
    "TaskRewriteCandidateV2",
    "TaskRewritePlanV2",
    "TraceCandidateDecisionV2",
    "TraceCandidateDispositionV2",
]
