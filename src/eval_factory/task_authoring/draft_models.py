from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    ContractModel,
    Identifier,
    ObjectRef,
    Sha256,
    TypedAttribute,
)
from eval_factory.contracts.task import (
    AttachmentCriticality,
    EvidencePriority,
    RequirementConflict,
)
from eval_factory.contracts.task_v2 import TaskDraftV2
from eval_factory.provenance.views import EvidenceProjectionMode
from eval_factory.task_authoring.models import is_safe_task_authoring_ref

TASK_DRAFT_AUTHORING_POLICY_VERSION: Literal["task-draft-authoring/r4-03-v1"] = (
    "task-draft-authoring/r4-03-v1"
)
TASK_DRAFT_EVIDENCE_PURPOSE: Literal["task-draft-authoring"] = "task-draft-authoring"


class TaskDraftAuthoringPolicyError(RuntimeError):
    pass


class TaskDraftAuthoringOutcome(StrEnum):
    DRAFTED = "DRAFTED"
    ABSTAIN = "ABSTAIN"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"


class TaskDraftAuthoringReason(StrEnum):
    AMBIGUOUS_TASK_INTENT = "AMBIGUOUS_TASK_INTENT"
    CONFLICTING_REQUIREMENTS = "CONFLICTING_REQUIREMENTS"
    MISSING_SAFE_AUTHORING_EVIDENCE = "MISSING_SAFE_AUTHORING_EVIDENCE"
    INCOMPLETE_REQUIREMENT_LINEAGE = "INCOMPLETE_REQUIREMENT_LINEAGE"
    MISSING_CAPABILITY = "MISSING_CAPABILITY"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"


class TaskDraftEvidenceView(ContractModel):
    schema_version: Literal["eval-factory/task-draft-evidence-view/r4-03"] = (
        "eval-factory/task-draft-evidence-view/r4-03"
    )
    evidence_ref_id: Identifier
    subject_ref: ObjectRef
    projection_mode: EvidenceProjectionMode
    content: str | None = Field(default=None, max_length=1_000_000)
    structure_fields: tuple[TypedAttribute, ...] = ()
    external_uri: str | None = Field(default=None, min_length=3, max_length=2048)
    capability: Identifier
    capability_complete: bool
    untrusted_data_marker: Literal[True] = True

    @field_validator("projection_mode", mode="before")
    @classmethod
    def parse_projection_mode(cls, value: object) -> EvidenceProjectionMode:
        if isinstance(value, EvidenceProjectionMode):
            return value
        if isinstance(value, str):
            return EvidenceProjectionMode(value)
        raise TypeError("projection_mode must be an EvidenceProjectionMode")

    @model_validator(mode="after")
    def validate_projection(self) -> TaskDraftEvidenceView:
        if not is_safe_task_authoring_ref(self.subject_ref):
            raise ValueError("unsafe task-author evidence reference")
        if self.projection_mode is EvidenceProjectionMode.CONTENT:
            if self.content is None:
                raise ValueError("CONTENT evidence view requires content")
        elif self.projection_mode is EvidenceProjectionMode.STRUCTURE:
            if not self.structure_fields:
                raise ValueError("STRUCTURE evidence view requires structure fields")
        elif self.projection_mode is EvidenceProjectionMode.EXTERNAL_LEAD:
            if self.external_uri is None:
                raise ValueError("EXTERNAL_LEAD evidence view requires external URI")
        elif self.projection_mode is EvidenceProjectionMode.AUDIT:
            raise ValueError("audit evidence cannot enter task authoring")
        return self


class TaskDraftAuthoringRequest(ContractModel):
    schema_version: Literal["eval-factory/task-draft-authoring-request/r4-03"] = (
        "eval-factory/task-draft-authoring-request/r4-03"
    )
    semantic_authoring_request_id: Identifier
    selection_context_ref: ObjectRef
    trace_envelope_ref: ObjectRef
    task_episode_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    selection_evidence_bundle_ref: ObjectRef
    authoring_evidence_bundle_ref: ObjectRef
    authoring_projection_policy_ref: ObjectRef
    prompt_boundary_enforcement_ref: ObjectRef
    evidence_views: tuple[TaskDraftEvidenceView, ...] = Field(min_length=1)
    allowed_capability_ids: tuple[Identifier, ...] = Field(min_length=1)
    allowed_tool_ids: tuple[Identifier, ...] = ()
    required_forbidden_outputs: tuple[str, ...] = Field(min_length=1)
    abstain_conditions: tuple[str, ...] = Field(min_length=1)
    model_profile: Identifier
    prompt_version: str = Field(min_length=1, max_length=128)
    returned_characters: int = Field(ge=0)
    max_characters: int = Field(ge=0)
    policy_version: Literal["task-draft-authoring/r4-03-v1"] = TASK_DRAFT_AUTHORING_POLICY_VERSION
    request_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_request(self) -> TaskDraftAuthoringRequest:
        _require_ref_type(
            self.selection_context_ref,
            "selection-context",
            "selection_context_ref",
        )
        _require_ref_type(
            self.trace_envelope_ref,
            "trace-envelope",
            "trace_envelope_ref",
        )
        for ref in self.task_episode_refs:
            _require_ref_type(ref, "task-episode", "task_episode_refs")
        _require_ref_type(
            self.selection_evidence_bundle_ref,
            "evidence-bundle",
            "selection_evidence_bundle_ref",
        )
        _require_ref_type(
            self.authoring_evidence_bundle_ref,
            "evidence-bundle",
            "authoring_evidence_bundle_ref",
        )
        _require_ref_type(
            self.authoring_projection_policy_ref,
            "projection-policy",
            "authoring_projection_policy_ref",
        )
        _require_ref_type(
            self.prompt_boundary_enforcement_ref,
            "prompt-boundary-enforcement",
            "prompt_boundary_enforcement_ref",
        )
        if self.returned_characters > self.max_characters:
            raise ValueError("task-draft authoring evidence exceeds budget")
        require_unique("task episode", (ref.object_id for ref in self.task_episode_refs))
        require_unique(
            "evidence view",
            (item.evidence_ref_id for item in self.evidence_views),
        )
        require_unique("allowed capability", self.allowed_capability_ids)
        require_unique("allowed tool", self.allowed_tool_ids)
        require_unique(
            "required forbidden output",
            self.required_forbidden_outputs,
        )
        if any(not item for item in self.required_forbidden_outputs):
            raise ValueError("required forbidden outputs must be non-empty")
        if any(not item for item in self.abstain_conditions):
            raise ValueError("abstain conditions must be non-empty")
        refs = (
            self.selection_context_ref,
            self.trace_envelope_ref,
            *self.task_episode_refs,
            self.selection_evidence_bundle_ref,
            self.authoring_evidence_bundle_ref,
            self.authoring_projection_policy_ref,
            self.prompt_boundary_enforcement_ref,
            *(item.subject_ref for item in self.evidence_views),
        )
        if any(not is_safe_task_authoring_ref(ref) for ref in refs):
            raise ValueError("unsafe task-draft authoring request reference")
        return self


class TaskDraftRequirementFixture(ContractModel):
    schema_version: Literal["eval-factory/task-draft-requirement-fixture/r4-03"] = (
        "eval-factory/task-draft-requirement-fixture/r4-03"
    )
    requirement_id: Identifier
    statement: str = Field(min_length=1, max_length=4000)
    criticality: AttachmentCriticality
    evidence_priority: EvidencePriority
    evidence_ref_ids: tuple[Identifier, ...] = Field(min_length=1)
    task_episode_ids: tuple[Identifier, ...] = Field(min_length=1)
    conflict_status: RequirementConflict

    @model_validator(mode="after")
    def validate_selection(self) -> TaskDraftRequirementFixture:
        require_unique("requirement evidence", self.evidence_ref_ids)
        require_unique("requirement task episode", self.task_episode_ids)
        return self


class TaskDraftAttachmentFixture(ContractModel):
    schema_version: Literal["eval-factory/task-draft-attachment-fixture/r4-03"] = (
        "eval-factory/task-draft-attachment-fixture/r4-03"
    )
    dependency_id: Identifier
    description: str = Field(min_length=1, max_length=2000)
    criticality: AttachmentCriticality
    evidence_priority: EvidencePriority
    evidence_ref_ids: tuple[Identifier, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_selection(self) -> TaskDraftAttachmentFixture:
        require_unique("attachment evidence", self.evidence_ref_ids)
        return self


class TaskDraftContentFixture(ContractModel):
    schema_version: Literal["eval-factory/task-draft-content-fixture/r4-03"] = (
        "eval-factory/task-draft-content-fixture/r4-03"
    )
    visible_prompt: str = Field(min_length=1, max_length=20000)
    task_intent: str = Field(min_length=1, max_length=4000)
    evaluation_claim: str = Field(min_length=1, max_length=4000)
    required_capability_ids: tuple[Identifier, ...] = Field(min_length=1)
    allowed_tool_ids: tuple[Identifier, ...] = ()
    forbidden_outputs: tuple[str, ...] = Field(min_length=1)
    attachment_dependencies: tuple[TaskDraftAttachmentFixture, ...] = ()
    requirements: tuple[TaskDraftRequirementFixture, ...] = Field(min_length=1)
    prompt_requirement_ids: tuple[Identifier, ...] = Field(min_length=1)
    uncertainties: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_content(self) -> TaskDraftContentFixture:
        require_unique("required capability", self.required_capability_ids)
        require_unique("allowed tool", self.allowed_tool_ids)
        require_unique("forbidden output", self.forbidden_outputs)
        require_unique(
            "attachment dependency",
            (item.dependency_id for item in self.attachment_dependencies),
        )
        requirement_ids = tuple(item.requirement_id for item in self.requirements)
        require_unique("requirement", requirement_ids)
        require_unique("prompt requirement", self.prompt_requirement_ids)
        if not set(self.prompt_requirement_ids).issubset(set(requirement_ids)):
            raise ValueError("prompt requirement IDs must resolve to requirements")
        return self


class FakeTaskDraftAuthoringFixture(ContractModel):
    schema_version: Literal["eval-factory/fake-task-draft-authoring-fixture/r4-03"] = (
        "eval-factory/fake-task-draft-authoring-fixture/r4-03"
    )
    fixture_id: Identifier
    outcome: TaskDraftAuthoringOutcome
    content: TaskDraftContentFixture | None = None
    unresolved_reasons: frozenset[TaskDraftAuthoringReason] = frozenset()
    model_available: bool = True

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> TaskDraftAuthoringOutcome:
        return parse_outcome(value)

    @field_validator("unresolved_reasons", mode="before")
    @classmethod
    def parse_reasons(
        cls,
        value: object,
    ) -> frozenset[TaskDraftAuthoringReason]:
        return parse_reasons(value)

    @model_validator(mode="after")
    def validate_shape(self) -> FakeTaskDraftAuthoringFixture:
        validate_outcome_shape(
            self.outcome,
            self.content,
            self.unresolved_reasons,
        )
        return self


class TaskDraftAuthoringProposal(ContractModel):
    schema_version: Literal["eval-factory/task-draft-authoring-proposal/r4-03"] = (
        "eval-factory/task-draft-authoring-proposal/r4-03"
    )
    semantic_authoring_proposal_id: Identifier
    request_ref: ObjectRef
    selection_context_ref: ObjectRef
    task_episode_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    selection_evidence_bundle_ref: ObjectRef
    authoring_evidence_bundle_ref: ObjectRef
    outcome: TaskDraftAuthoringOutcome
    content: TaskDraftContentFixture | None = None
    unresolved_reasons: frozenset[TaskDraftAuthoringReason] = frozenset()
    model_profile: Identifier
    prompt_version: str = Field(min_length=1, max_length=128)
    policy_version: Literal["task-draft-authoring/r4-03-v1"] = TASK_DRAFT_AUTHORING_POLICY_VERSION
    proposal_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> TaskDraftAuthoringOutcome:
        return parse_outcome(value)

    @field_validator("unresolved_reasons", mode="before")
    @classmethod
    def parse_reasons(
        cls,
        value: object,
    ) -> frozenset[TaskDraftAuthoringReason]:
        return parse_reasons(value)

    @model_validator(mode="after")
    def validate_proposal(self) -> TaskDraftAuthoringProposal:
        _require_ref_type(
            self.request_ref,
            "task-draft-authoring-request",
            "request_ref",
        )
        _require_ref_type(
            self.selection_context_ref,
            "selection-context",
            "selection_context_ref",
        )
        for ref in self.task_episode_refs:
            _require_ref_type(ref, "task-episode", "task_episode_refs")
        _require_ref_type(
            self.selection_evidence_bundle_ref,
            "evidence-bundle",
            "selection_evidence_bundle_ref",
        )
        _require_ref_type(
            self.authoring_evidence_bundle_ref,
            "evidence-bundle",
            "authoring_evidence_bundle_ref",
        )
        validate_outcome_shape(
            self.outcome,
            self.content,
            self.unresolved_reasons,
        )
        return self


class TaskDraftAuthoringResult(ContractModel):
    schema_version: Literal["eval-factory/task-draft-authoring-result/r4-03"] = (
        "eval-factory/task-draft-authoring-result/r4-03"
    )
    task_draft_authoring_result_id: Identifier
    request_ref: ObjectRef
    proposal_ref: ObjectRef
    outcome: TaskDraftAuthoringOutcome
    task_draft: TaskDraftV2 | None = None
    unresolved_reasons: frozenset[TaskDraftAuthoringReason] = frozenset()
    policy_version: Literal["task-draft-authoring/r4-03-v1"] = TASK_DRAFT_AUTHORING_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> TaskDraftAuthoringOutcome:
        return parse_outcome(value)

    @field_validator("unresolved_reasons", mode="before")
    @classmethod
    def parse_reasons(
        cls,
        value: object,
    ) -> frozenset[TaskDraftAuthoringReason]:
        return parse_reasons(value)

    @model_validator(mode="after")
    def validate_result(self) -> TaskDraftAuthoringResult:
        _require_ref_type(
            self.request_ref,
            "task-draft-authoring-request",
            "request_ref",
        )
        _require_ref_type(
            self.proposal_ref,
            "task-draft-authoring-proposal",
            "proposal_ref",
        )
        validate_outcome_shape(
            self.outcome,
            self.task_draft,
            self.unresolved_reasons,
        )
        return self


def validate_outcome_shape(
    outcome: TaskDraftAuthoringOutcome,
    content: object | None,
    reasons: frozenset[TaskDraftAuthoringReason],
) -> None:
    if outcome is TaskDraftAuthoringOutcome.DRAFTED:
        if content is None:
            raise ValueError("DRAFTED outcome requires task content")
        if reasons:
            raise ValueError("DRAFTED outcome cannot carry unresolved reasons")
        return
    if content is not None:
        raise ValueError(f"{outcome.value} outcome cannot carry task content")
    if not reasons:
        raise ValueError(f"{outcome.value} outcome requires unresolved reasons")
    if outcome is TaskDraftAuthoringOutcome.BLOCKED_CAPABILITY and not reasons.intersection(
        {
            TaskDraftAuthoringReason.MISSING_SAFE_AUTHORING_EVIDENCE,
            TaskDraftAuthoringReason.INCOMPLETE_REQUIREMENT_LINEAGE,
            TaskDraftAuthoringReason.MISSING_CAPABILITY,
            TaskDraftAuthoringReason.MODEL_UNAVAILABLE,
        }
    ):
        raise ValueError("BLOCKED_CAPABILITY requires evidence, lineage, capability, or model reason")


def parse_outcome(value: object) -> TaskDraftAuthoringOutcome:
    if isinstance(value, TaskDraftAuthoringOutcome):
        return value
    if isinstance(value, str):
        return TaskDraftAuthoringOutcome(value)
    raise TypeError("outcome must be a TaskDraftAuthoringOutcome")


def parse_reasons(
    value: object,
) -> frozenset[TaskDraftAuthoringReason]:
    if isinstance(value, (frozenset, set, tuple, list)):
        return frozenset(
            item if isinstance(item, TaskDraftAuthoringReason) else TaskDraftAuthoringReason(item)
            for item in value
        )
    raise TypeError("unresolved_reasons must be a collection")


def require_unique(label: str, values: Iterable[object]) -> None:
    seen: set[str] = set()
    for value in values:
        key = str(value)
        if key in seen:
            raise ValueError(f"{label} IDs must be unique")
        seen.add(key)


def _require_ref_type(ref: ObjectRef, expected: str, field_name: str) -> None:
    if ref.object_type != expected:
        raise ValueError(f"{field_name} must reference {expected}")
