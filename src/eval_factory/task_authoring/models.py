from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    ContractModel,
    EvidenceRef,
    Identifier,
    ObjectRef,
    Sha256,
)
from eval_factory.contracts.task_v2 import TaskEpisodeV2

TASK_EPISODE_GROUPING_POLICY_VERSION: Literal["task-episode-grouping/r4-01-v1"] = (
    "task-episode-grouping/r4-01-v1"
)
TASK_EPISODE_CONSUMER_STAGE: Literal["task-authoring"] = "task-authoring"
TASK_EPISODE_EVIDENCE_PURPOSE: Literal["task-episode-grouping"] = "task-episode-grouping"

_DENIED_REFERENCE_MARKERS = frozenset(
    {
        "answer-bearing",
        "completed-deliverable",
        "configured-pii",
        "final-answer",
        "final-output",
        "grader-rule",
        "hidden-pass-condition",
        "hidden-selection-signal",
        "private-reference",
        "quarantine",
        "raw-trace",
        "raw-traj",
        "restricted-pii",
        "secret",
        "sensitive-pii",
        "sensitive-secret",
        "trace-raw",
    }
)


class TaskEpisodeGroupingPolicyError(RuntimeError):
    pass


class TaskEpisodeGroupingOutcome(StrEnum):
    GROUPED = "GROUPED"
    ABSTAIN = "ABSTAIN"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"


class TaskEpisodeUnresolvedReason(StrEnum):
    AMBIGUOUS_TASK_BOUNDARY = "AMBIGUOUS_TASK_BOUNDARY"
    MISSING_SAFE_EVIDENCE = "MISSING_SAFE_EVIDENCE"
    INCOMPLETE_TRACE_CAPABILITY = "INCOMPLETE_TRACE_CAPABILITY"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"


class TaskEpisodeSegmentView(ContractModel):
    schema_version: Literal["eval-factory/task-episode-segment-view/r4-01"] = (
        "eval-factory/task-episode-segment-view/r4-01"
    )
    segment_id: Identifier
    boundary_method: Identifier
    sequence_start: int = Field(ge=0)
    sequence_end: int = Field(ge=0)
    segment_sha256: Sha256

    @model_validator(mode="after")
    def validate_range(self) -> TaskEpisodeSegmentView:
        if self.sequence_end < self.sequence_start:
            raise ValueError("segment sequence range must be ordered")
        return self


class TaskEpisodeSegmentSelection(ContractModel):
    schema_version: Literal["eval-factory/task-episode-segment-selection/r4-01"] = (
        "eval-factory/task-episode-segment-selection/r4-01"
    )
    segment_id: Identifier
    evidence_ref_ids: tuple[Identifier, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_evidence_ids(self) -> TaskEpisodeSegmentSelection:
        if len(self.evidence_ref_ids) != len(set(self.evidence_ref_ids)):
            raise ValueError("evidence_ref_ids must be unique")
        return self


class TaskEpisodeGroupingRequest(ContractModel):
    schema_version: Literal["eval-factory/task-episode-grouping-request/r4-01"] = (
        "eval-factory/task-episode-grouping-request/r4-01"
    )
    semantic_grouping_request_id: Identifier
    selection_context_ref: ObjectRef
    trace_envelope_ref: ObjectRef
    evidence_bundle_ref: ObjectRef
    projection_policy_ref: ObjectRef
    segments: tuple[TaskEpisodeSegmentView, ...] = Field(min_length=1)
    evidence_refs: tuple[EvidenceRef, ...] = Field(min_length=1)
    abstain_conditions: tuple[str, ...] = Field(min_length=1)
    model_profile: Identifier
    prompt_version: str = Field(min_length=1, max_length=128)
    returned_characters: int = Field(ge=0)
    max_characters: int = Field(ge=0)
    policy_version: Literal["task-episode-grouping/r4-01-v1"] = TASK_EPISODE_GROUPING_POLICY_VERSION
    request_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_request(self) -> TaskEpisodeGroupingRequest:
        _require_ref_type(self.selection_context_ref, "selection-context", "selection_context_ref")
        _require_ref_type(self.trace_envelope_ref, "trace-envelope", "trace_envelope_ref")
        _require_ref_type(self.evidence_bundle_ref, "evidence-bundle", "evidence_bundle_ref")
        _require_ref_type(self.projection_policy_ref, "projection-policy", "projection_policy_ref")
        if self.returned_characters > self.max_characters:
            raise ValueError("returned characters exceed task-episode grouping budget")
        require_unique("segment", (item.segment_id for item in self.segments))
        require_unique("evidence ref", (item.evidence_ref_id for item in self.evidence_refs))
        if any(not condition for condition in self.abstain_conditions):
            raise ValueError("abstain conditions must be non-empty")
        ensure_safe_task_authoring_ref(self.selection_context_ref, "SelectionContext")
        ensure_safe_task_authoring_ref(self.trace_envelope_ref, "trace envelope")
        ensure_safe_task_authoring_ref(self.evidence_bundle_ref, "evidence bundle")
        ensure_safe_task_authoring_ref(self.projection_policy_ref, "projection policy")
        for evidence in self.evidence_refs:
            ensure_safe_task_authoring_ref(evidence.subject_ref, "evidence")
        return self


class TaskEpisodeGroupFixture(ContractModel):
    schema_version: Literal["eval-factory/task-episode-group-fixture/r4-01"] = (
        "eval-factory/task-episode-group-fixture/r4-01"
    )
    group_id: Identifier
    selections: tuple[TaskEpisodeSegmentSelection, ...] = Field(min_length=1)
    rationale_ref: ObjectRef

    @model_validator(mode="after")
    def validate_selections(self) -> TaskEpisodeGroupFixture:
        selection_ids = [item.segment_id for item in self.selections]
        if len(selection_ids) != len(set(selection_ids)):
            raise ValueError("segment selections must be unique")
        return self


class FakeTaskEpisodeGroupingFixture(ContractModel):
    schema_version: Literal["eval-factory/fake-task-episode-grouping-fixture/r4-01"] = (
        "eval-factory/fake-task-episode-grouping-fixture/r4-01"
    )
    fixture_id: Identifier
    outcome: TaskEpisodeGroupingOutcome
    groups: tuple[TaskEpisodeGroupFixture, ...] = ()
    unresolved_reasons: frozenset[TaskEpisodeUnresolvedReason] = frozenset()
    model_available: bool = True

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> TaskEpisodeGroupingOutcome:
        return _parse_outcome(value)

    @field_validator("unresolved_reasons", mode="before")
    @classmethod
    def parse_unresolved_reasons(
        cls,
        value: object,
    ) -> frozenset[TaskEpisodeUnresolvedReason]:
        return _parse_unresolved_reasons(value)


class TaskEpisodeGroupProposal(ContractModel):
    schema_version: Literal["eval-factory/task-episode-group-proposal/r4-01"] = (
        "eval-factory/task-episode-group-proposal/r4-01"
    )
    group_id: Identifier
    selections: tuple[TaskEpisodeSegmentSelection, ...] = Field(min_length=1)
    rationale_ref: ObjectRef

    @model_validator(mode="after")
    def validate_group(self) -> TaskEpisodeGroupProposal:
        selection_ids = [item.segment_id for item in self.selections]
        if len(selection_ids) != len(set(selection_ids)):
            raise ValueError("segment selections must be unique")
        _require_ref_type(self.rationale_ref, "task-episode-rationale", "rationale_ref")
        ensure_safe_task_authoring_ref(self.rationale_ref, "rationale")
        return self


class TaskEpisodeGroupingProposal(ContractModel):
    schema_version: Literal["eval-factory/task-episode-grouping-proposal/r4-01"] = (
        "eval-factory/task-episode-grouping-proposal/r4-01"
    )
    semantic_grouping_proposal_id: Identifier
    request_ref: ObjectRef
    selection_context_ref: ObjectRef
    trace_envelope_ref: ObjectRef
    evidence_bundle_ref: ObjectRef
    outcome: TaskEpisodeGroupingOutcome
    groups: tuple[TaskEpisodeGroupProposal, ...] = ()
    unresolved_reasons: frozenset[TaskEpisodeUnresolvedReason] = frozenset()
    model_profile: Identifier
    prompt_version: str = Field(min_length=1, max_length=128)
    policy_version: Literal["task-episode-grouping/r4-01-v1"] = TASK_EPISODE_GROUPING_POLICY_VERSION
    proposal_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> TaskEpisodeGroupingOutcome:
        return _parse_outcome(value)

    @field_validator("unresolved_reasons", mode="before")
    @classmethod
    def parse_unresolved_reasons(
        cls,
        value: object,
    ) -> frozenset[TaskEpisodeUnresolvedReason]:
        return _parse_unresolved_reasons(value)

    @model_validator(mode="after")
    def validate_proposal(self) -> TaskEpisodeGroupingProposal:
        _require_ref_type(self.request_ref, "task-episode-grouping-request", "request_ref")
        _require_ref_type(self.selection_context_ref, "selection-context", "selection_context_ref")
        _require_ref_type(self.trace_envelope_ref, "trace-envelope", "trace_envelope_ref")
        _require_ref_type(self.evidence_bundle_ref, "evidence-bundle", "evidence_bundle_ref")
        validate_outcome_shape(self.outcome, self.groups, self.unresolved_reasons)
        require_unique("group", (item.group_id for item in self.groups))
        return self


class TaskEpisodeGroupingResult(ContractModel):
    schema_version: Literal["eval-factory/task-episode-grouping-result/r4-01"] = (
        "eval-factory/task-episode-grouping-result/r4-01"
    )
    task_episode_grouping_result_id: Identifier
    request_ref: ObjectRef
    proposal_ref: ObjectRef
    outcome: TaskEpisodeGroupingOutcome
    episodes: tuple[TaskEpisodeV2, ...] = ()
    unresolved_reasons: frozenset[TaskEpisodeUnresolvedReason] = frozenset()
    policy_version: Literal["task-episode-grouping/r4-01-v1"] = TASK_EPISODE_GROUPING_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> TaskEpisodeGroupingOutcome:
        return _parse_outcome(value)

    @field_validator("unresolved_reasons", mode="before")
    @classmethod
    def parse_unresolved_reasons(
        cls,
        value: object,
    ) -> frozenset[TaskEpisodeUnresolvedReason]:
        return _parse_unresolved_reasons(value)

    @model_validator(mode="after")
    def validate_result(self) -> TaskEpisodeGroupingResult:
        _require_ref_type(self.request_ref, "task-episode-grouping-request", "request_ref")
        _require_ref_type(self.proposal_ref, "task-episode-grouping-proposal", "proposal_ref")
        validate_outcome_shape(self.outcome, self.episodes, self.unresolved_reasons)
        require_unique("task episode", (item.task_episode_id for item in self.episodes))
        return self


def ensure_safe_task_authoring_ref(ref: ObjectRef, label: str) -> None:
    if not is_safe_task_authoring_ref(ref):
        raise TaskEpisodeGroupingPolicyError(f"unsafe {label} reference")


def is_safe_task_authoring_ref(ref: ObjectRef) -> bool:
    normalized = f"{ref.object_type}:{ref.object_id}".casefold().replace("_", "-")
    return not any(marker in normalized for marker in _DENIED_REFERENCE_MARKERS)


def validate_outcome_shape(
    outcome: TaskEpisodeGroupingOutcome,
    outputs: tuple[object, ...],
    reasons: frozenset[TaskEpisodeUnresolvedReason],
) -> None:
    if outcome is TaskEpisodeGroupingOutcome.GROUPED:
        if not outputs:
            raise ValueError("GROUPED outcome requires at least one task episode group")
        if reasons:
            raise ValueError("GROUPED outcome cannot carry unresolved reasons")
        return
    if outputs:
        raise ValueError(f"{outcome.value} outcome cannot carry task episode groups")
    if not reasons:
        raise ValueError(f"{outcome.value} outcome requires unresolved reasons")
    if outcome is TaskEpisodeGroupingOutcome.BLOCKED_CAPABILITY and not reasons.intersection(
        {
            TaskEpisodeUnresolvedReason.MISSING_SAFE_EVIDENCE,
            TaskEpisodeUnresolvedReason.INCOMPLETE_TRACE_CAPABILITY,
            TaskEpisodeUnresolvedReason.MODEL_UNAVAILABLE,
        }
    ):
        raise ValueError("BLOCKED_CAPABILITY requires an evidence, capability, or model reason")


def require_unique(label: str, values: Iterable[object]) -> None:
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text in seen:
            raise TaskEpisodeGroupingPolicyError(f"duplicate {label}: {text}")
        seen.add(text)


def _parse_outcome(value: object) -> TaskEpisodeGroupingOutcome:
    if isinstance(value, TaskEpisodeGroupingOutcome):
        return value
    if isinstance(value, str):
        return TaskEpisodeGroupingOutcome(value)
    raise TypeError("outcome must be a TaskEpisodeGroupingOutcome")


def _parse_unresolved_reasons(
    value: object,
) -> frozenset[TaskEpisodeUnresolvedReason]:
    if isinstance(value, frozenset):
        return frozenset(
            item if isinstance(item, TaskEpisodeUnresolvedReason) else TaskEpisodeUnresolvedReason(item)
            for item in value
        )
    if isinstance(value, (list, tuple, set)):
        return frozenset(TaskEpisodeUnresolvedReason(item) for item in value)
    raise TypeError("unresolved_reasons must be a collection")


def _require_ref_type(ref: ObjectRef, expected: str, field_name: str) -> None:
    if ref.object_type != expected:
        raise ValueError(f"{field_name} must reference {expected}")
