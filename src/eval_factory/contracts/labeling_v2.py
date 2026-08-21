from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    EvidenceRef,
    Identifier,
    ObjectRef,
    ScalarValue,
    Sha256,
)
from eval_factory.contracts.core_v2 import ContractModelV2


class PredicateOperatorV2(StrEnum):
    EXISTS = "EXISTS"
    NOT_EXISTS = "NOT_EXISTS"
    EQUALS = "EQUALS"
    CONTAINS = "CONTAINS"
    REGEX = "REGEX"
    SEQUENCE = "SEQUENCE"
    WITHIN_WINDOW = "WITHIN_WINDOW"
    ERROR_SIGNATURE = "ERROR_SIGNATURE"
    NOT_ERROR_SIGNATURE = "NOT_ERROR_SIGNATURE"


class StructuredPredicateV2(ContractModelV2):
    schema_version: Literal["eval-factory/structured-predicate/v2"] = "eval-factory/structured-predicate/v2"
    predicate_id: Identifier
    fact_type: Identifier
    field_path: str = Field(min_length=1, max_length=512)
    operator: PredicateOperatorV2
    expected_value: ScalarValue = None
    window_events: int | None = Field(default=None, ge=1)
    required_capability: Identifier
    rule_version: str = Field(min_length=1, max_length=128)

    @field_validator("operator", mode="before")
    @classmethod
    def parse_operator(cls, value: object) -> PredicateOperatorV2:
        if isinstance(value, PredicateOperatorV2):
            return value
        if isinstance(value, str):
            return PredicateOperatorV2(value)
        raise TypeError("operator must be a PredicateOperatorV2")


class SemanticResidualSpecV2(ContractModelV2):
    schema_version: Literal["eval-factory/semantic-residual-spec/v2"] = (
        "eval-factory/semantic-residual-spec/v2"
    )
    residual_id: Identifier
    question: str = Field(min_length=1, max_length=4000)
    evidence_bundle_purpose: Identifier
    allowed_evidence_types: tuple[Identifier, ...] = Field(min_length=1)
    abstain_conditions: tuple[str, ...] = Field(min_length=1)
    model_profile: Identifier
    prompt_version: str = Field(min_length=1, max_length=128)


class LabelSpecV2(ContractModelV2):
    schema_version: Literal["eval-factory/label-spec/v2"] = "eval-factory/label-spec/v2"
    label_spec_id: Identifier
    label_version: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=255)
    requirement: str = Field(min_length=1, max_length=8000)
    prerequisite_predicates: tuple[StructuredPredicateV2, ...] = ()
    positive_predicates: tuple[StructuredPredicateV2, ...] = ()
    negative_predicates: tuple[StructuredPredicateV2, ...] = ()
    semantic_residual: SemanticResidualSpecV2 | None = None
    decision_threshold: float = Field(ge=0, le=1)
    review_threshold: float = Field(ge=0, le=1)
    label_plan_ref: ObjectRef | None = None
    policy_version: str = Field(min_length=1, max_length=128)
    label_spec_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_spec(self) -> LabelSpecV2:
        if self.review_threshold > self.decision_threshold:
            raise ValueError("review threshold cannot exceed decision threshold")
        predicates = (
            *self.prerequisite_predicates,
            *self.positive_predicates,
            *self.negative_predicates,
        )
        if not predicates and self.semantic_residual is None:
            raise ValueError("LabelSpecV2 requires at least one structured predicate or semantic residual")
        predicate_ids = [item.predicate_id for item in predicates]
        if len(predicate_ids) != len(set(predicate_ids)):
            raise ValueError("predicate IDs must be unique")
        if self.label_plan_ref is not None and self.label_plan_ref.object_type != "label-plan":
            raise ValueError("label_plan_ref must reference label-plan")
        return self


class LabelDecisionValueV2(StrEnum):
    MATCH = "MATCH"
    NO_MATCH = "NO_MATCH"
    ABSTAIN = "ABSTAIN"


class LabelExecutionStatus(StrEnum):
    FINAL = "FINAL"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    UNRESOLVED = "UNRESOLVED"


class LabelUnresolvedReason(StrEnum):
    AMBIGUOUS_EVIDENCE = "AMBIGUOUS_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    INCOMPLETE_STRUCTURED_CAPABILITY = "INCOMPLETE_STRUCTURED_CAPABILITY"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    MISSING_EVIDENCE = "MISSING_EVIDENCE"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    USER_INSPECTION_REQUIRED = "USER_INSPECTION_REQUIRED"


class LabelDecisionV2(ContractModelV2):
    schema_version: Literal["eval-factory/label-decision/v2"] = "eval-factory/label-decision/v2"
    label_decision_id: Identifier
    label_spec_ref: ObjectRef
    trace_envelope_ref: ObjectRef
    decision: LabelDecisionValueV2
    execution_status: LabelExecutionStatus
    positive_evidence: tuple[EvidenceRef, ...] = ()
    negative_evidence: tuple[EvidenceRef, ...] = ()
    semantic_evidence: tuple[EvidenceRef, ...] = ()
    structured_capability_complete: bool
    confidence: float = Field(ge=0, le=1)
    rule_version: str | None = Field(default=None, min_length=1, max_length=128)
    model_profile: Identifier | None = None
    prompt_version: str | None = Field(default=None, min_length=1, max_length=128)
    unresolved_reasons: frozenset[LabelUnresolvedReason] = frozenset()
    policy_version: str = Field(min_length=1, max_length=128)
    decision_sha256: Sha256
    audit: ContractAudit

    @field_validator("decision", mode="before")
    @classmethod
    def parse_decision(cls, value: object) -> LabelDecisionValueV2:
        if isinstance(value, LabelDecisionValueV2):
            return value
        if isinstance(value, str):
            return LabelDecisionValueV2(value)
        raise TypeError("decision must be a LabelDecisionValueV2")

    @field_validator("execution_status", mode="before")
    @classmethod
    def parse_execution_status(cls, value: object) -> LabelExecutionStatus:
        if isinstance(value, LabelExecutionStatus):
            return value
        if isinstance(value, str):
            return LabelExecutionStatus(value)
        raise TypeError("execution_status must be a LabelExecutionStatus")

    @field_validator("unresolved_reasons", mode="before")
    @classmethod
    def parse_unresolved_reasons(cls, value: object) -> frozenset[LabelUnresolvedReason]:
        if isinstance(value, frozenset):
            return frozenset(
                item if isinstance(item, LabelUnresolvedReason) else LabelUnresolvedReason(item)
                for item in value
            )
        if isinstance(value, (list, tuple, set)):
            return frozenset(LabelUnresolvedReason(item) for item in value)
        raise TypeError("unresolved_reasons must be a collection")

    @model_validator(mode="after")
    def validate_decision(self) -> LabelDecisionV2:
        if self.label_spec_ref.object_type != "label-spec":
            raise ValueError("label_spec_ref must reference label-spec")
        if self.trace_envelope_ref.object_type != "trace-envelope":
            raise ValueError("trace_envelope_ref must reference trace-envelope")
        if self.decision is LabelDecisionValueV2.MATCH and not (
            self.positive_evidence or self.semantic_evidence
        ):
            raise ValueError("MATCH requires positive or semantic evidence")
        if self.decision is LabelDecisionValueV2.NO_MATCH:
            if not self.structured_capability_complete:
                raise ValueError("NO_MATCH requires complete structured capability")
            if not self.negative_evidence:
                raise ValueError("NO_MATCH requires negative evidence")
            if any(not evidence.capability_complete for evidence in self.negative_evidence):
                raise ValueError("NO_MATCH evidence must be capability-complete")
        if self.semantic_evidence and (self.model_profile is None or self.prompt_version is None):
            raise ValueError("semantic evidence requires model_profile and prompt_version")
        if self.decision is LabelDecisionValueV2.ABSTAIN and not self.unresolved_reasons:
            raise ValueError("ABSTAIN requires unresolved reason")
        if self.execution_status is LabelExecutionStatus.UNRESOLVED and not self.unresolved_reasons:
            raise ValueError("UNRESOLVED status requires unresolved reason")
        if self.execution_status is LabelExecutionStatus.FINAL and self.unresolved_reasons:
            raise ValueError("FINAL decisions cannot carry unresolved reasons")
        return self


_DENIED_SELECTION_OBJECT_TYPES = frozenset(
    {
        "answer-bearing",
        "final-answer",
        "final-output",
        "grader-rule",
        "hidden-pass-condition",
        "hidden-selection-signal",
        "private-reference",
        "quarantine",
        "quarantine-subject",
        "raw-trace",
        "raw-traj",
        "trace-raw",
    }
)


class SelectionContextV2(ContractModelV2):
    schema_version: Literal["eval-factory/selection-context/v2"] = "eval-factory/selection-context/v2"
    selection_context_id: Identifier
    candidate_id: Identifier
    approved_label_decision_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    safe_evidence_bundle_ref: ObjectRef
    task_authoring_note_refs: tuple[ObjectRef, ...] = ()
    excluded_signal_hashes: tuple[Sha256, ...] = Field(min_length=1)
    projection_policy_ref: ObjectRef
    policy_version: str = Field(min_length=1, max_length=128)
    selection_context_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_selection_context(self) -> SelectionContextV2:
        for ref in self.approved_label_decision_refs:
            if ref.object_type != "label-decision":
                raise ValueError("approved_label_decision_refs must reference label-decision")
        if self.safe_evidence_bundle_ref.object_type != "evidence-bundle":
            raise ValueError("safe_evidence_bundle_ref must reference evidence-bundle")
        if self.projection_policy_ref.object_type != "projection-policy":
            raise ValueError("projection_policy_ref must reference projection-policy")
        for ref in (
            *self.approved_label_decision_refs,
            self.safe_evidence_bundle_ref,
            *self.task_authoring_note_refs,
            self.projection_policy_ref,
        ):
            if ref.object_type in _DENIED_SELECTION_OBJECT_TYPES:
                raise ValueError("unsafe selection context reference")
        return self


def label_decision_ref(decision: LabelDecisionV2) -> ObjectRef:
    return ObjectRef(
        object_type="label-decision",
        object_id=decision.label_decision_id,
        object_version=decision.policy_version,
        object_sha256=decision.decision_sha256,
    )


def selection_context_ref(context: SelectionContextV2) -> ObjectRef:
    return ObjectRef(
        object_type="selection-context",
        object_id=context.selection_context_id,
        object_version="v2",
        object_sha256=context.selection_context_sha256,
    )
