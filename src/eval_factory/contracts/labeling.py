from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    ContractModel,
    EvidenceRef,
    Identifier,
    ObjectRef,
    ScalarValue,
    Sha256,
)


class PredicateOperator(StrEnum):
    EXISTS = "EXISTS"
    NOT_EXISTS = "NOT_EXISTS"
    EQUALS = "EQUALS"
    CONTAINS = "CONTAINS"
    REGEX = "REGEX"
    SEQUENCE = "SEQUENCE"
    WITHIN_WINDOW = "WITHIN_WINDOW"
    ERROR_SIGNATURE = "ERROR_SIGNATURE"


class StructuredPredicate(ContractModel):
    schema_version: Literal["eval-factory/structured-predicate/v1"] = "eval-factory/structured-predicate/v1"
    predicate_id: Identifier
    fact_type: Identifier
    field_path: str = Field(min_length=1, max_length=512)
    operator: PredicateOperator
    expected_value: ScalarValue = None
    window_events: int | None = Field(default=None, ge=1)
    required_capability: Identifier


class SemanticResidualSpec(ContractModel):
    schema_version: Literal["eval-factory/semantic-residual-spec/v1"] = (
        "eval-factory/semantic-residual-spec/v1"
    )
    residual_id: Identifier
    question: str = Field(min_length=1, max_length=4000)
    evidence_bundle_purpose: Identifier
    allowed_evidence_types: tuple[Identifier, ...] = Field(min_length=1)
    abstain_conditions: tuple[str, ...] = Field(min_length=1)
    model_profile: Identifier
    prompt_version: str = Field(min_length=1, max_length=128)


class LabelSpec(ContractModel):
    schema_version: Literal["eval-factory/label-spec/v1"] = "eval-factory/label-spec/v1"
    label_spec_id: Identifier
    label_version: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=255)
    requirement: str = Field(min_length=1, max_length=8000)
    positive_predicates: tuple[StructuredPredicate, ...]
    negative_predicates: tuple[StructuredPredicate, ...]
    semantic_residual: SemanticResidualSpec | None = None
    decision_threshold: float = Field(ge=0, le=1)
    review_threshold: float = Field(ge=0, le=1)
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_thresholds(self) -> LabelSpec:
        if self.review_threshold > self.decision_threshold:
            raise ValueError("review threshold cannot exceed decision threshold")
        return self


class LabelDecisionValue(StrEnum):
    MATCH = "MATCH"
    NO_MATCH = "NO_MATCH"
    ABSTAIN = "ABSTAIN"


class ReviewStatus(StrEnum):
    NOT_REVIEWED = "NOT_REVIEWED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class LabelDecision(ContractModel):
    schema_version: Literal["eval-factory/label-decision/v1"] = "eval-factory/label-decision/v1"
    label_decision_id: Identifier
    label_spec_ref: ObjectRef
    trace_envelope_ref: ObjectRef
    decision: LabelDecisionValue
    positive_evidence: tuple[EvidenceRef, ...] = ()
    negative_evidence: tuple[EvidenceRef, ...] = ()
    semantic_evidence: tuple[EvidenceRef, ...] = ()
    structured_capability_complete: bool
    confidence: float = Field(ge=0, le=1)
    rule_version: str | None = Field(default=None, max_length=128)
    model_profile: Identifier | None = None
    prompt_version: str | None = Field(default=None, max_length=128)
    review_status: ReviewStatus
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_evidence(self) -> LabelDecision:
        if self.decision is LabelDecisionValue.MATCH and not self.positive_evidence:
            raise ValueError("MATCH requires positive evidence")
        if self.decision is LabelDecisionValue.NO_MATCH:
            if not self.structured_capability_complete:
                raise ValueError("NO_MATCH requires complete structured capability")
            if not self.negative_evidence:
                raise ValueError("NO_MATCH requires negative evidence")
            if any(not evidence.capability_complete for evidence in self.negative_evidence):
                raise ValueError("NO_MATCH evidence must be capability-complete")
        return self


class SelectionContext(ContractModel):
    schema_version: Literal["eval-factory/selection-context/v1"] = "eval-factory/selection-context/v1"
    selection_context_id: Identifier
    candidate_id: Identifier
    approved_label_decision_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    safe_evidence_bundle_ref: ObjectRef
    task_authoring_note_refs: tuple[ObjectRef, ...] = ()
    excluded_signal_hashes: tuple[Sha256, ...] = Field(min_length=1)
    projection_policy_ref: ObjectRef
    audit: ContractAudit
