from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    ContractModel,
    EvidenceRef,
    Identifier,
    ObjectRef,
    Sha256,
)


class Severity(StrEnum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class FindingStatus(StrEnum):
    OPEN = "OPEN"
    RESOLVED = "RESOLVED"
    SUPERSEDED = "SUPERSEDED"


class FindingScope(StrEnum):
    ITEM = "ITEM"
    BATCH = "BATCH"


class ReviewerFinding(ContractModel):
    schema_version: Literal["eval-factory/reviewer-finding/v1"] = "eval-factory/reviewer-finding/v1"
    finding_id: Identifier
    scope: FindingScope
    severity: Severity
    category: Identifier
    status: FindingStatus
    description: str = Field(min_length=1, max_length=4000)
    subject_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1)
    non_waivable: bool
    introduced_round: int | None = Field(default=None, ge=1, le=3)
    resolution_ref: ObjectRef | None = None
    audit: ContractAudit


class SemanticReviewRound(StrEnum):
    COVERAGE_SOLVABILITY = "COVERAGE_SOLVABILITY"
    REALISM_CONSISTENCY = "REALISM_CONSISTENCY"
    LEAKAGE_EXECUTABILITY = "LEAKAGE_EXECUTABILITY"


class SemanticReviewResult(ContractModel):
    schema_version: Literal["eval-factory/semantic-review-result/v1"] = (
        "eval-factory/semantic-review-result/v1"
    )
    semantic_review_result_id: Identifier
    round: SemanticReviewRound
    stage_run_ref: ObjectRef
    clean_context_attestation_ref: ObjectRef
    evaluated_subject_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    prior_typed_finding_refs: tuple[ObjectRef, ...] = ()
    finding_refs: tuple[ObjectRef, ...]
    accepted: bool
    audit: ContractAudit


class QualityReport(ContractModel):
    schema_version: Literal["eval-factory/quality-report/v1"] = "eval-factory/quality-report/v1"
    quality_report_id: Identifier
    item_subject_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    deterministic_validation_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    semantic_round_refs: tuple[ObjectRef, ObjectRef, ObjectRef]
    finding_refs: tuple[ObjectRef, ...]
    open_p0_count: int = Field(ge=0)
    open_p1_count: int = Field(ge=0)
    unresolved_non_waivable_count: int = Field(ge=0)
    approvable: bool
    audit: ContractAudit

    @model_validator(mode="after")
    def enforce_approval_gate(self) -> QualityReport:
        blockers = self.open_p0_count + self.open_p1_count + self.unresolved_non_waivable_count
        if self.approvable and blockers:
            raise ValueError("QualityReport with blockers cannot be approvable")
        return self


class BatchQualityReport(ContractModel):
    schema_version: Literal["eval-factory/batch-quality-report/v1"] = "eval-factory/batch-quality-report/v1"
    batch_quality_report_id: Identifier
    item_quality_report_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    batch_finding_refs: tuple[ObjectRef, ...]
    open_p0_count: int = Field(ge=0)
    open_p1_count: int = Field(ge=0)
    approvable: bool
    audit: ContractAudit

    @model_validator(mode="after")
    def enforce_approval_gate(self) -> BatchQualityReport:
        if self.approvable and self.open_p0_count + self.open_p1_count:
            raise ValueError("BatchQualityReport with P0/P1 blockers cannot be approvable")
        return self


class ReviewConclusion(StrEnum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    REQUEST_CHANGES = "REQUEST_CHANGES"
    ABSTAIN = "ABSTAIN"
    CONFIRM_EXCEPTION = "CONFIRM_EXCEPTION"
    DENY_EXCEPTION = "DENY_EXCEPTION"


class HumanReviewRecord(ContractModel):
    schema_version: Literal["eval-factory/human-review-record/v1"] = "eval-factory/human-review-record/v1"
    review_record_id: Identifier
    case_id: Identifier
    subject_ref: ObjectRef
    reviewer_id: Identifier
    reviewer_role: Identifier
    review_policy_id: Identifier
    review_policy_version: str = Field(min_length=1, max_length=128)
    projection_ref: ObjectRef
    conclusion: ReviewConclusion
    reason: str = Field(min_length=1, max_length=4000)
    finding_refs: tuple[ObjectRef, ...] = ()
    diff_ref: ObjectRef | None = None
    supersedes: tuple[ObjectRef, ...] = ()
    decided_at: datetime
    record_sha256: Sha256


class ReviewRecordInvalidation(ContractModel):
    schema_version: Literal["eval-factory/review-record-invalidation/v1"] = (
        "eval-factory/review-record-invalidation/v1"
    )
    invalidation_id: Identifier
    review_record_ref: ObjectRef
    reason_code: Identifier
    source_ref: ObjectRef
    policy_version: str = Field(min_length=1, max_length=128)
    invalidated_at: datetime
    invalidation_sha256: Sha256


class ReviewQuorumStatus(StrEnum):
    PENDING = "PENDING"
    SATISFIED = "SATISFIED"
    DENIED = "DENIED"
    STALE = "STALE"


class ReviewQuorum(ContractModel):
    schema_version: Literal["eval-factory/review-quorum/v1"] = "eval-factory/review-quorum/v1"
    quorum_group_id: Identifier
    subject_ref: ObjectRef
    policy_version: str = Field(min_length=1, max_length=128)
    required_roles: tuple[Identifier, ...] = Field(min_length=1)
    required_distinct_identities: int = Field(ge=1)
    case_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    record_refs: tuple[ObjectRef, ...]
    status: ReviewQuorumStatus
    aggregate_sha256: Sha256
