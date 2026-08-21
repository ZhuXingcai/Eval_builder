from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
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
from eval_factory.contracts.labeling_v2 import (
    LabelDecisionV2,
    LabelDecisionValueV2,
    LabelExecutionStatus,
    LabelUnresolvedReason,
    label_decision_ref,
)
from eval_factory.labeling.batch import (
    LABEL_BATCH_EXPORT_ROW_FIELDS,
    LABEL_BATCH_POLICY_VERSION,
)

LABEL_CALIBRATION_POLICY_VERSION: Literal["label-calibration/r3-06-v1"] = "label-calibration/r3-06-v1"


class LabelCalibrationPolicyError(RuntimeError):
    pass


class LabelCalibrationClaimScope(StrEnum):
    DEVELOPMENT_CANARY_ONLY = "DEVELOPMENT_CANARY_ONLY"


class LabelCalibrationOutcome(StrEnum):
    EXACT_MATCH = "EXACT_MATCH"
    APPROVED_ABSTAIN = "APPROVED_ABSTAIN"
    BLOCKING_FINDING = "BLOCKING_FINDING"
    MISSING_REFERENCE = "MISSING_REFERENCE"
    MISSING_DECISION = "MISSING_DECISION"


class LabelCalibrationFindingCode(StrEnum):
    DECISION_MISMATCH = "DECISION_MISMATCH"
    EVIDENCE_MISMATCH = "EVIDENCE_MISMATCH"
    STRUCTURED_CAPABILITY_MISMATCH = "STRUCTURED_CAPABILITY_MISMATCH"
    CONFIDENCE_MISMATCH = "CONFIDENCE_MISMATCH"
    RULE_VERSION_MISMATCH = "RULE_VERSION_MISMATCH"
    UNEXPECTED_SEMANTIC_EVALUATION = "UNEXPECTED_SEMANTIC_EVALUATION"
    MISSING_SEMANTIC_METADATA = "MISSING_SEMANTIC_METADATA"
    SEMANTIC_METADATA_MISMATCH = "SEMANTIC_METADATA_MISMATCH"
    REFERENCE_NOT_APPROVED = "REFERENCE_NOT_APPROVED"
    UNRESOLVED_BLOCKING_OUTCOME = "UNRESOLVED_BLOCKING_OUTCOME"
    MISSING_REFERENCE = "MISSING_REFERENCE"
    MISSING_DECISION = "MISSING_DECISION"


class LabelCalibrationReviewStatus(StrEnum):
    NOT_REVIEWED = "NOT_REVIEWED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class LabelCalibrationSemanticResidualStatus(StrEnum):
    NOT_REQUIRED = "NOT_REQUIRED"
    REQUIRED = "REQUIRED"
    BLOCKED = "BLOCKED"


class LabelCalibrationObservationSource(StrEnum):
    DECISION_OBJECT = "DECISION_OBJECT"
    BATCH_EXPORT_ROW = "BATCH_EXPORT_ROW"


_DENIED_REFERENCE_MARKERS = frozenset(
    {
        "answer-bearing",
        "configured-pii",
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
        "secret",
        "sensitive-pii",
        "sensitive-secret",
        "trace-raw",
    }
)


class LabelCalibrationReference(ContractModel):
    schema_version: Literal["eval-factory/label-calibration-reference/r3-06"] = (
        "eval-factory/label-calibration-reference/r3-06"
    )
    annotation_ref: ObjectRef
    canary_instance_id: Identifier
    trace_envelope_ref: ObjectRef
    label_spec_ref: ObjectRef
    expected_decision: LabelDecisionValueV2
    positive_evidence_ref_ids: tuple[Identifier, ...] = ()
    negative_evidence_ref_ids: tuple[Identifier, ...] = ()
    semantic_evidence_ref_ids: tuple[Identifier, ...] = ()
    structured_capability_complete: bool
    semantic_residual_status: LabelCalibrationSemanticResidualStatus
    confidence: float = Field(ge=0, le=1)
    rule_version: str | None = Field(default=None, min_length=1, max_length=128)
    model_profile: Identifier | None = None
    prompt_version: str | None = Field(default=None, min_length=1, max_length=128)
    review_status: LabelCalibrationReviewStatus
    policy_version: str = Field(min_length=1, max_length=128)

    @field_validator("expected_decision", mode="before")
    @classmethod
    def parse_expected_decision(cls, value: object) -> LabelDecisionValueV2:
        if isinstance(value, LabelDecisionValueV2):
            return value
        if isinstance(value, str):
            return LabelDecisionValueV2(value)
        raise TypeError("expected_decision must be a LabelDecisionValueV2")

    @field_validator("semantic_residual_status", mode="before")
    @classmethod
    def parse_semantic_status(cls, value: object) -> LabelCalibrationSemanticResidualStatus:
        if isinstance(value, LabelCalibrationSemanticResidualStatus):
            return value
        if isinstance(value, str):
            return LabelCalibrationSemanticResidualStatus(value)
        raise TypeError("semantic_residual_status must be a LabelCalibrationSemanticResidualStatus")

    @field_validator("review_status", mode="before")
    @classmethod
    def parse_review_status(cls, value: object) -> LabelCalibrationReviewStatus:
        if isinstance(value, LabelCalibrationReviewStatus):
            return value
        if isinstance(value, str):
            return LabelCalibrationReviewStatus(value)
        raise TypeError("review_status must be a LabelCalibrationReviewStatus")

    @field_validator(
        "positive_evidence_ref_ids",
        "negative_evidence_ref_ids",
        "semantic_evidence_ref_ids",
        mode="before",
    )
    @classmethod
    def parse_evidence_ids(cls, value: object) -> tuple[str, ...]:
        return _sorted_identifiers(value, field_name="evidence ref IDs")

    @model_validator(mode="after")
    def validate_reference(self) -> LabelCalibrationReference:
        _require_ref_type(self.annotation_ref, "label-annotation", "annotation_ref")
        _require_ref_type(self.trace_envelope_ref, "trace-envelope", "trace_envelope_ref")
        _require_ref_type(self.label_spec_ref, "label-spec", "label_spec_ref")
        _validate_safe_ref(self.annotation_ref)
        _validate_safe_ref(self.trace_envelope_ref)
        _validate_safe_ref(self.label_spec_ref)
        _validate_safe_identifier(self.canary_instance_id)
        for evidence_id in (
            *self.positive_evidence_ref_ids,
            *self.negative_evidence_ref_ids,
            *self.semantic_evidence_ref_ids,
        ):
            _validate_safe_identifier(evidence_id)
        if self.expected_decision is LabelDecisionValueV2.MATCH and not (
            self.positive_evidence_ref_ids or self.semantic_evidence_ref_ids
        ):
            raise ValueError("MATCH calibration reference requires positive or semantic evidence")
        if self.expected_decision is LabelDecisionValueV2.NO_MATCH:
            if not self.structured_capability_complete:
                raise ValueError("NO_MATCH calibration reference requires complete structured capability")
            if not self.negative_evidence_ref_ids:
                raise ValueError("NO_MATCH calibration reference requires negative evidence")
        if self.semantic_residual_status is LabelCalibrationSemanticResidualStatus.REQUIRED:
            if not self.semantic_evidence_ref_ids:
                raise ValueError("REQUIRED semantic residual requires semantic evidence")
            if self.model_profile is None or self.prompt_version is None:
                raise ValueError("REQUIRED semantic residual requires model and prompt metadata")
        if self.semantic_residual_status is LabelCalibrationSemanticResidualStatus.NOT_REQUIRED and (
            self.semantic_evidence_ref_ids
            or self.model_profile is not None
            or self.prompt_version is not None
        ):
            raise ValueError("NOT_REQUIRED semantic residual cannot carry semantic metadata")
        if (
            self.semantic_residual_status is LabelCalibrationSemanticResidualStatus.BLOCKED
            and self.expected_decision is not LabelDecisionValueV2.ABSTAIN
        ):
            raise ValueError("BLOCKED semantic residual requires expected ABSTAIN")
        return self


class LabelCalibrationObservation(ContractModel):
    schema_version: Literal["eval-factory/label-calibration-observation/r3-06"] = (
        "eval-factory/label-calibration-observation/r3-06"
    )
    source: LabelCalibrationObservationSource
    source_row_id: Identifier | None = None
    decision_ref: ObjectRef
    trace_envelope_ref: ObjectRef
    label_spec_ref: ObjectRef
    decision: LabelDecisionValueV2
    execution_status: LabelExecutionStatus
    positive_evidence_ref_ids: tuple[Identifier, ...] = ()
    negative_evidence_ref_ids: tuple[Identifier, ...] = ()
    semantic_evidence_ref_ids: tuple[Identifier, ...] = ()
    structured_capability_complete: bool
    confidence: float = Field(ge=0, le=1)
    rule_version: str | None = Field(default=None, min_length=1, max_length=128)
    model_profile: Identifier | None = None
    prompt_version: str | None = Field(default=None, min_length=1, max_length=128)
    unresolved_reasons: frozenset[LabelUnresolvedReason] = frozenset()
    decision_policy_version: str = Field(min_length=1, max_length=128)

    @classmethod
    def from_decision(cls, decision: LabelDecisionV2) -> LabelCalibrationObservation:
        for evidence in (
            *decision.positive_evidence,
            *decision.negative_evidence,
            *decision.semantic_evidence,
        ):
            _validate_safe_ref(evidence.subject_ref)
            _validate_safe_identifier(evidence.evidence_ref_id)
        return cls(
            source=LabelCalibrationObservationSource.DECISION_OBJECT,
            decision_ref=label_decision_ref(decision),
            trace_envelope_ref=decision.trace_envelope_ref,
            label_spec_ref=decision.label_spec_ref,
            decision=decision.decision,
            execution_status=decision.execution_status,
            positive_evidence_ref_ids=_evidence_ids(decision.positive_evidence),
            negative_evidence_ref_ids=_evidence_ids(decision.negative_evidence),
            semantic_evidence_ref_ids=_evidence_ids(decision.semantic_evidence),
            structured_capability_complete=decision.structured_capability_complete,
            confidence=decision.confidence,
            rule_version=decision.rule_version,
            model_profile=decision.model_profile,
            prompt_version=decision.prompt_version,
            unresolved_reasons=decision.unresolved_reasons,
            decision_policy_version=decision.policy_version,
        )

    @classmethod
    def from_export_row(cls, row: Mapping[str, object]) -> LabelCalibrationObservation:
        expected_fields = set(LABEL_BATCH_EXPORT_ROW_FIELDS)
        observed_fields = set(row)
        if observed_fields != expected_fields:
            missing = sorted(expected_fields - observed_fields)
            unknown = sorted(observed_fields - expected_fields)
            raise LabelCalibrationPolicyError(
                f"label batch export row fields mismatch: missing={missing}, unknown={unknown}"
            )
        if _require_str(row, "batch_policy_version") != LABEL_BATCH_POLICY_VERSION:
            raise LabelCalibrationPolicyError("unsupported label batch export policy version")
        if _require_str(row, "pair_status") != "SUCCEEDED":
            raise LabelCalibrationPolicyError("calibration export row must be SUCCEEDED")
        decision_ref = _row_ref(row, "label_decision_ref")
        decision_sha256 = _require_str(row, "label_decision_sha256")
        if decision_ref.object_sha256 != decision_sha256:
            raise LabelCalibrationPolicyError("label decision ref hash mismatches decision hash")
        return cls(
            source=LabelCalibrationObservationSource.BATCH_EXPORT_ROW,
            source_row_id=_require_str(row, "row_id"),
            decision_ref=decision_ref,
            trace_envelope_ref=_row_ref(row, "trace_envelope_ref"),
            label_spec_ref=_row_ref(row, "label_spec_ref"),
            decision=LabelDecisionValueV2(_require_str(row, "decision")),
            execution_status=LabelExecutionStatus(_require_str(row, "execution_status")),
            positive_evidence_ref_ids=_require_str_sequence(
                row,
                "positive_evidence_ref_ids",
            ),
            negative_evidence_ref_ids=_require_str_sequence(
                row,
                "negative_evidence_ref_ids",
            ),
            semantic_evidence_ref_ids=_require_str_sequence(
                row,
                "semantic_evidence_ref_ids",
            ),
            structured_capability_complete=_require_bool(
                row,
                "structured_capability_complete",
            ),
            confidence=_require_float(row, "confidence"),
            rule_version=_optional_str(row, "rule_version"),
            model_profile=_optional_str(row, "model_profile"),
            prompt_version=_optional_str(row, "prompt_version"),
            unresolved_reasons=frozenset(
                LabelUnresolvedReason(value) for value in _require_str_sequence(row, "unresolved_reasons")
            ),
            decision_policy_version=_require_str(
                row,
                "label_decision_policy_version",
            ),
        )

    @field_validator("source", mode="before")
    @classmethod
    def parse_source(cls, value: object) -> LabelCalibrationObservationSource:
        if isinstance(value, LabelCalibrationObservationSource):
            return value
        if isinstance(value, str):
            return LabelCalibrationObservationSource(value)
        raise TypeError("source must be a LabelCalibrationObservationSource")

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

    @field_validator(
        "positive_evidence_ref_ids",
        "negative_evidence_ref_ids",
        "semantic_evidence_ref_ids",
        mode="before",
    )
    @classmethod
    def parse_evidence_ids(cls, value: object) -> tuple[str, ...]:
        return _sorted_identifiers(value, field_name="evidence ref IDs")

    @field_validator("unresolved_reasons", mode="before")
    @classmethod
    def parse_unresolved_reasons(cls, value: object) -> frozenset[LabelUnresolvedReason]:
        if isinstance(value, (frozenset, set, tuple, list)):
            return frozenset(
                item if isinstance(item, LabelUnresolvedReason) else LabelUnresolvedReason(item)
                for item in value
            )
        raise TypeError("unresolved_reasons must be a collection")

    @model_validator(mode="after")
    def validate_observation(self) -> LabelCalibrationObservation:
        _require_ref_type(self.decision_ref, "label-decision", "decision_ref")
        _require_ref_type(self.trace_envelope_ref, "trace-envelope", "trace_envelope_ref")
        _require_ref_type(self.label_spec_ref, "label-spec", "label_spec_ref")
        for ref in (self.decision_ref, self.trace_envelope_ref, self.label_spec_ref):
            _validate_safe_ref(ref)
        for evidence_id in (
            *self.positive_evidence_ref_ids,
            *self.negative_evidence_ref_ids,
            *self.semantic_evidence_ref_ids,
        ):
            _validate_safe_identifier(evidence_id)
        if self.source is LabelCalibrationObservationSource.BATCH_EXPORT_ROW and self.source_row_id is None:
            raise ValueError("batch export observation requires source_row_id")
        if self.source_row_id is not None:
            _validate_safe_identifier(self.source_row_id)
        if (
            self.source is LabelCalibrationObservationSource.DECISION_OBJECT
            and self.source_row_id is not None
        ):
            raise ValueError("decision object observation cannot carry source_row_id")
        return self


class LabelCalibrationInput(ContractModel):
    schema_version: Literal["eval-factory/label-calibration-input/r3-06"] = (
        "eval-factory/label-calibration-input/r3-06"
    )
    canary_manifest_ref: ObjectRef
    annotation_contract_ref: ObjectRef
    annotation_guide_ref: ObjectRef
    batch_export_ref: ObjectRef | None = None
    references: tuple[LabelCalibrationReference, ...] = ()
    observations: tuple[LabelCalibrationObservation, ...] = ()
    policy_version: Literal["label-calibration/r3-06-v1"] = LABEL_CALIBRATION_POLICY_VERSION
    audit: ContractAudit

    @classmethod
    def build(
        cls,
        *,
        canary_manifest_ref: ObjectRef,
        annotation_contract_ref: ObjectRef,
        annotation_guide_ref: ObjectRef,
        references: tuple[LabelCalibrationReference, ...],
        audit: ContractAudit,
        decisions: tuple[LabelDecisionV2, ...] = (),
        export_rows: tuple[Mapping[str, object], ...] = (),
        batch_export_ref: ObjectRef | None = None,
    ) -> LabelCalibrationInput:
        try:
            observations = (
                *(LabelCalibrationObservation.from_decision(decision) for decision in decisions),
                *(LabelCalibrationObservation.from_export_row(row) for row in export_rows),
            )
        except ValueError as exc:
            raise LabelCalibrationPolicyError(str(exc)) from exc
        if export_rows and batch_export_ref is None:
            raise LabelCalibrationPolicyError("batch export rows require batch_export_ref")
        return cls(
            canary_manifest_ref=canary_manifest_ref,
            annotation_contract_ref=annotation_contract_ref,
            annotation_guide_ref=annotation_guide_ref,
            batch_export_ref=batch_export_ref,
            references=references,
            observations=observations,
            audit=audit,
        )

    @model_validator(mode="after")
    def validate_input(self) -> LabelCalibrationInput:
        _require_ref_type(self.canary_manifest_ref, "canary-manifest", "canary_manifest_ref")
        _require_ref_type(
            self.annotation_contract_ref,
            "annotation-contract-manifest",
            "annotation_contract_ref",
        )
        _require_ref_type(self.annotation_guide_ref, "annotation-guide", "annotation_guide_ref")
        for ref in (
            self.canary_manifest_ref,
            self.annotation_contract_ref,
            self.annotation_guide_ref,
        ):
            _validate_safe_ref(ref)
        if self.batch_export_ref is not None:
            _require_ref_type(self.batch_export_ref, "label-batch-export", "batch_export_ref")
            _validate_safe_ref(self.batch_export_ref)
        if not self.references and not self.observations:
            raise ValueError("calibration input requires references or observations")
        if (
            any(
                observation.source is LabelCalibrationObservationSource.BATCH_EXPORT_ROW
                for observation in self.observations
            )
            and self.batch_export_ref is None
        ):
            raise ValueError("batch export observations require batch_export_ref")
        return self


class LabelCalibrationFindingCount(ContractModel):
    schema_version: Literal["eval-factory/label-calibration-finding-count/r3-06"] = (
        "eval-factory/label-calibration-finding-count/r3-06"
    )
    finding_code: LabelCalibrationFindingCode
    count: int = Field(ge=1)

    @field_validator("finding_code", mode="before")
    @classmethod
    def parse_finding_code(cls, value: object) -> LabelCalibrationFindingCode:
        if isinstance(value, LabelCalibrationFindingCode):
            return value
        if isinstance(value, str):
            return LabelCalibrationFindingCode(value)
        raise TypeError("finding_code must be a LabelCalibrationFindingCode")


class LabelCalibrationComparison(ContractModel):
    schema_version: Literal["eval-factory/label-calibration-comparison/r3-06"] = (
        "eval-factory/label-calibration-comparison/r3-06"
    )
    comparison_id: Identifier
    canary_instance_id: Identifier | None = None
    trace_envelope_ref: ObjectRef
    label_spec_ref: ObjectRef
    reference_ref: ObjectRef | None = None
    decision_ref: ObjectRef | None = None
    outcome: LabelCalibrationOutcome
    finding_codes: frozenset[LabelCalibrationFindingCode] = frozenset()
    expected_decision: LabelDecisionValueV2 | None = None
    observed_decision: LabelDecisionValueV2 | None = None
    expected_positive_evidence_ref_ids: tuple[Identifier, ...] = ()
    observed_positive_evidence_ref_ids: tuple[Identifier, ...] = ()
    expected_negative_evidence_ref_ids: tuple[Identifier, ...] = ()
    observed_negative_evidence_ref_ids: tuple[Identifier, ...] = ()
    expected_semantic_evidence_ref_ids: tuple[Identifier, ...] = ()
    observed_semantic_evidence_ref_ids: tuple[Identifier, ...] = ()
    expected_structured_capability_complete: bool | None = None
    observed_structured_capability_complete: bool | None = None
    expected_semantic_residual_status: LabelCalibrationSemanticResidualStatus | None = None
    observed_semantic_residual_status: LabelCalibrationSemanticResidualStatus | None = None
    expected_confidence: float | None = Field(default=None, ge=0, le=1)
    observed_confidence: float | None = Field(default=None, ge=0, le=1)
    expected_rule_version: str | None = Field(default=None, min_length=1, max_length=128)
    observed_rule_version: str | None = Field(default=None, min_length=1, max_length=128)
    expected_model_profile: Identifier | None = None
    observed_model_profile: Identifier | None = None
    expected_prompt_version: str | None = Field(default=None, min_length=1, max_length=128)
    observed_prompt_version: str | None = Field(default=None, min_length=1, max_length=128)
    expected_review_status: LabelCalibrationReviewStatus | None = None
    observed_unresolved_reasons: frozenset[LabelUnresolvedReason] = frozenset()
    comparison_sha256: Sha256

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> LabelCalibrationOutcome:
        if isinstance(value, LabelCalibrationOutcome):
            return value
        if isinstance(value, str):
            return LabelCalibrationOutcome(value)
        raise TypeError("outcome must be a LabelCalibrationOutcome")

    @field_validator("finding_codes", mode="before")
    @classmethod
    def parse_finding_codes(cls, value: object) -> frozenset[LabelCalibrationFindingCode]:
        if isinstance(value, (frozenset, set, tuple, list)):
            return frozenset(
                item if isinstance(item, LabelCalibrationFindingCode) else LabelCalibrationFindingCode(item)
                for item in value
            )
        raise TypeError("finding_codes must be a collection")

    @model_validator(mode="after")
    def validate_comparison(self) -> LabelCalibrationComparison:
        _require_ref_type(self.trace_envelope_ref, "trace-envelope", "trace_envelope_ref")
        _require_ref_type(self.label_spec_ref, "label-spec", "label_spec_ref")
        if self.reference_ref is not None:
            _require_ref_type(self.reference_ref, "label-annotation", "reference_ref")
        if self.decision_ref is not None:
            _require_ref_type(self.decision_ref, "label-decision", "decision_ref")
        if self.outcome in {
            LabelCalibrationOutcome.EXACT_MATCH,
            LabelCalibrationOutcome.APPROVED_ABSTAIN,
        }:
            if self.finding_codes:
                raise ValueError("non-blocking calibration outcome cannot carry findings")
            if self.reference_ref is None or self.decision_ref is None:
                raise ValueError("matched calibration outcome requires reference and decision")
        if self.outcome is LabelCalibrationOutcome.BLOCKING_FINDING and not self.finding_codes:
            raise ValueError("blocking calibration outcome requires findings")
        if self.outcome is LabelCalibrationOutcome.MISSING_REFERENCE and (
            self.reference_ref is not None
            or LabelCalibrationFindingCode.MISSING_REFERENCE not in self.finding_codes
        ):
            raise ValueError("missing-reference outcome requires missing reference finding")
        if self.outcome is LabelCalibrationOutcome.MISSING_DECISION and (
            self.decision_ref is not None
            or LabelCalibrationFindingCode.MISSING_DECISION not in self.finding_codes
        ):
            raise ValueError("missing-decision outcome requires missing decision finding")
        return self


class LabelCalibrationSummary(ContractModel):
    schema_version: Literal["eval-factory/label-calibration-summary/r3-06"] = (
        "eval-factory/label-calibration-summary/r3-06"
    )
    total: int = Field(ge=0)
    exact_matches: int = Field(ge=0)
    approved_abstentions: int = Field(ge=0)
    blocking_findings: int = Field(ge=0)
    missing_references: int = Field(ge=0)
    missing_decisions: int = Field(ge=0)
    finding_counts: tuple[LabelCalibrationFindingCount, ...] = ()

    @model_validator(mode="after")
    def validate_summary(self) -> LabelCalibrationSummary:
        counted = (
            self.exact_matches
            + self.approved_abstentions
            + self.blocking_findings
            + self.missing_references
            + self.missing_decisions
        )
        if counted != self.total:
            raise ValueError("calibration summary counts must equal total")
        codes = [item.finding_code for item in self.finding_counts]
        if len(codes) != len(set(codes)):
            raise ValueError("calibration finding counts must be unique")
        return self


class LabelCalibrationReport(ContractModel):
    schema_version: Literal["eval-factory/label-calibration-report/r3-06"] = (
        "eval-factory/label-calibration-report/r3-06"
    )
    label_calibration_report_id: Identifier
    claim_scope: Literal[LabelCalibrationClaimScope.DEVELOPMENT_CANARY_ONLY] = (
        LabelCalibrationClaimScope.DEVELOPMENT_CANARY_ONLY
    )
    canary_manifest_ref: ObjectRef
    annotation_contract_ref: ObjectRef
    annotation_guide_ref: ObjectRef
    batch_export_ref: ObjectRef | None = None
    compared_label_spec_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    comparisons: tuple[LabelCalibrationComparison, ...] = Field(min_length=1)
    summary: LabelCalibrationSummary
    policy_version: Literal["label-calibration/r3-06-v1"] = LABEL_CALIBRATION_POLICY_VERSION
    report_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_report(self) -> LabelCalibrationReport:
        if self.summary.total != len(self.comparisons):
            raise ValueError("calibration report summary must match comparisons")
        comparison_ids = [comparison.comparison_id for comparison in self.comparisons]
        if len(comparison_ids) != len(set(comparison_ids)):
            raise ValueError("calibration comparison IDs must be unique")
        return self


class LabelCalibrationRunner:
    policy_version = LABEL_CALIBRATION_POLICY_VERSION

    def compare(self, calibration_input: LabelCalibrationInput) -> LabelCalibrationReport:
        references = _reference_index(calibration_input.references)
        observations = _observation_index(calibration_input.observations)
        pair_keys = sorted(set(references) | set(observations))
        comparisons = tuple(
            _compare_pair(
                reference=references.get(pair_key),
                observation=observations.get(pair_key),
            )
            for pair_key in pair_keys
        )
        if not comparisons:
            raise LabelCalibrationPolicyError("calibration input produced no comparisons")
        summary = _summary(comparisons)
        label_refs = _unique_label_refs(comparisons)
        seed = {
            "claim_scope": LabelCalibrationClaimScope.DEVELOPMENT_CANARY_ONLY.value,
            "canary_manifest_ref": _ref_payload(calibration_input.canary_manifest_ref),
            "annotation_contract_ref": _ref_payload(calibration_input.annotation_contract_ref),
            "annotation_guide_ref": _ref_payload(calibration_input.annotation_guide_ref),
            "batch_export_ref": (
                None
                if calibration_input.batch_export_ref is None
                else _ref_payload(calibration_input.batch_export_ref)
            ),
            "compared_label_spec_refs": [_ref_payload(ref) for ref in label_refs],
            "comparisons": [_comparison_payload(comparison) for comparison in comparisons],
            "summary": _summary_payload(summary),
            "policy_version": LABEL_CALIBRATION_POLICY_VERSION,
        }
        return LabelCalibrationReport(
            label_calibration_report_id=_stable_id("label-calibration-report", seed),
            canary_manifest_ref=calibration_input.canary_manifest_ref,
            annotation_contract_ref=calibration_input.annotation_contract_ref,
            annotation_guide_ref=calibration_input.annotation_guide_ref,
            batch_export_ref=calibration_input.batch_export_ref,
            compared_label_spec_refs=label_refs,
            comparisons=comparisons,
            summary=summary,
            report_sha256=_stable_hash(seed),
            audit=calibration_input.audit,
        )


def _compare_pair(
    *,
    reference: LabelCalibrationReference | None,
    observation: LabelCalibrationObservation | None,
) -> LabelCalibrationComparison:
    if reference is None and observation is None:
        raise LabelCalibrationPolicyError("calibration pair cannot be empty")
    if reference is None:
        assert observation is not None
        return _comparison(
            reference=None,
            observation=observation,
            outcome=LabelCalibrationOutcome.MISSING_REFERENCE,
            findings=frozenset({LabelCalibrationFindingCode.MISSING_REFERENCE}),
        )
    if observation is None:
        return _comparison(
            reference=reference,
            observation=None,
            outcome=LabelCalibrationOutcome.MISSING_DECISION,
            findings=frozenset({LabelCalibrationFindingCode.MISSING_DECISION}),
        )
    findings = _comparison_findings(reference, observation)
    approved_abstain = _is_approved_abstention(reference, observation)
    if not findings:
        outcome = (
            LabelCalibrationOutcome.APPROVED_ABSTAIN
            if approved_abstain
            else LabelCalibrationOutcome.EXACT_MATCH
        )
    else:
        outcome = LabelCalibrationOutcome.BLOCKING_FINDING
    return _comparison(
        reference=reference,
        observation=observation,
        outcome=outcome,
        findings=findings,
    )


def _comparison_findings(
    reference: LabelCalibrationReference,
    observation: LabelCalibrationObservation,
) -> frozenset[LabelCalibrationFindingCode]:
    findings: set[LabelCalibrationFindingCode] = set()
    if reference.review_status is not LabelCalibrationReviewStatus.APPROVED:
        findings.add(LabelCalibrationFindingCode.REFERENCE_NOT_APPROVED)
    if reference.expected_decision is not observation.decision:
        findings.add(LabelCalibrationFindingCode.DECISION_MISMATCH)
    if (
        reference.positive_evidence_ref_ids != observation.positive_evidence_ref_ids
        or reference.negative_evidence_ref_ids != observation.negative_evidence_ref_ids
        or reference.semantic_evidence_ref_ids != observation.semantic_evidence_ref_ids
    ):
        findings.add(LabelCalibrationFindingCode.EVIDENCE_MISMATCH)
    if reference.structured_capability_complete != observation.structured_capability_complete:
        findings.add(LabelCalibrationFindingCode.STRUCTURED_CAPABILITY_MISMATCH)
    if reference.confidence != observation.confidence:
        findings.add(LabelCalibrationFindingCode.CONFIDENCE_MISMATCH)
    if reference.rule_version != observation.rule_version:
        findings.add(LabelCalibrationFindingCode.RULE_VERSION_MISMATCH)
    observed_semantic_status = _observed_semantic_status(observation)
    if reference.semantic_residual_status is not observed_semantic_status:
        findings.add(LabelCalibrationFindingCode.UNEXPECTED_SEMANTIC_EVALUATION)
    if observation.semantic_evidence_ref_ids and (
        observation.model_profile is None or observation.prompt_version is None
    ):
        findings.add(LabelCalibrationFindingCode.MISSING_SEMANTIC_METADATA)
    if reference.semantic_residual_status is LabelCalibrationSemanticResidualStatus.REQUIRED:
        if observation.model_profile is None or observation.prompt_version is None:
            findings.add(LabelCalibrationFindingCode.MISSING_SEMANTIC_METADATA)
        elif (
            reference.model_profile != observation.model_profile
            or reference.prompt_version != observation.prompt_version
        ):
            findings.add(LabelCalibrationFindingCode.SEMANTIC_METADATA_MISMATCH)
    if observation.execution_status is not LabelExecutionStatus.FINAL and not _is_approved_abstention(
        reference, observation
    ):
        findings.add(LabelCalibrationFindingCode.UNRESOLVED_BLOCKING_OUTCOME)
    if observation.decision is LabelDecisionValueV2.ABSTAIN and not observation.unresolved_reasons:
        findings.add(LabelCalibrationFindingCode.UNRESOLVED_BLOCKING_OUTCOME)
    return frozenset(findings)


def _is_approved_abstention(
    reference: LabelCalibrationReference,
    observation: LabelCalibrationObservation,
) -> bool:
    return bool(
        reference.expected_decision is LabelDecisionValueV2.ABSTAIN
        and observation.decision is LabelDecisionValueV2.ABSTAIN
        and reference.review_status is LabelCalibrationReviewStatus.APPROVED
        and observation.unresolved_reasons
    )


def _observed_semantic_status(
    observation: LabelCalibrationObservation,
) -> LabelCalibrationSemanticResidualStatus:
    if (
        observation.semantic_evidence_ref_ids
        or observation.model_profile is not None
        or observation.prompt_version is not None
    ):
        return LabelCalibrationSemanticResidualStatus.REQUIRED
    if LabelUnresolvedReason.MODEL_UNAVAILABLE in observation.unresolved_reasons:
        return LabelCalibrationSemanticResidualStatus.BLOCKED
    return LabelCalibrationSemanticResidualStatus.NOT_REQUIRED


def _comparison(
    *,
    reference: LabelCalibrationReference | None,
    observation: LabelCalibrationObservation | None,
    outcome: LabelCalibrationOutcome,
    findings: frozenset[LabelCalibrationFindingCode],
) -> LabelCalibrationComparison:
    subject = reference if reference is not None else observation
    assert subject is not None
    seed = {
        "reference": None if reference is None else _reference_payload(reference),
        "observation": None if observation is None else _observation_payload(observation),
        "outcome": outcome.value,
        "finding_codes": sorted(code.value for code in findings),
        "policy_version": LABEL_CALIBRATION_POLICY_VERSION,
    }
    return LabelCalibrationComparison(
        comparison_id=_stable_id("label-calibration-comparison", seed),
        canary_instance_id=None if reference is None else reference.canary_instance_id,
        trace_envelope_ref=subject.trace_envelope_ref,
        label_spec_ref=subject.label_spec_ref,
        reference_ref=None if reference is None else reference.annotation_ref,
        decision_ref=None if observation is None else observation.decision_ref,
        outcome=outcome,
        finding_codes=findings,
        expected_decision=None if reference is None else reference.expected_decision,
        observed_decision=None if observation is None else observation.decision,
        expected_positive_evidence_ref_ids=(() if reference is None else reference.positive_evidence_ref_ids),
        observed_positive_evidence_ref_ids=(
            () if observation is None else observation.positive_evidence_ref_ids
        ),
        expected_negative_evidence_ref_ids=(() if reference is None else reference.negative_evidence_ref_ids),
        observed_negative_evidence_ref_ids=(
            () if observation is None else observation.negative_evidence_ref_ids
        ),
        expected_semantic_evidence_ref_ids=(() if reference is None else reference.semantic_evidence_ref_ids),
        observed_semantic_evidence_ref_ids=(
            () if observation is None else observation.semantic_evidence_ref_ids
        ),
        expected_structured_capability_complete=(
            None if reference is None else reference.structured_capability_complete
        ),
        observed_structured_capability_complete=(
            None if observation is None else observation.structured_capability_complete
        ),
        expected_semantic_residual_status=(None if reference is None else reference.semantic_residual_status),
        observed_semantic_residual_status=(
            None if observation is None else _observed_semantic_status(observation)
        ),
        expected_confidence=None if reference is None else reference.confidence,
        observed_confidence=None if observation is None else observation.confidence,
        expected_rule_version=None if reference is None else reference.rule_version,
        observed_rule_version=None if observation is None else observation.rule_version,
        expected_model_profile=None if reference is None else reference.model_profile,
        observed_model_profile=None if observation is None else observation.model_profile,
        expected_prompt_version=None if reference is None else reference.prompt_version,
        observed_prompt_version=None if observation is None else observation.prompt_version,
        expected_review_status=None if reference is None else reference.review_status,
        observed_unresolved_reasons=(frozenset() if observation is None else observation.unresolved_reasons),
        comparison_sha256=_stable_hash(seed),
    )


def _summary(comparisons: tuple[LabelCalibrationComparison, ...]) -> LabelCalibrationSummary:
    findings = Counter(code for comparison in comparisons for code in comparison.finding_codes)
    return LabelCalibrationSummary(
        total=len(comparisons),
        exact_matches=sum(
            comparison.outcome is LabelCalibrationOutcome.EXACT_MATCH for comparison in comparisons
        ),
        approved_abstentions=sum(
            comparison.outcome is LabelCalibrationOutcome.APPROVED_ABSTAIN for comparison in comparisons
        ),
        blocking_findings=sum(
            comparison.outcome is LabelCalibrationOutcome.BLOCKING_FINDING for comparison in comparisons
        ),
        missing_references=sum(
            comparison.outcome is LabelCalibrationOutcome.MISSING_REFERENCE for comparison in comparisons
        ),
        missing_decisions=sum(
            comparison.outcome is LabelCalibrationOutcome.MISSING_DECISION for comparison in comparisons
        ),
        finding_counts=tuple(
            LabelCalibrationFindingCount(finding_code=code, count=count)
            for code, count in sorted(findings.items(), key=lambda item: item[0].value)
        ),
    )


def _reference_index(
    references: tuple[LabelCalibrationReference, ...],
) -> dict[tuple[str, ...], LabelCalibrationReference]:
    index: dict[tuple[str, ...], LabelCalibrationReference] = {}
    for reference in references:
        key = _pair_key(reference.trace_envelope_ref, reference.label_spec_ref)
        if key in index:
            raise LabelCalibrationPolicyError("duplicate reference trace-label pair")
        index[key] = reference
    return index


def _observation_index(
    observations: tuple[LabelCalibrationObservation, ...],
) -> dict[tuple[str, ...], LabelCalibrationObservation]:
    index: dict[tuple[str, ...], LabelCalibrationObservation] = {}
    for observation in observations:
        key = _pair_key(observation.trace_envelope_ref, observation.label_spec_ref)
        if key in index:
            raise LabelCalibrationPolicyError("duplicate decision trace-label pair")
        index[key] = observation
    return index


def _unique_label_refs(
    comparisons: tuple[LabelCalibrationComparison, ...],
) -> tuple[ObjectRef, ...]:
    refs = {_ref_key(comparison.label_spec_ref): comparison.label_spec_ref for comparison in comparisons}
    return tuple(refs[key] for key in sorted(refs))


def _pair_key(trace_ref: ObjectRef, label_ref: ObjectRef) -> tuple[str, ...]:
    return (*_ref_key(trace_ref), *_ref_key(label_ref))


def _reference_payload(reference: LabelCalibrationReference) -> dict[str, object]:
    return {
        "annotation_ref": _ref_payload(reference.annotation_ref),
        "canary_instance_id": reference.canary_instance_id,
        "trace_envelope_ref": _ref_payload(reference.trace_envelope_ref),
        "label_spec_ref": _ref_payload(reference.label_spec_ref),
        "expected_decision": reference.expected_decision.value,
        "positive_evidence_ref_ids": list(reference.positive_evidence_ref_ids),
        "negative_evidence_ref_ids": list(reference.negative_evidence_ref_ids),
        "semantic_evidence_ref_ids": list(reference.semantic_evidence_ref_ids),
        "structured_capability_complete": reference.structured_capability_complete,
        "semantic_residual_status": reference.semantic_residual_status.value,
        "confidence": reference.confidence,
        "rule_version": reference.rule_version,
        "model_profile": reference.model_profile,
        "prompt_version": reference.prompt_version,
        "review_status": reference.review_status.value,
        "policy_version": reference.policy_version,
    }


def _observation_payload(observation: LabelCalibrationObservation) -> dict[str, object]:
    return {
        "source": observation.source.value,
        "source_row_id": observation.source_row_id,
        "decision_ref": _ref_payload(observation.decision_ref),
        "trace_envelope_ref": _ref_payload(observation.trace_envelope_ref),
        "label_spec_ref": _ref_payload(observation.label_spec_ref),
        "decision": observation.decision.value,
        "execution_status": observation.execution_status.value,
        "positive_evidence_ref_ids": list(observation.positive_evidence_ref_ids),
        "negative_evidence_ref_ids": list(observation.negative_evidence_ref_ids),
        "semantic_evidence_ref_ids": list(observation.semantic_evidence_ref_ids),
        "structured_capability_complete": observation.structured_capability_complete,
        "confidence": observation.confidence,
        "rule_version": observation.rule_version,
        "model_profile": observation.model_profile,
        "prompt_version": observation.prompt_version,
        "unresolved_reasons": sorted(reason.value for reason in observation.unresolved_reasons),
        "decision_policy_version": observation.decision_policy_version,
    }


def _comparison_payload(comparison: LabelCalibrationComparison) -> dict[str, object]:
    return comparison.model_dump(mode="json", exclude={"schema_version"}, exclude_none=False)


def _summary_payload(summary: LabelCalibrationSummary) -> dict[str, object]:
    return summary.model_dump(mode="json", exclude={"schema_version"}, exclude_none=False)


def _evidence_ids(evidence: tuple[EvidenceRef, ...]) -> tuple[str, ...]:
    return tuple(sorted(item.evidence_ref_id for item in evidence))


def _row_ref(row: Mapping[str, object], prefix: str) -> ObjectRef:
    return ObjectRef(
        object_type=_require_str(row, f"{prefix}_object_type"),
        object_id=_require_str(row, f"{prefix}_object_id"),
        object_version=_require_str(row, f"{prefix}_object_version"),
        object_sha256=_require_str(row, f"{prefix}_sha256"),
    )


def _require_str(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise LabelCalibrationPolicyError(f"{field} must be a string")
    return value


def _optional_str(row: Mapping[str, object], field: str) -> str | None:
    value = row.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise LabelCalibrationPolicyError(f"{field} must be a string or null")
    return value


def _require_bool(row: Mapping[str, object], field: str) -> bool:
    value = row.get(field)
    if not isinstance(value, bool):
        raise LabelCalibrationPolicyError(f"{field} must be a boolean")
    return value


def _require_float(row: Mapping[str, object], field: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabelCalibrationPolicyError(f"{field} must be numeric")
    return float(value)


def _require_str_sequence(row: Mapping[str, object], field: str) -> tuple[str, ...]:
    value = row.get(field)
    if not isinstance(value, (tuple, list)):
        raise LabelCalibrationPolicyError(f"{field} must be a string sequence")
    if any(not isinstance(item, str) for item in value):
        raise LabelCalibrationPolicyError(f"{field} must contain only strings")
    return tuple(sorted(value))


def _sorted_identifiers(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list, set, frozenset)):
        raise TypeError(f"{field_name} must be a collection")
    if any(not isinstance(item, str) for item in value):
        raise TypeError(f"{field_name} must contain only strings")
    items = tuple(sorted(value))
    if len(items) != len(set(items)):
        raise ValueError(f"{field_name} must be unique")
    return items


def _require_ref_type(ref: ObjectRef, expected: str, field_name: str) -> None:
    if ref.object_type != expected:
        raise ValueError(f"{field_name} must reference {expected}")


def _validate_safe_ref(ref: ObjectRef) -> None:
    _validate_safe_identifier(ref.object_type)
    _validate_safe_identifier(ref.object_id)


def _validate_safe_identifier(value: str) -> None:
    normalized = value.casefold().replace("_", "-")
    if any(marker in normalized for marker in _DENIED_REFERENCE_MARKERS):
        raise ValueError("unsafe calibration reference")


def _ref_payload(ref: ObjectRef) -> dict[str, str]:
    return {
        "object_type": ref.object_type,
        "object_id": ref.object_id,
        "object_version": ref.object_version,
        "object_sha256": ref.object_sha256,
    }


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (ref.object_type, ref.object_id, ref.object_version, ref.object_sha256)


def _stable_id(kind: str, payload: object) -> str:
    return f"{kind}://sha256/{_stable_hash(payload)}"


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
