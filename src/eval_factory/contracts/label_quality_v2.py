from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from eval_factory.contracts.core import ContractAudit, Identifier, ObjectRef, Sha256
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.contracts.labeling_v2 import (
    LabelDecisionValueV2,
    LabelUnresolvedReason,
)
from eval_factory.contracts.statistics_v2 import LabelTestSetBalanceStatusV2

LABEL_QUALITY_POLICY_VERSION: Literal["label-quality-evaluation/r8-02-v1"] = (
    "label-quality-evaluation/r8-02-v1"
)
LABEL_QUALITY_BOOTSTRAP_REPLICATES: Literal[10000] = 10_000
LABEL_QUALITY_REPOSITORY_PENDING_SHA256: Literal[
    "428d9d3b16eec3a31014df9e9cf17fea8a84d3ebb81c5050cd9da67f243a1a17"
] = "428d9d3b16eec3a31014df9e9cf17fea8a84d3ebb81c5050cd9da67f243a1a17"

_DECISIONS = tuple(LabelDecisionValueV2)
_CLASS_METRICS = (
    "PRECISION",
    "RECALL",
    "F1",
)
_ALLOWED_ABSTAIN_REASONS = (
    LabelUnresolvedReason.AMBIGUOUS_EVIDENCE,
    LabelUnresolvedReason.CONFLICTING_EVIDENCE,
    LabelUnresolvedReason.LOW_CONFIDENCE,
    LabelUnresolvedReason.USER_INSPECTION_REQUIRED,
)


class LabelQualityEvidenceClassV2(StrEnum):
    PRODUCTION_INDEPENDENT_TEST = "PRODUCTION_INDEPENDENT_TEST"
    MECHANISM_VALIDATION_ONLY = "MECHANISM_VALIDATION_ONLY"
    REPOSITORY_PENDING_ONLY = "REPOSITORY_PENDING_ONLY"


class LabelQualityOutcomeV2(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    STATISTICAL_GATE_PENDING = "STATISTICAL_GATE_PENDING"


class LabelQualityLabelOutcomeV2(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    STATISTICAL_GATE_PENDING = "STATISTICAL_GATE_PENDING"


class LabelQualityPrerequisiteOutcomeV2(StrEnum):
    FROZEN = "FROZEN"
    STATISTICAL_GATE_PENDING = "STATISTICAL_GATE_PENDING"


class LabelQualityMetricAvailabilityV2(StrEnum):
    MEASURED = "MEASURED"
    UNDEFINED = "UNDEFINED"
    NOT_COMPUTED = "NOT_COMPUTED"


class LabelQualityMetricNameV2(StrEnum):
    PRECISION = "PRECISION"
    RECALL = "RECALL"
    F1 = "F1"
    MACRO_F1 = "MACRO_F1"


class LabelQualityReasonCodeV2(StrEnum):
    NONE = "NONE"
    FROZEN_DATASET_UNAVAILABLE = "FROZEN_DATASET_UNAVAILABLE"
    SAMPLE_SHORTAGE = "SAMPLE_SHORTAGE"
    OBSERVATION_MISSING = "OBSERVATION_MISSING"
    OBSERVATION_INCOMPLETE = "OBSERVATION_INCOMPLETE"
    SEMANTIC_EXECUTION_UNAVAILABLE = "SEMANTIC_EXECUTION_UNAVAILABLE"
    METRIC_UNDEFINED = "METRIC_UNDEFINED"
    STRUCTURED_THRESHOLD_NOT_MET = "STRUCTURED_THRESHOLD_NOT_MET"
    SEMANTIC_THRESHOLD_NOT_MET = "SEMANTIC_THRESHOLD_NOT_MET"


class LabelQualityIntervalMethodV2(StrEnum):
    WILSON_SCORE_95_V1 = "WILSON_SCORE_95_V1"
    STRATIFIED_SHA256_BOOTSTRAP_95_V1 = "STRATIFIED_SHA256_BOOTSTRAP_95_V1"


class LabelQualityEvaluationPolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/label-quality-evaluation-policy/v2"] = (
        "eval-factory/label-quality-evaluation-policy/v2"
    )
    policy_id: Identifier
    test_set_policy_ref: ObjectRef
    access_policy_ref: ObjectRef
    structured_label_spec_refs: tuple[ObjectRef, ObjectRef]
    semantic_label_spec_ref: ObjectRef
    repository_pending_evidence_ref: ObjectRef
    evidence_class: LabelQualityEvidenceClassV2
    structured_precision_threshold_basis_points: Literal[9900] = 9_900
    structured_recall_threshold_basis_points: Literal[9900] = 9_900
    semantic_macro_f1_threshold_basis_points: Literal[8500] = 8_500
    bootstrap_replicate_count: Literal[10000] = LABEL_QUALITY_BOOTSTRAP_REPLICATES
    structured_rule_version: Literal["structured-labeling/r3-02-v1"] = "structured-labeling/r3-02-v1"
    decision_policy_version: Literal["label-decision-merge/r3-04-v1"] = "label-decision-merge/r3-04-v1"
    allowed_semantic_abstain_reasons: tuple[LabelUnresolvedReason, ...] = _ALLOWED_ABSTAIN_REASONS
    max_members: int = Field(ge=300, le=10_000_000)
    max_observations: int = Field(ge=300, le=10_000_000)
    max_private_bytes: int = Field(ge=2, le=10**12)
    max_report_bytes: int = Field(ge=2, le=100_000_000)
    policy_version: Literal["label-quality-evaluation/r8-02-v1"] = LABEL_QUALITY_POLICY_VERSION
    policy_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        _require_ref(
            self.test_set_policy_ref,
            "independent-label-test-set-policy",
            "v2",
            "test_set_policy_ref",
        )
        _require_ref(
            self.access_policy_ref,
            "independent-label-test-set-access-policy",
            "v2",
            "access_policy_ref",
        )
        _require_sorted_unique_refs(
            self.structured_label_spec_refs,
            "structured_label_spec_refs",
        )
        for ref in (*self.structured_label_spec_refs, self.semantic_label_spec_ref):
            _require_ref(ref, "label-spec", "v2", "label spec")
        if self.semantic_label_spec_ref in self.structured_label_spec_refs:
            raise ValueError("semantic label must differ from structured labels")
        _require_ref(
            self.repository_pending_evidence_ref,
            "label-quality-repository-evidence",
            "json/v1",
            "repository_pending_evidence_ref",
        )
        if self.repository_pending_evidence_ref.object_sha256 != LABEL_QUALITY_REPOSITORY_PENDING_SHA256:
            raise ValueError("repository pending evidence is not approved")
        if self.allowed_semantic_abstain_reasons != _ALLOWED_ABSTAIN_REASONS:
            raise ValueError("semantic abstain reasons are not canonical")
        refs = (
            self.test_set_policy_ref,
            self.access_policy_ref,
            *self.structured_label_spec_refs,
            self.semantic_label_spec_ref,
            self.repository_pending_evidence_ref,
        )
        _require_audit(self.audit, refs, "label quality policy")
        _validate_identity(
            self.policy_id,
            self.policy_sha256,
            "label-quality-evaluation-policy",
            label_quality_evaluation_policy_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        test_set_policy_ref: ObjectRef,
        access_policy_ref: ObjectRef,
        structured_label_spec_refs: tuple[ObjectRef, ObjectRef],
        semantic_label_spec_ref: ObjectRef,
        repository_pending_evidence_ref: ObjectRef,
        evidence_class: LabelQualityEvidenceClassV2,
        max_members: int,
        max_observations: int,
        max_private_bytes: int,
        max_report_bytes: int,
        audit: ContractAudit,
    ) -> LabelQualityEvaluationPolicyV2:
        ordered_structured = sorted(structured_label_spec_refs, key=_ref_key)
        structured = (ordered_structured[0], ordered_structured[1])
        refs = (
            test_set_policy_ref,
            access_policy_ref,
            *structured,
            semantic_label_spec_ref,
            repository_pending_evidence_ref,
        )
        value = cls(
            policy_id="label-quality-evaluation-policy://pending",
            test_set_policy_ref=test_set_policy_ref,
            access_policy_ref=access_policy_ref,
            structured_label_spec_refs=structured,
            semantic_label_spec_ref=semantic_label_spec_ref,
            repository_pending_evidence_ref=repository_pending_evidence_ref,
            evidence_class=evidence_class,
            max_members=max_members,
            max_observations=max_observations,
            max_private_bytes=max_private_bytes,
            max_report_bytes=max_report_bytes,
            policy_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "policy_id",
            "policy_sha256",
            "label-quality-evaluation-policy",
            label_quality_evaluation_policy_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return label_quality_evaluation_policy_v2_ref(self)


class LabelQualityPrerequisiteSummaryV2(ContractModelV2):
    schema_version: Literal["eval-factory/label-quality-prerequisite-summary/v2"] = (
        "eval-factory/label-quality-prerequisite-summary/v2"
    )
    outcome: LabelQualityPrerequisiteOutcomeV2
    freeze_result_ref: ObjectRef | None = None
    repository_evidence_ref: ObjectRef | None = None
    dataset_manifest_ref: ObjectRef | None = None
    access_receipt_ref: ObjectRef | None = None
    dataset_version: str | None = Field(default=None, min_length=1, max_length=128)
    selected_member_count: int = Field(ge=0, le=10_000_000)
    available_semantic_count: int = Field(ge=0, le=10_000_000)
    required_semantic_count: Literal[100] = 100
    shortage_count: int = Field(ge=0, le=100)
    material_verified: bool

    @model_validator(mode="after")
    def validate_prerequisite(self) -> Self:
        if self.freeze_result_ref is not None:
            _require_ref(
                self.freeze_result_ref,
                "independent-label-test-set-freeze-result",
                "v2",
                "freeze_result_ref",
            )
        if self.repository_evidence_ref is not None:
            _require_ref(
                self.repository_evidence_ref,
                "label-quality-repository-evidence",
                "json/v1",
                "repository_evidence_ref",
            )
        if self.outcome is LabelQualityPrerequisiteOutcomeV2.FROZEN:
            if (
                self.freeze_result_ref is None
                or self.repository_evidence_ref is not None
                or self.dataset_manifest_ref is None
                or self.access_receipt_ref is None
                or self.dataset_version is None
                or self.selected_member_count < 300
                or self.available_semantic_count < self.required_semantic_count
                or self.shortage_count != 0
                or not self.material_verified
            ):
                raise ValueError("frozen prerequisite evidence is incomplete")
            _require_ref(
                self.dataset_manifest_ref,
                "independent-label-test-set-manifest",
                "v2",
                "dataset_manifest_ref",
            )
            _require_ref(
                self.access_receipt_ref,
                "independent-label-test-set-access-receipt",
                "v2",
                "access_receipt_ref",
            )
        elif (
            (self.freeze_result_ref is None) == (self.repository_evidence_ref is None)
            or self.dataset_manifest_ref is not None
            or self.access_receipt_ref is not None
            or self.dataset_version is not None
            or self.selected_member_count != 0
            or self.shortage_count < 1
            or self.material_verified
        ):
            raise ValueError("pending prerequisite cannot carry dataset authority")
        return self


class LabelQualityClassMetricV2(ContractModelV2):
    schema_version: Literal["eval-factory/label-quality-class-metric/v2"] = (
        "eval-factory/label-quality-class-metric/v2"
    )
    metric: LabelQualityMetricNameV2
    decision: LabelDecisionValueV2 | None = None
    numerator: int | None = Field(default=None, ge=0, le=10_000_000)
    denominator: int = Field(ge=0, le=10_000_000)
    value_basis_points: int | None = Field(default=None, ge=0, le=10_000)
    availability: LabelQualityMetricAvailabilityV2

    @model_validator(mode="after")
    def validate_metric(self) -> Self:
        if self.metric is LabelQualityMetricNameV2.MACRO_F1:
            if self.decision is not None:
                raise ValueError("macro-F1 cannot bind one decision class")
        elif self.decision is None:
            raise ValueError("class metric requires a decision class")
        if self.availability is LabelQualityMetricAvailabilityV2.MEASURED:
            if (
                self.numerator is None
                or self.denominator <= 0
                or self.numerator > self.denominator
                or self.value_basis_points != _round_basis_points(self.numerator, self.denominator)
            ):
                raise ValueError("measured metric differs from exact fraction")
        elif self.numerator is not None or self.denominator != 0 or self.value_basis_points is not None:
            raise ValueError("undefined metric values must remain null")
        return self

    @classmethod
    def measured(
        cls,
        *,
        metric: LabelQualityMetricNameV2,
        decision: LabelDecisionValueV2 | None,
        numerator: int,
        denominator: int,
    ) -> LabelQualityClassMetricV2:
        return cls(
            metric=metric,
            decision=decision,
            numerator=numerator,
            denominator=denominator,
            value_basis_points=_round_basis_points(numerator, denominator),
            availability=LabelQualityMetricAvailabilityV2.MEASURED,
        )

    @classmethod
    def undefined(
        cls,
        *,
        metric: LabelQualityMetricNameV2,
        decision: LabelDecisionValueV2 | None,
    ) -> LabelQualityClassMetricV2:
        return cls(
            metric=metric,
            decision=decision,
            numerator=None,
            denominator=0,
            value_basis_points=None,
            availability=LabelQualityMetricAvailabilityV2.UNDEFINED,
        )

    @classmethod
    def not_computed(
        cls,
        *,
        metric: LabelQualityMetricNameV2,
        decision: LabelDecisionValueV2 | None,
    ) -> LabelQualityClassMetricV2:
        return cls(
            metric=metric,
            decision=decision,
            numerator=None,
            denominator=0,
            value_basis_points=None,
            availability=LabelQualityMetricAvailabilityV2.NOT_COMPUTED,
        )


class LabelQualityConfidenceIntervalV2(ContractModelV2):
    schema_version: Literal["eval-factory/label-quality-confidence-interval/v2"] = (
        "eval-factory/label-quality-confidence-interval/v2"
    )
    method: LabelQualityIntervalMethodV2
    confidence_basis_points: Literal[9500] = 9_500
    lower_basis_points: int | None = Field(default=None, ge=0, le=10_000)
    upper_basis_points: int | None = Field(default=None, ge=0, le=10_000)
    sample_count: int = Field(ge=0, le=10_000_000)
    resample_count: int = Field(ge=0, le=1_000_000)
    algorithm_version: str = Field(min_length=1, max_length=128)
    availability: LabelQualityMetricAvailabilityV2

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        expected_algorithm = {
            LabelQualityIntervalMethodV2.WILSON_SCORE_95_V1: ("wilson-score-95/v1"),
            LabelQualityIntervalMethodV2.STRATIFIED_SHA256_BOOTSTRAP_95_V1: (
                "stratified-sha256-bootstrap-95/v1"
            ),
        }[self.method]
        expected_resamples = (
            LABEL_QUALITY_BOOTSTRAP_REPLICATES
            if (
                self.availability is LabelQualityMetricAvailabilityV2.MEASURED
                and self.method is LabelQualityIntervalMethodV2.STRATIFIED_SHA256_BOOTSTRAP_95_V1
            )
            else 0
        )
        if self.algorithm_version != expected_algorithm or self.resample_count != expected_resamples:
            raise ValueError("confidence interval algorithm is not canonical")
        if self.availability is LabelQualityMetricAvailabilityV2.MEASURED:
            if (
                self.lower_basis_points is None
                or self.upper_basis_points is None
                or self.sample_count <= 0
                or self.lower_basis_points > self.upper_basis_points
            ):
                raise ValueError("measured confidence interval is incomplete")
        elif (
            self.lower_basis_points is not None
            or self.upper_basis_points is not None
            or self.sample_count != 0
        ):
            raise ValueError("unavailable confidence interval must be empty")
        return self


class LabelQualityConfusionRowV2(ContractModelV2):
    schema_version: Literal["eval-factory/label-quality-confusion-row/v2"] = (
        "eval-factory/label-quality-confusion-row/v2"
    )
    reference_class: LabelDecisionValueV2
    predicted_match_count: int = Field(ge=0, le=10_000_000)
    predicted_no_match_count: int = Field(ge=0, le=10_000_000)
    predicted_abstain_count: int = Field(ge=0, le=10_000_000)

    @property
    def support(self) -> int:
        return self.predicted_match_count + self.predicted_no_match_count + self.predicted_abstain_count

    def predicted_count(self, decision: LabelDecisionValueV2) -> int:
        return {
            LabelDecisionValueV2.MATCH: self.predicted_match_count,
            LabelDecisionValueV2.NO_MATCH: self.predicted_no_match_count,
            LabelDecisionValueV2.ABSTAIN: self.predicted_abstain_count,
        }[decision]


class StructuredLabelQualitySummaryV2(ContractModelV2):
    schema_version: Literal["eval-factory/structured-label-quality-summary/v2"] = (
        "eval-factory/structured-label-quality-summary/v2"
    )
    label_spec_ref: ObjectRef
    reference_match_count: int = Field(ge=50, le=10_000_000)
    reference_no_match_count: int = Field(ge=50, le=10_000_000)
    true_positive_count: int = Field(ge=0, le=10_000_000)
    false_positive_count: int = Field(ge=0, le=10_000_000)
    true_negative_count: int = Field(ge=0, le=10_000_000)
    false_negative_count: int = Field(ge=0, le=10_000_000)
    observed_abstain_count: int = Field(ge=0, le=10_000_000)
    incomplete_observation_count: int = Field(ge=0, le=10_000_000)
    missing_observation_count: int = Field(ge=0, le=10_000_000)
    precision: LabelQualityClassMetricV2
    precision_interval: LabelQualityConfidenceIntervalV2
    recall: LabelQualityClassMetricV2
    recall_interval: LabelQualityConfidenceIntervalV2
    precision_threshold_basis_points: Literal[9900] = 9_900
    recall_threshold_basis_points: Literal[9900] = 9_900
    outcome: LabelQualityLabelOutcomeV2
    reason_codes: tuple[LabelQualityReasonCodeV2, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        _require_ref(self.label_spec_ref, "label-spec", "v2", "label_spec_ref")
        reference_total = self.reference_match_count + self.reference_no_match_count
        observed_total = (
            self.true_positive_count
            + self.false_positive_count
            + self.true_negative_count
            + self.false_negative_count
            + self.observed_abstain_count
            + self.incomplete_observation_count
        )
        if observed_total + self.missing_observation_count != reference_total:
            raise ValueError("structured counts do not close over references")
        if self.true_positive_count + self.false_negative_count > self.reference_match_count:
            raise ValueError("structured positive counts exceed references")
        if self.true_negative_count + self.false_positive_count > self.reference_no_match_count:
            raise ValueError("structured negative counts exceed references")
        incomplete = bool(
            self.observed_abstain_count or self.incomplete_observation_count or self.missing_observation_count
        )
        if incomplete:
            _require_not_computed_metric(
                self.precision,
                LabelQualityMetricNameV2.PRECISION,
                LabelDecisionValueV2.MATCH,
            )
            _require_not_computed_metric(
                self.recall,
                LabelQualityMetricNameV2.RECALL,
                LabelDecisionValueV2.MATCH,
            )
        else:
            _require_metric(
                self.precision,
                LabelQualityMetricNameV2.PRECISION,
                LabelDecisionValueV2.MATCH,
                self.true_positive_count,
                self.true_positive_count + self.false_positive_count,
            )
            _require_metric(
                self.recall,
                LabelQualityMetricNameV2.RECALL,
                LabelDecisionValueV2.MATCH,
                self.true_positive_count,
                self.true_positive_count + self.false_negative_count,
            )
        _require_interval_sample(
            self.precision_interval,
            self.precision,
            LabelQualityIntervalMethodV2.WILSON_SCORE_95_V1,
        )
        _require_interval_sample(
            self.recall_interval,
            self.recall,
            LabelQualityIntervalMethodV2.WILSON_SCORE_95_V1,
        )
        expected_outcome, expected_reasons = _structured_outcome(self)
        if self.outcome is not expected_outcome or self.reason_codes != expected_reasons:
            raise ValueError("structured outcome differs from aggregate evidence")
        return self

    @property
    def reference_count(self) -> int:
        return self.reference_match_count + self.reference_no_match_count

    @property
    def observation_count(self) -> int:
        return self.reference_count - self.missing_observation_count


class SemanticLabelQualitySummaryV2(ContractModelV2):
    schema_version: Literal["eval-factory/semantic-label-quality-summary/v2"] = (
        "eval-factory/semantic-label-quality-summary/v2"
    )
    label_spec_ref: ObjectRef
    model_profile: Identifier
    prompt_version: str = Field(min_length=1, max_length=128)
    confusion_rows: tuple[
        LabelQualityConfusionRowV2,
        LabelQualityConfusionRowV2,
        LabelQualityConfusionRowV2,
    ]
    class_metrics: tuple[LabelQualityClassMetricV2, ...]
    supported_class_count: int = Field(ge=1, le=3)
    balance_status: LabelTestSetBalanceStatusV2
    valid_prediction_abstain_count: int = Field(ge=0, le=10_000_000)
    incomplete_observation_count: int = Field(ge=0, le=10_000_000)
    model_unavailable_count: int = Field(ge=0, le=10_000_000)
    missing_observation_count: int = Field(ge=0, le=10_000_000)
    macro_f1: LabelQualityClassMetricV2
    macro_f1_interval: LabelQualityConfidenceIntervalV2
    macro_f1_threshold_basis_points: Literal[8500] = 8_500
    outcome: LabelQualityLabelOutcomeV2
    reason_codes: tuple[LabelQualityReasonCodeV2, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        _require_ref(self.label_spec_ref, "label-spec", "v2", "label_spec_ref")
        if tuple(row.reference_class for row in self.confusion_rows) != _DECISIONS:
            raise ValueError("semantic confusion rows are not canonical")
        supports = {row.reference_class: row.support for row in self.confusion_rows}
        supported = tuple(decision for decision in _DECISIONS if supports[decision] > 0)
        if len(supported) != self.supported_class_count:
            raise ValueError("semantic supported class count differs from matrix")
        expected_keys = tuple(
            (decision, metric)
            for decision in supported
            for metric in (
                LabelQualityMetricNameV2.PRECISION,
                LabelQualityMetricNameV2.RECALL,
                LabelQualityMetricNameV2.F1,
            )
        )
        observed_keys = tuple((metric.decision, metric.metric) for metric in self.class_metrics)
        if observed_keys != expected_keys:
            raise ValueError("semantic class metrics are not canonical")
        predicted_counts = {
            decision: sum(row.predicted_count(decision) for row in self.confusion_rows)
            for decision in _DECISIONS
        }
        diagonal = {
            row.reference_class: row.predicted_count(row.reference_class) for row in self.confusion_rows
        }
        by_key = {(metric.decision, metric.metric): metric for metric in self.class_metrics}
        for decision in supported:
            _require_metric(
                by_key[(decision, LabelQualityMetricNameV2.PRECISION)],
                LabelQualityMetricNameV2.PRECISION,
                decision,
                diagonal[decision],
                predicted_counts[decision],
            )
            _require_metric(
                by_key[(decision, LabelQualityMetricNameV2.RECALL)],
                LabelQualityMetricNameV2.RECALL,
                decision,
                diagonal[decision],
                supports[decision],
            )
            _require_metric(
                by_key[(decision, LabelQualityMetricNameV2.F1)],
                LabelQualityMetricNameV2.F1,
                decision,
                2 * diagonal[decision],
                supports[decision] + predicted_counts[decision],
            )
        reference_total = sum(supports.values()) + self.incomplete_observation_count
        if reference_total + self.missing_observation_count < 100:
            raise ValueError("semantic summary has fewer than 100 references")
        if self.model_unavailable_count > self.incomplete_observation_count:
            raise ValueError("model unavailable count exceeds incomplete observations")
        if self.valid_prediction_abstain_count != predicted_counts[LabelDecisionValueV2.ABSTAIN]:
            raise ValueError("semantic abstain count differs from matrix")
        incomplete = bool(self.incomplete_observation_count or self.missing_observation_count)
        if incomplete:
            _require_not_computed_metric(
                self.macro_f1,
                LabelQualityMetricNameV2.MACRO_F1,
                None,
            )
        else:
            expected_macro_f1 = _semantic_macro_f1(
                supported=supported,
                supports=supports,
                predicted_counts=predicted_counts,
                diagonal=diagonal,
            )
            _require_metric(
                self.macro_f1,
                LabelQualityMetricNameV2.MACRO_F1,
                None,
                expected_macro_f1.numerator,
                expected_macro_f1.denominator,
            )
        complete_reference_count = sum(supports.values())
        _require_interval_sample(
            self.macro_f1_interval,
            self.macro_f1,
            LabelQualityIntervalMethodV2.STRATIFIED_SHA256_BOOTSTRAP_95_V1,
            measured_sample_count=complete_reference_count,
        )
        expected_outcome, expected_reasons = _semantic_outcome(self)
        if self.outcome is not expected_outcome or self.reason_codes != expected_reasons:
            raise ValueError("semantic outcome differs from aggregate evidence")
        return self

    @property
    def reference_count(self) -> int:
        return (
            sum(row.support for row in self.confusion_rows)
            + self.incomplete_observation_count
            + self.missing_observation_count
        )

    @property
    def observation_count(self) -> int:
        return self.reference_count - self.missing_observation_count


class LabelQualityEvaluationReportV2(ContractModelV2):
    schema_version: Literal["eval-factory/label-quality-evaluation-report/v2"] = (
        "eval-factory/label-quality-evaluation-report/v2"
    )
    report_id: Identifier
    policy_ref: ObjectRef
    evidence_class: LabelQualityEvidenceClassV2
    prerequisite: LabelQualityPrerequisiteSummaryV2
    observation_closure_sha256: Sha256 | None = None
    result_closure_sha256: Sha256 | None = None
    structured_summaries: tuple[StructuredLabelQualitySummaryV2, ...] = Field(max_length=2)
    semantic_summaries: tuple[SemanticLabelQualitySummaryV2, ...] = Field(max_length=1)
    total_reference_count: int = Field(ge=0, le=30_000_000)
    total_observation_count: int = Field(ge=0, le=30_000_000)
    missing_observation_count: int = Field(ge=0, le=30_000_000)
    outcome: LabelQualityOutcomeV2
    reason_codes: tuple[LabelQualityReasonCodeV2, ...] = Field(min_length=1)
    satisfies_nfr_008: bool
    satisfies_sc_010: bool
    authorizes_sc_011: Literal[False] = False
    authorizes_sc_012: Literal[False] = False
    authorizes_sc_013: Literal[False] = False
    authorizes_approval: Literal[False] = False
    authorizes_attestation: Literal[False] = False
    authorizes_production_release: Literal[False] = False
    policy_version: Literal["label-quality-evaluation/r8-02-v1"] = LABEL_QUALITY_POLICY_VERSION
    report_sha256: Sha256
    audit: ContractAudit

    @field_validator("structured_summaries")
    @classmethod
    def validate_structured_order(
        cls,
        value: tuple[StructuredLabelQualitySummaryV2, ...],
    ) -> tuple[StructuredLabelQualitySummaryV2, ...]:
        _require_sorted_unique_summary_refs(value, "structured_summaries")
        return value

    @field_validator("semantic_summaries")
    @classmethod
    def validate_semantic_order(
        cls,
        value: tuple[SemanticLabelQualitySummaryV2, ...],
    ) -> tuple[SemanticLabelQualitySummaryV2, ...]:
        _require_sorted_unique_summary_refs(value, "semantic_summaries")
        return value

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        _require_ref(
            self.policy_ref,
            "label-quality-evaluation-policy",
            "v2",
            "policy_ref",
        )
        expected_reference_count = _report_reference_count(
            self.structured_summaries,
            self.semantic_summaries,
        )
        expected_observation_count = _report_observation_count(
            self.structured_summaries,
            self.semantic_summaries,
        )
        expected_missing = _report_missing_count(
            self.structured_summaries,
            self.semantic_summaries,
        )
        if self.prerequisite.outcome is LabelQualityPrerequisiteOutcomeV2.FROZEN:
            if (
                len(self.structured_summaries) != 2
                or len(self.semantic_summaries) != 1
                or self.observation_closure_sha256 is None
                or self.result_closure_sha256 is None
                or expected_reference_count != self.prerequisite.selected_member_count
            ):
                raise ValueError("frozen report closure is incomplete")
        else:
            if (
                self.structured_summaries
                or self.semantic_summaries
                or self.observation_closure_sha256 is not None
                or self.result_closure_sha256 is not None
            ):
                raise ValueError("pending report cannot carry metric closure")
            expected_reference_count = 0
            expected_observation_count = 0
            expected_missing = 0
        if (
            self.total_reference_count != expected_reference_count
            or self.total_observation_count != expected_observation_count
            or self.missing_observation_count != expected_missing
        ):
            raise ValueError("report counts differ from label evidence")
        expected_outcome, expected_reasons = _report_outcome(
            self.prerequisite,
            self.structured_summaries,
            self.semantic_summaries,
        )
        if self.outcome is not expected_outcome or self.reason_codes != expected_reasons:
            raise ValueError("report outcome differs from nested evidence")
        expected_nfr = (
            self.outcome is LabelQualityOutcomeV2.PASSED
            and self.evidence_class is LabelQualityEvidenceClassV2.PRODUCTION_INDEPENDENT_TEST
        )
        expected_sc = expected_nfr
        if self.satisfies_nfr_008 is not expected_nfr or self.satisfies_sc_010 is not expected_sc:
            raise ValueError("report authority differs from outcome and evidence class")
        if self.evidence_class is LabelQualityEvidenceClassV2.REPOSITORY_PENDING_ONLY and (
            self.prerequisite.outcome is not LabelQualityPrerequisiteOutcomeV2.STATISTICAL_GATE_PENDING
            or self.prerequisite.repository_evidence_ref is None
        ):
            raise ValueError("repository-pending evidence cannot carry a frozen result")
        if (
            self.prerequisite.repository_evidence_ref is not None
            and self.evidence_class is not LabelQualityEvidenceClassV2.REPOSITORY_PENDING_ONLY
        ):
            raise ValueError("repository evidence requires repository-pending claim scope")
        refs = (self.policy_ref, *_prerequisite_refs(self.prerequisite))
        _require_audit(self.audit, refs, "label quality report")
        _validate_identity(
            self.report_id,
            self.report_sha256,
            "label-quality-evaluation-report",
            label_quality_evaluation_report_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        policy_ref: ObjectRef,
        evidence_class: LabelQualityEvidenceClassV2,
        prerequisite: LabelQualityPrerequisiteSummaryV2,
        observation_closure_sha256: str | None,
        result_closure_sha256: str | None,
        structured_summaries: tuple[StructuredLabelQualitySummaryV2, ...],
        semantic_summaries: tuple[SemanticLabelQualitySummaryV2, ...],
        audit: ContractAudit,
    ) -> LabelQualityEvaluationReportV2:
        structured = tuple(sorted(structured_summaries, key=lambda value: _ref_key(value.label_spec_ref)))
        semantic = tuple(sorted(semantic_summaries, key=lambda value: _ref_key(value.label_spec_ref)))
        reference_count = _report_reference_count(structured, semantic)
        observation_count = _report_observation_count(structured, semantic)
        missing_count = _report_missing_count(structured, semantic)
        if prerequisite.outcome is LabelQualityPrerequisiteOutcomeV2.STATISTICAL_GATE_PENDING:
            reference_count = 0
            observation_count = 0
            missing_count = 0
        outcome, reasons = _report_outcome(prerequisite, structured, semantic)
        nfr = (
            outcome is LabelQualityOutcomeV2.PASSED
            and evidence_class is LabelQualityEvidenceClassV2.PRODUCTION_INDEPENDENT_TEST
        )
        value = cls(
            report_id="label-quality-evaluation-report://pending",
            policy_ref=policy_ref,
            evidence_class=evidence_class,
            prerequisite=prerequisite,
            observation_closure_sha256=observation_closure_sha256,
            result_closure_sha256=result_closure_sha256,
            structured_summaries=structured,
            semantic_summaries=semantic,
            total_reference_count=reference_count,
            total_observation_count=observation_count,
            missing_observation_count=missing_count,
            outcome=outcome,
            reason_codes=reasons,
            satisfies_nfr_008=nfr,
            satisfies_sc_010=nfr,
            report_sha256="0" * 64,
            audit=_safe_audit(
                audit,
                (policy_ref, *_prerequisite_refs(prerequisite)),
            ),
        )
        return _finalize(
            value,
            "report_id",
            "report_sha256",
            "label-quality-evaluation-report",
            label_quality_evaluation_report_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return label_quality_evaluation_report_v2_ref(self)


def label_quality_evaluation_policy_v2_carried_sha256(
    value: LabelQualityEvaluationPolicyV2,
) -> str:
    return _carried(value, {"policy_id", "policy_sha256", "audit"})


def label_quality_evaluation_report_v2_carried_sha256(
    value: LabelQualityEvaluationReportV2,
) -> str:
    return _carried(value, {"report_id", "report_sha256", "audit"})


def label_quality_evaluation_policy_v2_ref(
    value: LabelQualityEvaluationPolicyV2,
) -> ObjectRef:
    validate_label_quality_evaluation_policy_v2_identity(value)
    return _ref(
        "label-quality-evaluation-policy",
        value.policy_id,
        value.policy_sha256,
    )


def label_quality_evaluation_report_v2_ref(
    value: LabelQualityEvaluationReportV2,
) -> ObjectRef:
    validate_label_quality_evaluation_report_v2_identity(value)
    return _ref(
        "label-quality-evaluation-report",
        value.report_id,
        value.report_sha256,
    )


def validate_label_quality_evaluation_policy_v2_identity(
    value: LabelQualityEvaluationPolicyV2,
) -> None:
    _validate_identity(
        value.policy_id,
        value.policy_sha256,
        "label-quality-evaluation-policy",
        label_quality_evaluation_policy_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_label_quality_evaluation_report_v2_identity(
    value: LabelQualityEvaluationReportV2,
) -> None:
    _validate_identity(
        value.report_id,
        value.report_sha256,
        "label-quality-evaluation-report",
        label_quality_evaluation_report_v2_carried_sha256(value),
        allow_pending=False,
    )


def _structured_outcome(
    value: StructuredLabelQualitySummaryV2,
) -> tuple[LabelQualityLabelOutcomeV2, tuple[LabelQualityReasonCodeV2, ...]]:
    pending: set[LabelQualityReasonCodeV2] = set()
    if value.missing_observation_count:
        pending.add(LabelQualityReasonCodeV2.OBSERVATION_MISSING)
    if value.observed_abstain_count or value.incomplete_observation_count:
        pending.add(LabelQualityReasonCodeV2.OBSERVATION_INCOMPLETE)
    if pending:
        return (
            LabelQualityLabelOutcomeV2.STATISTICAL_GATE_PENDING,
            _ordered_reasons(pending),
        )
    failed: set[LabelQualityReasonCodeV2] = set()
    if (
        value.precision.availability is not LabelQualityMetricAvailabilityV2.MEASURED
        or value.recall.availability is not LabelQualityMetricAvailabilityV2.MEASURED
    ):
        failed.add(LabelQualityReasonCodeV2.METRIC_UNDEFINED)
    if not _metric_at_least(value.precision, value.precision_threshold_basis_points) or not _metric_at_least(
        value.recall, value.recall_threshold_basis_points
    ):
        failed.add(LabelQualityReasonCodeV2.STRUCTURED_THRESHOLD_NOT_MET)
    if failed:
        return LabelQualityLabelOutcomeV2.FAILED, _ordered_reasons(failed)
    return LabelQualityLabelOutcomeV2.PASSED, (LabelQualityReasonCodeV2.NONE,)


def _semantic_outcome(
    value: SemanticLabelQualitySummaryV2,
) -> tuple[LabelQualityLabelOutcomeV2, tuple[LabelQualityReasonCodeV2, ...]]:
    pending: set[LabelQualityReasonCodeV2] = set()
    if value.missing_observation_count:
        pending.add(LabelQualityReasonCodeV2.OBSERVATION_MISSING)
    if value.incomplete_observation_count:
        pending.add(LabelQualityReasonCodeV2.OBSERVATION_INCOMPLETE)
    if value.model_unavailable_count:
        pending.add(LabelQualityReasonCodeV2.SEMANTIC_EXECUTION_UNAVAILABLE)
    if pending:
        return (
            LabelQualityLabelOutcomeV2.STATISTICAL_GATE_PENDING,
            _ordered_reasons(pending),
        )
    failed: set[LabelQualityReasonCodeV2] = set()
    if value.macro_f1.availability is not LabelQualityMetricAvailabilityV2.MEASURED:
        failed.add(LabelQualityReasonCodeV2.METRIC_UNDEFINED)
    if not _metric_at_least(
        value.macro_f1,
        value.macro_f1_threshold_basis_points,
    ):
        failed.add(LabelQualityReasonCodeV2.SEMANTIC_THRESHOLD_NOT_MET)
    if failed:
        return LabelQualityLabelOutcomeV2.FAILED, _ordered_reasons(failed)
    return LabelQualityLabelOutcomeV2.PASSED, (LabelQualityReasonCodeV2.NONE,)


def _report_outcome(
    prerequisite: LabelQualityPrerequisiteSummaryV2,
    structured: tuple[StructuredLabelQualitySummaryV2, ...],
    semantic: tuple[SemanticLabelQualitySummaryV2, ...],
) -> tuple[LabelQualityOutcomeV2, tuple[LabelQualityReasonCodeV2, ...]]:
    if prerequisite.outcome is LabelQualityPrerequisiteOutcomeV2.STATISTICAL_GATE_PENDING:
        reasons = {LabelQualityReasonCodeV2.FROZEN_DATASET_UNAVAILABLE}
        if prerequisite.shortage_count:
            reasons.add(LabelQualityReasonCodeV2.SAMPLE_SHORTAGE)
        return LabelQualityOutcomeV2.STATISTICAL_GATE_PENDING, _ordered_reasons(reasons)
    summaries: tuple[
        StructuredLabelQualitySummaryV2 | SemanticLabelQualitySummaryV2,
        ...,
    ] = (*structured, *semantic)
    if any(summary.outcome is LabelQualityLabelOutcomeV2.STATISTICAL_GATE_PENDING for summary in summaries):
        reasons = {
            reason
            for summary in summaries
            if summary.outcome is LabelQualityLabelOutcomeV2.STATISTICAL_GATE_PENDING
            for reason in summary.reason_codes
        }
        return LabelQualityOutcomeV2.STATISTICAL_GATE_PENDING, _ordered_reasons(reasons)
    if any(summary.outcome is LabelQualityLabelOutcomeV2.FAILED for summary in summaries):
        reasons = {
            reason
            for summary in summaries
            if summary.outcome is LabelQualityLabelOutcomeV2.FAILED
            for reason in summary.reason_codes
        }
        return LabelQualityOutcomeV2.FAILED, _ordered_reasons(reasons)
    return LabelQualityOutcomeV2.PASSED, (LabelQualityReasonCodeV2.NONE,)


def _report_reference_count(
    structured: tuple[StructuredLabelQualitySummaryV2, ...],
    semantic: tuple[SemanticLabelQualitySummaryV2, ...],
) -> int:
    return sum(value.reference_count for value in structured) + sum(
        value.reference_count for value in semantic
    )


def _report_observation_count(
    structured: tuple[StructuredLabelQualitySummaryV2, ...],
    semantic: tuple[SemanticLabelQualitySummaryV2, ...],
) -> int:
    return sum(value.observation_count for value in structured) + sum(
        value.observation_count for value in semantic
    )


def _report_missing_count(
    structured: tuple[StructuredLabelQualitySummaryV2, ...],
    semantic: tuple[SemanticLabelQualitySummaryV2, ...],
) -> int:
    return sum(value.missing_observation_count for value in structured) + sum(
        value.missing_observation_count for value in semantic
    )


def _prerequisite_refs(
    value: LabelQualityPrerequisiteSummaryV2,
) -> tuple[ObjectRef, ...]:
    return tuple(
        ref
        for ref in (
            value.freeze_result_ref,
            value.repository_evidence_ref,
            value.dataset_manifest_ref,
            value.access_receipt_ref,
        )
        if ref is not None
    )


def _require_metric(
    value: LabelQualityClassMetricV2,
    metric: LabelQualityMetricNameV2,
    decision: LabelDecisionValueV2 | None,
    numerator: int,
    denominator: int,
) -> None:
    if value.metric is not metric or value.decision is not decision:
        raise ValueError("metric classification differs from aggregate evidence")
    if denominator == 0:
        if value.availability is not LabelQualityMetricAvailabilityV2.UNDEFINED:
            raise ValueError("zero denominator metric must be undefined")
    elif (
        value.availability is not LabelQualityMetricAvailabilityV2.MEASURED
        or value.numerator != numerator
        or value.denominator != denominator
    ):
        raise ValueError("metric fraction differs from aggregate evidence")


def _require_not_computed_metric(
    value: LabelQualityClassMetricV2,
    metric: LabelQualityMetricNameV2,
    decision: LabelDecisionValueV2 | None,
) -> None:
    if (
        value.metric is not metric
        or value.decision is not decision
        or value.availability is not LabelQualityMetricAvailabilityV2.NOT_COMPUTED
    ):
        raise ValueError("incomplete evidence requires a not-computed metric")


def _require_interval_sample(
    interval: LabelQualityConfidenceIntervalV2,
    metric: LabelQualityClassMetricV2,
    expected_method: LabelQualityIntervalMethodV2,
    *,
    measured_sample_count: int | None = None,
) -> None:
    if interval.method is not expected_method:
        raise ValueError("confidence interval method differs from metric")
    if metric.availability is LabelQualityMetricAvailabilityV2.MEASURED:
        expected_sample_count = metric.denominator if measured_sample_count is None else measured_sample_count
        if (
            interval.availability is not LabelQualityMetricAvailabilityV2.MEASURED
            or interval.sample_count != expected_sample_count
        ):
            raise ValueError("confidence interval sample differs from metric")
    elif interval.availability is LabelQualityMetricAvailabilityV2.MEASURED or interval.sample_count != 0:
        raise ValueError("unavailable metric cannot carry a measured interval")


def _semantic_macro_f1(
    *,
    supported: tuple[LabelDecisionValueV2, ...],
    supports: dict[LabelDecisionValueV2, int],
    predicted_counts: dict[LabelDecisionValueV2, int],
    diagonal: dict[LabelDecisionValueV2, int],
) -> Fraction:
    values = tuple(
        Fraction(
            2 * diagonal[decision],
            supports[decision] + predicted_counts[decision],
        )
        for decision in supported
    )
    if not values:
        raise ValueError("semantic macro-F1 requires a supported class")
    return sum(values, start=Fraction()) / len(values)


def _metric_at_least(
    value: LabelQualityClassMetricV2,
    threshold_basis_points: int,
) -> bool:
    return bool(
        value.availability is LabelQualityMetricAvailabilityV2.MEASURED
        and value.numerator is not None
        and value.numerator * 10_000 >= threshold_basis_points * value.denominator
    )


def _round_basis_points(numerator: int, denominator: int) -> int:
    if denominator <= 0 or numerator < 0 or numerator > denominator:
        raise ValueError("metric fraction is invalid")
    return (2 * numerator * 10_000 + denominator) // (2 * denominator)


def _ordered_reasons(
    values: set[LabelQualityReasonCodeV2],
) -> tuple[LabelQualityReasonCodeV2, ...]:
    return tuple(sorted(values, key=lambda value: value.value))


def _require_sorted_unique_summary_refs(
    values: tuple[StructuredLabelQualitySummaryV2, ...] | tuple[SemanticLabelQualitySummaryV2, ...],
    field_name: str,
) -> None:
    keys = tuple(_ref_key(value.label_spec_ref) for value in values)
    if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
        raise ValueError(f"{field_name} must be sorted and unique")


def _require_sorted_unique_refs(
    values: tuple[ObjectRef, ...],
    field_name: str,
) -> None:
    keys = tuple(_ref_key(value) for value in values)
    if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
        raise ValueError(f"{field_name} must be sorted and unique")


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    return audit.model_copy(update={"input_refs": _sorted_refs(refs)})


def _require_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
    label: str,
) -> None:
    if audit.input_refs != _sorted_refs(refs):
        raise ValueError(f"{label} audit refs are incomplete")


def _sorted_refs(values: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    unique = {_ref_key(value): value for value in values}
    return tuple(unique[key] for key in sorted(unique))


def _require_ref(
    value: ObjectRef,
    object_type: str,
    object_version: str,
    field_name: str,
) -> None:
    if value.object_type != object_type or value.object_version != object_version:
        raise ValueError(f"{field_name} must reference {object_type} {object_version}")


def _ref(
    object_type: str,
    object_id: str,
    object_sha256: str,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=object_id,
        object_version="v2",
        object_sha256=object_sha256,
    )


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _carried(
    value: ContractModelV2,
    exclude: set[str],
) -> str:
    return hashlib.sha256(
        json.dumps(
            canonical_value_v2(
                value.model_dump(
                    mode="python",
                    exclude=exclude,
                    exclude_none=False,
                )
            ),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _finalize[ModelT: ContractModelV2](
    value: ModelT,
    id_field: str,
    hash_field: str,
    prefix: str,
    digest: str,
) -> ModelT:
    return value.model_copy(
        update={
            id_field: f"{prefix}://sha256/{digest}",
            hash_field: digest,
        }
    )


def _validate_identity(
    object_id: str,
    object_sha256: str,
    prefix: str,
    observed: str,
    *,
    allow_pending: bool = True,
) -> None:
    if allow_pending and object_id == f"{prefix}://pending" and object_sha256 == "0" * 64:
        return
    if object_sha256 != observed or object_id != f"{prefix}://sha256/{observed}":
        raise ValueError(f"{prefix} identity is stale")
