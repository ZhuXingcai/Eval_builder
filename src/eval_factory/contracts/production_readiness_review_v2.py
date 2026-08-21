from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from eval_factory.contracts.canary_regression_v2 import (
    CanaryRegressionOutcomeV2,
)
from eval_factory.contracts.concurrency_experiment_v2 import (
    ConcurrencyExperimentOutcomeV2,
)
from eval_factory.contracts.core import ContractAudit, Identifier, ObjectRef, Sha256
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.contracts.label_quality_v2 import LabelQualityOutcomeV2
from eval_factory.contracts.real_trace_stability_v2 import (
    RealTraceStabilityOutcomeV2,
)

PRODUCTION_READINESS_REVIEW_POLICY_VERSION: Literal["production-readiness-review/r8-07-v1"] = (
    "production-readiness-review/r8-07-v1"
)
PRODUCTION_READINESS_REPOSITORY_PENDING_SHA256: Literal[
    "60f0652872f8b30cb78148e349f5c3ac89ccb2bba8206dfe8f3e0c7edcfb69ca"
] = "60f0652872f8b30cb78148e349f5c3ac89ccb2bba8206dfe8f3e0c7edcfb69ca"


class ProductionReadinessReviewEvidenceClassV2(StrEnum):
    PRODUCTION_ORGANIZATION_REVIEW = "PRODUCTION_ORGANIZATION_REVIEW"
    MECHANISM_VALIDATION_ONLY = "MECHANISM_VALIDATION_ONLY"
    REPOSITORY_PENDING_ONLY = "REPOSITORY_PENDING_ONLY"


class ProductionReadinessReviewDomainV2(StrEnum):
    SECURITY = "SECURITY"
    PRIVACY = "PRIVACY"
    RELEASE_PERMISSION = "RELEASE_PERMISSION"
    OPERATIONS_SLO = "OPERATIONS_SLO"


class ProductionReadinessApprovalDecisionV2(StrEnum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class ProductionReadinessDomainOutcomeV2(StrEnum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    APPROVAL_PENDING = "APPROVAL_PENDING"


class ProductionReadinessReviewOutcomeV2(StrEnum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    APPROVALS_PENDING = "APPROVALS_PENDING"


class ProductionReadinessReasonCodeV2(StrEnum):
    NONE = "NONE"
    APPROVAL_MISSING = "APPROVAL_MISSING"
    AUTHORITY_UNVERIFIED = "AUTHORITY_UNVERIFIED"
    AUTHORITY_NOT_INDEPENDENT = "AUTHORITY_NOT_INDEPENDENT"
    APPROVAL_EXPIRED = "APPROVAL_EXPIRED"
    APPROVAL_SCOPE_MISMATCH = "APPROVAL_SCOPE_MISMATCH"
    PREDECESSOR_EVIDENCE_PENDING = "PREDECESSOR_EVIDENCE_PENDING"
    OPERATIONS_SLO_POLICY_MISSING = "OPERATIONS_SLO_POLICY_MISSING"
    OPERATIONS_SLO_EVIDENCE_INCOMPLETE = "OPERATIONS_SLO_EVIDENCE_INCOMPLETE"
    OPERATIONS_SLO_THRESHOLD_NOT_MET = "OPERATIONS_SLO_THRESHOLD_NOT_MET"
    RELEASE_PERMISSION_MISSING = "RELEASE_PERMISSION_MISSING"
    TRUSTED_REJECTION = "TRUSTED_REJECTION"


class OperationsSLOMetricNameV2(StrEnum):
    THROUGHPUT_MILLIJOBS_PER_SECOND = "THROUGHPUT_MILLIJOBS_PER_SECOND"
    P50_LATENCY_MICROSECONDS = "P50_LATENCY_MICROSECONDS"
    P95_LATENCY_MICROSECONDS = "P95_LATENCY_MICROSECONDS"
    MODEL_REQUESTS_PER_JOB_MILLI = "MODEL_REQUESTS_PER_JOB_MILLI"
    COST_MICRO_USD_PER_JOB = "COST_MICRO_USD_PER_JOB"
    QUEUE_WAIT_P95_MICROSECONDS = "QUEUE_WAIT_P95_MICROSECONDS"
    ERROR_RATE_BASIS_POINTS = "ERROR_RATE_BASIS_POINTS"
    RESUME_SUCCESS_BASIS_POINTS = "RESUME_SUCCESS_BASIS_POINTS"


class OperationsSLOComparatorV2(StrEnum):
    GREATER_THAN_OR_EQUAL = "GREATER_THAN_OR_EQUAL"
    LESS_THAN_OR_EQUAL = "LESS_THAN_OR_EQUAL"


class OperationsSLOMetricAvailabilityV2(StrEnum):
    MEASURED = "MEASURED"
    UNAVAILABLE = "UNAVAILABLE"
    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"


class OperationsSLOAssessmentOutcomeV2(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    INCOMPLETE = "INCOMPLETE"


PRODUCTION_READINESS_DOMAIN_ORDER = tuple(ProductionReadinessReviewDomainV2)
OPERATIONS_SLO_METRIC_ORDER = tuple(OperationsSLOMetricNameV2)

_METRIC_SHAPES: dict[
    OperationsSLOMetricNameV2,
    tuple[str, OperationsSLOComparatorV2],
] = {
    OperationsSLOMetricNameV2.THROUGHPUT_MILLIJOBS_PER_SECOND: (
        "millijobs_per_second",
        OperationsSLOComparatorV2.GREATER_THAN_OR_EQUAL,
    ),
    OperationsSLOMetricNameV2.P50_LATENCY_MICROSECONDS: (
        "microseconds",
        OperationsSLOComparatorV2.LESS_THAN_OR_EQUAL,
    ),
    OperationsSLOMetricNameV2.P95_LATENCY_MICROSECONDS: (
        "microseconds",
        OperationsSLOComparatorV2.LESS_THAN_OR_EQUAL,
    ),
    OperationsSLOMetricNameV2.MODEL_REQUESTS_PER_JOB_MILLI: (
        "milli_requests_per_job",
        OperationsSLOComparatorV2.LESS_THAN_OR_EQUAL,
    ),
    OperationsSLOMetricNameV2.COST_MICRO_USD_PER_JOB: (
        "micro_usd_per_job",
        OperationsSLOComparatorV2.LESS_THAN_OR_EQUAL,
    ),
    OperationsSLOMetricNameV2.QUEUE_WAIT_P95_MICROSECONDS: (
        "microseconds",
        OperationsSLOComparatorV2.LESS_THAN_OR_EQUAL,
    ),
    OperationsSLOMetricNameV2.ERROR_RATE_BASIS_POINTS: (
        "basis_points",
        OperationsSLOComparatorV2.LESS_THAN_OR_EQUAL,
    ),
    OperationsSLOMetricNameV2.RESUME_SUCCESS_BASIS_POINTS: (
        "basis_points",
        OperationsSLOComparatorV2.GREATER_THAN_OR_EQUAL,
    ),
}


class ProductionReadinessReviewPolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/production-readiness-review-policy/v2"] = (
        "eval-factory/production-readiness-review-policy/v2"
    )
    policy_id: Identifier
    data_classification_policy_refs: tuple[ObjectRef, ObjectRef]
    user_approval_policy_ref: ObjectRef
    release_profile_decision_ref: ObjectRef
    repository_pending_evidence_ref: ObjectRef
    trusted_authority_registry_ref: ObjectRef | None = None
    evidence_class: ProductionReadinessReviewEvidenceClassV2
    required_domains: tuple[ProductionReadinessReviewDomainV2, ...] = PRODUCTION_READINESS_DOMAIN_ORDER
    required_operations_metrics: tuple[OperationsSLOMetricNameV2, ...] = OPERATIONS_SLO_METRIC_ORDER
    max_approvals: int = Field(ge=4, le=1_000)
    max_evidence_refs: int = Field(ge=4, le=10_000)
    max_private_bytes: int = Field(ge=2, le=10**12)
    max_report_bytes: int = Field(ge=2, le=100_000_000)
    policy_version: Literal["production-readiness-review/r8-07-v1"] = (
        PRODUCTION_READINESS_REVIEW_POLICY_VERSION
    )
    policy_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if self.required_domains != PRODUCTION_READINESS_DOMAIN_ORDER:
            raise ValueError("review domains are not canonical")
        if self.required_operations_metrics != OPERATIONS_SLO_METRIC_ORDER:
            raise ValueError("operations SLO metric inventory is not canonical")
        _require_sorted_unique_refs(
            self.data_classification_policy_refs,
            "data classification policy refs",
        )
        expected_versions = ("v1", "v2")
        if tuple(ref.object_version for ref in self.data_classification_policy_refs) != (expected_versions):
            raise ValueError("data classification policy versions are not exact")
        for ref in self.data_classification_policy_refs:
            _require_ref(
                ref,
                "data-classification-policy",
                ref.object_version,
                "data_classification_policy_refs",
            )
        _require_ref(
            self.user_approval_policy_ref,
            "user-approval-policy",
            "v2",
            "user_approval_policy_ref",
        )
        _require_ref(
            self.release_profile_decision_ref,
            "release-profile-decision",
            "v1",
            "release_profile_decision_ref",
        )
        _require_ref(
            self.repository_pending_evidence_ref,
            "production-readiness-review-repository-evidence",
            "json/v1",
            "repository_pending_evidence_ref",
        )
        if (
            self.repository_pending_evidence_ref.object_sha256
            != PRODUCTION_READINESS_REPOSITORY_PENDING_SHA256
        ):
            raise ValueError("repository pending evidence is not approved")
        if self.evidence_class is ProductionReadinessReviewEvidenceClassV2.REPOSITORY_PENDING_ONLY:
            if self.trusted_authority_registry_ref is not None:
                raise ValueError("repository-pending policy cannot bind a trusted registry")
        else:
            if self.trusted_authority_registry_ref is None:
                raise ValueError("full review policy requires a trusted registry")
            _require_ref(
                self.trusted_authority_registry_ref,
                "production-approval-authority-registry",
                "private-v1",
                "trusted_authority_registry_ref",
            )
        refs = (
            *self.data_classification_policy_refs,
            self.user_approval_policy_ref,
            self.release_profile_decision_ref,
            self.repository_pending_evidence_ref,
            *(
                (self.trusted_authority_registry_ref,)
                if self.trusted_authority_registry_ref is not None
                else ()
            ),
        )
        _require_audit(self.audit, refs, "production readiness review policy")
        _validate_identity(
            self.policy_id,
            self.policy_sha256,
            "production-readiness-review-policy",
            production_readiness_review_policy_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        data_classification_policy_refs: tuple[ObjectRef, ObjectRef],
        user_approval_policy_ref: ObjectRef,
        release_profile_decision_ref: ObjectRef,
        repository_pending_evidence_ref: ObjectRef,
        trusted_authority_registry_ref: ObjectRef | None,
        evidence_class: ProductionReadinessReviewEvidenceClassV2,
        max_approvals: int,
        max_evidence_refs: int,
        max_private_bytes: int,
        max_report_bytes: int,
        audit: ContractAudit,
    ) -> ProductionReadinessReviewPolicyV2:
        data_refs = tuple(sorted(data_classification_policy_refs, key=_ref_key))
        refs = (
            *data_refs,
            user_approval_policy_ref,
            release_profile_decision_ref,
            repository_pending_evidence_ref,
            *((trusted_authority_registry_ref,) if trusted_authority_registry_ref is not None else ()),
        )
        value = cls(
            policy_id="production-readiness-review-policy://pending",
            data_classification_policy_refs=data_refs,  # type: ignore[arg-type]
            user_approval_policy_ref=user_approval_policy_ref,
            release_profile_decision_ref=release_profile_decision_ref,
            repository_pending_evidence_ref=repository_pending_evidence_ref,
            trusted_authority_registry_ref=trusted_authority_registry_ref,
            evidence_class=evidence_class,
            max_approvals=max_approvals,
            max_evidence_refs=max_evidence_refs,
            max_private_bytes=max_private_bytes,
            max_report_bytes=max_report_bytes,
            policy_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "policy_id",
            "policy_sha256",
            "production-readiness-review-policy",
            production_readiness_review_policy_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return production_readiness_review_policy_v2_ref(self)


class ProductionReadinessPrerequisiteSummaryV2(ContractModelV2):
    schema_version: Literal["eval-factory/production-readiness-prerequisite-summary/v2"] = (
        "eval-factory/production-readiness-prerequisite-summary/v2"
    )
    label_quality_report_ref: ObjectRef
    canary_regression_report_ref: ObjectRef
    real_trace_stability_report_ref: ObjectRef
    concurrency_experiment_report_ref: ObjectRef
    release_profile_decision_ref: ObjectRef
    repository_evidence_ref: ObjectRef | None = None
    label_quality_outcome: LabelQualityOutcomeV2
    canary_regression_outcome: CanaryRegressionOutcomeV2
    real_trace_stability_outcome: RealTraceStabilityOutcomeV2
    concurrency_experiment_outcome: ConcurrencyExperimentOutcomeV2
    label_quality_claim_scope: Literal[
        "PRODUCTION_INDEPENDENT_TEST",
        "MECHANISM_VALIDATION_ONLY",
        "REPOSITORY_PENDING_ONLY",
    ] = "REPOSITORY_PENDING_ONLY"
    canary_regression_claim_scope: Literal["CANARY_REGRESSION_ONLY"] = "CANARY_REGRESSION_ONLY"
    real_trace_stability_claim_scope: Literal["REAL_TRACE_STABILITY_ONLY"] = "REAL_TRACE_STABILITY_ONLY"
    concurrency_experiment_claim_scope: Literal["CANARY_WORKER_STORAGE_ONLY"] = "CANARY_WORKER_STORAGE_ONLY"
    satisfies_sc_010: bool
    satisfies_sc_011: bool
    satisfies_sc_012: bool
    predecessor_authorizes_approval: Literal[False] = False
    predecessor_authorizes_attestation: Literal[False] = False
    predecessor_authorizes_production_release: Literal[False] = False

    @model_validator(mode="after")
    def validate_prerequisite(self) -> Self:
        _require_ref(
            self.label_quality_report_ref,
            "label-quality-evaluation-report",
            "v2",
            "label_quality_report_ref",
        )
        _require_ref(
            self.canary_regression_report_ref,
            "canary-regression-report",
            "v2",
            "canary_regression_report_ref",
        )
        _require_ref(
            self.real_trace_stability_report_ref,
            "real-trace-stability-report",
            "v2",
            "real_trace_stability_report_ref",
        )
        _require_ref(
            self.concurrency_experiment_report_ref,
            "concurrency-experiment-report",
            "v2",
            "concurrency_experiment_report_ref",
        )
        _require_ref(
            self.release_profile_decision_ref,
            "release-profile-decision",
            "v1",
            "release_profile_decision_ref",
        )
        if self.repository_evidence_ref is not None:
            _require_ref(
                self.repository_evidence_ref,
                "production-readiness-review-repository-evidence",
                "json/v1",
                "repository_evidence_ref",
            )
            if self.repository_evidence_ref.object_sha256 != PRODUCTION_READINESS_REPOSITORY_PENDING_SHA256:
                raise ValueError("repository evidence is not approved")
        if self.satisfies_sc_010 is not (
            self.label_quality_outcome is LabelQualityOutcomeV2.PASSED
            and self.label_quality_claim_scope == "PRODUCTION_INDEPENDENT_TEST"
        ):
            raise ValueError("SC-010 differs from label quality authority")
        if self.satisfies_sc_011 is not (
            self.real_trace_stability_outcome is RealTraceStabilityOutcomeV2.PASSED
        ):
            raise ValueError("SC-011 differs from stability authority")
        if self.satisfies_sc_012:
            raise ValueError("R8-06 evidence cannot satisfy SC-012")
        return self

    @property
    def refs(self) -> tuple[ObjectRef, ...]:
        return (
            self.label_quality_report_ref,
            self.canary_regression_report_ref,
            self.real_trace_stability_report_ref,
            self.concurrency_experiment_report_ref,
            self.release_profile_decision_ref,
            *((self.repository_evidence_ref,) if self.repository_evidence_ref is not None else ()),
        )


class OperationsSLOPolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/operations-slo-policy/v2"] = "eval-factory/operations-slo-policy/v2"
    policy_id: Identifier
    approval_policy_ref: ObjectRef
    minimum_throughput_millijobs_per_second: int = Field(ge=0)
    maximum_p50_latency_microseconds: int = Field(ge=0)
    maximum_p95_latency_microseconds: int = Field(ge=0)
    maximum_model_requests_per_job_milli: int = Field(ge=0)
    maximum_cost_micro_usd_per_job: int = Field(ge=0)
    maximum_queue_wait_p95_microseconds: int = Field(ge=0)
    maximum_error_rate_basis_points: int = Field(ge=0, le=10_000)
    minimum_resume_success_basis_points: int = Field(ge=0, le=10_000)
    minimum_sample_count: int = Field(ge=1, le=1_000_000_000)
    valid_from: datetime
    valid_until: datetime
    policy_version: Literal["production-readiness-review/r8-07-v1"] = (
        PRODUCTION_READINESS_REVIEW_POLICY_VERSION
    )
    policy_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        _require_ref(
            self.approval_policy_ref,
            "operations-slo-approval-policy",
            "v1",
            "approval_policy_ref",
        )
        if self.valid_until <= self.valid_from:
            raise ValueError("operations SLO validity window is invalid")
        _require_aware(self.valid_from, "operations SLO valid_from")
        _require_aware(self.valid_until, "operations SLO valid_until")
        _require_audit(
            self.audit,
            (self.approval_policy_ref,),
            "operations SLO policy",
        )
        _validate_identity(
            self.policy_id,
            self.policy_sha256,
            "operations-slo-policy",
            operations_slo_policy_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        approval_policy_ref: ObjectRef,
        minimum_throughput_millijobs_per_second: int,
        maximum_p50_latency_microseconds: int,
        maximum_p95_latency_microseconds: int,
        maximum_model_requests_per_job_milli: int,
        maximum_cost_micro_usd_per_job: int,
        maximum_queue_wait_p95_microseconds: int,
        maximum_error_rate_basis_points: int,
        minimum_resume_success_basis_points: int,
        minimum_sample_count: int,
        valid_from: datetime,
        valid_until: datetime,
        audit: ContractAudit,
    ) -> OperationsSLOPolicyV2:
        value = cls(
            policy_id="operations-slo-policy://pending",
            approval_policy_ref=approval_policy_ref,
            minimum_throughput_millijobs_per_second=(minimum_throughput_millijobs_per_second),
            maximum_p50_latency_microseconds=maximum_p50_latency_microseconds,
            maximum_p95_latency_microseconds=maximum_p95_latency_microseconds,
            maximum_model_requests_per_job_milli=(maximum_model_requests_per_job_milli),
            maximum_cost_micro_usd_per_job=maximum_cost_micro_usd_per_job,
            maximum_queue_wait_p95_microseconds=(maximum_queue_wait_p95_microseconds),
            maximum_error_rate_basis_points=maximum_error_rate_basis_points,
            minimum_resume_success_basis_points=minimum_resume_success_basis_points,
            minimum_sample_count=minimum_sample_count,
            valid_from=valid_from,
            valid_until=valid_until,
            policy_sha256="0" * 64,
            audit=_safe_audit(audit, (approval_policy_ref,)),
        )
        return _finalize(
            value,
            "policy_id",
            "policy_sha256",
            "operations-slo-policy",
            operations_slo_policy_v2_carried_sha256(value),
        )

    def threshold_for(self, metric: OperationsSLOMetricNameV2) -> int:
        return {
            OperationsSLOMetricNameV2.THROUGHPUT_MILLIJOBS_PER_SECOND: (
                self.minimum_throughput_millijobs_per_second
            ),
            OperationsSLOMetricNameV2.P50_LATENCY_MICROSECONDS: (self.maximum_p50_latency_microseconds),
            OperationsSLOMetricNameV2.P95_LATENCY_MICROSECONDS: (self.maximum_p95_latency_microseconds),
            OperationsSLOMetricNameV2.MODEL_REQUESTS_PER_JOB_MILLI: (
                self.maximum_model_requests_per_job_milli
            ),
            OperationsSLOMetricNameV2.COST_MICRO_USD_PER_JOB: (self.maximum_cost_micro_usd_per_job),
            OperationsSLOMetricNameV2.QUEUE_WAIT_P95_MICROSECONDS: (self.maximum_queue_wait_p95_microseconds),
            OperationsSLOMetricNameV2.ERROR_RATE_BASIS_POINTS: (self.maximum_error_rate_basis_points),
            OperationsSLOMetricNameV2.RESUME_SUCCESS_BASIS_POINTS: (self.minimum_resume_success_basis_points),
        }[metric]

    def to_ref(self) -> ObjectRef:
        return operations_slo_policy_v2_ref(self)


class OperationsSLOMetricV2(ContractModelV2):
    schema_version: Literal["eval-factory/operations-slo-metric/v2"] = "eval-factory/operations-slo-metric/v2"
    metric: OperationsSLOMetricNameV2
    unit: Identifier
    comparator: OperationsSLOComparatorV2
    threshold_value: int = Field(ge=0)
    observed_value: int | None = Field(default=None, ge=0)
    sample_count: int = Field(ge=0, le=1_000_000_000)
    required_sample_count: int = Field(ge=1, le=1_000_000_000)
    availability: OperationsSLOMetricAvailabilityV2
    meets_threshold: bool
    evidence_refs: tuple[ObjectRef, ...] = Field(max_length=1_000)

    @model_validator(mode="after")
    def validate_metric(self) -> Self:
        expected_unit, expected_comparator = _METRIC_SHAPES[self.metric]
        if self.unit != expected_unit or self.comparator is not expected_comparator:
            raise ValueError("operations SLO metric shape is invalid")
        _require_sorted_unique_refs(self.evidence_refs, "SLO metric evidence refs")
        if self.metric in {
            OperationsSLOMetricNameV2.ERROR_RATE_BASIS_POINTS,
            OperationsSLOMetricNameV2.RESUME_SUCCESS_BASIS_POINTS,
        } and (
            self.threshold_value > 10_000
            or (self.observed_value is not None and self.observed_value > 10_000)
        ):
            raise ValueError("basis-point SLO metric exceeds 10000")
        if self.availability is OperationsSLOMetricAvailabilityV2.UNAVAILABLE:
            if (
                self.observed_value is not None
                or self.sample_count != 0
                or self.meets_threshold
                or self.evidence_refs
            ):
                raise ValueError("unavailable SLO metric carries measured evidence")
            return self
        if self.observed_value is None or not self.evidence_refs:
            raise ValueError("observed SLO metric requires value and evidence")
        if self.availability is OperationsSLOMetricAvailabilityV2.INSUFFICIENT_SAMPLE:
            if self.sample_count >= self.required_sample_count or self.meets_threshold:
                raise ValueError("insufficient SLO metric sample is invalid")
            return self
        if self.sample_count < self.required_sample_count:
            raise ValueError("measured SLO metric sample is insufficient")
        expected = _compare(
            self.observed_value,
            self.threshold_value,
            self.comparator,
        )
        if self.meets_threshold is not expected:
            raise ValueError("SLO threshold result differs from metric evidence")
        return self

    @classmethod
    def measured(
        cls,
        *,
        policy: OperationsSLOPolicyV2,
        metric: OperationsSLOMetricNameV2,
        observed_value: int,
        sample_count: int,
        evidence_refs: tuple[ObjectRef, ...],
    ) -> OperationsSLOMetricV2:
        unit, comparator = _METRIC_SHAPES[metric]
        threshold = policy.threshold_for(metric)
        availability = (
            OperationsSLOMetricAvailabilityV2.MEASURED
            if sample_count >= policy.minimum_sample_count
            else OperationsSLOMetricAvailabilityV2.INSUFFICIENT_SAMPLE
        )
        meets = bool(
            availability is OperationsSLOMetricAvailabilityV2.MEASURED
            and _compare(observed_value, threshold, comparator)
        )
        return cls(
            metric=metric,
            unit=unit,
            comparator=comparator,
            threshold_value=threshold,
            observed_value=observed_value,
            sample_count=sample_count,
            required_sample_count=policy.minimum_sample_count,
            availability=availability,
            meets_threshold=meets,
            evidence_refs=_sorted_refs(evidence_refs),
        )

    @classmethod
    def unavailable(
        cls,
        *,
        policy: OperationsSLOPolicyV2,
        metric: OperationsSLOMetricNameV2,
    ) -> OperationsSLOMetricV2:
        unit, comparator = _METRIC_SHAPES[metric]
        return cls(
            metric=metric,
            unit=unit,
            comparator=comparator,
            threshold_value=policy.threshold_for(metric),
            observed_value=None,
            sample_count=0,
            required_sample_count=policy.minimum_sample_count,
            availability=OperationsSLOMetricAvailabilityV2.UNAVAILABLE,
            meets_threshold=False,
            evidence_refs=(),
        )


class OperationsSLOAssessmentV2(ContractModelV2):
    schema_version: Literal["eval-factory/operations-slo-assessment/v2"] = (
        "eval-factory/operations-slo-assessment/v2"
    )
    assessment_id: Identifier
    policy_ref: ObjectRef
    metrics: tuple[OperationsSLOMetricV2, ...] = Field(
        min_length=8,
        max_length=8,
    )
    evidence_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=1_000)
    outcome: OperationsSLOAssessmentOutcomeV2
    assessment_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_assessment(self) -> Self:
        _require_ref(
            self.policy_ref,
            "operations-slo-policy",
            "v2",
            "policy_ref",
        )
        names = tuple(metric.metric for metric in self.metrics)
        if names != OPERATIONS_SLO_METRIC_ORDER:
            raise ValueError("operations SLO metrics are not canonical")
        _require_sorted_unique_refs(
            self.evidence_refs,
            "operations SLO assessment evidence refs",
        )
        expected = _assessment_outcome(self.metrics)
        if self.outcome is not expected:
            raise ValueError("SLO assessment outcome differs from metrics")
        refs = (self.policy_ref, *self.evidence_refs)
        _require_audit(self.audit, refs, "operations SLO assessment")
        _validate_identity(
            self.assessment_id,
            self.assessment_sha256,
            "operations-slo-assessment",
            operations_slo_assessment_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        policy: OperationsSLOPolicyV2,
        metrics: tuple[OperationsSLOMetricV2, ...],
        evidence_refs: tuple[ObjectRef, ...],
        audit: ContractAudit,
    ) -> OperationsSLOAssessmentV2:
        ordered = tuple(sorted(metrics, key=lambda value: _metric_index(value.metric)))
        for metric in ordered:
            if (
                metric.threshold_value != policy.threshold_for(metric.metric)
                or metric.required_sample_count != policy.minimum_sample_count
            ):
                raise ValueError("SLO metric threshold differs from policy")
        refs = (policy.to_ref(), *_sorted_refs(evidence_refs))
        value = cls(
            assessment_id="operations-slo-assessment://pending",
            policy_ref=policy.to_ref(),
            metrics=ordered,
            evidence_refs=_sorted_refs(evidence_refs),
            outcome=_assessment_outcome(ordered),
            assessment_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "assessment_id",
            "assessment_sha256",
            "operations-slo-assessment",
            operations_slo_assessment_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return operations_slo_assessment_v2_ref(self)


class ProductionReadinessApprovalRecordV2(ContractModelV2):
    schema_version: Literal["eval-factory/production-readiness-approval-record/v2"] = (
        "eval-factory/production-readiness-approval-record/v2"
    )
    approval_record_id: Identifier
    domain: ProductionReadinessReviewDomainV2
    decision: ProductionReadinessApprovalDecisionV2
    review_series_id: Identifier
    review_version: int = Field(ge=1, le=1_000_000)
    authority_registry_ref: ObjectRef
    subject_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=1_000)
    evidence_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=1_000)
    policy_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=1_000)
    private_proof_commitment_sha256: Sha256
    issued_at: datetime
    valid_from: datetime
    valid_until: datetime
    approval_record_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        _require_ref(
            self.authority_registry_ref,
            "production-approval-authority-registry",
            "private-v1",
            "authority_registry_ref",
        )
        for name, refs in (
            ("approval subject refs", self.subject_refs),
            ("approval evidence refs", self.evidence_refs),
            ("approval policy refs", self.policy_refs),
        ):
            _require_sorted_unique_refs(refs, name)
        if self.valid_until <= self.valid_from or self.issued_at > self.valid_from:
            raise ValueError("approval validity window is invalid")
        _require_aware(self.issued_at, "approval issued_at")
        _require_aware(self.valid_from, "approval valid_from")
        _require_aware(self.valid_until, "approval valid_until")
        refs = (
            self.authority_registry_ref,
            *self.subject_refs,
            *self.evidence_refs,
            *self.policy_refs,
        )
        _require_audit(self.audit, refs, "production readiness approval")
        _validate_identity(
            self.approval_record_id,
            self.approval_record_sha256,
            "production-readiness-approval-record",
            production_readiness_approval_record_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        domain: ProductionReadinessReviewDomainV2,
        decision: ProductionReadinessApprovalDecisionV2,
        review_series_id: str,
        review_version: int,
        authority_registry_ref: ObjectRef,
        subject_refs: tuple[ObjectRef, ...],
        evidence_refs: tuple[ObjectRef, ...],
        policy_refs: tuple[ObjectRef, ...],
        private_proof_commitment_sha256: str,
        issued_at: datetime,
        valid_from: datetime,
        valid_until: datetime,
        audit: ContractAudit,
    ) -> ProductionReadinessApprovalRecordV2:
        subjects = _sorted_refs(subject_refs)
        evidence = _sorted_refs(evidence_refs)
        policies = _sorted_refs(policy_refs)
        refs = (
            authority_registry_ref,
            *subjects,
            *evidence,
            *policies,
        )
        value = cls(
            approval_record_id="production-readiness-approval-record://pending",
            domain=domain,
            decision=decision,
            review_series_id=review_series_id,
            review_version=review_version,
            authority_registry_ref=authority_registry_ref,
            subject_refs=subjects,
            evidence_refs=evidence,
            policy_refs=policies,
            private_proof_commitment_sha256=private_proof_commitment_sha256,
            issued_at=issued_at,
            valid_from=valid_from,
            valid_until=valid_until,
            approval_record_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "approval_record_id",
            "approval_record_sha256",
            "production-readiness-approval-record",
            production_readiness_approval_record_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return production_readiness_approval_record_v2_ref(self)


class ProductionReadinessDomainReviewV2(ContractModelV2):
    schema_version: Literal["eval-factory/production-readiness-domain-review/v2"] = (
        "eval-factory/production-readiness-domain-review/v2"
    )
    domain: ProductionReadinessReviewDomainV2
    evidence_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=1_000)
    approval_record: ProductionReadinessApprovalRecordV2 | None = None
    operations_slo_assessment_ref: ObjectRef | None = None
    operations_slo_outcome: OperationsSLOAssessmentOutcomeV2 | None = None
    outcome: ProductionReadinessDomainOutcomeV2
    reason_codes: tuple[ProductionReadinessReasonCodeV2, ...] = Field(min_length=1)
    valid_until: datetime | None = None

    @field_validator("reason_codes")
    @classmethod
    def validate_reason_order(
        cls,
        value: tuple[ProductionReadinessReasonCodeV2, ...],
    ) -> tuple[ProductionReadinessReasonCodeV2, ...]:
        expected = tuple(sorted(set(value), key=lambda item: item.value))
        if value != expected:
            raise ValueError("domain reason codes must be sorted and unique")
        if ProductionReadinessReasonCodeV2.NONE in value and value != (ProductionReadinessReasonCodeV2.NONE,):
            raise ValueError("NONE cannot accompany another domain reason")
        return value

    @model_validator(mode="after")
    def validate_review(self) -> Self:
        _require_sorted_unique_refs(self.evidence_refs, "domain evidence refs")
        if self.approval_record is not None and (self.approval_record.domain is not self.domain):
            raise ValueError("approval record domain differs from review")
        if self.domain is ProductionReadinessReviewDomainV2.OPERATIONS_SLO:
            if (self.operations_slo_assessment_ref is None) != (self.operations_slo_outcome is None):
                raise ValueError("operations review SLO closure must be all-or-none")
            if self.operations_slo_assessment_ref is not None:
                _require_ref(
                    self.operations_slo_assessment_ref,
                    "operations-slo-assessment",
                    "v2",
                    "operations_slo_assessment_ref",
                )
        elif self.operations_slo_assessment_ref is not None or self.operations_slo_outcome is not None:
            raise ValueError("non-operations review cannot carry SLO evidence")
        if self.outcome is ProductionReadinessDomainOutcomeV2.APPROVED:
            if (
                self.approval_record is None
                or self.approval_record.decision is not ProductionReadinessApprovalDecisionV2.APPROVED
                or self.reason_codes != (ProductionReadinessReasonCodeV2.NONE,)
                or self.valid_until != self.approval_record.valid_until
                or (
                    self.domain is ProductionReadinessReviewDomainV2.OPERATIONS_SLO
                    and self.operations_slo_outcome is not OperationsSLOAssessmentOutcomeV2.PASSED
                )
            ):
                raise ValueError("approved domain review is incomplete")
        elif self.outcome is ProductionReadinessDomainOutcomeV2.REJECTED:
            trusted_rejection = bool(
                self.approval_record is not None
                and self.approval_record.decision is ProductionReadinessApprovalDecisionV2.REJECTED
            )
            threshold_rejection = bool(
                self.domain is ProductionReadinessReviewDomainV2.OPERATIONS_SLO
                and self.operations_slo_outcome is OperationsSLOAssessmentOutcomeV2.FAILED
            )
            if (
                not (trusted_rejection or threshold_rejection)
                or self.valid_until
                != (self.approval_record.valid_until if self.approval_record is not None else None)
                or ProductionReadinessReasonCodeV2.NONE in self.reason_codes
            ):
                raise ValueError("rejected domain review lacks rejection evidence")
            if (
                trusted_rejection
                and ProductionReadinessReasonCodeV2.TRUSTED_REJECTION not in self.reason_codes
            ):
                raise ValueError("trusted rejection reason is missing")
            if (
                threshold_rejection
                and ProductionReadinessReasonCodeV2.OPERATIONS_SLO_THRESHOLD_NOT_MET not in self.reason_codes
            ):
                raise ValueError("SLO threshold rejection reason is missing")
        else:
            if ProductionReadinessReasonCodeV2.NONE in self.reason_codes or self.valid_until is not None:
                raise ValueError("pending domain review overclaims authority")
            if self.operations_slo_outcome is OperationsSLOAssessmentOutcomeV2.FAILED:
                raise ValueError("failed operations SLO cannot remain pending")
            if (
                self.domain is ProductionReadinessReviewDomainV2.RELEASE_PERMISSION
                and ProductionReadinessReasonCodeV2.RELEASE_PERMISSION_MISSING not in self.reason_codes
            ):
                raise ValueError("pending release permission reason is missing")
            if self.domain is ProductionReadinessReviewDomainV2.OPERATIONS_SLO:
                expected_reason = (
                    ProductionReadinessReasonCodeV2.OPERATIONS_SLO_POLICY_MISSING
                    if self.operations_slo_outcome is None
                    else ProductionReadinessReasonCodeV2.OPERATIONS_SLO_EVIDENCE_INCOMPLETE
                    if self.operations_slo_outcome is OperationsSLOAssessmentOutcomeV2.INCOMPLETE
                    else None
                )
                if expected_reason is not None and expected_reason not in self.reason_codes:
                    raise ValueError("pending operations SLO reason is missing")
        return self

    @classmethod
    def from_approval(
        cls,
        *,
        approval_record: ProductionReadinessApprovalRecordV2,
        evidence_refs: tuple[ObjectRef, ...],
        operations_slo_assessment: OperationsSLOAssessmentV2 | None,
    ) -> ProductionReadinessDomainReviewV2:
        domain = approval_record.domain
        slo_ref = operations_slo_assessment.to_ref() if operations_slo_assessment is not None else None
        slo_outcome = operations_slo_assessment.outcome if operations_slo_assessment is not None else None
        if approval_record.decision is ProductionReadinessApprovalDecisionV2.REJECTED:
            outcome = ProductionReadinessDomainOutcomeV2.REJECTED
            reasons = (ProductionReadinessReasonCodeV2.TRUSTED_REJECTION,)
            valid_until: datetime | None = approval_record.valid_until
        elif (
            domain is ProductionReadinessReviewDomainV2.OPERATIONS_SLO
            and slo_outcome is OperationsSLOAssessmentOutcomeV2.FAILED
        ):
            outcome = ProductionReadinessDomainOutcomeV2.REJECTED
            reasons = (ProductionReadinessReasonCodeV2.OPERATIONS_SLO_THRESHOLD_NOT_MET,)
            valid_until = approval_record.valid_until
        elif (
            domain is ProductionReadinessReviewDomainV2.OPERATIONS_SLO
            and slo_outcome is not OperationsSLOAssessmentOutcomeV2.PASSED
        ):
            outcome = ProductionReadinessDomainOutcomeV2.APPROVAL_PENDING
            reasons = (ProductionReadinessReasonCodeV2.OPERATIONS_SLO_EVIDENCE_INCOMPLETE,)
            valid_until = None
        else:
            outcome = ProductionReadinessDomainOutcomeV2.APPROVED
            reasons = (ProductionReadinessReasonCodeV2.NONE,)
            valid_until = approval_record.valid_until
        return cls(
            domain=domain,
            evidence_refs=_sorted_refs(evidence_refs),
            approval_record=approval_record,
            operations_slo_assessment_ref=slo_ref,
            operations_slo_outcome=slo_outcome,
            outcome=outcome,
            reason_codes=reasons,
            valid_until=valid_until,
        )

    @classmethod
    def pending(
        cls,
        *,
        domain: ProductionReadinessReviewDomainV2,
        evidence_refs: tuple[ObjectRef, ...],
        reason_codes: tuple[ProductionReadinessReasonCodeV2, ...],
        operations_slo_assessment: OperationsSLOAssessmentV2 | None = None,
    ) -> ProductionReadinessDomainReviewV2:
        return cls(
            domain=domain,
            evidence_refs=_sorted_refs(evidence_refs),
            approval_record=None,
            operations_slo_assessment_ref=(
                operations_slo_assessment.to_ref() if operations_slo_assessment is not None else None
            ),
            operations_slo_outcome=(
                operations_slo_assessment.outcome if operations_slo_assessment is not None else None
            ),
            outcome=ProductionReadinessDomainOutcomeV2.APPROVAL_PENDING,
            reason_codes=tuple(sorted(set(reason_codes), key=lambda value: value.value)),
            valid_until=None,
        )

    @property
    def refs(self) -> tuple[ObjectRef, ...]:
        return (
            *self.evidence_refs,
            *((self.approval_record.to_ref(),) if self.approval_record is not None else ()),
            *(
                (self.operations_slo_assessment_ref,)
                if self.operations_slo_assessment_ref is not None
                else ()
            ),
        )


class ProductionReadinessReviewReportV2(ContractModelV2):
    schema_version: Literal["eval-factory/production-readiness-review-report/v2"] = (
        "eval-factory/production-readiness-review-report/v2"
    )
    report_id: Identifier
    policy_ref: ObjectRef
    evidence_class: ProductionReadinessReviewEvidenceClassV2
    prerequisite: ProductionReadinessPrerequisiteSummaryV2
    review_series_id: Identifier
    review_version: int = Field(ge=1, le=1_000_000)
    previous_report_ref: ObjectRef | None = None
    private_request_closure_sha256: Sha256 | None = None
    private_approval_closure_sha256: Sha256 | None = None
    domain_reviews: tuple[ProductionReadinessDomainReviewV2, ...] = Field(
        min_length=4,
        max_length=4,
    )
    approved_domain_count: int = Field(ge=0, le=4)
    pending_domain_count: int = Field(ge=0, le=4)
    rejected_domain_count: int = Field(ge=0, le=4)
    outcome: ProductionReadinessReviewOutcomeV2
    reason_codes: tuple[ProductionReadinessReasonCodeV2, ...] = Field(min_length=1)
    valid_until: datetime | None = None
    satisfies_sc_013: bool
    authorizes_attestation: Literal[False] = False
    authorizes_production_release: Literal[False] = False
    policy_version: Literal["production-readiness-review/r8-07-v1"] = (
        PRODUCTION_READINESS_REVIEW_POLICY_VERSION
    )
    report_sha256: Sha256
    audit: ContractAudit

    @field_validator("domain_reviews")
    @classmethod
    def validate_review_order(
        cls,
        value: tuple[ProductionReadinessDomainReviewV2, ...],
    ) -> tuple[ProductionReadinessDomainReviewV2, ...]:
        if tuple(review.domain for review in value) != (PRODUCTION_READINESS_DOMAIN_ORDER):
            raise ValueError("domain reviews are not canonical")
        return value

    @field_validator("reason_codes")
    @classmethod
    def validate_reason_order(
        cls,
        value: tuple[ProductionReadinessReasonCodeV2, ...],
    ) -> tuple[ProductionReadinessReasonCodeV2, ...]:
        expected = tuple(sorted(set(value), key=lambda item: item.value))
        if value != expected:
            raise ValueError("report reason codes must be sorted and unique")
        if ProductionReadinessReasonCodeV2.NONE in value and value != (ProductionReadinessReasonCodeV2.NONE,):
            raise ValueError("NONE cannot accompany another report reason")
        return value

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        _require_ref(
            self.policy_ref,
            "production-readiness-review-policy",
            "v2",
            "policy_ref",
        )
        if self.review_version == 1:
            if self.previous_report_ref is not None:
                raise ValueError("review version 1 cannot have a predecessor")
        else:
            if self.previous_report_ref is None:
                raise ValueError("later review version requires previous report")
            _require_ref(
                self.previous_report_ref,
                "production-readiness-review-report",
                "v2",
                "previous_report_ref",
            )
        expected_counts = _domain_counts(self.domain_reviews)
        if (
            self.approved_domain_count,
            self.pending_domain_count,
            self.rejected_domain_count,
        ) != expected_counts:
            raise ValueError("domain review counts are not derived")
        expected_outcome, expected_reasons = _review_outcome(self.domain_reviews)
        if self.outcome is not expected_outcome or self.reason_codes != (expected_reasons):
            raise ValueError("review report outcome differs from domains")
        expected_valid_until = (
            min(review.valid_until for review in self.domain_reviews if review.valid_until is not None)
            if expected_outcome is ProductionReadinessReviewOutcomeV2.APPROVED
            else None
        )
        if self.valid_until != expected_valid_until:
            raise ValueError("review report validity differs from approvals")
        expected_sc_013 = bool(
            expected_outcome is ProductionReadinessReviewOutcomeV2.APPROVED
            and self.evidence_class is ProductionReadinessReviewEvidenceClassV2.PRODUCTION_ORGANIZATION_REVIEW
        )
        if self.satisfies_sc_013 is not expected_sc_013:
            raise ValueError("SC-013 authority differs from review evidence")
        if self.evidence_class is ProductionReadinessReviewEvidenceClassV2.REPOSITORY_PENDING_ONLY:
            if (
                self.prerequisite.repository_evidence_ref is None
                or self.private_request_closure_sha256 is not None
                or self.private_approval_closure_sha256 is not None
                or any(review.approval_record is not None for review in self.domain_reviews)
                or expected_outcome is not ProductionReadinessReviewOutcomeV2.APPROVALS_PENDING
            ):
                raise ValueError("repository-pending report carries private authority")
        else:
            if (
                self.prerequisite.repository_evidence_ref is not None
                or self.private_request_closure_sha256 is None
                or self.private_approval_closure_sha256 is None
            ):
                raise ValueError("full review report closure is incomplete")
        refs = (
            self.policy_ref,
            *self.prerequisite.refs,
            *((self.previous_report_ref,) if self.previous_report_ref is not None else ()),
            *(ref for review in self.domain_reviews for ref in review.refs),
        )
        _require_audit(self.audit, refs, "production readiness review report")
        _validate_identity(
            self.report_id,
            self.report_sha256,
            "production-readiness-review-report",
            production_readiness_review_report_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        policy_ref: ObjectRef,
        evidence_class: ProductionReadinessReviewEvidenceClassV2,
        prerequisite: ProductionReadinessPrerequisiteSummaryV2,
        review_series_id: str,
        review_version: int,
        previous_report_ref: ObjectRef | None,
        private_request_closure_sha256: str | None,
        private_approval_closure_sha256: str | None,
        domain_reviews: tuple[ProductionReadinessDomainReviewV2, ...],
        audit: ContractAudit,
    ) -> ProductionReadinessReviewReportV2:
        ordered = tuple(sorted(domain_reviews, key=lambda review: _domain_index(review.domain)))
        counts = _domain_counts(ordered)
        outcome, reasons = _review_outcome(ordered)
        valid_until = (
            min(review.valid_until for review in ordered if review.valid_until is not None)
            if outcome is ProductionReadinessReviewOutcomeV2.APPROVED
            else None
        )
        refs = (
            policy_ref,
            *prerequisite.refs,
            *((previous_report_ref,) if previous_report_ref is not None else ()),
            *(ref for review in ordered for ref in review.refs),
        )
        value = cls(
            report_id="production-readiness-review-report://pending",
            policy_ref=policy_ref,
            evidence_class=evidence_class,
            prerequisite=prerequisite,
            review_series_id=review_series_id,
            review_version=review_version,
            previous_report_ref=previous_report_ref,
            private_request_closure_sha256=private_request_closure_sha256,
            private_approval_closure_sha256=private_approval_closure_sha256,
            domain_reviews=ordered,
            approved_domain_count=counts[0],
            pending_domain_count=counts[1],
            rejected_domain_count=counts[2],
            outcome=outcome,
            reason_codes=reasons,
            valid_until=valid_until,
            satisfies_sc_013=bool(
                outcome is ProductionReadinessReviewOutcomeV2.APPROVED
                and evidence_class is ProductionReadinessReviewEvidenceClassV2.PRODUCTION_ORGANIZATION_REVIEW
            ),
            report_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "report_id",
            "report_sha256",
            "production-readiness-review-report",
            production_readiness_review_report_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return production_readiness_review_report_v2_ref(self)


def production_readiness_review_policy_v2_carried_sha256(
    value: ProductionReadinessReviewPolicyV2,
) -> str:
    return _carried(value, {"policy_id", "policy_sha256", "audit"})


def operations_slo_policy_v2_carried_sha256(
    value: OperationsSLOPolicyV2,
) -> str:
    return _carried(value, {"policy_id", "policy_sha256", "audit"})


def operations_slo_assessment_v2_carried_sha256(
    value: OperationsSLOAssessmentV2,
) -> str:
    return _carried(value, {"assessment_id", "assessment_sha256", "audit"})


def production_readiness_approval_record_v2_carried_sha256(
    value: ProductionReadinessApprovalRecordV2,
) -> str:
    return _carried(
        value,
        {"approval_record_id", "approval_record_sha256", "audit"},
    )


def production_readiness_review_report_v2_carried_sha256(
    value: ProductionReadinessReviewReportV2,
) -> str:
    return _carried(value, {"report_id", "report_sha256", "audit"})


def production_readiness_review_policy_v2_ref(
    value: ProductionReadinessReviewPolicyV2,
) -> ObjectRef:
    validate_production_readiness_review_policy_v2_identity(value)
    return _ref(
        "production-readiness-review-policy",
        value.policy_id,
        "v2",
        value.policy_sha256,
    )


def operations_slo_policy_v2_ref(
    value: OperationsSLOPolicyV2,
) -> ObjectRef:
    validate_operations_slo_policy_v2_identity(value)
    return _ref(
        "operations-slo-policy",
        value.policy_id,
        "v2",
        value.policy_sha256,
    )


def operations_slo_assessment_v2_ref(
    value: OperationsSLOAssessmentV2,
) -> ObjectRef:
    validate_operations_slo_assessment_v2_identity(value)
    return _ref(
        "operations-slo-assessment",
        value.assessment_id,
        "v2",
        value.assessment_sha256,
    )


def production_readiness_approval_record_v2_ref(
    value: ProductionReadinessApprovalRecordV2,
) -> ObjectRef:
    validate_production_readiness_approval_record_v2_identity(value)
    return _ref(
        "production-readiness-approval-record",
        value.approval_record_id,
        "v2",
        value.approval_record_sha256,
    )


def production_readiness_review_report_v2_ref(
    value: ProductionReadinessReviewReportV2,
) -> ObjectRef:
    validate_production_readiness_review_report_v2_identity(value)
    return _ref(
        "production-readiness-review-report",
        value.report_id,
        "v2",
        value.report_sha256,
    )


def validate_production_readiness_review_policy_v2_identity(
    value: ProductionReadinessReviewPolicyV2,
) -> None:
    _validate_identity(
        value.policy_id,
        value.policy_sha256,
        "production-readiness-review-policy",
        production_readiness_review_policy_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_operations_slo_policy_v2_identity(
    value: OperationsSLOPolicyV2,
) -> None:
    _validate_identity(
        value.policy_id,
        value.policy_sha256,
        "operations-slo-policy",
        operations_slo_policy_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_operations_slo_assessment_v2_identity(
    value: OperationsSLOAssessmentV2,
) -> None:
    _validate_identity(
        value.assessment_id,
        value.assessment_sha256,
        "operations-slo-assessment",
        operations_slo_assessment_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_production_readiness_approval_record_v2_identity(
    value: ProductionReadinessApprovalRecordV2,
) -> None:
    _validate_identity(
        value.approval_record_id,
        value.approval_record_sha256,
        "production-readiness-approval-record",
        production_readiness_approval_record_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_production_readiness_review_report_v2_identity(
    value: ProductionReadinessReviewReportV2,
) -> None:
    _validate_identity(
        value.report_id,
        value.report_sha256,
        "production-readiness-review-report",
        production_readiness_review_report_v2_carried_sha256(value),
        allow_pending=False,
    )


def _assessment_outcome(
    metrics: tuple[OperationsSLOMetricV2, ...],
) -> OperationsSLOAssessmentOutcomeV2:
    if any(metric.availability is not OperationsSLOMetricAvailabilityV2.MEASURED for metric in metrics):
        return OperationsSLOAssessmentOutcomeV2.INCOMPLETE
    if any(not metric.meets_threshold for metric in metrics):
        return OperationsSLOAssessmentOutcomeV2.FAILED
    return OperationsSLOAssessmentOutcomeV2.PASSED


def _domain_counts(
    reviews: tuple[ProductionReadinessDomainReviewV2, ...],
) -> tuple[int, int, int]:
    return (
        sum(review.outcome is ProductionReadinessDomainOutcomeV2.APPROVED for review in reviews),
        sum(review.outcome is ProductionReadinessDomainOutcomeV2.APPROVAL_PENDING for review in reviews),
        sum(review.outcome is ProductionReadinessDomainOutcomeV2.REJECTED for review in reviews),
    )


def _review_outcome(
    reviews: tuple[ProductionReadinessDomainReviewV2, ...],
) -> tuple[
    ProductionReadinessReviewOutcomeV2,
    tuple[ProductionReadinessReasonCodeV2, ...],
]:
    if any(review.outcome is ProductionReadinessDomainOutcomeV2.REJECTED for review in reviews):
        outcome = ProductionReadinessReviewOutcomeV2.REJECTED
    elif any(review.outcome is ProductionReadinessDomainOutcomeV2.APPROVAL_PENDING for review in reviews):
        outcome = ProductionReadinessReviewOutcomeV2.APPROVALS_PENDING
    else:
        outcome = ProductionReadinessReviewOutcomeV2.APPROVED
    reasons = {
        reason
        for review in reviews
        for reason in review.reason_codes
        if reason is not ProductionReadinessReasonCodeV2.NONE
    }
    return (
        outcome,
        (
            tuple(sorted(reasons, key=lambda value: value.value))
            if reasons
            else (ProductionReadinessReasonCodeV2.NONE,)
        ),
    )


def _compare(
    observed: int,
    threshold: int,
    comparator: OperationsSLOComparatorV2,
) -> bool:
    if comparator is OperationsSLOComparatorV2.GREATER_THAN_OR_EQUAL:
        return observed >= threshold
    return observed <= threshold


def _metric_index(value: OperationsSLOMetricNameV2) -> int:
    return OPERATIONS_SLO_METRIC_ORDER.index(value)


def _domain_index(value: ProductionReadinessReviewDomainV2) -> int:
    return PRODUCTION_READINESS_DOMAIN_ORDER.index(value)


def _require_aware(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")


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
    version: str,
    digest: str,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=object_id,
        object_version=version,
        object_sha256=digest,
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
