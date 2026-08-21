from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_factory.contracts.canary_regression_v2 import CanaryRegressionOutcomeV2
from eval_factory.contracts.concurrency_experiment_v2 import (
    ConcurrencyExperimentOutcomeV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.label_quality_v2 import LabelQualityOutcomeV2
from eval_factory.contracts.production_readiness_review_v2 import (
    OPERATIONS_SLO_METRIC_ORDER,
    PRODUCTION_READINESS_REPOSITORY_PENDING_SHA256,
    OperationsSLOAssessmentOutcomeV2,
    OperationsSLOAssessmentV2,
    OperationsSLOMetricNameV2,
    OperationsSLOMetricV2,
    OperationsSLOPolicyV2,
    ProductionReadinessApprovalDecisionV2,
    ProductionReadinessApprovalRecordV2,
    ProductionReadinessDomainOutcomeV2,
    ProductionReadinessDomainReviewV2,
    ProductionReadinessPrerequisiteSummaryV2,
    ProductionReadinessReasonCodeV2,
    ProductionReadinessReviewDomainV2,
    ProductionReadinessReviewEvidenceClassV2,
    ProductionReadinessReviewOutcomeV2,
    ProductionReadinessReviewPolicyV2,
    ProductionReadinessReviewReportV2,
    operations_slo_assessment_v2_ref,
    operations_slo_policy_v2_ref,
    production_readiness_approval_record_v2_ref,
    production_readiness_review_policy_v2_ref,
    production_readiness_review_report_v2_ref,
)
from eval_factory.contracts.real_trace_stability_v2 import (
    RealTraceStabilityOutcomeV2,
)

HASH = "a" * 64
SECOND_HASH = "b" * 64
THIRD_HASH = "c" * 64
NOW = datetime(2026, 8, 3, tzinfo=UTC)


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v2",
    digest: str = HASH,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version=version,
        object_sha256=digest,
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="r8-07-contract-test",
        governing_versions=(
            VersionBinding(
                component="production-readiness-review",
                version="production-readiness-review/r8-07-v1",
            ),
        ),
    )


def _repository_ref() -> ObjectRef:
    return _ref(
        "production-readiness-review-repository-evidence",
        "r8-07/v1",
        version="json/v1",
        digest=PRODUCTION_READINESS_REPOSITORY_PENDING_SHA256,
    )


def _policy(
    evidence_class: ProductionReadinessReviewEvidenceClassV2 = (
        ProductionReadinessReviewEvidenceClassV2.MECHANISM_VALIDATION_ONLY
    ),
) -> ProductionReadinessReviewPolicyV2:
    return ProductionReadinessReviewPolicyV2.create(
        data_classification_policy_refs=(
            _ref("data-classification-policy", "v1", version="v1"),
            _ref("data-classification-policy", "v2", version="v2"),
        ),
        user_approval_policy_ref=_ref(
            "user-approval-policy",
            "v2",
            version="v2",
        ),
        release_profile_decision_ref=_ref(
            "release-profile-decision",
            "lh-v1",
            version="v1",
        ),
        repository_pending_evidence_ref=_repository_ref(),
        trusted_authority_registry_ref=(
            None
            if evidence_class is ProductionReadinessReviewEvidenceClassV2.REPOSITORY_PENDING_ONLY
            else _ref(
                "production-approval-authority-registry",
                "test",
                version="private-v1",
            )
        ),
        evidence_class=evidence_class,
        max_approvals=4,
        max_evidence_refs=64,
        max_private_bytes=1_000_000,
        max_report_bytes=1_000_000,
        audit=_audit(),
    )


def _prerequisite(
    *,
    repository: bool = False,
) -> ProductionReadinessPrerequisiteSummaryV2:
    return ProductionReadinessPrerequisiteSummaryV2(
        label_quality_report_ref=_ref(
            "label-quality-evaluation-report",
            "accepted",
        ),
        canary_regression_report_ref=_ref(
            "canary-regression-report",
            "accepted",
        ),
        real_trace_stability_report_ref=_ref(
            "real-trace-stability-report",
            "accepted",
        ),
        concurrency_experiment_report_ref=_ref(
            "concurrency-experiment-report",
            "accepted",
        ),
        release_profile_decision_ref=_ref(
            "release-profile-decision",
            "lh-v1",
            version="v1",
        ),
        repository_evidence_ref=_repository_ref() if repository else None,
        label_quality_outcome=LabelQualityOutcomeV2.STATISTICAL_GATE_PENDING,
        canary_regression_outcome=CanaryRegressionOutcomeV2.PASSED,
        real_trace_stability_outcome=(RealTraceStabilityOutcomeV2.STATISTICAL_GATE_PENDING),
        concurrency_experiment_outcome=(ConcurrencyExperimentOutcomeV2.RECOMMENDATION_PENDING),
        satisfies_sc_010=False,
        satisfies_sc_011=False,
        satisfies_sc_012=False,
    )


def _slo_policy() -> OperationsSLOPolicyV2:
    return OperationsSLOPolicyV2.create(
        approval_policy_ref=_ref(
            "operations-slo-approval-policy",
            "test",
            version="v1",
        ),
        minimum_throughput_millijobs_per_second=100,
        maximum_p50_latency_microseconds=1_000,
        maximum_p95_latency_microseconds=2_000,
        maximum_model_requests_per_job_milli=1_000,
        maximum_cost_micro_usd_per_job=10_000,
        maximum_queue_wait_p95_microseconds=500,
        maximum_error_rate_basis_points=100,
        minimum_resume_success_basis_points=9_900,
        minimum_sample_count=10,
        valid_from=NOW,
        valid_until=NOW + timedelta(days=30),
        audit=_audit(),
    )


def _slo_metrics(
    policy: OperationsSLOPolicyV2,
) -> tuple[OperationsSLOMetricV2, ...]:
    return tuple(
        OperationsSLOMetricV2.measured(
            policy=policy,
            metric=name,
            observed_value=policy.threshold_for(name),
            sample_count=policy.minimum_sample_count,
            evidence_refs=(_ref("operations-metric-evidence", name.value.lower()),),
        )
        for name in OPERATIONS_SLO_METRIC_ORDER
    )


def _slo_assessment(
    *,
    metrics: tuple[OperationsSLOMetricV2, ...] | None = None,
) -> OperationsSLOAssessmentV2:
    policy = _slo_policy()
    return OperationsSLOAssessmentV2.create(
        policy=policy,
        metrics=_slo_metrics(policy) if metrics is None else metrics,
        evidence_refs=(_ref("operations-evidence", "test"),),
        audit=_audit(),
    )


def _approval(
    *,
    domain: ProductionReadinessReviewDomainV2,
    decision: ProductionReadinessApprovalDecisionV2 = (ProductionReadinessApprovalDecisionV2.APPROVED),
    operations: OperationsSLOAssessmentV2 | None = None,
) -> ProductionReadinessApprovalRecordV2:
    subject_refs = (
        _ref("production-readiness-review-subject", "test"),
        *((operations_slo_assessment_v2_ref(operations),) if operations is not None else ()),
    )
    return ProductionReadinessApprovalRecordV2.create(
        domain=domain,
        decision=decision,
        review_series_id="production-readiness-review-series://test",
        review_version=1,
        authority_registry_ref=_ref(
            "production-approval-authority-registry",
            "test",
            version="private-v1",
        ),
        subject_refs=subject_refs,
        evidence_refs=(_ref("review-evidence", domain.value.lower()),),
        policy_refs=(_ref("review-domain-policy", domain.value.lower()),),
        private_proof_commitment_sha256=SECOND_HASH,
        issued_at=NOW,
        valid_from=NOW,
        valid_until=NOW + timedelta(days=30),
        audit=_audit(),
    )


def _approved_reviews() -> tuple[ProductionReadinessDomainReviewV2, ...]:
    operations = _slo_assessment()
    return tuple(
        ProductionReadinessDomainReviewV2.from_approval(
            approval_record=_approval(
                domain=domain,
                operations=(
                    operations if domain is ProductionReadinessReviewDomainV2.OPERATIONS_SLO else None
                ),
            ),
            evidence_refs=(_ref("domain-review-evidence", domain.value.lower()),),
            operations_slo_assessment=(
                operations if domain is ProductionReadinessReviewDomainV2.OPERATIONS_SLO else None
            ),
        )
        for domain in ProductionReadinessReviewDomainV2
    )


def _pending_reasons(
    domain: ProductionReadinessReviewDomainV2,
) -> tuple[ProductionReadinessReasonCodeV2, ...]:
    values = {ProductionReadinessReasonCodeV2.APPROVAL_MISSING}
    if domain is ProductionReadinessReviewDomainV2.RELEASE_PERMISSION:
        values.add(ProductionReadinessReasonCodeV2.RELEASE_PERMISSION_MISSING)
    if domain is ProductionReadinessReviewDomainV2.OPERATIONS_SLO:
        values.add(ProductionReadinessReasonCodeV2.OPERATIONS_SLO_POLICY_MISSING)
    return tuple(sorted(values, key=lambda value: value.value))


def test_policy_is_strict_frozen_and_content_addressed() -> None:
    policy = _policy()
    assert production_readiness_review_policy_v2_ref(policy).object_sha256 == (policy.policy_sha256)
    with pytest.raises(ValidationError):
        policy.policy_version = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ProductionReadinessReviewPolicyV2.model_validate(
            {**policy.model_dump(mode="python"), "unexpected": True}
        )
    with pytest.raises(ValidationError, match="repository"):
        ProductionReadinessReviewPolicyV2.model_validate(
            {
                **policy.model_dump(mode="python"),
                "repository_pending_evidence_ref": _repository_ref().model_copy(
                    update={"object_sha256": THIRD_HASH}
                ),
            }
        )


def test_repository_policy_cannot_bind_a_trusted_registry() -> None:
    policy = _policy(ProductionReadinessReviewEvidenceClassV2.REPOSITORY_PENDING_ONLY)
    with pytest.raises(ValidationError, match="registry"):
        ProductionReadinessReviewPolicyV2.model_validate(
            {
                **policy.model_dump(mode="python"),
                "trusted_authority_registry_ref": _ref(
                    "production-approval-authority-registry",
                    "forged",
                    version="private-v1",
                ),
            }
        )


def test_slo_policy_metrics_and_assessment_are_exact() -> None:
    policy = _slo_policy()
    assert operations_slo_policy_v2_ref(policy).object_sha256 == (policy.policy_sha256)
    metrics = _slo_metrics(policy)
    assert tuple(metric.metric for metric in metrics) == OPERATIONS_SLO_METRIC_ORDER
    assert all(metric.meets_threshold for metric in metrics)
    assessment = _slo_assessment(metrics=metrics)
    assert assessment.outcome is OperationsSLOAssessmentOutcomeV2.PASSED
    assert operations_slo_assessment_v2_ref(assessment).object_sha256 == (assessment.assessment_sha256)

    incomplete = OperationsSLOMetricV2.unavailable(
        policy=policy,
        metric=metrics[0].metric,
    )
    pending = _slo_assessment(metrics=(incomplete, *metrics[1:]))
    assert pending.outcome is OperationsSLOAssessmentOutcomeV2.INCOMPLETE

    failed = OperationsSLOMetricV2.measured(
        policy=policy,
        metric=OperationsSLOMetricNameV2.ERROR_RATE_BASIS_POINTS,
        observed_value=policy.maximum_error_rate_basis_points + 1,
        sample_count=policy.minimum_sample_count,
        evidence_refs=(_ref("operations-metric-evidence", "error-rate-fail"),),
    )
    failed_metrics = tuple(
        failed if metric.metric is OperationsSLOMetricNameV2.ERROR_RATE_BASIS_POINTS else metric
        for metric in metrics
    )
    rejected = _slo_assessment(metrics=failed_metrics)
    assert rejected.outcome is OperationsSLOAssessmentOutcomeV2.FAILED


def test_slo_assessment_rejects_missing_duplicate_and_policy_drift() -> None:
    policy = _slo_policy()
    metrics = _slo_metrics(policy)
    with pytest.raises(ValidationError, match="canonical"):
        OperationsSLOAssessmentV2.create(
            policy=policy,
            metrics=(metrics[0], *metrics[:-1]),
            evidence_refs=(_ref("operations-evidence", "duplicate"),),
            audit=_audit(),
        )
    with pytest.raises(ValidationError, match="threshold"):
        OperationsSLOMetricV2.model_validate(
            {
                **metrics[0].model_dump(mode="python"),
                "threshold_value": metrics[0].threshold_value + 1,
            }
        )


def test_approval_record_is_content_addressed_and_time_bound() -> None:
    record = _approval(domain=ProductionReadinessReviewDomainV2.SECURITY)
    assert production_readiness_approval_record_v2_ref(record).object_sha256 == (
        record.approval_record_sha256
    )
    with pytest.raises(ValidationError, match="validity"):
        ProductionReadinessApprovalRecordV2.model_validate(
            {
                **record.model_dump(mode="python"),
                "valid_until": record.valid_from,
            }
        )


def test_approved_report_derives_sc_013_only_for_production_authority() -> None:
    reviews = _approved_reviews()
    mechanism = _policy(ProductionReadinessReviewEvidenceClassV2.MECHANISM_VALIDATION_ONLY)
    mechanism_report = ProductionReadinessReviewReportV2.create(
        policy_ref=mechanism.to_ref(),
        evidence_class=mechanism.evidence_class,
        prerequisite=_prerequisite(),
        review_series_id="production-readiness-review-series://test",
        review_version=1,
        previous_report_ref=None,
        private_request_closure_sha256=SECOND_HASH,
        private_approval_closure_sha256=THIRD_HASH,
        domain_reviews=reviews,
        audit=_audit(),
    )
    assert mechanism_report.outcome is ProductionReadinessReviewOutcomeV2.APPROVED
    assert mechanism_report.satisfies_sc_013 is False
    assert mechanism_report.authorizes_attestation is False
    assert mechanism_report.authorizes_production_release is False

    production = _policy(ProductionReadinessReviewEvidenceClassV2.PRODUCTION_ORGANIZATION_REVIEW)
    production_report = ProductionReadinessReviewReportV2.create(
        policy_ref=production.to_ref(),
        evidence_class=production.evidence_class,
        prerequisite=_prerequisite(),
        review_series_id="production-readiness-review-series://test",
        review_version=1,
        previous_report_ref=None,
        private_request_closure_sha256=SECOND_HASH,
        private_approval_closure_sha256=THIRD_HASH,
        domain_reviews=reviews,
        audit=_audit(),
    )
    assert production_report.satisfies_sc_013 is True
    assert (
        production_readiness_review_report_v2_ref(production_report).object_sha256
        == production_report.report_sha256
    )


def test_repository_pending_report_has_no_private_or_approval_authority() -> None:
    policy = _policy(ProductionReadinessReviewEvidenceClassV2.REPOSITORY_PENDING_ONLY)
    reviews = tuple(
        ProductionReadinessDomainReviewV2.pending(
            domain=domain,
            evidence_refs=(_repository_ref(),),
            reason_codes=_pending_reasons(domain),
        )
        for domain in ProductionReadinessReviewDomainV2
    )
    report = ProductionReadinessReviewReportV2.create(
        policy_ref=policy.to_ref(),
        evidence_class=policy.evidence_class,
        prerequisite=_prerequisite(repository=True),
        review_series_id="production-readiness-review-series://repository",
        review_version=1,
        previous_report_ref=None,
        private_request_closure_sha256=None,
        private_approval_closure_sha256=None,
        domain_reviews=reviews,
        audit=_audit(),
    )
    assert report.outcome is ProductionReadinessReviewOutcomeV2.APPROVALS_PENDING
    assert report.approved_domain_count == 0
    assert report.pending_domain_count == 4
    assert report.rejected_domain_count == 0
    assert report.satisfies_sc_013 is False
    assert all(
        review.outcome is ProductionReadinessDomainOutcomeV2.APPROVAL_PENDING
        and review.approval_record is None
        for review in report.domain_reviews
    )


def test_rejection_precedes_pending() -> None:
    rejected_record = _approval(
        domain=ProductionReadinessReviewDomainV2.SECURITY,
        decision=ProductionReadinessApprovalDecisionV2.REJECTED,
    )
    reviews = (
        ProductionReadinessDomainReviewV2.from_approval(
            approval_record=rejected_record,
            evidence_refs=(_ref("domain-review-evidence", "security"),),
            operations_slo_assessment=None,
        ),
        *tuple(
            ProductionReadinessDomainReviewV2.pending(
                domain=domain,
                evidence_refs=(_ref("domain-review-evidence", domain.value.lower()),),
                reason_codes=_pending_reasons(domain),
            )
            for domain in tuple(ProductionReadinessReviewDomainV2)[1:]
        ),
    )
    report = ProductionReadinessReviewReportV2.create(
        policy_ref=_policy().to_ref(),
        evidence_class=(ProductionReadinessReviewEvidenceClassV2.MECHANISM_VALIDATION_ONLY),
        prerequisite=_prerequisite(),
        review_series_id="production-readiness-review-series://rejected",
        review_version=1,
        previous_report_ref=None,
        private_request_closure_sha256=SECOND_HASH,
        private_approval_closure_sha256=THIRD_HASH,
        domain_reviews=reviews,
        audit=_audit(),
    )
    assert report.outcome is ProductionReadinessReviewOutcomeV2.REJECTED
    assert ProductionReadinessReasonCodeV2.TRUSTED_REJECTION in (report.reason_codes)


def test_public_contracts_exclude_sensitive_approval_surfaces() -> None:
    forbidden = {
        "authority_principal",
        "approver_identity",
        "signature",
        "signature_or_registry_proof",
        "incident_contact",
        "approval_reason",
        "private_ref",
        "physical_path",
        "credential",
        "raw_sample",
        "trace_id",
        "raw_sha256",
        "private_reference",
        "model_payload",
        "final_output",
        "grader_rule",
        "hidden_condition",
        "exception_text",
    }
    models = (
        ProductionReadinessReviewPolicyV2,
        ProductionReadinessPrerequisiteSummaryV2,
        OperationsSLOPolicyV2,
        OperationsSLOMetricV2,
        OperationsSLOAssessmentV2,
        ProductionReadinessApprovalRecordV2,
        ProductionReadinessDomainReviewV2,
        ProductionReadinessReviewReportV2,
    )
    for model in models:
        assert forbidden.isdisjoint(model.model_fields)


def test_content_free_gold_records_accepted_pending_authority() -> None:
    root = Path(__file__).resolve().parents[3]
    gold = json.loads(
        (root / "evals/golden/eval_factory/readiness/r8-07-production-readiness-review-v1.json").read_text()
    )
    pending = gold["repository_pending"]
    assert pending["accepted_report_sha256"] == (
        "659eed1f5157d18b8ffce7b3624eba01ee2f7f93fddb7d873073726cb96f4fa5"
    )
    assert pending["expected_outcome"] == "APPROVALS_PENDING"
    assert pending["pending_domain_count"] == 4
    assert pending["satisfies_sc_013"] is False
    assert pending["authorizes_attestation"] is False
    assert pending["authorizes_production_release"] is False
    assert pending["external_verification_count"] == 0
    assert pending["model_or_provider_execution_count"] == 0
    serialized = json.dumps(gold).casefold()
    for forbidden_value in (
        "principal://",
        "/private/",
        "bearer ",
        "api_key",
        "raw_trace_id",
        "private_reference://",
    ):
        assert forbidden_value not in serialized
