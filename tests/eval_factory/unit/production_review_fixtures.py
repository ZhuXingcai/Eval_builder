from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

from concurrency_experiment_fixtures import report as concurrency_report

from eval_factory.contracts.canary_regression_v2 import (
    CANARY_REGRESSION_POLICY_VERSION,
    CanaryRegressionCaseResultV2,
    CanaryRegressionExpectationV2,
    CanaryRegressionObservedOutcomeV2,
    CanaryRegressionPolicyV2,
    CanaryRegressionReasonCodeV2,
    CanaryRegressionReportV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.label_quality_v2 import (
    LABEL_QUALITY_REPOSITORY_PENDING_SHA256,
    LabelQualityEvaluationReportV2,
    LabelQualityEvidenceClassV2,
    LabelQualityPrerequisiteOutcomeV2,
    LabelQualityPrerequisiteSummaryV2,
)
from eval_factory.contracts.orchestration import ItemStatus, JobStatus
from eval_factory.contracts.orchestration_v2 import StageNameV2
from eval_factory.contracts.production_readiness_review_v2 import (
    OPERATIONS_SLO_METRIC_ORDER,
    PRODUCTION_READINESS_REPOSITORY_PENDING_SHA256,
    OperationsSLOAssessmentV2,
    OperationsSLOMetricNameV2,
    OperationsSLOMetricV2,
    OperationsSLOPolicyV2,
    ProductionReadinessApprovalDecisionV2,
    ProductionReadinessReviewDomainV2,
    ProductionReadinessReviewEvidenceClassV2,
    ProductionReadinessReviewPolicyV2,
    operations_slo_assessment_v2_carried_sha256,
)
from eval_factory.contracts.real_trace_stability_v2 import (
    REAL_TRACE_STABILITY_POLICY_VERSION,
    RealTraceStabilityCaseOutcomeV2,
    RealTraceStabilityCaseSummaryV2,
    RealTraceStabilityCategoryCountV2,
    RealTraceStabilityCorpusSummaryV2,
    RealTraceStabilityFaultPointV2,
    RealTraceStabilityPolicyV2,
    RealTraceStabilityReasonCodeV2,
    RealTraceStabilityReportV2,
)
from eval_factory.contracts.trace import ParseQuality
from eval_factory.readiness.production_review_models import (
    ProductionApprovalBundleV1,
    ProductionApprovalProofV1,
    ProductionReadinessReviewRequestV1,
    TrustedProductionApprovalAuthorityV1,
    TrustedProductionApprovalRegistryV1,
)

NOW = datetime(2026, 8, 3, tzinfo=UTC)
PROJECT_OWNER = "principal://project-owner"


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v2",
    value_digest: str | None = None,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://fixture/{suffix}",
        object_version=version,
        object_sha256=(
            value_digest if value_digest is not None else digest(f"{object_type}:{suffix}:{version}")
        ),
    )


def audit(
    component: str = "production-readiness-review",
    version: str = "production-readiness-review/r8-07-v1",
) -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="r8-07-fixture",
        governing_versions=(VersionBinding(component=component, version=version),),
    )


def label_report() -> LabelQualityEvaluationReportV2:
    evidence_ref = ref(
        "label-quality-repository-evidence",
        "r8-01",
        version="json/v1",
        value_digest=LABEL_QUALITY_REPOSITORY_PENDING_SHA256,
    )
    prerequisite = LabelQualityPrerequisiteSummaryV2(
        outcome=LabelQualityPrerequisiteOutcomeV2.STATISTICAL_GATE_PENDING,
        freeze_result_ref=None,
        repository_evidence_ref=evidence_ref,
        dataset_manifest_ref=None,
        access_receipt_ref=None,
        dataset_version=None,
        selected_member_count=0,
        available_semantic_count=91,
        required_semantic_count=100,
        shortage_count=1,
        material_verified=False,
    )
    return LabelQualityEvaluationReportV2.create(
        policy_ref=ref("label-quality-evaluation-policy", "fixture"),
        evidence_class=LabelQualityEvidenceClassV2.REPOSITORY_PENDING_ONLY,
        prerequisite=prerequisite,
        observation_closure_sha256=None,
        result_closure_sha256=None,
        structured_summaries=(),
        semantic_summaries=(),
        audit=audit("label-quality-evaluation", "label-quality-evaluation/r8-02-v1"),
    )


def _canary_cohort_ref() -> ObjectRef:
    return ObjectRef(
        object_type="development-canary-manifest",
        object_id="development-canary-manifest://v4",
        object_version="v4",
        object_sha256=("1a409bbb92ef0fc075d8dd2144f36cfd5829976ab6718c20db3a8e9f2d61a6a6"),
    )


def canary_report() -> CanaryRegressionReportV2:
    policy = CanaryRegressionPolicyV2.create(
        cohort_manifest_ref=_canary_cohort_ref(),
        max_case_refs=1_000,
        max_report_bytes=2_000_000,
        audit=audit("canary-regression", CANARY_REGRESSION_POLICY_VERSION),
    )
    cases = tuple(_canary_case(index) for index in range(1, 25))
    return CanaryRegressionReportV2.create(
        cohort_manifest_ref=_canary_cohort_ref(),
        policy_ref=policy.to_ref(),
        template_ref=ref(
            "canary-regression-template",
            "fixture",
            version="private-v1",
        ),
        case_results=cases,
        audit=audit("canary-regression", CANARY_REGRESSION_POLICY_VERSION),
    )


def _canary_case(index: int) -> CanaryRegressionCaseResultV2:
    instance_id = f"LH_{index:03d}"
    strict = index <= 15
    return CanaryRegressionCaseResultV2.create(
        instance_id=instance_id,
        source_trace_ref=ref("trace-source", instance_id, version="1.0.0"),
        signal_quality=("strict_events" if strict else "heuristic_requires_annotation"),
        expectation=(
            CanaryRegressionExpectationV2.MUST_SUCCEED
            if strict
            else CanaryRegressionExpectationV2.ANY_TYPED_TERMINAL
        ),
        observed_outcome=CanaryRegressionObservedOutcomeV2.SUCCEEDED,
        reason_code=CanaryRegressionReasonCodeV2.NONE,
        highest_reached_stage=StageNameV2.ITEM_QUALITY,
        child_manifest_ref=ref("r6-canary-execution-manifest", instance_id),
        child_dataset_result_ref=ref("r6-canary-dataset-result", instance_id),
        job_id=f"job://r8-07/{instance_id}",
        job_status=JobStatus.SUCCEEDED,
        item_id=f"item://r8-07/{instance_id}",
        item_status=ItemStatus.APPROVED,
        stage_result_refs=tuple(
            ref("stage-result", f"{instance_id}/{stage.value}", version="v1")
            for stage in (
                StageNameV2.TRACE_INDEX,
                StageNameV2.SAFETY,
                StageNameV2.LABEL,
                StageNameV2.TASK_AUTHORING,
                StageNameV2.ATTACHMENT,
                StageNameV2.ITEM_QUALITY,
            )
        ),
        quality_result_refs=(ref("item-quality-compilation-result", instance_id),),
        package_manifest_refs=(ref("final-package-manifest", instance_id),),
        audit_report_ref=ref("batch-audit-report", instance_id),
        work_unit_count=10,
        work_lease_count=10,
        stage_run_count=9,
        attempt_count=10,
        retry_count=0,
        resume_count=0,
        provider_invocation_count=1,
        audit=audit("canary-regression", CANARY_REGRESSION_POLICY_VERSION),
    )


def stability_report(
    *,
    eligible_unique_trace_count: int = 91,
) -> RealTraceStabilityReportV2:
    policy = RealTraceStabilityPolicyV2.create(
        source_manifest_ref=ObjectRef(
            object_type="real-trace-source-manifest",
            object_id="real-trace-source-manifest://manifest.csv",
            object_version="csv/v1",
            object_sha256=("0d9df6c3935e00e5e15d486c9afe8cc2f581b63cf8f99ec42c1d3216cc74ee04"),
        ),
        max_sources=10_000,
        max_source_bytes=100_000_000,
        max_total_source_bytes=10_000_000_000,
        max_private_bytes=32_000_000,
        max_report_bytes=8_000_000,
        audit=audit(
            "real-trace-stability",
            REAL_TRACE_STABILITY_POLICY_VERSION,
        ),
    )
    corpus = RealTraceStabilityCorpusSummaryV2.create(
        source_manifest_ref=policy.source_manifest_ref,
        policy_ref=policy.to_ref(),
        private_inventory_ref=ref(
            "real-trace-stability-inventory",
            "fixture",
            version="private-v1",
        ),
        eligible_unique_trace_count=eligible_unique_trace_count,
        unique_instance_count=eligible_unique_trace_count,
        unique_raw_hash_count=eligible_unique_trace_count,
        total_raw_bytes=eligible_unique_trace_count * 1_000_000,
        minimum_raw_bytes=900_000,
        maximum_raw_bytes=1_100_000,
        category_counts=(
            RealTraceStabilityCategoryCountV2(
                category="science",
                count=eligible_unique_trace_count,
            ),
        ),
        audit=audit(
            "real-trace-stability",
            REAL_TRACE_STABILITY_POLICY_VERSION,
        ),
    )
    cases = tuple(_stability_case(index) for index in range(eligible_unique_trace_count))
    return RealTraceStabilityReportV2.create(
        policy_ref=policy.to_ref(),
        corpus_summary=corpus,
        private_result_set_ref=ref(
            "real-trace-stability-result-set",
            "fixture",
            version="private-v1",
        ),
        case_summaries=cases,
        audit=audit(
            "real-trace-stability",
            REAL_TRACE_STABILITY_POLICY_VERSION,
        ),
    )


def _stability_case(index: int) -> RealTraceStabilityCaseSummaryV2:
    return RealTraceStabilityCaseSummaryV2.create(
        private_case_result_ref=ref(
            "real-trace-stability-case-result",
            f"case-{index:03d}",
            version="private-v1",
        ),
        outcome=RealTraceStabilityCaseOutcomeV2.STABLE,
        reason_code=RealTraceStabilityReasonCodeV2.NONE,
        parse_quality=ParseQuality.STRICT,
        assigned_fault_point=tuple(RealTraceStabilityFaultPointV2)[index % 4],
        fault_observed=True,
        resume_succeeded=True,
        replay_stable=True,
        attempt_count=1,
        unexpected_retry_count=0,
        resume_count=1,
        replay_count=1,
        stage_result_count=1,
        completion_witness_count=1,
        audit=audit(
            "real-trace-stability",
            REAL_TRACE_STABILITY_POLICY_VERSION,
        ),
    )


def slo_policy(*, expired: bool = False) -> OperationsSLOPolicyV2:
    return OperationsSLOPolicyV2.create(
        approval_policy_ref=ref(
            "operations-slo-approval-policy",
            "fixture",
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
        valid_from=NOW - timedelta(days=60) if expired else NOW,
        valid_until=NOW - timedelta(days=1) if expired else NOW + timedelta(days=30),
        audit=audit(),
    )


def slo_assessment(
    *,
    policy: OperationsSLOPolicyV2 | None = None,
    failed: bool = False,
    incomplete: bool = False,
    threshold_drift: bool = False,
) -> OperationsSLOAssessmentV2:
    policy = policy or slo_policy()
    metrics: list[OperationsSLOMetricV2] = []
    for name in OPERATIONS_SLO_METRIC_ORDER:
        if incomplete and name is OperationsSLOMetricNameV2.COST_MICRO_USD_PER_JOB:
            metric = OperationsSLOMetricV2.unavailable(
                policy=policy,
                metric=name,
            )
        else:
            observed = policy.threshold_for(name)
            if failed and name is OperationsSLOMetricNameV2.ERROR_RATE_BASIS_POINTS:
                observed += 1
            metric = OperationsSLOMetricV2.measured(
                policy=policy,
                metric=name,
                observed_value=observed,
                sample_count=policy.minimum_sample_count,
                evidence_refs=(ref("operations-metric-evidence", name.value),),
            )
        metrics.append(metric)
    assessment = OperationsSLOAssessmentV2.create(
        policy=policy,
        metrics=tuple(metrics),
        evidence_refs=(ref("operations-evidence", "fixture"),),
        audit=audit(),
    )
    if not threshold_drift:
        return assessment
    first = assessment.metrics[0]
    changed = first.model_copy(
        update={
            "threshold_value": first.threshold_value + 1,
            "observed_value": first.threshold_value + 1,
        }
    )
    drifted = assessment.model_copy(update={"metrics": (changed, *assessment.metrics[1:])})
    digest_value = operations_slo_assessment_v2_carried_sha256(drifted)
    return drifted.model_copy(
        update={
            "assessment_id": f"operations-slo-assessment://sha256/{digest_value}",
            "assessment_sha256": digest_value,
        }
    )


def registry(
    *,
    self_sign_domain: ProductionReadinessReviewDomainV2 | None = None,
) -> TrustedProductionApprovalRegistryV1:
    authorities = tuple(
        TrustedProductionApprovalAuthorityV1(
            authority_principal=(
                PROJECT_OWNER
                if domain is self_sign_domain
                else f"principal://{domain.value.lower()}-authority"
            ),
            authority_role=f"role://{domain.value.lower()}-approver",
            approved_domains=(domain,),
            organization_policy_id=f"organization-policy://{domain.value.lower()}",
            organization_policy_version="v1",
            valid_from=NOW,
            valid_until=NOW + timedelta(days=60),
            registry_evidence_sha256=digest(f"detached-registry-proof::{domain.value}::fixture"),
        )
        for domain in ProductionReadinessReviewDomainV2
    )
    return TrustedProductionApprovalRegistryV1.create(
        registry_version="r8-07-fixture-v1",
        project_owner_principal=PROJECT_OWNER,
        authorities=authorities,
        registry_proof_commitment_sha256=digest("registry-proof"),
        valid_from=NOW,
        valid_until=NOW + timedelta(days=60),
        audit=audit(),
    )


def review_policy(
    authority_registry: TrustedProductionApprovalRegistryV1,
    *,
    evidence_class: ProductionReadinessReviewEvidenceClassV2 = (
        ProductionReadinessReviewEvidenceClassV2.MECHANISM_VALIDATION_ONLY
    ),
) -> ProductionReadinessReviewPolicyV2:
    return ProductionReadinessReviewPolicyV2.create(
        data_classification_policy_refs=(
            ref("data-classification-policy", "v1", version="v1"),
            ref("data-classification-policy", "v2", version="v2"),
        ),
        user_approval_policy_ref=ref(
            "user-approval-policy",
            "v2",
            version="v2",
        ),
        release_profile_decision_ref=ref(
            "release-profile-decision",
            "lh-v1",
            version="v1",
        ),
        repository_pending_evidence_ref=ref(
            "production-readiness-review-repository-evidence",
            "r8-07",
            version="json/v1",
            value_digest=PRODUCTION_READINESS_REPOSITORY_PENDING_SHA256,
        ),
        trusted_authority_registry_ref=authority_registry.to_ref(),
        evidence_class=evidence_class,
        max_approvals=4,
        max_evidence_refs=64,
        max_private_bytes=100_000_000,
        max_report_bytes=10_000_000,
        audit=audit(),
    )


def full_inputs(
    *,
    missing_domain: ProductionReadinessReviewDomainV2 | None = None,
    rejected_domain: ProductionReadinessReviewDomainV2 | None = None,
    expired_domain: ProductionReadinessReviewDomainV2 | None = None,
    self_sign_domain: ProductionReadinessReviewDomainV2 | None = None,
    forged_proof_domain: ProductionReadinessReviewDomainV2 | None = None,
    failed_slo: bool = False,
    incomplete_slo: bool = False,
    expired_slo_policy: bool = False,
    drifted_slo_threshold: bool = False,
    review_version: int = 1,
    previous_report_ref: ObjectRef | None = None,
    evidence_class: ProductionReadinessReviewEvidenceClassV2 = (
        ProductionReadinessReviewEvidenceClassV2.MECHANISM_VALIDATION_ONLY
    ),
) -> tuple[
    ProductionReadinessReviewPolicyV2,
    ProductionReadinessReviewRequestV1,
    TrustedProductionApprovalRegistryV1,
]:
    authority_registry = registry(self_sign_domain=self_sign_domain)
    policy = review_policy(
        authority_registry,
        evidence_class=evidence_class,
    )
    label = label_report()
    canary = canary_report()
    stability = stability_report()
    concurrency = concurrency_report()
    operations_policy = slo_policy(expired=expired_slo_policy)
    operations_assessment = slo_assessment(
        policy=operations_policy,
        failed=failed_slo,
        incomplete=incomplete_slo,
        threshold_drift=drifted_slo_threshold,
    )
    subjects = tuple(
        sorted(
            (
                policy.to_ref(),
                label.to_ref(),
                canary.to_ref(),
                stability.to_ref(),
                concurrency.to_ref(),
                policy.release_profile_decision_ref,
                operations_policy.to_ref(),
                operations_assessment.to_ref(),
            ),
            key=_ref_key,
        )
    )
    proofs: list[ProductionApprovalProofV1] = []
    authority_by_domain = {
        authority.approved_domains[0]: authority for authority in authority_registry.authorities
    }
    for domain in ProductionReadinessReviewDomainV2:
        if domain is missing_domain:
            continue
        authority = authority_by_domain[domain]
        evidence_refs = _domain_evidence(
            domain,
            label=label.to_ref(),
            canary=canary.to_ref(),
            stability=stability.to_ref(),
            concurrency=concurrency.to_ref(),
            operations=operations_assessment.to_ref(),
        )
        policy_refs = _domain_policies(
            domain,
            policy=policy,
            operations_policy=operations_policy,
        )
        valid_until = NOW - timedelta(days=1) if domain is expired_domain else NOW + timedelta(days=30)
        valid_from = NOW - timedelta(days=30) if domain is expired_domain else NOW
        proofs.append(
            ProductionApprovalProofV1.create(
                authority_principal=authority.authority_principal,
                authority_role=authority.authority_role,
                organization_policy_id=authority.organization_policy_id,
                organization_policy_version=(authority.organization_policy_version),
                domain=domain,
                decision=(
                    ProductionReadinessApprovalDecisionV2.REJECTED
                    if domain is rejected_domain
                    else ProductionReadinessApprovalDecisionV2.APPROVED
                ),
                review_series_id="production-readiness-review-series://fixture",
                review_version=review_version,
                subject_refs=subjects,
                evidence_refs=evidence_refs,
                policy_refs=policy_refs,
                proof_value=(
                    f"detached-registry-proof::{domain.value}::forged"
                    if domain is forged_proof_domain
                    else f"detached-registry-proof::{domain.value}::fixture"
                ),
                issued_at=valid_from,
                valid_from=valid_from,
                valid_until=valid_until,
                audit=audit(),
            )
        )
    bundle = (
        ProductionApprovalBundleV1.create(
            review_series_id="production-readiness-review-series://fixture",
            review_version=review_version,
            proofs=tuple(proofs),
            audit=audit(),
        )
        if proofs
        else None
    )
    request = ProductionReadinessReviewRequestV1.create(
        review_series_id="production-readiness-review-series://fixture",
        review_version=review_version,
        previous_report_ref=previous_report_ref,
        label_quality_report=label,
        canary_regression_report=canary,
        real_trace_stability_report=stability,
        concurrency_experiment_report=concurrency,
        operations_slo_policy=operations_policy,
        operations_slo_assessment=operations_assessment,
        approval_bundle=bundle,
        evaluated_at=NOW,
        audit=audit(),
    )
    return policy, request, authority_registry


def _domain_evidence(
    domain: ProductionReadinessReviewDomainV2,
    *,
    label: ObjectRef,
    canary: ObjectRef,
    stability: ObjectRef,
    concurrency: ObjectRef,
    operations: ObjectRef,
) -> tuple[ObjectRef, ...]:
    return tuple(
        sorted(
            {
                ProductionReadinessReviewDomainV2.SECURITY: (
                    canary,
                    label,
                    stability,
                ),
                ProductionReadinessReviewDomainV2.PRIVACY: (
                    canary,
                    label,
                    stability,
                ),
                ProductionReadinessReviewDomainV2.RELEASE_PERMISSION: (canary,),
                ProductionReadinessReviewDomainV2.OPERATIONS_SLO: (
                    concurrency,
                    operations,
                ),
            }[domain],
            key=_ref_key,
        )
    )


def _domain_policies(
    domain: ProductionReadinessReviewDomainV2,
    *,
    policy: ProductionReadinessReviewPolicyV2,
    operations_policy: OperationsSLOPolicyV2,
) -> tuple[ObjectRef, ...]:
    return tuple(
        sorted(
            {
                ProductionReadinessReviewDomainV2.SECURITY: (
                    policy.to_ref(),
                    *policy.data_classification_policy_refs,
                ),
                ProductionReadinessReviewDomainV2.PRIVACY: (
                    policy.to_ref(),
                    *policy.data_classification_policy_refs,
                ),
                ProductionReadinessReviewDomainV2.RELEASE_PERMISSION: (
                    policy.to_ref(),
                    policy.release_profile_decision_ref,
                ),
                ProductionReadinessReviewDomainV2.OPERATIONS_SLO: (
                    policy.to_ref(),
                    operations_policy.to_ref(),
                    operations_policy.approval_policy_ref,
                ),
            }[domain],
            key=_ref_key,
        )
    )


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )
