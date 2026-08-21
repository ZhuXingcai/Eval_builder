from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from production_review_fixtures import (
    full_inputs,
    slo_assessment,
    slo_policy,
)

from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.production_readiness_review_v2 import (
    PRODUCTION_READINESS_REPOSITORY_PENDING_SHA256,
    ProductionReadinessDomainOutcomeV2,
    ProductionReadinessReasonCodeV2,
    ProductionReadinessReviewDomainV2,
    ProductionReadinessReviewEvidenceClassV2,
    ProductionReadinessReviewOutcomeV2,
    ProductionReadinessReviewPolicyV2,
)
from eval_factory.readiness.production_review import (
    ProductionReadinessReviewer,
)
from eval_factory.readiness.production_review_builder import (
    ProductionReadinessReviewAuthorizationError,
    ProductionReadinessReviewBuilder,
    ProductionReadinessReviewPolicyError,
)
from eval_factory.readiness.production_review_models import (
    ProductionReadinessReviewRequestV1,
    production_readiness_review_request_v1_carried_sha256,
)

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_PATH = ROOT / "evals/manifests/r8-07-repository-pending-evidence-v1.json"
HASH = "a" * 64
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
        created_by="r8-07-builder-test",
        governing_versions=(
            VersionBinding(
                component="production-readiness-review",
                version="production-readiness-review/r8-07-v1",
            ),
        ),
    )


def _policy(
    evidence_class: ProductionReadinessReviewEvidenceClassV2 = (
        ProductionReadinessReviewEvidenceClassV2.REPOSITORY_PENDING_ONLY
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
        repository_pending_evidence_ref=_ref(
            "production-readiness-review-repository-evidence",
            "r8-07/v1",
            version="json/v1",
            digest=PRODUCTION_READINESS_REPOSITORY_PENDING_SHA256,
        ),
        trusted_authority_registry_ref=(
            None
            if evidence_class is ProductionReadinessReviewEvidenceClassV2.REPOSITORY_PENDING_ONLY
            else _ref(
                "production-approval-authority-registry",
                "fixture",
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


def test_repository_pending_compilation_binds_exact_predecessors() -> None:
    payload = EVIDENCE_PATH.read_bytes()
    compilation = ProductionReadinessReviewBuilder().compile_repository_pending(
        payload=payload,
        policy=_policy(),
    )

    assert compilation.evidence_sha256 == (PRODUCTION_READINESS_REPOSITORY_PENDING_SHA256)
    assert compilation.request is None
    assert compilation.trusted_registry is None
    assert compilation.approval_bundle is None
    assert compilation.private_request_closure_sha256 is None
    assert compilation.private_approval_closure_sha256 is None
    assert len(compilation.domain_reviews) == 4
    assert all(
        review.outcome is ProductionReadinessDomainOutcomeV2.APPROVAL_PENDING
        and review.approval_record is None
        for review in compilation.domain_reviews
    )
    assert all(
        ProductionReadinessReasonCodeV2.APPROVAL_MISSING in review.reason_codes
        for review in compilation.domain_reviews
    )
    assert (
        compilation.prerequisite.label_quality_report_ref.object_sha256
        == "f5b6f5f6cab238f9c7a34196a06769393bbd4da6db75424c5dc43d34a98a58a7"
    )
    assert (
        compilation.prerequisite.canary_regression_report_ref.object_sha256
        == "375b519dc3b40d3c8aab09d9a04574deca73ec80138c2fcaa2f1714eb1edbd4f"
    )
    assert (
        compilation.prerequisite.real_trace_stability_report_ref.object_sha256
        == "730f151435d2472cfe456e9df350c2663e8702951bd1eb1e8f0dcb45b315374f"
    )
    assert (
        compilation.prerequisite.concurrency_experiment_report_ref.object_sha256
        == "842c375004468b7258ef5750ca39e778ec9a0be363e3729c74044b35c24b74a7"
    )


def test_repository_pending_rejects_changed_bytes_and_wrong_scope() -> None:
    payload = EVIDENCE_PATH.read_bytes()
    builder = ProductionReadinessReviewBuilder()
    with pytest.raises(
        ProductionReadinessReviewPolicyError,
        match="approved pending authority",
    ):
        builder.compile_repository_pending(
            payload=payload + b"\n",
            policy=_policy(),
        )
    with pytest.raises(
        ProductionReadinessReviewPolicyError,
        match="evidence class",
    ):
        builder.compile_repository_pending(
            payload=payload,
            policy=_policy(ProductionReadinessReviewEvidenceClassV2.MECHANISM_VALIDATION_ONLY),
        )


def test_repository_pending_rejects_semantic_manifest_drift() -> None:
    payload = EVIDENCE_PATH.read_bytes()
    changed = payload.replace(
        b'"organization_approval_authority_present": false',
        b'"organization_approval_authority_present": true ',
    )
    assert len(changed) == len(payload)
    with pytest.raises(ProductionReadinessReviewPolicyError):
        ProductionReadinessReviewBuilder().compile_repository_pending(
            payload=changed,
            policy=_policy(),
        )


def test_full_review_compiles_four_current_approvals() -> None:
    policy, request, registry = full_inputs()
    compilation = ProductionReadinessReviewBuilder().compile_full(
        policy=policy,
        request=request,
        trusted_registry=registry,
    )
    report = ProductionReadinessReviewer().evaluate(
        compilation=compilation,
        policy=policy,
        audit=request.audit,
    )

    assert all(
        review.outcome is ProductionReadinessDomainOutcomeV2.APPROVED and review.approval_record is not None
        for review in compilation.domain_reviews
    )
    assert report.outcome is ProductionReadinessReviewOutcomeV2.APPROVED
    assert report.satisfies_sc_013 is False
    assert compilation.private_request_closure_sha256 == request.request_sha256
    assert compilation.private_approval_closure_sha256 == (
        request.approval_bundle.bundle_sha256  # type: ignore[union-attr]
    )


@pytest.mark.parametrize(
    ("inputs", "expected"),
    (
        (
            {"missing_domain": ProductionReadinessReviewDomainV2.PRIVACY},
            ProductionReadinessReviewOutcomeV2.APPROVALS_PENDING,
        ),
        (
            {"expired_domain": ProductionReadinessReviewDomainV2.SECURITY},
            ProductionReadinessReviewOutcomeV2.APPROVALS_PENDING,
        ),
        (
            {"rejected_domain": (ProductionReadinessReviewDomainV2.RELEASE_PERMISSION)},
            ProductionReadinessReviewOutcomeV2.REJECTED,
        ),
        (
            {"failed_slo": True},
            ProductionReadinessReviewOutcomeV2.REJECTED,
        ),
        (
            {"incomplete_slo": True},
            ProductionReadinessReviewOutcomeV2.APPROVALS_PENDING,
        ),
    ),
)
def test_full_review_outcome_matrix(
    inputs: dict[str, object],
    expected: ProductionReadinessReviewOutcomeV2,
) -> None:
    policy, request, registry = full_inputs(**inputs)  # type: ignore[arg-type]
    compilation = ProductionReadinessReviewBuilder().compile_full(
        policy=policy,
        request=request,
        trusted_registry=registry,
    )
    report = ProductionReadinessReviewer().evaluate(
        compilation=compilation,
        policy=policy,
        audit=request.audit,
    )
    assert report.outcome is expected


def test_project_owner_cannot_self_sign() -> None:
    policy, request, registry = full_inputs(self_sign_domain=ProductionReadinessReviewDomainV2.SECURITY)
    with pytest.raises(
        ProductionReadinessReviewAuthorizationError,
        match="self-sign",
    ):
        ProductionReadinessReviewBuilder().compile_full(
            policy=policy,
            request=request,
            trusted_registry=registry,
        )


def test_submitted_proof_must_match_registry_pinned_digest() -> None:
    policy, request, registry = full_inputs(forged_proof_domain=ProductionReadinessReviewDomainV2.PRIVACY)
    with pytest.raises(
        ProductionReadinessReviewAuthorizationError,
        match="differs from proof",
    ):
        ProductionReadinessReviewBuilder().compile_full(
            policy=policy,
            request=request,
            trusted_registry=registry,
        )


@pytest.mark.parametrize(
    "case",
    ("threshold-drift", "expired-policy"),
)
def test_full_review_rejects_untrusted_operations_slo_policy_closure(
    case: str,
) -> None:
    policy, request, registry = full_inputs()
    operations_policy = (
        slo_policy(expired=True) if case == "expired-policy" else request.operations_slo_policy
    )
    assert operations_policy is not None
    operations_assessment = slo_assessment(
        policy=operations_policy,
        threshold_drift=case == "threshold-drift",
    )
    changed = request.model_copy(
        update={
            "operations_slo_policy": operations_policy,
            "operations_slo_assessment": operations_assessment,
        }
    )
    request_digest = production_readiness_review_request_v1_carried_sha256(changed)
    changed = changed.model_copy(
        update={
            "request_id": (f"production-readiness-review-request://sha256/{request_digest}"),
            "request_sha256": request_digest,
        }
    )
    assert isinstance(changed, ProductionReadinessReviewRequestV1)
    with pytest.raises(
        ProductionReadinessReviewPolicyError,
        match="operations SLO closure",
    ):
        ProductionReadinessReviewBuilder().compile_full(
            policy=policy,
            request=changed,
            trusted_registry=registry,
        )
