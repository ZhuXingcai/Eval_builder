from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.production_readiness_review_v2 import (
    PRODUCTION_READINESS_REPOSITORY_PENDING_SHA256,
    ProductionReadinessReasonCodeV2,
    ProductionReadinessReviewEvidenceClassV2,
    ProductionReadinessReviewOutcomeV2,
    ProductionReadinessReviewPolicyV2,
)
from eval_factory.readiness.production_review import (
    REPOSITORY_REVIEW_SERIES_ID,
    ProductionReadinessReviewer,
    ProductionReadinessReviewError,
)
from eval_factory.readiness.production_review_builder import (
    ProductionReadinessReviewBuilder,
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
        created_by="r8-07-review-test",
        governing_versions=(
            VersionBinding(
                component="production-readiness-review",
                version="production-readiness-review/r8-07-v1",
            ),
        ),
    )


def _policy() -> ProductionReadinessReviewPolicyV2:
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
        trusted_authority_registry_ref=None,
        evidence_class=(ProductionReadinessReviewEvidenceClassV2.REPOSITORY_PENDING_ONLY),
        max_approvals=4,
        max_evidence_refs=64,
        max_private_bytes=1_000_000,
        max_report_bytes=1_000_000,
        audit=_audit(),
    )


def _compilation():
    policy = _policy()
    return policy, ProductionReadinessReviewBuilder().compile_repository_pending(
        payload=EVIDENCE_PATH.read_bytes(),
        policy=policy,
    )


def test_repository_pending_evaluation_is_content_free_and_non_authoritative() -> None:
    policy, compilation = _compilation()
    report = ProductionReadinessReviewer().evaluate(
        compilation=compilation,
        policy=policy,
        audit=_audit(),
    )

    assert report.review_series_id == REPOSITORY_REVIEW_SERIES_ID
    assert report.review_version == 1
    assert report.outcome is ProductionReadinessReviewOutcomeV2.APPROVALS_PENDING
    assert report.pending_domain_count == 4
    assert report.approved_domain_count == 0
    assert report.rejected_domain_count == 0
    assert report.private_request_closure_sha256 is None
    assert report.private_approval_closure_sha256 is None
    assert report.satisfies_sc_013 is False
    assert report.authorizes_attestation is False
    assert report.authorizes_production_release is False
    assert ProductionReadinessReasonCodeV2.APPROVAL_MISSING in (report.reason_codes)


def test_repository_pending_report_is_stable_across_audit_time() -> None:
    policy, compilation = _compilation()
    first = ProductionReadinessReviewer().evaluate(
        compilation=compilation,
        policy=policy,
        audit=_audit(),
    )
    second_audit = _audit().model_copy(update={"created_at": datetime(2030, 1, 1, tzinfo=UTC)})
    second = ProductionReadinessReviewer().evaluate(
        compilation=compilation,
        policy=policy,
        audit=second_audit,
    )
    assert first.report_sha256 == second.report_sha256
    assert first.report_id == second.report_id


def test_evaluator_rejects_policy_promotion() -> None:
    policy, compilation = _compilation()
    promoted = ProductionReadinessReviewPolicyV2.create(
        data_classification_policy_refs=policy.data_classification_policy_refs,
        user_approval_policy_ref=policy.user_approval_policy_ref,
        release_profile_decision_ref=policy.release_profile_decision_ref,
        repository_pending_evidence_ref=policy.repository_pending_evidence_ref,
        trusted_authority_registry_ref=_ref(
            "production-approval-authority-registry",
            "forged",
            version="private-v1",
        ),
        evidence_class=(ProductionReadinessReviewEvidenceClassV2.MECHANISM_VALIDATION_ONLY),
        max_approvals=policy.max_approvals,
        max_evidence_refs=policy.max_evidence_refs,
        max_private_bytes=policy.max_private_bytes,
        max_report_bytes=policy.max_report_bytes,
        audit=_audit(),
    )
    with pytest.raises(
        ProductionReadinessReviewError,
        match="differs from policy",
    ):
        ProductionReadinessReviewer().evaluate(
            compilation=compilation,
            policy=promoted,
            audit=_audit(),
        )
