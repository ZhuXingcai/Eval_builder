from __future__ import annotations

from eval_factory.contracts.core import ContractAudit
from eval_factory.contracts.production_readiness_review_v2 import (
    ProductionReadinessReviewEvidenceClassV2,
    ProductionReadinessReviewPolicyV2,
    ProductionReadinessReviewReportV2,
    validate_production_readiness_review_policy_v2_identity,
)
from eval_factory.readiness.production_review_builder import (
    ProductionReadinessReviewCompilation,
)

REPOSITORY_REVIEW_SERIES_ID = "production-readiness-review-series://repository/r8-07"


class ProductionReadinessReviewError(RuntimeError):
    pass


class ProductionReadinessReviewer:
    def evaluate(
        self,
        *,
        compilation: ProductionReadinessReviewCompilation,
        policy: ProductionReadinessReviewPolicyV2,
        audit: ContractAudit,
    ) -> ProductionReadinessReviewReportV2:
        try:
            validate_production_readiness_review_policy_v2_identity(policy)
        except ValueError as exc:
            raise ProductionReadinessReviewError("production readiness review policy is stale") from exc
        if compilation.policy != policy or compilation.evidence_class is not policy.evidence_class:
            raise ProductionReadinessReviewError("production readiness compilation differs from policy")
        request = compilation.request
        if request is None:
            if (
                policy.evidence_class is not ProductionReadinessReviewEvidenceClassV2.REPOSITORY_PENDING_ONLY
                or compilation.trusted_registry is not None
                or compilation.approval_bundle is not None
            ):
                raise ProductionReadinessReviewError("pending compilation carries full review authority")
            series_id = REPOSITORY_REVIEW_SERIES_ID
            version = 1
            previous_report_ref = None
        else:
            if compilation.trusted_registry is None:
                raise ProductionReadinessReviewError("full compilation trusted registry is missing")
            series_id = request.review_series_id
            version = request.review_version
            previous_report_ref = request.previous_report_ref
        report = ProductionReadinessReviewReportV2.create(
            policy_ref=policy.to_ref(),
            evidence_class=policy.evidence_class,
            prerequisite=compilation.prerequisite,
            review_series_id=series_id,
            review_version=version,
            previous_report_ref=previous_report_ref,
            private_request_closure_sha256=(compilation.private_request_closure_sha256),
            private_approval_closure_sha256=(compilation.private_approval_closure_sha256),
            domain_reviews=compilation.domain_reviews,
            audit=audit,
        )
        if len(report.canonical_json()) > policy.max_report_bytes:
            raise ProductionReadinessReviewError("production readiness report exceeds policy limit")
        return report
