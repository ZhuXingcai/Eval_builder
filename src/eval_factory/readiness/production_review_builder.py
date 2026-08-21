from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

from pydantic import ValidationError

from eval_factory.contracts.canary_regression_v2 import (
    CanaryRegressionOutcomeV2,
)
from eval_factory.contracts.concurrency_experiment_v2 import (
    ConcurrencyExperimentOutcomeV2,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.label_quality_v2 import LabelQualityOutcomeV2
from eval_factory.contracts.production_readiness_review_v2 import (
    PRODUCTION_READINESS_DOMAIN_ORDER,
    PRODUCTION_READINESS_REPOSITORY_PENDING_SHA256,
    ProductionReadinessApprovalRecordV2,
    ProductionReadinessDomainReviewV2,
    ProductionReadinessPrerequisiteSummaryV2,
    ProductionReadinessReasonCodeV2,
    ProductionReadinessReviewDomainV2,
    ProductionReadinessReviewEvidenceClassV2,
    ProductionReadinessReviewPolicyV2,
    validate_production_readiness_review_policy_v2_identity,
)
from eval_factory.contracts.real_trace_stability_v2 import (
    RealTraceStabilityOutcomeV2,
)
from eval_factory.readiness.production_review_models import (
    ProductionApprovalBundleV1,
    ProductionApprovalProofV1,
    ProductionReadinessReviewRequestV1,
    RepositoryPendingReviewEvidenceV1,
    TrustedProductionApprovalAuthorityV1,
    TrustedProductionApprovalRegistryV1,
    production_readiness_review_request_v1_ref,
    trusted_production_approval_registry_v1_ref,
    validate_production_readiness_review_request_v1_slo_closure,
)

_ACCEPTED_REPORTS = {
    "R8_02": (
        "label-quality-evaluation-report",
        "f5b6f5f6cab238f9c7a34196a06769393bbd4da6db75424c5dc43d34a98a58a7",
        "STATISTICAL_GATE_PENDING",
        "REPOSITORY_PENDING_ONLY",
        False,
    ),
    "R8_03": (
        "canary-regression-report",
        "375b519dc3b40d3c8aab09d9a04574deca73ec80138c2fcaa2f1714eb1edbd4f",
        "PASSED",
        "CANARY_REGRESSION_ONLY",
        True,
    ),
    "R8_04": (
        "real-trace-stability-report",
        "730f151435d2472cfe456e9df350c2663e8702951bd1eb1e8f0dcb45b315374f",
        "STATISTICAL_GATE_PENDING",
        "REAL_TRACE_STABILITY_ONLY",
        False,
    ),
    "R8_06": (
        "concurrency-experiment-report",
        "842c375004468b7258ef5750ca39e778ec9a0be363e3729c74044b35c24b74a7",
        "RECOMMENDATION_PENDING",
        "CANARY_WORKER_STORAGE_ONLY",
        False,
    ),
}


class ProductionReadinessReviewBuilderError(RuntimeError):
    pass


class ProductionReadinessReviewPolicyError(ProductionReadinessReviewBuilderError):
    pass


class ProductionReadinessReviewIntegrityError(ProductionReadinessReviewBuilderError):
    pass


class ProductionReadinessReviewAuthorizationError(ProductionReadinessReviewBuilderError):
    pass


@dataclass(frozen=True, slots=True)
class ProductionReadinessReviewCompilation:
    policy: ProductionReadinessReviewPolicyV2
    prerequisite: ProductionReadinessPrerequisiteSummaryV2
    trusted_registry: TrustedProductionApprovalRegistryV1 | None
    request: ProductionReadinessReviewRequestV1 | None
    approval_bundle: ProductionApprovalBundleV1 | None
    domain_reviews: tuple[ProductionReadinessDomainReviewV2, ...]
    evidence_sha256: str
    evidence_class: ProductionReadinessReviewEvidenceClassV2
    private_request_closure_sha256: str | None
    private_approval_closure_sha256: str | None


class ProductionReadinessReviewBuilder:
    def compile_repository_pending(
        self,
        *,
        payload: bytes,
        policy: ProductionReadinessReviewPolicyV2,
    ) -> ProductionReadinessReviewCompilation:
        _validate_policy(policy)
        if policy.evidence_class is not ProductionReadinessReviewEvidenceClassV2.REPOSITORY_PENDING_ONLY:
            raise ProductionReadinessReviewPolicyError(
                "repository pending evidence class differs from policy"
            )
        digest = hashlib.sha256(payload).hexdigest()
        if (
            digest != PRODUCTION_READINESS_REPOSITORY_PENDING_SHA256
            or policy.repository_pending_evidence_ref.object_sha256 != digest
        ):
            raise ProductionReadinessReviewPolicyError(
                "repository evidence differs from approved pending authority"
            )
        try:
            evidence = RepositoryPendingReviewEvidenceV1.model_validate_json(payload)
        except (ValidationError, ValueError) as exc:
            raise ProductionReadinessReviewPolicyError("repository pending evidence is invalid") from exc
        self._validate_repository_evidence(evidence)
        refs = {value.stage: _accepted_report_ref(value.stage) for value in evidence.predecessors}
        prerequisite = ProductionReadinessPrerequisiteSummaryV2(
            label_quality_report_ref=refs["R8_02"],
            canary_regression_report_ref=refs["R8_03"],
            real_trace_stability_report_ref=refs["R8_04"],
            concurrency_experiment_report_ref=refs["R8_06"],
            release_profile_decision_ref=policy.release_profile_decision_ref,
            repository_evidence_ref=policy.repository_pending_evidence_ref,
            label_quality_outcome=(LabelQualityOutcomeV2.STATISTICAL_GATE_PENDING),
            canary_regression_outcome=CanaryRegressionOutcomeV2.PASSED,
            real_trace_stability_outcome=(RealTraceStabilityOutcomeV2.STATISTICAL_GATE_PENDING),
            concurrency_experiment_outcome=(ConcurrencyExperimentOutcomeV2.RECOMMENDATION_PENDING),
            satisfies_sc_010=False,
            satisfies_sc_011=False,
            satisfies_sc_012=False,
        )
        domain_reviews = tuple(
            ProductionReadinessDomainReviewV2.pending(
                domain=domain,
                evidence_refs=(policy.repository_pending_evidence_ref,),
                reason_codes=_pending_repository_reasons(domain),
            )
            for domain in PRODUCTION_READINESS_DOMAIN_ORDER
        )
        return ProductionReadinessReviewCompilation(
            policy=policy,
            prerequisite=prerequisite,
            trusted_registry=None,
            request=None,
            approval_bundle=None,
            domain_reviews=domain_reviews,
            evidence_sha256=digest,
            evidence_class=policy.evidence_class,
            private_request_closure_sha256=None,
            private_approval_closure_sha256=None,
        )

    def compile_full(
        self,
        *,
        policy: ProductionReadinessReviewPolicyV2,
        request: ProductionReadinessReviewRequestV1,
        trusted_registry: TrustedProductionApprovalRegistryV1,
    ) -> ProductionReadinessReviewCompilation:
        _validate_policy(policy)
        if policy.evidence_class is ProductionReadinessReviewEvidenceClassV2.REPOSITORY_PENDING_ONLY:
            raise ProductionReadinessReviewPolicyError("repository pending policy cannot admit a full review")
        try:
            request_ref = production_readiness_review_request_v1_ref(request)
            registry_ref = trusted_production_approval_registry_v1_ref(trusted_registry)
        except ValueError as exc:
            raise ProductionReadinessReviewIntegrityError("full review authority is stale") from exc
        try:
            validate_production_readiness_review_request_v1_slo_closure(request)
        except ValueError as exc:
            raise ProductionReadinessReviewPolicyError(
                "operations SLO closure differs from review policy"
            ) from exc
        if policy.trusted_authority_registry_ref != registry_ref:
            raise ProductionReadinessReviewAuthorizationError(
                "trusted authority registry differs from policy"
            )
        if not _contains_time(
            trusted_registry.valid_from,
            trusted_registry.valid_until,
            request.evaluated_at,
        ):
            raise ProductionReadinessReviewAuthorizationError("trusted authority registry is not current")
        if request.approval_bundle is not None and len(request.approval_bundle.proofs) > policy.max_approvals:
            raise ProductionReadinessReviewPolicyError("approval bundle exceeds policy limits")
        if (
            len(request.canonical_json()) > policy.max_private_bytes
            or len(trusted_registry.canonical_json()) > policy.max_private_bytes
        ):
            raise ProductionReadinessReviewPolicyError("full review material exceeds policy limits")
        prerequisite = _full_prerequisite(policy, request)
        authorities = {value.authority_principal: value for value in trusted_registry.authorities}
        proofs = (
            {value.domain: value for value in request.approval_bundle.proofs}
            if request.approval_bundle is not None
            else {}
        )
        domain_reviews = tuple(
            self._compile_domain(
                domain=domain,
                policy=policy,
                request=request,
                registry=trusted_registry,
                authority_by_principal=authorities,
                proof=proofs.get(domain),
            )
            for domain in PRODUCTION_READINESS_DOMAIN_ORDER
        )
        approval_digest = (
            request.approval_bundle.bundle_sha256
            if request.approval_bundle is not None
            else hashlib.sha256(b"[]").hexdigest()
        )
        return ProductionReadinessReviewCompilation(
            policy=policy,
            prerequisite=prerequisite,
            trusted_registry=trusted_registry,
            request=request,
            approval_bundle=request.approval_bundle,
            domain_reviews=domain_reviews,
            evidence_sha256=request.request_sha256,
            evidence_class=policy.evidence_class,
            private_request_closure_sha256=request_ref.object_sha256,
            private_approval_closure_sha256=approval_digest,
        )

    @staticmethod
    def _validate_repository_evidence(
        evidence: RepositoryPendingReviewEvidenceV1,
    ) -> None:
        for value in evidence.predecessors:
            expected = _ACCEPTED_REPORTS[value.stage]
            if (
                value.report_sha256,
                value.outcome,
                value.claim_scope,
                value.satisfies_stage_gate,
            ) != expected[1:]:
                raise ProductionReadinessReviewPolicyError(
                    "repository predecessor differs from accepted authority"
                )

    def _compile_domain(
        self,
        *,
        domain: ProductionReadinessReviewDomainV2,
        policy: ProductionReadinessReviewPolicyV2,
        request: ProductionReadinessReviewRequestV1,
        registry: TrustedProductionApprovalRegistryV1,
        authority_by_principal: dict[
            str,
            TrustedProductionApprovalAuthorityV1,
        ],
        proof: ProductionApprovalProofV1 | None,
    ) -> ProductionReadinessDomainReviewV2:
        evidence_refs = _domain_evidence_refs(domain, request)
        if len(evidence_refs) > policy.max_evidence_refs:
            raise ProductionReadinessReviewPolicyError("domain evidence exceeds policy limits")
        operations = (
            request.operations_slo_assessment
            if domain is ProductionReadinessReviewDomainV2.OPERATIONS_SLO
            else None
        )
        if proof is None:
            return ProductionReadinessDomainReviewV2.pending(
                domain=domain,
                evidence_refs=evidence_refs,
                reason_codes=_missing_domain_reasons(
                    domain,
                    operations_present=operations is not None,
                ),
                operations_slo_assessment=operations,
            )
        authority = authority_by_principal.get(proof.authority_principal)
        if authority is None:
            raise ProductionReadinessReviewAuthorizationError("approval authority is not trusted")
        self._validate_authority(
            domain=domain,
            proof=proof,
            authority=authority,
            registry=registry,
            evaluated_at=request.evaluated_at,
        )
        expected_subjects = _review_subject_refs(policy, request)
        expected_policies = _domain_policy_refs(domain, policy, request)
        if (
            proof.review_series_id != request.review_series_id
            or proof.review_version != request.review_version
            or proof.subject_refs != expected_subjects
            or proof.evidence_refs != evidence_refs
            or proof.policy_refs != expected_policies
        ):
            raise ProductionReadinessReviewPolicyError("approval proof scope differs from review authority")
        if not _contains_time(
            proof.valid_from,
            proof.valid_until,
            request.evaluated_at,
        ):
            return ProductionReadinessDomainReviewV2.pending(
                domain=domain,
                evidence_refs=evidence_refs,
                reason_codes=(ProductionReadinessReasonCodeV2.APPROVAL_EXPIRED,),
                operations_slo_assessment=operations,
            )
        record = ProductionReadinessApprovalRecordV2.create(
            domain=domain,
            decision=proof.decision,
            review_series_id=proof.review_series_id,
            review_version=proof.review_version,
            authority_registry_ref=registry.to_ref(),
            subject_refs=proof.subject_refs,
            evidence_refs=proof.evidence_refs,
            policy_refs=proof.policy_refs,
            private_proof_commitment_sha256=proof.proof_value_sha256,
            issued_at=proof.issued_at,
            valid_from=proof.valid_from,
            valid_until=proof.valid_until,
            audit=proof.audit,
        )
        return ProductionReadinessDomainReviewV2.from_approval(
            approval_record=record,
            evidence_refs=evidence_refs,
            operations_slo_assessment=operations,
        )

    @staticmethod
    def _validate_authority(
        *,
        domain: ProductionReadinessReviewDomainV2,
        proof: ProductionApprovalProofV1,
        authority: TrustedProductionApprovalAuthorityV1,
        registry: TrustedProductionApprovalRegistryV1,
        evaluated_at: datetime,
    ) -> None:
        if proof.authority_principal == registry.project_owner_principal:
            raise ProductionReadinessReviewAuthorizationError(
                "project owner cannot self-sign production approval"
            )
        if (
            domain not in authority.approved_domains
            or proof.domain is not domain
            or proof.authority_role != authority.authority_role
            or proof.organization_policy_id != authority.organization_policy_id
            or proof.organization_policy_version != authority.organization_policy_version
            or proof.verification_method != authority.verification_method
            or proof.proof_value_sha256 != authority.registry_evidence_sha256
        ):
            raise ProductionReadinessReviewAuthorizationError("approval authority grant differs from proof")
        if not _contains_time(
            authority.valid_from,
            authority.valid_until,
            evaluated_at,
        ):
            raise ProductionReadinessReviewAuthorizationError("approval authority grant is not current")


def _validate_policy(policy: ProductionReadinessReviewPolicyV2) -> None:
    try:
        validate_production_readiness_review_policy_v2_identity(policy)
    except ValueError as exc:
        raise ProductionReadinessReviewPolicyError("production readiness review policy is stale") from exc


def _accepted_report_ref(stage: str) -> ObjectRef:
    object_type, digest, *_ = _ACCEPTED_REPORTS[stage]
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://sha256/{digest}",
        object_version="v2",
        object_sha256=digest,
    )


def _pending_repository_reasons(
    domain: ProductionReadinessReviewDomainV2,
) -> tuple[ProductionReadinessReasonCodeV2, ...]:
    values = {ProductionReadinessReasonCodeV2.APPROVAL_MISSING}
    if domain is ProductionReadinessReviewDomainV2.RELEASE_PERMISSION:
        values.add(ProductionReadinessReasonCodeV2.RELEASE_PERMISSION_MISSING)
    if domain is ProductionReadinessReviewDomainV2.OPERATIONS_SLO:
        values.add(ProductionReadinessReasonCodeV2.OPERATIONS_SLO_POLICY_MISSING)
        values.add(ProductionReadinessReasonCodeV2.PREDECESSOR_EVIDENCE_PENDING)
    return tuple(sorted(values, key=lambda value: value.value))


def _missing_domain_reasons(
    domain: ProductionReadinessReviewDomainV2,
    *,
    operations_present: bool,
) -> tuple[ProductionReadinessReasonCodeV2, ...]:
    values = {ProductionReadinessReasonCodeV2.APPROVAL_MISSING}
    if domain is ProductionReadinessReviewDomainV2.RELEASE_PERMISSION:
        values.add(ProductionReadinessReasonCodeV2.RELEASE_PERMISSION_MISSING)
    if domain is ProductionReadinessReviewDomainV2.OPERATIONS_SLO and not operations_present:
        values.add(ProductionReadinessReasonCodeV2.OPERATIONS_SLO_POLICY_MISSING)
    return tuple(sorted(values, key=lambda value: value.value))


def _full_prerequisite(
    policy: ProductionReadinessReviewPolicyV2,
    request: ProductionReadinessReviewRequestV1,
) -> ProductionReadinessPrerequisiteSummaryV2:
    return ProductionReadinessPrerequisiteSummaryV2(
        label_quality_report_ref=request.label_quality_report.to_ref(),
        canary_regression_report_ref=request.canary_regression_report.to_ref(),
        real_trace_stability_report_ref=(request.real_trace_stability_report.to_ref()),
        concurrency_experiment_report_ref=(request.concurrency_experiment_report.to_ref()),
        release_profile_decision_ref=policy.release_profile_decision_ref,
        repository_evidence_ref=None,
        label_quality_outcome=request.label_quality_report.outcome,
        canary_regression_outcome=request.canary_regression_report.outcome,
        real_trace_stability_outcome=(request.real_trace_stability_report.outcome),
        concurrency_experiment_outcome=(request.concurrency_experiment_report.outcome),
        label_quality_claim_scope=request.label_quality_report.evidence_class.value,
        satisfies_sc_010=request.label_quality_report.satisfies_sc_010,
        satisfies_sc_011=(request.real_trace_stability_report.outcome is RealTraceStabilityOutcomeV2.PASSED),
        satisfies_sc_012=(request.concurrency_experiment_report.satisfies_sc_012),
    )


def _review_subject_refs(
    policy: ProductionReadinessReviewPolicyV2,
    request: ProductionReadinessReviewRequestV1,
) -> tuple[ObjectRef, ...]:
    return _sorted_refs(
        (
            policy.to_ref(),
            request.label_quality_report.to_ref(),
            request.canary_regression_report.to_ref(),
            request.real_trace_stability_report.to_ref(),
            request.concurrency_experiment_report.to_ref(),
            policy.release_profile_decision_ref,
            *((request.operations_slo_policy.to_ref(),) if request.operations_slo_policy is not None else ()),
            *(
                (request.operations_slo_assessment.to_ref(),)
                if request.operations_slo_assessment is not None
                else ()
            ),
        )
    )


def _domain_evidence_refs(
    domain: ProductionReadinessReviewDomainV2,
    request: ProductionReadinessReviewRequestV1,
) -> tuple[ObjectRef, ...]:
    values = {
        ProductionReadinessReviewDomainV2.SECURITY: (
            request.canary_regression_report.to_ref(),
            request.label_quality_report.to_ref(),
            request.real_trace_stability_report.to_ref(),
        ),
        ProductionReadinessReviewDomainV2.PRIVACY: (
            request.canary_regression_report.to_ref(),
            request.label_quality_report.to_ref(),
            request.real_trace_stability_report.to_ref(),
        ),
        ProductionReadinessReviewDomainV2.RELEASE_PERMISSION: (request.canary_regression_report.to_ref(),),
        ProductionReadinessReviewDomainV2.OPERATIONS_SLO: (
            request.concurrency_experiment_report.to_ref(),
            *(
                (request.operations_slo_assessment.to_ref(),)
                if request.operations_slo_assessment is not None
                else ()
            ),
        ),
    }[domain]
    return _sorted_refs(values)


def _domain_policy_refs(
    domain: ProductionReadinessReviewDomainV2,
    policy: ProductionReadinessReviewPolicyV2,
    request: ProductionReadinessReviewRequestV1,
) -> tuple[ObjectRef, ...]:
    values = {
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
            *(
                (
                    request.operations_slo_policy.to_ref(),
                    request.operations_slo_policy.approval_policy_ref,
                )
                if request.operations_slo_policy is not None
                else ()
            ),
        ),
    }[domain]
    return _sorted_refs(values)


def _contains_time(
    valid_from: datetime,
    valid_until: datetime,
    current: datetime,
) -> bool:
    return valid_from <= current < valid_until


def _sorted_refs(values: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    unique = {_ref_key(value): value for value in values}
    return tuple(unique[key] for key in sorted(unique))


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )
