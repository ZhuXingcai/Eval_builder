from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

from pydantic import ValidationError

from eval_factory.contracts.concurrency_experiment_v2 import (
    ConcurrencyExperimentOutcomeV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.label_quality_v2 import (
    LabelQualityEvidenceClassV2,
    LabelQualityOutcomeV2,
)
from eval_factory.contracts.production_attestation_v2 import (
    PRODUCTION_ATTESTATION_GATE_ORDER,
    ProductionAttestationEvidenceClassV2,
    ProductionReadinessAttestationPolicyV2,
    ProductionReadinessAttestationPrerequisiteV2,
    ProductionReadinessAttestationReasonCodeV2,
    ProductionReadinessGateAssessmentV2,
    ProductionReadinessGateOutcomeV2,
    ProductionReadinessGateV2,
    ProductionReadinessInvalidationPolicyV2,
    ProductionReadinessVersionSetV2,
    validate_production_readiness_attestation_policy_v2_identity,
    validate_production_readiness_invalidation_policy_v2_identity,
)
from eval_factory.contracts.production_readiness_review_v2 import (
    ProductionReadinessReviewEvidenceClassV2,
    ProductionReadinessReviewOutcomeV2,
)
from eval_factory.contracts.real_trace_stability_v2 import (
    RealTraceStabilityOutcomeV2,
)
from eval_factory.readiness.production_attestation_models import (
    AttestationIssuanceBundleV1,
    AttestationIssuanceProofV1,
    ProductionReadinessAttestationRequestV1,
    RepositoryPendingAttestationEvidenceV1,
    TrustedAttestationIssuerRegistryV1,
    TrustedAttestationIssuerV1,
    production_readiness_attestation_request_v1_ref,
    trusted_attestation_issuer_registry_v1_ref,
)

PRODUCTION_ATTESTATION_REPOSITORY_PENDING_SHA256 = (
    "4627db5d82f3da0cf52f53ae1a3df9580e36fbe140223dffd3bfb80f0d02d068"
)

_ACCEPTED_REPORTS = {
    "R8_02": (
        "label-quality-evaluation-report",
        "f5b6f5f6cab238f9c7a34196a06769393bbd4da6db75424c5dc43d34a98a58a7",
        "STATISTICAL_GATE_PENDING",
        "REPOSITORY_PENDING_ONLY",
    ),
    "R8_04": (
        "real-trace-stability-report",
        "730f151435d2472cfe456e9df350c2663e8702951bd1eb1e8f0dcb45b315374f",
        "STATISTICAL_GATE_PENDING",
        "REAL_TRACE_STABILITY_ONLY",
    ),
    "R8_06": (
        "concurrency-experiment-report",
        "842c375004468b7258ef5750ca39e778ec9a0be363e3729c74044b35c24b74a7",
        "RECOMMENDATION_PENDING",
        "CANARY_WORKER_STORAGE_ONLY",
    ),
    "R8_07": (
        "production-readiness-review-report",
        "659eed1f5157d18b8ffce7b3624eba01ee2f7f93fddb7d873073726cb96f4fa5",
        "APPROVALS_PENDING",
        "REPOSITORY_PENDING_ONLY",
    ),
}


class ProductionAttestationBuilderError(RuntimeError):
    pass


class ProductionAttestationPolicyError(ProductionAttestationBuilderError):
    pass


class ProductionAttestationIntegrityError(ProductionAttestationBuilderError):
    pass


class ProductionAttestationAuthorizationError(ProductionAttestationBuilderError):
    pass


@dataclass(frozen=True, slots=True)
class ProductionAttestationCompilation:
    policy: ProductionReadinessAttestationPolicyV2
    invalidation_policy: ProductionReadinessInvalidationPolicyV2
    version_set: ProductionReadinessVersionSetV2
    prerequisite: ProductionReadinessAttestationPrerequisiteV2
    trusted_registry: TrustedAttestationIssuerRegistryV1 | None
    request: ProductionReadinessAttestationRequestV1 | None
    issuance_bundle: AttestationIssuanceBundleV1 | None
    approved_issuer_principals: tuple[str, ...]
    issuance_valid_until: datetime | None
    evidence_sha256: str
    evidence_class: ProductionAttestationEvidenceClassV2
    private_request_closure_sha256: str | None
    private_issuer_closure_sha256: str | None


class ProductionAttestationBuilder:
    def compile_repository_pending(
        self,
        *,
        payload: bytes,
        policy: ProductionReadinessAttestationPolicyV2,
    ) -> ProductionAttestationCompilation:
        _validate_policy(policy)
        if policy.evidence_class is not (ProductionAttestationEvidenceClassV2.REPOSITORY_PENDING_ONLY):
            raise ProductionAttestationPolicyError("repository pending evidence class differs from policy")
        digest = hashlib.sha256(payload).hexdigest()
        if (
            digest != PRODUCTION_ATTESTATION_REPOSITORY_PENDING_SHA256
            or policy.repository_pending_evidence_ref.object_sha256 != digest
        ):
            raise ProductionAttestationPolicyError(
                "repository evidence differs from approved pending authority"
            )
        try:
            evidence = RepositoryPendingAttestationEvidenceV1.model_validate_json(payload)
        except (ValidationError, ValueError) as exc:
            raise ProductionAttestationPolicyError(
                "repository pending attestation evidence is invalid"
            ) from exc
        self._validate_repository_evidence(evidence)
        version_set = ProductionReadinessVersionSetV2.create(
            system_version=policy.expected_system_version,
            base_contract_manifest_ref=policy.base_contract_manifest_ref,
            overlay_contract_manifest_ref=policy.overlay_contract_manifest_ref,
            schema_manifest_refs=policy.required_schema_manifest_refs,
            policy_refs=policy.required_policy_refs,
            release_profile_decision_ref=policy.release_profile_decision_ref,
            resource_policy_approval_ref=None,
            lifecycle_approval_ref=None,
            incident_route_ref=None,
            audit=policy.audit,
        )
        gate_assessments = tuple(
            ProductionReadinessGateAssessmentV2.create(
                gate=gate,
                outcome=ProductionReadinessGateOutcomeV2.PENDING,
                reason_codes=(_pending_reason(gate),),
                evidence_refs=(policy.repository_pending_evidence_ref,),
                valid_until=None,
            )
            for gate in PRODUCTION_ATTESTATION_GATE_ORDER
        )
        prerequisite = ProductionReadinessAttestationPrerequisiteV2.create(
            gate_assessments=gate_assessments,
            audit=policy.audit,
        )
        return ProductionAttestationCompilation(
            policy=policy,
            invalidation_policy=policy.invalidation_policy,
            version_set=version_set,
            prerequisite=prerequisite,
            trusted_registry=None,
            request=None,
            issuance_bundle=None,
            approved_issuer_principals=(),
            issuance_valid_until=None,
            evidence_sha256=digest,
            evidence_class=policy.evidence_class,
            private_request_closure_sha256=None,
            private_issuer_closure_sha256=None,
        )

    def compile_full(
        self,
        *,
        policy: ProductionReadinessAttestationPolicyV2,
        request: ProductionReadinessAttestationRequestV1,
        trusted_registry: TrustedAttestationIssuerRegistryV1,
    ) -> ProductionAttestationCompilation:
        try:
            policy = ProductionReadinessAttestationPolicyV2.model_validate_json(policy.canonical_json())
            request = ProductionReadinessAttestationRequestV1.model_validate_json(request.canonical_json())
            trusted_registry = TrustedAttestationIssuerRegistryV1.model_validate_json(
                trusted_registry.canonical_json()
            )
        except (ValidationError, ValueError) as exc:
            raise ProductionAttestationIntegrityError("full attestation authority is invalid") from exc
        _validate_policy(policy)
        if policy.evidence_class is not (ProductionAttestationEvidenceClassV2.PRODUCTION_VERIFIED):
            raise ProductionAttestationPolicyError("full attestation requires production-verified policy")
        try:
            request_ref = production_readiness_attestation_request_v1_ref(request)
            registry_ref = trusted_attestation_issuer_registry_v1_ref(trusted_registry)
        except ValueError as exc:
            raise ProductionAttestationIntegrityError("full attestation authority is stale") from exc
        if policy.trusted_issuer_registry_ref != registry_ref:
            raise ProductionAttestationAuthorizationError("trusted issuer registry differs from policy")
        if policy.minimum_issuer_count != trusted_registry.minimum_independent_issuer_count:
            raise ProductionAttestationAuthorizationError(
                "attestation issuer minimum differs from trusted registry"
            )
        requested_duration = (request.requested_valid_until - request.requested_valid_from).total_seconds()
        if requested_duration > policy.maximum_validity_seconds:
            raise ProductionAttestationPolicyError("requested attestation validity exceeds policy")
        if requested_duration > policy.maximum_validity_seconds:
            raise ProductionAttestationPolicyError("requested attestation validity exceeds policy")
        self._validate_version_set(policy, request.version_set)
        if any(
            len(values) > policy.max_evidence_refs
            for values in (
                request.version_set.schema_manifest_refs,
                request.version_set.policy_refs,
                request.sc_014_closure.evidence_refs,
                request.issuance_bundle.proofs,
            )
        ):
            raise ProductionAttestationPolicyError("attestation evidence inventory exceeds policy")
        prerequisite = self._compile_full_prerequisite(request)
        if prerequisite != request.expected_prerequisite:
            raise ProductionAttestationPolicyError(
                "request prerequisite differs from exact predecessor authority"
            )
        approved, valid_until, rejected, issuer_expired = self._validate_issuance_bundle(
            policy=policy,
            request=request,
            registry=trusted_registry,
            prerequisite=prerequisite,
        )
        if rejected:
            prerequisite = _replace_final_gate(
                prerequisite,
                outcome=ProductionReadinessGateOutcomeV2.FAILED,
                reason=ProductionReadinessAttestationReasonCodeV2.TRUSTED_REJECTION,
                audit=policy.audit,
            )
        elif issuer_expired:
            prerequisite = _replace_final_gate(
                prerequisite,
                outcome=ProductionReadinessGateOutcomeV2.PENDING,
                reason=(ProductionReadinessAttestationReasonCodeV2.ISSUER_AUTHORITY_EXPIRED),
                audit=policy.audit,
            )
            valid_until = None
        if len(request.canonical_json()) > policy.max_private_bytes:
            raise ProductionAttestationPolicyError("attestation request exceeds private byte limit")
        return ProductionAttestationCompilation(
            policy=policy,
            invalidation_policy=policy.invalidation_policy,
            version_set=request.version_set,
            prerequisite=prerequisite,
            trusted_registry=trusted_registry,
            request=request,
            issuance_bundle=request.issuance_bundle,
            approved_issuer_principals=approved,
            issuance_valid_until=valid_until,
            evidence_sha256=request.request_sha256,
            evidence_class=policy.evidence_class,
            private_request_closure_sha256=request_ref.object_sha256,
            private_issuer_closure_sha256=request.issuance_bundle.bundle_sha256,
        )

    @staticmethod
    def _validate_repository_evidence(
        evidence: RepositoryPendingAttestationEvidenceV1,
    ) -> None:
        for item in evidence.predecessors:
            expected = _ACCEPTED_REPORTS[item.stage]
            if (
                item.report_sha256,
                item.outcome,
                item.claim_scope,
            ) != expected[1:]:
                raise ProductionAttestationPolicyError(
                    "repository predecessor differs from accepted authority"
                )

    @staticmethod
    def _validate_version_set(
        policy: ProductionReadinessAttestationPolicyV2,
        version_set: ProductionReadinessVersionSetV2,
    ) -> None:
        try:
            validate_production_readiness_attestation_policy_v2_identity(policy)
        except ValueError as exc:
            raise ProductionAttestationPolicyError("attestation policy is stale") from exc
        if (
            version_set.system_version != policy.expected_system_version
            or version_set.base_contract_manifest_ref != policy.base_contract_manifest_ref
            or version_set.overlay_contract_manifest_ref != policy.overlay_contract_manifest_ref
            or version_set.schema_manifest_refs != policy.required_schema_manifest_refs
            or version_set.policy_refs != policy.required_policy_refs
            or version_set.release_profile_decision_ref != policy.release_profile_decision_ref
            or version_set.resource_policy_approval_ref is None
            or version_set.lifecycle_approval_ref is None
            or version_set.incident_route_ref is None
        ):
            raise ProductionAttestationPolicyError("attestation version set differs from production policy")

    @staticmethod
    def _compile_full_prerequisite(
        request: ProductionReadinessAttestationRequestV1,
    ) -> ProductionReadinessAttestationPrerequisiteV2:
        evaluated_at = request.evaluated_at
        valid_until = min(
            request.requested_valid_until,
            request.sc_014_closure.valid_until,
        )
        label = request.label_quality_report
        label_pass = bool(
            label.outcome is LabelQualityOutcomeV2.PASSED
            and label.evidence_class is LabelQualityEvidenceClassV2.PRODUCTION_INDEPENDENT_TEST
            and label.satisfies_sc_010
        )
        label_failed = bool(
            label.outcome is LabelQualityOutcomeV2.FAILED
            and label.evidence_class is LabelQualityEvidenceClassV2.PRODUCTION_INDEPENDENT_TEST
        )
        stability = request.real_trace_stability_report
        stability_pass = bool(
            stability.outcome is RealTraceStabilityOutcomeV2.PASSED
            and stability.corpus_summary.eligible_unique_trace_count >= 100
        )
        stability_failed = bool(
            stability.outcome is RealTraceStabilityOutcomeV2.STABILITY_THRESHOLD_NOT_MET
            and stability.corpus_summary.eligible_unique_trace_count >= 100
        )
        concurrency = request.concurrency_experiment_report
        resource_pass = bool(
            concurrency.outcome is ConcurrencyExperimentOutcomeV2.NON_MODEL_RECOMMENDED
            and concurrency.recommendation is not None
            and request.version_set.resource_policy_approval_ref is not None
        )
        review = request.production_readiness_review_report
        review_pass = bool(
            review.outcome is ProductionReadinessReviewOutcomeV2.APPROVED
            and review.evidence_class
            is ProductionReadinessReviewEvidenceClassV2.PRODUCTION_ORGANIZATION_REVIEW
            and review.satisfies_sc_013
            and review.valid_until is not None
            and evaluated_at < review.valid_until
        )
        review_failed = bool(
            review.outcome is ProductionReadinessReviewOutcomeV2.REJECTED
            and review.evidence_class
            is ProductionReadinessReviewEvidenceClassV2.PRODUCTION_ORGANIZATION_REVIEW
        )
        sc014 = request.sc_014_closure
        sc014_pass = bool(
            sc014.open_count == 0
            and sc014.lifecycle_approval_ref == request.version_set.lifecycle_approval_ref
            and sc014.incident_route_ref == request.version_set.incident_route_ref
            and sc014.evaluated_at <= evaluated_at < sc014.valid_until
        )
        values = (
            (
                ProductionReadinessGateV2.SC_010_LABEL_QUALITY,
                label_pass,
                label_failed,
                label.to_ref(),
                ProductionReadinessAttestationReasonCodeV2.SC_010_PENDING,
            ),
            (
                ProductionReadinessGateV2.SC_011_REAL_TRACE_STABILITY,
                stability_pass,
                stability_failed,
                stability.to_ref(),
                ProductionReadinessAttestationReasonCodeV2.SC_011_PENDING,
            ),
            (
                ProductionReadinessGateV2.SC_012_RESOURCE_POLICY,
                resource_pass,
                False,
                concurrency.to_ref(),
                ProductionReadinessAttestationReasonCodeV2.SC_012_PENDING,
            ),
            (
                ProductionReadinessGateV2.SC_013_ORGANIZATION_APPROVALS,
                review_pass,
                review_failed,
                review.to_ref(),
                ProductionReadinessAttestationReasonCodeV2.SC_013_PENDING,
            ),
            (
                ProductionReadinessGateV2.SC_014_ZERO_OPEN_GATES,
                sc014_pass,
                False,
                sc014.to_ref(),
                ProductionReadinessAttestationReasonCodeV2.SC_014_UNVERIFIED,
            ),
        )
        gates = tuple(
            ProductionReadinessGateAssessmentV2.create(
                gate=gate,
                outcome=(
                    ProductionReadinessGateOutcomeV2.PASSED
                    if passed
                    else (
                        ProductionReadinessGateOutcomeV2.FAILED
                        if failed
                        else ProductionReadinessGateOutcomeV2.PENDING
                    )
                ),
                reason_codes=(
                    (ProductionReadinessAttestationReasonCodeV2.NONE,)
                    if passed
                    else ((ProductionReadinessAttestationReasonCodeV2.GATE_FAILED,) if failed else (reason,))
                ),
                evidence_refs=(evidence_ref,),
                valid_until=valid_until if passed else None,
            )
            for gate, passed, failed, evidence_ref, reason in values
        )
        return ProductionReadinessAttestationPrerequisiteV2.create(
            gate_assessments=gates,
            audit=request.audit,
        )

    @staticmethod
    def _validate_issuance_bundle(
        *,
        policy: ProductionReadinessAttestationPolicyV2,
        request: ProductionReadinessAttestationRequestV1,
        registry: TrustedAttestationIssuerRegistryV1,
        prerequisite: ProductionReadinessAttestationPrerequisiteV2,
    ) -> tuple[tuple[str, ...], datetime | None, bool, bool]:
        bundle = request.issuance_bundle
        if len(bundle.proofs) != policy.minimum_issuer_count:
            raise ProductionAttestationAuthorizationError(
                "issuance bundle does not have exact independent issuer count"
            )
        issuer_by_principal = {item.issuer_principal: item for item in registry.issuers}
        expected_evidence = _sorted_refs(
            (
                request.label_quality_report.to_ref(),
                request.real_trace_stability_report.to_ref(),
                request.concurrency_experiment_report.to_ref(),
                request.production_readiness_review_report.to_ref(),
                request.sc_014_closure.to_ref(),
            )
        )
        expected_policies = _sorted_refs(
            (
                policy.to_ref(),
                request.version_set.release_profile_decision_ref,
                request.version_set.resource_policy_approval_ref,
                request.version_set.lifecycle_approval_ref,
                request.version_set.incident_route_ref,
                *request.version_set.policy_refs,
            )
        )
        approved: list[str] = []
        organizations: set[str] = set()
        valid_until = min(
            request.requested_valid_until,
            request.sc_014_closure.valid_until,
            registry.valid_until,
            prerequisite.valid_until or request.requested_valid_until,
        )
        rejected = False
        issuer_expired = not _covers_start(
            registry.valid_from,
            registry.valid_until,
            requested_valid_from=request.requested_valid_from,
            evaluated_at=request.evaluated_at,
        )
        for proof in bundle.proofs:
            issuer = issuer_by_principal.get(proof.issuer_principal)
            if issuer is None:
                raise ProductionAttestationAuthorizationError("attestation issuer is not trusted")
            _validate_issuer_proof(
                issuer=issuer,
                proof=proof,
                request=request,
                prerequisite=prerequisite,
                expected_evidence=expected_evidence,
                expected_policies=expected_policies,
            )
            proof_current = _covers_start(
                proof.valid_from,
                proof.valid_until,
                requested_valid_from=request.requested_valid_from,
                evaluated_at=request.evaluated_at,
            )
            issuer_current = _covers_start(
                issuer.valid_from,
                issuer.valid_until,
                requested_valid_from=request.requested_valid_from,
                evaluated_at=request.evaluated_at,
            )
            issuer_expired = issuer_expired or not proof_current or not issuer_current
            organizations.add(issuer.issuer_organization)
            valid_until = min(valid_until, issuer.valid_until, proof.valid_until)
            if proof.decision == "REJECT":
                rejected = True
            elif proof_current and issuer_current:
                approved.append(proof.issuer_principal)
        if len(organizations) < policy.minimum_issuer_count:
            raise ProductionAttestationAuthorizationError(
                "attestation issuers are not organizationally independent"
            )
        return (
            tuple(sorted(approved)),
            (None if issuer_expired else valid_until),
            rejected,
            issuer_expired,
        )


def _validate_policy(
    policy: ProductionReadinessAttestationPolicyV2,
) -> None:
    try:
        validate_production_readiness_attestation_policy_v2_identity(policy)
        validate_production_readiness_invalidation_policy_v2_identity(policy.invalidation_policy)
    except ValueError as exc:
        raise ProductionAttestationPolicyError("production attestation policy identity is stale") from exc


def _validate_issuer_proof(
    *,
    issuer: TrustedAttestationIssuerV1,
    proof: AttestationIssuanceProofV1,
    request: ProductionReadinessAttestationRequestV1,
    prerequisite: ProductionReadinessAttestationPrerequisiteV2,
    expected_evidence: tuple[ObjectRef, ...],
    expected_policies: tuple[ObjectRef, ...],
) -> None:
    if (
        proof.issuer_organization != issuer.issuer_organization
        or proof.issuer_role != issuer.issuer_role
        or proof.issuance_scope != issuer.issuance_scope
        or proof.issuance_scope_version != issuer.issuance_scope_version
        or proof.verification_method != issuer.verification_method
        or proof.proof_value_sha256 != issuer.registry_evidence_sha256
    ):
        raise ProductionAttestationAuthorizationError("attestation issuer grant differs from proof")
    if (
        proof.attestation_series_id != request.attestation_series_id
        or proof.attestation_version != request.attestation_version
        or proof.version_set_ref != request.version_set.to_ref()
        or proof.prerequisite_ref != prerequisite.to_ref()
        or proof.evidence_refs != expected_evidence
        or proof.policy_refs != expected_policies
    ):
        raise ProductionAttestationPolicyError("attestation issuance proof scope differs from request")


def _pending_reason(
    gate: ProductionReadinessGateV2,
) -> ProductionReadinessAttestationReasonCodeV2:
    return {
        ProductionReadinessGateV2.SC_010_LABEL_QUALITY: (
            ProductionReadinessAttestationReasonCodeV2.SC_010_PENDING
        ),
        ProductionReadinessGateV2.SC_011_REAL_TRACE_STABILITY: (
            ProductionReadinessAttestationReasonCodeV2.SC_011_PENDING
        ),
        ProductionReadinessGateV2.SC_012_RESOURCE_POLICY: (
            ProductionReadinessAttestationReasonCodeV2.SC_012_PENDING
        ),
        ProductionReadinessGateV2.SC_013_ORGANIZATION_APPROVALS: (
            ProductionReadinessAttestationReasonCodeV2.SC_013_PENDING
        ),
        ProductionReadinessGateV2.SC_014_ZERO_OPEN_GATES: (
            ProductionReadinessAttestationReasonCodeV2.SC_014_UNVERIFIED
        ),
    }[gate]


def _covers_start(
    valid_from: datetime,
    valid_until: datetime,
    *,
    requested_valid_from: datetime,
    evaluated_at: datetime,
) -> bool:
    return valid_from <= requested_valid_from <= evaluated_at < valid_until


def _replace_final_gate(
    prerequisite: ProductionReadinessAttestationPrerequisiteV2,
    *,
    outcome: ProductionReadinessGateOutcomeV2,
    reason: ProductionReadinessAttestationReasonCodeV2,
    audit: ContractAudit,
) -> ProductionReadinessAttestationPrerequisiteV2:
    gates = list(prerequisite.gate_assessments)
    final = gates[-1]
    gates[-1] = ProductionReadinessGateAssessmentV2.create(
        gate=ProductionReadinessGateV2.SC_014_ZERO_OPEN_GATES,
        outcome=outcome,
        reason_codes=(reason,),
        evidence_refs=final.evidence_refs,
        valid_until=None,
    )
    return ProductionReadinessAttestationPrerequisiteV2.create(
        gate_assessments=tuple(gates),
        audit=audit,
    )


def _sorted_refs(values: tuple[ObjectRef | None, ...]) -> tuple[ObjectRef, ...]:
    present = tuple(item for item in values if item is not None)
    unique = {_ref_key(value): value for value in present}
    return tuple(unique[key] for key in sorted(unique))


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


__all__ = [
    "PRODUCTION_ATTESTATION_REPOSITORY_PENDING_SHA256",
    "ProductionAttestationAuthorizationError",
    "ProductionAttestationBuilder",
    "ProductionAttestationBuilderError",
    "ProductionAttestationCompilation",
    "ProductionAttestationIntegrityError",
    "ProductionAttestationPolicyError",
]
