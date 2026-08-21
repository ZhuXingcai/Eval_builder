from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

from concurrency_experiment_fixtures import report as concurrency_report
from label_quality_fixtures import (
    fixture_audit,
    frozen_authority,
    frozen_request,
    observation_set,
    quality_policy,
)
from production_review_fixtures import (
    full_inputs as review_inputs,
)
from production_review_fixtures import (
    stability_report,
)

from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.label_quality_v2 import (
    LabelQualityEvidenceClassV2,
)
from eval_factory.contracts.production_attestation_v2 import (
    PRODUCTION_ATTESTATION_GATE_ORDER,
    ProductionAttestationEvidenceClassV2,
    ProductionReadinessAttestationPolicyV2,
    ProductionReadinessAttestationPrerequisiteV2,
    ProductionReadinessAttestationReasonCodeV2,
    ProductionReadinessGateAssessmentV2,
    ProductionReadinessGateOutcomeV2,
    ProductionReadinessInvalidationPolicyV2,
    ProductionReadinessVersionSetV2,
)
from eval_factory.contracts.production_readiness_review_v2 import (
    ProductionReadinessReviewEvidenceClassV2,
)
from eval_factory.readiness.production_attestation_builder import (
    PRODUCTION_ATTESTATION_REPOSITORY_PENDING_SHA256,
    ProductionAttestationBuilder,
    ProductionAttestationCompilation,
)
from eval_factory.readiness.production_attestation_models import (
    AttestationIssuanceBundleV1,
    AttestationIssuanceProofV1,
    ProductionReadinessAttestationRequestV1,
    ProductionReadinessSC014ClosureV1,
    TrustedAttestationIssuerRegistryV1,
    TrustedAttestationIssuerV1,
)
from eval_factory.readiness.production_review import ProductionReadinessReviewer
from eval_factory.readiness.production_review_builder import (
    ProductionReadinessReviewBuilder,
)
from eval_factory.statistics.label_quality import LabelQualityEvaluator
from eval_factory.statistics.label_quality_builder import (
    LabelQualityEvaluationBuilder,
)

NOW = datetime(2026, 8, 4, tzinfo=UTC)


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
        object_sha256=value_digest or digest(f"{object_type}:{suffix}:{version}"),
    )


def audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="r8-08-fixture",
        governing_versions=(
            VersionBinding(
                component="production-readiness-attestation",
                version="production-readiness-attestation/r8-08-v1",
            ),
        ),
    )


def invalidation_policy() -> ProductionReadinessInvalidationPolicyV2:
    return ProductionReadinessInvalidationPolicyV2.create(
        max_projection_revisions=1_000,
        max_invalidation_refs=128,
        audit=audit(),
    )


def version_set(
    *,
    production: bool,
    system_version: str = "env-mock-agent/0.1.0",
    extra_policy_refs: tuple[ObjectRef, ...] = (),
    release_profile_decision_ref: ObjectRef | None = None,
) -> ProductionReadinessVersionSetV2:
    return ProductionReadinessVersionSetV2.create(
        system_version=system_version,
        base_contract_manifest_ref=ref("contract-manifest", "v1", version="v1"),
        overlay_contract_manifest_ref=ref("contract-manifest", "v2", version="v2"),
        schema_manifest_refs=(
            ref("contract-schema-manifest", "v1", version="v1"),
            ref("contract-schema-manifest", "v2", version="v2"),
        ),
        policy_refs=(
            ref("compatibility-policy", "v1", version="v1"),
            ref("compatibility-policy", "v2", version="v2"),
            ref("data-classification-policy", "v1", version="v1"),
            ref("data-classification-policy", "v2", version="v2"),
            ref("user-approval-policy", "v2", version="v2"),
            *extra_policy_refs,
        ),
        release_profile_decision_ref=(
            release_profile_decision_ref
            or ref(
                "release-profile-decision",
                "lh-v1",
                version="v1",
            )
        ),
        resource_policy_approval_ref=(
            ref("production-resource-policy-approval", "fixture", version="v1") if production else None
        ),
        lifecycle_approval_ref=(
            ref("data-lifecycle-approval", "fixture", version="v1") if production else None
        ),
        incident_route_ref=(ref("incident-route", "fixture", version="v1") if production else None),
        audit=audit(),
    )


def policy(
    *,
    evidence_class: ProductionAttestationEvidenceClassV2,
    version: ProductionReadinessVersionSetV2,
    registry_ref: ObjectRef | None,
) -> ProductionReadinessAttestationPolicyV2:
    invalidation = invalidation_policy()
    return ProductionReadinessAttestationPolicyV2.create(
        expected_system_version=version.system_version,
        base_contract_manifest_ref=version.base_contract_manifest_ref,
        overlay_contract_manifest_ref=version.overlay_contract_manifest_ref,
        required_schema_manifest_refs=version.schema_manifest_refs,
        required_policy_refs=version.policy_refs,
        release_profile_decision_ref=version.release_profile_decision_ref,
        repository_pending_evidence_ref=ref(
            "production-attestation-repository-evidence",
            "r8-08",
            version="json/v1",
            value_digest=PRODUCTION_ATTESTATION_REPOSITORY_PENDING_SHA256,
        ),
        invalidation_policy=invalidation,
        trusted_issuer_registry_ref=registry_ref,
        evidence_class=evidence_class,
        minimum_issuer_count=2,
        maximum_validity_seconds=30 * 86_400,
        max_evidence_refs=128,
        max_private_bytes=100_000_000,
        max_report_bytes=10_000_000,
        audit=audit(),
    )


def passed_prerequisite(
    *,
    label_ref: ObjectRef,
    stability_ref: ObjectRef,
    concurrency_ref: ObjectRef,
    review_ref: ObjectRef,
    closure_ref: ObjectRef,
) -> ProductionReadinessAttestationPrerequisiteV2:
    evidence_by_gate = {
        PRODUCTION_ATTESTATION_GATE_ORDER[0]: label_ref,
        PRODUCTION_ATTESTATION_GATE_ORDER[1]: stability_ref,
        PRODUCTION_ATTESTATION_GATE_ORDER[2]: concurrency_ref,
        PRODUCTION_ATTESTATION_GATE_ORDER[3]: review_ref,
        PRODUCTION_ATTESTATION_GATE_ORDER[4]: closure_ref,
    }
    gates = tuple(
        ProductionReadinessGateAssessmentV2.create(
            gate=gate,
            outcome=ProductionReadinessGateOutcomeV2.PASSED,
            reason_codes=(ProductionReadinessAttestationReasonCodeV2.NONE,),
            evidence_refs=(evidence_by_gate[gate],),
            valid_until=NOW + timedelta(days=30),
        )
        for gate in PRODUCTION_ATTESTATION_GATE_ORDER
    )
    return ProductionReadinessAttestationPrerequisiteV2.create(
        gate_assessments=gates,
        audit=audit(),
    )


def registry() -> TrustedAttestationIssuerRegistryV1:
    issuers = tuple(
        TrustedAttestationIssuerV1(
            issuer_principal=f"principal://issuer-{index}",
            issuer_organization=f"organization://issuer-{index}",
            issuer_role="role://production-attestation-issuer",
            issuance_scope="PRODUCTION_READINESS_ATTESTATION",
            issuance_scope_version="v1",
            valid_from=NOW,
            valid_until=NOW + timedelta(days=60),
            registry_evidence_sha256=digest(f"issuer-proof-value-{index}"),
        )
        for index in (1, 2)
    )
    return TrustedAttestationIssuerRegistryV1.create(
        registry_version="fixture-v1",
        project_owner_principal="principal://project-owner",
        issuers=issuers,
        minimum_independent_issuer_count=2,
        registry_proof_commitment_sha256=digest("registry"),
        valid_from=NOW,
        valid_until=NOW + timedelta(days=60),
        audit=audit(),
    )


def production_inputs(
    tmp_path: Path,
    *,
    rejected: bool = False,
    expired_proof: bool = False,
    attestation_version: int = 1,
    previous_result_ref: ObjectRef | None = None,
    previous_projection_ref: ObjectRef | None = None,
    extra_policy_refs: tuple[ObjectRef, ...] = (),
    release_profile_decision_ref: ObjectRef | None = None,
) -> tuple[
    ProductionReadinessAttestationPolicyV2,
    ProductionReadinessAttestationRequestV1,
    TrustedAttestationIssuerRegistryV1,
]:
    trusted = registry()
    versions = version_set(
        production=True,
        extra_policy_refs=extra_policy_refs,
        release_profile_decision_ref=release_profile_decision_ref,
    )
    attestation_policy = policy(
        evidence_class=ProductionAttestationEvidenceClassV2.PRODUCTION_VERIFIED,
        version=versions,
        registry_ref=trusted.to_ref(),
    )
    resource_policy_approval_ref = versions.resource_policy_approval_ref
    lifecycle_approval_ref = versions.lifecycle_approval_ref
    incident_route_ref = versions.incident_route_ref
    assert resource_policy_approval_ref is not None
    assert lifecycle_approval_ref is not None
    assert incident_route_ref is not None
    persistence, specs, frozen = frozen_authority(tmp_path / "label-quality")
    manifest = frozen.dataset_manifest
    assert manifest is not None
    material = persistence.material_store.get_test_set(manifest.private_material_ref)
    observations = observation_set(frozen, specs, material.members)
    label_policy = quality_policy(
        frozen,
        specs,
        evidence_class=LabelQualityEvidenceClassV2.PRODUCTION_INDEPENDENT_TEST,
    )
    label_compilation = LabelQualityEvaluationBuilder().compile_frozen(
        persistence=persistence,
        expected_result=frozen,
        policy=label_policy,
        request=frozen_request(frozen, specs, observations),
        audit=fixture_audit(),
    )
    label = LabelQualityEvaluator().evaluate(
        compilation=label_compilation,
        policy=label_policy,
        audit=fixture_audit(),
    )
    stability = stability_report(eligible_unique_trace_count=100)
    concurrency = concurrency_report()
    review_policy, review_request, review_registry = review_inputs(
        evidence_class=(ProductionReadinessReviewEvidenceClassV2.PRODUCTION_ORGANIZATION_REVIEW),
    )
    review_compilation = ProductionReadinessReviewBuilder().compile_full(
        policy=review_policy,
        request=review_request,
        trusted_registry=review_registry,
    )
    review = ProductionReadinessReviewer().evaluate(
        compilation=review_compilation,
        policy=review_policy,
        audit=review_request.audit,
    )
    closure = ProductionReadinessSC014ClosureV1.create(
        statistical_gate_open_count=0,
        open_p0_count=0,
        open_p1_count=0,
        open_non_waivable_count=0,
        required_checkpoint_open_count=0,
        release_integrity_open_count=0,
        evidence_refs=(
            ref("batch-quality-report", "fixture"),
            ref("release-projection-result", "fixture"),
        ),
        lifecycle_approval_ref=lifecycle_approval_ref,
        incident_route_ref=incident_route_ref,
        evaluated_at=NOW,
        valid_until=NOW + timedelta(days=30),
        audit=audit(),
    )
    prerequisite = passed_prerequisite(
        label_ref=label.to_ref(),
        stability_ref=stability.to_ref(),
        concurrency_ref=concurrency.to_ref(),
        review_ref=review.to_ref(),
        closure_ref=closure.to_ref(),
    )
    evidence_refs = tuple(
        sorted(
            (
                label.to_ref(),
                stability.to_ref(),
                concurrency.to_ref(),
                review.to_ref(),
                closure.to_ref(),
            ),
            key=lambda item: (
                item.object_type,
                item.object_id,
                item.object_version,
                item.object_sha256,
            ),
        )
    )
    policy_refs = tuple(
        sorted(
            (
                attestation_policy.to_ref(),
                versions.release_profile_decision_ref,
                resource_policy_approval_ref,
                lifecycle_approval_ref,
                incident_route_ref,
                *versions.policy_refs,
            ),
            key=lambda item: (
                item.object_type,
                item.object_id,
                item.object_version,
                item.object_sha256,
            ),
        )
    )
    proofs = tuple(
        AttestationIssuanceProofV1.create(
            issuer=issuer,
            attestation_series_id="production-attestation-series://fixture",
            attestation_version=attestation_version,
            version_set_ref=versions.to_ref(),
            prerequisite_ref=prerequisite.to_ref(),
            evidence_refs=evidence_refs,
            policy_refs=policy_refs,
            decision=("REJECT" if rejected and index == 1 else "APPROVE"),
            proof_value=f"issuer-proof-value-{index}",
            issued_at=(NOW - timedelta(days=1) if expired_proof else NOW),
            valid_from=(NOW - timedelta(days=1) if expired_proof else NOW),
            valid_until=(NOW if expired_proof else NOW + timedelta(days=30)),
            audit=audit(),
        )
        for index, issuer in enumerate(trusted.issuers, start=1)
    )
    bundle = AttestationIssuanceBundleV1.create(
        attestation_series_id="production-attestation-series://fixture",
        attestation_version=attestation_version,
        proofs=proofs,
        audit=audit(),
    )
    request = ProductionReadinessAttestationRequestV1.create(
        attestation_series_id="production-attestation-series://fixture",
        attestation_version=attestation_version,
        previous_result_ref=previous_result_ref,
        previous_projection_ref=previous_projection_ref,
        label_quality_report=label,
        real_trace_stability_report=stability,
        concurrency_experiment_report=concurrency,
        production_readiness_review_report=review,
        version_set=versions,
        expected_prerequisite=prerequisite,
        sc_014_closure=closure,
        issuance_bundle=bundle,
        requested_valid_from=NOW,
        requested_valid_until=NOW + timedelta(days=30),
        evaluated_at=NOW,
        audit=audit(),
    )
    return attestation_policy, request, trusted


def production_compilation(
    tmp_path: Path,
    *,
    rejected: bool = False,
    expired_proof: bool = False,
    attestation_version: int = 1,
    previous_result_ref: ObjectRef | None = None,
    previous_projection_ref: ObjectRef | None = None,
) -> ProductionAttestationCompilation:
    attestation_policy, request, trusted = production_inputs(
        tmp_path,
        rejected=rejected,
        expired_proof=expired_proof,
        attestation_version=attestation_version,
        previous_result_ref=previous_result_ref,
        previous_projection_ref=previous_projection_ref,
    )
    return ProductionAttestationBuilder().compile_full(
        policy=attestation_policy,
        request=request,
        trusted_registry=trusted,
    )


def pending_compilation(payload: bytes) -> ProductionAttestationCompilation:
    versions = version_set(production=False)
    invalidation = invalidation_policy()
    attestation_policy = ProductionReadinessAttestationPolicyV2.create(
        expected_system_version=versions.system_version,
        base_contract_manifest_ref=versions.base_contract_manifest_ref,
        overlay_contract_manifest_ref=versions.overlay_contract_manifest_ref,
        required_schema_manifest_refs=versions.schema_manifest_refs,
        required_policy_refs=versions.policy_refs,
        release_profile_decision_ref=versions.release_profile_decision_ref,
        repository_pending_evidence_ref=ref(
            "production-attestation-repository-evidence",
            "r8-08",
            version="json/v1",
            value_digest=PRODUCTION_ATTESTATION_REPOSITORY_PENDING_SHA256,
        ),
        invalidation_policy=invalidation,
        trusted_issuer_registry_ref=None,
        evidence_class=ProductionAttestationEvidenceClassV2.REPOSITORY_PENDING_ONLY,
        minimum_issuer_count=2,
        maximum_validity_seconds=30 * 86_400,
        max_evidence_refs=128,
        max_private_bytes=100_000_000,
        max_report_bytes=10_000_000,
        audit=audit(),
    )
    return ProductionAttestationBuilder().compile_repository_pending(
        payload=payload,
        policy=attestation_policy,
    )
