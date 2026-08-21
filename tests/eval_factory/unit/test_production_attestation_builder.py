from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from production_attestation_fixtures import (
    audit,
    digest,
    pending_compilation,
    production_compilation,
    production_inputs,
    registry,
)
from pydantic import ValidationError

from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.label_quality_v2 import (
    LabelQualityOutcomeV2,
    label_quality_evaluation_report_v2_carried_sha256,
)
from eval_factory.contracts.production_attestation_v2 import (
    ProductionReadinessAttestationOutcomeV2,
    ProductionReadinessAttestationPolicyV2,
    ProductionReadinessAttestationReasonCodeV2,
    ProductionReadinessGateOutcomeV2,
)
from eval_factory.readiness.production_attestation_builder import (
    ProductionAttestationBuilder,
    ProductionAttestationIntegrityError,
    ProductionAttestationPolicyError,
)
from eval_factory.readiness.production_attestation_models import (
    AttestationIssuanceProofV1,
    TrustedAttestationIssuerRegistryV1,
    production_readiness_attestation_request_v1_carried_sha256,
)

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_PATH = ROOT / "evals/manifests/r8-08-repository-pending-attestation-evidence-v1.json"


def test_repository_pending_compilation_is_exact_and_private_free() -> None:
    payload = EVIDENCE_PATH.read_bytes()
    compilation = pending_compilation(payload)

    assert compilation.request is None
    assert compilation.trusted_registry is None
    assert compilation.issuance_bundle is None
    assert compilation.approved_issuer_principals == ()
    assert compilation.private_request_closure_sha256 is None
    assert compilation.private_issuer_closure_sha256 is None
    assert compilation.prerequisite.outcome is (ProductionReadinessAttestationOutcomeV2.ATTESTATION_PENDING)
    assert compilation.prerequisite.pending_gate_count == 5
    assert all(
        item.outcome is ProductionReadinessGateOutcomeV2.PENDING
        for item in compilation.prerequisite.gate_assessments
    )


def test_repository_pending_rejects_changed_bytes() -> None:
    payload = EVIDENCE_PATH.read_bytes()
    with pytest.raises(
        ProductionAttestationPolicyError,
        match="approved pending authority",
    ):
        pending_compilation(payload + b"\n")


def test_registry_rejects_project_owner_and_duplicate_authority() -> None:
    trusted = registry()
    issuer = trusted.issuers[0]
    with pytest.raises(ValidationError, match="project owner"):
        TrustedAttestationIssuerRegistryV1.create(
            registry_version="invalid",
            project_owner_principal=issuer.issuer_principal,
            issuers=trusted.issuers,
            minimum_independent_issuer_count=2,
            registry_proof_commitment_sha256=digest("registry"),
            valid_from=trusted.valid_from,
            valid_until=trusted.valid_until,
            audit=audit(),
        )
    with pytest.raises(ValidationError, match="sorted and unique"):
        TrustedAttestationIssuerRegistryV1.create(
            registry_version="invalid",
            project_owner_principal="principal://project-owner",
            issuers=(issuer, issuer),
            minimum_independent_issuer_count=2,
            registry_proof_commitment_sha256=digest("registry"),
            valid_from=trusted.valid_from,
            valid_until=trusted.valid_until,
            audit=audit(),
        )


def test_issuance_proof_value_is_hash_bound() -> None:
    trusted = registry()
    issuer = trusted.issuers[0]
    proof = AttestationIssuanceProofV1.create(
        issuer=issuer,
        attestation_series_id="production-attestation-series://fixture",
        attestation_version=1,
        version_set_ref=EVIDENCE_PATH_REF,
        prerequisite_ref=PREREQUISITE_REF,
        evidence_refs=tuple(
            ObjectRef(
                object_type="production-readiness-evidence",
                object_id=f"production-readiness-evidence://fixture/{index}",
                object_version="v2",
                object_sha256=f"{index:064x}",
            )
            for index in range(1, 5)
        ),
        policy_refs=(POLICY_REF,),
        decision="APPROVE",
        proof_value="issuer-proof-value-1",
        issued_at=trusted.valid_from,
        valid_from=trusted.valid_from,
        valid_until=trusted.valid_until,
        audit=audit(),
    )
    with pytest.raises(ValidationError, match="hash is stale"):
        AttestationIssuanceProofV1.model_validate(
            {
                **proof.model_dump(mode="python"),
                "proof_value": "changed-proof-value",
            }
        )


def test_full_builder_rejects_forged_nested_predecessor(
    tmp_path: Path,
) -> None:
    policy, request, trusted = production_inputs(tmp_path)
    forged_label = request.label_quality_report.model_copy(
        update={
            "report_id": "label-quality-evaluation-report://pending",
            "outcome": LabelQualityOutcomeV2.FAILED,
            "report_sha256": "0" * 64,
        }
    )
    label_digest = label_quality_evaluation_report_v2_carried_sha256(forged_label)
    forged_label = forged_label.model_copy(
        update={
            "report_id": f"label-quality-evaluation-report://sha256/{label_digest}",
            "report_sha256": label_digest,
        }
    )
    forged_request = request.model_copy(
        update={
            "request_id": "production-readiness-attestation-request://pending",
            "label_quality_report": forged_label,
            "request_sha256": "0" * 64,
        }
    )
    request_digest = production_readiness_attestation_request_v1_carried_sha256(forged_request)
    forged_request = forged_request.model_copy(
        update={
            "request_id": (f"production-readiness-attestation-request://sha256/{request_digest}"),
            "request_sha256": request_digest,
        }
    )

    with pytest.raises(
        ProductionAttestationIntegrityError,
        match="invalid",
    ):
        ProductionAttestationBuilder().compile_full(
            policy=policy,
            request=forged_request,
            trusted_registry=trusted,
        )


def test_full_builder_enforces_policy_maximum_validity(
    tmp_path: Path,
) -> None:
    policy, request, trusted = production_inputs(tmp_path)
    short_policy = ProductionReadinessAttestationPolicyV2.create(
        expected_system_version=policy.expected_system_version,
        base_contract_manifest_ref=policy.base_contract_manifest_ref,
        overlay_contract_manifest_ref=policy.overlay_contract_manifest_ref,
        required_schema_manifest_refs=policy.required_schema_manifest_refs,
        required_policy_refs=policy.required_policy_refs,
        release_profile_decision_ref=policy.release_profile_decision_ref,
        repository_pending_evidence_ref=policy.repository_pending_evidence_ref,
        invalidation_policy=policy.invalidation_policy,
        trusted_issuer_registry_ref=policy.trusted_issuer_registry_ref,
        evidence_class=policy.evidence_class,
        minimum_issuer_count=policy.minimum_issuer_count,
        maximum_validity_seconds=int(timedelta(days=1).total_seconds()),
        max_evidence_refs=policy.max_evidence_refs,
        max_private_bytes=policy.max_private_bytes,
        max_report_bytes=policy.max_report_bytes,
        audit=policy.audit,
    )
    assert request.requested_valid_until - request.requested_valid_from > (
        timedelta(seconds=short_policy.maximum_validity_seconds)
    )

    with pytest.raises(ProductionAttestationPolicyError, match="validity"):
        ProductionAttestationBuilder().compile_full(
            policy=short_policy,
            request=request,
            trusted_registry=trusted,
        )


def test_expired_trusted_proof_remains_pending_without_issuance(
    tmp_path: Path,
) -> None:
    compilation = production_compilation(
        tmp_path,
        expired_proof=True,
    )

    assert compilation.prerequisite.outcome is (ProductionReadinessAttestationOutcomeV2.ATTESTATION_PENDING)
    assert compilation.prerequisite.reason_codes == (
        ProductionReadinessAttestationReasonCodeV2.ISSUER_AUTHORITY_EXPIRED,
    )
    assert compilation.issuance_valid_until is None


EVIDENCE_PATH_REF = ObjectRef(
    object_type="production-readiness-version-set",
    object_id="production-readiness-version-set://fixture",
    object_version="v2",
    object_sha256="a" * 64,
)
PREREQUISITE_REF = ObjectRef(
    object_type="production-readiness-attestation-prerequisite",
    object_id="production-readiness-attestation-prerequisite://fixture",
    object_version="v2",
    object_sha256="b" * 64,
)
POLICY_REF = ObjectRef(
    object_type="production-readiness-attestation-policy",
    object_id="production-readiness-attestation-policy://fixture",
    object_version="v2",
    object_sha256="c" * 64,
)
