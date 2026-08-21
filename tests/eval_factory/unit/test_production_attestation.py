from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from production_attestation_fixtures import (
    NOW,
    audit,
    pending_compilation,
    production_compilation,
    ref,
    version_set,
)

from eval_factory.contracts.production_attestation_v2 import (
    ProductionReadinessAttestationOutcomeV2,
    ProductionReadinessAttestationStateV2,
    ProductionReadinessInvalidationTriggerV2,
    ProductionReadinessVersionSetV2,
)
from eval_factory.readiness.production_attestation import (
    ProductionAttestationError,
    ProductionAttestationEvaluator,
    production_readiness_attestation_v1_ref,
)

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_PATH = ROOT / "evals/manifests/r8-08-repository-pending-attestation-evidence-v1.json"


def test_repository_pending_evaluation_creates_no_frozen_attestation() -> None:
    compilation = pending_compilation(EVIDENCE_PATH.read_bytes())
    result, frozen = ProductionAttestationEvaluator().evaluate(
        compilation=compilation,
        audit=audit(),
    )

    assert result.outcome is (ProductionReadinessAttestationOutcomeV2.ATTESTATION_PENDING)
    assert result.pending_gate_count == 5
    assert result.projection is None
    assert result.active_attestation is False
    assert result.satisfies_sc_015 is False
    assert result.authorizes_production_release is False
    assert frozen is None


def test_production_shaped_compilation_issues_private_frozen_v1(
    tmp_path: Path,
) -> None:
    compilation = production_compilation(tmp_path)
    result, frozen = ProductionAttestationEvaluator().evaluate(
        compilation=compilation,
        audit=audit(),
    )

    assert result.outcome is ProductionReadinessAttestationOutcomeV2.ISSUED
    assert result.active_attestation is True
    assert result.projection is not None
    assert result.projection.state is ProductionReadinessAttestationStateV2.ACTIVE
    assert result.projection.issuer_count == 2
    assert result.satisfies_sc_015 is False
    assert result.authorizes_production_release is False
    assert frozen is not None
    assert frozen.approved_by == compilation.approved_issuer_principals
    assert set(frozen.schema_refs) == {
        compilation.version_set.base_contract_manifest_ref,
        compilation.version_set.overlay_contract_manifest_ref,
        *compilation.version_set.schema_manifest_refs,
    }
    assert production_readiness_attestation_v1_ref(frozen) == (result.projection.frozen_attestation_ref)


def test_currentness_noop_and_version_drift_invalidation(
    tmp_path: Path,
) -> None:
    compilation = production_compilation(tmp_path)
    result, _ = ProductionAttestationEvaluator().evaluate(
        compilation=compilation,
        audit=audit(),
    )
    assert result.projection is not None
    registry = compilation.trusted_registry
    assert registry is not None
    evaluator = ProductionAttestationEvaluator()
    record, current = evaluator.evaluate_currentness(
        projection=result.projection,
        prior_version_set=compilation.version_set,
        observed_version_set=compilation.version_set,
        observed_issuer_registry_ref=registry.to_ref(),
        observed_at=NOW + timedelta(days=1),
        audit=audit(),
    )
    assert record is None
    assert current == result.projection

    observed = version_set(
        production=True,
        system_version="env-mock-agent/0.1.1",
    )
    with pytest.raises(ProductionAttestationError, match="prior version set"):
        evaluator.evaluate_currentness(
            projection=result.projection,
            prior_version_set=observed,
            observed_version_set=observed,
            observed_issuer_registry_ref=registry.to_ref(),
            observed_at=NOW + timedelta(days=1),
            audit=audit(),
        )
    with pytest.raises(ProductionAttestationError, match="validity"):
        evaluator.evaluate_currentness(
            projection=result.projection,
            prior_version_set=compilation.version_set,
            observed_version_set=compilation.version_set,
            observed_issuer_registry_ref=registry.to_ref(),
            observed_at=NOW - timedelta(seconds=1),
            audit=audit(),
        )
    record, invalidated = evaluator.evaluate_currentness(
        projection=result.projection,
        prior_version_set=compilation.version_set,
        observed_version_set=observed,
        observed_issuer_registry_ref=registry.to_ref(),
        observed_at=NOW + timedelta(days=1),
        audit=audit(),
    )
    assert record is not None
    assert record.triggers == (ProductionReadinessInvalidationTriggerV2.SYSTEM_VERSION_CHANGED,)
    assert invalidated.state is ProductionReadinessAttestationStateV2.INVALIDATED
    with pytest.raises(ProductionAttestationError, match="cannot reactivate"):
        evaluator.evaluate_currentness(
            projection=invalidated,
            prior_version_set=compilation.version_set,
            observed_version_set=compilation.version_set,
            observed_issuer_registry_ref=registry.to_ref(),
            observed_at=NOW + timedelta(days=2),
            audit=audit(),
        )


def test_expiry_and_explicit_revocation_are_closed(
    tmp_path: Path,
) -> None:
    compilation = production_compilation(tmp_path)
    result, _ = ProductionAttestationEvaluator().evaluate(
        compilation=compilation,
        audit=audit(),
    )
    assert result.projection is not None
    registry = compilation.trusted_registry
    assert registry is not None
    evaluator = ProductionAttestationEvaluator()
    record, expired = evaluator.evaluate_currentness(
        projection=result.projection,
        prior_version_set=compilation.version_set,
        observed_version_set=compilation.version_set,
        observed_issuer_registry_ref=registry.to_ref(),
        observed_at=result.projection.valid_until,
        audit=audit(),
    )
    assert record is not None
    assert record.triggers == (ProductionReadinessInvalidationTriggerV2.ATTESTATION_EXPIRED,)
    assert expired.state is ProductionReadinessAttestationStateV2.EXPIRED

    with pytest.raises(ProductionAttestationError, match="immutable ref"):
        evaluator.evaluate_currentness(
            projection=result.projection,
            prior_version_set=compilation.version_set,
            observed_version_set=compilation.version_set,
            observed_issuer_registry_ref=registry.to_ref(),
            observed_at=NOW + timedelta(days=1),
            explicit_revocation_kind="INCIDENT",
            audit=audit(),
        )
    record, revoked = evaluator.evaluate_currentness(
        projection=result.projection,
        prior_version_set=compilation.version_set,
        observed_version_set=compilation.version_set,
        observed_issuer_registry_ref=registry.to_ref(),
        observed_at=NOW + timedelta(days=1),
        explicit_revocation_kind="INCIDENT",
        explicit_revocation_ref=ref(
            "production-attestation-revocation",
            "incident",
            version="v1",
        ),
        audit=audit(),
    )
    assert record is not None
    assert record.triggers == (ProductionReadinessInvalidationTriggerV2.EXPLICIT_INCIDENT_REVOCATION,)
    assert revoked.state is ProductionReadinessAttestationStateV2.INVALIDATED


def test_currentness_detects_complete_drift_and_release_revocation(
    tmp_path: Path,
) -> None:
    compilation = production_compilation(tmp_path)
    result, _ = ProductionAttestationEvaluator().evaluate(
        compilation=compilation,
        audit=audit(),
    )
    assert result.projection is not None
    changed = ProductionReadinessVersionSetV2.create(
        system_version="env-mock-agent/0.2.0",
        base_contract_manifest_ref=ref(
            "contract-manifest",
            "changed-v1",
            version="v1",
        ),
        overlay_contract_manifest_ref=ref(
            "contract-manifest",
            "changed-v2",
            version="v2",
        ),
        schema_manifest_refs=(
            ref("contract-schema-manifest", "changed-v1", version="v1"),
            ref("contract-schema-manifest", "changed-v2", version="v2"),
        ),
        policy_refs=(ref("compatibility-policy", "changed-v1", version="v1"),),
        release_profile_decision_ref=ref(
            "release-profile-decision",
            "changed",
            version="v1",
        ),
        resource_policy_approval_ref=ref(
            "production-resource-policy-approval",
            "changed",
            version="v1",
        ),
        lifecycle_approval_ref=ref(
            "data-lifecycle-approval",
            "changed",
            version="v1",
        ),
        incident_route_ref=ref(
            "incident-route",
            "changed",
            version="v1",
        ),
        audit=audit(),
    )
    record, successor = ProductionAttestationEvaluator().evaluate_currentness(
        projection=result.projection,
        prior_version_set=compilation.version_set,
        observed_version_set=changed,
        observed_issuer_registry_ref=ref(
            "attestation-issuer-registry",
            "changed",
            version="private-v1",
        ),
        observed_at=NOW + timedelta(days=1),
        evidence_superseded=True,
        issuer_revoked_or_expired=True,
        explicit_revocation_kind="RELEASE",
        explicit_revocation_ref=ref(
            "production-attestation-revocation",
            "release",
            version="v1",
        ),
        audit=audit(),
    )

    assert record is not None
    assert record.triggers == (
        ProductionReadinessInvalidationTriggerV2.SYSTEM_VERSION_CHANGED,
        ProductionReadinessInvalidationTriggerV2.BASE_CONTRACT_MANIFEST_CHANGED,
        ProductionReadinessInvalidationTriggerV2.OVERLAY_CONTRACT_MANIFEST_CHANGED,
        ProductionReadinessInvalidationTriggerV2.SCHEMA_SET_CHANGED,
        ProductionReadinessInvalidationTriggerV2.POLICY_SET_CHANGED,
        ProductionReadinessInvalidationTriggerV2.EVIDENCE_SUPERSEDED,
        ProductionReadinessInvalidationTriggerV2.ISSUER_REGISTRY_CHANGED,
        ProductionReadinessInvalidationTriggerV2.ISSUER_REVOKED_OR_EXPIRED,
        ProductionReadinessInvalidationTriggerV2.EXPLICIT_RELEASE_REVOCATION,
    )
    assert successor.state is ProductionReadinessAttestationStateV2.INVALIDATED

    with pytest.raises(ProductionAttestationError, match="invalid type"):
        ProductionAttestationEvaluator().evaluate_currentness(
            projection=result.projection,
            prior_version_set=compilation.version_set,
            observed_version_set=compilation.version_set,
            observed_issuer_registry_ref=ref(
                "wrong-registry-type",
                "fixture",
                version="private-v1",
            ),
            observed_at=NOW + timedelta(days=1),
            audit=audit(),
        )
    with pytest.raises(ProductionAttestationError, match="timezone-aware"):
        ProductionAttestationEvaluator().evaluate_currentness(
            projection=result.projection,
            prior_version_set=compilation.version_set,
            observed_version_set=compilation.version_set,
            observed_issuer_registry_ref=result.projection.issuer_registry_ref,
            observed_at=(NOW + timedelta(days=1)).replace(tzinfo=None),
            audit=audit(),
        )
    with pytest.raises(ProductionAttestationError, match="no closed revocation"):
        ProductionAttestationEvaluator().evaluate_currentness(
            projection=result.projection,
            prior_version_set=compilation.version_set,
            observed_version_set=compilation.version_set,
            observed_issuer_registry_ref=result.projection.issuer_registry_ref,
            observed_at=NOW + timedelta(days=1),
            explicit_revocation_ref=ref(
                "production-attestation-revocation",
                "orphan",
                version="v1",
            ),
            audit=audit(),
        )
