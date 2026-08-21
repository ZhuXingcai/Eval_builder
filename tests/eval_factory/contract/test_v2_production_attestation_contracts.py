from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.production_attestation_v2 import (
    PRODUCTION_ATTESTATION_GATE_ORDER,
    PRODUCTION_ATTESTATION_INVALIDATION_TRIGGER_ORDER,
    ProductionAttestationEvidenceClassV2,
    ProductionReadinessAttestationOutcomeV2,
    ProductionReadinessAttestationPolicyV2,
    ProductionReadinessAttestationPrerequisiteV2,
    ProductionReadinessAttestationProjectionV2,
    ProductionReadinessAttestationReasonCodeV2,
    ProductionReadinessAttestationResultV2,
    ProductionReadinessAttestationStateV2,
    ProductionReadinessGateAssessmentV2,
    ProductionReadinessGateOutcomeV2,
    ProductionReadinessGateV2,
    ProductionReadinessInvalidationPolicyV2,
    ProductionReadinessInvalidationRecordV2,
    ProductionReadinessInvalidationTriggerV2,
    ProductionReadinessVersionSetV2,
    production_readiness_attestation_result_v2_ref,
    validate_production_readiness_attestation_result_v2_identity,
)

NOW = datetime(2026, 8, 4, tzinfo=UTC)
HASH = "a" * 64


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


def _audit(*refs: ObjectRef) -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="r8-08-contract-test",
        governing_versions=(
            VersionBinding(
                component="production-readiness-attestation",
                version="production-readiness-attestation/r8-08-v1",
            ),
        ),
        input_refs=tuple(
            sorted(
                refs,
                key=lambda value: (
                    value.object_type,
                    value.object_id,
                    value.object_version,
                    value.object_sha256,
                ),
            )
        ),
    )


def _version_set(
    *,
    production: bool = True,
    system_version: str = "env-mock-agent/0.1.0",
) -> ProductionReadinessVersionSetV2:
    return ProductionReadinessVersionSetV2.create(
        system_version=system_version,
        base_contract_manifest_ref=_ref(
            "contract-manifest",
            "v1",
            version="v1",
        ),
        overlay_contract_manifest_ref=_ref(
            "contract-manifest",
            "v2",
            version="v2",
        ),
        schema_manifest_refs=(
            _ref("contract-schema-manifest", "v1", version="v1"),
            _ref("contract-schema-manifest", "v2", version="v2"),
        ),
        policy_refs=(
            _ref("compatibility-policy", "v1", version="v1"),
            _ref("compatibility-policy", "v2", version="v2"),
            _ref("data-classification-policy", "v1", version="v1"),
            _ref("data-classification-policy", "v2", version="v2"),
            _ref("user-approval-policy", "v2", version="v2"),
        ),
        release_profile_decision_ref=_ref(
            "release-profile-decision",
            "lh-v1",
            version="v1",
        ),
        resource_policy_approval_ref=(
            _ref(
                "production-resource-policy-approval",
                "production",
                version="v1",
            )
            if production
            else None
        ),
        lifecycle_approval_ref=(
            _ref("data-lifecycle-approval", "production", version="v1") if production else None
        ),
        incident_route_ref=(_ref("incident-route", "production", version="v1") if production else None),
        audit=_audit(),
    )


def _gates(
    outcome: ProductionReadinessGateOutcomeV2,
) -> tuple[ProductionReadinessGateAssessmentV2, ...]:
    pending_reasons = {
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
    }
    return tuple(
        ProductionReadinessGateAssessmentV2.create(
            gate=gate,
            outcome=outcome,
            reason_codes=(
                ("NONE",)
                if outcome is ProductionReadinessGateOutcomeV2.PASSED
                else (pending_reasons[gate],)
                if outcome is ProductionReadinessGateOutcomeV2.PENDING
                else ("GATE_FAILED",)
            ),
            evidence_refs=(_ref("readiness-evidence", gate.value.lower()),),
            valid_until=(
                NOW + timedelta(days=30) if outcome is ProductionReadinessGateOutcomeV2.PASSED else None
            ),
        )
        for gate in PRODUCTION_ATTESTATION_GATE_ORDER
    )


def _invalidation_policy() -> ProductionReadinessInvalidationPolicyV2:
    return ProductionReadinessInvalidationPolicyV2.create(
        max_projection_revisions=1_000,
        max_invalidation_refs=64,
        audit=_audit(),
    )


def _policy(
    evidence_class: ProductionAttestationEvidenceClassV2,
    *,
    version_set: ProductionReadinessVersionSetV2,
    invalidation_policy: ProductionReadinessInvalidationPolicyV2,
) -> ProductionReadinessAttestationPolicyV2:
    return ProductionReadinessAttestationPolicyV2.create(
        expected_system_version="env-mock-agent/0.1.0",
        base_contract_manifest_ref=version_set.base_contract_manifest_ref,
        overlay_contract_manifest_ref=version_set.overlay_contract_manifest_ref,
        required_schema_manifest_refs=version_set.schema_manifest_refs,
        required_policy_refs=version_set.policy_refs,
        release_profile_decision_ref=version_set.release_profile_decision_ref,
        repository_pending_evidence_ref=_ref(
            "production-attestation-repository-evidence",
            "r8-08",
            version="json/v1",
        ),
        invalidation_policy=invalidation_policy,
        trusted_issuer_registry_ref=(
            _ref("attestation-issuer-registry", "production", version="private-v1")
            if evidence_class is ProductionAttestationEvidenceClassV2.PRODUCTION_VERIFIED
            else None
        ),
        evidence_class=evidence_class,
        minimum_issuer_count=2,
        maximum_validity_seconds=86_400 * 30,
        max_evidence_refs=128,
        max_private_bytes=100_000_000,
        max_report_bytes=10_000_000,
        audit=_audit(),
    )


def test_public_contract_inventory_and_version_set_are_strict() -> None:
    assert tuple(ProductionReadinessGateV2) == PRODUCTION_ATTESTATION_GATE_ORDER
    assert tuple(ProductionReadinessInvalidationTriggerV2) == (
        PRODUCTION_ATTESTATION_INVALIDATION_TRIGGER_ORDER
    )
    version_set = _version_set()
    assert version_set.version_set_id.endswith(version_set.version_set_sha256)
    assert version_set.lifecycle_approval_ref is not None
    with pytest.raises(ValidationError):
        ProductionReadinessVersionSetV2.model_validate(
            {**version_set.model_dump(mode="python"), "unexpected": True}
        )
    with pytest.raises(ValidationError, match="sorted and unique"):
        ProductionReadinessVersionSetV2.model_validate(
            {
                **version_set.model_dump(mode="python"),
                "policy_refs": tuple(reversed(version_set.policy_refs)),
            }
        )


def test_gate_prerequisite_derives_pending_failed_and_passed() -> None:
    pending = ProductionReadinessAttestationPrerequisiteV2.create(
        gate_assessments=_gates(ProductionReadinessGateOutcomeV2.PENDING),
        audit=_audit(),
    )
    assert pending.pending_gate_count == 5
    assert pending.outcome is ProductionReadinessAttestationOutcomeV2.ATTESTATION_PENDING
    assert pending.valid_until is None

    failed_gates = list(_gates(ProductionReadinessGateOutcomeV2.PASSED))
    failed_gates[0] = ProductionReadinessGateAssessmentV2.create(
        gate=failed_gates[0].gate,
        outcome=ProductionReadinessGateOutcomeV2.FAILED,
        reason_codes=("GATE_FAILED",),
        evidence_refs=failed_gates[0].evidence_refs,
        valid_until=None,
    )
    failed = ProductionReadinessAttestationPrerequisiteV2.create(
        gate_assessments=tuple(failed_gates),
        audit=_audit(),
    )
    assert failed.outcome is ProductionReadinessAttestationOutcomeV2.REJECTED

    passed = ProductionReadinessAttestationPrerequisiteV2.create(
        gate_assessments=_gates(ProductionReadinessGateOutcomeV2.PASSED),
        audit=_audit(),
    )
    assert passed.passed_gate_count == 5
    assert passed.outcome is ProductionReadinessAttestationOutcomeV2.ISSUED
    assert passed.valid_until == NOW + timedelta(days=30)


def test_pending_and_mechanism_results_cannot_carry_attestation() -> None:
    version_set = _version_set(production=False)
    invalidation_policy = _invalidation_policy()
    prerequisite = ProductionReadinessAttestationPrerequisiteV2.create(
        gate_assessments=_gates(ProductionReadinessGateOutcomeV2.PENDING),
        audit=_audit(),
    )
    policy = _policy(
        ProductionAttestationEvidenceClassV2.REPOSITORY_PENDING_ONLY,
        version_set=version_set,
        invalidation_policy=invalidation_policy,
    )
    result = ProductionReadinessAttestationResultV2.create(
        policy_ref=policy.to_ref(),
        version_set=version_set,
        prerequisite=prerequisite,
        evidence_class=policy.evidence_class,
        attestation_series_id="production-attestation-series://repository/r8-08",
        attestation_version=1,
        previous_result_ref=None,
        projection=None,
        invalidation_record=None,
        private_request_closure_sha256=None,
        private_issuer_closure_sha256=None,
        private_attestation_closure_sha256=None,
        audit=_audit(),
    )
    assert result.outcome is ProductionReadinessAttestationOutcomeV2.ATTESTATION_PENDING
    assert result.projection is None
    assert result.active_attestation is False
    assert result.satisfies_sc_015 is False
    assert result.authorizes_production_release is False
    validate_production_readiness_attestation_result_v2_identity(result)
    assert production_readiness_attestation_result_v2_ref(result).object_sha256 == result.result_sha256

    passed = ProductionReadinessAttestationPrerequisiteV2.create(
        gate_assessments=_gates(ProductionReadinessGateOutcomeV2.PASSED),
        audit=_audit(),
    )
    mechanism_policy = _policy(
        ProductionAttestationEvidenceClassV2.MECHANISM_VALIDATION_ONLY,
        version_set=version_set,
        invalidation_policy=invalidation_policy,
    )
    mechanism = ProductionReadinessAttestationResultV2.create(
        policy_ref=mechanism_policy.to_ref(),
        version_set=version_set,
        prerequisite=passed,
        evidence_class=mechanism_policy.evidence_class,
        attestation_series_id="production-attestation-series://mechanism",
        attestation_version=1,
        previous_result_ref=None,
        projection=None,
        invalidation_record=None,
        private_request_closure_sha256=None,
        private_issuer_closure_sha256=None,
        private_attestation_closure_sha256=None,
        audit=_audit(),
    )
    assert mechanism.outcome is (ProductionReadinessAttestationOutcomeV2.ATTESTATION_PENDING)
    assert mechanism.reason_codes == (ProductionReadinessAttestationReasonCodeV2.MECHANISM_ONLY,)


def test_active_projection_and_invalidation_successor_are_exact() -> None:
    version_set = _version_set()
    prerequisite = ProductionReadinessAttestationPrerequisiteV2.create(
        gate_assessments=_gates(ProductionReadinessGateOutcomeV2.PASSED),
        audit=_audit(),
    )
    attestation_ref = _ref(
        "production-readiness-attestation",
        "issued",
        version="v1",
    )
    registry_ref = _ref(
        "attestation-issuer-registry",
        "production",
        version="private-v1",
    )
    active = ProductionReadinessAttestationProjectionV2.create_active(
        attestation_series_id="production-attestation-series://fixture",
        attestation_version=1,
        frozen_attestation_ref=attestation_ref,
        version_set_ref=version_set.to_ref(),
        prerequisite_ref=prerequisite.to_ref(),
        issuer_registry_ref=registry_ref,
        issuer_count=2,
        valid_from=NOW,
        valid_until=NOW + timedelta(days=30),
        audit=_audit(),
    )
    assert active.state is ProductionReadinessAttestationStateV2.ACTIVE

    observed = _version_set(
        system_version="env-mock-agent/0.1.1",
    )
    record = ProductionReadinessInvalidationRecordV2.create(
        prior_projection_ref=active.to_ref(),
        prior_version_set_ref=version_set.to_ref(),
        observed_version_set_ref=observed.to_ref(),
        triggers=(ProductionReadinessInvalidationTriggerV2.SYSTEM_VERSION_CHANGED,),
        explicit_revocation_ref=None,
        observed_at=NOW + timedelta(days=1),
        audit=_audit(),
    )
    invalidated = ProductionReadinessAttestationProjectionV2.create_successor(
        prior=active,
        state=ProductionReadinessAttestationStateV2.INVALIDATED,
        invalidation_record=record,
        audit=_audit(),
    )
    assert invalidated.projection_revision == 2
    assert invalidated.previous_projection_ref == active.to_ref()
    assert invalidated.frozen_attestation_ref == active.frozen_attestation_ref
    assert invalidated.invalidation_record_ref == record.to_ref()
    with pytest.raises(ValueError, match="reactivate"):
        ProductionReadinessAttestationProjectionV2.create_successor(
            prior=invalidated,
            state=ProductionReadinessAttestationStateV2.ACTIVE,
            invalidation_record=record,
            audit=_audit(),
        )


def test_projection_and_result_reject_cross_series_authority() -> None:
    version_set = _version_set()
    prerequisite = ProductionReadinessAttestationPrerequisiteV2.create(
        gate_assessments=_gates(ProductionReadinessGateOutcomeV2.PASSED),
        audit=_audit(),
    )
    active = ProductionReadinessAttestationProjectionV2.create_active(
        attestation_series_id="production-attestation-series://fixture",
        attestation_version=1,
        frozen_attestation_ref=_ref(
            "production-readiness-attestation",
            "issued",
            version="v1",
        ),
        version_set_ref=version_set.to_ref(),
        prerequisite_ref=prerequisite.to_ref(),
        issuer_registry_ref=_ref(
            "attestation-issuer-registry",
            "production",
            version="private-v1",
        ),
        issuer_count=2,
        valid_from=NOW,
        valid_until=NOW + timedelta(days=30),
        audit=_audit(),
    )
    with pytest.raises(ValidationError, match="projection authority"):
        ProductionReadinessAttestationResultV2.create(
            policy_ref=_ref("production-readiness-attestation-policy", "fixture"),
            version_set=version_set,
            prerequisite=prerequisite,
            evidence_class=ProductionAttestationEvidenceClassV2.PRODUCTION_VERIFIED,
            attestation_series_id="production-attestation-series://other",
            attestation_version=1,
            previous_result_ref=None,
            projection=active,
            invalidation_record=None,
            private_request_closure_sha256="a" * 64,
            private_issuer_closure_sha256="b" * 64,
            private_attestation_closure_sha256="c" * 64,
            audit=_audit(),
        )

    mismatched = ProductionReadinessInvalidationRecordV2.create(
        prior_projection_ref=_ref(
            "production-readiness-attestation-projection",
            "other",
        ),
        prior_version_set_ref=version_set.to_ref(),
        observed_version_set_ref=version_set.to_ref(),
        triggers=(ProductionReadinessInvalidationTriggerV2.EVIDENCE_SUPERSEDED,),
        explicit_revocation_ref=None,
        observed_at=NOW + timedelta(days=1),
        audit=_audit(),
    )
    with pytest.raises(ValueError, match="prior projection"):
        ProductionReadinessAttestationProjectionV2.create_successor(
            prior=active,
            state=ProductionReadinessAttestationStateV2.INVALIDATED,
            invalidation_record=mismatched,
            audit=_audit(),
        )


def test_public_contracts_exclude_sensitive_issuer_surfaces() -> None:
    forbidden = {
        "issuer_principal",
        "issuer_organization",
        "issuer_role",
        "proof_value",
        "approved_by",
        "organization_contact",
        "approval_reason",
        "incident_route_body",
        "lifecycle_approval_body",
        "private_ref",
        "physical_path",
        "credential",
        "raw_trace",
        "private_reference",
        "model_payload",
        "final_output",
        "grader_rule",
        "hidden_condition",
        "exception",
    }
    models = (
        ProductionReadinessAttestationPolicyV2,
        ProductionReadinessVersionSetV2,
        ProductionReadinessGateAssessmentV2,
        ProductionReadinessAttestationPrerequisiteV2,
        ProductionReadinessInvalidationPolicyV2,
        ProductionReadinessInvalidationRecordV2,
        ProductionReadinessAttestationProjectionV2,
        ProductionReadinessAttestationResultV2,
    )
    for model in models:
        assert forbidden.isdisjoint(model.model_fields)


def test_r8_08_gold_preserves_pending_authority_and_invalidation_scope() -> None:
    root = Path(__file__).resolve().parents[3]
    gold = json.loads(
        (root / "evals/golden/eval_factory/readiness/r8-08-production-attestation-v1.json").read_text()
    )
    pending = gold["repository_pending"]
    assert pending["accepted_result_sha256"] == (
        "af5025277bf0404ea1253299538292a68406fc7bab08e2bdcb9990d94e611418"
    )
    assert (
        pending["passed_gate_count"],
        pending["failed_gate_count"],
        pending["pending_gate_count"],
    ) == (0, 0, 5)
    assert pending["attestation_projection"] is None
    assert pending["private_request_closure"] is None
    assert pending["private_issuer_closure"] is None
    assert pending["private_attestation_closure"] is None
    assert not any(
        (
            pending["satisfies_sc_010"],
            pending["satisfies_sc_011"],
            pending["satisfies_sc_012"],
            pending["satisfies_sc_013"],
            pending["satisfies_sc_014"],
            pending["satisfies_sc_015"],
            pending["active_attestation"],
            pending["authorizes_production_release"],
        )
    )
    assert pending["private_and_public_store_file_count"] == 13
    assert pending["private_and_public_store_bytes"] == 42348
    assert pending["exact_replays"] == 2
    assert gold["invalidation_triggers"] == [
        trigger.value for trigger in PRODUCTION_ATTESTATION_INVALIDATION_TRIGGER_ORDER
    ]
