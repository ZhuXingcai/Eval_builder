from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.production_release_v2 import (
    ProductionReleaseEvidenceClassV2,
    ProductionReleaseOutcomeV2,
    ProductionReleasePolicyV2,
    ProductionReleaseReasonCodeV2,
    ProductionReleaseResultV2,
)

NOW = datetime(2026, 8, 4, tzinfo=UTC)


def _digest(value: str) -> str:
    return (value.encode().hex() + "0" * 64)[:64]


def _ref(
    object_type: str,
    name: str,
    *,
    version: str = "v2",
    digest: str | None = None,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{name}",
        object_version=version,
        object_sha256=digest or _digest(f"{object_type}:{name}:{version}"),
    )


def _audit(*refs: ObjectRef) -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="production-release-contract-test",
        governing_versions=(
            VersionBinding(
                component="production-release",
                version="production-release/r8-09-v1",
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


def _policy() -> ProductionReleasePolicyV2:
    base = _ref("contract-manifest", "v1", version="v1")
    overlay = _ref("contract-manifest", "v2")
    schemas = (
        _ref("contract-schema-manifest", "v1", version="v1"),
        _ref("contract-schema-manifest", "v2"),
    )
    baseline = (
        _ref("compatibility-policy", "v1", version="v1"),
        _ref("data-classification-policy", "v2"),
    )
    profile = _ref(
        "release-profile-decision",
        "r0-11/v1",
        version="v1",
        digest=_digest("release-profile"),
    )
    evidence = _ref(
        "production-release-repository-evidence",
        "r8-09",
        version="json/v1",
    )
    pending = _ref("production-readiness-attestation-result", "r8-08")
    return ProductionReleasePolicyV2.create(
        evidence_class=ProductionReleaseEvidenceClassV2.REPOSITORY_PENDING_ONLY,
        expected_system_version="env-mock-agent/0.1.0",
        base_contract_manifest_ref=base,
        overlay_contract_manifest_ref=overlay,
        required_schema_manifest_refs=schemas,
        baseline_attested_policy_refs=baseline,
        required_attestation_policy_version=("production-readiness-attestation/r8-08-v1"),
        release_profile_decision_ref=profile,
        release_profile_decision_sha256=profile.object_sha256,
        repository_evidence_ref=evidence,
        repository_attestation_result_ref=pending,
        allowed_production_registry_ids=frozenset({"registry://production/lh-v1"}),
        forbidden_nonproduction_registry_ids=frozenset({"registry://canary", "registry://internal-review"}),
        max_items=10,
        max_workspace_members_per_item=100,
        max_workspace_bytes_per_item=1_000_000,
        max_manifest_refs=1_000,
        max_registry_bytes=10_000_000,
        max_report_bytes=1_000_000,
        audit=_audit(base, overlay, *schemas, *baseline, profile, evidence, pending),
    )


def test_closed_enums_and_pending_policy_identity() -> None:
    assert tuple(ProductionReleaseEvidenceClassV2) == (
        ProductionReleaseEvidenceClassV2.PRODUCTION_VERIFIED,
        ProductionReleaseEvidenceClassV2.MECHANISM_VALIDATION_ONLY,
        ProductionReleaseEvidenceClassV2.REPOSITORY_PENDING_ONLY,
    )
    assert tuple(ProductionReleaseOutcomeV2) == (
        ProductionReleaseOutcomeV2.PUBLISHED,
        ProductionReleaseOutcomeV2.PRODUCTION_RELEASE_BLOCKED,
    )
    assert tuple(ProductionReleaseReasonCodeV2) == (
        ProductionReleaseReasonCodeV2.NONE,
        ProductionReleaseReasonCodeV2.ATTESTATION_PENDING,
        ProductionReleaseReasonCodeV2.ATTESTATION_NOT_ISSUED,
        ProductionReleaseReasonCodeV2.ATTESTATION_NOT_CURRENT,
        ProductionReleaseReasonCodeV2.ATTESTATION_EXPIRED,
        ProductionReleaseReasonCodeV2.MECHANISM_ONLY,
    )
    policy = _policy()
    assert policy.to_ref().object_sha256 == policy.policy_sha256
    assert policy.policy_id.endswith(policy.policy_sha256)


def test_repository_blocked_result_has_zero_production_authority() -> None:
    policy = _policy()
    result = ProductionReleaseResultV2.create_blocked(
        policy_ref=policy.to_ref(),
        evidence_class=ProductionReleaseEvidenceClassV2.REPOSITORY_PENDING_ONLY,
        reason_codes=(ProductionReleaseReasonCodeV2.ATTESTATION_PENDING,),
        repository_evidence_ref=policy.repository_evidence_ref,
        audit=_audit(policy.to_ref(), policy.repository_evidence_ref),
    )

    assert result.outcome is (ProductionReleaseOutcomeV2.PRODUCTION_RELEASE_BLOCKED)
    assert result.attestation_authority is None
    assert result.release_manifest is None
    assert result.export_receipts == ()
    assert result.published_decisions == ()
    assert result.published_items == ()
    assert result.item_projections == ()
    assert result.registry_entry is None
    assert result.satisfies_sc_015 is False
    assert result.to_ref().object_sha256 == result.result_sha256


def test_policy_is_strict_and_production_verified_forbids_pending_refs() -> None:
    payload = _policy().model_dump(mode="python")
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        ProductionReleasePolicyV2.model_validate(payload)

    payload = _policy().model_dump(mode="python")
    payload["evidence_class"] = "PRODUCTION_VERIFIED"
    with pytest.raises(
        ValidationError,
        match="production-verified policy cannot bind repository pending",
    ):
        ProductionReleasePolicyV2.model_validate(payload)


def test_public_contract_surface_is_content_free() -> None:
    serialized = _policy().model_dump_json().casefold()
    for forbidden in (
        "approved_by",
        "credential",
        "final_answer",
        "grader_rule",
        "issuer_principal",
        "private_reference",
        "proof_value",
        "raw_trace",
        "store_path",
    ):
        assert forbidden not in serialized


def test_result_rejects_naive_audit_time() -> None:
    policy = _policy()
    audit = _audit(policy.to_ref(), policy.repository_evidence_ref).model_copy(
        update={"created_at": (NOW + timedelta(seconds=1)).replace(tzinfo=None)}
    )
    with pytest.raises((ValidationError, ValueError), match="timezone-aware"):
        ProductionReleaseResultV2.create_blocked(
            policy_ref=policy.to_ref(),
            evidence_class=(ProductionReleaseEvidenceClassV2.REPOSITORY_PENDING_ONLY),
            reason_codes=(ProductionReleaseReasonCodeV2.ATTESTATION_PENDING,),
            repository_evidence_ref=policy.repository_evidence_ref,
            audit=audit,
        )
