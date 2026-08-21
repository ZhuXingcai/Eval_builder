from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from production_attestation_fixtures import (
    audit as attestation_audit,
)
from production_attestation_fixtures import (
    pending_compilation,
    production_inputs,
    version_set,
)
from test_release_publication import _source

from env_mock_agent.facade import LHWorkspaceExportResultV2
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.production_release_v2 import (
    ProductionAttestationAuthorityV2,
    ProductionReleaseEvidenceClassV2,
    ProductionReleaseOutcomeV2,
    ProductionReleasePolicyV2,
    ProductionReleaseReasonCodeV2,
)
from eval_factory.dataset.export import ReleaseBundleFacts
from eval_factory.dataset.production_release import (
    ProductionReleaseAttestationGate,
    ProductionReleaseCompiler,
    ProductionReleasePolicyError,
    ProductionReleasePredecessor,
)
from eval_factory.readiness.production_attestation import (
    ProductionAttestationEvaluator,
)
from eval_factory.readiness.production_attestation_builder import (
    ProductionAttestationBuilder,
)
from eval_factory.readiness.production_attestation_store import (
    ProductionAttestationAcceptedRunStore,
    ProductionAttestationMaterialStore,
    ProductionAttestationReportStore,
)

NOW = datetime(2026, 8, 4, tzinfo=UTC)
ACCEPTED_ATTESTATION_SHA = "af5025277bf0404ea1253299538292a68406fc7bab08e2bdcb9990d94e611418"
PROFILE_SHA = "6b4c0f86d76757c2ea7a6331022e2d6926cfb79f1424dfc664929ff6357d5ace"
R7_GOLD_SHA = "9a554ce55063b603051ce6948f0abb8aec5638af1773c1253afa49f5be0631f7"
ROOT = Path(__file__).resolve().parents[3]


def _ref(
    object_type: str,
    name: str,
    digest: str,
    *,
    version: str = "v2",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{name}",
        object_version=version,
        object_sha256=digest,
    )


def _audit(*refs: ObjectRef) -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="production-release-unit-test",
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


def _payload() -> bytes:
    value = {
        "schema_version": ("eval-factory/repository-pending-production-release-evidence/v1"),
        "evidence_class": "REPOSITORY_PENDING_ONLY",
        "attestation_result_sha256": ACCEPTED_ATTESTATION_SHA,
        "attestation_outcome": "ATTESTATION_PENDING",
        "attestation_evidence_class": "REPOSITORY_PENDING_ONLY",
        "passed_gate_count": 0,
        "failed_gate_count": 0,
        "pending_gate_count": 5,
        "attestation_projection_present": False,
        "frozen_attestation_present": False,
        "satisfies_sc_010_through_sc_015": False,
        "authorizes_production_release": False,
        "release_profile_decision_sha256": PROFILE_SHA,
        "lh_profile": "LH",
        "lh_profile_version": "v1",
        "lh_production_enabled": False,
        "lh_production_gate": "R8_PRODUCTION_READINESS_ATTESTATION",
        "generic_production_enabled": False,
        "r7_nonproduction_gold_sha256": R7_GOLD_SHA,
        "r7_claim_scope": "NONPRODUCTION_ONLY",
        "expected_outcome": "PRODUCTION_RELEASE_BLOCKED",
        "expected_reason": "ATTESTATION_PENDING",
    }
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


def _policy(payload: bytes) -> ProductionReleasePolicyV2:
    digest = hashlib.sha256(payload).hexdigest()
    base = _ref("contract-manifest", "v1", "1" * 64, version="v1")
    overlay = _ref("contract-manifest", "v2", "2" * 64)
    schemas = (
        _ref("contract-schema-manifest", "v1", "3" * 64, version="v1"),
        _ref("contract-schema-manifest", "v2", "4" * 64),
    )
    baseline = (
        _ref("compatibility-policy", "v1", "5" * 64, version="v1"),
        _ref("data-classification-policy", "v2", "6" * 64),
    )
    profile = _ref(
        "release-profile-decision",
        "r0-11/v1",
        PROFILE_SHA,
        version="v1",
    )
    evidence = _ref(
        "production-release-repository-evidence",
        "r8-09",
        digest,
        version="json/v1",
    )
    result = _ref(
        "production-readiness-attestation-result",
        "r8-08",
        ACCEPTED_ATTESTATION_SHA,
    )
    return ProductionReleasePolicyV2.create(
        evidence_class=ProductionReleaseEvidenceClassV2.REPOSITORY_PENDING_ONLY,
        expected_system_version="env-mock-agent/0.1.0",
        base_contract_manifest_ref=base,
        overlay_contract_manifest_ref=overlay,
        required_schema_manifest_refs=schemas,
        baseline_attested_policy_refs=baseline,
        required_attestation_policy_version=("production-readiness-attestation/r8-08-v1"),
        release_profile_decision_ref=profile,
        release_profile_decision_sha256=PROFILE_SHA,
        repository_evidence_ref=evidence,
        repository_attestation_result_ref=result,
        allowed_production_registry_ids=frozenset({"registry://production/lh-v1"}),
        forbidden_nonproduction_registry_ids=frozenset({"registry://canary", "registry://internal-review"}),
        max_items=10,
        max_workspace_members_per_item=100,
        max_workspace_bytes_per_item=1_000_000,
        max_manifest_refs=1_000,
        max_registry_bytes=10_000_000,
        max_report_bytes=1_000_000,
        audit=_audit(base, overlay, *schemas, *baseline, profile, evidence, result),
    )


def _production_policy() -> ProductionReleasePolicyV2:
    pending = _policy(_payload())
    return ProductionReleasePolicyV2.create(
        evidence_class=ProductionReleaseEvidenceClassV2.PRODUCTION_VERIFIED,
        expected_system_version=pending.expected_system_version,
        base_contract_manifest_ref=pending.base_contract_manifest_ref,
        overlay_contract_manifest_ref=pending.overlay_contract_manifest_ref,
        required_schema_manifest_refs=pending.required_schema_manifest_refs,
        baseline_attested_policy_refs=pending.baseline_attested_policy_refs,
        required_attestation_policy_version=(pending.required_attestation_policy_version),
        release_profile_decision_ref=pending.release_profile_decision_ref,
        release_profile_decision_sha256=(pending.release_profile_decision_sha256),
        repository_evidence_ref=None,
        repository_attestation_result_ref=None,
        allowed_production_registry_ids=(pending.allowed_production_registry_ids),
        forbidden_nonproduction_registry_ids=(pending.forbidden_nonproduction_registry_ids),
        max_items=pending.max_items,
        max_workspace_members_per_item=(pending.max_workspace_members_per_item),
        max_workspace_bytes_per_item=(pending.max_workspace_bytes_per_item),
        max_manifest_refs=pending.max_manifest_refs,
        max_registry_bytes=pending.max_registry_bytes,
        max_report_bytes=pending.max_report_bytes,
        audit=pending.audit,
    )


def _authority() -> ProductionAttestationAuthorityV2:
    return ProductionAttestationAuthorityV2.create(
        attestation_result_ref=_ref(
            "production-readiness-attestation-result",
            "issued",
            "7" * 64,
        ),
        current_projection_ref=_ref(
            "production-readiness-attestation-projection",
            "issued",
            "8" * 64,
        ),
        frozen_attestation_ref=_ref(
            "production-readiness-attestation",
            "issued",
            "9" * 64,
            version="v1",
        ),
        version_set_ref=_ref(
            "production-readiness-version-set",
            "issued",
            "a" * 64,
        ),
        prerequisite_ref=_ref(
            "production-readiness-attestation-prerequisite",
            "issued",
            "b" * 64,
        ),
        attestation_series_id="production-attestation-series://test",
        attestation_version=2,
        valid_from=NOW,
        valid_until=datetime(2026, 9, 4, tzinfo=UTC),
        verified_at=NOW,
        audit=_audit(),
    )


def _gate_policy() -> ProductionReleasePolicyV2:
    profile = _ref(
        "release-profile-decision",
        "r0-11/v1",
        PROFILE_SHA,
        version="v1",
    )
    versions = version_set(
        production=True,
        release_profile_decision_ref=profile,
    )
    return ProductionReleasePolicyV2.create(
        evidence_class=ProductionReleaseEvidenceClassV2.PRODUCTION_VERIFIED,
        expected_system_version=versions.system_version,
        base_contract_manifest_ref=versions.base_contract_manifest_ref,
        overlay_contract_manifest_ref=versions.overlay_contract_manifest_ref,
        required_schema_manifest_refs=versions.schema_manifest_refs,
        baseline_attested_policy_refs=versions.policy_refs,
        required_attestation_policy_version=("production-readiness-attestation/r8-08-v1"),
        release_profile_decision_ref=profile,
        release_profile_decision_sha256=PROFILE_SHA,
        repository_evidence_ref=None,
        repository_attestation_result_ref=None,
        allowed_production_registry_ids=frozenset({"registry://production/lh-v1"}),
        forbidden_nonproduction_registry_ids=frozenset({"registry://canary", "registry://internal-review"}),
        max_items=10,
        max_workspace_members_per_item=100,
        max_workspace_bytes_per_item=1_000_000,
        max_manifest_refs=1_000,
        max_registry_bytes=10_000_000,
        max_report_bytes=1_000_000,
        audit=_audit(),
    )


def test_repository_pending_compiles_zero_authority_block() -> None:
    payload = _payload()
    compilation = ProductionReleaseCompiler().compile_repository_blocked(
        payload=payload,
        policy=_policy(payload),
    )

    result = compilation.result
    assert result.outcome is (ProductionReleaseOutcomeV2.PRODUCTION_RELEASE_BLOCKED)
    assert result.reason_codes == (ProductionReleaseReasonCodeV2.ATTESTATION_PENDING,)
    assert result.evidence_class is (ProductionReleaseEvidenceClassV2.REPOSITORY_PENDING_ONLY)
    assert result.attestation_authority is None
    assert result.release_manifest is None
    assert result.registry_entry is None
    assert result.satisfies_sc_015 is False


def test_repository_pending_rejects_changed_bytes_and_claims() -> None:
    payload = _payload()
    policy = _policy(payload)
    changed = payload.replace(b'"pending_gate_count":5', b'"pending_gate_count":4')
    with pytest.raises(ProductionReleasePolicyError, match="digest"):
        ProductionReleaseCompiler().compile_repository_blocked(
            payload=changed,
            policy=policy,
        )

    value = json.loads(payload)
    value["attestation_projection_present"] = True
    forged = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    forged_policy = _policy(forged)
    with pytest.raises(ProductionReleasePolicyError, match="contract"):
        ProductionReleaseCompiler().compile_repository_blocked(
            payload=forged,
            policy=forged_policy,
        )


def test_repository_compilation_revalidates_policy_identity() -> None:
    payload = _payload()
    policy = _policy(payload).model_copy(update={"policy_sha256": "f" * 64})
    with pytest.raises(ProductionReleasePolicyError, match="policy"):
        ProductionReleaseCompiler().compile_repository_blocked(
            payload=payload,
            policy=policy,
        )


def test_production_compiler_creates_linked_decision_and_registry() -> None:
    source, _ = _source()
    policy = _production_policy()
    authority = _authority()
    profile_payload = (ROOT / "specs/002-eval-dataset-factory/release-profiles/v1/decision.json").read_bytes()
    predecessor = ProductionReleasePredecessor.direct(source.approved_result)

    manifest = ProductionReleaseCompiler().compile_manifest(
        job_id=source.approved_result.release_subject.job_id,
        item_sources=(source,),
        predecessors=(predecessor,),
        attestation=authority,
        release_profile_payload=profile_payload,
        registry="registry://production/lh-v1",
        policy=policy,
        audit=_audit(),
    )
    request = manifest.items[0].workspace_request
    workspace = LHWorkspaceExportResultV2.exported(
        request=request,
        observed_members=request.members,
        copied_output_refs=request.output_refs,
    )
    result = (
        ProductionReleaseCompiler()
        .compile_publication(
            manifest=manifest,
            workspace_results=(workspace,),
            bundle_facts=(
                ReleaseBundleFacts(
                    bundle_sha256="c" * 64,
                    bundle_file_count=3,
                    bundle_total_bytes=100,
                ),
            ),
            policy=policy,
            published_at=NOW,
            audit=_audit(),
        )
        .result
    )

    decision = result.published_decisions[0]
    assert result.outcome is ProductionReleaseOutcomeV2.PUBLISHED
    assert result.satisfies_sc_015 is True
    assert decision.previous_decision_ref == (source.approved_result.release_decision_ref)
    assert decision.channel.value == "PRODUCTION"
    assert decision.registry == "registry://production/lh-v1"
    assert decision.production_attestation_ref == (authority.frozen_attestation_ref)
    assert result.registry_entry is not None


def test_attestation_gate_returns_closed_pending_without_authority(
    tmp_path: Path,
) -> None:
    payload = (ROOT / "evals/manifests/r8-08-repository-pending-attestation-evidence-v1.json").read_bytes()
    compilation = pending_compilation(payload)
    result, frozen = ProductionAttestationEvaluator().evaluate(
        compilation=compilation,
        audit=attestation_audit(),
    )
    material = ProductionAttestationMaterialStore(
        tmp_path / "attestation-private",
        max_private_bytes=10_000_000,
        max_members=1_000,
    )
    reports = ProductionAttestationReportStore(
        tmp_path / "attestation-public",
        max_report_bytes=10_000_000,
    )
    accepted = ProductionAttestationAcceptedRunStore(
        material_store=material,
        report_store=reports,
    )
    accepted.persist(
        acceptance_key="production-attestation-acceptance://pending",
        request_sha256="d" * 64,
        compilation=compilation,
        result=result,
        frozen=frozen,
        audit=attestation_audit(),
    )

    with ProductionReleaseAttestationGate(accepted).hold_current(
        acceptance_key="production-attestation-acceptance://pending",
        previous_acceptance_key=None,
        policy=_production_policy(),
        observed_at=NOW,
        audit=_audit(),
    ) as admission:
        assert admission.authority is None
        assert admission.reason_code is (ProductionReleaseReasonCodeV2.ATTESTATION_PENDING)


def test_attestation_gate_derives_authority_from_full_accepted_closure(
    tmp_path: Path,
) -> None:
    policy = _gate_policy()
    attestation_policy, request, registry = production_inputs(
        tmp_path / "inputs",
        extra_policy_refs=(policy.to_ref(),),
        release_profile_decision_ref=policy.release_profile_decision_ref,
    )
    compilation = ProductionAttestationBuilder().compile_full(
        policy=attestation_policy,
        request=request,
        trusted_registry=registry,
    )
    result, frozen = ProductionAttestationEvaluator().evaluate(
        compilation=compilation,
        audit=request.audit,
    )
    material = ProductionAttestationMaterialStore(
        tmp_path / "attestation-private",
        max_private_bytes=100_000_000,
        max_members=1_000,
    )
    reports = ProductionAttestationReportStore(
        tmp_path / "attestation-public",
        max_report_bytes=10_000_000,
    )
    accepted = ProductionAttestationAcceptedRunStore(
        material_store=material,
        report_store=reports,
    )
    accepted.persist(
        acceptance_key="production-attestation-acceptance://issued",
        request_sha256=request.request_sha256,
        compilation=compilation,
        result=result,
        frozen=frozen,
        audit=request.audit,
    )

    with ProductionReleaseAttestationGate(accepted).hold_current(
        acceptance_key="production-attestation-acceptance://issued",
        previous_acceptance_key=None,
        policy=policy,
        observed_at=NOW,
        audit=_audit(),
    ) as admission:
        assert admission.reason_code is ProductionReleaseReasonCodeV2.NONE
        assert admission.authority is not None
        assert admission.authority.attestation_result_ref == result.to_ref()
        assert admission.authority.frozen_attestation_ref == (result.projection.frozen_attestation_ref)
