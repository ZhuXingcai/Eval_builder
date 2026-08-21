from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from production_attestation_fixtures import (
    NOW,
    audit,
    pending_compilation,
    production_compilation,
    version_set,
)

from eval_factory.contracts.core import ObjectRef
from eval_factory.readiness.production_attestation import (
    ProductionAttestationEvaluator,
)
from eval_factory.readiness.production_attestation_store import (
    ProductionAttestationAcceptedRunStore,
    ProductionAttestationMaterialStore,
    ProductionAttestationReportStore,
    ProductionAttestationStoreCodec,
    ProductionAttestationStoreConflictError,
    ProductionAttestationStoreIntegrityError,
    ProductionAttestationStoreTypeError,
)

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_PATH = ROOT / "evals/manifests/r8-08-repository-pending-attestation-evidence-v1.json"


def _stores(
    tmp_path: Path,
) -> tuple[
    ProductionAttestationMaterialStore,
    ProductionAttestationReportStore,
    ProductionAttestationAcceptedRunStore,
]:
    material = ProductionAttestationMaterialStore(
        tmp_path / "private",
        max_private_bytes=100_000_000,
        max_members=1_000,
    )
    reports = ProductionAttestationReportStore(
        tmp_path / "public",
        max_report_bytes=10_000_000,
    )
    return (
        material,
        reports,
        ProductionAttestationAcceptedRunStore(
            material_store=material,
            report_store=reports,
        ),
    )


def _tree(root: Path) -> tuple[tuple[str, int], ...]:
    return tuple(
        (str(path.relative_to(root)), path.stat().st_size)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )


def test_pending_first_authority_and_exact_replay_have_no_private_issue_material(
    tmp_path: Path,
) -> None:
    compilation = pending_compilation(EVIDENCE_PATH.read_bytes())
    result, frozen = ProductionAttestationEvaluator().evaluate(
        compilation=compilation,
        audit=audit(),
    )
    material, reports, accepted = _stores(tmp_path)
    key = "production-attestation-acceptance://repository/r8-08/v1"
    first = accepted.persist(
        acceptance_key=key,
        request_sha256=compilation.evidence_sha256,
        compilation=compilation,
        result=result,
        frozen=frozen,
        audit=audit(),
    )
    before = _tree(tmp_path)
    second = accepted.persist(
        acceptance_key=key,
        request_sha256=compilation.evidence_sha256,
        compilation=compilation,
        result=result,
        frozen=frozen,
        audit=audit(),
    )
    after = _tree(tmp_path)

    assert first == second
    assert before == after
    assert first.registry_ref is None
    assert first.request_ref is None
    assert first.bundle_ref is None
    assert first.frozen_attestation_ref is None
    assert first.current_projection_ref is None
    assert accepted.get_accepted_result(key) == result
    assert reports.find_current_projection(result.attestation_series_id) is None
    assert material.get_acceptance(key) == first


def test_full_material_codecs_and_current_projection_round_trip(
    tmp_path: Path,
) -> None:
    compilation = production_compilation(tmp_path / "fixtures")
    result, frozen = ProductionAttestationEvaluator().evaluate(
        compilation=compilation,
        audit=audit(),
    )
    assert frozen is not None
    assert result.projection is not None
    material, reports, _ = _stores(tmp_path)
    assert compilation.trusted_registry is not None
    assert compilation.request is not None
    assert compilation.issuance_bundle is not None
    for proof in compilation.issuance_bundle.proofs:
        material.put(proof)
        assert (
            material.get(
                proof.to_ref(),
                ProductionAttestationStoreCodec.ISSUANCE_PROOF,
                type(proof),
            )
            == proof
        )
    for value in (
        compilation.trusted_registry,
        compilation.issuance_bundle,
        compilation.request.sc_014_closure,
        compilation.request,
        frozen,
    ):
        material.put(value)
    for value in (
        compilation.policy,
        compilation.invalidation_policy,
        compilation.version_set,
        compilation.prerequisite,
        result.projection,
        result,
    ):
        reports.put(value)
    current = reports.publish_current_projection(
        series_key=result.attestation_series_id,
        projection=result.projection,
        expected_prior_ref=None,
    )

    assert current == result.projection
    assert reports.find_current_projection(result.attestation_series_id) == (result.projection)
    with pytest.raises(
        ProductionAttestationStoreTypeError,
        match="series key",
    ):
        reports.publish_current_projection(
            series_key="production-attestation-series://other",
            projection=result.projection,
            expected_prior_ref=None,
        )
    assert compilation.trusted_registry is not None
    record, successor = ProductionAttestationEvaluator().evaluate_currentness(
        projection=current,
        prior_version_set=compilation.version_set,
        observed_version_set=version_set(
            production=True,
            system_version="env-mock-agent/0.1.1",
        ),
        observed_issuer_registry_ref=compilation.trusted_registry.to_ref(),
        observed_at=NOW + timedelta(days=1),
        audit=audit(),
    )
    assert record is not None
    published = reports.publish_invalidation(
        series_key=result.attestation_series_id,
        invalidation_policy=compilation.invalidation_policy,
        prior=current,
        invalidation_record=record,
        successor=successor,
    )
    assert published == successor
    assert reports.find_current_projection(result.attestation_series_id) == successor
    assert (
        reports.get(
            record.to_ref(),
            ProductionAttestationStoreCodec.INVALIDATION_RECORD,
            type(record),
        )
        == record
    )


def test_changed_request_conflicts_and_overlapping_roots_fail(
    tmp_path: Path,
) -> None:
    compilation = pending_compilation(EVIDENCE_PATH.read_bytes())
    result, frozen = ProductionAttestationEvaluator().evaluate(
        compilation=compilation,
        audit=audit(),
    )
    _, _, accepted = _stores(tmp_path)
    key = "production-attestation-acceptance://repository/r8-08/v1"
    accepted.persist(
        acceptance_key=key,
        request_sha256=compilation.evidence_sha256,
        compilation=compilation,
        result=result,
        frozen=frozen,
        audit=audit(),
    )
    with pytest.raises(
        ProductionAttestationStoreConflictError,
        match="request changed",
    ):
        accepted.persist(
            acceptance_key=key,
            request_sha256="f" * 64,
            compilation=compilation,
            result=result,
            frozen=frozen,
            audit=audit(),
        )

    material = ProductionAttestationMaterialStore(
        tmp_path / "shared",
        max_private_bytes=1_000_000,
        max_members=100,
    )
    reports = ProductionAttestationReportStore(
        tmp_path / "shared" / "public",
        max_report_bytes=1_000_000,
    )
    with pytest.raises(ProductionAttestationStoreTypeError, match="overlap"):
        ProductionAttestationAcceptedRunStore(
            material_store=material,
            report_store=reports,
        )


def test_full_rejected_authority_persists_without_frozen_attestation(
    tmp_path: Path,
) -> None:
    compilation = production_compilation(
        tmp_path / "fixtures",
        rejected=True,
    )
    result, frozen = ProductionAttestationEvaluator().evaluate(
        compilation=compilation,
        audit=audit(),
    )
    assert result.outcome.value == "REJECTED"
    assert result.projection is None
    assert frozen is None
    _, _, accepted = _stores(tmp_path / "stores")
    key = "production-attestation-acceptance://fixture/rejected/v1"

    first = accepted.persist(
        acceptance_key=key,
        request_sha256=compilation.evidence_sha256,
        compilation=compilation,
        result=result,
        frozen=frozen,
        audit=audit(),
        previous_acceptance_key=None,
    )

    assert first.registry_ref is not None
    assert first.request_ref is not None
    assert first.bundle_ref is not None
    assert first.frozen_attestation_ref is None
    assert first.current_projection_ref is None
    assert (
        accepted.get_accepted_result(
            key,
            previous_acceptance_key=None,
        )
        == result
    )


def test_full_replay_requires_independent_policy_and_sc014_closure(
    tmp_path: Path,
) -> None:
    compilation = production_compilation(tmp_path / "fixtures")
    result, frozen = ProductionAttestationEvaluator().evaluate(
        compilation=compilation,
        audit=audit(),
    )
    material, reports, accepted = _stores(tmp_path / "stores")
    key = "production-attestation-acceptance://fixture/issued/v1"
    accepted.persist(
        acceptance_key=key,
        request_sha256=compilation.evidence_sha256,
        compilation=compilation,
        result=result,
        frozen=frozen,
        audit=audit(),
        previous_acceptance_key=None,
    )
    invalidation_ref = compilation.invalidation_policy.to_ref()
    invalidation_path = (
        reports.root
        / "envelopes"
        / "invalidation-policy"
        / "sha256"
        / invalidation_ref.object_sha256[:2]
        / f"{invalidation_ref.object_sha256}.json"
    )
    invalidation_path.unlink()
    with pytest.raises(
        ProductionAttestationStoreIntegrityError,
        match="missing",
    ):
        accepted.get_accepted_result(
            key,
            previous_acceptance_key=None,
        )

    reports.put(compilation.invalidation_policy)
    assert compilation.request is not None
    closure_ref = compilation.request.sc_014_closure.to_ref()
    closure_path = (
        material.root
        / "envelopes"
        / "sc014-closure"
        / "sha256"
        / closure_ref.object_sha256[:2]
        / f"{closure_ref.object_sha256}.json"
    )
    closure_path.unlink()
    with pytest.raises(
        ProductionAttestationStoreIntegrityError,
        match="missing",
    ):
        accepted.get_accepted_result(
            key,
            previous_acceptance_key=None,
        )


def test_version_two_requires_exact_accepted_previous_authority(
    tmp_path: Path,
) -> None:
    first_compilation = production_compilation(tmp_path / "fixtures-v1")
    first_result, first_frozen = ProductionAttestationEvaluator().evaluate(
        compilation=first_compilation,
        audit=audit(),
    )
    assert first_result.projection is not None
    _, reports, accepted = _stores(tmp_path / "stores")
    first_key = "production-attestation-acceptance://fixture/issued/v1"
    accepted.persist(
        acceptance_key=first_key,
        request_sha256=first_compilation.evidence_sha256,
        compilation=first_compilation,
        result=first_result,
        frozen=first_frozen,
        audit=audit(),
        previous_acceptance_key=None,
    )
    assert first_compilation.trusted_registry is not None
    invalidation_record, invalidated = ProductionAttestationEvaluator().evaluate_currentness(
        projection=first_result.projection,
        prior_version_set=first_compilation.version_set,
        observed_version_set=first_compilation.version_set,
        observed_issuer_registry_ref=(first_compilation.trusted_registry.to_ref()),
        observed_at=NOW + timedelta(days=1),
        explicit_revocation_kind="INCIDENT",
        explicit_revocation_ref=ObjectRef(
            object_type="production-attestation-revocation",
            object_id="production-attestation-revocation://fixture/incident",
            object_version="v1",
            object_sha256="f" * 64,
        ),
        audit=audit(),
    )
    assert invalidation_record is not None
    reports.publish_invalidation(
        series_key=first_result.attestation_series_id,
        invalidation_policy=first_compilation.invalidation_policy,
        prior=first_result.projection,
        invalidation_record=invalidation_record,
        successor=invalidated,
    )
    second_compilation = production_compilation(
        tmp_path / "fixtures-v2",
        attestation_version=2,
        previous_result_ref=first_result.to_ref(),
        previous_projection_ref=invalidated.to_ref(),
    )
    second_result, second_frozen = ProductionAttestationEvaluator().evaluate(
        compilation=second_compilation,
        audit=audit(),
    )
    second_key = "production-attestation-acceptance://fixture/issued/v2"
    with pytest.raises(
        ProductionAttestationStoreIntegrityError,
        match="previous authority",
    ):
        accepted.persist(
            acceptance_key=second_key,
            request_sha256=second_compilation.evidence_sha256,
            compilation=second_compilation,
            result=second_result,
            frozen=second_frozen,
            audit=audit(),
            previous_acceptance_key=None,
        )

    accepted.persist(
        acceptance_key=second_key,
        request_sha256=second_compilation.evidence_sha256,
        compilation=second_compilation,
        result=second_result,
        frozen=second_frozen,
        audit=audit(),
        previous_acceptance_key=first_key,
    )
    assert (
        accepted.get_accepted_result(
            second_key,
            previous_acceptance_key=first_key,
        )
        == second_result
    )


def test_explicit_replay_repairs_missing_current_head_after_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compilation = production_compilation(tmp_path / "fixtures")
    result, frozen = ProductionAttestationEvaluator().evaluate(
        compilation=compilation,
        audit=audit(),
    )
    assert result.projection is not None
    material, reports, accepted = _stores(tmp_path / "stores")
    key = "production-attestation-acceptance://fixture/crash/v1"
    publish = reports.publish_current_projection

    def crash_before_current_head(**_: object) -> None:
        raise RuntimeError("injected current-head crash")

    monkeypatch.setattr(
        reports,
        "publish_current_projection",
        crash_before_current_head,
    )
    with pytest.raises(RuntimeError, match="current-head crash"):
        accepted.persist(
            acceptance_key=key,
            request_sha256=compilation.evidence_sha256,
            compilation=compilation,
            result=result,
            frozen=frozen,
            audit=audit(),
            previous_acceptance_key=None,
        )
    assert material.find_acceptance(key) is not None
    assert reports.find_current_projection(result.attestation_series_id) is None

    monkeypatch.setattr(reports, "publish_current_projection", publish)
    assert (
        accepted.replay_accepted_result(
            key,
            previous_acceptance_key=None,
        )
        == result
    )
    assert reports.find_current_projection(result.attestation_series_id) == (result.projection)
