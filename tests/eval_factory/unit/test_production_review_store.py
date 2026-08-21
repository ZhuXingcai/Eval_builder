from __future__ import annotations

from pathlib import Path

import pytest
from production_review_fixtures import audit, full_inputs

from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.production_readiness_review_v2 import (
    PRODUCTION_READINESS_REPOSITORY_PENDING_SHA256,
    ProductionReadinessReviewDomainV2,
    ProductionReadinessReviewEvidenceClassV2,
    ProductionReadinessReviewPolicyV2,
)
from eval_factory.readiness.production_review import (
    REPOSITORY_REVIEW_SERIES_ID,
    ProductionReadinessReviewer,
)
from eval_factory.readiness.production_review_builder import (
    ProductionReadinessReviewBuilder,
)
from eval_factory.readiness.production_review_store import (
    ProductionReadinessAcceptedRunStore,
    ProductionReadinessMaterialStore,
    ProductionReadinessReportStore,
    ProductionReadinessStoreConflictError,
    ProductionReadinessStoreFaultPoint,
    ProductionReadinessStoreInjectedCrash,
    ProductionReadinessStoreIntegrityError,
    ProductionReadinessStoreLimitError,
    ProductionReadinessStoreTypeError,
    StaticProductionReadinessStoreFaultInjector,
)

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_PATH = ROOT / "evals/manifests/r8-07-repository-pending-evidence-v1.json"
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


def _pending_policy() -> ProductionReadinessReviewPolicyV2:
    return ProductionReadinessReviewPolicyV2.create(
        data_classification_policy_refs=(
            _ref("data-classification-policy", "v1", version="v1"),
            _ref("data-classification-policy", "v2", version="v2"),
        ),
        user_approval_policy_ref=_ref(
            "user-approval-policy",
            "v2",
            version="v2",
        ),
        release_profile_decision_ref=_ref(
            "release-profile-decision",
            "lh-v1",
            version="v1",
        ),
        repository_pending_evidence_ref=_ref(
            "production-readiness-review-repository-evidence",
            "r8-07/v1",
            version="json/v1",
            digest=PRODUCTION_READINESS_REPOSITORY_PENDING_SHA256,
        ),
        trusted_authority_registry_ref=None,
        evidence_class=(ProductionReadinessReviewEvidenceClassV2.REPOSITORY_PENDING_ONLY),
        max_approvals=4,
        max_evidence_refs=64,
        max_private_bytes=100_000_000,
        max_report_bytes=10_000_000,
        audit=audit(),
    )


def _stores(
    tmp_path: Path,
) -> tuple[
    ProductionReadinessMaterialStore,
    ProductionReadinessReportStore,
    ProductionReadinessAcceptedRunStore,
]:
    material = ProductionReadinessMaterialStore(
        tmp_path / "private",
        max_private_bytes=100_000_000,
        max_members=1_000,
    )
    reports = ProductionReadinessReportStore(
        tmp_path / "public",
        max_report_bytes=10_000_000,
    )
    return (
        material,
        reports,
        ProductionReadinessAcceptedRunStore(
            material_store=material,
            report_store=reports,
        ),
    )


def test_pending_first_authority_and_exact_replay_are_physical_noops(
    tmp_path: Path,
) -> None:
    policy = _pending_policy()
    compilation = ProductionReadinessReviewBuilder().compile_repository_pending(
        payload=EVIDENCE_PATH.read_bytes(),
        policy=policy,
    )
    report = ProductionReadinessReviewer().evaluate(
        compilation=compilation,
        policy=policy,
        audit=audit(),
    )
    material, reports, accepted = _stores(tmp_path)
    key = f"{REPOSITORY_REVIEW_SERIES_ID}/v1"

    first = accepted.persist(
        acceptance_key=key,
        request_sha256=compilation.evidence_sha256,
        compilation=compilation,
        report=report,
        audit=audit(),
    )
    before = _tree_facts(tmp_path)
    replayed = accepted.persist(
        acceptance_key=key,
        request_sha256=compilation.evidence_sha256,
        compilation=compilation,
        report=report,
        audit=audit(),
    )
    after = _tree_facts(tmp_path)

    assert first == replayed
    assert before == after
    assert accepted.get_accepted_report(key) == report
    assert first.registry_ref is None
    assert first.request_ref is None
    assert first.approval_bundle_ref is None
    assert reports.get(report.to_ref()) == report
    assert material.get_acceptance(key) == first


def test_full_first_authority_reloads_complete_private_public_closure(
    tmp_path: Path,
) -> None:
    policy, request, registry = full_inputs()
    compilation = ProductionReadinessReviewBuilder().compile_full(
        policy=policy,
        request=request,
        trusted_registry=registry,
    )
    report = ProductionReadinessReviewer().evaluate(
        compilation=compilation,
        policy=policy,
        audit=request.audit,
    )
    material, _, accepted = _stores(tmp_path)
    key = f"{request.review_series_id}/v{request.review_version}"
    stored = accepted.persist(
        acceptance_key=key,
        request_sha256=request.request_sha256,
        compilation=compilation,
        report=report,
        audit=audit(),
    )

    assert stored.registry_ref == registry.to_ref()
    assert stored.request_ref == request.to_ref()
    assert stored.approval_bundle_ref == request.approval_bundle.to_ref()  # type: ignore[union-attr]
    assert material.get_registry(registry.to_ref()) == registry
    assert material.get_request(request.to_ref()) == request
    assert accepted.get_accepted_report(key) == report


def test_later_review_version_requires_accepted_previous_authority(
    tmp_path: Path,
) -> None:
    previous_ref = _ref(
        "production-readiness-review-report",
        "not-accepted",
    )
    policy, request, registry = full_inputs(
        review_version=2,
        previous_report_ref=previous_ref,
    )
    compilation = ProductionReadinessReviewBuilder().compile_full(
        policy=policy,
        request=request,
        trusted_registry=registry,
    )
    report = ProductionReadinessReviewer().evaluate(
        compilation=compilation,
        policy=policy,
        audit=request.audit,
    )
    _, _, accepted = _stores(tmp_path)

    with pytest.raises(
        ProductionReadinessStoreIntegrityError,
        match="previous",
    ):
        accepted.persist(
            acceptance_key=f"{request.review_series_id}/v2",
            request_sha256=request.request_sha256,
            compilation=compilation,
            report=report,
            audit=audit(),
        )


def test_later_review_version_accepts_exact_previous_authority_and_replays(
    tmp_path: Path,
) -> None:
    policy_v1, request_v1, registry_v1 = full_inputs()
    compilation_v1 = ProductionReadinessReviewBuilder().compile_full(
        policy=policy_v1,
        request=request_v1,
        trusted_registry=registry_v1,
    )
    report_v1 = ProductionReadinessReviewer().evaluate(
        compilation=compilation_v1,
        policy=policy_v1,
        audit=request_v1.audit,
    )
    _, _, accepted = _stores(tmp_path)
    key_v1 = f"{request_v1.review_series_id}/v1"
    accepted.persist(
        acceptance_key=key_v1,
        request_sha256=request_v1.request_sha256,
        compilation=compilation_v1,
        report=report_v1,
        audit=audit(),
    )

    policy_v2, request_v2, registry_v2 = full_inputs(
        review_version=2,
        previous_report_ref=report_v1.to_ref(),
    )
    compilation_v2 = ProductionReadinessReviewBuilder().compile_full(
        policy=policy_v2,
        request=request_v2,
        trusted_registry=registry_v2,
    )
    report_v2 = ProductionReadinessReviewer().evaluate(
        compilation=compilation_v2,
        policy=policy_v2,
        audit=request_v2.audit,
    )
    key_v2 = f"{request_v2.review_series_id}/v2"
    stored = accepted.persist(
        acceptance_key=key_v2,
        request_sha256=request_v2.request_sha256,
        compilation=compilation_v2,
        report=report_v2,
        audit=audit(),
        previous_acceptance_key=key_v1,
    )
    replayed = accepted.persist(
        acceptance_key=key_v2,
        request_sha256=request_v2.request_sha256,
        compilation=compilation_v2,
        report=report_v2,
        audit=audit(),
        previous_acceptance_key=key_v1,
    )

    assert stored == replayed
    assert (
        accepted.get_accepted_report(
            key_v2,
            previous_acceptance_key=key_v1,
        )
        == report_v2
    )


def test_changed_request_under_first_authority_conflicts(
    tmp_path: Path,
) -> None:
    policy = _pending_policy()
    compilation = ProductionReadinessReviewBuilder().compile_repository_pending(
        payload=EVIDENCE_PATH.read_bytes(),
        policy=policy,
    )
    report = ProductionReadinessReviewer().evaluate(
        compilation=compilation,
        policy=policy,
        audit=audit(),
    )
    _, _, accepted = _stores(tmp_path)
    key = f"{REPOSITORY_REVIEW_SERIES_ID}/v1"
    accepted.persist(
        acceptance_key=key,
        request_sha256=compilation.evidence_sha256,
        compilation=compilation,
        report=report,
        audit=audit(),
    )
    with pytest.raises(
        ProductionReadinessStoreConflictError,
        match="request changed",
    ):
        accepted.persist(
            acceptance_key=key,
            request_sha256="f" * 64,
            compilation=compilation,
            report=report,
            audit=audit(),
        )


def test_public_report_must_match_private_compilation(
    tmp_path: Path,
) -> None:
    policy, request, registry = full_inputs()
    compilation = ProductionReadinessReviewBuilder().compile_full(
        policy=policy,
        request=request,
        trusted_registry=registry,
    )
    other_policy, other_request, other_registry = full_inputs(
        missing_domain=next(iter(ProductionReadinessReviewDomainV2))
    )
    other_compilation = ProductionReadinessReviewBuilder().compile_full(
        policy=other_policy,
        request=other_request,
        trusted_registry=other_registry,
    )
    forged = ProductionReadinessReviewer().evaluate(
        compilation=other_compilation,
        policy=other_policy,
        audit=other_request.audit,
    )
    _, _, accepted = _stores(tmp_path)
    with pytest.raises(
        ProductionReadinessStoreIntegrityError,
        match="differs from private evaluation",
    ):
        accepted.persist(
            acceptance_key=f"{request.review_series_id}/v1",
            request_sha256=request.request_sha256,
            compilation=compilation,
            report=forged,
            audit=audit(),
        )


def test_store_roots_reject_overlap_and_symlink(tmp_path: Path) -> None:
    material = ProductionReadinessMaterialStore(
        tmp_path / "shared",
        max_private_bytes=100_000_000,
        max_members=1_000,
    )
    reports = ProductionReadinessReportStore(
        tmp_path / "shared" / "public",
        max_report_bytes=10_000_000,
    )
    with pytest.raises(ProductionReadinessStoreTypeError, match="overlap"):
        ProductionReadinessAcceptedRunStore(
            material_store=material,
            report_store=reports,
        )

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ProductionReadinessStoreTypeError, match="non-symlink"):
        ProductionReadinessMaterialStore(
            link,
            max_private_bytes=100_000_000,
            max_members=1_000,
        )


def test_report_store_fault_type_limit_and_missing_material(
    tmp_path: Path,
) -> None:
    policy = _pending_policy()
    compilation = ProductionReadinessReviewBuilder().compile_repository_pending(
        payload=EVIDENCE_PATH.read_bytes(),
        policy=policy,
    )
    report = ProductionReadinessReviewer().evaluate(
        compilation=compilation,
        policy=policy,
        audit=audit(),
    )
    faulted = ProductionReadinessReportStore(
        tmp_path / "faulted",
        max_report_bytes=10_000_000,
        fault_injector=StaticProductionReadinessStoreFaultInjector(
            frozenset({ProductionReadinessStoreFaultPoint.AFTER_CAS_WRITE})
        ),
    )
    with pytest.raises(ProductionReadinessStoreInjectedCrash):
        faulted.put(report)

    recovered = ProductionReadinessReportStore(
        tmp_path / "faulted",
        max_report_bytes=10_000_000,
    )
    assert recovered.put(report).written is True
    assert recovered.get(report.to_ref()) == report
    with pytest.raises(ProductionReadinessStoreTypeError):
        recovered.get(_ref("wrong-report", "wrong"))

    empty = ProductionReadinessReportStore(
        tmp_path / "empty",
        max_report_bytes=10_000_000,
    )
    with pytest.raises(ProductionReadinessStoreIntegrityError):
        empty.get(report.to_ref())

    limited = ProductionReadinessReportStore(
        tmp_path / "limited",
        max_report_bytes=2,
    )
    with pytest.raises(ProductionReadinessStoreLimitError):
        limited.put(report)


def _tree_facts(root: Path) -> tuple[int, int]:
    files = tuple(path for path in root.rglob("*") if path.is_file())
    return len(files), sum(path.stat().st_size for path in files)
