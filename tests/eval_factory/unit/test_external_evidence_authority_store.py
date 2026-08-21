from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from test_external_evidence_composition import _inputs
from test_external_evidence_store import _audit

from eval_factory.contracts.core import ObjectRef, TypedAttribute
from eval_factory.contracts.external_evidence_v2 import (
    ExternalEvidenceAdmissionReportV2,
    ExternalEvidencePackageManifestV2,
    ExternalLabelObservationSummaryV2,
    ExternalSc010Sc011EvidenceIndexV2,
)
from eval_factory.contracts.external_stability_v2 import (
    ExternalRealTraceStabilityOutcomeV2,
)
from eval_factory.contracts.label_quality_v2 import (
    LabelQualityEvidenceClassV2,
    LabelQualityOutcomeV2,
)
from eval_factory.readiness.external_evidence_authority_store import (
    ExternalEvidenceAuthorityStore,
    ExternalEvidenceAuthorityStoreFaultInjector,
    ExternalEvidenceAuthorityStoreFaultPoint,
    ExternalEvidenceAuthorityStoreInjectedCrash,
    ExternalEvidenceAuthorityStoreIntegrityError,
    ExternalEvidenceAuthorityStoreLimitError,
    ExternalEvidenceAuthorityStoreTypeError,
    StaticExternalEvidenceAuthorityStoreFaultInjector,
)
from eval_factory.readiness.external_evidence_composition import (
    ExternalEvidenceComposer,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v2",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://external-authority/{suffix}",
        object_version=version,
        object_sha256=_digest(f"{object_type}:{suffix}:{version}"),
    )


def _values() -> tuple[
    ExternalEvidencePackageManifestV2,
    ExternalEvidenceAdmissionReportV2,
    ExternalLabelObservationSummaryV2,
    ExternalSc010Sc011EvidenceIndexV2,
]:
    requirement, corpus, partitions, references, observations = _inputs()
    package = ExternalEvidenceComposer().compile_package(
        requirement=requirement,
        corpus=corpus,
        partitions=partitions,
        reference_set=references,
        observation_set=observations,
        audit=_audit(),
    )
    index = ExternalSc010Sc011EvidenceIndexV2.create(
        package_manifest_ref=package.package_manifest.to_ref(),
        admission_report_ref=package.admission_report.to_ref(),
        requirement_spec_ref=(package.package_manifest.requirement_spec_ref),
        observation_summary_ref=package.observation_summary.to_ref(),
        freeze_result_ref=_ref(
            "independent-label-test-set-freeze-result",
            "freeze",
        ),
        dataset_manifest_ref=None,
        label_quality_report_ref=_ref(
            "label-quality-evaluation-report",
            "quality",
        ),
        label_quality_outcome=(LabelQualityOutcomeV2.STATISTICAL_GATE_PENDING),
        label_evidence_class=(LabelQualityEvidenceClassV2.PRODUCTION_INDEPENDENT_TEST),
        label_report_satisfies_sc_010=False,
        stability_report_ref=_ref(
            "external-real-trace-stability-report",
            "stability",
        ),
        stability_outcome=(ExternalRealTraceStabilityOutcomeV2.PASSED),
        unique_real_trace_count=100,
        observed_stability_threshold_met=True,
        evidence_class_counts=(
            TypedAttribute(key="DEVELOPMENT", value=91),
            TypedAttribute(key="REAL", value=100),
            TypedAttribute(key="REPLAY", value=100),
            TypedAttribute(key="SYNTHETIC", value=0),
        ),
        audit=_audit(),
    )
    return (
        package.package_manifest,
        package.admission_report,
        package.observation_summary,
        index,
    )


def _store(
    root: Path,
    *,
    max_authority_bytes: int = 100_000_000,
    fault_injector: ExternalEvidenceAuthorityStoreFaultInjector | None = None,
) -> ExternalEvidenceAuthorityStore:
    return ExternalEvidenceAuthorityStore(
        root,
        max_authority_bytes=max_authority_bytes,
        fault_injector=fault_injector,
    )


def test_external_authority_store_round_trip_and_exact_replay(
    tmp_path: Path,
) -> None:
    values = _values()
    root = tmp_path / "authority"
    store = _store(root)

    first = tuple(store.put(value) for value in values)
    first_files = tuple(sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file()))
    replay = tuple(store.put(value) for value in values)
    replay_files = tuple(sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file()))

    assert all(write.written for write in first)
    assert not any(write.written for write in replay)
    assert replay_files == first_files
    assert tuple(store.get(value.to_ref()) for value in values) == values
    assert not hasattr(store, "list")
    assert not hasattr(store, "path_for")


def test_external_authority_store_limits_types_and_corruption(
    tmp_path: Path,
) -> None:
    package = _values()[0]
    with pytest.raises(ExternalEvidenceAuthorityStoreLimitError):
        _store(tmp_path / "invalid", max_authority_bytes=1)
    with pytest.raises(ExternalEvidenceAuthorityStoreLimitError):
        _store(
            tmp_path / "small",
            max_authority_bytes=2,
        ).put(package)
    with pytest.raises(ExternalEvidenceAuthorityStoreTypeError):
        _store(tmp_path / "authority").get(_ref("wrong", "wrong"))
    root_file = tmp_path / "file"
    root_file.write_text("x", encoding="utf-8")
    with pytest.raises(ExternalEvidenceAuthorityStoreTypeError):
        _store(root_file)

    root = tmp_path / "corrupt"
    store = _store(root)
    store.put(package)
    blob = next(path for path in (root / "cas" / "sha256").rglob("*") if path.is_file())
    blob.write_bytes(b"{}")
    with pytest.raises(
        ExternalEvidenceAuthorityStoreIntegrityError,
        match="corrupt",
    ):
        store.get(package.to_ref())


def test_external_authority_store_recovers_cas_orphan(
    tmp_path: Path,
) -> None:
    package = _values()[0]
    root = tmp_path / "orphan"
    crashing = _store(
        root,
        fault_injector=StaticExternalEvidenceAuthorityStoreFaultInjector(
            crash_points=frozenset(
                {
                    ExternalEvidenceAuthorityStoreFaultPoint.AFTER_CAS_WRITE,
                }
            )
        ),
    )
    with pytest.raises(ExternalEvidenceAuthorityStoreInjectedCrash):
        crashing.put(package)

    write = _store(root).put(package)
    assert write.written is True
    assert write.content_blob_written is False
