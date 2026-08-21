from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.external_stability_v2 import (
    ExternalRealTraceStabilityPolicyV2,
    ExternalRealTraceStabilityReportV2,
)
from eval_factory.contracts.real_trace_stability_v2 import (
    RealTraceStabilityCaseOutcomeV2,
    RealTraceStabilityFaultPointV2,
    RealTraceStabilityReasonCodeV2,
)
from eval_factory.readiness.external_stability import (
    ExternalRealTraceStabilityRunResult,
)
from eval_factory.readiness.external_stability_models import (
    ExternalRealTraceStabilityCaseResultV1,
    ExternalRealTraceStabilityResultSetV1,
)
from eval_factory.readiness.external_stability_store import (
    ExternalRealTraceStabilityMaterialStore,
    ExternalRealTraceStabilityPersistenceService,
    ExternalRealTraceStabilityReportStore,
    ExternalRealTraceStabilityStoreCodec,
    ExternalRealTraceStabilityStoreFaultInjector,
    ExternalRealTraceStabilityStoreFaultPoint,
    ExternalRealTraceStabilityStoreInjectedCrash,
    ExternalRealTraceStabilityStoreIntegrityError,
    ExternalRealTraceStabilityStoreLimitError,
    ExternalRealTraceStabilityStoreTypeError,
    StaticExternalRealTraceStabilityStoreFaultInjector,
)


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str,
) -> ObjectRef:
    digest = (suffix * 64)[:64]
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://external-store/{suffix}",
        object_version=version,
        object_sha256=digest,
    )


def _audit(*, kernel: bool = False) -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 9, tzinfo=UTC),
        created_by="external-stability-store-test",
        governing_versions=(
            VersionBinding(
                component=("real-trace-stability" if kernel else "external-real-trace-stability"),
                version=(
                    "real-trace-stability/r8-04-v1" if kernel else "external-real-trace-stability/r8-10-v1"
                ),
            ),
        ),
    )


def _values() -> ExternalRealTraceStabilityRunResult:
    inventory_ref = _ref(
        "external-corpus-inventory",
        "a",
        version="private-v1",
    )
    package_ref = _ref(
        "external-evidence-package-manifest",
        "b",
        version="v2",
    )
    policy = ExternalRealTraceStabilityPolicyV2.create(
        package_manifest_ref=package_ref,
        private_inventory_ref=inventory_ref,
        source_count=1,
        inventory_sha256=inventory_ref.object_sha256,
        max_source_bytes=1_000_000,
        max_total_source_bytes=1_000_000,
        max_private_bytes=10_000_000,
        max_report_bytes=10_000_000,
        audit=_audit(),
    )
    case = ExternalRealTraceStabilityCaseResultV1.create(
        member_ref=_ref(
            "real-trace-stability-member",
            "c",
            version="private-v1",
        ),
        dataset_job_spec_ref=_ref(
            "dataset-job-spec",
            "d",
            version="v2",
        ),
        source_trace_ref=_ref(
            "trace-source",
            "e",
            version="runtime_snapshot_v1/1.0.0",
        ),
        job_id="job://external-stability/store-test",
        job_status=None,
        item_id=None,
        item_status=None,
        stage_run_refs=(),
        stage_result_refs=(),
        output_refs=(),
        checkpoint_ref=None,
        stored_manifest_ref=None,
        completion_outbox_refs=(),
        parse_quality=None,
        assigned_fault_point=(RealTraceStabilityFaultPointV2.BEFORE_SOURCE_REGISTRATION),
        fault_observed=False,
        resume_succeeded=False,
        replay_stable=False,
        attempt_count=0,
        unexpected_retry_count=0,
        resume_count=0,
        replay_count=0,
        outcome=RealTraceStabilityCaseOutcomeV2.INCOMPLETE,
        reason_code=(RealTraceStabilityReasonCodeV2.CHILD_STATE_INCOMPLETE),
        audit=_audit(kernel=True),
    )
    result_set = ExternalRealTraceStabilityResultSetV1.create(
        policy_ref=policy.to_ref(),
        private_inventory_ref=inventory_ref,
        case_results=(case,),
        audit=_audit(),
    )
    report = ExternalRealTraceStabilityReportV2.create(
        policy_ref=policy.to_ref(),
        package_manifest_ref=package_ref,
        private_inventory_ref=inventory_ref,
        private_result_set_ref=result_set.to_ref(),
        unique_real_trace_count=1,
        case_summaries=result_set.case_summaries,
        audit=_audit(),
    )
    return ExternalRealTraceStabilityRunResult(
        case_results=(case,),
        result_set=result_set,
        report=report,
    )


def _material_store(
    root: Path,
    *,
    max_private_bytes: int = 10_000_000,
    fault_injector: (ExternalRealTraceStabilityStoreFaultInjector | None) = None,
) -> ExternalRealTraceStabilityMaterialStore:
    return ExternalRealTraceStabilityMaterialStore(
        root,
        max_private_bytes=max_private_bytes,
        max_members=100,
        fault_injector=fault_injector,
    )


def _report_store(
    root: Path,
    *,
    max_report_bytes: int = 10_000_000,
    fault_injector: (ExternalRealTraceStabilityStoreFaultInjector | None) = None,
) -> ExternalRealTraceStabilityReportStore:
    return ExternalRealTraceStabilityReportStore(
        root,
        max_report_bytes=max_report_bytes,
        max_cases=100,
        fault_injector=fault_injector,
    )


def test_external_stability_closure_round_trip_and_exact_replay(
    tmp_path: Path,
) -> None:
    run = _values()
    material_root = tmp_path / "material"
    report_root = tmp_path / "reports"
    service = ExternalRealTraceStabilityPersistenceService(
        material_store=_material_store(material_root),
        report_store=_report_store(report_root),
    )

    first = service.persist(run)
    first_files = tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()))
    replay = service.persist(run)
    replay_files = tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()))

    assert first == run
    assert replay == run
    assert replay_files == first_files
    assert service.get(run.report.to_ref()) == run
    assert not hasattr(service.material_store, "root")
    assert not hasattr(service.report_store, "root")


def test_external_stability_store_limits_types_and_roots_fail_closed(
    tmp_path: Path,
) -> None:
    run = _values()
    with pytest.raises(ExternalRealTraceStabilityStoreLimitError):
        ExternalRealTraceStabilityMaterialStore(
            tmp_path / "invalid",
            max_private_bytes=1,
            max_members=1,
        )
    with pytest.raises(
        ExternalRealTraceStabilityStoreLimitError,
        match="byte",
    ):
        _material_store(
            tmp_path / "small",
            max_private_bytes=2,
        ).put_case_result(run.case_results[0])
    wrong = _ref("wrong", "f", version="v2")
    with pytest.raises(ExternalRealTraceStabilityStoreTypeError):
        _material_store(tmp_path / "material").get_case_result(wrong)
    with pytest.raises(ExternalRealTraceStabilityStoreTypeError):
        _report_store(tmp_path / "reports").get(wrong)
    root_file = tmp_path / "file"
    root_file.write_text("x", encoding="utf-8")
    with pytest.raises(
        ExternalRealTraceStabilityStoreTypeError,
        match="non-symlink",
    ):
        _report_store(root_file)
    with pytest.raises(
        ExternalRealTraceStabilityStoreTypeError,
        match="overlap",
    ):
        ExternalRealTraceStabilityPersistenceService(
            material_store=_material_store(tmp_path / "same"),
            report_store=_report_store(tmp_path / "same" / "reports"),
        )


def test_external_stability_store_corruption_and_cas_orphan(
    tmp_path: Path,
) -> None:
    run = _values()
    material_root = tmp_path / "material"
    crashing = _material_store(
        material_root,
        fault_injector=StaticExternalRealTraceStabilityStoreFaultInjector(
            crash_points=frozenset(
                {
                    ExternalRealTraceStabilityStoreFaultPoint.AFTER_CAS_WRITE,
                }
            )
        ),
    )
    with pytest.raises(ExternalRealTraceStabilityStoreInjectedCrash):
        crashing.put_case_result(run.case_results[0])
    clean = _material_store(material_root)
    with pytest.raises(
        ExternalRealTraceStabilityStoreIntegrityError,
        match="envelope",
    ):
        clean.get_case_result(run.case_results[0].to_ref())
    assert clean.put_case_result(run.case_results[0]).written is True

    report_root = tmp_path / "reports"
    reports = _report_store(report_root)
    write = reports.put(run.report)
    envelope_path = (
        report_root
        / "envelopes"
        / ExternalRealTraceStabilityStoreCodec.REPORT.value
        / "sha256"
        / write.object_ref.object_sha256[:2]
        / f"{write.object_ref.object_sha256}.json"
    )
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    digest = envelope["content_blob_ref"]["object_sha256"]
    content_path = report_root / "cas" / "sha256" / digest[:2] / digest
    content_path.write_bytes(b"{}")
    with pytest.raises(
        ExternalRealTraceStabilityStoreIntegrityError,
        match="corrupt",
    ):
        reports.get(run.report.to_ref())
