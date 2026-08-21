from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path

import pytest
from test_canary_regression_builder import CANARY_PATH, _policy

from eval_factory.contracts.canary_regression_v2 import (
    CanaryRegressionCaseResultV2,
    CanaryRegressionObservedOutcomeV2,
    CanaryRegressionReasonCodeV2,
    CanaryRegressionReportV2,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.readiness.canary_regression_builder import FrozenCanaryCohortLoader
from eval_factory.readiness.canary_regression_store import (
    CanaryRegressionReportInjectedCrash,
    CanaryRegressionReportIntegrityError,
    CanaryRegressionReportLimitError,
    CanaryRegressionReportStore,
    CanaryRegressionReportStoreFaultPoint,
    CanaryRegressionReportTypeError,
    StaticCanaryRegressionReportStoreFaultInjector,
)


def _report() -> CanaryRegressionReportV2:
    policy = _policy()
    cohort = FrozenCanaryCohortLoader().load(CANARY_PATH, policy=policy)
    cases = tuple(
        CanaryRegressionCaseResultV2.create(
            instance_id=case.instance_id,
            source_trace_ref=ObjectRef(
                object_type="trace-source",
                object_id=f"source-trace://canary-regression/{case.instance_id}",
                object_version="1.0.0",
                object_sha256=case.raw_sha256,
            ),
            signal_quality=case.signal_quality,
            expectation=case.expectation,
            observed_outcome=(CanaryRegressionObservedOutcomeV2.INFRASTRUCTURE_ERROR),
            reason_code=CanaryRegressionReasonCodeV2.INTERNAL_ERROR,
            highest_reached_stage=None,
            child_manifest_ref=ObjectRef(
                object_type="r6-canary-execution-manifest",
                object_id=(
                    "r6-canary-execution-manifest://sha256/"
                    f"{hashlib.sha256(case.instance_id.encode()).hexdigest()}"
                ),
                object_version="v2",
                object_sha256=hashlib.sha256(case.instance_id.encode()).hexdigest(),
            ),
            child_dataset_result_ref=None,
            job_id=f"job://canary-regression/{case.instance_id}",
            job_status=None,
            item_id=None,
            item_status=None,
            stage_result_refs=(),
            quality_result_refs=(),
            package_manifest_refs=(),
            audit_report_ref=None,
            work_unit_count=0,
            work_lease_count=0,
            stage_run_count=0,
            attempt_count=0,
            retry_count=0,
            resume_count=0,
            provider_invocation_count=0,
            audit=policy.audit,
        )
        for case in cohort.cases
    )
    return CanaryRegressionReportV2.create(
        cohort_manifest_ref=cohort.manifest_ref,
        policy_ref=policy.to_ref(),
        template_ref=ObjectRef(
            object_type="canary-regression-template",
            object_id="canary-regression-template://r8-03/store-test",
            object_version="private-v1",
            object_sha256="a" * 64,
        ),
        case_results=cases,
        audit=policy.audit,
    )


def _store(
    root: Path,
    *,
    max_report_bytes: int = 2_000_000,
    fault_injector: StaticCanaryRegressionReportStoreFaultInjector | None = None,
) -> CanaryRegressionReportStore:
    return CanaryRegressionReportStore(
        root,
        max_report_bytes=max_report_bytes,
        max_case_refs=30,
        fault_injector=fault_injector,
    )


def test_report_store_round_trips_replay_and_first_audit_authority(
    tmp_path: Path,
) -> None:
    report = _report()
    changed = CanaryRegressionReportV2.create(
        cohort_manifest_ref=report.cohort_manifest_ref,
        policy_ref=report.policy_ref,
        template_ref=report.template_ref,
        case_results=report.case_results,
        audit=report.audit.model_copy(
            update={
                "created_at": report.audit.created_at + timedelta(days=1),
                "created_by": "another-r8-03-actor",
            }
        ),
    )
    assert changed.to_ref() == report.to_ref()
    store = _store(tmp_path / "reports")

    first = store.put(report)
    replay = store.put(report)
    audit_replay = store.put(changed)

    assert first.written is True
    assert replay.written is False
    assert audit_replay.written is False
    assert store.get(report.to_ref()) == report
    assert not hasattr(store, "root")
    assert not hasattr(store, "envelope_path")


def test_report_store_limits_types_missing_and_corruption_fail_closed(
    tmp_path: Path,
) -> None:
    report = _report()
    with pytest.raises(CanaryRegressionReportLimitError):
        CanaryRegressionReportStore(
            tmp_path / "invalid",
            max_report_bytes=1,
            max_case_refs=23,
        )
    with pytest.raises(CanaryRegressionReportLimitError, match="byte"):
        _store(tmp_path / "limited", max_report_bytes=2).put(report)
    store = _store(tmp_path / "reports")
    with pytest.raises(CanaryRegressionReportTypeError):
        store.get(
            ObjectRef(
                object_type="wrong",
                object_id="wrong://report",
                object_version="v2",
                object_sha256="a" * 64,
            )
        )
    with pytest.raises(CanaryRegressionReportIntegrityError, match="envelope"):
        store.get(report.to_ref())

    write = store.put(report)
    envelope_path = (
        tmp_path
        / "reports"
        / "envelopes"
        / "sha256"
        / write.object_ref.object_sha256[:2]
        / f"{write.object_ref.object_sha256}.json"
    )
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    digest = envelope["content_blob_ref"]["object_sha256"]
    content_path = tmp_path / "reports" / "cas" / "sha256" / digest[:2] / digest
    content_path.write_bytes(b"{}")
    with pytest.raises(CanaryRegressionReportIntegrityError, match="corrupt"):
        store.get(write.object_ref)

    envelope_path.write_text("{}", encoding="utf-8")
    with pytest.raises(CanaryRegressionReportIntegrityError, match="invalid"):
        store.get(write.object_ref)


def test_report_store_cas_orphan_is_invisible_and_reused(
    tmp_path: Path,
) -> None:
    report = _report()
    root = tmp_path / "reports"
    crashing = _store(
        root,
        fault_injector=StaticCanaryRegressionReportStoreFaultInjector(
            crash_points=frozenset({CanaryRegressionReportStoreFaultPoint.AFTER_CAS_WRITE})
        ),
    )

    with pytest.raises(CanaryRegressionReportInjectedCrash):
        crashing.put(report)

    clean = _store(root)
    with pytest.raises(CanaryRegressionReportIntegrityError, match="envelope"):
        clean.get(report.to_ref())
    retried = clean.put(report)
    assert retried.written is True
    assert retried.content_blob_written is False


def test_report_store_replays_published_envelope_and_rejects_unsafe_roots(
    tmp_path: Path,
) -> None:
    report = _report()
    root = tmp_path / "reports"
    crashing = _store(
        root,
        fault_injector=StaticCanaryRegressionReportStoreFaultInjector(
            crash_points=frozenset({CanaryRegressionReportStoreFaultPoint.AFTER_ENVELOPE_WRITE})
        ),
    )

    with pytest.raises(CanaryRegressionReportInjectedCrash):
        crashing.put(report)

    clean = _store(root)
    replay = clean.put(report)
    assert replay.written is False
    assert clean.get(report.to_ref()) == report

    root_file = tmp_path / "report-file"
    root_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(CanaryRegressionReportTypeError, match="non-symlink"):
        _store(root_file)
    root_link = tmp_path / "report-link"
    root_link.symlink_to(root, target_is_directory=True)
    with pytest.raises(CanaryRegressionReportTypeError, match="non-symlink"):
        _store(root_link)
