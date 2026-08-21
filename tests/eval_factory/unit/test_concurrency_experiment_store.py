from __future__ import annotations

import json
from pathlib import Path

import pytest
from concurrency_experiment_fixtures import policy, report
from test_concurrency_experiment import (
    _Clock,
    _Executor,
    _prepared,
    _RecoveryExecutor,
)
from test_concurrency_experiment_builder import _prepared_inputs

from eval_factory.contracts.concurrency_experiment_v2 import (
    concurrency_experiment_report_v2_ref,
)
from eval_factory.readiness.concurrency_experiment import (
    ConcurrencyExperimentRunner,
)
from eval_factory.readiness.concurrency_experiment_store import (
    ConcurrencyExperimentMaterialStore,
    ConcurrencyExperimentReportStore,
    ConcurrencyExperimentStoreConflictError,
    ConcurrencyExperimentStoreFaultPoint,
    ConcurrencyExperimentStoreInjectedCrash,
    ConcurrencyExperimentStoreIntegrityError,
    ConcurrencyExperimentStoreLimitError,
    ConcurrencyExperimentStoreTypeError,
    StaticConcurrencyExperimentStoreFaultInjector,
)
from eval_factory.storage import ContentAddressedByteStoreConflictError


def test_private_workload_store_replays_and_round_trips(tmp_path: Path) -> None:
    _builder, _cohort, _template, _host, workload = _prepared_inputs(tmp_path)
    store = ConcurrencyExperimentMaterialStore(
        tmp_path / "private",
        max_private_bytes=1_000_000,
        max_members=1_000,
    )

    first = store.put_workload(workload)
    replay = store.put_workload(workload)

    assert first.written is True
    assert replay.written is False
    assert store.get_workload(workload.to_ref()) == workload


def test_public_report_store_reuses_first_authority(tmp_path: Path) -> None:
    value = report()
    store = ConcurrencyExperimentReportStore(
        tmp_path / "report",
        max_report_bytes=10_000_000,
    )

    first = store.put(value)
    replay = store.put(
        value.model_copy(update={"audit": value.audit.model_copy(update={"created_by": "audit-only-replay"})})
    )

    assert first.written is True
    assert replay.written is False
    assert store.get(concurrency_experiment_report_v2_ref(value)) == value


def test_cas_orphan_is_invisible_and_retry_publishes_once(
    tmp_path: Path,
) -> None:
    _builder, _cohort, _template, _host, workload = _prepared_inputs(tmp_path)
    root = tmp_path / "private"
    broken = ConcurrencyExperimentMaterialStore(
        root,
        max_private_bytes=1_000_000,
        max_members=1_000,
        fault_injector=StaticConcurrencyExperimentStoreFaultInjector(
            crash_points=frozenset({ConcurrencyExperimentStoreFaultPoint.AFTER_CAS_WRITE})
        ),
    )

    with pytest.raises(ConcurrencyExperimentStoreInjectedCrash):
        broken.put_workload(workload)
    with pytest.raises(ConcurrencyExperimentStoreIntegrityError):
        broken.get_workload(workload.to_ref())

    recovered = ConcurrencyExperimentMaterialStore(
        root,
        max_private_bytes=1_000_000,
        max_members=1_000,
    )
    assert recovered.put_workload(workload).written is True
    assert recovered.get_workload(workload.to_ref()) == workload


def test_report_corruption_and_wrong_ref_fail_closed(tmp_path: Path) -> None:
    value = report()
    root = tmp_path / "report"
    store = ConcurrencyExperimentReportStore(
        root,
        max_report_bytes=10_000_000,
    )
    ref = concurrency_experiment_report_v2_ref(value)
    store.put(value)
    envelope = next((root / "envelopes").rglob("*.json"))
    envelope.write_text("{}", encoding="utf-8")

    with pytest.raises(ConcurrencyExperimentStoreIntegrityError):
        store.get(ref)
    with pytest.raises(ConcurrencyExperimentStoreTypeError):
        store.get(ref.model_copy(update={"object_type": "scheduler-load-report"}))


def test_store_rejects_symlink_root(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)

    with pytest.raises(ConcurrencyExperimentStoreTypeError):
        ConcurrencyExperimentMaterialStore(
            alias,
            max_private_bytes=1_000_000,
            max_members=1_000,
        )


def test_envelope_first_authority_survives_post_write_crash(
    tmp_path: Path,
) -> None:
    _builder, _cohort, _template, _host, workload = _prepared_inputs(tmp_path)
    root = tmp_path / "private"
    broken = ConcurrencyExperimentMaterialStore(
        root,
        max_private_bytes=1_000_000,
        max_members=1_000,
        fault_injector=StaticConcurrencyExperimentStoreFaultInjector(
            crash_points=frozenset({ConcurrencyExperimentStoreFaultPoint.AFTER_ENVELOPE_WRITE})
        ),
    )

    with pytest.raises(ConcurrencyExperimentStoreInjectedCrash):
        broken.put_workload(workload)

    recovered = ConcurrencyExperimentMaterialStore(
        root,
        max_private_bytes=1_000_000,
        max_members=1_000,
    )
    assert recovered.get_workload(workload.to_ref()) == workload
    assert recovered.put_workload(workload).written is False


def test_store_enforces_private_and_public_byte_limits(
    tmp_path: Path,
) -> None:
    _builder, _cohort, _template, _host, workload = _prepared_inputs(tmp_path)
    private = ConcurrencyExperimentMaterialStore(
        tmp_path / "private",
        max_private_bytes=2,
        max_members=1_000,
    )
    public = ConcurrencyExperimentReportStore(
        tmp_path / "report",
        max_report_bytes=2,
    )

    with pytest.raises(ConcurrencyExperimentStoreLimitError):
        private.put_workload(workload)
    with pytest.raises(ConcurrencyExperimentStoreLimitError):
        public.put(report())


def test_store_constructor_rejects_invalid_limits_and_file_root(
    tmp_path: Path,
) -> None:
    with pytest.raises(ConcurrencyExperimentStoreLimitError, match="byte limit"):
        ConcurrencyExperimentReportStore(
            tmp_path / "report",
            max_report_bytes=1,
        )
    with pytest.raises(ConcurrencyExperimentStoreLimitError, match="member limit"):
        ConcurrencyExperimentMaterialStore(
            tmp_path / "private-small",
            max_private_bytes=1_000_000,
            max_members=383,
        )
    with pytest.raises(ConcurrencyExperimentStoreLimitError, match="member limit"):
        ConcurrencyExperimentMaterialStore(
            tmp_path / "private-large",
            max_private_bytes=1_000_000,
            max_members=1_000_001,
        )
    file_root = tmp_path / "file-root"
    file_root.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ConcurrencyExperimentStoreTypeError, match="directory"):
        ConcurrencyExperimentReportStore(
            file_root,
            max_report_bytes=1_000_000,
        )


def test_store_rejects_wrong_material_ref_and_stale_report(
    tmp_path: Path,
) -> None:
    _builder, _cohort, _template, host, workload = _prepared_inputs(tmp_path)
    private = ConcurrencyExperimentMaterialStore(
        tmp_path / "private",
        max_private_bytes=1_000_000,
        max_members=1_000,
    )
    host_ref = private.put_host(host).object_ref

    with pytest.raises(ConcurrencyExperimentStoreTypeError, match="wrong type"):
        private.get_workload(host_ref)

    public = ConcurrencyExperimentReportStore(
        tmp_path / "report",
        max_report_bytes=10_000_000,
    )
    stale = report().model_copy(update={"report_sha256": "0" * 64})
    with pytest.raises(ConcurrencyExperimentStoreConflictError, match="stale"):
        public.put(stale)

    private.put_workload(workload)


def test_report_verify_and_content_corruption_fail_closed(
    tmp_path: Path,
) -> None:
    value = report()
    root = tmp_path / "report"
    store = ConcurrencyExperimentReportStore(
        root,
        max_report_bytes=10_000_000,
    )
    ref = concurrency_experiment_report_v2_ref(value)
    store.put(value)
    store.verify(ref)
    blob = next((root / "cas" / "sha256").glob("*/*"))
    blob.write_bytes(b"{}")

    with pytest.raises(ConcurrencyExperimentStoreIntegrityError, match="corrupt"):
        store.get(ref)


def test_store_wraps_cas_write_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _builder, _cohort, _template, _host, workload = _prepared_inputs(tmp_path)
    store = ConcurrencyExperimentMaterialStore(
        tmp_path / "private",
        max_private_bytes=1_000_000,
        max_members=1_000,
    )

    def conflict(**kwargs: object) -> bool:
        del kwargs
        raise ContentAddressedByteStoreConflictError("private CAS detail")

    monkeypatch.setattr(store._store._cas, "write", conflict)
    with pytest.raises(ConcurrencyExperimentStoreConflictError, match="conflicted"):
        store.put_workload(workload)


def test_store_rejects_noncanonical_envelope_bytes(tmp_path: Path) -> None:
    _builder, _cohort, _template, _host, workload = _prepared_inputs(tmp_path)
    root = tmp_path / "private"
    store = ConcurrencyExperimentMaterialStore(
        root,
        max_private_bytes=1_000_000,
        max_members=1_000,
    )
    store.put_workload(workload)
    envelope = next((root / "envelopes").rglob("*.json"))
    payload = json.loads(envelope.read_text(encoding="utf-8"))
    envelope.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(
        ConcurrencyExperimentStoreIntegrityError,
        match="canonical",
    ):
        store.get_workload(workload.to_ref())


def test_store_rejects_nested_symlink_envelope_root(tmp_path: Path) -> None:
    _builder, _cohort, _template, _host, workload = _prepared_inputs(tmp_path)
    root = tmp_path / "private"
    store = ConcurrencyExperimentMaterialStore(
        root,
        max_private_bytes=1_000_000,
        max_members=1_000,
    )
    envelopes = root / "envelopes"
    envelopes.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (envelopes / "workload").symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        ConcurrencyExperimentStoreTypeError,
        match="symlink",
    ):
        store.put_workload(workload)

    assert tuple(outside.rglob("*.json")) == ()


def test_material_store_round_trips_all_runner_codecs(
    tmp_path: Path,
) -> None:
    workload, host, _cohort, trials, recovery = _prepared(tmp_path)
    root = tmp_path / "private"
    store = ConcurrencyExperimentMaterialStore(
        root,
        max_private_bytes=100_000_000,
        max_members=1_000,
    )
    store.put_host(host)
    store.put_workload(workload)
    result = ConcurrencyExperimentRunner(
        executor=_Executor(),
        recovery_executor=_RecoveryExecutor(),
        clock_ns=_Clock(),
        material_store=store,
    ).run(
        workload=workload,
        policy=policy(),
        host_summary=host,
        prepared_trials=trials,
        recovery_cases=recovery,
        run_root=tmp_path / "run",
        audit=workload.audit,
    )

    assert store.get_case_sample(result.case_samples[0].to_ref()) == (result.case_samples[0])
    assert store.get_trial_result(result.trial_results[0].to_ref()) == (result.trial_results[0])
    assert store.get_level_result(result.level_results[0].to_ref()) == (result.level_results[0])
    assert store.get_recovery_result(result.recovery_results[0].to_ref()) == (result.recovery_results[0])
    assert store.get_result_set(result.result_set.to_ref()) == result.result_set
    assert store.get_run_ledger(result.run_ledger.to_ref()) == result.run_ledger

    closure = store.get_result_set_closure(result.result_set.to_ref())
    assert closure.result_set == result.result_set
    assert len(closure.level_results) == 5
    assert len(closure.recovery_results) == 24

    first_trial = store.get_trial_result(closure.level_results[0].trial_result_refs[0])
    missing_case_ref = first_trial.case_sample_refs[0]
    missing_envelope = next(
        (root / "envelopes" / "case-sample").rglob(f"{missing_case_ref.object_sha256}.json")
    )
    missing_envelope.unlink()
    with pytest.raises(
        ConcurrencyExperimentStoreIntegrityError,
        match="envelope is missing",
    ):
        store.get_result_set_closure(result.result_set.to_ref())
