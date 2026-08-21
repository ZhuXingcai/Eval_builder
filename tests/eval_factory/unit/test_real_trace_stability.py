from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_serial_trace_runner import _write_trace

from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.orchestration import ItemStatus, JobStatus
from eval_factory.contracts.real_trace_stability_v2 import (
    REAL_TRACE_STABILITY_POLICY_VERSION,
    RealTraceStabilityCaseOutcomeV2,
    RealTraceStabilityCategoryCountV2,
    RealTraceStabilityCorpusSummaryV2,
    RealTraceStabilityFaultPointV2,
    RealTraceStabilityOutcomeV2,
    RealTraceStabilityPolicyV2,
    RealTraceStabilityReasonCodeV2,
)
from eval_factory.contracts.trace import ParseQuality
from eval_factory.orchestration.job_store import JobStore
from eval_factory.readiness.real_trace_stability import (
    R1RealTraceStabilityCaseExecutor,
    RealTraceStabilityRunError,
    RealTraceStabilityRunner,
    _CaseEvidence,
    _classify,
    _private_root,
    _validate_stored_manifest,
)
from eval_factory.readiness.real_trace_stability_builder import (
    RealTraceStabilityBuilder,
    RealTraceStabilityInventoryCompilation,
)
from eval_factory.readiness.real_trace_stability_models import (
    PreparedRealTraceStabilityCase,
    RealTraceStabilityInventoryV1,
    RealTraceStabilityMemberV1,
)
from eval_factory.trace import TraceIndexStore

NOW = datetime(2026, 8, 2, tzinfo=UTC)


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="r8-04-runner-test",
        governing_versions=(
            VersionBinding(
                component="real-trace-stability",
                version=REAL_TRACE_STABILITY_POLICY_VERSION,
            ),
        ),
    )


def _manifest_ref() -> ObjectRef:
    return ObjectRef(
        object_type="real-trace-source-manifest",
        object_id="real-trace-source-manifest://manifest.csv",
        object_version="csv/v1",
        object_sha256="0d9df6c3935e00e5e15d486c9afe8cc2f581b63cf8f99ec42c1d3216cc74ee04",
    )


def _policy() -> RealTraceStabilityPolicyV2:
    return RealTraceStabilityPolicyV2.create(
        source_manifest_ref=_manifest_ref(),
        max_sources=10_000,
        max_source_bytes=100_000_000,
        max_total_source_bytes=10_000_000_000,
        max_private_bytes=32_000_000,
        max_report_bytes=8_000_000,
        audit=_audit(),
    )


def _prepared(
    tmp_path: Path,
    fault: RealTraceStabilityFaultPointV2,
) -> PreparedRealTraceStabilityCase:
    source = tmp_path / "source.jsonl"
    trace = _write_trace(source)
    member = RealTraceStabilityMemberV1.create(
        instance_id="LH_001",
        sid="sid-001",
        business="AgentPlan",
        category="science",
        pool_id="1",
        pool_category="science",
        raw_sha256=trace.raw_sha256,
        size_bytes=source.stat().st_size,
        assigned_fault_point=fault,
    )
    return RealTraceStabilityBuilder().prepare_case(
        member,
        raw_path=source,
        policy=_policy(),
        audit=_audit(),
    )


@pytest.mark.parametrize("fault", tuple(RealTraceStabilityFaultPointV2))
def test_executor_crashes_resumes_and_replays_each_r1_fault(
    tmp_path: Path,
    fault: RealTraceStabilityFaultPointV2,
) -> None:
    prepared = _prepared(tmp_path, fault)
    child = tmp_path / "child"
    executor = R1RealTraceStabilityCaseExecutor()

    first = executor.execute(prepared, child_root=child, audit=_audit())
    replay = executor.execute(prepared, child_root=child, audit=_audit())

    assert first.outcome is RealTraceStabilityCaseOutcomeV2.STABLE
    assert first.reason_code is RealTraceStabilityReasonCodeV2.NONE
    assert first.parse_quality is ParseQuality.STRICT
    assert first.fault_observed is True
    assert first.resume_succeeded is True
    assert first.replay_stable is True
    assert first.attempt_count == 1
    assert first.unexpected_retry_count == 0
    assert first.resume_count == 1
    assert first.replay_count == 1
    assert len(first.stage_result_refs) == 1
    assert len(first.completion_outbox_refs) == 1
    assert first.output_refs[0].object_version == "v2"
    assert first.to_ref() == replay.to_ref()
    public = first.to_public_summary().model_dump_json().casefold()
    assert prepared.member.instance_id.casefold() not in public
    assert prepared.member.raw_sha256 not in public
    job_store = JobStore(child / "job.sqlite3")
    assert sum(event.event_type == "stage-run-completed" for event in job_store.list_outbox()) == 1
    stored = TraceIndexStore(child / "trace-store").load(first.output_refs[0].object_id)
    _validate_stored_manifest(
        prepared,
        output_ref=first.output_refs[0],
        manifest=stored.manifest,
    )
    with pytest.raises(ValueError, match="differs from stability policy"):
        _validate_stored_manifest(
            prepared,
            output_ref=first.output_refs[0].model_copy(update={"object_sha256": "f" * 64}),
            manifest=stored.manifest,
        )
    with pytest.raises(ValueError, match="differs from stability policy"):
        _validate_stored_manifest(
            prepared,
            output_ref=first.output_refs[0].model_copy(
                update={"object_version": prepared.policy.storage_policy_version},
            ),
            manifest=stored.manifest,
        )


def _ref(object_type: str, suffix: str) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r8-04/{suffix}",
        object_version="record/v1",
        object_sha256=(suffix.encode().hex() + "a" * 64)[:64],
    )


def test_classification_matrix_is_strict_and_closed() -> None:
    output = _ref("trace-envelope", "output")
    base = _CaseEvidence(
        job_status=JobStatus.SUCCEEDED,
        item_id="item://r8-04/test",
        item_status=ItemStatus.APPROVED,
        stage_run_refs=(_ref("stage-run", "run"),),
        stage_result_refs=(_ref("stage-result", "result"),),
        output_refs=(output,),
        checkpoint_ref=output,
        stored_manifest_ref=output,
        completion_outbox_refs=(_ref("outbox-event", "event"),),
        parse_quality=ParseQuality.STRICT,
        attempt_count=1,
        unexpected_retry_count=0,
    )

    assert _classify(
        base,
        fault_observed=True,
        resume_succeeded=True,
        replay_stable=True,
        error=None,
    ) == (
        RealTraceStabilityCaseOutcomeV2.STABLE,
        RealTraceStabilityReasonCodeV2.NONE,
    )
    assert (
        _classify(
            base,
            fault_observed=False,
            resume_succeeded=True,
            replay_stable=True,
            error=None,
        )[1]
        is RealTraceStabilityReasonCodeV2.EXPECTED_FAULT_NOT_OBSERVED
    )
    assert (
        _classify(
            base,
            fault_observed=True,
            resume_succeeded=False,
            replay_stable=True,
            error=None,
        )[1]
        is RealTraceStabilityReasonCodeV2.RESUME_FAILED
    )
    assert (
        _classify(
            base,
            fault_observed=True,
            resume_succeeded=True,
            replay_stable=False,
            error=None,
        )[1]
        is RealTraceStabilityReasonCodeV2.REPLAY_MISMATCH
    )
    duplicate = replace(
        base,
        completion_outbox_refs=(
            _ref("outbox-event", "a"),
            _ref("outbox-event", "b"),
        ),
    )
    assert (
        _classify(
            duplicate,
            fault_observed=True,
            resume_succeeded=True,
            replay_stable=True,
            error=None,
        )[1]
        is RealTraceStabilityReasonCodeV2.DUPLICATE_COMPLETION
    )


def test_runner_compiles_pending_report_and_exact_ledger_replay(
    tmp_path: Path,
) -> None:
    prepared = _prepared(
        tmp_path,
        RealTraceStabilityFaultPointV2.BEFORE_SOURCE_REGISTRATION,
    )
    inventory = RealTraceStabilityInventoryV1.create(
        source_manifest_ref=_manifest_ref(),
        members=(prepared.member,),
        audit=_audit(),
    )
    corpus = RealTraceStabilityCorpusSummaryV2.create(
        source_manifest_ref=_manifest_ref(),
        policy_ref=_policy().to_ref(),
        private_inventory_ref=inventory.to_ref(),
        eligible_unique_trace_count=1,
        unique_instance_count=1,
        unique_raw_hash_count=1,
        total_raw_bytes=prepared.member.size_bytes,
        minimum_raw_bytes=prepared.member.size_bytes,
        maximum_raw_bytes=prepared.member.size_bytes,
        category_counts=(
            RealTraceStabilityCategoryCountV2(
                category="science",
                count=1,
            ),
        ),
        audit=_audit(),
    )
    compilation = RealTraceStabilityInventoryCompilation(
        inventory=inventory,
        corpus_summary=corpus,
        raw_paths=((prepared.member.instance_id, prepared.raw_path),),
    )
    runner = RealTraceStabilityRunner()

    first = runner.run(
        compilation=compilation,
        policy=_policy(),
        run_root=tmp_path / "run",
        audit=_audit(),
    )
    replay = runner.run(
        compilation=compilation,
        policy=_policy(),
        run_root=tmp_path / "run",
        audit=_audit(),
    )

    assert first.report.outcome is RealTraceStabilityOutcomeV2.STATISTICAL_GATE_PENDING
    assert first.report.observed_threshold_met is True
    assert first.report.to_ref() == replay.report.to_ref()
    assert first.result_set.to_ref() == replay.result_set.to_ref()


def test_private_root_rejects_file_and_symlink(tmp_path: Path) -> None:
    root_file = tmp_path / "file"
    root_file.write_text("x", encoding="utf-8")
    with pytest.raises(RealTraceStabilityRunError, match="non-symlink"):
        _private_root(root_file)
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(RealTraceStabilityRunError, match="non-symlink"):
        _private_root(link)
