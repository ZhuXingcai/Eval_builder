from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.core_v2 import canonical_value_v2
from eval_factory.contracts.orchestration import (
    ItemStatus,
    JobStatus,
    StageRunStatus,
)
from eval_factory.contracts.orchestration_v2 import StageNameV2
from eval_factory.contracts.real_trace_stability_v2 import (
    RealTraceStabilityCaseOutcomeV2,
    RealTraceStabilityPolicyV2,
    RealTraceStabilityReasonCodeV2,
    RealTraceStabilityReportV2,
    validate_real_trace_stability_policy_v2_identity,
)
from eval_factory.contracts.trace import ParseQuality
from eval_factory.orchestration.job_store import (
    JobStore,
    JobStoreError,
    RecordNotFoundError,
)
from eval_factory.orchestration.models import (
    OutboxEvent,
    stage_result_record_ref,
    stage_run_record_ref,
)
from eval_factory.orchestration.runner import (
    SerialTraceRunner,
    StaticTraceRunnerFaultInjector,
    TraceRunnerFaultPoint,
    TraceRunnerInjectedCrash,
    TraceRunnerStageError,
)
from eval_factory.readiness.real_trace_stability_builder import (
    RealTraceStabilityBuilder,
    RealTraceStabilityInventoryCompilation,
)
from eval_factory.readiness.real_trace_stability_models import (
    PreparedRealTraceStabilityCase,
    RealTraceStabilityCaseResultV1,
    RealTraceStabilityFaultObservationV1,
    RealTraceStabilityResultSetV1,
    RealTraceStabilityRunLedgerV1,
)
from eval_factory.trace import (
    CuratedTrajectoryV1Adapter,
    CuratedTrajectoryV1Parser,
    CuratedTrajectoryV1Recovery,
    RawTrajRecovery,
    RawTrajV1Normalizer,
    RuntimeSnapshotV1Adapter,
    RuntimeSnapshotV1Parser,
    TraceAdapter,
    TraceIndexStore,
    TraceSourceRegistry,
)
from eval_factory.trace.storage.models import StoredTraceManifest, TraceStoreError


class RealTraceStabilityRunError(RuntimeError):
    pass


class RealTraceStabilityRunConflictError(RealTraceStabilityRunError):
    pass


class RealTraceStabilityCaseExecutor(Protocol):
    def execute(
        self,
        prepared: PreparedRealTraceStabilityCase,
        *,
        child_root: Path,
        audit: ContractAudit,
    ) -> RealTraceStabilityCaseResultV1: ...


@dataclass(frozen=True, slots=True)
class RealTraceStabilityRunResult:
    case_results: tuple[RealTraceStabilityCaseResultV1, ...]
    result_set: RealTraceStabilityResultSetV1
    report: RealTraceStabilityReportV2


class R1RealTraceStabilityCaseExecutor:
    def execute(
        self,
        prepared: PreparedRealTraceStabilityCase,
        *,
        child_root: Path,
        audit: ContractAudit,
    ) -> RealTraceStabilityCaseResultV1:
        root = _private_root(child_root)
        job_store, source_registry, trace_store = _stores(root, prepared)
        fault_observed = _read_fault_observation(root, prepared) is not None
        resume_succeeded = False
        replay_stable = False
        resume_count = 0
        replay_count = 0
        caught_error: Exception | None = None

        try:
            job = job_store.get_job(prepared.job_spec.job_id)
        except RecordNotFoundError:
            runner = _serial_runner(
                prepared,
                job_store=job_store,
                source_registry=source_registry,
                trace_store=trace_store,
                fault_injector=StaticTraceRunnerFaultInjector(
                    crash_points=frozenset(
                        {TraceRunnerFaultPoint(prepared.member.assigned_fault_point.value)}
                    )
                ),
            )
            try:
                runner.run(prepared.job_spec)
            except TraceRunnerInjectedCrash:
                _admit_fault_observation(root, prepared)
                fault_observed = True
            except Exception as exc:
                caught_error = exc
            try:
                job = job_store.get_job(prepared.job_spec.job_id)
            except RecordNotFoundError:
                return _closed_case_result(
                    prepared,
                    audit=audit,
                    outcome=RealTraceStabilityCaseOutcomeV2.INCOMPLETE,
                    reason=(RealTraceStabilityReasonCodeV2.CHILD_STATE_INCOMPLETE),
                    fault_observed=fault_observed,
                )

        clean_runner = _serial_runner(
            prepared,
            job_store=job_store,
            source_registry=source_registry,
            trace_store=trace_store,
        )
        if job.status is not JobStatus.SUCCEEDED:
            resume_count = 1
            try:
                clean_runner.resume(prepared.job_spec.job_id)
                resume_succeeded = job_store.get_job(prepared.job_spec.job_id).status is JobStatus.SUCCEEDED
            except Exception as exc:
                caught_error = caught_error or exc
        else:
            resume_count = 1 if fault_observed else 0
            resume_succeeded = fault_observed

        replay_summary = None
        if job_store.get_job(prepared.job_spec.job_id).status is JobStatus.SUCCEEDED:
            replay_count = 1
            try:
                replay_summary = clean_runner.run(prepared.job_spec)
            except Exception as exc:
                caught_error = caught_error or exc

        try:
            evidence = _read_case_evidence(
                prepared,
                job_store=job_store,
                trace_store=trace_store,
            )
        except (JobStoreError, TraceStoreError, ValueError):
            return _closed_case_result(
                prepared,
                audit=audit,
                outcome=(RealTraceStabilityCaseOutcomeV2.INFRASTRUCTURE_ERROR),
                reason=(RealTraceStabilityReasonCodeV2.MATERIAL_INTEGRITY_ERROR),
                fault_observed=fault_observed,
                resume_succeeded=resume_succeeded,
                resume_count=resume_count,
                replay_count=replay_count,
            )

        replay_stable = (
            replay_summary is not None
            and replay_summary.job_status is JobStatus.SUCCEEDED
            and replay_summary.executed_count == 0
            and replay_summary.skipped_completed_count == 1
            and len(replay_summary.item_summaries) == 1
            and replay_summary.item_summaries[0].output_refs == evidence.output_refs
            and replay_summary.item_summaries[0].checkpoint_ref == evidence.checkpoint_ref
        )
        outcome, reason = _classify(
            evidence,
            fault_observed=fault_observed,
            resume_succeeded=resume_succeeded,
            replay_stable=replay_stable,
            error=caught_error,
        )
        return RealTraceStabilityCaseResultV1.create(
            member_ref=prepared.member.to_ref(),
            dataset_job_spec_ref=prepared.dataset_job_spec_ref,
            source_trace_ref=_source_trace_ref(prepared),
            job_id=prepared.job_spec.job_id,
            job_status=evidence.job_status,
            item_id=evidence.item_id,
            item_status=evidence.item_status,
            stage_run_refs=evidence.stage_run_refs,
            stage_result_refs=evidence.stage_result_refs,
            output_refs=evidence.output_refs,
            checkpoint_ref=evidence.checkpoint_ref,
            stored_manifest_ref=evidence.stored_manifest_ref,
            completion_outbox_refs=evidence.completion_outbox_refs,
            parse_quality=evidence.parse_quality,
            assigned_fault_point=prepared.member.assigned_fault_point,
            fault_observed=fault_observed,
            resume_succeeded=resume_succeeded,
            replay_stable=replay_stable,
            attempt_count=evidence.attempt_count,
            unexpected_retry_count=evidence.unexpected_retry_count,
            resume_count=resume_count,
            replay_count=replay_count,
            outcome=outcome,
            reason_code=reason,
            audit=audit,
        )


@dataclass(frozen=True, slots=True)
class _CaseEvidence:
    job_status: JobStatus | None = None
    item_id: str | None = None
    item_status: ItemStatus | None = None
    stage_run_refs: tuple[ObjectRef, ...] = ()
    stage_result_refs: tuple[ObjectRef, ...] = ()
    output_refs: tuple[ObjectRef, ...] = ()
    checkpoint_ref: ObjectRef | None = None
    stored_manifest_ref: ObjectRef | None = None
    completion_outbox_refs: tuple[ObjectRef, ...] = ()
    parse_quality: ParseQuality | None = None
    attempt_count: int = 0
    unexpected_retry_count: int = 0


class RealTraceStabilityRunner:
    def __init__(
        self,
        *,
        builder: RealTraceStabilityBuilder | None = None,
        executor: RealTraceStabilityCaseExecutor | None = None,
    ) -> None:
        self._builder = builder or RealTraceStabilityBuilder()
        self._executor = executor or R1RealTraceStabilityCaseExecutor()

    def run(
        self,
        *,
        compilation: RealTraceStabilityInventoryCompilation,
        policy: RealTraceStabilityPolicyV2,
        run_root: Path,
        audit: ContractAudit,
    ) -> RealTraceStabilityRunResult:
        try:
            validate_real_trace_stability_policy_v2_identity(policy)
            inventory_ref = compilation.inventory.to_ref()
        except ValueError as exc:
            raise RealTraceStabilityRunError("real-trace stability authority is stale") from exc
        root = _private_root(run_root)
        prepared = self._builder.prepare_cases(
            compilation,
            policy=policy,
            audit=audit,
        )
        ledger = RealTraceStabilityRunLedgerV1.create(
            inventory_ref=inventory_ref,
            policy_ref=policy.to_ref(),
            prepared=prepared,
        )
        _admit_ledger(root, ledger)

        results: list[RealTraceStabilityCaseResultV1] = []
        for case in prepared:
            try:
                result = self._executor.execute(
                    case,
                    child_root=root / "children" / case.member.case_key.rsplit("/", 1)[-1],
                    audit=audit,
                )
                _validate_case_binding(case, result)
            except Exception:
                result = _closed_case_result(
                    case,
                    audit=audit,
                    outcome=(RealTraceStabilityCaseOutcomeV2.INFRASTRUCTURE_ERROR),
                    reason=RealTraceStabilityReasonCodeV2.INTERNAL_ERROR,
                    fault_observed=False,
                )
            results.append(result)

        result_set = RealTraceStabilityResultSetV1.create(
            inventory_ref=inventory_ref,
            policy_ref=policy.to_ref(),
            case_results=tuple(results),
            audit=audit,
        )
        report = RealTraceStabilityReportV2.create(
            policy_ref=policy.to_ref(),
            corpus_summary=compilation.corpus_summary,
            private_result_set_ref=result_set.to_ref(),
            case_summaries=result_set.case_summaries,
            audit=audit,
        )
        return RealTraceStabilityRunResult(
            case_results=tuple(results),
            result_set=result_set,
            report=report,
        )


def _read_case_evidence(
    prepared: PreparedRealTraceStabilityCase,
    *,
    job_store: JobStore,
    trace_store: TraceIndexStore,
) -> _CaseEvidence:
    job = job_store.get_job(prepared.job_spec.job_id)
    items = job_store.list_items(prepared.job_spec.job_id)
    if len(items) != 1:
        raise ValueError("isolated stability child has ambiguous Item state")
    item = items[0]
    runs = job_store.list_stage_runs(
        job_id=prepared.job_spec.job_id,
        item_id=item.item_id,
        stage=StageNameV2.TRACE_INDEX,
    )
    results = tuple(
        result for run in runs if (result := job_store.get_stage_result_for_run(run.stage_run_id)) is not None
    )
    stage_run_refs = tuple(
        sorted(
            (stage_run_record_ref(run) for run in runs),
            key=_ref_key,
        )
    )
    stage_result_refs = tuple(
        sorted(
            (stage_result_record_ref(result) for result in results),
            key=_ref_key,
        )
    )
    successful = tuple(result for result in results if result.status is StageRunStatus.SUCCEEDED)
    output_refs: tuple[ObjectRef, ...] = ()
    checkpoint_ref: ObjectRef | None = None
    stored_manifest_ref: ObjectRef | None = None
    parse_quality: ParseQuality | None = None
    if len(successful) == 1:
        output_refs = successful[0].output_refs
        checkpoint_ref = successful[0].checkpoint_ref
        if len(output_refs) == 1:
            stored = trace_store.load(output_refs[0].object_id)
            _validate_stored_manifest(
                prepared,
                output_ref=output_refs[0],
                manifest=stored.manifest,
            )
            parse_quality = ParseQuality(stored.manifest.parse_quality)
            stored_manifest_ref = output_refs[0]
    completion_events = tuple(
        event
        for event in job_store.list_outbox()
        if event.event_type == "stage-run-completed"
        and event.aggregate_id in {run.stage_run_id for run in runs}
    )
    return _CaseEvidence(
        job_status=job.status,
        item_id=item.item_id,
        item_status=item.status,
        stage_run_refs=stage_run_refs,
        stage_result_refs=stage_result_refs,
        output_refs=output_refs,
        checkpoint_ref=checkpoint_ref,
        stored_manifest_ref=stored_manifest_ref,
        completion_outbox_refs=tuple(
            sorted((_outbox_ref(event) for event in completion_events), key=_ref_key)
        ),
        parse_quality=parse_quality,
        attempt_count=len(runs),
        unexpected_retry_count=max(len(runs) - 1, 0),
    )


def _validate_stored_manifest(
    prepared: PreparedRealTraceStabilityCase,
    *,
    output_ref: ObjectRef,
    manifest: StoredTraceManifest,
) -> None:
    trace = prepared.trace
    policy = prepared.policy
    if (
        output_ref.object_type != "trace-envelope"
        or output_ref.object_version != "v2"
        or manifest.trace_ir_version_id != output_ref.object_id
        or manifest.canonical_sha256() != output_ref.object_sha256
        or manifest.source_trace_id != trace.source_trace_id
        or manifest.source_uri != trace.source_uri
        or manifest.raw_sha256 != trace.raw_sha256
        or manifest.adapter_name != trace.adapter_name
        or manifest.adapter_version != trace.adapter_version
        or f"{manifest.adapter_name}/{manifest.adapter_version}" != policy.adapter_version
        or manifest.trace_ir_schema_version != policy.trace_ir_version
        or manifest.repair_policy_version != policy.repair_policy_version
        or manifest.recovery_policy_version != policy.recovery_policy_version
        or manifest.normalization_policy_version != policy.normalization_policy_version
        or manifest.tool_family_policy_version != policy.tool_family_policy_version
        or manifest.file_observation_policy_version != policy.file_observation_policy_version
        or manifest.segmentation_policy_version != policy.segmentation_policy_version
    ):
        raise ValueError("stored trace manifest differs from stability policy or source")


def _classify(
    evidence: _CaseEvidence,
    *,
    fault_observed: bool,
    resume_succeeded: bool,
    replay_stable: bool,
    error: Exception | None,
) -> tuple[
    RealTraceStabilityCaseOutcomeV2,
    RealTraceStabilityReasonCodeV2,
]:
    if error is not None and isinstance(error, (JobStoreError, TraceStoreError)):
        return (
            RealTraceStabilityCaseOutcomeV2.INFRASTRUCTURE_ERROR,
            RealTraceStabilityReasonCodeV2.MATERIAL_INTEGRITY_ERROR,
        )
    if (
        evidence.job_status is not JobStatus.SUCCEEDED
        or evidence.item_status is not ItemStatus.APPROVED
        or evidence.parse_quality is None
        or len(evidence.stage_result_refs) != 1
        or len(evidence.output_refs) != 1
        or evidence.checkpoint_ref != evidence.output_refs[0]
        or evidence.stored_manifest_ref != evidence.output_refs[0]
    ):
        return (
            RealTraceStabilityCaseOutcomeV2.UNSTABLE,
            (
                RealTraceStabilityReasonCodeV2.RESUME_FAILED
                if error is not None and isinstance(error, TraceRunnerStageError)
                else RealTraceStabilityReasonCodeV2.TRACE_RUN_FAILED
            ),
        )
    if not fault_observed:
        return (
            RealTraceStabilityCaseOutcomeV2.UNSTABLE,
            RealTraceStabilityReasonCodeV2.EXPECTED_FAULT_NOT_OBSERVED,
        )
    if not resume_succeeded:
        return (
            RealTraceStabilityCaseOutcomeV2.UNSTABLE,
            RealTraceStabilityReasonCodeV2.RESUME_FAILED,
        )
    if evidence.unexpected_retry_count:
        return (
            RealTraceStabilityCaseOutcomeV2.UNSTABLE,
            RealTraceStabilityReasonCodeV2.UNEXPECTED_RETRY,
        )
    if not replay_stable:
        return (
            RealTraceStabilityCaseOutcomeV2.UNSTABLE,
            RealTraceStabilityReasonCodeV2.REPLAY_MISMATCH,
        )
    if len(evidence.completion_outbox_refs) != 1:
        return (
            RealTraceStabilityCaseOutcomeV2.UNSTABLE,
            RealTraceStabilityReasonCodeV2.DUPLICATE_COMPLETION,
        )
    return (
        RealTraceStabilityCaseOutcomeV2.STABLE,
        RealTraceStabilityReasonCodeV2.NONE,
    )


def _closed_case_result(
    prepared: PreparedRealTraceStabilityCase,
    *,
    audit: ContractAudit,
    outcome: RealTraceStabilityCaseOutcomeV2,
    reason: RealTraceStabilityReasonCodeV2,
    fault_observed: bool,
    resume_succeeded: bool = False,
    resume_count: int = 0,
    replay_count: int = 0,
) -> RealTraceStabilityCaseResultV1:
    return RealTraceStabilityCaseResultV1.create(
        member_ref=prepared.member.to_ref(),
        dataset_job_spec_ref=prepared.dataset_job_spec_ref,
        source_trace_ref=_source_trace_ref(prepared),
        job_id=prepared.job_spec.job_id,
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
        assigned_fault_point=prepared.member.assigned_fault_point,
        fault_observed=fault_observed,
        resume_succeeded=resume_succeeded,
        replay_stable=False,
        attempt_count=0,
        unexpected_retry_count=0,
        resume_count=resume_count,
        replay_count=replay_count,
        outcome=outcome,
        reason_code=reason,
        audit=audit,
    )


def _validate_case_binding(
    prepared: PreparedRealTraceStabilityCase,
    result: RealTraceStabilityCaseResultV1,
) -> None:
    if (
        result.member_ref != prepared.member.to_ref()
        or result.dataset_job_spec_ref != prepared.dataset_job_spec_ref
        or result.source_trace_ref != _source_trace_ref(prepared)
        or result.job_id != prepared.job_spec.job_id
        or result.assigned_fault_point is not prepared.member.assigned_fault_point
    ):
        raise RealTraceStabilityRunError("stability case result crosses private child identity")


def _source_trace_ref(
    prepared: PreparedRealTraceStabilityCase,
) -> ObjectRef:
    trace = prepared.trace
    return ObjectRef(
        object_type="trace-source",
        object_id=trace.source_trace_id,
        object_version=f"{trace.adapter_name}/{trace.adapter_version}",
        object_sha256=trace.canonical_sha256(),
    )


def _stores(
    root: Path,
    prepared: PreparedRealTraceStabilityCase,
) -> tuple[JobStore, TraceSourceRegistry, TraceIndexStore]:
    adapter: TraceAdapter | None = None
    if prepared.trace.adapter_name == RuntimeSnapshotV1Adapter.name:
        adapter = RuntimeSnapshotV1Adapter()
    elif prepared.trace.adapter_name == CuratedTrajectoryV1Adapter.name:
        adapter = CuratedTrajectoryV1Adapter()
    return (
        JobStore(root / "job.sqlite3"),
        TraceSourceRegistry(root / "source.sqlite3", adapter=adapter),
        TraceIndexStore(root / "trace-store"),
    )


def _serial_runner(
    prepared: PreparedRealTraceStabilityCase,
    *,
    job_store: JobStore,
    source_registry: TraceSourceRegistry,
    trace_store: TraceIndexStore,
    fault_injector: StaticTraceRunnerFaultInjector | None = None,
) -> SerialTraceRunner:
    if prepared.trace.adapter_name == CuratedTrajectoryV1Adapter.name:
        curated_adapter = CuratedTrajectoryV1Adapter()
        return SerialTraceRunner(
            job_store=job_store,
            source_registry=source_registry,
            trace_store=trace_store,
            fault_injector=fault_injector,
            parser=CuratedTrajectoryV1Parser(),
            recovery=CuratedTrajectoryV1Recovery(),
            normalizer=RawTrajV1Normalizer(adapter=curated_adapter),
        )
    if prepared.trace.adapter_name != RuntimeSnapshotV1Adapter.name:
        return SerialTraceRunner(
            job_store=job_store,
            source_registry=source_registry,
            trace_store=trace_store,
            fault_injector=fault_injector,
        )
    runtime_adapter = RuntimeSnapshotV1Adapter()
    return SerialTraceRunner(
        job_store=job_store,
        source_registry=source_registry,
        trace_store=trace_store,
        fault_injector=fault_injector,
        parser=RuntimeSnapshotV1Parser(),
        recovery=RawTrajRecovery(adapter=runtime_adapter),
        normalizer=RawTrajV1Normalizer(adapter=runtime_adapter),
    )


def _private_root(path: Path) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink() or (candidate.exists() and not candidate.is_dir()):
        raise RealTraceStabilityRunError("real-trace stability root must be a non-symlink directory")
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate.resolve()


def _fault_path(root: Path) -> Path:
    return root / "fault-observation.json"


def _read_fault_observation(
    root: Path,
    prepared: PreparedRealTraceStabilityCase,
) -> RealTraceStabilityFaultObservationV1 | None:
    path = _fault_path(root)
    if not path.exists():
        return None
    try:
        value = RealTraceStabilityFaultObservationV1.model_validate_json(path.read_bytes())
    except (OSError, ValidationError, ValueError) as exc:
        raise RealTraceStabilityRunConflictError("fault observation is invalid") from exc
    expected = RealTraceStabilityFaultObservationV1.create(
        member_ref=prepared.member.to_ref(),
        fault_point=prepared.member.assigned_fault_point,
    )
    if value != expected:
        raise RealTraceStabilityRunConflictError("fault observation conflicts with this case")
    return value


def _admit_fault_observation(
    root: Path,
    prepared: PreparedRealTraceStabilityCase,
) -> None:
    value = RealTraceStabilityFaultObservationV1.create(
        member_ref=prepared.member.to_ref(),
        fault_point=prepared.member.assigned_fault_point,
    )
    path = _fault_path(root)
    encoded = value.canonical_json() + b"\n"
    if path.exists():
        _read_fault_observation(root, prepared)
        return
    temporary = root / f".fault-{value.observation_sha256}-{os.getpid()}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _read_fault_observation(root, prepared)
    finally:
        temporary.unlink(missing_ok=True)


def _admit_ledger(
    root: Path,
    ledger: RealTraceStabilityRunLedgerV1,
) -> None:
    path = root / "run-ledger.json"
    encoded = ledger.canonical_json() + b"\n"
    if path.exists():
        _require_ledger(path, ledger)
        return
    temporary = root / f".ledger-{ledger.ledger_sha256}-{os.getpid()}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _require_ledger(path, ledger)
    finally:
        temporary.unlink(missing_ok=True)


def _require_ledger(
    path: Path,
    expected: RealTraceStabilityRunLedgerV1,
) -> None:
    try:
        observed = RealTraceStabilityRunLedgerV1.model_validate_json(path.read_bytes())
    except (OSError, ValidationError, ValueError) as exc:
        raise RealTraceStabilityRunConflictError("real-trace stability run ledger is invalid") from exc
    if observed != expected:
        raise RealTraceStabilityRunConflictError("real-trace stability run ledger conflicts with this run")


def _outbox_ref(value: OutboxEvent) -> ObjectRef:
    encoded = json.dumps(
        canonical_value_v2(
            value.model_dump(
                mode="python",
                exclude={"event_id", "published_at"},
            )
        ),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    return ObjectRef(
        object_type="outbox-event",
        object_id=f"outbox-event://sha256/{digest}",
        object_version="record/v1",
        object_sha256=digest,
    )


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


__all__ = [
    "R1RealTraceStabilityCaseExecutor",
    "RealTraceStabilityCaseExecutor",
    "RealTraceStabilityRunConflictError",
    "RealTraceStabilityRunError",
    "RealTraceStabilityRunResult",
    "RealTraceStabilityRunner",
]
