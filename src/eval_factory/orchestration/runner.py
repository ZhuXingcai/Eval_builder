from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import unquote, urlparse

from pydantic import Field

from eval_factory.contracts.core import (
    ContractAudit,
    ContractModel,
    FailureClass,
    FailureRecord,
    Identifier,
    ObjectRef,
    TypedAttribute,
)
from eval_factory.contracts.orchestration import ItemStatus, JobStatus, StageRunStatus, TraceSourceRef
from eval_factory.contracts.orchestration_v2 import DatasetJobSpecV2, StageNameV2
from eval_factory.orchestration.job_store import IllegalTransitionError, JobStore
from eval_factory.orchestration.models import ItemRecord, JobRecord, StageRunRecord
from eval_factory.trace.indexing import TraceIndexBuilder
from eval_factory.trace.normalization import RawTrajV1Normalizer
from eval_factory.trace.parsing import RawTrajRecovery, RawTrajV1Parser
from eval_factory.trace.source_registry import TraceSourceRegistry
from eval_factory.trace.storage import TraceIndexStore
from eval_factory.trace.storage.models import StoredTraceIndex


class TraceRunnerError(RuntimeError):
    pass


class TraceRunnerCancelledError(TraceRunnerError):
    pass


class TraceRunnerStageError(TraceRunnerError):
    def __init__(self, code: str, message: str, *, retryable: bool) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(message)


class TraceRunnerInjectedCrash(TraceRunnerError):
    pass


class TraceRunnerFaultPoint(StrEnum):
    BEFORE_SOURCE_REGISTRATION = "before_source_registration"
    AFTER_SOURCE_REGISTRATION = "after_source_registration"
    AFTER_TRACE_STORE_PERSIST = "after_trace_store_persist"
    BEFORE_STAGE_COMPLETION = "before_stage_completion"


class TraceRunnerFaultInjector(Protocol):
    def maybe_raise(
        self,
        point: TraceRunnerFaultPoint,
        *,
        job_id: str,
        source_trace_id: str,
        attempt: int,
    ) -> None: ...


@dataclass(frozen=True)
class StaticTraceRunnerFaultInjector:
    failure_points: frozenset[TraceRunnerFaultPoint] = frozenset()
    crash_points: frozenset[TraceRunnerFaultPoint] = frozenset()

    def maybe_raise(
        self,
        point: TraceRunnerFaultPoint,
        *,
        job_id: str,
        source_trace_id: str,
        attempt: int,
    ) -> None:
        del job_id, source_trace_id, attempt
        if point in self.crash_points:
            raise TraceRunnerInjectedCrash(f"injected runner crash at {point.value}")
        if point in self.failure_points:
            raise TraceRunnerStageError(
                f"injected-{point.value.replace('_', '-')}",
                f"Injected runner failure at {point.value}.",
                retryable=True,
            )


@dataclass(frozen=True)
class TraceIndexStageExecution:
    trace_envelope_ref: ObjectRef
    stored_index: StoredTraceIndex


class TraceIndexStageService:
    def __init__(
        self,
        *,
        source_registry: TraceSourceRegistry,
        trace_store: TraceIndexStore,
        parser: RawTrajV1Parser | None = None,
        recovery: RawTrajRecovery | None = None,
        normalizer: RawTrajV1Normalizer | None = None,
        fault_injector: TraceRunnerFaultInjector | None = None,
    ) -> None:
        self.source_registry = source_registry
        self.trace_store = trace_store
        self.parser = parser or RawTrajV1Parser()
        self.recovery = recovery or RawTrajRecovery()
        self.normalizer = normalizer or RawTrajV1Normalizer()
        self.fault_injector = fault_injector

    def execute(
        self,
        *,
        trace: TraceSourceRef,
        audit: ContractAudit,
        job_id: Identifier,
        attempt: int,
    ) -> TraceIndexStageExecution:
        source_path = _source_path(trace)
        self._maybe_fault(
            TraceRunnerFaultPoint.BEFORE_SOURCE_REGISTRATION,
            job_id,
            trace.source_trace_id,
            attempt,
        )
        registered = self.source_registry.register(
            source_path,
            source_trace_id=trace.source_trace_id,
            source_uri=trace.source_uri,
        )
        if registered.source != trace:
            raise TraceRunnerStageError(
                "trace-source-mismatch",
                "registered trace source does not match DatasetJobSpecV2 source reference",
                retryable=False,
            )
        self._maybe_fault(
            TraceRunnerFaultPoint.AFTER_SOURCE_REGISTRATION,
            job_id,
            trace.source_trace_id,
            attempt,
        )
        parsed = self.parser.parse(
            source_path,
            registered_source=registered,
            audit=audit,
        )
        recovered = self.recovery.recover(
            source_path,
            parse_result=parsed,
            audit=audit,
        )
        normalized = self.normalizer.normalize(
            source_path,
            recovery_result=recovered,
            audit=audit,
        )
        indexed = TraceIndexBuilder().build(
            normalization_result=normalized,
            audit=audit,
        )
        self.trace_store.persist(indexed, audit=audit)
        self._maybe_fault(
            TraceRunnerFaultPoint.AFTER_TRACE_STORE_PERSIST,
            job_id,
            trace.source_trace_id,
            attempt,
        )
        stored = self.trace_store.load(indexed.normalization_result.trace_ir_version_id)
        return TraceIndexStageExecution(
            trace_envelope_ref=ObjectRef(
                object_type="trace-envelope",
                object_id=stored.manifest.trace_ir_version_id,
                object_version="v2",
                object_sha256=stored.manifest.canonical_sha256(),
            ),
            stored_index=stored,
        )

    def _maybe_fault(
        self,
        point: TraceRunnerFaultPoint,
        job_id: str,
        source_trace_id: str,
        attempt: int,
    ) -> None:
        if self.fault_injector is not None:
            self.fault_injector.maybe_raise(
                point,
                job_id=job_id,
                source_trace_id=source_trace_id,
                attempt=attempt,
            )


class TraceRunnerItemSummary(ContractModel):
    schema_version: Literal["eval-factory/trace-runner-item-summary/v1"] = (
        "eval-factory/trace-runner-item-summary/v1"
    )
    source_trace_id: Identifier
    item_id: Identifier
    item_status: ItemStatus
    stage_run_id: Identifier | None = None
    stage_status: StageRunStatus | None = None
    output_refs: tuple[ObjectRef, ...] = ()
    checkpoint_ref: ObjectRef | None = None
    skipped: bool = False
    failure_code: Identifier | None = None


class TraceRunnerSummary(ContractModel):
    schema_version: Literal["eval-factory/trace-runner-summary/v1"] = "eval-factory/trace-runner-summary/v1"
    job_id: Identifier
    job_status: JobStatus
    item_summaries: tuple[TraceRunnerItemSummary, ...]
    skipped_completed_count: int = Field(ge=0)
    executed_count: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    cancelled: bool = False
    failure_code: Identifier | None = None


@dataclass
class _RunCounters:
    skipped: int = 0
    executed: int = 0
    retries: int = 0


def _utc_now() -> datetime:
    return datetime.now(UTC)


class SerialTraceRunner:
    def __init__(
        self,
        *,
        job_store: JobStore,
        source_registry: TraceSourceRegistry,
        trace_store: TraceIndexStore,
        clock: Callable[[], datetime] | None = None,
        fault_injector: TraceRunnerFaultInjector | None = None,
        parser: RawTrajV1Parser | None = None,
        recovery: RawTrajRecovery | None = None,
        normalizer: RawTrajV1Normalizer | None = None,
    ) -> None:
        self.job_store = job_store
        self.source_registry = source_registry
        self.trace_store = trace_store
        self.clock = clock or _utc_now
        self.fault_injector = fault_injector
        self.trace_index_stage = TraceIndexStageService(
            source_registry=source_registry,
            trace_store=trace_store,
            parser=parser,
            recovery=recovery,
            normalizer=normalizer,
            fault_injector=fault_injector,
        )

    def run(self, spec: DatasetJobSpecV2) -> TraceRunnerSummary:
        self.job_store.create_job(spec)
        return self._drive(spec)

    def resume(self, job_id: Identifier) -> TraceRunnerSummary:
        job = self.job_store.get_job(job_id)
        if job.status is JobStatus.CANCELLED:
            raise TraceRunnerCancelledError(f"job is cancelled and cannot be resumed: {job_id}")
        spec = self.job_store.get_job_spec(job_id)
        return self._drive(spec)

    def cancel(self, job_id: Identifier) -> TraceRunnerSummary:
        job = self.job_store.get_job(job_id)
        if job.status is JobStatus.CANCELLED:
            return self._summary(job_id, _RunCounters(), cancelled=True)
        if job.status in {JobStatus.SUCCEEDED, JobStatus.FAILED}:
            raise TraceRunnerError(f"terminal job cannot be cancelled: {job_id}")

        self._cancel_incomplete_stages(job_id)
        current = self.job_store.get_job(job_id)
        self.job_store.transition_job(
            job_id,
            JobStatus.CANCELLED,
            expected_version=current.row_version,
            idempotency_key=_key("job-cancelled", job_id),
            reason="cancelled by operator",
        )
        return self._summary(job_id, _RunCounters(), cancelled=True)

    def _drive(self, spec: DatasetJobSpecV2) -> TraceRunnerSummary:
        if spec.requested_stages[0] is not StageNameV2.TRACE_INDEX:
            raise TraceRunnerError("DatasetJobSpecV2 must begin with trace_index")
        job = self.job_store.get_job(spec.job_id)
        if job.status is JobStatus.CANCELLED:
            raise TraceRunnerCancelledError(f"job is cancelled and cannot be resumed: {spec.job_id}")
        if job.status is JobStatus.FAILED:
            raise TraceRunnerError(f"failed job cannot be resumed: {spec.job_id}")
        if job.status is JobStatus.SUCCEEDED:
            return self._summary(spec.job_id, _RunCounters())
        self._ensure_job_running(job)

        counters = _RunCounters()
        for trace in spec.traces:
            self._raise_if_cancelled(spec.job_id)
            self._execute_trace(spec, trace, counters)
        self._complete_job_if_ready(spec.job_id)
        return self._summary(spec.job_id, counters)

    def _execute_trace(
        self,
        spec: DatasetJobSpecV2,
        trace: TraceSourceRef,
        counters: _RunCounters,
    ) -> None:
        item = self._ensure_item_running(spec.job_id, trace.source_trace_id)
        stage = self._stage_for_item(spec, trace, item.item_id, counters)
        if stage.status is StageRunStatus.SUCCEEDED:
            counters.skipped += 1
            self._approve_item_if_needed(item.item_id)
            return
        if stage.status in {StageRunStatus.TERMINAL_FAILURE, StageRunStatus.CANCELLED}:
            raise TraceRunnerStageError(
                "trace-index-stage-terminal",
                "trace_index stage is already terminal and cannot resume",
                retryable=False,
            )
        running = self._start_stage(stage)
        try:
            self._run_trace_index(spec, trace, running)
        except TraceRunnerInjectedCrash:
            raise
        except TraceRunnerStageError as exc:
            self._complete_failed_stage(running.stage_run_id, exc)
            raise
        except Exception as exc:
            wrapped = TraceRunnerStageError(
                "trace-index-runner-error",
                f"trace_index runner failed with {type(exc).__name__}",
                retryable=True,
            )
            self._complete_failed_stage(running.stage_run_id, wrapped)
            raise wrapped from exc
        counters.executed += 1
        self._approve_item_if_needed(item.item_id)

    def _run_trace_index(
        self,
        spec: DatasetJobSpecV2,
        trace: TraceSourceRef,
        stage: StageRunRecord,
    ) -> None:
        execution = self.trace_index_stage.execute(
            trace=trace,
            audit=spec.audit,
            job_id=spec.job_id,
            attempt=stage.attempt,
        )
        output_ref = execution.trace_envelope_ref
        self._maybe_fault(
            TraceRunnerFaultPoint.BEFORE_STAGE_COMPLETION,
            spec.job_id,
            trace.source_trace_id,
            stage.attempt,
        )
        self.job_store.complete_stage_run(
            stage_result_id=_stage_result_id(stage.stage_run_id, StageRunStatus.SUCCEEDED),
            stage_run_id=stage.stage_run_id,
            status=StageRunStatus.SUCCEEDED,
            expected_version=self.job_store.get_stage_run(stage.stage_run_id).row_version,
            idempotency_key=_key("complete-stage", stage.stage_run_id),
            output_refs=(output_ref,),
            checkpoint_ref=output_ref,
        )

    def _stage_for_item(
        self,
        spec: DatasetJobSpecV2,
        trace: TraceSourceRef,
        item_id: str,
        counters: _RunCounters,
    ) -> StageRunRecord:
        runs = self.job_store.list_stage_runs(
            job_id=spec.job_id,
            item_id=item_id,
            stage=StageNameV2.TRACE_INDEX,
        )
        if not runs:
            return self._create_stage_run(spec, trace, item_id, attempt=1, retry_of_stage_run_id=None)
        latest = runs[-1]
        result = self.job_store.get_stage_result_for_run(latest.stage_run_id)
        if latest.status is StageRunStatus.SUCCEEDED and result is not None:
            return latest
        if latest.status in {
            StageRunStatus.BLOCKED_POLICY,
            StageRunStatus.BLOCKED_CAPABILITY,
            StageRunStatus.RETRYABLE_FAILURE,
        }:
            counters.retries += 1
            return self._create_stage_run(
                spec,
                trace,
                item_id,
                attempt=latest.attempt + 1,
                retry_of_stage_run_id=latest.stage_run_id,
            )
        return latest

    def _create_stage_run(
        self,
        spec: DatasetJobSpecV2,
        trace: TraceSourceRef,
        item_id: str,
        *,
        attempt: int,
        retry_of_stage_run_id: str | None,
    ) -> StageRunRecord:
        return self.job_store.create_stage_run(
            stage_run_id=_stage_run_id(spec.job_id, trace.source_trace_id, attempt),
            job_id=spec.job_id,
            item_id=item_id,
            stage=StageNameV2.TRACE_INDEX,
            attempt=attempt,
            principal_ref=_principal_ref(),
            input_refs=(_trace_source_ref(trace),),
            idempotency_key=_key("create-stage", spec.job_id, trace.source_trace_id, str(attempt)),
            retry_of_stage_run_id=retry_of_stage_run_id,
        )

    def _start_stage(self, stage: StageRunRecord) -> StageRunRecord:
        current = self.job_store.get_stage_run(stage.stage_run_id)
        if current.status is StageRunStatus.RUNNING:
            return current
        if current.status is not StageRunStatus.PENDING:
            return current
        return self.job_store.transition_stage_run(
            current.stage_run_id,
            StageRunStatus.RUNNING,
            expected_version=current.row_version,
            idempotency_key=_key("start-stage", current.stage_run_id),
        )

    def _complete_failed_stage(self, stage_run_id: str, error: TraceRunnerStageError) -> None:
        current = self.job_store.get_stage_run(stage_run_id)
        if self.job_store.get_stage_result_for_run(stage_run_id) is not None:
            return
        status = StageRunStatus.RETRYABLE_FAILURE if error.retryable else StageRunStatus.TERMINAL_FAILURE
        self.job_store.complete_stage_run(
            stage_result_id=_stage_result_id(stage_run_id, status),
            stage_run_id=stage_run_id,
            status=status,
            expected_version=current.row_version,
            idempotency_key=_key("complete-stage-failure", stage_run_id, status.value),
            failure=_failure(error.code, str(error), retryable=error.retryable),
        )

    def _cancel_incomplete_stages(self, job_id: str) -> None:
        for stage in self.job_store.list_stage_runs(job_id=job_id, stage=StageNameV2.TRACE_INDEX):
            if stage.status not in {StageRunStatus.PENDING, StageRunStatus.RUNNING}:
                continue
            if self.job_store.get_stage_result_for_run(stage.stage_run_id) is not None:
                continue
            self.job_store.complete_stage_run(
                stage_result_id=_stage_result_id(stage.stage_run_id, StageRunStatus.CANCELLED),
                stage_run_id=stage.stage_run_id,
                status=StageRunStatus.CANCELLED,
                expected_version=stage.row_version,
                idempotency_key=_key("complete-stage-cancelled", stage.stage_run_id),
                failure=_failure("trace-index-cancelled", "trace_index stage cancelled", retryable=False),
            )

    def _ensure_job_running(self, job: JobRecord) -> JobRecord:
        if job.status is JobStatus.RUNNING:
            return job
        if job.status not in {JobStatus.CREATED, JobStatus.BLOCKED}:
            return job
        return self.job_store.transition_job(
            job.job_id,
            JobStatus.RUNNING,
            expected_version=job.row_version,
            idempotency_key=_key("job-running", job.job_id),
        )

    def _complete_job_if_ready(self, job_id: str) -> None:
        job = self.job_store.get_job(job_id)
        if job.status is not JobStatus.RUNNING:
            return
        items = self.job_store.list_items(job_id)
        if not items or any(item.status is not ItemStatus.APPROVED for item in items):
            return
        self.job_store.transition_job(
            job_id,
            JobStatus.SUCCEEDED,
            expected_version=job.row_version,
            idempotency_key=_key("job-succeeded", job_id),
        )

    def _ensure_item_running(self, job_id: str, source_trace_id: str) -> ItemRecord:
        item_id = _item_id(job_id, source_trace_id)
        try:
            item = self.job_store.create_item(
                job_id,
                item_id,
                idempotency_key=_key("create-item", job_id, source_trace_id),
            )
        except IllegalTransitionError:
            item = self.job_store.get_item(item_id)
        if item.status is ItemStatus.CANDIDATE:
            return self.job_store.transition_item(
                item.item_id,
                ItemStatus.RUNNING,
                expected_version=item.row_version,
                idempotency_key=_key("item-running", item.item_id),
            )
        return item

    def _approve_item_if_needed(self, item_id: str) -> ItemRecord:
        item = self.job_store.get_item(item_id)
        if item.status is ItemStatus.APPROVED:
            return item
        if item.status is ItemStatus.CANDIDATE:
            item = self.job_store.transition_item(
                item.item_id,
                ItemStatus.RUNNING,
                expected_version=item.row_version,
                idempotency_key=_key("item-running", item.item_id),
            )
        if item.status is ItemStatus.RUNNING:
            return self.job_store.transition_item(
                item.item_id,
                ItemStatus.APPROVED,
                expected_version=item.row_version,
                idempotency_key=_key("item-approved", item.item_id),
            )
        return item

    def _raise_if_cancelled(self, job_id: str) -> None:
        if self.job_store.get_job(job_id).status is JobStatus.CANCELLED:
            raise TraceRunnerCancelledError(f"job is cancelled and cannot be resumed: {job_id}")

    def _maybe_fault(
        self,
        point: TraceRunnerFaultPoint,
        job_id: str,
        source_trace_id: str,
        attempt: int,
    ) -> None:
        if self.fault_injector is None:
            return
        self.fault_injector.maybe_raise(
            point,
            job_id=job_id,
            source_trace_id=source_trace_id,
            attempt=attempt,
        )

    def _summary(
        self,
        job_id: str,
        counters: _RunCounters,
        *,
        cancelled: bool = False,
        failure_code: str | None = None,
    ) -> TraceRunnerSummary:
        job = self.job_store.get_job(job_id)
        spec = self.job_store.get_job_spec(job_id)
        items = {item.item_id: item for item in self.job_store.list_items(job_id)}
        summaries: list[TraceRunnerItemSummary] = []
        skipped = counters.skipped
        for trace in spec.traces:
            item_id = _item_id(job_id, trace.source_trace_id)
            item = items.get(item_id)
            stage = _latest(
                self.job_store.list_stage_runs(
                    job_id=job_id,
                    item_id=item_id,
                    stage=StageNameV2.TRACE_INDEX,
                )
            )
            result = (
                self.job_store.get_stage_result_for_run(stage.stage_run_id) if stage is not None else None
            )
            if counters.executed == 0 and result is not None and result.status is StageRunStatus.SUCCEEDED:
                skipped += 1
            summaries.append(
                TraceRunnerItemSummary(
                    source_trace_id=trace.source_trace_id,
                    item_id=item_id,
                    item_status=item.status if item is not None else ItemStatus.CANDIDATE,
                    stage_run_id=stage.stage_run_id if stage is not None else None,
                    stage_status=stage.status if stage is not None else None,
                    output_refs=result.output_refs if result is not None else (),
                    checkpoint_ref=result.checkpoint_ref if result is not None else None,
                    skipped=result is not None
                    and result.status is StageRunStatus.SUCCEEDED
                    and counters.executed == 0,
                    failure_code=result.failure.code
                    if result is not None and result.failure is not None
                    else None,
                )
            )
        return TraceRunnerSummary(
            job_id=job_id,
            job_status=job.status,
            item_summaries=tuple(summaries),
            skipped_completed_count=skipped,
            executed_count=counters.executed,
            retry_count=counters.retries,
            cancelled=cancelled or job.status is JobStatus.CANCELLED,
            failure_code=failure_code,
        )


def _latest(stage_runs: tuple[StageRunRecord, ...]) -> StageRunRecord | None:
    return stage_runs[-1] if stage_runs else None


def _source_path(trace: TraceSourceRef) -> Path:
    parsed = urlparse(trace.source_uri)
    if parsed.scheme != "file":
        raise TraceRunnerStageError(
            "trace-source-uri-unsupported",
            "R1 trace runner supports local file trace sources only",
            retryable=False,
        )
    if parsed.netloc not in {"", "localhost"}:
        raise TraceRunnerStageError(
            "trace-source-uri-unsupported",
            "R1 trace runner does not support remote file authorities",
            retryable=False,
        )
    return Path(unquote(parsed.path))


def _trace_source_ref(trace: TraceSourceRef) -> ObjectRef:
    return ObjectRef(
        object_type="trace-source",
        object_id=trace.source_trace_id,
        object_version=f"{trace.adapter_name}/{trace.adapter_version}",
        object_sha256=trace.canonical_sha256(),
    )


def _principal_ref() -> ObjectRef:
    payload = {"principal": "r1-09-serial-trace-runner", "stage": StageNameV2.TRACE_INDEX.value}
    return ObjectRef(
        object_type="stage-principal",
        object_id="stage-principal://r1-09-serial-trace-runner",
        object_version="v1",
        object_sha256=_hash(payload),
    )


def _failure(code: str, message: str, *, retryable: bool) -> FailureRecord:
    return FailureRecord(
        failure_class=FailureClass.INTERNAL,
        code=code,
        message=message[:2000],
        retryable=retryable,
        detail=(TypedAttribute(key="stage", value=StageNameV2.TRACE_INDEX.value),),
    )


def _stage_run_id(job_id: str, source_trace_id: str, attempt: int) -> str:
    return f"stage-run://r1-09/{_hash({'job_id': job_id, 'source_trace_id': source_trace_id, 'attempt': attempt})}"


def _stage_result_id(stage_run_id: str, status: StageRunStatus) -> str:
    return f"stage-result://r1-09/{_hash({'stage_run_id': stage_run_id, 'status': status.value})}"


def _item_id(job_id: str, source_trace_id: str) -> str:
    return f"item://r1-09/{_hash({'job_id': job_id, 'source_trace_id': source_trace_id})}"


def _key(*parts: str) -> str:
    return "r1-09:" + ":".join(parts)


def _hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
