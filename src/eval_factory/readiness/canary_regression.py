from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from eval_factory.contracts.canary_execution_v2 import (
    R6_CANARY_STAGES,
    R6CanaryDatasetResultV2,
    R6CanaryObjectCodecV2,
    r6_canary_execution_manifest_v2_ref,
)
from eval_factory.contracts.canary_regression_v2 import (
    CanaryRegressionCaseResultV2,
    CanaryRegressionObservedOutcomeV2,
    CanaryRegressionPolicyV2,
    CanaryRegressionReasonCodeV2,
    CanaryRegressionReportV2,
    r6_canary_dataset_result_v2_ref,
    validate_canary_regression_policy_v2_identity,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.orchestration import (
    ItemStatus,
    JobStatus,
    StageRunStatus,
)
from eval_factory.contracts.orchestration_v2 import StageNameV2
from eval_factory.contracts.resource_v2 import ResourceAdmissionOutcomeV2
from eval_factory.orchestration.canary_driver import (
    CanaryPipelineDriver,
    create_canary_stage_store,
)
from eval_factory.orchestration.canary_errors import (
    CanaryProfileError,
    CanaryResourceAdmissionError,
)
from eval_factory.orchestration.job_store import (
    JobStore,
    JobStoreError,
    RecordNotFoundError,
    StaleWorkLeaseError,
)
from eval_factory.orchestration.lifecycle import PipelineLifecyclePolicyError
from eval_factory.orchestration.models import stage_result_record_ref
from eval_factory.orchestration.stage_object_store import (
    StageObjectStore,
    StageObjectStoreError,
)
from eval_factory.readiness.canary_regression_builder import (
    CanaryRegressionBuilder,
)
from eval_factory.readiness.canary_regression_models import (
    CanaryRegressionRunLedgerV1,
    CanaryRegressionTemplateV1,
    FrozenCanaryCohortV1,
    PreparedCanaryRegressionCase,
)
from eval_factory.trace import TraceIndexStore, TraceSourceRegistry


@dataclass(frozen=True, slots=True)
class _ChildEvidence:
    job_status: JobStatus | None = None
    item_id: str | None = None
    item_status: ItemStatus | None = None
    stage_result_refs: tuple[ObjectRef, ...] = ()
    highest_reached_stage: StageNameV2 | None = None
    terminal_stage_status: StageRunStatus | None = None
    work_unit_count: int = 0
    work_lease_count: int = 0
    stage_run_count: int = 0
    attempt_count: int = 0
    retry_count: int = 0
    provider_invocation_count: int = 0


class CanaryRegressionCaseEvidenceCompiler:
    def compile(
        self,
        prepared: PreparedCanaryRegressionCase,
        *,
        job_store: JobStore,
        stage_store: StageObjectStore,
        child_result: R6CanaryDatasetResultV2 | None,
        error: Exception | None,
        audit: ContractAudit,
    ) -> CanaryRegressionCaseResultV2:
        manifest = prepared.child_manifest
        binding = manifest.trace_bindings[0]
        child_manifest_ref = r6_canary_execution_manifest_v2_ref(manifest)
        evidence_error: Exception | None = None
        try:
            evidence = self._read_evidence(
                job_id=manifest.job_spec.job_id,
                job_store=job_store,
                stage_store=stage_store,
            )
        except (JobStoreError, StageObjectStoreError, ValueError) as exc:
            evidence = _ChildEvidence()
            evidence_error = exc

        outcome, reason = self._classify(
            evidence=evidence,
            child_result=child_result,
            child_manifest_ref=child_manifest_ref,
            error=evidence_error or error,
        )
        successful = outcome is CanaryRegressionObservedOutcomeV2.SUCCEEDED
        return CanaryRegressionCaseResultV2.create(
            instance_id=prepared.cohort_case.instance_id,
            source_trace_ref=binding.source_trace_ref,
            signal_quality=prepared.cohort_case.signal_quality,
            expectation=prepared.expectation,
            observed_outcome=outcome,
            reason_code=reason,
            highest_reached_stage=evidence.highest_reached_stage,
            child_manifest_ref=child_manifest_ref,
            child_dataset_result_ref=(
                r6_canary_dataset_result_v2_ref(child_result)
                if successful and child_result is not None
                else None
            ),
            job_id=manifest.job_spec.job_id,
            job_status=evidence.job_status,
            item_id=evidence.item_id,
            item_status=evidence.item_status,
            stage_result_refs=evidence.stage_result_refs,
            quality_result_refs=(
                child_result.quality_result_refs if successful and child_result is not None else ()
            ),
            package_manifest_refs=(
                child_result.package_manifest_refs if successful and child_result is not None else ()
            ),
            audit_report_ref=(
                child_result.audit_report_ref if successful and child_result is not None else None
            ),
            work_unit_count=evidence.work_unit_count,
            work_lease_count=evidence.work_lease_count,
            stage_run_count=evidence.stage_run_count,
            attempt_count=evidence.attempt_count,
            retry_count=evidence.retry_count,
            resume_count=0,
            provider_invocation_count=evidence.provider_invocation_count,
            audit=audit,
        )

    @staticmethod
    def _read_evidence(
        *,
        job_id: str,
        job_store: JobStore,
        stage_store: StageObjectStore,
    ) -> _ChildEvidence:
        try:
            job = job_store.get_job(job_id)
        except RecordNotFoundError:
            return _ChildEvidence()
        items = job_store.list_items(job_id)
        if len(items) > 1:
            raise ValueError("isolated child has ambiguous Item state")
        item = items[0] if items else None
        units = job_store.list_work_units(job_id=job_id)
        leases = job_store.list_work_leases(job_id=job_id)
        runs = job_store.list_stage_runs(job_id=job_id)
        completed = tuple(
            result
            for run in runs
            if (result := job_store.get_stage_result_for_run(run.stage_run_id)) is not None
        )
        stage_refs = tuple(stage_result_record_ref(result) for result in completed)
        highest = _highest_stage(
            tuple(
                run.stage for run in runs if job_store.get_stage_result_for_run(run.stage_run_id) is not None
            )
        )
        terminal_status = None
        if highest is not None:
            statuses = tuple(
                result.status
                for run, result in (
                    (run, job_store.get_stage_result_for_run(run.stage_run_id)) for run in runs
                )
                if result is not None and run.stage is highest
            )
            if statuses:
                terminal_status = statuses[-1]
        provider_count = sum(
            envelope.codec is R6CanaryObjectCodecV2.PROVIDER_OPERATION_OUTPUT
            for envelope in stage_store.list_envelopes()
        )
        return _ChildEvidence(
            job_status=job.status,
            item_id=item.item_id if item is not None else None,
            item_status=item.status if item is not None else None,
            stage_result_refs=tuple(sorted(stage_refs, key=_ref_key)),
            highest_reached_stage=highest,
            terminal_stage_status=terminal_status,
            work_unit_count=len(units),
            work_lease_count=len(leases),
            stage_run_count=len(runs),
            attempt_count=len(leases),
            retry_count=sum(lease.attempt > 1 for lease in leases),
            provider_invocation_count=provider_count,
        )

    @staticmethod
    def _classify(
        *,
        evidence: _ChildEvidence,
        child_result: R6CanaryDatasetResultV2 | None,
        child_manifest_ref: ObjectRef,
        error: Exception | None,
    ) -> tuple[
        CanaryRegressionObservedOutcomeV2,
        CanaryRegressionReasonCodeV2,
    ]:
        if _is_complete_success(
            evidence,
            child_result=child_result,
            child_manifest_ref=child_manifest_ref,
        ):
            return (
                CanaryRegressionObservedOutcomeV2.SUCCEEDED,
                CanaryRegressionReasonCodeV2.NONE,
            )
        if (
            evidence.item_status is ItemStatus.REJECTED
            and evidence.highest_reached_stage is StageNameV2.LABEL
            and evidence.stage_result_refs
        ):
            return (
                CanaryRegressionObservedOutcomeV2.BLOCKED,
                CanaryRegressionReasonCodeV2.LABEL_NO_MATCH,
            )
        if (
            evidence.item_status is ItemStatus.FAILED
            and evidence.highest_reached_stage is not None
            and evidence.stage_result_refs
        ):
            return (
                CanaryRegressionObservedOutcomeV2.FAILED,
                _terminal_reason(evidence.terminal_stage_status),
            )
        if error is not None:
            return (
                CanaryRegressionObservedOutcomeV2.INFRASTRUCTURE_ERROR,
                _error_reason(error),
            )
        return (
            CanaryRegressionObservedOutcomeV2.INCOMPLETE,
            CanaryRegressionReasonCodeV2.CHILD_STATE_INCOMPLETE,
        )


class CanaryRegressionCaseExecutor(Protocol):
    async def execute(
        self,
        prepared: PreparedCanaryRegressionCase,
        *,
        child_root: Path,
        audit: ContractAudit,
    ) -> CanaryRegressionCaseResultV2: ...


class R6CanaryRegressionCaseExecutor:
    def __init__(
        self,
        *,
        compiler: CanaryRegressionCaseEvidenceCompiler | None = None,
    ) -> None:
        self._compiler = compiler or CanaryRegressionCaseEvidenceCompiler()

    async def execute(
        self,
        prepared: PreparedCanaryRegressionCase,
        *,
        child_root: Path,
        audit: ContractAudit,
    ) -> CanaryRegressionCaseResultV2:
        root = _private_root(child_root)
        driver = CanaryPipelineDriver(
            job_store=JobStore(root / "job.sqlite3"),
            source_registry=TraceSourceRegistry(root / "source.sqlite3"),
            trace_store=TraceIndexStore(root / "trace-store"),
            stage_store=create_canary_stage_store(root / "stage-store"),
        )
        child_result: R6CanaryDatasetResultV2 | None = None
        error: Exception | None = None
        try:
            child_result = await driver.run(prepared.child_manifest)
        except Exception as exc:
            error = exc
        return self._compiler.compile(
            prepared,
            job_store=driver.job_store,
            stage_store=driver.stage_store,
            child_result=child_result,
            error=error,
            audit=audit,
        )


class CanaryRegressionRunError(RuntimeError):
    pass


class CanaryRegressionRunConflictError(CanaryRegressionRunError):
    pass


class CanaryRegressionRunner:
    def __init__(
        self,
        *,
        builder: CanaryRegressionBuilder | None = None,
        executor: CanaryRegressionCaseExecutor | None = None,
    ) -> None:
        self._builder = builder or CanaryRegressionBuilder()
        self._executor = executor or R6CanaryRegressionCaseExecutor()

    async def run(
        self,
        *,
        cohort: FrozenCanaryCohortV1,
        template: CanaryRegressionTemplateV1,
        policy: CanaryRegressionPolicyV2,
        raw_root: Path,
        run_root: Path,
        audit: ContractAudit,
    ) -> CanaryRegressionReportV2:
        try:
            validate_canary_regression_policy_v2_identity(policy)
            cohort_ref = cohort.to_ref()
            template_ref = template.to_ref()
        except ValueError as exc:
            raise CanaryRegressionRunError("canary regression authority is stale") from exc
        if cohort.manifest_ref != policy.cohort_manifest_ref:
            raise CanaryRegressionRunError("canary cohort differs from regression policy")
        root = _private_root(run_root)
        prepared = tuple(
            self._builder.prepare_case(
                case,
                template=template,
                raw_root=raw_root,
                preparation_root=root / "preparation",
                policy=policy,
                audit=audit,
            )
            for case in cohort.cases
        )
        ledger = CanaryRegressionRunLedgerV1.create(
            cohort_ref=cohort_ref,
            template_ref=template_ref,
            policy_ref=policy.to_ref(),
            child_manifest_refs=tuple(
                r6_canary_execution_manifest_v2_ref(case.child_manifest) for case in prepared
            ),
        )
        _admit_ledger(root, ledger)

        results: list[CanaryRegressionCaseResultV2] = []
        for case in prepared:
            try:
                result = await self._executor.execute(
                    case,
                    child_root=root / "children" / case.cohort_case.instance_id,
                    audit=audit,
                )
                _validate_case_binding(case, result)
            except Exception:
                result = _internal_error_result(case, audit=audit)
            results.append(result)
        return CanaryRegressionReportV2.create(
            cohort_manifest_ref=cohort.manifest_ref,
            policy_ref=policy.to_ref(),
            template_ref=template_ref,
            case_results=tuple(results),
            audit=audit,
        )


def _is_complete_success(
    evidence: _ChildEvidence,
    *,
    child_result: R6CanaryDatasetResultV2 | None,
    child_manifest_ref: ObjectRef,
) -> bool:
    return (
        child_result is not None
        and child_result.manifest_ref == child_manifest_ref
        and child_result.job_status is JobStatus.SUCCEEDED
        and evidence.job_status is JobStatus.SUCCEEDED
        and evidence.item_id is not None
        and evidence.item_status is ItemStatus.APPROVED
        and child_result.item_ids == (evidence.item_id,)
        and child_result.succeeded_item_ids == (evidence.item_id,)
        and not child_result.blocked_item_ids
        and not child_result.failed_item_ids
        and evidence.highest_reached_stage is StageNameV2.ITEM_QUALITY
        and bool(evidence.stage_result_refs)
        and len(child_result.quality_result_refs) == 1
        and len(child_result.package_manifest_refs) == 1
    )


def _highest_stage(stages: tuple[StageNameV2, ...]) -> StageNameV2 | None:
    admitted = tuple(stage for stage in stages if stage in R6_CANARY_STAGES)
    if not admitted:
        return None
    return max(admitted, key=R6_CANARY_STAGES.index)


def _terminal_reason(
    status: StageRunStatus | None,
) -> CanaryRegressionReasonCodeV2:
    if status is StageRunStatus.BLOCKED_POLICY:
        return CanaryRegressionReasonCodeV2.STAGE_BLOCKED_POLICY
    if status is StageRunStatus.BLOCKED_CAPABILITY:
        return CanaryRegressionReasonCodeV2.STAGE_BLOCKED_CAPABILITY
    return CanaryRegressionReasonCodeV2.STAGE_RETRY_EXHAUSTED


def _error_reason(error: Exception) -> CanaryRegressionReasonCodeV2:
    if isinstance(error, CanaryResourceAdmissionError):
        return {
            ResourceAdmissionOutcomeV2.WAITING_CAPACITY: (CanaryRegressionReasonCodeV2.RESOURCE_WAITING),
            ResourceAdmissionOutcomeV2.BUDGET_EXHAUSTED: (
                CanaryRegressionReasonCodeV2.RESOURCE_BUDGET_EXHAUSTED
            ),
            ResourceAdmissionOutcomeV2.UNSATISFIABLE_DEMAND: (
                CanaryRegressionReasonCodeV2.RESOURCE_UNSATISFIABLE
            ),
            ResourceAdmissionOutcomeV2.ADMITTED: (CanaryRegressionReasonCodeV2.INTERNAL_ERROR),
        }[error.outcome]
    if isinstance(error, CanaryProfileError):
        return CanaryRegressionReasonCodeV2.PROFILE_POLICY_ERROR
    if isinstance(error, PipelineLifecyclePolicyError):
        return CanaryRegressionReasonCodeV2.LIFECYCLE_POLICY_ERROR
    if isinstance(error, StaleWorkLeaseError):
        return CanaryRegressionReasonCodeV2.INTERNAL_ERROR
    if isinstance(error, (StageObjectStoreError, JobStoreError)):
        return CanaryRegressionReasonCodeV2.MATERIAL_INTEGRITY_ERROR
    return CanaryRegressionReasonCodeV2.INTERNAL_ERROR


def _private_root(path: Path) -> Path:
    if path.is_symlink():
        raise CanaryRegressionRunError("canary regression root cannot be a symlink")
    if path.exists() and not path.is_dir():
        raise CanaryRegressionRunError("canary regression root is not a directory")
    path.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve()
    if not resolved.is_dir():
        raise CanaryRegressionRunError("canary regression root is not a directory")
    return resolved


def _admit_ledger(
    root: Path,
    ledger: CanaryRegressionRunLedgerV1,
) -> None:
    path = root / "run-ledger.json"
    encoded = ledger.canonical_json() + b"\n"
    if path.exists():
        _require_ledger(path, ledger)
        return
    temporary = root / f".run-ledger-{ledger.ledger_sha256}-{os.getpid()}.tmp"
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
    expected: CanaryRegressionRunLedgerV1,
) -> None:
    try:
        observed = CanaryRegressionRunLedgerV1.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise CanaryRegressionRunConflictError("canary regression run ledger is invalid") from exc
    if observed != expected:
        raise CanaryRegressionRunConflictError("canary regression run ledger conflicts with this run")


def _validate_case_binding(
    prepared: PreparedCanaryRegressionCase,
    result: CanaryRegressionCaseResultV2,
) -> None:
    binding = prepared.child_manifest.trace_bindings[0]
    if (
        result.instance_id != prepared.cohort_case.instance_id
        or result.expectation is not prepared.expectation
        or result.signal_quality != prepared.cohort_case.signal_quality
        or result.source_trace_ref != binding.source_trace_ref
        or result.child_manifest_ref != r6_canary_execution_manifest_v2_ref(prepared.child_manifest)
        or result.job_id != prepared.child_manifest.job_spec.job_id
    ):
        raise CanaryRegressionRunError("canary regression case result crosses child identity")


def _internal_error_result(
    prepared: PreparedCanaryRegressionCase,
    *,
    audit: ContractAudit,
) -> CanaryRegressionCaseResultV2:
    manifest = prepared.child_manifest
    return CanaryRegressionCaseResultV2.create(
        instance_id=prepared.cohort_case.instance_id,
        source_trace_ref=manifest.trace_bindings[0].source_trace_ref,
        signal_quality=prepared.cohort_case.signal_quality,
        expectation=prepared.expectation,
        observed_outcome=(CanaryRegressionObservedOutcomeV2.INFRASTRUCTURE_ERROR),
        reason_code=CanaryRegressionReasonCodeV2.INTERNAL_ERROR,
        highest_reached_stage=None,
        child_manifest_ref=r6_canary_execution_manifest_v2_ref(manifest),
        child_dataset_result_ref=None,
        job_id=manifest.job_spec.job_id,
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
        audit=audit,
    )


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )
