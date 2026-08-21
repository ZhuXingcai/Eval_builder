from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from eval_factory.contracts.approval import ApprovalMode
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.external_stability_v2 import (
    ExternalRealTraceStabilityPolicyV2,
    ExternalRealTraceStabilityReportV2,
)
from eval_factory.contracts.orchestration import (
    ConcurrencyLimit,
    ExportTarget,
    JobStatus,
    ResourceBudget,
    TraceSourceRef,
)
from eval_factory.contracts.orchestration_v2 import (
    DatasetJobSpecV2,
    StageNameV2,
    dataset_job_spec_v2_ref,
)
from eval_factory.contracts.real_trace_stability_v2 import (
    RealTraceStabilityCaseOutcomeV2,
    RealTraceStabilityFaultPointV2,
    RealTraceStabilityPolicyV2,
    RealTraceStabilityReasonCodeV2,
)
from eval_factory.orchestration.job_store import RecordNotFoundError
from eval_factory.orchestration.runner import (
    StaticTraceRunnerFaultInjector,
    TraceRunnerFaultPoint,
    TraceRunnerInjectedCrash,
)
from eval_factory.readiness.external_evidence_admission import (
    ExternalCorpusInventoryCompilation,
)
from eval_factory.readiness.external_stability_models import (
    ExternalRealTraceStabilityCaseResultV1,
    ExternalRealTraceStabilityResultSetV1,
)
from eval_factory.readiness.real_trace_stability import (
    _admit_fault_observation,
    _classify,
    _read_case_evidence,
    _read_fault_observation,
    _serial_runner,
    _source_trace_ref,
    _stores,
)
from eval_factory.readiness.real_trace_stability import (
    _private_root as _kernel_private_root,
)
from eval_factory.readiness.real_trace_stability_models import (
    PreparedRealTraceStabilityCase,
    RealTraceStabilityMemberV1,
)


class ExternalRealTraceStabilityError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ExternalRealTraceStabilityRunResult:
    case_results: tuple[ExternalRealTraceStabilityCaseResultV1, ...]
    result_set: ExternalRealTraceStabilityResultSetV1
    report: ExternalRealTraceStabilityReportV2


class ExternalRealTraceStabilityCaseExecutor:
    def execute(
        self,
        prepared: PreparedRealTraceStabilityCase,
        *,
        child_root: Path,
        audit: ContractAudit,
    ) -> ExternalRealTraceStabilityCaseResultV1:
        root = _kernel_private_root(child_root)
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
                return _external_closed_result(
                    prepared,
                    audit=audit,
                    fault_observed=fault_observed,
                )
        clean = _serial_runner(
            prepared,
            job_store=job_store,
            source_registry=source_registry,
            trace_store=trace_store,
        )
        if job.status is not JobStatus.SUCCEEDED:
            resume_count = 1
            try:
                clean.resume(prepared.job_spec.job_id)
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
                replay_summary = clean.run(prepared.job_spec)
            except Exception as exc:
                caught_error = caught_error or exc
        evidence = _read_case_evidence(
            prepared,
            job_store=job_store,
            trace_store=trace_store,
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
        return ExternalRealTraceStabilityCaseResultV1.create(
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


class ExternalRealTraceStabilityRunner:
    def __init__(
        self,
        *,
        executor: ExternalRealTraceStabilityCaseExecutor | None = None,
    ) -> None:
        self.executor = executor or ExternalRealTraceStabilityCaseExecutor()

    def prepare_cases(
        self,
        *,
        compilation: ExternalCorpusInventoryCompilation,
        policy: ExternalRealTraceStabilityPolicyV2,
        kernel_audit: ContractAudit,
    ) -> tuple[PreparedRealTraceStabilityCase, ...]:
        if (
            policy.private_inventory_ref != compilation.inventory.to_ref()
            or policy.source_count != len(compilation.inventory.members)
            or policy.inventory_sha256 != compilation.inventory.inventory_sha256
        ):
            raise ExternalRealTraceStabilityError("external stability policy differs from admitted inventory")
        faults = tuple(RealTraceStabilityFaultPointV2)
        prepared: list[PreparedRealTraceStabilityCase] = []
        for index, external_member in enumerate(compilation.inventory.members):
            member = RealTraceStabilityMemberV1.create(
                instance_id=(f"EXT_{external_member.raw_sha256[:16].upper()}"),
                sid=external_member.sid,
                business=external_member.business,
                category=external_member.runtime.value,
                pool_id=f"{index + 1}",
                pool_category=external_member.runtime.value,
                raw_sha256=external_member.raw_sha256,
                size_bytes=external_member.size_bytes,
                assigned_fault_point=faults[index % len(faults)],
            )
            raw_path = compilation.path_for(external_member.source_trace_id)
            trace = TraceSourceRef(
                source_trace_id=external_member.source_trace_id,
                source_uri=raw_path.resolve().as_uri(),
                raw_sha256=external_member.raw_sha256,
                adapter_name=external_member.adapter_name,
                adapter_version=external_member.adapter_version,
                processing_class="RESTRICTED_TRACE_RAW",
            )
            digest = member.case_key.rsplit("/", 1)[-1]
            member_ref = member.to_ref()
            spec_audit = kernel_audit.model_copy(
                update={
                    "input_refs": tuple(
                        sorted(
                            {
                                policy.to_ref(),
                                policy.package_manifest_ref,
                                member_ref,
                            },
                            key=_ref_key,
                        )
                    )
                }
            )
            spec = DatasetJobSpecV2(
                job_id=f"job://external-real-trace-stability/{digest}",
                traces=(trace,),
                requested_stages=(StageNameV2.TRACE_INDEX,),
                privacy_profile="trusted-monitored-local",
                model_profiles=(),
                budget=ResourceBudget(
                    max_model_requests=0,
                    max_model_tokens=0,
                    max_processes=1,
                    max_renderers=0,
                    max_network_requests=0,
                    max_storage_bytes=policy.max_source_bytes * 4,
                ),
                concurrency=ConcurrencyLimit(
                    model_requests=1,
                    processes=1,
                    renderers=1,
                    network_requests=1,
                    artifacts_per_item=1,
                    items=1,
                ),
                approval_policy_ref=_static_ref(
                    "user-approval-policy",
                    "external-real-trace-stability-none",
                ),
                approval_mode=ApprovalMode.NONE,
                enabled_checkpoints=frozenset(),
                export_target=ExportTarget(
                    profile="LH",
                    profile_version="v1",
                    channel="CANARY",
                    registry=("registry://external-real-trace-stability-unused"),
                ),
                idempotency_key=(f"external-real-trace-stability-{digest}"),
                audit=spec_audit,
            )
            prepared.append(
                PreparedRealTraceStabilityCase(
                    member=member,
                    raw_path=raw_path,
                    trace=trace,
                    policy=cast(RealTraceStabilityPolicyV2, policy),
                    job_spec=spec,
                    dataset_job_spec_ref=dataset_job_spec_v2_ref(spec),
                )
            )
        return tuple(prepared)

    def run(
        self,
        *,
        compilation: ExternalCorpusInventoryCompilation,
        policy: ExternalRealTraceStabilityPolicyV2,
        run_root: Path,
        kernel_audit: ContractAudit,
        wrapper_audit: ContractAudit,
    ) -> ExternalRealTraceStabilityRunResult:
        root = _private_root(run_root)
        prepared = self.prepare_cases(
            compilation=compilation,
            policy=policy,
            kernel_audit=kernel_audit,
        )
        results: list[ExternalRealTraceStabilityCaseResultV1] = []
        for case in prepared:
            try:
                result = self.executor.execute(
                    case,
                    child_root=(root / "children" / case.member.case_key.rsplit("/", 1)[-1]),
                    audit=kernel_audit,
                )
            except Exception:
                result = _external_closed_result(
                    case,
                    audit=kernel_audit,
                    fault_observed=False,
                )
            results.append(result)
        result_set = ExternalRealTraceStabilityResultSetV1.create(
            policy_ref=policy.to_ref(),
            private_inventory_ref=compilation.inventory.to_ref(),
            case_results=tuple(results),
            audit=wrapper_audit,
        )
        report = ExternalRealTraceStabilityReportV2.create(
            policy_ref=policy.to_ref(),
            package_manifest_ref=policy.package_manifest_ref,
            private_inventory_ref=compilation.inventory.to_ref(),
            private_result_set_ref=result_set.to_ref(),
            unique_real_trace_count=len(results),
            case_summaries=result_set.case_summaries,
            audit=wrapper_audit,
        )
        return ExternalRealTraceStabilityRunResult(
            case_results=tuple(results),
            result_set=result_set,
            report=report,
        )


def _private_root(path: Path) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink() or (candidate.exists() and not candidate.is_dir()):
        raise ExternalRealTraceStabilityError("external stability root must be a non-symlink directory")
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate.resolve()


def _external_closed_result(
    prepared: PreparedRealTraceStabilityCase,
    *,
    audit: ContractAudit,
    fault_observed: bool,
) -> ExternalRealTraceStabilityCaseResultV1:
    return ExternalRealTraceStabilityCaseResultV1.create(
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
        resume_succeeded=False,
        replay_stable=False,
        attempt_count=0,
        unexpected_retry_count=0,
        resume_count=0,
        replay_count=0,
        outcome=RealTraceStabilityCaseOutcomeV2.INFRASTRUCTURE_ERROR,
        reason_code=RealTraceStabilityReasonCodeV2.INTERNAL_ERROR,
        audit=audit,
    )


def _static_ref(object_type: str, suffix: str) -> ObjectRef:
    digest = hashlib.sha256(f"{object_type}:{suffix}:v1".encode()).hexdigest()
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version="v1",
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
    "ExternalRealTraceStabilityCaseExecutor",
    "ExternalRealTraceStabilityError",
    "ExternalRealTraceStabilityRunResult",
    "ExternalRealTraceStabilityRunner",
]
