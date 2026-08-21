from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from eval_factory.contracts.canary_execution_v2 import R6CanaryObjectCodecV2
from eval_factory.contracts.canary_regression_v2 import (
    CanaryRegressionPolicyV2,
)
from eval_factory.contracts.concurrency_experiment_v2 import (
    ConcurrencyExperimentFaultPointV2,
    ConcurrencyExperimentHostSummaryV2,
    ConcurrencyExperimentPolicyV2,
    validate_concurrency_experiment_policy_v2_identity,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.orchestration.canary_profile import R6CanaryTaskFixture
from eval_factory.readiness.canary_regression_builder import (
    CanaryRegressionBuilder,
    CanaryRegressionExecutionIdentity,
    FrozenCanaryCohortLoader,
)
from eval_factory.readiness.canary_regression_models import (
    CanaryRegressionTemplateV1,
    FrozenCanaryCaseV1,
    FrozenCanaryCohortV1,
)
from eval_factory.readiness.concurrency_experiment_models import (
    ConcurrencyExperimentTrialSpecV1,
    ConcurrencyExperimentWorkloadV1,
    PreparedConcurrencyExperimentCase,
    PreparedConcurrencyRecoveryCase,
)

ACCEPTED_CANARY_POLICY_SHA256 = "ee4c13adeb90dfd446de834424a0b6c482d421111bbd52cdaf3e9e9fd5580365"


class ConcurrencyExperimentBuilderError(RuntimeError):
    pass


class ConcurrencyExperimentPolicyError(ConcurrencyExperimentBuilderError):
    pass


class ConcurrencyExperimentHostError(ConcurrencyExperimentBuilderError):
    pass


class ConcurrencyExperimentStorageError(ConcurrencyExperimentBuilderError):
    pass


@dataclass(frozen=True, slots=True)
class HostCapacityObservation:
    logical_cpu_count: int
    physical_cpu_count: int
    memory_bytes: int
    python_version: str

    def __post_init__(self) -> None:
        if (
            self.logical_cpu_count < 1
            or self.physical_cpu_count < 1
            or self.physical_cpu_count > self.logical_cpu_count
            or self.memory_bytes < 1
            or not self.python_version
        ):
            raise ValueError("host capacity observation is invalid")


@dataclass(frozen=True, slots=True)
class ConcurrencyRecoveryExecutionIdentity:
    execution_identity: CanaryRegressionExecutionIdentity
    fault_point: ConcurrencyExperimentFaultPointV2


class ConcurrencyExperimentBuilder:
    def __init__(
        self,
        *,
        canary_builder: CanaryRegressionBuilder | None = None,
    ) -> None:
        self._canary_builder = canary_builder or CanaryRegressionBuilder()

    @staticmethod
    def load_cohort(
        path: Path,
        *,
        template: CanaryRegressionTemplateV1,
    ) -> FrozenCanaryCohortV1:
        return FrozenCanaryCohortLoader().load(
            path,
            policy=_accepted_canary_policy(template),
        )

    def observe_host(
        self,
        *,
        audit: ContractAudit,
        observation: HostCapacityObservation | None = None,
    ) -> ConcurrencyExperimentHostSummaryV2:
        observed = observation or _observe_current_host()
        machine = platform.machine().casefold()
        if sys.platform != "darwin" or machine not in {"arm64", "aarch64"}:
            raise ConcurrencyExperimentHostError("R8-06 accepted host profile requires Darwin ARM64")
        return ConcurrencyExperimentHostSummaryV2.create(
            platform_family="DARWIN",
            architecture="ARM64",
            logical_cpu_count=observed.logical_cpu_count,
            physical_cpu_count=observed.physical_cpu_count,
            memory_bytes=observed.memory_bytes,
            python_version=observed.python_version,
            audit=audit,
        )

    def compile_workload(
        self,
        *,
        policy: ConcurrencyExperimentPolicyV2,
        host_summary: ConcurrencyExperimentHostSummaryV2,
        cohort: FrozenCanaryCohortV1,
        template: CanaryRegressionTemplateV1,
        audit: ContractAudit,
    ) -> ConcurrencyExperimentWorkloadV1:
        try:
            validate_concurrency_experiment_policy_v2_identity(policy)
            cohort_ref = cohort.to_ref()
            template_ref = template.to_ref()
        except ValueError as exc:
            raise ConcurrencyExperimentPolicyError("concurrency experiment authority is stale") from exc
        if cohort.manifest_ref != policy.cohort_manifest_ref:
            raise ConcurrencyExperimentPolicyError("concurrency cohort differs from experiment policy")
        storage_per_case = _retained_storage_bytes(template)
        storage_per_cohort = storage_per_case * len(cohort.cases)
        if template.template_manifest.control_plane.resource_capacity.storage_bytes < storage_per_cohort:
            raise ConcurrencyExperimentStorageError("shared canary pool cannot retain the complete cohort")
        return ConcurrencyExperimentWorkloadV1.create(
            policy_ref=policy.to_ref(),
            cohort_ref=cohort_ref,
            template_ref=template_ref,
            host_summary=host_summary,
            retained_storage_bytes_per_case=storage_per_case,
            retained_storage_bytes_per_cohort=storage_per_cohort,
            audit=audit,
        )

    def prepare_trial(
        self,
        *,
        workload: ConcurrencyExperimentWorkloadV1,
        trial: ConcurrencyExperimentTrialSpecV1,
        cohort: FrozenCanaryCohortV1,
        template: CanaryRegressionTemplateV1,
        raw_root: Path,
        preparation_root: Path,
        audit: ContractAudit,
    ) -> tuple[PreparedConcurrencyExperimentCase, ...]:
        if trial not in workload.trials:
            raise ConcurrencyExperimentPolicyError("trial is not part of the admitted workload")
        prepared = tuple(
            self.prepare_case(
                workload=workload,
                trial=trial,
                case_ordinal=ordinal,
                case=case,
                template=template,
                raw_root=raw_root,
                preparation_root=preparation_root,
                audit=audit,
            )
            for ordinal, case in enumerate(cohort.cases)
        )
        if len({value.prepared_case.child_manifest.manifest_id for value in prepared}) != len(prepared):
            raise ConcurrencyExperimentPolicyError("prepared trial child manifests are not unique")
        return prepared

    def prepare_case(
        self,
        *,
        workload: ConcurrencyExperimentWorkloadV1,
        trial: ConcurrencyExperimentTrialSpecV1,
        case_ordinal: int,
        case: FrozenCanaryCaseV1,
        template: CanaryRegressionTemplateV1,
        raw_root: Path,
        preparation_root: Path,
        audit: ContractAudit,
    ) -> PreparedConcurrencyExperimentCase:
        if trial not in workload.trials:
            raise ConcurrencyExperimentPolicyError("trial is not part of the admitted workload")
        execution_identity = _trial_execution_identity(
            workload=workload,
            trial=trial,
            case=case,
        )
        prepared = self._canary_builder.prepare_case(
            case,
            template=template,
            raw_root=raw_root,
            preparation_root=(
                preparation_root / f"workers-{trial.worker_count}" / f"trial-{trial.trial_index}"
            ),
            policy=_accepted_canary_policy(template),
            audit=audit,
            execution_identity=execution_identity,
        )
        return PreparedConcurrencyExperimentCase(
            trial=trial,
            case_ordinal=case_ordinal,
            frozen_case=case,
            prepared_case=prepared,
        )

    def prepare_recovery_cases(
        self,
        *,
        workload: ConcurrencyExperimentWorkloadV1,
        worker_count: int,
        cohort: FrozenCanaryCohortV1,
        template: CanaryRegressionTemplateV1,
        raw_root: Path,
        preparation_root: Path,
        audit: ContractAudit,
    ) -> tuple[PreparedConcurrencyRecoveryCase, ...]:
        if worker_count not in {trial.worker_count for trial in workload.trials}:
            raise ConcurrencyExperimentPolicyError(
                "recovery worker count is not part of the admitted workload"
            )
        prepared: list[PreparedConcurrencyRecoveryCase] = []
        for ordinal, case in enumerate(cohort.cases):
            derived = derive_recovery_execution_identity(
                workload=workload,
                worker_count=worker_count,
                case_ordinal=ordinal,
                case=case,
            )
            child = self._canary_builder.prepare_case(
                case,
                template=template,
                raw_root=raw_root,
                preparation_root=(
                    preparation_root
                    / "recovery"
                    / f"workers-{worker_count}"
                    / derived.fault_point.value.casefold()
                ),
                policy=_accepted_canary_policy(template),
                audit=audit,
                execution_identity=derived.execution_identity,
            )
            prepared.append(
                PreparedConcurrencyRecoveryCase(
                    worker_count=worker_count,
                    case_ordinal=ordinal,
                    fault_point=derived.fault_point,
                    frozen_case=case,
                    prepared_case=child,
                )
            )
        if len({value.prepared_case.child_manifest.manifest_id for value in prepared}) != len(prepared):
            raise ConcurrencyExperimentPolicyError("prepared recovery child manifests are not unique")
        return tuple(prepared)


def derive_recovery_execution_identity(
    *,
    workload: ConcurrencyExperimentWorkloadV1,
    worker_count: int,
    case_ordinal: int,
    case: FrozenCanaryCaseV1,
) -> ConcurrencyRecoveryExecutionIdentity:
    if worker_count not in {trial.worker_count for trial in workload.trials}:
        raise ConcurrencyExperimentPolicyError("recovery worker count is not part of the admitted workload")
    if not 0 <= case_ordinal < len(tuple(ConcurrencyExperimentFaultPointV2)) * 6:
        raise ConcurrencyExperimentPolicyError("recovery case ordinal is out of range")
    points = tuple(ConcurrencyExperimentFaultPointV2)
    point = points[case_ordinal % len(points)]
    prefix = f"{workload.workload_sha256}-recovery-{worker_count}-{point.value.casefold()}-{case.instance_id}"
    return ConcurrencyRecoveryExecutionIdentity(
        execution_identity=CanaryRegressionExecutionIdentity(
            job_id=(
                "job://concurrency-experiment/"
                f"{workload.workload_sha256}/recovery/{worker_count}/"
                f"{point.value.casefold()}/{case.instance_id}"
            ),
            idempotency_key=f"concurrency-experiment-{prefix}",
        ),
        fault_point=point,
    )


def _trial_execution_identity(
    *,
    workload: ConcurrencyExperimentWorkloadV1,
    trial: ConcurrencyExperimentTrialSpecV1,
    case: FrozenCanaryCaseV1,
) -> CanaryRegressionExecutionIdentity:
    prefix = f"{workload.workload_sha256}-{trial.worker_count}-{trial.trial_index}-{case.instance_id}"
    return CanaryRegressionExecutionIdentity(
        job_id=(
            "job://concurrency-experiment/"
            f"{workload.workload_sha256}/{trial.worker_count}/"
            f"{trial.trial_index}/{case.instance_id}"
        ),
        idempotency_key=f"concurrency-experiment-{prefix}",
    )


def _accepted_canary_policy(
    template: CanaryRegressionTemplateV1,
) -> CanaryRegressionPolicyV2:
    value = CanaryRegressionPolicyV2.create(
        cohort_manifest_ref=ObjectRef(
            object_type="development-canary-manifest",
            object_id="development-canary-manifest://v4",
            object_version="v4",
            object_sha256=("1a409bbb92ef0fc075d8dd2144f36cfd5829976ab6718c20db3a8e9f2d61a6a6"),
        ),
        max_case_refs=1_000,
        max_report_bytes=2_000_000,
        audit=template.audit,
    )
    if value.policy_sha256 != ACCEPTED_CANARY_POLICY_SHA256:
        raise ConcurrencyExperimentPolicyError("reconstructed R8-03 policy differs from accepted authority")
    return value


def _retained_storage_bytes(
    template: CanaryRegressionTemplateV1,
) -> int:
    matches = tuple(
        seed
        for seed in template.template_manifest.seed_objects
        if seed.codec is R6CanaryObjectCodecV2.CANARY_ATTACHMENT_FIXTURE
    )
    if len(matches) != 1:
        raise ConcurrencyExperimentStorageError("canary template requires one attachment fixture")
    try:
        fixture = R6CanaryTaskFixture.model_validate_json(matches[0].payload_json)
    except ValueError as exc:
        raise ConcurrencyExperimentStorageError("canary attachment fixture is invalid") from exc
    if fixture.text_provider_content is None:
        raise ConcurrencyExperimentStorageError("canary attachment fixture has no retained content")
    return len((fixture.text_provider_content.rstrip() + "\n").encode("utf-8"))


def _observe_current_host() -> HostCapacityObservation:
    logical = os.cpu_count() or 0
    try:
        physical = int(
            subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True,
                check=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
        memory = int(
            subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                check=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise ConcurrencyExperimentHostError("host capacity probe failed") from exc
    try:
        return HostCapacityObservation(
            logical_cpu_count=logical,
            physical_cpu_count=physical,
            memory_bytes=memory,
            python_version=platform.python_version(),
        )
    except ValueError as exc:
        raise ConcurrencyExperimentHostError("host capacity probe returned invalid values") from exc


__all__ = [
    "ACCEPTED_CANARY_POLICY_SHA256",
    "ConcurrencyExperimentBuilder",
    "ConcurrencyExperimentBuilderError",
    "ConcurrencyExperimentHostError",
    "ConcurrencyExperimentPolicyError",
    "ConcurrencyExperimentStorageError",
    "ConcurrencyRecoveryExecutionIdentity",
    "HostCapacityObservation",
    "derive_recovery_execution_identity",
]
