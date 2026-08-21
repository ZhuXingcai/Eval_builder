from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_canary_regression_builder import (
    CANARY_PATH,
    RAW_ROOT,
    _template,
)
from test_canary_regression_builder import (
    _policy as _canary_policy,
)

import eval_factory.readiness.concurrency_experiment_builder as builder_module
from eval_factory.contracts.canary_execution_v2 import (
    R6CanaryExecutionManifestV2,
)
from eval_factory.contracts.concurrency_experiment_v2 import (
    ConcurrencyExperimentFaultPointV2,
    ConcurrencyExperimentPolicyV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.readiness.canary_regression_builder import (
    CanaryRegressionExecutionIdentity,
    FrozenCanaryCohortLoader,
)
from eval_factory.readiness.canary_regression_models import (
    CanaryRegressionTemplateV1,
    FrozenCanaryCaseV1,
    PreparedCanaryRegressionCase,
)
from eval_factory.readiness.concurrency_experiment_builder import (
    ConcurrencyExperimentBuilder,
    ConcurrencyExperimentHostError,
    ConcurrencyExperimentPolicyError,
    ConcurrencyExperimentStorageError,
    HostCapacityObservation,
    derive_recovery_execution_identity,
)
from eval_factory.readiness.concurrency_experiment_models import (
    CONCURRENCY_TRIAL_SCHEDULE,
)

NOW = datetime(2026, 8, 3, tzinfo=UTC)


class _FakeCanaryBuilder:
    def prepare_case(
        self,
        case: FrozenCanaryCaseV1,
        *,
        template: CanaryRegressionTemplateV1,
        raw_root: Path,
        preparation_root: Path,
        policy: object,
        audit: ContractAudit,
        execution_identity: CanaryRegressionExecutionIdentity | None = None,
    ) -> PreparedCanaryRegressionCase:
        del preparation_root, policy, audit
        assert execution_identity is not None
        base = template.template_manifest
        job_spec = base.job_spec.model_validate(
            {
                **base.job_spec.model_dump(mode="python"),
                "job_id": execution_identity.job_id,
                "idempotency_key": execution_identity.idempotency_key,
            }
        )
        manifest = R6CanaryExecutionManifestV2.create(
            job_spec=job_spec,
            control=base.control,
            control_plane=base.control_plane,
            seed_objects=base.seed_objects,
            trace_bindings=base.trace_bindings,
            audit=base.audit,
        )
        return PreparedCanaryRegressionCase(
            cohort_case=case,
            expectation=case.expectation,
            raw_path=raw_root / f"{case.instance_id}.jsonl",
            child_manifest=manifest,
        )


def _ref(object_type: str, digest: str, *, version: str = "v2") -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://sha256/{digest}",
        object_version=version,
        object_sha256=digest,
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="r8-06-builder-test",
        governing_versions=(
            VersionBinding(
                component="concurrency-experiment",
                version="concurrency-experiment/r8-06-v1",
            ),
        ),
    )


def _policy() -> ConcurrencyExperimentPolicyV2:
    return ConcurrencyExperimentPolicyV2.create(
        canary_regression_report_ref=_ref(
            "canary-regression-report",
            "375b519dc3b40d3c8aab09d9a04574deca73ec80138c2fcaa2f1714eb1edbd4f",
        ),
        real_trace_stability_report_ref=_ref(
            "real-trace-stability-report",
            "730f151435d2472cfe456e9df350c2663e8702951bd1eb1e8f0dcb45b315374f",
        ),
        scheduler_load_report_ref=_ref(
            "scheduler-load-report",
            "52061b69b7b418874709bddc71450111680e7931f4e04a337103b34b9f379bac",
        ),
        cohort_manifest_ref=ObjectRef(
            object_type="development-canary-manifest",
            object_id="development-canary-manifest://v4",
            object_version="v4",
            object_sha256="1a409bbb92ef0fc075d8dd2144f36cfd5829976ab6718c20db3a8e9f2d61a6a6",
        ),
        max_capacity_readmissions=24,
        max_private_bytes=1_000_000_000,
        max_report_bytes=100_000_000,
        audit=_audit(),
    )


def _prepared_inputs(tmp_path: Path):
    cohort = FrozenCanaryCohortLoader().load(
        CANARY_PATH,
        policy=_canary_policy(),
    )
    template = _template(tmp_path)
    builder = ConcurrencyExperimentBuilder()
    host = builder.observe_host(
        audit=_audit(),
        observation=HostCapacityObservation(
            logical_cpu_count=10,
            physical_cpu_count=10,
            memory_bytes=17_179_869_184,
            python_version="3.12.13",
        ),
    )
    workload = builder.compile_workload(
        policy=_policy(),
        host_summary=host,
        cohort=cohort,
        template=template,
        audit=_audit(),
    )
    return builder, cohort, template, host, workload


def test_host_summary_is_content_free_and_deterministic(tmp_path: Path) -> None:
    builder, _cohort, _template_value, host, _workload = _prepared_inputs(tmp_path)
    replay = builder.observe_host(
        audit=_audit().model_copy(
            update={
                "created_at": datetime(2026, 8, 4, tzinfo=UTC),
                "created_by": "another-actor",
            }
        ),
        observation=HostCapacityObservation(
            logical_cpu_count=10,
            physical_cpu_count=10,
            memory_bytes=17_179_869_184,
            python_version="3.12.13",
        ),
    )

    assert replay.host_summary_sha256 == host.host_summary_sha256
    serialized = host.model_dump_json().casefold()
    for forbidden in ("hostname", "username", "path", "serial", "process_id"):
        assert forbidden not in serialized


def test_workload_uses_exact_interleaved_schedule_and_storage(
    tmp_path: Path,
) -> None:
    _builder, _cohort, _template_value, _host, workload = _prepared_inputs(tmp_path)

    assert (
        tuple((trial.worker_count, trial.trial_index) for trial in workload.trials)
        == CONCURRENCY_TRIAL_SCHEDULE
    )
    assert len(workload.trials) == 15
    assert workload.retained_storage_bytes_per_case > 0
    assert workload.retained_storage_bytes_per_cohort == (workload.retained_storage_bytes_per_case * 24)


def test_builder_prepares_trial_specific_final_child_identity(
    tmp_path: Path,
) -> None:
    builder, cohort, template, _host, workload = _prepared_inputs(tmp_path)
    trial = next(value for value in workload.trials if value.worker_count == 2)
    case = cohort.cases[0]

    prepared = builder.prepare_case(
        workload=workload,
        trial=trial,
        case_ordinal=0,
        case=case,
        template=template,
        raw_root=RAW_ROOT,
        preparation_root=tmp_path / "preparation",
        audit=_audit(),
    )

    spec = prepared.prepared_case.child_manifest.job_spec
    assert spec.job_id == (
        f"job://concurrency-experiment/{workload.workload_sha256}/2/{trial.trial_index}/{case.instance_id}"
    )
    assert spec.idempotency_key == (
        f"concurrency-experiment-{workload.workload_sha256}-2-{trial.trial_index}-{case.instance_id}"
    )
    assert prepared.case_ordinal == 0
    assert prepared.frozen_case == case


def test_recovery_execution_identities_are_unique_and_six_by_four(
    tmp_path: Path,
) -> None:
    _builder, cohort, _template_value, _host, workload = _prepared_inputs(tmp_path)

    derived = tuple(
        derive_recovery_execution_identity(
            workload=workload,
            worker_count=2,
            case_ordinal=ordinal,
            case=case,
        )
        for ordinal, case in enumerate(cohort.cases)
    )

    assert len({value.execution_identity.job_id for value in derived}) == 24
    assert all(workload.workload_sha256 in value.execution_identity.job_id for value in derived)
    assert {
        point: sum(value.fault_point is point for value in derived)
        for point in ConcurrencyExperimentFaultPointV2
    } == {point: 6 for point in ConcurrencyExperimentFaultPointV2}
    assert all("/recovery/2/" in value.execution_identity.job_id for value in derived)


def test_builder_prepares_complete_trial_and_recovery_partitions(
    tmp_path: Path,
) -> None:
    _builder, cohort, template, _host, workload = _prepared_inputs(tmp_path)
    builder = ConcurrencyExperimentBuilder(
        canary_builder=_FakeCanaryBuilder(),  # type: ignore[arg-type]
    )
    trial = workload.trials[0]

    prepared = builder.prepare_trial(
        workload=workload,
        trial=trial,
        cohort=cohort,
        template=template,
        raw_root=tmp_path / "raw",
        preparation_root=tmp_path / "preparation",
        audit=_audit(),
    )
    recovery = builder.prepare_recovery_cases(
        workload=workload,
        worker_count=1,
        cohort=cohort,
        template=template,
        raw_root=tmp_path / "raw",
        preparation_root=tmp_path / "preparation",
        audit=_audit(),
    )

    assert len(prepared) == 24
    assert len(recovery) == 24
    assert len({value.prepared_case.child_manifest.manifest_id for value in (*prepared, *recovery)}) == 48
    assert {
        point: sum(value.fault_point is point for value in recovery)
        for point in ConcurrencyExperimentFaultPointV2
    } == {point: 6 for point in ConcurrencyExperimentFaultPointV2}


def test_builder_rejects_unknown_recovery_level_and_ordinal(
    tmp_path: Path,
) -> None:
    _builder, cohort, template, _host, workload = _prepared_inputs(tmp_path)
    builder = ConcurrencyExperimentBuilder(
        canary_builder=_FakeCanaryBuilder(),  # type: ignore[arg-type]
    )

    with pytest.raises(ConcurrencyExperimentPolicyError):
        builder.prepare_recovery_cases(
            workload=workload,
            worker_count=3,
            cohort=cohort,
            template=template,
            raw_root=tmp_path / "raw",
            preparation_root=tmp_path / "preparation",
            audit=_audit(),
        )
    with pytest.raises(ConcurrencyExperimentPolicyError):
        derive_recovery_execution_identity(
            workload=workload,
            worker_count=1,
            case_ordinal=24,
            case=cohort.cases[0],
        )


def test_builder_observes_current_host_profile() -> None:
    host = ConcurrencyExperimentBuilder().observe_host(audit=_audit())

    assert host.platform_family == "DARWIN"
    assert host.architecture == "ARM64"
    assert host.logical_cpu_count >= host.physical_cpu_count >= 1


def test_builder_public_cohort_loader_and_host_input_validation(
    tmp_path: Path,
) -> None:
    template = _template(tmp_path)
    loaded = ConcurrencyExperimentBuilder.load_cohort(
        CANARY_PATH,
        template=template,
    )

    assert len(loaded.cases) == 24
    with pytest.raises(ValueError):
        HostCapacityObservation(
            logical_cpu_count=0,
            physical_cpu_count=1,
            memory_bytes=0,
            python_version="",
        )


def test_recovery_identity_rejects_unknown_worker_level(
    tmp_path: Path,
) -> None:
    _builder, cohort, _template_value, _host, workload = _prepared_inputs(tmp_path)

    with pytest.raises(ConcurrencyExperimentPolicyError):
        derive_recovery_execution_identity(
            workload=workload,
            worker_count=3,
            case_ordinal=0,
            case=cohort.cases[0],
        )


def test_builder_rejects_stale_policy_and_cohort_authority(
    tmp_path: Path,
) -> None:
    builder, cohort, template, host, _workload = _prepared_inputs(tmp_path)
    stale_policy = _policy().model_copy(update={"policy_sha256": "0" * 64})

    with pytest.raises(ConcurrencyExperimentPolicyError, match="stale"):
        builder.compile_workload(
            policy=stale_policy,
            host_summary=host,
            cohort=cohort,
            template=template,
            audit=_audit(),
        )

    changed_cohort = cohort.model_copy(
        update={
            "manifest_ref": _ref(
                "development-canary-manifest",
                "f" * 64,
                version="v4",
            )
        }
    )
    with pytest.raises(ConcurrencyExperimentPolicyError, match="stale"):
        builder.compile_workload(
            policy=_policy(),
            host_summary=host,
            cohort=changed_cohort,
            template=template,
            audit=_audit(),
        )


def test_builder_rejects_missing_fixture_and_insufficient_storage(
    tmp_path: Path,
) -> None:
    builder, cohort, template, host, _workload = _prepared_inputs(tmp_path)
    manifest = template.template_manifest
    without_fixture = manifest.model_copy(
        update={
            "seed_objects": tuple(
                seed for seed in manifest.seed_objects if seed.codec.value != "CANARY_ATTACHMENT_FIXTURE"
            )
        }
    )
    with pytest.raises(ConcurrencyExperimentStorageError, match="one attachment"):
        builder_module._retained_storage_bytes(
            template.model_copy(update={"template_manifest": without_fixture})
        )

    capacity = manifest.control_plane.resource_capacity.model_copy(update={"storage_bytes": 1})
    control_plane = manifest.control_plane.model_copy(update={"resource_capacity": capacity})
    constrained_manifest = R6CanaryExecutionManifestV2.create(
        job_spec=manifest.job_spec,
        control=manifest.control,
        control_plane=control_plane,
        seed_objects=manifest.seed_objects,
        trace_bindings=manifest.trace_bindings,
        audit=manifest.audit,
    )
    constrained_template = CanaryRegressionTemplateV1.create(
        template_manifest=constrained_manifest,
        audit=template.audit,
    )
    with pytest.raises(ConcurrencyExperimentStorageError, match="complete cohort"):
        builder.compile_workload(
            policy=_policy(),
            host_summary=host,
            cohort=cohort,
            template=constrained_template,
            audit=_audit(),
        )


def test_builder_rejects_foreign_trial_for_trial_and_case(
    tmp_path: Path,
) -> None:
    builder, cohort, template, _host, workload = _prepared_inputs(tmp_path)
    foreign = workload.trials[0].model_copy(update={"trial_id": "concurrency-experiment-trial://foreign"})

    with pytest.raises(ConcurrencyExperimentPolicyError, match="not part"):
        builder.prepare_trial(
            workload=workload,
            trial=foreign,
            cohort=cohort,
            template=template,
            raw_root=tmp_path / "raw",
            preparation_root=tmp_path / "preparation",
            audit=_audit(),
        )
    with pytest.raises(ConcurrencyExperimentPolicyError, match="not part"):
        builder.prepare_case(
            workload=workload,
            trial=foreign,
            case_ordinal=0,
            case=cohort.cases[0],
            template=template,
            raw_root=tmp_path / "raw",
            preparation_root=tmp_path / "preparation",
            audit=_audit(),
        )


def test_host_profile_rejects_unsupported_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = HostCapacityObservation(
        logical_cpu_count=10,
        physical_cpu_count=10,
        memory_bytes=17_179_869_184,
        python_version="3.12.13",
    )
    monkeypatch.setattr(builder_module.sys, "platform", "linux")

    with pytest.raises(ConcurrencyExperimentHostError, match="Darwin ARM64"):
        ConcurrencyExperimentBuilder().observe_host(
            audit=_audit(),
            observation=observation,
        )


def test_host_probe_failures_are_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(builder_module.os, "cpu_count", lambda: 10)

    def failed_run(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise subprocess.SubprocessError("private probe detail")

    monkeypatch.setattr(builder_module.subprocess, "run", failed_run)
    with pytest.raises(ConcurrencyExperimentHostError, match="probe failed"):
        builder_module._observe_current_host()

    outputs = iter(("0", "17179869184"))

    def invalid_run(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return subprocess.CompletedProcess(
            args=("sysctl",),
            returncode=0,
            stdout=next(outputs),
        )

    monkeypatch.setattr(builder_module.subprocess, "run", invalid_run)
    with pytest.raises(
        ConcurrencyExperimentHostError,
        match="invalid values",
    ):
        builder_module._observe_current_host()
