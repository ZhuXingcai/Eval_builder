from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.contracts.approval import ApprovalMode
from eval_factory.contracts.core import ContractAudit, VersionBinding
from eval_factory.contracts.orchestration_v2 import StageNameV2
from eval_factory.contracts.scheduler_load_v2 import (
    SchedulerLoadFaultPointV2,
    SchedulerLoadPolicyV2,
)
from eval_factory.orchestration import (
    DatasetJobPlanCompiler,
    DatasetJobWorkGraphCompiler,
)
from eval_factory.readiness.scheduler_load_builder import (
    SchedulerLoadBuilder,
    SchedulerLoadPolicyError,
)
from eval_factory.readiness.scheduler_load_models import (
    SchedulerLoadMemberV1,
    SchedulerLoadWorkloadV1,
)

NOW = datetime(2026, 8, 2, tzinfo=UTC)


def _audit(
    created_at: datetime = NOW,
) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="scheduler-load-builder-test",
        governing_versions=(
            VersionBinding(
                component="scheduler-load",
                version="scheduler-load/r8-05-v1",
            ),
        ),
    )


def _policy() -> SchedulerLoadPolicyV2:
    return SchedulerLoadPolicyV2.create(
        lease_duration_seconds=300,
        heartbeat_extension_seconds=60,
        max_private_bytes=100_000_000,
        max_report_bytes=100_000_000,
        audit=_audit(),
    )


def test_builder_compiles_exact_unique_workload_and_balanced_faults() -> None:
    workload = SchedulerLoadBuilder().compile_workload(
        policy=_policy(),
        audit=_audit(),
    )

    assert isinstance(workload, SchedulerLoadWorkloadV1)
    assert len(workload.members) == 1000
    assert tuple(member.ordinal for member in workload.members) == tuple(range(1000))
    assert len({member.case_key for member in workload.members}) == 1000
    assert len({member.descriptor_sha256 for member in workload.members}) == 1000
    assert Counter(member.assigned_fault_point for member in workload.members) == Counter(
        {fault: 125 for fault in SchedulerLoadFaultPointV2}
    )
    assert workload.policy_ref == _policy().to_ref()


def test_private_member_identity_binds_ordinal_descriptor_and_fault() -> None:
    member = SchedulerLoadMemberV1.create(
        ordinal=7,
        descriptor_sha256="a" * 64,
        assigned_fault_point=SchedulerLoadFaultPointV2.AFTER_OBSERVABILITY_REFRESH,
    )

    assert member.case_key.startswith("scheduler-load-case://sha256/")
    assert member.to_ref().object_version == "private-v1"
    with pytest.raises(ValueError, match="case key"):
        member.model_copy(update={"ordinal": 8}).model_validate(
            member.model_copy(update={"ordinal": 8}).model_dump(mode="python")
        )


def test_prepare_case_uses_minimal_current_r6_authorities() -> None:
    builder = SchedulerLoadBuilder()
    policy = _policy()
    workload = builder.compile_workload(policy=policy, audit=_audit())
    prepared = builder.prepare_case(
        workload.members[0],
        workload=workload,
        policy=policy,
        audit=_audit(),
    )

    spec = prepared.job_spec
    assert spec.requested_stages == (StageNameV2.TRACE_INDEX,)
    assert len(spec.traces) == 1
    assert spec.traces[0].source_uri.startswith("scheduler-load://")
    assert spec.traces[0].processing_class == "RESTRICTED_TRACE_RAW"
    assert spec.model_profiles == ()
    assert spec.budget.max_model_requests == 0
    assert spec.budget.max_model_tokens == 0
    assert spec.budget.max_processes == 0
    assert spec.budget.max_renderers == 0
    assert spec.budget.max_network_requests == 0
    assert spec.budget.max_storage_bytes == 0
    assert spec.concurrency.model_requests == 1
    assert spec.concurrency.processes == 1
    assert spec.concurrency.items == 1
    assert spec.approval_mode is ApprovalMode.NONE
    assert spec.enabled_checkpoints == frozenset()
    assert spec.export_target.channel == "CANARY"

    DatasetJobPlanCompiler().validate_current(
        prepared.resolved_plan,
        job_spec=spec,
    )
    DatasetJobWorkGraphCompiler().validate_current(
        prepared.work_graph,
        job_spec=spec,
        resolved_plan=prepared.resolved_plan,
    )
    assert len(prepared.work_graph.item_ids) == 1
    assert prepared.work_graph.work_units == (prepared.root_work_unit,)
    assert prepared.root_work_unit.stage is StageNameV2.TRACE_INDEX
    assert prepared.control_policy.max_attempts == 1
    assert prepared.control_policy.retry_delay_seconds == ()
    assert prepared.control_policy.retry_lease_expiry is False
    assert prepared.holder_ref.object_type == "worker-principal"
    assert prepared.output_ref.object_type == "scheduler-load-output"


def test_prepare_cases_cover_exact_workload_without_identity_aliases() -> None:
    builder = SchedulerLoadBuilder()
    policy = _policy()
    workload = builder.compile_workload(policy=policy, audit=_audit())
    prepared = builder.prepare_cases(
        workload,
        policy=policy,
        audit=_audit(),
    )

    assert len(prepared) == 1000
    assert len({case.job_spec.job_id for case in prepared}) == 1000
    assert len({case.job_spec.idempotency_key for case in prepared}) == 1000
    assert len({case.dataset_job_spec_ref for case in prepared}) == 1000
    assert len({case.work_graph.resolved_job_work_graph_id for case in prepared}) == 1000
    assert tuple(case.member for case in prepared) == workload.members


def test_workload_identity_ignores_audit_time() -> None:
    policy = _policy()
    first = SchedulerLoadBuilder().compile_workload(
        policy=policy,
        audit=_audit(datetime(2026, 8, 2, tzinfo=UTC)),
    )
    second = SchedulerLoadBuilder().compile_workload(
        policy=policy,
        audit=_audit(datetime(2026, 8, 3, tzinfo=UTC)),
    )

    assert first.workload_id == second.workload_id
    assert first.workload_sha256 == second.workload_sha256
    assert first.members == second.members
    assert first.canonical_sha256() != second.canonical_sha256()


def test_changed_policy_cannot_prepare_existing_workload() -> None:
    builder = SchedulerLoadBuilder()
    policy = _policy()
    workload = builder.compile_workload(policy=policy, audit=_audit())
    changed = policy.model_copy(
        update={
            "policy_id": "scheduler-load-policy://sha256/" + "b" * 64,
            "policy_sha256": "b" * 64,
        }
    )

    with pytest.raises(SchedulerLoadPolicyError, match="policy is stale"):
        builder.prepare_case(
            workload.members[0],
            workload=workload,
            policy=changed,
            audit=_audit(),
        )


def test_workload_is_stable_across_python_hash_seeds(tmp_path: Path) -> None:
    script = """
import json
from datetime import UTC, datetime
from eval_factory.contracts.core import ContractAudit, VersionBinding
from eval_factory.contracts.scheduler_load_v2 import SchedulerLoadPolicyV2
from eval_factory.readiness.scheduler_load_builder import SchedulerLoadBuilder

audit = ContractAudit(
    created_at=datetime(2026, 8, 2, tzinfo=UTC),
    created_by="seed-check",
    governing_versions=(
        VersionBinding(
            component="scheduler-load",
            version="scheduler-load/r8-05-v1",
        ),
    ),
)
policy = SchedulerLoadPolicyV2.create(
    lease_duration_seconds=300,
    heartbeat_extension_seconds=60,
    max_private_bytes=100_000_000,
    max_report_bytes=100_000_000,
    audit=audit,
)
value = SchedulerLoadBuilder().compile_workload(policy=policy, audit=audit)
print(json.dumps({
    "workload_id": value.workload_id,
    "workload_sha256": value.workload_sha256,
    "first": value.members[0].case_key,
    "last": value.members[-1].case_key,
}, sort_keys=True))
"""
    outputs = []
    for seed in ("1", "321"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )
        outputs.append(json.loads(result.stdout))

    assert outputs[0] == outputs[1]
