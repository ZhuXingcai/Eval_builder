from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from eval_factory.contracts.core import (
    ContractAudit,
    FailureClass,
    FailureRecord,
    ObjectRef,
    VersionBinding,
)
from eval_factory.contracts.orchestration_v2 import (
    R6_WORK_CONTROL_FAILURE_MESSAGE,
    R6_WORK_CONTROL_POLICY_VERSION,
    WorkCancellationRecordV2,
    WorkControlPolicyV2,
    WorkDispatchDecisionV2,
    WorkLeaseEventKindV2,
    WorkLeaseEventV2,
    WorkLeaseV2,
    WorkReadinessSnapshotV2,
    WorkReadinessV2,
    WorkRetryDecisionKindV2,
    WorkRetryDecisionV2,
    work_cancellation_record_v2_ref,
    work_control_policy_v2_ref,
    work_dispatch_decision_v2_ref,
    work_lease_event_v2_ref,
    work_lease_v2_ref,
    work_retry_decision_v2_ref,
)

HASH = "a" * 64
NOW = datetime(2026, 7, 31, tzinfo=UTC)


def _ref(object_type: str, suffix: str = "current", *, version: str = "v2") -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r6-05/{suffix}",
        object_version=version,
        object_sha256=HASH,
    )


def _audit(*refs: ObjectRef, created_at: datetime = NOW) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="r6-05-contract-test",
        governing_versions=(
            VersionBinding(
                component="work-control",
                version=R6_WORK_CONTROL_POLICY_VERSION,
            ),
        ),
        input_refs=tuple(sorted(refs, key=_ref_key)),
    )


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _policy(*, created_at: datetime = NOW) -> WorkControlPolicyV2:
    graph_ref = _ref("resolved-job-work-graph")
    return WorkControlPolicyV2.create(
        resolved_job_work_graph_ref=graph_ref,
        lease_duration_seconds=30,
        heartbeat_extension_seconds=20,
        max_attempts=3,
        retry_delay_seconds=(0, 5),
        retry_lease_expiry=True,
        audit=_audit(graph_ref, created_at=created_at),
    )


def _dispatch(policy: WorkControlPolicyV2) -> WorkDispatchDecisionV2:
    graph_ref = policy.resolved_job_work_graph_ref
    unit_ref = _ref("resolved-work-unit")
    readiness_ref = _ref("work-readiness-snapshot")
    policy_ref = work_control_policy_v2_ref(policy)
    return WorkDispatchDecisionV2.create(
        resolved_job_work_graph_ref=graph_ref,
        work_unit_ref=unit_ref,
        work_readiness_snapshot_ref=readiness_ref,
        work_control_policy_ref=policy_ref,
        retry_decision_ref=None,
        attempt=1,
        eligible_at=NOW,
        audit=_audit(graph_ref, unit_ref, readiness_ref, policy_ref),
    )


def _lease(policy: WorkControlPolicyV2) -> WorkLeaseV2:
    dispatch = _dispatch(policy)
    dispatch_ref = work_dispatch_decision_v2_ref(dispatch)
    unit_ref = dispatch.work_unit_ref
    policy_ref = work_control_policy_v2_ref(policy)
    holder_ref = _ref("worker-principal", version="v1")
    stage_run_ref = _ref("stage-run", version="identity/v1")
    return WorkLeaseV2.create(
        work_dispatch_decision_ref=dispatch_ref,
        work_unit_ref=unit_ref,
        work_control_policy_ref=policy_ref,
        holder_ref=holder_ref,
        fencing_token=1,
        attempt=1,
        stage_run_ref=stage_run_ref,
        acquired_at=NOW,
        expires_at=NOW + timedelta(seconds=30),
        audit=_audit(dispatch_ref, unit_ref, policy_ref, holder_ref, stage_run_ref),
    )


def _failure(*, retryable: bool) -> FailureRecord:
    return FailureRecord(
        failure_class=FailureClass.INTERNAL,
        code="r6-05-injected-failure",
        message=R6_WORK_CONTROL_FAILURE_MESSAGE,
        retryable=retryable,
    )


def test_control_policy_is_strict_frozen_and_audit_time_independent() -> None:
    first = _policy()
    second = _policy(created_at=NOW + timedelta(days=1))

    assert first.work_control_policy_id == second.work_control_policy_id
    assert first.work_control_policy_sha256 == second.work_control_policy_sha256
    assert work_control_policy_v2_ref(first).object_sha256 == first.work_control_policy_sha256
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        WorkControlPolicyV2.model_validate(
            {
                **first.model_dump(mode="python"),
                "worker_secret": "must-not-cross",
            }
        )
    with pytest.raises(ValidationError, match="frozen"):
        first.max_attempts = 4  # type: ignore[misc]


def test_control_policy_requires_an_explicit_complete_retry_schedule() -> None:
    policy = _policy()
    values = policy.model_dump(mode="python")
    values["retry_delay_seconds"] = (0,)

    with pytest.raises(ValidationError, match="max_attempts - 1"):
        WorkControlPolicyV2.model_validate(values)


def test_lease_and_event_contracts_enforce_terminal_field_matrices() -> None:
    policy = _policy()
    lease = _lease(policy)
    lease_ref = work_lease_v2_ref(lease)
    policy_ref = work_control_policy_v2_ref(policy)

    with pytest.raises(ValidationError, match="after acquired_at"):
        WorkLeaseV2.model_validate(
            {
                **lease.model_dump(mode="python"),
                "expires_at": NOW,
            }
        )

    heartbeat = WorkLeaseEventV2.create(
        work_lease_ref=lease_ref,
        work_unit_ref=lease.work_unit_ref,
        event_kind=WorkLeaseEventKindV2.HEARTBEAT,
        lease_version=1,
        fencing_token=1,
        effective_expires_at=NOW + timedelta(seconds=45),
        result_refs=(),
        failure=None,
        work_control_policy_ref=policy_ref,
        audit=_audit(lease_ref, lease.work_unit_ref, policy_ref),
    )
    assert work_lease_event_v2_ref(heartbeat).object_sha256 == heartbeat.work_lease_event_sha256

    with pytest.raises(ValidationError, match="heartbeat cannot carry result or failure"):
        WorkLeaseEventV2.create(
            work_lease_ref=lease_ref,
            work_unit_ref=lease.work_unit_ref,
            event_kind=WorkLeaseEventKindV2.HEARTBEAT,
            lease_version=1,
            fencing_token=1,
            effective_expires_at=NOW + timedelta(seconds=45),
            result_refs=(),
            failure=_failure(retryable=True),
            work_control_policy_ref=policy_ref,
            audit=_audit(lease_ref, lease.work_unit_ref, policy_ref),
        )

    result_ref = _ref("stage-result", version="record/v1")
    succeeded = WorkLeaseEventV2.create(
        work_lease_ref=lease_ref,
        work_unit_ref=lease.work_unit_ref,
        event_kind=WorkLeaseEventKindV2.SUCCEEDED,
        lease_version=1,
        fencing_token=1,
        effective_expires_at=None,
        result_refs=(result_ref,),
        failure=None,
        work_control_policy_ref=policy_ref,
        audit=_audit(lease_ref, lease.work_unit_ref, policy_ref, result_ref),
    )
    assert succeeded.result_refs == (result_ref,)


def test_retry_decision_requires_exact_scheduled_or_exhausted_shape() -> None:
    policy = _policy()
    lease = _lease(policy)
    lease_ref = work_lease_v2_ref(lease)
    policy_ref = work_control_policy_v2_ref(policy)
    result_ref = _ref("stage-result", version="record/v1")
    event = WorkLeaseEventV2.create(
        work_lease_ref=lease_ref,
        work_unit_ref=lease.work_unit_ref,
        event_kind=WorkLeaseEventKindV2.RETRYABLE_FAILURE,
        lease_version=1,
        fencing_token=1,
        effective_expires_at=None,
        result_refs=(result_ref,),
        failure=_failure(retryable=True),
        work_control_policy_ref=policy_ref,
        audit=_audit(lease_ref, lease.work_unit_ref, policy_ref, result_ref),
    )
    event_ref = work_lease_event_v2_ref(event)
    unsafe_failure = _failure(retryable=True).model_copy(
        update={"message": "runtime transcript must not cross"}
    )
    with pytest.raises(ValidationError, match="content-free"):
        WorkLeaseEventV2.create(
            work_lease_ref=lease_ref,
            work_unit_ref=lease.work_unit_ref,
            event_kind=WorkLeaseEventKindV2.RETRYABLE_FAILURE,
            lease_version=1,
            fencing_token=1,
            effective_expires_at=None,
            result_refs=(result_ref,),
            failure=unsafe_failure,
            work_control_policy_ref=policy_ref,
            audit=_audit(
                lease_ref,
                lease.work_unit_ref,
                policy_ref,
                result_ref,
            ),
        )
    decision = WorkRetryDecisionV2.create(
        work_unit_ref=lease.work_unit_ref,
        prior_lease_event_ref=event_ref,
        prior_result_refs=(result_ref,),
        work_control_policy_ref=policy_ref,
        decision=WorkRetryDecisionKindV2.RETRY_SCHEDULED,
        completed_attempt=1,
        next_attempt=2,
        eligible_at=NOW + timedelta(seconds=5),
        audit=_audit(lease.work_unit_ref, event_ref, result_ref, policy_ref),
    )
    assert work_retry_decision_v2_ref(decision).object_sha256 == (decision.work_retry_decision_sha256)

    with pytest.raises(ValidationError, match="EXHAUSTED"):
        WorkRetryDecisionV2.create(
            work_unit_ref=lease.work_unit_ref,
            prior_lease_event_ref=event_ref,
            prior_result_refs=(result_ref,),
            work_control_policy_ref=policy_ref,
            decision=WorkRetryDecisionKindV2.EXHAUSTED,
            completed_attempt=1,
            next_attempt=2,
            eligible_at=NOW + timedelta(seconds=5),
            audit=_audit(lease.work_unit_ref, event_ref, result_ref, policy_ref),
        )


def test_readiness_accepts_only_typed_control_evidence() -> None:
    graph_ref = _ref("resolved-job-work-graph")
    unit_ref = _ref("resolved-work-unit")
    retry_ref = _ref("work-retry-decision")
    event_ref = _ref("work-lease-event")
    snapshot = WorkReadinessSnapshotV2(
        work_readiness_snapshot_id="work-readiness-snapshot://pending",
        resolved_job_work_graph_ref=graph_ref,
        work_unit_ref=unit_ref,
        readiness=WorkReadinessV2.BLOCKED_DEPENDENCY,
        terminal_non_success_result_refs=(event_ref, retry_ref),
        work_readiness_snapshot_sha256="0" * 64,
        audit=_audit(graph_ref, unit_ref, retry_ref, event_ref),
    )
    assert snapshot.terminal_non_success_result_refs == tuple(sorted((event_ref, retry_ref), key=_ref_key))

    invalid = _ref("untyped-control-result")
    with pytest.raises(ValidationError, match="invalid ref"):
        WorkReadinessSnapshotV2.model_validate(
            {
                **snapshot.model_dump(mode="python"),
                "terminal_non_success_result_refs": (invalid,),
                "audit": _audit(graph_ref, unit_ref, invalid),
            }
        )


def test_cancellation_record_binds_exact_cancelled_and_preserved_evidence() -> None:
    policy = _policy()
    graph_ref = policy.resolved_job_work_graph_ref
    policy_ref = work_control_policy_v2_ref(policy)
    event_ref = _ref("work-lease-event", "cancelled")
    result_ref = _ref("stage-result", "preserved", version="record/v1")
    record = WorkCancellationRecordV2.create(
        resolved_job_work_graph_ref=graph_ref,
        work_control_policy_ref=policy_ref,
        reason_code="operator-cancelled",
        cancelled_lease_event_refs=(event_ref,),
        preserved_result_refs=(result_ref,),
        audit=_audit(graph_ref, policy_ref, event_ref, result_ref),
    )

    assert work_cancellation_record_v2_ref(record).object_sha256 == (record.work_cancellation_record_sha256)
