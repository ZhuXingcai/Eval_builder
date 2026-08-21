from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_user_approval_requests import (
    _compile as _compile_request,
)
from test_user_approval_requests import (
    _generation_policy,
    _job_spec,
    _label_source,
    _policy,
)
from test_user_decisions import (
    _audit,
    _authentication,
    _handling_policy,
    _label_request,
)

from eval_factory.approval.persistence import (
    UserDecisionConflictError,
    UserDecisionPersistenceService,
)
from eval_factory.contracts.approval import (
    ApprovalCheckpoint,
    ApprovalMode,
    UserDecision,
)
from eval_factory.contracts.approval_v2 import (
    user_approval_request_carried_sha256,
    user_approval_request_ref,
)
from eval_factory.orchestration.job_store import (
    IdempotencyConflictError,
    ImmutableResultError,
    JobStore,
    RecordNotFoundError,
)

NOW = datetime(2026, 8, 1, tzinfo=UTC)
JOB_ID = "job://r7-04/requests"


def _store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "factory.sqlite3", clock=lambda: NOW)


def _commit(
    service: UserDecisionPersistenceService,
    *,
    decision: UserDecision = UserDecision.ACCEPT,
    idempotency_key: str = "commit-label-decision",
    reason: str = "Approved current label plan.",
):
    policy, source, request_result, request = _label_request()
    return service.commit(
        job_spec=_job_spec(policy),
        approval_policy=policy,
        generation_policy=_generation_policy(),
        request_compilation=request_result,
        checkpoint_sources=(source,),
        request_ref=user_approval_request_ref(request),
        handling_policy=_handling_policy(),
        authentication=_authentication(),
        decision=decision,
        adjustments=(),
        adjustment_effect=None,
        environment_decisions=(),
        query_packaging=None,
        reason=reason,
        idempotency_key=idempotency_key,
        audit=_audit(),
    )


def _prepared_service(
    tmp_path: Path,
) -> tuple[JobStore, UserDecisionPersistenceService]:
    store = _store(tmp_path)
    policy, _, _, _ = _label_request()
    store.create_job(_job_spec(policy))
    return store, UserDecisionPersistenceService(store)


def test_commit_is_atomic_replay_safe_and_queryable(tmp_path: Path) -> None:
    store, service = _prepared_service(tmp_path)

    first = _commit(service)
    replay = _commit(service)

    assert replay == first
    assert service.get_commit(first.commit_id) == first
    assert service.get_decision_for_request(first.request_ref.object_id) == first.decision_record
    assert service.list_job_commits(first.job_id) == (first,)
    events = tuple(event for event in store.list_outbox() if event.event_type == "user-decision-committed")
    assert len(events) == 1
    assert {attribute.key: attribute.value for attribute in events[0].attributes} == {
        "checkpoint": first.decision_record.checkpoint.value,
        "commit-id": first.commit_id,
        "decision-record-id": first.decision_record.decision_record_id,
        "outcome": first.outcome.value,
    }


def test_same_key_changed_submission_and_second_key_conflict(
    tmp_path: Path,
) -> None:
    _, service = _prepared_service(tmp_path)
    _commit(service)

    with pytest.raises(IdempotencyConflictError, match="different request"):
        _commit(
            service,
            decision=UserDecision.REJECT,
            idempotency_key="commit-label-decision",
            reason="Reject current label plan.",
        )
    with pytest.raises(UserDecisionConflictError, match="already has"):
        _commit(
            service,
            idempotency_key="different-key",
        )


def test_concurrent_conflicting_decisions_have_one_winner(
    tmp_path: Path,
) -> None:
    store, _ = _prepared_service(tmp_path)

    def attempt(
        decision: UserDecision,
        key: str,
    ) -> str:
        service = UserDecisionPersistenceService(JobStore(store.path, clock=lambda: NOW))
        try:
            return _commit(
                service,
                decision=decision,
                idempotency_key=key,
                reason=f"{decision.value} current label plan.",
            ).outcome.value
        except UserDecisionConflictError:
            return "CONFLICT"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(
            executor.map(
                lambda value: attempt(*value),
                (
                    (UserDecision.ACCEPT, "accept-key"),
                    (UserDecision.REJECT, "reject-key"),
                ),
            )
        )

    assert outcomes.count("CONFLICT") == 1
    assert len(UserDecisionPersistenceService(store).list_job_commits(JOB_ID)) == 1


def test_outbox_failure_rolls_back_every_decision_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, service = _prepared_service(tmp_path)

    def fail_outbox(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected user decision outbox failure")

    monkeypatch.setattr(store, "_append_outbox", fail_outbox)
    with pytest.raises(RuntimeError, match="outbox failure"):
        _commit(service)

    with pytest.raises(RecordNotFoundError):
        service.get_decision_for_request(_label_request()[3].request_id)
    assert service.list_job_commits(JOB_ID) == ()


def test_reopen_existing_store_preserves_decision_and_additive_schema(
    tmp_path: Path,
) -> None:
    store, service = _prepared_service(tmp_path)
    committed = _commit(service)

    reopened = UserDecisionPersistenceService(JobStore(store.path))

    assert reopened.get_commit(committed.commit_id) == committed
    with sqlite3.connect(store.path) as connection:
        names = {
            str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert {
        "user_approval_request_compilations",
        "user_approval_requests",
        "user_approval_request_revisions",
        "user_decision_records",
        "user_decision_commits",
    }.issubset(names)


def test_materialized_column_drift_fails_closed(tmp_path: Path) -> None:
    store, service = _prepared_service(tmp_path)
    committed = _commit(service)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE user_decision_commits
            SET outcome = 'REJECTED'
            WHERE commit_id = ?
            """,
            (committed.commit_id,),
        )

    with pytest.raises(ImmutableResultError, match="columns"):
        service.get_commit(committed.commit_id)


def test_decision_job_column_drift_fails_closed(tmp_path: Path) -> None:
    store, service = _prepared_service(tmp_path)
    committed = _commit(service)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE user_decision_records
            SET job_id = 'job://other'
            WHERE decision_record_id = ?
            """,
            (committed.decision_record.decision_record_id,),
        )

    with pytest.raises(ImmutableResultError, match="decision row"):
        service.get_commit(committed.commit_id)


def test_multiple_requests_reuse_one_exact_compilation(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    policy = _policy(ApprovalMode.PLAN_GATES)
    sources = (_label_source("a"), _label_source("b"))
    compilation = _compile_request(
        policy=policy,
        checkpoint=ApprovalCheckpoint.LABEL_PLAN,
        sources=sources,
    )
    store.create_job(_job_spec(policy))
    service = UserDecisionPersistenceService(store)

    commits = tuple(
        service.commit(
            job_spec=_job_spec(policy),
            approval_policy=policy,
            generation_policy=_generation_policy(),
            request_compilation=compilation,
            checkpoint_sources=sources,
            request_ref=user_approval_request_ref(request),
            handling_policy=_handling_policy(),
            authentication=_authentication(),
            decision=UserDecision.ACCEPT,
            adjustments=(),
            adjustment_effect=None,
            environment_decisions=(),
            query_packaging=None,
            reason="Accept current label plan.",
            idempotency_key=f"accept-label-{index}",
            audit=_audit(),
        )
        for index, request in enumerate(compilation.requests)
    )

    assert len(commits) == 2
    assert service.list_job_commits(JOB_ID) == tuple(sorted(commits, key=lambda value: value.commit_id))


@pytest.mark.parametrize(
    ("table", "column", "expected"),
    (
        (
            "user_approval_requests",
            "request_id",
            "missing its approval request",
        ),
        (
            "user_approval_request_compilations",
            "result_id",
            "missing its request compilation",
        ),
    ),
)
def test_missing_nested_request_authority_fails_closed(
    tmp_path: Path,
    table: str,
    column: str,
    expected: str,
) -> None:
    store, service = _prepared_service(tmp_path)
    committed = _commit(service)
    identity = (
        committed.request_ref.object_id
        if column == "request_id"
        else committed.request_compilation_result_ref.object_id
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            f"DELETE FROM {table} WHERE {column} = ?",
            (identity,),
        )

    with pytest.raises(ImmutableResultError, match=expected):
        service.get_commit(committed.commit_id)


def test_malformed_nested_json_fails_as_immutable_corruption(
    tmp_path: Path,
) -> None:
    store, service = _prepared_service(tmp_path)
    committed = _commit(service)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE user_decision_commits
            SET record_json = '{}'
            WHERE commit_id = ?
            """,
            (committed.commit_id,),
        )

    with pytest.raises(ImmutableResultError, match="malformed"):
        service.get_commit(committed.commit_id)


def test_missing_commit_and_revision_are_not_found(tmp_path: Path) -> None:
    _, service = _prepared_service(tmp_path)

    with pytest.raises(RecordNotFoundError, match="UserDecisionCommitResultV2"):
        service.get_commit("user-decision-commit://missing")
    with pytest.raises(RecordNotFoundError, match="UserApprovalRequestRevisionV2"):
        service.get_request_revision("user-approval-request-revision://missing")


def test_commit_missing_decision_row_fails_closed(tmp_path: Path) -> None:
    store, service = _prepared_service(tmp_path)
    committed = _commit(service)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            DELETE FROM user_decision_records
            WHERE decision_record_id = ?
            """,
            (committed.decision_record.decision_record_id,),
        )

    with pytest.raises(ImmutableResultError, match="missing its decision record"):
        service.get_commit(committed.commit_id)


@pytest.mark.parametrize(
    ("column", "value", "expected"),
    (
        ("record_json", "{}", "malformed"),
        ("authenticated_user", "other-user", "columns"),
    ),
)
def test_decision_row_corruption_fails_closed(
    tmp_path: Path,
    column: str,
    value: str,
    expected: str,
) -> None:
    store, service = _prepared_service(tmp_path)
    committed = _commit(service)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            f"UPDATE user_decision_records SET {column} = ? WHERE decision_record_id = ?",
            (value, committed.decision_record.decision_record_id),
        )

    with pytest.raises(ImmutableResultError, match=expected):
        service.get_decision_for_request(committed.request_ref.object_id)


def test_request_revision_materialization_is_immutable_and_replay_safe(
    tmp_path: Path,
) -> None:
    store, service = _prepared_service(tmp_path)
    pending_commit = _commit(
        service,
        decision=UserDecision.REQUEST_MORE_EXAMPLES,
        idempotency_key="request-more-label",
        reason="More examples are required.",
    )
    pending = pending_commit.request_revision
    assert pending is not None
    source_request = _label_request()[3]
    successor = source_request.model_copy(
        update={
            "request_id": "user-approval-request://pending",
            "idempotency_key": "successor-label-request",
        }
    )
    digest = user_approval_request_carried_sha256(successor)
    successor = successor.model_copy(update={"request_id": f"user-approval-request://sha256/{digest}"})

    materialized = service.materialize_request_revision(
        pending_revision=pending,
        successor_request=successor,
        handling_policy=_handling_policy(),
        idempotency_key="materialize-label-request",
        audit=_audit(),
    )
    replay = service.materialize_request_revision(
        pending_revision=pending,
        successor_request=successor,
        handling_policy=_handling_policy(),
        idempotency_key="materialize-label-request",
        audit=_audit(),
    )

    assert replay == materialized
    assert service.get_request_revision(materialized.revision_id) == (materialized)
    assert [event.event_type for event in store.list_outbox()].count(
        "user-approval-request-revision-materialized"
    ) == 1


@pytest.mark.parametrize(
    ("column", "value", "expected"),
    (
        ("record_json", "{}", "malformed"),
        ("state", "MATERIALIZED", "columns"),
    ),
)
def test_request_revision_corruption_fails_closed(
    tmp_path: Path,
    column: str,
    value: str,
    expected: str,
) -> None:
    store, service = _prepared_service(tmp_path)
    committed = _commit(
        service,
        decision=UserDecision.REQUEST_MORE_EXAMPLES,
        idempotency_key="request-more-label",
        reason="More examples are required.",
    )
    revision = committed.request_revision
    assert revision is not None
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            f"UPDATE user_approval_request_revisions SET {column} = ? WHERE revision_id = ?",
            (value, revision.revision_id),
        )

    with pytest.raises(ImmutableResultError, match=expected):
        service.get_request_revision(revision.revision_id)
