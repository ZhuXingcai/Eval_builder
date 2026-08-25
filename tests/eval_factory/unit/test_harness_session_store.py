from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.harness import (
    GatewayJournalStateV1,
    HarnessGatewayJournalV1,
    HarnessMessageRoleV1,
    HarnessMessageV1,
    HarnessSessionConcurrencyError,
    HarnessSessionConflictError,
    HarnessSessionIntegrityError,
    HarnessSessionNotFoundError,
    HarnessSessionStatusV1,
    HarnessSessionStore,
    HarnessTurnOutcomeV1,
    ProviderEvidenceClassV1,
    RequirementInterpretationOutcomeV1,
    RequirementInterpretationV1,
    SessionLifecycleEventKindV1,
)

HASH = "a" * 64
NOW = datetime(2026, 8, 17, tzinfo=UTC)


def _ref(object_type: str, *, version: str = "v1") -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://example/{version}",
        object_version=version,
        object_sha256=HASH,
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="harness-session-store-test",
        governing_versions=(VersionBinding(component="evaluation-agent-harness", version="v1", sha256=HASH),),
    )


def _store(path: Path) -> HarnessSessionStore:
    return HarnessSessionStore(path, clock=lambda: NOW)


def test_session_store_reopens_replays_and_rebuilds_projection(tmp_path: Path) -> None:
    path = tmp_path / "harness.sqlite3"
    store = _store(path)
    projection = store.create_session(
        session_id="session-store-001",
        incarnation_id="session-store-incarnation-001",
        composition_ref=_ref("harness-composition"),
        created_by="session-user",
        idempotency_key="create-session-store-001",
        audit=_audit(),
    )
    assert projection.session.session_version == 1
    assert len(projection.event_refs) == 1
    assert store.get_projection_by_ref(projection.session.session_ref) == projection

    replay = store.create_session(
        session_id="session-store-001",
        incarnation_id="session-store-incarnation-001",
        composition_ref=_ref("harness-composition"),
        created_by="session-user",
        idempotency_key="create-session-store-001",
        audit=_audit(),
    )
    assert replay == projection

    start = store.start_turn(
        session_id="session-store-001",
        expected_session_version=1,
        principal_ref=_ref("principal"),
        content="需要构建通用 Agent 评测数据。",
        artifact_envelope_refs=(),
        idempotency_key="turn-store-001",
        audit=_audit(),
    )
    assert start.state.session_version == 2
    assert start.replay is None

    prepared = HarnessGatewayJournalV1.create(
        journal_id="journal-store-prepared",
        session_ref=start.state.session_ref,
        command_ref=start.command.to_ref(),
        state=GatewayJournalStateV1.PREPARED,
        evidence_class=ProviderEvidenceClassV1.MECHANISM_FIXTURE,
        route_ref=_ref("model-route-decision", version="v2"),
        invocation_request_ref=_ref("gateway-invocation-request", version="v2"),
        prompt_template_ref=_ref("prompt-template", version="v2"),
        model_profile_ref=_ref("model-capability-profile", version="v2"),
        predecessor_journal_ref=None,
        receipt_ref=None,
        invocation_result_ref=None,
        output_ref=None,
        gateway_status=None,
        usage=None,
        failure_code=None,
        audit=_audit(),
    )
    store.record_prepared_journal(
        session_id=start.state.session_id,
        command=start.command,
        journal=prepared,
    )
    unknown = HarnessGatewayJournalV1.create(
        journal_id="journal-store-unknown",
        session_ref=start.state.session_ref,
        command_ref=start.command.to_ref(),
        state=GatewayJournalStateV1.UNKNOWN_OUTCOME,
        evidence_class=ProviderEvidenceClassV1.MECHANISM_FIXTURE,
        route_ref=prepared.route_ref,
        invocation_request_ref=prepared.invocation_request_ref,
        prompt_template_ref=prepared.prompt_template_ref,
        model_profile_ref=prepared.model_profile_ref,
        predecessor_journal_ref=prepared.to_ref(),
        receipt_ref=None,
        invocation_result_ref=None,
        output_ref=None,
        gateway_status=None,
        usage=None,
        failure_code="GATEWAY_OUTCOME_UNKNOWN",
        audit=_audit(),
    )
    final = HarnessMessageV1.create(
        message_id="harness-message-store-final",
        session_ref=start.state.session_ref,
        role=HarnessMessageRoleV1.ASSISTANT,
        content="调用结果无法被证明, 已停止自动重试。",
        artifact_envelope_refs=(),
        chunk_index=None,
        final=True,
        created_at=NOW,
        audit=_audit(),
    )
    interpretation = RequirementInterpretationV1.create(
        interpretation_id="requirement-interpretation-store-blocked",
        session_ref=start.state.session_ref,
        command_ref=start.command.to_ref(),
        user_message_ref=start.user_message.to_ref(),
        assistant_message_ref=final.to_ref(),
        proposal_ref=None,
        gateway_result_ref=None,
        outcome=RequirementInterpretationOutcomeV1.BLOCKED_CAPABILITY,
        evidence_class=ProviderEvidenceClassV1.MECHANISM_FIXTURE,
        goals=(),
        constraints=(),
        assumptions=(),
        source_expectations=(),
        target_capabilities=(),
        quality_intent=None,
        delivery_intent=None,
        missing_field_codes=(),
        clarification_questions=(),
        reason_codes=("GATEWAY_OUTCOME_UNKNOWN",),
        audit=_audit(),
    )
    turn = store.complete_turn(
        session_id=start.state.session_id,
        command=start.command,
        prepared_journal=prepared,
        committed_journal=unknown,
        assistant_messages=(final,),
        interpretation=interpretation,
        requirement_policy=None,
        outcome=HarnessTurnOutcomeV1.UNKNOWN_OUTCOME,
        audit=_audit(),
    )
    assert turn.outcome is HarnessTurnOutcomeV1.UNKNOWN_OUTCOME

    reopened = _store(path)
    current = reopened.get_projection("session-store-001")
    assert current.session.status is HarnessSessionStatusV1.BLOCKED
    assert tuple(item.role for item in current.transcript) == (
        HarnessMessageRoleV1.USER,
        HarnessMessageRoleV1.ASSISTANT,
    )
    assert current.latest_gateway is not None
    assert current.latest_gateway.state is GatewayJournalStateV1.UNKNOWN_OUTCOME

    exact_replay = reopened.start_turn(
        session_id="session-store-001",
        expected_session_version=1,
        principal_ref=_ref("principal"),
        content="需要构建通用 Agent 评测数据。",
        artifact_envelope_refs=(),
        idempotency_key="turn-store-001",
        audit=_audit(),
    )
    assert exact_replay.replay == turn

    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            UPDATE harness_session_projections
            SET source_fingerprint = ?
            WHERE session_id = ?
            """,
            ("b" * 64, "session-store-001"),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(HarnessSessionIntegrityError):
        reopened.get_projection("session-store-001")
    rebuilt = reopened.rebuild_projection("session-store-001")
    assert rebuilt == reopened.get_projection("session-store-001")


def test_session_store_closes_without_deleting_audit_history(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "harness.sqlite3")
    created = store.create_session(
        session_id="session-store-close",
        incarnation_id="session-store-close-incarnation",
        composition_ref=_ref("harness-composition"),
        created_by="session-user",
        idempotency_key="create-session-store-close",
        audit=_audit(),
    )
    with pytest.raises(HarnessSessionConcurrencyError):
        store.close_session(
            session_id=created.session.session_id,
            expected_session_version=created.session.session_version + 1,
            principal_ref=_ref("principal"),
            idempotency_key="close-session-store-stale",
            audit=_audit(),
        )
    closed = store.close_session(
        session_id=created.session.session_id,
        expected_session_version=created.session.session_version,
        principal_ref=_ref("principal"),
        idempotency_key="close-session-store",
        audit=_audit(),
    )

    assert closed.session.status is HarnessSessionStatusV1.CLOSED
    assert closed.session.session_version == 2
    events = store.list_events(
        created.session.session_id,
        after_sequence=0,
        limit=100,
    ).events
    assert events[-1].payload.event_kind is (
        SessionLifecycleEventKindV1.SESSION_CLOSED
    )
    assert store.list_sessions(offset=0, limit=100).total == 0
    assert store.list_sessions(
        offset=0,
        limit=100,
        include_closed=True,
    ).sessions == (closed.session,)

    replay = store.close_session(
        session_id=created.session.session_id,
        expected_session_version=created.session.session_version,
        principal_ref=_ref("principal"),
        idempotency_key="close-session-store",
        audit=_audit(),
    )
    assert replay == closed
    with pytest.raises(HarnessSessionConflictError):
        store.close_session(
            session_id=created.session.session_id,
            expected_session_version=created.session.session_version,
            principal_ref=_ref("principal", version="v2"),
            idempotency_key="close-session-store",
            audit=_audit(),
        )
    with pytest.raises(HarnessSessionConflictError):
        store.close_session(
            session_id=created.session.session_id,
            expected_session_version=created.session.session_version,
            principal_ref=_ref("principal"),
            idempotency_key="close-session-store-second",
            audit=_audit(),
        )


def test_session_store_close_fault_rolls_back(tmp_path: Path) -> None:
    path = tmp_path / "harness-close-fault.sqlite3"
    created = _store(path).create_session(
        session_id="session-store-close-fault",
        incarnation_id="session-store-close-fault-incarnation",
        composition_ref=_ref("harness-composition"),
        created_by="session-user",
        idempotency_key="create-session-store-close-fault",
        audit=_audit(),
    )
    faulted = HarnessSessionStore(
        path,
        clock=lambda: NOW,
        fault_injector=lambda point: (
            (_ for _ in ()).throw(RuntimeError("injected"))
            if point == "before_close_session_commit"
            else None
        ),
    )

    with pytest.raises(RuntimeError, match="injected"):
        faulted.close_session(
            session_id=created.session.session_id,
            expected_session_version=created.session.session_version,
            principal_ref=_ref("principal"),
            idempotency_key="close-session-store-fault",
            audit=_audit(),
        )

    reopened = _store(path)
    assert reopened.get_projection(created.session.session_id) == created
    assert reopened.list_sessions(offset=0, limit=100).total == 1
    assert len(
        reopened.list_events(
            created.session.session_id,
            after_sequence=0,
            limit=100,
        ).events
    ) == 1


def test_session_store_rejects_changed_idempotency_request(tmp_path: Path) -> None:
    store = _store(tmp_path / "harness.sqlite3")
    store.create_session(
        session_id="session-store-002",
        incarnation_id="session-store-incarnation-002",
        composition_ref=_ref("harness-composition"),
        created_by="session-user",
        idempotency_key="create-session-store-002",
        audit=_audit(),
    )
    store.start_turn(
        session_id="session-store-002",
        expected_session_version=1,
        principal_ref=_ref("principal"),
        content="first",
        artifact_envelope_refs=(),
        idempotency_key="turn-store-002",
        audit=_audit(),
    )
    with pytest.raises(HarnessSessionConflictError):
        store.start_turn(
            session_id="session-store-002",
            expected_session_version=1,
            principal_ref=_ref("principal"),
            content="changed",
            artifact_envelope_refs=(),
            idempotency_key="turn-store-002",
            audit=_audit(),
        )


def test_session_store_bounds_faults_and_prepared_replay_are_closed(
    tmp_path: Path,
) -> None:
    faulted = HarnessSessionStore(
        tmp_path / "faulted.sqlite3",
        clock=lambda: NOW,
        fault_injector=lambda point: (
            (_ for _ in ()).throw(RuntimeError("injected"))
            if point == "before_create_session_commit"
            else None
        ),
    )
    with pytest.raises(RuntimeError, match="injected"):
        faulted.create_session(
            session_id="session-faulted",
            incarnation_id="session-faulted-incarnation",
            composition_ref=_ref("harness-composition"),
            created_by="session-user",
            idempotency_key="create-session-faulted",
            audit=_audit(),
        )
    with pytest.raises(HarnessSessionNotFoundError):
        faulted.get_projection("session-faulted")

    store = _store(tmp_path / "harness.sqlite3")
    created = store.create_session(
        session_id="session-store-003",
        incarnation_id="session-store-incarnation-003",
        composition_ref=_ref("harness-composition"),
        created_by="session-user",
        idempotency_key="create-session-store-003",
        audit=_audit(),
    )
    assert store.list_sessions(offset=0, limit=10).total == 1
    assert len(store.list_events("session-store-003", after_sequence=0, limit=1).events) == 1
    with pytest.raises(ValueError):
        store.list_sessions(offset=-1, limit=10)
    with pytest.raises(ValueError):
        store.list_events("session-store-003", after_sequence=-1, limit=10)
    with pytest.raises(HarnessSessionConcurrencyError):
        store.start_turn(
            session_id="session-store-003",
            expected_session_version=created.session.session_version + 1,
            principal_ref=_ref("principal"),
            content="stale",
            artifact_envelope_refs=(),
            idempotency_key="turn-store-stale",
            audit=_audit(),
        )
    start = store.start_turn(
        session_id="session-store-003",
        expected_session_version=created.session.session_version,
        principal_ref=_ref("principal"),
        content="prepare",
        artifact_envelope_refs=(),
        idempotency_key="turn-store-prepare",
        audit=_audit(),
    )
    prepared = HarnessGatewayJournalV1.create(
        journal_id="journal-store-prepared-replay",
        session_ref=start.state.session_ref,
        command_ref=start.command.to_ref(),
        state=GatewayJournalStateV1.PREPARED,
        evidence_class=ProviderEvidenceClassV1.MECHANISM_FIXTURE,
        route_ref=_ref("model-route-decision", version="v2"),
        invocation_request_ref=_ref("gateway-invocation-request", version="v2"),
        prompt_template_ref=_ref("prompt-template", version="v2"),
        model_profile_ref=_ref("model-capability-profile", version="v2"),
        predecessor_journal_ref=None,
        receipt_ref=None,
        invocation_result_ref=None,
        output_ref=None,
        gateway_status=None,
        usage=None,
        failure_code=None,
        audit=_audit(),
    )
    assert (
        store.record_prepared_journal(
            session_id=start.state.session_id,
            command=start.command,
            journal=prepared,
        )
        == prepared
    )
    assert (
        store.record_prepared_journal(
            session_id=start.state.session_id,
            command=start.command,
            journal=prepared,
        )
        == prepared
    )
    assert store.get_journal(prepared.to_ref()) == prepared
    assert store.get_turn_start(start.command.to_ref()).command == start.command

    changed = prepared.model_copy(update={"object_id": "harness-gateway-journal://sha256/" + "b" * 64})
    with pytest.raises(HarnessSessionConflictError):
        store.record_prepared_journal(
            session_id=start.state.session_id,
            command=start.command,
            journal=changed,
        )

    connection = sqlite3.connect(store.path)
    try:
        connection.execute(
            "DELETE FROM harness_session_projections WHERE session_id = ?",
            ("session-store-003",),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(HarnessSessionIntegrityError):
        store.get_projection("session-store-003")
    assert store.rebuild_projection("session-store-003").session.session_id == ("session-store-003")


def test_shell_reconcile_intent_is_idempotent_and_restart_safe(
    tmp_path: Path,
) -> None:
    path = tmp_path / "harness-reconcile.sqlite3"
    store = _store(path)
    created = store.create_session(
        session_id="session-shell-reconcile",
        incarnation_id="session-shell-reconcile-incarnation",
        composition_ref=_ref("harness-composition"),
        created_by="session-user",
        idempotency_key="create-session-shell-reconcile",
        audit=_audit(),
    )
    values = {
        "session_id": created.session.session_id,
        "expected_session_version": created.session.session_version,
        "principal_ref": _ref("principal"),
        "idempotency_key": "reconcile-session-shell",
        "effect_ref": _ref("graph-binding"),
    }

    claimed = store.claim_shell_reconcile(**values)
    replayed = _store(path).claim_shell_reconcile(
        **{
            **values,
            "effect_ref": _ref(
                "graph-binding",
                version="v2",
            ),
        }
    )
    assert claimed.replay is False
    assert claimed.effect_ref == values["effect_ref"]
    assert replayed.replay is True
    assert replayed.effect_ref == values["effect_ref"]

    with pytest.raises(HarnessSessionConflictError, match="idempotency"):
        store.claim_shell_reconcile(
            **{
                **values,
                "principal_ref": _ref(
                    "principal",
                    version="v2",
                ),
            }
        )
    with pytest.raises(HarnessSessionConcurrencyError, match="stale"):
        store.claim_shell_reconcile(
            **{
                **values,
                "expected_session_version": 2,
                "idempotency_key": "reconcile-session-shell-stale",
            }
        )


def test_shell_reconcile_intent_fault_rolls_back(
    tmp_path: Path,
) -> None:
    path = tmp_path / "harness-reconcile-fault.sqlite3"
    store = HarnessSessionStore(
        path,
        clock=lambda: NOW,
        fault_injector=lambda point: (
            (_ for _ in ()).throw(RuntimeError("injected"))
            if point == "before_shell_reconcile_claim_commit"
            else None
        ),
    )
    created = store.create_session(
        session_id="session-shell-reconcile-fault",
        incarnation_id="session-shell-reconcile-fault-incarnation",
        composition_ref=_ref("harness-composition"),
        created_by="session-user",
        idempotency_key="create-session-shell-reconcile-fault",
        audit=_audit(),
    )
    values = {
        "session_id": created.session.session_id,
        "expected_session_version": created.session.session_version,
        "principal_ref": _ref("principal"),
        "idempotency_key": "reconcile-session-shell-fault",
        "effect_ref": None,
    }

    with pytest.raises(RuntimeError, match="injected"):
        store.claim_shell_reconcile(**values)

    recovered = _store(path).claim_shell_reconcile(**values)
    assert recovered.replay is False
    assert recovered.effect_ref is None
