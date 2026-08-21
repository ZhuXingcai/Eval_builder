from __future__ import annotations

import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_factory.contracts.core import ObjectRef
from eval_factory.memory import (
    AgentMemoryService,
    AgentMemoryStore,
    MemoryAccessContextV1,
    MemoryAuthorizationError,
    MemoryCandidateV1,
    MemoryCapabilityUnavailableError,
    MemoryConcurrencyError,
    MemoryIdempotencyConflictError,
    MemoryIntegrityError,
    MemoryKindV1,
    MemoryNamespaceV1,
    MemoryPolicyError,
    MemoryRecallQueryV1,
    MemoryRetrievalModeV1,
    MemorySensitivityV1,
    MemoryTombstonedError,
    MemoryVisibilityV1,
    StoredMemoryV1,
)

NOW = datetime(2026, 8, 14, 12, tzinfo=UTC)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ref(
    object_type: str = "agent-result-envelope",
    suffix: str = "source",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://memory-test/{suffix}",
        object_version="v2",
        object_sha256=_digest(f"{object_type}:{suffix}"),
    )


def _access(
    *,
    tenant_id: str = "tenant-a",
    project_id: str = "project-a",
    subject_id: str = "evaluation-domain-a",
    requester_agent_id: str = "planner-agent",
) -> MemoryAccessContextV1:
    return MemoryAccessContextV1(
        tenant_id=tenant_id,
        project_id=project_id,
        subject_id=subject_id,
        requester_agent_id=requester_agent_id,
    )


def _namespace(
    *,
    visibility: MemoryVisibilityV1 = MemoryVisibilityV1.PROJECT_SHARED,
    owner_agent_id: str | None = None,
    tenant_id: str = "tenant-a",
    project_id: str = "project-a",
    subject_id: str = "evaluation-domain-a",
) -> MemoryNamespaceV1:
    return MemoryNamespaceV1(
        tenant_id=tenant_id,
        project_id=project_id,
        subject_id=subject_id,
        visibility=visibility,
        owner_agent_id=owner_agent_id,
    )


def _candidate(
    memory_id: str = "agent-memory://evaluation-domain-a/powershell-recovery",
    *,
    content: str = "PowerShell recovery succeeds after inspecting the paired tool error.",
    namespace: MemoryNamespaceV1 | None = None,
    kind: MemoryKindV1 = MemoryKindV1.EPISODIC,
    tags: tuple[str, ...] = ("powershell", "recovery"),
    source_refs: tuple[ObjectRef, ...] | None = None,
    valid_from: datetime = NOW - timedelta(days=1),
    valid_until: datetime | None = None,
    expires_at: datetime | None = None,
    importance_basis_points: int = 7_500,
    sensitivity: MemorySensitivityV1 = MemorySensitivityV1.INTERNAL,
    approval_ref: ObjectRef | None = None,
) -> MemoryCandidateV1:
    return MemoryCandidateV1(
        memory_id=memory_id,
        namespace=namespace or _namespace(),
        kind=kind,
        content=content,
        tags=tags,
        source_refs=source_refs or (_ref(),),
        valid_from=valid_from,
        valid_until=valid_until,
        expires_at=expires_at,
        importance_basis_points=importance_basis_points,
        sensitivity=sensitivity,
        approval_ref=approval_ref,
    )


def _store(
    path: Path,
    *,
    trusted_access_contexts: frozenset[MemoryAccessContextV1] | None = None,
    trusted_approval_refs: frozenset[ObjectRef] = frozenset(),
) -> AgentMemoryStore:
    return AgentMemoryStore(
        path,
        trusted_access_contexts=(
            trusted_access_contexts if trusted_access_contexts is not None else frozenset({_access()})
        ),
        trusted_approval_refs=trusted_approval_refs,
    )


def _remember(
    store: AgentMemoryStore,
    candidate: MemoryCandidateV1,
    *,
    access: MemoryAccessContextV1 | None = None,
    expected_revision: int = 0,
    idempotency_key: str | None = None,
    created_at: datetime = NOW,
):
    return store.remember(
        candidate,
        access=access or _access(),
        expected_revision=expected_revision,
        idempotency_key=idempotency_key or f"remember-{candidate.memory_id}-{expected_revision}",
        created_at=created_at,
    )


def test_memory_models_are_strict_and_scope_sensitive() -> None:
    with pytest.raises(ValidationError, match="owner_agent_id"):
        _namespace(
            visibility=MemoryVisibilityV1.AGENT_PRIVATE,
            owner_agent_id=None,
        )
    with pytest.raises(ValidationError, match="project-shared"):
        _namespace(owner_agent_id="planner-agent")
    with pytest.raises(ValidationError, match="sorted and unique"):
        _candidate(tags=("recovery", "powershell"))
    with pytest.raises(ValidationError, match="memory-admission-approval"):
        _candidate(
            sensitivity=MemorySensitivityV1.RESTRICTED,
            approval_ref=None,
        )
    with pytest.raises(ValidationError, match="extra"):
        MemoryAccessContextV1.model_validate(
            {
                **_access().model_dump(mode="python"),
                "unknown": True,
            }
        )


def test_memory_round_trip_revision_reopen_and_exact_replay(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.sqlite"
    store = _store(path)
    first_candidate = _candidate()
    first = _remember(store, first_candidate, idempotency_key="remember-1")
    replay = _remember(store, first_candidate, idempotency_key="remember-1")

    assert replay == first
    assert first.revision == 1
    assert store.get_current(first.memory_id, access=_access()).content == first_candidate.content

    successor_candidate = _candidate(
        content="PowerShell recovery succeeds after pairing the error and correcting the command.",
        tags=("corrected-command", "powershell", "recovery"),
        source_refs=(_ref(suffix="successor"),),
    )
    successor = _remember(
        store,
        successor_candidate,
        expected_revision=1,
        idempotency_key="remember-2",
        created_at=NOW + timedelta(minutes=1),
    )

    assert successor.revision == 2
    assert successor.predecessor_ref == first.to_ref()
    reopened = _store(path)
    assert reopened.get_current(first.memory_id, access=_access()) == StoredMemoryV1(
        record=successor,
        content=successor_candidate.content,
    )
    with pytest.raises(MemoryConcurrencyError, match="stale"):
        _remember(
            reopened,
            successor_candidate,
            expected_revision=1,
            idempotency_key="remember-stale",
        )
    with pytest.raises(MemoryConcurrencyError, match="timestamp"):
        _remember(
            reopened,
            successor_candidate,
            expected_revision=2,
            idempotency_key="remember-time-travel",
            created_at=NOW,
        )


def test_memory_idempotency_conflict_and_tombstoned_id_never_reactivates(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "memory.sqlite")
    candidate = _candidate()
    record = _remember(store, candidate, idempotency_key="same-key")
    with pytest.raises(MemoryIdempotencyConflictError):
        _remember(
            store,
            _candidate(content="A changed request must not reuse the key."),
            idempotency_key="same-key",
        )

    tombstone = store.forget(
        record.memory_id,
        access=_access(),
        expected_revision=1,
        reason_code="user-requested",
        idempotency_key="forget-key",
        deleted_at=NOW + timedelta(hours=1),
    )
    assert (
        store.forget(
            record.memory_id,
            access=_access(),
            expected_revision=1,
            reason_code="user-requested",
            idempotency_key="forget-key",
            deleted_at=NOW + timedelta(hours=1),
        )
        == tombstone
    )
    with pytest.raises(MemoryTombstonedError):
        _remember(
            store,
            candidate,
            idempotency_key="same-key",
        )


def test_concurrent_successor_writes_have_one_authority(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "memory.sqlite")
    first = _remember(store, _candidate(), idempotency_key="first")
    candidates = (
        _candidate(
            content="PowerShell recovery variant alpha.",
            source_refs=(_ref(suffix="alpha"),),
        ),
        _candidate(
            content="PowerShell recovery variant beta.",
            source_refs=(_ref(suffix="beta"),),
        ),
    )

    def write(index: int) -> object:
        try:
            return _remember(
                store,
                candidates[index],
                expected_revision=first.revision,
                idempotency_key=f"concurrent-{index}",
                created_at=NOW + timedelta(minutes=1),
            )
        except MemoryConcurrencyError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(write, range(2)))

    assert sum(not isinstance(result, Exception) for result in results) == 1
    assert sum(isinstance(result, MemoryConcurrencyError) for result in results) == 1
    current = store.get_current(first.memory_id, access=_access())
    assert current.record.revision == 2
    assert current.content in {candidate.content for candidate in candidates}


def test_shared_private_and_cross_scope_access_isolation(
    tmp_path: Path,
) -> None:
    other_agent = _access(requester_agent_id="criteria-agent")
    store = _store(
        tmp_path / "memory.sqlite",
        trusted_access_contexts=frozenset({_access(), other_agent}),
    )
    shared = _candidate(memory_id="agent-memory://evaluation-domain-a/shared")
    private = _candidate(
        memory_id="agent-memory://evaluation-domain-a/private",
        namespace=_namespace(
            visibility=MemoryVisibilityV1.AGENT_PRIVATE,
            owner_agent_id="planner-agent",
        ),
    )
    _remember(store, shared)
    _remember(store, private)

    with pytest.raises(MemoryAuthorizationError, match="not trusted"):
        _store(
            tmp_path / "untrusted.sqlite",
            trusted_access_contexts=frozenset(),
        ).list_current(access=_access())
    assert tuple(value.record.memory_id for value in store.list_current(access=other_agent)) == (
        shared.memory_id,
    )
    with pytest.raises(MemoryAuthorizationError):
        store.get_current(private.memory_id, access=other_agent)
    with pytest.raises(MemoryAuthorizationError):
        store.get_current(
            shared.memory_id,
            access=_access(project_id="project-b"),
        )
    with pytest.raises(MemoryAuthorizationError):
        _remember(
            store,
            _candidate(
                memory_id="agent-memory://evaluation-domain-a/cross-tenant",
                namespace=_namespace(tenant_id="tenant-b"),
            ),
        )


def test_memory_admission_rejects_forbidden_sensitive_and_secret_content(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "memory.sqlite")
    with pytest.raises(MemoryPolicyError, match="forbidden"):
        _remember(
            store,
            _candidate(sensitivity=MemorySensitivityV1.FORBIDDEN),
        )
    with pytest.raises(MemoryPolicyError, match="credential"):
        _remember(
            store,
            _candidate(content="api_key = abcdefghijklmnopqrstuvwxyz123456"),
        )
    with pytest.raises(MemoryPolicyError, match="source type"):
        _remember(
            store,
            _candidate(
                source_refs=(_ref("final-output", "unsafe"),),
            ),
        )

    approval = _ref("memory-admission-approval", "approved")
    restricted = _candidate(
        memory_id="agent-memory://evaluation-domain-a/restricted",
        sensitivity=MemorySensitivityV1.RESTRICTED,
        approval_ref=approval,
    )
    with pytest.raises(MemoryAuthorizationError, match="not trusted"):
        _remember(store, restricted)
    trusted = _store(
        tmp_path / "trusted-memory.sqlite",
        trusted_approval_refs=frozenset({approval}),
    )
    assert _remember(trusted, restricted).sensitivity is MemorySensitivityV1.RESTRICTED


def test_lexical_recall_is_deterministic_bounded_and_temporally_current(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "memory.sqlite")
    _remember(
        store,
        _candidate(
            memory_id="agent-memory://evaluation-domain-a/exact",
            content="PowerShell recovery should inspect the paired error before retry.",
            importance_basis_points=9_000,
        ),
        created_at=NOW - timedelta(days=2),
    )
    _remember(
        store,
        _candidate(
            memory_id="agent-memory://evaluation-domain-a/partial",
            content="Recovery should preserve provenance.",
            importance_basis_points=2_000,
        ),
        created_at=NOW - timedelta(days=60),
    )
    _remember(
        store,
        _candidate(
            memory_id="agent-memory://evaluation-domain-a/future",
            content="PowerShell recovery from a future policy.",
            valid_from=NOW + timedelta(days=1),
        ),
    )
    _remember(
        store,
        _candidate(
            memory_id="agent-memory://evaluation-domain-a/expired",
            content="PowerShell recovery from an expired policy.",
            valid_from=NOW - timedelta(days=10),
            expires_at=NOW - timedelta(seconds=1),
        ),
    )
    _remember(
        store,
        _candidate(
            memory_id="agent-memory://evaluation-domain-a/procedure",
            content="PowerShell recovery procedure.",
            kind=MemoryKindV1.PROCEDURAL,
        ),
    )
    _remember(
        store,
        _candidate(
            memory_id="agent-memory://evaluation-domain-a/not-yet-observed",
            content="PowerShell recovery with a retroactive business date.",
        ),
        created_at=NOW + timedelta(days=1),
    )
    service = AgentMemoryService(store=store)
    query = MemoryRecallQueryV1(
        access=_access(),
        query_text="PowerShell recovery",
        mode=MemoryRetrievalModeV1.LEXICAL,
        kinds=(MemoryKindV1.EPISODIC,),
        required_tags=("powershell",),
        limit=10,
        evaluated_at=NOW,
    )

    first = service.recall(query)
    second = service.recall(query)

    assert first == second
    assert first.query_sha256 == query.canonical_sha256()
    assert first.candidate_count == 2
    assert tuple(match.memory.record.memory_id for match in first.matches) == (
        "agent-memory://evaluation-domain-a/exact",
        "agent-memory://evaluation-domain-a/partial",
    )
    assert first.matches[0].total_score_basis_points > first.matches[1].total_score_basis_points
    assert all(match.semantic_score_basis_points is None for match in first.matches)


class _StaticSemanticIndex:
    def score(
        self,
        *,
        query_text: str,
        candidates: tuple[StoredMemoryV1, ...],
    ) -> dict[str, int]:
        assert query_text
        return {
            item.record.memory_record_id: (9_000 if "paired" in item.content else 1_000)
            for item in candidates
        }


class _EscapingSemanticIndex:
    def score(
        self,
        *,
        query_text: str,
        candidates: tuple[StoredMemoryV1, ...],
    ) -> dict[str, int]:
        return {"agent-memory-record://unauthorized": 10_000}


def test_required_semantic_recall_fails_closed_and_adapter_cannot_expand_scope(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "memory.sqlite")
    candidate = _candidate()
    _remember(store, candidate)
    query = MemoryRecallQueryV1(
        access=_access(),
        query_text="diagnose shell failure",
        mode=MemoryRetrievalModeV1.SEMANTIC_REQUIRED,
        evaluated_at=NOW,
    )

    with pytest.raises(MemoryCapabilityUnavailableError, match="unavailable"):
        AgentMemoryService(store=store).recall(query)
    result = AgentMemoryService(
        store=store,
        semantic_index=_StaticSemanticIndex(),
    ).recall(query)
    assert len(result.matches) == 1
    assert result.matches[0].semantic_score_basis_points == 9_000
    with pytest.raises(MemoryCapabilityUnavailableError, match="unauthorized"):
        AgentMemoryService(
            store=store,
            semantic_index=_EscapingSemanticIndex(),
        ).recall(query)


def test_forget_physically_erases_private_revision_material(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.sqlite"
    store = _store(path)
    unique_private_content = "PRIVATE-MEMORY-CONTENT-ERASURE-MARKER-8d41fa93"
    first = _remember(
        store,
        _candidate(content=unique_private_content),
        idempotency_key="remember-1",
    )
    second_candidate = _candidate(
        content="PowerShell recovery now uses a corrected command.",
        tags=("corrected-command", "powershell", "recovery"),
        source_refs=(_ref(suffix="second"),),
    )
    second = _remember(
        store,
        second_candidate,
        expected_revision=1,
        idempotency_key="remember-2",
        created_at=NOW + timedelta(minutes=1),
    )

    tombstone = store.forget(
        first.memory_id,
        access=_access(),
        expected_revision=2,
        reason_code="retention-expired",
        idempotency_key="forget",
        deleted_at=NOW + timedelta(days=1),
    )

    assert tombstone.prior_record_ref == second.to_ref()
    assert store.content_row_count(first.memory_id) == 0
    assert store.erased_revision_digest_count(first.memory_id) == 2
    assert store.get_tombstone(first.memory_id, access=_access()) == tombstone
    with pytest.raises(MemoryTombstonedError):
        store.get_current(first.memory_id, access=_access())
    with pytest.raises(MemoryAuthorizationError):
        store.get_current(
            first.memory_id,
            access=_access(project_id="project-b"),
        )

    connection = sqlite3.connect(path)
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_revisions WHERE memory_id = ?",
            (first.memory_id,),
        ).fetchone() == (0,)
        assert connection.execute(
            """
            SELECT COUNT(*) FROM memory_idempotency
            WHERE response_type = 'agent-memory-record'
            """
        ).fetchone() == (0,)
        retained = "\n".join(
            str(value)
            for row in connection.execute(
                """
                SELECT response_json FROM memory_idempotency
                UNION ALL SELECT record_json FROM memory_tombstones
                """
            )
            for value in row
        )
        assert second_candidate.content not in retained
        assert "corrected-command" not in retained
    finally:
        connection.close()
    for artifact in (
        path,
        Path(f"{path}-wal"),
        Path(f"{path}-shm"),
    ):
        if artifact.exists():
            assert unique_private_content.encode("utf-8") not in artifact.read_bytes()
            assert second_candidate.content.encode("utf-8") not in artifact.read_bytes()


def test_memory_corruption_and_materialized_drift_fail_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.sqlite"
    store = _store(path)
    record = _remember(store, _candidate())
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            UPDATE memory_contents
            SET content_text = 'corrupt'
            WHERE memory_id = ? AND revision = 1
            """,
            (record.memory_id,),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(MemoryIntegrityError, match="hash"):
        store.get_current(record.memory_id, access=_access())

    second_path = tmp_path / "head.sqlite"
    second_store = _store(second_path)
    second_record = _remember(second_store, _candidate())
    connection = sqlite3.connect(second_path)
    try:
        connection.execute(
            "UPDATE memory_heads SET kind = 'SEMANTIC' WHERE memory_id = ?",
            (second_record.memory_id,),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(MemoryIntegrityError, match="head"):
        second_store.get_current(second_record.memory_id, access=_access())
