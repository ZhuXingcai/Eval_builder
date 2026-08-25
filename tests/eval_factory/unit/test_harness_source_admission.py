from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests.eval_factory.integration.test_harness_agent_loop import (
    _components,
)

from eval_factory.console_api import (
    AgentShellSourceAdmissionCommandV1,
    AgentShellSourceFileClaimV1,
    HarnessSourceAdmissionService,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.harness import (
    HarnessSourceAdmissionConflictError,
    HarnessSourceAdmissionHashError,
    HarnessSourceAdmissionIntegrityError,
    HarnessSourceAdmissionInventoryError,
    HarnessSourceAdmissionLimitError,
    HarnessSourceAdmissionLimits,
    HarnessSourceAdmissionManifestError,
    HarnessSourceAdmissionMetadataError,
    HarnessSourceAdmissionNotFoundError,
    HarnessSourceAdmissionSourceError,
    HarnessSourceAdmissionStore,
    RequirementInterpretationProposalV1,
    SourceUploadPart,
)
from eval_factory.trace import TraceSourceRegistry

NOW = datetime(2026, 8, 23, tzinfo=UTC)
HASH = "a" * 64
MANIFEST_COLUMNS = (
    "instance_id",
    "sid",
    "p_date",
    "business",
    "category",
    "pool_id",
    "pool_category",
)


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v1",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version=version,
        object_sha256=HASH,
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="user://source-admission-test",
        governing_versions=(
            VersionBinding(
                component="source-admission",
                version="v1",
                sha256=HASH,
            ),
        ),
    )


def _trace(
    *,
    sid: str = "sid-001",
    p_date: str = "2026-08-23",
    business: str = "CodingPlan",
    marker: str = "one",
) -> bytes:
    value = {
        "account": "account",
        "api_type": "chat",
        "business": business,
        "endpoint": "internal",
        "event_time": "2026-08-23T00:00:00Z",
        "extra": "{}",
        "mm_urls": "",
        "model": "model",
        "p_date": p_date,
        "request": json.dumps(
            {"messages": [{"role": "user", "content": marker}]},
            sort_keys=True,
        ),
        "response": "[]",
        "sid": sid,
        "source": None,
    }
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def _manifest(
    rows: tuple[dict[str, str], ...] | None = None,
    *,
    columns: tuple[str, ...] = MANIFEST_COLUMNS,
) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        rows
        or (
            {
                "instance_id": "LH_001",
                "sid": "sid-001",
                "p_date": "2026-08-23",
                "business": "CodingPlan",
                "category": "agent",
                "pool_id": "pool-001",
                "pool_category": "evaluation",
            },
        )
    )
    return output.getvalue().encode()


def _claim(name: str, payload: bytes) -> AgentShellSourceFileClaimV1:
    return AgentShellSourceFileClaimV1(
        relative_name=name,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_size_bytes=len(payload),
    )


def _setup(
    tmp_path: Path,
    *,
    fault: str | None = None,
    limits: HarnessSourceAdmissionLimits | None = None,
    session_ids: tuple[str, ...] = ("session-source-admission",),
) -> tuple[HarnessSourceAdmissionService, HarnessSourceAdmissionStore]:
    proposal = RequirementInterpretationProposalV1(
        outcome="CLARIFICATION_REQUIRED",
        assistant_message="请补充数据来源。",
        missing_field_codes=("SOURCE_EXPECTATION_MISSING",),
        clarification_questions=("数据来源是什么?",),
    )
    sessions, _, _, _ = _components(
        tmp_path / "harness",
        proposal,
    )
    for session_id in session_ids:
        sessions.create_session(
            session_id=session_id,
            incarnation_id=f"{session_id}-incarnation",
            composition_ref=_ref("harness-composition", "source-admission"),
            created_by="user://source-admission-test",
            idempotency_key=f"create-{session_id}",
            audit=_audit(),
        )
    store = HarnessSourceAdmissionStore(
        tmp_path / "source-store",
        limits=limits,
        fault_injector=(
            (lambda point: (_ for _ in ()).throw(RuntimeError("injected")) if point == fault else None)
            if fault is not None
            else None
        ),
    )
    service = HarnessSourceAdmissionService(
        sessions=sessions,
        store=store,
        trace_schema_ref=_ref("json-schema", "raw-traj-v1"),
        producer_capability_ref=_ref(
            "harness-capability-definition",
            "trace-ingestion",
        ),
        governing_versions=_audit().governing_versions,
        clock=lambda: NOW,
    )
    return service, store


def _admit(
    service: HarnessSourceAdmissionService,
    *,
    manifest: bytes | None = None,
    traces: tuple[tuple[str, bytes], ...] | None = None,
    key: str = "admit-source-files",
    expected_session_version: int = 1,
    principal: str = "user://source-admission-test",
    session_id: str = "session-source-admission",
):
    manifest_bytes = manifest or _manifest()
    trace_values = traces or (("LH_001_sid-001.jsonl", _trace()),)
    command = AgentShellSourceAdmissionCommandV1(
        expected_session_version=expected_session_version,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        manifest_size_bytes=len(manifest_bytes),
        trace_files=tuple(_claim(name, payload) for name, payload in sorted(trace_values)),
        idempotency_key=key,
    )
    return service.admit(
        session_id,
        command,
        manifest=SourceUploadPart(
            filename="manifest.csv",
            stream=io.BytesIO(manifest_bytes),
        ),
        traces=tuple(
            SourceUploadPart(filename=name, stream=io.BytesIO(payload)) for name, payload in trace_values
        ),
        principal=principal,
    )


def test_source_admission_persists_replays_and_reopens_without_paths(
    tmp_path: Path,
) -> None:
    service, store = _setup(tmp_path)

    first = _admit(service)
    replay = _admit(service)
    second_key = _admit(service, key="admit-source-files-second-key")

    assert replay == first
    assert second_key == first
    assert first.source_count == 1
    assert first.files[0].relative_name == "manifest.csv"
    assert first.files[1].relative_name == "LH_001_sid-001.jsonl"
    assert first.files[1].source_ref is not None
    assert first.files[1].artifact_envelope_ref is not None
    rendered = first.model_dump_json()
    assert str(tmp_path) not in rendered
    assert '"request"' not in rendered
    source_path = store.source_path(first.files[1].source_ref)
    assert source_path.suffix == ".jsonl"
    assert source_path.read_bytes() == _trace()
    registered = TraceSourceRegistry(
        tmp_path / "trace-registry.sqlite3",
    ).register(
        source_path,
        source_trace_id=first.files[1].source_ref.object_id,
        source_uri=(f"source-admission://{first.files[1].source_ref.object_sha256}"),
    )
    assert registered.source.raw_sha256 == first.files[1].sha256
    assert not tuple(store.staging.iterdir())
    connection = sqlite3.connect(store.path)
    try:
        persisted = "\n".join(
            str(value)
            for row in connection.execute(
                """
                SELECT admission_json FROM harness_source_admissions
                UNION ALL
                SELECT member_json
                FROM harness_source_admission_members
                UNION ALL
                SELECT envelope_json
                FROM harness_source_admission_members
                """
            ).fetchall()
            for value in row
        )
    finally:
        connection.close()
    assert str(tmp_path) not in persisted
    assert '"messages"' not in persisted
    assert '"content"' not in persisted
    assert '"one"' not in persisted

    reopened = HarnessSourceAdmissionStore(
        store.root,
        limits=store.limits,
    )
    assert service.get("session-source-admission") == first
    assert reopened.get(first.admission_ref).admission.to_ref() == first.admission_ref


def test_source_admission_reuses_content_authority_across_sessions(
    tmp_path: Path,
) -> None:
    service, store = _setup(
        tmp_path,
        session_ids=(
            "session-source-admission",
            "session-source-admission-second",
        ),
    )

    first = _admit(service)
    second = _admit(
        service,
        key="admit-source-files-second-session",
        session_id="session-source-admission-second",
    )

    assert first.admission_ref != second.admission_ref
    assert first.files[1].source_ref == second.files[1].source_ref
    assert first.files[1].artifact_envelope_ref == second.files[1].artifact_envelope_ref
    source_ref = first.files[1].source_ref
    assert source_ref is not None
    assert store.source_path(source_ref).read_bytes() == _trace()


def test_source_execution_view_replays_and_rejects_inventory_drift(
    tmp_path: Path,
) -> None:
    service, store = _setup(tmp_path)
    admitted = _admit(service)

    first = store.materialize_for_execution(
        "session-source-admission",
    )
    replay = HarnessSourceAdmissionStore(
        store.root,
        limits=store.limits,
    ).materialize_for_execution(
        "session-source-admission",
    )

    assert replay == first
    assert first.admission_ref == admitted.admission_ref
    assert first.manifest_path == first.raw_root / "manifest.csv"
    assert first.manifest_path.read_bytes() == _manifest()
    assert tuple(path.name for path in first.trace_paths) == ("LH_001_sid-001.jsonl",)
    assert first.trace_paths[0].read_bytes() == _trace()
    source_ref = admitted.files[1].source_ref
    assert source_ref is not None
    assert first.trace_paths[0].stat().st_ino == store.source_path(source_ref).stat().st_ino

    (first.raw_root / "unexpected.jsonl").write_bytes(_trace())
    with pytest.raises(
        HarnessSourceAdmissionIntegrityError,
        match="inventory",
    ):
        store.materialize_for_execution(
            "session-source-admission",
        )


def test_source_admission_changed_replay_and_second_authority_conflict(
    tmp_path: Path,
) -> None:
    service, _ = _setup(tmp_path)
    _admit(service)
    changed = _trace(marker="changed")

    with pytest.raises(
        HarnessSourceAdmissionConflictError,
        match="idempotency",
    ):
        _admit(
            service,
            traces=(("LH_001_sid-001.jsonl", changed),),
        )
    with pytest.raises(
        HarnessSourceAdmissionConflictError,
        match="another",
    ):
        _admit(
            service,
            traces=(("LH_001_sid-001.jsonl", changed),),
            key="different-source-admission-key",
        )
    with pytest.raises(
        HarnessSourceAdmissionConflictError,
        match="another",
    ):
        _admit(
            service,
            key="different-source-admission-principal",
            principal="user://another-source-owner",
        )


@pytest.mark.parametrize(
    ("manifest", "traces", "error"),
    [
        (
            b"instance_id,sid\nLH_001,sid-001\n",
            (("LH_001_sid-001.jsonl", _trace()),),
            HarnessSourceAdmissionManifestError,
        ),
        (
            (
                b"instance_id,sid,p_date,business,category,pool_id,pool_category\n"
                b'LH_001,sid-001,2026-08-23,"unterminated\n'
            ),
            (("LH_001_sid-001.jsonl", _trace()),),
            HarnessSourceAdmissionManifestError,
        ),
        (
            _manifest(),
            (("LH_002_sid-002.jsonl", _trace()),),
            HarnessSourceAdmissionInventoryError,
        ),
        (
            _manifest(),
            (("LH_001_sid-001.jsonl", b"not-json\n"),),
            HarnessSourceAdmissionSourceError,
        ),
        (
            _manifest(),
            (("LH_001_sid-001.jsonl", _trace(sid="sid-other")),),
            HarnessSourceAdmissionMetadataError,
        ),
    ],
)
def test_source_admission_rejects_invalid_manifest_inventory_and_trace(
    tmp_path: Path,
    manifest: bytes,
    traces: tuple[tuple[str, bytes], ...],
    error: type[Exception],
) -> None:
    service, store = _setup(tmp_path)

    with pytest.raises(error):
        _admit(service, manifest=manifest, traces=traces)

    assert not tuple(store.staging.iterdir())
    with pytest.raises(HarnessSourceAdmissionNotFoundError):
        store.get_for_session("session-source-admission")


def test_source_admission_rejects_hash_size_alias_and_limits(
    tmp_path: Path,
) -> None:
    service, store = _setup(
        tmp_path,
        limits=HarnessSourceAdmissionLimits(
            max_source_files=2,
            max_manifest_bytes=1024,
            max_source_bytes=1024,
            max_total_source_bytes=2048,
            max_request_bytes=4096,
        ),
    )
    trace = _trace()
    manifest = _manifest()
    bad_command = AgentShellSourceAdmissionCommandV1(
        expected_session_version=1,
        manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        manifest_size_bytes=len(manifest),
        trace_files=(
            AgentShellSourceFileClaimV1(
                relative_name="LH_001_sid-001.jsonl",
                expected_sha256="b" * 64,
                expected_size_bytes=len(trace),
            ),
        ),
        idempotency_key="bad-hash",
    )
    with pytest.raises(HarnessSourceAdmissionHashError):
        service.admit(
            "session-source-admission",
            bad_command,
            manifest=SourceUploadPart(
                filename="manifest.csv",
                stream=io.BytesIO(manifest),
            ),
            traces=(
                SourceUploadPart(
                    filename="LH_001_sid-001.jsonl",
                    stream=io.BytesIO(trace),
                ),
            ),
            principal="user://source-admission-test",
        )

    rows = (
        {
            "instance_id": "LH_001",
            "sid": "sid-001",
            "p_date": "2026-08-23",
            "business": "CodingPlan",
            "category": "agent",
            "pool_id": "pool-001",
            "pool_category": "evaluation",
        },
        {
            "instance_id": "LH_002",
            "sid": "sid-002",
            "p_date": "2026-08-23",
            "business": "CodingPlan",
            "category": "agent",
            "pool_id": "pool-002",
            "pool_category": "evaluation",
        },
    )
    with pytest.raises(HarnessSourceAdmissionInventoryError, match="aliased"):
        _admit(
            service,
            manifest=_manifest(rows),
            traces=(
                ("LH_001_sid-001.jsonl", trace),
                ("LH_002_sid-002.jsonl", trace),
            ),
            key="aliased-content",
        )
    with pytest.raises(HarnessSourceAdmissionLimitError):
        _admit(
            service,
            traces=(
                (
                    "LH_001_sid-001.jsonl",
                    trace + (b"x" * 1024),
                ),
            ),
            key="oversized-source",
        )
    lying_command = AgentShellSourceAdmissionCommandV1(
        expected_session_version=1,
        manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        manifest_size_bytes=len(manifest),
        trace_files=(
            AgentShellSourceFileClaimV1(
                relative_name="LH_001_sid-001.jsonl",
                expected_sha256=hashlib.sha256(trace).hexdigest(),
                expected_size_bytes=len(trace),
            ),
        ),
        idempotency_key="lying-source-size",
    )
    with pytest.raises(HarnessSourceAdmissionLimitError):
        service.admit(
            "session-source-admission",
            lying_command,
            manifest=SourceUploadPart(
                filename="manifest.csv",
                stream=io.BytesIO(manifest),
            ),
            traces=(
                SourceUploadPart(
                    filename="LH_001_sid-001.jsonl",
                    stream=io.BytesIO(trace + (b"x" * 1024)),
                ),
            ),
            principal="user://source-admission-test",
        )
    assert not tuple(store.staging.iterdir())


def test_source_admission_orders_multiple_sources_and_envelopes(
    tmp_path: Path,
) -> None:
    service, store = _setup(tmp_path)
    rows = (
        {
            "instance_id": "LH_002",
            "sid": "sid-002",
            "p_date": "2026-08-23",
            "business": "CodingPlan",
            "category": "agent",
            "pool_id": "pool-002",
            "pool_category": "evaluation",
        },
        {
            "instance_id": "LH_001",
            "sid": "sid-001",
            "p_date": "2026-08-23",
            "business": "CodingPlan",
            "category": "agent",
            "pool_id": "pool-001",
            "pool_category": "evaluation",
        },
    )
    result = _admit(
        service,
        manifest=_manifest(rows),
        traces=(
            (
                "LH_002_sid-002.jsonl",
                _trace(sid="sid-002", marker="two"),
            ),
            ("LH_001_sid-001.jsonl", _trace()),
        ),
        key="multi-source-admission",
    )

    assert tuple(item.relative_name for item in result.files[1:]) == (
        "LH_001_sid-001.jsonl",
        "LH_002_sid-002.jsonl",
    )
    assert result.artifact_envelope_refs == tuple(
        sorted(
            result.artifact_envelope_refs,
            key=lambda item: (
                item.object_type,
                item.object_id,
                item.object_version,
                item.object_sha256,
            ),
        )
    )
    assert all(
        store.source_path(item.source_ref).is_file()
        for item in result.files[1:]
        if item.source_ref is not None
    )


@pytest.mark.parametrize(
    "fault",
    (
        "after_source_admission_stage",
        "after_source_admission_cas",
        "before_source_admission_commit",
    ),
)
def test_source_admission_faults_leave_no_partial_authority_and_retry(
    tmp_path: Path,
    fault: str,
) -> None:
    service, store = _setup(tmp_path, fault=fault)

    with pytest.raises(RuntimeError, match="injected"):
        _admit(service)

    with pytest.raises(HarnessSourceAdmissionNotFoundError):
        store.get_for_session("session-source-admission")
    assert not tuple(store.staging.iterdir())
    connection = sqlite3.connect(store.path)
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM harness_source_admissions",
        ).fetchone() == (0,)
    finally:
        connection.close()

    recovered_store = HarnessSourceAdmissionStore(
        store.root,
        limits=store.limits,
    )
    recovered_service, _ = _setup(tmp_path / "recovery-harness")
    recovered_service.store = recovered_store
    result = _admit(recovered_service)
    assert result.source_count == 1


def test_source_admission_detects_sqlite_and_cas_drift(
    tmp_path: Path,
) -> None:
    service, store = _setup(tmp_path)
    result = _admit(service)

    connection = sqlite3.connect(store.path)
    try:
        connection.execute(
            """
            UPDATE harness_source_admissions
            SET source_count = 2
            WHERE admission_object_id = ?
            """,
            (result.admission_ref.object_id,),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(
        HarnessSourceAdmissionIntegrityError,
        match="columns drifted",
    ):
        store.get(result.admission_ref)

    connection = sqlite3.connect(store.path)
    try:
        connection.execute(
            """
            UPDATE harness_source_admissions
            SET source_count = 1
            WHERE admission_object_id = ?
            """,
            (result.admission_ref.object_id,),
        )
        connection.commit()
    finally:
        connection.close()
    source_ref = result.files[1].source_ref
    assert source_ref is not None
    store.source_path(source_ref).write_bytes(b"corrupt")
    with pytest.raises(
        HarnessSourceAdmissionIntegrityError,
        match="integrity",
    ):
        store.get(result.admission_ref)


def test_source_admission_staging_recovery_is_explicit_and_bounded(
    tmp_path: Path,
) -> None:
    _, store = _setup(tmp_path)
    directory = store.staging / "abandoned"
    directory.mkdir()
    (directory / "part").write_bytes(b"partial")
    file = store.staging / "orphan"
    file.write_bytes(b"partial")
    link = store.staging / "link"
    link.symlink_to(file)

    assert store.recover_staging() == 3
    assert not tuple(store.staging.iterdir())

    root_target = tmp_path / "real-root"
    root_target.mkdir()
    root_link = tmp_path / "linked-root"
    root_link.symlink_to(root_target, target_is_directory=True)
    with pytest.raises(
        HarnessSourceAdmissionIntegrityError,
        match="non-symlink",
    ):
        HarnessSourceAdmissionStore(root_link)
