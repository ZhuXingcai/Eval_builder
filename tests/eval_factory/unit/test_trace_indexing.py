from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.contracts.core import ContractAudit, VersionBinding
from eval_factory.contracts.trace import (
    Completeness,
    FileOperation,
    ToolFamily,
)
from eval_factory.trace import (
    RawTrajRecovery,
    RawTrajV1Normalizer,
    RawTrajV1Parser,
    TraceIndexBuilder,
    TraceSourceRegistry,
)
from eval_factory.trace.indexing import TraceIndexResult
from eval_factory.trace.indexing.paths import project_logical_path
from eval_factory.trace.normalization import TraceNormalizationResult


def _outer(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "account": "synthetic",
        "api_type": "chat",
        "business": "IndexingGold",
        "endpoint": "offline",
        "event_time": "2026-07-22T00:00:00Z",
        "extra": '{"req_lost_number":0,"resp_lost_number":0}',
        "mm_urls": "",
        "model": "offline-model",
        "p_date": "2026-07-22",
        "request": '{"messages":[]}',
        "response": '{"content":[],"role":"assistant"}',
        "sid": "synthetic-session",
        "source": None,
    }
    value.update(overrides)
    return value


def _write_outer(path: Path, *records: dict[str, object]) -> None:
    path.write_text(
        "\n".join(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )


def _audit(
    created_at: datetime = datetime(2026, 7, 22, tzinfo=UTC),
) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="trace-indexing-test",
        governing_versions=(VersionBinding(component="raw-traj-file-observation", version="v1"),),
    )


def _normalize(
    source: Path,
    tmp_path: Path,
    *,
    source_trace_id: str = "source-trace://indexing-test",
    audit: ContractAudit | None = None,
) -> TraceNormalizationResult:
    active_audit = audit or _audit()
    registry = TraceSourceRegistry(tmp_path / "registry.sqlite3")
    registered = registry.register(
        source,
        source_trace_id=source_trace_id,
        source_uri=f"raw-traj://{source_trace_id.rsplit('/', 1)[-1]}",
    )
    parsed = RawTrajV1Parser().parse(source, registered_source=registered, audit=active_audit)
    recovered = RawTrajRecovery().recover(source, parse_result=parsed, audit=active_audit)
    return RawTrajV1Normalizer().normalize(source, recovery_result=recovered, audit=active_audit)


def _index(
    source: Path,
    tmp_path: Path,
    *,
    source_trace_id: str = "source-trace://indexing-test",
) -> TraceIndexResult:
    normalized = _normalize(source, tmp_path, source_trace_id=source_trace_id)
    return TraceIndexBuilder().build(normalization_result=normalized, audit=_audit())


def test_logical_path_projection_is_safe_and_deterministic() -> None:
    relative = project_logical_path("relative/file.txt")
    posix = project_logical_path("/workspace/a.txt")
    windows = project_logical_path("C:\\work\\a.txt")
    home = project_logical_path("~/project/a.txt")
    assert relative is not None and relative.value == "relative/file.txt"
    assert posix is not None and posix.value == "workspace/a.txt"
    assert windows is not None and windows.value == "C/work/a.txt"
    assert home is not None and home.value == "home/project/a.txt"
    assert project_logical_path("src/**/*.py") is None
    assert project_logical_path("../secret.txt") is None
    assert project_logical_path("https://example.com/file.txt") is None
    assert project_logical_path("") is None


def test_builds_file_observations_from_structured_file_tools(tmp_path: Path) -> None:
    request = {
        "messages": [
            {"role": "user", "content": "inspect files"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "read", "name": "Read", "input": {"file_path": "input.txt"}},
                    {
                        "type": "tool_use",
                        "id": "write",
                        "name": "Write",
                        "input": {"file_path": "out.txt", "content": "hello"},
                    },
                    {
                        "type": "tool_use",
                        "id": "edit",
                        "name": "Edit",
                        "input": {"file_path": "out.txt", "old_string": "h", "new_string": "H"},
                    },
                    {
                        "type": "tool_use",
                        "id": "glob",
                        "name": "Glob",
                        "input": {"pattern": "src/**/*.py"},
                    },
                    {"type": "tool_use", "id": "grep", "name": "Grep", "input": {"pattern": "x"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "read", "content": "file text"},
                    {"type": "tool_result", "tool_use_id": "write", "content": "ok"},
                    {"type": "tool_result", "tool_use_id": "edit", "content": "ok"},
                    {"type": "tool_result", "tool_use_id": "glob", "content": ["src/a.py"]},
                    {"type": "tool_result", "tool_use_id": "grep", "content": "match"},
                ],
            },
        ]
    }
    source = tmp_path / "files.jsonl"
    _write_outer(source, _outer(request=json.dumps(request)))

    result = _index(source, tmp_path)

    observed = {(item.logical_path, item.operation): item for item in result.file_observations}
    assert set(observed) == {
        ("input.txt", FileOperation.READ),
        ("out.txt", FileOperation.WRITE),
        ("out.txt", FileOperation.EDIT),
    }
    assert observed[("input.txt", FileOperation.READ)].completeness is Completeness.PARTIAL
    assert observed[("out.txt", FileOperation.WRITE)].completeness is Completeness.COMPLETE
    assert observed[("out.txt", FileOperation.EDIT)].completeness is Completeness.COMPLETE
    assert any(item.code == "file-observation-path-unsafe" for item in result.diagnostics)
    for observation in result.file_observations:
        assert observation.source_event_refs
        assert observation.raw_path_ref.object_id.startswith("content://sha256/")
        if observation.content_ref is not None:
            assert observation.content_sha256 == observation.content_ref.object_sha256
        assert observation.file_version_id.startswith("file-version://sha256/")
    assert all(
        diagnostic.code != "file-observation-ineligible-tool-family"
        for diagnostic in result.diagnostics
        if diagnostic.attributes
    )


def test_skips_missing_and_unsafe_paths_without_guessing(tmp_path: Path) -> None:
    request = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "missing", "name": "Read", "input": {"limit": 10}},
                    {
                        "type": "tool_use",
                        "id": "unsafe",
                        "name": "Write",
                        "input": {"file_path": "../secret"},
                    },
                    {
                        "type": "tool_use",
                        "id": "shell",
                        "name": "Bash",
                        "input": {"command": "cat output.txt"},
                    },
                ],
            }
        ]
    }
    source = tmp_path / "unsafe.jsonl"
    _write_outer(source, _outer(request=json.dumps(request)))

    result = _index(source, tmp_path)

    assert result.file_observations == ()
    assert Counter(item.code for item in result.diagnostics) == Counter(
        {
            "file-observation-path-missing": 1,
            "file-observation-path-unsafe": 1,
        }
    )


def test_segments_cover_user_tool_file_and_context_windows(tmp_path: Path) -> None:
    request = {
        "messages": [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "read it"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "ok"},
                    {"type": "tool_use", "id": "read", "name": "Read", "input": {"file_path": "input.txt"}},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "read", "content": "text"}]},
            {"role": "assistant", "content": [{"type": "continuation_marker", "reason": "more"}]},
        ]
    }
    source = tmp_path / "segments.jsonl"
    _write_outer(source, _outer(request=json.dumps(request)))

    result = _index(source, tmp_path)

    methods = Counter(segment.boundary_method for segment in result.interaction_segments)
    assert methods["user_turn"] == 1
    assert methods["tool_window"] == 1
    assert methods["file_operation"] == 1
    assert methods["continuation"] == 1
    assert methods["context"] >= 1
    event_refs = {event.event_id for event in result.normalization_result.events}
    for segment in result.interaction_segments:
        assert segment.sequence_start <= segment.sequence_end
        assert segment.member_event_refs
        assert {ref.object_id for ref in segment.member_event_refs} <= event_refs


def test_index_ids_ignore_audit_time(tmp_path: Path) -> None:
    request = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "read", "name": "Read", "input": {"file_path": "input.txt"}},
                ],
            }
        ]
    }
    source = tmp_path / "audit.jsonl"
    _write_outer(source, _outer(request=json.dumps(request)))
    normalized = _normalize(
        source,
        tmp_path,
        source_trace_id="source-trace://audit",
        audit=_audit(datetime(2026, 7, 22, tzinfo=UTC)),
    )
    first = TraceIndexBuilder().build(
        normalization_result=normalized,
        audit=_audit(datetime(2026, 7, 22, tzinfo=UTC)),
    )
    second = TraceIndexBuilder().build(
        normalization_result=normalized,
        audit=_audit(datetime(2026, 7, 23, tzinfo=UTC)),
    )

    assert [item.observation_id for item in first.file_observations] == [
        item.observation_id for item in second.file_observations
    ]
    assert [item.segment_id for item in first.interaction_segments] == [
        item.segment_id for item in second.interaction_segments
    ]


def test_local_corpus_observations_match_structured_eligible_file_tools(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    paths = sorted((root.parent / "raw_traj").glob("*.jsonl"))
    if len(paths) != 91:
        pytest.skip("complete local raw_traj corpus is unavailable")
    registry = TraceSourceRegistry(tmp_path / "corpus.sqlite3")
    parser = RawTrajV1Parser()
    recovery = RawTrajRecovery()
    normalizer = RawTrajV1Normalizer()
    builder = TraceIndexBuilder()
    eligible = 0
    observed = 0

    for index, source in enumerate(paths):
        audit = _audit()
        registered = registry.register(
            source,
            source_trace_id=f"source-trace://indexing-corpus-{index:03d}",
            source_uri=f"raw-traj://indexing-corpus-{index:03d}",
        )
        parsed = parser.parse(source, registered_source=registered, audit=audit)
        recovered = recovery.recover(source, parse_result=parsed, audit=audit)
        normalized = normalizer.normalize(source, recovery_result=recovered, audit=audit)
        result = builder.build(normalization_result=normalized, audit=audit)
        eligible += sum(
            1
            for record in normalized.tool_call_records
            if record.tool_family
            in {
                ToolFamily.FILE_READ,
                ToolFamily.FILE_WRITE,
                ToolFamily.FILE_EDIT,
                ToolFamily.FILE_LIST,
            }
        )
        observed += len(result.file_observations)

    assert observed > 0
    assert observed <= eligible
