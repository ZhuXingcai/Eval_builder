from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from eval_factory.contracts.core import ContractAudit, VersionBinding
from eval_factory.trace import (
    RawTrajRecovery,
    RawTrajV1Normalizer,
    RawTrajV1Parser,
    TraceSourceRegistry,
)


def _outer(request: str) -> dict[str, object]:
    return {
        "account": "synthetic",
        "api_type": "chat",
        "business": "NormalizationProperty",
        "endpoint": "offline",
        "event_time": "2026-07-22T00:00:00Z",
        "extra": '{"req_lost_number":0,"resp_lost_number":0}',
        "mm_urls": "",
        "model": "offline-model",
        "p_date": "2026-07-22",
        "request": request,
        "response": '{"content":[],"role":"assistant"}',
        "sid": "synthetic-session",
        "source": None,
    }


def _write_outer(path: Path, request: dict[str, object]) -> None:
    path.write_text(
        json.dumps(
            _outer(json.dumps(request, sort_keys=True, separators=(",", ":"))),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 7, 22, tzinfo=UTC),
        created_by="trace-normalization-property-test",
        governing_versions=(VersionBinding(component="raw-traj-event-normalization", version="v1"),),
    )


def test_normalization_is_stable_across_python_hash_seeds(tmp_path: Path) -> None:
    source = tmp_path / "hash-seed.jsonl"
    _write_outer(
        source,
        {
            "messages": [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "call-1", "name": "Read", "input": {"path": "a.txt"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "call-1", "content": "ok"},
                    ],
                },
            ]
        },
    )
    script = """
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from eval_factory.contracts.core import ContractAudit, VersionBinding
from eval_factory.trace import RawTrajRecovery, RawTrajV1Normalizer, RawTrajV1Parser, TraceSourceRegistry

source = Path(sys.argv[1])
registry = TraceSourceRegistry(Path(sys.argv[2]))
registered = registry.register(
    source,
    source_trace_id="source-trace://normalization-hash-seed",
    source_uri="raw-traj://normalization-hash-seed",
)
audit = ContractAudit(
    created_at=datetime(2026, 7, 22, tzinfo=UTC),
    created_by="trace-normalization-property-test",
    governing_versions=(VersionBinding(component="raw-traj-event-normalization", version="v1"),),
)
parsed = RawTrajV1Parser().parse(source, registered_source=registered, audit=audit)
recovered = RawTrajRecovery().recover(source, parse_result=parsed, audit=audit)
result = RawTrajV1Normalizer().normalize(source, recovery_result=recovered, audit=audit)
print(json.dumps({
    "trace_ir_version_id": result.trace_ir_version_id,
    "events": [[event.sequence, event.event_id, event.event_type] for event in result.events],
    "records": [[record.sequence, record.tool_call_record_id, record.status] for record in result.tool_call_records],
    "content": [blob.content_ref.object_sha256 for blob in result.content_blobs],
}, sort_keys=True))
"""
    outputs: set[bytes] = set()
    for seed in ("1", "2", "77"):
        outputs.add(
            subprocess.check_output(
                [
                    sys.executable,
                    "-c",
                    script,
                    str(source),
                    str(tmp_path / f"registry-{seed}.sqlite3"),
                ],
                cwd=Path(__file__).resolve().parents[3],
                env={**os.environ, "PYTHONHASHSEED": seed},
            )
        )

    assert len(outputs) == 1


def test_object_refs_resolve_to_matching_payloads(tmp_path: Path) -> None:
    source = tmp_path / "refs.jsonl"
    _write_outer(
        source,
        {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "call-1", "name": "Read", "input": {"path": "a.txt"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "call-1", "content": "ok"},
                    ],
                },
            ]
        },
    )
    registry = TraceSourceRegistry(tmp_path / "registry.sqlite3")
    registered = registry.register(
        source,
        source_trace_id="source-trace://refs",
        source_uri="raw-traj://refs",
    )
    audit = _audit()
    parsed = RawTrajV1Parser().parse(source, registered_source=registered, audit=audit)
    recovered = RawTrajRecovery().recover(source, parse_result=parsed, audit=audit)
    result = RawTrajV1Normalizer().normalize(source, recovery_result=recovered, audit=audit)

    spans = {span.span_id: span for span in result.source_spans}
    blobs = {blob.content_ref.object_id: blob for blob in result.content_blobs}
    events = {event.event_id: event for event in result.events}
    for event in result.events:
        for ref in event.source_spans:
            ref.require_matching_hash(spans[ref.object_id])
        if event.content_ref is not None:
            assert blobs[event.content_ref.object_id].content_ref == event.content_ref
    for record in result.tool_call_records:
        if record.call_event_ref is not None:
            record.call_event_ref.require_matching_hash(events[record.call_event_ref.object_id])
        if record.result_event_ref is not None:
            record.result_event_ref.require_matching_hash(events[record.result_event_ref.object_id])
