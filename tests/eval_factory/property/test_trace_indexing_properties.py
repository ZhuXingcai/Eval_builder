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
    TraceIndexBuilder,
    TraceSourceRegistry,
)


def _outer(request: str) -> dict[str, object]:
    return {
        "account": "synthetic",
        "api_type": "chat",
        "business": "IndexingProperty",
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
        created_by="trace-indexing-property-test",
        governing_versions=(VersionBinding(component="raw-traj-file-observation", version="v1"),),
    )


def test_indexing_is_stable_across_python_hash_seeds(tmp_path: Path) -> None:
    source = tmp_path / "hash-seed.jsonl"
    _write_outer(
        source,
        {
            "messages": [
                {"role": "user", "content": "read"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "call-1", "name": "Read", "input": {"file_path": "a.txt"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "ok"}],
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
from eval_factory.trace import RawTrajRecovery, RawTrajV1Normalizer, RawTrajV1Parser, TraceIndexBuilder, TraceSourceRegistry

source = Path(sys.argv[1])
registry = TraceSourceRegistry(Path(sys.argv[2]))
registered = registry.register(
    source,
    source_trace_id="source-trace://indexing-hash-seed",
    source_uri="raw-traj://indexing-hash-seed",
)
audit = ContractAudit(
    created_at=datetime(2026, 7, 22, tzinfo=UTC),
    created_by="trace-indexing-property-test",
    governing_versions=(VersionBinding(component="raw-traj-file-observation", version="v1"),),
)
parsed = RawTrajV1Parser().parse(source, registered_source=registered, audit=audit)
recovered = RawTrajRecovery().recover(source, parse_result=parsed, audit=audit)
normalized = RawTrajV1Normalizer().normalize(source, recovery_result=recovered, audit=audit)
result = TraceIndexBuilder().build(normalization_result=normalized, audit=audit)
print(json.dumps({
    "observations": [[item.sequence, item.observation_id, item.file_version_id] for item in result.file_observations],
    "segments": [[item.sequence_start, item.sequence_end, item.boundary_method, item.segment_id] for item in result.interaction_segments],
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


def test_index_contract_refs_are_self_consistent(tmp_path: Path) -> None:
    source = tmp_path / "refs.jsonl"
    _write_outer(
        source,
        {
            "messages": [
                {"role": "user", "content": "write"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call-1",
                            "name": "Write",
                            "input": {"file_path": "a.txt", "content": "hello"},
                        },
                    ],
                },
            ]
        },
    )
    registry = TraceSourceRegistry(tmp_path / "registry.sqlite3")
    registered = registry.register(
        source, source_trace_id="source-trace://refs", source_uri="raw-traj://refs"
    )
    audit = _audit()
    parsed = RawTrajV1Parser().parse(source, registered_source=registered, audit=audit)
    recovered = RawTrajRecovery().recover(source, parse_result=parsed, audit=audit)
    normalized = RawTrajV1Normalizer().normalize(source, recovery_result=recovered, audit=audit)
    result = TraceIndexBuilder().build(normalization_result=normalized, audit=audit)

    events = {event.event_id: event for event in result.normalization_result.events}
    blobs = {blob.content_ref.object_id: blob for blob in result.normalization_result.content_blobs}
    for observation in result.file_observations:
        for ref in observation.source_event_refs:
            ref.require_matching_hash(events[ref.object_id])
        assert observation.raw_path_ref.object_id in blobs
        if observation.content_ref is not None:
            assert observation.content_ref.object_id in blobs
    for segment in result.interaction_segments:
        for ref in segment.member_event_refs:
            ref.require_matching_hash(events[ref.object_id])
