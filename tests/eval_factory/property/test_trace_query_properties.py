from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_trace_query_results_are_stable_across_python_hash_seeds(tmp_path: Path) -> None:
    source = tmp_path / "query-property.jsonl"
    request = {
        "messages": [
            {"role": "user", "content": "NeedleDelta"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "read-1", "name": "Read", "input": {"file_path": "a.txt"}}
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "read-1", "content": "NeedleDelta"}],
            },
        ]
    }
    outer = {
        "account": "synthetic",
        "api_type": "chat",
        "business": "QueryProperty",
        "endpoint": "offline",
        "event_time": "2026-07-22T00:00:00Z",
        "extra": '{"req_lost_number":0,"resp_lost_number":0}',
        "mm_urls": "",
        "model": "offline-model",
        "p_date": "2026-07-22",
        "request": json.dumps(request, sort_keys=True, separators=(",", ":")),
        "response": '{"content":[],"role":"assistant"}',
        "sid": "synthetic-session",
        "source": None,
    }
    source.write_text(json.dumps(outer, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    script = """
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from eval_factory.contracts.core import ContractAudit, VersionBinding
from eval_factory.trace import RawTrajRecovery, RawTrajV1Normalizer, RawTrajV1Parser, TraceIndexBuilder, TraceIndexStore, TraceSourceRegistry
from eval_factory.trace.query import QueryKind, TraceQueryRequest, TraceQueryService, TrustedTracePrincipal

source = Path(sys.argv[1])
root = Path(sys.argv[2])
audit = ContractAudit(
    created_at=datetime(2026, 7, 22, tzinfo=UTC),
    created_by="trace-query-property-test",
    governing_versions=(VersionBinding(component="trace-query-service", version="v1"),),
)
registered = TraceSourceRegistry(root / "registry.sqlite3").register(
    source,
    source_trace_id="source-trace://query-property",
    source_uri="raw-traj://query-property",
)
parsed = RawTrajV1Parser().parse(source, registered_source=registered, audit=audit)
recovered = RawTrajRecovery().recover(source, parse_result=parsed, audit=audit)
normalized = RawTrajV1Normalizer().normalize(source, recovery_result=recovered, audit=audit)
indexed = TraceIndexBuilder().build(normalization_result=normalized, audit=audit)
store = TraceIndexStore(root / "store")
store.persist(indexed, audit=audit)
trace_id = indexed.normalization_result.trace_ir_version_id
principal = TrustedTracePrincipal(
    principal_id="trace-principal://query-property",
    consumer_stage="label",
    consumer_agent="structured-labeler",
    allowed_purposes=frozenset({"labeling"}),
    allowed_trace_ir_version_ids=frozenset({trace_id}),
    max_events=10,
    max_characters=100,
)
service = TraceQueryService(store)
request = TraceQueryRequest(
    principal=principal,
    trace_ir_version_id=trace_id,
    purpose="labeling",
    query_kind=QueryKind.TEXT_SEARCH,
    filters={"text": "NeedleDelta"},
    limit=10,
    max_characters=100,
)
response = service.query(request)
print(json.dumps({
    "refs": [(item.object_type, item.object_id, item.object_sha256) for item in response.result_refs],
    "spans": [(item.span_id, item.raw_sha256, item.approximate) for item in response.source_span_refs],
    "chars": response.returned_characters,
    "truncated": response.truncated,
}, sort_keys=True))
"""
    outputs: set[bytes] = set()
    for seed in ("1", "2", "77"):
        root = tmp_path / f"run-{seed}"
        root.mkdir()
        outputs.add(
            subprocess.check_output(
                [sys.executable, "-c", script, str(source), str(root)],
                cwd=Path(__file__).resolve().parents[3],
                env={**os.environ, "PYTHONHASHSEED": seed},
            )
        )

    assert len(outputs) == 1
