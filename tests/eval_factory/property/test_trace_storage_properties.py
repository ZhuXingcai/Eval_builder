from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_trace_storage_is_stable_across_python_hash_seeds(tmp_path: Path) -> None:
    source = tmp_path / "storage-hash-seed.jsonl"
    request = {
        "messages": [
            {"role": "user", "content": "NeedleBeta"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "read-1",
                        "name": "Read",
                        "input": {"file_path": "a.txt"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "read-1", "content": "NeedleBeta"}],
            },
        ]
    }
    outer = {
        "account": "synthetic",
        "api_type": "chat",
        "business": "StorageProperty",
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
    source.write_text(
        json.dumps(outer, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    script = """
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from eval_factory.contracts.core import ContractAudit, VersionBinding
from eval_factory.trace import RawTrajRecovery, RawTrajV1Normalizer, RawTrajV1Parser, TraceIndexBuilder, TraceSourceRegistry
from eval_factory.trace.storage import TraceIndexStore

source = Path(sys.argv[1])
root = Path(sys.argv[2])
audit = ContractAudit(
    created_at=datetime(2026, 7, 22, tzinfo=UTC),
    created_by="trace-storage-property-test",
    governing_versions=(VersionBinding(component="trace-index-storage", version="v1"),),
)
registered = TraceSourceRegistry(root / "registry.sqlite3").register(
    source,
    source_trace_id="source-trace://storage-property",
    source_uri="raw-traj://storage-property",
)
parsed = RawTrajV1Parser().parse(source, registered_source=registered, audit=audit)
recovered = RawTrajRecovery().recover(source, parse_result=parsed, audit=audit)
normalized = RawTrajV1Normalizer().normalize(source, recovery_result=recovered, audit=audit)
indexed = TraceIndexBuilder().build(normalization_result=normalized, audit=audit)
store = TraceIndexStore(root / "store")
commit = store.persist(indexed, audit=audit)
loaded = store.load(indexed.normalization_result.trace_ir_version_id)
rebuilt = store.rebuild_projection()
print(json.dumps({
    "batch": commit.batch_sha256,
    "events": loaded.event_summary(),
    "files": loaded.file_observation_summary(),
    "segments": loaded.segment_summary(),
    "search": [(item.object_id, item.object_sha256) for item in store.search_text(indexed.normalization_result.trace_ir_version_id, "NeedleBeta")],
    "rebuild": rebuilt.projection_sha256,
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
