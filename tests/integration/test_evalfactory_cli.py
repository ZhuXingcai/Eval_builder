from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from eval_factory.contracts.core import ContractAudit, VersionBinding
from eval_factory.trace import (
    RawTrajRecovery,
    RawTrajV1Normalizer,
    RawTrajV1Parser,
    TraceIndexBuilder,
    TraceIndexStore,
    TraceSourceRegistry,
)
from eval_factory.trace.query.cli import app


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 7, 22, tzinfo=UTC),
        created_by="evalfactory-cli-test",
        governing_versions=(VersionBinding(component="trace-query-service", version="v1"),),
    )


def _build_store(tmp_path: Path) -> tuple[Path, str, Path]:
    request = {
        "messages": [
            {"role": "user", "content": "NeedleCli"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "read-1", "name": "Read", "input": {"file_path": "cli.txt"}}
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "read-1", "content": "NeedleCli"}],
            },
        ]
    }
    outer = {
        "account": "synthetic",
        "api_type": "chat",
        "business": "QueryCli",
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
    source = tmp_path / "cli.jsonl"
    source.write_text(json.dumps(outer, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    audit = _audit()
    registered = TraceSourceRegistry(tmp_path / "registry.sqlite3").register(
        source,
        source_trace_id="source-trace://query-cli",
        source_uri="raw-traj://query-cli",
    )
    parsed = RawTrajV1Parser().parse(source, registered_source=registered, audit=audit)
    recovered = RawTrajRecovery().recover(source, parse_result=parsed, audit=audit)
    normalized = RawTrajV1Normalizer().normalize(source, recovery_result=recovered, audit=audit)
    indexed = TraceIndexBuilder().build(normalization_result=normalized, audit=audit)
    store_root = tmp_path / "store"
    TraceIndexStore(store_root).persist(indexed, audit=audit)
    trace_id = indexed.normalization_result.trace_ir_version_id
    principal_path = tmp_path / "principal.json"
    principal_path.write_text(
        json.dumps(
            {
                "principal_id": "trace-principal://cli",
                "consumer_stage": "label",
                "consumer_agent": "structured-labeler",
                "allowed_purposes": ["labeling"],
                "allowed_trace_ir_version_ids": [trace_id],
                "max_events": 10,
                "max_characters": 100,
                "audit": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return store_root, trace_id, principal_path


def test_evalfactory_trace_query_cli_outputs_json(tmp_path: Path) -> None:
    store, trace_id, principal = _build_store(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "trace",
            "query",
            "--store",
            str(store),
            "--principal",
            str(principal),
            "--trace",
            trace_id,
            "--kind",
            "events",
            "--purpose",
            "labeling",
            "--limit",
            "10",
            "--max-characters",
            "100",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["trace_ir_version_id"] == trace_id
    assert payload["result_refs"]
    assert payload["returned_characters"] <= 100


def test_evalfactory_trace_search_cli_rejects_unauthorized_purpose(tmp_path: Path) -> None:
    store, trace_id, principal = _build_store(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "trace",
            "search",
            "--store",
            str(store),
            "--principal",
            str(principal),
            "--trace",
            trace_id,
            "--purpose",
            "task-authoring",
            "--text",
            "NeedleCli",
            "--json",
        ],
    )

    assert result.exit_code != 0
    assert "purpose" in result.output
