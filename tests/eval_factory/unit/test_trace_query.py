from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.contracts.core import ContractAudit, VersionBinding
from eval_factory.trace import (
    RawTrajRecovery,
    RawTrajV1Normalizer,
    RawTrajV1Parser,
    TraceIndexBuilder,
    TraceIndexStore,
    TraceSourceRegistry,
)
from eval_factory.trace.query import (
    QueryAuthorizationError,
    QueryBudgetExceededError,
    QueryKind,
    TraceEvidenceRequest,
    TraceQueryRequest,
    TraceQueryService,
    TrustedTracePrincipal,
)


def _outer(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "account": "synthetic",
        "api_type": "chat",
        "business": "QueryGold",
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


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 7, 22, tzinfo=UTC),
        created_by="trace-query-test",
        governing_versions=(VersionBinding(component="trace-query-service", version="v1"),),
    )


def _store_fixture(tmp_path: Path) -> tuple[TraceIndexStore, str]:
    request = {
        "messages": [
            {"role": "system", "content": "query policy"},
            {"role": "user", "content": "NeedleGamma inspect input.txt"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Reading input."},
                    {"type": "tool_use", "id": "read-1", "name": "Read", "input": {"file_path": "input.txt"}},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "read-1", "content": "NeedleGamma body"}],
            },
        ]
    }
    source = tmp_path / "query.jsonl"
    source.write_text(
        json.dumps(_outer(request=json.dumps(request)), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    audit = _audit()
    registry = TraceSourceRegistry(tmp_path / "registry.sqlite3")
    registered = registry.register(
        source,
        source_trace_id="source-trace://query-test",
        source_uri="raw-traj://query-test",
    )
    parsed = RawTrajV1Parser().parse(source, registered_source=registered, audit=audit)
    recovered = RawTrajRecovery().recover(source, parse_result=parsed, audit=audit)
    normalized = RawTrajV1Normalizer().normalize(source, recovery_result=recovered, audit=audit)
    indexed = TraceIndexBuilder().build(normalization_result=normalized, audit=audit)
    store = TraceIndexStore(tmp_path / "store")
    store.persist(indexed, audit=audit)
    return store, indexed.normalization_result.trace_ir_version_id


def _principal(trace_id: str, *, audit: bool = False) -> TrustedTracePrincipal:
    return TrustedTracePrincipal(
        principal_id="trace-principal://query-test",
        consumer_stage="label",
        consumer_agent="structured-labeler",
        allowed_purposes=frozenset({"labeling", "audit"}),
        allowed_trace_ir_version_ids=frozenset({trace_id}),
        max_events=10,
        max_characters=80,
        audit=audit,
    )


def _request(
    trace_id: str,
    kind: QueryKind,
    *,
    filters: dict[str, object] | None = None,
    limit: int = 10,
    max_characters: int = 80,
    principal: TrustedTracePrincipal | None = None,
) -> TraceQueryRequest:
    active = principal or _principal(trace_id)
    return TraceQueryRequest(
        principal=active,
        trace_ir_version_id=trace_id,
        purpose="labeling",
        query_kind=kind,
        filters=filters or {},
        limit=limit,
        max_characters=max_characters,
    )


def test_authorization_rejects_forged_scope_purpose_budget_and_taint(tmp_path: Path) -> None:
    store, trace_id = _store_fixture(tmp_path)
    service = TraceQueryService(store)
    principal = _principal(trace_id)

    with pytest.raises(QueryAuthorizationError, match="purpose"):
        service.query(
            TraceQueryRequest(
                principal=principal,
                trace_ir_version_id=trace_id,
                purpose="task-authoring",
                query_kind=QueryKind.MANIFEST,
                filters={},
                limit=1,
                max_characters=10,
            )
        )
    with pytest.raises(QueryAuthorizationError, match="trace scope"):
        service.query(
            TraceQueryRequest(
                principal=principal,
                trace_ir_version_id="trace-ir://other",
                purpose="labeling",
                query_kind=QueryKind.MANIFEST,
                filters={},
                limit=1,
                max_characters=10,
            )
        )
    with pytest.raises(QueryBudgetExceededError, match="event limit"):
        service.query(_request(trace_id, QueryKind.EVENTS, limit=11))
    with pytest.raises(QueryBudgetExceededError, match="character budget"):
        service.query(_request(trace_id, QueryKind.EVENTS, max_characters=81))
    with pytest.raises(QueryAuthorizationError, match="tainted"):
        service.query(
            TraceQueryRequest(
                principal=principal,
                trace_ir_version_id=trace_id,
                purpose="labeling",
                query_kind=QueryKind.EVENTS,
                filters={},
                limit=1,
                max_characters=10,
                allow_tainted=True,
            )
        )


def test_queries_manifest_events_span_tool_file_segment_and_text_search(tmp_path: Path) -> None:
    store, trace_id = _store_fixture(tmp_path)
    service = TraceQueryService(store)
    manifest = service.query(_request(trace_id, QueryKind.MANIFEST))
    events = service.query(_request(trace_id, QueryKind.EVENTS, filters={"event_type": "user_text"}))
    event = service.query(
        _request(trace_id, QueryKind.EVENT, filters={"event_id": events.result_refs[0].object_id})
    )
    span = service.query(
        _request(trace_id, QueryKind.SOURCE_SPAN, filters={"span_id": events.source_span_refs[0].span_id})
    )
    tools = service.query(_request(trace_id, QueryKind.TOOL_CALLS, filters={"tool_family": "file_read"}))
    files = service.query(
        _request(trace_id, QueryKind.FILE_OBSERVATIONS, filters={"logical_path": "input.txt"})
    )
    segments = service.query(_request(trace_id, QueryKind.SEGMENTS, filters={"boundary_method": "user_turn"}))
    search = service.query(_request(trace_id, QueryKind.TEXT_SEARCH, filters={"text": "NeedleGamma"}))

    assert manifest.result_refs[0].object_type == "trace-envelope"
    assert [item.object_type for item in event.result_refs] == ["trace-event"]
    assert span.source_span_refs[0].span_id == events.source_span_refs[0].span_id
    assert tools.result_refs and tools.result_refs[0].object_type == "tool-call-record"
    assert files.result_refs and files.result_refs[0].object_type == "file-observation"
    assert segments.result_refs and segments.result_refs[0].object_type == "interaction-segment"
    assert search.result_refs
    assert all(response.returned_characters <= response.max_characters for response in (events, search))
    assert events.consumer_stage == "label"
    assert events.consumer_agent == "structured-labeler"


def test_text_budget_truncates_without_exceeding_limit(tmp_path: Path) -> None:
    store, trace_id = _store_fixture(tmp_path)
    service = TraceQueryService(store)

    response = service.query(
        _request(trace_id, QueryKind.TEXT_SEARCH, filters={"text": "NeedleGamma"}, max_characters=5)
    )

    assert response.truncated is True
    assert response.returned_characters <= 5
    assert response.result_refs


def test_audit_principal_can_request_tainted_flag(tmp_path: Path) -> None:
    store, trace_id = _store_fixture(tmp_path)
    service = TraceQueryService(store)

    response = service.query(
        TraceQueryRequest(
            principal=_principal(trace_id, audit=True),
            trace_ir_version_id=trace_id,
            purpose="audit",
            query_kind=QueryKind.EVENTS,
            filters={},
            limit=2,
            max_characters=20,
            allow_tainted=True,
        )
    )

    assert response.audit is True
    assert response.result_refs


def test_evidence_bundle_binds_refs_spans_and_budget(tmp_path: Path) -> None:
    store, trace_id = _store_fixture(tmp_path)
    service = TraceQueryService(store)
    request = TraceEvidenceRequest(
        query=_request(trace_id, QueryKind.EVENTS, filters={"event_type": "user_text"}, max_characters=30)
    )

    bundle = service.evidence_bundle(request)

    assert bundle.trace_ir_version_id == trace_id
    assert bundle.consumer_stage == "label"
    assert bundle.purpose == "labeling"
    assert bundle.returned_characters <= bundle.max_characters
    assert bundle.evidence
    assert bundle.tainted_content_included is False
