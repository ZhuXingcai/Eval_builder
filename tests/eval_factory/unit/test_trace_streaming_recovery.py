from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.contracts.core import ContractAudit, VersionBinding
from eval_factory.contracts.trace import CapabilityStatus, ParseQuality
from eval_factory.trace import (
    AdmissionMode,
    RawTrajRecovery,
    RawTrajV1Parser,
    RecoveryFieldStatus,
    RecoveryUnitKind,
    TraceSourceRegistry,
    TraceUse,
    evaluate_admission,
)


def _outer(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "account": "synthetic",
        "api_type": "chat",
        "business": "StreamingGold",
        "endpoint": "offline",
        "event_time": "2026-07-21T00:00:00Z",
        "extra": '{"req_lost_number":0,"resp_lost_number":0}',
        "mm_urls": "",
        "model": "offline-model",
        "p_date": "2026-07-21",
        "request": '{"messages":[]}',
        "response": '{"content":[],"role":"assistant"}',
        "sid": "synthetic-session",
        "source": None,
    }
    value.update(overrides)
    return value


def _write_outer(path: Path, *records: dict[str, object]) -> bytes:
    raw = (
        "\n".join(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for record in records
        )
        + "\n"
    ).encode()
    path.write_bytes(raw)
    return raw


def _audit(
    created_at: datetime = datetime(2026, 7, 21, tzinfo=UTC),
) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="streaming-recovery-test",
        governing_versions=(
            VersionBinding(
                component="raw-traj-streaming-recovery",
                version="v1",
            ),
        ),
    )


def _recover(
    source: Path,
    tmp_path: Path,
    *,
    source_trace_id: str = "source-trace://streaming-test",
    audit: ContractAudit | None = None,
):
    registry = TraceSourceRegistry(tmp_path / "registry.sqlite3")
    registered = registry.register(
        source,
        source_trace_id=source_trace_id,
        source_uri=f"raw-traj://{source_trace_id.rsplit('/', 1)[-1]}",
    )
    active_audit = audit or _audit()
    parsed = RawTrajV1Parser().parse(
        source,
        registered_source=registered,
        audit=active_audit,
    )
    recovered = RawTrajRecovery().recover(
        source,
        parse_result=parsed,
        audit=active_audit,
    )
    return parsed, recovered


def _capability(result, name: str):
    return next(item for item in result.capabilities if item.capability == name)


def _units(result):
    return tuple(unit for record in result.records for field in record.fields for unit in field.units)


def test_complete_inputs_bypass_streaming_and_keep_complete_capabilities(
    tmp_path: Path,
) -> None:
    strict_source = tmp_path / "strict.jsonl"
    _write_outer(strict_source, _outer())
    repaired_source = tmp_path / "repaired.jsonl"
    _write_outer(
        repaired_source,
        _outer(request='{"messages":[{"role":"user","content":"C:\\qa"}]}'),
    )

    strict_parse, strict = _recover(
        strict_source,
        tmp_path,
        source_trace_id="source-trace://strict",
    )
    repaired_parse, repaired = _recover(
        repaired_source,
        tmp_path,
        source_trace_id="source-trace://repaired",
    )

    assert strict_parse.outcome.value == "STRICT"
    assert strict.parse_quality is ParseQuality.STRICT
    assert repaired_parse.outcome.value == "REPAIRED"
    assert repaired.parse_quality is ParseQuality.REPAIRED
    for result in (strict, repaired):
        assert _units(result) == ()
        assert result.unrecoverable_spans == ()
        assert {item.status for item in result.capabilities} == {CapabilityStatus.COMPLETE}
        assert all(
            field.status is RecoveryFieldStatus.COMPLETE
            for record in result.records
            for field in record.fields
        )


def test_streaming_gold_cases(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    gold = json.loads(
        (root / "evals/golden/eval_factory/parser/v1/streaming-cases.json").read_text(encoding="utf-8")
    )

    for index, case in enumerate(gold["cases"]):
        source = tmp_path / f"{case['case_id']}.jsonl"
        _write_outer(
            source,
            _outer(
                request=case["request"],
                response=case["response"],
                extra=case["extra"],
            ),
        )
        _, result = _recover(
            source,
            tmp_path,
            source_trace_id=f"source-trace://streaming-gold-{index}",
        )

        assert result.parse_quality.value == case["expected_quality"]
        assert len(_units(result)) == case["expected_units"]


def test_truncated_request_recovers_message_and_incomplete_message_blocks(
    tmp_path: Path,
) -> None:
    request = (
        '{"messages":['
        '{"role":"user","content":[{"type":"text","text":"first"}]},'
        '{"role":"assistant","content":['
        '{"type":"text","text":"second"},'
        '{"type":"tool_use","id":"call-1","name":"tool","input":{}},'
        '{"type":"text","text":"broken"'
    )
    source = tmp_path / "request.jsonl"
    raw = _write_outer(source, _outer(request=request))

    _, result = _recover(source, tmp_path)

    units = _units(result)
    assert tuple(item.kind for item in units) == (
        RecoveryUnitKind.MESSAGE,
        RecoveryUnitKind.CONTENT_BLOCK,
        RecoveryUnitKind.CONTENT_BLOCK,
    )
    assert units[1].observed_role == "assistant"
    assert units[2].observed_role == "assistant"
    assert _capability(result, "conversation_events").status is CapabilityStatus.PARTIAL
    assert _capability(result, "final_response").status is CapabilityStatus.COMPLETE
    assert result.parse_quality is ParseQuality.PARTIAL
    assert result.unrecoverable_spans
    for unit in units:
        assert raw[unit.source_span.raw_byte_start : unit.source_span.raw_byte_end]
    _assert_non_overlapping(result)


def test_truncated_response_recovers_complete_content_blocks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "response.jsonl"
    _write_outer(
        source,
        _outer(response=('{"content":[{"type":"text","text":"complete"},{"type":"text","text":"broken"')),
    )

    _, result = _recover(source, tmp_path)

    units = _units(result)
    assert len(units) == 1
    assert units[0].kind is RecoveryUnitKind.CONTENT_BLOCK
    assert units[0].observed_role == "assistant"
    assert _capability(result, "conversation_events").status is CapabilityStatus.COMPLETE
    assert _capability(result, "final_response").status is CapabilityStatus.PARTIAL


def test_recovered_unit_with_invalid_backslash_has_repair_map(
    tmp_path: Path,
) -> None:
    source = tmp_path / "repair.jsonl"
    raw = _write_outer(
        source,
        _outer(request='{"messages":[{"role":"user","content":"C:\\qa"}]'),
    )

    _, result = _recover(source, tmp_path)

    unit = _units(result)[0]
    assert unit.repair_map is not None
    assert unit.source_span.approximate is True
    assert unit.source_span.repair_map_ref is not None
    unit.source_span.repair_map_ref.require_matching_hash(unit.repair_map)
    assert len(unit.repair_map.segments) == 1
    segment = unit.repair_map.segments[0]
    assert raw[segment.raw_byte_start : segment.raw_byte_end] == b"\\\\"
    assert result.unrecoverable_spans
    missing_delimiter = result.unrecoverable_spans[-1]
    assert missing_delimiter.raw_byte_start == missing_delimiter.raw_byte_end


def test_balanced_invalid_and_scalar_candidates_are_not_emitted(
    tmp_path: Path,
) -> None:
    source = tmp_path / "invalid-candidates.jsonl"
    _write_outer(
        source,
        _outer(request='{"messages":[{"role":},42]'),
    )

    _, result = _recover(source, tmp_path)

    assert _units(result) == ()
    assert _capability(result, "conversation_events").status is CapabilityStatus.MISSING
    assert result.unrecoverable_spans


def test_extra_metadata_alone_cannot_prevent_unparseable(tmp_path: Path) -> None:
    source = tmp_path / "unparseable.jsonl"
    _write_outer(
        source,
        _outer(
            request=('{"messages":[{"role":"user","content":[{"type":"text","text":"broken'),
            response='{"content":[{"type":"text","text":"broken',
        ),
    )

    _, result = _recover(source, tmp_path)

    assert result.parse_quality is ParseQuality.UNPARSEABLE
    assert _units(result) == ()
    assert {item.status for item in result.capabilities} == {CapabilityStatus.MISSING}
    for use in TraceUse:
        decision = evaluate_admission(
            result,
            use=use,
            required_capabilities=("conversation_events",),
        )
        assert decision.mode is AdmissionMode.BLOCKED
        assert decision.reason_codes == ("trace-unparseable",)


def test_repair_map_remains_resolvable_when_no_unit_is_recovered(
    tmp_path: Path,
) -> None:
    source = tmp_path / "repair-no-unit.jsonl"
    _write_outer(
        source,
        _outer(
            request='{"messages":[{"role":"user","content":"C:\\qa',
        ),
    )

    _, result = _recover(source, tmp_path)

    request = result.records[0].fields[0]
    assert request.status is RecoveryFieldStatus.MISSING
    assert request.units == ()
    assert request.repair_map is not None
    assert result.recovery_maps == (request.repair_map,)
    assert request.unrecoverable_spans
    for span in request.unrecoverable_spans:
        assert span.repair_map_ref is not None
        span.repair_map_ref.require_matching_hash(request.repair_map)


@pytest.mark.parametrize(
    "extra",
    [
        '{"req_lost_number":1,"resp_lost_number":0}',
        '{"resp_lost_number":0}',
        '{"req_lost_number":"0","resp_lost_number":0}',
        '{"req_lost_number":-1,"resp_lost_number":0}',
        '{"req_lost_number":0',
    ],
)
def test_nonzero_or_invalid_loss_metadata_prevents_complete_request_capability(
    tmp_path: Path,
    extra: str,
) -> None:
    source = tmp_path / "loss.jsonl"
    _write_outer(source, _outer(extra=extra))

    _, result = _recover(source, tmp_path)

    conversation = _capability(result, "conversation_events")
    assert conversation.status is CapabilityStatus.PARTIAL
    assert conversation.reason_codes
    assert result.parse_quality is ParseQuality.PARTIAL


def test_partial_capability_admission_is_positive_only(tmp_path: Path) -> None:
    source = tmp_path / "admission.jsonl"
    _write_outer(
        source,
        _outer(request='{"messages":[{"role":"user","content":"complete"}]'),
    )
    _, result = _recover(source, tmp_path)

    positive = evaluate_admission(
        result,
        use=TraceUse.OBSERVED_POSITIVE_FACT,
        required_capabilities=("conversation_events",),
    )
    negative = evaluate_admission(
        result,
        use=TraceUse.NEGATIVE_PREDICATE,
        required_capabilities=("conversation_events",),
    )
    stage = evaluate_admission(
        result,
        use=TraceUse.STAGE_REQUIRED,
        required_capabilities=("conversation_events",),
    )
    unrelated_complete = evaluate_admission(
        result,
        use=TraceUse.NEGATIVE_PREDICATE,
        required_capabilities=("final_response",),
    )

    assert positive.mode is AdmissionMode.POSITIVE_ONLY
    assert negative.mode is AdmissionMode.BLOCKED
    assert stage.mode is AdmissionMode.BLOCKED
    assert unrelated_complete.mode is AdmissionMode.FULL


def test_recovery_ids_ignore_audit_time(tmp_path: Path) -> None:
    source = tmp_path / "audit-time.jsonl"
    _write_outer(
        source,
        _outer(request='{"messages":[{"role":"user","content":"C:\\qa"}]'),
    )
    registry = TraceSourceRegistry(tmp_path / "registry.sqlite3")
    registered = registry.register(
        source,
        source_trace_id="source-trace://recovery-audit-time",
    )
    parser = RawTrajV1Parser()
    first_audit = _audit(datetime(2026, 7, 21, tzinfo=UTC))
    second_audit = _audit(datetime(2026, 7, 22, tzinfo=UTC))
    first_parse = parser.parse(
        source,
        registered_source=registered,
        audit=first_audit,
    )
    second_parse = parser.parse(
        source,
        registered_source=registered,
        audit=second_audit,
    )

    first = RawTrajRecovery().recover(
        source,
        parse_result=first_parse,
        audit=first_audit,
    )
    second = RawTrajRecovery().recover(
        source,
        parse_result=second_parse,
        audit=second_audit,
    )

    assert tuple(item.unit_id for item in _units(first)) == tuple(item.unit_id for item in _units(second))
    assert tuple(item.source_span.span_id for item in _units(first)) == tuple(
        item.source_span.span_id for item in _units(second)
    )
    assert first.recovery_maps[0].repair_map_id == second.recovery_maps[0].repair_map_id
    assert first.recovery_maps[0].canonical_sha256() != second.recovery_maps[0].canonical_sha256()


def test_recovery_is_stable_across_python_hash_seeds(tmp_path: Path) -> None:
    source = tmp_path / "hash-seed.jsonl"
    _write_outer(
        source,
        _outer(request='{"messages":[{"role":"user","content":"C:\\qa"}]'),
    )
    script = """
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from eval_factory.contracts.core import ContractAudit, VersionBinding
from eval_factory.trace import RawTrajRecovery, RawTrajV1Parser, TraceSourceRegistry

source = Path(sys.argv[1])
registry = TraceSourceRegistry(Path(sys.argv[2]))
registered = registry.register(
    source,
    source_trace_id="source-trace://hash-seed",
    source_uri="raw-traj://hash-seed",
)
audit = ContractAudit(
    created_at=datetime(2026, 7, 21, tzinfo=UTC),
    created_by="hash-seed-test",
    governing_versions=(
        VersionBinding(component="raw-traj-streaming-recovery", version="v1"),
    ),
)
parsed = RawTrajV1Parser().parse(source, registered_source=registered, audit=audit)
result = RawTrajRecovery().recover(source, parse_result=parsed, audit=audit)
units = [
    unit
    for record in result.records
    for field in record.fields
    for unit in field.units
]
print(json.dumps({
    "quality": result.parse_quality,
    "unit_ids": [unit.unit_id for unit in units],
    "span_ids": [unit.source_span.span_id for unit in units],
    "repair_ids": [item.repair_map_id for item in result.recovery_maps],
    "capabilities": [
        [item.capability, item.status, list(item.reason_codes)]
        for item in result.capabilities
    ],
}, sort_keys=True))
"""
    outputs: set[bytes] = set()
    for seed in ("1", "2", "77"):
        environment = {**os.environ, "PYTHONHASHSEED": seed}
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
                env=environment,
            )
        )

    assert len(outputs) == 1


def test_local_corpus_bypasses_streaming_with_complete_capabilities(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[3]
    paths = sorted((root.parent / "raw_traj").glob("*.jsonl"))
    if len(paths) != 91:
        pytest.skip("complete local raw_traj corpus is unavailable")
    registry = TraceSourceRegistry(tmp_path / "corpus.sqlite3")
    parser = RawTrajV1Parser()
    recovery = RawTrajRecovery()

    for index, source in enumerate(paths):
        registered = registry.register(
            source,
            source_trace_id=f"source-trace://streaming-corpus-{index:03d}",
            source_uri=f"raw-traj://streaming-corpus-{index:03d}",
        )
        parsed = parser.parse(
            source,
            registered_source=registered,
            audit=_audit(),
        )
        result = recovery.recover(
            source,
            parse_result=parsed,
            audit=_audit(),
        )
        assert result.parse_quality in {ParseQuality.STRICT, ParseQuality.REPAIRED}
        assert _units(result) == ()
        assert result.unrecoverable_spans == ()
        assert {item.status for item in result.capabilities} == {CapabilityStatus.COMPLETE}


def _assert_non_overlapping(result) -> None:
    recovered = sorted(
        (unit.source_span.raw_byte_start, unit.source_span.raw_byte_end) for unit in _units(result)
    )
    unrecoverable = sorted((span.raw_byte_start, span.raw_byte_end) for span in result.unrecoverable_spans)
    for recovered_start, recovered_end in recovered:
        for missing_start, missing_end in unrecoverable:
            assert recovered_end <= missing_start or recovered_start >= missing_end
