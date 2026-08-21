from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.contracts.core import ContractAudit, VersionBinding
from eval_factory.contracts.trace import (
    CapabilityStatus,
    ParseQuality,
    ToolCallStatus,
)
from eval_factory.trace import (
    CuratedTrajectoryV1Adapter,
    CuratedTrajectoryV1Parser,
    CuratedTrajectoryV1Recovery,
    RawTrajV1Normalizer,
    TraceProbeStatus,
    TraceSourceRegistry,
)


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 13, tzinfo=UTC),
        created_by="curated-trajectory-adapter-test",
        governing_versions=(
            VersionBinding(
                component="curated-trajectory-adapter",
                version="curated-trajectory/v1",
            ),
        ),
    )


def _trajectory() -> dict[str, object]:
    return {
        "system": "trusted runtime context",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "inspect failure"}],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "powershell-1",
                        "name": "PowerShell",
                        "input": {"command": "Get-Item missing"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "powershell-1",
                        "is_error": True,
                        "content": "structured failure",
                    }
                ],
            },
        ],
    }


def _write_trajectory(path: Path) -> bytes:
    raw = json.dumps(
        _trajectory(),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    path.write_bytes(raw)
    return raw


def test_curated_trajectory_probe_preserves_exact_logical_member_bytes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "trajectory.json"
    raw = _write_trajectory(source)

    result = CuratedTrajectoryV1Adapter().probe(source)

    assert result.status is TraceProbeStatus.SUPPORTED
    assert result.raw_sha256 == hashlib.sha256(raw).hexdigest()
    assert result.size_bytes == len(raw)
    assert result.record_count == 1
    assert result.outer_fields == ("messages", "system")
    assert source.read_bytes() == raw


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"messages": []},
        {"system": 1, "messages": []},
        {"system": "context", "messages": "not-a-list"},
        {"system": "context", "messages": [], "extra": "closed"},
        {"system": "context", "messages": [{"role": "user"}]},
    ],
)
def test_curated_trajectory_probe_rejects_invalid_shapes(
    tmp_path: Path,
    payload: object,
) -> None:
    source = tmp_path / "trajectory.json"
    source.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    result = CuratedTrajectoryV1Adapter().probe(source)

    assert result.status is TraceProbeStatus.INVALID
    assert result.diagnostics


def test_curated_trajectory_probe_rejects_symlink(
    tmp_path: Path,
) -> None:
    source = tmp_path / "trajectory.json"
    _write_trajectory(source)
    link = tmp_path / "linked.json"
    link.symlink_to(source)

    result = CuratedTrajectoryV1Adapter().probe(link)

    assert result.status is TraceProbeStatus.UNSUPPORTED


def test_curated_trajectory_parse_and_normalize_keep_missing_final_typed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "trajectory.json"
    raw = _write_trajectory(source)
    adapter = CuratedTrajectoryV1Adapter()
    registered = TraceSourceRegistry(
        tmp_path / "registry.sqlite3",
        adapter=adapter,
    ).register(
        source,
        source_trace_id="source-trace://curated/example",
        source_uri="curated-trajectory://container/example",
    )

    parsed = CuratedTrajectoryV1Parser().parse(
        source,
        registered_source=registered,
        audit=_audit(),
    )
    recovered = CuratedTrajectoryV1Recovery().recover(
        source,
        parse_result=parsed,
        audit=_audit(),
    )
    normalized = RawTrajV1Normalizer(adapter=adapter).normalize(
        source,
        recovery_result=recovered,
        audit=_audit(),
    )

    capabilities = {capability.capability: capability.status for capability in recovered.capabilities}
    assert registered.source.raw_sha256 == hashlib.sha256(raw).hexdigest()
    assert recovered.parse_quality is ParseQuality.PARTIAL
    assert capabilities["call_result_pairing"] is CapabilityStatus.COMPLETE
    assert capabilities["final_response"] is CapabilityStatus.MISSING
    assert len(normalized.tool_call_records) == 1
    record = normalized.tool_call_records[0]
    assert record.status is ToolCallStatus.PAIRED_ERROR
    assert record.error_signature is not None
    assert record.error_signature.startswith("powershell:")
    assert record.error_signature.endswith(":paired_error")
    assert source.read_bytes() == raw
