from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from env_mock_agent.facade.execution_adapter import (
    MappingAttachmentExecutionMaterialResolver,
    RegistryAttachmentExecutionFacade,
)
from env_mock_agent.facade.telemetry_v2 import (
    ExecutionTelemetryV2,
    ReportedCostV2,
    TelemetryAvailabilityV2,
    ToolFamilyCountV2,
    ToolFamilyV2,
    execution_telemetry_ref,
    normalize_reported_cost_usd,
    normalize_tool_family_counts,
)
from env_mock_agent.providers import ProviderRegistry
from env_mock_agent.runtimes import (
    ClaudeCodeCliRuntime,
    PiRpcRuntime,
    RuntimeRegistry,
)
from env_mock_agent.schemas import (
    RuntimeEvent,
    RuntimeEventType,
    RuntimeName,
    RuntimeUsage,
)

NOW = datetime(2026, 8, 1, tzinfo=UTC)


def test_unavailable_telemetry_is_explicit_and_deterministic() -> None:
    first = ExecutionTelemetryV2.unavailable(observed_at=NOW)
    second = ExecutionTelemetryV2.unavailable(observed_at=NOW)

    assert first == second
    assert first.duration_us is None
    assert first.duration_availability is TelemetryAvailabilityV2.UNAVAILABLE
    assert first.reported_cost.amount_microusd is None
    assert first.reported_cost.availability is TelemetryAvailabilityV2.UNAVAILABLE
    assert execution_telemetry_ref(first).object_sha256 == first.execution_telemetry_sha256


def test_reported_zero_cost_differs_from_unavailable() -> None:
    reported = normalize_reported_cost_usd(0.0)
    unavailable = normalize_reported_cost_usd(None)

    assert reported == ReportedCostV2.reported(amount_microusd=0)
    assert reported.availability is TelemetryAvailabilityV2.REPORTED
    assert unavailable == ReportedCostV2.unavailable()
    assert reported != unavailable


def test_reported_cost_uses_half_even_micro_usd_rounding() -> None:
    assert normalize_reported_cost_usd(Decimal("1.2345675")).amount_microusd == 1_234_568
    assert normalize_reported_cost_usd("0.0000005").amount_microusd == 0
    assert normalize_reported_cost_usd("0.0000015").amount_microusd == 2


@pytest.mark.parametrize(
    "value",
    [
        True,
        -1,
        -0.1,
        float("nan"),
        float("inf"),
        "-Infinity",
        "not-a-number",
        "1e1000000",
    ],
)
def test_reported_cost_rejects_invalid_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        normalize_reported_cost_usd(value)


def test_tool_names_are_reduced_to_closed_families() -> None:
    counts = normalize_tool_family_counts(
        ("Read", "Write", "Bash", "WebSearch", "WebFetch", "browser_click", "unknown-provider-tool")
    )

    assert counts == (
        ToolFamilyCountV2(tool_family=ToolFamilyV2.FILE, calls=2),
        ToolFamilyCountV2(tool_family=ToolFamilyV2.SHELL, calls=1),
        ToolFamilyCountV2(tool_family=ToolFamilyV2.SEARCH, calls=1),
        ToolFamilyCountV2(tool_family=ToolFamilyV2.FETCH, calls=1),
        ToolFamilyCountV2(tool_family=ToolFamilyV2.RENDERER, calls=1),
        ToolFamilyCountV2(tool_family=ToolFamilyV2.OTHER, calls=1),
    )


def test_telemetry_is_strict_and_rejects_sensitive_extra_fields() -> None:
    telemetry = ExecutionTelemetryV2.create(
        duration_us=12_000,
        duration_availability=TelemetryAvailabilityV2.REPORTED,
        tool_family_counts=(ToolFamilyCountV2(tool_family=ToolFamilyV2.FILE, calls=2),),
        tool_availability=TelemetryAvailabilityV2.DERIVED,
        reported_cost=ReportedCostV2.reported(amount_microusd=123),
        observed_at=NOW,
    )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExecutionTelemetryV2.model_validate(
            {
                **telemetry.model_dump(mode="python"),
                "tool_arguments": {"path": "/private/file"},
            }
        )
    with pytest.raises(ValidationError, match="duration"):
        ExecutionTelemetryV2.model_validate(
            {
                **telemetry.model_dump(mode="python"),
                "duration_us": None,
            }
        )


def test_runtime_parsers_preserve_missing_and_reported_zero_cost() -> None:
    assert ClaudeCodeCliRuntime._usage({"usage": {}}, 0).cost_usd is None
    assert (
        ClaudeCodeCliRuntime._usage(
            {
                "usage": {},
                "total_cost_usd": None,
                "duration_ms": None,
            },
            0,
        ).cost_usd
        is None
    )
    assert (
        ClaudeCodeCliRuntime._usage(
            {
                "usage": {},
                "total_cost_usd": 0,
                "duration_ms": 0,
            },
            0,
        ).cost_usd
        == 0.0
    )
    assert PiRpcRuntime._usage({}).cost_usd is None
    assert PiRpcRuntime._usage({"costUsd": None}).cost_usd is None
    assert PiRpcRuntime._usage({"costUsd": 0}).cost_usd == 0.0
    with pytest.raises(ValueError, match="numeric"):
        ClaudeCodeCliRuntime._usage(
            {
                "usage": {},
                "total_cost_usd": "not-a-number",
            },
            0,
        )
    with pytest.raises(ValueError, match="numeric"):
        PiRpcRuntime._usage({"costUsd": False})


def test_runtime_events_compile_content_free_reported_telemetry(
    tmp_path: Path,
) -> None:
    facade = RegistryAttachmentExecutionFacade(
        ProviderRegistry(),
        RuntimeRegistry(),
        resolver=MappingAttachmentExecutionMaterialResolver(world_ledgers={}),
        staging_root=tmp_path,
        clock=lambda: NOW,
    )
    shared = {
        "run_id": "run-1",
        "artifact_id": "artifact-1",
        "runtime": RuntimeName.FAKE,
    }
    telemetry = facade._runtime_telemetry(
        [
            RuntimeEvent(
                **shared,
                event_type=RuntimeEventType.TOOL_STARTED,
                tool_name="Read",
            ),
            RuntimeEvent(
                **shared,
                event_type=RuntimeEventType.TOOL_STARTED,
                tool_name="WebSearch",
            ),
            RuntimeEvent(
                **shared,
                event_type=RuntimeEventType.USAGE,
                usage=RuntimeUsage(
                    cost_usd=0.0,
                    duration_ms=12,
                ),
            ),
        ]
    )

    assert telemetry.duration_us == 12_000
    assert telemetry.duration_availability is TelemetryAvailabilityV2.REPORTED
    assert telemetry.tool_family_counts == (
        ToolFamilyCountV2(tool_family=ToolFamilyV2.FILE, calls=1),
        ToolFamilyCountV2(tool_family=ToolFamilyV2.SEARCH, calls=1),
    )
    assert telemetry.reported_cost == ReportedCostV2.reported(amount_microusd=0)
    serialized = telemetry.model_dump_json()
    assert "Read" not in serialized
    assert "WebSearch" not in serialized
