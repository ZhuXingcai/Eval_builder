from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from env_mock_agent.facade.contracts import (
    FacadeModel,
    FacadeObjectRef,
    Identifier,
    Sha256,
)

EXECUTION_TELEMETRY_POLICY_VERSION: Literal["execution-telemetry/r6-06-v1"] = "execution-telemetry/r6-06-v1"
_MICRO_USD = Decimal("0.000001")
_MICRO_USD_FACTOR = Decimal(1_000_000)


class TelemetryAvailabilityV2(StrEnum):
    REPORTED = "REPORTED"
    DERIVED = "DERIVED"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ToolFamilyV2(StrEnum):
    FILE = "FILE"
    SHELL = "SHELL"
    SEARCH = "SEARCH"
    FETCH = "FETCH"
    RENDERER = "RENDERER"
    OTHER = "OTHER"


class ToolFamilyCountV2(FacadeModel):
    tool_family: ToolFamilyV2
    calls: int = Field(ge=1)


class ReportedCostV2(FacadeModel):
    availability: TelemetryAvailabilityV2
    currency: Literal["USD"] = "USD"
    amount_microusd: int | None = Field(default=None, ge=0)
    rounding: Literal["HALF_EVEN_MICRO_USD"] | None = None

    @model_validator(mode="after")
    def validate_cost(self) -> Self:
        reported = self.availability is TelemetryAvailabilityV2.REPORTED
        if reported != (self.amount_microusd is not None):
            raise ValueError("reported cost availability and amount_microusd disagree")
        if reported != (self.rounding is not None):
            raise ValueError("reported cost availability and rounding disagree")
        if self.availability not in {
            TelemetryAvailabilityV2.REPORTED,
            TelemetryAvailabilityV2.UNAVAILABLE,
        }:
            raise ValueError("cost availability must be REPORTED or UNAVAILABLE")
        return self

    @classmethod
    def reported(cls, *, amount_microusd: int) -> ReportedCostV2:
        return cls(
            availability=TelemetryAvailabilityV2.REPORTED,
            amount_microusd=amount_microusd,
            rounding="HALF_EVEN_MICRO_USD",
        )

    @classmethod
    def unavailable(cls) -> ReportedCostV2:
        return cls(availability=TelemetryAvailabilityV2.UNAVAILABLE)


class ExecutionTelemetryV2(FacadeModel):
    schema_version: Literal["env-mock-agent/execution-telemetry/v2"] = "env-mock-agent/execution-telemetry/v2"
    execution_telemetry_id: Identifier
    duration_us: int | None = Field(default=None, ge=0)
    duration_availability: TelemetryAvailabilityV2
    tool_family_counts: tuple[ToolFamilyCountV2, ...] = ()
    tool_availability: TelemetryAvailabilityV2
    reported_cost: ReportedCostV2
    observed_at: datetime
    policy_version: Literal["execution-telemetry/r6-06-v1"] = EXECUTION_TELEMETRY_POLICY_VERSION
    execution_telemetry_sha256: Sha256

    @model_validator(mode="after")
    def validate_telemetry(self) -> Self:
        duration_known = self.duration_availability in {
            TelemetryAvailabilityV2.REPORTED,
            TelemetryAvailabilityV2.DERIVED,
        }
        if duration_known != (self.duration_us is not None):
            raise ValueError("duration availability and duration_us disagree")
        families = tuple(item.tool_family for item in self.tool_family_counts)
        if len(families) != len(set(families)):
            raise ValueError("tool family counts must be unique")
        if families != tuple(sorted(families, key=_tool_family_rank)):
            raise ValueError("tool family counts must use canonical order")
        if self.tool_family_counts and self.tool_availability not in {
            TelemetryAvailabilityV2.REPORTED,
            TelemetryAvailabilityV2.DERIVED,
        }:
            raise ValueError("tool counts require reported or derived availability")
        if self.observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        validate_execution_telemetry_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        duration_us: int | None,
        duration_availability: TelemetryAvailabilityV2,
        tool_family_counts: tuple[ToolFamilyCountV2, ...],
        tool_availability: TelemetryAvailabilityV2,
        reported_cost: ReportedCostV2,
        observed_at: datetime,
    ) -> ExecutionTelemetryV2:
        value = cls(
            execution_telemetry_id="execution-telemetry://pending",
            duration_us=duration_us,
            duration_availability=duration_availability,
            tool_family_counts=tuple(
                sorted(tool_family_counts, key=lambda item: _tool_family_rank(item.tool_family))
            ),
            tool_availability=tool_availability,
            reported_cost=reported_cost,
            observed_at=observed_at,
            execution_telemetry_sha256="0" * 64,
        )
        digest = execution_telemetry_carried_sha256(value)
        return value.model_copy(
            update={
                "execution_telemetry_id": f"execution-telemetry://sha256/{digest}",
                "execution_telemetry_sha256": digest,
            }
        )

    @classmethod
    def unavailable(cls, *, observed_at: datetime) -> ExecutionTelemetryV2:
        return cls.create(
            duration_us=None,
            duration_availability=TelemetryAvailabilityV2.UNAVAILABLE,
            tool_family_counts=(),
            tool_availability=TelemetryAvailabilityV2.UNAVAILABLE,
            reported_cost=ReportedCostV2.unavailable(),
            observed_at=observed_at,
        )


def normalize_reported_cost_usd(value: object | None) -> ReportedCostV2:
    if value is None:
        return ReportedCostV2.unavailable()
    if isinstance(value, bool):
        raise TypeError("reported cost cannot be boolean")
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("reported cost must be a decimal number") from exc
    if not decimal_value.is_finite() or decimal_value < 0:
        raise ValueError("reported cost must be finite and non-negative")
    try:
        quantized = decimal_value.quantize(
            _MICRO_USD,
            rounding=ROUND_HALF_EVEN,
        )
    except InvalidOperation as exc:
        raise ValueError("reported cost exceeds supported decimal precision") from exc
    return ReportedCostV2.reported(amount_microusd=int(quantized * _MICRO_USD_FACTOR))


def normalize_tool_family_counts(tool_names: Iterable[str]) -> tuple[ToolFamilyCountV2, ...]:
    counts = Counter(_tool_family(name) for name in tool_names)
    return tuple(
        ToolFamilyCountV2(tool_family=family, calls=counts[family])
        for family in ToolFamilyV2
        if counts[family]
    )


def execution_telemetry_carried_sha256(value: ExecutionTelemetryV2) -> str:
    payload = value.model_dump(
        mode="json",
        exclude={
            "execution_telemetry_id",
            "execution_telemetry_sha256",
        },
        exclude_none=False,
    )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def execution_telemetry_ref(value: ExecutionTelemetryV2) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type="execution-telemetry",
        object_id=value.execution_telemetry_id,
        object_version="v2",
        object_sha256=value.execution_telemetry_sha256,
    )


def validate_execution_telemetry_identity(value: ExecutionTelemetryV2) -> None:
    if (
        value.execution_telemetry_id == "execution-telemetry://pending"
        and value.execution_telemetry_sha256 == "0" * 64
    ):
        return
    digest = execution_telemetry_carried_sha256(value)
    if (
        value.execution_telemetry_id != f"execution-telemetry://sha256/{digest}"
        or value.execution_telemetry_sha256 != digest
    ):
        raise ValueError("execution telemetry identity is stale or invalid")


def _tool_family(name: str) -> ToolFamilyV2:
    normalized = name.strip().casefold().replace("-", "_")
    if normalized in {
        "read",
        "write",
        "edit",
        "multiedit",
        "glob",
        "grep",
        "ls",
        "file",
    }:
        return ToolFamilyV2.FILE
    if normalized in {
        "bash",
        "shell",
        "terminal",
        "exec",
        "exec_command",
        "powershell",
    }:
        return ToolFamilyV2.SHELL
    if normalized in {"websearch", "web_search", "search"}:
        return ToolFamilyV2.SEARCH
    if normalized in {"webfetch", "web_fetch", "fetch"}:
        return ToolFamilyV2.FETCH
    if normalized.startswith("browser_") or normalized in {
        "browser",
        "renderer",
        "screenshot",
    }:
        return ToolFamilyV2.RENDERER
    return ToolFamilyV2.OTHER


def _tool_family_rank(value: ToolFamilyV2) -> int:
    return tuple(ToolFamilyV2).index(value)
