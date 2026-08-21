from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Literal

from eval_factory.contracts.core import Sha256
from eval_factory.contracts.trace import RepairMap, SourceSpan
from eval_factory.trace.models import RegisteredTraceSource

type FrozenJsonValue = (
    None | bool | int | float | str | tuple[FrozenJsonValue, ...] | Mapping[str, FrozenJsonValue]
)
type FrozenJsonObject = Mapping[str, FrozenJsonValue]
type NestedFieldName = Literal["request", "response", "extra"]


class NestedFieldStatus(StrEnum):
    STRICT = "STRICT"
    REPAIRED = "REPAIRED"
    NEEDS_STREAMING = "NEEDS_STREAMING"


class RawTrajParseOutcome(StrEnum):
    STRICT = "STRICT"
    REPAIRED = "REPAIRED"
    NEEDS_STREAMING = "NEEDS_STREAMING"


@dataclass(frozen=True)
class ParseDiagnostic:
    code: str
    message: str
    error_position: int | None = None


@dataclass(frozen=True)
class ParsedNestedField:
    field: NestedFieldName
    status: NestedFieldStatus
    value: FrozenJsonObject | None
    canonical_value_sha256: Sha256 | None
    source_span: SourceSpan
    repair_map: RepairMap | None
    raw_sha256: Sha256
    diagnostics: tuple[ParseDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        successful = self.status in {
            NestedFieldStatus.STRICT,
            NestedFieldStatus.REPAIRED,
        }
        if successful != (self.value is not None and self.canonical_value_sha256 is not None):
            raise ValueError("successful fields require a parsed value and canonical hash")
        if (self.status is NestedFieldStatus.REPAIRED) != (self.repair_map is not None):
            raise ValueError("only repaired fields may contain a RepairMap")
        if self.status is NestedFieldStatus.NEEDS_STREAMING and not self.diagnostics:
            raise ValueError("NEEDS_STREAMING fields require a diagnostic")
        if successful and self.diagnostics:
            raise ValueError("successful fields cannot contain diagnostics")
        if self.source_span.raw_sha256 != self.raw_sha256:
            raise ValueError("source span and parsed field must bind the same raw hash")
        if self.source_span.field != self.field:
            raise ValueError("source span and parsed field must bind the same field")
        if self.repair_map is None:
            if self.source_span.repair_map_ref is not None or self.source_span.approximate:
                raise ValueError("unrepaired fields require an exact span without a repair reference")
        else:
            if self.source_span.repair_map_ref is None or not self.source_span.approximate:
                raise ValueError("repaired fields require an approximate span and repair reference")
            self.source_span.repair_map_ref.require_matching_hash(self.repair_map)


@dataclass(frozen=True)
class ParsedRawTrajRecord:
    outer_record_index: int
    fields: tuple[ParsedNestedField, ...]

    def __post_init__(self) -> None:
        if self.outer_record_index < 0:
            raise ValueError("outer_record_index must be non-negative")
        names = tuple(item.field for item in self.fields)
        if names not in {
            ("request", "response"),
            ("request", "response", "extra"),
        }:
            raise ValueError("record fields must be ordered as request, response, with optional extra")


@dataclass(frozen=True)
class RawTrajParseResult:
    registered_source: RegisteredTraceSource
    repair_policy_version: str
    outcome: RawTrajParseOutcome
    records: tuple[ParsedRawTrajRecord, ...]
    repair_maps: tuple[RepairMap, ...]

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("parse result requires at least one record")
        observed = tuple(
            field.repair_map
            for record in self.records
            for field in record.fields
            if field.repair_map is not None
        )
        if observed != self.repair_maps:
            raise ValueError("repair_maps must match repaired fields in source order")
        statuses = {field.status for record in self.records for field in record.fields}
        expected = RawTrajParseOutcome.STRICT
        if NestedFieldStatus.NEEDS_STREAMING in statuses:
            expected = RawTrajParseOutcome.NEEDS_STREAMING
        elif NestedFieldStatus.REPAIRED in statuses:
            expected = RawTrajParseOutcome.REPAIRED
        if self.outcome is not expected:
            raise ValueError(f"parse outcome must be {expected}")


def freeze_json(value: object) -> FrozenJsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return tuple(freeze_json(item) for item in value)
    if isinstance(value, dict):
        frozen = {str(key): freeze_json(item) for key, item in sorted(value.items())}
        return MappingProxyType(frozen)
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")
