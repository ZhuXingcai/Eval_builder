from __future__ import annotations

import hashlib
import io
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.trace import RepairMap, RepairSegment, SourceSpan
from eval_factory.trace.adapters import (
    RawTrajV1Adapter,
    RuntimeSnapshotV1Adapter,
    TraceAdapter,
)
from eval_factory.trace.models import RegisteredTraceSource, TraceProbeStatus
from eval_factory.trace.parsing.models import (
    NestedFieldName,
    NestedFieldStatus,
    ParseDiagnostic,
    ParsedNestedField,
    ParsedRawTrajRecord,
    RawTrajParseOutcome,
    RawTrajParseResult,
    freeze_json,
)
from eval_factory.trace.parsing.raw_json import LocatedJsonString, locate_top_level_strings
from eval_factory.trace.parsing.repair import (
    INVALID_BACKSLASH_RULE_ID,
    INVALID_BACKSLASH_TRANSFORM,
    REPAIR_POLICY_VERSION,
    LocalRepairResult,
    repair_invalid_json_string_backslashes,
)

NESTED_FIELDS: tuple[NestedFieldName, ...] = ("request", "response", "extra")


class TraceParseError(RuntimeError):
    pass


class RegisteredSourceMismatchError(TraceParseError):
    pass


class RawTrajV1Parser:
    repair_policy_version = REPAIR_POLICY_VERSION

    def __init__(
        self,
        *,
        adapter: TraceAdapter | None = None,
        nested_fields: tuple[NestedFieldName, ...] = NESTED_FIELDS,
    ) -> None:
        self.adapter = adapter or RawTrajV1Adapter()
        self.nested_fields = nested_fields

    def parse(
        self,
        source_path: Path,
        *,
        registered_source: RegisteredTraceSource,
        audit: ContractAudit,
    ) -> RawTrajParseResult:
        raw = self._read_registered_bytes(source_path, registered_source)
        records: list[ParsedRawTrajRecord] = []
        repair_maps: list[RepairMap] = []
        line_byte_start = 0
        outer_record_index = 0

        for raw_line in io.BytesIO(raw):
            if raw_line.strip():
                outer = decode_outer_object(raw_line)
                located = locate_top_level_strings(
                    raw_line,
                    line_byte_start=line_byte_start,
                    expected_outer=outer,
                    target_fields=frozenset(self.nested_fields),
                )
                fields = tuple(
                    self._parse_nested_field(
                        field=field,
                        located=located[field],
                        outer_record_index=outer_record_index,
                        registered_source=registered_source,
                        audit=audit,
                    )
                    for field in self.nested_fields
                )
                record = ParsedRawTrajRecord(
                    outer_record_index=outer_record_index,
                    fields=fields,
                )
                records.append(record)
                repair_maps.extend(item.repair_map for item in fields if item.repair_map is not None)
                outer_record_index += 1
            line_byte_start += len(raw_line)

        if outer_record_index != registered_source.record_count:
            raise RegisteredSourceMismatchError(
                "parsed record count does not match immutable source registration"
            )
        statuses = {field.status for record in records for field in record.fields}
        outcome = RawTrajParseOutcome.STRICT
        if NestedFieldStatus.NEEDS_STREAMING in statuses:
            outcome = RawTrajParseOutcome.NEEDS_STREAMING
        elif NestedFieldStatus.REPAIRED in statuses:
            outcome = RawTrajParseOutcome.REPAIRED
        return RawTrajParseResult(
            registered_source=registered_source,
            repair_policy_version=self.repair_policy_version,
            outcome=outcome,
            records=tuple(records),
            repair_maps=tuple(repair_maps),
        )

    def _read_registered_bytes(
        self,
        source_path: Path,
        registered_source: RegisteredTraceSource,
    ) -> bytes:
        return read_registered_source_bytes(
            source_path,
            registered_source=registered_source,
            adapter=self.adapter,
        )

    def _parse_nested_field(
        self,
        *,
        field: NestedFieldName,
        located: LocatedJsonString,
        outer_record_index: int,
        registered_source: RegisteredTraceSource,
        audit: ContractAudit,
    ) -> ParsedNestedField:
        parsed, diagnostic = decode_nested_object(located.value)
        if parsed is not None:
            return self._successful_field(
                field=field,
                value=parsed,
                decoded_source=located.value,
                located=located,
                outer_record_index=outer_record_index,
                registered_source=registered_source,
                repair=None,
                audit=audit,
            )

        repair = repair_invalid_json_string_backslashes(located.value)
        if repair.edits:
            repaired, repaired_diagnostic = decode_nested_object(repair.value)
            if repaired is not None:
                return self._successful_field(
                    field=field,
                    value=repaired,
                    decoded_source=repair.value,
                    located=located,
                    outer_record_index=outer_record_index,
                    registered_source=registered_source,
                    repair=repair,
                    audit=audit,
                )
            diagnostic = ParseDiagnostic(
                code="nested-json-local-repair-insufficient",
                message="nested JSON remains invalid after the allowlisted local repair",
                error_position=(
                    repaired_diagnostic.error_position if repaired_diagnostic is not None else None
                ),
            )
        assert diagnostic is not None
        source_span = build_source_span(
            registered_source=registered_source,
            outer_record_index=outer_record_index,
            field=field,
            located=located,
            decoded_char_end=len(located.value),
            repair_map=None,
        )
        return ParsedNestedField(
            field=field,
            status=NestedFieldStatus.NEEDS_STREAMING,
            value=None,
            canonical_value_sha256=None,
            source_span=source_span,
            repair_map=None,
            raw_sha256=registered_source.source.raw_sha256,
            diagnostics=(diagnostic,),
        )

    def _successful_field(
        self,
        *,
        field: NestedFieldName,
        value: dict[str, Any],
        decoded_source: str,
        located: LocatedJsonString,
        outer_record_index: int,
        registered_source: RegisteredTraceSource,
        repair: LocalRepairResult | None,
        audit: ContractAudit,
    ) -> ParsedNestedField:
        repair_map = (
            build_repair_map(
                registered_source=registered_source,
                outer_record_index=outer_record_index,
                field=field,
                located=located,
                repair=repair,
                audit=audit,
            )
            if repair is not None
            else None
        )
        source_span = build_source_span(
            registered_source=registered_source,
            outer_record_index=outer_record_index,
            field=field,
            located=located,
            decoded_char_end=len(decoded_source),
            repair_map=repair_map,
        )
        frozen = freeze_json(value)
        if not isinstance(frozen, Mapping):
            raise TraceParseError("nested JSON object did not freeze as an object")
        return ParsedNestedField(
            field=field,
            status=(NestedFieldStatus.REPAIRED if repair_map is not None else NestedFieldStatus.STRICT),
            value=frozen,
            canonical_value_sha256=canonical_json_sha256(value),
            source_span=source_span,
            repair_map=repair_map,
            raw_sha256=registered_source.source.raw_sha256,
        )


def decode_outer_object(raw_line: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            raw_line,
            parse_constant=_reject_json_constant,
            parse_float=_finite_float,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise TraceParseError(f"registered outer record failed strict decode: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise TraceParseError("registered outer record is not an object")
    return value


def decode_nested_object(
    value: str,
) -> tuple[dict[str, Any] | None, ParseDiagnostic | None]:
    try:
        decoded = json.loads(
            value,
            parse_constant=_reject_json_constant,
            parse_float=_finite_float,
        )
    except json.JSONDecodeError as exc:
        return None, ParseDiagnostic(
            code="nested-json-invalid",
            message=f"nested JSON failed strict decoding: {exc.msg}",
            error_position=exc.pos,
        )
    except ValueError as exc:
        return None, ParseDiagnostic(
            code="nested-json-non-standard-number",
            message=f"nested JSON failed strict decoding: {type(exc).__name__}",
        )
    if not isinstance(decoded, dict):
        return None, ParseDiagnostic(
            code="nested-json-root-not-object",
            message=f"nested JSON root must be an object, observed {type(decoded).__name__}",
        )
    return decoded, None


def build_repair_map(
    *,
    registered_source: RegisteredTraceSource,
    outer_record_index: int,
    field: NestedFieldName,
    located: LocatedJsonString,
    repair: LocalRepairResult,
    audit: ContractAudit,
) -> RepairMap:
    segments = tuple(
        RepairSegment(
            raw_byte_start=located.raw_range_for_char(edit.source_char_start)[0],
            raw_byte_end=located.raw_range_for_char(edit.source_char_start)[1],
            decoded_char_start=edit.repaired_char_start,
            decoded_char_end=edit.repaired_char_end,
            transform=INVALID_BACKSLASH_TRANSFORM,
            rule_id=INVALID_BACKSLASH_RULE_ID,
            exact=False,
        )
        for edit in repair.edits
    )
    repair_map_id = stable_trace_id(
        "repair-map",
        {
            "source_trace_id": registered_source.source.source_trace_id,
            "raw_sha256": registered_source.source.raw_sha256,
            "outer_record_index": outer_record_index,
            "field": field,
            "repair_policy_version": REPAIR_POLICY_VERSION,
            "segments": [segment.model_dump(mode="json", exclude_none=False) for segment in segments],
        },
    )
    return RepairMap(
        repair_map_id=repair_map_id,
        source_trace_id=registered_source.source.source_trace_id,
        field=field,
        segments=segments,
        semantic_critical=True,
        audit=audit,
    )


def build_source_span(
    *,
    registered_source: RegisteredTraceSource,
    outer_record_index: int,
    field: NestedFieldName,
    located: LocatedJsonString,
    source_char_start: int = 0,
    source_char_end: int | None = None,
    decoded_char_start: int = 0,
    decoded_char_end: int,
    repair_map: RepairMap | None,
) -> SourceSpan:
    source = registered_source.source
    source_end = len(located.value) if source_char_end is None else source_char_end
    raw_byte_start, raw_byte_end = located.raw_range_for_chars(
        source_char_start,
        source_end,
    )
    span_id = stable_trace_id(
        "source-span",
        {
            "source_trace_id": source.source_trace_id,
            "raw_sha256": source.raw_sha256,
            "outer_record_index": outer_record_index,
            "field": field,
            "raw_byte_start": raw_byte_start,
            "raw_byte_end": raw_byte_end,
        },
    )
    repair_map_ref = (
        ObjectRef(
            object_type="repair-map",
            object_id=repair_map.repair_map_id,
            object_version=REPAIR_POLICY_VERSION,
            object_sha256=repair_map.canonical_sha256(),
        )
        if repair_map is not None
        else None
    )
    return SourceSpan(
        span_id=span_id,
        source_uri=source.source_uri,
        source_trace_id=source.source_trace_id,
        outer_record_index=outer_record_index,
        field=field,
        raw_byte_start=raw_byte_start,
        raw_byte_end=raw_byte_end,
        decoded_char_start=decoded_char_start,
        decoded_char_end=decoded_char_end,
        repair_map_ref=repair_map_ref,
        approximate=repair_map is not None,
        raw_sha256=source.raw_sha256,
    )


def canonical_json_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def stable_trace_id(prefix: str, value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return f"{prefix}://sha256/{hashlib.sha256(encoded).hexdigest()}"


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def read_registered_source_bytes(
    source_path: Path,
    *,
    registered_source: RegisteredTraceSource,
    adapter: TraceAdapter | None = None,
) -> bytes:
    active_adapter = adapter or RawTrajV1Adapter()
    source = registered_source.source
    if source.adapter_name != active_adapter.name or source.adapter_version != active_adapter.version:
        raise RegisteredSourceMismatchError("registered adapter identity does not match parser adapter")
    probe = active_adapter.probe(source_path)
    if probe.status is not TraceProbeStatus.SUPPORTED:
        codes = ", ".join(item.code for item in probe.diagnostics) or probe.status
        raise RegisteredSourceMismatchError(f"source is no longer parseable: {codes}")
    expected_probe = (
        source.raw_sha256,
        registered_source.size_bytes,
        registered_source.record_count,
        registered_source.outer_fields,
    )
    observed_probe = (
        probe.raw_sha256,
        probe.size_bytes,
        probe.record_count,
        probe.outer_fields,
    )
    if observed_probe != expected_probe:
        raise RegisteredSourceMismatchError(
            "current source probe does not match immutable source registration"
        )
    try:
        raw = source_path.expanduser().absolute().read_bytes()
    except OSError as exc:
        raise RegisteredSourceMismatchError(
            f"registered source could not be read: {type(exc).__name__}"
        ) from exc
    if len(raw) != registered_source.size_bytes:
        raise RegisteredSourceMismatchError(
            "current source byte size does not match immutable source registration"
        )
    if hashlib.sha256(raw).hexdigest() != source.raw_sha256:
        raise RegisteredSourceMismatchError(
            "current source hash does not match immutable source registration"
        )
    return raw


class RuntimeSnapshotV1Parser(RawTrajV1Parser):
    def __init__(self) -> None:
        super().__init__(
            adapter=RuntimeSnapshotV1Adapter(),
            nested_fields=("request", "response"),
        )
