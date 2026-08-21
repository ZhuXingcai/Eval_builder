from __future__ import annotations

import io
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from eval_factory.contracts.core import ContractAudit, Sha256
from eval_factory.contracts.trace import (
    CapabilityStatus,
    ParseQuality,
    RepairMap,
    SourceSpan,
    TraceCapability,
)
from eval_factory.trace.adapters import (
    RawTrajV1Adapter,
    RuntimeSnapshotV1Adapter,
    TraceAdapter,
)
from eval_factory.trace.parsing.models import (
    FrozenJsonObject,
    NestedFieldName,
    NestedFieldStatus,
    ParseDiagnostic,
    ParsedNestedField,
    RawTrajParseOutcome,
    RawTrajParseResult,
    freeze_json,
)
from eval_factory.trace.parsing.raw_json import LocatedJsonString, locate_top_level_strings
from eval_factory.trace.parsing.raw_traj_v1 import (
    NESTED_FIELDS,
    build_repair_map,
    build_source_span,
    canonical_json_sha256,
    decode_nested_object,
    decode_outer_object,
    read_registered_source_bytes,
    stable_trace_id,
)
from eval_factory.trace.parsing.repair import (
    repair_invalid_json_string_backslashes,
    repaired_boundary_to_source,
)
from eval_factory.trace.parsing.streaming import (
    RecoveryCandidate,
    RecoveryUnitKind,
    scan_request_units,
    scan_response_units,
)

type TraceCapabilityName = Literal[
    "conversation_events",
    "tool_events",
    "call_result_pairing",
    "file_timeline",
    "final_response",
    "attachment_observations",
]

RECOVERY_POLICY_VERSION = "raw-traj-streaming-recovery/v1"
CAPABILITY_ORDER: tuple[TraceCapabilityName, ...] = (
    "conversation_events",
    "tool_events",
    "call_result_pairing",
    "file_timeline",
    "final_response",
    "attachment_observations",
)
REQUEST_CAPABILITIES: tuple[TraceCapabilityName, ...] = (
    "conversation_events",
    "tool_events",
    "call_result_pairing",
    "file_timeline",
    "attachment_observations",
)


class RecoveryFieldStatus(StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"


@dataclass(frozen=True)
class RecoveredJsonUnit:
    unit_id: str
    kind: RecoveryUnitKind
    outer_record_index: int
    field: NestedFieldName
    source_order: int
    observed_role: str | None
    value: FrozenJsonObject
    canonical_value_sha256: Sha256
    source_span: SourceSpan
    repair_map: RepairMap | None

    def __post_init__(self) -> None:
        if self.outer_record_index < 0 or self.source_order < 0:
            raise ValueError("recovery unit indexes must be non-negative")
        if (self.repair_map is None) != (self.source_span.repair_map_ref is None):
            raise ValueError("recovery unit repair map and source span reference must agree")
        if self.repair_map is not None:
            assert self.source_span.repair_map_ref is not None
            self.source_span.repair_map_ref.require_matching_hash(self.repair_map)
        if self.source_span.field != self.field:
            raise ValueError("recovery unit and source span field must agree")


@dataclass(frozen=True)
class RecoveredNestedField:
    original: ParsedNestedField
    status: RecoveryFieldStatus
    units: tuple[RecoveredJsonUnit, ...]
    unrecoverable_spans: tuple[SourceSpan, ...]
    repair_map: RepairMap | None
    target_array_found: bool
    target_array_complete: bool
    diagnostics: tuple[ParseDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if self.original.status is not NestedFieldStatus.NEEDS_STREAMING:
            if self.status is not RecoveryFieldStatus.COMPLETE:
                raise ValueError("strict/repaired fields must remain complete")
            if self.units or self.unrecoverable_spans or self.repair_map or self.diagnostics:
                raise ValueError("complete fields cannot contain streaming recovery output")
        elif self.status is RecoveryFieldStatus.COMPLETE:
            raise ValueError("NEEDS_STREAMING fields cannot be upgraded to complete")
        if self.status is RecoveryFieldStatus.PARTIAL and not (self.units or self.target_array_found):
            raise ValueError("partial fields require recovered or structural evidence")
        if self.status is RecoveryFieldStatus.MISSING and self.units:
            raise ValueError("missing fields cannot contain recovery units")
        if self.repair_map is not None:
            for unit in self.units:
                if unit.repair_map != self.repair_map:
                    raise ValueError("recovery units must bind the field repair map")
            for span in self.unrecoverable_spans:
                if span.repair_map_ref is None:
                    raise ValueError("repaired field spans require a repair reference")
                span.repair_map_ref.require_matching_hash(self.repair_map)
        for unit in self.units:
            for span in self.unrecoverable_spans:
                if not (
                    unit.source_span.raw_byte_end <= span.raw_byte_start
                    or unit.source_span.raw_byte_start >= span.raw_byte_end
                ):
                    raise ValueError("recovered and unrecoverable spans cannot overlap")


@dataclass(frozen=True)
class RecoveredRawTrajRecord:
    outer_record_index: int
    fields: tuple[RecoveredNestedField, ...]

    def __post_init__(self) -> None:
        if self.outer_record_index < 0:
            raise ValueError("outer_record_index must be non-negative")
        if tuple(item.original.field for item in self.fields) not in {
            ("request", "response"),
            NESTED_FIELDS,
        }:
            raise ValueError("recovered record fields must use canonical order")


@dataclass(frozen=True)
class TraceRecoveryResult:
    base_parse_result: RawTrajParseResult
    recovery_policy_version: str
    parse_quality: ParseQuality
    records: tuple[RecoveredRawTrajRecord, ...]
    capabilities: tuple[TraceCapability, ...]
    recovery_maps: tuple[RepairMap, ...]
    unrecoverable_spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if len(self.records) != len(self.base_parse_result.records):
            raise ValueError("recovery records must match base parse records")
        for recovered, parsed in zip(
            self.records,
            self.base_parse_result.records,
            strict=True,
        ):
            if recovered.outer_record_index != parsed.outer_record_index:
                raise ValueError("recovery record indexes must match base parse records")
            if tuple(field.original for field in recovered.fields) != parsed.fields:
                raise ValueError("recovery fields must bind base parsed fields")
        if tuple(item.capability for item in self.capabilities) != CAPABILITY_ORDER:
            raise ValueError("capabilities must use canonical order")
        if self.parse_quality is not _derive_parse_quality(
            self.base_parse_result,
            self.capabilities,
        ):
            raise ValueError("parse quality does not match capability evidence")
        observed_maps = tuple(
            field.repair_map
            for record in self.records
            for field in record.fields
            if field.repair_map is not None
        )
        if observed_maps != self.recovery_maps:
            raise ValueError("recovery_maps must match recovered fields")
        observed_spans = tuple(
            span for record in self.records for field in record.fields for span in field.unrecoverable_spans
        )
        if observed_spans != self.unrecoverable_spans:
            raise ValueError("unrecoverable_spans must match recovered fields")


class RawTrajRecovery:
    recovery_policy_version = RECOVERY_POLICY_VERSION

    def __init__(
        self,
        *,
        adapter: TraceAdapter | None = None,
    ) -> None:
        self.adapter = adapter or RawTrajV1Adapter()

    def recover(
        self,
        source_path: Path,
        *,
        parse_result: RawTrajParseResult,
        audit: ContractAudit,
    ) -> TraceRecoveryResult:
        registered = parse_result.registered_source
        raw = read_registered_source_bytes(
            source_path,
            registered_source=registered,
            adapter=self.adapter,
        )
        target_fields = frozenset(field.field for record in parse_result.records for field in record.fields)
        located_records = _locate_records(
            raw,
            target_fields=target_fields,
        )
        if len(located_records) != len(parse_result.records):
            raise ValueError("located source records do not match base parse result")

        recovered_records: list[RecoveredRawTrajRecord] = []
        for parsed_record, located in zip(
            parse_result.records,
            located_records,
            strict=True,
        ):
            fields = tuple(
                self._recover_field(
                    original=original,
                    located=located[original.field],
                    outer_record_index=parsed_record.outer_record_index,
                    parse_result=parse_result,
                    audit=audit,
                )
                for original in parsed_record.fields
            )
            recovered_records.append(
                RecoveredRawTrajRecord(
                    outer_record_index=parsed_record.outer_record_index,
                    fields=fields,
                )
            )

        records = tuple(recovered_records)
        capabilities = _derive_capabilities(
            records,
            adapter_name=registered.source.adapter_name,
        )
        parse_quality = _derive_parse_quality(parse_result, capabilities)
        recovery_maps = tuple(
            field.repair_map for record in records for field in record.fields if field.repair_map is not None
        )
        unrecoverable_spans = tuple(
            span for record in records for field in record.fields for span in field.unrecoverable_spans
        )
        return TraceRecoveryResult(
            base_parse_result=parse_result,
            recovery_policy_version=self.recovery_policy_version,
            parse_quality=parse_quality,
            records=records,
            capabilities=capabilities,
            recovery_maps=recovery_maps,
            unrecoverable_spans=unrecoverable_spans,
        )

    def _recover_field(
        self,
        *,
        original: ParsedNestedField,
        located: LocatedJsonString,
        outer_record_index: int,
        parse_result: RawTrajParseResult,
        audit: ContractAudit,
    ) -> RecoveredNestedField:
        if original.status is not NestedFieldStatus.NEEDS_STREAMING:
            return RecoveredNestedField(
                original=original,
                status=RecoveryFieldStatus.COMPLETE,
                units=(),
                unrecoverable_spans=(),
                repair_map=None,
                target_array_found=True,
                target_array_complete=True,
            )
        if original.field == "extra":
            return RecoveredNestedField(
                original=original,
                status=RecoveryFieldStatus.MISSING,
                units=(),
                unrecoverable_spans=(original.source_span,),
                repair_map=None,
                target_array_found=False,
                target_array_complete=False,
                diagnostics=(
                    ParseDiagnostic(
                        code="streaming-recovery-unsupported-field",
                        message="streaming recovery is not defined for this nested field",
                    ),
                ),
            )

        repair = repair_invalid_json_string_backslashes(located.value)
        boundary_map = repaired_boundary_to_source(located.value, repair)
        scan = (
            scan_request_units(repair.value)
            if original.field == "request"
            else scan_response_units(repair.value)
        )
        repair_map = (
            build_repair_map(
                registered_source=parse_result.registered_source,
                outer_record_index=outer_record_index,
                field=original.field,
                located=located,
                repair=repair,
                audit=audit,
            )
            if repair.edits
            else None
        )
        units, invalid_candidates = _build_units(
            candidates=scan.candidates,
            repaired_value=repair.value,
            boundary_map=boundary_map,
            repair_map=repair_map,
            original=original,
            located=located,
            outer_record_index=outer_record_index,
            parse_result=parse_result,
        )
        ranges = _merge_ranges([*scan.unrecoverable_ranges, *invalid_candidates])
        if scan.target_array_complete and not ranges:
            ranges = ((len(repair.value), len(repair.value)),)
        spans = tuple(
            _range_span(
                start=start,
                end=end,
                boundary_map=boundary_map,
                repair_map=repair_map,
                original=original,
                located=located,
                outer_record_index=outer_record_index,
                parse_result=parse_result,
            )
            for start, end in ranges
        )
        clean_empty_array = (
            scan.target_array_complete
            and not scan.candidates
            and not scan.unrecoverable_ranges
            and not invalid_candidates
        )
        has_evidence = bool(units) or clean_empty_array
        return RecoveredNestedField(
            original=original,
            status=(RecoveryFieldStatus.PARTIAL if has_evidence else RecoveryFieldStatus.MISSING),
            units=units,
            unrecoverable_spans=spans,
            repair_map=repair_map,
            target_array_found=scan.target_array_found,
            target_array_complete=scan.target_array_complete,
            diagnostics=(
                ParseDiagnostic(
                    code=(
                        "streaming-recovery-partial"
                        if has_evidence
                        else "streaming-recovery-no-complete-unit"
                    ),
                    message=(
                        "complete structural units were recovered from an incomplete field"
                        if has_evidence
                        else "no complete structural unit could be recovered"
                    ),
                ),
            ),
        )


def _locate_records(
    raw: bytes,
    *,
    target_fields: frozenset[str] = frozenset(NESTED_FIELDS),
) -> tuple[dict[str, LocatedJsonString], ...]:
    records: list[dict[str, LocatedJsonString]] = []
    line_byte_start = 0
    for raw_line in io.BytesIO(raw):
        if raw_line.strip():
            outer = decode_outer_object(raw_line)
            records.append(
                locate_top_level_strings(
                    raw_line,
                    line_byte_start=line_byte_start,
                    expected_outer=outer,
                    target_fields=target_fields,
                )
            )
        line_byte_start += len(raw_line)
    return tuple(records)


def _build_units(
    *,
    candidates: tuple[RecoveryCandidate, ...],
    repaired_value: str,
    boundary_map: tuple[int, ...],
    repair_map: RepairMap | None,
    original: ParsedNestedField,
    located: LocatedJsonString,
    outer_record_index: int,
    parse_result: RawTrajParseResult,
) -> tuple[tuple[RecoveredJsonUnit, ...], tuple[tuple[int, int], ...]]:
    units: list[RecoveredJsonUnit] = []
    invalid: list[tuple[int, int]] = []
    source_order = 0
    for candidate in candidates:
        decoded, _ = decode_nested_object(repaired_value[candidate.start : candidate.end])
        if decoded is None:
            invalid.append((candidate.start, candidate.end))
            continue
        frozen = freeze_json(decoded)
        if not isinstance(frozen, Mapping):
            invalid.append((candidate.start, candidate.end))
            continue
        span = _range_span(
            start=candidate.start,
            end=candidate.end,
            boundary_map=boundary_map,
            repair_map=repair_map,
            original=original,
            located=located,
            outer_record_index=outer_record_index,
            parse_result=parse_result,
        )
        unit_id = stable_trace_id(
            "recovery-unit",
            {
                "source_trace_id": parse_result.registered_source.source.source_trace_id,
                "raw_sha256": parse_result.registered_source.source.raw_sha256,
                "outer_record_index": outer_record_index,
                "field": original.field,
                "kind": candidate.kind,
                "source_order": source_order,
                "raw_byte_start": span.raw_byte_start,
                "raw_byte_end": span.raw_byte_end,
                "recovery_policy_version": RECOVERY_POLICY_VERSION,
            },
        )
        units.append(
            RecoveredJsonUnit(
                unit_id=unit_id,
                kind=candidate.kind,
                outer_record_index=outer_record_index,
                field=original.field,
                source_order=source_order,
                observed_role=candidate.observed_role,
                value=frozen,
                canonical_value_sha256=canonical_json_sha256(decoded),
                source_span=span,
                repair_map=repair_map,
            )
        )
        source_order += 1
    return tuple(units), tuple(invalid)


def _range_span(
    *,
    start: int,
    end: int,
    boundary_map: tuple[int, ...],
    repair_map: RepairMap | None,
    original: ParsedNestedField,
    located: LocatedJsonString,
    outer_record_index: int,
    parse_result: RawTrajParseResult,
) -> SourceSpan:
    if start < 0 or end < start or end >= len(boundary_map):
        raise ValueError(f"recovery range is out of bounds: {start}:{end}")
    return build_source_span(
        registered_source=parse_result.registered_source,
        outer_record_index=outer_record_index,
        field=original.field,
        located=located,
        source_char_start=boundary_map[start],
        source_char_end=boundary_map[end],
        decoded_char_start=start,
        decoded_char_end=end,
        repair_map=repair_map,
    )


def _derive_capabilities(
    records: tuple[RecoveredRawTrajRecord, ...],
    *,
    adapter_name: str,
) -> tuple[TraceCapability, ...]:
    request_statuses: list[tuple[CapabilityStatus, str | None]] = []
    response_statuses: list[tuple[CapabilityStatus, str | None]] = []
    for record in records:
        by_name = {field.original.field: field for field in record.fields}
        request_statuses.append(
            _field_capability(
                by_name["request"],
                loss=_loss_count_for_adapter(
                    by_name.get("extra"),
                    key="req_lost_number",
                    adapter_name=adapter_name,
                ),
                expected_members=("messages", "input"),
            )
        )
        response_statuses.append(
            _field_capability(
                by_name["response"],
                loss=_loss_count_for_adapter(
                    by_name.get("extra"),
                    key="resp_lost_number",
                    adapter_name=adapter_name,
                ),
                expected_members=("content", "output", "choices"),
            )
        )
    request_status, request_reasons = _aggregate_status(request_statuses)
    response_status, response_reasons = _aggregate_status(response_statuses)
    return tuple(
        TraceCapability(
            capability=name,
            status=(response_status if name == "final_response" else request_status),
            reason_codes=(response_reasons if name == "final_response" else request_reasons),
        )
        for name in CAPABILITY_ORDER
    )


def _field_capability(
    field: RecoveredNestedField,
    *,
    loss: tuple[int | None, str | None],
    expected_members: tuple[str, ...],
) -> tuple[CapabilityStatus, str | None]:
    loss_count, loss_reason = loss
    if field.original.value is not None:
        member = next(
            (field.original.value[name] for name in expected_members if name in field.original.value),
            None,
        )
        structurally_complete = isinstance(member, tuple) and all(
            isinstance(item, Mapping) for item in member
        )
        usable = structurally_complete or (
            isinstance(member, tuple) and any(isinstance(item, Mapping) for item in member)
        )
        if structurally_complete and loss_count == 0:
            return CapabilityStatus.COMPLETE, None
        if usable or structurally_complete:
            return (
                CapabilityStatus.PARTIAL,
                loss_reason or "nested-structure-partial",
            )
        return CapabilityStatus.MISSING, "expected-structure-missing"
    if field.status is RecoveryFieldStatus.PARTIAL:
        return CapabilityStatus.PARTIAL, "streaming-recovered-partial"
    return CapabilityStatus.MISSING, "streaming-no-complete-unit"


def _loss_count(
    extra: RecoveredNestedField,
    key: str,
) -> tuple[int | None, str | None]:
    if extra.original.value is None:
        return None, "loss-metadata-unavailable"
    value = extra.original.value.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None, "loss-metadata-invalid"
    if value > 0:
        return value, "source-loss-reported"
    return 0, None


def _loss_count_for_adapter(
    extra: RecoveredNestedField | None,
    *,
    key: str,
    adapter_name: str,
) -> tuple[int | None, str | None]:
    if extra is not None:
        return _loss_count(extra, key)
    if adapter_name == RuntimeSnapshotV1Adapter.name:
        return 0, None
    return None, "loss-metadata-unavailable"


def _aggregate_status(
    statuses: list[tuple[CapabilityStatus, str | None]],
) -> tuple[CapabilityStatus, tuple[str, ...]]:
    observed = [status for status, _ in statuses]
    reasons = tuple(sorted({reason for _, reason in statuses if reason is not None}))
    if all(status is CapabilityStatus.COMPLETE for status in observed):
        return CapabilityStatus.COMPLETE, ()
    if all(status is CapabilityStatus.MISSING for status in observed):
        return CapabilityStatus.MISSING, reasons or ("capability-missing",)
    return CapabilityStatus.PARTIAL, reasons or ("capability-partial",)


def _derive_parse_quality(
    parse_result: RawTrajParseResult,
    capabilities: tuple[TraceCapability, ...],
) -> ParseQuality:
    statuses = {capability.status for capability in capabilities}
    if statuses == {CapabilityStatus.COMPLETE}:
        return (
            ParseQuality.STRICT
            if parse_result.outcome is RawTrajParseOutcome.STRICT
            else ParseQuality.REPAIRED
        )
    usable = any(
        capability.status in {CapabilityStatus.COMPLETE, CapabilityStatus.PARTIAL}
        for capability in capabilities
    )
    return ParseQuality.PARTIAL if usable else ParseQuality.UNPARSEABLE


def _merge_ranges(
    ranges: list[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted((start, end) for start, end in ranges if end >= start):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return tuple(merged)
