from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from eval_factory.contracts.core import ContractAudit
from eval_factory.contracts.trace import (
    CapabilityStatus,
    ParseQuality,
    SourceSpan,
    TraceCapability,
)
from eval_factory.trace.adapters.curated_trajectory_v1 import (
    CuratedTrajectoryV1Adapter,
)
from eval_factory.trace.models import RegisteredTraceSource
from eval_factory.trace.parsing.models import (
    NestedFieldStatus,
    ParseDiagnostic,
    ParsedNestedField,
    ParsedRawTrajRecord,
    RawTrajParseOutcome,
    RawTrajParseResult,
    freeze_json,
)
from eval_factory.trace.parsing.raw_traj_v1 import (
    RawTrajV1Parser,
    canonical_json_sha256,
    read_registered_source_bytes,
    stable_trace_id,
)
from eval_factory.trace.parsing.recovery import (
    CAPABILITY_ORDER,
    RECOVERY_POLICY_VERSION,
    REQUEST_CAPABILITIES,
    RawTrajRecovery,
    RecoveredNestedField,
    RecoveredRawTrajRecord,
    RecoveryFieldStatus,
    TraceRecoveryResult,
)


class CuratedTrajectoryV1Parser(RawTrajV1Parser):
    def __init__(self) -> None:
        super().__init__(
            adapter=CuratedTrajectoryV1Adapter(),
            nested_fields=("request", "response"),
        )

    def parse(
        self,
        source_path: Path,
        *,
        registered_source: RegisteredTraceSource,
        audit: ContractAudit,
    ) -> RawTrajParseResult:
        del audit
        raw = read_registered_source_bytes(
            source_path,
            registered_source=registered_source,
            adapter=self.adapter,
        )
        value = json.loads(raw, parse_constant=_reject_json_constant)
        if not isinstance(value, dict):
            raise ValueError("accepted curated trajectory is not an object")
        frozen = freeze_json(value)
        if not isinstance(frozen, Mapping):
            raise ValueError("accepted curated trajectory did not freeze as an object")
        decoded_length = len(raw.decode("utf-8"))
        request_span = _source_span(
            registered_source=registered_source,
            field="request",
            raw_byte_start=0,
            raw_byte_end=len(raw),
            decoded_char_end=decoded_length,
        )
        response_span = _source_span(
            registered_source=registered_source,
            field="response",
            raw_byte_start=len(raw),
            raw_byte_end=len(raw),
            decoded_char_end=0,
        )
        request = ParsedNestedField(
            field="request",
            status=NestedFieldStatus.STRICT,
            value=frozen,
            canonical_value_sha256=canonical_json_sha256(value),
            source_span=request_span,
            repair_map=None,
            raw_sha256=registered_source.source.raw_sha256,
        )
        response = ParsedNestedField(
            field="response",
            status=NestedFieldStatus.NEEDS_STREAMING,
            value=None,
            canonical_value_sha256=None,
            source_span=response_span,
            repair_map=None,
            raw_sha256=registered_source.source.raw_sha256,
            diagnostics=(
                ParseDiagnostic(
                    code="curated-trajectory-separate-response-unavailable",
                    message="logical trajectory has no separately authoritative final response",
                ),
            ),
        )
        return RawTrajParseResult(
            registered_source=registered_source,
            repair_policy_version=self.repair_policy_version,
            outcome=RawTrajParseOutcome.NEEDS_STREAMING,
            records=(
                ParsedRawTrajRecord(
                    outer_record_index=0,
                    fields=(request, response),
                ),
            ),
            repair_maps=(),
        )


class CuratedTrajectoryV1Recovery(RawTrajRecovery):
    def __init__(self) -> None:
        super().__init__(adapter=CuratedTrajectoryV1Adapter())

    def recover(
        self,
        source_path: Path,
        *,
        parse_result: RawTrajParseResult,
        audit: ContractAudit,
    ) -> TraceRecoveryResult:
        del audit
        read_registered_source_bytes(
            source_path,
            registered_source=parse_result.registered_source,
            adapter=self.adapter,
        )
        if len(parse_result.records) != 1:
            raise ValueError("curated trajectory parse must contain exactly one record")
        parsed = parse_result.records[0]
        request, response = parsed.fields
        if (
            request.field != "request"
            or request.status is not NestedFieldStatus.STRICT
            or request.value is None
            or response.field != "response"
            or response.status is not NestedFieldStatus.NEEDS_STREAMING
        ):
            raise ValueError("curated trajectory parse fields are invalid")
        recovered_request = RecoveredNestedField(
            original=request,
            status=RecoveryFieldStatus.COMPLETE,
            units=(),
            unrecoverable_spans=(),
            repair_map=None,
            target_array_found=True,
            target_array_complete=True,
        )
        recovered_response = RecoveredNestedField(
            original=response,
            status=RecoveryFieldStatus.MISSING,
            units=(),
            unrecoverable_spans=(response.source_span,),
            repair_map=None,
            target_array_found=False,
            target_array_complete=False,
            diagnostics=response.diagnostics,
        )
        capabilities = tuple(
            TraceCapability(
                capability=capability,
                status=(
                    CapabilityStatus.COMPLETE
                    if capability in REQUEST_CAPABILITIES
                    else CapabilityStatus.MISSING
                ),
                reason_codes=(
                    () if capability in REQUEST_CAPABILITIES else ("separate-final-response-unavailable",)
                ),
            )
            for capability in CAPABILITY_ORDER
        )
        return TraceRecoveryResult(
            base_parse_result=parse_result,
            recovery_policy_version=RECOVERY_POLICY_VERSION,
            parse_quality=ParseQuality.PARTIAL,
            records=(
                RecoveredRawTrajRecord(
                    outer_record_index=0,
                    fields=(recovered_request, recovered_response),
                ),
            ),
            capabilities=capabilities,
            recovery_maps=(),
            unrecoverable_spans=(response.source_span,),
        )


def _source_span(
    *,
    registered_source: RegisteredTraceSource,
    field: Literal["request", "response"],
    raw_byte_start: int,
    raw_byte_end: int,
    decoded_char_end: int,
) -> SourceSpan:
    source = registered_source.source
    span_id = stable_trace_id(
        "source-span",
        {
            "source_trace_id": source.source_trace_id,
            "raw_sha256": source.raw_sha256,
            "outer_record_index": 0,
            "field": field,
            "raw_byte_start": raw_byte_start,
            "raw_byte_end": raw_byte_end,
            "adapter_name": source.adapter_name,
            "adapter_version": source.adapter_version,
        },
    )
    return SourceSpan(
        span_id=span_id,
        source_uri=source.source_uri,
        source_trace_id=source.source_trace_id,
        outer_record_index=0,
        field=field,
        raw_byte_start=raw_byte_start,
        raw_byte_end=raw_byte_end,
        decoded_char_start=0,
        decoded_char_end=decoded_char_end,
        repair_map_ref=None,
        approximate=False,
        raw_sha256=source.raw_sha256,
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


__all__ = [
    "CuratedTrajectoryV1Parser",
    "CuratedTrajectoryV1Recovery",
]
