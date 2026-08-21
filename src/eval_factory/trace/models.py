from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from eval_factory.contracts.core import Identifier, Sha256
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.orchestration import TraceSourceRef


def registered_trace_source_sha256(
    source: TraceSourceRef,
    size_bytes: int,
    record_count: int,
    outer_fields: tuple[str, ...],
) -> str:
    payload = {
        "source": source.model_dump(mode="json", exclude_none=False),
        "size_bytes": size_bytes,
        "record_count": record_count,
        "outer_fields": list(outer_fields),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class TraceProbeStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    INVALID = "INVALID"
    UNREADABLE = "UNREADABLE"


class TraceProbeDiagnostic(ContractModelV2):
    schema_version: Literal["eval-factory/trace-probe-diagnostic/v1"] = (
        "eval-factory/trace-probe-diagnostic/v1"
    )
    code: Identifier
    message: str = Field(min_length=1, max_length=2000)
    line_number: int | None = Field(default=None, ge=1)


class TraceProbeResult(ContractModelV2):
    schema_version: Literal["eval-factory/trace-probe-result/v1"] = "eval-factory/trace-probe-result/v1"
    adapter_name: Literal["raw_traj_v1"] = "raw_traj_v1"
    adapter_version: Literal["1.0.0"] = "1.0.0"
    source_uri: str = Field(min_length=3, max_length=1024)
    status: TraceProbeStatus
    raw_sha256: Sha256 | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    record_count: int = Field(ge=0)
    outer_fields: tuple[str, ...] = ()
    diagnostics: tuple[TraceProbeDiagnostic, ...] = ()

    @model_validator(mode="after")
    def validate_supported(self) -> TraceProbeResult:
        if self.status is TraceProbeStatus.SUPPORTED:
            if self.raw_sha256 is None or self.size_bytes is None:
                raise ValueError("supported probe requires exact source hash and size")
            if self.record_count < 1:
                raise ValueError("supported probe requires at least one record")
            if self.diagnostics:
                raise ValueError("supported probe cannot contain diagnostics")
        elif self.status is TraceProbeStatus.INVALID and not self.diagnostics:
            raise ValueError("invalid probe requires diagnostics")
        return self


class TraceProbeResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/trace-probe-result/v2"] = "eval-factory/trace-probe-result/v2"
    adapter_name: Identifier
    adapter_version: str = Field(min_length=1, max_length=128)
    source_uri: str = Field(min_length=3, max_length=1024)
    status: TraceProbeStatus
    raw_sha256: Sha256 | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    record_count: int = Field(ge=0)
    outer_fields: tuple[str, ...] = ()
    diagnostics: tuple[TraceProbeDiagnostic, ...] = ()

    @model_validator(mode="after")
    def validate_supported(self) -> TraceProbeResultV2:
        if self.status is TraceProbeStatus.SUPPORTED:
            if self.raw_sha256 is None or self.size_bytes is None:
                raise ValueError("supported probe requires exact source hash and size")
            if self.record_count < 1:
                raise ValueError("supported probe requires at least one record")
            if self.diagnostics:
                raise ValueError("supported probe cannot contain diagnostics")
        elif self.status is TraceProbeStatus.INVALID and not self.diagnostics:
            raise ValueError("invalid probe requires diagnostics")
        return self


class RegisteredTraceSource(ContractModelV2):
    schema_version: Literal["eval-factory/registered-trace-source/v1"] = (
        "eval-factory/registered-trace-source/v1"
    )
    source: TraceSourceRef
    size_bytes: int = Field(ge=0)
    record_count: int = Field(ge=1)
    outer_fields: tuple[str, ...] = Field(min_length=1)
    registration_sha256: Sha256

    @model_validator(mode="after")
    def validate_registration_hash(self) -> RegisteredTraceSource:
        observed = registered_trace_source_sha256(
            self.source,
            self.size_bytes,
            self.record_count,
            self.outer_fields,
        )
        if observed != self.registration_sha256:
            raise ValueError(
                f"registration hash mismatch: expected {self.registration_sha256}, observed {observed}"
            )
        return self
