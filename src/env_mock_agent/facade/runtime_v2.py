from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from env_mock_agent.facade.contracts import (
    FacadeModel,
    FacadeObjectRef,
    Identifier,
    Sha256,
)
from env_mock_agent.facade.telemetry_v2 import (
    ReportedCostV2,
    ToolFamilyV2,
)

RUNTIME_PROTOCOL_VERSION: Literal["runtime-protocol/stage5-0-v1"] = "runtime-protocol/stage5-0-v1"


class RuntimeReadinessStatusV2(StrEnum):
    READY = "READY"
    DEGRADED = "DEGRADED"
    BLOCKED = "BLOCKED"


class RuntimeCredentialStatusV2(StrEnum):
    READY = "READY"
    NOT_REQUIRED = "NOT_REQUIRED"
    UNKNOWN = "UNKNOWN"
    MISSING = "MISSING"
    INVALID = "INVALID"


class RuntimeSandboxEnforcementV2(StrEnum):
    FULL = "FULL"
    PARTIAL = "PARTIAL"
    NONE = "NONE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class RuntimeFeatureV2(StrEnum):
    CANCEL = "CANCEL"
    EVENT_STREAMING = "EVENT_STREAMING"
    RESUME = "RESUME"
    TOOL_EVENTS = "TOOL_EVENTS"
    TOOL_PROGRESS = "TOOL_PROGRESS"
    USAGE = "USAGE"


class RuntimeReadinessReasonV2(StrEnum):
    CREDENTIAL_INVALID = "CREDENTIAL_INVALID"
    CREDENTIAL_MISSING = "CREDENTIAL_MISSING"
    CREDENTIAL_NOT_PROBED = "CREDENTIAL_NOT_PROBED"
    EVENT_STREAMING_UNAVAILABLE = "EVENT_STREAMING_UNAVAILABLE"
    PROTOCOL_UNAVAILABLE = "PROTOCOL_UNAVAILABLE"
    SANDBOX_PARTIAL = "SANDBOX_PARTIAL"
    SANDBOX_UNAVAILABLE = "SANDBOX_UNAVAILABLE"


class RuntimeEventKindV2(StrEnum):
    RUN_STARTED = "RUN_STARTED"
    MODEL_REQUEST_STARTED = "MODEL_REQUEST_STARTED"
    MODEL_OUTPUT_DELTA = "MODEL_OUTPUT_DELTA"
    MODEL_RESPONSE_AVAILABLE = "MODEL_RESPONSE_AVAILABLE"
    TOOL_CALL_STARTED = "TOOL_CALL_STARTED"
    TOOL_CALL_PROGRESS = "TOOL_CALL_PROGRESS"
    TOOL_CALL_COMPLETED = "TOOL_CALL_COMPLETED"
    ARTIFACT_CHANGED = "ARTIFACT_CHANGED"
    WARNING = "WARNING"
    USAGE_REPORTED = "USAGE_REPORTED"
    RUN_COMPLETED = "RUN_COMPLETED"
    RUN_FAILED = "RUN_FAILED"


class RuntimeFailureCodeV2(StrEnum):
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    CANCELLED = "CANCELLED"
    CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"
    INVALID_REQUEST = "INVALID_REQUEST"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    OVERLOADED = "OVERLOADED"
    PROTOCOL_ERROR = "PROTOCOL_ERROR"
    PROVIDER_FAILED = "PROVIDER_FAILED"
    RATE_LIMITED = "RATE_LIMITED"
    SANDBOX_UNAVAILABLE = "SANDBOX_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    TOOL_DENIED = "TOOL_DENIED"
    TOOL_FAILED = "TOOL_FAILED"
    TRANSPORT_FAILED = "TRANSPORT_FAILED"
    UNKNOWN = "UNKNOWN"


class UnifiedRuntimeUsageV2(FacadeModel):
    schema_version: Literal["env-mock-agent/unified-runtime-usage/v2"] = (
        "env-mock-agent/unified-runtime-usage/v2"
    )
    model_requests: int = Field(ge=0, le=10_000_000)
    input_tokens: int = Field(ge=0, le=10_000_000_000)
    output_tokens: int = Field(ge=0, le=10_000_000_000)
    cache_creation_input_tokens: int = Field(ge=0, le=10_000_000_000)
    cache_read_input_tokens: int = Field(ge=0, le=10_000_000_000)
    tool_calls: int = Field(ge=0, le=10_000_000)
    turns: int = Field(ge=0, le=10_000_000)
    duration_ms: int | None = Field(default=None, ge=0, le=10**15)
    reported_cost: ReportedCostV2


class UnifiedRuntimeReadinessV2(FacadeModel):
    schema_version: Literal["env-mock-agent/unified-runtime-readiness/v2"] = (
        "env-mock-agent/unified-runtime-readiness/v2"
    )
    runtime_id: Identifier
    runtime_version: str | None = Field(default=None, min_length=1, max_length=256)
    status: RuntimeReadinessStatusV2
    credential_status: RuntimeCredentialStatusV2
    features: tuple[RuntimeFeatureV2, ...] = Field(default=(), max_length=32)
    tool_families: tuple[ToolFamilyV2, ...] = Field(default=(), max_length=16)
    sandbox_enforcement: RuntimeSandboxEnforcementV2
    reason_codes: tuple[RuntimeReadinessReasonV2, ...] = Field(
        default=(),
        max_length=16,
    )
    observed_at: datetime
    protocol_version: Literal["runtime-protocol/stage5-0-v1"] = RUNTIME_PROTOCOL_VERSION

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("runtime readiness timestamp must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_readiness(self) -> Self:
        if self.features != tuple(sorted(set(self.features), key=lambda item: item.value)):
            raise ValueError("runtime features must be sorted and unique")
        if self.tool_families != tuple(
            family for family in ToolFamilyV2 if family in set(self.tool_families)
        ):
            raise ValueError("runtime tool families must use canonical order")
        if self.reason_codes != tuple(sorted(set(self.reason_codes), key=lambda item: item.value)):
            raise ValueError("runtime readiness reasons must be sorted and unique")
        if self.status is RuntimeReadinessStatusV2.READY:
            if self.credential_status not in {
                RuntimeCredentialStatusV2.READY,
                RuntimeCredentialStatusV2.NOT_REQUIRED,
            }:
                raise ValueError("READY runtime requires verified or unnecessary credentials")
            if self.reason_codes:
                raise ValueError("READY runtime cannot carry readiness reasons")
        elif not self.reason_codes:
            raise ValueError("non-ready runtime requires readiness reasons")
        if (
            self.credential_status
            in {
                RuntimeCredentialStatusV2.MISSING,
                RuntimeCredentialStatusV2.INVALID,
            }
            and self.status is not RuntimeReadinessStatusV2.BLOCKED
        ):
            raise ValueError("missing or invalid credentials must block runtime readiness")
        return self


class UnifiedRuntimeEventV2(FacadeModel):
    schema_version: Literal["env-mock-agent/unified-runtime-event/v2"] = (
        "env-mock-agent/unified-runtime-event/v2"
    )
    runtime_event_id: Identifier
    runtime_event_sha256: Sha256
    run_id: Identifier
    work_id: Identifier
    runtime_id: Identifier
    runtime_version: str | None = Field(default=None, min_length=1, max_length=256)
    sequence: int = Field(ge=1, le=10_000_000)
    kind: RuntimeEventKindV2
    occurred_at: datetime
    source_refs: tuple[FacadeObjectRef, ...] = Field(min_length=1, max_length=16)
    tool_call_sha256: Sha256 | None = None
    tool_family: ToolFamilyV2 | None = None
    usage: UnifiedRuntimeUsageV2 | None = None
    failure_code: RuntimeFailureCodeV2 | None = None
    protocol_version: Literal["runtime-protocol/stage5-0-v1"] = RUNTIME_PROTOCOL_VERSION

    @field_validator("occurred_at")
    @classmethod
    def validate_occurred_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("runtime event timestamp must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_event(self) -> Self:
        if self.source_refs != _sorted_refs(self.source_refs):
            raise ValueError("runtime event source refs must be sorted and unique")
        tool_event = self.kind in {
            RuntimeEventKindV2.TOOL_CALL_STARTED,
            RuntimeEventKindV2.TOOL_CALL_PROGRESS,
            RuntimeEventKindV2.TOOL_CALL_COMPLETED,
        }
        has_tool_call = self.tool_call_sha256 is not None
        has_tool_family = self.tool_family is not None
        if tool_event and not (has_tool_call and has_tool_family):
            raise ValueError("tool event fields must exactly match a tool event")
        if not tool_event and (has_tool_call or has_tool_family):
            raise ValueError("tool event fields must exactly match a tool event")
        if (self.kind is RuntimeEventKindV2.USAGE_REPORTED) != (self.usage is not None):
            raise ValueError("usage is allowed only on USAGE_REPORTED")
        if (self.kind is RuntimeEventKindV2.RUN_FAILED) != (self.failure_code is not None):
            raise ValueError("failure code must exactly match RUN_FAILED")
        validate_unified_runtime_event_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        work_id: str,
        runtime_id: str,
        runtime_version: str | None,
        sequence: int,
        kind: RuntimeEventKindV2,
        occurred_at: datetime,
        source_refs: tuple[FacadeObjectRef, ...],
        tool_call_sha256: str | None = None,
        tool_family: ToolFamilyV2 | None = None,
        usage: UnifiedRuntimeUsageV2 | None = None,
        failure_code: RuntimeFailureCodeV2 | None = None,
    ) -> UnifiedRuntimeEventV2:
        value = cls(
            runtime_event_id="runtime-event://pending",
            runtime_event_sha256="0" * 64,
            run_id=run_id,
            work_id=work_id,
            runtime_id=runtime_id,
            runtime_version=runtime_version,
            sequence=sequence,
            kind=kind,
            occurred_at=occurred_at,
            source_refs=_sorted_refs(source_refs),
            tool_call_sha256=tool_call_sha256,
            tool_family=tool_family,
            usage=usage,
            failure_code=failure_code,
        )
        digest = unified_runtime_event_carried_sha256(value)
        return value.model_copy(
            update={
                "runtime_event_id": f"runtime-event://sha256/{digest}",
                "runtime_event_sha256": digest,
            }
        )

    def to_ref(self) -> FacadeObjectRef:
        return FacadeObjectRef(
            object_type="runtime-event",
            object_id=self.runtime_event_id,
            object_version="v2",
            object_sha256=self.runtime_event_sha256,
        )


def unified_runtime_event_carried_sha256(
    value: UnifiedRuntimeEventV2,
) -> str:
    payload = value.model_dump(
        mode="json",
        exclude={
            "runtime_event_id",
            "runtime_event_sha256",
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


def validate_unified_runtime_event_identity(
    value: UnifiedRuntimeEventV2,
) -> None:
    if value.runtime_event_id == "runtime-event://pending" and value.runtime_event_sha256 == "0" * 64:
        return
    digest = unified_runtime_event_carried_sha256(value)
    if value.runtime_event_id != f"runtime-event://sha256/{digest}" or value.runtime_event_sha256 != digest:
        raise ValueError("unified runtime event identity is stale or invalid")


def _ref_key(value: FacadeObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _sorted_refs(
    values: tuple[FacadeObjectRef, ...],
) -> tuple[FacadeObjectRef, ...]:
    keys = tuple(_ref_key(value) for value in values)
    if len(keys) != len(set(keys)):
        raise ValueError("runtime event source refs must be unique")
    return tuple(sorted(values, key=_ref_key))


__all__ = [
    "RUNTIME_PROTOCOL_VERSION",
    "RuntimeCredentialStatusV2",
    "RuntimeEventKindV2",
    "RuntimeFailureCodeV2",
    "RuntimeFeatureV2",
    "RuntimeReadinessReasonV2",
    "RuntimeReadinessStatusV2",
    "RuntimeSandboxEnforcementV2",
    "UnifiedRuntimeEventV2",
    "UnifiedRuntimeReadinessV2",
    "UnifiedRuntimeUsageV2",
    "unified_runtime_event_carried_sha256",
    "validate_unified_runtime_event_identity",
]
