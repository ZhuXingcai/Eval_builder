from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Protocol

from env_mock_agent.facade.contracts import FacadeObjectRef
from env_mock_agent.facade.runtime_v2 import (
    RuntimeCredentialStatusV2,
    RuntimeEventKindV2,
    RuntimeFailureCodeV2,
    RuntimeFeatureV2,
    RuntimeReadinessReasonV2,
    RuntimeReadinessStatusV2,
    RuntimeSandboxEnforcementV2,
    UnifiedRuntimeEventV2,
    UnifiedRuntimeReadinessV2,
    UnifiedRuntimeUsageV2,
)
from env_mock_agent.facade.telemetry_v2 import (
    ReportedCostV2,
    normalize_reported_cost_usd,
    normalize_tool_family,
    normalize_tool_family_counts,
)
from env_mock_agent.runtimes.base import AgentRuntime
from env_mock_agent.schemas import (
    RuntimeCapabilities,
    RuntimeErrorCode,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeRequest,
)


class RuntimeProtocolError(RuntimeError):
    pass


class RuntimeEventPrivateSink(Protocol):
    def put_runtime_event(
        self,
        payload: bytes,
    ) -> FacadeObjectRef: ...


def runtime_readiness_from_legacy(
    capabilities: RuntimeCapabilities,
    *,
    observed_at: datetime,
) -> UnifiedRuntimeReadinessV2:
    features: set[RuntimeFeatureV2] = {RuntimeFeatureV2.USAGE}
    if capabilities.supports_event_streaming:
        features.add(RuntimeFeatureV2.EVENT_STREAMING)
    if capabilities.tools:
        features.add(RuntimeFeatureV2.TOOL_EVENTS)
    if capabilities.supports_tool_progress:
        features.add(RuntimeFeatureV2.TOOL_PROGRESS)
    if capabilities.supports_resume:
        features.add(RuntimeFeatureV2.RESUME)
    if capabilities.supports_cancel:
        features.add(RuntimeFeatureV2.CANCEL)

    credential_status = RuntimeCredentialStatusV2(
        capabilities.credential_status.value,
    )
    sandbox_enforcement = RuntimeSandboxEnforcementV2(
        capabilities.sandbox_enforcement.value,
    )
    reasons: set[RuntimeReadinessReasonV2] = set()
    if not capabilities.available:
        status = RuntimeReadinessStatusV2.BLOCKED
        reasons.add(RuntimeReadinessReasonV2.PROTOCOL_UNAVAILABLE)
    else:
        if credential_status is RuntimeCredentialStatusV2.UNKNOWN:
            reasons.add(RuntimeReadinessReasonV2.CREDENTIAL_NOT_PROBED)
        elif credential_status is RuntimeCredentialStatusV2.MISSING:
            reasons.add(RuntimeReadinessReasonV2.CREDENTIAL_MISSING)
        elif credential_status is RuntimeCredentialStatusV2.INVALID:
            reasons.add(RuntimeReadinessReasonV2.CREDENTIAL_INVALID)
        if RuntimeFeatureV2.EVENT_STREAMING not in features:
            reasons.add(
                RuntimeReadinessReasonV2.EVENT_STREAMING_UNAVAILABLE,
            )
        if sandbox_enforcement is RuntimeSandboxEnforcementV2.NONE:
            reasons.add(RuntimeReadinessReasonV2.SANDBOX_UNAVAILABLE)
        elif sandbox_enforcement is RuntimeSandboxEnforcementV2.PARTIAL:
            reasons.add(RuntimeReadinessReasonV2.SANDBOX_PARTIAL)
        blocked = credential_status in {
            RuntimeCredentialStatusV2.MISSING,
            RuntimeCredentialStatusV2.INVALID,
        }
        status = (
            RuntimeReadinessStatusV2.BLOCKED
            if blocked
            else (RuntimeReadinessStatusV2.DEGRADED if reasons else RuntimeReadinessStatusV2.READY)
        )

    return UnifiedRuntimeReadinessV2(
        runtime_id=capabilities.name.value,
        runtime_version=capabilities.version,
        status=status,
        credential_status=credential_status,
        features=tuple(sorted(features, key=lambda item: item.value)),
        tool_families=tuple(item.tool_family for item in normalize_tool_family_counts(capabilities.tools)),
        sandbox_enforcement=sandbox_enforcement,
        reason_codes=tuple(sorted(reasons, key=lambda item: item.value)),
        observed_at=observed_at,
    )


class LegacyRuntimeProtocolAdapter:
    def __init__(
        self,
        *,
        runtime: AgentRuntime,
        readiness: UnifiedRuntimeReadinessV2,
        private_sink: RuntimeEventPrivateSink,
    ) -> None:
        self._runtime = runtime
        self._readiness = readiness
        self._private_sink = private_sink

    async def run(
        self,
        request: RuntimeRequest,
    ) -> AsyncIterator[UnifiedRuntimeEventV2]:
        expected_sequence = 1
        started = False
        usage_seen = False
        terminal_seen = False
        async for source in self._runtime.run(request):
            if terminal_seen:
                raise RuntimeProtocolError(
                    "runtime emitted an event after its terminal event",
                )
            if source.sequence != expected_sequence:
                raise RuntimeProtocolError(
                    "runtime event sequence must be contiguous from one",
                )
            if (
                source.run_id != request.run_id
                or source.artifact_id != request.artifact_id
                or source.runtime.value != self._readiness.runtime_id
            ):
                raise RuntimeProtocolError(
                    "runtime event identity differs from the selected request",
                )
            kind = _event_kind(source.event_type)
            if expected_sequence == 1 and kind is not RuntimeEventKindV2.RUN_STARTED:
                raise RuntimeProtocolError("runtime stream must start with RUN_STARTED")
            if kind is RuntimeEventKindV2.RUN_STARTED:
                if started:
                    raise RuntimeProtocolError("runtime stream contains duplicate RUN_STARTED")
                started = True
            if kind is RuntimeEventKindV2.USAGE_REPORTED:
                usage_seen = True
            if kind in {
                RuntimeEventKindV2.RUN_COMPLETED,
                RuntimeEventKindV2.RUN_FAILED,
            }:
                if not usage_seen:
                    raise RuntimeProtocolError(
                        "runtime terminal event requires prior usage",
                    )
                terminal_seen = True
            private_ref = self._private_sink.put_runtime_event(
                source.model_dump_json(exclude_none=False).encode(),
            )
            if (
                private_ref.object_type != "runtime-event-content"
                or private_ref.object_version != "private-v1"
            ):
                raise RuntimeProtocolError(
                    "runtime event sink returned an invalid private reference",
                )
            yield _normalize_event(
                source,
                readiness=self._readiness,
                kind=kind,
                private_ref=private_ref,
            )
            expected_sequence += 1
        if not started or not terminal_seen:
            raise RuntimeProtocolError(
                "runtime stream requires one start and one terminal event",
            )

    async def cancel(self, run_handle: str) -> None:
        await self._runtime.cancel(run_handle)


def _normalize_event(
    source: RuntimeEvent,
    *,
    readiness: UnifiedRuntimeReadinessV2,
    kind: RuntimeEventKindV2,
    private_ref: FacadeObjectRef,
) -> UnifiedRuntimeEventV2:
    tool_event = kind in {
        RuntimeEventKindV2.TOOL_CALL_STARTED,
        RuntimeEventKindV2.TOOL_CALL_PROGRESS,
        RuntimeEventKindV2.TOOL_CALL_COMPLETED,
    }
    tool_call_sha256: str | None = None
    tool_family = None
    if tool_event:
        if not source.tool_call_id:
            raise RuntimeProtocolError("tool event requires a tool call identity")
        tool_call_sha256 = hashlib.sha256(
            f"{source.runtime.value}\0{source.tool_call_id}".encode(),
        ).hexdigest()
        tool_family = normalize_tool_family(source.tool_name or "")
    usage = _usage(source) if kind is RuntimeEventKindV2.USAGE_REPORTED else None
    failure_code = _failure_code(source.error_code) if kind is RuntimeEventKindV2.RUN_FAILED else None
    return UnifiedRuntimeEventV2.create(
        run_id=source.run_id,
        work_id=source.artifact_id,
        runtime_id=readiness.runtime_id,
        runtime_version=readiness.runtime_version,
        sequence=source.sequence,
        kind=kind,
        occurred_at=source.timestamp,
        source_refs=(private_ref,),
        tool_call_sha256=tool_call_sha256,
        tool_family=tool_family,
        usage=usage,
        failure_code=failure_code,
    )


def _usage(source: RuntimeEvent) -> UnifiedRuntimeUsageV2:
    if source.usage is None:
        raise RuntimeProtocolError("usage event has no usage payload")
    return UnifiedRuntimeUsageV2(
        model_requests=max(source.usage.turns, 1),
        input_tokens=source.usage.input_tokens,
        output_tokens=source.usage.output_tokens,
        cache_creation_input_tokens=(source.usage.cache_creation_input_tokens),
        cache_read_input_tokens=source.usage.cache_read_input_tokens,
        tool_calls=source.usage.tool_calls,
        turns=source.usage.turns,
        duration_ms=source.usage.duration_ms,
        reported_cost=(
            normalize_reported_cost_usd(source.usage.cost_usd)
            if source.usage.cost_usd is not None
            else ReportedCostV2.unavailable()
        ),
    )


def _event_kind(value: RuntimeEventType) -> RuntimeEventKindV2:
    return {
        RuntimeEventType.RUNTIME_STARTED: RuntimeEventKindV2.RUN_STARTED,
        RuntimeEventType.MESSAGE_DELTA: RuntimeEventKindV2.MODEL_OUTPUT_DELTA,
        RuntimeEventType.TOOL_STARTED: RuntimeEventKindV2.TOOL_CALL_STARTED,
        RuntimeEventType.TOOL_PROGRESS: RuntimeEventKindV2.TOOL_CALL_PROGRESS,
        RuntimeEventType.TOOL_FINISHED: RuntimeEventKindV2.TOOL_CALL_COMPLETED,
        RuntimeEventType.ARTIFACT_WRITTEN: RuntimeEventKindV2.ARTIFACT_CHANGED,
        RuntimeEventType.WARNING: RuntimeEventKindV2.WARNING,
        RuntimeEventType.USAGE: RuntimeEventKindV2.USAGE_REPORTED,
        RuntimeEventType.RUNTIME_FINISHED: RuntimeEventKindV2.RUN_COMPLETED,
        RuntimeEventType.RUNTIME_FAILED: RuntimeEventKindV2.RUN_FAILED,
    }[value]


def _failure_code(
    value: RuntimeErrorCode | None,
) -> RuntimeFailureCodeV2:
    if value is None:
        return RuntimeFailureCodeV2.UNKNOWN
    return {
        RuntimeErrorCode.TIMEOUT: RuntimeFailureCodeV2.TIMEOUT,
        RuntimeErrorCode.CANCELLED: RuntimeFailureCodeV2.CANCELLED,
        RuntimeErrorCode.AUTH_UNAVAILABLE: (RuntimeFailureCodeV2.AUTHENTICATION_FAILED),
        RuntimeErrorCode.MODEL_UNAVAILABLE: (RuntimeFailureCodeV2.MODEL_UNAVAILABLE),
        RuntimeErrorCode.TOOL_DENIED: RuntimeFailureCodeV2.TOOL_DENIED,
        RuntimeErrorCode.PATH_ESCAPE: RuntimeFailureCodeV2.TOOL_DENIED,
        RuntimeErrorCode.BUDGET_EXCEEDED: (RuntimeFailureCodeV2.INVALID_REQUEST),
        RuntimeErrorCode.PROTOCOL_ERROR: RuntimeFailureCodeV2.PROTOCOL_ERROR,
        RuntimeErrorCode.PROCESS_ERROR: RuntimeFailureCodeV2.TRANSPORT_FAILED,
    }[value]


__all__ = [
    "LegacyRuntimeProtocolAdapter",
    "RuntimeEventPrivateSink",
    "RuntimeProtocolError",
    "runtime_readiness_from_legacy",
]
