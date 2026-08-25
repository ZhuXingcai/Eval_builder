from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from env_mock_agent.facade import FacadeObjectRef
from env_mock_agent.facade.runtime_adapter import (
    LegacyRuntimeProtocolAdapter,
    runtime_readiness_from_legacy,
)
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
    ToolFamilyV2,
)
from env_mock_agent.runtimes import FakeRuntime
from env_mock_agent.runtimes.base import AgentRuntime, RuntimeEventSequencer
from env_mock_agent.schemas import (
    ModelProfile,
    RuntimeCapabilities,
    RuntimeCredentialStatus,
    RuntimeErrorCode,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeName,
    RuntimeRequest,
    RuntimeSandboxEnforcement,
    RuntimeUsage,
)

NOW = datetime(2026, 8, 23, tzinfo=UTC)


def _ref(suffix: str) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type="runtime-event-content",
        object_id=f"runtime-event-content://sha256/{suffix}",
        object_version="private-v1",
        object_sha256=suffix,
    )


def _request(tmp_path: Path) -> RuntimeRequest:
    return RuntimeRequest(
        run_id="run-stage5",
        artifact_id="artifact-stage5",
        role="builder",
        workspace=str(tmp_path),
        prompt="private prompt",
        model_profile=ModelProfile(provider="fake", model="fake"),
    )


class _Sink:
    def __init__(self) -> None:
        self.payloads: list[bytes] = []

    def put_runtime_event(self, payload: bytes) -> FacadeObjectRef:
        self.payloads.append(payload)
        return _ref(hashlib.sha256(payload).hexdigest())


class _SequenceGapRuntime(AgentRuntime):
    async def probe(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            name=RuntimeName.FAKE,
            available=True,
            version="1",
            credential_status="NOT_REQUIRED",
        )

    async def run(self, request: RuntimeRequest) -> AsyncIterator[RuntimeEvent]:
        sequencer = RuntimeEventSequencer(
            request,
            RuntimeName.FAKE,
            "fake:run-stage5:artifact-stage5",
        )
        yield sequencer.event(RuntimeEventType.RUNTIME_STARTED)
        sequencer.event(RuntimeEventType.WARNING)
        yield sequencer.event(RuntimeEventType.RUNTIME_FINISHED)

    async def cancel(self, run_handle: str) -> None:
        return None


class _ScriptedRuntime(AgentRuntime):
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.cancelled: list[str] = []

    async def probe(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            name=RuntimeName.FAKE,
            available=True,
            version="1",
            credential_status="NOT_REQUIRED",
            sandbox_enforcement="NOT_APPLICABLE",
        )

    async def run(self, request: RuntimeRequest) -> AsyncIterator[RuntimeEvent]:
        events = RuntimeEventSequencer(request, RuntimeName.FAKE, "scripted-handle")
        if self.mode == "wrong-start":
            yield events.event(RuntimeEventType.WARNING)
            return
        started = events.event(RuntimeEventType.RUNTIME_STARTED)
        if self.mode == "wrong-identity":
            started = started.model_copy(update={"run_id": "other-run"})
        yield started
        if self.mode == "no-terminal":
            yield events.event(RuntimeEventType.USAGE, usage=RuntimeUsage(turns=1))
            return
        if self.mode == "no-usage":
            yield events.event(RuntimeEventType.RUNTIME_FINISHED)
            return
        if self.mode == "missing-tool-id":
            yield events.event(RuntimeEventType.TOOL_STARTED, tool_name="write")
            return
        if self.mode == "failure":
            yield events.event(
                RuntimeEventType.TOOL_STARTED,
                tool_name="bash",
                tool_call_id="call-1",
            )
            yield events.event(
                RuntimeEventType.TOOL_PROGRESS,
                tool_name="bash",
                tool_call_id="call-1",
            )
            yield events.event(
                RuntimeEventType.TOOL_FINISHED,
                tool_name="bash",
                tool_call_id="call-1",
            )
            yield events.event(
                RuntimeEventType.USAGE,
                usage=RuntimeUsage(turns=1, cost_usd=0.000001),
            )
            yield events.event(
                RuntimeEventType.RUNTIME_FAILED,
                error_code=RuntimeErrorCode.AUTH_UNAVAILABLE,
            )
            return
        yield events.event(RuntimeEventType.USAGE, usage=RuntimeUsage(turns=1))
        yield events.event(RuntimeEventType.RUNTIME_FINISHED)
        if self.mode == "after-terminal":
            yield events.event(RuntimeEventType.WARNING)

    async def cancel(self, run_handle: str) -> None:
        self.cancelled.append(run_handle)


class _InvalidSink:
    def put_runtime_event(self, payload: bytes) -> FacadeObjectRef:
        return FacadeObjectRef(
            object_type="wrong",
            object_id="wrong://runtime-event",
            object_version="v1",
            object_sha256=hashlib.sha256(payload).hexdigest(),
        )


def test_runtime_readiness_never_promotes_unknown_credentials() -> None:
    degraded = runtime_readiness_from_legacy(
        RuntimeCapabilities(
            name=RuntimeName.PI_RPC,
            available=True,
            version="0.80.10",
            tools=["read", "bash"],
            supports_resume=True,
            credential_status="UNKNOWN",
            sandbox_enforcement="NONE",
        ),
        observed_at=NOW,
    )

    assert degraded.status is RuntimeReadinessStatusV2.DEGRADED
    assert degraded.credential_status is RuntimeCredentialStatusV2.UNKNOWN
    assert degraded.reason_codes == (
        RuntimeReadinessReasonV2.CREDENTIAL_NOT_PROBED,
        RuntimeReadinessReasonV2.SANDBOX_UNAVAILABLE,
    )
    assert RuntimeFeatureV2.RESUME in degraded.features

    with pytest.raises(ValidationError, match="READY runtime"):
        UnifiedRuntimeReadinessV2(
            runtime_id="pi_rpc",
            runtime_version="0.80.10",
            status=RuntimeReadinessStatusV2.READY,
            credential_status=RuntimeCredentialStatusV2.UNKNOWN,
            features=(RuntimeFeatureV2.EVENT_STREAMING,),
            tool_families=(),
            sandbox_enforcement=RuntimeSandboxEnforcementV2.NONE,
            reason_codes=(),
            observed_at=NOW,
        )


def test_runtime_event_is_strict_hash_bound_and_closed() -> None:
    usage = UnifiedRuntimeUsageV2(
        model_requests=1,
        input_tokens=10,
        output_tokens=5,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        tool_calls=0,
        turns=1,
        duration_ms=20,
        reported_cost=ReportedCostV2.unavailable(),
    )
    event = UnifiedRuntimeEventV2.create(
        run_id="run-stage5",
        work_id="artifact-stage5",
        runtime_id="gateway",
        runtime_version="v1",
        sequence=3,
        kind=RuntimeEventKindV2.USAGE_REPORTED,
        occurred_at=NOW,
        source_refs=(_ref("a" * 64),),
        usage=usage,
    )
    assert event.runtime_event_id.endswith(event.runtime_event_sha256)

    stale = event.model_dump(mode="python")
    stale["runtime_event_sha256"] = "b" * 64
    with pytest.raises(ValidationError, match="identity"):
        UnifiedRuntimeEventV2.model_validate(stale)

    extra = event.model_dump(mode="python")
    extra["raw_payload"] = {"secret": "not allowed"}
    with pytest.raises(ValidationError):
        UnifiedRuntimeEventV2.model_validate(extra)

    with pytest.raises(ValidationError, match="tool event"):
        UnifiedRuntimeEventV2.create(
            run_id="run-stage5",
            work_id="artifact-stage5",
            runtime_id="pi_rpc",
            runtime_version="0.80.10",
            sequence=2,
            kind=RuntimeEventKindV2.TOOL_CALL_STARTED,
            occurred_at=NOW,
            source_refs=(_ref("a" * 64),),
        )


@pytest.mark.parametrize(
    ("credential", "sandbox", "available", "expected"),
    [
        (
            RuntimeCredentialStatus.MISSING,
            RuntimeSandboxEnforcement.FULL,
            True,
            RuntimeReadinessStatusV2.BLOCKED,
        ),
        (
            RuntimeCredentialStatus.INVALID,
            RuntimeSandboxEnforcement.FULL,
            True,
            RuntimeReadinessStatusV2.BLOCKED,
        ),
        (
            RuntimeCredentialStatus.READY,
            RuntimeSandboxEnforcement.PARTIAL,
            True,
            RuntimeReadinessStatusV2.DEGRADED,
        ),
        (
            RuntimeCredentialStatus.READY,
            RuntimeSandboxEnforcement.FULL,
            False,
            RuntimeReadinessStatusV2.BLOCKED,
        ),
    ],
)
def test_runtime_readiness_closes_failure_dimensions(
    credential: RuntimeCredentialStatus,
    sandbox: RuntimeSandboxEnforcement,
    available: bool,
    expected: RuntimeReadinessStatusV2,
) -> None:
    readiness = runtime_readiness_from_legacy(
        RuntimeCapabilities(
            name=RuntimeName.CLAUDE_AGENT_SDK,
            available=available,
            version="1",
            supports_event_streaming=True,
            credential_status=credential,
            sandbox_enforcement=sandbox,
        ),
        observed_at=NOW,
    )
    assert readiness.status is expected


@pytest.mark.asyncio
async def test_legacy_runtime_adapter_emits_content_safe_protocol(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime()
    readiness = runtime_readiness_from_legacy(
        await runtime.probe(),
        observed_at=NOW,
    )
    sink = _Sink()
    events = [
        event
        async for event in LegacyRuntimeProtocolAdapter(
            runtime=runtime,
            readiness=readiness,
            private_sink=sink,
        ).run(_request(tmp_path))
    ]

    assert readiness.status is RuntimeReadinessStatusV2.READY
    assert readiness.credential_status is RuntimeCredentialStatusV2.NOT_REQUIRED
    assert [event.kind for event in events] == [
        RuntimeEventKindV2.RUN_STARTED,
        RuntimeEventKindV2.ARTIFACT_CHANGED,
        RuntimeEventKindV2.USAGE_REPORTED,
        RuntimeEventKindV2.RUN_COMPLETED,
    ]
    assert [event.sequence for event in events] == [1, 2, 3, 4]
    assert len(sink.payloads) == len(events)
    public_json = "\n".join(event.model_dump_json() for event in events)
    assert str(tmp_path) not in public_json
    assert "fake-runtime.txt" not in public_json
    assert "private prompt" not in public_json


@pytest.mark.asyncio
async def test_legacy_runtime_adapter_rejects_sequence_gaps(
    tmp_path: Path,
) -> None:
    runtime = _SequenceGapRuntime()
    readiness = runtime_readiness_from_legacy(
        await runtime.probe(),
        observed_at=NOW,
    )
    adapter = LegacyRuntimeProtocolAdapter(
        runtime=runtime,
        readiness=readiness,
        private_sink=_Sink(),
    )

    with pytest.raises(RuntimeError, match="contiguous"):
        _ = [event async for event in adapter.run(_request(tmp_path))]


@pytest.mark.asyncio
async def test_legacy_runtime_adapter_maps_tools_failure_cost_and_cancel(
    tmp_path: Path,
) -> None:
    runtime = _ScriptedRuntime("failure")
    readiness = runtime_readiness_from_legacy(
        await runtime.probe(),
        observed_at=NOW,
    )
    adapter = LegacyRuntimeProtocolAdapter(
        runtime=runtime,
        readiness=readiness,
        private_sink=_Sink(),
    )
    events = [event async for event in adapter.run(_request(tmp_path))]

    assert [event.kind for event in events[1:4]] == [
        RuntimeEventKindV2.TOOL_CALL_STARTED,
        RuntimeEventKindV2.TOOL_CALL_PROGRESS,
        RuntimeEventKindV2.TOOL_CALL_COMPLETED,
    ]
    assert all(event.tool_family is ToolFamilyV2.SHELL for event in events[1:4])
    assert events[4].usage is not None
    assert events[4].usage.reported_cost.amount_microusd == 1
    assert events[-1].failure_code is RuntimeFailureCodeV2.AUTHENTICATION_FAILED
    await adapter.cancel("scripted-handle")
    assert runtime.cancelled == ["scripted-handle"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("wrong-start", "start with RUN_STARTED"),
        ("wrong-identity", "identity differs"),
        ("no-usage", "requires prior usage"),
        ("no-terminal", "requires one start and one terminal"),
        ("missing-tool-id", "tool event requires"),
        ("after-terminal", "after its terminal"),
    ],
)
async def test_legacy_runtime_adapter_fails_closed_on_invalid_streams(
    tmp_path: Path,
    mode: str,
    message: str,
) -> None:
    runtime = _ScriptedRuntime(mode)
    readiness = runtime_readiness_from_legacy(
        await runtime.probe(),
        observed_at=NOW,
    )
    adapter = LegacyRuntimeProtocolAdapter(
        runtime=runtime,
        readiness=readiness,
        private_sink=_Sink(),
    )
    with pytest.raises(RuntimeError, match=message):
        _ = [event async for event in adapter.run(_request(tmp_path))]


@pytest.mark.asyncio
async def test_legacy_runtime_adapter_rejects_non_private_sink_ref(
    tmp_path: Path,
) -> None:
    runtime = _ScriptedRuntime("success")
    readiness = runtime_readiness_from_legacy(
        await runtime.probe(),
        observed_at=NOW,
    )
    adapter = LegacyRuntimeProtocolAdapter(
        runtime=runtime,
        readiness=readiness,
        private_sink=_InvalidSink(),
    )
    with pytest.raises(RuntimeError, match="invalid private reference"):
        _ = [event async for event in adapter.run(_request(tmp_path))]
