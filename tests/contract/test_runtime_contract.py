from __future__ import annotations

from pathlib import Path

import pytest

from env_mock_agent.runtimes import FakeRuntime
from env_mock_agent.schemas import (
    ModelProfile,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeName,
    RuntimeRequest,
    RuntimeResumeRequest,
)


def make_request(workspace: Path) -> RuntimeRequest:
    return RuntimeRequest(
        run_id="run-1",
        artifact_id="artifact-1",
        role="builder",
        workspace=str(workspace),
        prompt="Create the input fixture.",
        model_profile=ModelProfile(provider="fake", model="fake"),
    )


async def collect(runtime: FakeRuntime, request: RuntimeRequest) -> list[RuntimeEvent]:
    return [event async for event in runtime.run(request)]


def assert_event_contract(events: list[RuntimeEvent]) -> None:
    assert sum(event.event_type == RuntimeEventType.RUNTIME_STARTED for event in events) == 1
    terminals = [
        event
        for event in events
        if event.event_type in {RuntimeEventType.RUNTIME_FINISHED, RuntimeEventType.RUNTIME_FAILED}
    ]
    assert len(terminals) == 1
    assert terminals[0] == events[-1]
    assert any(event.event_type == RuntimeEventType.USAGE for event in events)
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert all(event.run_handle for event in events)


@pytest.mark.asyncio
async def test_fake_runtime_satisfies_shared_event_contract(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    events = await collect(runtime, make_request(tmp_path))
    assert_event_contract(events)
    artifacts = [event for event in events if event.event_type == RuntimeEventType.ARTIFACT_WRITTEN]
    assert [event.message for event in artifacts] == ["fake-runtime.txt"]
    assert artifacts[0].payload["sha256"]


@pytest.mark.asyncio
async def test_fake_runtime_resumes_from_persisted_session(tmp_path: Path) -> None:
    request = make_request(tmp_path)
    first_events = await collect(FakeRuntime(), request)
    session = first_events[-1].session
    assert session is not None

    resumed_events = [
        event
        async for event in FakeRuntime().resume(
            RuntimeResumeRequest(
                request=request,
                session=session,
                prompt="Continue the input fixture.",
            )
        )
    ]
    assert_event_contract(resumed_events)
    assert resumed_events[-1].message == "resumed"
    assert resumed_events[-1].session == session


@pytest.mark.asyncio
async def test_fake_runtime_cancel_is_addressed_by_run_handle(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    handle = "fake:run-1:artifact-1"
    await runtime.cancel(handle)
    assert handle in runtime.cancelled


@pytest.mark.asyncio
async def test_fake_probe_declares_real_resume_support() -> None:
    capabilities = await FakeRuntime().probe()
    assert capabilities.name == RuntimeName.FAKE
    assert capabilities.available is True
    assert capabilities.supports_resume is True
