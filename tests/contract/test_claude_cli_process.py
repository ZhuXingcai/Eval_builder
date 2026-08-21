from __future__ import annotations

import time
from pathlib import Path

import pytest

from env_mock_agent.runtimes import ClaudeCodeCliRuntime
from env_mock_agent.schemas import (
    ModelProfile,
    RuntimeErrorCode,
    RuntimeEventType,
    RuntimeRequest,
    RuntimeResumeRequest,
)


def make_request(workspace: Path) -> RuntimeRequest:
    return RuntimeRequest(
        run_id="run",
        artifact_id="artifact",
        role="builder",
        workspace=str(workspace),
        prompt="Create an input.",
        model_profile=ModelProfile(provider="anthropic", model="test"),
        timeout_seconds=10,
    )


def write_executable(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\nset -eu\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_cli_command_terminates_variadic_tool_options_before_prompt(tmp_path: Path) -> None:
    request = make_request(tmp_path / "workspace").model_copy(update={"disallowed_tools": ["Bash", "Write"]})
    command = ClaudeCodeCliRuntime()._command(request, None)

    assert command[-2:] == ["--", request.prompt]
    assert command.index("--disallowed-tools") < command.index("--")


@pytest.mark.asyncio
async def test_cli_process_emits_session_usage_artifact_and_terminal(tmp_path: Path) -> None:
    executable = write_executable(
        tmp_path / "fake-claude",
        """
printf 'input\\n' > source.txt
printf '%s\\n' '{"type":"system","subtype":"init","session_id":"session-1"}'
printf '%s\\n' '{"type":"result","result":"done","usage":{"input_tokens":2,"output_tokens":3},"num_turns":1}'
""",
    )
    request = make_request(tmp_path / "workspace")
    events = [event async for event in ClaudeCodeCliRuntime(str(executable)).run(request)]
    assert events[-1].event_type == RuntimeEventType.RUNTIME_FINISHED
    assert events[-1].session is not None
    assert events[-1].session.external_session_id == "session-1"
    assert any(event.event_type == RuntimeEventType.USAGE for event in events)
    assert any(
        event.event_type == RuntimeEventType.ARTIFACT_WRITTEN and event.message == "source.txt"
        for event in events
    )

    resumed = [
        event
        async for event in ClaudeCodeCliRuntime(str(executable)).resume(
            RuntimeResumeRequest(
                request=request,
                session=events[-1].session,
                prompt="Continue.",
            )
        )
    ]
    assert resumed[-1].event_type == RuntimeEventType.RUNTIME_FINISHED


@pytest.mark.asyncio
async def test_cli_authentication_retry_fails_fast(tmp_path: Path) -> None:
    executable = write_executable(
        tmp_path / "fake-claude",
        """
printf '%s\\n' '{"type":"system","subtype":"init","session_id":"session-1"}'
printf '%s\\n' '{"type":"system","subtype":"api_retry","error":"authentication_failed"}'
sleep 30
""",
    )
    started = time.monotonic()
    events = [
        event
        async for event in ClaudeCodeCliRuntime(str(executable)).run(make_request(tmp_path / "workspace"))
    ]
    assert time.monotonic() - started < 8
    assert events[-1].event_type == RuntimeEventType.RUNTIME_FAILED
    assert events[-1].error_code == RuntimeErrorCode.AUTH_UNAVAILABLE


@pytest.mark.asyncio
async def test_cli_timeout_cleans_up_process(tmp_path: Path) -> None:
    executable = write_executable(
        tmp_path / "fake-claude",
        """
sleep 30
""",
    )
    request = make_request(tmp_path / "workspace").model_copy(update={"timeout_seconds": 1})
    runtime = ClaudeCodeCliRuntime(str(executable))
    events = [event async for event in runtime.run(request)]
    assert events[-1].event_type == RuntimeEventType.RUNTIME_FAILED
    assert events[-1].error_code == RuntimeErrorCode.TIMEOUT
    assert runtime._processes == {}
