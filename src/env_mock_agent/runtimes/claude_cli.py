from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import AsyncIterator
from pathlib import Path

from env_mock_agent.runtimes.base import AgentRuntime, RuntimeEventSequencer
from env_mock_agent.runtimes.process import drain_stream, isolated_environment, terminate_process
from env_mock_agent.runtimes.workspace_manifest import changed_files, snapshot_workspace
from env_mock_agent.schemas import (
    RuntimeCapabilities,
    RuntimeErrorCode,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeName,
    RuntimeRequest,
    RuntimeResumeRequest,
    RuntimeSessionRef,
    RuntimeUsage,
)


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float | str):
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    return 0


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(
        value,
        int | float | str,
    ):
        raise ValueError("optional numeric value is invalid")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("optional numeric value is invalid") from exc


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(
        value,
        int | float | str,
    ):
        raise ValueError("optional integer value is invalid")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("optional integer value is invalid") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("optional integer value is invalid")
    return parsed


class ClaudeCodeCliRuntime(AgentRuntime):
    def __init__(self, executable: str = "claude") -> None:
        self.executable = executable
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._cancelled: set[str] = set()

    async def probe(self) -> RuntimeCapabilities:
        path = shutil.which(self.executable)
        if path is None:
            return RuntimeCapabilities(
                name=RuntimeName.CLAUDE_CODE_CLI,
                available=False,
                reason=f"{self.executable} not found",
            )
        process = await asyncio.create_subprocess_exec(
            path,
            "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        version = (stdout or stderr).decode(errors="replace").strip()
        return RuntimeCapabilities(
            name=RuntimeName.CLAUDE_CODE_CLI,
            available=process.returncode == 0,
            version=version,
            models=[],
            tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebSearch", "WebFetch"],
            supports_resume=True,
            reason="CLI protocol is available; credential readiness is not probed",
        )

    async def run(self, request: RuntimeRequest) -> AsyncIterator[RuntimeEvent]:
        async for event in self._execute(request):
            yield event

    async def resume(self, resume_request: RuntimeResumeRequest) -> AsyncIterator[RuntimeEvent]:
        if not resume_request.session.external_session_id:
            raise ValueError("Claude CLI resume requires external_session_id")
        request = resume_request.request.model_copy(update={"prompt": resume_request.prompt})
        async for event in self._execute(request, resume_request.session):
            yield event

    async def _execute(
        self,
        request: RuntimeRequest,
        resume_session: RuntimeSessionRef | None = None,
    ) -> AsyncIterator[RuntimeEvent]:
        workspace = Path(request.workspace).expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        handle = f"claude-cli:{request.run_id}:{request.artifact_id}"
        events = RuntimeEventSequencer(request, RuntimeName.CLAUDE_CODE_CLI, handle)
        before = snapshot_workspace(workspace)
        command = self._command(request, resume_session)
        environment = isolated_environment(workspace)
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=workspace,
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        self._processes[handle] = process
        stderr_task = asyncio.create_task(drain_stream(process.stderr))
        yield events.event(RuntimeEventType.RUNTIME_STARTED, session=resume_session)

        session = resume_session
        usage: RuntimeUsage | None = None
        result_message = ""
        result_error = False
        tool_calls = 0
        failure_message = ""
        failure_code: RuntimeErrorCode | None = None
        try:
            assert process.stdout is not None
            async with asyncio.timeout(request.timeout_seconds):
                async for raw in process.stdout:
                    line = raw.decode(errors="replace").strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        yield events.event(
                            RuntimeEventType.WARNING,
                            message=line,
                            raw_type="non_json_stdout",
                        )
                        continue
                    if not isinstance(payload, dict):
                        yield events.event(
                            RuntimeEventType.WARNING,
                            raw_type="invalid_payload",
                            payload={"raw": payload},
                        )
                        continue
                    raw_type = str(payload.get("type") or "")
                    subtype = str(payload.get("subtype") or "")
                    normalized_payload = {str(key): value for key, value in payload.items()}
                    if raw_type == "system" and subtype == "init":
                        external_session_id = str(payload.get("session_id") or "")
                        if external_session_id:
                            session = RuntimeSessionRef(
                                runtime=RuntimeName.CLAUDE_CODE_CLI,
                                run_handle=handle,
                                external_session_id=external_session_id,
                                workspace=str(workspace),
                                model_profile=request.model_profile,
                            )
                        continue
                    if (
                        raw_type == "system"
                        and subtype == "api_retry"
                        and str(payload.get("error") or "") == "authentication_failed"
                    ):
                        failure_message = "Claude CLI authentication failed"
                        failure_code = RuntimeErrorCode.AUTH_UNAVAILABLE
                        await terminate_process(process)
                        break
                    if raw_type == "result":
                        usage = self._usage(payload, tool_calls)
                        result_message = str(payload.get("result") or "")
                        result_error = bool(payload.get("is_error")) or subtype.startswith("error")
                        continue
                    event = self._normalize_event(events, payload)
                    if event.event_type == RuntimeEventType.TOOL_STARTED:
                        tool_calls += 1
                        if tool_calls > request.max_tool_events:
                            raise RuntimeError(f"Claude CLI exceeded {request.max_tool_events} tool calls")
                    event.payload = normalized_payload
                    yield event
                return_code = await process.wait()
                stderr = await stderr_task
                if (return_code != 0 or result_error) and not failure_message:
                    failure_message = result_message or stderr or f"Claude CLI exited with {return_code}"
                    failure_code = self._error_code(failure_message)
        except TimeoutError:
            await terminate_process(process)
            failure_message = f"Claude CLI timed out after {request.timeout_seconds}s"
            failure_code = RuntimeErrorCode.TIMEOUT
        except Exception as exc:
            await terminate_process(process)
            failure_message = str(exc)
            failure_code = RuntimeErrorCode.PROCESS_ERROR
        finally:
            if not stderr_task.done():
                await terminate_process(process)
                await stderr_task
            self._processes.pop(handle, None)

        if handle in self._cancelled:
            self._cancelled.discard(handle)
            failure_message = "Claude CLI cancelled"
            failure_code = RuntimeErrorCode.CANCELLED
        for entry in changed_files(before, snapshot_workspace(workspace)):
            yield events.event(
                RuntimeEventType.ARTIFACT_WRITTEN,
                message=entry.relative_path,
                payload=entry.as_payload(),
            )
        yield events.event(
            RuntimeEventType.USAGE,
            usage=usage or RuntimeUsage(tool_calls=tool_calls),
            session=session,
        )
        if failure_message:
            yield events.event(
                RuntimeEventType.RUNTIME_FAILED,
                message=failure_message,
                session=session,
                error_code=failure_code,
            )
        else:
            yield events.event(
                RuntimeEventType.RUNTIME_FINISHED,
                message=result_message,
                session=session,
            )

    def _command(
        self,
        request: RuntimeRequest,
        resume_session: RuntimeSessionRef | None,
    ) -> list[str]:
        tools = request.allowed_tools or ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]
        command = [
            self.executable,
            "-p",
            "--verbose",
            "--output-format",
            "stream-json",
            "--include-hook-events",
            "--permission-mode",
            "acceptEdits",
            "--allowed-tools",
            ",".join(tools),
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--model",
            request.model_profile.model,
            "--effort",
            request.model_profile.effort or "high",
        ]
        if os.environ.get("ANTHROPIC_API_KEY"):
            command.insert(1, "--bare")
        if resume_session and resume_session.external_session_id:
            command.extend(["--resume", resume_session.external_session_id])
        if request.disallowed_tools:
            command.extend(["--disallowed-tools", ",".join(request.disallowed_tools)])
        if request.system_policy:
            command.extend(["--append-system-prompt", request.system_policy])
        if request.max_budget_usd is not None:
            command.extend(["--max-budget-usd", str(request.max_budget_usd)])
        command.extend(["--", request.prompt])
        return command

    @staticmethod
    def _normalize_event(
        events: RuntimeEventSequencer,
        payload: dict[object, object],
    ) -> RuntimeEvent:
        raw_type = str(payload.get("type") or "")
        subtype = str(payload.get("subtype") or "")
        hook_event = str(payload.get("hook_event") or "")
        lowered = f"{raw_type}:{subtype}:{hook_event}".lower()
        tool_name = str(payload.get("tool_name") or "") or None
        tool_call_id = (
            str(payload.get("tool_use_id") or payload.get("tool_call_id") or payload.get("uuid") or "")
            or None
        )
        if raw_type == "system" and subtype == "hook_started" and "tooluse" in lowered:
            event_type = RuntimeEventType.TOOL_STARTED
        elif raw_type == "system" and subtype == "hook_response" and "tooluse" in lowered:
            event_type = RuntimeEventType.TOOL_FINISHED
        elif raw_type in {"assistant", "stream_event"}:
            event_type = RuntimeEventType.MESSAGE_DELTA
        else:
            event_type = RuntimeEventType.WARNING
        return events.event(
            event_type,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            raw_type=f"{raw_type}:{subtype}".rstrip(":"),
        )

    @staticmethod
    def _usage(payload: dict[object, object], tool_calls: int) -> RuntimeUsage:
        raw = payload.get("usage")
        usage = raw if isinstance(raw, dict) else {}
        return RuntimeUsage(
            input_tokens=_as_int(usage.get("input_tokens")),
            output_tokens=_as_int(usage.get("output_tokens")),
            cache_creation_input_tokens=_as_int(usage.get("cache_creation_input_tokens")),
            cache_read_input_tokens=_as_int(usage.get("cache_read_input_tokens")),
            cost_usd=_optional_float(payload.get("total_cost_usd")),
            tool_calls=tool_calls,
            turns=_as_int(payload.get("num_turns")),
            duration_ms=_optional_int(payload.get("duration_ms")),
        )

    @staticmethod
    def _error_code(message: str) -> RuntimeErrorCode:
        lowered = message.lower()
        if "auth" in lowered or "api key" in lowered:
            return RuntimeErrorCode.AUTH_UNAVAILABLE
        if "budget" in lowered:
            return RuntimeErrorCode.BUDGET_EXCEEDED
        if "model" in lowered and ("not found" in lowered or "unavailable" in lowered):
            return RuntimeErrorCode.MODEL_UNAVAILABLE
        return RuntimeErrorCode.PROCESS_ERROR

    async def cancel(self, run_handle: str) -> None:
        self._cancelled.add(run_handle)
        process = self._processes.get(run_handle)
        if process is not None:
            await terminate_process(process)
