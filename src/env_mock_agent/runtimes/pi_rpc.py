from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

from env_mock_agent.config import find_project_root
from env_mock_agent.runtimes.base import AgentRuntime, RuntimeEventSequencer
from env_mock_agent.runtimes.process import drain_stream, isolated_environment, terminate_process
from env_mock_agent.runtimes.workspace_manifest import changed_files, snapshot_workspace
from env_mock_agent.schemas import (
    ModelProfile,
    RuntimeCapabilities,
    RuntimeCredentialStatus,
    RuntimeErrorCode,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeName,
    RuntimeRequest,
    RuntimeResumeRequest,
    RuntimeSandboxEnforcement,
    RuntimeSessionRef,
    RuntimeUsage,
)


def _number(value: object, *, integer: bool = True) -> int | float:
    if isinstance(value, bool):
        return int(value) if integer else float(value)
    if isinstance(value, int | float | str):
        try:
            return int(value) if integer else float(value)
        except (TypeError, ValueError):
            pass
    return 0 if integer else 0.0


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


class PiRpcRuntime(AgentRuntime):
    def __init__(self, bridge_path: Path | None = None) -> None:
        root = find_project_root()
        self.bridge_path = (bridge_path or root / "node/pi_bridge/dist/index.js").resolve()
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._cancelled: set[str] = set()

    async def probe(self) -> RuntimeCapabilities:
        if not self.bridge_path.is_file():
            return RuntimeCapabilities(
                name=RuntimeName.PI_RPC,
                available=False,
                reason=f"Pi bridge is not built: {self.bridge_path}",
            )
        process = await asyncio.create_subprocess_exec(
            "node",
            str(self.bridge_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write(b'{"type":"probe","requestId":"probe"}\n')
            await process.stdin.drain()
            raw = await asyncio.wait_for(process.stdout.readline(), 10)
            payload = json.loads(raw.decode())
            await self._shutdown(process, "probe-shutdown")
            raw_details = payload.get("payload") if isinstance(payload, dict) else None
            details = raw_details if isinstance(raw_details, dict) else {}
            return RuntimeCapabilities(
                name=RuntimeName.PI_RPC,
                available=bool(details.get("available", True)),
                version=str(details.get("piVersion") or ""),
                tools=["read", "write", "edit", "bash", "grep", "find", "ls"],
                supports_resume=bool(details.get("supportsResume")),
                supports_event_streaming=bool(details.get("supportsEventStreaming", True)),
                supports_tool_progress=bool(details.get("supportsToolProgress")),
                credential_status=RuntimeCredentialStatus(str(details.get("credentialStatus") or "UNKNOWN")),
                sandbox_enforcement=RuntimeSandboxEnforcement(
                    str(details.get("sandboxEnforcement") or "NONE")
                ),
                reason="protocol readiness verified; credential readiness is reported separately",
            )
        except Exception as exc:
            await terminate_process(process)
            return RuntimeCapabilities(
                name=RuntimeName.PI_RPC,
                available=False,
                reason=str(exc),
            )

    async def run(self, request: RuntimeRequest) -> AsyncIterator[RuntimeEvent]:
        async for event in self._execute(request):
            yield event

    async def resume(self, resume_request: RuntimeResumeRequest) -> AsyncIterator[RuntimeEvent]:
        if not resume_request.session.session_path:
            raise ValueError("Pi resume requires session_path")
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
        self._prepare_model_config(workspace, request.model_profile)
        handle = f"pi-rpc:{request.run_id}:{request.artifact_id}"
        events = RuntimeEventSequencer(request, RuntimeName.PI_RPC, handle)
        before = snapshot_workspace(workspace)
        environment = isolated_environment(workspace)
        environment["PI_CONFIG_DIR"] = str(workspace / ".pi-home")
        process = await asyncio.create_subprocess_exec(
            "node",
            str(self.bridge_path),
            cwd=find_project_root(),
            env=environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        self._processes[handle] = process
        stderr_task = asyncio.create_task(drain_stream(process.stderr))
        command = self._command(request, handle, resume_session)
        yield events.event(RuntimeEventType.RUNTIME_STARTED, session=resume_session)

        session = resume_session
        usage: RuntimeUsage | None = None
        failure_message = ""
        failure_code: RuntimeErrorCode | None = None
        result_message = ""
        try:
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write((json.dumps(command) + "\n").encode())
            await process.stdin.drain()
            async with asyncio.timeout(request.timeout_seconds):
                async for raw in process.stdout:
                    if not raw.strip():
                        continue
                    payload = json.loads(raw.decode())
                    raw_type, details = self._payload(payload)
                    updated_session = self._session(request, workspace, handle, details)
                    if updated_session is not None:
                        session = updated_session
                    if raw_type == "runtime_started":
                        continue
                    if raw_type == "usage":
                        usage = self._usage(details)
                        continue
                    if raw_type == "runtime_finished":
                        result_message = str(details.get("message") or "")
                        break
                    if raw_type == "runtime_failed":
                        failure_message = str(details.get("error") or "Pi runtime failed")
                        failure_code = self._error_code(details.get("errorCode"))
                        break
                    yield self._normalize(events, raw_type, details)
                await self._shutdown(process, f"{handle}:shutdown")
                stderr = await stderr_task
                if process.returncode not in {0, None} and not failure_message:
                    failure_message = stderr or f"Pi bridge exited with {process.returncode}"
                    failure_code = RuntimeErrorCode.PROCESS_ERROR
        except TimeoutError:
            await terminate_process(process)
            failure_message = f"Pi runtime timed out after {request.timeout_seconds}s"
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
            failure_message = "Pi runtime cancelled"
            failure_code = RuntimeErrorCode.CANCELLED
        for entry in changed_files(before, snapshot_workspace(workspace)):
            yield events.event(
                RuntimeEventType.ARTIFACT_WRITTEN,
                message=entry.relative_path,
                payload=entry.as_payload(),
            )
        yield events.event(
            RuntimeEventType.USAGE,
            usage=usage or RuntimeUsage(),
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

    @staticmethod
    def _command(
        request: RuntimeRequest,
        handle: str,
        resume_session: RuntimeSessionRef | None,
    ) -> dict[str, object]:
        command: dict[str, object] = {
            "type": "resume" if resume_session else "start",
            "requestId": handle,
            "workspace": str(Path(request.workspace).expanduser().resolve()),
            "prompt": request.prompt,
            "systemPolicy": request.system_policy,
            "provider": request.model_profile.provider,
            "model": request.model_profile.model,
            "thinkingLevel": request.model_profile.thinking or "high",
            "tools": (
                [tool.lower() for tool in request.allowed_tools]
                if request.allowed_tools
                else ["read", "write", "edit", "bash", "grep", "find", "ls"]
            ),
            "excludeTools": [tool.lower() for tool in request.disallowed_tools],
            "maxTurns": request.max_turns,
            "maxToolEvents": request.max_tool_events,
        }
        if resume_session and resume_session.session_path:
            command["sessionFile"] = resume_session.session_path
        return command

    @staticmethod
    def _prepare_model_config(workspace: Path, profile: ModelProfile) -> None:
        if profile.provider != "ark":
            return
        agent_dir = workspace / ".pi-home"
        agent_dir.mkdir(parents=True, exist_ok=True)
        configuration = {
            "providers": {
                "ark": {
                    "baseUrl": "https://ark.cn-beijing.volces.com/api/v3",
                    "api": "openai-completions",
                    "apiKey": "$ARK_API_KEY",
                    "compat": {
                        "supportsDeveloperRole": False,
                        "supportsReasoningEffort": False,
                    },
                    "models": [
                        {
                            "id": str(profile.model),
                            "name": f"Ark {profile.model}",
                            "reasoning": False,
                            "input": ["text"],
                            "contextWindow": 200000,
                            "maxTokens": 64000,
                        }
                    ],
                }
            }
        }
        (agent_dir / "models.json").write_text(
            json.dumps(configuration, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _payload(payload: object) -> tuple[str, dict[str, object]]:
        if not isinstance(payload, dict):
            return "invalid_payload", {"raw": payload}
        raw_type = str(payload.get("type") or "")
        details = payload.get("payload")
        normalized = {str(key): value for key, value in details.items()} if isinstance(details, dict) else {}
        return raw_type, normalized

    @staticmethod
    def _normalize(
        events: RuntimeEventSequencer,
        raw_type: str,
        details: dict[str, object],
    ) -> RuntimeEvent:
        tool_name = str(details.get("toolName") or "") or None
        tool_call_id = str(details.get("toolCallId") or "") or None
        if raw_type == "tool_execution_start":
            event_type = RuntimeEventType.TOOL_STARTED
        elif raw_type == "tool_execution_update":
            event_type = RuntimeEventType.TOOL_PROGRESS
        elif raw_type == "tool_execution_end":
            event_type = RuntimeEventType.TOOL_FINISHED
        elif raw_type in {"message_update", "message_end"}:
            event_type = RuntimeEventType.MESSAGE_DELTA
        else:
            event_type = RuntimeEventType.WARNING
        return events.event(
            event_type,
            message=str(details.get("error") or ""),
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            raw_type=raw_type,
            payload=details,
        )

    @staticmethod
    def _usage(details: dict[str, object]) -> RuntimeUsage:
        return RuntimeUsage(
            input_tokens=int(_number(details.get("inputTokens"))),
            output_tokens=int(_number(details.get("outputTokens"))),
            cache_creation_input_tokens=int(_number(details.get("cacheWriteTokens"))),
            cache_read_input_tokens=int(_number(details.get("cacheReadTokens"))),
            cost_usd=_optional_float(details.get("costUsd")),
            tool_calls=int(_number(details.get("toolCalls"))),
            turns=int(_number(details.get("turns"))),
        )

    @staticmethod
    def _session(
        request: RuntimeRequest,
        workspace: Path,
        handle: str,
        details: dict[str, object],
    ) -> RuntimeSessionRef | None:
        session_path = str(details.get("sessionFile") or "")
        session_id = str(details.get("sessionId") or "")
        if not session_path and not session_id:
            return None
        return RuntimeSessionRef(
            runtime=RuntimeName.PI_RPC,
            run_handle=handle,
            external_session_id=session_id or None,
            session_path=session_path or None,
            workspace=str(workspace),
            model_profile=request.model_profile,
        )

    @staticmethod
    def _error_code(value: object) -> RuntimeErrorCode:
        try:
            return RuntimeErrorCode(str(value))
        except ValueError:
            return RuntimeErrorCode.PROCESS_ERROR

    @staticmethod
    async def _shutdown(process: asyncio.subprocess.Process, request_id: str) -> None:
        if process.returncode is not None:
            return
        assert process.stdin is not None
        process.stdin.write((json.dumps({"type": "shutdown", "requestId": request_id}) + "\n").encode())
        await process.stdin.drain()
        process.stdin.close()
        await process.stdin.wait_closed()
        await asyncio.wait_for(process.wait(), 10)

    async def cancel(self, run_handle: str) -> None:
        self._cancelled.add(run_handle)
        process = self._processes.get(run_handle)
        if process is None or process.returncode is not None or process.stdin is None:
            return
        process.stdin.write((json.dumps({"type": "cancel", "requestId": run_handle}) + "\n").encode())
        await process.stdin.drain()
