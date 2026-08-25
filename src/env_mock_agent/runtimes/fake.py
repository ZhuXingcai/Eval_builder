from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from env_mock_agent.runtimes.base import AgentRuntime, RuntimeEventSequencer
from env_mock_agent.runtimes.workspace_manifest import changed_files, snapshot_workspace
from env_mock_agent.schemas import (
    RuntimeCapabilities,
    RuntimeCredentialStatus,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeName,
    RuntimeRequest,
    RuntimeResumeRequest,
    RuntimeSandboxEnforcement,
    RuntimeSessionRef,
    RuntimeUsage,
)


class FakeRuntime(AgentRuntime):
    def __init__(self) -> None:
        self.cancelled: set[str] = set()

    async def probe(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            name=RuntimeName.FAKE,
            available=True,
            version="1",
            models=["fake"],
            tools=["write"],
            supports_resume=True,
            credential_status=RuntimeCredentialStatus.NOT_REQUIRED,
            sandbox_enforcement=RuntimeSandboxEnforcement.NOT_APPLICABLE,
        )

    async def run(self, request: RuntimeRequest) -> AsyncIterator[RuntimeEvent]:
        handle = f"fake:{request.run_id}:{request.artifact_id}"
        events = RuntimeEventSequencer(request, RuntimeName.FAKE, handle)
        workspace = Path(request.workspace).expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        before = snapshot_workspace(workspace)
        session = RuntimeSessionRef(
            runtime=RuntimeName.FAKE,
            run_handle=handle,
            external_session_id=f"{request.run_id}:{request.artifact_id}",
            workspace=str(workspace),
            model_profile=request.model_profile,
        )
        yield events.event(RuntimeEventType.RUNTIME_STARTED, session=session)
        output = workspace / "fake-runtime.txt"
        output.write_text(request.prompt + "\n", encoding="utf-8")
        for entry in changed_files(before, snapshot_workspace(workspace)):
            yield events.event(
                RuntimeEventType.ARTIFACT_WRITTEN,
                message=entry.relative_path,
                payload=entry.as_payload(),
            )
        yield events.event(
            RuntimeEventType.USAGE,
            usage=RuntimeUsage(output_tokens=len(request.prompt.split()), turns=1),
        )
        yield events.event(RuntimeEventType.RUNTIME_FINISHED, session=session)

    async def resume(self, resume_request: RuntimeResumeRequest) -> AsyncIterator[RuntimeEvent]:
        request = resume_request.request.model_copy(update={"prompt": resume_request.prompt})
        async for event in self.run(request):
            if event.event_type == RuntimeEventType.RUNTIME_FINISHED:
                event.message = "resumed"
                event.session = resume_request.session
            yield event

    async def cancel(self, run_handle: str) -> None:
        self.cancelled.add(run_handle)
