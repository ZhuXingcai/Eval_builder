from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

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


class RuntimeEventSequencer:
    def __init__(
        self,
        request: RuntimeRequest,
        runtime: RuntimeName,
        run_handle: str,
    ) -> None:
        self.request = request
        self.runtime = runtime
        self.run_handle = run_handle
        self._sequence = 0

    def event(
        self,
        event_type: RuntimeEventType,
        *,
        message: str = "",
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        raw_type: str | None = None,
        payload: dict[str, object] | None = None,
        usage: RuntimeUsage | None = None,
        session: RuntimeSessionRef | None = None,
        error_code: RuntimeErrorCode | None = None,
    ) -> RuntimeEvent:
        self._sequence += 1
        return RuntimeEvent(
            event_type=event_type,
            run_id=self.request.run_id,
            artifact_id=self.request.artifact_id,
            runtime=self.runtime,
            sequence=self._sequence,
            message=message,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            raw_type=raw_type,
            payload=payload or {},
            usage=usage,
            session=session,
            error_code=error_code,
            run_handle=self.run_handle,
        )


class AgentRuntime(ABC):
    @abstractmethod
    async def probe(self) -> RuntimeCapabilities:
        raise NotImplementedError

    @abstractmethod
    def run(self, request: RuntimeRequest) -> AsyncIterator[RuntimeEvent]:
        raise NotImplementedError

    def resume(self, request: RuntimeResumeRequest) -> AsyncIterator[RuntimeEvent]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support resume: {request.session.run_handle}"
        )

    @abstractmethod
    async def cancel(self, run_handle: str) -> None:
        raise NotImplementedError
