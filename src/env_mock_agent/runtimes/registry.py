from __future__ import annotations

from env_mock_agent.runtimes.base import AgentRuntime
from env_mock_agent.runtimes.claude_agent_sdk import ClaudeAgentSdkRuntime
from env_mock_agent.runtimes.claude_cli import ClaudeCodeCliRuntime
from env_mock_agent.runtimes.fake import FakeRuntime
from env_mock_agent.runtimes.pi_rpc import PiRpcRuntime
from env_mock_agent.schemas import RuntimeName


class RuntimeRegistry:
    def __init__(self) -> None:
        self._runtimes: dict[RuntimeName, AgentRuntime] = {}

    def register(self, name: RuntimeName, runtime: AgentRuntime) -> None:
        self._runtimes[name] = runtime

    def get(self, name: RuntimeName) -> AgentRuntime:
        try:
            return self._runtimes[name]
        except KeyError as exc:
            raise KeyError(f"runtime is not registered: {name}") from exc

    def items(self) -> list[tuple[RuntimeName, AgentRuntime]]:
        return list(self._runtimes.items())

    @classmethod
    def default(cls) -> RuntimeRegistry:
        registry = cls()
        registry.register(RuntimeName.CLAUDE_AGENT_SDK, ClaudeAgentSdkRuntime())
        registry.register(RuntimeName.CLAUDE_CODE_CLI, ClaudeCodeCliRuntime())
        registry.register(RuntimeName.PI_RPC, PiRpcRuntime())
        registry.register(RuntimeName.FAKE, FakeRuntime())
        return registry
