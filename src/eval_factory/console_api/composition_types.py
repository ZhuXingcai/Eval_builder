from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from eval_factory.contracts.core import ContractAudit
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetRunViewV2,
)
from eval_factory.harness.graph_models import (
    HarnessGraphExecutionBindingV1,
)
from eval_factory.harness.runtime_models import (
    HarnessTurnResultV1,
)


class AgentShellCompositionError(RuntimeError):
    code = "AGENT_SHELL_COMPOSITION_ERROR"


class AgentShellCompositionNotConfiguredError(AgentShellCompositionError):
    code = "AGENT_SHELL_COMPOSITION_NOT_CONFIGURED"


class AgentShellCompositionBlockedError(AgentShellCompositionError):
    code = "AGENT_SHELL_COMPOSITION_BLOCKED"

    def __init__(self, reason_codes: tuple[str, ...]) -> None:
        self.reason_codes = reason_codes
        super().__init__("Agent Shell composition is blocked")


class AgentShellCompositionConflictError(AgentShellCompositionError):
    code = "AGENT_SHELL_COMPOSITION_CONFLICT"


class AgentShellCompositionNotFoundError(AgentShellCompositionError):
    code = "AGENT_SHELL_COMPOSITION_NOT_FOUND"


class AgentShellCompositionIntegrityError(AgentShellCompositionError):
    code = "AGENT_SHELL_COMPOSITION_INTEGRITY"


@dataclass(frozen=True, slots=True)
class AgentShellCompositionPreflight:
    ready: bool
    reason_codes: tuple[str, ...]


class AgentShellGraphStarter(Protocol):
    def preflight(
        self,
        session_id: str,
    ) -> AgentShellCompositionPreflight: ...

    async def start(
        self,
        session_id: str,
        *,
        turn: HarnessTurnResultV1,
        principal: str,
        audit: ContractAudit,
    ) -> FactoryDatasetRunViewV2: ...

    async def advance(
        self,
        session_id: str,
        *,
        binding: HarnessGraphExecutionBindingV1,
        audit: ContractAudit,
    ) -> FactoryDatasetRunViewV2: ...

    def current_view(
        self,
        session_id: str,
        *,
        binding: HarnessGraphExecutionBindingV1,
    ) -> FactoryDatasetRunViewV2: ...


__all__ = [
    "AgentShellCompositionBlockedError",
    "AgentShellCompositionConflictError",
    "AgentShellCompositionError",
    "AgentShellCompositionIntegrityError",
    "AgentShellCompositionNotConfiguredError",
    "AgentShellCompositionNotFoundError",
    "AgentShellCompositionPreflight",
    "AgentShellGraphStarter",
]
