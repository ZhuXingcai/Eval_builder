from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, Field

from env_mock_agent.runtimes.security import is_path_inside
from env_mock_agent.schemas import (
    ArtifactPlan,
    ArtifactResult,
    SourceEvidence,
    WorldLedger,
)


class ProviderCapability(BaseModel):
    name: str
    asset_types: list[str] = Field(default_factory=list)
    available: bool = True
    reason: str = ""


class ProviderRequest(BaseModel):
    plan: ArtifactPlan
    staging_root: Path
    evidence: list[SourceEvidence] = Field(default_factory=list)
    world_ledger: WorldLedger = Field(default_factory=WorldLedger)

    def destination(self) -> Path:
        root = self.staging_root.expanduser().resolve()
        target = (root / self.plan.relative_path).resolve()
        if not is_path_inside(root, target):
            raise ValueError(f"provider path escapes staging root: {self.plan.relative_path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target


class ArtifactProvider(ABC):
    name: str
    asset_types: tuple[str, ...]

    def probe(self) -> ProviderCapability:
        return ProviderCapability(
            name=self.name,
            asset_types=list(self.asset_types),
        )

    @abstractmethod
    def generate(self, request: ProviderRequest) -> ArtifactResult:
        raise NotImplementedError
