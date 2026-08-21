from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, Field

from env_mock_agent.schemas import (
    ArtifactPlan,
    ArtifactResult,
    DependencySpec,
    FindingCategory,
    ForbiddenOutputSpec,
    Severity,
    SourceEvidence,
    ValidationFinding,
    WorldLedger,
)


class ValidationRequest(BaseModel):
    plan: ArtifactPlan
    result: ArtifactResult
    dependency: DependencySpec | None = None
    evidence: list[SourceEvidence] = Field(default_factory=list)
    forbidden_outputs: list[ForbiddenOutputSpec] = Field(default_factory=list)
    world_ledger: WorldLedger = Field(default_factory=WorldLedger)

    def artifact_path(self) -> Path:
        if not self.result.path:
            raise ValueError("artifact result has no path")
        return Path(self.result.path).expanduser().resolve()


class ArtifactValidator(ABC):
    name: str

    @abstractmethod
    def validate(self, request: ValidationRequest) -> list[ValidationFinding]:
        raise NotImplementedError


def finding(
    validator: str,
    request: ValidationRequest,
    severity: Severity,
    category: FindingCategory,
    description: str,
    *,
    evidence: list[str] | None = None,
    repair_action: str = "",
) -> ValidationFinding:
    stable = "|".join(
        (
            validator,
            request.plan.artifact_id,
            category.value,
            description,
        )
    )
    finding_id = f"{validator}-{hashlib.sha256(stable.encode()).hexdigest()[:12]}"
    return ValidationFinding(
        finding_id=finding_id,
        severity=severity,
        category=category,
        description=description,
        evidence=evidence or [],
        artifact_id=request.plan.artifact_id,
        path=request.result.path,
        repair_action=repair_action,
    )
