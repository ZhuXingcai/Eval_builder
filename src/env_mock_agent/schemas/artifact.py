from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ArtifactStatus(StrEnum):
    PLANNED = "planned"
    BUILDING = "building"
    BUILT = "built"
    VALIDATING = "validating"
    VALIDATED = "validated"
    FAILED = "failed"
    BLOCKED_CAPABILITY = "blocked_capability"


class ArtifactPlan(BaseModel):
    model_config = ConfigDict(extra="allow")

    artifact_id: str
    dependency_id: str
    relative_path: str
    asset_type: str
    provider: str | None = None
    runtime_role: str | None = None
    runtime_preference: list[str] = Field(default_factory=list)
    content_contract: dict[str, object] = Field(default_factory=dict)
    render_contract: dict[str, object] = Field(default_factory=dict)
    validators: list[str] = Field(default_factory=list)
    source_evidence_ids: list[str] = Field(default_factory=list)
    seed: int = 0
    status: ArtifactStatus = ArtifactStatus.PLANNED


class ArtifactResult(BaseModel):
    artifact_id: str
    status: ArtifactStatus
    path: str | None = None
    sha256: str | None = None
    bytes_count: int = 0
    runtime: str | None = None
    model: str | None = None
    attempts: int = 0
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)
