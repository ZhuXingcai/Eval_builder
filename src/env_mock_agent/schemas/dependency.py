from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DependencyKind(StrEnum):
    FILE = "file"
    DIRECTORY = "dir"
    PROJECT = "project"
    DATASET = "dataset"
    GLOB = "glob"
    UNKNOWN = "unknown"


class Criticality(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EvidenceLevel(StrEnum):
    CONFIRMED = "confirmed"
    PARTIAL = "partial"
    CLUE_ONLY = "clue_only"
    REQUIRED_MISSING = "required_missing"
    QUESTIONABLE = "questionable"


class ReconstructionStrategy(StrEnum):
    SEARCH_DOWNLOAD = "search_download"
    SYNTHESIZE = "synthesize"
    SYNTHESIZE_GROUNDED = "synthesize_grounded"
    CODE_SCAFFOLD = "code_scaffold"
    MIXED = "mixed"
    COPY_FROM_SOURCE = "copy_from_source"
    NOT_NEEDED = "not_needed"


class DependencyStatus(StrEnum):
    PLANNED = "planned"
    READY = "ready"
    READY_WITH_APPROVED_FALLBACK = "ready_with_approved_fallback"
    BLOCKED_CAPABILITY = "blocked_capability"
    BUILT = "built"
    VALIDATED = "validated"
    FAILED = "failed"


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    locator: str | None = None
    excerpt: str | None = None


class DependencySpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    dependency_id: str
    path: str
    kind: DependencyKind = DependencyKind.UNKNOWN
    must_exist_before_start: bool = True
    criticality: Criticality = Criticality.MEDIUM
    evidence_level: EvidenceLevel = EvidenceLevel.CLUE_ONLY
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    expected_content: str | dict[str, object] = ""
    reconstruction_strategy: ReconstructionStrategy = ReconstructionStrategy.SYNTHESIZE
    asset_type: str = "file_unspecified"
    risk_notes: str = ""
    minimum_quality: int = Field(default=60, ge=0, le=100)
    status: DependencyStatus = DependencyStatus.PLANNED

    @field_validator("dependency_id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("dependency_id must not be empty")
        return value

    @field_validator("path")
    @classmethod
    def normalize_path(cls, value: str) -> str:
        value = value.strip().replace("\\", "/")
        while value.startswith("./"):
            value = value[2:]
        if not value:
            raise ValueError("dependency path must not be empty")
        if value == ".." or value.startswith("../") or "/../" in f"/{value}/":
            raise ValueError("dependency path must not traverse outside the workspace")
        return value
