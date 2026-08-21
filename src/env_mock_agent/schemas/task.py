from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from env_mock_agent.schemas.dependency import DependencySpec


class InputAdapterType(StrEnum):
    GENERIC = "generic"
    CC = "cc"
    LH = "lh"


class ProductionSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    asset_type: str
    path: str | None = None
    description: str = ""


class ForbiddenOutputSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    forbidden_id: str
    description: str
    path_patterns: list[str] = Field(default_factory=list)
    semantic_patterns: list[str] = Field(default_factory=list)
    reason: str = ""


class RubricSource(BaseModel):
    model_config = ConfigDict(extra="allow")

    path: str | None = None
    raw: object | None = None
    count: int = 0


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: int = 1
    task_id: str
    request_id: str | None = None
    task_name: str
    prompt: str
    category: str | None = None
    business: str | None = None
    source_adapter: InputAdapterType = InputAdapterType.GENERIC
    source_path: str | None = None
    trace_sources: list[str] = Field(default_factory=list)
    original_attachments: list[str] = Field(default_factory=list)
    dependencies: list[DependencySpec] = Field(default_factory=list)
    production: list[ProductionSpec] = Field(default_factory=list)
    forbidden_outputs: list[ForbiddenOutputSpec] = Field(default_factory=list)
    rubrics: RubricSource | None = None
    profile: str = "generic"
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("task_id", "task_name", "prompt")
    @classmethod
    def required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("required text field must not be empty")
        return value

    @field_validator("source_path")
    @classmethod
    def normalize_source_path(cls, value: str | None) -> str | None:
        return str(Path(value).expanduser().resolve()) if value else None

    def dependency_by_id(self, dependency_id: str) -> DependencySpec:
        for dependency in self.dependencies:
            if dependency.dependency_id == dependency_id:
                return dependency
        raise KeyError(dependency_id)
