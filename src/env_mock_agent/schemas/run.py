from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class RunStatus(StrEnum):
    CREATED = "created"
    PLANNING = "planning"
    RETRIEVING = "retrieving"
    BUILDING = "building"
    VALIDATING = "validating"
    REVIEWING = "reviewing"
    REPAIRING = "repairing"
    ASSEMBLING = "assembling"
    PACKAGED = "packaged"
    EXPORTED = "exported"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED_CAPABILITY = "blocked_capability"


class RunRecord(BaseModel):
    schema_version: int = 1
    run_id: str
    task_id: str
    profile: str
    status: RunStatus = RunStatus.CREATED
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    current_node: str | None = None
    task_path: str | None = None
    package_path: str | None = None
    export_path: str | None = None
    review_round: int = 0
    error: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class WorldLedger(BaseModel):
    schema_version: int = 1
    entities: dict[str, dict[str, object]] = Field(default_factory=dict)
    timeline: list[dict[str, object]] = Field(default_factory=list)
    relationships: list[dict[str, object]] = Field(default_factory=list)
    locked_facts: dict[str, object] = Field(default_factory=dict)
    allowed_conflicts: list[str] = Field(default_factory=list)
