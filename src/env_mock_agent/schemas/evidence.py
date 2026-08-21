from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class SourceType(StrEnum):
    DIRECT_FILE = "direct_file"
    OFFICIAL = "official"
    AUTHORITATIVE = "authoritative"
    SECONDARY = "secondary"
    SYNTHETIC_GROUNDING = "synthetic_grounding"


class UsagePolicy(StrEnum):
    DIRECT_USE_ALLOWED = "direct_use_allowed"
    FACTS_ONLY = "facts_only"
    STRUCTURE_AND_STYLE_ONLY = "structure_and_style_only"
    NEEDS_REVIEW = "needs_review"
    PROHIBITED = "prohibited"


class SourceEvidence(BaseModel):
    model_config = ConfigDict(extra="allow")

    source_id: str
    url: str | None = None
    local_path: str | None = None
    title: str = ""
    publisher: str = ""
    source_type: SourceType
    usage_policy: UsagePolicy = UsagePolicy.NEEDS_REVIEW
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    sha256: str | None = None
    media_type: str | None = None
    relevant_excerpt: str = ""
    used_by: list[str] = Field(default_factory=list)
    supported_dependencies: list[str] = Field(default_factory=list)
    usage_basis: str = ""
    license_or_usage: str = ""
    retrieval_error: str | None = None
    attempts: int = Field(default=1, ge=0)
    metadata: dict[str, object] = Field(default_factory=dict)


class SearchHit(BaseModel):
    title: str
    url: str
    snippet: str = ""
    rank: int = 0
    provider: str = ""


class FetchResult(BaseModel):
    url: str
    status_code: int
    media_type: str | None = None
    sha256: str
    content_path: str
    bytes_count: int
    headers: dict[str, str] = Field(default_factory=dict)
