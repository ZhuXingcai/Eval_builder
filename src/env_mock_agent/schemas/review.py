from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Severity(StrEnum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class FindingCategory(StrEnum):
    STRUCTURE = "structure"
    COVERAGE = "coverage"
    SOLVABILITY = "solvability"
    AUTHENTICITY = "authenticity"
    CONSISTENCY = "consistency"
    EXECUTABILITY = "executability"
    TRACEABILITY = "traceability"
    SECURITY = "security"
    ANSWER_LEAKAGE = "answer_leakage"
    CAPABILITY = "capability"
    METADATA = "metadata"


class ProblemStatus(StrEnum):
    OPEN = "open"
    FIXED = "fixed"
    WONT_FIX = "wontfix"
    NEEDS_USER = "needs_user"


class ValidationFinding(BaseModel):
    model_config = ConfigDict(extra="allow")

    finding_id: str
    severity: Severity
    category: FindingCategory
    description: str
    evidence: list[str] = Field(default_factory=list)
    artifact_id: str | None = None
    path: str | None = None
    repair_action: str = ""


class Problem(BaseModel):
    model_config = ConfigDict(extra="allow")

    problem_id: str
    severity: Severity
    category: FindingCategory
    description: str
    evidence: list[str] = Field(default_factory=list)
    repair_action: str = ""
    status: ProblemStatus = ProblemStatus.OPEN
    introduced_round: int = Field(ge=0, le=3)
    fixed_round: int | None = Field(default=None, ge=1, le=3)
    regression_of: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ReviewRoundResult(BaseModel):
    round_number: int = Field(ge=1, le=3)
    name: str
    findings: list[ValidationFinding] = Field(default_factory=list)
    score: float = Field(default=0, ge=0, le=100)
    accepted: bool = False
    summary: str = ""


class ProblemLedger(BaseModel):
    schema_version: int = 1
    problems: list[Problem] = Field(default_factory=list)

    @property
    def has_release_blocker(self) -> bool:
        return any(
            item.status == ProblemStatus.OPEN and item.severity in {Severity.P0, Severity.P1}
            for item in self.problems
        )
