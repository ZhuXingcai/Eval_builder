from __future__ import annotations

import filetype  # type: ignore[import-untyped]

from env_mock_agent.schemas import FindingCategory, Severity, ValidationFinding
from env_mock_agent.validators.base import (
    ArtifactValidator,
    ValidationRequest,
    finding,
)


class CommonValidator(ArtifactValidator):
    name = "common"

    def validate(self, request: ValidationRequest) -> list[ValidationFinding]:
        path = request.artifact_path()
        findings: list[ValidationFinding] = []
        if not path.exists():
            return [
                finding(
                    self.name,
                    request,
                    Severity.P0,
                    FindingCategory.STRUCTURE,
                    "artifact path does not exist",
                    repair_action="generate the missing artifact",
                )
            ]
        if path.is_file() and path.stat().st_size == 0:
            findings.append(
                finding(
                    self.name,
                    request,
                    Severity.P0,
                    FindingCategory.STRUCTURE,
                    "artifact file is empty",
                    repair_action="regenerate non-empty content",
                )
            )
        if path.is_dir() and not any(item.is_file() for item in path.rglob("*")):
            findings.append(
                finding(
                    self.name,
                    request,
                    Severity.P0,
                    FindingCategory.STRUCTURE,
                    "artifact directory contains no files",
                    repair_action="populate the project or directory fixture",
                )
            )
        if path.is_file() and path.suffix.lower() in {".pdf", ".docx", ".xlsx"}:
            guessed = filetype.guess(str(path))
            if guessed is None:
                findings.append(
                    finding(
                        self.name,
                        request,
                        Severity.P1,
                        FindingCategory.STRUCTURE,
                        "binary artifact magic bytes are not recognized",
                        evidence=[str(path)],
                        repair_action="regenerate with the format-specific provider",
                    )
                )
        return findings
