from __future__ import annotations

import ast

from env_mock_agent.schemas import FindingCategory, Severity, ValidationFinding
from env_mock_agent.validators.base import (
    ArtifactValidator,
    ValidationRequest,
    finding,
)


class CodeProjectValidator(ArtifactValidator):
    name = "code_project"

    def validate(self, request: ValidationRequest) -> list[ValidationFinding]:
        root = request.artifact_path()
        findings: list[ValidationFinding] = []
        required_files = request.plan.content_contract.get("required_files")
        if not isinstance(required_files, list):
            files = request.plan.content_contract.get("files")
            required_files = list(files) if isinstance(files, dict) else []
        for relative_path in required_files:
            if not (root / str(relative_path)).is_file():
                findings.append(
                    finding(
                        self.name,
                        request,
                        Severity.P1,
                        FindingCategory.COVERAGE,
                        f"required project file is missing: {relative_path}",
                    )
                )
        for python_file in root.rglob("*.py"):
            try:
                ast.parse(python_file.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError) as exc:
                findings.append(
                    finding(
                        self.name,
                        request,
                        Severity.P1,
                        FindingCategory.EXECUTABILITY,
                        f"Python source cannot be parsed: {python_file.name}: {exc}",
                        evidence=[str(python_file)],
                    )
                )
        return findings
