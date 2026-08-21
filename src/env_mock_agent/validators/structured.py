from __future__ import annotations

import json
from pathlib import Path

import yaml

from env_mock_agent.schemas import FindingCategory, Severity, ValidationFinding
from env_mock_agent.validators.base import (
    ArtifactValidator,
    ValidationRequest,
    finding,
)


class StructuredDataValidator(ArtifactValidator):
    name = "structured"

    def validate(self, request: ValidationRequest) -> list[ValidationFinding]:
        path = request.artifact_path()
        try:
            data = self._load(path)
        except (json.JSONDecodeError, yaml.YAMLError, UnicodeDecodeError) as exc:
            return [
                finding(
                    self.name,
                    request,
                    Severity.P0,
                    FindingCategory.STRUCTURE,
                    f"structured artifact cannot be parsed: {exc}",
                    repair_action="regenerate valid structured data",
                )
            ]
        findings: list[ValidationFinding] = []
        required_keys = request.plan.content_contract.get("required_keys")
        if isinstance(required_keys, list):
            if not isinstance(data, dict):
                findings.append(
                    finding(
                        self.name,
                        request,
                        Severity.P1,
                        FindingCategory.COVERAGE,
                        "required keys were specified but root value is not an object",
                    )
                )
            else:
                for key in required_keys:
                    if str(key) not in data:
                        findings.append(
                            finding(
                                self.name,
                                request,
                                Severity.P1,
                                FindingCategory.COVERAGE,
                                f"required key is missing: {key}",
                            )
                        )
        return findings

    @staticmethod
    def _load(path: Path) -> object:
        text = path.read_text(encoding="utf-8")
        return json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
