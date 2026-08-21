from __future__ import annotations

import fnmatch
import re

from env_mock_agent.schemas import FindingCategory, Severity, ValidationFinding
from env_mock_agent.validators.base import (
    ArtifactValidator,
    ValidationRequest,
    finding,
)
from env_mock_agent.validators.extract import extract_text


class LeakageValidator(ArtifactValidator):
    name = "leakage"

    def validate(self, request: ValidationRequest) -> list[ValidationFinding]:
        text = extract_text(request.artifact_path())
        relative_path = request.plan.relative_path
        findings: list[ValidationFinding] = []
        for forbidden in request.forbidden_outputs:
            for pattern in forbidden.path_patterns:
                if fnmatch.fnmatch(relative_path, pattern):
                    findings.append(
                        finding(
                            self.name,
                            request,
                            Severity.P0,
                            FindingCategory.ANSWER_LEAKAGE,
                            f"artifact path matches forbidden output {forbidden.forbidden_id}: {pattern}",
                            repair_action="remove the final-deliverable artifact",
                        )
                    )
            for pattern in forbidden.semantic_patterns:
                matched = (
                    re.search(pattern[3:], text, flags=re.IGNORECASE) is not None
                    if pattern.startswith("re:")
                    else pattern.casefold() in text.casefold()
                )
                if matched:
                    findings.append(
                        finding(
                            self.name,
                            request,
                            Severity.P0,
                            FindingCategory.ANSWER_LEAKAGE,
                            f"artifact content matches forbidden output {forbidden.forbidden_id}",
                            evidence=[f"pattern={pattern}"],
                            repair_action="remove the answer-bearing content and restore input-state material",
                        )
                    )
        return findings
