from __future__ import annotations

from env_mock_agent.schemas import FindingCategory, Severity, ValidationFinding
from env_mock_agent.validators.base import (
    ArtifactValidator,
    ValidationRequest,
    finding,
)
from env_mock_agent.validators.extract import extract_text


class TextValidator(ArtifactValidator):
    name = "text"

    def validate(self, request: ValidationRequest) -> list[ValidationFinding]:
        text = extract_text(request.artifact_path())
        contract = request.plan.content_contract
        findings: list[ValidationFinding] = []
        minimum = contract.get("min_characters", 1)
        minimum_characters = int(minimum) if isinstance(minimum, int | float | str) else 1
        if len(text.strip()) < minimum_characters:
            findings.append(
                finding(
                    self.name,
                    request,
                    Severity.P1,
                    FindingCategory.COVERAGE,
                    f"text has {len(text.strip())} characters; expected at least {minimum_characters}",
                    repair_action="add the missing source material and required detail",
                )
            )
        required_terms = contract.get("required_terms")
        if isinstance(required_terms, list):
            for term in required_terms:
                value = str(term)
                if value.casefold() not in text.casefold():
                    findings.append(
                        finding(
                            self.name,
                            request,
                            Severity.P1,
                            FindingCategory.COVERAGE,
                            f"required term is missing: {value}",
                            repair_action=f"add evidence-grounded content covering {value}",
                        )
                    )
        return findings
